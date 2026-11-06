//! Raw-encoding helpers for control flow and the few instruction forms that
//! function discovery and jump-table recovery look at. Matching bit patterns
//! here is exact and much cheaper than a full decode.

/// Where execution goes after an instruction.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Flow {
    /// Falls through to the next instruction.
    Next,
    /// B: always goes to the target.
    Jump(u64),
    /// B.cond, CBZ/CBNZ, TBZ/TBNZ: the target or the next instruction.
    Branch(u64),
    /// BL: calls the target, then falls through.
    Call(u64),
    /// BLR and its PAC forms: calls through a register, then falls through.
    CallIndirect,
    /// BR and its PAC forms: jumps through register n.
    JumpIndirect(u8),
    /// RET, RETAA/RETAB, ERET.
    Return,
    /// UDF, BRK, HLT.
    Trap,
}

impl Flow {
    /// The next instruction is not reached from this one.
    pub fn ends(self) -> bool {
        matches!(
            self,
            Flow::Jump(_) | Flow::JumpIndirect(_) | Flow::Return | Flow::Trap
        )
    }
}

pub fn sext(v: u64, bits: u32) -> i64 {
    ((v << (64 - bits)) as i64) >> (64 - bits)
}
fn rd(w: u32) -> u8 {
    (w & 31) as u8
}
fn rn(w: u32) -> u8 {
    ((w >> 5) & 31) as u8
}
fn rm(w: u32) -> u8 {
    ((w >> 16) & 31) as u8
}
/// Mask bit of x0..x30; register 31 (zr or sp) is not tracked.
pub fn bit(r: u8) -> u32 {
    if r < 31 { 1 << r } else { 0 }
}

pub fn flow(w: u32, pc: u64) -> Flow {
    let rel = |imm: u32, bits: u32| pc.wrapping_add((sext(u64::from(imm), bits) << 2) as u64);
    match w {
        _ if w & 0xFC00_0000 == 0x1400_0000 => Flow::Jump(rel(w & 0x03FF_FFFF, 26)),
        _ if w & 0xFC00_0000 == 0x9400_0000 => Flow::Call(rel(w & 0x03FF_FFFF, 26)),
        _ if w & 0xFF00_0010 == 0x5400_0000 || w & 0x7E00_0000 == 0x3400_0000 => {
            Flow::Branch(rel((w >> 5) & 0x7FFFF, 19))
        }
        _ if w & 0x7E00_0000 == 0x3600_0000 => Flow::Branch(rel((w >> 5) & 0x3FFF, 14)),
        _ if w & 0xFFFF_0000 == 0
            || w & 0xFFE0_001F == 0xD420_0000
            || w & 0xFFE0_001F == 0xD440_0000 =>
        {
            Flow::Trap
        }
        _ if w == 0xD69F_03E0 => Flow::Return, // ERET
        _ if w >> 25 == 0b110_1011 && (w >> 16) & 31 == 31 => {
            let (z, opc, op3, m) = ((w >> 24) & 1, (w >> 21) & 7, (w >> 10) & 0x3F, w & 31);
            match (z, opc, op3) {
                (0, 0, 0) if m == 0 => Flow::JumpIndirect(rn(w)),
                (0, 1, 0) if m == 0 => Flow::CallIndirect,
                (0, 2, 0) if m == 0 => Flow::Return,
                (0, 0, 2 | 3) if m == 31 => Flow::JumpIndirect(rn(w)),
                (0, 1, 2 | 3) if m == 31 => Flow::CallIndirect,
                (0, 2, 2 | 3) if m == 31 && rn(w) == 31 => Flow::Return,
                (1, 0, 2 | 3) => Flow::JumpIndirect(rn(w)),
                (1, 1, 2 | 3) => Flow::CallIndirect,
                _ => Flow::Next,
            }
        }
        _ => Flow::Next,
    }
}

/// x0..x18 and x30: what a call may change.
const CALL_CLOBBER: u32 = 0x4007_FFFF;

/// The general registers an instruction may write. Over-approximates: a
/// spurious write only makes a later analysis give up, never lie.
pub fn writes(w: u32) -> u32 {
    let v = (w >> 26) & 1;
    match (w >> 25) & 0xF {
        // Data processing, immediate.
        0b1000 | 0b1001 => bit(rd(w)),
        // Branches and system.
        0b1010 | 0b1011 => match flow(w, 0) {
            Flow::Call(_) | Flow::CallIndirect => CALL_CLOBBER,
            _ if w & 0xFFF0_0000 == 0xD530_0000 || w & 0xFFF8_0000 == 0xD528_0000 => bit(rd(w)), // MRS, SYSL
            _ => 0,
        },
        // Data processing, register; conditional compares write only flags.
        0b0101 | 0b1101 => {
            if (w >> 21) & 0xFF == 0b1101_0010 {
                0
            } else {
                bit(rd(w))
            }
        }
        // SIMD and FP: only moves and conversions to general registers.
        0b0111 | 0b1111 => {
            if w & 0x5F20_FC00 == 0x1E20_0000 {
                if matches!((w >> 16) & 7, 2 | 3 | 7) {
                    0
                } else {
                    bit(rd(w))
                }
            } else if w & 0x5F20_0000 == 0x1E00_0000 {
                if (w >> 16) & 7 < 2 { bit(rd(w)) } else { 0 }
            } else if w & 0x9FE0_8400 == 0x0E00_0400
                && (w >> 29) & 1 == 0
                && matches!((w >> 11) & 0xF, 5 | 7)
            {
                bit(rd(w)) // SMOV, UMOV
            } else {
                0
            }
        }
        // Loads and stores.
        0b0100 | 0b0110 | 0b1100 | 0b1110 => match (w >> 28) & 3 {
            0 if v == 1 => {
                if (w >> 23) & 1 == 1 { bit(rn(w)) } else { 0 } // LD/ST multiple structures, post-index
            }
            0 => {
                // Exclusives, ordered, compare-and-swap: status or Rs (pair), then Rt and Rt2 for loads.
                let mut m = bit(rm(w)) | bit(rm(w).wrapping_add(1));
                if (w >> 22) & 1 == 1 {
                    m |= bit(rd(w)) | bit(rd(w).wrapping_add(1)) | bit(((w >> 10) & 31) as u8);
                }
                m
            }
            1 if (w >> 24) & 1 == 0 => {
                if v == 0 && w >> 30 != 3 {
                    bit(rd(w))
                } else {
                    0
                } // LDR literal
            }
            1 => bit(rd(w)) | bit(rn(w)) | bit(rm(w)), // LDAPUR, memory copy and set
            2 => {
                let mut m = 0;
                if (w >> 22) & 1 == 1 && v == 0 {
                    m |= bit(rd(w)) | bit(((w >> 10) & 31) as u8);
                }
                if matches!((w >> 23) & 3, 1 | 3) {
                    m |= bit(rn(w));
                }
                m
            }
            _ => {
                let (size, opc) = (w >> 30, (w >> 22) & 3);
                let load = v == 0 && opc != 0 && !(size == 3 && opc == 2);
                let rt = if load { bit(rd(w)) } else { 0 };
                if (w >> 24) & 1 == 1 {
                    rt
                } else if (w >> 21) & 1 == 0 {
                    rt | if matches!((w >> 10) & 3, 1 | 3) {
                        bit(rn(w))
                    } else {
                        0
                    }
                } else {
                    match (w >> 10) & 3 {
                        2 => rt,
                        0 => bit(rd(w)), // atomics
                        _ => bit(rd(w)) | if (w >> 11) & 1 == 1 { bit(rn(w)) } else { 0 }, // LDRAA/LDRAB
                    }
                }
            }
        },
        _ => bit(rd(w)) | bit(rn(w)) | bit(rm(w)),
    }
}

/// ADRP: (rd, page address).
pub fn adrp(w: u32, pc: u64) -> Option<(u8, u64)> {
    (w & 0x9F00_0000 == 0x9000_0000).then(|| {
        let imm = u64::from(((w >> 5) & 0x7FFFF) << 2 | (w >> 29) & 3);
        (
            rd(w),
            (pc & !0xFFF).wrapping_add((sext(imm, 21) << 12) as u64),
        )
    })
}

/// ADR: (rd, address).
pub fn adr(w: u32, pc: u64) -> Option<(u8, u64)> {
    (w & 0x9F00_0000 == 0x1000_0000).then(|| {
        let imm = u64::from(((w >> 5) & 0x7FFFF) << 2 | (w >> 29) & 3);
        (rd(w), pc.wrapping_add(sext(imm, 21) as u64))
    })
}

/// MOVZ: (rd, value).
pub fn movz(w: u32) -> Option<(u8, u64)> {
    (w & 0x7F80_0000 == 0x5280_0000).then(|| {
        (
            rd(w),
            u64::from((w >> 5) & 0xFFFF) << (16 * ((w >> 21) & 3)),
        )
    })
}

/// ADD/SUB immediate without flags: (rd, rn, signed immediate, 64-bit).
pub fn add_imm(w: u32) -> Option<(u8, u8, i64, bool)> {
    ((w >> 23) & 0x3F == 0b10_0010 && (w >> 29) & 1 == 0).then(|| {
        let imm = i64::from((w >> 10) & 0xFFF) << (12 * ((w >> 22) & 1));
        (
            rd(w),
            rn(w),
            if (w >> 30) & 1 == 1 { -imm } else { imm },
            w >> 31 == 1,
        )
    })
}

/// CMP immediate (SUBS to the zero register): (rn, immediate).
pub fn cmp_imm(w: u32) -> Option<(u8, u64)> {
    (w & 0x7F80_001F == 0x7100_001F).then(|| {
        (
            rn(w),
            u64::from((w >> 10) & 0xFFF) << (12 * ((w >> 22) & 1)),
        )
    })
}

/// MOV between general registers (ORR with the zero register): (rd, rm, 64-bit).
pub fn mov_reg(w: u32) -> Option<(u8, u8, bool)> {
    (w & 0x7FE0_FFE0 == 0x2A00_03E0).then(|| (rd(w), rm(w), w >> 31 == 1))
}

/// LDRB/LDRH with an immediate offset (unsigned or unscaled): rt, a value
/// below 2^8 or 2^16. (rt, cap)
pub fn load_narrow(w: u32) -> Option<(u8, u64)> {
    let size = w >> 30;
    let form = w & 0x3FC0_0000 == 0x3940_0000 || w & 0x3FE0_0C00 == 0x3840_0000;
    (form && size < 2).then(|| (rd(w), 1 << (8 << size)))
}

/// UXTW or SXTW (as UBFM/SBFM #0, #31): (rd, rn). Index-preserving copies.
pub fn extend_word(w: u32) -> Option<(u8, u8)> {
    (w & 0xFFFF_FC00 == 0xD340_7C00 || w & 0xFFFF_FC00 == 0x9340_7C00).then(|| (rd(w), rn(w)))
}

/// Narrowing forms: AND #(2^k - 1) and UXTB/UXTH keep the low bits of rn
/// (`low` is set), UBFX/LSR extract a field. (rd, rn, largest value + 1, low)
pub fn narrows(w: u32) -> Option<(u8, u8, u64, bool)> {
    let (n, immr, imms) = ((w >> 22) & 1, (w >> 16) & 0x3F, (w >> 10) & 0x3F);
    let sf = w >> 31 == 1;
    let fits = |imms: u32| if sf { imms < 63 } else { imms < 31 };
    // AND immediate with a low mask (N must match sf for 64-bit masks).
    if (w >> 23) & 0x3F == 0b10_0100
        && (w >> 29) & 3 == 0
        && immr == 0
        && (n == 1 || !sf)
        && fits(imms)
    {
        return Some((rd(w), rn(w), 1 << (imms + 1), true));
    }
    // UBFM: UXTB/UXTH (immr = 0) and UBFX/LSR (imms >= immr).
    if (w >> 23) & 0x3F == 0b10_0110 && (w >> 29) & 3 == 2 && imms >= immr && fits(imms - immr) {
        return Some((rd(w), rn(w), 1 << (imms - immr + 1), immr == 0));
    }
    None
}

/// Load, register offset: (rt, rn, rm, size log2, sign-extends, to a 64-bit
/// register, index extend option, index scaled).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LoadIndexed {
    pub rt: u8,
    pub rn: u8,
    pub rm: u8,
    pub size: u8,
    pub signed: bool,
    pub wide: bool,
    pub option: u8,
    pub scaled: bool,
}

pub fn load_indexed(w: u32) -> Option<LoadIndexed> {
    if w & 0x3F20_0C00 != 0x3820_0800 {
        return None;
    }
    let (size, opc) = ((w >> 30) as u8, ((w >> 22) & 3) as u8);
    let (signed, wide) = match (size, opc) {
        (_, 0) => return None,
        (3, 1) => (false, true),
        (_, 1) => (false, false),
        (0..=1, 2) => (true, true),
        (2, 2) => (true, true),
        (0..=1, 3) => (true, false),
        _ => return None,
    };
    let option = ((w >> 13) & 7) as u8;
    Some(LoadIndexed {
        rt: rd(w),
        rn: rn(w),
        rm: rm(w),
        size,
        signed,
        wide,
        option,
        scaled: (w >> 12) & 1 == 1,
    })
}

/// How ADD (register) transforms its second source.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Extend {
    Lsl(u8),
    Uxtw(u8),
    Sxtw(u8),
}

impl Extend {
    pub fn apply(self, v: u64) -> u64 {
        match self {
            Extend::Lsl(s) => v << s,
            Extend::Uxtw(s) => u64::from(v as u32) << s,
            Extend::Sxtw(s) => (i64::from(v as u32 as i32) as u64) << s,
        }
    }
}

/// 64-bit ADD (shifted or extended register), no flags: (rd, rn, rm, extend of rm).
pub fn add_reg(w: u32) -> Option<(u8, u8, u8, Extend)> {
    if w & 0xFF20_0000 == 0x8B00_0000 && (w >> 22) & 3 == 0 {
        let s = ((w >> 10) & 0x3F) as u8;
        return Some((rd(w), rn(w), rm(w), Extend::Lsl(s)));
    }
    if w & 0xFFE0_0000 == 0x8B20_0000 && (w >> 10) & 7 <= 4 {
        let s = ((w >> 10) & 7) as u8;
        let e = match (w >> 13) & 7 {
            2 => Extend::Uxtw(s),
            6 => Extend::Sxtw(s),
            3 | 7 => Extend::Lsl(s), // UXTX, SXTX
            _ => return None,
        };
        return Some((rd(w), rn(w), rm(w), e));
    }
    None
}

/// Condition code of B.cond and the CSEL family.
pub fn condition(w: u32) -> Option<u8> {
    if w & 0xFF00_0010 == 0x5400_0000 {
        Some((w & 0xF) as u8)
    } else if w & 0x3FE0_0000 == 0x1A80_0000 {
        Some(((w >> 12) & 0xF) as u8)
    } else {
        None
    }
}

/// Reads or writes the flags, other than through `condition`.
pub fn touches_flags(w: u32) -> bool {
    let s = (w >> 29) & 1 == 1;
    w & 0x3FE0_0000 == 0x3A40_0000 // CCMP/CCMN
        || w & 0x1FE0_0000 == 0x1A00_0000 // ADC/SBC
        || s && ((w >> 23) & 0x3F == 0b10_0010 // ADDS/SUBS immediate
            || w & 0x1F00_0000 == 0x0B00_0000 // ADDS/SUBS register
            || (w >> 23) & 0x3F == 0b10_0100 && (w >> 29) & 3 == 3 // ANDS immediate
            || w & 0x1F00_0000 == 0x0A00_0000 && (w >> 29) & 3 == 3) // ANDS register
        || w & 0xFFF0_0000 == 0xD510_0000 || w & 0xFFF0_0000 == 0xD530_0000 // MSR/MRS (NZCV)
        || w & 0xFF20_FC07 == 0x1E20_2000 || w & 0xFF20_0C00 == 0x1E20_0400 // FCMP, FCCMP
}
