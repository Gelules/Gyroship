//! From a decoded instruction to one IR operation.

use crate::ir::{Addr, Alu, Cond, FBin, FUn, Op, Operand, Reduce, Reg, Rmw, Select, Shift, Unary, VCmp, VFp, VInt, VIntUn, Writeback, W, ZERO};
use gyroship_analysis::decode::{decode, Instruction, Opcode, Operand as O, SIMDSizeCode, ShiftStyle, SizeCode};

/// MRS/MSR encodings of TPIDR_EL0 and TPIDRRO_EL0 as yaxpeax numbers them.
const TPIDR_EL0: u16 = 0x5E82;
const TPIDRRO_EL0: u16 = 0x5E83;
const CNTPCT_EL0: u16 = 0x5F01;
const CNTVCT_EL0: u16 = 0x5F02;

pub fn lift(word: u32, pc: u64) -> Op {
    // yield, wfe, wfi: hints for the decoder, waits for the runtime.
    if let 0xD503_203F | 0xD503_205F | 0xD503_207F = word {
        return Op::Yield(((word >> 5) & 7) as u8);
    }
    match decode(word) {
        Some(i) => lift_decoded(&i, pc).unwrap_or(Op::Unhandled),
        None => Op::Unhandled,
    }
}

fn width(s: SizeCode) -> W {
    match s {
        SizeCode::X => W::W64,
        SizeCode::W => W::W32,
    }
}

/// A general register, with 31 as the zero register (`Register`) or SP (`RegisterOrSP`).
fn reg(o: &O) -> Option<(Reg, W)> {
    match *o {
        O::Register(s, 31) => Some((Reg::Zr, width(s))),
        O::Register(s, n) => Some((Reg::X(n as u8), width(s))),
        O::RegisterOrSP(s, 31) => Some((Reg::Sp, width(s))),
        O::RegisterOrSP(s, n) => Some((Reg::X(n as u8), width(s))),
        _ => None,
    }
}

fn r(o: &O) -> Option<Reg> {
    reg(o).map(|(r, _)| r)
}

fn x(n: u16) -> Reg {
    if n == 31 { Reg::Zr } else { Reg::X(n as u8) }
}

fn base(n: u16) -> Reg {
    if n == 31 { Reg::Sp } else { Reg::X(n as u8) }
}

fn imm(o: &O) -> Option<u64> {
    match *o {
        O::Immediate(v) => Some(u64::from(v)),
        O::Imm64(v) => Some(v),
        O::Imm16(v) => Some(u64::from(v)),
        O::ImmShift(v, s) => Some(u64::from(v) << s),
        _ => None,
    }
}

fn operand(o: &O) -> Option<Operand> {
    if let Some(v) = imm(o) {
        return Some(Operand::Imm(v));
    }
    match *o {
        O::Register(_, n) => Some(Operand::Shifted(x(n), Shift::Lsl, 0)),
        O::RegShift(style, amount, _, n) => {
            let ext = |signed, bits| Some(Operand::Extended { reg: x(n), signed, bits, lsl: amount });
            match style {
                ShiftStyle::LSL => Some(Operand::Shifted(x(n), Shift::Lsl, amount)),
                ShiftStyle::LSR => Some(Operand::Shifted(x(n), Shift::Lsr, amount)),
                ShiftStyle::ASR => Some(Operand::Shifted(x(n), Shift::Asr, amount)),
                ShiftStyle::ROR => Some(Operand::Shifted(x(n), Shift::Ror, amount)),
                ShiftStyle::UXTB => ext(false, 8),
                ShiftStyle::UXTH => ext(false, 16),
                ShiftStyle::UXTW => ext(false, 32),
                ShiftStyle::UXTX => ext(false, 64),
                ShiftStyle::SXTB => ext(true, 8),
                ShiftStyle::SXTH => ext(true, 16),
                ShiftStyle::SXTW => ext(true, 32),
                ShiftStyle::SXTX => ext(true, 64),
            }
        }
        _ => None,
    }
}

fn addr(o: &O) -> Option<Addr> {
    Some(match *o {
        O::RegPreIndex(b, off, wb) => {
            Addr { base: base(b), offset: i64::from(off), index: None, writeback: if wb { Writeback::Pre } else { Writeback::None } }
        }
        O::RegPostIndex(b, off) => Addr { base: base(b), offset: i64::from(off), index: None, writeback: Writeback::Post },
        O::RegRegOffset(b, i, _, style, amount) => {
            let (signed, bits) = match style {
                ShiftStyle::LSL | ShiftStyle::UXTX => (false, 64),
                ShiftStyle::SXTX => (true, 64),
                ShiftStyle::UXTW => (false, 32),
                ShiftStyle::SXTW => (true, 32),
                _ => return None,
            };
            Addr { base: base(b), offset: 0, index: Some((x(i), signed, bits, amount)), writeback: Writeback::None }
        }
        _ => return None,
    })
}

fn pc_offset(o: &O, pc: u64) -> Option<u64> {
    match *o {
        O::PCOffset(off) => Some(pc.wrapping_add(off as u64)),
        _ => None,
    }
}

fn cond(o: &O) -> Option<u8> {
    match *o {
        O::ConditionCode(c) => Some(c),
        _ => None,
    }
}

/// An LSE atomic's operation, and its size when the opcode fixes it (B, H);
/// the ordering bits do not matter, every host atomic being sequentially consistent.
fn lse(op: Opcode) -> Option<(Rmw, Option<u8>)> {
    use Opcode as C;
    Some(match op {
        C::LDADD(_) => (Rmw::Add, None),
        C::LDADDB(_) => (Rmw::Add, Some(0)),
        C::LDADDH(_) => (Rmw::Add, Some(1)),
        C::LDCLR(_) => (Rmw::Clr, None),
        C::LDCLRB(_) => (Rmw::Clr, Some(0)),
        C::LDCLRH(_) => (Rmw::Clr, Some(1)),
        C::LDEOR(_) => (Rmw::Eor, None),
        C::LDEORB(_) => (Rmw::Eor, Some(0)),
        C::LDEORH(_) => (Rmw::Eor, Some(1)),
        C::LDSET(_) => (Rmw::Set, None),
        C::LDSETB(_) => (Rmw::Set, Some(0)),
        C::LDSETH(_) => (Rmw::Set, Some(1)),
        C::LDSMAX(_) => (Rmw::Smax, None),
        C::LDSMAXB(_) => (Rmw::Smax, Some(0)),
        C::LDSMAXH(_) => (Rmw::Smax, Some(1)),
        C::LDSMIN(_) => (Rmw::Smin, None),
        C::LDSMINB(_) => (Rmw::Smin, Some(0)),
        C::LDSMINH(_) => (Rmw::Smin, Some(1)),
        C::LDUMAX(_) => (Rmw::Umax, None),
        C::LDUMAXB(_) => (Rmw::Umax, Some(0)),
        C::LDUMAXH(_) => (Rmw::Umax, Some(1)),
        C::LDUMIN(_) => (Rmw::Umin, None),
        C::LDUMINB(_) => (Rmw::Umin, Some(0)),
        C::LDUMINH(_) => (Rmw::Umin, Some(1)),
        C::SWP(_) => (Rmw::Swp, None),
        C::SWPB(_) => (Rmw::Swp, Some(0)),
        C::SWPH(_) => (Rmw::Swp, Some(1)),
        _ => return None,
    })
}

fn lift_decoded(i: &Instruction, pc: u64) -> Option<Op> {
    use Opcode as C;
    let o = &i.operands;
    let alu = |op: Alu, flags: bool| -> Option<Op> {
        let (dst, w) = reg(&o[0])?;
        Some(Op::Alu { op, w, dst, a: r(&o[1])?, b: operand(&o[2])?, flags })
    };
    // Loads and stores: (size log2, signed) from the opcode; None means "size of Rt".
    let load = |size: Option<u8>, signed: bool| -> Option<Op> {
        let (dst, w) = reg(&o[0])?;
        let size = size.unwrap_or(if w == W::W64 { 3 } else { 2 });
        let ordered = matches!(
            i.opcode,
            C::LDAR | C::LDARB | C::LDARH | C::LDAPR | C::LDAPRB | C::LDAPRH | C::LDAPUR | C::LDAPURB | C::LDAPURH | C::LDAPURSB | C::LDAPURSH | C::LDAPURSW
        );
        Some(Op::Load { dst, w, size, signed, ordered, addr: addr(&o[1])? })
    };
    let store = |size: Option<u8>| -> Option<Op> {
        let (src, w) = reg(&o[0])?;
        let size = size.unwrap_or(if w == W::W64 { 3 } else { 2 });
        let ordered = matches!(i.opcode, C::STLR | C::STLRB | C::STLRH | C::STLUR | C::STLURB | C::STLURH);
        Some(Op::Store { src, size, ordered, addr: addr(&o[1])? })
    };
    if let Some(op) = lift_simd(i) {
        return Some(op);
    }
    Some(match i.opcode {
        C::ADD => alu(Alu::Add, false)?,
        C::ADDS => alu(Alu::Add, true)?,
        C::SUB => alu(Alu::Sub, false)?,
        C::SUBS => alu(Alu::Sub, true)?,
        C::AND => alu(Alu::And, false)?,
        C::ANDS => alu(Alu::And, true)?,
        C::ORR => alu(Alu::Orr, false)?,
        C::EOR => alu(Alu::Eor, false)?,
        C::BIC => alu(Alu::Bic, false)?,
        C::BICS => alu(Alu::Bic, true)?,
        C::ORN => alu(Alu::Orn, false)?,
        C::EON => alu(Alu::Eon, false)?,
        C::MOVZ | C::MOVN | C::MOVK => {
            let (dst, w) = reg(&o[0])?;
            let O::ImmShift(imm, shift) = o[1] else { return None };
            Op::MovWide { w, dst, imm, shift, invert: i.opcode == C::MOVN, keep: i.opcode == C::MOVK }
        }
        C::UBFM | C::SBFM | C::BFM => {
            let (dst, w) = reg(&o[0])?;
            Op::Bitfield {
                w,
                dst,
                src: r(&o[1])?,
                immr: imm(&o[2])? as u8,
                imms: imm(&o[3])? as u8,
                signed: i.opcode == C::SBFM,
                insert: i.opcode == C::BFM,
            }
        }
        C::EXTR => {
            let (dst, w) = reg(&o[0])?;
            Op::Extr { w, dst, a: r(&o[1])?, b: r(&o[2])?, lsb: imm(&o[3])? as u8 }
        }
        C::LSLV | C::LSRV | C::ASRV | C::RORV => {
            let (dst, w) = reg(&o[0])?;
            let op = match i.opcode {
                C::LSLV => Shift::Lsl,
                C::LSRV => Shift::Lsr,
                C::ASRV => Shift::Asr,
                _ => Shift::Ror,
            };
            Op::ShiftVar { op, w, dst, a: r(&o[1])?, b: r(&o[2])? }
        }
        C::MADD | C::MSUB => {
            let (dst, w) = reg(&o[0])?;
            Op::MulAdd { w, dst, a: r(&o[1])?, b: r(&o[2])?, acc: r(&o[3])?, sub: i.opcode == C::MSUB }
        }
        C::SMADDL | C::SMSUBL | C::UMADDL | C::UMSUBL => Op::MulAddLong {
            dst: r(&o[0])?,
            a: r(&o[1])?,
            b: r(&o[2])?,
            acc: r(&o[3])?,
            signed: matches!(i.opcode, C::SMADDL | C::SMSUBL),
            sub: matches!(i.opcode, C::SMSUBL | C::UMSUBL),
        },
        C::SMULH | C::UMULH => Op::MulHigh { dst: r(&o[0])?, a: r(&o[1])?, b: r(&o[2])?, signed: i.opcode == C::SMULH },
        C::UDIV | C::SDIV => {
            let (dst, w) = reg(&o[0])?;
            Op::Div { w, dst, a: r(&o[1])?, b: r(&o[2])?, signed: i.opcode == C::SDIV }
        }
        C::CSEL | C::CSINC | C::CSINV | C::CSNEG => {
            let (dst, w) = reg(&o[0])?;
            Op::CondSel {
                w,
                dst,
                a: r(&o[1])?,
                b: r(&o[2])?,
                cond: cond(&o[3])?,
                inc: matches!(i.opcode, C::CSINC | C::CSNEG),
                inv: matches!(i.opcode, C::CSINV | C::CSNEG),
            }
        }
        C::CCMP | C::CCMN => {
            let (a, w) = reg(&o[0])?;
            Op::CondCmp { w, a, b: operand(&o[1])?, nzcv: imm(&o[2])? as u8, cond: cond(&o[3])?, add: i.opcode == C::CCMN }
        }
        C::CLZ | C::RBIT | C::REV | C::REV16 | C::REV32 => {
            let (dst, w) = reg(&o[0])?;
            let op = match i.opcode {
                C::CLZ => Unary::Clz,
                C::RBIT => Unary::Rbit,
                C::REV => Unary::Rev,
                C::REV16 => Unary::Rev16,
                _ => Unary::Rev32,
            };
            Op::Unary { op, w, dst, src: r(&o[1])? }
        }
        C::CRC32B | C::CRC32H | C::CRC32W | C::CRC32X | C::CRC32CB | C::CRC32CH | C::CRC32CW | C::CRC32CX => Op::Crc {
            dst: r(&o[0])?,
            acc: r(&o[1])?,
            data: r(&o[2])?,
            size: match i.opcode {
                C::CRC32B | C::CRC32CB => 0,
                C::CRC32H | C::CRC32CH => 1,
                C::CRC32W | C::CRC32CW => 2,
                _ => 3,
            },
            castagnoli: matches!(i.opcode, C::CRC32CB | C::CRC32CH | C::CRC32CW | C::CRC32CX),
        },
        C::ADR => Op::Const { dst: r(&o[0])?, value: pc_offset(&o[1], pc)? },
        C::ADRP => Op::Const { dst: r(&o[0])?, value: (pc & !0xFFF).wrapping_add(pc_offset(&o[1], 0)?) },
        C::LDR | C::LDUR | C::LDAR | C::LDAPR | C::LDAPUR => load(None, false)?,
        C::LDRB | C::LDURB | C::LDARB | C::LDAPRB | C::LDAPURB => load(Some(0), false)?,
        C::LDRH | C::LDURH | C::LDARH | C::LDAPRH | C::LDAPURH => load(Some(1), false)?,
        C::LDRSB | C::LDURSB | C::LDAPURSB => load(Some(0), true)?,
        C::LDRSH | C::LDURSH | C::LDAPURSH => load(Some(1), true)?,
        C::LDRSW | C::LDURSW | C::LDAPURSW => load(Some(2), true)?,
        C::STR | C::STUR | C::STLR | C::STLUR => store(None)?,
        C::STRB | C::STURB | C::STLRB | C::STLURB => store(Some(0))?,
        C::STRH | C::STURH | C::STLRH | C::STLURH => store(Some(1))?,
        C::LDP | C::LDNP | C::LDPSW => {
            let (a, w) = reg(&o[0])?;
            let signed = i.opcode == C::LDPSW;
            let size = if signed || w == W::W32 { 2 } else { 3 };
            Op::LoadPair { a, b: r(&o[1])?, w, size, signed, addr: addr(&o[2])? }
        }
        C::STP | C::STNP => {
            let (a, w) = reg(&o[0])?;
            Op::StorePair { a, b: r(&o[1])?, size: if w == W::W64 { 3 } else { 2 }, addr: addr(&o[2])? }
        }
        C::MRS => match o[1] {
            O::SystemReg(TPIDR_EL0) => Op::ReadThread { dst: r(&o[0])?, ro: false },
            O::SystemReg(TPIDRRO_EL0) => Op::ReadThread { dst: r(&o[0])?, ro: true },
            O::SystemReg(CNTPCT_EL0 | CNTVCT_EL0) => Op::Counter(r(&o[0])?),
            _ => return None,
        },
        // A DMB or DSB with the LD or ST option (CRm & 3 == 1 or 2) orders what x86
        // keeps in order anyway; any other option is a full barrier.
        C::DMB(option) | C::DSB(option) if !matches!(option & 3, 1 | 2) => Op::Fence,
        // Hints, the other barriers and pointer authentication: the host needs none of
        // them (B-key signing is never checked, so stripping a pointer is a no-op too).
        // CLREX goes to the interpreter, which owns the exclusive monitor.
        C::HINT | C::PACIBSP | C::AUTIBSP | C::XPACI | C::XPACLRI | C::DMB(_) | C::DSB(_) | C::ISB | C::PRFM | C::PRFUM => Op::Nop,
        C::B => Op::Jump(pc_offset(&o[0], pc)?),
        C::BL => Op::Call(pc_offset(&o[0], pc)?),
        C::Bcc(c) => Op::Branch(Cond::Flags(c), pc_offset(&o[0], pc)?),
        C::CBZ | C::CBNZ => {
            let (reg, w) = reg(&o[0])?;
            Op::Branch(Cond::Zero { reg, w, nonzero: i.opcode == C::CBNZ }, pc_offset(&o[1], pc)?)
        }
        C::TBZ | C::TBNZ => Op::Branch(Cond::Bit { reg: r(&o[0])?, bit: imm(&o[1])? as u8, set: i.opcode == C::TBNZ }, pc_offset(&o[2], pc)?),
        C::BR => Op::JumpIndirect(r(&o[0])?),
        C::BLR => Op::CallIndirect(r(&o[0])?),
        C::RET => match r(&o[0])? {
            Reg::X(30) => Op::Return,
            other => Op::JumpIndirect(other),
        },
        C::RETAB | C::RETAA => Op::Return,
        C::UDF | C::BRK | C::HLT => Op::Trap(imm(&o[0])? as u16),
        C::SVC => Op::Svc(imm(&o[0])? as u16),
        // LSE atomics, [Rs, Rt, [Xn|SP]]: host atomics of the same width.
        op if lse(op).is_some() => {
            let (rmw, size) = lse(op)?;
            let (src, w) = reg(&o[0])?;
            Op::AtomicRmw { op: rmw, size: size.unwrap_or(if w == W::W64 { 3 } else { 2 }), src, dst: r(&o[1])?, addr: addr(&o[2])? }
        }
        C::CAS(_) | C::CASB(_) | C::CASH(_) => {
            let (cmp, w) = reg(&o[0])?;
            let size = match i.opcode {
                C::CASB(_) => 0,
                C::CASH(_) => 1,
                _ if w == W::W64 => 3,
                _ => 2,
            };
            Op::AtomicCas { size, cmp, new: r(&o[1])?, addr: addr(&o[2])? }
        }
        // Cache maintenance by address (DC CIVAC and the like, IC IVAU): nothing to do
        // on the host. DC ZVA (CRm 4) writes memory: the interpreter does it.
        C::SYS(ops) if ops.op1() == 3 && matches!(o[1], O::ControlReg(7)) && !matches!(o[2], O::ControlReg(4)) => Op::Nop,
        _ => return None,
    })
}

fn simd_size(s: SIMDSizeCode) -> u8 {
    match s {
        SIMDSizeCode::B => 0,
        SIMDSizeCode::H => 1,
        SIMDSizeCode::S => 2,
        SIMDSizeCode::D => 3,
        SIMDSizeCode::Q => 4,
    }
}

/// A scalar FP/SIMD register: (number, log2 bytes).
fn vreg(o: &O) -> Option<(u8, u8)> {
    match *o {
        O::SIMDRegister(s, n) => Some((n as u8, simd_size(s))),
        _ => None,
    }
}

/// A single or double precision register: (number, is double).
fn freg(o: &O) -> Option<(u8, bool)> {
    match vreg(o)? {
        (n, 2) => Some((n, false)),
        (n, 3) => Some((n, true)),
        _ => None,
    }
}

/// A whole vector: (number, 16 bytes, log2 of the element bytes).
fn vector(o: &O) -> Option<(u8, bool, u8)> {
    match *o {
        O::SIMDRegisterElements(v, n, e) => Some((n as u8, v == SIMDSizeCode::Q, simd_size(e))),
        _ => None,
    }
}

/// `lane` (of `esize` log2 bytes) repeated over 8 or 16 bytes.
fn replicate(lane: u64, esize: u8, q: bool) -> u128 {
    let bits = 8u32 << esize;
    let lane = u128::from(if bits == 64 { lane } else { lane & ((1u64 << bits) - 1) });
    let total = if q { 128 } else { 64 };
    (0..total / bits).fold(0u128, |v, k| v | lane << (k * bits))
}

/// FP lanes of a vector: single (false) or double; None for half precision and .1d.
fn fp_lanes(e: u8, q: bool) -> Option<bool> {
    match (e, q) {
        (2, _) => Some(false),
        (3, true) => Some(true),
        _ => None,
    }
}

/// A shuffle whose lane j (of `eb` bytes, in the low `width` bytes; the
/// others are zero) is the lane of a:b that starts at byte `src(j)`.
fn shuffle(dst: u8, a: u8, b: u8, eb: u32, width: u32, src: impl Fn(u32) -> u32) -> Op {
    let mut mask = [ZERO; 16];
    for i in 0..width {
        mask[i as usize] = (src(i / eb) + i % eb) as u8;
    }
    Op::VShuffle { dst, a, b, mask }
}

/// Vector-lane forms; None for everything else.
fn lift_lanes(i: &Instruction) -> Option<Op> {
    use Opcode as C;
    let o = &i.operands;
    let vec_op = matches!(o[0], O::SIMDRegisterElements(..));
    Some(match i.opcode {
        C::LD1 | C::LD1R => {
            let a = addr(&o[1])?;
            match o[0] {
                O::SIMDRegisterGroupLane(t, e, 1, lane) if i.opcode == C::LD1 => Op::LoadLane { dst: t as u8, size: simd_size(e), lane: Some(lane), q: true, addr: a },
                O::SIMDRegisterGroup(v, t, e, 1) if i.opcode == C::LD1R => Op::LoadLane { dst: t as u8, size: simd_size(e), lane: None, q: v == SIMDSizeCode::Q, addr: a },
                _ => return None,
            }
        }
        C::ST1 => match o[0] {
            O::SIMDRegisterGroupLane(t, e, 1, lane) => Op::StoreLane { src: t as u8, size: simd_size(e), lane, addr: addr(&o[1])? },
            _ => return None,
        },
        C::INS => match (&o[0], &o[1]) {
            (&O::SIMDRegisterElementsLane(_, d, e, l), &O::SIMDRegisterElementsLane(_, s, _, k)) => {
                let eb = 1u32 << simd_size(e);
                shuffle(d as u8, d as u8, s as u8, eb, 16, |j| if j == u32::from(l) { 16 + u32::from(k) * eb } else { j * eb })
            }
            (&O::SIMDRegisterElementsLane(_, d, e, l), O::Register(..)) => Op::ToLane { dst: d as u8, src: r(&o[1])?, size: simd_size(e), lane: Some(l), q: true },
            _ => return None,
        },
        C::DUP => match (&o[0], &o[1]) {
            (&O::SIMDRegisterElements(v, d, e), O::Register(..)) => Op::ToLane { dst: d as u8, src: r(&o[1])?, size: simd_size(e), lane: None, q: v == SIMDSizeCode::Q },
            (&O::SIMDRegisterElements(v, d, e), &O::SIMDRegisterElementsLane(_, s, _, k)) => {
                let eb = 1u32 << simd_size(e);
                shuffle(d as u8, s as u8, s as u8, eb, if v == SIMDSizeCode::Q { 16 } else { 8 }, |_| u32::from(k) * eb)
            }
            // DUP (scalar), the MOV of an element into a scalar register.
            (&O::SIMDRegister(e, d), &O::SIMDRegisterElementsLane(_, s, _, k)) => {
                let eb = 1u32 << simd_size(e);
                shuffle(d as u8, s as u8, s as u8, eb, eb, |_| u32::from(k) * eb)
            }
            _ => return None,
        },
        C::EXT => {
            let ((dst, q, _), (a, _, _), (b, _, _)) = (vector(&o[0])?, vector(&o[1])?, vector(&o[2])?);
            let (k, width) = (imm(&o[3])? as u32, if q { 16 } else { 8 });
            shuffle(dst, a, b, 1, width, |j| if j + k < width { j + k } else { 16 + j + k - width })
        }
        C::REV64 | C::REV32 | C::REV16 if vec_op => {
            let ((dst, q, e), (src, _, _)) = (vector(&o[0])?, vector(&o[1])?);
            let eb = 1u32 << e;
            let group = match i.opcode {
                C::REV64 => 8,
                C::REV32 => 4,
                _ => 2,
            };
            if eb >= group {
                return None; // reserved
            }
            let per = group / eb;
            shuffle(dst, src, src, eb, if q { 16 } else { 8 }, |j| (j / per * per + per - 1 - j % per) * eb)
        }
        C::ZIP1 | C::ZIP2 | C::UZP1 | C::UZP2 | C::TRN1 | C::TRN2 => {
            let ((dst, q, e), (a, _, _), (b, _, _)) = (vector(&o[0])?, vector(&o[1])?, vector(&o[2])?);
            let (eb, width) = (1u32 << e, if q { 16 } else { 8 });
            let n = width / eb;
            // Lane k of the concatenation a:b.
            let cat = |k: u32| if k < n { k * eb } else { 16 + (k - n) * eb };
            let op = i.opcode;
            shuffle(dst, a, b, eb, width, |j| {
                let odd = j % 2;
                cat(match op {
                    C::ZIP1 => odd * n + j / 2,
                    C::ZIP2 => odd * n + n / 2 + j / 2,
                    C::UZP1 => 2 * j,
                    C::UZP2 => 2 * j + 1,
                    C::TRN1 => j + odd * (n - 1),
                    _ => j + 1 - odd + odd * n,
                })
            })
        }
        C::FADD | C::FSUB | C::FMUL | C::FDIV | C::FMLA | C::FMLS | C::FADDP if vec_op => {
            let ((dst, q, e), (a, _, _)) = (vector(&o[0])?, vector(&o[1])?);
            let (b, lane) = match o[2] {
                O::SIMDRegisterElementsLane(_, m, _, l) if !matches!(i.opcode, C::FADDP | C::FDIV) => (m as u8, Some(l)),
                O::SIMDRegisterElements(_, m, _) => (m as u8, None),
                _ => return None,
            };
            let op = match i.opcode {
                C::FADD => VFp::Add,
                C::FSUB => VFp::Sub,
                C::FMUL => VFp::Mul,
                C::FDIV => VFp::Div,
                C::FMLA => VFp::Mla,
                C::FMLS => VFp::Mls,
                _ => VFp::AddP,
            };
            Op::VFp { op, f64: fp_lanes(e, q)?, q, dst, a, b, lane, scalar: false }
        }
        // Scalar by element: FMUL s0, s1, v2.s[1] and the like.
        C::FMUL | C::FMLA | C::FMLS if matches!((&o[0], &o[2]), (O::SIMDRegister(..), O::SIMDRegisterElementsLane(..))) => {
            let ((dst, d), (a, _)) = (freg(&o[0])?, freg(&o[1])?);
            let O::SIMDRegisterElementsLane(_, b, _, lane) = o[2] else { return None };
            let op = match i.opcode {
                C::FMUL => VFp::Mul,
                C::FMLA => VFp::Mla,
                _ => VFp::Mls,
            };
            Op::VFp { op, f64: d, q: false, dst, a, b: b as u8, lane: Some(lane), scalar: true }
        }
        // Scalar pairwise: FADDP s0, v1.2s (d0, v1.2d: 16 bytes).
        C::FADDP if matches!(o[0], O::SIMDRegister(..)) => {
            let ((dst, d), (src, _, _)) = (freg(&o[0])?, vector(&o[1])?);
            Op::VFp { op: VFp::AddP, f64: d, q: d, dst, a: src, b: src, lane: None, scalar: true }
        }
        C::FMAX | C::FMIN | C::FMAXNM | C::FMINNM if vec_op => {
            let ((dst, q, e), (a, _, _), (b, _, _)) = (vector(&o[0])?, vector(&o[1])?, vector(&o[2])?);
            let op = match i.opcode {
                C::FMAX => VFp::Max,
                C::FMIN => VFp::Min,
                C::FMAXNM => VFp::MaxNm,
                _ => VFp::MinNm,
            };
            Op::VFp { op, f64: fp_lanes(e, q)?, q, dst, a, b, lane: None, scalar: false }
        }
        C::FNEG | C::FABS | C::FSQRT | C::FRINTM | C::FRINTP | C::FRINTZ | C::FRINTN if vec_op => {
            let ((dst, q, e), (src, _, _)) = (vector(&o[0])?, vector(&o[1])?);
            let op = match i.opcode {
                C::FNEG => FUn::Neg,
                C::FABS => FUn::Abs,
                C::FSQRT => FUn::Sqrt,
                C::FRINTM => FUn::Floor,
                C::FRINTP => FUn::Ceil,
                C::FRINTZ => FUn::Trunc,
                _ => FUn::Nearest,
            };
            Op::VFUn { op, f64: fp_lanes(e, q)?, q, dst, src }
        }
        C::FCMEQ | C::FCMGE | C::FCMGT | C::FCMLE | C::FCMLT if vec_op => {
            let ((dst, q, e), (a, _, _)) = (vector(&o[0])?, vector(&o[1])?);
            let b = match o[2] {
                O::ImmediateDouble(_) => None, // only +0.0 is encodable
                _ => Some(vector(&o[2])?.0),
            };
            let cmp = match i.opcode {
                C::FCMEQ => VCmp::Eq,
                C::FCMGE => VCmp::Ge,
                C::FCMGT => VCmp::Gt,
                C::FCMLE => VCmp::Le,
                _ => VCmp::Lt,
            };
            Op::VFCmp { cmp, f64: fp_lanes(e, q)?, q, dst, a, b }
        }
        C::ADD | C::SUB | C::UMIN | C::UMAX | C::SMIN | C::SMAX | C::ADDP | C::UMINP | C::UMAXP | C::SMINP | C::SMAXP if vec_op => {
            let ((dst, q, size), (a, _, _), (b, _, _)) = (vector(&o[0])?, vector(&o[1])?, vector(&o[2])?);
            let (op, pairwise) = match i.opcode {
                C::ADD => (VInt::Add, false),
                C::SUB => (VInt::Sub, false),
                C::UMIN => (VInt::UMin, false),
                C::UMAX => (VInt::UMax, false),
                C::SMIN => (VInt::SMin, false),
                C::SMAX => (VInt::SMax, false),
                C::ADDP => (VInt::Add, true),
                C::UMINP => (VInt::UMin, true),
                C::UMAXP => (VInt::UMax, true),
                C::SMINP => (VInt::SMin, true),
                _ => (VInt::SMax, true),
            };
            // 64-bit lanes: only ADD/SUB/ADDP, on 16 bytes.
            (size < 3 || (q && matches!(op, VInt::Add | VInt::Sub))).then_some(Op::VInt { op, size, q, pairwise, dst, a, b })?
        }
        C::SCVTF | C::UCVTF if vec_op && matches!(o[2], O::Nothing) => {
            let ((dst, q, e), (src, _, _)) = (vector(&o[0])?, vector(&o[1])?);
            Op::VIntToF { f64: fp_lanes(e, q)?, q, signed: i.opcode == C::SCVTF, dst, src }
        }
        // The low half of each lane, into the low 8 bytes (XTN2: the high 8, the low kept).
        C::XTN | C::XTN2 => {
            let ((dst, _, e), (src, _, _)) = (vector(&o[0])?, vector(&o[1])?);
            let eb = 1u32 << e;
            if e > 2 {
                return None;
            }
            if i.opcode == C::XTN {
                shuffle(dst, src, src, eb, 8, |j| 2 * j * eb)
            } else {
                let mask = std::array::from_fn(|k| if k < 8 { k as u8 } else { (16 + 2 * ((k as u32 - 8) / eb) * eb + (k as u32 - 8) % eb) as u8 });
                Op::VShuffle { dst, a: dst, b: src, mask }
            }
        }
        C::CNT if vec_op => {
            let ((dst, q, size), (src, _, _)) = (vector(&o[0])?, vector(&o[1])?);
            (size == 0).then_some(Op::VIntUn { op: VIntUn::Cnt, size, q, dst, src })?
        }
        C::SHL if vec_op => {
            let ((dst, q, size), (src, _, _)) = (vector(&o[0])?, vector(&o[1])?);
            let k = u8::try_from(imm(&o[2])?).ok()?;
            (k < 8 << size && (size < 3 || q)).then_some(Op::VIntUn { op: VIntUn::Shl(k), size, q, dst, src })?
        }
        C::CMEQ | C::CMGE | C::CMGT | C::CMLE | C::CMLT if vec_op && matches!(o[2], O::Imm64(0)) => {
            let ((dst, q, size), (src, _, _)) = (vector(&o[0])?, vector(&o[1])?);
            let cmp = match i.opcode {
                C::CMEQ => VCmp::Eq,
                C::CMGE => VCmp::Ge,
                C::CMGT => VCmp::Gt,
                C::CMLE => VCmp::Le,
                _ => VCmp::Lt,
            };
            (size < 3 || q).then_some(Op::VIntUn { op: VIntUn::CmpZero(cmp), size, q, dst, src })?
        }
        C::ADDV | C::UMAXV | C::UMINV | C::SMAXV | C::SMINV | C::UADDLV | C::SADDLV => {
            let ((dst, _), (src, q, size)) = (vreg(&o[0])?, vector(&o[1])?);
            let op = match i.opcode {
                C::ADDV => Reduce::Add,
                C::UMAXV => Reduce::UMax,
                C::UMINV => Reduce::UMin,
                C::SMAXV => Reduce::SMax,
                C::SMINV => Reduce::SMin,
                C::UADDLV => Reduce::UAddL,
                _ => Reduce::SAddL,
            };
            // Four lanes at least: 4S takes 16 bytes; no 64-bit lanes.
            (size < 2 || (size == 2 && q)).then_some(Op::VReduce { op, size, q, dst, src })?
        }
        _ => return None,
    })
}

/// FP and SIMD forms the backends translate; None for everything else.
fn lift_simd(i: &Instruction) -> Option<Op> {
    use Opcode as C;
    let o = &i.operands;
    if let Some(op) = lift_lanes(i) {
        return Some(op);
    }
    let f2 = |op: FBin| -> Option<Op> {
        let ((dst, d), (a, da), (b, db)) = (freg(&o[0])?, freg(&o[1])?, freg(&o[2])?);
        (d == da && d == db).then_some(Op::FBin { op, f64: d, dst, a, b })
    };
    let f1 = |op: FUn| -> Option<Op> {
        let ((dst, d), (src, ds)) = (freg(&o[0])?, freg(&o[1])?);
        (d == ds).then_some(Op::FUn { op, f64: d, dst, src })
    };
    let fma = |neg_acc: bool, neg_product: bool| -> Option<Op> {
        let ((dst, d), (a, _), (b, _), (acc, _)) = (freg(&o[0])?, freg(&o[1])?, freg(&o[2])?, freg(&o[3])?);
        Some(Op::FMulAdd { f64: d, dst, a, b, acc, neg_acc, neg_product })
    };
    Some(match i.opcode {
        C::LDR | C::LDUR | C::STR | C::STUR => {
            let (n, size) = vreg(&o[0])?;
            let a = addr(&o[1])?;
            if matches!(i.opcode, C::LDR | C::LDUR) { Op::LoadV { dst: n, size, addr: a } } else { Op::StoreV { src: n, size, addr: a } }
        }
        C::LDP | C::LDNP | C::STP | C::STNP => {
            let ((a, size), (b, _)) = (vreg(&o[0])?, vreg(&o[1])?);
            let at = addr(&o[2])?;
            if matches!(i.opcode, C::LDP | C::LDNP) { Op::LoadPairV { a, b, size, addr: at } } else { Op::StorePairV { a, b, size, addr: at } }
        }
        C::FMOV => match (&o[0], &o[1]) {
            (O::SIMDRegister(..), O::SIMDRegister(..)) => {
                let ((dst, size), (src, _)) = (vreg(&o[0])?, vreg(&o[1])?);
                Op::VMov { dst, src, size }
            }
            (O::SIMDRegister(..), O::Register(..)) => {
                let (dst, size) = vreg(&o[0])?;
                (size >= 2).then_some(Op::ToV { dst, src: r(&o[1])?, size })?
            }
            (O::Register(..), O::SIMDRegister(..)) => {
                let (src, size) = vreg(&o[1])?;
                (size >= 2).then_some(Op::FromV { dst: r(&o[0])?, src, size, lane: 0 })?
            }
            (O::Register(..), O::SIMDRegisterElementsLane(_, src, SIMDSizeCode::D, lane)) => {
                Op::FromV { dst: r(&o[0])?, src: *src as u8, size: 3, lane: *lane }
            }
            (O::SIMDRegister(..), O::ImmediateDouble(x)) => {
                let (dst, d) = freg(&o[0])?;
                let bits = if d { x.to_bits() } else { u64::from((*x as f32).to_bits()) };
                Op::VConst { dst, value: u128::from(bits) }
            }
            (O::SIMDRegisterElements(..), O::ImmediateDouble(x)) => {
                let (dst, q, e) = vector(&o[0])?;
                let lane = match e {
                    3 => x.to_bits(),
                    2 => u64::from((*x as f32).to_bits()),
                    _ => return None,
                };
                Op::VConst { dst, value: replicate(lane, e, q) }
            }
            _ => return None,
        },
        C::MOVI | C::MVNI => {
            let invert = i.opcode == C::MVNI;
            let (dst, value) = match (&o[0], &o[1]) {
                (O::SIMDRegister(SIMDSizeCode::D, n), O::Imm64(v)) => (*n as u8, u128::from(*v)),
                (O::SIMDRegisterElements(..), imm) => {
                    let (dst, q, e) = vector(&o[0])?;
                    let lane = match *imm {
                        O::Imm64(v) => v,
                        O::Immediate(v) => u64::from(v),
                        O::ImmShift(v, s) => u64::from(v) << s,
                        O::ImmShiftMSL(v, s) => (u64::from(v) << s) | ((1u64 << s) - 1),
                        _ => return None,
                    };
                    (dst, replicate(if invert { !lane } else { lane }, e, q))
                }
                _ => return None,
            };
            Op::VConst { dst, value }
        }
        C::UMOV => {
            let O::SIMDRegisterElementsLane(_, src, e, lane) = o[1] else { return None };
            Op::FromV { dst: r(&o[0])?, src: src as u8, size: simd_size(e), lane }
        }
        C::ORR | C::AND | C::EOR | C::BIC | C::ORN => {
            let ((dst, q, _), (a, _, _), (b, _, _)) = (vector(&o[0])?, vector(&o[1])?, vector(&o[2])?);
            let op = match i.opcode {
                C::ORR => Alu::Orr,
                C::AND => Alu::And,
                C::EOR => Alu::Eor,
                C::BIC => Alu::Bic,
                _ => Alu::Orn,
            };
            if op == Alu::Orr && a == b { Op::VMov { dst, src: a, size: if q { 4 } else { 3 } } } else { Op::VLogic { op, q, dst, a, b } }
        }
        C::FADD => f2(FBin::Add)?,
        C::FSUB => f2(FBin::Sub)?,
        C::FMUL => f2(FBin::Mul)?,
        C::FDIV => f2(FBin::Div)?,
        C::FMAX => f2(FBin::Max)?,
        C::FMIN => f2(FBin::Min)?,
        C::FMAXNM => f2(FBin::MaxNm)?,
        C::FMINNM => f2(FBin::MinNm)?,
        C::FNMUL => f2(FBin::Nmul)?,
        C::FABD => f2(FBin::Abd)?,
        C::FNEG => f1(FUn::Neg)?,
        C::FABS => f1(FUn::Abs)?,
        C::FSQRT => f1(FUn::Sqrt)?,
        C::FRINTM => f1(FUn::Floor)?,
        C::FRINTP => f1(FUn::Ceil)?,
        C::FRINTZ => f1(FUn::Trunc)?,
        C::FRINTN | C::FRINTX | C::FRINTI => f1(FUn::Nearest)?,
        C::FRINTA => f1(FUn::Round)?,
        C::BSL | C::BIT | C::BIF => {
            let ((dst, q, _), (a, _, _), (b, _, _)) = (vector(&o[0])?, vector(&o[1])?, vector(&o[2])?);
            let kind = match i.opcode {
                C::BSL => Select::Bsl,
                C::BIT => Select::Bit,
                _ => Select::Bif,
            };
            Op::VSelect { kind, q, dst, a, b }
        }
        C::FMADD => fma(false, false)?,
        C::FMSUB => fma(false, true)?,
        C::FNMADD => fma(true, true)?,
        C::FNMSUB => fma(true, false)?,
        C::FCMP | C::FCMPE => {
            let (a, d) = freg(&o[0])?;
            match o[1] {
                O::ImmediateDouble(_) => Op::FCmp { f64: d, a, b: None }, // FCMP #imm only encodes +0.0
                _ => Op::FCmp { f64: d, a, b: Some(freg(&o[1])?.0) },
            }
        }
        C::FCCMP | C::FCCMPE => {
            let ((a, d), (b, _)) = (freg(&o[0])?, freg(&o[1])?);
            Op::FCondCmp { f64: d, a, b, nzcv: imm(&o[2])? as u8, cond: cond(&o[3])? }
        }
        C::FCSEL => {
            let ((dst, d), (a, _), (b, _)) = (freg(&o[0])?, freg(&o[1])?, freg(&o[2])?);
            Op::FCondSel { f64: d, dst, a, b, cond: cond(&o[3])? }
        }
        C::FCVT => {
            let ((dst, d), (src, s)) = (freg(&o[0])?, freg(&o[1])?);
            (d != s).then_some(Op::FCvt { dst, src, to_f64: d })?
        }
        C::SCVTF | C::UCVTF => {
            let (dst, d) = freg(&o[0])?;
            let (src, w) = reg(&o[1])?;
            let fbits = match o[2] {
                O::Nothing => 0,
                ref x => u8::try_from(imm(x)?).ok()?,
            };
            // A fixed-point 64-bit integer to a single stays interpreted: the interpreter
            // rounds it twice (through a double), which a native conversion would not.
            if fbits > 0 && w == W::W64 && !d {
                return None;
            }
            Op::IntToF { dst, src, w, f64: d, signed: i.opcode == C::SCVTF, fbits }
        }
        C::FCVTZS | C::FCVTZU => {
            if !matches!(o[2], O::Nothing) {
                return None;
            }
            let (dst, w) = reg(&o[0])?;
            let (src, d) = freg(&o[1])?;
            Op::FToInt { dst, w, src, f64: d, signed: i.opcode == C::FCVTZS }
        }
        _ => return None,
    })
}
