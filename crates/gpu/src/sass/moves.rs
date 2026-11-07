//! Moves family: register moves, predicate logic, system registers,
//! uniform-datapath (UR) ALU, warp votes, reductions and shuffles.
//!
//! Contract for the integration:
//! - `pub enum MovesOp` with typed variants (written from the encoding facts
//!   in Mesa NAK's `sm70_encode.rs`, SM 8.x branch; no code copied);
//! - `pub fn decode(w: Word) -> Option<MovesOp>`: Some for every opcode in
//!   `OPCODES`, None for any other;
//! - `pub fn explained(opcode: u16) -> u128`: the bits `decode` reads for that
//!   opcode; ALWAYS_EXPLAINED is added by the check;
//! - `pub const OPCODES: &[u16]`: the 12-bit opcodes the family owns.
//!
//! Register conventions: GPR fields are 8 bits (255 = RZ); before SM 10.0 a
//! UGPR field is 6 bits (63 = URZ); predicate fields are 3 bits (7 = PT, or
//! UPT for uniform predicates).
//!
//! Uniform ALU layout (NAK `encode_ualu_base`): destination UR 16..22,
//! a = UR 24..30, b = UR 32..38 (form 1) or a 32-bit immediate 32..64
//! (form 4), c = UR 64..70; bit 91 is always set. The UR operands carry no
//! abs/neg bits except where an op says so.

use super::prelude::*;

pub const OPCODES: &[u16] = &[
    0x202, 0x802, 0xa02, 0xab9, 0x805, 0xf89, 0x81c, 0x899, 0x292, 0x886, 0x919, 0x804, 0x882, 0xc82, 0x803, 0x891, 0x890, 0x290,
    0x989, 0x2bd, 0x589, 0x88c, 0x3c4, 0xabb, 0x806, 0xc02,
];

/// Load width, 3 bits (NAK `set_mem_type`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MemType {
    U8,
    S8,
    U16,
    S16,
    B32,
    B64,
    B128,
    Reserved7,
}

impl MemType {
    fn at(w: Word, lo: u32) -> MemType {
        [Self::U8, Self::S8, Self::U16, Self::S16, Self::B32, Self::B64, Self::B128, Self::Reserved7][w.field(lo, 3) as usize]
    }
}

/// VOTE mode, bits 72..74 (NAK `OpVote::encode`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum VoteMode {
    All,
    Any,
    Eq,
    Reserved3,
}

/// SHFL mode, bits 58..60 (NAK `OpShfl::encode`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ShflMode {
    Idx,
    Up,
    Down,
    Bfly,
}

/// REDUX operation, bits 78..81 (NAK `OpRedux::encode`); Min/Max are signed
/// when bit 73 is set.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ReduxOp {
    And,
    Or,
    Xor,
    Sum,
    Min,
    Max,
    Reserved(u8),
}

/// USHF data type, bits 73..75 (NAK `OpShf::encode`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ShfType {
    S64,
    U64,
    S32,
    U32,
}

/// How a compare result combines with the accumulator predicate, 2 bits
/// (NAK `set_pred_set_op`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PredSetOp {
    And,
    Or,
    Xor,
    Reserved3,
}

/// Integer comparison, 3 bits (NAK `set_int_cmp_op`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum IntCmp {
    False,
    Lt,
    Eq,
    Le,
    Gt,
    Ne,
    Ge,
    True,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MovesOp {
    /// MOV (0x002; forms 1 reg, 4 imm, 5 cbuf, 6 UR): dst 16..24, source in the
    /// generic ALU b slot, lane mask 72..76 (0xf = all four lanes of a quad).
    /// NAK `OpMov::encode` (encode_alu with only src1).
    Mov { dst: u8, src: Operand, quad_lanes: u8 },
    /// UMOV (0x082; forms 4 imm, 6 UR): UR dst 16..22, source in the ALU b
    /// slot. NAK `OpMov::encode`, uniform branch.
    UMov { dst: u8, src: Operand },
    /// ULDC (0xab9): UR dst 16..22, c[bank 54..59][offset 38..54], or with
    /// bit 91 set the bindless cx[UR 24..30][offset]; width 73..76.
    /// NAK `OpLdc::encode`, uniform branch for SM < 10.0.
    Uldc { dst: u8, src: Src, mem: MemType },
    /// Guess: ULDC with a UR offset, c[bank][UR + offset] (0xabb, one sample).
    /// Same fields as ULDC plus UR 24..30; NAK has no Ampere encoding, but its
    /// SM 10.0 uniform LDC puts the UR offset at 24 with bit 91 set and keeps
    /// the bank at 54..59, which is this word's shape. `bit91` is kept raw.
    UldcIndexed { dst: u8, bank: u8, index: u8, offset: u16, mem: MemType, bit91: bool },
    /// CS2R (0x805): dst 16..24, system register 72..80, bit 80 = .64 (register
    /// pair). In the samples the source is always SRZ (0xff): zeroing a pair.
    /// NAK `OpCS2R::encode`.
    Cs2r { dst: u8, sr: u8, wide: bool },
    /// S2R (0x919): dst 16..24, system register 72..80. NAK `OpS2R::encode`.
    /// Indices (NAK nak_private.h): 0x00 LANEID, 0x21..0x23 TID.XYZ,
    /// 0x25..0x27 CTAID.XYZ, 0x38..0x3c LANEMASK_EQ/LT/LE/GT/GE, 0x50/0x51 CLOCK.
    S2r { dst: u8, sr: u8 },
    /// VOTE (0x806, GPR ballot 16..24) and VOTEU (0x886, UR ballot 16..22):
    /// mode 72..74, vote result predicate 81..84, source predicate 87..90 (not
    /// 90). NAK `OpVote::encode`.
    Vote { uniform: bool, dst: u8, mode: VoteMode, vote: u8, pred: Pred },
    /// SHFL (0x589, 0x989, 0xf89): dst 16..24, value 24..32, lane = R 32..40
    /// or imm 53..58, clamp/segment mask c = imm 40..53 or R 64..72, mode
    /// 58..60, in-bounds predicate 81..84. NAK `OpShfl::encode`.
    Shfl { dst: u8, src: u8, lane: Src, c: Src, mode: ShflMode, in_bounds: u8 },
    /// PLOP3 (0x81c): dst[0] = lut[0](src), dst[1] = lut[1](src).
    /// dst 81..84, 84..87; src[0] 87..90 (not 90), src[1] 77..80 (not 80),
    /// src[2] 68..71 (not 71, a uniform predicate when bit 67 is set);
    /// lut[0] = 64..67 | 72..77 << 3, lut[1] = 16..24. NAK `OpPLop3::encode`.
    Plop3 { dst: [u8; 2], src: [Pred; 3], src2_uniform: bool, lut: [u8; 2] },
    /// Guess: P2R (0x803, ALU form 4): dst 16..24, a = R 24..32 (RZ in every
    /// sample), predicate mask imm 32..64 (single-bit masks in the samples).
    P2r { dst: u8, a: u8, mask: u32 },
    /// Guess: R2P (0x804, ALU form 4): source R 24..32, predicate mask imm
    /// 32..64 (0x3, 0x7, 0x17 in the samples); no GPR destination.
    R2p { src: u8, mask: u32 },
    /// REDUX (0x3c4): UR dst 16..22, source R 24..32, signed 73, op 78..81.
    /// NAK `OpRedux::encode`.
    Redux { dst: u8, src: u8, op: ReduxOp, signed: bool },
    /// UIADD3 (0x090; forms 1, 4): neg a 72, neg b 63 (form 1 only), neg c 75;
    /// .X 74; carry-in UP 87..90 (not 90) and 77..80 (not 80); carry-out UP
    /// 81..84 and 84..87. NAK `OpIAdd3::encode` / `OpIAdd3X::encode`.
    UIAdd3 { dst: u8, a: Operand, b: Operand, c: Operand, x: bool, carry: [Pred; 2], overflow: [u8; 2] },
    /// ULEA (0x091, form 4): dst = (a << shift) + b, a = low, c = high half
    /// (URZ unless .HI); bit 72 negates (or with .X inverts) the shifted
    /// value; .X 74; shift 75..80; .HI 80; carry-out UP 81..84; carry-in UP
    /// 87..90 (not 90). NAK `OpLea::encode` / `OpLeaX::encode`.
    ULea { dst: u8, a: Operand, b: Operand, c: Operand, neg_shifted: bool, x: bool, shift: u8, hi: bool, overflow: u8, carry: Pred },
    /// ULOP3 (0x092, form 1): lut 72..80, .PAND 80, predicate dst UP 81..84,
    /// predicate src UP 87..90 (not 90). NAK `OpLop3::encode`.
    ULop3 { dst: u8, a: Operand, b: Operand, c: Operand, lut: u8, pand: bool, pdst: u8, psrc: Pred },
    /// USHF (0x099, form 4): a = low, b = shift, c = high; type 73..75, .W 75,
    /// .R 76 (else left), .HI 80. NAK `OpShf::encode`.
    UShf { dst: u8, low: Operand, shift: Operand, high: Operand, ty: ShfType, wrap: bool, right: bool, hi: bool },
    /// UISETP (0x08c, form 4): no register dst; .EX 72, signed 73, set op
    /// 74..76, compare 76..79; dst UP 81..84 and 84..87; low-compare UP 68..71
    /// (not 71); accumulator UP 87..90 (not 90). NAK `OpISetP::encode`.
    UISetP { dst: [u8; 2], a: Operand, b: Operand, cmp: IntCmp, set_op: PredSetOp, signed: bool, ex: bool, low_cmp: Pred, accum: Pred },
    /// UFLO (0x0bd, form 1): source in the b slot UR 32..38, bit 63 inverts
    /// it, signed 73, .SH 74 (return a shift amount), predicate dst UP 81..84.
    /// NAK `OpFlo::encode`.
    UFlo { dst: u8, src: Operand, invert: bool, signed: bool, shift_amount: bool, pdst: u8 },
}

use MovesOp::*;

fn pdst(w: Word, lo: u32) -> u8 {
    w.field(lo, 3) as u8
}

fn ur(w: Word, lo: u32) -> Operand {
    Operand::plain(Src::UReg(ureg(w, lo)))
}

fn negated(mut o: Operand, w: Word, bit: u32) -> Operand {
    o.neg = w.bit(bit);
    o
}

/// The uniform ALU b slot: UR 32..38 (form 1) or imm 32..64 (form 4).
fn ub(w: Word) -> Operand {
    if w.field(9, 3) == 4 { Operand::plain(Src::Imm(w.field(32, 32) as u32)) } else { ur(w, 32) }
}

pub fn decode(w: Word) -> Option<MovesOp> {
    let op = w.opcode();
    let p = |lo: u32, not: u32| Pred::at(w, lo, not);
    Some(match op {
        0x202 | 0x802 | 0xa02 | 0xc02 => Mov { dst: dst(w), src: alu(w, false)?.b, quad_lanes: w.field(72, 4) as u8 },
        0x882 | 0xc82 => UMov { dst: ureg(w, 16), src: alu(w, false)?.b },
        0xab9 => {
            let offset = w.field(38, 16) as u16;
            let src = if w.bit(91) { Src::CBufU { ureg: ureg(w, 24), offset } } else { Src::CBuf { bank: w.field(54, 5) as u8, offset } };
            Uldc { dst: ureg(w, 16), src, mem: MemType::at(w, 73) }
        }
        0xabb => UldcIndexed {
            dst: ureg(w, 16),
            bank: w.field(54, 5) as u8,
            index: ureg(w, 24),
            offset: w.field(38, 16) as u16,
            mem: MemType::at(w, 73),
            bit91: w.bit(91),
        },
        0x805 => Cs2r { dst: dst(w), sr: w.field(72, 8) as u8, wide: w.bit(80) },
        0x919 => S2r { dst: dst(w), sr: w.field(72, 8) as u8 },
        0x806 | 0x886 => {
            let uniform = op == 0x886;
            Vote {
                uniform,
                dst: if uniform { ureg(w, 16) } else { dst(w) },
                mode: [VoteMode::All, VoteMode::Any, VoteMode::Eq, VoteMode::Reserved3][w.field(72, 2) as usize],
                vote: pdst(w, 81),
                pred: p(87, 90),
            }
        }
        0x589 | 0x989 | 0xf89 => {
            // form (bits 9..12): 2 = R lane + imm c, 4 = imm lane + R c, 7 = both imm
            let lane_imm = op & 0x800 != 0;
            let c_imm = op & 0x400 != 0;
            Shfl {
                dst: dst(w),
                src: reg(w, 24),
                lane: if lane_imm { Src::Imm(w.field(53, 5) as u32) } else { Src::Reg(reg(w, 32)) },
                c: if c_imm { Src::Imm(w.field(40, 13) as u32) } else { Src::Reg(reg(w, 64)) },
                mode: [ShflMode::Idx, ShflMode::Up, ShflMode::Down, ShflMode::Bfly][w.field(58, 2) as usize],
                in_bounds: pdst(w, 81),
            }
        }
        0x81c => Plop3 {
            dst: [pdst(w, 81), pdst(w, 84)],
            src: [p(87, 90), p(77, 80), p(68, 71)],
            src2_uniform: w.bit(67),
            lut: [(w.field(64, 3) | w.field(72, 5) << 3) as u8, w.field(16, 8) as u8],
        },
        0x803 => P2r { dst: dst(w), a: reg(w, 24), mask: w.field(32, 32) as u32 },
        0x804 => R2p { src: reg(w, 24), mask: w.field(32, 32) as u32 },
        0x3c4 => Redux {
            dst: ureg(w, 16),
            src: reg(w, 24),
            op: match w.field(78, 3) as u8 {
                0 => ReduxOp::And,
                1 => ReduxOp::Or,
                2 => ReduxOp::Xor,
                3 => ReduxOp::Sum,
                4 => ReduxOp::Min,
                5 => ReduxOp::Max,
                v => ReduxOp::Reserved(v),
            },
            signed: w.bit(73),
        },
        0x290 | 0x890 => UIAdd3 {
            dst: ureg(w, 16),
            a: negated(ur(w, 24), w, 72),
            b: if op == 0x290 { negated(ub(w), w, 63) } else { ub(w) },
            c: negated(ur(w, 64), w, 75),
            x: w.bit(74),
            carry: [p(87, 90), p(77, 80)],
            overflow: [pdst(w, 81), pdst(w, 84)],
        },
        0x891 => ULea {
            dst: ureg(w, 16),
            a: ur(w, 24),
            b: ub(w),
            c: ur(w, 64),
            neg_shifted: w.bit(72),
            x: w.bit(74),
            shift: w.field(75, 5) as u8,
            hi: w.bit(80),
            overflow: pdst(w, 81),
            carry: p(87, 90),
        },
        0x292 => ULop3 {
            dst: ureg(w, 16),
            a: ur(w, 24),
            b: ub(w),
            c: ur(w, 64),
            lut: w.field(72, 8) as u8,
            pand: w.bit(80),
            pdst: pdst(w, 81),
            psrc: p(87, 90),
        },
        0x899 => UShf {
            dst: ureg(w, 16),
            low: ur(w, 24),
            shift: ub(w),
            high: ur(w, 64),
            ty: [ShfType::S64, ShfType::U64, ShfType::S32, ShfType::U32][w.field(73, 2) as usize],
            wrap: w.bit(75),
            right: w.bit(76),
            hi: w.bit(80),
        },
        0x88c => UISetP {
            dst: [pdst(w, 81), pdst(w, 84)],
            a: ur(w, 24),
            b: ub(w),
            cmp: [IntCmp::False, IntCmp::Lt, IntCmp::Eq, IntCmp::Le, IntCmp::Gt, IntCmp::Ne, IntCmp::Ge, IntCmp::True][w.field(76, 3) as usize],
            set_op: [PredSetOp::And, PredSetOp::Or, PredSetOp::Xor, PredSetOp::Reserved3][w.field(74, 2) as usize],
            signed: w.bit(73),
            ex: w.bit(72),
            low_cmp: p(68, 71),
            accum: p(87, 90),
        },
        0x2bd => UFlo { dst: ureg(w, 16), src: ub(w), invert: w.bit(63), signed: w.bit(73), shift_amount: w.bit(74), pdst: pdst(w, 81) },
        _ => return None,
    })
}

pub fn explained(opcode: u16) -> u128 {
    const B62_63: u128 = bits(62, 2);
    const UDST: u128 = bits(16, 6);
    // a, b, c of the uniform ALU layout (form 1 or form 4)
    const UABC1: u128 = UDST | bits(24, 6) | bits(32, 6) | bits(64, 6);
    const UABC4: u128 = UDST | bits(24, 6) | bits(32, 32) | bits(64, 6);
    const PRED_SRC: u128 = bits(87, 4);
    const VOTE: u128 = bits(72, 2) | bits(81, 3) | PRED_SRC;
    const SHFL: u128 = bits(16, 16) | bits(58, 2) | bits(81, 3);
    const ULDC: u128 = UDST | bits(24, 6) | bits(38, 21) | bits(73, 3) | bits(91, 1);
    const UIADD3: u128 = bits(72, 1) | bits(74, 2) | bits(77, 14);
    match opcode {
        0x202 => bits(16, 8) | bits(32, 8) | B62_63 | bits(72, 4),
        0x802 => bits(16, 8) | bits(32, 32) | bits(72, 4),
        0xa02 => bits(16, 8) | bits(32, 27) | B62_63 | bits(91, 1) | bits(72, 4),
        0xc02 => bits(16, 8) | bits(32, 6) | B62_63 | bits(72, 4),
        0x882 => UDST | bits(32, 32),
        0xc82 => UDST | bits(32, 6) | B62_63,
        0xab9 | 0xabb => ULDC,
        0x805 => bits(16, 8) | bits(72, 9),
        0x919 => bits(16, 8) | bits(72, 8),
        0x806 => bits(16, 8) | VOTE,
        0x886 => UDST | VOTE,
        0x589 => SHFL | bits(32, 21),
        0x989 => SHFL | bits(53, 5) | bits(64, 8),
        0xf89 => SHFL | bits(40, 18),
        0x81c => bits(16, 8) | bits(64, 27),
        0x803 => bits(16, 48),
        0x804 => bits(24, 40),
        0x3c4 => UDST | bits(24, 8) | bits(73, 1) | bits(78, 3),
        0x290 => UABC1 | bits(63, 1) | UIADD3,
        0x890 => UABC4 | UIADD3,
        0x891 => UABC4 | bits(72, 1) | bits(74, 10) | PRED_SRC,
        0x292 => UABC1 | bits(72, 12) | PRED_SRC,
        0x899 => UABC4 | bits(73, 4) | bits(80, 1),
        0x88c => (UABC4 & !UDST & !bits(64, 6)) | bits(68, 4) | bits(72, 7) | bits(81, 10),
        0x2bd => UDST | bits(32, 6) | bits(63, 1) | bits(73, 2) | bits(81, 3),
        _ => 0,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A word with the given opcode and guard PT, plus (lo, len, value) fields.
    fn word(opcode: u16, fields: &[(u32, u32, u64)]) -> Word {
        let mut v = opcode as u128 | 7 << 12;
        for &(lo, len, val) in fields {
            assert!(len == 64 || val >> len == 0);
            v |= (val as u128) << lo;
        }
        Word(v)
    }

    fn pred(reg: u8, not: bool) -> Pred {
        Pred { reg, not }
    }

    fn reg_op(r: u8) -> Operand {
        Operand::plain(Src::Reg(r))
    }

    fn ur_op(r: u8) -> Operand {
        Operand::plain(Src::UReg(r))
    }

    fn imm_op(v: u32) -> Operand {
        Operand::plain(Src::Imm(v))
    }

    #[test]
    fn mov() {
        let w = word(0x202, &[(16, 8, 3), (32, 8, 9), (63, 1, 1), (72, 4, 0xf)]);
        assert_eq!(decode(w), Some(Mov { dst: 3, src: Operand { neg: true, ..reg_op(9) }, quad_lanes: 0xf }));
        let w = word(0x802, &[(16, 8, 4), (32, 32, 0x3f80_0000), (72, 4, 0x5)]);
        assert_eq!(decode(w), Some(Mov { dst: 4, src: imm_op(0x3f80_0000), quad_lanes: 5 }));
        let w = word(0xa02, &[(16, 8, 5), (38, 16, 0x160), (54, 5, 3), (72, 4, 0xf)]);
        assert_eq!(decode(w), Some(Mov { dst: 5, src: Operand::plain(Src::CBuf { bank: 3, offset: 0x160 }), quad_lanes: 0xf }));
        let w = word(0xa02, &[(16, 8, 5), (32, 6, 7), (38, 16, 0x10), (91, 1, 1), (72, 4, 0xf)]);
        assert_eq!(decode(w), Some(Mov { dst: 5, src: Operand::plain(Src::CBufU { ureg: 7, offset: 0x10 }), quad_lanes: 0xf }));
        let w = word(0xc02, &[(16, 8, 6), (32, 6, 5), (91, 1, 1), (72, 4, 0xf)]);
        assert_eq!(decode(w), Some(Mov { dst: 6, src: ur_op(5), quad_lanes: 0xf }));
    }

    #[test]
    fn umov() {
        let w = word(0x882, &[(16, 6, 4), (32, 32, 0x8000_1234)]);
        assert_eq!(decode(w), Some(UMov { dst: 4, src: imm_op(0x8000_1234) }));
        let w = word(0xc82, &[(16, 6, 5), (32, 6, 6), (91, 1, 1)]);
        assert_eq!(decode(w), Some(UMov { dst: 5, src: ur_op(6) }));
    }

    #[test]
    fn uldc() {
        let w = word(0xab9, &[(16, 6, 4), (38, 16, 0x118), (54, 5, 0xe), (73, 3, 5)]);
        assert_eq!(decode(w), Some(Uldc { dst: 4, src: Src::CBuf { bank: 0xe, offset: 0x118 }, mem: MemType::B64 }));
        let w = word(0xab9, &[(16, 6, 6), (24, 6, 2), (38, 16, 0x20), (73, 3, 2), (91, 1, 1)]);
        assert_eq!(decode(w), Some(Uldc { dst: 6, src: Src::CBufU { ureg: 2, offset: 0x20 }, mem: MemType::U16 }));
        let w = word(0xabb, &[(16, 6, 4), (24, 6, 5), (38, 16, 0x10), (54, 5, 6), (73, 3, 4), (91, 1, 1)]);
        assert_eq!(decode(w), Some(UldcIndexed { dst: 4, bank: 6, index: 5, offset: 0x10, mem: MemType::B32, bit91: true }));
    }

    #[test]
    fn system_registers() {
        let w = word(0x805, &[(16, 8, 2), (72, 8, 0xff), (80, 1, 1)]);
        assert_eq!(decode(w), Some(Cs2r { dst: 2, sr: 0xff, wide: true }));
        let w = word(0x919, &[(16, 8, 7), (72, 8, 0x25)]);
        assert_eq!(decode(w), Some(S2r { dst: 7, sr: 0x25 }));
    }

    #[test]
    fn vote() {
        let w = word(0x806, &[(16, 8, 9), (72, 2, 2), (81, 3, 1), (87, 3, 3), (90, 1, 1)]);
        assert_eq!(decode(w), Some(Vote { uniform: false, dst: 9, mode: VoteMode::Eq, vote: 1, pred: pred(3, true) }));
        let w = word(0x886, &[(16, 6, 4), (72, 2, 1), (81, 3, 7), (87, 3, 7)]);
        assert_eq!(decode(w), Some(Vote { uniform: true, dst: 4, mode: VoteMode::Any, vote: 7, pred: Pred::TRUE }));
    }

    #[test]
    fn shfl() {
        let w = word(0x589, &[(16, 8, 1), (24, 8, 2), (32, 8, 3), (40, 13, 0x1f), (58, 2, 0), (81, 3, 7)]);
        assert_eq!(
            decode(w),
            Some(Shfl { dst: 1, src: 2, lane: Src::Reg(3), c: Src::Imm(0x1f), mode: ShflMode::Idx, in_bounds: 7 })
        );
        let w = word(0x989, &[(16, 8, 4), (24, 8, 5), (53, 5, 1), (58, 2, 1), (64, 8, 0xff), (81, 3, 2)]);
        assert_eq!(
            decode(w),
            Some(Shfl { dst: 4, src: 5, lane: Src::Imm(1), c: Src::Reg(0xff), mode: ShflMode::Up, in_bounds: 2 })
        );
        let w = word(0xf89, &[(16, 8, 6), (24, 8, 7), (40, 13, 0x1c03), (53, 5, 16), (58, 2, 3), (81, 3, 7)]);
        assert_eq!(
            decode(w),
            Some(Shfl { dst: 6, src: 7, lane: Src::Imm(16), c: Src::Imm(0x1c03), mode: ShflMode::Bfly, in_bounds: 7 })
        );
    }

    #[test]
    fn plop3() {
        let lut0: u64 = 0xa8;
        let w = word(
            0x81c,
            &[
                (16, 8, 0xc8),
                (64, 3, lut0 & 7),
                (67, 1, 1),
                (68, 3, 2),
                (71, 1, 1),
                (72, 5, lut0 >> 3),
                (77, 3, 1),
                (81, 3, 0),
                (84, 3, 7),
                (87, 3, 3),
                (90, 1, 1),
            ],
        );
        assert_eq!(
            decode(w),
            Some(Plop3 { dst: [0, 7], src: [pred(3, true), pred(1, false), pred(2, true)], src2_uniform: true, lut: [0xa8, 0xc8] })
        );
    }

    #[test]
    fn predicate_registers() {
        let w = word(0x803, &[(16, 8, 0x1b), (24, 8, 0xff), (32, 32, 0x40)]);
        assert_eq!(decode(w), Some(P2r { dst: 0x1b, a: 0xff, mask: 0x40 }));
        let w = word(0x804, &[(24, 8, 0x11), (32, 32, 0x17)]);
        assert_eq!(decode(w), Some(R2p { src: 0x11, mask: 0x17 }));
    }

    #[test]
    fn redux() {
        let w = word(0x3c4, &[(16, 6, 8), (24, 8, 1), (73, 1, 1), (78, 3, 5)]);
        assert_eq!(decode(w), Some(Redux { dst: 8, src: 1, op: ReduxOp::Max, signed: true }));
        let w = word(0x3c4, &[(16, 6, 5), (24, 8, 5), (78, 3, 7)]);
        assert_eq!(decode(w), Some(Redux { dst: 5, src: 5, op: ReduxOp::Reserved(7), signed: false }));
    }

    #[test]
    fn uiadd3() {
        let w = word(
            0x290,
            &[(16, 6, 1), (24, 6, 2), (32, 6, 3), (63, 1, 1), (64, 6, 63), (72, 1, 1), (74, 1, 1), (75, 1, 1), (77, 3, 7), (80, 1, 1), (81, 3, 7), (84, 3, 6), (87, 3, 4)],
        );
        assert_eq!(
            decode(w),
            Some(UIAdd3 {
                dst: 1,
                a: Operand { neg: true, ..ur_op(2) },
                b: Operand { neg: true, ..ur_op(3) },
                c: Operand { neg: true, ..ur_op(63) },
                x: true,
                carry: [pred(4, false), pred(7, true)],
                overflow: [7, 6],
            })
        );
        let w = word(0x890, &[(16, 6, 4), (24, 6, 5), (32, 32, 0xffff_fff0), (64, 6, 63), (77, 3, 7), (80, 1, 1), (81, 3, 0), (84, 3, 7), (87, 3, 7), (90, 1, 1)]);
        assert_eq!(
            decode(w),
            Some(UIAdd3 {
                dst: 4,
                a: ur_op(5),
                b: imm_op(0xffff_fff0),
                c: ur_op(63),
                x: false,
                carry: [pred(7, true), pred(7, true)],
                overflow: [0, 7],
            })
        );
    }

    #[test]
    fn ulea() {
        let w = word(
            0x891,
            &[(16, 6, 4), (24, 6, 5), (32, 32, 0x100), (64, 6, 63), (72, 1, 1), (74, 1, 1), (75, 5, 3), (80, 1, 1), (81, 3, 2), (87, 3, 1), (90, 1, 1)],
        );
        assert_eq!(
            decode(w),
            Some(ULea { dst: 4, a: ur_op(5), b: imm_op(0x100), c: ur_op(63), neg_shifted: true, x: true, shift: 3, hi: true, overflow: 2, carry: pred(1, true) })
        );
    }

    #[test]
    fn ulop3() {
        let w = word(0x292, &[(16, 6, 1), (24, 6, 2), (32, 6, 3), (64, 6, 63), (72, 8, 0xc0), (80, 1, 1), (81, 3, 7), (87, 3, 7), (90, 1, 1)]);
        assert_eq!(
            decode(w),
            Some(ULop3 { dst: 1, a: ur_op(2), b: ur_op(3), c: ur_op(63), lut: 0xc0, pand: true, pdst: 7, psrc: pred(7, true) })
        );
    }

    #[test]
    fn ushf() {
        let w = word(0x899, &[(16, 6, 4), (24, 6, 5), (32, 32, 2), (64, 6, 6), (73, 2, 3), (75, 1, 1), (76, 1, 1), (80, 1, 1)]);
        assert_eq!(
            decode(w),
            Some(UShf { dst: 4, low: ur_op(5), shift: imm_op(2), high: ur_op(6), ty: ShfType::U32, wrap: true, right: true, hi: true })
        );
    }

    #[test]
    fn uisetp() {
        let w = word(0x88c, &[(24, 6, 4), (32, 32, 7), (68, 3, 7), (72, 1, 1), (73, 1, 1), (74, 2, 1), (76, 3, 6), (81, 3, 0), (84, 3, 7), (87, 3, 2), (90, 1, 1)]);
        assert_eq!(
            decode(w),
            Some(UISetP {
                dst: [0, 7],
                a: ur_op(4),
                b: imm_op(7),
                cmp: IntCmp::Ge,
                set_op: PredSetOp::Or,
                signed: true,
                ex: true,
                low_cmp: Pred::TRUE,
                accum: pred(2, true),
            })
        );
    }

    #[test]
    fn uflo() {
        let w = word(0x2bd, &[(16, 6, 4), (32, 6, 5), (63, 1, 1), (73, 1, 1), (74, 1, 1), (81, 3, 7), (91, 1, 1)]);
        assert_eq!(decode(w), Some(UFlo { dst: 4, src: ur_op(5), invert: true, signed: true, shift_amount: true, pdst: 7 }));
    }

    #[test]
    fn others_are_not_decoded() {
        for op in [0x002u16, 0x389, 0x210, 0x482, 0x9c3, 0x3c2, 0xb82, 0x006, 0x89c] {
            assert!(!OPCODES.contains(&op));
            assert_eq!(decode(word(op, &[])), None, "{op:#05x}");
        }
        for &op in OPCODES {
            assert!(decode(word(op, &[])).is_some(), "{op:#05x}");
            assert_ne!(explained(op), 0, "{op:#05x}");
        }
    }
}
