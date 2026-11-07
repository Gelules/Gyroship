//! FP32 family: arithmetic (FADD, FMUL, FFMA, FSWZADD), selection and
//! comparison (FSEL, FMNMX, FSET, FSETP), the multi-function unit (MUFU) and
//! the float/integer conversions and rounding (F2I, I2F, FRND).
//!
//! Contract for the integration:
//! - `pub enum Fp32Op` with typed variants (written from the encoding facts,
//!   not copied from Mesa);
//! - `pub fn decode(w: Word) -> Option<Fp32Op>`: Some for every owned opcode
//!   (12-bit, all forms), None for any other;
//! - `pub fn explained(opcode: u16) -> u128`: the bits decode() reads for
//!   that opcode; ALWAYS_EXPLAINED is added by the check;
//! - `pub const OPCODES: &[u16]`: the 12-bit opcodes the family owns.
//!
//! Sources: Mesa NAK `sm70_encode.rs` (MIT), the `encode()` of OpFAdd, OpFMul,
//! OpFFma, OpFSwzAdd, OpFMnMx, OpFSet, OpFSetP, OpMuFu, OpF2I, OpI2F, OpFRnd
//! and OpSel, the helpers `encode_alu_base` (operand forms), `set_rnd_mode`,
//! `set_float_cmp_op`, `set_pred_set_op`; checked against the game's samples.
//!
//! Except FSWZADD (whose 12-bit opcode 0x822 is fixed), every op uses the
//! generic ALU layout of `core::alu`: low 9 bits = operation, bits 9..12 =
//! operand form. In the game's code the operand slot an op does not use is
//! left at 0 (NAK writes RZ there); the decoder ignores it.

use super::prelude::*;

pub const OPCODES: &[u16] = &[
    0x220, 0x223, 0xa23, 0x221, 0x308, 0xa20, 0x423, 0x421, 0x820, 0x621, 0x823, 0x20b, 0x809, 0x209, 0x208, 0x307, 0x80b, 0x623,
    0x808, 0x822, 0xb08, 0x305, 0xa0b, 0xa09, 0x20a, 0x306, 0xa08, 0xa0a, 0x80a, 0xc20, 0xb07, 0xc23, 0xd06, 0xb05, 0xe21,
];

/// Rounding mode, 2 bits at 78..80 (NAK `set_rnd_mode`). For F2I and FRND
/// the same values are the round-to-integer modes: nearest-even, floor,
/// ceil, trunc.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Rnd {
    NearestEven,
    NegInf,
    PosInf,
    Zero,
}

/// Float comparison, 4 bits at 76..80 (NAK `set_float_cmp_op` gives 1..14;
/// 0 = always false and 15 = always true complete the ordered/unordered
/// pattern, NAK never emits them and the samples never use them). The `*u`
/// forms are unordered (also true when an operand is NaN).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FCmp {
    F,
    Lt,
    Eq,
    Le,
    Gt,
    Ne,
    Ge,
    Num,
    Nan,
    Ltu,
    Equ,
    Leu,
    Gtu,
    Neu,
    Geu,
    T,
}

/// How the comparison is combined with the accumulator predicate, 2 bits at
/// 74..76 (NAK `set_pred_set_op`: 0 and, 1 or, 2 xor; 3 is never emitted
/// nor seen and is kept as is).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BoolOp {
    And,
    Or,
    Xor,
    Reserved3,
}

/// One FSWZADD quad-lane operation (NAK `FSwzAddOp`, 0..3): add = a + b,
/// sub = a - b, subr = b - a (NAK's ddx/ddy lowering with a = the other
/// lane's value), mov2 (NAK MoveLeft; exact semantics not established).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SwzOp {
    Add,
    Sub,
    Subr,
    Mov2,
}

/// MUFU function, 6 bits at 74..80 (NAK `OpMuFu::encode`, values 0..9).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MufuFn {
    Cos,
    Sin,
    Ex2,
    Lg2,
    Rcp,
    Rsq,
    Rcp64h,
    Rsq64h,
    Sqrt,
    Tanh,
    Other(u8),
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Fp32Op {
    /// FADD d = a + b (operation 0x021, forms 1, 2, 3, 7; NAK OpFAdd). The
    /// second source sits in the slot at 32..: register (form 1, the b slot),
    /// immediate, constant or uniform register (forms 2, 3, 7, the c slot).
    /// sat 77, rnd 78..80, ftz 80.
    Fadd { dst: u8, a: Operand, b: Operand, sat: bool, rnd: Rnd, ftz: bool },
    /// FMUL d = a * b (0x020, forms 1, 4, 5, 6; NAK OpFMul): b in the b slot.
    /// dnz 76, sat 77, rnd 78..80, ftz 80. `scale` 84..87 is NAK's PDIV
    /// field: 4 = no scaling (NAK always writes 4, 99% of samples); the game
    /// also uses 2 and 3, whose power-of-two factors are not established.
    Fmul { dst: u8, a: Operand, b: Operand, sat: bool, rnd: Rnd, ftz: bool, dnz: bool, scale: u8 },
    /// FFMA d = a * b + c (0x023, forms 1..6; NAK OpFFma): the three generic
    /// slots. dnz 76, sat 77, rnd 78..80, ftz 80.
    Ffma { dst: u8, a: Operand, b: Operand, c: Operand, sat: bool, rnd: Rnd, ftz: bool, dnz: bool },
    /// FSWZADD (fixed 12-bit opcode 0x822; NAK OpFSwzAdd): per quad lane,
    /// d = op[lane](a, b). a register 24..32, b register 64..72, the four
    /// lane ops in 32..40 (lane 0 in bits 38..40, lane 3 in 32..34), ndv 77,
    /// rnd 78..80, ftz 80.
    Fswzadd { dst: u8, a: u8, b: u8, ops: [SwzOp; 4], ndv: bool, rnd: Rnd, ftz: bool },
    /// FSEL d = cond ? a : b (0x008, forms 1, 4, 5). NAK has no encoder:
    /// laid out like integer SEL 0x007 (NAK OpSel: condition 87..90, not
    /// 90); the samples use the float neg/abs bits of a and b.
    Fsel { dst: u8, a: Operand, b: Operand, cond: Pred },
    /// FMNMX d = min ? min(a, b) : max(a, b) (0x009, forms 1, 4, 5; NAK
    /// OpFMnMx): min predicate 87..90, not 90, ftz 80.
    Fmnmx { dst: u8, a: Operand, b: Operand, min: Pred, ftz: bool },
    /// FSET d = ((a cmp b) bop acc) ? 1.0 : 0.0 (0x00a, forms 1, 4, 5; NAK
    /// OpFSet, which nvdisasm prints as FSET.BF): cmp 76..80, ftz 80, acc
    /// 87..90, not 90. Assumed, not proved: `bop` at 74..76 and how `acc`
    /// combines come from FSETP's layout (NAK's FSET writes nothing at 74..76
    /// and forces acc to PT); 74..76 is 0 in every instruction of the game.
    Fset { dst: u8, a: Operand, b: Operand, cmp: FCmp, bop: BoolOp, acc: Pred, ftz: bool },
    /// FSETP p = (a cmp b) bop acc (0x00b, forms 1, 4, 5; NAK OpFSetP): bop
    /// 74..76, cmp 76..80, ftz 80, p 81..84, q 84..87 (second destination,
    /// NAK writes PT and so do all samples), acc 87..90, not 90.
    Fsetp { p: u8, q: u8, a: Operand, b: Operand, cmp: FCmp, bop: BoolOp, acc: Pred, ftz: bool },
    /// MUFU d = func(src) (0x108, forms 1, 5; NAK OpMuFu): src in the b slot;
    /// f16 73 (the SM 8.x branch), func 74..80. With f16 the source half is
    /// the b swizzle at 60..62 (0 low, 1 high).
    Mufu { dst: u8, func: MufuFn, src: Operand, f16: bool },
    /// F2I (0x105, forms 1, 5; NAK OpF2I): src in the b slot (half selected
    /// by the swizzle at 60..62 when src_bits is 16); signed 72, dst size
    /// 75..77, ntz 77, rnd 78..80, ftz 80, src size 84..86. Sizes are
    /// log2(bytes) in the word, decoded here as bit counts.
    F2i { dst: u8, src: Operand, signed: bool, dst_bits: u8, src_bits: u8, ntz: bool, rnd: Rnd, ftz: bool },
    /// I2F (0x106, forms 1, 6; NAK OpI2F): src in the b slot; `sel` 60..62
    /// (NAK "subop", always 0 there) picks the source byte or half: the
    /// samples use 0..2 with 8-bit sources and 0..1 with 16-bit ones, so the
    /// integer read is bits sel*src_bits.. of the register. signed 74, dst
    /// size 75..77, rnd 78..80, src size 84..86.
    I2f { dst: u8, src: Operand, signed: bool, dst_bits: u8, src_bits: u8, sel: u8, rnd: Rnd },
    /// FRND (0x107, forms 1, 5; NAK OpFRnd): round to an integral float with
    /// `rnd`. src in the b slot; dst size 75..77, rnd 78..80, ftz 80, src
    /// size 84..86.
    Frnd { dst: u8, src: Operand, dst_bits: u8, src_bits: u8, rnd: Rnd, ftz: bool },
}

fn rnd(w: Word) -> Rnd {
    [Rnd::NearestEven, Rnd::NegInf, Rnd::PosInf, Rnd::Zero][w.field(78, 2) as usize]
}

fn fcmp(w: Word) -> FCmp {
    use FCmp::*;
    [F, Lt, Eq, Le, Gt, Ne, Ge, Num, Nan, Ltu, Equ, Leu, Gtu, Neu, Geu, T][w.field(76, 4) as usize]
}

fn bool_op(w: Word) -> BoolOp {
    [BoolOp::And, BoolOp::Or, BoolOp::Xor, BoolOp::Reserved3][w.field(74, 2) as usize]
}

/// A 2-bit log2(bytes) size field as a bit count (8, 16, 32, 64).
fn size_bits(w: Word, lo: u32) -> u8 {
    8 << w.field(lo, 2)
}

fn mufu_fn(f: u8) -> MufuFn {
    use MufuFn::*;
    [Cos, Sin, Ex2, Lg2, Rcp, Rsq, Rcp64h, Rsq64h, Sqrt, Tanh].get(f as usize).copied().unwrap_or(Other(f))
}

pub fn decode(w: Word) -> Option<Fp32Op> {
    let op = w.opcode();
    if !OPCODES.contains(&op) {
        return None;
    }
    let dst = dst(w);
    let (sat, dnz, ftz) = (w.bit(77), w.bit(76), w.bit(80));
    let p87 = Pred::at(w, 87, 90);
    if op == 0x822 {
        let lane = |i: u32| [SwzOp::Add, SwzOp::Sub, SwzOp::Subr, SwzOp::Mov2][w.field(38 - 2 * i, 2) as usize];
        let ops = [lane(0), lane(1), lane(2), lane(3)];
        return Some(Fp32Op::Fswzadd { dst, a: reg(w, 24), b: reg(w, 64), ops, ndv: w.bit(77), rnd: rnd(w), ftz });
    }
    let x = alu(w, false)?;
    let (a, b, c) = (x.a, x.b, x.c);
    // Single-source ops with a half-float source read its swizzle (b slot).
    let half_src = |f16: bool| alu(w, f16).map(|h| h.b);
    Some(match op & 0x1ff {
        0x020 => Fp32Op::Fmul { dst, a, b, sat, rnd: rnd(w), ftz, dnz, scale: w.field(84, 3) as u8 },
        0x021 => Fp32Op::Fadd { dst, a, b: if x.form == 1 { b } else { c }, sat, rnd: rnd(w), ftz },
        0x023 => Fp32Op::Ffma { dst, a, b, c, sat, rnd: rnd(w), ftz, dnz },
        0x008 => Fp32Op::Fsel { dst, a, b, cond: p87 },
        0x009 => Fp32Op::Fmnmx { dst, a, b, min: p87, ftz },
        0x00a => Fp32Op::Fset { dst, a, b, cmp: fcmp(w), bop: bool_op(w), acc: p87, ftz },
        0x00b => Fp32Op::Fsetp { p: w.field(81, 3) as u8, q: w.field(84, 3) as u8, a, b, cmp: fcmp(w), bop: bool_op(w), acc: p87, ftz },
        0x108 => Fp32Op::Mufu { dst, func: mufu_fn(w.field(74, 6) as u8), src: half_src(w.bit(73))?, f16: w.bit(73) },
        0x105 => {
            let src_bits = size_bits(w, 84);
            let src = half_src(src_bits == 16)?;
            Fp32Op::F2i { dst, src, signed: w.bit(72), dst_bits: size_bits(w, 75), src_bits, ntz: w.bit(77), rnd: rnd(w), ftz }
        }
        0x106 => Fp32Op::I2f {
            dst,
            src: b,
            signed: w.bit(74),
            dst_bits: size_bits(w, 75),
            src_bits: size_bits(w, 84),
            sel: w.field(60, 2) as u8,
            rnd: rnd(w),
        },
        0x107 => {
            let src_bits = size_bits(w, 84);
            Fp32Op::Frnd { dst, src: half_src(src_bits == 16)?, dst_bits: size_bits(w, 75), src_bits, rnd: rnd(w), ftz }
        }
        _ => return None,
    })
}

// ---- explained bits ----

/// Source a: register 24..32, neg 72, abs 73.
const A: u128 = bits(24, 8) | bits(72, 2);
/// The register slot at 64..72 with its abs 74 and neg 75.
const LOW: u128 = bits(64, 8) | bits(74, 2);

/// The slot at 32.. for an operand form: register, immediate, constant
/// (with the uniform-bank bit 91) or uniform register, with abs 62 / neg 63.
fn mid(form: u16) -> u128 {
    match form {
        1 => bits(32, 8) | bits(62, 2),
        2 | 4 => bits(32, 32),
        3 | 5 => bits(32, 27) | bits(62, 2) | bits(91, 1),
        6 | 7 => bits(32, 6) | bits(62, 2),
        _ => 0,
    }
}

/// The generic b slot: the 64.. register in forms 2, 3, 7, else the 32.. slot.
fn b_slot(form: u16) -> u128 {
    if matches!(form, 2 | 3 | 7) { LOW } else { mid(form) }
}

pub fn explained(opcode: u16) -> u128 {
    if !OPCODES.contains(&opcode) {
        return 0;
    }
    let form = opcode >> 9;
    let d = bits(16, 8);
    let rnd_ftz = bits(78, 3);
    let p87 = bits(87, 4);
    let swizzle = bits(60, 2);
    match opcode & 0x1ff {
        0x022 => d | bits(24, 8) | bits(32, 8) | bits(64, 8) | bits(77, 4),
        0x020 => d | A | b_slot(form) | bits(76, 2) | rnd_ftz | bits(84, 3),
        0x021 => d | A | mid(form) | bits(77, 1) | rnd_ftz,
        0x023 => d | A | mid(form) | LOW | bits(76, 2) | rnd_ftz,
        0x008 => d | A | b_slot(form) | p87,
        0x009 => d | A | b_slot(form) | bits(80, 1) | p87,
        0x00a => d | A | b_slot(form) | bits(74, 7) | p87,
        0x00b => A | b_slot(form) | bits(74, 7) | bits(81, 6) | p87,
        0x108 => d | b_slot(form) | swizzle | bits(73, 7),
        0x105 => d | b_slot(form) | swizzle | bits(72, 1) | bits(75, 6) | bits(84, 2),
        0x106 => d | b_slot(form) | swizzle | bits(74, 6) | bits(84, 2),
        0x107 => d | b_slot(form) | swizzle | bits(75, 6) | bits(84, 2),
        _ => 0,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A word with the given opcode, guard PT and fields (lo, len, value).
    fn word(op: u16, fields: &[(u32, u32, u64)]) -> Word {
        let mut v = op as u128 | (7u128 << 12);
        for &(lo, len, x) in fields {
            v |= (x as u128 & ((1u128 << len) - 1)) << lo;
        }
        Word(v)
    }

    fn op(src: Src, neg: bool, abs: bool) -> Operand {
        Operand { src, neg, abs, swizzle: 0 }
    }

    fn r(n: u8) -> Operand {
        Operand::plain(Src::Reg(n))
    }

    const IMM: u64 = 0x3f80_0000;
    const D: (u32, u32, u64) = (16, 8, 0x11);
    const RA: (u32, u32, u64) = (24, 8, 0x22);
    const RB: (u32, u32, u64) = (32, 8, 0x33);
    const RC: (u32, u32, u64) = (64, 8, 0x44);
    const IM: (u32, u32, u64) = (32, 32, IMM);
    const CB: [(u32, u32, u64); 2] = [(38, 16, 0x1234), (54, 5, 3)];
    const UR: [(u32, u32, u64); 2] = [(32, 6, 5), (91, 1, 1)];
    fn cb() -> Src {
        Src::CBuf { bank: 3, offset: 0x1234 }
    }

    fn cases() -> Vec<(Word, Fp32Op)> {
        use Fp32Op::*;
        let ra = r(0x22);
        let imm = Operand::plain(Src::Imm(IMM as u32));
        let ur = Operand::plain(Src::UReg(5));
        let p = |reg, not| Pred { reg, not };
        vec![
            // FMUL, forms 1, 5, 4, 6
            (
                word(0x220, &[D, RA, RB, (72, 1, 1), (62, 1, 1), (76, 1, 1), (77, 1, 1), (78, 2, 1), (80, 1, 1), (84, 3, 4)]),
                Fmul { dst: 0x11, a: op(Src::Reg(0x22), true, false), b: op(Src::Reg(0x33), false, true), sat: true, rnd: Rnd::NegInf, ftz: true, dnz: true, scale: 4 },
            ),
            (
                word(0xa20, &[D, RA, CB[0], CB[1], (63, 1, 1), (78, 2, 3), (84, 3, 2)]),
                Fmul { dst: 0x11, a: ra, b: op(cb(), true, false), sat: false, rnd: Rnd::Zero, ftz: false, dnz: false, scale: 2 },
            ),
            (word(0x820, &[D, RA, IM, (84, 3, 4)]), Fmul { dst: 0x11, a: ra, b: imm, sat: false, rnd: Rnd::NearestEven, ftz: false, dnz: false, scale: 4 }),
            (word(0xc20, &[D, RA, UR[0], UR[1], (73, 1, 1)]), Fmul { dst: 0x11, a: op(Src::Reg(0x22), false, true), b: ur, sat: false, rnd: Rnd::NearestEven, ftz: false, dnz: false, scale: 0 }),
            // FADD, forms 1, 2, 3, 7
            (
                word(0x221, &[D, RA, RB, (63, 1, 1), (73, 1, 1), (78, 2, 2), (80, 1, 1)]),
                Fadd { dst: 0x11, a: op(Src::Reg(0x22), false, true), b: op(Src::Reg(0x33), true, false), sat: false, rnd: Rnd::PosInf, ftz: true },
            ),
            (word(0x421, &[D, RA, IM, (77, 1, 1)]), Fadd { dst: 0x11, a: ra, b: imm, sat: true, rnd: Rnd::NearestEven, ftz: false }),
            (word(0x621, &[D, RA, CB[0], CB[1], (63, 1, 1)]), Fadd { dst: 0x11, a: ra, b: op(cb(), true, false), sat: false, rnd: Rnd::NearestEven, ftz: false }),
            (word(0xe21, &[D, RA, UR[0], UR[1], (62, 1, 1)]), Fadd { dst: 0x11, a: ra, b: op(Src::UReg(5), false, true), sat: false, rnd: Rnd::NearestEven, ftz: false }),
            // FFMA, forms 1..6
            (
                word(0x223, &[D, RA, RB, RC, (74, 1, 1), (75, 1, 1), (76, 1, 1), (77, 1, 1), (78, 2, 3), (80, 1, 1)]),
                Ffma { dst: 0x11, a: ra, b: r(0x33), c: op(Src::Reg(0x44), true, true), sat: true, rnd: Rnd::Zero, ftz: true, dnz: true },
            ),
            (word(0x423, &[D, RA, RC, IM, (75, 1, 1)]), Ffma { dst: 0x11, a: ra, b: op(Src::Reg(0x44), true, false), c: imm, sat: false, rnd: Rnd::NearestEven, ftz: false, dnz: false }),
            (word(0x623, &[D, RA, RC, CB[0], CB[1], (63, 1, 1)]), Ffma { dst: 0x11, a: ra, b: r(0x44), c: op(cb(), true, false), sat: false, rnd: Rnd::NearestEven, ftz: false, dnz: false }),
            (word(0x823, &[D, RA, IM, RC, (72, 1, 1)]), Ffma { dst: 0x11, a: op(Src::Reg(0x22), true, false), b: imm, c: r(0x44), sat: false, rnd: Rnd::NearestEven, ftz: false, dnz: false }),
            (word(0xa23, &[D, RA, CB[0], CB[1], RC, (62, 1, 1)]), Ffma { dst: 0x11, a: ra, b: op(cb(), false, true), c: r(0x44), sat: false, rnd: Rnd::NearestEven, ftz: false, dnz: false }),
            (word(0xc23, &[D, RA, UR[0], UR[1], RC]), Ffma { dst: 0x11, a: ra, b: ur, c: r(0x44), sat: false, rnd: Rnd::NearestEven, ftz: false, dnz: false }),
            // FSWZADD: 0x66 = sub, subr, sub, subr (ddx); 0xc4 = mov2, add, sub, add
            (
                word(0x822, &[D, RA, RC, (32, 8, 0x66)]),
                Fswzadd { dst: 0x11, a: 0x22, b: 0x44, ops: [SwzOp::Sub, SwzOp::Subr, SwzOp::Sub, SwzOp::Subr], ndv: false, rnd: Rnd::NearestEven, ftz: false },
            ),
            (
                word(0x822, &[D, RA, RC, (32, 8, 0xc4), (77, 1, 1), (78, 2, 1), (80, 1, 1)]),
                Fswzadd { dst: 0x11, a: 0x22, b: 0x44, ops: [SwzOp::Mov2, SwzOp::Add, SwzOp::Sub, SwzOp::Add], ndv: true, rnd: Rnd::NegInf, ftz: true },
            ),
            // FSEL, forms 1, 4, 5
            (word(0x208, &[D, RA, RB, (72, 1, 1), (87, 3, 3), (90, 1, 1)]), Fsel { dst: 0x11, a: op(Src::Reg(0x22), true, false), b: r(0x33), cond: p(3, true) }),
            (word(0x808, &[D, RA, IM, (87, 3, 1)]), Fsel { dst: 0x11, a: ra, b: imm, cond: p(1, false) }),
            (word(0xa08, &[D, RA, CB[0], CB[1], (63, 1, 1), (87, 3, 7)]), Fsel { dst: 0x11, a: ra, b: op(cb(), true, false), cond: Pred::TRUE }),
            // FMNMX, forms 1, 4, 5
            (word(0x209, &[D, RA, RB, (80, 1, 1), (87, 3, 7)]), Fmnmx { dst: 0x11, a: ra, b: r(0x33), min: Pred::TRUE, ftz: true }),
            (word(0x809, &[D, RA, IM, (87, 3, 7), (90, 1, 1)]), Fmnmx { dst: 0x11, a: ra, b: imm, min: p(7, true), ftz: false }),
            (word(0xa09, &[D, RA, CB[0], CB[1], (73, 1, 1), (87, 3, 2)]), Fmnmx { dst: 0x11, a: op(Src::Reg(0x22), false, true), b: Operand::plain(cb()), min: p(2, false), ftz: false }),
            // FSET, forms 1, 4, 5
            (
                word(0x20a, &[D, RA, RB, (74, 2, 1), (76, 4, 11), (80, 1, 1), (87, 3, 2)]),
                Fset { dst: 0x11, a: ra, b: r(0x33), cmp: FCmp::Leu, bop: BoolOp::Or, acc: p(2, false), ftz: true },
            ),
            (word(0x80a, &[D, RA, IM, (76, 4, 9), (87, 3, 7)]), Fset { dst: 0x11, a: ra, b: imm, cmp: FCmp::Ltu, bop: BoolOp::And, acc: Pred::TRUE, ftz: false }),
            (word(0xa0a, &[D, RA, CB[0], CB[1], (76, 4, 14), (87, 3, 7)]), Fset { dst: 0x11, a: ra, b: Operand::plain(cb()), cmp: FCmp::Geu, bop: BoolOp::And, acc: Pred::TRUE, ftz: false }),
            // FSETP, forms 1, 4, 5
            (
                word(0x20b, &[RA, RB, (74, 2, 2), (76, 4, 5), (80, 1, 1), (81, 3, 1), (84, 3, 7), (87, 3, 4), (90, 1, 1)]),
                Fsetp { p: 1, q: 7, a: ra, b: r(0x33), cmp: FCmp::Ne, bop: BoolOp::Xor, acc: p(4, true), ftz: true },
            ),
            (word(0x80b, &[RA, IM, (76, 4, 1), (81, 3, 0), (84, 3, 7), (87, 3, 7)]), Fsetp { p: 0, q: 7, a: ra, b: imm, cmp: FCmp::Lt, bop: BoolOp::And, acc: Pred::TRUE, ftz: false }),
            (
                word(0xa0b, &[RA, CB[0], CB[1], (74, 2, 1), (76, 4, 12), (81, 3, 6), (84, 3, 7), (87, 3, 7)]),
                Fsetp { p: 6, q: 7, a: ra, b: Operand::plain(cb()), cmp: FCmp::Gtu, bop: BoolOp::Or, acc: Pred::TRUE, ftz: false },
            ),
            // MUFU, forms 1, 5
            (word(0x308, &[D, RB, (63, 1, 1), (74, 6, 4)]), Mufu { dst: 0x11, func: MufuFn::Rcp, src: op(Src::Reg(0x33), true, false), f16: false }),
            (word(0x308, &[D, RB, (60, 2, 1), (73, 1, 1), (74, 6, 1)]), Mufu { dst: 0x11, func: MufuFn::Sin, src: Operand { swizzle: 1, ..r(0x33) }, f16: true }),
            (word(0x308, &[D, RB, (74, 6, 12)]), Mufu { dst: 0x11, func: MufuFn::Other(12), src: r(0x33), f16: false }),
            (word(0xb08, &[D, CB[0], CB[1], (74, 6, 3)]), Mufu { dst: 0x11, func: MufuFn::Lg2, src: Operand::plain(cb()), f16: false }),
            // F2I, forms 1, 5
            (
                word(0x305, &[D, RB, (60, 2, 1), (72, 1, 1), (75, 2, 2), (77, 1, 1), (78, 2, 3), (80, 1, 1), (84, 2, 1)]),
                F2i { dst: 0x11, src: Operand { swizzle: 1, ..r(0x33) }, signed: true, dst_bits: 32, src_bits: 16, ntz: true, rnd: Rnd::Zero, ftz: true },
            ),
            (
                word(0xb05, &[D, CB[0], CB[1], (75, 2, 1), (78, 2, 1), (84, 2, 2)]),
                F2i { dst: 0x11, src: Operand::plain(cb()), signed: false, dst_bits: 16, src_bits: 32, ntz: false, rnd: Rnd::NegInf, ftz: false },
            ),
            // I2F, forms 1, 6
            (
                word(0x306, &[D, RB, (60, 2, 2), (74, 1, 1), (75, 2, 2), (78, 2, 2)]),
                I2f { dst: 0x11, src: r(0x33), signed: true, dst_bits: 32, src_bits: 8, sel: 2, rnd: Rnd::PosInf },
            ),
            (word(0xd06, &[D, UR[0], UR[1], (60, 2, 1), (75, 2, 2), (84, 2, 1)]), I2f { dst: 0x11, src: ur, signed: false, dst_bits: 32, src_bits: 16, sel: 1, rnd: Rnd::NearestEven }),
            // FRND, forms 1, 5
            (word(0x307, &[D, RB, (75, 2, 2), (78, 2, 1), (80, 1, 1), (84, 2, 2)]), Frnd { dst: 0x11, src: r(0x33), dst_bits: 32, src_bits: 32, rnd: Rnd::NegInf, ftz: true }),
            (word(0xb07, &[D, CB[0], CB[1], (63, 1, 1), (75, 2, 2), (78, 2, 2), (84, 2, 2)]), Frnd { dst: 0x11, src: op(cb(), true, false), dst_bits: 32, src_bits: 32, rnd: Rnd::PosInf, ftz: false }),
        ]
    }

    #[test]
    fn synthetic() {
        let cases = cases();
        for (w, want) in &cases {
            assert_eq!(decode(*w), Some(*want), "{:#05x}", w.opcode());
        }
        for &op in OPCODES {
            assert!(cases.iter().any(|(w, _)| w.opcode() == op), "no synthetic case for {op:#05x}");
        }
        for op in [0x207, 0x222, 0x224, 0x104, 0x109, 0x420, 0x021, 0x000] {
            assert_eq!(decode(word(op, &[])), None, "{op:#05x}");
        }
    }

    /// The explained mask is complete: flipping any bit outside it does not
    /// change the decoding (synthetic words; the inventory checks real code).
    #[test]
    fn explained_is_complete() {
        let words: Vec<Word> = cases().iter().map(|c| c.0).collect();
        for w in words {
            let keep = explained(w.opcode()) | ALWAYS_EXPLAINED;
            for i in (0..128).filter(|i| keep >> i & 1 == 0) {
                assert_eq!(decode(Word(w.0 ^ 1 << i)), decode(w), "{:#05x} bit {i}", w.opcode());
            }
        }
    }
}
