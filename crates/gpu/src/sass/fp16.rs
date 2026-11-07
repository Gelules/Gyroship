//! The "fp16" family: packed half-float ALU (HADD2, HFMA2, HMUL2, HSETP2), the
//! conversions next to them (F2FP, I2FP), and UIMAD, which sits in the 0x0a4
//! slot that was suspected to be fp16 but is the uniform integer multiply-add.
//!
//! Contract for the integration:
//! - `pub enum Fp16Op` with typed variants (our own code, written from the
//!   encoding facts read in Mesa NAK's `sm70_encode.rs`);
//! - `pub fn decode(w: Word) -> Option<Fp16Op>`: Some for every opcode in
//!   `OPCODES`, None for any other;
//! - `pub fn explained(opcode: u16) -> u128`: the bits the decoder reads;
//! - `pub const OPCODES: &[u16]`: the 12-bit opcodes the family owns.
//!
//! Operand layout (NAK `encode_alu_base` / `encode_fp16v2_alu`, generic ALU
//! forms decoded by `core::alu`): for the packed-half ops each register or
//! constant operand carries a 2-bit lane selector (see `SWZ_*`), a at 74..76,
//! the slot at 32.. at 60..62, the register at 64.. at 81..83; neg/abs for the
//! slot at 64.. are 84/83 instead of 75/74.

use super::prelude::*;

pub const OPCODES: &[u16] = &[0x230, 0x245, 0x234, 0x8a4, 0x23e, 0x231, 0x232, 0xa45, 0xc45, 0x2a4, 0x4a4, 0x630, 0xa3e, 0x430, 0x832, 0x434, 0x240, 0x23c, 0x23a];

/// Lane selector values of a packed-half operand (`Operand::swizzle`), from
/// NAK `set_swizzle` for F16v2 sources: 0 keeps both lanes, 2 broadcasts the
/// low half, 3 the high half.
pub const SWZ_H1_H0: u8 = 0;
/// Value 1 is never emitted by NAK for F16v2; nvdisasm prints it as `.F32`
/// (the operand is a full f32). Not seen in the game's samples.
pub const SWZ_F32: u8 = 1;
pub const SWZ_H0_H0: u8 = 2;
pub const SWZ_H1_H1: u8 = 3;

/// Float comparison, 4 bits (NAK `set_float_cmp_op`). 0 and 15 are not
/// emitted by NAK; they are the always-false / always-true tests of the
/// FSETP family (assumed, not seen in the samples).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FloatCmp {
    False,
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
    True,
}

impl FloatCmp {
    fn from_bits(v: u64) -> FloatCmp {
        use FloatCmp::*;
        [False, Lt, Eq, Le, Gt, Ne, Ge, Num, Nan, Ltu, Equ, Leu, Gtu, Neu, Geu, True][v as usize & 15]
    }
}

/// How a compare result is combined with the accumulator predicate, 2 bits
/// (NAK `set_pred_set_op`); 3 is not emitted by NAK.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BoolOp {
    And,
    Or,
    Xor,
    Reserved3,
}

impl BoolOp {
    fn from_bits(v: u64) -> BoolOp {
        [BoolOp::And, BoolOp::Or, BoolOp::Xor, BoolOp::Reserved3][v as usize & 3]
    }
}

/// Float rounding mode, 2 bits (NAK `set_rnd_mode`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Round {
    NearestEven,
    NegInf,
    PosInf,
    Zero,
}

impl Round {
    fn at(w: Word, lo: u32) -> Round {
        [Round::NearestEven, Round::NegInf, Round::PosInf, Round::Zero][w.field(lo, 2) as usize]
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Fp16Op {
    /// HADD2 (0x030), per lane `dst = a + b`. NAK `OpHAdd2::encode`: the
    /// second source is in the slot at 32.. when it is a register (form 1),
    /// else in the "c" position (forms 2 imm, 3 cbuf, 7 ureg). sat 77,
    /// f32 78 (one f32 result instead of two halves), ftz 80.
    Hadd2 { dst: u8, a: Operand, b: Operand, sat: bool, f32: bool, ftz: bool },
    /// HFMA2 (0x031), per lane `dst = a * b + c`. NAK `OpHFma2::encode`:
    /// all three ALU slots; dnz 76, sat 77, f32 78, relu 79 (SM86+), ftz 80,
    /// bf16 85 (SM86+). `bit86` is not written by NAK: it is set in 2 of 20
    /// samples, always with every lane selector at H1_H0; probably the
    /// `.MMA` variant (same math, issued to the tensor pipe), unconfirmed.
    Hfma2 { dst: u8, a: Operand, b: Operand, c: Operand, dnz: bool, sat: bool, f32: bool, relu: bool, ftz: bool, bf16: bool, bit86: bool },
    /// HMUL2 (0x032), per lane `dst = a * b`. NAK `OpHMul2::encode`: b in
    /// the slot at 32.. (forms 1, 4, 5, 6); dnz 76, sat 77, relu 79 (SM86+),
    /// ftz 80. Bit 78 (.F32) is SM70-75 only and not read.
    Hmul2 { dst: u8, a: Operand, b: Operand, dnz: bool, sat: bool, relu: bool, ftz: bool },
    /// HSETP2 (0x034): compare both lanes, write predicates. NAK
    /// `OpHSetP2::encode`: b placed like HADD2's; set op 69..71, h_and 71
    /// (dst0 = both lanes true, dst1 = its negation), cmp 76..80, ftz 80,
    /// dst0 81..84, dst1 84..87 (7 = PT), accumulator 87..90 with not 90.
    Hsetp2 { dst0: u8, dst1: u8, a: Operand, b: Operand, cmp: FloatCmp, bop: BoolOp, accum: Pred, h_and: bool, ftz: bool },
    /// F2FP.PACK_AB (0x03e): `dst = half(a) << 16 | half(b)` (a goes to the
    /// high lane, from NAK's f2f16 lowering). NAK `OpF2FP::encode`, generic
    /// (non-fp16) ALU layout with c = RZ: c is the MERGE_C source, h1 72
    /// (merge from c's high half, MERGE_C only, so a's neg bit is not read),
    /// merge_c 78, rounding 79..81.
    F2fp { dst: u8, a: Operand, b: Operand, c: u8, h1: bool, merge_c: bool, rnd: Round },
    /// I2FP (0x045): integer to float conversion, the SM86+ variant of I2F.
    /// Not encoded by NAK: identified because its modifiers match NAK's
    /// `OpI2F::encode` exactly (signed 74, dst size 75..77, rounding 78..80,
    /// src size 84..86, source in the slot at 32.., a unused) and the game
    /// only uses it as S32/U32 -> F32. Sizes are in bits.
    I2fp { dst: u8, src: Operand, signed: bool, dst_bits: u8, src_bits: u8, rnd: Round },
    /// UIMAD (0x0a4): uniform `dst = a * b + c` on uniform registers. NAK
    /// `OpIMad::encode` (uniform branch, `encode_ualu`): 6-bit uniform
    /// registers dst 16, a 24, the slot at 32.. (form 1) and the one at 64..
    /// (b in form 2, c in forms 1 and 4), immediates at 32..64; neg bits 72,
    /// 63, 75; signed 73; carry-out uniform predicate 81..84. Assumed, not
    /// proved: `x` (74) and `carry_in` (87..90, not 90) are not in NAK's IMAD
    /// encoder and come from its IADD3.X layout; in the game 74 is always 0
    /// and 87..91 always !UPT.
    Uimad { dst: u8, a: Operand, b: Operand, c: Operand, signed: bool, x: bool, carry_out: u8, carry_in: Pred },
    /// HMNMX2 (0x040, SM 8.x): per lane min (`min` true) or max of a and b.
    /// NAK `OpHMnMx2::encode`: the packed-half ALU layout of HADD2; ftz 80,
    /// the min/max predicate 87..90 with not 90. Bits 78 (.F32), 81 (.NAN),
    /// 82 (.XORSIGN) and 85 (.BF16_V2) are kept to refuse those forms.
    Hmnmx2 { dst: u8, a: Operand, b: Operand, min: Pred, ftz: bool, f32: bool, nan: bool, xorsign: bool, bf16: bool },
    /// HMMA (0x03c): a warp's matrix multiply-add D = A × B + C on half
    /// floats, register fragments as PTX `mma.m16n8k{4,8,16}`. NAK
    /// `OpHmma::encode`: dst 16, A 24, B 32, C 64 (RZ: no C); the shape in
    /// bits 75 (low) and 78 (high): 0 = m16n8k8, 1 = m16n8k16, 2 = m16n8k4;
    /// F32 accumulators 76; source type 82..84 (0 = F16).
    Hmma { dst: u8, a: u8, b: u8, c: u8, k: u8, f32: bool, src_type: u8 },
    /// MOVM (0x03a): transpose an 8×8 matrix of 16-bit values held as one
    /// register per lane (lane t: row t / 4, columns 2 (t % 4) and + 1).
    /// NAK `OpMovm::encode`: dst 16, source 24; shape 78..80 (0 = MT88).
    Movm { dst: u8, src: u8, shape: u8 },
}

/// The second source of a two-operand op whose encoder puts a non-register
/// operand in the "c" position (forms 2, 3, 7) and a register at 32.. (form 1).
fn second(alu: &Alu) -> Operand {
    match alu.form {
        2 | 3 | 7 => alu.c,
        _ => alu.b,
    }
}

fn uniform(w: Word, lo: u32, neg_bit: u32) -> Operand {
    Operand { src: Src::UReg(ureg(w, lo)), neg: w.bit(neg_bit), abs: false, swizzle: 0 }
}

pub fn decode(w: Word) -> Option<Fp16Op> {
    if !OPCODES.contains(&w.opcode()) {
        return None;
    }
    let half = alu(w, true)?;
    let on = |i| w.bit(i);
    Some(match w.opcode() & 0x1ff {
        0x030 => Fp16Op::Hadd2 { dst: dst(w), a: half.a, b: second(&half), sat: on(77), f32: on(78), ftz: on(80) },
        0x031 => Fp16Op::Hfma2 {
            dst: dst(w),
            a: half.a,
            b: half.b,
            c: half.c,
            dnz: on(76),
            sat: on(77),
            f32: on(78),
            relu: on(79),
            ftz: on(80),
            bf16: on(85),
            bit86: on(86),
        },
        0x032 => Fp16Op::Hmul2 { dst: dst(w), a: half.a, b: half.b, dnz: on(76), sat: on(77), relu: on(79), ftz: on(80) },
        0x034 => Fp16Op::Hsetp2 {
            dst0: w.field(81, 3) as u8,
            dst1: w.field(84, 3) as u8,
            a: half.a,
            b: second(&half),
            cmp: FloatCmp::from_bits(w.field(76, 4)),
            bop: BoolOp::from_bits(w.field(69, 2)),
            accum: Pred::at(w, 87, 90),
            h_and: on(71),
            ftz: on(80),
        },
        0x03e => {
            let full = alu(w, false)?;
            Fp16Op::F2fp { dst: dst(w), a: Operand { neg: false, ..full.a }, b: full.b, c: reg(w, 64), h1: on(72), merge_c: on(78), rnd: Round::at(w, 79) }
        }
        0x040 => Fp16Op::Hmnmx2 { dst: dst(w), a: half.a, b: second(&half), min: Pred::at(w, 87, 90), ftz: on(80), f32: on(78), nan: on(81), xorsign: on(82), bf16: on(85) },
        0x03c => {
            let k = match w.field(75, 1) | w.field(78, 1) << 1 {
                0 => 8,
                1 => 16,
                2 => 4,
                _ => 0,
            };
            Fp16Op::Hmma { dst: dst(w), a: reg(w, 24), b: reg(w, 32), c: reg(w, 64), k, f32: on(76), src_type: w.field(82, 2) as u8 }
        }
        0x03a => Fp16Op::Movm { dst: dst(w), src: reg(w, 24), shape: w.field(78, 2) as u8 },
        0x045 => Fp16Op::I2fp {
            dst: dst(w),
            src: alu(w, false)?.b,
            signed: on(74),
            dst_bits: 8 << w.field(75, 2) as u8,
            src_bits: 8 << w.field(84, 2) as u8,
            rnd: Round::at(w, 78),
        },
        _ => {
            // 0x0a4, forms 1, 2, 4 (the only owned ones).
            let imm = Operand::plain(Src::Imm(w.field(32, 32) as u32));
            let low = uniform(w, 64, 75);
            let (b, c) = match half.form {
                1 => (uniform(w, 32, 63), low),
                2 => (low, imm),
                _ => (imm, low),
            };
            Fp16Op::Uimad { dst: ureg(w, 16), a: uniform(w, 24, 72), b, c, signed: on(73), x: on(74), carry_out: w.field(81, 3) as u8, carry_in: Pred::at(w, 87, 90) }
        }
    })
}

const fn bit(i: u32) -> u128 {
    1 << i
}

/// Register a with neg/abs (and lane selector for packed halves).
const fn a_bits(fp16: bool) -> u128 {
    bits(24, 8) | bits(72, 2) | if fp16 { bits(74, 2) } else { 0 }
}

/// The slot at 32..64 for each form, as `core::alu` reads it.
const fn mid_bits(form: u16, fp16: bool) -> u128 {
    let mods = bits(62, 2) | if fp16 { bits(60, 2) } else { 0 };
    match form {
        1 => bits(32, 8) | mods,
        2 | 4 => bits(32, 32),
        3 | 5 => bits(32, 27) | bit(91) | mods,
        _ => bits(32, 6) | mods,
    }
}

/// The register at 64.. with its modifiers.
const fn low_bits(fp16: bool) -> u128 {
    bits(64, 8) | if fp16 { bits(81, 4) } else { bits(74, 2) }
}

pub fn explained(opcode: u16) -> u128 {
    if !OPCODES.contains(&opcode) {
        return 0;
    }
    let form = opcode >> 9;
    let dst = bits(16, 8);
    match opcode & 0x1ff {
        0x030 => dst | a_bits(true) | mid_bits(form, true) | bits(77, 2) | bit(80),
        0x031 => dst | a_bits(true) | mid_bits(form, true) | low_bits(true) | bits(76, 5) | bits(85, 2),
        0x032 => dst | a_bits(true) | mid_bits(form, true) | bits(76, 2) | bits(79, 2),
        0x034 => a_bits(true) | mid_bits(form, true) | bits(69, 3) | bits(76, 15),
        0x03e => dst | bits(24, 8) | bits(72, 2) | mid_bits(form, false) | bits(64, 8) | bits(78, 3),
        0x045 => dst | mid_bits(form, false) | bits(74, 6) | bits(84, 2),
        0x040 => dst | a_bits(true) | mid_bits(form, true) | bit(78) | bits(80, 3) | bit(85) | bits(87, 4),
        0x03c => dst | bits(24, 8) | bits(32, 8) | bits(64, 8) | bits(75, 2) | bit(78) | bits(82, 2),
        0x03a => dst | bits(24, 8) | bits(78, 2),
        _ => {
            let b = if form == 1 { bits(32, 6) | bit(63) } else { bits(32, 32) };
            bits(16, 6) | bits(24, 6) | bits(72, 4) | b | bits(64, 6) | bits(81, 3) | bits(87, 4)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A word from (lo, value) fields, with the guard set to PT.
    fn word(op: u16, fields: &[(u32, u128)]) -> Word {
        Word(fields.iter().fold(op as u128 | 7 << 12, |acc, &(lo, v)| acc | v << lo))
    }

    fn reg_op(r: u8, neg: bool, abs: bool, swizzle: u8) -> Operand {
        Operand { src: Src::Reg(r), neg, abs, swizzle }
    }

    #[test]
    fn hadd2_reg() {
        // HADD2.F32.SAT.FTZ R5, -RZ.H0_H0, R7.H1_H1
        let w = word(0x230, &[(16, 5), (24, 0xff), (72, 1), (74, 2), (32, 7), (60, 3), (77, 1), (78, 1), (80, 1), (64, 9)]);
        assert_eq!(
            decode(w),
            Some(Fp16Op::Hadd2 { dst: 5, a: reg_op(RZ, true, false, SWZ_H0_H0), b: reg_op(7, false, false, SWZ_H1_H1), sat: true, f32: true, ftz: true })
        );
    }

    #[test]
    fn hadd2_imm_and_cbuf() {
        let w = word(0x430, &[(16, 2), (24, 3), (73, 1), (74, 3), (32, 0x3c00_4000)]);
        assert_eq!(
            decode(w),
            Some(Fp16Op::Hadd2 { dst: 2, a: reg_op(3, false, true, SWZ_H1_H1), b: Operand::plain(Src::Imm(0x3c00_4000)), sat: false, f32: false, ftz: false })
        );
        let w = word(0x630, &[(16, 4), (24, 0xff), (38, 0x1a0), (54, 3), (60, 2), (63, 1), (78, 1)]);
        let Some(Fp16Op::Hadd2 { dst: 4, b, f32: true, .. }) = decode(w) else { panic!() };
        assert_eq!(b, Operand { src: Src::CBuf { bank: 3, offset: 0x1a0 }, neg: true, abs: false, swizzle: SWZ_H0_H0 });
    }

    #[test]
    fn hfma2() {
        let w = word(0x231, &[(16, 9), (24, 1), (32, 2), (60, 2), (63, 1), (64, 3), (81, 3), (83, 1), (84, 1), (76, 1), (79, 1), (85, 1), (86, 1)]);
        assert_eq!(
            decode(w),
            Some(Fp16Op::Hfma2 {
                dst: 9,
                a: reg_op(1, false, false, SWZ_H1_H0),
                b: reg_op(2, true, false, SWZ_H0_H0),
                c: reg_op(3, true, true, SWZ_H1_H1),
                dnz: true,
                sat: false,
                f32: false,
                relu: true,
                ftz: false,
                bf16: true,
                bit86: true,
            })
        );
    }

    #[test]
    fn hmul2() {
        let w = word(0x232, &[(16, 1), (24, 4), (74, 3), (32, 5), (62, 1), (77, 1), (80, 1)]);
        assert_eq!(
            decode(w),
            Some(Fp16Op::Hmul2 { dst: 1, a: reg_op(4, false, false, SWZ_H1_H1), b: reg_op(5, false, true, 0), dnz: false, sat: true, relu: false, ftz: true })
        );
        let w = word(0x832, &[(16, 1), (24, 4), (32, 0x3800_3800), (76, 1)]);
        let Some(Fp16Op::Hmul2 { b, dnz: true, .. }) = decode(w) else { panic!() };
        assert_eq!(b.src, Src::Imm(0x3800_3800));
    }

    #[test]
    fn hsetp2() {
        // HSETP2.GTU.XOR.H_AND.FTZ P1, P2, R6.H0_H0, R8, !P3
        let w = word(0x234, &[(24, 6), (74, 2), (32, 8), (69, 2), (71, 1), (76, 0xc), (80, 1), (81, 1), (84, 2), (87, 3), (90, 1)]);
        assert_eq!(
            decode(w),
            Some(Fp16Op::Hsetp2 {
                dst0: 1,
                dst1: 2,
                a: reg_op(6, false, false, SWZ_H0_H0),
                b: reg_op(8, false, false, 0),
                cmp: FloatCmp::Gtu,
                bop: BoolOp::Xor,
                accum: Pred { reg: 3, not: true },
                h_and: true,
                ftz: true,
            })
        );
        let w = word(0x434, &[(24, 6), (32, 0x0000_3c00), (76, 4), (81, 0), (84, 7), (87, 7)]);
        let Some(Fp16Op::Hsetp2 { b, cmp: FloatCmp::Gt, dst0: 0, dst1: PT, accum: Pred::TRUE, bop: BoolOp::And, .. }) = decode(w) else { panic!() };
        assert_eq!(b.src, Src::Imm(0x3c00));
    }

    #[test]
    fn f2fp() {
        let w = word(0x23e, &[(16, 3), (24, 10), (73, 1), (32, 11), (63, 1), (64, 0xff), (72, 1), (78, 1), (79, 3)]);
        assert_eq!(
            decode(w),
            Some(Fp16Op::F2fp { dst: 3, a: reg_op(10, false, true, 0), b: reg_op(11, true, false, 0), c: RZ, h1: true, merge_c: true, rnd: Round::Zero })
        );
        let w = word(0xa3e, &[(16, 3), (24, 10), (38, 0x40), (54, 2), (64, 0xff)]);
        let Some(Fp16Op::F2fp { b, rnd: Round::NearestEven, h1: false, merge_c: false, .. }) = decode(w) else { panic!() };
        assert_eq!(b.src, Src::CBuf { bank: 2, offset: 0x40 });
    }

    #[test]
    fn i2fp() {
        // I2FP.F32.S32.RM R7, R9
        let w = word(0x245, &[(16, 7), (32, 9), (74, 1), (75, 2), (78, 1), (84, 2)]);
        assert_eq!(decode(w), Some(Fp16Op::I2fp { dst: 7, src: reg_op(9, false, false, 0), signed: true, dst_bits: 32, src_bits: 32, rnd: Round::NegInf }));
        let w = word(0xa45, &[(16, 7), (38, 0x10), (54, 1), (75, 2), (84, 2)]);
        let Some(Fp16Op::I2fp { src, signed: false, .. }) = decode(w) else { panic!() };
        assert_eq!(src.src, Src::CBuf { bank: 1, offset: 0x10 });
        let w = word(0xc45, &[(16, 7), (32, 5), (91, 1), (75, 2), (84, 2)]);
        let Some(Fp16Op::I2fp { src, .. }) = decode(w) else { panic!() };
        assert_eq!(src.src, Src::UReg(5));
    }

    #[test]
    fn uimad() {
        let u = |r: u8, neg: bool| Operand { src: Src::UReg(r), neg, abs: false, swizzle: 0 };
        let common = [(16, 4), (24, 6), (72, 1), (73, 1), (81, 7), (87, 7), (90, 1), (91, 1)];
        let w = word(0x2a4, &[&common[..], &[(32, 5), (63, 1), (64, 0x3f), (75, 1)]].concat());
        assert_eq!(
            decode(w),
            Some(Fp16Op::Uimad { dst: 4, a: u(6, true), b: u(5, true), c: u(URZ, true), signed: true, x: false, carry_out: PT, carry_in: Pred { reg: PT, not: true } })
        );
        let w = word(0x4a4, &[&common[..], &[(32, 0x100), (64, 2), (74, 1)]].concat());
        let Some(Fp16Op::Uimad { b, c, x: true, .. }) = decode(w) else { panic!() };
        assert_eq!((b, c), (u(2, false), Operand::plain(Src::Imm(0x100))));
        let w = word(0x8a4, &[&common[..], &[(32, 3), (64, 7)]].concat());
        let Some(Fp16Op::Uimad { b, c, .. }) = decode(w) else { panic!() };
        assert_eq!((b, c), (Operand::plain(Src::Imm(3)), u(7, false)));
    }

    #[test]
    fn tensor_ops() {
        // HMNMX2 R4, R6.H1_H1, R8, !PT (max)
        let w = word(0x240, &[(16, 4), (24, 6), (74, 3), (32, 8), (87, 7), (90, 1)]);
        let Some(Fp16Op::Hmnmx2 { dst: 4, a, b, min, f32: false, .. }) = decode(w) else { panic!("{:?}", decode(w)) };
        assert_eq!((a, b, min), (reg_op(6, false, false, SWZ_H1_H1), reg_op(8, false, false, 0), Pred { reg: PT, not: true }));
        // HMMA.16816.F16 R10, R20, R24, R10 and HMMA.1688.F32 R2, R4, R6, RZ
        let w = word(0x23c, &[(16, 10), (24, 20), (32, 24), (64, 10), (75, 1)]);
        assert_eq!(decode(w), Some(Fp16Op::Hmma { dst: 10, a: 20, b: 24, c: 10, k: 16, f32: false, src_type: 0 }));
        let w = word(0x23c, &[(16, 2), (24, 4), (32, 6), (64, 0xff), (76, 1)]);
        assert_eq!(decode(w), Some(Fp16Op::Hmma { dst: 2, a: 4, b: 6, c: RZ, k: 8, f32: true, src_type: 0 }));
        // MOVM.16.MT88 R3, R5
        assert_eq!(decode(word(0x23a, &[(16, 3), (24, 5)])), Some(Fp16Op::Movm { dst: 3, src: 5, shape: 0 }));
    }

    #[test]
    fn foreign_opcodes() {
        for op in [0x030, 0xe30, 0x033, 0x233, 0x024, 0x224, 0x040, 0x440, 0x0a4, 0x645] {
            assert_eq!(decode(word(op, &[])), None, "{op:#05x}");
            assert_eq!(explained(op), 0);
        }
    }
}
