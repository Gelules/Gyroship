//! The "int" family: integer arithmetic, logic, shifts and comparisons
//! (SM 8.x encodings).
//!
//! Contract for the integration:
//! - `pub enum IntOp` with typed variants (our own code, written from the
//!   encoding facts in Mesa NAK's `sm70_encode.rs`, MIT);
//! - `pub fn decode(w: Word) -> Option<IntOp>`: Some for every opcode
//!   (12-bit, all forms) the family owns, None for any other;
//! - `pub fn explained(opcode: u16) -> u128`: the bits the decoder reads for
//!   that opcode (with `core::bits`); ALWAYS_EXPLAINED is added by the check;
//! - `pub const OPCODES: &[u16]`: the 12-bit opcodes the family owns.
//!
//! Every instruction here uses the generic ALU layout (`core::alu`): opcode
//! in bits 0..9, operand form in 9..12, destination register 16..24, a at
//! 24..32, b and c in the slots at 32.. and 64.. depending on the form.
//! Integer operations reuse the float modifier bits for their own fields, so
//! the decoder never takes `abs` from `alu()` and keeps `neg` only where the
//! instruction really has a negation (IADD3, IMAD, LEA's b).
//!
//! Sources of the bit positions (NAK, `impl SM70Op for Op*::encode`, SM 8.x
//! path): OpSel, OpISetP (+ `set_pred_set_op`, `set_int_cmp_op`), OpIAdd3 /
//! OpIAdd3X, OpLea / OpLeaX, OpLop3, OpIAbs, OpPrmt, OpIMnMx, OpShf, OpSgxt,
//! OpBMsk, OpIMad, OpFlo, OpBRev, OpPopC. IMAD.HI (0x027) has no NAK encoder:
//! it is identified from the samples (see `IntOp::IMad`).

use super::prelude::*;

pub const OPCODES: &[u16] = &[
    0x819, 0x812, 0x20c, 0x81a, 0x824, 0x80c, 0x210, 0x810, 0xa0c, 0x807, 0x424, 0x212, 0x816, 0x224, 0x811, 0x211, 0xa12, 0x207, 0x817, 0xa10, 0xc10, 0x217, 0x219,
    0xc17, 0x624, 0x827, 0x300, 0xc0c, 0xa11, 0xa17, 0xa24, 0x213, 0xd09, 0x309, 0xc12, 0xd00, 0xe24, 0x21a, 0x301, 0xb09, 0x21b, 0xc24,
];

/// ISETP comparison (bits 76..79, NAK `set_int_cmp_op`).
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

/// How ISETP combines its comparison with the accumulator predicate (bits
/// 74..76, NAK `set_pred_set_op`). Value 3 has no known meaning.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BoolOp {
    And,
    Or,
    Xor,
    Unknown3,
}

/// PRMT selection mode (bits 72..75, NAK OpPrmt). Value 7 has no known meaning.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PrmtMode {
    /// Plain byte index selection (no suffix).
    Index,
    F4e,
    B4e,
    Rc8,
    Ecl,
    Ecr,
    Rc16,
    Unknown7,
}

/// SHF data type (bits 73..75, NAK OpShf).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ShfType {
    S64,
    U64,
    S32,
    U32,
}

/// A predicate destination register (7 = PT, the result is discarded).
pub type PDst = u8;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum IntOp {
    /// SEL (0x007): dst = cond ? a : b. cond at 87..90, its negation at 90.
    Sel { dst: u8, a: Operand, b: Operand, cond: Pred },
    /// ISETP (0x00c): dst = (a cmp b) op accum; dst1 gets the complementary
    /// result. No GPR destination. `ex` (bit 72) chains a 64-bit comparison
    /// through `low_cmp` (68..71, negation 71). signed 73, op 74..76, cmp
    /// 76..79, dst 81..84, dst1 84..87, accum 87..90 (negation 90).
    ISetP { dst: PDst, dst1: PDst, a: Operand, b: Operand, cmp: IntCmp, op: BoolOp, signed: bool, ex: bool, accum: Pred, low_cmp: Pred },
    /// IADD3 (0x010): dst = a + b + c. Negations: a 72, slot 32.. 63, slot
    /// 64.. 75 (as in `alu`); with `x` (bit 74) they mean bitwise NOT and the
    /// carries come in from `carry[0]` (87..90, neg 90) and `carry[1]` (77..80,
    /// neg 80). Carry/overflow outputs: `overflow[0]` 81..84, `[1]` 84..87.
    IAdd3 { dst: u8, a: Operand, b: Operand, c: Operand, x: bool, carry: [Pred; 2], overflow: [PDst; 2] },
    /// LEA (0x011): dst = (a << shift) + b, or with `hi` (bit 80) the high
    /// half of the 64-bit c:a shifted. `neg_shifted` (72) negates (with `x`,
    /// inverts) the shifted a; b may be negated (bit 63, mid slot). x 74,
    /// shift 75..80, overflow 81..84, carry 87..90 (neg 90).
    Lea { dst: u8, a: Operand, b: Operand, c: Operand, shift: u8, hi: bool, x: bool, neg_shifted: bool, overflow: PDst, carry: Pred },
    /// LOP3.LUT (0x012): dst = lut(a, b, c). lut 72..80, `pand` 80, predicate
    /// output 81..84, predicate input 87..90 (neg 90).
    Lop3 { dst: u8, a: Operand, b: Operand, c: Operand, lut: u8, pand: bool, pdst: PDst, psrc: Pred },
    /// IABS (0x013): dst = |src|, src in the b slot.
    IAbs { dst: u8, src: Operand },
    /// PRMT (0x016): byte permute of a and c, selected by b; mode 72..75.
    Prmt { dst: u8, a: Operand, sel: Operand, c: Operand, mode: PrmtMode },
    /// IMNMX (0x017): dst = min/max(a, b); `min` (87..90, neg 90) true picks
    /// the minimum; signed 73.
    IMnMx { dst: u8, a: Operand, b: Operand, signed: bool, min: Pred },
    /// SHF (0x019): funnel shift of c:a (high:low) by b. type 73..75, wrap
    /// 75, right 76, hi (keep the high word) 80.
    Shf { dst: u8, lo: Operand, shift: Operand, hi: Operand, ty: ShfType, wrap: bool, right: bool, dst_hi: bool },
    /// SGXT (0x01a): sign/zero-extend a from b bits; signed 73, wrap 75.
    Sgxt { dst: u8, a: Operand, bits: Operand, signed: bool, wrap: bool },
    /// BMSK (0x01b): bit mask of width b at position a; wrap 75.
    BMsk { dst: u8, pos: Operand, width: Operand, wrap: bool },
    /// IMAD (0x024) and IMAD.HI (0x027, `hi`): dst = a * b + c (low or high
    /// 32 bits of the product). Negations as in `alu` (a 72, slot 32.. 63,
    /// slot 64.. 75); signed 73 and `pdst` 81..84 (always PT in NAK;
    /// presumably carry-out) are NAK's. Assumed, not proved: `x` at 74 and
    /// the carry-in at 87..90 (neg 90) come from IADD3.X's layout; NAK's IMAD
    /// writes neither (74 is the slot-64 abs bit there). In the game, 74 is
    /// always 0 and 87..91 always !PT (no carry).
    /// 0x027 is not in NAK: its samples carry division magic numbers
    /// (0xcccccccd...) with c = RZ and odd destinations (not WIDE), and the
    /// same fixed fields as IMAD.
    IMad { dst: u8, a: Operand, b: Operand, c: Operand, hi: bool, signed: bool, x: bool, pdst: PDst, carry: Pred },
    /// FLO (0x100): find leading one of src (b slot); `not` 63 inverts src,
    /// signed 73, `sh` 74 returns the shift amount, pdst 81..84.
    Flo { dst: u8, src: Operand, not: bool, signed: bool, sh: bool, pdst: PDst },
    /// BREV (0x101): bit reverse of src (b slot).
    BRev { dst: u8, src: Operand },
    /// POPC (0x109): population count of src (b slot); `not` 63 inverts src.
    PopC { dst: u8, src: Operand, not: bool },
}

/// No modifiers: these bits carry other fields in integer instructions.
fn bare(o: Operand) -> Operand {
    Operand { neg: false, abs: false, ..o }
}

/// Keeps the negation, drops `abs` (bits 62/73/74 are not abs here).
fn negated(o: Operand) -> Operand {
    Operand { abs: false, ..o }
}

fn pdst(w: Word, lo: u32) -> PDst {
    w.field(lo, 3) as u8
}

pub fn decode(w: Word) -> Option<IntOp> {
    if !OPCODES.contains(&w.opcode()) {
        return None;
    }
    let Alu { a, b, c, .. } = alu(w, false)?;
    let dst = dst(w);
    let p87 = Pred::at(w, 87, 90);
    Some(match w.opcode() & 0x1ff {
        0x007 => IntOp::Sel { dst, a: bare(a), b: bare(b), cond: p87 },
        0x00c => IntOp::ISetP {
            dst: pdst(w, 81),
            dst1: pdst(w, 84),
            a: bare(a),
            b: bare(b),
            cmp: [IntCmp::False, IntCmp::Lt, IntCmp::Eq, IntCmp::Le, IntCmp::Gt, IntCmp::Ne, IntCmp::Ge, IntCmp::True][w.field(76, 3) as usize],
            op: [BoolOp::And, BoolOp::Or, BoolOp::Xor, BoolOp::Unknown3][w.field(74, 2) as usize],
            signed: w.bit(73),
            ex: w.bit(72),
            accum: p87,
            low_cmp: Pred::at(w, 68, 71),
        },
        0x010 => IntOp::IAdd3 {
            dst,
            a: negated(a),
            b: negated(b),
            c: negated(c),
            x: w.bit(74),
            carry: [p87, Pred::at(w, 77, 80)],
            overflow: [pdst(w, 81), pdst(w, 84)],
        },
        // ponytail: b's negation comes from alu(), right for the owned forms
        // (1, 4, 5: b in the slot at 32..); forms 2/3/7 would put it on bit 75,
        // which LEA uses for the shift.
        0x011 => IntOp::Lea {
            dst,
            a: bare(a),
            b: negated(b),
            c: bare(c),
            shift: w.field(75, 5) as u8,
            hi: w.bit(80),
            x: w.bit(74),
            neg_shifted: w.bit(72),
            overflow: pdst(w, 81),
            carry: p87,
        },
        0x012 => IntOp::Lop3 { dst, a: bare(a), b: bare(b), c: bare(c), lut: w.field(72, 8) as u8, pand: w.bit(80), pdst: pdst(w, 81), psrc: p87 },
        0x013 => IntOp::IAbs { dst, src: bare(b) },
        0x016 => IntOp::Prmt {
            dst,
            a: bare(a),
            sel: bare(b),
            c: bare(c),
            mode: [PrmtMode::Index, PrmtMode::F4e, PrmtMode::B4e, PrmtMode::Rc8, PrmtMode::Ecl, PrmtMode::Ecr, PrmtMode::Rc16, PrmtMode::Unknown7][w.field(72, 3) as usize],
        },
        0x017 => IntOp::IMnMx { dst, a: bare(a), b: bare(b), signed: w.bit(73), min: p87 },
        0x019 => IntOp::Shf {
            dst,
            lo: bare(a),
            shift: bare(b),
            hi: bare(c),
            ty: [ShfType::S64, ShfType::U64, ShfType::S32, ShfType::U32][w.field(73, 2) as usize],
            wrap: w.bit(75),
            right: w.bit(76),
            dst_hi: w.bit(80),
        },
        0x01a => IntOp::Sgxt { dst, a: bare(a), bits: bare(b), signed: w.bit(73), wrap: w.bit(75) },
        0x01b => IntOp::BMsk { dst, pos: bare(a), width: bare(b), wrap: w.bit(75) },
        op @ (0x024 | 0x027) => IntOp::IMad {
            dst,
            a: negated(a),
            b: negated(b),
            c: negated(c),
            hi: op == 0x027,
            signed: w.bit(73),
            x: w.bit(74),
            pdst: pdst(w, 81),
            carry: p87,
        },
        0x100 => IntOp::Flo { dst, src: bare(b), not: w.bit(63), signed: w.bit(73), sh: w.bit(74), pdst: pdst(w, 81) },
        0x101 => IntOp::BRev { dst, src: bare(b) },
        0x109 => IntOp::PopC { dst, src: bare(b), not: w.bit(63) },
        _ => unreachable!("OPCODES lists only the opcodes matched above"),
    })
}

/// Operand bits for a form: a at 24..32, and b / c wherever the form puts
/// them (see `core::alu`).
fn src_bits(form: u8, a: bool, b: bool, c: bool) -> u128 {
    let mid = match form {
        1 => bits(32, 8),
        2 | 4 => bits(32, 32),
        3 | 5 => bits(32, 27) | bits(91, 1),
        _ => bits(32, 6),
    };
    let low = bits(64, 8);
    let (bm, cm) = if matches!(form, 2 | 3 | 7) { (low, mid) } else { (mid, low) };
    [(a, bits(24, 8)), (b, bm), (c, cm)].iter().filter(|x| x.0).fold(0, |m, x| m | x.1)
}

pub fn explained(opcode: u16) -> u128 {
    if !OPCODES.contains(&opcode) {
        return 0;
    }
    let f = (opcode >> 9) as u8;
    let dst = bits(16, 8);
    let negs = bits(63, 1) | bits(72, 1) | bits(75, 1);
    let p87 = bits(87, 4);
    match opcode & 0x1ff {
        0x007 => dst | src_bits(f, true, true, false) | p87,
        0x00c => src_bits(f, true, true, false) | bits(68, 4) | bits(72, 7) | bits(81, 6) | p87,
        0x010 => dst | src_bits(f, true, true, true) | negs | bits(74, 1) | bits(77, 4) | bits(81, 6) | p87,
        0x011 => dst | src_bits(f, true, true, true) | bits(63, 1) | bits(72, 1) | bits(74, 7) | bits(81, 3) | p87,
        0x012 => dst | src_bits(f, true, true, true) | bits(72, 12) | p87,
        0x013 | 0x101 => dst | src_bits(f, false, true, false),
        0x016 => dst | src_bits(f, true, true, true) | bits(72, 3),
        0x017 => dst | src_bits(f, true, true, false) | bits(73, 1) | p87,
        0x019 => dst | src_bits(f, true, true, true) | bits(73, 4) | bits(80, 1),
        0x01a => dst | src_bits(f, true, true, false) | bits(73, 1) | bits(75, 1),
        0x01b => dst | src_bits(f, true, true, false) | bits(75, 1),
        0x024 | 0x027 => dst | src_bits(f, true, true, true) | negs | bits(73, 2) | bits(81, 3) | p87,
        0x100 => dst | src_bits(f, false, true, false) | bits(63, 1) | bits(73, 2) | bits(81, 3),
        0x109 => dst | src_bits(f, false, true, false) | bits(63, 1),
        _ => 0,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // ---- synthetic words ----

    /// Value of bits 32..64 in every synthetic word (bit 63 clear).
    const V: u64 = 0x0234_5605;

    /// dst R0x11, a R0x22, slot 32.. = V, slot 64.. = R0x33, then `extra`
    /// fields (lo, len, value) written over them.
    fn word(op: u16, extra: &[(u32, u32, u64)]) -> Word {
        let mut w = op as u128 | 0x11 << 16 | 0x22 << 24 | (V as u128) << 32 | 0x33 << 64;
        for &(lo, len, v) in extra {
            w = w & !bits(lo, len) | (v as u128) << lo;
        }
        Word(w)
    }

    fn reg(r: u8) -> Operand {
        Operand::plain(Src::Reg(r))
    }

    /// Expected b and c for V and R0x33, by hand from the form table.
    fn bc(op: u16) -> (Operand, Operand) {
        let form = (op >> 9) as u8;
        let mid = Operand::plain(match form {
            1 => Src::Reg(5),
            2 | 4 => Src::Imm(V as u32),
            3 | 5 => Src::CBuf { bank: (V >> 22 & 31) as u8, offset: (V >> 6) as u16 },
            _ => Src::UReg(5),
        });
        if matches!(form, 2 | 3 | 7) { (reg(0x33), mid) } else { (mid, reg(0x33)) }
    }

    fn forms(low: u16) -> impl Iterator<Item = u16> {
        OPCODES.iter().copied().filter(move |op| op & 0x1ff == low)
    }

    /// Every owned opcode is covered by one of the tests below.
    #[test]
    fn coverage() {
        let tested = [0x007, 0x00c, 0x010, 0x011, 0x012, 0x013, 0x016, 0x017, 0x019, 0x01a, 0x01b, 0x024, 0x027, 0x100, 0x101, 0x109];
        for &op in OPCODES {
            assert!(tested.contains(&(op & 0x1ff)), "{op:#05x}");
            assert!(decode(word(op, &[])).is_some());
        }
        // not owned: other opcode, or an owned instruction in a foreign form
        for op in [0x000, 0x225, 0x407, 0xa19, 0x202] {
            assert_eq!(decode(word(op, &[])), None, "{op:#05x}");
            assert_eq!(explained(op), 0);
        }
    }

    #[test]
    fn sel() {
        for op in forms(0x007) {
            let (b, _) = bc(op);
            let got = decode(word(op, &[(87, 4, 0b1011)])).unwrap();
            assert_eq!(got, IntOp::Sel { dst: 0x11, a: reg(0x22), b, cond: Pred { reg: 3, not: true } }, "{op:#05x}");
        }
    }

    #[test]
    fn isetp() {
        for op in forms(0x00c) {
            let (b, _) = bc(op);
            let w = word(op, &[(64, 8, 0), (68, 4, 0b1010), (72, 1, 1), (73, 1, 1), (74, 2, 2), (76, 3, 6), (81, 3, 1), (84, 3, 7), (87, 4, 0b0100)]);
            let got = decode(w).unwrap();
            let want = IntOp::ISetP {
                dst: 1,
                dst1: 7,
                a: reg(0x22),
                b,
                cmp: IntCmp::Ge,
                op: BoolOp::Xor,
                signed: true,
                ex: true,
                accum: Pred { reg: 4, not: false },
                low_cmp: Pred { reg: 2, not: true },
            };
            assert_eq!(got, want, "{op:#05x}");
        }
        // comparison table
        let cmp = |v| match decode(word(0x20c, &[(76, 3, v)])).unwrap() {
            IntOp::ISetP { cmp, .. } => cmp,
            _ => unreachable!(),
        };
        assert_eq!([cmp(0), cmp(1), cmp(2), cmp(3), cmp(4), cmp(5), cmp(7)], [IntCmp::False, IntCmp::Lt, IntCmp::Eq, IntCmp::Le, IntCmp::Gt, IntCmp::Ne, IntCmp::True]);
    }

    #[test]
    fn iadd3() {
        for op in forms(0x010) {
            let (mut b, mut c) = bc(op);
            // bit 75 negates the operand in the slot at 64..
            if b.src == Src::Reg(0x33) { b.neg = true } else { c.neg = true }
            let w = word(op, &[(72, 1, 1), (74, 1, 1), (75, 1, 1), (77, 4, 0b0101), (81, 3, 2), (84, 3, 6), (87, 4, 0b1001)]);
            let want = IntOp::IAdd3 {
                dst: 0x11,
                a: Operand { neg: true, ..reg(0x22) },
                b,
                c,
                x: true,
                carry: [Pred { reg: 1, not: true }, Pred { reg: 5, not: false }],
                overflow: [2, 6],
            };
            assert_eq!(decode(w).unwrap(), want, "{op:#05x}");
        }
        // bit 63 negates a register b (form 1)
        let IntOp::IAdd3 { b, .. } = decode(word(0x210, &[(63, 1, 1)])).unwrap() else { panic!() };
        assert_eq!(b, Operand { neg: true, ..reg(5) });
    }

    #[test]
    fn lea() {
        for op in forms(0x011) {
            let (b, c) = bc(op);
            let w = word(op, &[(72, 1, 1), (74, 1, 1), (75, 5, 0x1d), (80, 1, 1), (81, 3, 3), (87, 4, 0b0110)]);
            let want = IntOp::Lea { dst: 0x11, a: reg(0x22), b, c, shift: 0x1d, hi: true, x: true, neg_shifted: true, overflow: 3, carry: Pred { reg: 6, not: false } };
            assert_eq!(decode(w).unwrap(), want, "{op:#05x}");
        }
        let IntOp::Lea { b, .. } = decode(word(0xa11, &[(63, 1, 1)])).unwrap() else { panic!() };
        assert!(b.neg);
    }

    #[test]
    fn lop3() {
        for op in forms(0x012) {
            let (b, c) = bc(op);
            let w = word(op, &[(72, 8, 0x96), (80, 1, 1), (81, 3, 4), (87, 4, 0b1111)]);
            let want = IntOp::Lop3 { dst: 0x11, a: reg(0x22), b, c, lut: 0x96, pand: true, pdst: 4, psrc: Pred { reg: 7, not: true } };
            assert_eq!(decode(w).unwrap(), want, "{op:#05x}");
        }
    }

    #[test]
    fn one_source() {
        for op in forms(0x013).chain(forms(0x101)) {
            let (src, _) = bc(op);
            let want = if op & 0x1ff == 0x013 { IntOp::IAbs { dst: 0x11, src } } else { IntOp::BRev { dst: 0x11, src } };
            assert_eq!(decode(word(op, &[])).unwrap(), want, "{op:#05x}");
        }
        for op in forms(0x100) {
            let (src, _) = bc(op);
            let w = word(op, &[(63, 1, 1), (73, 1, 1), (74, 1, 1), (81, 3, 5)]);
            assert_eq!(decode(w).unwrap(), IntOp::Flo { dst: 0x11, src, not: true, signed: true, sh: true, pdst: 5 }, "{op:#05x}");
        }
        for op in forms(0x109) {
            let (src, _) = bc(op);
            assert_eq!(decode(word(op, &[(63, 1, 1)])).unwrap(), IntOp::PopC { dst: 0x11, src, not: true }, "{op:#05x}");
            assert_eq!(decode(word(op, &[])).unwrap(), IntOp::PopC { dst: 0x11, src, not: false }, "{op:#05x}");
        }
    }

    #[test]
    fn prmt() {
        for op in forms(0x016) {
            let (sel, c) = bc(op);
            let w = word(op, &[(72, 3, 6)]);
            assert_eq!(decode(w).unwrap(), IntOp::Prmt { dst: 0x11, a: reg(0x22), sel, c, mode: PrmtMode::Rc16 }, "{op:#05x}");
        }
    }

    #[test]
    fn imnmx() {
        for op in forms(0x017) {
            let (b, _) = bc(op);
            let w = word(op, &[(73, 1, 1), (87, 4, 0b0111)]);
            assert_eq!(decode(w).unwrap(), IntOp::IMnMx { dst: 0x11, a: reg(0x22), b, signed: true, min: Pred::TRUE }, "{op:#05x}");
        }
    }

    #[test]
    fn shf() {
        for op in forms(0x019) {
            let (shift, hi) = bc(op);
            let w = word(op, &[(73, 2, 2), (75, 1, 1), (76, 1, 1), (80, 1, 1)]);
            let want = IntOp::Shf { dst: 0x11, lo: reg(0x22), shift, hi, ty: ShfType::S32, wrap: true, right: true, dst_hi: true };
            assert_eq!(decode(w).unwrap(), want, "{op:#05x}");
        }
    }

    #[test]
    fn sgxt_bmsk() {
        for op in forms(0x01a) {
            let (bits, _) = bc(op);
            let w = word(op, &[(73, 1, 1), (75, 1, 1)]);
            assert_eq!(decode(w).unwrap(), IntOp::Sgxt { dst: 0x11, a: reg(0x22), bits, signed: true, wrap: true }, "{op:#05x}");
        }
        for op in forms(0x01b) {
            let (width, _) = bc(op);
            let w = word(op, &[(75, 1, 1)]);
            assert_eq!(decode(w).unwrap(), IntOp::BMsk { dst: 0x11, pos: reg(0x22), width, wrap: true }, "{op:#05x}");
        }
    }

    #[test]
    fn imad() {
        for op in forms(0x024).chain(forms(0x027)) {
            let (mut b, mut c) = bc(op);
            if b.src == Src::Reg(0x33) { b.neg = true } else { c.neg = true }
            let w = word(op, &[(72, 1, 1), (73, 1, 1), (74, 1, 1), (75, 1, 1), (81, 3, 0), (87, 4, 0b0010)]);
            let want = IntOp::IMad {
                dst: 0x11,
                a: Operand { neg: true, ..reg(0x22) },
                b,
                c,
                hi: op & 0x1ff == 0x027,
                signed: true,
                x: true,
                pdst: 0,
                carry: Pred { reg: 2, not: false },
            };
            assert_eq!(decode(w).unwrap(), want, "{op:#05x}");
        }
    }

    /// `explained` covers exactly the fields the tests above set.
    #[test]
    fn explained_masks() {
        assert_eq!(explained(0x819), bits(16, 56) | bits(73, 4) | bits(80, 1));
        assert_eq!(explained(0x20c), bits(24, 16) | bits(68, 11) | bits(81, 10));
        assert_eq!(explained(0xa12), bits(16, 8) | bits(24, 8) | bits(32, 27) | bits(64, 20) | bits(87, 5));
        assert_eq!(explained(0xd00), bits(16, 8) | bits(32, 6) | bits(63, 1) | bits(73, 2) | bits(81, 3));
    }
}
