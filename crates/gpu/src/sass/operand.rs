//! Operands the instruction families share, and the generic ALU operand
//! layout: the operand form (bits 9..12) says where the second and third
//! sources are.
//!
//! | form | b (second source)        | c (third source)         |
//! |------|--------------------------|--------------------------|
//! | 1    | register, bits 32..40    | register, bits 64..72    |
//! | 2    | register, bits 64..72    | immediate, bits 32..64   |
//! | 3    | register, bits 64..72    | constant, bits 32..59    |
//! | 4    | immediate, bits 32..64   | register, bits 64..72    |
//! | 5    | constant, bits 32..59    | register, bits 64..72    |
//! | 6    | uniform reg, bits 32..38 | register, bits 64..72    |
//! | 7    | register, bits 64..72    | uniform reg, bits 32..38 |
//!
//! The first source is always the register at 24..32. Modifiers: a has abs
//! 73 and neg 72; the slot at 32.. has abs 62 and neg 63; the slot at 64..
//! has abs 74 and neg 75 (83 and 84 for packed half floats). A constant is
//! c[bank 54..59][byte offset 38..54], or c[UR at 32..38][offset] when bit
//! 91 is set. Public encoding facts (Mesa NAK); this is our own decoder.

use super::Word;

/// The zero register (R255), uniform zero (UR63), true predicate (P7).
pub const RZ: u8 = 255;
pub const URZ: u8 = 63;
pub const PT: u8 = 7;

/// A predicate operand: P0..P6, or PT (7); `not` negates it.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Pred {
    pub reg: u8,
    pub not: bool,
}

impl Pred {
    /// The predicate at `lo..lo+3` with its negation bit.
    pub fn at(w: Word, lo: u32, not_bit: u32) -> Pred {
        Pred { reg: w.field(lo, 3) as u8, not: w.bit(not_bit) }
    }

    pub const TRUE: Pred = Pred { reg: PT, not: false };
}

/// Where a source value comes from.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Src {
    /// R0..R254, or RZ (255).
    Reg(u8),
    /// UR0..UR62, or URZ (63).
    UReg(u8),
    Imm(u32),
    /// c[bank][byte offset].
    CBuf { bank: u8, offset: u16 },
    /// c[UR][byte offset]: a bindless constant buffer.
    CBufU { ureg: u8, offset: u16 },
}

/// A source with its modifiers and, for packed half floats, the raw 2-bit
/// swizzle field (its meaning belongs to the instruction family).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Operand {
    pub src: Src,
    pub neg: bool,
    pub abs: bool,
    pub swizzle: u8,
}

impl Operand {
    pub fn plain(src: Src) -> Operand {
        Operand { src, neg: false, abs: false, swizzle: 0 }
    }
}

/// The three sources of an ALU instruction.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Alu {
    pub form: u8,
    pub a: Operand,
    pub b: Operand,
    pub c: Operand,
}

/// The register at `lo..lo+8`.
pub fn reg(w: Word, lo: u32) -> u8 {
    w.field(lo, 8) as u8
}

/// The uniform register at `lo..lo+6`.
pub fn ureg(w: Word, lo: u32) -> u8 {
    w.field(lo, 6) as u8
}

/// The destination register (16..24).
pub fn dst(w: Word) -> u8 {
    reg(w, 16)
}

/// The constant at 32..59 (bindless when bit 91 is set).
pub fn cbuf(w: Word) -> Src {
    let offset = w.field(38, 16) as u16;
    if w.bit(91) { Src::CBufU { ureg: ureg(w, 32), offset } } else { Src::CBuf { bank: w.field(54, 5) as u8, offset } }
}

/// Decodes the generic ALU sources; `fp16` selects the packed half float
/// modifier and swizzle bits. `None` for forms 0 and outside 1..=7.
pub fn alu(w: Word, fp16: bool) -> Option<Alu> {
    let form = w.field(9, 3) as u8;
    let with = |src: Src, abs: u32, neg: u32, swizzle: Option<u32>| Operand { src, abs: w.bit(abs), neg: w.bit(neg), swizzle: swizzle.map_or(0, |lo| w.field(lo, 2) as u8) };
    let half = |lo: u32| fp16.then_some(lo);
    let a = with(Src::Reg(reg(w, 24)), 73, 72, half(74));
    // The slot at 32.. (register, uniform register or constant) and the one at 64..
    let mid = |src: Src| with(src, 62, 63, half(60));
    let (abs64, neg64) = if fp16 { (83, 84) } else { (74, 75) };
    let low = with(Src::Reg(reg(w, 64)), abs64, neg64, half(81));
    let imm = Operand::plain(Src::Imm(w.field(32, 32) as u32));
    let (b, c) = match form {
        1 => (mid(Src::Reg(reg(w, 32))), low),
        2 => (low, imm),
        3 => (low, mid(cbuf(w))),
        4 => (imm, low),
        5 => (mid(cbuf(w)), low),
        6 => (mid(Src::UReg(ureg(w, 32))), low),
        7 => (low, mid(Src::UReg(ureg(w, 32)))),
        _ => return None,
    };
    Some(Alu { form, a, b, c })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn word(parts: &[(u32, u128)]) -> Word {
        Word(parts.iter().fold(0, |w, (lo, v)| w | v << lo))
    }

    #[test]
    fn each_form_puts_its_sources_where_the_table_says() {
        // a = R1, slot 32 = R2 / imm / c[3][0x40] / UR4, slot 64 = R5; -|x| on
        // slot 32 but for immediates, whose bits 62 and 63 are the value's.
        let at = |form: u128, slot: &[(u32, u128)]| {
            let mut p = vec![(24, 1), (64, 5), (9, form)];
            if form != 2 && form != 4 {
                p.extend([(62, 1), (63, 1)]);
            }
            p.extend_from_slice(slot);
            alu(word(&p), false).unwrap()
        };
        let r1 = at(1, &[(32, 2)]);
        assert_eq!((r1.a.src, r1.b.src, r1.b.neg, r1.b.abs, r1.c.src), (Src::Reg(1), Src::Reg(2), true, true, Src::Reg(5)));
        let r4 = at(4, &[(32, 0x3f80_0000)]);
        assert_eq!((r4.b.src, r4.c.src), (Src::Imm(0x3f80_0000), Src::Reg(5)));
        let r2 = at(2, &[(32, 0x3f80_0000)]);
        assert_eq!((r2.b.src, r2.c.src), (Src::Reg(5), Src::Imm(0x3f80_0000)));
        let r5 = at(5, &[(38, 0x40), (54, 3)]);
        assert_eq!(r5.b.src, Src::CBuf { bank: 3, offset: 0x40 });
        let r5u = at(5, &[(38, 0x40), (32, 4), (91, 1)]);
        assert_eq!(r5u.b.src, Src::CBufU { ureg: 4, offset: 0x40 });
        let r6 = at(6, &[(32, 4), (91, 1)]);
        assert_eq!(r6.b.src, Src::UReg(4));
        let r7 = at(7, &[(32, 4), (91, 1)]);
        assert_eq!((r7.b.src, r7.c.src), (Src::Reg(5), Src::UReg(4)));
        assert!(alu(word(&[(9, 0)]), false).is_none());
    }
}
