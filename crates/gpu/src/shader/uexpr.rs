//! Uniform-register values the renderer can compute at a draw: words of
//! constant banks and what the uniform datapath makes of them before a
//! texture instruction reads its handle (e.g. a texture handle or'ed with a
//! sampler handle).

use std::collections::BTreeSet;

/// A uniform register's value as an expression over constant-bank words.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum UExpr {
    Imm(u32),
    /// c[bank][byte offset].
    CBuf(u8, u16),
    /// LOP3 truth table over a (0xf0), b (0xcc), c (0xaa).
    Lop3(u8, Box<[UExpr; 3]>),
    /// a + b + c.
    Add(Box<[UExpr; 3]>),
    /// Funnel shift of high:low by `amount`, as SHF: masked by `limit` when
    /// `wrap`, else clamped to `limit + 1`; the high word when `hi`.
    Shf { parts: Box<[UExpr; 3]>, limit: u32, wrap: bool, right: bool, signed: bool, hi: bool },
}

impl UExpr {
    /// The value, given the constant banks' words; None if one is unknown.
    pub fn eval(&self, word: &dyn Fn(u8, u16) -> Option<u32>) -> Option<u32> {
        let all = |x: &[UExpr; 3]| -> Option<[u32; 3]> { Some([x[0].eval(word)?, x[1].eval(word)?, x[2].eval(word)?]) };
        Some(match self {
            UExpr::Imm(v) => *v,
            UExpr::CBuf(bank, offset) => word(*bank, *offset)?,
            UExpr::Lop3(lut, x) => {
                let [a, b, c] = all(x)?;
                let pick = |v: u32, on: bool| if on { v } else { !v };
                (0..8).filter(|k| lut >> k & 1 != 0).fold(0, |r, k| r | pick(a, k & 4 != 0) & pick(b, k & 2 != 0) & pick(c, k & 1 != 0))
            }
            UExpr::Add(x) => {
                let [a, b, c] = all(x)?;
                a.wrapping_add(b).wrapping_add(c)
            }
            UExpr::Shf { parts, limit, wrap, right, signed, hi } => {
                let [low, shift, high] = all(parts)?;
                let v = u64::from(high) << 32 | u64::from(low);
                let s = if *wrap { shift & limit } else { shift.min(limit + 1) };
                let r = match (right, signed) {
                    (true, true) => ((v as i64) >> s.min(63)) as u64,
                    (true, false) => v.checked_shr(s).unwrap_or(0),
                    (false, _) => v.checked_shl(s).unwrap_or(0),
                };
                if *hi { (r >> 32) as u32 } else { r as u32 }
            }
        })
    }

    /// The constant banks it reads.
    pub fn banks(&self, out: &mut BTreeSet<u8>) {
        match self {
            UExpr::Imm(_) => {}
            UExpr::CBuf(bank, _) => {
                out.insert(*bank);
            }
            UExpr::Lop3(_, x) | UExpr::Add(x) | UExpr::Shf { parts: x, .. } => x.iter().for_each(|e| e.banks(out)),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A texture handle or'ed with a sampler handle, and the sampler index
    /// taken from it (ULOP3 0xfc, then USHF.R.U32.HI by 20 of URZ:UR).
    #[test]
    fn a_combined_handle_and_its_sampler_index() {
        let words = |bank: u8, offset: u16| match (bank, offset) {
            (2, 0x40) => Some(0x0000_0042),
            (2, 0x44) => Some(0x0230_0000),
            _ => None,
        };
        let handle = UExpr::Lop3(0xfc, Box::new([UExpr::CBuf(2, 0x40), UExpr::CBuf(2, 0x44), UExpr::Imm(0)]));
        assert_eq!(handle.eval(&words), Some(0x0230_0042));
        let sampler = UExpr::Shf { parts: Box::new([UExpr::Imm(0), UExpr::Imm(20), handle.clone()]), limit: 31, wrap: false, right: true, signed: false, hi: true };
        assert_eq!(sampler.eval(&words), Some(0x23));
        assert_eq!(UExpr::Add(Box::new([UExpr::Imm(u32::MAX), UExpr::Imm(2), UExpr::CBuf(2, 0x40)])).eval(&words), Some(0x43));
        assert_eq!(UExpr::CBuf(3, 0).eval(&words), None);
        let mut banks = BTreeSet::new();
        sampler.banks(&mut banks);
        assert_eq!(banks, BTreeSet::from([2]));
    }
}
