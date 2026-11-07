//! What a compute dispatch's queue meta data (QMD V03_00, clc7c0qmd.h)
//! releases when the grid is done: up to two semaphore writes or
//! reductions. The compute program itself is not run here.

/// Bytes of a QMD V03_00.
pub const SIZE: usize = 0x100;

/// Bits hi..=lo of the QMD (clc7c0qmd.h `MW(hi:lo)`), at most 32 wide.
fn mw(q: &[u8], hi: u32, lo: u32) -> u32 {
    (lo..=hi).rev().fold(0, |v, b| v << 1 | u32::from(q.get((b / 8) as usize).is_some_and(|x| x >> (b % 8) & 1 != 0)))
}

/// `RELEASEn_REDUCTION_OP` (when `RELEASEn_REDUCTION_ENABLE`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Reduction {
    Add,
    Min,
    Max,
    /// Wrapping increment: 0 once the value reaches the payload.
    Inc,
    /// Wrapping decrement: the payload once the value is 0 or above it.
    Dec,
    And,
    Or,
    Xor,
}

/// One semaphore release of a QMD.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Release {
    pub addr: u64,
    pub payload: u64,
    pub reduction: Option<Reduction>,
    /// `RELEASEn_REDUCTION_FORMAT` SIGNED_32 (min and max compare signed).
    pub signed: bool,
    /// `RELEASEn_PAYLOAD64B`: a 64-bit payload.
    pub wide: bool,
    /// `RELEASEn_STRUCTURE_SIZE` FOUR_WORDS: a timestamp follows at +8.
    pub timestamp: bool,
}

impl Release {
    /// The memory value after the release, given the one before.
    pub fn apply(&self, old: u64) -> u64 {
        let p = self.payload;
        let (o, q) = if self.wide { (old, p) } else { (old & 0xffff_ffff, p & 0xffff_ffff) };
        let mask = if self.wide { u64::MAX } else { 0xffff_ffff };
        let lt = |a: u64, b: u64| if self.signed { (a as i64) << (if self.wide { 0 } else { 32 }) < (b as i64) << (if self.wide { 0 } else { 32 }) } else { a < b };
        match self.reduction {
            None => q,
            Some(Reduction::Add) => o.wrapping_add(q) & mask,
            Some(Reduction::Min) => if lt(q, o) { q } else { o },
            Some(Reduction::Max) => if lt(o, q) { q } else { o },
            Some(Reduction::Inc) => if o >= q { 0 } else { o + 1 },
            Some(Reduction::Dec) => if o == 0 || o > q { q } else { o - 1 },
            Some(Reduction::And) => o & q,
            Some(Reduction::Or) => o | q,
            Some(Reduction::Xor) => o ^ q,
        }
    }
}

/// `QMD_MAJOR_VERSION` and `QMD_VERSION`.
pub fn version(q: &[u8]) -> (u32, u32) {
    (mw(q, 583, 580), mw(q, 579, 576))
}

/// The enabled releases (RELEASE0 at bit 768, RELEASE1 at bit 896).
pub fn releases(q: &[u8]) -> Vec<Release> {
    [768u32, 896]
        .into_iter()
        .filter(|&b| mw(q, b + 55, b + 55) != 0) // RELEASEn_ENABLE
        .map(|b| Release {
            addr: u64::from(mw(q, b + 39, b + 32)) << 32 | u64::from(mw(q, b + 31, b)),
            payload: u64::from(mw(q, b + 127, b + 96)) << 32 | u64::from(mw(q, b + 95, b + 64)),
            reduction: (mw(q, b + 58, b + 58) != 0).then(|| {
                use Reduction::*;
                [Add, Min, Max, Inc, Dec, And, Or, Xor][mw(q, b + 54, b + 52) as usize]
            }),
            signed: mw(q, b + 57, b + 56) == 1,
            wide: mw(q, b + 61, b + 61) != 0,
            timestamp: mw(q, b + 63, b + 62) == 0,
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn put(q: &mut [u8], hi: u32, lo: u32, v: u32) {
        for (i, b) in (lo..=hi).enumerate() {
            let byte = &mut q[(b / 8) as usize];
            *byte = *byte & !(1 << (b % 8)) | ((v >> i & 1) as u8) << (b % 8);
        }
    }

    #[test]
    fn releases_and_reductions() {
        let mut q = [0u8; SIZE];
        put(&mut q, 583, 580, 3);
        // RELEASE0: one word, add 1 at 0x12_3456_7000.
        put(&mut q, 799, 768, 0x3456_7000);
        put(&mut q, 807, 800, 0x12);
        put(&mut q, 823, 823, 1);
        put(&mut q, 826, 826, 1);
        put(&mut q, 831, 830, 1);
        put(&mut q, 863, 832, 1);
        // RELEASE1: four words (timestamp), plain write of 0x99 at 0x8000.
        put(&mut q, 927, 896, 0x8000);
        put(&mut q, 951, 951, 1);
        put(&mut q, 991, 960, 0x99);
        assert_eq!(version(&q).0, 3);
        let r = releases(&q);
        assert_eq!(r.len(), 2);
        assert_eq!((r[0].addr, r[0].payload, r[0].reduction, r[0].timestamp), (0x12_3456_7000, 1, Some(Reduction::Add), false));
        assert_eq!((r[1].addr, r[1].payload, r[1].reduction, r[1].timestamp), (0x8000, 0x99, None, true));
        assert_eq!(r[0].apply(0x9a), 0x9b);
        assert_eq!(r[0].apply(0xffff_ffff), 0);
        assert_eq!(r[1].apply(5), 0x99);
        let inc = Release { reduction: Some(Reduction::Inc), payload: 3, ..r[0] };
        assert_eq!((inc.apply(2), inc.apply(3)), (3, 0));
        let min = Release { reduction: Some(Reduction::Min), payload: 0xffff_fffe, signed: true, ..r[0] };
        assert_eq!(min.apply(1), 0xffff_fffe);
        assert!(releases(&[0u8; SIZE]).is_empty());
    }
}
