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

/// What a dispatch runs (QMD V03_00 fields).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Launch {
    /// PROGRAM_ADDRESS_LOWER/UPPER: the SASS, without an SPH.
    pub program: u64,
    /// CTA_RASTER_WIDTH/HEIGHT/DEPTH: blocks in the grid.
    pub grid: [u32; 3],
    /// CTA_THREAD_DIMENSION0..2: threads in a block.
    pub block: [u32; 3],
    /// SHARED_MEMORY_SIZE, bytes.
    pub shared: u32,
    /// The valid CONSTANT_BUFFER(i): bank, address, size in bytes.
    pub cbufs: Vec<(u8, u64, u32)>,
}

pub fn launch(q: &[u8]) -> Launch {
    Launch {
        program: u64::from(mw(q, 1584, 1568)) << 32 | u64::from(mw(q, 1567, 1536)),
        grid: [mw(q, 415, 384), mw(q, 431, 416), mw(q, 463, 448)],
        block: [mw(q, 607, 592), mw(q, 623, 608), mw(q, 639, 624)],
        shared: mw(q, 561, 544),
        cbufs: (0..8u32)
            .filter(|i| mw(q, 640 + i, 640 + i) != 0)
            .map(|i| {
                let b = 1024 + 64 * i;
                (i as u8, u64::from(mw(q, b + 48, b + 32)) << 32 | u64::from(mw(q, b + 31, b)), mw(q, b + 63, b + 51) << 4)
            })
            .collect(),
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
        // A launch: program, a 4x2x1 grid of 8x8x1 blocks, bank 1 of 0x100 bytes.
        put(&mut q, 1567, 1536, 0x5678_9a00);
        put(&mut q, 1584, 1568, 0x11);
        put(&mut q, 415, 384, 4);
        put(&mut q, 431, 416, 2);
        put(&mut q, 463, 448, 1);
        put(&mut q, 607, 592, 8);
        put(&mut q, 623, 608, 8);
        put(&mut q, 639, 624, 1);
        put(&mut q, 561, 544, 0x400);
        put(&mut q, 641, 641, 1);
        put(&mut q, 1024 + 64 + 31, 1024 + 64, 0x2000);
        put(&mut q, 1024 + 64 + 48, 1024 + 64 + 32, 0x12);
        put(&mut q, 1024 + 64 + 63, 1024 + 64 + 51, 0x10);
        let l = launch(&q);
        assert_eq!(l, Launch { program: 0x11_5678_9a00, grid: [4, 2, 1], block: [8, 8, 1], shared: 0x400, cbufs: vec![(1, 0x12_0000_2000, 0x100)] });
    }
}
