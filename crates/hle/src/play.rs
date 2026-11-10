//! The play engine (D41): each guest thread runs on a host thread of its own,
//! scheduled by Windows, on the real clock, with the kernel behind its one
//! lock. Never a reference run (D38). Nothing here runs in the deterministic
//! engine (D19): `me()` is None on every one of its host threads.

use std::cell::Cell;
use std::sync::LazyLock;
use std::time::{Duration, Instant};

thread_local! {
    /// The guest thread (its index) this host thread runs.
    static ME: Cell<Option<usize>> = const { Cell::new(None) };
}

/// The guest thread the calling host thread runs: None on the boot and timer
/// threads, and on every host thread of the deterministic engine.
pub fn me() -> Option<usize> {
    ME.get()
}

/// A kernel scope that holds the lock longer than this (40 ms, about one audio
/// ring) is counted in the summary line: t17's GPU work holds it until Stage 4.
pub(crate) const LONG_HOLD: u64 = 40 * 31_250;

/// Real time since the process first read it, in ticks of 31.25 MHz, the rate
/// the SDK assumes. No lock: guest counter reads call it on any host thread.
// ponytail: Instant::now() and a division, ~60 ns; a raw QPC with a fixed-point scale if counter reads ever show in a profile.
pub fn real_ticks() -> u64 {
    static ORIGIN: LazyLock<Instant> = LazyLock::new(Instant::now);
    ticks_in(ORIGIN.elapsed())
}

/// A duration in ticks of 32 ns, rounded down. Exact: a second is 31,250,000.
fn ticks_in(d: Duration) -> u64 {
    d.as_secs() * 31_250_000 + u64::from(d.subsec_nanos() / 32)
}

/// Host time left until real tick `at`; zero once it has passed.
pub fn real_until(at: u64) -> Duration {
    Duration::from_nanos(at.saturating_sub(real_ticks()).saturating_mul(32))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ticks_are_32_ns() {
        for (s, n, ticks) in [(0, 0, 0), (0, 31, 0), (0, 32, 1), (0, 999_999_999, 31_249_999), (1, 0, 31_250_000), (1, 63, 31_250_001), (3_153_600_000, 123_456_789, 3_153_600_000 * 31_250_000 + 3_858_024)] {
            assert_eq!(ticks_in(Duration::new(s, n)), ticks, "{s} s {n} ns");
        }
    }

    #[test]
    fn the_real_clock_runs_at_31_25_mhz() {
        let (i0, t0) = (Instant::now(), real_ticks());
        std::thread::sleep(Duration::from_millis(50));
        let (t1, i1) = (real_ticks(), Instant::now());
        // The ticks' interval lies inside the Instants' one, give or take the rounding.
        assert!(Duration::from_nanos((t1 - t0) * 32) <= i1 - i0 + Duration::from_nanos(64));
        assert!(t1 - t0 >= 50 * 31_250 - 1);
        assert_eq!(real_until(t0), Duration::ZERO);
        assert!(real_until(real_ticks() + 31_250) <= Duration::from_millis(1));
        assert!(real_until(u64::MAX) > Duration::from_secs(1 << 32));
    }
}
