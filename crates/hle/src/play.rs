//! The play engine (D41): each guest thread runs on a host thread of its own,
//! scheduled by Windows, on the real clock, with the kernel behind its one
//! lock. Never a reference run (D38). Nothing here runs in the deterministic
//! engine (D19): `me()` is None on every one of its host threads.

use crate::Kernel;
use crate::thread::{Outcome, State};
use gyroship_runtime::{self as rt, Cpu};
use std::cell::Cell;
use std::sync::LazyLock;
use std::time::{Duration, Instant};

thread_local! {
    /// The guest thread (its index) this host thread runs.
    static ME: Cell<Option<usize>> = const { Cell::new(None) };
    /// Guest instructions this thread's preemption points set aside since its
    /// last SVC, which counts them (`--perf`).
    static PREEMPTED: Cell<u64> = const { Cell::new(0) };
}

/// How often the boot thread looks at the run's end and its time limit.
const CHECK: Duration = Duration::from_millis(250);

/// The play engine's boot thread: starts the ready threads (the main thread)
/// on host threads of their own, then waits for the last started thread to
/// end and returns how the run ended. Every other end exits the process from
/// the thread that stops it (`Kernel::stop`). Past `Config::time_limit`, in
/// real time from this call, this thread stops the run: the watchdog of
/// headless runs.
pub fn run_play() -> Outcome {
    let start = real_ticks();
    let limit = crate::with(|k| {
        assert!(k.config.play, "the play engine runs with Config::play");
        k.sched.boot_thread = Some(std::thread::current());
        let ready: Vec<usize> = (0..k.sched.threads.len()).filter(|&t| k.sched.threads[t].state == State::Ready).collect();
        if ready.is_empty() {
            k.finish();
        }
        for t in ready {
            k.start_host_thread(t);
        }
        k.config.time_limit.map(|l| start.saturating_add(l))
    });
    loop {
        if let Some(outcome) = crate::with(|k| k.sched.outcome.clone()) {
            return outcome;
        }
        match limit {
            Some(at) if real_ticks() >= at => crate::with(|k| k.stop(crate::STOP_EXIT, "the run reached its time limit")),
            Some(at) => std::thread::park_timeout(real_until(at).min(CHECK)),
            None => std::thread::park_timeout(CHECK),
        }
    }
}

impl Kernel {
    /// Started thread `t` gets a host thread of its own, named after it, with
    /// the fibers' stack reservation. Its code may start at once; its first
    /// kernel scope waits for this lock.
    pub(crate) fn start_host_thread(&mut self, t: usize) {
        let (id, cpu) = (self.sched.threads[t].id, self.sched.threads[t].cpu as usize);
        let kernel = crate::kernel_ref();
        let spawned = std::thread::Builder::new().name(format!("guest {id}")).stack_size(rt::fiber::STACK_RESERVE).spawn(move || host_main(kernel, t, cpu));
        match spawned {
            Ok(h) => self.sched.threads[t].host_thread = Some(h),
            Err(e) => self.stop(crate::STOP_EXIT, &format!("guest thread {id}: no host thread: {e}")),
        }
    }
}

/// The host thread of guest thread `me`: its code, then its end.
fn host_main(kernel: crate::KernelRef, me: usize, cpu: usize) {
    crate::attach(kernel);
    ME.set(Some(me));
    // SAFETY: the thread's own context, owned by the thread table for the
    // kernel's life; only this host thread runs it.
    unsafe { rt::run(cpu as *mut Cpu) };
    // Its code returned to EXIT: it ends as with ExitThread, and its host thread with it.
    crate::with(|k| k.end_current());
}

/// The play engine's SVC (from `on_svc`): the SVC under the kernel lock, its
/// deferred work after the lock, then what its caller's state asks for.
/// Windows schedules the host threads: no scheduler runs.
pub(crate) unsafe fn svc(cpu: *mut Cpu, pc: u64, imm: u32, me: usize) {
    // SleepThread 0, -1 or -2: the host's own yield, once the lock is free.
    // SAFETY: the calling thread's own context, during its own hook.
    let yields = imm == 0x0B && unsafe { (*cpu).x[0] } as i64 <= 0;
    let (job, state) = crate::with(|k| {
        // SAFETY: as above: what its preemption points set aside counts here.
        unsafe { (*cpu).work += PREEMPTED.take() };
        k.take_work(cpu);
        k.census(cpu, imm, None);
        // SAFETY: as above, for the time of the call.
        k.svc(unsafe { &mut *cpu }, pc, imm);
        (k.deferred.take(), k.sched.threads[me].state)
    });
    if let Some(why) = job.and_then(|job| job.run().err()) {
        crate::with(|k| k.stop(crate::STOP_EXIT, &why));
    }
    match state {
        State::Waiting => crate::with(|k| k.stop(crate::STOP_EXIT, "play engine: blocking SVCs are not implemented yet")),
        // ExitThread inside guest frames, which must never run again: they stay on
        // this host stack, as a deleted fiber's do, and the thread parks for good.
        // ponytail: one parked host thread per ExitThread (one in the tutorial); a fiber per host thread frees them if a session piles them up.
        State::Done => loop {
            std::thread::park();
        },
        _ if yields => std::thread::yield_now(),
        _ => {}
    }
}

/// The play engine's `yield`, `wfe`, `wfi` and preemption points (from
/// `on_wait`): no kernel lock and no switch.
pub(crate) unsafe fn hint(cpu: *mut Cpu, kind: u32) {
    match kind {
        // Generated code calls again at every loop head while its count stays past
        // PREEMPT_WORK: the count moves aside for the next SVC.
        // SAFETY: the calling thread's own context, during its own hook.
        rt::YIELD_PREEMPT => PREEMPTED.set(PREEMPTED.get() + unsafe { std::mem::take(&mut (*cpu).work) }),
        // wfi waits for an interrupt: the host core goes to another thread.
        3 => std::thread::yield_now(),
        // yield and wfe sit in spin loops whose other side runs on another core.
        _ => std::hint::spin_loop(),
    }
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
