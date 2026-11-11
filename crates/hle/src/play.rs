//! The play engine (D41): each guest thread runs on a host thread of its own,
//! scheduled by Windows, on the real clock, with the kernel behind its one
//! lock. Never a reference run (D38). Nothing here runs in the deterministic
//! engine (D19): `me()` is None on every one of its host threads.

use crate::Kernel;
use crate::thread::{Outcome, State, Timer, TimerAction, Wait};
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
    rt::fine_timers();
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
    let timers = spawn_timers(crate::kernel_ref());
    let done = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false));
    let watchdog = spawn_watchdog(done.clone());
    loop {
        if let Some(outcome) = crate::with(|k| k.sched.outcome.clone()) {
            done.store(true, std::sync::atomic::Ordering::Relaxed);
            // They see the outcome too, and leave the kernel before it is uninstalled.
            for t in timers.into_iter().chain([watchdog]) {
                t.thread().unpark();
                let _ = t.join();
            }
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
        State::Waiting => wait(me),
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

/// After its SVC, the calling host thread waits here while its guest thread
/// `me` is Waiting. Each round runs under the kernel lock:
/// - a waker made it Ready, writing x0/x1 under the lock: it returns, and
///   having taken the lock again makes those writes visible to its code;
/// - past its deadline, it ends its own wait (x0 = `on_timeout`, as `pick`
///   does), so a timeout and a wake can never both win;
/// - otherwise the round releases the lock and waits: `park` without a
///   deadline, `park_timeout` (a waker may end it early; to the millisecond
///   once `fine_timers` ran), or `sleep` for a Sleep, which nothing ends early.
///
/// Wakers queue the unpark, and `with` issues it once the lock is free; std
/// keeps an unpark that comes before the park, so no wake-up is lost, and a
/// spurious return just goes round again.
fn wait(me: usize) {
    enum Then {
        Run,
        Park,
        ParkFor(Duration),
        Sleep(Duration),
    }
    loop {
        let then = crate::with(|k| {
            let now = k.now();
            let th = &k.sched.threads[me];
            let (state, sleep, deadline, cpu, on_timeout) = (th.state, th.wait == Wait::Sleep, th.deadline, th.cpu, th.on_timeout);
            if state != State::Waiting {
                return Then::Run;
            }
            match deadline {
                None => Then::Park,
                Some(d) if now < d => {
                    let left = Duration::from_nanos((d - now).saturating_mul(32));
                    if sleep { Then::Sleep(left) } else { Then::ParkFor(left) }
                }
                Some(d) => {
                    // SAFETY: this thread's own context; its code is stopped in this hook.
                    unsafe { (*cpu).x[0] = u64::from(on_timeout) };
                    k.ready(me, now);
                    // How late timed waits end, for the summary line.
                    k.count(if sleep { "play sleeps" } else { "play timed waits" });
                    if now - d > 31_250 {
                        k.count(if sleep { "play sleeps over 1 ms late" } else { "play timed waits over 1 ms late" });
                    }
                    Then::Run
                }
            }
        });
        match then {
            Then::Run => return,
            Then::Park => std::thread::park(),
            Then::ParkFor(left) => std::thread::park_timeout(left),
            Then::Sleep(left) => std::thread::sleep(left),
        }
    }
}

/// The longest wait of a timer thread: a timer the game creates meanwhile
/// starts within it.
// ponytail: a new timer may start up to 100 ms late (twice a run); unpark the timer threads where timers are pushed if that ever matters.
const TIMER_IDLE: u64 = 3_125_000;

/// Waits until real tick `at`, TIMER_IDLE at most (to the millisecond, as
/// timed waits are); the boot thread unparks it at the run's end.
fn wait_until(at: u64) {
    std::thread::park_timeout(real_until(at.min(real_ticks() + TIMER_IDLE)));
}

/// The play engine's timers, each on a host thread of its own: the audio
/// sinks and hid's sampler. Each returns once the run's outcome is set.
fn spawn_timers(kernel: crate::KernelRef) -> [std::thread::JoinHandle<()>; 2] {
    let spawn = |name: &str, run: fn()| {
        std::thread::Builder::new()
            .name(name.into())
            .spawn(move || {
                crate::attach(kernel);
                run()
            })
            .expect("a timer thread")
    };
    [spawn("audio sink", run_sinks), spawn("hid sampler", run_hid)]
}

/// How long the kernel lock may stay held before the watchdog shows where its
/// holder stands (a long GPU batch takes seconds, until Stage 4).
const WATCHDOG: Duration = Duration::from_secs(5);

/// The watchdog: every half second it reads the kernel lock's holder without
/// the lock; a holder past WATCHDOG, then past 4 WATCHDOG, gets its host
/// stack written to stderr (the journal needs the lock). Returns once `done`.
fn spawn_watchdog(done: std::sync::Arc<std::sync::atomic::AtomicBool>) -> std::thread::JoinHandle<()> {
    use std::sync::atomic::Ordering::Relaxed;
    std::thread::Builder::new()
        .name("watchdog".into())
        .spawn(move || {
            let (mut hold, mut shown) = (0, 0);
            while !done.load(Relaxed) {
                std::thread::park_timeout(Duration::from_millis(500));
                let (since, tid, guest) = (crate::HELD_SINCE.load(Relaxed), crate::HOLDER.load(Relaxed), crate::HOLDER_GUEST.load(Relaxed));
                if tid == 0 || crate::HELD_SINCE.load(Relaxed) != since {
                    continue;
                }
                if since != hold {
                    (hold, shown) = (since, 0);
                }
                let held = real_ticks().saturating_sub(since);
                let due = [WATCHDOG, 4 * WATCHDOG].into_iter().filter(|&d| held >= ticks_in(d)).count();
                if due > shown {
                    shown = due;
                    eprintln!("gyroship: watchdog: the kernel lock has been held for {:.1} s by guest thread {guest} (host thread {tid}, 0: a timer or the boot thread); where it stands:", held as f64 / 31_250_000.0);
                    for line in rt::profile::stack_of(tid) {
                        eprintln!("gyroship:   {line}");
                    }
                }
            }
        })
        .expect("the watchdog thread")
}

/// The audio sinks, on their grid of real time. A tick copies the ring and
/// hands it to the host outside the kernel lock, so that the sound does not
/// wait for a holder (thread 17 holds it through its GPU work until Stage 4);
/// then one scope signals the streams' events, takes new sinks and the gain.
/// A holder past a whole period still delays the next tick.
// ponytail: QPC's grid, which the sound device's own clock drifts from (a gap or a dropped tick every few minutes); pace on the device's queue level if that is heard.
fn run_sinks() {
    let (mem, mut out) = crate::with(|k| (k.mem.clone(), k.audio_out.take()));
    // (timer index, the timer as it ticks here); timers are never removed.
    let mut sinks: Vec<(usize, Timer)> = Vec::new();
    let (mut seen, mut gain) = (0, 1.0);
    // For the journal: ticks so far, and the frames played since the last line.
    let (mut ticks, mut played, mut since) = (0u64, 0usize, Instant::now());
    loop {
        let now = real_ticks();
        let mut ticked = Vec::new();
        for (i, timer) in sinks.iter_mut().filter(|s| s.1.at() <= now) {
            let TimerAction::AudioSink { ring, slots, data, channels, event } = timer.action else { unreachable!("only sinks are kept") };
            let (_, frames) = crate::audio::sink_take(&mem, ring, slots, data, channels, true);
            if let (Some(out), Some(f)) = (out.as_mut(), &frames) {
                out(f, gain);
            }
            played += frames.as_ref().map_or(0, |f| f.len() / channels.max(1) as usize);
            // A tick takes all that was written: the ticks a late wake-up missed go with it.
            while timer.at() <= now {
                timer.k += 1;
            }
            ticked.push((*i, event, frames, timer.k));
        }
        let done = crate::with(|k| {
            for (i, t) in k.timers.iter().enumerate().skip(seen) {
                if matches!(t.action, TimerAction::AudioSink { .. }) {
                    sinks.push((i, *t));
                }
            }
            seen = k.timers.len();
            for (i, event, frames, next) in ticked {
                if let (Some(all), Some(f)) = (k.audio_samples.as_mut(), frames) {
                    all.extend(f);
                }
                k.signal_event(event);
                k.timers[i].k = next;
                ticks += 1;
                if ticks.is_multiple_of(100) {
                    let line = format!("audio sink: tick {ticks}: {played} frames in {:.2} s", since.elapsed().as_secs_f64());
                    k.log(&line);
                    (played, since) = (0, Instant::now());
                }
            }
            gain = k.audio_gain;
            k.sched.outcome.is_some()
        });
        if done {
            return;
        }
        wait_until(sinks.iter().map(|s| s.1.at()).min().unwrap_or(u64::MAX));
    }
}

/// hid's samples on their 60 Hz grid of real time, on a host thread of its
/// own: a sample needs the kernel lock, which the sink must never wait behind.
/// Samples the lock made late are taken at once, one after the other, so that
/// sample k still stands for k / 60 s (`--replay` and `--record` count samples).
fn run_hid() {
    loop {
        let next = crate::with(|k| {
            if k.sched.outcome.is_some() {
                return None;
            }
            let now = k.now();
            for i in 0..k.timers.len() {
                let TimerAction::HidSample { shared } = k.timers[i].action else { continue };
                while k.timers[i].at() <= now {
                    let n = k.timers[i].k;
                    k.hid_sample(shared, n);
                    k.timers[i].k += 1;
                }
            }
            Some(k.timers.iter().filter(|t| matches!(t.action, TimerAction::HidSample { .. })).map(Timer::at).min().unwrap_or(u64::MAX))
        });
        let Some(next) = next else { return };
        wait_until(next);
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

    /// The sink's copy and its sound go out on time while another thread holds
    /// the kernel: only the stream's event waits for the lock.
    #[test]
    fn the_sink_plays_while_another_thread_holds_the_kernel() {
        use crate::journal::Journal;
        use gyroship_runtime::mem::{Block, Memory, perm, state};
        use std::sync::{Arc, Mutex};
        const RING: u64 = crate::layout::ALIAS.0;
        let mut k = Kernel::new(Memory::new().unwrap(), Journal::memory(), crate::Config { play: true, ..crate::Config::default() });
        k.mem.map(RING, 0x2000, Block::new(state::NORMAL, perm::RW)).unwrap();
        let (w, r) = k.create_event().unwrap();
        k.handles.close(w);
        let event = k.event_of(r).unwrap();
        // A tick every 50 ms from now, on an empty ring.
        k.timers.push(Timer { start: real_ticks(), k: 1, num: 50 * 31_250, den: 1, action: TimerAction::AudioSink { ring: RING, slots: 9, data: RING + 0x1040, channels: 1, event } });
        let ticks = Arc::new(Mutex::new(Vec::new()));
        let heard = ticks.clone();
        k.audio_out = Some(Box::new(move |_: &[f32], _: f32| {
            heard.lock().unwrap().push(Instant::now());
            0
        }));
        let timers = spawn_timers(crate::install(k));
        // Once the first tick's scope is over (the sink sleeps), the kernel is held through the next tick.
        while crate::with(|k| k.timers[0].k) < 2 {
            std::thread::sleep(Duration::from_millis(1));
        }
        let held = crate::with(|_| {
            let from = Instant::now();
            std::thread::sleep(Duration::from_millis(80));
            from..Instant::now()
        });
        while ticks.lock().unwrap().len() < 3 {
            std::thread::sleep(Duration::from_millis(1));
        }
        crate::with(|k| k.sched.outcome = Some(Outcome::Finished));
        for t in timers {
            t.join().unwrap();
        }
        let k = crate::uninstall().unwrap();
        assert!(ticks.lock().unwrap().iter().any(|t| held.contains(t)), "no sound while the kernel was held");
        assert!(k.events[event].signaled);
    }
}
