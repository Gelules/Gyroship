//! The play engine (D41): guest threads on host threads of their own, through
//! the real hooks. Guest code is Rust with the generated signature; results
//! go to guest words (several host threads write them), and BAD counts what
//! must never happen.

mod common;

use common::*;
use gyroship_hle::result::{INVALID_STATE, SUCCESS, TIMED_OUT};
use gyroship_hle::svc::CURRENT_THREAD;
use gyroship_hle::sync::HAS_LISTENERS;
use gyroship_hle::thread::{Outcome, State, Wait};
use gyroship_hle::{Config, STOP_EXIT};
use gyroship_runtime::{self as rt, Cpu};
use std::sync::atomic::{AtomicU32, Ordering::SeqCst};
use std::time::{Duration, Instant};

const BAD: u64 = DATA;
const ARRIVED: u64 = DATA + 4;
const HELD: u64 = DATA + 8;
const RELEASED: u64 = DATA + 0xc;
const TURN: u64 = DATA + 0x10;
const MUTEX: u64 = DATA + 0x14;
const COUNTER: u64 = DATA + 0x18;
/// A word that stays 0, for timed waits nothing ends.
const ZERO: u64 = DATA + 0x1c;
/// The racer's round, the signaler's last round done, and the word they race on.
const GATE: u64 = DATA + 0x20;
const DONE: u64 = DATA + 0x24;
const SPOT: u64 = DATA + 0x28;
/// A handle list of one, for WaitSynchronization.
const HANDLES: u64 = DATA + 0x100;
/// Handles of started threads, by their argument.
const SLOTS: u64 = DATA + 0x200;
/// A timeout of -1: no deadline.
const FOREVER: u64 = u64::MAX;

const MAIN_AT_ONCE: u64 = 0x1000;
const WORKER_AT_ONCE: u64 = 0x1100;
const MAIN_CALLERS: u64 = 0x1200;
const WORKER_CALLERS: u64 = 0x1300;
const MAIN_ENDS: u64 = 0x1400;
const EXITER: u64 = 0x1500;
const RETURNER: u64 = 0x1600;
const MAIN_QUICK: u64 = 0x1700;
const CREATOR: u64 = 0x1800;
const MAIN_SPIN: u64 = 0x1900;
const SPINNER: u64 = 0x1a00;
const MAIN_LOCKED: u64 = 0x1b00;
const HOLDER: u64 = 0x1c00;
const PING: u64 = 0x1d00;
const PONG: u64 = 0x1e00;
const MAIN_ADDERS: u64 = 0x1f00;
const ADDER: u64 = 0x2000;
const TIMED: u64 = 0x2100;
const RACER: u64 = 0x2200;
const SIGNALER: u64 = 0x2300;
const MAIN_JOINS: u64 = 0x2400;
const YIELDER: u64 = 0x2500;

const FUNCTIONS: &[(u64, rt::Guest)] = &[
    (MAIN_AT_ONCE, main_at_once),
    (WORKER_AT_ONCE, worker_at_once),
    (MAIN_CALLERS, main_callers),
    (WORKER_CALLERS, worker_callers),
    (MAIN_ENDS, main_ends),
    (EXITER, exiter),
    (RETURNER, returner),
    (MAIN_QUICK, main_quick),
    (CREATOR, creator),
    (MAIN_SPIN, main_spin),
    (SPINNER, spinner),
    (MAIN_LOCKED, main_locked),
    (HOLDER, holder),
    (PING, ping),
    (PONG, pong),
    (MAIN_ADDERS, main_adders),
    (ADDER, adder),
    (TIMED, timed),
    (RACER, racer),
    (SIGNALER, signaler),
    (MAIN_JOINS, main_joins),
    (YIELDER, yielder),
];

/// A word of the test's data page, as host threads share it.
unsafe fn word(cpu: *mut Cpu, addr: u64) -> &'static AtomicU32 {
    // SAFETY: an aligned word of the data page, mapped for the kernel's life.
    unsafe { AtomicU32::from_ptr((*cpu).mem.add(addr as usize).cast()) }
}

unsafe fn bad(cpu: *mut Cpu) {
    unsafe { word(cpu, BAD).fetch_add(1, SeqCst) };
}

/// Waits for the end of thread `h` without blocking in the kernel:
/// WaitSynchronization with timeout 0, then a host yield; 10 s at most.
unsafe fn poll_join(cpu: *mut Cpu, h: u64) {
    unsafe {
        store(cpu, HANDLES, h as u32);
        let start = Instant::now();
        while svc(cpu, 0x18, &[0, HANDLES, 1, 0]) != 0 {
            if start.elapsed() > Duration::from_secs(10) {
                return bad(cpu);
            }
            std::thread::yield_now();
        }
    }
}

/// Runs the play engine from `entry` on a thread of its own, so that a hang
/// fails this test instead of the whole binary; returns how the run ended
/// and BAD.
fn play_within(entry: u64, config: Config) -> (Outcome, u32) {
    install(FUNCTIONS);
    let (tx, rx) = std::sync::mpsc::channel();
    std::thread::spawn(move || {
        let (outcome, k) = play(entry, config);
        let _ = tx.send((outcome, k.mem.load_u32(BAD).unwrap()));
    });
    rx.recv_timeout(Duration::from_secs(60)).expect("the play run hung or panicked")
}

unsafe extern "C" fn main_at_once(cpu: *mut Cpu, _: u64) {
    unsafe {
        let threads: Vec<u64> = (0..4).map(|c| spawn(cpu, WORKER_AT_ONCE, 0, 44, c)).collect();
        for h in threads {
            poll_join(cpu, h);
        }
        ret(cpu);
    }
}

unsafe extern "C" fn worker_at_once(cpu: *mut Cpu, _: u64) {
    unsafe {
        svc(cpu, 0x25, &[0, CURRENT_THREAD]);
        let name = format!("guest {}", (*cpu).x[1]);
        if std::thread::current().name() != Some(name.as_str()) {
            bad(cpu);
        }
        // The four of them at once: each spins here until all have arrived.
        word(cpu, ARRIVED).fetch_add(1, SeqCst);
        let start = Instant::now();
        while word(cpu, ARRIVED).load(SeqCst) < 4 {
            if start.elapsed() > Duration::from_secs(10) {
                bad(cpu);
                break;
            }
            rt::gyroship_yield(cpu, PC, 1);
        }
        ret(cpu);
    }
}

/// Guest threads run at once, each on a host thread named after it.
#[test]
fn guest_threads_run_at_once_on_host_threads_of_their_own() {
    assert_eq!(play_within(MAIN_AT_ONCE, Config::default()), (Outcome::Finished, 0));
}

unsafe extern "C" fn main_callers(cpu: *mut Cpu, _: u64) {
    unsafe {
        let threads: Vec<u64> = (0..4).map(|i| spawn(cpu, WORKER_CALLERS, 40 + i, 40 + i, i)).collect();
        for h in threads {
            poll_join(cpu, h);
        }
        ret(cpu);
    }
}

unsafe extern "C" fn worker_callers(cpu: *mut Cpu, _: u64) {
    unsafe {
        // Its argument is its priority.
        let priority = (*cpu).x[0];
        svc(cpu, 0x25, &[0, CURRENT_THREAD]);
        let id = (*cpu).x[1];
        for _ in 0..20_000 {
            svc(cpu, 0x0C, &[0, CURRENT_THREAD]);
            let p = (*cpu).x[1];
            svc(cpu, 0x25, &[0, CURRENT_THREAD]);
            if p != priority || (*cpu).x[1] != id {
                bad(cpu);
            }
        }
        ret(cpu);
    }
}

/// Each kernel scope runs for its caller: its own id and priority, through
/// the current-thread handle, while three other threads do the same.
#[test]
fn every_kernel_scope_runs_for_its_caller() {
    assert_eq!(play_within(MAIN_CALLERS, Config::default()), (Outcome::Finished, 0));
}

unsafe extern "C" fn main_ends(cpu: *mut Cpu, _: u64) {
    unsafe {
        let threads: Vec<u64> = (0..6).map(|i| spawn(cpu, if i < 3 { EXITER } else { RETURNER }, 0, 44, i % 4)).collect();
        for h in threads {
            poll_join(cpu, h);
        }
        ret(cpu);
    }
}

/// ExitThread `depth` frames down: nothing after it runs again.
#[inline(never)]
unsafe fn exit_from(cpu: *mut Cpu, depth: u32) {
    unsafe {
        if depth == 0 {
            svc(cpu, 0x0A, &[]);
        } else {
            exit_from(cpu, depth - 1);
        }
        bad(cpu);
    }
}

unsafe extern "C" fn exiter(cpu: *mut Cpu, _: u64) {
    unsafe {
        exit_from(cpu, 3);
        ret(cpu);
    }
}

unsafe extern "C" fn returner(cpu: *mut Cpu, _: u64) {
    unsafe { ret(cpu) }
}

/// A thread ends by ExitThread from nested frames, which never return, or
/// by returning to EXIT; its joiner sees both ends.
#[test]
fn threads_end_by_exit_thread_or_by_returning() {
    assert_eq!(play_within(MAIN_ENDS, Config::default()), (Outcome::Finished, 0));
}

unsafe extern "C" fn main_quick(cpu: *mut Cpu, _: u64) {
    unsafe {
        spawn(cpu, RETURNER, 0, 44, 1);
        spawn(cpu, RETURNER, 0, 44, 2);
        ret(cpu);
    }
}

/// The run ends with its last thread, whichever it is and however its end
/// meets the boot thread's wait.
#[test]
fn the_run_ends_with_its_last_guest_thread() {
    for _ in 0..50 {
        assert_eq!(play_within(MAIN_QUICK, Config::default()), (Outcome::Finished, 0));
    }
}

unsafe extern "C" fn creator(cpu: *mut Cpu, _: u64) {
    unsafe {
        if svc(cpu, 0x08, &[0, RETURNER, 0, 0, 44, 0]) != 0 {
            bad(cpu);
        }
        ret(cpu);
    }
}

/// A thread created and never started: the same outcome in both engines.
#[test]
fn a_thread_never_started_ends_the_run_as_in_the_deterministic_engine() {
    install(FUNCTIONS);
    let deadlock = Outcome::Deadlock("thread 2 (Created)".into());
    assert_eq!(run(CREATOR).0, deadlock);
    assert_eq!(play_within(CREATOR, Config::default()), (deadlock, 0));
}

unsafe extern "C" fn main_spin(cpu: *mut Cpu, _: u64) {
    unsafe {
        for core in 1..4 {
            spawn(cpu, SPINNER, 0, 44, core);
        }
        spinner(cpu, 0);
    }
}

unsafe extern "C" fn spinner(_: *mut Cpu, _: u64) {
    loop {
        std::hint::spin_loop();
    }
}

/// Only does something when started by `the_time_limit_stops_a_run_that_makes_no_svc`.
#[test]
fn time_limit_child() {
    if std::env::var("GYROSHIP_TIME_LIMIT_CHILD").is_err() {
        return;
    }
    std::thread::spawn(|| {
        std::thread::sleep(Duration::from_secs(20));
        std::process::exit(99);
    });
    install(FUNCTIONS);
    play(MAIN_SPIN, Config { time_limit: Some(31_250_000 * 3 / 10), ..Config::default() });
}

/// The boot thread is the watchdog: past the time limit the run stops, even
/// while no guest thread makes an SVC.
#[test]
fn the_time_limit_stops_a_run_that_makes_no_svc() {
    let start = Instant::now();
    let out = std::process::Command::new(std::env::current_exe().unwrap())
        .args(["--exact", "time_limit_child", "--nocapture", "--test-threads=1"])
        .env("GYROSHIP_TIME_LIMIT_CHILD", "1")
        .output()
        .unwrap();
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(out.status.code(), Some(STOP_EXIT), "{stderr}");
    assert!(stderr.contains("the run reached its time limit"), "{stderr}");
    assert!(start.elapsed() < Duration::from_secs(10), "{:?}", start.elapsed());
}

unsafe extern "C" fn main_locked(cpu: *mut Cpu, _: u64) {
    unsafe {
        let holder = spawn(cpu, HOLDER, 0, 44, 1);
        while word(cpu, HELD).load(SeqCst) == 0 {
            std::hint::spin_loop();
        }
        let mut last = 0;
        for i in 0..10_000 {
            let t = now(cpu);
            if t < last {
                bad(cpu);
            }
            last = t;
            rt::gyroship_yield(cpu, PC, 1 + i % 3);
        }
        // All of that while the other thread held the kernel.
        if word(cpu, RELEASED).load(SeqCst) != 0 {
            bad(cpu);
        }
        poll_join(cpu, holder);
        ret(cpu);
    }
}

unsafe extern "C" fn holder(cpu: *mut Cpu, _: u64) {
    unsafe {
        let (held, released) = (word(cpu, HELD), word(cpu, RELEASED));
        // Holding the kernel is the point here; production code never sleeps in a scope.
        gyroship_hle::with(|_| {
            held.store(1, SeqCst);
            std::thread::sleep(Duration::from_secs(1));
            released.store(1, SeqCst);
        });
        ret(cpu);
    }
}

/// Counter reads and hints take no kernel lock: they go on while another
/// thread holds it.
#[test]
fn play_hooks_never_wait_for_the_kernel() {
    assert_eq!(play_within(MAIN_LOCKED, Config::default()), (Outcome::Finished, 0));
}

/// CreateThread at `entry` with argument `arg`, its handle stored at
/// SLOTS + 4 × arg for the thread to read, then StartThread.
unsafe fn start(cpu: *mut Cpu, entry: u64, arg: u64, core: u64) -> u64 {
    unsafe {
        if svc(cpu, 0x08, &[0, entry, arg, 0, 44, core]) != 0 {
            bad(cpu);
        }
        let h = (*cpu).x[1];
        word(cpu, SLOTS + 4 * arg).store(h as u32, SeqCst);
        svc(cpu, 0x09, &[h]);
        h
    }
}

/// Waits in the kernel for the end of thread `h`: WaitSynchronization, no timeout.
unsafe fn join(cpu: *mut Cpu, h: u64) {
    unsafe {
        store(cpu, HANDLES, h as u32);
        if svc(cpu, 0x18, &[0, HANDLES, 1, FOREVER]) as u32 != SUCCESS || (*cpu).x[1] != 0 {
            bad(cpu);
        }
    }
}

/// Waits until the word at `addr` is `want`, as a futex: WaitForAddress
/// (wait if equal) on the value it saw.
unsafe fn wait_for(cpu: *mut Cpu, addr: u64, want: u32) {
    unsafe {
        loop {
            let v = word(cpu, addr).load(SeqCst);
            if v == want {
                return;
            }
            let r = svc(cpu, 0x34, &[addr, 2, u64::from(v), FOREVER]) as u32;
            if r != SUCCESS && r != INVALID_STATE {
                bad(cpu);
            }
        }
    }
}

const ROUNDS: u32 = 20_000;

unsafe extern "C" fn ping(cpu: *mut Cpu, _: u64) {
    unsafe {
        let pong = start(cpu, PONG, 0, 1);
        for _ in 0..ROUNDS {
            wait_for(cpu, TURN, 0);
            word(cpu, TURN).store(1, SeqCst);
            svc(cpu, 0x35, &[TURN, 0, 0, 1]);
        }
        join(cpu, pong);
        ret(cpu);
    }
}

unsafe extern "C" fn pong(cpu: *mut Cpu, _: u64) {
    unsafe {
        for _ in 0..ROUNDS {
            wait_for(cpu, TURN, 1);
            word(cpu, TURN).store(0, SeqCst);
            svc(cpu, 0x35, &[TURN, 0, 0, 1]);
        }
        ret(cpu);
    }
}

/// Two host threads hand a word back and forth through WaitForAddress and
/// SignalToAddress: a lost wake-up would hang them.
#[test]
fn no_wake_up_is_lost_between_host_threads() {
    assert_eq!(play_within(PING, Config::default()), (Outcome::Finished, 0));
}

/// The userspace half of a Horizon mutex, on host atomics: `tag` is the
/// caller's handle.
unsafe fn lock(cpu: *mut Cpu, tag: u32) {
    unsafe {
        let m = word(cpu, MUTEX);
        loop {
            let Err(v) = m.compare_exchange(0, tag, SeqCst, SeqCst) else { return };
            // Held: marked as waited on, then the kernel hands it over.
            if v & HAS_LISTENERS == 0 && m.compare_exchange(v, v | HAS_LISTENERS, SeqCst, SeqCst).is_err() {
                continue;
            }
            svc(cpu, 0x1A, &[u64::from(v & !HAS_LISTENERS), MUTEX, u64::from(tag)]);
            if m.load(SeqCst) & !HAS_LISTENERS == tag {
                return;
            }
        }
    }
}

unsafe fn unlock(cpu: *mut Cpu, tag: u32) {
    unsafe {
        // Waiters: the kernel hands it to the best one.
        if word(cpu, MUTEX).compare_exchange(tag, 0, SeqCst, SeqCst).is_err() {
            svc(cpu, 0x1B, &[MUTEX]);
        }
    }
}

unsafe extern "C" fn main_adders(cpu: *mut Cpu, _: u64) {
    unsafe {
        let threads: Vec<u64> = (0..4).map(|i| start(cpu, ADDER, i, i)).collect();
        for h in threads {
            join(cpu, h);
        }
        if load(cpu, COUNTER) != 4 * 20_000 || load(cpu, MUTEX) != 0 {
            bad(cpu);
        }
        ret(cpu);
    }
}

unsafe extern "C" fn adder(cpu: *mut Cpu, _: u64) {
    unsafe {
        let tag = word(cpu, SLOTS + 4 * (*cpu).x[0]).load(SeqCst);
        for _ in 0..20_000 {
            lock(cpu, tag);
            // A plain read and write on purpose: only the mutex keeps the threads apart.
            store(cpu, COUNTER, load(cpu, COUNTER) + 1);
            unlock(cpu, tag);
        }
        ret(cpu);
    }
}

/// A guest mutex, its waiters parked by ArbitrateLock and handed over by
/// ArbitrateUnlock: four host threads, no count lost.
#[test]
fn a_guest_mutex_excludes_across_host_threads() {
    assert_eq!(play_within(MAIN_ADDERS, Config::default()), (Outcome::Finished, 0));
}

unsafe extern "C" fn timed(cpu: *mut Cpu, _: u64) {
    unsafe {
        for ns in [100_000u64, 1_000_000, 2_500_000, 16_666_667, 33_280_000] {
            for sleep in [false, true] {
                let before = now(cpu);
                let r = if sleep { svc(cpu, 0x0B, &[ns]) } else { svc(cpu, 0x34, &[ZERO, 2, 0, ns]) } as u32;
                let elapsed = now(cpu) - before;
                let ticks = ns.div_ceil(32);
                eprintln!("play timing: {} of {ns} ns ended {:.3} ms late", if sleep { "a sleep" } else { "a timed wait" }, elapsed.saturating_sub(ticks) as f64 * 32e-6);
                // Never early; within 15 ms late even on a loaded machine.
                if r != if sleep { SUCCESS } else { TIMED_OUT } || elapsed < ticks || elapsed > ticks + 15 * 31_250 {
                    bad(cpu);
                }
            }
        }
        ret(cpu);
    }
}

/// Timed waits and sleeps end at their deadline, never before it.
#[test]
fn deadlines_end_waits_on_time() {
    assert_eq!(play_within(TIMED, Config::default()), (Outcome::Finished, 0));
}

const RACE_ROUNDS: u32 = 500;

unsafe extern "C" fn racer(cpu: *mut Cpu, _: u64) {
    unsafe {
        let signaler = start(cpu, SIGNALER, 0, 1);
        let (mut woken, mut timed_out) = (0, 0);
        for round in 1..=RACE_ROUNDS {
            word(cpu, GATE).store(round, SeqCst);
            match svc(cpu, 0x34, &[SPOT, 2, 0, 1_000_000]) as u32 {
                SUCCESS => woken += 1,
                TIMED_OUT => timed_out += 1,
                _ => bad(cpu),
            }
            // One winner: the thread's next scope finds nothing left of the wait.
            let (state, wait, deadline) = gyroship_hle::with(|k| {
                let th = &k.sched.threads[k.sched.current.unwrap()];
                (th.state, th.wait.clone(), th.deadline)
            });
            if state != State::Ready || wait != Wait::None || deadline.is_some() {
                bad(cpu);
            }
            while word(cpu, DONE).load(SeqCst) < round {
                std::hint::spin_loop();
            }
        }
        // The signal came both before and after the deadline.
        if woken == 0 || timed_out == 0 {
            bad(cpu);
        }
        join(cpu, signaler);
        ret(cpu);
    }
}

unsafe extern "C" fn signaler(cpu: *mut Cpu, _: u64) {
    unsafe {
        for round in 1..=RACE_ROUNDS {
            while word(cpu, GATE).load(SeqCst) < round {
                std::hint::spin_loop();
            }
            // 0 to 2 ms after the racer starts its 1 ms wait, by round.
            let at = Instant::now() + Duration::from_micros(u64::from(round * 397 % 2_000));
            while Instant::now() < at {
                std::hint::spin_loop();
            }
            svc(cpu, 0x35, &[SPOT, 0, 0, 1]);
            word(cpu, DONE).store(round, SeqCst);
        }
        ret(cpu);
    }
}

/// A timeout racing a wake: exactly one of them ends the wait.
#[test]
fn a_timeout_racing_a_wake_has_one_winner() {
    assert_eq!(play_within(RACER, Config::default()), (Outcome::Finished, 0));
}

unsafe extern "C" fn main_joins(cpu: *mut Cpu, _: u64) {
    unsafe {
        let (a, b) = (start(cpu, RETURNER, 0, 1), start(cpu, RETURNER, 1, 2));
        join(cpu, a);
        join(cpu, b);
        ret(cpu);
    }
}

/// Joins that block in the kernel, against threads that end at once: their
/// end races the wait, and no wake-up is lost.
#[test]
fn joins_that_block_lose_no_wake_up() {
    for _ in 0..200 {
        assert_eq!(play_within(MAIN_JOINS, Config::default()), (Outcome::Finished, 0));
    }
}

unsafe extern "C" fn yielder(cpu: *mut Cpu, _: u64) {
    unsafe {
        // SleepThread 0, -1 and -2: yields, which never wait.
        for i in 0..3_000u64 {
            if svc(cpu, 0x0B, &[0u64.wrapping_sub(i % 3)]) as u32 != SUCCESS {
                bad(cpu);
            }
        }
        ret(cpu);
    }
}

#[test]
fn yields_return_at_once() {
    assert_eq!(play_within(YIELDER, Config::default()), (Outcome::Finished, 0));
}
