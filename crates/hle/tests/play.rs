//! The play engine (D41): guest threads on host threads of their own, through
//! the real hooks. Guest code is Rust with the generated signature; results
//! go to guest words (several host threads write them), and BAD counts what
//! must never happen.

mod common;

use common::*;
use gyroship_hle::svc::CURRENT_THREAD;
use gyroship_hle::thread::Outcome;
use gyroship_hle::{Config, STOP_EXIT};
use gyroship_runtime::{self as rt, Cpu};
use std::sync::atomic::{AtomicU32, Ordering::SeqCst};
use std::time::{Duration, Instant};

const BAD: u64 = DATA;
const ARRIVED: u64 = DATA + 4;
const HELD: u64 = DATA + 8;
const RELEASED: u64 = DATA + 0xc;
/// A handle list of one, for WaitSynchronization.
const HANDLES: u64 = DATA + 0x100;

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
