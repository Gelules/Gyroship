//! Guest threads and the scheduler. The "guest code" here is Rust with the
//! generated signature, installed in the dispatch table; it makes SVCs
//! through the runtime, so each test goes through the real hooks and fiber
//! switches.

use gyroship_hle::journal::Journal;
use gyroship_hle::svc::CURRENT_THREAD;
use gyroship_hle::thread::{self, Outcome, STEP};
use gyroship_hle::{Config, Kernel};
use gyroship_runtime::mem::Memory;
use gyroship_runtime::{self as rt, Cpu};
use std::cell::{Cell, RefCell};
use std::sync::Once;

const PC: u64 = 0x1000;

thread_local! {
    /// What the guest threads did: (who, virtual time).
    static EVENTS: RefCell<Vec<(&'static str, u64)>> = const { RefCell::new(Vec::new()) };
    static FLAG: Cell<bool> = const { Cell::new(false) };
}

/// svc #imm with x0.. = args; returns x0 (and leaves x1, x2 in the context).
unsafe fn svc(cpu: *mut Cpu, imm: u32, args: &[u64]) -> u64 {
    unsafe {
        for (i, a) in args.iter().enumerate() {
            (*cpu).x[i] = *a;
        }
        rt::gyroship_svc(cpu, PC, imm);
        (*cpu).x[0]
    }
}

unsafe fn now(cpu: *mut Cpu) -> u64 {
    unsafe {
        rt::gyroship_counter(cpu, PC, 9);
        (*cpu).x[9]
    }
}

unsafe fn event(cpu: *mut Cpu, who: &'static str) {
    let t = unsafe { now(cpu) };
    EVENTS.with(|e| e.borrow_mut().push((who, t)));
}

/// RET: the thread's first function returns to EXIT, and the thread ends.
unsafe fn ret(cpu: *mut Cpu) {
    unsafe { (*cpu).pc = (*cpu).x[30] };
}

/// Creates and starts a thread at `entry`; returns its handle.
unsafe fn spawn(cpu: *mut Cpu, entry: u64, arg: u64, priority: u64, core: u64) -> u64 {
    unsafe {
        assert_eq!(svc(cpu, 0x08, &[0, entry, arg, 0, priority, core]), 0, "CreateThread");
        let h = (*cpu).x[1];
        assert_eq!(svc(cpu, 0x09, &[h]), 0, "StartThread");
        h
    }
}

// Case 1: two threads that sleep.
const SLEEPERS: u64 = 0x1000;
const SLEEPER: u64 = 0x1100;
unsafe extern "C" fn sleepers(cpu: *mut Cpu, _: u64) {
    unsafe {
        spawn(cpu, SLEEPER, 0, 44, 1);
        for _ in 0..3 {
            event(cpu, "main");
            svc(cpu, 0x0B, &[1000]);
        }
        ret(cpu);
    }
}
unsafe extern "C" fn sleeper(cpu: *mut Cpu, _: u64) {
    unsafe {
        for _ in 0..3 {
            event(cpu, "worker");
            svc(cpu, 0x0B, &[1500]);
        }
        ret(cpu);
    }
}

// Case 2: priorities on one core.
const HIGH: u64 = 0x1200;
const LOW: u64 = 0x1300;
unsafe extern "C" fn high(cpu: *mut Cpu, _: u64) {
    unsafe {
        spawn(cpu, LOW, 0, 50, 0);
        // A yield only gives way to the same priority.
        svc(cpu, 0x0B, &[0]);
        event(cpu, "high after yield");
        svc(cpu, 0x0B, &[1000]);
        event(cpu, "high after sleep");
        ret(cpu);
    }
}
unsafe extern "C" fn low(cpu: *mut Cpu, _: u64) {
    unsafe {
        event(cpu, "low");
        ret(cpu);
    }
}

// Case 3: a thread spins on another core's flag.
const SPINNER: u64 = 0x1400;
const SETTER: u64 = 0x1500;
unsafe extern "C" fn spinner(cpu: *mut Cpu, _: u64) {
    unsafe {
        FLAG.set(false);
        spawn(cpu, SETTER, 0, 45, 1);
        while !FLAG.get() {
            rt::gyroship_yield(cpu, PC, 1);
        }
        event(cpu, "spinner saw the flag");
        ret(cpu);
    }
}
unsafe extern "C" fn setter(cpu: *mut Cpu, _: u64) {
    unsafe {
        FLAG.set(true);
        event(cpu, "setter");
        ret(cpu);
    }
}

// Case 4: an idle core takes a thread that waits behind another.
const BUSY: u64 = 0x1600;
const MIGRANT: u64 = 0x1700;
unsafe extern "C" fn busy(cpu: *mut Cpu, _: u64) {
    unsafe {
        assert_eq!(svc(cpu, 0x08, &[0, MIGRANT, 0, 0, 44, 0]), 0);
        let h = (*cpu).x[1];
        assert_eq!(svc(cpu, 0x0F, &[h, u64::MAX - 1 /* -2 */, 0x3F]), 0, "SetThreadCoreMask");
        svc(cpu, 0x09, &[h]);
        for _ in 0..200 {
            now(cpu);
        }
        ret(cpu);
    }
}
unsafe extern "C" fn migrant(cpu: *mut Cpu, _: u64) {
    unsafe {
        let core = svc(cpu, 0x10, &[]);
        EVENTS.with(|e| e.borrow_mut().push(("migrant core", core)));
        ret(cpu);
    }
}

// Case 5: CPU time through TLR + 0x108 leaves out sleeping.
const TICKS: u64 = 0x1800;
unsafe extern "C" fn ticks(cpu: *mut Cpu, _: u64) {
    unsafe {
        let tlr = (*cpu).tpidrro;
        let tick_count = |cpu: *mut Cpu| now(cpu).wrapping_add(rt::ld64(&*cpu, tlr + 0x108));
        let before = tick_count(cpu);
        svc(cpu, 0x0B, &[100_000]);
        let after = tick_count(cpu);
        EVENTS.with(|e| e.borrow_mut().push(("cpu ticks", after.wrapping_sub(before))));
        EVENTS.with(|e| e.borrow_mut().push(("id", svc(cpu, 0x25, &[0, u64::from(CURRENT_THREAD as u32)]) | (*cpu).x[1] << 8)));
        ret(cpu);
    }
}

fn setup() {
    static ONCE: Once = Once::new();
    ONCE.call_once(|| {
        let f: [(u64, rt::Guest); 9] = [
            (SLEEPERS, sleepers),
            (SLEEPER, sleeper),
            (HIGH, high),
            (LOW, low),
            (SPINNER, spinner),
            (SETTER, setter),
            (BUSY, busy),
            (MIGRANT, migrant),
            (TICKS, ticks),
        ];
        rt::install(f.iter().map(|&(addr, function)| rt::Entry { addr, len: 0x100, function }).collect());
    });
}

/// Runs the main thread at `entry` to the end; returns how it ended and the events.
fn run(entry: u64) -> (Outcome, Vec<(&'static str, u64)>, String) {
    setup();
    EVENTS.with(|e| e.borrow_mut().clear());
    let mut k = Kernel::new(Memory::new().unwrap(), Journal::memory(), Config::default());
    k.journal.all_svcs = true;
    k.create_main(entry, 0);
    gyroship_hle::install(k);
    let outcome = thread::run();
    let k = gyroship_hle::uninstall().unwrap();
    (outcome, EVENTS.with(|e| e.borrow().clone()), k.journal.text().to_string())
}

#[test]
fn sleeping_threads_follow_virtual_time() {
    let (outcome, events, journal) = run(SLEEPERS);
    assert_eq!(outcome, Outcome::Finished, "{journal}");
    assert_eq!(events.len(), 6, "{events:?}");
    // Execution order is time order: each thread ran only between two sleeps.
    assert!(events.windows(2).all(|w| w[0].1 <= w[1].1), "{events:?}");
    let times = |who| events.iter().filter(|e| e.0 == who).map(|e| e.1).collect::<Vec<_>>();
    for (who, sleep) in [("main", 1000u64.div_ceil(32)), ("worker", 1500u64.div_ceil(32))] {
        assert!(times(who).windows(2).all(|w| w[1] - w[0] >= sleep), "{who}: {events:?}");
    }
    // Both threads are interleaved, not one after the other.
    assert_ne!(events[1].0, events[2].0, "{events:?}");
    // The same run gives the same journal.
    assert_eq!(journal, run(SLEEPERS).2);
}

#[test]
fn a_yield_gives_way_to_the_same_priority_only() {
    let (outcome, events, journal) = run(HIGH);
    assert_eq!(outcome, Outcome::Finished, "{journal}");
    let names: Vec<_> = events.iter().map(|e| e.0).collect();
    assert_eq!(names, ["high after yield", "low", "high after sleep"]);
}

#[test]
fn a_spinning_thread_lets_another_core_run() {
    let (outcome, events, journal) = run(SPINNER);
    assert_eq!(outcome, Outcome::Finished, "{journal}");
    let names: Vec<_> = events.iter().map(|e| e.0).collect();
    assert_eq!(names, ["setter", "spinner saw the flag"]);
}

#[test]
fn an_idle_core_takes_a_waiting_thread() {
    let (outcome, events, journal) = run(BUSY);
    assert_eq!(outcome, Outcome::Finished, "{journal}");
    assert_eq!(events, [("migrant core", 1)], "{journal}");
}

#[test]
fn thread_tick_count_leaves_out_sleep() {
    let (outcome, events, journal) = run(TICKS);
    assert_eq!(outcome, Outcome::Finished, "{journal}");
    let ticks = events[0].1;
    // 100 µs of sleep are 3125 ticks; the thread itself ran a few steps.
    assert!(ticks < 10 * STEP, "{ticks} ticks: {journal}");
    assert_eq!(events[1].1, 1 << 8, "GetThreadId of the main thread: success, id 1");
}
