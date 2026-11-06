//! Synchronization between guest threads.

mod common;

use common::*;
use gyroship_hle::result::*;
use gyroship_hle::svc::CURRENT_THREAD;
use gyroship_hle::sync::HAS_LISTENERS;
use gyroship_hle::thread::Outcome;
use gyroship_runtime::Cpu;

const TURN: u64 = DATA;
const MUTEX: u64 = DATA + 0x10;
const KEY: u64 = DATA + 0x14;
const READY: u64 = DATA + 0x18;
/// A thread handle, for a thread that needs its own.
const OTHER: u64 = DATA + 0x20;
/// Handle lists for WaitSynchronization.
const LIST: u64 = DATA + 0x100;
const FOREVER: u64 = u64::MAX; // -1

/// Waits until the word at `addr` is `want`, as a futex.
unsafe fn wait_for(cpu: *mut Cpu, addr: u64, want: u32) {
    unsafe {
        loop {
            let v = load(cpu, addr);
            if v == want {
                return;
            }
            svc(cpu, 0x34, &[addr, 2, u64::from(v), FOREVER]);
        }
    }
}

/// Waits for the end of the thread whose handle is `h`.
unsafe fn join(cpu: *mut Cpu, h: u64) {
    unsafe {
        store(cpu, LIST, h as u32);
        assert_eq!(svc(cpu, 0x18, &[0, LIST, 1, FOREVER]), 0, "WaitSynchronization");
        assert_eq!((*cpu).x[1], 0, "index");
    }
}

/// The userspace half of a Horizon mutex.
unsafe fn lock(cpu: *mut Cpu, tag: u32) {
    unsafe {
        loop {
            let v = load(cpu, MUTEX);
            if v == 0 {
                store(cpu, MUTEX, tag);
                return;
            }
            store(cpu, MUTEX, v | HAS_LISTENERS);
            svc(cpu, 0x1A, &[u64::from(v & !HAS_LISTENERS), MUTEX, u64::from(tag)]);
            if load(cpu, MUTEX) & !HAS_LISTENERS == tag {
                return;
            }
        }
    }
}

unsafe fn unlock(cpu: *mut Cpu) {
    unsafe {
        if load(cpu, MUTEX) & HAS_LISTENERS != 0 {
            svc(cpu, 0x1B, &[MUTEX]);
        } else {
            store(cpu, MUTEX, 0);
        }
    }
}

// Ping-pong through WaitForAddress and SignalToAddress.
const PING: u64 = 0x1000;
const PONG: u64 = 0x1100;
unsafe extern "C" fn ping(cpu: *mut Cpu, _: u64) {
    unsafe {
        let h = spawn(cpu, PONG, 0, 44, 1);
        for _ in 0..1000 {
            wait_for(cpu, TURN, 0);
            store(cpu, TURN, 1);
            svc(cpu, 0x35, &[TURN, 0, 0, 1]);
        }
        record("ping rounds", 1000);
        join(cpu, h);
        ret(cpu);
    }
}
unsafe extern "C" fn pong(cpu: *mut Cpu, _: u64) {
    unsafe {
        for _ in 0..1000 {
            wait_for(cpu, TURN, 1);
            store(cpu, TURN, 0);
            svc(cpu, 0x35, &[TURN, 0, 0, 1]);
        }
        record("pong rounds", 1000);
        ret(cpu);
    }
}

// A condition variable.
const CONSUMER: u64 = 0x1200;
const PRODUCER: u64 = 0x1300;
unsafe extern "C" fn consumer(cpu: *mut Cpu, _: u64) {
    unsafe {
        let me = (*cpu).x[1] as u32;
        assert_eq!(svc(cpu, 0x08, &[0, PRODUCER, 0, 0, 44, 1]), 0);
        let h = (*cpu).x[1];
        store(cpu, OTHER, h as u32);
        svc(cpu, 0x09, &[h]);
        lock(cpu, me);
        while load(cpu, READY) == 0 {
            assert_eq!(svc(cpu, 0x1C, &[MUTEX, KEY, u64::from(me), FOREVER]), 0);
        }
        record("consumer holds the mutex", u64::from(load(cpu, MUTEX) & !HAS_LISTENERS == me));
        unlock(cpu);
        join(cpu, h);
        record("key after", u64::from(load(cpu, KEY)));
        ret(cpu);
    }
}
unsafe extern "C" fn producer(cpu: *mut Cpu, _: u64) {
    unsafe {
        let me = load(cpu, OTHER);
        svc(cpu, 0x0B, &[5000]);
        lock(cpu, me);
        store(cpu, READY, 1);
        svc(cpu, 0x1D, &[KEY, 1]);
        unlock(cpu);
        record("produced", 1);
        ret(cpu);
    }
}

// Results without a partner.
const ALONE: u64 = 0x1400;
unsafe extern "C" fn alone(cpu: *mut Cpu, _: u64) {
    unsafe {
        store(cpu, TURN, 7);
        let t0 = now(cpu);
        record("timed out", svc(cpu, 0x34, &[TURN, 2, 7, 10_000]));
        record("waited", now(cpu) - t0);
        record("changed", svc(cpu, 0x34, &[TURN, 2, 8, FOREVER]));
        // Decrement and wait if less than, without waiting.
        store(cpu, TURN, 5);
        record("decrement", svc(cpu, 0x34, &[TURN, 1, 6, 0]));
        record("decremented", u64::from(load(cpu, TURN)));
        record("increment if equal (no)", svc(cpu, 0x35, &[TURN, 1, 3, 1]));
        record("increment if equal", svc(cpu, 0x35, &[TURN, 1, 4, 1]));
        // Modify by waiters: none wait, so value + 1.
        record("modify", svc(cpu, 0x35, &[TURN, 2, 5, 1]));
        record("modified", u64::from(load(cpu, TURN)));
        // A cancel before the wait makes it return at once.
        svc(cpu, 0x19, &[u64::from(CURRENT_THREAD as u32)]);
        record("cancelled", svc(cpu, 0x18, &[0, LIST, 0, FOREVER]));
        record("not cancelled again", svc(cpu, 0x18, &[0, LIST, 0, 0]));
        ret(cpu);
    }
}

// A thread that nothing will wake.
const STUCK: u64 = 0x1500;
unsafe extern "C" fn stuck(cpu: *mut Cpu, _: u64) {
    unsafe {
        svc(cpu, 0x34, &[TURN, 2, 0, FOREVER]);
        ret(cpu);
    }
}

// Events.
const SIGNALER: u64 = 0x1600;
const WAITER: u64 = 0x1700;
unsafe extern "C" fn signaler(cpu: *mut Cpu, _: u64) {
    unsafe {
        let (w, r) = gyroship_hle::with(|k| k.create_event()).unwrap();
        store(cpu, LIST + 0x10, r);
        let h = spawn(cpu, WAITER, 0, 44, 1);
        svc(cpu, 0x0B, &[2000]);
        record("signal", svc(cpu, 0x11, &[u64::from(w)]));
        join(cpu, h);
        record("reset", svc(cpu, 0x17, &[u64::from(r)]));
        record("reset again", svc(cpu, 0x17, &[u64::from(r)]));
        ret(cpu);
    }
}
unsafe extern "C" fn waiter(cpu: *mut Cpu, _: u64) {
    unsafe {
        let r = svc(cpu, 0x18, &[0, LIST + 0x10, 1, FOREVER]);
        record("woken", r);
        record("index", (*cpu).x[1]);
        ret(cpu);
    }
}

fn go(entry: u64) -> (Outcome, Vec<(&'static str, u64)>, String) {
    install(&[
        (PING, ping),
        (PONG, pong),
        (CONSUMER, consumer),
        (PRODUCER, producer),
        (ALONE, alone),
        (STUCK, stuck),
        (SIGNALER, signaler),
        (WAITER, waiter),
    ]);
    run(entry)
}

#[test]
fn ping_pong_through_addresses() {
    let (outcome, mut events, journal) = go(PING);
    assert_eq!(outcome, Outcome::Finished, "{journal}");
    events.sort();
    assert_eq!(events, [("ping rounds", 1000), ("pong rounds", 1000)]);
    assert!(journal.matches("svc 0x34 WaitForAddress").count() >= 1000, "the threads waited");
    assert_eq!(journal, go(PING).2, "two runs, one journal");
}

#[test]
fn a_condition_variable_hands_the_mutex_back() {
    let (outcome, mut events, journal) = go(CONSUMER);
    assert_eq!(outcome, Outcome::Finished, "{journal}");
    events.sort();
    assert_eq!(events, [("consumer holds the mutex", 1), ("key after", 0), ("produced", 1)], "{journal}");
    // The consumer really waited on the condition, and got the mutex from the arbiter.
    for svc in ["svc 0x1c WaitProcessWideKeyAtomic", "svc 0x1d SignalProcessWideKey", "svc 0x1b ArbitrateUnlock"] {
        assert!(journal.contains(svc), "{svc}: {journal}");
    }
}

#[test]
fn results_the_sdk_expects() {
    let (outcome, events, journal) = go(ALONE);
    assert_eq!(outcome, Outcome::Finished, "{journal}");
    let get = |what| events.iter().find(|e| e.0 == what).map(|e| e.1).unwrap_or_else(|| panic!("{what}: {events:?}"));
    assert_eq!(get("timed out"), u64::from(TIMED_OUT));
    assert_eq!(TIMED_OUT, 0xEA01);
    assert!(get("waited") >= 10_000 / 32, "{events:?}");
    assert_eq!(get("changed"), u64::from(INVALID_STATE));
    assert_eq!(INVALID_STATE, 0xFA01);
    assert_eq!((get("decrement"), get("decremented")), (u64::from(TIMED_OUT), 4));
    assert_eq!(get("increment if equal (no)"), u64::from(INVALID_STATE));
    assert_eq!(get("increment if equal"), 0);
    assert_eq!((get("modify"), get("modified")), (0, 6));
    assert_eq!(get("cancelled"), u64::from(CANCELLED));
    assert_eq!(CANCELLED, 0xEC01);
    assert_eq!(get("not cancelled again"), u64::from(TIMED_OUT));
}

#[test]
fn a_thread_nothing_wakes_is_a_deadlock() {
    let (outcome, _, journal) = go(STUCK);
    assert_eq!(outcome, Outcome::Deadlock("thread 1 (Waiting)".to_string()), "{journal}");
}

#[test]
fn events_wake_their_waiters() {
    let (outcome, events, journal) = go(SIGNALER);
    assert_eq!(outcome, Outcome::Finished, "{journal}");
    let mut events = events;
    events.sort();
    assert_eq!(events, [("index", 0), ("reset", 0), ("reset again", u64::from(INVALID_STATE)), ("signal", 0), ("woken", 0)], "{journal}");
}
