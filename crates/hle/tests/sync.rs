//! Synchronization between guest threads.

mod common;

use common::*;
use gyroship_hle::result::*;
use gyroship_hle::svc::CURRENT_THREAD;
use gyroship_hle::sync::HAS_LISTENERS;
use gyroship_hle::thread::{Outcome, State, Wait};
use gyroship_hle::{Config, Kernel};
use gyroship_runtime::Cpu;
use std::sync::atomic::{AtomicBool, AtomicU32, Ordering::SeqCst};
use std::time::{Duration, Instant};

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

// A timer: the silent audio sink over a ring at DATA (read index +0x44,
// write index +0x1004), every 1000 ticks.
const TIMED: u64 = 0x1800;
const RING: u64 = DATA;
unsafe extern "C" fn timed(cpu: *mut Cpu, _: u64) {
    unsafe {
        use gyroship_hle::thread::{Timer, TimerAction};
        use gyroship_runtime::mem::{Block, perm, state};
        let start = now(cpu);
        let r = gyroship_hle::with(|k| {
            k.mem.map(DATA + 0x1000, 0x1000, Block::new(state::NORMAL, perm::RW)).unwrap();
            let (w, r) = k.create_event().unwrap();
            k.handles.close(w);
            let event = k.event_of(r).unwrap();
            k.timers.push(Timer { start, k: 1, num: 1000, den: 1, action: TimerAction::AudioSink { ring: RING, slots: 4, data: RING + 0x1040, channels: 2, event } });
            r
        });
        store(cpu, RING + 0x1004, 2);
        store(cpu, LIST + 0x20, r);
        record("start", start);
        record("woken", svc(cpu, 0x18, &[0, LIST + 0x20, 1, FOREVER]));
        event(cpu, "at");
        record("read index", u64::from(load(cpu, RING + 0x44)));
        ret(cpu);
    }
}

fn go(entry: u64) -> (Outcome, Vec<(&'static str, u64)>, String) {
    install(&[
        (TIMED, timed),
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
fn a_timer_wakes_its_waiter_at_its_deadline_then_the_run_ends() {
    let (outcome, events, journal) = go(TIMED);
    // With nobody left waiting for it, the timer does not keep the run alive.
    assert_eq!(outcome, Outcome::Finished, "{journal}");
    let get = |what| events.iter().find(|e| e.0 == what).map(|e| e.1).unwrap_or_else(|| panic!("{what}: {events:?}"));
    assert_eq!(get("woken"), 0);
    // Woken at the deadline (the read of the counter itself costs a step).
    let late = get("at") - (get("start") + 1000);
    assert!(late <= 64, "{events:?}");
    assert_eq!(get("read index"), 2, "the sink caught up with the write index");
    assert!(journal.contains("audio sink: tick 1 read index -> 2"), "{journal}");
}

#[test]
fn events_wake_their_waiters() {
    let (outcome, events, journal) = go(SIGNALER);
    assert_eq!(outcome, Outcome::Finished, "{journal}");
    let mut events = events;
    events.sort();
    assert_eq!(events, [("index", 0), ("reset", 0), ("reset again", u64::from(INVALID_STATE)), ("signal", 0), ("woken", 0)], "{journal}");
}

// Guest words that other host threads update while the kernel runs: the tests call
// the handlers from one host thread, with no fiber; the other host threads stand
// for guest code (LSE and STXR are host atomics since Stage 1).

/// A kernel with the data page mapped.
fn kernel() -> Kernel {
    use gyroship_hle::journal::Journal;
    use gyroship_runtime::mem::{Block, Memory, perm, state};
    let k = Kernel::new(Memory::new().unwrap(), Journal::memory(), Config::default());
    k.mem.map(DATA, 0x1000, Block::new(state::NORMAL, perm::RW)).unwrap();
    k
}

/// Guest word `addr`, as guest code on another host thread updates it.
fn word(k: &Kernel, addr: u64) -> &'static AtomicU32 {
    // SAFETY: an aligned word of a mapped page; the kernel outlives the test's uses.
    unsafe { AtomicU32::from_ptr(k.mem.base().add(addr as usize).cast()) }
}

/// A thread waiting on condition `key` after releasing `mutex`, as
/// WaitProcessWideKeyAtomic leaves it; returns its index.
fn cond_waiter(k: &mut Kernel, mutex: u64, key: u64, tag: u32) -> usize {
    k.create_thread(PC, 0, 0, 44, 0).unwrap();
    let t = k.sched.threads.len() - 1;
    k.sched.current = Some(t);
    assert_eq!(k.wait_key(mutex, key, tag, -1), SUCCESS);
    k.sched.current = None;
    t
}

/// WaitForAddress type 1 and SignalToAddress type 1 update a word that other host
/// threads add to with atomics, as the guest's LSE code does: no update is lost.
#[test]
fn address_updates_lose_nothing_to_other_host_threads() {
    const HOSTS: i32 = 3;
    const ADDS: i32 = 100_000;
    const ROUNDS: i32 = 100_000;
    let mut k = kernel();
    let w = word(&k, DATA);
    let mut incremented = 0;
    std::thread::scope(|s| {
        for _ in 0..HOSTS {
            s.spawn(move || {
                for _ in 0..ADDS {
                    w.fetch_add(1, SeqCst);
                }
            });
        }
        for _ in 0..ROUNDS {
            // Decrement if less than (always: the word stays far below i32::MAX), no wait.
            assert_eq!(k.wait_address(DATA, 1, i32::MAX as u64, 0), TIMED_OUT);
            // Increment if equal: InvalidState when a host thread added in between.
            match k.signal_address(DATA, 1, w.load(SeqCst) as i32, -1) {
                SUCCESS => incremented += 1,
                r => assert_eq!(r, INVALID_STATE),
            }
        }
    });
    assert_eq!(w.load(SeqCst) as i32, HOSTS * ADDS - ROUNDS + incremented);
}

/// SignalProcessWideKey takes the mutex word over in one atomic step while a guest
/// thread on another host thread locks and unlocks it by compare-exchange (the SDK's
/// fast paths): the waiter gets the mutex free, or queues behind its holder, whose
/// unlock then sees the has-listeners bit. Nothing else may come out.
#[test]
fn a_signal_takes_the_mutex_over_in_one_step() {
    const HOLDER: u32 = 0x100;
    const WAITER: u32 = 0x200;
    let mut k = kernel();
    for i in 0..200 {
        let (mutex, key) = (DATA + 8 * i, DATA + 8 * i + 4);
        let t = cond_waiter(&mut k, mutex, key, WAITER);
        let m = word(&k, mutex);
        let (running, stop) = (AtomicBool::new(false), AtomicBool::new(false));
        let holds = std::thread::scope(|s| {
            let guest = s.spawn(|| {
                let mut holds = false;
                while !holds && !stop.load(SeqCst) {
                    running.store(true, SeqCst);
                    // An unlock fails only if the kernel set the bit meanwhile: the
                    // holder keeps the mutex (the SDK would call ArbitrateUnlock).
                    holds = m.compare_exchange(0, HOLDER, SeqCst, SeqCst).is_ok() && m.compare_exchange(HOLDER, 0, SeqCst, SeqCst).is_err();
                }
                holds
            });
            while !running.load(SeqCst) {
                std::hint::spin_loop();
            }
            k.signal_key(key, 1);
            stop.store(true, SeqCst);
            guest.join().unwrap()
        });
        let th = &k.sched.threads[t];
        if th.state == State::Ready {
            assert_eq!((m.load(SeqCst), holds), (WAITER, false), "round {i}");
        } else {
            assert_eq!((m.load(SeqCst), &th.wait, holds), (HOLDER | HAS_LISTENERS, &Wait::Lock { addr: mutex, tag: WAITER }, true), "round {i}");
        }
        assert_eq!(word(&k, key).load(SeqCst), 0, "no waiter is left on the key");
    }
}

/// WaitProcessWideKeyAtomic sets the key before it frees the mutex, as Horizon does:
/// a guest thread that takes the mutex on another host thread, then reads the key,
/// finds it set (an SDK that signals only when it is set loses no wake-up).
#[test]
fn the_key_is_set_before_the_mutex_is_free() {
    const WAITER: u32 = 0x100;
    const TAKER: u32 = 0x200;
    let mut k = kernel();
    for i in 0..200 {
        let (mutex, key) = (DATA + 8 * i, DATA + 8 * i + 4);
        let (m, flag) = (word(&k, mutex), word(&k, key));
        m.store(WAITER, SeqCst);
        k.create_thread(PC, 0, 0, 44, 0).unwrap();
        k.sched.current = Some(k.sched.threads.len() - 1);
        let running = AtomicBool::new(false);
        let seen = std::thread::scope(|s| {
            let guest = s.spawn(|| {
                running.store(true, SeqCst);
                let deadline = Instant::now() + Duration::from_secs(10);
                while m.compare_exchange(0, TAKER, SeqCst, SeqCst).is_err() {
                    assert!(Instant::now() < deadline, "round {i}: the mutex was never freed");
                    std::hint::spin_loop();
                }
                flag.load(SeqCst)
            });
            while !running.load(SeqCst) {
                std::hint::spin_loop();
            }
            assert_eq!(k.wait_key(mutex, key, WAITER, -1), SUCCESS);
            guest.join().unwrap()
        });
        k.sched.current = None;
        assert_eq!(seen, 1, "round {i}");
    }
}

/// A failed WaitProcessWideKeyAtomic has the effects it had before the key moved
/// first: a mutex word it cannot write changes nothing, the key included; a bad
/// key still frees the mutex.
#[test]
fn a_failed_condition_wait_keeps_its_effects() {
    use gyroship_runtime::mem::{Block, perm, state};
    const OWNER: u32 = 0x100;
    let mut k = kernel();
    let read_only = DATA + 0x1000;
    k.mem.map(read_only, 0x1000, Block::new(state::NORMAL, perm::R)).unwrap();
    let (m, key) = (word(&k, MUTEX), word(&k, KEY));
    assert_eq!(k.wait_key(read_only, KEY, OWNER, -1), INVALID_CURRENT_MEMORY);
    assert_eq!(key.load(SeqCst), 0, "the key is untouched");
    for (bad, result) in [(KEY + 2, INVALID_ADDRESS), (read_only, INVALID_CURRENT_MEMORY)] {
        m.store(OWNER, SeqCst);
        assert_eq!(k.wait_key(MUTEX, bad, OWNER, -1), result);
        assert_eq!(m.load(SeqCst), 0, "the mutex is free");
    }
}
