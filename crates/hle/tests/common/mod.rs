//! Helpers for tests whose "guest code" is Rust with the generated
//! signature: it makes SVCs through the runtime, so each test goes through
//! the real hooks and fiber switches.

#![allow(dead_code)]

use gyroship_hle::journal::Journal;
use gyroship_hle::thread::{self, Outcome};
use gyroship_hle::{Config, Kernel};
use gyroship_runtime::mem::{Block, Memory, perm, state};
use gyroship_runtime::{self as rt, Cpu};
use std::cell::RefCell;
use std::sync::Once;

pub const PC: u64 = 0x1000;
/// A data page for the guest threads' words.
pub const DATA: u64 = gyroship_hle::layout::ALIAS.0;

thread_local! {
    /// What the guest threads did: (what, a value).
    pub static EVENTS: RefCell<Vec<(&'static str, u64)>> = const { RefCell::new(Vec::new()) };
}

/// svc #imm with x0.. = args; returns x0 (x1 and x2 stay in the context).
pub unsafe fn svc(cpu: *mut Cpu, imm: u32, args: &[u64]) -> u64 {
    unsafe {
        for (i, a) in args.iter().enumerate() {
            (*cpu).x[i] = *a;
        }
        rt::gyroship_svc(cpu, PC, imm);
        (*cpu).x[0]
    }
}

/// Reads the virtual counter.
pub unsafe fn now(cpu: *mut Cpu) -> u64 {
    unsafe {
        rt::gyroship_counter(cpu, PC, 9);
        (*cpu).x[9]
    }
}

pub fn record(what: &'static str, v: u64) {
    EVENTS.with(|e| e.borrow_mut().push((what, v)));
}

/// Records `what` at the current virtual time.
pub unsafe fn event(cpu: *mut Cpu, what: &'static str) {
    let t = unsafe { now(cpu) };
    record(what, t);
}

/// RET: the thread's first function returns to EXIT, and the thread ends.
pub unsafe fn ret(cpu: *mut Cpu) {
    unsafe { (*cpu).pc = (*cpu).x[30] };
}

/// Creates and starts a thread at `entry`; returns its handle.
pub unsafe fn spawn(cpu: *mut Cpu, entry: u64, arg: u64, priority: u64, core: u64) -> u64 {
    unsafe {
        assert_eq!(svc(cpu, 0x08, &[0, entry, arg, 0, priority, core]), 0, "CreateThread");
        let h = (*cpu).x[1];
        assert_eq!(svc(cpu, 0x09, &[h]), 0, "StartThread");
        h
    }
}

pub unsafe fn load(cpu: *mut Cpu, addr: u64) -> u32 {
    unsafe { rt::ld32(&*cpu, addr) as u32 }
}

pub unsafe fn store(cpu: *mut Cpu, addr: u64, v: u32) {
    unsafe { rt::st32(&*cpu, addr, u64::from(v)) };
}

/// Installs the test binary's guest functions, once.
pub fn install(functions: &[(u64, rt::Guest)]) {
    static ONCE: Once = Once::new();
    ONCE.call_once(|| rt::install(functions.iter().map(|&(addr, function)| rt::Entry { addr, len: 0x100, function }).collect()));
}

/// Runs the main thread at `entry` to the end; returns how it ended, the
/// events and the journal.
pub fn run(entry: u64) -> (Outcome, Vec<(&'static str, u64)>, String) {
    EVENTS.with(|e| e.borrow_mut().clear());
    let mut k = Kernel::new(Memory::new().unwrap(), Journal::memory(), Config::default());
    k.journal.all_svcs = true;
    k.mem.map(DATA, 0x1000, Block::new(state::NORMAL, perm::RW)).unwrap();
    k.create_main(entry, 0);
    gyroship_hle::install(k);
    let outcome = thread::run();
    let k = gyroship_hle::uninstall().unwrap();
    (outcome, EVENTS.with(|e| e.borrow().clone()), k.journal.text().to_string())
}
