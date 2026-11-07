//! High-level emulation of OS services: SVCs, IPC services, fs (romfs), hid, audio, time; network always offline.
//!
//! The kernel state lives in a thread-local: everything guest runs on one
//! host thread (D19), and each test thread gets a kernel of its own. A borrow
//! of it must never stay open across a guest fiber switch.

pub mod account;
pub mod am;
pub mod audio;
pub mod boot;
pub mod bsd;
pub mod fs;
pub mod gs;
pub mod hid;
pub mod ipc;
pub mod journal;
pub mod nvdrv;
pub mod save;
pub mod service;
pub mod shared;
pub mod svc;
pub mod sync;
pub mod thread;
pub mod time;

use gyroship_runtime::{Cpu, Hooks};
use gyroship_runtime::mem::Memory;
use journal::Journal;
use std::cell::RefCell;

/// Horizon result codes of the kernel module (1); the description is the
/// public one from switchbrew.
pub mod result {
    const fn kernel(desc: u32) -> u32 {
        1 | desc << 9
    }
    pub const SUCCESS: u32 = 0;
    pub const NOT_IMPLEMENTED: u32 = kernel(33);
    pub const OUT_OF_HANDLES: u32 = kernel(105);
    pub const INVALID_PRIORITY: u32 = kernel(112);
    pub const INVALID_CORE_ID: u32 = kernel(113);
    pub const INVALID_SIZE: u32 = kernel(101);
    pub const INVALID_ADDRESS: u32 = kernel(102);
    pub const INVALID_CURRENT_MEMORY: u32 = kernel(106);
    pub const INVALID_NEW_MEMORY_PERMISSION: u32 = kernel(108);
    pub const INVALID_MEMORY_REGION: u32 = kernel(110);
    pub const INVALID_HANDLE: u32 = kernel(114);
    pub const INVALID_POINTER: u32 = kernel(115);
    pub const INVALID_COMBINATION: u32 = kernel(116);
    pub const TIMED_OUT: u32 = kernel(117);
    pub const CANCELLED: u32 = kernel(118);
    pub const INVALID_ENUM_VALUE: u32 = kernel(120);
    pub const NOT_FOUND: u32 = kernel(121);
    pub const SESSION_CLOSED: u32 = kernel(123);
    pub const OUT_OF_RANGE: u32 = kernel(119);
    pub const INVALID_STATE: u32 = kernel(125);
}

/// The fixed regions of the 39-bit address space (start, size), aligned to
/// 2 MiB and below the D12 module bases. What GetInfo reports.
pub mod layout {
    pub const ASLR: (u64, u64) = (0x00_0800_0000, 0x7F_F800_0000);
    /// Never used: the game takes its memory from the alias region (VAMM).
    pub const HEAP: (u64, u64) = (0x08_0000_0000, 8 << 30);
    pub const ALIAS: (u64, u64) = (0x10_0000_0000, 64 << 30);
    pub const STACK: (u64, u64) = (0x20_0000_0000, 2 << 30);
    /// Thread-local regions and the process-local region; not reported.
    pub const THREAD_LOCAL: (u64, u64) = (0x30_0000_0000, 64 << 30);
    /// Switch 2 region of GetInfo 0x1D/0x1E, with its own allocator in the SDK.
    pub const REMOTE: (u64, u64) = (0x40_0000_0000, 32 << 30);

    pub fn contains(region: (u64, u64), addr: u64, size: u64) -> bool {
        addr >= region.0 && addr.checked_add(size).is_some_and(|end| end <= region.0 + region.1)
    }
}

/// Values a run depends on, recorded in the journal header.
#[derive(Clone, Debug)]
pub struct Config {
    /// Source of GetInfo RandomEntropy (and later of csrng).
    pub seed: u64,
    /// Unknown SVCs and commands return NotImplemented instead of stopping:
    /// for collecting what is missing in one run, never for validation.
    pub explore: bool,
    /// GetInfo 0x15 TotalNonSystemMemorySize: 9 793 MiB, Lichii's value, the
    /// one known to reach the title screen (an arena of 128 MiB).
    pub non_system_memory: u64,
    /// Stop the run past this virtual time (ticks).
    pub time_limit: Option<u64>,
    /// End the run after this many presented frames.
    pub frames: Option<u64>,
    /// `--save DIR`: saves are loaded from and committed to DIR; otherwise
    /// every run starts from empty saves, kept in memory.
    pub save_dir: Option<std::path::PathBuf>,
    /// vi2 presents wait for the display's next vsync (60 Hz); `--no-vsync` turns it off.
    pub vsync: bool,
    /// `--dump-gpfifo DIR`: the first submits' pushbuffer segments go to DIR, raw.
    pub gpfifo_dump: Option<std::path::PathBuf>,
    /// `--dump-frames DIR`: at each present, the render targets the frame
    /// touched go to DIR as PNG files.
    pub frames_dump: Option<std::path::PathBuf>,
}

impl Default for Config {
    fn default() -> Config {
        Config { seed: 0x5EED, explore: false, non_system_memory: 0x2_6410_0000, time_limit: None, frames: None, save_dir: None, vsync: true, gpfifo_dump: None, frames_dump: None }
    }
}

pub struct Kernel {
    pub mem: Memory,
    pub journal: Journal,
    pub config: Config,
    pub handles: thread::Handles,
    pub sched: thread::Sched,
    pub events: Vec<sync::Event>,
    /// `link()`'s view of memory, compared at the oracle probe (boot).
    pub oracle: Option<gyroship_loader::link::Program>,
    pub sessions: Vec<Option<ipc::Session>>,
    /// The service objects sessions and domains refer to.
    pub objects: Vec<service::Service>,
    pub domains: Vec<ipc::Domain>,
    /// The services the NPDM grants (its service access control).
    pub allowed_services: Vec<String>,
    pub shared: Vec<shared::SharedMemory>,
    pub transfers: Vec<shared::TransferMemory>,
    /// time's shared memory, once a client asked for it.
    pub time_shared: Option<usize>,
    /// (pc, lr, fp) of the last SVC, for backtraces.
    pub frame: (u64, u64, u64),
    /// Exported function symbols of every module, sorted: (address, name).
    pub symbols: Vec<(u64, String)>,
    /// The game's RomFS, served by fsp-srv.
    pub romfs: Option<fs::RomFs>,
    /// State the graphics sessions share.
    pub gs: gs::State,
    /// hid's shared memory, once a client asked for it.
    pub hid_shared: Option<usize>,
    /// Spin detector: each thread's counter reads and waits since its last SVC.
    pub spin: Vec<u32>,
    /// Open saves, by key (user id and save type).
    pub saves: std::collections::BTreeMap<String, save::SaveFs>,
    /// Virtual timers (the audio sink).
    pub timers: Vec<thread::Timer>,
    /// The host render targets, when something renders (`--dump-frames`).
    pub surfaces: Option<gyroship_gpu::surfaces::Surfaces>,
}

/// Exit code of a run the HLE stops (an unknown or forbidden SVC, a guest
/// abort).
pub const STOP_EXIT: i32 = 0x48;

impl Kernel {
    pub fn new(mem: Memory, mut journal: Journal, config: Config) -> Kernel {
        let mode = if config.explore { "exploration (not a validation run)" } else { "strict" };
        journal.line(0, 0, &format!("gyroship hle: mode {mode}, seed {:#x}, non-system memory {:#x}", config.seed, config.non_system_memory));
        Kernel { mem, journal, config, handles: thread::Handles::new(), sched: thread::Sched::default(), events: Vec::new(), oracle: None, sessions: Vec::new(), objects: Vec::new(), domains: Vec::new(), allowed_services: Vec::new(), shared: Vec::new(), transfers: Vec::new(), time_shared: None, frame: (0, 0, 0), symbols: Vec::new(), romfs: None, gs: gs::State::default(), hid_shared: None, spin: Vec::new(), saves: Default::default(), timers: Vec::new(), surfaces: None }
    }

    /// Logs a line for the running thread (0 when none runs).
    pub fn log(&mut self, text: &str) {
        let id = self.sched.current.map_or(0, |t| self.sched.threads[t].id);
        self.journal.line(self.now(), id, text);
    }

    /// Ends the run: logs why, flushes the journal, exits with `code`.
    pub fn stop(&mut self, code: i32, why: &str) -> ! {
        self.log(why);
        for line in self.backtrace() {
            self.log(&line);
        }
        // The running thread's own lr and frame chain (the stop may come from one of its hooks).
        if let Some(t) = self.sched.current {
            // SAFETY: the running thread's context; it is stopped in this hook.
            let (lr, fp) = unsafe { ((*self.sched.threads[t].cpu).x[30], (*self.sched.threads[t].cpu).x[29]) };
            self.log("  running thread (lr, then the frame chain):");
            for line in self.backtrace_from((lr, lr, fp)).into_iter().skip(1) {
                self.log(&line);
            }
        }
        // How many times guest code passed each probe (e.g. the engine stages' ticks).
        for (addr, name, n) in gyroship_runtime::probe_counts() {
            if n > 0 {
                self.log(&format!("  probe {name} at {addr:#x}: {n} passes"));
            }
        }
        // Every thread: who holds the cores, who waits.
        for i in 0..self.sched.threads.len() {
            let t = &self.sched.threads[i];
            let line = format!("  thread {} {:?} priority {} core {} affinity {:#x} cpu time {} wait {:?}", t.id, t.state, t.priority, t.core, t.affinity, t.cpu_time, t.wait);
            self.log(&line);
        }
        self.journal.flush();
        eprintln!("gyroship: {why}");
        std::process::exit(code);
    }

    /// The guest call stack at the last SVC: its pc, its lr, then the frame
    /// chain (x29 -> [previous x29, return address]), each with the nearest
    /// exported symbol.
    pub fn backtrace(&self) -> Vec<String> {
        self.backtrace_from(self.frame)
    }

    /// The same, from a given (pc, lr, x29).
    pub fn backtrace_from(&self, (pc, lr, mut fp): (u64, u64, u64)) -> Vec<String> {
        let mut addrs = vec![pc, lr];
        for _ in 0..24 {
            let mut b = [0; 16];
            if fp == 0 || self.mem.read(fp, &mut b).is_err() {
                break;
            }
            let (next, ret) = (u64::from_le_bytes(b[0..8].try_into().unwrap()), u64::from_le_bytes(b[8..16].try_into().unwrap()));
            addrs.push(ret);
            if next <= fp {
                break;
            }
            fp = next;
        }
        addrs.iter().enumerate().map(|(i, &a)| format!("  #{i} {a:#x} {}", self.symbolize(a))).collect()
    }

    /// `module+offset (symbol+offset)` for a guest address.
    pub fn symbolize(&self, a: u64) -> String {
        let module = gyroship_loader::link::LOAD_ORDER.iter().filter_map(|n| Some((*n, gyroship_loader::link::base_of(n)?))).filter(|(_, b)| *b <= a).max_by_key(|(_, b)| *b);
        let place = module.map_or(String::new(), |(n, b)| format!("{n}+{:#x}", a - b));
        let i = self.symbols.partition_point(|(s, _)| *s <= a);
        match i.checked_sub(1).map(|i| &self.symbols[i]) {
            Some((s, name)) if a - s < 0x10000 => format!("{place} ({name}+{:#x})", a - s),
            _ => place,
        }
    }
}

thread_local! {
    static KERNEL: RefCell<Option<Kernel>> = const { RefCell::new(None) };
}

/// Makes `kernel` the one this thread's guest code talks to.
pub fn install(kernel: Kernel) {
    KERNEL.with(|k| *k.borrow_mut() = Some(kernel));
    gyroship_runtime::set_hooks(Hooks { svc: on_svc, counter: on_counter, wait: on_wait, probe: boot::on_probe, fault: on_fault });
}

/// A guest fault: the message and the stack at the last SVC go to the
/// journal, which is flushed (the runtime exits next). Guest code faults
/// outside any kernel borrow; if one is held, nothing is written.
fn on_fault(what: &str) {
    KERNEL.with(|k| {
        if let Ok(mut k) = k.try_borrow_mut()
            && let Some(k) = k.as_mut()
        {
            k.log(what);
            // The faulting thread's lr and frame chain (registers live in its context).
            if let Some(t) = k.sched.current {
                // SAFETY: the running thread's context; its code is stopped in the fault.
                let (lr, fp) = unsafe { ((*k.sched.threads[t].cpu).x[30], (*k.sched.threads[t].cpu).x[29]) };
                k.log("stack at the fault (lr, then the frame chain):");
                for line in k.backtrace_from((lr, lr, fp)).into_iter().skip(1) {
                    k.log(&line);
                }
            }
            k.log("stack at the last SVC:");
            for line in k.backtrace() {
                k.log(&line);
            }
            k.journal.flush();
        }
    });
}

/// Takes this thread's kernel back.
pub fn uninstall() -> Option<Kernel> {
    KERNEL.with(|k| k.borrow_mut().take())
}

/// Runs `f` on this thread's kernel.
pub fn with<R>(f: impl FnOnce(&mut Kernel) -> R) -> R {
    KERNEL.with(|k| f(k.borrow_mut().as_mut().expect("no kernel installed on this thread")))
}

// Each hook releases the kernel before it switches fibers: the next thread
// borrows it too.

unsafe fn on_svc(cpu: *mut Cpu, pc: u64, imm: u32) {
    let next = with(|k| {
        k.tick();
        if let Some(n) = k.sched.current.and_then(|t| k.spin.get_mut(t)) {
            *n = 0;
        }
        // SAFETY: the running thread's context, for the time of the call.
        k.svc(unsafe { &mut *cpu }, pc, imm);
        k.reschedule()
    });
    thread::switch_to(next);
}

/// Counter reads without an SVC in between that mean a thread spins.
const SPIN_READS: u32 = 10_000;

impl Kernel {
    /// A thread that reads the counter or waits (yield, wfe, wfi) SPIN_READS
    /// times without an SVC spins on something: its frame chain goes to the
    /// journal, once per thread.
    fn count_spin(&mut self, cpu: *mut Cpu, what: &str) {
        let Some(t) = self.sched.current else { return };
        if self.spin.len() <= t {
            self.spin.resize(t + 1, 0);
        }
        self.spin[t] += 1;
        if self.spin[t] == SPIN_READS {
            let id = self.sched.current.map_or(0, |t| self.sched.threads[t].id);
            if self.journal.first(&format!("spin {id}")) {
                // SAFETY: the running thread's context, stopped in the hook.
                let (lr, fp) = unsafe { ((*cpu).x[30], (*cpu).x[29]) };
                self.log(&format!("thread {id} spins: {SPIN_READS} {what} without an SVC; lr, then its frame chain:"));
                for line in self.backtrace_from((lr, lr, fp)).into_iter().skip(1) {
                    self.log(&line);
                }
            }
        }
    }
}

unsafe fn on_counter(cpu: *mut Cpu, rt: u32) {
    let next = with(|k| {
        k.tick();
        k.count_spin(cpu, "counter reads or waits");
        if rt < 31 {
            // SAFETY: as above.
            unsafe { (*cpu).x[rt as usize] = k.now() };
        }
        k.reschedule()
    });
    thread::switch_to(next);
}

unsafe fn on_wait(cpu: *mut Cpu, _kind: u32) {
    let next = with(|k| {
        k.tick();
        k.count_spin(cpu, "counter reads or waits");
        k.reschedule()
    });
    thread::switch_to(next);
}
