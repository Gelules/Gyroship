//! High-level emulation of OS services: SVCs, IPC services, fs (romfs), hid, audio, time; network always offline.
//!
//! The kernel state lives in a thread-local: everything guest runs on one
//! host thread (D19), and each test thread gets a kernel of its own. A borrow
//! of it must never stay open across a guest fiber switch.

pub mod journal;
pub mod svc;

use gyroship_runtime::Cpu;
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
}

impl Default for Config {
    fn default() -> Config {
        Config { seed: 0x5EED, explore: false, non_system_memory: 0x2_6410_0000 }
    }
}

pub struct Kernel {
    pub mem: Memory,
    pub journal: Journal,
    pub config: Config,
    /// The running guest thread's id (the main thread is 1).
    pub thread: u64,
}

/// Exit code of a run the HLE stops (an unknown or forbidden SVC, a guest
/// abort).
pub const STOP_EXIT: i32 = 0x48;

impl Kernel {
    pub fn new(mem: Memory, mut journal: Journal, config: Config) -> Kernel {
        let mode = if config.explore { "exploration (not a validation run)" } else { "strict" };
        journal.line(0, &format!("gyroship hle: mode {mode}, seed {:#x}, non-system memory {:#x}", config.seed, config.non_system_memory));
        Kernel { mem, journal, config, thread: 1 }
    }

    /// Logs a line for the running thread.
    pub fn log(&mut self, text: &str) {
        self.journal.line(self.thread, text);
    }

    /// Ends the run: logs why, flushes the journal, exits with `code`.
    pub fn stop(&mut self, code: i32, why: &str) -> ! {
        self.log(why);
        self.journal.flush();
        eprintln!("gyroship: {why}");
        std::process::exit(code);
    }
}

thread_local! {
    static KERNEL: RefCell<Option<Kernel>> = const { RefCell::new(None) };
}

/// Makes `kernel` the one this thread's guest code talks to.
pub fn install(kernel: Kernel) {
    KERNEL.with(|k| *k.borrow_mut() = Some(kernel));
    gyroship_runtime::set_svc_handler(on_svc);
}

/// Runs `f` on this thread's kernel.
pub fn with<R>(f: impl FnOnce(&mut Kernel) -> R) -> R {
    KERNEL.with(|k| f(k.borrow_mut().as_mut().expect("no kernel installed on this thread")))
}

fn on_svc(cpu: &mut Cpu, pc: u64, imm: u32) {
    with(|k| k.svc(cpu, pc, imm));
}
