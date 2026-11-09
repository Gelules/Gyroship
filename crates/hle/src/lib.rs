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
pub mod opus;
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
    /// `--work-time`: guest instructions cost virtual time (D40), so that loads
    /// spread over frames; off, only switch points do (D19).
    pub work_time: bool,
    /// `--dump-gpfifo DIR`: the first submits' pushbuffer segments go to DIR, raw.
    pub gpfifo_dump: Option<std::path::PathBuf>,
    /// `--dump-frames DIR`: at each present, the render targets the frame
    /// touched go to DIR as PNG files.
    pub frames_dump: Option<std::path::PathBuf>,
    /// `--dump-every K`: only every K-th frame is dumped (all are rendered).
    pub dump_every: u64,
    /// `--dump-draws N`: in frame N, the targets of each draw are dumped after it.
    pub dump_draws: Option<u64>,
    /// `--dump-shaders DIR`: the code of every program a draw uses, once, as
    /// `{key}.bin`.
    pub shaders_dump: Option<std::path::PathBuf>,
    /// `--replay FILE`: the controller's states by hid sample (D29).
    pub replay: Option<Vec<(u64, hid::Step)>>,
    /// `--profile FILE` and `--profile-from N`: sample the host thread from
    /// present N on; the report goes to FILE when the run ends.
    pub profile: Option<(std::path::PathBuf, u64)>,
    /// `--perf FILE`: each present's host wall time, as CSV (never journaled).
    pub perf: Option<std::path::PathBuf>,
    /// `--hashes FILE`: a hash of each presented image, as CSV (never journaled).
    pub hashes: Option<std::path::PathBuf>,
    /// `--record FILE`: the controller's states, in `--replay`'s format.
    pub record: Option<std::path::PathBuf>,
    /// `--dump-audio FILE`: every frame the game plays, as a float WAV.
    pub audio_dump: Option<std::path::PathBuf>,
}

impl Default for Config {
    fn default() -> Config {
        Config { seed: 0x5EED, explore: false, non_system_memory: 0x2_6410_0000, time_limit: None, frames: None, save_dir: None, vsync: true, work_time: false, gpfifo_dump: None, frames_dump: None, dump_every: 1, dump_draws: None, shaders_dump: None, replay: None, profile: None, perf: None, hashes: None, record: None, audio_dump: None }
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
    /// Where the controller's replay script stands.
    pub input: hid::Input,
    /// Spin detector: each thread's counter reads and waits since its last SVC.
    pub spin: Vec<u32>,
    /// Switch points by thread and kind, for `--perf` (host statistics).
    pub census: Census,
    /// Open saves, by key (user id and save type).
    pub saves: std::collections::BTreeMap<String, save::SaveFs>,
    /// Virtual timers (the audio sink).
    pub timers: Vec<thread::Timer>,
    /// The host render targets, when something renders (`--dump-frames`).
    pub surfaces: Option<gyroship_gpu::surfaces::Surfaces>,
    /// The GPU channels and what they ran.
    pub gpu: nvdrv::Gpu,
    /// Normal activity, by name (submits, draws…), for the summary line.
    pub counts: std::collections::BTreeMap<&'static str, u64>,
    /// Named anomalies seen (D32), for the summary line.
    pub anomalies: std::collections::BTreeMap<&'static str, u64>,
    /// `--hashes`: the presents with no image to hash (the others come back from the GPU).
    pub frame_hashes: Vec<(u64, Option<(u64, u64)>)>,
    /// The host window's pace hook and its last size (`boot::HostWindow`).
    pub window: Option<(Pace, (u32, u32))>,
    /// The host's live controller state, read at each hid sample.
    pub live_input: Option<LiveInput>,
    /// `--record`: the script lines so far.
    pub recorded: Option<Vec<String>>,
    /// The host's sound output: each sink tick's frames, and the stream's gain.
    pub audio_out: Option<AudioOut>,
    /// `--dump-audio`: every frame played so far.
    pub audio_samples: Option<Vec<f32>>,
    /// hwopus decoders, by manager (its results ring) and instance.
    pub opus: std::collections::BTreeMap<(u64, u16), opus::Decoder>,
    /// The stream's gain (SetStreamGain), and its channels and rate.
    pub audio_gain: f32,
    pub audio_format: (u16, u32),
}

/// The host's sound output (`boot::HostWindow::audio`): interleaved f32 frames and the gain.
pub type AudioOut = Box<dyn FnMut(&[f32], f32) + Send>;

/// A host window's pace hook (`boot::HostWindow::pace`).
pub type Pace = Box<dyn FnMut(u64) -> Option<(u32, u32)> + Send>;

/// The host's live controller state (`boot::HostWindow::input`).
pub type LiveInput = Box<dyn FnMut() -> hid::Pad + Send>;

/// Exit code of a run the HLE stops (an unknown or forbidden SVC, a guest
/// abort, an anomaly in strict mode).
pub const STOP_EXIT: i32 = 0x48;

/// Anomalies strict mode lets pass for now, each until a known fix (D32).
const TOLERATED: &[&str] = &[];

impl Kernel {
    pub fn new(mem: Memory, mut journal: Journal, config: Config) -> Kernel {
        let mode = if config.explore { "exploration (not a validation run)" } else { "strict" };
        journal.line(0, 0, &format!("gyroship hle: mode {mode}, seed {:#x}, non-system memory {:#x}", config.seed, config.non_system_memory));
        let recorded = config.record.as_ref().map(|_| Vec::new());
        let audio_samples = config.audio_dump.as_ref().map(|_| Vec::new());
        Kernel { mem, journal, config, handles: thread::Handles::new(), sched: thread::Sched::default(), events: Vec::new(), oracle: None, sessions: Vec::new(), objects: Vec::new(), domains: Vec::new(), allowed_services: Vec::new(), shared: Vec::new(), transfers: Vec::new(), time_shared: None, frame: (0, 0, 0), symbols: Vec::new(), romfs: None, gs: gs::State::default(), hid_shared: None, input: Default::default(), spin: Vec::new(), census: Census::default(), saves: Default::default(), timers: Vec::new(), surfaces: None, gpu: Default::default(), counts: Default::default(), anomalies: Default::default(), frame_hashes: Vec::new(), window: None, live_input: None, recorded, audio_out: None, audio_samples, opus: Default::default(), audio_gain: 1.0, audio_format: (2, 48_000) }
    }

    /// Counts one more `name` (normal activity); returns the new count.
    pub fn count(&mut self, name: &'static str) -> u64 {
        let c = self.counts.entry(name).or_default();
        *c += 1;
        *c
    }

    /// A named anomaly (D32): counted, journaled the first times, and fatal
    /// in strict mode unless tolerated.
    pub fn anomaly(&mut self, name: &'static str, text: &str) {
        *self.anomalies.entry(name).or_default() += 1;
        let line = format!("anomaly {name}: {text}");
        if !self.config.explore && !TOLERATED.contains(&name) {
            self.stop(STOP_EXIT, &line);
        }
        if self.journal.few(name, 4) {
            self.log(&line);
        }
    }

    /// One line: presents, activity, then the anomalies (or none).
    pub fn summary(&self) -> String {
        let list = |m: &std::collections::BTreeMap<&str, u64>| m.iter().map(|(n, c)| format!("{n} {c}")).collect::<Vec<_>>().join(", ");
        let anomalies = if self.anomalies.is_empty() { "none".to_string() } else { list(&self.anomalies) };
        let host = self.surfaces.as_ref().map_or(String::new(), |s| format!("; host: {}", s.cache_sizes()));
        format!("summary: {} presents; activity: {}; anomalies: {anomalies}{host}", self.gs.presents, list(&self.counts))
    }

    /// Logs a line for the running thread (0 when none runs).
    pub fn log(&mut self, text: &str) {
        let id = self.sched.current.map_or(0, |t| self.sched.threads[t].id);
        self.journal.line(self.now(), id, text);
    }

    /// Ends the run: logs why, flushes the journal, exits with `code`.
    pub fn stop(&mut self, code: i32, why: &str) -> ! {
        self.log(why);
        let summary = self.summary();
        self.log(&summary);
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
        self.host_reports();
        eprintln!("gyroship: {why}");
        std::process::exit(code);
    }

    /// Writes the host-side reports a run asked for (`--profile`, `--perf`).
    pub fn host_reports(&mut self) {
        gyroship_runtime::profile::finish(&|a| self.symbolize(a));
        if let Some(path) = &self.config.perf {
            let (gpu, stalls) = self.surfaces.as_mut().map_or((Vec::new(), 0), |s| s.frame_times());
            gyroship_runtime::profile::write_perf(path, self.config.profile.as_ref().map_or(600, |p| p.1 as usize), &gpu);
            if stalls > 0 {
                eprintln!("gyroship: GPU frame times: {stalls} flushes waited for the GPU to free a timing slot");
            }
            if let Some(s) = &self.surfaces {
                eprintln!("gyroship: staging over {} presents: {:?}", self.gs.presents, s.staging);
            }
            // Host time by guest thread, most first (HLE work on a thread's behalf included).
            let total: f64 = self.sched.threads.iter().map(|t| t.host.as_secs_f64()).sum();
            let mut by: Vec<(usize, f64)> = self.sched.threads.iter().enumerate().map(|(i, t)| (i + 1, t.host.as_secs_f64())).collect();
            by.sort_by(|a, b| b.1.total_cmp(&a.1));
            let rows: Vec<String> = by.iter().filter(|r| r.1 > 0.0005).map(|(i, s)| format!("thread {i} {:.1} s ({:.1}%)", s, 100.0 * s / total.max(1e-9))).collect();
            eprintln!("gyroship: host time by guest thread over {} presents: {}", self.gs.presents, rows.join(", "));
            // Guest instructions by thread: per present, and per second of the thread's host time.
            let presents = self.gs.presents.max(1) as f64;
            let mut work: Vec<(u64, u64, f64)> = self.sched.threads.iter().map(|t| (t.id, t.instructions, t.host.as_secs_f64())).filter(|w| w.1 > 0).collect();
            work.sort_by_key(|w| std::cmp::Reverse(w.1));
            let all: u64 = work.iter().map(|w| w.1).sum();
            let rows: Vec<String> = work.iter().take(12).map(|&(id, n, s)| format!("thread {id} {:.0} M ({:.2} M/present, {:.0} M/s)", n as f64 / 1e6, n as f64 / presents / 1e6, n as f64 / s.max(1e-9) / 1e6)).collect();
            eprintln!("gyroship: guest instructions over {} presents: {:.0} M in all; {}", self.gs.presents, all as f64 / 1e6, rows.join(", "));
            let census = path.with_extension("census.txt");
            if let Err(e) = std::fs::write(&census, self.census_report()) {
                eprintln!("gyroship: --perf {}: {e}", census.display());
            }
            for (name, header, rows) in [("rewrites.csv", "present,gpu,size\n", &self.census.rewrites), ("romfs.csv", "present,offset,size\n", &self.census.romfs)] {
                let file = path.with_extension(name);
                let csv: String = std::iter::once(header.to_string()).chain(rows.iter().map(|(n, at, size)| format!("{n},{at:#x},{size:#x}\n"))).collect();
                if let Err(e) = std::fs::write(&file, csv) {
                    eprintln!("gyroship: --perf {}: {e}", file.display());
                }
            }
            let m = self.gpu.mme_stats();
            eprintln!(
                "gyroship: macros over {} presents: {} calls, {:.1}% repeating their macro's last parameters without reading registers or memory; {} zero-count LOOPs; {} taken branches with bit 14 ({} with an END pending)",
                self.gs.presents,
                m.calls,
                100.0 * m.repeats as f64 / m.calls.max(1) as f64,
                m.zero_loops,
                m.bit14,
                m.bit14_end
            );
        }
        if let (Some(path), Some(samples)) = (&self.config.audio_dump, &self.audio_samples)
            && let Err(e) = path.parent().map_or(Ok(()), std::fs::create_dir_all).and_then(|()| std::fs::write(path, audio::wav(samples, self.audio_format.0, self.audio_format.1)))
        {
            eprintln!("gyroship: --dump-audio {}: {e}", path.display());
        }
        if let (Some(path), Some(lines)) = (&self.config.record, &self.recorded) {
            let end = format!("{} end", self.input.last + 1);
            let text = lines.iter().chain(std::iter::once(&end)).cloned().collect::<Vec<_>>().join("
") + "
";
            if let Err(e) = std::fs::write(path, text) {
                eprintln!("gyroship: --record {}: {e}", path.display());
            }
        }
        if let Some(path) = &self.config.hashes {
            let mut rows = std::mem::take(&mut self.frame_hashes);
            if let Some(s) = self.surfaces.as_mut() {
                rows.extend(s.presented_hashes().into_iter().map(|(n, target, hash)| (n, Some((target, hash)))));
            }
            rows.sort_unstable();
            let lines = rows.iter().map(|(n, h)| match h {
                Some((target, hash)) => format!("{n},{target:#x},{hash:016x}"),
                None => format!("{n},,"),
            });
            let csv = std::iter::once("present,target,hash".to_string()).chain(lines).collect::<Vec<_>>().join("\n") + "\n";
            if let Err(e) = path.parent().map_or(Ok(()), std::fs::create_dir_all).and_then(|()| std::fs::write(path, csv)) {
                eprintln!("gyroship: --hashes {}: {e}", path.display());
            }
        }
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
        k.take_work(cpu);
        k.census(cpu, imm, None);
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

/// Guest instructions a virtual tick (31.25 MHz) costs with `--work-time`:
/// 48 is 1.5 billion a second per core, an estimate for the console's
/// Cortex-A78C cores near 1 GHz on game code (not measured on hardware).
const WORK_PER_TICK: u64 = 48;

/// Switch points by guest thread and kind, counter reads and waits by site,
/// and each thread's frame chain at its first switch point: `--perf` host
/// statistics, never journaled and not guest state.
#[derive(Default)]
pub struct Census {
    /// (thread id, kind) -> count; a kind is an SVC number, [`COUNTER`], or [`WAIT`] plus the hint.
    by_kind: std::collections::HashMap<(u64, u32), u64>,
    /// (thread id, pc) -> count, for counter reads and waits.
    sites: std::collections::HashMap<(u64, u64), u64>,
    /// Thread id -> its frame chain at its first switch point (its body is at the end).
    first: std::collections::BTreeMap<u64, Vec<String>>,
    /// Textures found rewritten by the CPU: (present, GPU address, size).
    rewrites: Vec<(u64, u64, u64)>,
    /// RomFS reads: (present, offset, size).
    romfs: Vec<(u64, u64, u64)>,
}

const COUNTER: u32 = 0x100;
const WAIT: u32 = 0x200;

impl Kernel {
    /// Takes the guest instructions the running thread ran since its last
    /// switch point (the generated code counts them in its context). With
    /// `--work-time` they cost its core virtual time, at WORK_PER_TICK a tick:
    /// a long computation (a load) then spreads over frames as on the
    /// console, instead of running between two of them.
    fn take_work(&mut self, cpu: *mut Cpu) {
        // SAFETY: the running thread's context, stopped in the hook.
        let work = unsafe { std::mem::take(&mut (*cpu).work) };
        let Some(t) = self.sched.current else { return };
        let th = &mut self.sched.threads[t];
        th.instructions += work;
        if self.config.work_time {
            th.work_rem += work;
            let ticks = th.work_rem / WORK_PER_TICK;
            th.work_rem %= WORK_PER_TICK;
            self.sched.clocks[th.core] += ticks;
        }
    }

    /// Counts a switch point of the running thread for `--perf`; `pc` is
    /// given for counter reads and waits.
    fn census(&mut self, cpu: *mut Cpu, kind: u32, pc: Option<u64>) {
        let (Some(_), Some(t)) = (&self.config.perf, self.sched.current) else { return };
        let id = self.sched.threads[t].id;
        *self.census.by_kind.entry((id, kind)).or_insert(0) += 1;
        if let Some(pc) = pc {
            *self.census.sites.entry((id, pc)).or_insert(0) += 1;
        }
        if !self.census.first.contains_key(&id) {
            // SAFETY: the running thread's context, stopped in the hook.
            let (lr, fp) = unsafe { ((*cpu).x[30], (*cpu).x[29]) };
            let chain = self.backtrace_from((lr, lr, fp)).into_iter().skip(1).collect();
            self.census.first.insert(id, chain);
        }
    }

    /// The census as text: switch points per present by thread and kind, the
    /// busiest counter-read and wait sites, and each thread's first frame chain.
    fn census_report(&self) -> String {
        use std::fmt::Write as _;
        let presents = self.gs.presents.max(1) as f64;
        let name = |k: u32| match k {
            COUNTER => "counter".to_string(),
            k if k > WAIT => ["yield", "wfe", "wfi"].get((k - WAIT - 1) as usize).map_or(format!("hint {k:#x}"), |s| s.to_string()),
            k => format!("svc {k:#x}"),
        };
        let mut by_thread: std::collections::BTreeMap<u64, Vec<(u32, u64)>> = Default::default();
        for (&(id, kind), &n) in &self.census.by_kind {
            by_thread.entry(id).or_default().push((kind, n));
        }
        let mut out = format!("# switch points per present ({} presents), by guest thread and kind\n", self.gs.presents);
        for (id, mut kinds) in by_thread {
            kinds.sort_by(|a, b| b.1.cmp(&a.1).then(a.0.cmp(&b.0)));
            let total: u64 = kinds.iter().map(|k| k.1).sum();
            let top: Vec<String> = kinds.iter().take(8).map(|&(k, n)| format!("{} {:.1}", name(k), n as f64 / presents)).collect();
            let _ = writeln!(out, "thread {id}: {:.1} per present: {}", total as f64 / presents, top.join(", "));
        }
        let mut sites: Vec<(&(u64, u64), &u64)> = self.census.sites.iter().collect();
        sites.sort_by(|a, b| b.1.cmp(a.1).then(a.0.cmp(b.0)));
        out += "# counter reads and waits per present, by site\n";
        for (&(id, pc), &n) in sites.into_iter().take(30) {
            let _ = writeln!(out, "{:8.1}  thread {id}  {pc:#x} {}", n as f64 / presents, self.symbolize(pc));
        }
        out += "# each thread's frame chain at its first switch point (its body is last)\n";
        for (id, chain) in &self.census.first {
            let _ = writeln!(out, "thread {id}:");
            for line in chain {
                let _ = writeln!(out, "{line}");
            }
        }
        out
    }

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

unsafe fn on_counter(cpu: *mut Cpu, pc: u64, rt: u32) {
    let next = with(|k| {
        k.tick();
        k.take_work(cpu);
        k.census(cpu, COUNTER, Some(pc));
        k.count_spin(cpu, "counter reads or waits");
        if rt < 31 {
            // SAFETY: as above.
            unsafe { (*cpu).x[rt as usize] = k.now() };
        }
        k.reschedule()
    });
    thread::switch_to(next);
}

unsafe fn on_wait(cpu: *mut Cpu, pc: u64, kind: u32) {
    let next = with(|k| {
        k.tick();
        k.take_work(cpu);
        k.census(cpu, WAIT + kind, Some(pc));
        k.count_spin(cpu, "counter reads or waits");
        k.reschedule()
    });
    thread::switch_to(next);
}
