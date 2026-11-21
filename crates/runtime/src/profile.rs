//! A sampling profiler for the host thread that runs the guest (`--profile`):
//! every millisecond another thread suspends it, reads its instruction
//! pointer and resumes it. At the end, dbghelp names each address from the
//! executable's PDB (recompiled functions are `f_<guest address>`), and a
//! report lists the samples by module, then by symbol, most first. A module
//! without a PDB (a driver) names its addresses after its nearest export: its
//! module name is the reliable part.
//!
//! The sampler never allocates nor locks while the thread is suspended (it
//! could hold the allocator's lock).

use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::Mutex;
use std::sync::atomic::{AtomicBool, AtomicPtr, AtomicU64, Ordering};
use std::thread::JoinHandle;

/// Samples by (guest thread, host instruction pointer, guest x30, host caller:
/// the first return address on the host stack, 0 if none).
type Samples = HashMap<(u64, u64, u64, u64), u64>;

struct Profiler {
    path: PathBuf,
    stop: &'static AtomicBool,
    sampler: JoinHandle<Samples>,
}

static PROFILER: Mutex<Option<Profiler>> = Mutex::new(None);
/// The interpreter counts its instructions (a profile of a guest thread).
static COUNTING: AtomicBool = AtomicBool::new(false);

/// The guest thread the guest's host thread runs, and its CPU context: the
/// sampler tags each sample with the thread and its x30 (the caller, in a
/// leaf function). Contexts live until the end of the run.
static GUEST_THREAD: AtomicU64 = AtomicU64::new(0);
static GUEST_CPU: AtomicPtr<crate::Cpu> = AtomicPtr::new(std::ptr::null_mut());

/// Guest thread `id`, whose context is `cpu`, runs from now on.
pub fn set_guest(id: u64, cpu: *mut crate::Cpu) {
    GUEST_THREAD.store(id, Ordering::Relaxed);
    GUEST_CPU.store(cpu, Ordering::Relaxed);
}

thread_local! {
    /// Interpreted instruction words and how often each ran (guest thread only).
    static WORDS: std::cell::RefCell<HashMap<u32, u64>> = std::cell::RefCell::new(HashMap::new());
}

/// Host wall-clock time of each present (`--perf`), never in the journal.
static FRAMES: Mutex<Vec<std::time::Instant>> = Mutex::new(Vec::new());

/// Marks a present.
pub fn mark_frame() {
    FRAMES.lock().unwrap_or_else(|e| e.into_inner()).push(std::time::Instant::now());
}

/// Writes each frame's wall time and the GPU time of its work (`gpu`, by
/// present; empty if not measured) to `path` (CSV: present, ms, gpu_ms) and
/// prints their median, 95th percentile and mean from present `from` on.
pub fn write_perf(path: &std::path::Path, from: usize, gpu: &[f64]) {
    let marks = std::mem::take(&mut *FRAMES.lock().unwrap_or_else(|e| e.into_inner()));
    let ms: Vec<f64> = marks.windows(2).map(|w| (w[1] - w[0]).as_secs_f64() * 1e3).collect();
    let mut csv = String::from("present,ms,gpu_ms\n");
    for (i, t) in ms.iter().enumerate() {
        csv += &format!("{},{t:.3},{:.3}\n", i + 2, gpu.get(i + 2).copied().unwrap_or(0.0));
    }
    if let Some(dir) = path.parent() {
        let _ = std::fs::create_dir_all(dir);
    }
    if let Err(e) = std::fs::write(path, csv) {
        eprintln!("gyroship: --perf {}: {e}", path.display());
    }
    let stats = |label: &str, values: &[f64]| {
        let mut steady: Vec<f64> = values.iter().copied().skip(from.saturating_sub(1)).collect();
        if steady.is_empty() {
            return;
        }
        steady.sort_by(f64::total_cmp);
        let at = |q: f64| steady[((steady.len() - 1) as f64 * q) as usize];
        let mean = steady.iter().sum::<f64>() / steady.len() as f64;
        eprintln!("gyroship: {label} from present {from}: {} frames, median {:.2} ms, p95 {:.2} ms, mean {mean:.2} ms", steady.len(), at(0.5), at(0.95));
    };
    stats("perf", &ms);
    // The GPU's own time, aligned with the wall times (index 0: present 2).
    if gpu.len() > 2 {
        stats("GPU time", &gpu[2..]);
    }
}

/// CPU time (user and kernel) of the host thread behind `handle`, which the
/// caller keeps open: `--perf` in the play engine, where threads run at once.
/// Windows counts it in clock ticks: exact enough for a run, not for a frame.
#[cfg(windows)]
pub fn cpu_time(handle: usize) -> Option<std::time::Duration> {
    use windows_sys::Win32::Foundation::FILETIME;
    use windows_sys::Win32::System::Threading::GetThreadTimes;
    let zero = FILETIME { dwLowDateTime: 0, dwHighDateTime: 0 };
    let (mut created, mut exited, mut kernel, mut user) = (zero, zero, zero, zero);
    // SAFETY: a thread handle the caller keeps open, and four FILETIMEs to fill.
    if unsafe { GetThreadTimes(handle as _, &mut created, &mut exited, &mut kernel, &mut user) } == 0 {
        return None;
    }
    let hundreds = |f: FILETIME| u64::from(f.dwHighDateTime) << 32 | u64::from(f.dwLowDateTime);
    Some(std::time::Duration::from_nanos((hundreds(kernel) + hundreds(user)) * 100))
}

/// The calling host thread's CPU time, as [`cpu_time`].
#[cfg(windows)]
pub fn own_cpu_time() -> Option<std::time::Duration> {
    // SAFETY: the calling thread's pseudo-handle, which needs no closing.
    cpu_time(unsafe { windows_sys::Win32::System::Threading::GetCurrentThread() } as usize)
}

#[cfg(not(windows))]
pub fn own_cpu_time() -> Option<std::time::Duration> {
    None
}

/// The calling host thread's Windows id, for [`stack_of`].
#[cfg(windows)]
pub fn thread_id() -> u32 {
    // SAFETY: reads the calling thread's id.
    unsafe { windows_sys::Win32::System::Threading::GetCurrentThreadId() }
}

#[cfg(not(windows))]
pub fn thread_id() -> u32 {
    0
}

/// Where host thread `tid` (a Windows thread id) stands, for a watchdog: its
/// instruction pointer, then the return addresses on its stack (words just
/// after a call instruction, up to 64 KiB above its stack pointer, innermost
/// first; a stale one may show up), symbolized. The thread is suspended only
/// to read its context and copy its stack: nothing allocates or locks meanwhile.
#[cfg(windows)]
pub fn stack_of(tid: u32) -> Vec<String> {
    use windows_sys::Win32::Foundation::CloseHandle;
    use windows_sys::Win32::System::Diagnostics::Debug::{CONTEXT, CONTEXT_CONTROL_AMD64, GetThreadContext};
    use windows_sys::Win32::System::Memory::{MEMORY_BASIC_INFORMATION, VirtualQuery};
    use windows_sys::Win32::System::Threading::{OpenThread, ResumeThread, SuspendThread, THREAD_GET_CONTEXT, THREAD_SUSPEND_RESUME};
    const WORDS: usize = 8192;
    let mut stack = vec![0u64; WORDS];
    #[repr(C, align(16))]
    struct Aligned(CONTEXT);
    // SAFETY: a zeroed CONTEXT is valid; only its control registers are asked for.
    let mut aligned: Aligned = unsafe { std::mem::zeroed() };
    aligned.0.ContextFlags = CONTEXT_CONTROL_AMD64;
    // SAFETY: opens a thread of this process; closed below.
    let handle = unsafe { OpenThread(THREAD_SUSPEND_RESUME | THREAD_GET_CONTEXT, 0, tid) };
    if handle.is_null() {
        return vec![format!("cannot open thread {tid}: {}", std::io::Error::last_os_error())];
    }
    let mut copied = None;
    // SAFETY: while the thread is suspended, its context is read and its stack
    // copied from its stack pointer to the end of that committed region at
    // most, into a buffer allocated before; then it resumes.
    unsafe {
        if SuspendThread(handle) != u32::MAX {
            if GetThreadContext(handle, &mut aligned.0) != 0 {
                let rsp = aligned.0.Rsp as usize;
                let mut info: MEMORY_BASIC_INFORMATION = std::mem::zeroed();
                let n = if VirtualQuery(rsp as *const _, &mut info, size_of::<MEMORY_BASIC_INFORMATION>()) != 0 { ((info.BaseAddress as usize + info.RegionSize - rsp) / 8).min(WORDS) } else { 0 };
                std::ptr::copy_nonoverlapping(rsp as *const u64, stack.as_mut_ptr(), n);
                copied = Some(n);
            }
            ResumeThread(handle);
        }
        CloseHandle(handle);
    }
    let Some(n) = copied else { return vec![format!("cannot read thread {tid}: {}", std::io::Error::last_os_error())] };
    let rip = aligned.0.Rip;
    let line = |a: u64| format!("{a:#x} [{}] {}", module(a), symbol(a));
    std::iter::once(line(rip)).chain(stack[..n].iter().filter(|&&w| after_call(w)).take(40).map(|&w| line(w))).collect()
}

#[cfg(not(windows))]
pub fn stack_of(_tid: u32) -> Vec<String> {
    vec!["no stack on this platform".into()]
}

/// Whether `w` is the address right after a call instruction in executable
/// memory (a return address): call rel32, or call through a register or
/// memory operand (the common encodings).
#[cfg(windows)]
fn after_call(w: u64) -> bool {
    use windows_sys::Win32::System::Memory::{MEM_COMMIT, MEMORY_BASIC_INFORMATION, PAGE_EXECUTE, PAGE_EXECUTE_READ, PAGE_EXECUTE_READWRITE, PAGE_EXECUTE_WRITECOPY, VirtualQuery};
    let at = w as usize;
    if at < 0x10000 {
        return false;
    }
    // SAFETY: a query; it reads nothing at `at`.
    let mut info: MEMORY_BASIC_INFORMATION = unsafe { std::mem::zeroed() };
    if unsafe { VirtualQuery(at as *const _, &mut info, size_of::<MEMORY_BASIC_INFORMATION>()) } == 0 {
        return false;
    }
    let exec = PAGE_EXECUTE | PAGE_EXECUTE_READ | PAGE_EXECUTE_READWRITE | PAGE_EXECUTE_WRITECOPY;
    if info.State != MEM_COMMIT || info.Protect & exec == 0 || at - 6 < info.BaseAddress as usize {
        return false;
    }
    // SAFETY: the 6 bytes before `at` lie in its committed executable region.
    let b: [u8; 6] = unsafe { std::ptr::read_unaligned((at - 6) as *const [u8; 6]) };
    b[1] == 0xE8
        || b[0] == 0xFF && (b[1] == 0x15 || b[1] & 0xF8 == 0x90)
        || b[2] == 0xFF && b[3] == 0x54
        || b[3] == 0xFF && b[4] & 0xF8 == 0x50
        || b[3] == 0x41 && b[4] == 0xFF && b[5] & 0xF8 == 0xD0
        || b[4] == 0xFF && (b[5] & 0xF8 == 0xD0 || b[5] & 0xF8 == 0x10)
}

/// Counts an instruction the interpreter runs, while a guest thread's profile runs.
#[inline]
pub fn count_interpreted(word: u32) {
    if COUNTING.load(Ordering::Relaxed) {
        WORDS.with(|w| *w.borrow_mut().entry(word).or_insert(0) += 1);
    }
}

/// Starts sampling the calling thread; the report goes to `path` at [`finish`].
/// With `count`, the interpreter counts its instructions meanwhile, a hash per
/// instruction on every guest thread: only for a profile of the guest thread
/// that runs them, not of a thread they would slow down.
#[cfg(windows)]
pub fn start(path: PathBuf, count: bool) {
    use windows_sys::Win32::System::Diagnostics::Debug::{CONTEXT, CONTEXT_CONTROL_AMD64, GetThreadContext};
    use windows_sys::Win32::System::Memory::{MEMORY_BASIC_INFORMATION, VirtualQuery};
    use windows_sys::Win32::System::Threading::{GetCurrentThreadId, OpenThread, ResumeThread, SuspendThread, THREAD_GET_CONTEXT, THREAD_SUSPEND_RESUME};
    // SAFETY: opens a handle to this thread; it stays open for the process's life.
    let handle = unsafe { OpenThread(THREAD_SUSPEND_RESUME | THREAD_GET_CONTEXT, 0, GetCurrentThreadId()) } as usize;
    if handle == 0 {
        eprintln!("gyroship: --profile: cannot open the thread");
        return;
    }
    let stop: &'static AtomicBool = Box::leak(Box::new(AtomicBool::new(false)));
    let sampler = std::thread::spawn(move || {
        let mut samples = HashMap::new();
        // GetThreadContext wants a 16-byte aligned CONTEXT.
        #[repr(C, align(16))]
        struct Aligned(CONTEXT);
        // SAFETY: a zeroed CONTEXT is valid; only its control registers are asked for.
        let mut aligned: Aligned = unsafe { std::mem::zeroed() };
        let context = &mut aligned.0;
        // The words at the top of the thread's stack, copied while it is suspended:
        // a leaf function's caller, or the current function's, is the first return
        // address among them (found after the thread resumes).
        let mut top = [0u64; 16];
        while !stop.load(Ordering::Relaxed) {
            std::thread::sleep(std::time::Duration::from_millis(1));
            context.ContextFlags = CONTEXT_CONTROL_AMD64;
            let mut words = 0;
            // SAFETY: the handle is valid; the thread resumes right after the reads.
            // The words copied lie between its stack pointer and the end of that
            // committed region.
            let read = unsafe {
                if SuspendThread(handle as _) == u32::MAX {
                    eprintln!("gyroship: --profile: SuspendThread: {}", std::io::Error::last_os_error());
                    break;
                }
                let ok = GetThreadContext(handle as _, context) != 0;
                if ok {
                    let rsp = context.Rsp as usize;
                    let mut info: MEMORY_BASIC_INFORMATION = std::mem::zeroed();
                    if VirtualQuery(rsp as *const _, &mut info, size_of::<MEMORY_BASIC_INFORMATION>()) != 0 {
                        words = ((info.BaseAddress as usize + info.RegionSize).saturating_sub(rsp) / 8).min(top.len());
                        std::ptr::copy_nonoverlapping(rsp as *const u64, top.as_mut_ptr(), words);
                    }
                }
                ResumeThread(handle as _);
                ok
            };
            if read {
                let cpu = GUEST_CPU.load(Ordering::Relaxed);
                // SAFETY: the context outlives the run; its thread is suspended or parked.
                let lr = if cpu.is_null() { 0 } else { unsafe { std::ptr::read_volatile(&raw const (*cpu).x[30]) } };
                let caller = top[..words].iter().copied().find(|&w| after_call(w)).unwrap_or(0);
                *samples.entry((GUEST_THREAD.load(Ordering::Relaxed), context.Rip, lr, caller)).or_insert(0u64) += 1;
            } else if samples.is_empty() {
                eprintln!("gyroship: --profile: GetThreadContext: {}", std::io::Error::last_os_error());
                break;
            }
        }
        samples
    });
    *PROFILER.lock().unwrap_or_else(|e| e.into_inner()) = Some(Profiler { path, stop, sampler });
    COUNTING.store(count, Ordering::Relaxed);
}

#[cfg(not(windows))]
pub fn start(_path: PathBuf, _count: bool) {
    eprintln!("gyroship: --profile is only implemented on Windows");
}

/// Stops sampling and writes the report; `guest` names a guest address
/// (for `f_` symbols). Does nothing if no profile runs.
pub fn finish(guest: &dyn Fn(u64) -> String) {
    let Some(p) = PROFILER.lock().unwrap_or_else(|e| e.into_inner()).take() else { return };
    p.stop.store(true, Ordering::Relaxed);
    COUNTING.store(false, Ordering::Relaxed);
    let Ok(samples) = p.sampler.join() else { return };
    let total: u64 = samples.values().sum();
    let pct = |n: u64| 100.0 * n as f64 / total.max(1) as f64;
    // A recompiled function's guest name, after its symbol.
    let guest_of = |name: &str| {
        let bare = name.split_once("] ").map_or(name, |(_, s)| s);
        bare.strip_prefix("f_").and_then(|a| u64::from_str_radix(a, 16).ok()).map_or(String::new(), |a| format!("  {}", guest(a)))
    };
    let mut names: HashMap<u64, (String, String)> = HashMap::new();
    let mut by_symbol: HashMap<String, u64> = HashMap::new();
    let mut by_module: HashMap<String, u64> = HashMap::new();
    let mut by_thread: HashMap<u64, HashMap<String, u64>> = HashMap::new();
    let mut callers: HashMap<String, HashMap<u64, u64>> = HashMap::new();
    let mut hosts: HashMap<String, HashMap<u64, u64>> = HashMap::new();
    for ((thread, rip, lr, caller), n) in samples {
        let (module, name) = names
            .entry(rip)
            .or_insert_with(|| {
                let m = module(rip);
                let s = format!("[{m}] {}", symbol(rip));
                (m, s)
            })
            .clone();
        *by_module.entry(module).or_insert(0) += n;
        *by_symbol.entry(name.clone()).or_insert(0) += n;
        *by_thread.entry(thread).or_default().entry(name.clone()).or_insert(0) += n;
        *hosts.entry(name.clone()).or_default().entry(caller).or_insert(0) += n;
        *callers.entry(name).or_default().entry(lr).or_insert(0) += n;
    }
    let mut out = format!("# gyroship profile: {total} samples of 1 ms of the guest's host thread\n# samples  percent  module\n");
    let mut modules: Vec<(String, u64)> = by_module.into_iter().collect();
    modules.sort_by(|a, b| b.1.cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
    for (name, n) in modules {
        out += &format!("{n:9} {:7.2}%  {name}\n", pct(n));
    }
    out += "# samples  percent  [module] symbol\n";
    let mut rows: Vec<(String, u64)> = by_symbol.into_iter().collect();
    rows.sort_by(|a, b| b.1.cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
    for (name, n) in &rows {
        out += &format!("{n:9} {:7.2}%  {name}{}\n", pct(*n), guest_of(name));
    }
    // By guest thread (0: the boot fiber, before any thread), then the
    // hottest symbols of each thread with 1% of the samples or more.
    let mut threads: Vec<(u64, u64, HashMap<String, u64>)> = by_thread.into_iter().map(|(t, m)| (t, m.values().sum(), m)).collect();
    threads.sort_by(|a, b| b.1.cmp(&a.1).then(a.0.cmp(&b.0)));
    out += "# samples  percent  guest thread\n";
    for (t, n, _) in &threads {
        out += &format!("{n:9} {:7.2}%  thread {t}\n", pct(*n));
    }
    // Down to 0.1% of the thread's own samples, so its time can be split by
    // category; the rest in one line.
    for (t, n, m) in threads.iter().filter(|r| pct(r.1) >= 1.0) {
        out += &format!("# thread {t}: {n} samples, its symbols down to 0.1% of them\n");
        let mut hot: Vec<(&String, &u64)> = m.iter().collect();
        hot.sort_by(|a, b| b.1.cmp(a.1).then_with(|| a.0.cmp(b.0)));
        let (shown, rest): (Vec<_>, Vec<_>) = hot.into_iter().partition(|(_, k)| **k * 1000 >= *n);
        for (name, k) in shown {
            out += &format!("{k:9} {:7.2}%  {name}{}\n", pct(*k), guest_of(name));
        }
        let rest: u64 = rest.iter().map(|(_, k)| **k).sum();
        out += &format!("{rest:9} {:7.2}%  (the rest of thread {t})\n", pct(rest));
    }
    // Where the hottest symbols were called from: the guest's x30 at the
    // sample, the return address of the last call; exact in a leaf function only.
    out += "# callers of the 25 hottest symbols (x30 when sampled; exact in leaf functions only)\n";
    for (name, n) in rows.iter().take(25) {
        out += &format!("# {name}: {n} samples\n");
        let mut sites: Vec<(u64, u64)> = callers.get(name).map_or(Vec::new(), |c| c.iter().map(|(&a, &k)| (a, k)).collect());
        sites.sort_by(|a, b| b.1.cmp(&a.1).then(a.0.cmp(&b.0)));
        for (lr, k) in sites.into_iter().take(6) {
            out += &format!("{k:9} {:7.2}%  x30 {lr:#x}  {}\n", pct(k), guest(lr));
        }
    }
    // Their host callers: the first return address on the host stack, the caller
    // of a leaf function (memcpy, the allocator), or of the current function; a
    // stale one may show up. By calling function, then its call sites.
    out += "# host callers of the 25 hottest symbols (first return address on the stack)\n";
    for (name, n) in rows.iter().take(25) {
        out += &format!("# {name}: {n} samples\n");
        let mut by_function: HashMap<String, u64> = HashMap::new();
        for (&at, &k) in hosts.get(name).into_iter().flatten() {
            let function = if at == 0 { "(none found)".to_string() } else { format!("[{}] {}", module(at), symbol(at)) };
            *by_function.entry(function).or_insert(0) += k;
        }
        let mut sites: Vec<(String, u64)> = by_function.into_iter().collect();
        sites.sort_by(|a, b| b.1.cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
        for (function, k) in sites.into_iter().take(8) {
            out += &format!("{k:9} {:7.2}%  {function}\n", pct(k));
        }
    }
    // The interpreter's instructions, by opcode, then the most frequent words.
    let words: Vec<(u32, u64)> = WORDS.with(|w| w.borrow().iter().map(|(&w, &n)| (w, n)).collect());
    let interpreted: u64 = words.iter().map(|w| w.1).sum();
    let mut by_opcode: HashMap<String, u64> = HashMap::new();
    for &(w, n) in &words {
        let op = crate::interp::opcode_name(w);
        *by_opcode.entry(op).or_insert(0) += n;
    }
    let mut ops: Vec<(String, u64)> = by_opcode.into_iter().collect();
    ops.sort_by(|a, b| b.1.cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
    out += &format!("# interpreted instructions: {interpreted}\n# count  percent  opcode\n");
    for (op, n) in ops {
        out += &format!("{n:12} {:7.2}%  {op}\n", 100.0 * n as f64 / interpreted.max(1) as f64);
    }
    let mut top = words;
    top.sort_by(|a, b| b.1.cmp(&a.1).then(a.0.cmp(&b.0)));
    out += "# most frequent words\n";
    for (w, n) in top.into_iter().take(80) {
        out += &format!("{n:12}  {w:08x}  {}\n", crate::interp::disassemble(w));
    }
    if let Some(dir) = p.path.parent() {
        let _ = std::fs::create_dir_all(dir);
    }
    if let Err(e) = std::fs::write(&p.path, out) {
        eprintln!("gyroship: --profile {}: {e}", p.path.display());
    }
}

/// The name of the symbol at host address `rip`, from the PDB; `?` and the
/// address when dbghelp has none.
#[cfg(windows)]
fn symbol(rip: u64) -> String {
    use windows_sys::Win32::System::Diagnostics::Debug::{SYMBOL_INFO, SymFromAddr, SymInitialize};
    use windows_sys::Win32::System::Threading::GetCurrentProcess;
    static INIT: std::sync::Once = std::sync::Once::new();
    // SAFETY: dbghelp is initialized once for this process, before any lookup;
    // it looks for the PDB next to the executable.
    INIT.call_once(|| unsafe {
        let dir = std::env::current_exe().ok().and_then(|e| Some(e.parent()?.to_string_lossy().into_owned() + "\0"));
        SymInitialize(GetCurrentProcess(), dir.as_deref().map_or(std::ptr::null(), |d| d.as_ptr()), 1);
    });
    const MAX: usize = 512;
    #[repr(C)]
    struct Buffer {
        info: SYMBOL_INFO,
        name: [u8; MAX],
    }
    // SAFETY: a zeroed SYMBOL_INFO followed by MAX bytes for its name, as dbghelp asks.
    let mut b: Buffer = unsafe { std::mem::zeroed() };
    b.info.SizeOfStruct = size_of::<SYMBOL_INFO>() as u32;
    b.info.MaxNameLen = MAX as u32;
    let mut displacement = 0u64;
    // SAFETY: the buffer is as large as MaxNameLen says.
    if unsafe { SymFromAddr(GetCurrentProcess(), rip, &mut displacement, &mut b.info) } == 0 {
        static SAID: std::sync::Once = std::sync::Once::new();
        SAID.call_once(|| eprintln!("gyroship: --profile: no symbol at {rip:#x}: {}", std::io::Error::last_os_error()));
        return format!("? {rip:#x}");
    }
    let len = (b.info.NameLen as usize).min(MAX);
    // SAFETY: Name is the first byte of a NameLen-byte string inside the buffer.
    let name = unsafe { std::slice::from_raw_parts(b.info.Name.as_ptr().cast::<u8>(), len) };
    String::from_utf8_lossy(name).into_owned()
}

#[cfg(not(windows))]
fn symbol(rip: u64) -> String {
    format!("? {rip:#x}")
}

/// The name of the module (executable or DLL) holding host address `rip`, as
/// dbghelp loaded it ([`symbol`] runs first and initializes dbghelp).
#[cfg(windows)]
fn module(rip: u64) -> String {
    use windows_sys::Win32::System::Diagnostics::Debug::{IMAGEHLP_MODULEW64, SymGetModuleInfoW64};
    use windows_sys::Win32::System::Threading::GetCurrentProcess;
    let _ = symbol(0);
    // SAFETY: a zeroed IMAGEHLP_MODULEW64 with its size set, as dbghelp asks.
    let mut info: IMAGEHLP_MODULEW64 = unsafe { std::mem::zeroed() };
    info.SizeOfStruct = size_of::<IMAGEHLP_MODULEW64>() as u32;
    // SAFETY: the struct is as large as SizeOfStruct says.
    if unsafe { SymGetModuleInfoW64(GetCurrentProcess(), rip, &mut info) } == 0 {
        return "?".into();
    }
    let len = info.ModuleName.iter().position(|&c| c == 0).unwrap_or(info.ModuleName.len());
    String::from_utf16_lossy(&info.ModuleName[..len])
}

#[cfg(not(windows))]
fn module(_rip: u64) -> String {
    "?".into()
}

#[cfg(all(test, windows))]
mod tests {
    use std::os::windows::io::AsRawHandle;
    use std::time::{Duration, Instant};

    /// A thread that spins has CPU time; one that sleeps has almost none.
    #[test]
    fn cpu_time_counts_running_not_sleeping() {
        let (tx, rx) = std::sync::mpsc::channel();
        let (busy_tx, idle_tx) = (tx.clone(), tx);
        let busy = std::thread::spawn(move || {
            let start = Instant::now();
            while start.elapsed() < Duration::from_millis(300) {
                std::hint::spin_loop();
            }
            busy_tx.send(()).unwrap();
        });
        let idle = std::thread::spawn(move || {
            std::thread::sleep(Duration::from_millis(300));
            idle_tx.send(()).unwrap();
        });
        rx.recv().unwrap();
        rx.recv().unwrap();
        let busy_time = super::cpu_time(busy.as_raw_handle() as usize).unwrap();
        let idle_time = super::cpu_time(idle.as_raw_handle() as usize).unwrap();
        assert!(busy_time > Duration::from_millis(100) && idle_time < Duration::from_millis(50), "busy {busy_time:?}, idle {idle_time:?}");
        busy.join().unwrap();
        idle.join().unwrap();
    }

    /// A thread spinning in a known function: its instruction pointer is in
    /// that function, and the return address into its caller is on its stack.
    #[test]
    fn a_watched_thread_shows_where_it_spins() {
        use std::sync::atomic::{AtomicBool, AtomicU32, Ordering};
        static STOP: AtomicBool = AtomicBool::new(false);
        static TID: AtomicU32 = AtomicU32::new(0);
        #[inline(never)]
        fn spin() -> u64 {
            let mut n = 0u64;
            while !STOP.load(Ordering::Relaxed) {
                n = std::hint::black_box(n + 1);
            }
            n
        }
        #[inline(never)]
        fn caller() -> u64 {
            TID.store(super::thread_id(), Ordering::Relaxed);
            std::hint::black_box(spin()) + 1
        }
        let t = std::thread::spawn(caller);
        while TID.load(Ordering::Relaxed) == 0 {
            std::thread::yield_now();
        }
        std::thread::sleep(Duration::from_millis(20));
        let lines = super::stack_of(TID.load(Ordering::Relaxed));
        STOP.store(true, Ordering::Relaxed);
        t.join().unwrap();
        // By name, anywhere in the stack: a debug build samples the spin in the atomic
        // load it does not inline, above `spin`.
        let at = |name: &str| lines.iter().position(|l| l.ends_with(name));
        assert!(at("::spin").is_some_and(|s| at("::caller").is_some_and(|c| s < c)), "{lines:?}");
    }
}
