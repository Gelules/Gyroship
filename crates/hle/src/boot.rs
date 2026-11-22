//! Boot (D17): the seven modules are mapped raw at their D12 bases, as
//! Horizon's loader would, and the main thread starts at rtld's entry. rtld
//! relocates itself and every module, then calls `nn::init::Start`.
//!
//! `link()` is the oracle: when the probe named `nn::init::Start` first
//! passes, guest memory is compared with the linker's image.

use crate::journal::Journal;
use crate::{Config, Kernel, STOP_EXIT, layout, thread};
use gyroship_loader::Module;
use gyroship_loader::link::{self, LOAD_ORDER, Program, R_AARCH64_JUMP_SLOT};
use gyroship_runtime::mem::{self, Block, Memory, perm, state};
use gyroship_runtime::{self as rt, Cpu};
use std::path::Path;

/// The main thread's stack, alone in its block of the stack region.
const MAIN_STACK: (u64, u64) = (layout::STACK.0 + 0x10_0000, 0x10_0000);

/// The probe that triggers the comparison with `link()`.
pub const ORACLE_PROBE: &str = "nn::init::Start";

/// The modules of an exefs, in load order.
pub fn read_modules(exefs: &Path) -> Result<Vec<(String, Module)>, String> {
    let mut out = Vec::new();
    for name in LOAD_ORDER {
        let path = exefs.join(name);
        if path.is_file() {
            let file = std::fs::read(&path).map_err(|e| format!("{}: {e}", path.display()))?;
            out.push((name.to_string(), Module::parse_nso(&file).map_err(|e| format!("{name}: {}", e.0))?));
        }
    }
    // Nothing checks the dump before the game starts: a wrong folder is said here, in words.
    if out.is_empty() {
        return Err(format!("{}: no module here (main, rtld, sdk...): give the exefs folder of your dump, with its romfs folder next to it", exefs.display()));
    }
    Ok(out)
}

/// The client services an NPDM grants: the service access control of its
/// ACI0 (offset at ACI0+0x28, size at +0x2C), whose entries are a control
/// byte (bits 0-2: length - 1, bit 7: server) and the name.
pub fn npdm_services(npdm: &[u8]) -> Option<Vec<String>> {
    let u32_at = |o: usize| Some(u32::from_le_bytes(npdm.get(o..o + 4)?.try_into().ok()?) as usize);
    if npdm.get(0..4)? != b"META" {
        return None;
    }
    let aci = u32_at(0x70)?;
    if npdm.get(aci..aci + 4)? != b"ACI0" {
        return None;
    }
    let (off, size) = (u32_at(aci + 0x28)?, u32_at(aci + 0x2C)?);
    let sac = npdm.get(aci + off..aci + off + size)?;
    let mut names = Vec::new();
    let mut i = 0;
    while i < sac.len() {
        let control = sac[i];
        let len = usize::from(control & 7) + 1;
        let name = sac.get(i + 1..i + 1 + len)?;
        if control & 0x80 == 0 {
            names.push(String::from_utf8_lossy(name).into_owned());
        }
        i += 1 + len;
    }
    Some(names)
}

fn round_up(v: u64) -> u64 {
    v.div_ceil(mem::PAGE) * mem::PAGE
}

/// Maps one module raw at `base`: text R-X and ro R in state Code, data and
/// bss RW in state CodeData. Returns its end.
pub fn map_module(m: &Memory, base: u64, module: &Module) -> Result<u64, String> {
    let data_end = round_up(module.mem_end());
    let segments = [
        (module.text.addr, round_up(module.text.end()), state::CODE, perm::RX),
        (module.ro.addr, round_up(module.ro.end()), state::CODE, perm::R),
        (module.data.addr, data_end, state::CODE_DATA, perm::RW),
    ];
    for (start, end, st, p) in segments {
        let (at, size) = (base + start, end - start);
        // Written while RW, then given its permission.
        m.map(at, size, Block::new(st, perm::RW)).map_err(|e| format!("mapping {at:#x}+{size:#x}: {e:?}"))?;
        let bytes = &module.image[start as usize..(end as usize).min(module.image.len())];
        m.write(at, bytes).map_err(|e| format!("writing {at:#x}: {e:?}"))?;
        if p != perm::RW {
            m.protect(at, size, p).map_err(|e| format!("protecting {at:#x}: {e:?}"))?;
        }
    }
    Ok(base + data_end)
}

/// The dump's romfs folder: next to the exefs.
fn romfs_dir(exefs: &Path) -> Option<std::path::PathBuf> {
    exefs.parent().map(|p| p.join("romfs")).filter(|p| p.is_dir())
}

/// A kernel with the modules mapped and the main thread ready at rtld's entry.
pub fn boot(exefs: &Path, journal: Journal, config: Config) -> Result<Kernel, String> {
    let modules = read_modules(exefs)?;
    let m = Memory::new().map_err(|e| format!("reserving guest memory: {e:?}"))?;
    let mut k_lines = Vec::new();
    for (name, module) in &modules {
        let base = link::base_of(name).ok_or_else(|| format!("no base for {name}"))?;
        let end = map_module(&m, base, module)?;
        k_lines.push(format!("module {name} at {base:#x}..{end:#x}"));
    }
    m.map(MAIN_STACK.0, MAIN_STACK.1, Block::new(state::STACK, perm::RW)).map_err(|e| format!("main stack: {e:?}"))?;
    let mut k = Kernel::new(m, journal, config);
    for line in k_lines {
        k.log(&line);
    }
    let npdm = std::fs::read(exefs.join("main.npdm")).map_err(|e| format!("{}: {e}", exefs.join("main.npdm").display()))?;
    k.allowed_services = npdm_services(&npdm).ok_or("main.npdm: no service access control")?;
    let services = format!("npdm: {} services granted: {}", k.allowed_services.len(), k.allowed_services.join(" "));
    k.log(&services);
    let romfs = romfs_dir(exefs).ok_or("no romfs folder next to the exefs")?;
    let layer = k.config.romfs_layer.clone();
    let (image, replaced) = crate::fs::RomFs::layered(&romfs, layer.as_deref()).map_err(|e| format!("{}: {e}", romfs.display()))?;
    k.log(&format!("romfs: an image of {} bytes", image.size()));
    if let Some(dir) = layer {
        // A run with a layer is never a reference run: the journal says what changed.
        k.log(&format!("romfs: layer {} replaces {}", dir.display(), if replaced.is_empty() { "nothing".to_string() } else { replaced.join(", ") }));
    }
    k.romfs = Some(std::sync::Arc::new(image));
    let rtld = link::base_of("rtld").expect("rtld's base");
    k.create_main(rtld, MAIN_STACK.0 + MAIN_STACK.1);
    let program = Program::link(modules).map_err(|e| format!("link: {}", e.0))?;
    k.symbols = program.modules.iter().flat_map(|p| p.module.symbols.iter().filter(|s| s.is_defined() && s.is_function()).map(move |s| (p.base + s.value, s.name.clone()))).collect();
    k.symbols.sort();
    k.oracle = Some(program);
    Ok(k)
}

impl Kernel {
    /// Compares the ro and data segments of every module with `link()`'s
    /// image. The 973 lazy JUMP_SLOTs of `main` differ by design (rtld binds
    /// them at the first call); everything else is listed.
    pub fn check_oracle(&mut self) {
        let Some(program) = self.oracle.take() else { return };
        let mut lines = Vec::new();
        for p in &program.modules {
            let lazy: std::collections::HashSet<u64> =
                if p.name == "main" { p.module.relocations.iter().filter(|r| r.kind == R_AARCH64_JUMP_SLOT).map(|r| p.base + r.offset).collect() } else { Default::default() };
            let (mut slots, mut other) = (0, Vec::new());
            for seg in [p.module.ro, p.module.data] {
                let mut guest = vec![0; seg.size as usize];
                if self.mem.read(p.base + seg.addr, &mut guest).is_err() {
                    lines.push(format!("oracle: {} segment at {:#x} is not readable", p.name, p.base + seg.addr));
                    continue;
                }
                let linked = &p.module.image[seg.addr as usize..seg.end() as usize];
                for (i, (g, l)) in guest.chunks(8).zip(linked.chunks(8)).enumerate() {
                    if g != l {
                        let at = p.base + seg.addr + 8 * i as u64;
                        if lazy.contains(&at) {
                            slots += 1;
                        } else {
                            other.push(format!("{at:#x}: guest {} link {}", hex(g), hex(l)));
                        }
                    }
                }
            }
            lines.push(format!("oracle: {}: {} lazy JUMP_SLOTs differ, {} other words differ", p.name, slots, other.len()));
            lines.extend(other.iter().take(64).map(|o| format!("oracle:   {o}")));
        }
        for l in lines {
            self.log(&l);
        }
    }
}

fn hex(b: &[u8]) -> String {
    let mut v = [0; 8];
    v[..b.len()].copy_from_slice(b);
    format!("{:#x}", u64::from_le_bytes(v))
}

/// A `--trace` line's stamp: the virtual time, the guest thread, the present.
fn stamp(k: &Kernel) -> (u64, u64, u64) {
    (k.now(), k.sched.current.map_or(0, |t| k.sched.threads[t].id), k.gs.presents)
}

/// Whether `--trace` writes its file. Without it, a `trace:` probe's pass leaves at once:
/// job threads pass them hundreds of times a frame, and the kernel lock is all they would take.
static TRACING: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

/// The probe hook: the first pass is journaled; the oracle runs at `nn::init::Start`.
/// Every pass of a `trace:` probe goes to the `--trace` file only, then to the practice
/// actions (`--practice`), which run without the kernel lock.
pub(crate) unsafe fn on_probe(cpu: *mut Cpu, pc: u64, name: &str) {
    if name.starts_with("trace:") {
        // SAFETY: the running thread's context, which this hook and the actions only read.
        let cpu = unsafe { &*cpu };
        let practice = crate::practice::actions();
        // The speed line's reading and the markers' vehicle (any overlay option) write nothing: they may come
        // before the trace line.
        let speed = practice.and_then(|p| p.read_speed(pc, cpu));
        if let Some(p) = practice {
            p.mark_pass(pc, cpu);
        }
        // The trace line first: the state the game left at the probe; then the reading.
        let tracing = TRACING.load(std::sync::atomic::Ordering::Relaxed);
        if tracing {
            crate::with(|k| {
                let stamp = stamp(k);
                if let Some(t) = k.trace.as_mut() {
                    t.pass(pc, name, cpu, &k.mem, stamp);
                    if let Some([s, m, b, v]) = speed {
                        t.note(stamp, &format!("hud speed={v} s={s} m={m} b={b}"));
                    }
                }
            });
        }
        // Then the actions that write (--boost, ms lines), lock-free; a perfect release's guard
        // speaks once.
        if let Some(line) = practice.and_then(|p| p.act(pc, cpu)) {
            eprintln!("gyroship: {line}");
            crate::with(|k| k.log(&line));
        }
        if let Some(text) = practice.and_then(|p| p.time_ms(pc, cpu))
            && tracing
        {
            crate::with(|k| {
                let stamp = stamp(k);
                if let Some(t) = k.trace.as_mut() {
                    t.note(stamp, &format!("ms text={text}"));
                }
            });
        }
        return;
    }
    // SAFETY: the running thread's context, read before anything else runs.
    let (x, lr) = unsafe { ((*cpu).x, (*cpu).x[30]) };
    crate::with(|k| {
        let caller = k.symbolize(lr);
        k.log(&format!("probe {name} at {pc:#x}: x0={:#x} x1={:#x} x2={:#x} x3={:#x} x8={:#x}, called from {lr:#x} {caller}", x[0], x[1], x[2], x[3], x[8]));
        // Probes named `regs:...` also log every register, `stack:...` their frame chain;
        // an `each:` prefix (every pass is reported) combines with them.
        let name = name.strip_prefix("each:").unwrap_or(name);
        if name.starts_with("regs:") {
            let all: Vec<String> = x.iter().enumerate().map(|(i, v)| format!("x{i}={v:#x}")).collect();
            k.log(&format!("  {}", all.join(" ")));
        }
        // `dump:...`: the 0x700 bytes at x0, as u64 words, 8 per line.
        if name.starts_with("dump:") {
            let mut raw = vec![0u8; 0x700];
            if k.mem.read(x[0], &mut raw).is_ok() {
                for (i, row) in raw.as_chunks::<64>().0.iter().enumerate() {
                    let words: Vec<String> = row.as_chunks::<8>().0.iter().map(|w| format!("{:x}", u64::from_le_bytes(*w))).collect();
                    k.log(&format!("  +{:#05x}: {}", i * 64, words.join(" ")));
                }
            }
        }
        if name.starts_with("stack:") {
            for line in k.backtrace_from((pc, lr, x[29])) {
                k.log(&line);
            }
        }
        if name == ORACLE_PROBE {
            k.check_oracle();
        }
    });
}

/// Every option after `boot <exefs>`, the host window's too, and whether it takes a
/// value. Anything else, or a valued option given twice, ends the start: a misspelt option
/// used to be ignored without a word.
pub const OPTIONS: &[(&str, bool)] = &[
    // The host (crates/gyroship).
    ("--headless", false),
    ("--backend", true),
    ("--renderer", true),
    ("--sdl", true),
    ("--fullscreen", false),
    ("--joycon", false),
    ("--joycon-mouse", false),
    // The run.
    ("--log", true),
    ("--all-svcs", false),
    ("--explore", false),
    ("--frames", true),
    ("--seconds", true),
    ("--replay", true),
    ("--save", true),
    ("--record", true),
    ("--play", false),
    ("--no-vsync", false),
    ("--late-frames", false),
    ("--gpu-sync", false),
    ("--work-time", false),
    ("--gpu", false),
    ("--pipeline-cache", true),
    ("--romfs-layer", true),
    ("--hooks", true),
    ("--language", true),
    ("--practice", true),
    ("--boost", true),
    ("--hud", false),
    ("--hud-speed", false),
    ("--hud-fps", false),
    ("--hud-markers", false),
    // Reports and dumps.
    ("--perf", true),
    ("--hashes", true),
    ("--profile", true),
    ("--profile-from", true),
    ("--trace", true),
    ("--dump-frames", true),
    ("--dump-every", true),
    ("--dump-from", true),
    ("--dump-draws", true),
    ("--dump-shaders", true),
    ("--dump-gpfifo", true),
    ("--dump-audio", true),
    ("--video", true),
    ("--help", false),
];

/// Checks `boot <exefs> [options]` (the program's arguments, without its name): every
/// option is known, each valued one has its value and comes once.
pub fn check_options(args: &[String]) -> Result<(), String> {
    let mut seen = std::collections::BTreeSet::new();
    let mut rest = args.iter().skip(2);
    while let Some(a) = rest.next() {
        let Some(&(name, valued)) = OPTIONS.iter().find(|(o, _)| o == a) else {
            return Err(format!("unknown option {a} (--help lists them)"));
        };
        if valued {
            if rest.next().is_none() {
                return Err(format!("{name} needs a value"));
            }
            if !seen.insert(name) {
                return Err(format!("{name} given twice"));
            }
        }
    }
    Ok(())
}

/// `precompile <exefs> --pipeline-cache DIR` (D110 step 8): the shaders of the dump's data.zip,
/// and those past play translated, made into pipeline libraries in DIR's cache before play.
/// Progress `done/total` on stdout (the launcher's), the report on stderr. Exit code 0 when done
/// or when there was nothing to do, 1 on a failure, 2 on a misuse.
fn precompile(args: &[String]) -> i32 {
    let cache = args.iter().position(|a| a == "--pipeline-cache").and_then(|i| args.get(i + 1));
    let (Some(exefs), Some(cache)) = (args.first(), cache) else {
        eprintln!("usage: gyroship precompile <exefs> --pipeline-cache DIR");
        return 2;
    };
    let Some(zip) = romfs_dir(Path::new(exefs)).map(|r| r.join("data.zip")) else {
        eprintln!("gyroship: precompile: no romfs folder next to {exefs}");
        return 2;
    };
    // A line per hundredth at most: the launcher shows them all.
    let mut progress = |done: usize, total: usize| {
        if done == total || done % (total / 100).max(1) == 0 {
            println!("{done}/{total}");
            let _ = std::io::Write::flush(&mut std::io::stdout());
        }
    };
    match gyroship_gpu::precompile::run(&zip, Path::new(cache), &mut progress) {
        Ok(report) => {
            eprintln!("gyroship: precompile: {report}");
            0
        }
        Err(e) => {
            eprintln!("gyroship: precompile: {e}");
            1
        }
    }
}

/// `--help`: the options, one per line.
pub fn usage() -> String {
    let mut s = String::from("usage: gyroship boot <exefs> [options]\n       gyroship precompile <exefs> --pipeline-cache DIR\noptions:\n");
    for (o, valued) in OPTIONS {
        s += &format!("  {o}{}\n", if *valued { " VALUE" } else { "" });
    }
    s
}

/// The option names a source text quotes, as the option readers spell them (for the tests
/// that check `OPTIONS` against the code reading them).
pub fn literals(src: &str) -> Vec<&str> {
    src.match_indices("\"--")
        .filter_map(|(i, _)| {
            let s = &src[i + 1..];
            let end = 2 + s[2..].find(|c: char| !(c.is_ascii_lowercase() || c.is_ascii_digit() || c == '-'))?;
            (end > 2 && s.as_bytes()[end] == b'"').then(|| &s[..end])
        })
        .collect()
}

/// The generated harness's `main`: `boot <exefs> [--log FILE] [--explore]
/// [--all-svcs] [--seconds N] [--frames N] [--gpu] [--backend auto|vulkan|dx12]
/// [--renderer wgpu|vulkan] [--play] ...` runs the game (until N frames are presented, or for N
/// seconds of virtual time, or of real time with `--play`: 60 when neither is
/// given); any other mode is the runtime's. A GPU that lacks what the game
/// needs ends the run with exit code 3.
pub fn main(table: Vec<rt::Entry>) {
    run(table, None)
}

/// What a host window gives the HLE (D28): where to draw, its size, and the
/// pace. The HLE itself has no window, wall clock or input device.
pub struct HostWindow {
    /// The window's surface, and the instance it was made from (on the window's thread).
    pub instance: gyroship_gpu::wgpu::Instance,
    pub surface: gyroship_gpu::wgpu::Surface<'static>,
    pub size: (u32, u32),
    /// Called after each frame shown, with its virtual time (ticks): waits for
    /// that time on the wall clock (D27), then gives the window's size, or
    /// None once the window is closed. In play mode it never waits: it runs
    /// on the GPU thread, under the GPU side's lock, and guest time is the
    /// wall clock already.
    pub pace: crate::Pace,
    /// The controller's live state (keyboard now, controllers later), read at each hid sample.
    pub input: crate::LiveInput,
    /// Where the game's sound goes (None: silent).
    pub audio: Option<crate::AudioOut>,
}

/// [`main`], showing the frames in a host window when one is given.
pub fn run(mut table: Vec<rt::Entry>, window: Option<HostWindow>) {
    let args: Vec<String> = std::env::args().skip(1).collect();
    if args.first().map(String::as_str) == Some("precompile") {
        std::process::exit(precompile(&args[1..]));
    }
    if args.first().map(String::as_str) != Some("boot") {
        return rt::harness::main(table);
    }
    let Some(exefs) = args.get(1) else {
        eprintln!("usage: gyroship boot <exefs> [--log FILE] [--explore] [--all-svcs]");
        std::process::exit(2);
    };
    let flag = |f: &str| args.iter().any(|a| a == f);
    let log = args.iter().position(|a| a == "--log").and_then(|i| args.get(i + 1)).map_or("boot.log", String::as_str);
    let mut journal = Journal::file(Path::new(log)).unwrap_or_else(|e| panic!("{log}: {e}"));
    journal.all_svcs = flag("--all-svcs");
    let value = |f: &str| args.iter().position(|a| a == f).and_then(|i| args.get(i + 1)).cloned();
    let frames: Option<u64> = value("--frames").and_then(|s| s.parse().ok());
    // A virtual time limit: --seconds, else 60 s unless --frames, a window or a replay (its end) bounds the run.
    let seconds: Option<u64> = value("--seconds").and_then(|s| s.parse().ok()).or(if frames.is_some() || window.is_some() || value("--replay").is_some() { None } else { Some(60) });
    let save_dir = value("--save").map(std::path::PathBuf::from);
    let config = Config {
        explore: flag("--explore"),
        time_limit: seconds.map(|s| s * 31_250_000),
        frames,
        save_dir,
        vsync: !flag("--no-vsync"),
        work_time: flag("--work-time"),
        gpfifo_dump: value("--dump-gpfifo").map(std::path::PathBuf::from),
        frames_dump: value("--dump-frames").map(std::path::PathBuf::from),
        dump_draws: value("--dump-draws").and_then(|s| s.parse().ok()),
        dump_every: value("--dump-every").and_then(|s| s.parse().ok()).filter(|&k| k > 0).unwrap_or(1),
        dump_from: value("--dump-from").and_then(|s| s.parse().ok()).unwrap_or(0),
        shaders_dump: value("--dump-shaders").map(std::path::PathBuf::from),
        replay: value("--replay").map(|f| {
            let script = std::fs::read_to_string(&f).map_err(|e| e.to_string()).and_then(|t| crate::hid::parse_script(&t));
            script.unwrap_or_else(|e| {
                eprintln!("gyroship: --replay {f}: {e}");
                std::process::exit(2);
            })
        }),
        profile: value("--profile").map(|f| (f.into(), value("--profile-from").and_then(|n| n.parse().ok()).unwrap_or(600))),
        perf: value("--perf").map(Into::into),
        hashes: value("--hashes").map(Into::into),
        record: value("--record").map(Into::into),
        trace: value("--trace").map(Into::into),
        audio_dump: value("--dump-audio").map(Into::into),
        play: flag("--play"),
        pipeline_cache: value("--pipeline-cache").map(Into::into),
        late_frames: flag("--late-frames"),
        romfs_layer: value("--romfs-layer").map(Into::into),
        hooks: value("--hooks").map(Into::into),
        video: value("--video").map(Into::into),
        gpu_sync: flag("--gpu-sync"),
        language: value("--language").map(|v| {
            gyroship_practice::texts::language_code(&v).unwrap_or_else(|| {
                eprintln!("gyroship: --language {v}: not a console language code ({})", gyroship_practice::texts::LANGUAGE_CODES.join(", "));
                std::process::exit(2);
            })
        }),
        // --hud: the three overlays; the others, one each.
        hud: crate::Overlays { speed: flag("--hud") || flag("--hud-speed"), fps: flag("--hud") || flag("--hud-fps"), markers: flag("--hud") || flag("--hud-markers") },
        ..Config::default()
    };
    if config.play {
        eprintln!("gyroship: play mode (experimental): a host thread per guest thread, on the wall clock; runs do not repeat and are never a reference");
        if config.replay.is_some() {
            eprintln!("gyroship: --replay in play mode follows the wall clock: the game sees the inputs at other moments than in the recorded run");
        }
        if config.record.is_some() {
            eprintln!("gyroship: --record in play mode: the file will not replay this run exactly");
        }
    }
    let hooked = config.hooks.as_ref().map(|f| {
        std::fs::read_to_string(f).map_err(|e| e.to_string()).and_then(|t| crate::hooks::apply(&mut table, &t)).unwrap_or_else(|e| {
            eprintln!("gyroship: --hooks {}: {e}", f.display());
            std::process::exit(2);
        })
    });
    let installed = table.len();
    rt::install(table);
    mem::install_fault_handler();
    let mut k = boot(Path::new(exefs), journal, config).unwrap_or_else(|e| {
        eprintln!("gyroship: boot failed: {e}");
        std::process::exit(2);
    });
    k.log(&format!("{installed} recompiled functions installed"));
    // A run with hooks is never a reference run: the journal says what changed.
    for h in hooked.unwrap_or_default() {
        k.log(&format!("hook: {h}"));
    }
    if let Some(path) = k.config.trace.clone() {
        k.trace = Some(crate::trace::Trace::open(&path).unwrap_or_else(|e| {
            eprintln!("gyroship: --trace {}: {e}", path.display());
            std::process::exit(2);
        }));
        TRACING.store(true, std::sync::atomic::Ordering::Relaxed);
    }
    // M6: practice actions at trace: probes; the journal says so, as a run with them is never
    // a reference.
    let boost = value("--boost").map(|b| {
        crate::practice::Boost::parse(&b).unwrap_or_else(|| {
            eprintln!("gyroship: --boost {b}: not a boost mode (infinite, perfect)");
            std::process::exit(2);
        })
    });
    match value("--practice") {
        Some(path) => {
            let text = std::fs::read_to_string(&path).unwrap_or_else(|e| {
                eprintln!("gyroship: --practice {path}: {e}");
                std::process::exit(2);
            });
            let practice = crate::practice::Practice::parse(&text, boost, k.config.hud.any(), k.mem.clone(), &rt::probe_counts()).unwrap_or_else(|e| {
                eprintln!("gyroship: --practice {path}: {e}");
                std::process::exit(2);
            });
            // The overlays: the font is read once, here; they are drawn on the GPU side.
            if k.config.hud.any() {
                let zip = romfs_dir(Path::new(exefs)).map(|r| r.join("data.zip")).ok_or_else(|| "no romfs folder next to the exefs".to_string());
                let hud = zip.and_then(|zip| crate::practice::Hud::load(&text, practice.hud_at(), &zip, k.config.language, k.config.hud)).unwrap_or_else(|e| {
                    eprintln!("gyroship: the --hud options: {e}");
                    std::process::exit(2);
                });
                k.gpu_side().hud = Some(hud);
            }
            // A run whose actions write is never a reference: the journal says so.
            if practice.writes_ms() {
                k.log("practice: milliseconds after the game's level time (ms lines)");
            }
            crate::practice::install(practice);
            if let Some(b) = boost {
                k.log(&format!("practice: boost {} (--boost)", b.name()));
            }
        }
        None if boost.is_some() || k.config.hud.any() => {
            eprintln!("gyroship: {} --practice FILE", if boost.is_some() { "--boost needs" } else { "the --hud options need" });
            std::process::exit(2);
        }
        None => {}
    }
    // The host GPU runs the game's work when frames are dumped, filmed or measured, or with --gpu.
    if window.is_some() || k.config.frames_dump.is_some() || k.config.video.is_some() || k.config.perf.is_some() || k.config.hashes.is_some() || flag("--gpu") {
        // D110 step 6b: every draw, full clear and fill recorded in raw Vulkan, on the Vulkan backend.
        let vulkan = match value("--renderer").as_deref() {
            None | Some("wgpu") => false,
            Some("vulkan") => true,
            Some(other) => {
                eprintln!("gyroship: --renderer {other}: expected wgpu or vulkan");
                std::process::exit(2);
            }
        };
        let backend = match value("--backend").as_deref() {
            None | Some("auto") if vulkan => gyroship_gpu::Backend::Vulkan,
            None | Some("auto") => gyroship_gpu::Backend::Auto,
            Some("vulkan") => gyroship_gpu::Backend::Vulkan,
            Some("dx12") if vulkan => {
                eprintln!("gyroship: --renderer vulkan needs the Vulkan backend, not --backend dx12");
                std::process::exit(2);
            }
            Some("dx12") => gyroship_gpu::Backend::Dx12,
            Some(other) => {
                eprintln!("gyroship: --backend {other}: expected auto, vulkan or dx12");
                std::process::exit(2);
            }
        };
        let surfaces = match window {
            Some(w) => {
                k.gpu_side().window = Some((w.pace, w.size));
                k.live_input = Some(w.input);
                k.audio_out = w.audio;
                gyroship_gpu::surfaces::Surfaces::windowed(&w.instance, w.surface, w.size)
            }
            None => gyroship_gpu::surfaces::Surfaces::headless(backend),
        };
        match surfaces {
            Ok(mut s) => {
                k.log(&format!("gpu: host renderer {}", s.describe()));
                // Said on stderr only, the journal staying comparable with the wgpu path's.
                if vulkan {
                    s.record_raw();
                }
                // D42: in play mode a pipeline the driver takes seconds over holds no frame.
                s.background = k.config.play;
                // Said on stderr only: the journal of a run with the cache stays comparable.
                if let Some(dir) = &k.config.pipeline_cache {
                    match s.use_pipeline_cache(dir) {
                        Ok((file, size)) => eprintln!("gyroship: pipeline cache {} ({} KiB)", file.display(), size / 1024),
                        Err(e) => eprintln!("gyroship: --pipeline-cache {}: {e}; pipelines are compiled every run", dir.display()),
                    }
                    // D72: the translated shaders too; not while a debugging override replaces translations.
                    match std::env::var_os("GYROSHIP_WGSL_OVERRIDE") {
                        Some(_) => eprintln!("gyroship: shader cache off: GYROSHIP_WGSL_OVERRIDE is set"),
                        None => match gyroship_gpu::shader::cache::ShaderCache::open(&dir.join("shaders")) {
                            Ok(c) => {
                                eprintln!("gyroship: shader cache {}", c.describe());
                                s.shader_cache = Some(c);
                            }
                            Err(e) => eprintln!("gyroship: shader cache: {e}; shaders are translated every run"),
                        },
                    }
                }
                if k.config.perf.is_some() && !s.measure_frames() {
                    eprintln!("gyroship: --perf: this GPU has no timestamps inside command encoders: no GPU frame times");
                }
                // D65: in play mode each frame is submitted on a thread of its own (the frame clock goes with it).
                if k.config.play
                    && let Err(e) = s.start_submit_thread()
                {
                    eprintln!("gyroship: cannot start the GPU submit thread: {e}");
                    std::process::exit(2);
                }
                // S12c: and recorded on another while the GPU thread resolves the next draws.
                // GYROSHIP_RECORDER=thread: in a deterministic run too, which proves it bit for bit
                // while wgpu raises no error (the two threads' errors may come in either order).
                if (k.config.play || std::env::var_os("GYROSHIP_RECORDER").is_some_and(|v| v == "thread"))
                    && let Err(e) = s.start_recorder_thread()
                {
                    eprintln!("gyroship: cannot start the GPU recorder thread: {e}");
                    std::process::exit(2);
                }
                k.gpu_side().surfaces = Some(Box::new(s));
            }
            Err(e) => {
                k.log(&format!("gpu: no host renderer: {e}"));
                k.journal.flush();
                eprintln!("gyroship: this GPU cannot run the game: {e}");
                std::process::exit(3);
            }
        }
    }
    let play = k.config.play;
    // In play mode a thread of its own applies the GPU ops while the submitting thread decodes.
    // `run_play` must run on this thread: the GPU thread wakes it.
    if play && let Err(e) = k.start_gpu_thread() {
        eprintln!("gyroship: cannot start the GPU thread: {e}");
        std::process::exit(2);
    }
    crate::install(k);
    let outcome = if play { crate::play::run_play() } else { thread::run() };
    crate::with(|k| match &outcome {
        thread::Outcome::Finished => {
            // Play mode: the GPU thread's queued work first; a stop it met ends the run with its code.
            if let Some((code, why)) = k.drain_gpu() {
                k.stop(code, &why);
            }
            k.log("every guest thread ended");
            let summary = k.summary();
            k.log(&summary);
            k.journal.flush();
            k.host_reports();
        }
        thread::Outcome::Deadlock(who) => k.stop(STOP_EXIT, &format!("deadlock: {who}")),
    });
    // The kernel ends with the run, as the thread-local one did with its thread.
    drop(crate::uninstall());
}

#[cfg(test)]
mod tests {
    #[test]
    fn a_folder_without_modules_is_said() {
        let said = super::read_modules(std::path::Path::new("no-such-folder")).err().unwrap();
        assert!(said.starts_with("no-such-folder: no module here") && said.contains("exefs folder"), "{said}");
    }

    #[test]
    fn options_are_known_and_valued_ones_come_once() {
        let line = |s: &str| s.split(' ').map(String::from).collect::<Vec<_>>();
        assert_eq!(super::check_options(&line("boot exefs --play --log a --perf b --language fr")), Ok(()));
        assert!(super::check_options(&line("boot exefs --languge fr")).unwrap_err().contains("--languge"));
        assert!(super::check_options(&line("boot exefs --hooks a --hooks b")).unwrap_err().contains("twice"));
        assert!(super::check_options(&line("boot exefs --log")).unwrap_err().contains("needs a value"));
        // Every option the boot reads is in the list.
        let code = include_str!("boot.rs").split("#[cfg(test)]").next().unwrap();
        let read = super::literals(code);
        assert!(read.contains(&"--log") && read.contains(&"--language") && read.contains(&"--video"), "{read:?}");
        for o in read {
            assert!(super::OPTIONS.iter().any(|(n, _)| *n == o), "{o} is read but not listed");
        }
    }
}
