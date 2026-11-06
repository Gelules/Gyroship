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
pub fn map_module(m: &mut Memory, base: u64, module: &Module) -> Result<u64, String> {
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

/// A kernel with the modules mapped and the main thread ready at rtld's entry.
pub fn boot(exefs: &Path, journal: Journal, config: Config) -> Result<Kernel, String> {
    let modules = read_modules(exefs)?;
    let mut m = Memory::new().map_err(|e| format!("reserving guest memory: {e:?}"))?;
    let mut k_lines = Vec::new();
    for (name, module) in &modules {
        let base = link::base_of(name).ok_or_else(|| format!("no base for {name}"))?;
        let end = map_module(&mut m, base, module)?;
        k_lines.push(format!("module {name} at {base:#x}..{end:#x}"));
    }
    m.map(MAIN_STACK.0, MAIN_STACK.1, Block::new(state::STACK, perm::RW)).map_err(|e| format!("main stack: {e:?}"))?;
    let mut k = Kernel::new(m, journal, config);
    for line in k_lines {
        k.log(&line);
    }
    let npdm = std::fs::read(exefs.join("main.npdm")).map_err(|e| format!("main.npdm: {e}"))?;
    k.allowed_services = npdm_services(&npdm).ok_or("main.npdm: no service access control")?;
    let services = format!("npdm: {} services granted: {}", k.allowed_services.len(), k.allowed_services.join(" "));
    k.log(&services);
    let romfs = exefs.parent().map(|p| p.join("romfs")).filter(|p| p.is_dir()).ok_or("no romfs folder next to the exefs")?;
    let image = crate::fs::RomFs::build(&romfs).map_err(|e| format!("{}: {e}", romfs.display()))?;
    k.log(&format!("romfs: an image of {} bytes", image.size()));
    k.romfs = Some(image);
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

/// The probe hook: the first pass is journaled; the oracle runs at `nn::init::Start`.
pub(crate) unsafe fn on_probe(cpu: *mut Cpu, pc: u64, name: &str) {
    // SAFETY: the running thread's context, read before anything else runs.
    let (x, lr) = unsafe { ((*cpu).x, (*cpu).x[30]) };
    crate::with(|k| {
        let caller = k.symbolize(lr);
        k.log(&format!("probe {name} at {pc:#x}: x0={:#x} x1={:#x} x2={:#x} x3={:#x} x8={:#x}, called from {lr:#x} {caller}", x[0], x[1], x[2], x[3], x[8]));
        // Probes named `regs:...` also log every register, `stack:...` their frame chain.
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

/// The generated harness's `main`: `boot <exefs> [--log FILE] [--explore]
/// [--all-svcs] [--seconds N] [--frames N]` runs the game (for N virtual
/// seconds, 60 by default, or until N frames are presented); any other mode
/// is the runtime's.
pub fn main(table: Vec<rt::Entry>) {
    let args: Vec<String> = std::env::args().skip(1).collect();
    if args.first().map(String::as_str) != Some("boot") {
        return rt::harness::main(table);
    }
    let Some(exefs) = args.get(1) else {
        eprintln!("usage: harness boot <exefs> [--log FILE] [--explore] [--all-svcs]");
        std::process::exit(2);
    };
    let flag = |f: &str| args.iter().any(|a| a == f);
    let log = args.iter().position(|a| a == "--log").and_then(|i| args.get(i + 1)).map_or("boot.log", String::as_str);
    let mut journal = Journal::file(Path::new(log)).unwrap_or_else(|e| panic!("{log}: {e}"));
    journal.all_svcs = flag("--all-svcs");
    // A virtual time limit, 60 s unless --seconds says otherwise.
    let seconds: u64 = args.iter().position(|a| a == "--seconds").and_then(|i| args.get(i + 1)).and_then(|s| s.parse().ok()).unwrap_or(60);
    let frames: Option<u64> = args.iter().position(|a| a == "--frames").and_then(|i| args.get(i + 1)).and_then(|s| s.parse().ok());
    let value = |f: &str| args.iter().position(|a| a == f).and_then(|i| args.get(i + 1)).cloned();
    let save_dir = value("--save").map(std::path::PathBuf::from);
    let config = Config { explore: flag("--explore"), time_limit: Some(seconds * 31_250_000), frames, save_dir, vsync: !flag("--no-vsync"), gpfifo_dump: value("--dump-gpfifo").map(std::path::PathBuf::from), ..Config::default() };
    let installed = table.len();
    rt::install(table);
    mem::install_fault_handler();
    let mut k = boot(Path::new(exefs), journal, config).unwrap_or_else(|e| {
        eprintln!("gyroship: boot failed: {e}");
        std::process::exit(2);
    });
    k.log(&format!("{installed} recompiled functions installed"));
    crate::install(k);
    let outcome = thread::run();
    crate::with(|k| match &outcome {
        thread::Outcome::Finished => {
            k.log("every guest thread ended");
            k.journal.flush();
        }
        thread::Outcome::Deadlock(who) => k.stop(STOP_EXIT, &format!("deadlock: {who}")),
    });
}
