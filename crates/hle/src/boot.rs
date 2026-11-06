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
    let rtld = link::base_of("rtld").expect("rtld's base");
    k.create_main(rtld, MAIN_STACK.0 + MAIN_STACK.1);
    k.oracle = Some(Program::link(modules).map_err(|e| format!("link: {}", e.0))?);
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
pub(crate) unsafe fn on_probe(_cpu: *mut Cpu, pc: u64, name: &str) {
    crate::with(|k| {
        k.log(&format!("probe {name} at {pc:#x}"));
        if name == ORACLE_PROBE {
            k.check_oracle();
        }
    });
}

/// The generated harness's `main`: `boot <exefs> [--log FILE] [--explore]
/// [--all-svcs]` runs the game; any other mode is the runtime's.
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
    let config = Config { explore: flag("--explore"), ..Config::default() };
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
