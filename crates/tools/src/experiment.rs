//! The backend experiment of M1 (phase 0 report, section 4): generate a
//! sample of real functions of `main` with one backend, as a project that
//! `cargo build` turns into a binary, so build time and memory can be measured.

use gyroship_analysis::{cfg, decode, discover};
use gyroship_loader::link::Program;
use gyroship_recomp::ir::Op;
use gyroship_recomp::{Callee, Input, lift, rust};
use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet};
use std::fmt::Write as _;
use std::path::Path;

pub struct Sample<'a> {
    pub program: &'a Program,
    pub functions: Vec<(&'a discover::Function, cfg::Cfg)>,
    stubs: HashMap<u64, u64>,
    set: HashSet<u64>,
}

impl Sample<'_> {
    pub fn word(&self, a: u64) -> u32 {
        self.program.read(a, 4).map_or(0, |b| u32::from_le_bytes(b.try_into().expect("4 bytes")))
    }
    pub fn callee(&self, t: u64) -> Callee {
        let t = self.stubs.get(&t).copied().unwrap_or(t);
        if self.set.contains(&t) { Callee::Direct(t) } else { Callee::Dispatch(t) }
    }
    /// Generated functions that `f` calls or tail-calls by name.
    fn callees(&self, f: &discover::Function, g: &cfg::Cfg) -> BTreeSet<u64> {
        let mut out = BTreeSet::new();
        for b in &g.blocks {
            for pc in (b.start..b.start + b.len).step_by(4) {
                let target = match lift::lift(self.word(pc), pc) {
                    Op::Call(t) | Op::Jump(t) | Op::Branch(_, t) => t,
                    _ => continue,
                };
                let local = (f.start..f.end()).contains(&target) && g.blocks.binary_search_by_key(&target, |b| b.start).is_ok();
                if let (false, Callee::Direct(t)) = (local, self.callee(target)) {
                    out.insert(t);
                }
            }
            // Falling off the end of a block that ends the function is a tail call too.
            let end = b.start + b.len;
            if let (false, Callee::Direct(t)) = ((f.start..f.end()).contains(&end), self.callee(end)) {
                out.insert(t);
            }
        }
        out
    }
}

/// Every `every`-th FDE function of `main`, so sizes follow the module's.
pub fn sample<'a>(program: &'a Program, d: &'a discover::Discovery, every: usize) -> super::Result<Sample<'a>> {
    let main = program.modules.iter().find(|m| m.name == "main").ok_or("no main module")?;
    let picked: Vec<&discover::Function> =
        d.functions.iter().filter(|f| f.unwind && (main.base..main.end()).contains(&f.start)).step_by(every.max(1)).collect();
    picked_sample(program, d, picked)
}

/// The functions that contain the given guest addresses, and with `callees`
/// every function they reach by direct calls and jumps, transitively.
pub fn chosen<'a>(program: &'a Program, d: &'a discover::Discovery, addrs: &[u64], callees: bool) -> super::Result<Sample<'a>> {
    let mut picked: Vec<&discover::Function> =
        addrs.iter().map(|&a| d.function_at(a).ok_or_else(|| format!("no function at {a:#x}"))).collect::<Result<_, _>>()?;
    if callees {
        let stubs: HashMap<u64, u64> = d.stubs.iter().map(|s| (s.addr, s.target)).collect();
        let mut seen: HashSet<u64> = picked.iter().map(|f| f.start).collect();
        let mut i = 0;
        while i < picked.len() {
            let f = picked[i];
            i += 1;
            let m = program.module_at(f.start).ok_or("function outside every module")?;
            for b in &cfg::build(&m.module.image, m.base, f).blocks {
                for pc in (b.start..b.start + b.len).step_by(4) {
                    let w = program.read(pc, 4).map_or(0, |b| u32::from_le_bytes(b.try_into().expect("4 bytes")));
                    let (Op::Call(t) | Op::Jump(t) | Op::Branch(_, t)) = lift::lift(w, pc) else { continue };
                    let t = stubs.get(&t).copied().unwrap_or(t);
                    if let Some(g) = d.function_at(t).filter(|g| g.start == t && seen.insert(t)) {
                        picked.push(g);
                    }
                }
            }
        }
    }
    picked_sample(program, d, picked)
}

fn picked_sample<'a>(program: &'a Program, d: &'a discover::Discovery, picked: Vec<&'a discover::Function>) -> super::Result<Sample<'a>> {
    let mut functions = Vec::new();
    for f in &picked {
        let m = program.module_at(f.start).ok_or("function outside every module")?;
        functions.push((*f, cfg::build(&m.module.image, m.base, f)));
    }
    let set = picked.iter().map(|f| f.start).collect();
    let stubs = d.stubs.iter().map(|s| (s.addr, s.target)).collect();
    Ok(Sample { program, functions, stubs, set })
}

/// The harness `main`: installs the dispatch table (`table`, an expression
/// of type `Vec<rt::Entry>` that `prelude` may define), then runs the
/// runtime's test modes (see `gyroship_runtime::harness`).
pub fn harness_main(prelude: &str, table: &str) -> String {
    format!("use gyroship_runtime as rt;\n{prelude}\nfn main() {{\n    rt::harness::main({table});\n}}\n")
}

/// Instruction count, untranslated count, and the most frequent untranslated opcodes.
pub fn coverage(s: &Sample) -> (u64, u64, Vec<(String, u64)>) {
    let (mut total, mut unhandled) = (0, 0);
    let mut by: BTreeMap<String, u64> = BTreeMap::new();
    for (_, g) in &s.functions {
        for b in &g.blocks {
            for pc in (b.start..b.start + b.len).step_by(4) {
                total += 1;
                let w = s.word(pc);
                if lift::lift(w, pc) == Op::Unhandled {
                    unhandled += 1;
                    let name = decode::decode(w).map_or("(undecodable)".to_string(), |i| format!("{:?}", i.opcode));
                    *by.entry(name).or_default() += 1;
                }
            }
        }
    }
    let mut top: Vec<(String, u64)> = by.into_iter().collect();
    top.sort_by_key(|e| std::cmp::Reverse(e.1));
    top.truncate(12);
    (total, unhandled, top)
}

/// A path as Cargo manifests and linker arguments accept it: forward slashes, no `\\?\` prefix.
pub fn plain(p: &Path) -> String {
    p.display().to_string().replace('\\', "/").trim_start_matches("//?/").to_string()
}

pub fn runtime_path() -> String {
    // The runtime crate next to this one; generated projects are never committed.
    let p = Path::new(env!("CARGO_MANIFEST_DIR")).join("../runtime");
    plain(&p.canonicalize().unwrap_or(p))
}

/// Writes a cargo workspace: crates `g0`, `g1`... of `files_per_crate` files of
/// `per_file` functions each, and a `harness` binary that installs them all in
/// the dispatch table. Returns the bytes of Rust written.
pub fn write_rust(s: &Sample, out: &Path, per_file: usize, files_per_crate: usize, opt_level: u8) -> super::Result<u64> {
    std::fs::create_dir_all(out)?;
    let runtime = runtime_path();
    let chunks: Vec<&[(&discover::Function, cfg::Cfg)]> = s.functions.chunks(per_file.max(1)).collect();
    let crates: Vec<&[&[(&discover::Function, cfg::Cfg)]]> = chunks.chunks(files_per_crate.max(1)).collect();
    let mut bytes = 0u64;
    let mut table = String::new();
    let word = |a: u64| s.word(a);
    let callee = |t: u64| s.callee(t);
    for (ci, files) in crates.iter().enumerate() {
        let dir = out.join(format!("g{ci}/src"));
        std::fs::create_dir_all(&dir)?;
        std::fs::write(
            out.join(format!("g{ci}/Cargo.toml")),
            format!("[package]\nname = \"g{ci}\"\nversion = \"0.0.0\"\nedition = \"2024\"\npublish = false\n\n[dependencies]\ngyroship-runtime = {{ path = \"{runtime}\" }}\n"),
        )?;
        let mut lib = String::from("#![allow(warnings, clippy::all)]\n");
        for (fi, file) in files.iter().enumerate() {
            let _ = writeln!(lib, "pub mod f{fi};");
            let defined: HashSet<u64> = file.iter().map(|(f, _)| f.start).collect();
            let mut external = BTreeSet::new();
            for (f, g) in file.iter() {
                external.extend(s.callees(f, g).into_iter().filter(|t| !defined.contains(t)));
            }
            let mut src = String::from("use gyroship_runtime::{self as rt, Cpu};\n\n");
            if !external.is_empty() {
                src.push_str("unsafe extern \"C\" {\n");
                for t in external {
                    let _ = writeln!(src, "    fn {}(c: *mut Cpu, resume: u64);", rust::name(t));
                }
                src.push_str("}\n\n");
            }
            for (f, g) in file.iter() {
                rust::function(&mut src, &Input { function: f, cfg: g, word: &word, callee: &callee });
                let _ = writeln!(table, "    rt::Entry {{ addr: {:#x}, len: {:#x}, function: g{ci}::f{fi}::{} }},", f.start, f.len, rust::name(f.start));
            }
            bytes += src.len() as u64;
            std::fs::write(dir.join(format!("f{fi}.rs")), src)?;
        }
        std::fs::write(dir.join("lib.rs"), lib)?;
    }
    let members: Vec<String> = (0..crates.len()).map(|i| format!("\"g{i}\"")).chain(["\"harness\"".to_string()]).collect();
    std::fs::write(
        out.join("Cargo.toml"),
        format!(
            "[workspace]\nresolver = \"3\"\nmembers = [{}]\n\n[profile.release]\nopt-level = {opt_level}\ndebug = false\nincremental = false\n",
            members.join(", ")
        ),
    )?;
    std::fs::create_dir_all(out.join("harness/src"))?;
    let deps: String = (0..crates.len()).map(|i| format!("g{i} = {{ path = \"../g{i}\" }}\n")).collect();
    std::fs::write(
        out.join("harness/Cargo.toml"),
        format!("[package]\nname = \"harness\"\nversion = \"0.0.0\"\nedition = \"2024\"\npublish = false\n\n[dependencies]\ngyroship-runtime = {{ path = \"{runtime}\" }}\n{deps}"),
    )?;
    std::fs::write(
        out.join("harness/src/main.rs"),
        harness_main(&format!("\nstatic TABLE: &[rt::Entry] = &[\n{table}];\n"), "TABLE.to_vec()"),
    )?;
    Ok(bytes)
}

/// Writes `guest.obj` (every sampled function, compiled by Cranelift) and a
/// `harness` crate that links it with the runtime. Returns the object size
/// and the code generation time in seconds.
pub fn write_clif(s: &Sample, out: &Path, verify: bool) -> super::Result<(u64, f64)> {
    use gyroship_recomp::clif;
    std::fs::create_dir_all(out.join("harness/src"))?;
    let start = std::time::Instant::now();
    let defined: Vec<(u64, u64)> = s.functions.iter().map(|(f, _)| (f.start, f.len)).collect();
    let mut obj = clif::Object::new(&defined, verify)?;
    let word = |a: u64| s.word(a);
    let callee = |t: u64| s.callee(t);
    for (f, g) in &s.functions {
        obj.function(&Input { function: f, cfg: g, word: &word, callee: &callee }).map_err(|e| format!("{:#x}: {e}", f.start))?;
    }
    let bytes = obj.finish()?;
    let seconds = start.elapsed().as_secs_f64();
    let path = out.join("guest.obj");
    std::fs::write(&path, &bytes)?;
    let object = plain(&path.canonicalize()?);
    let runtime = runtime_path();
    std::fs::write(
        out.join("Cargo.toml"),
        "[workspace]\nresolver = \"3\"\nmembers = [\"harness\"]\n\n[profile.release]\ndebug = false\nincremental = false\n",
    )?;
    std::fs::write(
        out.join("harness/Cargo.toml"),
        format!("[package]\nname = \"harness\"\nversion = \"0.0.0\"\nedition = \"2024\"\npublish = false\n\n[dependencies]\ngyroship-runtime = {{ path = \"{runtime}\" }}\n"),
    )?;
    std::fs::write(
        out.join("harness/build.rs"),
        format!("fn main() {{\n    println!(\"cargo:rustc-link-arg={object}\");\n    println!(\"cargo:rerun-if-changed={object}\");\n}}\n"),
    )?;
    let mut decls = String::new();
    let mut table = String::new();
    for &(a, len) in &defined {
        let _ = writeln!(decls, "    fn {}(c: *mut rt::Cpu, resume: u64);", rust::name(a));
        let _ = writeln!(table, "    rt::Entry {{ addr: {a:#x}, len: {len:#x}, function: {} }},", rust::name(a));
    }
    std::fs::write(
        out.join("harness/src/main.rs"),
        harness_main(&format!("\nunsafe extern \"C\" {{\n{decls}}}\n\nstatic TABLE: &[rt::Entry] = &[\n{table}];\n"), "TABLE.to_vec()"),
    )?;
    Ok((bytes.len() as u64, seconds))
}
