//! Recompiles every discovered function of an exefs with Cranelift: one
//! object per thread, each with its part of the dispatch table, and a
//! `harness` crate that links them all with the runtime.

use crate::experiment::{harness_main, plain, runtime_path};
use gyroship_analysis::{cfg, decode, discover};
use gyroship_recomp::ir::Op;
use gyroship_recomp::{Callee, Input, clif, lift};
use std::collections::{BTreeMap, HashMap, HashSet};
use std::fmt::Write as _;
use std::path::Path;
use std::time::Instant;

#[derive(Default)]
struct Part {
    functions: usize,
    instructions: u64,
    untranslated: BTreeMap<String, u64>,
    object: u64,
}

pub fn run(exefs: &Path, out: &Path, threads: usize, verify: bool) -> super::Result<()> {
    let p = super::load(exefs)?;
    let d = discover::discover(&p)?;
    let starts: HashSet<u64> = d.functions.iter().map(|f| f.start).collect();
    let stubs: HashMap<u64, u64> = d.stubs.iter().map(|s| (s.addr, s.target)).collect();
    let callee = |t: u64| {
        let t = stubs.get(&t).copied().unwrap_or(t);
        if starts.contains(&t) { Callee::Direct(t) } else { Callee::Dispatch(t) }
    };
    let word = |a: u64| p.read(a, 4).map_or(0, |b| u32::from_le_bytes(b.try_into().expect("4 bytes")));

    // Contiguous parts of about the same code size.
    let threads = threads.max(1);
    let total: u64 = d.functions.iter().map(|f| f.len).sum();
    let mut parts: Vec<&[discover::Function]> = Vec::new();
    let (mut from, mut acc) = (0, 0);
    for (i, f) in d.functions.iter().enumerate() {
        acc += f.len;
        if acc * threads as u64 >= total * (parts.len() as u64 + 1) && parts.len() + 1 < threads {
            parts.push(&d.functions[from..=i]);
            from = i + 1;
        }
    }
    parts.push(&d.functions[from..]);

    std::fs::create_dir_all(out.join("harness/src"))?;
    let start = Instant::now();
    let results: Vec<Result<Part, String>> = std::thread::scope(|scope| {
        let handles: Vec<_> = parts
            .iter()
            .enumerate()
            .map(|(i, part)| {
                let (callee, word) = (&callee, &word);
                let p = &p;
                scope.spawn(move || -> Result<Part, String> {
                    let defined: Vec<(u64, u64)> = part.iter().map(|f| (f.start, f.len)).collect();
                    let mut obj = clif::Object::new(&defined, verify)?;
                    let mut stats = Part { functions: part.len(), ..Part::default() };
                    for f in part.iter() {
                        let m = p.module_at(f.start).ok_or("function outside every module")?;
                        let g = cfg::build(&m.module.image, m.base, f);
                        for b in &g.blocks {
                            for pc in (b.start..b.start + b.len).step_by(4) {
                                stats.instructions += 1;
                                let w = word(pc);
                                if lift::lift(w, pc) == Op::Unhandled {
                                    let name = decode::decode(w).map_or("(undecodable)".to_string(), |i| format!("{:?}", i.opcode));
                                    *stats.untranslated.entry(name).or_default() += 1;
                                }
                            }
                        }
                        obj.function(&Input { function: f, cfg: &g, word, callee }).map_err(|e| format!("{:#x}: {e}", f.start))?;
                    }
                    obj.table(&format!("gyroship_table_{i}"))?;
                    let bytes = obj.finish()?;
                    stats.object = bytes.len() as u64;
                    std::fs::write(out.join(format!("guest{i}.obj")), bytes).map_err(|e| e.to_string())?;
                    Ok(stats)
                })
            })
            .collect();
        handles.into_iter().map(|h| h.join().unwrap_or_else(|_| Err("a compiler thread panicked".into()))).collect()
    });
    let seconds = start.elapsed().as_secs_f64();
    let mut all = Part::default();
    for r in results {
        let r = r?;
        all.functions += r.functions;
        all.instructions += r.instructions;
        all.object += r.object;
        for (k, v) in r.untranslated {
            *all.untranslated.entry(k).or_default() += v;
        }
    }
    let untranslated: u64 = all.untranslated.values().sum();
    let mut top: Vec<(String, u64)> = all.untranslated.into_iter().collect();
    top.sort_by_key(|e| std::cmp::Reverse(e.1));
    let top: Vec<String> = top.iter().take(16).map(|(n, c)| format!("{n} {c}")).collect();
    println!(
        "recompiled  {} functions, {} instructions, {untranslated} untranslated ({:.2}%), in {seconds:.1} s on {} threads",
        all.functions,
        all.instructions,
        100.0 * untranslated as f64 / all.instructions as f64,
        parts.len()
    );
    println!("untranslated: {}", top.join(", "));
    println!("objects     {:.0} MiB in {} files -> {}", all.object as f64 / (1u64 << 20) as f64, parts.len(), out.display());

    // The harness links every object and installs every table part.
    let mut build = String::from("fn main() {\n");
    let mut decls = String::from("\nunsafe extern \"C\" {\n");
    let mut parts_expr = Vec::new();
    for i in 0..parts.len() {
        let object = plain(&out.join(format!("guest{i}.obj")).canonicalize()?);
        let _ = writeln!(build, "    println!(\"cargo:rustc-link-arg={object}\");\n    println!(\"cargo:rerun-if-changed={object}\");");
        let _ = writeln!(decls, "    static gyroship_table_{i}: rt::Entry;\n    static gyroship_table_{i}_len: u64;");
        parts_expr.push(format!("rt::entries(&raw const gyroship_table_{i}, gyroship_table_{i}_len)"));
    }
    build.push_str("}\n");
    decls.push_str("}\n");
    let table = format!("unsafe {{ {}.collect() }}", parts_expr.join(".chain(") + &")".repeat(parts.len() - 1));
    std::fs::write(
        out.join("Cargo.toml"),
        "[workspace]\nresolver = \"3\"\nmembers = [\"harness\"]\n\n[profile.release]\ndebug = false\nincremental = false\n",
    )?;
    std::fs::write(
        out.join("harness/Cargo.toml"),
        format!(
            "[package]\nname = \"harness\"\nversion = \"0.0.0\"\nedition = \"2024\"\npublish = false\n\n[dependencies]\ngyroship-runtime = {{ path = \"{}\" }}\n",
            runtime_path()
        ),
    )?;
    std::fs::write(out.join("harness/build.rs"), build)?;
    std::fs::write(out.join("harness/src/main.rs"), harness_main(&decls, &table))?;
    Ok(())
}
