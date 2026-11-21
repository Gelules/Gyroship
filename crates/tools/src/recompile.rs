//! Recompiles every discovered function of an exefs with Cranelift: one
//! object per thread, each with its part of the dispatch table, and a
//! `harness` crate that links them all with the runtime into the game's
//! executable, `gyroship`.
//!
//! PLT stubs and each module's PLT0 are compiled too, as small functions of
//! their own: direct calls skip them (resolved at generation), but an
//! indirect jump to one runs the real code, which reads its GOT slot. A slot
//! that rtld binds lazily points at PLT0 until then.

use crate::experiment::{crate_path, plain, repository, runtime_path, vendor_path};
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
    resume_points: usize,
    untranslated: BTreeMap<String, u64>,
    object: u64,
}

pub fn run(exefs: &Path, out: &Path, threads: usize, verify: bool, probes: Option<&Path>) -> super::Result<()> {
    // The repository first: a misplaced executable fails before minutes of work.
    let (naga, runtime, game, lock) = (vendor_path("naga")?, runtime_path()?, crate_path("gyroship")?, repository()?.join("Cargo.lock"));
    if !lock.is_file() {
        return Err(format!("{}: not found (is the repository whole?)", lock.display()).into());
    }
    // The game's icon is the launcher's: both executables show the same logo.
    let icon = repository()?.join("crates").join("launcher").join("icons").join("icon.ico");
    let icon = icon_res(&std::fs::read(&icon).map_err(|e| format!("{}: {e}", icon.display()))?).map_err(|e| format!("{}: {e}", icon.display()))?;
    let p = super::load(exefs)?;
    let d = discover::discover(&p)?;
    let probes = match probes {
        Some(file) => read_probes(&std::fs::read_to_string(file)?)?,
        None => Vec::new(),
    };
    let probed: HashSet<u64> = probes.iter().map(|p| p.0).collect();
    let probe = |a: u64| probed.contains(&a);
    let word = |a: u64| p.read(a, 4).map_or(0, |b| u32::from_le_bytes(b.try_into().expect("4 bytes")));
    let plt = plt_functions(&d, &word);
    let mut functions = d.functions.clone();
    functions.extend(plt.iter().cloned());
    functions.sort_unstable_by_key(|f| f.start);
    let starts: HashSet<u64> = functions.iter().map(|f| f.start).collect();
    // A call to a stub goes straight to its import; an unresolved one runs the stub.
    let stubs: HashMap<u64, u64> = d.stubs.iter().filter(|s| s.target != 0).map(|s| (s.addr, s.target)).collect();
    let callee = |t: u64| {
        let t = stubs.get(&t).copied().unwrap_or(t);
        if starts.contains(&t) { Callee::Direct(t) } else { Callee::Dispatch(t) }
    };
    let d_functions = &functions;

    // Contiguous parts of about the same code size.
    let threads = threads.max(1);
    let total: u64 = d_functions.iter().map(|f| f.len).sum();
    let mut parts: Vec<&[discover::Function]> = Vec::new();
    let (mut from, mut acc) = (0, 0);
    for (i, f) in d_functions.iter().enumerate() {
        acc += f.len;
        if acc * threads as u64 >= total * (parts.len() as u64 + 1) && parts.len() + 1 < threads {
            parts.push(&d_functions[from..=i]);
            from = i + 1;
        }
    }
    parts.push(&d_functions[from..]);

    std::fs::create_dir_all(out.join("harness/src"))?;
    let start = Instant::now();
    let results: Vec<Result<Part, String>> = std::thread::scope(|scope| {
        let handles: Vec<_> = parts
            .iter()
            .enumerate()
            .map(|(i, part)| {
                let (callee, word, probe) = (&callee, &word, &probe);
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
                        obj.function(&Input { function: f, cfg: &g, word, callee, probe }).map_err(|e| format!("{:#x}: {e}", f.start))?;
                    }
                    stats.resume_points = obj.resume_points();
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
        all.resume_points += r.resume_points;
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
    println!(
        "            {} of them PLT stubs and PLT0s, {} resume points, {} probes",
        plt.len(),
        all.resume_points,
        probes.len()
    );
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
    // The icon as a resource file, which link.exe takes as it takes the objects.
    std::fs::write(out.join("gyroship.res"), &icon)?;
    let res = plain(&out.join("gyroship.res").canonicalize()?);
    let _ = writeln!(build, "    println!(\"cargo:rustc-link-arg={res}\");\n    println!(\"cargo:rerun-if-changed={res}\");");
    build.push_str("}\n");
    decls.push_str("}\n");
    let names: String = probes.iter().map(|(a, n)| format!("({a:#x}, {n:?}.to_string()), ")).collect();
    let table = format!(
        "{{ rt::install_probes(vec![{names}]); unsafe {{ {}.collect() }} }}",
        parts_expr.join(".chain(") + &")".repeat(parts.len() - 1)
    );
    // Line tables: `--profile` then names every host function from the PDB, not
    // only its public symbols (each private function was charged to the public
    // one before it). The guest's own objects are linked as they are. The
    // repository's patched naga, as in its own workspace.
    std::fs::write(
        out.join("Cargo.toml"),
        format!("[workspace]\nresolver = \"3\"\nmembers = [\"harness\"]\n\n[profile.release]\ndebug = \"line-tables-only\"\nincremental = false\n\n[patch.crates-io]\nnaga = {{ path = \"{naga}\" }}\n"),
    )?;
    // The repository's lock, at each port: the game builds with the crate versions its replays
    // checked, not the newest ones (Cargo adds `harness` to it).
    std::fs::copy(&lock, out.join("Cargo.lock"))?;
    std::fs::write(
        out.join("harness/Cargo.toml"),
        format!(
            "[package]\nname = \"harness\"\nversion = \"0.0.0\"\nedition = \"2024\"\npublish = false\n\n[[bin]]\nname = \"gyroship\"\npath = \"src/main.rs\"\n\n[dependencies]\ngyroship-runtime = {{ path = \"{runtime}\" }}\ngyroship = {{ path = \"{game}\" }}\n"
        ),
    )?;
    std::fs::write(out.join("harness/build.rs"), build)?;
    // The game's harness: `boot <exefs>` in a window, or --headless (D28); its other modes are the runtime's.
    std::fs::write(out.join("harness/src/main.rs"), format!("use gyroship_runtime as rt;\n{decls}\nfn main() {{\n    gyroship::main({table});\n}}\n"))?;
    Ok(())
}

/// `<hex address> <name>` per line; `#` starts a comment.
fn read_probes(text: &str) -> super::Result<Vec<(u64, String)>> {
    let mut out = Vec::new();
    for line in text.lines() {
        let line = line.split('#').next().unwrap_or("").trim();
        if line.is_empty() {
            continue;
        }
        let (a, name) = line.split_once(char::is_whitespace).ok_or_else(|| format!("probe without a name: {line}"))?;
        out.push((u64::from_str_radix(a.trim_start_matches("0x"), 16).map_err(|e| format!("{line}: {e}"))?, name.trim().to_string()));
    }
    Ok(out)
}

/// The PLT stubs, and the PLT0 just before the first stub of each module
/// (`stp x16, x30, [sp, #-16]!` then the jump to rtld's resolver), as
/// functions with no unwind information.
fn plt_functions(d: &discover::Discovery, word: &dyn Fn(u64) -> u32) -> Vec<discover::Function> {
    const PLT0_FIRST: u32 = 0xA9BF_7BF0;
    let function = |start: u64, len: u64| discover::Function { start, len, unwind: false, entries: vec![start] };
    let mut out: Vec<discover::Function> = d.stubs.iter().map(|s| function(s.addr, 16)).collect();
    // Stubs are contiguous: the first of a run has no stub 16 bytes before it.
    let addrs: HashSet<u64> = d.stubs.iter().map(|s| s.addr).collect();
    for first in d.stubs.iter().map(|s| s.addr).filter(|a| !addrs.contains(&(a - 16))) {
        let plt0 = first - 0x20;
        if word(plt0) == PLT0_FIRST && d.function_at(plt0).is_none() {
            out.push(function(plt0, 0x20));
        }
    }
    out
}

/// An .ico file as a Windows resource file (.res): each image an icon (RT_ICON, from 1), and
/// the group of them as icon 1 (RT_GROUP_ICON), the one Explorer shows for the executable and
/// the game's window loads (`Icon::from_resource(1, …)`).
fn icon_res(ico: &[u8]) -> Result<Vec<u8>, String> {
    let word = |at: usize| ico.get(at..at + 2).map(|b| u16::from_le_bytes([b[0], b[1]]));
    if (word(0), word(2)) != (Some(0), Some(1)) {
        return Err("not an icon file".into());
    }
    let count = word(4).ok_or("not an icon file")?;
    // A resource file starts with an empty entry.
    let mut res = resource(0, 0, 0, &[]);
    let mut group = [0u16, 1, count].iter().flat_map(|w| w.to_le_bytes()).collect::<Vec<u8>>();
    for id in 1..=count {
        let entry = ico.get(6 + 16 * usize::from(id - 1)..).and_then(|e| e.get(..16)).ok_or("icon directory cut short")?;
        let [size, at] = [8, 12].map(|i| u32::from_le_bytes(entry[i..i + 4].try_into().expect("4 bytes")) as usize);
        res.extend(resource(3, id, 0x1010, ico.get(at..at + size).ok_or("icon image cut short")?));
        // The group's entry: the directory's, with the image's id in place of its offset.
        group.extend(&entry[..12]);
        group.extend(id.to_le_bytes());
    }
    res.extend(resource(14, 1, 0x1030, &group));
    Ok(res)
}

/// One entry of a resource file: its header (data size, header size, type and name as numbers,
/// data version, memory flags, language neutral, version, characteristics), then its data,
/// padded to 4 bytes.
fn resource(kind: u16, name: u16, flags: u16, data: &[u8]) -> Vec<u8> {
    let mut out: Vec<u8> = [data.len() as u32, 32].iter().flat_map(|d| d.to_le_bytes()).collect();
    out.extend([0xffff, kind, 0xffff, name].iter().flat_map(|w| w.to_le_bytes()));
    out.extend([0u8; 4]);
    out.extend(flags.to_le_bytes());
    out.extend([0u8; 10]);
    out.extend(data);
    out.resize(out.len().next_multiple_of(4), 0);
    out
}

#[cfg(test)]
mod tests {
    /// Two images become two icons and a group that names them, each entry where a resource
    /// file puts it; a file that is no icon is refused.
    #[test]
    fn an_icon_becomes_a_resource_file() {
        let images: [&[u8]; 2] = [b"first image", b"second"];
        let mut ico = vec![0, 0, 1, 0, 2, 0];
        let mut at = 6 + 2 * 16;
        for (size, image) in [16u8, 32].into_iter().zip(images) {
            ico.extend([size, size, 0, 0, 1, 0, 32, 0]);
            ico.extend((image.len() as u32).to_le_bytes());
            ico.extend((at as u32).to_le_bytes());
            at += image.len();
        }
        ico.extend(images.concat());
        let res = super::icon_res(&ico).unwrap();
        // The entries in order: (data size, type, name, data), each from a 4-byte boundary.
        let mut entries = Vec::new();
        let mut at = 0;
        while at < res.len() {
            let dword = |i: usize| u32::from_le_bytes(res[at + i..at + i + 4].try_into().unwrap());
            let (size, header) = (dword(0) as usize, dword(4) as usize);
            assert_eq!((header, dword(8) & 0xffff, dword(12) & 0xffff), (32, 0xffff, 0xffff), "header at {at}");
            entries.push((size, dword(8) >> 16, dword(12) >> 16, res[at + header..at + header + size].to_vec()));
            at = (at + header + size).next_multiple_of(4);
        }
        assert_eq!(entries[..3], [(0, 0, 0, vec![]), (11, 3, 1, b"first image".to_vec()), (6, 3, 2, b"second".to_vec())]);
        let group = &entries[3];
        assert_eq!((group.1, group.2), (14, 1));
        assert_eq!(group.3, [&[0u8, 0, 1, 0, 2, 0][..], &[16, 16, 0, 0, 1, 0, 32, 0, 11, 0, 0, 0, 1, 0], &[32, 32, 0, 0, 1, 0, 32, 0, 6, 0, 0, 0, 2, 0]].concat());
        assert!(super::icon_res(b"PNG, not an icon").is_err());
    }
}
