//! `shaders`: the offline inventory of the game's shaders. Every
//! `nvn_gen2.raw` blob of `data.zip` is parsed, its program found (up to
//! the closing self-branch), and its instructions counted by opcode and
//! stage. With `--samples DIR`, real instruction words go to one file per
//! opcode (game data: never committed).

use gyroship_gpu::datazip::DataZip;
use gyroship_gpu::sass::{self, Blob, NOP, Stage, Word};
use std::collections::BTreeMap;
use std::io::Write;
use std::path::Path;

/// Instruction words kept per opcode with `--samples`.
const SAMPLES: usize = 400;

pub fn run(zip: &Path, samples: Option<&Path>) -> super::Result<bool> {
    let mut z = DataZip::open(zip)?;
    let entries = z.entries.clone();
    let mut stages: BTreeMap<Stage, usize> = BTreeMap::new();
    let mut opcodes: BTreeMap<u16, [usize; 6]> = BTreeMap::new();
    let mut kept: BTreeMap<u16, Vec<String>> = BTreeMap::new();
    // Per opcode: a first word, the bits that vary from it, and how many fail to decode.
    let mut seen: BTreeMap<u16, (u128, u128, usize)> = BTreeMap::new();
    let (mut blobs, mut no_end, mut instructions, mut trailer) = (0usize, Vec::new(), 0usize, 0usize);
    for e in &entries {
        if e.size < 0x18 || z.head(e, 4)? != sass::BLOB_MAGIC.to_le_bytes() {
            continue;
        }
        let bytes = z.read(e)?;
        let Some(blob) = Blob::parse(&bytes) else {
            println!("{}: shader magic but a bad header", e.name);
            continue;
        };
        blobs += 1;
        let stage = blob.stage();
        *stages.entry(stage).or_default() += 1;
        let sass = blob.sass();
        let Some(program) = sass::program(sass) else {
            no_end.push(e.name.clone());
            continue;
        };
        // After the program: NOP padding, zeros, then data (tables).
        trailer += sass::words(&sass[program.len()..]).take_while(|w| w.opcode() == NOP || w.0 == 0).count();
        for (i, w) in sass::words(program).enumerate() {
            instructions += 1;
            opcodes.entry(w.opcode()).or_default()[stage as usize] += 1;
            let s = seen.entry(w.opcode()).or_insert((w.0, 0, 0));
            s.1 |= s.0 ^ w.0;
            if sass::decode(w).is_none() {
                s.2 += 1;
            }
            if samples.is_some() {
                let v = kept.entry(w.opcode()).or_default();
                if v.len() < SAMPLES && !v.iter().any(|s| s.ends_with(&hex(w))) {
                    v.push(format!("{} {stage:?} {:#06x} {}", e.name, i * 16, hex(w)));
                }
            }
        }
    }
    println!("blobs        {blobs}: {}", stages.iter().map(|(s, n)| format!("{s:?} {n}")).collect::<Vec<_>>().join(", "));
    println!("programs     {} ending in the self-branch, {} without ({})", blobs - no_end.len(), no_end.len(), no_end.iter().take(8).cloned().collect::<Vec<_>>().join(" "));
    println!("instructions {instructions} in programs, in {} distinct opcodes; {trailer} padding words after them", opcodes.len());
    println!("opcode    total   vertex  tessctl tesseval geometry pixel   compute");
    let mut sorted: Vec<_> = opcodes.iter().collect();
    sorted.sort_by_key(|(op, n)| (std::cmp::Reverse(n.iter().sum::<usize>()), **op));
    for (op, n) in sorted {
        println!("{op:#05x} {:>9} {:>8} {:>8} {:>8} {:>8} {:>7} {:>7}", n.iter().sum::<usize>(), n[0], n[1], n[2], n[3], n[4], n[5]);
    }
    // The decoder on every instruction: failures, and bits that vary but are never read.
    let failed: usize = seen.values().map(|s| s.2).sum();
    println!("decoder      {} of {instructions} instructions decoded ({failed} not)", instructions - failed);
    let mut clean = true;
    for (op, (_, varying, fails)) in &seen {
        let unexplained = varying & !sass::explained(*op);
        if *fails > 0 || unexplained != 0 {
            clean = false;
            println!("  {op:#05x}: {fails} not decoded; varying bits not read: {}", sass::ranges(unexplained));
        }
    }
    if clean {
        println!("             every opcode decodes, and every varying bit is read");
    }
    if let Some(dir) = samples {
        std::fs::create_dir_all(dir)?;
        for (op, lines) in &kept {
            let mut f = std::fs::File::create(dir.join(format!("op-{op:03x}.txt")))?;
            writeln!(f, "# entry stage offset word (128 bits, hex, high first)")?;
            for l in lines {
                writeln!(f, "{l}")?;
            }
        }
        println!("samples      up to {SAMPLES} words per opcode in {}", dir.display());
    }
    Ok(no_end.is_empty() && clean)
}

/// Lists a dumped graphics program (SPH then SASS): the SPH words, then each
/// instruction, decoded.
pub fn list(path: &Path) -> super::Result<()> {
    let code = std::fs::read(path)?;
    // Compute programs have no SPH.
    let start = if sass::stage(&code) == sass::Stage::Compute { 0 } else { sass::SPH_SIZE };
    let sph: Vec<String> = code.get(..start).unwrap_or_default().as_chunks::<4>().0.iter().map(|w| format!("{:08x}", u32::from_le_bytes(*w))).collect();
    println!("sph {}", if sph.is_empty() { "none (compute)".to_string() } else { sph.join(" ") });
    let program = sass::program(code.get(start..).unwrap_or_default()).ok_or("no closing self-branch")?;
    for (i, w) in sass::words(program).enumerate() {
        let (p, not) = w.guard();
        let guard = if w.always() { String::new() } else { format!("@{}P{p} ", if not { "!" } else { "" }) };
        match sass::decode(w) {
            Some(op) => println!("{:#06x} {guard}{op:?}", i * 16),
            None => println!("{:#06x} {guard}unknown {:032x}", i * 16, w.0),
        }
    }
    Ok(())
}

/// Translates dumped graphics programs to WGSL, validates each with naga,
/// and writes `{key}.wgsl` into `out` (game-derived: keep it under Data).
pub fn translate(files: &[std::path::PathBuf], out: Option<&Path>) -> super::Result<bool> {
    let mut all_ok = true;
    for path in files {
        let code = std::fs::read(path)?;
        let name = path.display();
        match gyroship_gpu::shader::translate(&code) {
            Err(e) => {
                all_ok = false;
                println!("{name}: not translated: {e}");
                // Every refused instruction, grouped by what is missing.
                let mut gaps: std::collections::BTreeMap<String, Vec<(usize, String)>> = Default::default();
                for g in gyroship_gpu::shader::gaps(&code, &Default::default()) {
                    let what = g.what.split(" {").next().unwrap_or(&g.what).to_string();
                    gaps.entry(what).or_default().push((g.at.unwrap_or(0), g.what));
                }
                for (what, seen) in gaps {
                    let (at, example) = &seen[0];
                    println!("  {} x {what}, first at {at:#x}: {}", seen.len(), example.chars().take(400).collect::<String>());
                }
            }
            Ok(t) => {
                let valid = gyroship_gpu::shader::validate(&t.wgsl);
                let i = &t.interface;
                println!(
                    "{name}: {:?} {:016x}: {} lines of WGSL, {}; cbufs {:?}, textures {:?}, attributes {:?}, varyings {:?}, targets {:?}",
                    t.stage,
                    t.key,
                    t.wgsl.lines().count(),
                    if valid.is_ok() { "valid" } else { "INVALID" },
                    i.cbufs,
                    i.textures.iter().map(|x| x.handle.clone()).collect::<Vec<_>>(),
                    i.attributes,
                    i.varyings,
                    i.targets
                );
                if let Err(e) = &valid {
                    all_ok = false;
                    println!("{e}");
                }
                if let Some(dir) = out {
                    std::fs::create_dir_all(dir)?;
                    std::fs::write(dir.join(format!("{:016x}.wgsl", t.key)), &t.wgsl)?;
                }
            }
        }
    }
    Ok(all_ok)
}

fn hex(w: Word) -> String {
    format!("{:032x}", w.0)
}
