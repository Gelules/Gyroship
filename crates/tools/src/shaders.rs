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
    Ok(no_end.is_empty())
}

fn hex(w: Word) -> String {
    format!("{:032x}", w.0)
}
