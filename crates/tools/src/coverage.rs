//! `shader-coverage` (D110, step 1): how much of the game's shaders can be made before play.
//! - Every program of data.zip, translated as it is (graphics stages with no pairing, compute
//!   with a placeholder block size).
//! - The jump tables of those that have some, read from their blob at a few candidate places,
//!   against the targets real play found: the entries of the shader cache, every build's.
//! - The programs play used that data.zip lacks, and where their code lies in the game's
//!   modules.
//! - How many variants play made of each vertex, pixel and compute program, and what a vertex
//!   program's variants depend on (its pixel program's varyings, integer attributes, clip
//!   planes).
//!
//! Its output names programs by their keys and offsets only; keep it under Data all the same.

use gyroship_gpu::datazip::DataZip;
use gyroship_gpu::draw::{TABLE_MAX, jump_targets};
use gyroship_gpu::sass::{self, Blob, Stage};
use gyroship_gpu::shader::{self, JumpTable, Options, cache, sph::Sph};
use std::collections::{BTreeMap, BTreeSet, HashSet};
use std::path::Path;

/// A program of data.zip: its stage, its code as the GPU reads it (SPH then program for the
/// graphics stages), the blob's whole code section (the program, padding, then data), its
/// control section, and how many blobs hold it.
struct Program {
    stage: Stage,
    code: Vec<u8>,
    section: Vec<u8>,
    control: Vec<u8>,
    blobs: usize,
}

/// Where a program's table bank may start in its blob, by name: the start of the data after
/// the program's padding, the program's end, the section's start, the program's end rounded
/// up to 256 bytes, and the control section's start.
const RULES: [&str; 5] = ["data", "end", "section", "end256", "control"];

pub fn run(zip: &Path, root: &Path, exefs: Option<&Path>) -> super::Result<bool> {
    let programs = inventory(zip)?;
    let by_stage = |pick: &dyn Fn(&u64) -> bool| -> String {
        let mut n: BTreeMap<Stage, usize> = BTreeMap::new();
        for (_, p) in programs.iter().filter(|(k, _)| pick(k)) {
            *n.entry(p.stage).or_default() += 1;
        }
        n.iter().map(|(s, c)| format!("{s:?} {c}")).collect::<Vec<_>>().join(", ")
    };
    let blobs: usize = programs.values().map(|p| p.blobs).sum();
    println!("data.zip     {blobs} blobs, {} distinct programs: {}", programs.len(), by_stage(&|_| true));

    // Each program translated once, as it is: its jump tables, if any.
    let mut tables: BTreeMap<u64, Vec<JumpTable>> = BTreeMap::new();
    let mut failed: Vec<(u64, Stage)> = Vec::new();
    for (key, p) in &programs {
        let options = if p.stage == Stage::Compute { Options { workgroup: [32, 1, 1], ..Default::default() } } else { Options::default() };
        match shader::translate_with(&p.code, &options) {
            Ok(t) => {
                if let Some(t) = t.interface.jump_tables.filter(|t| !t.is_empty()) {
                    tables.insert(*key, t);
                }
            }
            Err(_) => failed.push((*key, p.stage)),
        }
    }
    let failed_keys: BTreeSet<u64> = failed.iter().map(|f| f.0).collect();
    println!("translated   {}: {}", programs.len() - failed.len(), by_stage(&|k| !failed_keys.contains(k)));
    println!("not          {}: {}", failed.len(), failed.iter().map(|(k, s)| format!("{k:016x} {s:?}")).collect::<Vec<_>>().join(", "));

    // Real play: every entry of the cache that decodes.
    let (stored, unread) = cache::entries(root);
    let builds: BTreeSet<&str> = stored.iter().map(|s| s.build.as_str()).collect();
    println!("cache        {} entries read from {} builds ({} unread: older builds' types, or damaged); this build is {}", stored.len(), builds.len(), unread, cache::BUILD);

    // The jump tables: play's targets against the blob's data at each rule's place.
    let mut truths: BTreeMap<u64, BTreeSet<BTreeSet<usize>>> = BTreeMap::new();
    for s in &stored {
        if let Some(t) = &s.options.jump_targets {
            truths.entry(s.key).or_default().insert(t.clone());
        }
    }
    let mut held = [0usize; RULES.len()];
    let (mut checked, mut unseen) = (0, 0);
    println!("jump tables  {} programs with tables: {}", tables.len(), by_stage(&|k| tables.contains_key(k)));
    for (key, t) in &tables {
        let p = &programs[key];
        let Some(sets) = truths.get(key) else {
            unseen += 1;
            continue;
        };
        checked += 1;
        let end = p.code.len();
        let data = end + 16 * sass::words(&p.section[end.min(p.section.len())..]).take_while(|w| w.opcode() == sass::NOP || w.0 == 0).count();
        let places = [(&p.section, data), (&p.section, end), (&p.section, 0), (&p.section, end.next_multiple_of(256)), (&p.control, 0)];
        let mut line = String::new();
        for (i, (bytes, at)) in places.into_iter().enumerate() {
            let found = sets.contains(&jump_targets(t, &table_bytes(t, bytes, at), end));
            held[i] += usize::from(found);
            line += &format!(" {}={}", RULES[i], if found { "yes" } else { "no" });
        }
        // Every place in the code section that gives one of play's sets, after the program's end.
        let anywhere: Vec<String> = (end..p.section.len()).step_by(4).filter(|&at| sets.contains(&jump_targets(t, &table_bytes(t, &p.section, at), end))).take(4).map(|at| format!("end+{:#x}", at - end)).collect();
        println!("  {key:016x} {:?}: {} tables, {} sets in play, data at end+{:#x};{line}; anywhere: {}", p.stage, t.len(), sets.len(), data - end, if anywhere.is_empty() { "none".to_string() } else { anywhere.join(" ") });
    }
    println!(
        "  rules, of {checked} programs play drew ({unseen} never drawn): {}",
        RULES.iter().zip(held).map(|(r, n)| format!("{r} {n}")).collect::<Vec<_>>().join(", ")
    );

    // Play's programs, by stage: in data.zip or not, and their variants.
    let mut variants: BTreeMap<u64, (Stage, HashSet<Options>)> = BTreeMap::new();
    let mut code: BTreeMap<u64, &[u8]> = BTreeMap::new();
    for s in &stored {
        variants.entry(s.key).or_insert_with(|| (s.stage, HashSet::new())).1.insert(s.options.clone());
        code.entry(s.key).or_insert(&s.code);
    }
    let mut stages: BTreeMap<Stage, (usize, usize, usize)> = BTreeMap::new();
    for (key, (stage, sets)) in &variants {
        let n = stages.entry(*stage).or_default();
        n.0 += 1;
        n.1 += usize::from(programs.contains_key(key));
        n.2 += sets.len();
    }
    println!("play         {} programs: {}", variants.len(), stages.iter().map(|(s, (n, inzip, sets))| format!("{s:?} {n} ({inzip} in data.zip, {sets} variants)")).collect::<Vec<_>>().join(", "));

    // Vertex programs: what their variants depend on.
    let (mut sets, mut wide, mut wide_keys, mut ints, mut int_keys) = (0, 0, 0, 0, 0);
    let mut clips: BTreeMap<u8, usize> = BTreeMap::new();
    let mut most = 0;
    for (key, (stage, options)) in &variants {
        if *stage != Stage::Vertex {
            continue;
        }
        let written: BTreeSet<u8> = Sph::new(Stage::Vertex, code[key]).map_or_else(BTreeSet::new, |s| s.outputs().iter().enumerate().filter(|(_, m)| **m != 0).map(|(v, _)| v as u8).collect());
        let (mut w, mut i) = (false, false);
        for o in options {
            sets += 1;
            let reads_unwritten = o.varyings.keys().any(|v| !written.contains(v));
            wide += usize::from(reads_unwritten);
            w |= reads_unwritten;
            ints += usize::from(!o.int_attributes.is_empty());
            i |= !o.int_attributes.is_empty();
            *clips.entry(o.clip_planes).or_default() += 1;
        }
        wide_keys += usize::from(w);
        int_keys += usize::from(i);
        let masks: BTreeSet<Vec<(u8, bool)>> = options.iter().map(|o| o.int_attributes.iter().map(|(l, s)| (*l, *s)).collect()).collect();
        most = most.max(masks.len());
    }
    println!("vertex       {sets} variants; their pixel program reads a varying they do not write: {wide} ({wide_keys} programs)");
    println!("             integer attributes: {ints} variants ({int_keys} programs, at most {most} masks for one); clip planes: {}", clips.iter().map(|(c, n)| format!("{c:#04x} {n}")).collect::<Vec<_>>().join(", "));

    // Compute programs: the block sizes, shared memory and image formats play used.
    for (key, (stage, options)) in variants.iter().filter(|(_, (s, _))| *s == Stage::Compute) {
        let seen: BTreeSet<String> = options.iter().map(|o| format!("{:?}/{}/{}/{}", o.workgroup, o.shared, o.images.len(), o.queried.len())).collect();
        println!("compute      {key:016x} {stage:?}: {} variants: {}", options.len(), seen.into_iter().collect::<Vec<_>>().join(" "));
    }

    // Play's programs that data.zip lacks, and where their code lies in the game's modules.
    let modules = exefs.map(super::load).transpose()?;
    let missing: Vec<(&u64, &(Stage, HashSet<Options>))> = variants.iter().filter(|(k, _)| !programs.contains_key(k)).collect();
    println!("not in data.zip: {}", missing.len());
    for (key, (stage, options)) in missing {
        let bytes = code[key];
        let sass = if *stage == Stage::Compute { bytes } else { bytes.get(sass::SPH_SIZE..).unwrap_or_default() };
        let needle = &sass[..sass.len().min(128)];
        let found: Vec<String> = modules
            .iter()
            .flat_map(|p| &p.modules)
            .flat_map(|m| memchr::memmem::find_iter(&m.module.image, needle).take(2).map(move |at| format!("{}+{at:#x}", m.name)))
            .collect();
        println!("  {key:016x} {stage:?}: {} bytes, {} variants; in the modules: {}", bytes.len(), options.len(), if modules.is_none() { "not looked for".to_string() } else if found.is_empty() { "no".to_string() } else { found.join(" ") });
    }
    Ok(true)
}

/// The programs of data.zip's shader blobs, by key: the first blob of each, and how many.
fn inventory(zip: &Path) -> super::Result<BTreeMap<u64, Program>> {
    let mut z = DataZip::open(zip)?;
    let entries = z.entries.clone();
    let mut programs: BTreeMap<u64, Program> = BTreeMap::new();
    for e in &entries {
        if e.size < 0x18 || z.head(e, 4)? != sass::BLOB_MAGIC.to_le_bytes() {
            continue;
        }
        let bytes = z.read(e)?;
        let Some(blob) = Blob::parse(&bytes) else { continue };
        let stage = blob.stage();
        let found = if stage == Stage::Compute { sass::compute_key(blob.code) } else { sass::program_key(blob.code) };
        let Some((key, len)) = found else { continue };
        programs.entry(key).and_modify(|p| p.blobs += 1).or_insert_with(|| Program { stage, code: blob.code[..len].to_vec(), section: blob.code.to_vec(), control: blob.control.to_vec(), blobs: 1 });
    }
    Ok(programs)
}

/// Each table's bytes when its bank starts at `at` in `data`, read as the renderer reads a
/// bound bank: its entries when known, else up to `TABLE_MAX` bytes, never past the data.
fn table_bytes(tables: &[JumpTable], data: &[u8], at: usize) -> Vec<Vec<u8>> {
    tables
        .iter()
        .map(|&(_, _, offset, entries)| {
            let start = (at + usize::from(offset)).min(data.len());
            let len = entries.map_or(TABLE_MAX, |n| 4 * n as usize).min(data.len() - start);
            data[start..start + len].to_vec()
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A table read where the blob holds it gives the targets; read a word further, others.
    #[test]
    fn tables_are_read_where_the_bank_starts() {
        // Two entries, absolute: instructions 3 and 5 (byte offsets 0x30 and 0x50).
        let mut data = vec![0u8; 8];
        data.extend(0x30u32.to_le_bytes());
        data.extend(0x50u32.to_le_bytes());
        let tables: Vec<JumpTable> = vec![(1, 0, 0, Some(2))];
        assert_eq!(jump_targets(&tables, &table_bytes(&tables, &data, 8), 0x100), BTreeSet::from([3, 5]));
        assert_ne!(jump_targets(&tables, &table_bytes(&tables, &data, 12), 0x100), BTreeSet::from([3, 5]));
        // Past the data: an empty table, no panic.
        assert!(jump_targets(&tables, &table_bytes(&tables, &data, 64), 0x100).is_empty());
    }
}
