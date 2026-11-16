//! `text-sheet` (M6 step 3b, P0): every text Gyroship draws, in the 14 regions, with the font
//! each region's `font` line names and the speed line's own raster, white over dark grey, one
//! row per region. The fonts come from the dump's data.zip: the sheet is game output, to keep
//! out of the repository.

use gyroship_gpu::datazip::DataZip;
use gyroship_practice::font::{self, Face};
use gyroship_practice::texts::{self, REGIONS};
use std::collections::HashMap;
use std::collections::hash_map::Entry;
use std::path::Path;

/// Draws the sheet into `out` and prints each region's line, its font and the characters the
/// font cannot draw. Ok(false) when a region misses one.
pub fn run(zip: &Path, practice: &Path, out: &Path) -> Result<bool, Box<dyn std::error::Error>> {
    let text = std::fs::read_to_string(practice).map_err(|e| format!("{}: {e}", practice.display()))?;
    let mut z = DataZip::open(zip).map_err(|e| format!("{}: {e}", zip.display()))?;
    let mut faces: HashMap<u64, Face> = HashMap::new();
    let (mut rows, mut complete) = (Vec::new(), true);
    for r in 0..REGIONS {
        let id = font::entry(&text, r)?;
        let face = match faces.entry(id) {
            Entry::Occupied(o) => o.into_mut(),
            Entry::Vacant(v) => {
                let e = z.entries.iter().find(|e| u64::from_str_radix(&e.name, 16) == Ok(id)).cloned().ok_or_else(|| format!("{}: no entry {id:x}", zip.display()))?;
                v.insert(Face::load(&z.read(&e)?)?)
            }
        };
        let line = texts::ALL.iter().map(|&t| texts::text(t, r)).collect::<Vec<_>>().join("  ") + "  1230";
        let missing: Vec<String> = face.missing(&font::needed(r)).iter().map(|c| format!("U+{:04X}", u32::from(*c))).collect();
        println!("{:7} {}: {line}{}", texts::tag(r), face.source, if missing.is_empty() { String::new() } else { format!("  (cannot draw {})", missing.join(" ")) });
        complete &= missing.is_empty();
        rows.push(face.mask(&line, font::px(1080)));
    }
    let (margin, gap) = (16u32, 8u32);
    let width = rows.iter().map(|m| m.width).max().unwrap_or(1) + 2 * margin;
    let height = rows.iter().map(|m| m.height + gap).sum::<u32>() + 2 * margin;
    let mut rgba = [40u8, 40, 40, 255].repeat((width * height) as usize);
    let mut top = margin;
    for m in &rows {
        for (i, &a) in m.bytes.iter().enumerate() {
            let (x, y) = (margin + i as u32 % m.width, top + i as u32 / m.width);
            let at = (y * width + x) as usize * 4;
            for c in &mut rgba[at..at + 3] {
                *c = (f32::from(*c) + (255.0 - f32::from(*c)) * f32::from(a) / 255.0).round() as u8;
            }
        }
        top += m.height + gap;
    }
    if let Some(dir) = out.parent() {
        std::fs::create_dir_all(dir)?;
    }
    gyroship_gpu::write_png(out, width, height, &rgba)?;
    println!("{}: {width}x{height}", out.display());
    Ok(complete)
}
