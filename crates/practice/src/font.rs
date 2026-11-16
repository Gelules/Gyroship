//! The dump's fonts for Gyroship's in-game texts (D73, D74): practice.txt's `font` lines name a
//! data.zip entry per text region, or `latin` for the others; the OpenType file inside the
//! entry is cut out and read by ab_glyph, and a line of text becomes a tight coverage mask at
//! the size of the image it is drawn on.

use crate::texts::{self, REGIONS};
use ab_glyph::{Font as _, FontVec, GlyphId, PxScale, ScaleFont as _, point};
use std::collections::HashMap;

/// The text's size, in pixels per em, on a 1,080-pixel-high image.
pub const HUD_PX: f32 = 32.0;

/// The size for an image `height` pixels high: the text scales with the image.
pub fn px(height: u32) -> u16 {
    ((HUD_PX * height as f32 / 1080.0).round() as u16).max(1)
}

/// What a region's font must draw: its texts, a space and the digits.
pub fn needed(region: usize) -> String {
    texts::ALL.iter().map(|&t| texts::text(t, region)).chain([" 0123456789"]).collect()
}

/// The data.zip entry that practice.txt's `font <region tag | latin> <entry>` lines give a
/// region: its own line, else the `latin` one. Entry names are hex hashes, compared by value.
pub fn entry(practice: &str, region: usize) -> Result<u64, String> {
    let (mut own, mut latin) = (None, None);
    for (n, line) in practice.lines().enumerate() {
        let mut words = line.split_whitespace();
        if words.next() != Some("font") {
            continue;
        }
        let at = n + 1;
        let (Some(key), Some(name), None) = (words.next(), words.next(), words.next()) else { return Err(format!("line {at}: font takes a region and an entry")) };
        let id = u64::from_str_radix(name, 16).map_err(|_| format!("line {at}: {name} is not a data.zip entry name"))?;
        let slot = match key {
            "latin" => &mut latin,
            k if k == texts::tag(region) => &mut own,
            k if (0..REGIONS).any(|r| texts::tag(r) == k) => continue,
            k => return Err(format!("line {at}: {k} is neither latin nor a text region")),
        };
        if slot.replace(id).is_some() {
            return Err(format!("line {at}: a second font for {key}"));
        }
    }
    own.or(latin).ok_or_else(|| format!("no font line for {} or latin", texts::tag(region)))
}

/// The OpenType file inside a font entry of data.zip, and the source path its header names:
/// the first `OTTO` or 1.0 table directory with 4 to 64 printable tags and a cmap, cut at the
/// end of its last table (offsets count from the font's start), after a header naming a path
/// under `/fonts/`.
pub fn sfnt(d: &[u8]) -> Option<(String, &[u8])> {
    (0..d.len()).find_map(|p| {
        let magic = d.get(p..p + 4)?;
        if magic != b"OTTO" && magic != [0, 1, 0, 0] {
            return None;
        }
        let n = usize::from(u16::from_be_bytes(d.get(p + 4..p + 6)?.try_into().ok()?));
        let records = d.get(p + 12..p + 12 + 16 * n).filter(|_| (4..=64).contains(&n))?;
        let (mut end, mut cmap) = (0usize, false);
        for r in records.as_chunks::<16>().0 {
            if !r[..4].iter().all(|c| (b' '..=b'~').contains(c)) {
                return None;
            }
            cmap |= &r[..4] == b"cmap";
            let be = |i: usize| u32::from_be_bytes([r[i], r[i + 1], r[i + 2], r[i + 3]]) as usize;
            end = end.max(be(8).checked_add(be(12))?);
        }
        let font = d.get(p..p.checked_add(end)?).filter(|_| cmap)?;
        let head = String::from_utf8_lossy(&d[..p]);
        let at = head.find("/fonts/")?;
        let start = head[..at].rfind(['"', '\0']).map_or(0, |i| i + 1);
        let stop = head[at..].find(['"', '\0']).map_or(head.len(), |i| at + i);
        Some((head[start..stop].to_string(), font))
    })
}

/// One glyph at one size: its coverage, placed from the pen on the baseline (`top` is negative
/// above it), and its advance. Empty for a glyph without an outline (a space).
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Coverage {
    pub left: i32,
    pub top: i32,
    pub width: u32,
    pub height: u32,
    pub alpha: Vec<u8>,
    pub advance: f32,
}

/// A tight 8-bit coverage mask, rows top down, 0 clear and 255 full, with an empty last column
/// and row for the shadow drawn one pixel to the lower right.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Mask {
    pub width: u32,
    pub height: u32,
    pub bytes: Vec<u8>,
}

/// A region's font, its glyphs at the sizes drawn so far, and the path the dump names it by.
pub struct Face {
    font: FontVec,
    cache: HashMap<(GlyphId, u16), Coverage>,
    pub source: String,
}

impl Face {
    /// The font of a decoded data.zip font entry.
    pub fn load(entry: &[u8]) -> Result<Face, String> {
        let (source, font) = sfnt(entry).ok_or("no OpenType font under /fonts/ in the entry")?;
        let font = FontVec::try_from_vec(font.to_vec()).map_err(|e| format!("{source}: {e}"))?;
        Ok(Face { font, cache: HashMap::new(), source })
    }

    /// The characters of `text` the font cannot draw, each once: unmapped, or, spaces aside,
    /// without an outline (ttf-parser drops a CFF table it cannot read without a word).
    pub fn missing(&self, text: &str) -> Vec<char> {
        let mut out: Vec<char> = text
            .chars()
            .filter(|&c| {
                let id = self.font.glyph_id(c);
                id.0 == 0 || !c.is_whitespace() && self.font.outline(id).is_none()
            })
            .collect();
        out.sort_unstable();
        out.dedup();
        out
    }

    /// `text` at `px` pixels per em, one line per `\n`, each in a line box of the font's height
    /// (an empty line keeps its place). ab_glyph's scale is the height from descent to ascent,
    /// so a CJK font with a taller box would come out smaller at the same scale.
    pub fn mask(&mut self, text: &str, px: u16) -> Mask {
        let scale = PxScale::from(f32::from(px) * self.font.height_unscaled() / self.font.units_per_em().unwrap_or(1000.0));
        // ponytail: the cache keeps every size drawn (a window dragged through many sizes); clear it if that ever shows.
        for c in text.chars().filter(|&c| c != '\n') {
            let id = self.font.glyph_id(c);
            self.cache.entry((id, px)).or_insert_with(|| coverage(&self.font, id, scale));
        }
        let scaled = self.font.as_scaled(scale);
        let (ascent, descent) = (scaled.ascent(), scaled.descent());
        let lines: Vec<Mask> = text.split('\n').map(|line| lay_out(line, ascent.ceil() as i32, (ascent - descent).ceil() as u32, |c| &self.cache[&(self.font.glyph_id(c), px)])).collect();
        stack(&lines)
    }
}

/// Lines laid out by [`lay_out`], stacked top down: the widest sets the width.
pub fn stack(lines: &[Mask]) -> Mask {
    let width = lines.iter().map(|m| m.width).max().unwrap_or(1);
    let height = lines.iter().map(|m| m.height).sum::<u32>().max(1);
    let mut bytes = vec![0u8; (width * height) as usize];
    let mut top = 0;
    for m in lines {
        for (y, row) in m.bytes.chunks(m.width as usize).enumerate() {
            let at = (top + y) * width as usize;
            bytes[at..at + row.len()].copy_from_slice(row);
        }
        top += m.height as usize;
    }
    Mask { width, height, bytes }
}

/// One glyph's coverage at `scale`, drawn at the pen (0, 0).
fn coverage(font: &FontVec, id: GlyphId, scale: PxScale) -> Coverage {
    let advance = font.as_scaled(scale).h_advance(id);
    let Some(g) = font.outline_glyph(id.with_scale_and_position(scale, point(0.0, 0.0))) else { return Coverage { advance, ..Coverage::default() } };
    let b = g.px_bounds();
    let (width, height) = (b.width() as u32, b.height() as u32);
    let mut alpha = vec![0u8; (width * height) as usize];
    g.draw(|x, y, c| {
        if let Some(a) = alpha.get_mut((y * width + x) as usize) {
            // The coverage can pass 1.0 where contours overlap.
            *a = (c.min(1.0) * 255.0).round() as u8;
        }
    });
    Coverage { left: b.min.x as i32, top: b.min.y as i32, width, height, alpha, advance }
}

/// `text` laid out on one line: the pen starts at 0 on a baseline `ascent` pixels under the
/// top, advances in f32 and is rounded per glyph; the line is `height` pixels high, plus the
/// shadow's row and column. Ink outside the line is clipped; where glyphs overlap, the larger
/// coverage wins. `glyph` gives each character's coverage, so the tests need no font.
pub fn lay_out<'a>(text: &str, ascent: i32, height: u32, glyph: impl Fn(char) -> &'a Coverage) -> Mask {
    let (mut pen, mut right, mut placed) = (0f32, 0i32, Vec::new());
    for c in text.chars() {
        let g = glyph(c);
        let x = pen.round() as i32 + g.left;
        placed.push((x, ascent + g.top, g));
        right = right.max(x + g.width as i32);
        pen += g.advance;
    }
    let (width, height) = (right.max(pen.round() as i32).max(0) as u32 + 1, height + 1);
    let mut bytes = vec![0u8; (width * height) as usize];
    for (x0, y0, g) in placed {
        for (i, &a) in g.alpha.iter().enumerate() {
            let (x, y) = (x0 + (i as u32 % g.width) as i32, y0 + (i as u32 / g.width) as i32);
            if (0..width as i32 - 1).contains(&x) && (0..height as i32 - 1).contains(&y) {
                let at = (y as u32 * width + x as u32) as usize;
                bytes[at] = bytes[at].max(a);
            }
        }
    }
    Mask { width, height, bytes }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A font entry: a header, then an `OTTO` directory of four 8-byte tables, then junk.
    fn font_entry(header: &str, tags: [&[u8; 4]; 4]) -> (Vec<u8>, usize) {
        let mut d = header.as_bytes().to_vec();
        let p = d.len();
        d.extend_from_slice(b"OTTO\0\x04\0\0\0\0\0\0");
        for (i, tag) in (0u32..).zip(tags) {
            d.extend_from_slice(tag);
            d.extend_from_slice(&[0; 4]);
            d.extend_from_slice(&(76 + 8 * i).to_be_bytes());
            d.extend_from_slice(&8u32.to_be_bytes());
        }
        d.extend_from_slice(&[7; 32]);
        d.extend_from_slice(b"junk after the font");
        (d, p)
    }

    const TABLES: [&[u8; 4]; 4] = [b"cmap", b"head", b"hhea", b"maxp"];

    #[test]
    fn the_font_is_cut_out_of_its_entry_after_a_fonts_path() {
        let (d, p) = font_entry("{\"path\": \"/data/eclipse/fonts/x.otf\", \"sizes\": [12]}\0", TABLES);
        assert_eq!(sfnt(&d), Some(("/data/eclipse/fonts/x.otf".to_string(), &d[p..p + 108])));
        assert_eq!(sfnt(&font_entry("{\"path\": \"/data/x.otf\"}", TABLES).0), None, "no /fonts/ path");
        assert_eq!(sfnt(&font_entry("/fonts/", [b"cmaq", b"head", b"hhea", b"maxp"]).0), None, "no cmap");
        assert_eq!(sfnt(&d[..p + 12 + 16 * 3]), None, "a directory cut short");
    }

    #[test]
    fn a_region_takes_its_own_font_line_else_the_latin_one() {
        let file = "boost 1 x\nfont latin 51f6\nfont ja-JP 0e2b1\n# font ko-KR 1\nfont zh-CN ff09\n";
        assert_eq!(entry(file, 10), Ok(0xe2b1));
        assert_eq!(entry(file, 13), Ok(0xff09));
        assert_eq!(entry(file, 8), Ok(0x51f6));
        assert_eq!(entry("font ja-JP 1", 0), Err("no font line for en-US or latin".into()));
        assert!(entry("font xx-YY 1", 0).unwrap_err().contains("neither latin nor a text region"));
        assert!(entry("font latin", 0).unwrap_err().contains("takes a region and an entry"));
        assert!(entry("font latin 1 2", 0).unwrap_err().contains("takes a region and an entry"));
        assert!(entry("font latin zz", 0).unwrap_err().contains("not a data.zip entry name"));
        assert!(entry("font latin 1\nfont latin 2", 0).unwrap_err().contains("a second font for latin"));
    }

    #[test]
    fn glyphs_are_laid_out_from_the_baseline_with_room_for_the_shadow() {
        let a = Coverage { left: 0, top: -3, width: 2, height: 3, alpha: vec![255; 6], advance: 2.5 };
        let space = Coverage { advance: 1.5, ..Coverage::default() };
        let m = lay_out("A A", 4, 5, |c| if c == 'A' { &a } else { &space });
        let mut want = vec![0u8; 8 * 6];
        for y in 1..4 {
            for x in [0, 1, 4, 5] {
                want[y * 8 + x] = 255;
            }
        }
        assert_eq!(m, Mask { width: 8, height: 6, bytes: want });
        // Overlapping glyphs: the larger coverage wins.
        let (half, full) = (Coverage { width: 2, height: 1, alpha: vec![128, 128], advance: 1.0, ..a.clone() }, Coverage { width: 1, height: 1, alpha: vec![255], advance: 1.0, ..a.clone() });
        let m = lay_out("hf", 3, 1, |c| if c == 'h' { &half } else { &full });
        assert_eq!(&m.bytes[..3], &[128, 255, 0]);
    }

    #[test]
    fn lines_stack_in_their_boxes_and_an_empty_one_keeps_its_place() {
        // A line as lay_out makes it: ink in its first row, then the shadow's column and row.
        let line = |width: usize, ink: u8| {
            let mut bytes = vec![0u8; width * 2];
            bytes[..width - 1].fill(ink);
            Mask { width: width as u32, height: 2, bytes }
        };
        let m = stack(&[line(3, 9), line(2, 7)]);
        assert_eq!(m, Mask { width: 3, height: 4, bytes: vec![9, 9, 0, 0, 0, 0, 7, 0, 0, 0, 0, 0] });
        // An empty first line (its own box, one column wide) keeps the second one down.
        let m = stack(&[line(1, 0), line(2, 7)]);
        assert_eq!(m, Mask { width: 2, height: 4, bytes: vec![0, 0, 0, 0, 7, 0, 0, 0] });
    }

    #[test]
    fn the_size_scales_with_the_image() {
        assert_eq!([1080, 720, 2160, 0].map(px), [32, 21, 64, 1]);
    }
}
