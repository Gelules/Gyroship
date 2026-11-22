//! `gyroship-tools mods` (D113): the mods' layers, built on the player's PC from their own dump,
//! so that no file of the game is ever distributed. A layer is a folder the game reads over its
//! romfs (`--romfs-layer`): `about.txt` (the launcher's label for it), `veclipse_build.cvar`, and
//! `data.zip.entries/<entry name>`, each replacing the archive's entry of that very name.
//!
//! The entries edited are the engine's JSON text, one key per line, indented with tabs. Each edit,
//! as the Python scripts it was ported from, inserts or replaces a few lines and leaves every other
//! byte as it is; it checks what it relies on (each anchor found exactly once, the imports' order)
//! and that the result still reads as JSON. The values read in the game that the edits need (entry
//! names, GUIDs, the entities' names) are in templates/mods.txt (D106), none in this code.

mod demo;
mod json;
mod mouse;
mod quit;

use gyroship_gpu::datazip::DataZip;
use std::collections::{BTreeMap, HashMap};
use std::fs;
use std::path::Path;

/// Builds every layer into `out`, from the dump whose exefs folder is `exefs`.
pub fn run(exefs: &Path, out: &Path) -> super::Result<bool> {
    let repo = crate::experiment::repository()?;
    let recipe = Recipe::read(&repo.join("templates").join("mods.txt"))?;
    let romfs = exefs.parent().map(|p| p.join("romfs")).ok_or_else(|| format!("{}: no folder above it", exefs.display()))?;
    let mut zip = Archive::open(&romfs.join("data.zip"))?;
    let cvar_file = romfs.join("veclipse_build.cvar");
    let cvar = fs::read_to_string(&cvar_file).map_err(|e| format!("{}: {e}", cvar_file.display()))?;
    fs::create_dir_all(out).map_err(|e| format!("{}: {e}", out.display()))?;
    // The practice layer first: every other mod is made on it, and adds QUIT GAME and the timer too.
    let base = quit::layer(&recipe, &mut zip, &cvar)?;
    let demo = demo::layer(&recipe, &base)?;
    let mouse = mouse::layer(&recipe, &mut zip, &base)?;
    for (folder, layer) in [("layer", &base), ("layer-demo", &demo), ("layer-mouse", &mouse)] {
        layer.write(&out.join(folder))?;
        println!("mods: {folder}, {} entries: {}", layer.entries.len(), layer.about);
    }
    Ok(true)
}

/// The values read in the game that the mods need (templates/mods.txt): one `key value...` per
/// line; a comment is a whole line that starts with #.
#[derive(Debug)]
pub struct Recipe(HashMap<String, Vec<String>>);

impl Recipe {
    pub fn read(file: &Path) -> Result<Recipe, String> {
        Recipe::parse(&fs::read_to_string(file).map_err(|e| format!("{}: {e}", file.display()))?)
    }

    pub fn parse(text: &str) -> Result<Recipe, String> {
        let mut map = HashMap::new();
        for (i, line) in text.lines().enumerate() {
            let mut words = line.split_whitespace();
            let Some(key) = words.next().filter(|k| !k.starts_with('#')) else { continue };
            let values: Vec<String> = words.map(String::from).collect();
            if values.is_empty() {
                return Err(format!("mods.txt, line {}: {key} has no value", i + 1));
            }
            if map.insert(key.to_string(), values).is_some() {
                return Err(format!("mods.txt, line {}: {key} a second time", i + 1));
            }
        }
        Ok(Recipe(map))
    }

    /// The one value of `key`.
    pub fn one(&self, key: &str) -> Result<&str, String> {
        match self.all(key)? {
            [value] => Ok(value),
            values => Err(format!("mods.txt: {key} has {} values, 1 expected", values.len())),
        }
    }

    /// The values of `key`.
    pub fn all(&self, key: &str) -> Result<&[String], String> {
        self.0.get(key).map(Vec::as_slice).ok_or_else(|| format!("mods.txt has no {key}"))
    }
}

/// The dump's data.zip, its entries found by name or by path.
pub struct Archive {
    zip: DataZip,
    by_value: HashMap<u64, usize>,
}

impl Archive {
    pub fn open(path: &Path) -> Result<Archive, String> {
        let zip = DataZip::open(path).map_err(|e| format!("{}: {e}", path.display()))?;
        let by_value = zip.entries.iter().enumerate().filter_map(|(i, e)| Some((u64::from_str_radix(&e.name, 16).ok()?, i))).collect();
        Ok(Archive { zip, by_value })
    }

    /// The name and the decoded bytes of the entry named `name` (hex digits, as the archive or a
    /// layer writes them).
    pub fn named(&mut self, name: &str) -> Result<(String, Vec<u8>), String> {
        let value = u64::from_str_radix(name, 16).map_err(|e| format!("{name}: no entry name ({e})"))?;
        self.read(value, name)
    }

    /// The name and the decoded bytes of the entry of `path`, the FNV-1a hash of the path without
    /// its first slash.
    #[allow(dead_code)]
    pub fn at(&mut self, path: &str) -> Result<(String, Vec<u8>), String> {
        self.read(fnv1a(path.trim_start_matches('/')), path)
    }

    fn read(&mut self, value: u64, what: &str) -> Result<(String, Vec<u8>), String> {
        let e = self.by_value.get(&value).map(|&i| self.zip.entries[i].clone()).ok_or_else(|| format!("data.zip has no entry {what}"))?;
        let bytes = self.zip.read(&e).map_err(|err| format!("data.zip entry {}: {err}", e.name))?;
        Ok((e.name, bytes))
    }
}

/// The archive's hash of a path: FNV-1a, 64 bits.
pub fn fnv1a(s: &str) -> u64 {
    s.bytes().fold(0xcbf2_9ce4_8422_2325, |h, b| (h ^ u64::from(b)).wrapping_mul(0x0100_0000_01b3))
}

/// A layer as the game reads it: its label, its settings file, its entries by name.
#[derive(Clone, Debug, PartialEq)]
pub struct Layer {
    pub about: String,
    pub cvar: String,
    pub entries: BTreeMap<String, Vec<u8>>,
}

impl Layer {
    /// Writes the layer as `folder`: whole in `<folder>.part` first, then in place of any earlier
    /// build of it, so that a stopped build never leaves half a layer to play.
    pub fn write(&self, folder: &Path) -> Result<(), String> {
        let fail = |p: &Path, e: std::io::Error| format!("{}: {e}", p.display());
        let part = folder.with_file_name(format!("{}.part", folder.file_name().and_then(|n| n.to_str()).ok_or("a layer's folder has no name")?));
        if part.exists() {
            fs::remove_dir_all(&part).map_err(|e| fail(&part, e))?;
        }
        let entries = part.join("data.zip.entries");
        fs::create_dir_all(&entries).map_err(|e| fail(&entries, e))?;
        fs::write(part.join("about.txt"), format!("{}\n", self.about)).map_err(|e| fail(&part, e))?;
        fs::write(part.join("veclipse_build.cvar"), &self.cvar).map_err(|e| fail(&part, e))?;
        for (name, bytes) in &self.entries {
            fs::write(entries.join(name), bytes).map_err(|e| fail(&entries.join(name), e))?;
        }
        if folder.exists() {
            fs::remove_dir_all(folder).map_err(|e| fail(folder, e))?;
        }
        fs::rename(&part, folder).map_err(|e| fail(folder, e))
    }
}

/// An entry's text, as the edits want it: ASCII, with LF line ends only.
fn ascii(bytes: Vec<u8>, what: &str) -> Result<String, String> {
    let text = String::from_utf8(bytes).ok().filter(|t| t.is_ascii()).ok_or_else(|| format!("{what} is not ASCII: not the entry this tool knows"))?;
    if text.contains('\r') {
        return Err(format!("{what} has CR line ends; the dump's entry has LF only"));
    }
    Ok(text)
}

/// Where `anchor` is in `text`, when it is there exactly once.
fn once(text: &str, anchor: &str, what: &str) -> Result<usize, String> {
    match text.matches(anchor).count() {
        1 => Ok(text.find(anchor).expect("found once")),
        n => Err(format!("the anchor for {what} found {n} times, expected exactly once")),
    }
}

/// The one line of `lines[span]` that `pred` takes.
fn one(lines: &[String], span: impl IntoIterator<Item = usize>, pred: impl Fn(&str) -> bool, what: &str) -> Result<usize, String> {
    let hits: Vec<usize> = span.into_iter().filter(|&i| pred(&lines[i])).collect();
    match hits[..] {
        [i] => Ok(i),
        _ => Err(format!("{what}: {} lines, expected exactly one", hits.len())),
    }
}

/// A GUID, lower case: 8-4-4-4-12 hex digits.
fn is_guid(s: &str) -> bool {
    let parts: Vec<&str> = s.split('-').collect();
    parts.iter().map(|p| p.len()).eq([8, 4, 4, 4, 12]) && parts.iter().all(|p| p.bytes().all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b)))
}

/// An engine type: listed first in a level's imports, in an order of their own.
fn is_native(g: &str) -> bool {
    g.split('-').skip(1).take(2).eq(["0000", "0000"])
}

/// How a level sorts its other imports: the time-based GUID's fields, most significant first.
fn import_key(g: &str) -> [&str; 5] {
    let p: Vec<&str> = g.split('-').chain(std::iter::repeat("")).take(5).collect();
    [p[2], p[1], p[0], p[3], p[4]]
}

/// The engine's 32-bit hash of a string (a localized string's version_hash): CRC-32C, reflected,
/// initial value 0, no final xor.
fn crc32c(data: &[u8]) -> u32 {
    data.iter().fold(0u32, |mut c, &b| {
        c ^= u32::from(b);
        for _ in 0..8 {
            c = if c & 1 != 0 { (c >> 1) ^ 0x82f6_3b78 } else { c >> 1 };
        }
        c
    })
}

/// A name-based GUID (UUID v5, RFC 4122) as Python's uuid.uuid5 makes it: SHA-1 of the
/// namespace's 16 bytes and the name.
fn uuid5(namespace: &str, name: &str) -> String {
    let hex: String = namespace.chars().filter(|c| *c != '-').collect();
    let mut data: Vec<u8> = (0..16).map(|i| u8::from_str_radix(&hex[2 * i..2 * i + 2], 16).expect("a GUID")).collect();
    data.extend_from_slice(name.as_bytes());
    let mut d = sha1(&data);
    d[6] = (d[6] & 0x0f) | 0x50;
    d[8] = (d[8] & 0x3f) | 0x80;
    let h: String = d[..16].iter().map(|b| format!("{b:02x}")).collect();
    format!("{}-{}-{}-{}-{}", &h[..8], &h[8..12], &h[12..16], &h[16..20], &h[20..])
}

/// SHA-1 (FIPS 180-4), for `uuid5` only.
fn sha1(data: &[u8]) -> [u8; 20] {
    let mut h: [u32; 5] = [0x6745_2301, 0xefcd_ab89, 0x98ba_dcfe, 0x1032_5476, 0xc3d2_e1f0];
    let mut msg = data.to_vec();
    msg.push(0x80);
    while msg.len() % 64 != 56 {
        msg.push(0);
    }
    msg.extend_from_slice(&(data.len() as u64 * 8).to_be_bytes());
    for chunk in msg.chunks(64) {
        let mut w = [0u32; 80];
        for i in 0..80 {
            w[i] = if i < 16 { u32::from_be_bytes(chunk[4 * i..4 * i + 4].try_into().expect("4 bytes")) } else { (w[i - 3] ^ w[i - 8] ^ w[i - 14] ^ w[i - 16]).rotate_left(1) };
        }
        let [mut a, mut b, mut c, mut d, mut e] = h;
        for (i, &wi) in w.iter().enumerate() {
            let (f, k) = match i {
                0..20 => ((b & c) | (!b & d), 0x5a82_7999),
                20..40 => (b ^ c ^ d, 0x6ed9_eba1),
                40..60 => ((b & c) | (b & d) | (c & d), 0x8f1b_bcdc),
                _ => (b ^ c ^ d, 0xca62_c1d6),
            };
            let t = a.rotate_left(5).wrapping_add(f).wrapping_add(e).wrapping_add(k).wrapping_add(wi);
            (e, d, c, b, a) = (d, c, b.rotate_left(30), a, t);
        }
        for (x, y) in h.iter_mut().zip([a, b, c, d, e]) {
            *x = x.wrapping_add(y);
        }
    }
    let mut out = [0u8; 20];
    for (i, x) in h.iter().enumerate() {
        out[4 * i..4 * i + 4].copy_from_slice(&x.to_be_bytes());
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The hashes the archive and the engine use, on known values: FNV-1a's empty string, the
    /// CRC-32C of PAUSE_MENU_QUIT's en-US text (its version_hash), Python's own uuid5 example.
    #[test]
    fn hashes_match_their_references() {
        assert_eq!(fnv1a(""), 0xcbf2_9ce4_8422_2325);
        assert_eq!(fnv1a("a"), 0xaf63_dc4c_8601_ec8c);
        assert_eq!(crc32c(b"QUIT"), 4_106_313_554);
        assert_eq!(crc32c(b"vtable"), 2_930_123_390);
        assert_eq!(sha1(b"abc").iter().map(|b| format!("{b:02x}")).collect::<String>(), "a9993e364706816aba3e25717850c26c9cd0d89d");
        assert_eq!(uuid5("6ba7b810-9dad-11d1-80b4-00c04fd430c8", "python.org"), "886313e1-3b8a-5372-9b90-0c9aee199e5d");
    }

    /// Engine types first, then the others by their time fields.
    #[test]
    fn imports_sort_by_their_time_fields() {
        assert!(is_native("00000001-0000-0000-1c9d-d96cfbdace5f") && !is_native("2e28028d-50a7-11ee-9484-5ed8aab988d3"));
        assert!(import_key("0981fca4-2198-11ee-98de-d800c63bd261") < import_key("2e28028d-50a7-11ee-9484-5ed8aab988d3"));
        assert!(is_guid("83fbc756-fe56-4f74-87ae-29f6bec0e440") && !is_guid("83FBC756-fe56-4f74-87ae-29f6bec0e440") && !is_guid("83fbc756fe564f7487ae29f6bec0e440"));
    }

    #[test]
    fn a_recipe_reads_its_keys_once() {
        let r = Recipe::parse("# a comment\nquit.strings 5139d98edd181587\nquit.fields a b\n\n").unwrap();
        assert_eq!(r.one("quit.strings").unwrap(), "5139d98edd181587");
        assert_eq!(r.all("quit.fields").unwrap(), ["a", "b"]);
        assert!(r.one("quit.fields").unwrap_err().contains("2 values"));
        assert!(r.one("missing").unwrap_err().contains("no missing"));
        assert!(Recipe::parse("a 1\na 2").unwrap_err().contains("second time"));
        assert!(Recipe::parse("a").unwrap_err().contains("no value"));
    }
}
