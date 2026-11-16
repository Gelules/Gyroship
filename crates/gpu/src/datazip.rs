//! The game engine's asset archive (romfs `data.zip`, about 15 GiB): a zip
//! (zip64 sizes and offsets) of entries named by a 64-bit hash. An entry is
//! either stored raw (its compressed size equals its size) or a chain of LZ4
//! blocks, each a u32 compressed length then the block, decoding to 64 KiB
//! but for the last; a block may refer back into the previous one.
//!
//! Before its first entry the archive holds an index of the engine's own, the
//! only part the engine reads to find an entry: "VZIP", a record count, the
//! size of the names; then 40-byte records (the offset of a name, at +8 the
//! offset of the entry's bytes, their stored and decoded sizes, 1 for LZ4
//! chunks or 0 for raw bytes, a hash of the name); then the names, each ended
//! by a zero. An entry has a record per name it answers to.

use std::fs::File;
use std::io::{self, Read, Seek, SeekFrom};
use std::path::Path;

/// One entry of the central directory.
#[derive(Clone, Debug)]
pub struct Entry {
    pub name: String,
    /// Offset of the local header.
    pub header: u64,
    pub compressed: u64,
    pub size: u64,
}

pub struct DataZip {
    file: File,
    pub entries: Vec<Entry>,
}

const CHUNK: usize = 0x10000;
/// The index block: its header's size, and a record's.
const INDEX: usize = 12;
const RECORD: usize = 40;

fn invalid(what: impl Into<String>) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidData, what.into())
}

fn u16_at(b: &[u8], o: usize) -> u16 {
    u16::from_le_bytes([b[o], b[o + 1]])
}

fn u32_at(b: &[u8], o: usize) -> u32 {
    u32::from_le_bytes(b[o..o + 4].try_into().expect("4 bytes"))
}

fn u64_at(b: &[u8], o: usize) -> u64 {
    u64::from_le_bytes(b[o..o + 8].try_into().expect("8 bytes"))
}

impl DataZip {
    /// Reads the central directory (through the zip64 end records when present).
    pub fn open(path: &Path) -> io::Result<DataZip> {
        let mut file = File::open(path)?;
        let len = file.seek(SeekFrom::End(0))?;
        let tail_len = len.min(0x1_0000 + 22);
        let mut tail = vec![0; tail_len as usize];
        file.seek(SeekFrom::Start(len - tail_len))?;
        file.read_exact(&mut tail)?;
        let eocd = (0..=tail.len() - 22).rev().find(|&i| u32_at(&tail, i) == 0x0605_4b50).ok_or_else(|| invalid("no end of central directory"))?;
        let (mut count, mut size, mut start) = (u64::from(u16_at(&tail, eocd + 10)), u64::from(u32_at(&tail, eocd + 12)), u64::from(u32_at(&tail, eocd + 16)));
        // zip64: the locator just before the end record points to the zip64 end record.
        if eocd >= 20 && u32_at(&tail, eocd - 20) == 0x0706_4b50 {
            let at = u64_at(&tail, eocd - 20 + 8);
            let mut rec = [0u8; 56];
            file.seek(SeekFrom::Start(at))?;
            file.read_exact(&mut rec)?;
            if u32_at(&rec, 0) != 0x0606_4b50 {
                return Err(invalid("bad zip64 end of central directory"));
            }
            (count, size, start) = (u64_at(&rec, 32), u64_at(&rec, 40), u64_at(&rec, 48));
        }
        let mut dir = vec![0; size as usize];
        file.seek(SeekFrom::Start(start))?;
        file.read_exact(&mut dir)?;
        let mut entries = Vec::with_capacity(count as usize);
        let mut i = 0;
        while entries.len() < count as usize {
            if i + 46 > dir.len() || u32_at(&dir, i) != 0x0201_4b50 {
                return Err(invalid(format!("bad central directory entry at {i:#x}")));
            }
            let (name_len, extra_len, comment_len) = (usize::from(u16_at(&dir, i + 28)), usize::from(u16_at(&dir, i + 30)), usize::from(u16_at(&dir, i + 32)));
            let mut compressed = u64::from(u32_at(&dir, i + 20));
            let mut size = u64::from(u32_at(&dir, i + 24));
            let mut header = u64::from(u32_at(&dir, i + 42));
            let name = String::from_utf8_lossy(&dir[i + 46..i + 46 + name_len]).into_owned();
            // zip64 extra field: the 64-bit values of the fields that read 0xffffffff, in order.
            let mut x = i + 46 + name_len;
            let extra_end = x + extra_len;
            while x + 4 <= extra_end {
                let (id, n) = (u16_at(&dir, x), usize::from(u16_at(&dir, x + 2)));
                if id == 1 {
                    let mut v = x + 4;
                    for field in [&mut size, &mut compressed, &mut header] {
                        if *field == 0xffff_ffff && v + 8 <= x + 4 + n {
                            *field = u64_at(&dir, v);
                            v += 8;
                        }
                    }
                }
                x += 4 + n;
            }
            entries.push(Entry { name, header, compressed, size });
            i = extra_end + comment_len;
        }
        Ok(DataZip { file, entries })
    }

    /// The stored bytes of an entry (`limit` bytes at most).
    fn stored(&mut self, e: &Entry, limit: u64) -> io::Result<Vec<u8>> {
        let mut h = [0u8; 30];
        self.file.seek(SeekFrom::Start(e.header))?;
        self.file.read_exact(&mut h)?;
        if u32_at(&h, 0) != 0x0403_4b50 {
            return Err(invalid(format!("{}: bad local header", e.name)));
        }
        let skip = u64::from(u16_at(&h, 26)) + u64::from(u16_at(&h, 28));
        self.file.seek(SeekFrom::Start(e.header + 30 + skip))?;
        let mut raw = vec![0; e.compressed.min(limit) as usize];
        self.file.read_exact(&mut raw)?;
        Ok(raw)
    }

    /// The content of an entry, decoded.
    pub fn read(&mut self, e: &Entry) -> io::Result<Vec<u8>> {
        let raw = self.stored(e, u64::MAX)?;
        if e.compressed == e.size {
            return Ok(raw);
        }
        decode_chunks(&raw, e.size as usize).map_err(|m| invalid(format!("{}: {m}", e.name)))
    }

    /// The first bytes of an entry, without decoding all of it: the raw bytes,
    /// or the first decoded chunk.
    pub fn head(&mut self, e: &Entry, n: usize) -> io::Result<Vec<u8>> {
        if e.compressed == e.size {
            return self.stored(e, n as u64);
        }
        let raw = self.stored(e, 4 + CHUNK as u64 + CHUNK as u64 / 255 + 16)?;
        let first = u32_at(&raw, 0) as usize;
        let block = raw.get(4..4 + first).ok_or_else(|| invalid(format!("{}: first chunk past the entry", e.name)))?;
        let mut out = vec![0; CHUNK.min(e.size as usize)];
        let got = lz4_flex::block::decompress_into(block, &mut out).map_err(|m| invalid(format!("{}: {m}", e.name)))?;
        out.truncate(got.min(n));
        Ok(out)
    }
}

/// Decodes a chain of (u32 length, LZ4 block) chunks into `size` bytes; each
/// block may refer back into the 64 KiB before it.
pub fn decode_chunks(raw: &[u8], size: usize) -> Result<Vec<u8>, String> {
    let mut out = Vec::with_capacity(size);
    let mut chunk = vec![0u8; CHUNK];
    let mut i = 0;
    while out.len() < size {
        let len = raw.get(i..i + 4).map(|b| u32::from_le_bytes(b.try_into().expect("4 bytes")) as usize).ok_or("chunk length past the entry")?;
        let block = raw.get(i + 4..i + 4 + len).ok_or_else(|| format!("chunk of {len:#x} bytes past the entry"))?;
        let want = CHUNK.min(size - out.len());
        let dict = &out[out.len().saturating_sub(CHUNK)..];
        let n = lz4_flex::block::decompress_into_with_dict(block, &mut chunk[..want], dict).map_err(|e| format!("LZ4 at {i:#x}: {e}"))?;
        if n == 0 {
            return Err(format!("empty chunk at {i:#x}"));
        }
        out.extend_from_slice(&chunk[..n]);
        i += 4 + len;
    }
    Ok(out)
}

/// An archive in which some entries took new contents, as bytes over the file's own.
#[derive(Debug, Default)]
pub struct Overlay {
    /// What follows the file's bytes: the new contents, as the archive stores them.
    pub end: Vec<u8>,
    /// Bytes that take the file's place, by offset: in the index records of those
    /// entries, where their bytes are and their sizes.
    pub patches: Vec<(u64, Vec<u8>)>,
}

/// The archive in which the entries `replace` names take new contents, as the engine
/// reads it: through its index block. Each content follows the file's bytes, on a
/// multiple of 16 as the archive's own are, in LZ4 chunks or raw as the entry was;
/// every index record of the entry (one per name it has) points at it. The old
/// contents stay in the file. Refuses a name the index lacks.
pub fn overlay(path: &Path, replace: &[(String, Vec<u8>)]) -> io::Result<Overlay> {
    let mut file = File::open(path)?;
    let len = file.seek(SeekFrom::End(0))?;
    let mut read = |at: usize, n: usize| -> io::Result<Vec<u8>> {
        let mut b = vec![0; n];
        file.seek(SeekFrom::Start(at as u64))?;
        file.read_exact(&mut b)?;
        Ok(b)
    };
    let head = read(0, INDEX)?;
    if !head.starts_with(b"VZIP") {
        return Err(invalid("no index block"));
    }
    let records = read(INDEX, RECORD * u32_at(&head, 4) as usize)?;
    let names = read(INDEX + records.len(), u32_at(&head, 8) as usize)?;
    let records = records.as_chunks::<RECORD>().0;
    let mut o = Overlay::default();
    for (name, data) in replace {
        // The entry's record under that name: where its bytes are, and in which form.
        let named = |r: &&[u8; RECORD]| names.get(u32_at(*r, 0) as usize..).is_some_and(|n| n.starts_with(name.as_bytes()) && n.get(name.len()) == Some(&0));
        let r = records.iter().find(named).ok_or_else(|| invalid(format!("no entry named {name}")))?;
        let stored = match u32_at(r, 0x20) {
            0 => data.clone(),
            1 => chunks(data),
            form => return Err(invalid(format!("{name}: stored in form {form}"))),
        };
        let pad = (len + o.end.len() as u64).wrapping_neg() % 16;
        o.end.resize(o.end.len() + pad as usize, 0);
        let new: Vec<u8> = [len + o.end.len() as u64, stored.len() as u64, data.len() as u64].iter().flat_map(|v| v.to_le_bytes()).collect();
        o.end.extend(stored);
        let old = u64_at(r, 8);
        for (k, _) in records.iter().enumerate().filter(|(_, r)| u64_at(*r, 8) == old) {
            o.patches.push(((INDEX + RECORD * k + 8) as u64, new.clone()));
        }
    }
    Ok(o)
}

/// A content as a chain of LZ4 chunks.
fn chunks(data: &[u8]) -> Vec<u8> {
    let mut out = Vec::new();
    for chunk in data.chunks(CHUNK) {
        let block = lz4_flex::block::compress(chunk);
        out.extend((block.len() as u32).to_le_bytes());
        out.extend(block);
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn chunked(parts: &[&[u8]]) -> Vec<u8> {
        let mut raw = Vec::new();
        let mut before: Vec<u8> = Vec::new();
        for p in parts {
            let block = lz4_flex::block::compress_with_dict(p, &before[before.len().saturating_sub(CHUNK)..]);
            raw.extend((block.len() as u32).to_le_bytes());
            raw.extend(block);
            before.extend_from_slice(p);
        }
        raw
    }

    #[test]
    fn chunks_decode_and_may_refer_to_the_previous_one() {
        let first: Vec<u8> = (0..CHUNK).map(|i| (i * 7 % 251) as u8).collect();
        // The second chunk repeats the first one's end: its matches reach back.
        let second = first[CHUNK - 1000..].to_vec();
        let raw = chunked(&[&first, &second]);
        let out = decode_chunks(&raw, CHUNK + second.len()).unwrap();
        assert_eq!(&out[..CHUNK], &first[..]);
        assert_eq!(&out[CHUNK..], &second[..]);
        assert!(decode_chunks(&raw[..raw.len() - 3], CHUNK + second.len()).is_err());
    }

    fn tmp(name: &str) -> std::path::PathBuf {
        let dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../target/tmp");
        std::fs::create_dir_all(&dir).unwrap();
        dir.join(name)
    }

    /// A minimal zip: one stored entry and one chunked entry.
    #[test]
    fn the_central_directory_and_entries_read_back() {
        let path = tmp("datazip-test.zip");
        let plain = b"RIFF stored as is".to_vec();
        let packed_content = b"0123456789".repeat(50);
        let packed = chunked(&[&packed_content]);
        let mut zip = Vec::new();
        let mut central = Vec::new();
        for (name, data, size) in [("aa", &plain, plain.len()), ("bb", &packed, packed_content.len())] {
            let header = zip.len() as u32;
            zip.extend(0x0403_4b50u32.to_le_bytes());
            zip.extend([0u8; 14]);
            zip.extend((data.len() as u32).to_le_bytes());
            zip.extend((size as u32).to_le_bytes());
            zip.extend((name.len() as u16).to_le_bytes());
            zip.extend(0u16.to_le_bytes());
            zip.extend(name.as_bytes());
            zip.extend(data.iter());
            central.extend(0x0201_4b50u32.to_le_bytes());
            central.extend([0u8; 16]);
            central.extend((data.len() as u32).to_le_bytes());
            central.extend((size as u32).to_le_bytes());
            central.extend((name.len() as u16).to_le_bytes());
            central.extend([0u8; 12]);
            central.extend(header.to_le_bytes());
            central.extend(name.as_bytes());
        }
        let start = zip.len() as u32;
        zip.extend(&central);
        zip.extend(0x0605_4b50u32.to_le_bytes());
        zip.extend([0u8; 6]);
        zip.extend(2u16.to_le_bytes());
        zip.extend((central.len() as u32).to_le_bytes());
        zip.extend(start.to_le_bytes());
        zip.extend(0u16.to_le_bytes());
        std::fs::write(&path, &zip).unwrap();
        let mut z = DataZip::open(&path).unwrap();
        assert_eq!(z.entries.iter().map(|e| e.name.as_str()).collect::<Vec<_>>(), ["aa", "bb"]);
        let (a, b) = (z.entries[0].clone(), z.entries[1].clone());
        assert_eq!(z.read(&a).unwrap(), plain);
        assert_eq!(z.read(&b).unwrap(), packed_content);
        assert_eq!(z.head(&b, 4).unwrap(), b"0123");
    }

    /// An archive as the engine reads it: the index block, a record per name, then each
    /// entry's bytes on a multiple of 16. An entry: its names, its stored bytes, its
    /// decoded size, 1 for chunks or 0 for raw bytes.
    fn indexed(entries: &[(&[&str], &[u8], usize, u32)]) -> Vec<u8> {
        let names: Vec<u8> = entries.iter().flat_map(|e| e.0).flat_map(|n| [n.as_bytes(), &[0]].concat()).collect();
        let count = entries.iter().map(|e| e.0.len()).sum::<usize>();
        let mut zip = [&b"VZIP"[..], &(count as u32).to_le_bytes(), &(names.len() as u32).to_le_bytes(), &vec![0; RECORD * count][..], &names[..]].concat();
        let (mut record, mut name) = (INDEX, 0u32);
        for (its_names, stored, size, form) in entries {
            zip.resize(zip.len().next_multiple_of(16), 0);
            let at = zip.len() as u64;
            zip.extend(*stored);
            for n in *its_names {
                zip[record..record + 4].copy_from_slice(&name.to_le_bytes());
                zip[record + 8..record + 0x20].copy_from_slice(&[at, stored.len() as u64, *size as u64].map(u64::to_le_bytes).concat());
                zip[record + 0x20..record + 0x24].copy_from_slice(&form.to_le_bytes());
                (record, name) = (record + RECORD, name + n.len() as u32 + 1);
            }
        }
        zip
    }

    /// New contents for a chunked entry with two names and for a raw one: they follow
    /// the file's bytes, each in its entry's form; every record of those entries points
    /// at them, and nothing else changes.
    #[test]
    fn an_overlay_points_every_record_of_an_entry_at_its_new_content() {
        let packed_content = b"0123456789".repeat(50);
        let packed = chunked(&[&packed_content]);
        // More than one chunk of text.
        let long: Vec<u8> = (0..CHUNK + 5000).map(|i| b"{\"entity\": 1}\n"[i % 14]).collect();
        let zip = indexed(&[(&["aa"], b"stays", 5, 0), (&["bb", "b2"], &packed, packed_content.len(), 1), (&["cc"], b"raw bytes", 9, 0)]);
        let path = tmp("datazip-overlay.zip");
        std::fs::write(&path, &zip).unwrap();
        let o = overlay(&path, &[("cc".to_string(), b"new raw".to_vec()), ("b2".to_string(), long.clone())]).unwrap();
        let mut served = [&zip[..], &o.end[..]].concat();
        let mut patched = vec![false; zip.len()];
        for (at, bytes) in &o.patches {
            let span = *at as usize..*at as usize + bytes.len();
            served[span.clone()].copy_from_slice(bytes);
            patched[span].fill(true);
        }
        let record = |zip: &[u8], k: usize| [8, 0x10, 0x18].map(|o| u64_at(zip, INDEX + RECORD * k + o));
        // Both names of the second entry give the new content, in chunks.
        for k in [1, 2] {
            let [at, stored, size] = record(&served, k);
            assert!(at as usize >= zip.len() && at % 16 == 0, "record {k}: past the file's bytes, on a multiple of 16");
            assert_eq!(decode_chunks(&served[at as usize..][..stored as usize], size as usize).unwrap(), long, "record {k}");
        }
        let [at, stored, size] = record(&served, 3);
        assert_eq!((at % 16, stored, size), (0, 7, 7));
        assert_eq!(&served[at as usize..][..7], b"new raw", "raw as it was");
        // Three records changed, by 24 bytes each; the other bytes are the file's.
        assert_eq!(patched.iter().filter(|p| **p).count(), 3 * 24);
        assert!((0..zip.len()).all(|i| patched[i] || served[i] == zip[i]));
        // A name the index lacks, a name's start, a file without an index.
        assert!(overlay(&path, &[("dd".to_string(), Vec::new())]).unwrap_err().to_string().contains("no entry named dd"));
        assert!(overlay(&path, &[("b".to_string(), Vec::new())]).is_err());
        std::fs::write(&path, &zip[4..]).unwrap();
        assert!(overlay(&path, &[("aa".to_string(), Vec::new())]).unwrap_err().to_string().contains("no index block"));
    }
}
