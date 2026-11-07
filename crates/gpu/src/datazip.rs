//! The game engine's asset archive (romfs `data.zip`, about 15 GiB): a zip
//! (zip64 sizes and offsets) of entries named by a 64-bit hash. An entry is
//! either stored raw (its compressed size equals its size) or a chain of LZ4
//! blocks, each a u32 compressed length then the block, decoding to 64 KiB
//! but for the last; a block may refer back into the previous one.

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

    /// A minimal zip: one stored entry and one chunked entry.
    #[test]
    fn the_central_directory_and_entries_read_back() {
        let dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../target/tmp");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("datazip-test.zip");
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
}
