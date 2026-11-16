//! fs: `fsp-srv` and the RomFS the game reads raw through IStorage.
//!
//! The RomFS image is built over the files of the dump's `romfs` folder, in
//! the public format (switchbrew): a 0x50-byte header, the file data from
//! 0x200 (each file 16-byte aligned), then the directory hash table, the
//! directories, the file hash table and the files. Entries are sorted by
//! name, so every run builds the same image; file bytes are read from the
//! host files when the guest asks for them. For the targeted dump this gives
//! the size Lichii reports, 15 611 574 120 bytes.

use crate::Kernel;
use crate::ipc::{Reply, Request};
use crate::service::Service;
use gyroship_gpu::datazip::Overlay;
use gyroship_runtime::mem::perm;
use std::path::{Path, PathBuf};
use std::sync::OnceLock;

const HEADER: u64 = 0x50;
const DATA: u64 = 0x200;
const EMPTY: u32 = u32::MAX;

/// A file of the image: where its bytes are on the host and in the image.
#[derive(Debug)]
struct File {
    path: PathBuf,
    offset: u64,
    size: u64,
    /// An archive with entries replaced: what the image serves over the host file.
    overlay: Option<Overlay>,
}

#[derive(Debug)]
pub struct RomFs {
    /// The header, then (at `tables`) the four tables.
    header: Vec<u8>,
    tables: Vec<u8>,
    tables_at: u64,
    files: Vec<File>,
    size: u64,
    /// Each file's host handle, opened at its first read and kept (host state,
    /// rebuilt on demand: never part of the guest's). Host threads share them:
    /// reads are positional.
    // ponytail: one handle per file, so concurrent reads of one file (data.zip)
    // serialize on it (a synchronous file object); a handle per host thread if
    // play-mode loads measure it.
    open: Vec<OnceLock<std::fs::File>>,
}

/// RomFS hash of an entry name under its parent's offset.
fn hash(parent: u32, name: &[u8]) -> u32 {
    let mut h = parent ^ 123_456_789;
    for &c in name {
        h = h.rotate_right(5) ^ u32::from(c);
    }
    h
}

/// The bucket count Nintendo's builder uses for `n` entries.
fn buckets(n: usize) -> usize {
    match n {
        0..3 => 3,
        3..19 => n | 1,
        _ => (n..).find(|c| [2, 3, 5, 7, 11, 13, 17].iter().all(|p| c % p != 0)).expect("some count"),
    }
}

fn pad4(v: &mut Vec<u8>) {
    v.resize(v.len().next_multiple_of(4), 0);
}

/// A directory being laid out.
struct Dir {
    name: Vec<u8>,
    parent: usize,
    dirs: Vec<usize>,
    files: Vec<(Vec<u8>, PathBuf, u64, Option<Overlay>)>,
}

impl RomFs {
    /// The image of the files under `root`.
    pub fn build(root: &Path) -> std::io::Result<RomFs> {
        RomFs::layered(root, None).map(|(r, _)| r)
    }

    /// The image of the files under `root`, where a file at the same path
    /// under `layer` takes the dump file's place, bytes and size (the dump is
    /// only read, rule 4). A folder `<path>.entries` under `layer` replaces
    /// entries of the archive at `<path>` instead, each by the file of its
    /// name, decoded; a name the archive lacks is refused. Also returns what
    /// the layer replaced.
    // ponytail: a layer only replaces files; one the dump lacks is ignored, and
    // the caller logs what was replaced so a misplaced file shows.
    pub fn layered(root: &Path, layer: Option<&Path>) -> std::io::Result<(RomFs, Vec<String>)> {
        let mut dirs = vec![Dir { name: Vec::new(), parent: 0, dirs: Vec::new(), files: Vec::new() }];
        let mut replaced = Vec::new();
        let mut todo = vec![(0, root.to_path_buf(), PathBuf::new())];
        while let Some((d, path, rel)) = todo.pop() {
            let mut entries: Vec<_> = std::fs::read_dir(&path)?.collect::<Result<_, _>>()?;
            entries.sort_by_key(|e| e.file_name());
            for e in entries {
                let name = e.file_name().to_string_lossy().into_owned().into_bytes();
                let rel = rel.join(e.file_name());
                if e.file_type()?.is_dir() {
                    let i = dirs.len();
                    dirs.push(Dir { name, parent: d, dirs: Vec::new(), files: Vec::new() });
                    dirs[d].dirs.push(i);
                    todo.push((i, e.path(), rel));
                } else {
                    let shown = rel.to_string_lossy().replace('\\', "/");
                    let over = layer.map(|l| l.join(&rel)).filter(|p| p.is_file());
                    if over.is_some() {
                        replaced.push(shown.clone());
                    }
                    let host = over.unwrap_or_else(|| e.path());
                    let mut size = std::fs::metadata(&host)?.len();
                    let mut overlay = None;
                    if let Some(folder) = layer.map(|l| l.join(format!("{shown}.entries"))).filter(|p| p.is_dir()) {
                        let mut entries = Vec::new();
                        for f in std::fs::read_dir(&folder)? {
                            let f = f?;
                            entries.push((f.file_name().to_string_lossy().into_owned(), std::fs::read(f.path())?));
                        }
                        entries.sort();
                        let o = gyroship_gpu::datazip::overlay(&host, &entries)?;
                        replaced.push(entries.iter().fold(format!("{shown} entries"), |s, (n, _)| s + " " + n));
                        size += o.end.len() as u64;
                        overlay = Some(o);
                    }
                    dirs[d].files.push((name, host, size, overlay));
                }
            }
        }
        // Offsets of every directory and file entry in their tables.
        let entry = |name: &[u8], fixed: usize| fixed + name.len().next_multiple_of(4);
        let mut dir_at = vec![0u32; dirs.len()];
        let mut at = 0;
        for (i, d) in dirs.iter().enumerate() {
            dir_at[i] = at as u32;
            at += entry(&d.name, 0x18);
        }
        let mut file_at = Vec::new();
        let mut at = 0;
        let mut files = Vec::new();
        let mut data = 0u64;
        for d in &mut dirs {
            let mut here = Vec::new();
            for (name, path, size, overlay) in &mut d.files {
                here.push(at as u32);
                at += entry(name, 0x20);
                files.push(File { path: path.clone(), offset: DATA + data, size: *size, overlay: overlay.take() });
                data = (data + *size).next_multiple_of(16);
            }
            file_at.push(here);
        }
        let data_end = files.last().map_or(DATA, |f| f.offset + f.size);
        // The tables.
        let (nd, nf) = (dirs.len(), files.len());
        let (mut dir_hash, mut file_hash) = (vec![EMPTY; buckets(nd)], vec![EMPTY; buckets(nf)]);
        let (mut dir_meta, mut file_meta) = (Vec::new(), Vec::new());
        let mut file_index = 0;
        for (i, d) in dirs.iter().enumerate() {
            let parent = dir_at[d.parent];
            let h = hash(parent, &d.name) as usize % dir_hash.len();
            let sibling = dirs[d.parent].dirs.iter().position(|&x| x == i).and_then(|p| dirs[d.parent].dirs.get(p + 1)).map_or(EMPTY, |&s| dir_at[s]);
            let sibling = if i == 0 { EMPTY } else { sibling };
            let child_dir = d.dirs.first().map_or(EMPTY, |&c| dir_at[c]);
            let child_file = file_at[i].first().copied().unwrap_or(EMPTY);
            for v in [parent, sibling, child_dir, child_file, dir_hash[h], d.name.len() as u32] {
                dir_meta.extend(v.to_le_bytes());
            }
            dir_meta.extend(&d.name);
            pad4(&mut dir_meta);
            dir_hash[h] = dir_at[i];
            for (k, (name, _, size, _)) in d.files.iter().enumerate() {
                let me = file_at[i][k];
                let h = hash(dir_at[i], name) as usize % file_hash.len();
                let sibling = file_at[i].get(k + 1).copied().unwrap_or(EMPTY);
                file_meta.extend(dir_at[i].to_le_bytes());
                file_meta.extend(sibling.to_le_bytes());
                file_meta.extend((files[file_index].offset - DATA).to_le_bytes());
                file_meta.extend(size.to_le_bytes());
                file_meta.extend(file_hash[h].to_le_bytes());
                file_meta.extend((name.len() as u32).to_le_bytes());
                file_meta.extend(name);
                pad4(&mut file_meta);
                file_hash[h] = me;
                file_index += 1;
            }
        }
        let tables_at = data_end.next_multiple_of(4);
        let mut tables = Vec::new();
        let mut header = Vec::new();
        header.extend(HEADER.to_le_bytes());
        for t in [dir_hash.iter().flat_map(|v| v.to_le_bytes()).collect::<Vec<u8>>(), dir_meta, file_hash.iter().flat_map(|v| v.to_le_bytes()).collect(), file_meta] {
            header.extend((tables_at + tables.len() as u64).to_le_bytes());
            header.extend((t.len() as u64).to_le_bytes());
            tables.extend(t);
        }
        header.extend(DATA.to_le_bytes());
        let size = tables_at + tables.len() as u64;
        let open = files.iter().map(|_| OnceLock::new()).collect();
        replaced.sort();
        Ok((RomFs { header, tables, tables_at, files, size, open }, replaced))
    }

    pub fn size(&self) -> u64 {
        self.size
    }

    /// Reads `out.len()` bytes of the image at `offset`; past the end reads zeros.
    pub fn read(&self, offset: u64, out: &mut [u8]) -> std::io::Result<()> {
        out.fill(0);
        let end = offset + out.len() as u64;
        let mut copy = |start: u64, bytes: &[u8]| {
            let (lo, hi) = (offset.max(start), end.min(start + bytes.len() as u64));
            if lo < hi {
                out[(lo - offset) as usize..(hi - offset) as usize].copy_from_slice(&bytes[(lo - start) as usize..(hi - start) as usize]);
            }
        };
        copy(0, &self.header);
        copy(self.tables_at, &self.tables);
        for (f, open) in self.files.iter().zip(&self.open) {
            let (lo, hi) = (offset.max(f.offset), end.min(f.offset + f.size));
            if lo < hi {
                let out = &mut out[(lo - offset) as usize..(hi - offset) as usize];
                // The host file's bytes, all of them before what an overlay adds.
                let (at, cut) = (lo - f.offset, f.size - f.overlay.as_ref().map_or(0, |o| o.end.len() as u64));
                let host = cut.saturating_sub(at).min(out.len() as u64) as usize;
                if host > 0 {
                    if open.get().is_none() {
                        let _ = open.set(std::fs::File::open(&f.path)?);
                    }
                    let file = open.get().ok_or_else(|| std::io::Error::other("RomFS file not open"))?;
                    read_at(file, at, &mut out[..host])?;
                }
                // Over them, the overlay's patches; after them, its end.
                if let Some(o) = &f.overlay {
                    for (p, bytes) in &o.patches {
                        let (lo, hi) = (at.max(*p), (at + host as u64).min(p + bytes.len() as u64));
                        if lo < hi {
                            out[(lo - at) as usize..(hi - at) as usize].copy_from_slice(&bytes[(lo - p) as usize..(hi - p) as usize]);
                        }
                    }
                    let from = (at + host as u64).saturating_sub(cut) as usize;
                    let rest = out.len() - host;
                    out[host..].copy_from_slice(&o.end[from..from + rest]);
                }
            }
        }
        Ok(())
    }
}

/// Fills `buf` from `file` at `at`, a positional read (the handle stays open
/// between requests, the OS's read-ahead with it).
fn read_at(file: &std::fs::File, mut at: u64, mut buf: &mut [u8]) -> std::io::Result<()> {
    while !buf.is_empty() {
        #[cfg(windows)]
        let n = std::os::windows::fs::FileExt::seek_read(file, buf, at)?;
        #[cfg(unix)]
        let n = std::os::unix::fs::FileExt::read_at(file, buf, at)?;
        if n == 0 {
            return Err(std::io::ErrorKind::UnexpectedEof.into());
        }
        buf = &mut buf[n..];
        at += n as u64;
    }
    Ok(())
}

#[derive(Debug)]
pub enum Fs {
    /// `fsp-srv`: IFileSystemProxy.
    Proxy,
    /// IStorage over the RomFS.
    RomStorage,
}

impl Fs {
    pub fn label(&self) -> String {
        match self {
            Fs::Proxy => "fsp-srv".into(),
            Fs::RomStorage => "fs IStorage (RomFS)".into(),
        }
    }

    pub fn request(&mut self, k: &mut Kernel, req: &Request) -> Option<Reply> {
        Some(match (self, req.command) {
            // SetCurrentProcess; GetGlobalAccessLogMode (no access log); OpenDataStorageByCurrentProcess.
            (Fs::Proxy, 1) => Reply::ok(),
            (Fs::Proxy, 1005) => Reply::with(&0u32.to_le_bytes()),
            (Fs::Proxy, 200) => Reply::object(Service::Fs(Fs::RomStorage)),
            // OpenSaveDataFileSystem(space id, attribute).
            (Fs::Proxy, 51) => return k.open_save(req),
            // Read(offset, size) into the B buffer: checked and counted here; the
            // bytes are read and land after the kernel lock, before the caller
            // resumes (`crate::Deferred`). GetSize.
            (Fs::RomStorage, 0) => {
                let (offset, size) = (req.u64_at(0)?, req.u64_at(8)?);
                let b = req.out_buffer(0)?;
                let len = size.min(b.size);
                let romfs = k.romfs.clone()?;
                if k.mem.accessible(b.addr, len, perm::W).is_err() {
                    k.anomaly("romfs read buffer", &format!("{size:#x} bytes at {offset:#x} into {:#x}+{:#x}: not writable", b.addr, b.size));
                    return Some(Reply::error(crate::result::INVALID_POINTER));
                }
                k.count("romfs reads");
                // Each read, to see what loads read and whether any is read again (host statistics).
                if k.config.perf.is_some() {
                    k.census.romfs.push((k.gs.presents, offset, size));
                }
                k.deferred = Some(crate::Deferred::RomFsRead { romfs, mem: k.mem.clone(), offset, addr: b.addr, len });
                Reply::ok()
            }
            (Fs::RomStorage, 4) => Reply::with(&k.romfs.as_ref()?.size().to_le_bytes()),
            _ => return None,
        })
    }
}
