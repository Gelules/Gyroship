//! Save data: the file systems fsp-srv's OpenSaveDataFileSystem hands out,
//! with IFileSystem and IFile as the public documentation describes them
//! (switchbrew). Each save lives in memory. A validation run starts from an
//! empty save every time, so two runs give the same journal; with
//! `--save DIR` a save is loaded from `DIR/<key>` when first opened and
//! written back there at every Commit (D22).

use std::collections::{BTreeMap, BTreeSet};
use std::path::Path;

use crate::Kernel;
use crate::ipc::{Reply, Request};
use crate::service::Service;

/// fs results (module 2): PathNotFound, PathAlreadyExists.
pub const PATH_NOT_FOUND: u32 = 1 << 9 | 2;
pub const PATH_ALREADY_EXISTS: u32 = 2 << 9 | 2;
/// What GetFreeSpaceSize and GetTotalSpaceSize answer.
const SPACE: u64 = 0x4000_0000;

/// One save's tree: files by path, and the directories (`/` always exists).
#[derive(Debug, Default)]
pub struct SaveFs {
    pub files: BTreeMap<String, Vec<u8>>,
    pub dirs: BTreeSet<String>,
}

impl SaveFs {
    fn is_dir(&self, path: &str) -> bool {
        path == "/" || self.dirs.contains(path)
    }

    /// Loads the tree under `root` (missing: empty).
    fn load(root: &Path) -> std::io::Result<SaveFs> {
        let mut fs = SaveFs::default();
        if !root.exists() {
            return Ok(fs);
        }
        let mut todo = vec![(root.to_path_buf(), String::new())];
        while let Some((dir, prefix)) = todo.pop() {
            for e in std::fs::read_dir(&dir)? {
                let e = e?;
                let path = format!("{prefix}/{}", e.file_name().to_string_lossy());
                if e.file_type()?.is_dir() {
                    fs.dirs.insert(path.clone());
                    todo.push((e.path(), path));
                } else {
                    fs.files.insert(path, std::fs::read(e.path())?);
                }
            }
        }
        Ok(fs)
    }

    /// Writes the tree to `root`, through a sibling directory, then replaces it.
    fn store(&self, root: &Path) -> std::io::Result<()> {
        let tmp = root.with_extension("tmp");
        if tmp.exists() {
            std::fs::remove_dir_all(&tmp)?;
        }
        std::fs::create_dir_all(&tmp)?;
        for d in &self.dirs {
            std::fs::create_dir_all(tmp.join(&d[1..]))?;
        }
        for (p, bytes) in &self.files {
            std::fs::write(tmp.join(&p[1..]), bytes)?;
        }
        if root.exists() {
            std::fs::remove_dir_all(root)?;
        }
        std::fs::rename(&tmp, root)
    }
}

/// A path as the guest writes it, normalized: one leading `/`, no empty or
/// trailing component.
fn normalize(raw: &str) -> String {
    let parts: Vec<&str> = raw.split('/').filter(|p| !p.is_empty() && *p != ".").collect();
    format!("/{}", parts.join("/"))
}

fn parent(path: &str) -> &str {
    match path.rfind('/') {
        Some(0) | None => "/",
        Some(i) => &path[..i],
    }
}

#[derive(Debug)]
pub enum Save {
    /// IFileSystem over the save `key`.
    FileSystem { key: String },
    /// IFile: a file of the save `key`.
    File { key: String, path: String },
}

impl Save {
    pub fn label(&self) -> String {
        match self {
            Save::FileSystem { .. } => "fs IFileSystem (save)".into(),
            Save::File { .. } => "fs IFile (save)".into(),
        }
    }

    pub fn request(&mut self, k: &mut Kernel, req: &Request) -> Option<Reply> {
        match self {
            Save::FileSystem { key } => {
                let key = key.clone();
                k.save_fs_request(&key, req)
            }
            Save::File { key, path } => {
                let (key, path) = (key.clone(), path.clone());
                k.save_file_request(&key, &path, req)
            }
        }
    }
}

impl Kernel {
    /// fsp-srv 51: the save the attribute names (user id and save type),
    /// opened (and loaded with `--save`) on first use.
    pub fn open_save(&mut self, req: &Request) -> Option<Reply> {
        let space = *req.data.first()?;
        let attr = req.data.get(8..0x48)?;
        let user = u128::from_le_bytes(attr[8..0x18].try_into().ok()?);
        let key = format!("{user:032x}-{}", attr[0x20]);
        if !self.saves.contains_key(&key) {
            let fs = match &self.config.save_dir {
                Some(dir) => match SaveFs::load(&dir.join(&key)) {
                    Ok(fs) => fs,
                    Err(e) => self.stop(crate::STOP_EXIT, &format!("save {key}: cannot load: {e}")),
                },
                None => SaveFs::default(),
            };
            self.log(&format!("save: open {key} (space {space}, {} files{})", fs.files.len(), if self.config.save_dir.is_some() { ", persistent" } else { ", in memory" }));
            self.saves.insert(key.clone(), fs);
        }
        Some(Reply::object(Service::Save(Save::FileSystem { key })))
    }

    /// The path in X buffer `i`, normalized.
    fn path_arg(&self, req: &Request, i: usize) -> Option<String> {
        let b = req.send_buffer(i)?;
        let mut raw = vec![0; b.size.min(0x301) as usize];
        self.mem.read(b.addr, &mut raw).ok()?;
        let end = raw.iter().position(|&c| c == 0).unwrap_or(raw.len());
        Some(normalize(&String::from_utf8_lossy(&raw[..end])))
    }

    fn save_fs_request(&mut self, key: &str, req: &Request) -> Option<Reply> {
        let path = self.path_arg(req, 0);
        let commit = req.command == 10;
        let fs = self.saves.get_mut(key)?;
        let result = |ok: bool, err: u32| if ok { Reply::ok() } else { Reply::error(err) };
        let reply = match req.command {
            // CreateFile(option, size): its parent must exist.
            0 => {
                let (p, size) = (path?, req.u64_at(8)?);
                if fs.files.contains_key(&p) || fs.is_dir(&p) {
                    Reply::error(PATH_ALREADY_EXISTS)
                } else if !fs.is_dir(parent(&p)) {
                    Reply::error(PATH_NOT_FOUND)
                } else {
                    fs.files.insert(p, vec![0; size.min(0x1000_0000) as usize]);
                    Reply::ok()
                }
            }
            1 => result(fs.files.remove(&path?).is_some(), PATH_NOT_FOUND),
            // CreateDirectory.
            2 => {
                let p = path?;
                if fs.files.contains_key(&p) || fs.is_dir(&p) {
                    Reply::error(PATH_ALREADY_EXISTS)
                } else if !fs.is_dir(parent(&p)) {
                    Reply::error(PATH_NOT_FOUND)
                } else {
                    fs.dirs.insert(p);
                    Reply::ok()
                }
            }
            // DeleteDirectory (empty) and DeleteDirectoryRecursively.
            3 | 4 => {
                let p = path?;
                let prefix = format!("{p}/");
                let empty = !fs.files.keys().chain(fs.dirs.iter()).any(|x| x.starts_with(&prefix));
                if !fs.dirs.contains(&p) {
                    Reply::error(PATH_NOT_FOUND)
                } else if req.command == 3 && !empty {
                    Reply::error(crate::result::INVALID_STATE)
                } else {
                    fs.files.retain(|x, _| !x.starts_with(&prefix));
                    fs.dirs.retain(|x| x != &p && !x.starts_with(&prefix));
                    Reply::ok()
                }
            }
            // RenameFile(old, new).
            5 => {
                let (old, new) = (path?, self.path_arg(req, 1)?);
                let fs = self.saves.get_mut(key)?;
                match fs.files.remove(&old) {
                    Some(bytes) if !fs.files.contains_key(&new) => {
                        fs.files.insert(new, bytes);
                        Reply::ok()
                    }
                    Some(bytes) => {
                        fs.files.insert(old, bytes);
                        Reply::error(PATH_ALREADY_EXISTS)
                    }
                    None => Reply::error(PATH_NOT_FOUND),
                }
            }
            // GetEntryType: 0 directory, 1 file.
            7 => {
                let p = path?;
                match (fs.is_dir(&p), fs.files.contains_key(&p)) {
                    (true, _) => Reply::with(&0u32.to_le_bytes()),
                    (_, true) => Reply::with(&1u32.to_le_bytes()),
                    _ => Reply::error(PATH_NOT_FOUND),
                }
            }
            // OpenFile(mode).
            8 => {
                let p = path?;
                if fs.files.contains_key(&p) {
                    Reply::object(Service::Save(Save::File { key: key.to_string(), path: p }))
                } else {
                    Reply::error(PATH_NOT_FOUND)
                }
            }
            10 => Reply::ok(),
            11 | 12 => Reply::with(&SPACE.to_le_bytes()),
            _ => return None,
        };
        if self.journal.few(&format!("save fs {} {:#x}", req.command, reply.result), 4) {
            let p = self.path_arg(req, 0).unwrap_or_default();
            self.log(&format!("save: {key} command {} {p:?} -> {:#x}", req.command, reply.result));
        }
        if commit && let Some(dir) = self.config.save_dir.clone() {
            if let Err(e) = self.saves[key].store(&dir.join(key)) {
                self.stop(crate::STOP_EXIT, &format!("save {key}: cannot write: {e}"));
            }
            self.log(&format!("save: {key} committed to the save directory"));
        }
        Some(reply)
    }

    fn save_file_request(&mut self, key: &str, path: &str, req: &Request) -> Option<Reply> {
        Some(match req.command {
            // Read(option, offset, size) into the B buffer: the byte count.
            0 => {
                let (offset, size, b) = (req.u64_at(8)?, req.u64_at(0x10)?, req.out_buffer(0)?);
                let file = self.saves.get(key)?.files.get(path)?;
                let start = (offset as usize).min(file.len());
                let end = (start + size.min(b.size) as usize).min(file.len());
                let bytes = file[start..end].to_vec();
                if self.mem.write(b.addr, &bytes).is_err() {
                    return Some(Reply::error(crate::result::INVALID_POINTER));
                }
                Reply::with(&(bytes.len() as u64).to_le_bytes())
            }
            // Write(option, offset, size) from the A buffer; the file grows if needed.
            1 => {
                let (offset, size, a) = (req.u64_at(8)? as usize, req.u64_at(0x10)?, req.send_buffer(0)?);
                let mut bytes = vec![0; size.min(a.size).min(0x1000_0000) as usize];
                if self.mem.read(a.addr, &mut bytes).is_err() {
                    return Some(Reply::error(crate::result::INVALID_POINTER));
                }
                let file = self.saves.get_mut(key)?.files.get_mut(path)?;
                if file.len() < offset + bytes.len() {
                    file.resize(offset + bytes.len(), 0);
                }
                file[offset..offset + bytes.len()].copy_from_slice(&bytes);
                Reply::ok()
            }
            2 => Reply::ok(),
            3 => {
                let size = req.u64_at(0)?.min(0x1000_0000) as usize;
                self.saves.get_mut(key)?.files.get_mut(path)?.resize(size, 0);
                Reply::ok()
            }
            4 => Reply::with(&(self.saves.get(key)?.files.get(path)?.len() as u64).to_le_bytes()),
            _ => return None,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn paths_and_store_round_trip() {
        assert_eq!(normalize("save:/a//b/"), "/save:/a/b");
        assert_eq!(normalize("/save_game_data.save"), "/save_game_data.save");
        assert_eq!(parent("/a/b"), "/a");
        assert_eq!(parent("/a"), "/");

        let mut fs = SaveFs::default();
        fs.dirs.insert("/d".into());
        fs.files.insert("/d/f".into(), vec![1, 2, 3]);
        fs.files.insert("/g".into(), vec![4]);
        // Inside the workspace's target directory (ignored by Git).
        let root = Path::new(env!("CARGO_MANIFEST_DIR")).join(format!("../../target/tmp/save-test-{}", std::process::id()));
        fs.store(&root).expect("store");
        fs.store(&root).expect("store again over the first");
        let back = SaveFs::load(&root).expect("load");
        std::fs::remove_dir_all(&root).expect("clean up");
        assert_eq!(back.files, fs.files);
        assert_eq!(back.dirs, fs.dirs);
    }
}
