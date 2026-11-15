//! A disk cache of translated shaders (D72), in `<--pipeline-cache DIR>/shaders`:
//! a program's translation for a set of options, with the naga module wgpu parses
//! from its WGSL. A later run makes the module from that naga module: it skips the
//! translation and naga's WGSL front, most of a new material's cost; wgpu still
//! validates it.
//!
//! - Entries sit in a folder named after this build (`BUILD`, from build.rs: the
//!   translator's and naga's sources and the compiler): another build never reads them.
//! - An entry serves only when its checksum holds and its stored code and options
//!   equal the live ones.
//! - Only modules wgpu accepted are queued; a thread of their own writes them,
//!   through a temporary file and a rename. A run never loads what it queued: its
//!   first use of a program takes the same path with or without the cache.

use super::{Options, Translation};
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex, PoisonError, mpsc};
use std::time::{Duration, Instant};

/// A hash of the sources the entries depend on (build.rs).
const BUILD: &str = env!("GYROSHIP_SHADER_BUILD");

/// The largest body decompressed: its size comes first, and a wrong one would be
/// allocated (only our own writer could get one past the checksum).
const MAX_BODY: u32 = 64 << 20;

/// An entry's body: the program's code, the options, how long translating and making
/// the module took (µs), the translation, and the naga module of its WGSL.
type Body = (Vec<u8>, Options, u64, Translation, naga::Module);

/// An entry to write: its file, then the body's parts but the naga module.
type Job = (PathBuf, Vec<u8>, Options, Translation, Duration);

/// What the writer thread did: entries written, their bytes, entries it failed to write.
#[derive(Default)]
struct Written {
    entries: u32,
    bytes: u64,
    failed: u32,
}

pub struct ShaderCache {
    dir: PathBuf,
    queue: Option<mpsc::Sender<Job>>,
    writer: Option<std::thread::JoinHandle<()>>,
    written: Arc<Mutex<Written>>,
    /// Entries queued this run.
    queued: std::collections::HashSet<String>,
    /// Entries made into modules, and entries rejected: unreadable, another program
    /// or other options in the file, or a module wgpu refused.
    pub loaded: u32,
    pub rejected: u32,
    /// Time spent reading entries (and looking for missing ones).
    pub read: Duration,
}

impl ShaderCache {
    /// The cache in `dir`: this build's folder in it, made if needed, and the writer
    /// thread started.
    pub fn open(dir: &Path) -> Result<ShaderCache, String> {
        let dir = dir.join(BUILD);
        std::fs::create_dir_all(&dir).map_err(|e| format!("{}: {e}", dir.display()))?;
        let (queue, jobs) = mpsc::channel::<Job>();
        let written = Arc::<Mutex<Written>>::default();
        let stats = written.clone();
        // It parses WGSL: a stack the size of the pipeline workers'.
        let writer = std::thread::Builder::new()
            .name("shader cache".into())
            .stack_size(16 << 20)
            .spawn(move || {
                for (file, code, options, t, took) in jobs {
                    let done = write(&file, &code, &options, &t, took);
                    let mut w = stats.lock().unwrap_or_else(PoisonError::into_inner);
                    match done {
                        Ok(bytes) => {
                            w.entries += 1;
                            w.bytes += bytes;
                        }
                        Err(e) => {
                            if w.failed == 0 {
                                eprintln!("gyroship: shader cache {}: {e}", file.display());
                            }
                            w.failed += 1;
                        }
                    }
                }
            })
            .map_err(|e| format!("no writer thread: {e}"))?;
        Ok(ShaderCache { dir, queue: Some(queue), writer: Some(writer), written, queued: Default::default(), loaded: 0, rejected: 0, read: Duration::ZERO })
    }

    /// Its folder, its entries, and the folders of other builds beside it (never
    /// removed: deleting `shaders` empties the cache).
    pub fn describe(&self) -> String {
        let count = |d: &Path| std::fs::read_dir(d).map_or(0, Iterator::count);
        let others = self.dir.parent().map_or(0, count).saturating_sub(1);
        format!("{} ({} entries; {others} folders of other builds)", self.dir.display(), count(&self.dir))
    }

    /// The translation, naga module and making time stored for program `key` with this
    /// code and these options. None without such an entry (a missing file is not
    /// counted as rejected), and for an entry this run queued.
    pub fn load(&mut self, key: u64, code: &[u8], options: &Options) -> Option<(Translation, naga::Module, Duration)> {
        let name = name(key, options);
        if self.queued.contains(&name) {
            return None;
        }
        let since = Instant::now();
        let bytes = std::fs::read(self.dir.join(&name));
        self.read += since.elapsed();
        match read(&bytes.ok()?) {
            Some((c, o, took, t, module)) if c == code && o == *options => Some((t, module, Duration::from_micros(took))),
            _ => {
                self.rejected += 1;
                None
            }
        }
    }

    /// Queues a module wgpu accepted: translation `t` of program `key` (this code, these
    /// options), made in `took`. Once per entry and run.
    pub fn store(&mut self, key: u64, code: &[u8], options: Options, t: Translation, took: Duration) {
        let name = name(key, &options);
        if let Some(queue) = &self.queue
            && self.queued.insert(name.clone())
        {
            let _ = queue.send((self.dir.join(name), code.to_vec(), options, t, took));
        }
    }

    /// The run's end: closes the queue and waits for the writer `patience` at most.
    /// Returns what the cache did, for stderr.
    pub fn finish(&mut self, patience: Duration) -> String {
        self.queue = None;
        let since = Instant::now();
        while self.writer.as_ref().is_some_and(|w| !w.is_finished()) && since.elapsed() < patience {
            std::thread::sleep(Duration::from_millis(10));
        }
        let w = self.written.lock().unwrap_or_else(PoisonError::into_inner);
        let left = (self.queued.len() as u32).saturating_sub(w.entries + w.failed);
        format!("{} loaded, {} rejected, {} ms reading; {} written ({} KiB), {} failed, {left} not written", self.loaded, self.rejected, self.read.as_millis(), w.entries, w.bytes / 1024, w.failed)
    }
}

/// An entry's file name: the program key, and a hash of the options.
fn name(key: u64, options: &Options) -> String {
    format!("{key:016x}-{:016x}.bin", crate::sass::fnv(&postcard::to_stdvec(options).unwrap_or_default()))
}

/// Writes an entry, with the naga module parsed from the translation's WGSL as wgpu
/// parses it, through a temporary file and a rename. Returns its size.
fn write(file: &Path, code: &[u8], options: &Options, t: &Translation, took: Duration) -> Result<u64, String> {
    let module = naga::front::wgsl::parse_str(&t.wgsl).map_err(|e| e.message().to_string())?;
    let bytes = entry(&body(code, options, took.as_micros() as u64, t, &module)?);
    let tmp = file.with_extension(format!("{}.tmp", std::process::id()));
    std::fs::write(&tmp, &bytes).and_then(|()| std::fs::rename(&tmp, file)).map_err(|e| {
        let _ = std::fs::remove_file(&tmp);
        e.to_string()
    })?;
    Ok(bytes.len() as u64)
}

/// An entry's body: its parts, then lz4 with the size first.
fn body(code: &[u8], options: &Options, took: u64, t: &Translation, module: &naga::Module) -> Result<Vec<u8>, String> {
    Ok(lz4_flex::block::compress_prepend_size(&postcard::to_stdvec(&(code, options, took, t, module)).map_err(|e| e.to_string())?))
}

/// An entry: a checksum of its body (FNV-1a 64), then the body.
fn entry(body: &[u8]) -> Vec<u8> {
    [&crate::sass::fnv(body).to_le_bytes()[..], body].concat()
}

/// An entry's body, if its checksum holds.
fn read(entry: &[u8]) -> Option<Body> {
    let (sum, body) = entry.split_at_checked(8)?;
    let size = u32::from_le_bytes(body.get(..4)?.try_into().ok()?);
    if *sum != crate::sass::fnv(body).to_le_bytes() || size > MAX_BODY {
        return None;
    }
    postcard::from_bytes(&lz4_flex::block::decompress_size_prepended(body).ok()?).ok()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A cache folder of its own for test `name`, next to the test binary.
    fn folder(name: &str) -> PathBuf {
        let dir = std::env::current_exe().unwrap().parent().unwrap().join(format!("shader-cache-{name}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        dir
    }

    #[test]
    fn an_entry_serves_only_its_code_and_options_once_written() {
        let dir = folder("entry");
        let (code, options) = super::super::tests::samples().swap_remove(0);
        let t = super::super::translate_with(&code, &options).unwrap();
        let took = Duration::from_millis(5);
        let mut c = ShaderCache::open(&dir).unwrap();
        c.store(t.key, &code, options.clone(), t.clone(), took);
        c.store(t.key, &code, options.clone(), t.clone(), took);
        let line = c.finish(Duration::from_secs(10));
        assert!(line.contains("; 1 written") && line.contains(" 0 failed, 0 not written"), "{line}");
        // Written, but never loaded back by the run that queued it.
        assert!(c.load(t.key, &code, &options).is_none());
        let mut c = ShaderCache::open(&dir).unwrap();
        let (back, module, stored) = c.load(t.key, &code, &options).expect("the entry");
        assert_eq!((&back, stored), (&t, took));
        naga::valid::Validator::new(naga::valid::ValidationFlags::all(), naga::valid::Capabilities::all()).validate(&module).unwrap();
        // Another program under its name, or its file under other options' name: rejected.
        assert!(c.load(t.key, &[&code[..], &[0; 16]].concat(), &options).is_none());
        let other = Options { clip_planes: 0, ..options.clone() };
        let file = dir.join(BUILD).join(name(t.key, &options));
        std::fs::copy(&file, dir.join(BUILD).join(name(t.key, &other))).unwrap();
        assert!(c.load(t.key, &code, &other).is_none());
        // Another valid body under the old checksum: rejected.
        let bytes = std::fs::read(&file).unwrap();
        std::fs::write(&file, [&bytes[..8], &body(&code, &options, 6000, &t, &module).unwrap()[..]].concat()).unwrap();
        assert!(c.load(t.key, &code, &options).is_none());
        assert_eq!(c.rejected, 3);
        // No file: a plain miss.
        assert!(c.load(t.key ^ 1, &code, &options).is_none());
        assert_eq!(c.rejected, 3);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn the_build_stamp_covers_the_compiler_the_translator_and_naga() {
        let list = include_str!(concat!(env!("OUT_DIR"), "/shader-sources.txt"));
        assert!(list.starts_with("rustc "), "{list}");
        for want in ["Cargo.toml", "src/shader/mod.rs", "src/shader/cache.rs", "src/sass/mod.rs", "../../vendor/naga/Cargo.toml", "../../vendor/naga/src/front/mod.rs"] {
            assert!(list.lines().any(|l| l == want), "{want}\n{list}");
        }
    }
}
