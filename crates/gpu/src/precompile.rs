//! `gyroship precompile` (D110 step 8): the game's shaders made into pipeline libraries before
//! play, in the raw path's pipeline cache (`--pipeline-cache DIR`'s `vulkan/libraries.cache`),
//! where the game then finds them instead of compiling them:
//! - every pixel program of data.zip that translates, with the options the game gives it: the
//!   jump targets its blob holds when it has tables (never variant 0, every instruction a block,
//!   which the driver once took minutes over);
//! - the vertex programs past play translated (this build's shader cache entries: their pairing
//!   is in their options), in both layouts a vertex stage may take (with and without dynamic
//!   offsets: the pair decides), as nothing tells which yet.
//!
//! Each library is made exactly as the game makes it ([`crate::raw::Maker`], layouts from
//! [`crate::draw::interface_layout`]), first without compiling: one a cache has is not made again.
//! The interfaces and the links cost nothing at the first draw: they are left to it. A one-line
//! signature ends a run at once when nothing changed. A journal of the libraries in flight lets a
//! run that the driver brought down leave out, once it happened twice, the library it was making.

use crate::datazip::DataZip;
use crate::draw::{TABLE_MAX, interface_layout, jump_targets};
use crate::raw::{Maker, shader_part};
use crate::sass::{self, Blob, Stage};
use crate::shader::{self, JumpTable, Options, Translation};
use ash::vk;
use std::collections::{BTreeMap, BTreeSet};
use std::io::Write;
use std::path::Path;
use std::sync::{Mutex, PoisonError, mpsc};
use wgpu::hal::api::Vulkan;

/// A program of data.zip: its stage, its code as the GPU reads it (SPH then program for the
/// graphics stages), the blob's whole code section (the program, padding, then data), its
/// control section, and how many blobs hold it.
pub struct Program {
    pub stage: Stage,
    pub code: Vec<u8>,
    pub section: Vec<u8>,
    pub control: Vec<u8>,
    pub blobs: usize,
}

impl Program {
    /// The offset in the code section where a jump table's bank starts: the first data word
    /// after the program and its padding (the rule play confirmed, 111 programs of 111).
    pub fn data_start(&self) -> usize {
        let end = self.code.len();
        end + 16 * sass::words(&self.section[end.min(self.section.len())..]).take_while(|w| w.opcode() == sass::NOP || w.0 == 0).count()
    }

    /// The targets of `tables` as the blob holds them (the rule of `data_start`).
    pub fn blob_targets(&self, tables: &[JumpTable]) -> BTreeSet<usize> {
        jump_targets(tables, &table_bytes(tables, &self.section, self.data_start()), self.code.len())
    }
}

/// The programs of data.zip's shader blobs, by key: the first blob of each, and how many.
pub fn inventory(zip: &Path) -> std::io::Result<BTreeMap<u64, Program>> {
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
pub fn table_bytes(tables: &[JumpTable], data: &[u8], at: usize) -> Vec<Vec<u8>> {
    tables
        .iter()
        .map(|&(_, _, offset, entries)| {
            let start = (at + usize::from(offset)).min(data.len());
            let len = entries.map_or(TABLE_MAX, |n| 4 * n as usize).min(data.len() - start);
            data[start..start + len].to_vec()
        })
        .collect()
}

/// The threads making libraries: the probe's measure (D110 step 2: 3 threads, 293 MB more at most).
const THREADS: usize = 3;

/// A library to make: its name (the journal's), whether it is a vertex program's, its SPIR-V,
/// and its layout.
struct Job {
    name: String,
    vertex: bool,
    words: Vec<u32>,
    layout: vk::PipelineLayout,
}

/// What `run` did, counted; `done`: the jobs dealt with, of `total`.
#[derive(Default)]
struct Counts {
    made: usize,
    compiled: usize,
    failed: usize,
    left_out: usize,
    untranslated: usize,
    done: usize,
    total: usize,
}

impl Counts {
    /// A job dealt with: made (`Ok`, compiled or found in a cache) or failed, told to `progress`.
    fn tell(&mut self, result: Result<bool, String>, name: &str, progress: &mut dyn FnMut(usize, usize)) {
        match result {
            Ok(compiled) => {
                self.made += 1;
                self.compiled += usize::from(compiled);
            }
            Err(e) => {
                self.failed += 1;
                eprintln!("gyroship: precompile {name}: {e}");
            }
        }
        self.skip(progress);
    }

    /// A job dealt with without a library.
    fn skip(&mut self, progress: &mut dyn FnMut(usize, usize)) {
        self.done += 1;
        progress(self.done, self.total);
    }
}

/// Precompiles into `dir` (a `--pipeline-cache` folder) the programs of `zip` and of past play,
/// telling `progress` (done, total) as libraries are made. Returns the report: what was made, or
/// why nothing had to be (up to date, or a GPU without pipeline libraries linked fast).
pub fn run(zip: &Path, dir: &Path, progress: &mut dyn FnMut(usize, usize)) -> Result<String, String> {
    let since = std::time::Instant::now();
    // The signature, written at each run that ends well (the launcher's light: newer than the game).
    let folder = dir.join("vulkan").join(shader::cache::BUILD);
    let stamp = folder.join("precompiled.txt");
    let write = |text: &str| std::fs::create_dir_all(&folder).and_then(|()| std::fs::write(&stamp, text)).map_err(|e| format!("{}: {e}", stamp.display()));
    const UNUSED: &str = "not used on this GPU: it links no pipeline libraries fast";
    // The adapter alone first (the device's, `Surfaces::open`'s choice): nothing to do is said at once.
    let adapter = pollster::block_on(crate::Renderer::instance_for(crate::Backend::Vulkan).request_adapter(&wgpu::RequestAdapterOptions { power_preference: wgpu::PowerPreference::HighPerformance, compatible_surface: None, force_fallback_adapter: false, apply_limit_buckets: false }))
        .map_err(|e| format!("no Vulkan adapter: {e}"))?;
    if !crate::vk::extras(&adapter).is_some_and(|e| e.pipeline_library && e.fast_linking && e.dynamic_rendering) {
        write(&format!("{UNUSED}\n"))?;
        return Ok(UNUSED.into());
    }
    let (signature, driver) = signature(&adapter, zip)?;
    drop(adapter);
    // What earlier runs made under this signature: data.zip's programs, then the play entries the
    // signature's file names. Nothing new: said at once; new play entries: those alone are made.
    let shaders = dir.join("shaders");
    let names = shader::cache::this_build_names(&shaders);
    let text = std::fs::read_to_string(&stamp).unwrap_or_default();
    let mut lines = text.lines();
    let signed = lines.next() == Some(signature.as_str());
    let seen: BTreeSet<String> = if signed { lines.map(str::to_string).collect() } else { BTreeSet::new() };
    if signed && names.is_subset(&seen) {
        write(&text)?;
        return Ok("up to date".into());
    }
    let mut s = crate::surfaces::Surfaces::headless(crate::Backend::Vulkan)?;
    if !s.record_raw() || s.raw.as_ref().is_none_or(|m| m.libraries.is_none()) {
        write(&format!("{UNUSED}\n"))?;
        return Ok(UNUSED.into());
    }
    s.use_pipeline_cache(dir).map(drop)?;
    std::fs::create_dir_all(&folder).map_err(|e| format!("{}: {e}", folder.display()))?;
    let (journal, left_out) = flight(&folder, driver)?;
    let programs = if signed { BTreeMap::new() } else { inventory(zip).map_err(|e| format!("{}: {e}", zip.display()))? };
    let pixels: Vec<(u64, &Program)> = programs.iter().filter(|(_, p)| p.stage == Stage::Pixel).map(|(&k, p)| (k, p)).collect();
    let fresh: BTreeSet<String> = names.difference(&seen).cloned().collect();
    let vertices: Vec<(String, Options, Translation)> = shader::cache::this_build(&shaders, &fresh).into_iter().filter(|(_, _, t)| t.stage == Stage::Vertex).collect();
    let limit = s.dynamic_max as usize;
    let Some(m) = s.raw.as_mut() else { return Err("no raw path".into()) };
    let (maker, options) = (m.raw.maker(), m.options.clone());
    let mut c = Counts { total: pixels.len() + 2 * vertices.len(), ..Default::default() };
    let (jobs_tx, jobs_rx) = mpsc::sync_channel::<Job>(2 * THREADS);
    let (jobs_rx, journal) = (Mutex::new(jobs_rx), Mutex::new(journal));
    let (done_tx, done_rx) = mpsc::channel::<(String, Result<bool, String>)>();
    std::thread::scope(|scope| -> Result<(), String> {
        for _ in 0..THREADS {
            let (jobs, results, maker, journal) = (&jobs_rx, done_tx.clone(), maker.clone(), &journal);
            scope.spawn(move || {
                loop {
                    // The lock goes with this statement: the threads make their libraries at once.
                    let next = jobs.lock().unwrap_or_else(PoisonError::into_inner).recv();
                    let Ok(job) = next else { break };
                    note(journal, '+', &job.name);
                    let result = library(&maker, &job);
                    note(journal, '-', &job.name);
                    let _ = results.send((job.name, result));
                }
            });
        }
        drop(done_tx);
        // A job to the threads (unless left out), then what they finished meanwhile.
        let send = |c: &mut Counts, progress: &mut dyn FnMut(usize, usize), job: Job| -> Result<(), String> {
            if left_out.contains(&job.name) {
                c.left_out += 1;
                c.skip(progress);
            } else {
                jobs_tx.send(job).map_err(|_| "the threads making libraries ended".to_string())?;
            }
            while let Ok((name, result)) = done_rx.try_recv() {
                c.tell(result, &name, progress);
            }
            Ok(())
        };
        for (key, p) in &pixels {
            let name = format!("p{key:016x}");
            let Some(t) = pixel_translation(p) else {
                c.untranslated += 1;
                c.skip(progress);
                continue;
            };
            let dynamic = t.interface.cbufs.len() + usize::from(t.interface.swizzles) <= limit;
            match prepare(m, &options, &t, false, dynamic) {
                Ok((words, layout)) => send(&mut c, progress, Job { name, vertex: false, words, layout })?,
                Err(e) => c.tell(Err(e), &name, progress),
            }
        }
        for (entry, _, t) in &vertices {
            let id = format!("v{}", entry.trim_end_matches(".bin"));
            let mut layouts = BTreeSet::new();
            for dynamic in [true, false] {
                let name = format!("{id}-{}", if dynamic { "d" } else { "s" });
                match prepare(m, &options, t, true, dynamic) {
                    // A stage without buffers has one layout either way: made once.
                    Ok((_, layout)) if !layouts.insert(layout) => c.skip(progress),
                    Ok((words, layout)) => send(&mut c, progress, Job { name, vertex: true, words, layout })?,
                    Err(e) => c.tell(Err(e), &name, progress),
                }
            }
        }
        drop(jobs_tx);
        for (name, result) in done_rx.iter() {
            c.tell(result, &name, progress);
        }
        Ok(())
    })?;
    let written = m.raw.save_cache()?.unwrap_or(0);
    write(&format!("{signature}\n{}", names.iter().map(|n| format!("{n}\n")).collect::<String>()))?;
    let _ = std::fs::remove_file(folder.join("journal.txt"));
    Ok(format!(
        "{} {} made in {:.0} s ({} compiled, {} found in a cache); {} failed, {} left out, {} programs that do not translate; cache {}",
        c.made,
        if c.made == 1 { "library" } else { "libraries" },
        since.elapsed().as_secs_f64(),
        c.compiled,
        c.made - c.compiled,
        c.failed,
        c.left_out,
        c.untranslated,
        if written > 0 { format!("written ({} KiB)", written / 1024) } else { "unchanged".into() }
    ))
}

/// A pixel program translated as the game translates it for drawing: without options, or with
/// the jump targets its blob holds when it has tables.
fn pixel_translation(p: &Program) -> Option<Translation> {
    let t = shader::translate_with(&p.code, &Options::default()).ok()?;
    match t.interface.jump_tables.as_ref().filter(|t| !t.is_empty()) {
        Some(tables) => shader::translate_with(&p.code, &Options { jump_targets: Some(p.blob_targets(tables)), ..Default::default() }).ok(),
        None => Some(t),
    }
}

/// A module's SPIR-V and its library's layout, as the game makes them: the translation's WGSL read
/// by naga as `keep_ir` reads it, written with the device's options and its stage's group
/// numbered densely (group 0 for a vertex program, 1 for a pixel program); the layout of its own
/// set only (a vertex program's, empty without bindings; a pixel program's after a null set 0, or
/// none without bindings).
fn prepare(m: &mut crate::raw::RawMode, options: &naga::back::spv::Options<'static>, t: &Translation, vertex: bool, dynamic: bool) -> Result<(Vec<u32>, vk::PipelineLayout), String> {
    let module = naga::front::wgsl::parse_str(&t.wgsl).map_err(|e| e.message().to_string())?;
    let info = naga::valid::Validator::new(naga::valid::ValidationFlags::all(), naga::valid::Capabilities::all()).validate(&module).map_err(|e| e.to_string())?;
    let (visibility, stage) = if vertex { (wgpu::ShaderStages::VERTEX, naga::ShaderStage::Vertex) } else { (wgpu::ShaderStages::FRAGMENT, naga::ShaderStage::Fragment) };
    let entries = interface_layout(t, visibility, dynamic);
    let groups: Vec<&[wgpu::BindGroupLayoutEntry]> = if vertex { vec![entries.as_slice()] } else { vec![&[][..], entries.as_slice()] };
    let words = crate::vk::spirv(&module, &info, stage, "main", options, &crate::vk::binding_map(&groups))?;
    let set = m.raw.set_layout(&entries)?;
    let sets = match (vertex, entries.is_empty()) {
        (true, _) => vec![set],
        (false, true) => Vec::new(),
        (false, false) => vec![vk::DescriptorSetLayout::null(), set],
    };
    Ok((words, m.raw.pipeline_layout(&sets, true)?))
}

/// Makes a job's library as the game makes it ([`shader_part`]), then destroys it (the cache keeps
/// it): whether it compiled.
fn library(maker: &Maker, job: &Job) -> Result<bool, String> {
    let ash = &maker.ash;
    let module = unsafe { ash.create_shader_module(&vk::ShaderModuleCreateInfo::default().code(&job.words), None) }.map_err(|e| format!("shader module: {e}"))?;
    let made = maker.make(&shader_part(job.vertex, Some(module), job.layout));
    unsafe {
        if let Ok(m) = &made {
            ash.destroy_pipeline(m.pipeline, None);
        }
        ash.destroy_shader_module(module, None);
    }
    made.map(|m| m.compiled)
}

/// What the libraries depend on, in one line: the GPU, its driver and its cache's UUID, this
/// build, and data.zip (size and modification time); and the driver's version.
fn signature(adapter: &wgpu::Adapter, zip: &Path) -> Result<(String, u32), String> {
    let p = unsafe { adapter.as_hal::<Vulkan>() }.map(|a| a.physical_device_capabilities().properties()).ok_or("not Vulkan")?;
    let meta = std::fs::metadata(zip).map_err(|e| format!("{}: {e}", zip.display()))?;
    let modified = meta.modified().ok().and_then(|t| t.duration_since(std::time::UNIX_EPOCH).ok()).map_or(0, |d| d.as_secs());
    let uuid: String = p.pipeline_cache_uuid.iter().map(|b| format!("{b:02x}")).collect();
    let line = format!("{} {:04x}:{:04x} driver {:08x} cache {uuid} build {} data.zip {} {modified}", adapter.get_info().name, p.vendor_id, p.device_id, p.driver_version, shader::cache::BUILD, meta.len());
    Ok((line, p.driver_version))
}

/// The journal of this run's libraries in flight, made anew, and the libraries left out under
/// `driver`: those an earlier run left in flight twice. What a run left in flight when it ended
/// early (the driver brought it down, or it was stopped) is suspected once more.
fn flight(folder: &Path, driver: u32) -> Result<(std::fs::File, BTreeSet<String>), String> {
    let (journal, file) = (folder.join("journal.txt"), folder.join(format!("suspects-{driver:08x}.txt")));
    let mut suspects: BTreeMap<String, u32> = std::fs::read_to_string(&file).unwrap_or_default().lines().filter_map(|l| l.split_once(' ').and_then(|(name, n)| Some((name.to_string(), n.parse().ok()?)))).collect();
    if let Ok(text) = std::fs::read_to_string(&journal) {
        let mut flying = BTreeSet::new();
        for line in text.lines() {
            match line.split_at_checked(1) {
                Some(("+", name)) => flying.insert(name.to_string()),
                Some(("-", name)) => flying.remove(name),
                _ => false,
            };
        }
        for name in flying {
            *suspects.entry(name).or_default() += 1;
        }
        let lines: String = suspects.iter().map(|(name, n)| format!("{name} {n}\n")).collect();
        std::fs::write(&file, lines).map_err(|e| format!("{}: {e}", file.display()))?;
    }
    let left_out = suspects.into_iter().filter(|&(_, n)| n >= 2).map(|(name, _)| name).collect();
    Ok((std::fs::File::create(&journal).map_err(|e| format!("{}: {e}", journal.display()))?, left_out))
}

/// One line of the journal, written at once (the run may end with the next call; a file write
/// outlives the process that made it).
fn note(journal: &Mutex<std::fs::File>, sign: char, name: &str) {
    let _ = writeln!(journal.lock().unwrap_or_else(PoisonError::into_inner), "{sign}{name}");
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

    /// A library left in flight by a run that ended early is suspected; left in flight by a
    /// second run, it is left out, under that driver only; one that ended is not suspected.
    #[test]
    fn a_library_in_flight_twice_is_left_out() {
        let folder = std::env::current_exe().unwrap().parent().unwrap().join(format!("precompile-flight-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&folder);
        std::fs::create_dir_all(&folder).unwrap();
        let journal = folder.join("journal.txt");
        let (_, left_out) = flight(&folder, 7).unwrap();
        assert!(left_out.is_empty(), "no journal yet");
        std::fs::write(&journal, "+a\n+b\n-b\n").unwrap();
        let (_, left_out) = flight(&folder, 7).unwrap();
        assert!(left_out.is_empty(), "a suspected once: {left_out:?}");
        std::fs::write(&journal, "+c\n-c\n+a\n").unwrap();
        let (_, left_out) = flight(&folder, 7).unwrap();
        assert_eq!(left_out, BTreeSet::from(["a".to_string()]));
        std::fs::write(&journal, "").unwrap();
        assert!(flight(&folder, 8).unwrap().1.is_empty(), "another driver suspects nothing");
        let _ = std::fs::remove_dir_all(&folder);
    }

    /// A shader blob of `code` (SPH then program), as data.zip holds them.
    fn blob(code: &[u8]) -> Vec<u8> {
        let mut b = Vec::new();
        for v in [sass::BLOB_MAGIC, code.len() as u32, 0, 0] {
            b.extend(v.to_le_bytes());
        }
        b.extend(0x1234u64.to_le_bytes());
        b.extend(code);
        b
    }

    /// A zip of `entries` stored as they are, as `DataZip` reads one: local headers, the central
    /// directory, its end record.
    fn zip(entries: &[(String, Vec<u8>)]) -> Vec<u8> {
        let (mut out, mut dir) = (Vec::new(), Vec::new());
        for (name, data) in entries {
            let (at, len, name_len) = (out.len() as u32, data.len() as u32, name.len() as u16);
            out.extend(0x0403_4b50u32.to_le_bytes());
            out.extend([20, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]); // version, flags, method, time, date, crc
            out.extend(len.to_le_bytes());
            out.extend(len.to_le_bytes());
            out.extend(name_len.to_le_bytes());
            out.extend([0, 0]);
            out.extend(name.as_bytes());
            out.extend(data);
            dir.extend(0x0201_4b50u32.to_le_bytes());
            dir.extend([20, 0, 20, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]); // versions, flags, method, time, date, crc
            dir.extend(len.to_le_bytes());
            dir.extend(len.to_le_bytes());
            dir.extend(name_len.to_le_bytes());
            dir.extend([0; 12]); // extra, comment, disk, attributes
            dir.extend(at.to_le_bytes());
            dir.extend(name.as_bytes());
        }
        let start = out.len() as u32;
        out.extend(&dir);
        out.extend(0x0605_4b50u32.to_le_bytes());
        out.extend([0, 0, 0, 0]);
        out.extend((entries.len() as u16).to_le_bytes());
        out.extend((entries.len() as u16).to_le_bytes());
        out.extend((dir.len() as u32).to_le_bytes());
        out.extend(start.to_le_bytes());
        out.extend([0, 0]);
        out
    }

    /// A data.zip of the translator's sample pixel programs is precompiled into libraries; a
    /// second run, nothing changed, says so in less than a second, without a device; a vertex
    /// program play put in the shader cache since is made alone, then nothing again; a data.zip
    /// changed makes the libraries again, found in the cache then. Needs a GPU with pipeline
    /// libraries linked fast.
    #[test]
    fn precompiling_makes_the_libraries_and_says_when_nothing_changed() {
        let root = std::env::current_exe().unwrap().parent().unwrap().join(format!("precompile-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&root);
        std::fs::create_dir_all(&root).unwrap();
        let mut entries: Vec<(String, Vec<u8>)> = shader::tests::samples().into_iter().filter(|(code, _)| sass::stage(code) == Stage::Pixel).enumerate().map(|(i, (code, _))| (format!("{i:016x}"), blob(&code))).collect();
        assert_eq!(entries.len(), 2, "the sample pixel programs");
        let (zip_path, cache) = (root.join("data.zip"), root.join("cache"));
        std::fs::write(&zip_path, zip(&entries)).unwrap();
        let mut ticks = Vec::new();
        let first = match run(&zip_path, &cache, &mut |done, total| ticks.push((done, total))) {
            Ok(r) if r.starts_with("not used") => {
                eprintln!("{r}: test skipped");
                return;
            }
            Ok(r) => r,
            Err(e) => {
                eprintln!("{e}: test skipped");
                return;
            }
        };
        assert!(first.starts_with("2 libraries made"), "{first}");
        assert!(first.contains("0 failed, 0 left out, 0 programs that do not translate"), "{first}");
        assert_eq!(ticks.last(), Some(&(2, 2)));
        let since = std::time::Instant::now();
        assert_eq!(run(&zip_path, &cache, &mut |_, _| {}).unwrap(), "up to date");
        assert!(since.elapsed() < std::time::Duration::from_secs(1), "{:?}", since.elapsed());
        // Past play: a vertex program's entry in the shader cache, as the game writes them.
        let (code, options) = shader::tests::samples().into_iter().find(|(code, _)| sass::stage(code) == Stage::Vertex).unwrap();
        let t = shader::translate_with(&code, &options).unwrap();
        let mut played = shader::cache::ShaderCache::open(&cache.join("shaders")).unwrap();
        played.store(t.key, &code, options, t, std::time::Duration::ZERO);
        played.finish(std::time::Duration::from_secs(10));
        let vertex = run(&zip_path, &cache, &mut |_, _| {}).unwrap();
        assert!(vertex.starts_with("1 library made") || vertex.starts_with("2 libraries made"), "{vertex}");
        assert_eq!(run(&zip_path, &cache, &mut |_, _| {}).unwrap(), "up to date");
        entries.push(("ffffffffffffffff".into(), b"not a blob".to_vec()));
        std::fs::write(&zip_path, zip(&entries)).unwrap();
        // The pixel programs and the vertex one made again, every one found in the cache.
        let again = run(&zip_path, &cache, &mut |_, _| {}).unwrap();
        assert!(again.contains(" made in ") && again.contains("(0 compiled, "), "{again}");
        let _ = std::fs::remove_dir_all(&root);
    }
}
