//! Draws on wgpu: a 3D-class draw (`pushbuf::Draw`) becomes a render pass
//! into its targets. The game's programs are translated to WGSL; vertices,
//! indices and constant buffers are read from guest memory at the draw;
//! textures come from a render target at the same address, else from
//! guest memory.
//!
//! ponytail: one encoder and fresh buffers per draw (ordering for free; a
//! ring buffer when draw counts make it slow); textures cached by address
//! and header, dropped when the GPU writes their memory (not when the CPU
//! does); float vertex attributes; depth-compare textures only from depth
//! targets; pipelines keyed by a Debug string. Each draw not done counts its reason
//! in `Surfaces::skipped`.

use crate::pushbuf::{Draw, Surface};
use crate::sass::texture::TexDim;
use crate::shader::{self, Handle, Options, Translation};
use crate::state::resources::{Pool, SamplerBinding, TexHandle};
use crate::state::textures::{self, Layout, TextureType};
use crate::state::vertex::{AttrKind, wgpu_vertex_format};
use crate::surfaces::{FxMap, Surfaces, color_format, depth_format};
use std::collections::{BTreeMap, BTreeSet};

/// Reads guest memory at a GPU address; false if a byte is not readable.
pub type Mem<'a> = &'a mut dyn FnMut(u64, &mut [u8]) -> bool;

/// A constant bank as read: its guest address and bytes (shared by the draws of a submit).
pub(crate) type Bank = (u64, std::sync::Arc<Vec<u8>>);

/// What a draw's render pass is opened on: the color targets with their formats,
/// the depth target with its format, and whether colors are attached (not
/// without a pixel program). Draws with the same key share the open pass.
pub(crate) type PassKey = (Vec<Option<(Surface, wgpu::TextureFormat)>>, Option<(Surface, wgpu::TextureFormat)>, bool);

/// What the open pass was last given: a draw sets only what changed (wgpu
/// validates and encodes each call again, and consecutive draws mostly repeat).
/// Pipelines and bind groups wgpu already skips when repeated.
#[derive(Default)]
pub(crate) struct PassState {
    viewport: Option<[f32; 6]>,
    scissor: Option<[u32; 4]>,
    blend: Option<[f32; 4]>,
    stencil: Option<u32>,
    vertex: Vec<Option<crate::surfaces::Slice>>,
    index: Option<(crate::surfaces::Slice, wgpu::IndexFormat)>,
}

/// Stores `new` in `old`; whether it differed.
fn changed<T: PartialEq>(old: &mut Option<T>, new: T) -> bool {
    let differs = old.as_ref() != Some(&new);
    if differs {
        *old = Some(new);
    }
    differs
}

/// The bytes of a constant bank as the translated programs declare it.
const CBUF_BYTES: usize = 0x10000;

/// The key and code (SPH then SASS, up to its closing self-branch) of the
/// program at `gpu`.
pub fn read_program(mem: Mem, gpu: u64) -> Option<(u64, Vec<u8>)> {
    read_until(mem, gpu, crate::sass::program_key)
}

/// The key and code of the compute program at `gpu` (no SPH).
pub fn read_compute_program(mem: Mem, gpu: u64) -> Option<(u64, Vec<u8>)> {
    read_until(mem, gpu, crate::sass::compute_key)
}

fn read_until(mem: Mem, gpu: u64, key: fn(&[u8]) -> Option<(u64, usize)>) -> Option<(u64, Vec<u8>)> {
    let mut code = Vec::new();
    while code.len() < 0x4_0000 {
        // To the end of a page: never past the page that holds the closing self-branch.
        let at = gpu + code.len() as u64;
        let mut chunk = vec![0u8; (0x1000 - (at & 0xfff)) as usize];
        if !mem(at, &mut chunk) {
            return None;
        }
        code.extend(chunk);
        if let Some((key, len)) = key(&code) {
            code.truncate(len);
            return Some((key, code));
        }
    }
    None
}

/// Reads `out` from `addr` page by page: the bytes readable before the first page that is not.
pub(crate) fn read_pages(mem: Mem, addr: u64, out: &mut [u8]) -> usize {
    let mut done = 0;
    while done < out.len() {
        let at = addr + done as u64;
        let n = ((0x1000 - (at & 0xfff)) as usize).min(out.len() - done);
        if !mem(at, &mut out[done..done + n]) {
            break;
        }
        done += n;
    }
    done
}

/// A translation and its module, shared: draws take a reference count, not a copy
/// of the WGSL (some programs are 300 KB of text, drawn a dozen times a frame).
type Module = Result<std::sync::Arc<(Translation, wgpu::ShaderModule)>, String>;

/// Work for the pipeline worker threads (play mode, D42).
pub(crate) type Job = Box<dyn FnOnce() + Send>;

/// How many pipelines are made at once off the GPU thread: one that never
/// returns (the driver once worked on one for minutes) holds its thread only.
const WORKERS: usize = 3;

/// A pipeline, or one a worker thread is making: the worker sets the cell;
/// since when, its label (`p<program>+<pixel program>`), and whether it was
/// reported as slow.
pub(crate) enum Made<T> {
    Ready(T),
    Pending(std::sync::Arc<std::sync::OnceLock<T>>, std::time::Instant, String, bool),
}

impl<T: Clone> Made<T> {
    /// The pipeline, once made (a pending one done since becomes ready).
    pub(crate) fn ready(&mut self) -> Option<T> {
        if let Made::Pending(cell, ..) = self
            && let Some(p) = cell.get()
        {
            *self = Made::Ready(p.clone());
        }
        match self {
            Made::Ready(p) => Some(p.clone()),
            Made::Pending(..) => None,
        }
    }

    /// The pipeline, waited for until `until` at most: one the driver has in
    /// its cache is made in a few milliseconds, sooner than a frame without it.
    /// ponytail: polls every 200 µs; a condition variable if that shows up.
    pub(crate) fn wait(&mut self, until: std::time::Instant) -> Option<T> {
        loop {
            if let Some(p) = self.ready() {
                return Some(p);
            }
            if std::time::Instant::now() >= until {
                return None;
            }
            std::thread::sleep(std::time::Duration::from_micros(200));
        }
    }
}

/// The worker threads and their queue. The SPIR-V of a program is written
/// inside the pipeline call (wgpu-hal): their stacks are the size of the
/// guest threads', which made pipelines until then.
fn pipeline_workers() -> std::sync::mpsc::Sender<Job> {
    let (tx, rx) = std::sync::mpsc::channel::<Job>();
    let rx = std::sync::Arc::new(std::sync::Mutex::new(rx));
    for i in 0..WORKERS {
        let rx = rx.clone();
        let spawned = std::thread::Builder::new().name(format!("pipelines {i}")).stack_size(16 << 20).spawn(move || {
            loop {
                // A statement of its own: the queue's lock is released before the job runs.
                let job = rx.lock().unwrap_or_else(std::sync::PoisonError::into_inner).recv();
                match job {
                    Ok(job) => job(),
                    Err(_) => return,
                }
            }
        });
        if let Err(e) = spawned {
            eprintln!("gyroship: no pipeline worker thread: {e}");
        }
    }
    tx
}

/// The caches below are hashed with `Fx`: SipHash cost more than the lookups,
/// every draw hashing a pipeline key, texture and sampler headers.
#[derive(Default)]
pub struct Cache {
    /// Program key by GPU address, and code by key.
    programs: FxMap<u64, u64>,
    pub(crate) code: FxMap<u64, Vec<u8>>,
    /// Compute program keys by GPU address, compiled compute programs, and the
    /// images compute programs wrote, by GPU address (see `compute.rs`).
    pub(crate) compute_programs: FxMap<u64, u64>,
    pub(crate) compute: FxMap<String, crate::compute::Compiled>,
    pub(crate) compute_pipelines: FxMap<String, (wgpu::ComputePipeline, wgpu::BindGroupLayout)>,
    pub(crate) images: FxMap<u64, wgpu::Texture>,
    /// Translations by (program key, key of the pixel program it pairs with,
    /// mask of the attributes fetched as integers, hash of the jump targets or
    /// 0, user clip planes enabled).
    modules: FxMap<ModuleId, Module>,
    pipelines: FxMap<PipelineKey, (Made<wgpu::RenderPipeline>, Vec<wgpu::BindGroupLayout>)>,
    /// Bind group layouts by their entries, shared by the pipelines.
    layouts: FxMap<Vec<wgpu::BindGroupLayoutEntry>, wgpu::BindGroupLayout>,
    /// Uploaded textures by address and header, with the end of the memory they came from.
    textures: FxMap<(u64, [u8; 32]), (wgpu::Texture, u64)>,
    /// Samplers by header, filtering and compare.
    samplers: FxMap<([u8; 32], bool, bool), wgpu::Sampler>,
    /// Stand-ins for textures only queried, by width, height, depth and levels.
    queried: FxMap<(u32, u32, u32, u32), wgpu::Texture>,
    /// The pass that writes R32Float texels as depth (ZF32 uploads).
    depth_copy: Option<wgpu::RenderPipeline>,
    /// Float twins of integer compute images, by address, until the next dispatch.
    pub(crate) twins: FxMap<u64, wgpu::Texture>,
    /// Color textures from memory read with a depth compare: their R channel as
    /// depth, by address and header; forgotten with any invalidation.
    compare_depths: FxMap<(u64, [u8; 32]), wgpu::Texture>,
    /// Per program (key, paired pixel key, integer mask): the jump tables' bank
    /// bytes last seen and the module variant (id.3) they gave; the same bytes
    /// give the same variant without scanning them again.
    jumps: JumpMemo,
    /// GPU writes to guest memory not yet applied to `textures`, as (start, end):
    /// applied once before the next read of the cache (`drain_invalidations`).
    pending: Vec<(u64, u64)>,
    /// `textures`' ranges as [`spans`] indexes them; None once `textures` changed.
    spans: Option<Vec<(u64, u64)>>,
    /// Color targets read through a smaller texture header, cropped to it, by
    /// (address, format, width, height), with the target's write stamp when copied.
    crops: FxMap<(u64, u32, u32, u32), (u64, wgpu::Texture)>,
}

/// How far a jump table of unknown length is read: 256 entries.
const TABLE_MAX: usize = 1024;

/// The bytes of each jump table, in table order, from its offset in its bank as
/// bound: its entries when known, else up to TABLE_MAX bytes. None if a bank is
/// unbound or unreadable. The rest of the bank (other constants, which may
/// change every frame) never reaches the targets: a program keeps one variant.
fn jump_banks(tables: &[shader::JumpTable], banks: &[Option<(u64, u32)>], mem: Mem) -> Option<Vec<Vec<u8>>> {
    let mut out = Vec::with_capacity(tables.len());
    for &(bank, _, offset, entries) in tables {
        let (addr, size) = (*banks.get(usize::from(bank))?)?;
        let start = usize::from(offset).min(size as usize);
        let len = entries.map_or(TABLE_MAX, |n| 4 * n as usize).min((size as usize).min(CBUF_BYTES).saturating_sub(start));
        let mut data = vec![0u8; len];
        if !mem(addr + start as u64, &mut data) {
            return None;
        }
        out.push(data);
    }
    Some(out)
}

/// The instructions a program's jump tables may send it to: each entry that,
/// added to the jump's base, is an instruction's byte offset (a 16-byte multiple
/// inside the program). A table of unknown length ends at its first word that is
/// not one.
fn jump_targets(tables: &[shader::JumpTable], bytes: &[Vec<u8>], code_len: usize) -> BTreeSet<usize> {
    let mut targets = BTreeSet::new();
    for (&(_, base, _, entries), data) in tables.iter().zip(bytes) {
        for w in data.as_chunks::<4>().0 {
            let at = u32::from_le_bytes(*w).wrapping_add(base);
            if at % 16 == 0 && (at as usize) < code_len {
                targets.insert(at as usize / 16);
            } else if entries.is_none() {
                break;
            }
        }
    }
    targets
}

/// `ranges` (start, end) sorted and merged where they touch or overlap.
fn merged(ranges: &[(u64, u64)]) -> Vec<(u64, u64)> {
    let mut sorted = ranges.to_vec();
    sorted.sort_unstable();
    let mut out: Vec<(u64, u64)> = Vec::with_capacity(sorted.len());
    for (start, end) in sorted {
        match out.last_mut() {
            Some(last) if start <= last.1 => last.1 = last.1.max(end),
            _ => out.push((start, end)),
        }
    }
    out
}

/// Whether `[a, b)` overlaps one of the sorted, merged `ranges` (the test
/// `invalidate` applied range by range: `b > start && a < end`).
fn overlaps(ranges: &[(u64, u64)], a: u64, b: u64) -> bool {
    let i = ranges.partition_point(|r| r.1 <= a);
    ranges.get(i).is_some_and(|r| r.0 < b)
}

/// Ranges [a, b) sorted by start, each end replaced by the furthest end so far.
fn spans(ranges: impl Iterator<Item = (u64, u64)>) -> Vec<(u64, u64)> {
    let mut s: Vec<(u64, u64)> = ranges.collect();
    s.sort_unstable();
    s.iter_mut().fold(0, |max, r| {
        r.1 = r.1.max(max);
        r.1
    });
    s
}

/// Whether [s, e) overlaps one of the ranges `spans` indexed (as [`overlaps`]: a < e && b > s).
fn hits(spans: &[(u64, u64)], s: u64, e: u64) -> bool {
    let i = spans.partition_point(|r| r.0 < e);
    i > 0 && spans[i - 1].1 > s
}

/// Which bind groups (0 vertex, 1 pixel) take their uniform buffers at dynamic
/// offsets, given each group's uniform count and the device's limit for a pipeline:
/// all when they fit; over the limit, the pixel group first (it holds the textures,
/// dozens of entries made again otherwise), else the vertex group alone.
fn dynamic_groups(uniforms: &[usize], limit: usize) -> Vec<bool> {
    match uniforms {
        [v, p] if v + p > limit => vec![*p > limit && *v <= limit, *p <= limit],
        all => vec![all.iter().sum::<usize>() <= limit; all.len()],
    }
}

/// Texels of `slices` slices of `from` (columns, rows) elements of `bpe` bytes,
/// laid out as `to` (columns, rows): what fits is copied, the rest is zero.
fn repack(src: &[u8], from: (u32, u32), to: (u32, u32), slices: u32, bpe: u32) -> Vec<u8> {
    let (fc, fr, tc, tr, bpe) = (from.0 as usize, from.1 as usize, to.0 as usize, to.1 as usize, bpe as usize);
    let mut out = vec![0u8; tc * tr * slices as usize * bpe];
    let n = fc.min(tc) * bpe;
    for s in 0..slices as usize {
        for r in 0..fr.min(tr) {
            let at = ((s * fr + r) * fc) * bpe;
            if let Some(row) = src.get(at..at + n) {
                let to = ((s * tr + r) * tc) * bpe;
                out[to..to + n].copy_from_slice(row);
            }
        }
    }
    out
}

/// An index of `size` little-endian bytes.
fn index_value(c: &[u8]) -> u32 {
    c.iter().rev().fold(0, |v, b| v << 8 | u32::from(*b))
}

/// The largest of the little-endian indices in `raw` (`size` bytes each), the
/// restart value excluded when there is one; 0 when none is left. The restart
/// value counts as 0, which gives the same maximum without a branch per index.
fn index_max(raw: &[u8], size: usize, restart: Option<u32>) -> u32 {
    let r = restart.unwrap_or(u32::MAX);
    let skip = |i: u32| if restart.is_some() && i == r { 0 } else { i };
    match size {
        4 => raw.as_chunks::<4>().0.iter().map(|b| skip(u32::from_le_bytes(*b))).max(),
        2 => raw.as_chunks::<2>().0.iter().map(|b| skip(u32::from(u16::from_le_bytes(*b)))).max(),
        1 => raw.iter().map(|&b| skip(u32::from(b))).max(),
        _ => raw.chunks_exact(size).map(|c| skip(index_value(c))).max(),
    }
    .unwrap_or(0)
}

/// The float format with the bits of a 32-bit integer format.
fn float_format(f: wgpu::TextureFormat) -> Option<wgpu::TextureFormat> {
    use wgpu::TextureFormat as F;
    Some(match f {
        F::R32Uint | F::R32Sint => F::R32Float,
        F::Rg32Uint | F::Rg32Sint => F::Rg32Float,
        F::Rgba32Uint | F::Rgba32Sint => F::Rgba32Float,
        _ => return None,
    })
}

/// Per program (key, paired pixel key, integer mask): jump-table bank bytes and the
/// module variant they gave.
type JumpMemo = FxMap<(u64, Option<u64>, u64, u8), (Vec<Vec<u8>>, u64)>;

/// A module's key in the cache: see `Cache::modules`.
type ModuleId = (u64, Option<u64>, u64, u64, u8);

/// What a render pipeline is made from, as a hashable key: the vertex program
/// (key, WGSL length), the pixel program's, the vertex buffer layouts (stream,
/// step, stride, attributes), the primitive, color and depth states, and the
/// bind group layouts. The WGSL lengths tell a program's module with exact
/// jump-table blocks from its first one.
type PipelineKey = (
    (u64, usize),
    Option<(u64, usize)>,
    Vec<(u8, wgpu::VertexStepMode, u64, Vec<wgpu::VertexAttribute>)>,
    wgpu::PrimitiveState,
    Vec<Option<wgpu::ColorTargetState>>,
    Option<wgpu::DepthStencilState>,
    Vec<Vec<wgpu::BindGroupLayoutEntry>>,
);

/// A stage's bindings.
pub(crate) struct Bindings<'t> {
    /// Constant buffers by bank.
    pub(crate) cbufs: BTreeMap<u8, Bank>,
    pub(crate) textures: Vec<Slot>,
    /// Each slot's header swizzle, when the program reads them.
    pub(crate) swizzles: Option<Vec<u32>>,
    /// The program's `Interface::cbuf_sizes`, the bytes of each bank it may
    /// read (all 64 KiB of a bank it has no entry for): staged as bound, then
    /// zeros up to there, as the console reads past a bank's size. The binding
    /// spans 64 KiB; what lies past these bytes is never read.
    pub(crate) sizes: &'t BTreeMap<u8, u32>,
}

/// A texture slot: its view, its sampler unless fetched, and whether it is
/// a depth target (read as unfilterable floats, with a non-filtering sampler)
/// or a depth target sampled with a comparison (depth texture and sampler).
pub(crate) struct Slot {
    pub(crate) view: wgpu::TextureView,
    pub(crate) sampler: Option<wgpu::Sampler>,
    depth: bool,
    compare: bool,
    dimension: wgpu::TextureViewDimension,
}

impl Bindings<'_> {
    /// The bytes of bank `bank` the program may read.
    pub(crate) fn window(&self, bank: u8) -> u64 {
        self.sizes.get(&bank).map_or(CBUF_BYTES as u64, |&s| u64::from(s))
    }

    /// The bank's bytes to stage (what the program may read of them) and the
    /// bytes that must read as the bank then zeros.
    pub(crate) fn staged(&self, bank: u8, data: &[u8]) -> (usize, usize) {
        let window = self.window(bank) as usize;
        (data.len().min(window), window)
    }

    /// The layout these bindings take in a stage.
    pub(crate) fn layout(&self, visibility: wgpu::ShaderStages) -> Vec<wgpu::BindGroupLayoutEntry> {
        let buffer = |binding: u32, size: u64| wgpu::BindGroupLayoutEntry {
            binding,
            visibility,
            ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Uniform, has_dynamic_offset: false, min_binding_size: wgpu::BufferSize::new(size) },
            count: None,
        };
        let mut entries: Vec<_> = self.cbufs.keys().map(|bank| buffer(u32::from(*bank), CBUF_BYTES as u64)).collect();
        if let Some(codes) = &self.swizzles {
            entries.push(buffer(shader::SWIZZLE_BINDING, 4 * codes.len() as u64));
        }
        for (i, s) in self.textures.iter().enumerate() {
            let filterable = s.sampler.is_some() && !s.depth;
            let sample_type = if s.compare { wgpu::TextureSampleType::Depth } else { wgpu::TextureSampleType::Float { filterable } };
            entries.push(wgpu::BindGroupLayoutEntry { binding: 32 + 2 * i as u32, visibility, ty: wgpu::BindingType::Texture { sample_type, view_dimension: s.dimension, multisampled: false }, count: None });
            if s.sampler.is_some() {
                let ty = match (s.compare, s.depth) {
                    (true, _) => wgpu::SamplerBindingType::Comparison,
                    (false, true) => wgpu::SamplerBindingType::NonFiltering,
                    (false, false) => wgpu::SamplerBindingType::Filtering,
                };
                entries.push(wgpu::BindGroupLayoutEntry { binding: 33 + 2 * i as u32, visibility, ty: wgpu::BindingType::Sampler(ty), count: None });
            }
        }
        entries
    }
}

impl<T> Made<T> {
    /// Its label, the first time it is found still being made `after` it was asked for.
    fn overdue(&mut self, after: std::time::Duration) -> Option<String> {
        match self {
            Made::Pending(cell, since, label, reported) if !*reported && cell.get().is_none() && since.elapsed() >= after => {
                *reported = true;
                Some(label.clone())
            }
            _ => None,
        }
    }
}

impl Surfaces {
    /// A pipeline `make` makes: at once in the deterministic engine; on a
    /// worker thread in play mode (D42), pending until it is done.
    pub(crate) fn make<T: Send + Sync + 'static>(&mut self, make: impl FnOnce() -> T + Send + 'static, label: String) -> Made<T> {
        if !self.background {
            return Made::Ready(make());
        }
        let cell = std::sync::Arc::new(std::sync::OnceLock::new());
        let (done, finished, name) = (cell.clone(), self.finished.clone(), label.clone());
        let workers = self.workers.get_or_insert_with(pipeline_workers);
        let _ = workers.send(Box::new(move || {
            let since = std::time::Instant::now();
            let _ = done.set(make());
            if let Ok(mut f) = finished.lock() {
                f.push((name, since.elapsed()));
            }
        }));
        Made::Pending(cell, std::time::Instant::now(), label, false)
    }

    /// The pipelines still being made `after` they were asked for, each named
    /// once (play mode: the driver may take seconds, or never return).
    pub fn overdue(&mut self, after: std::time::Duration) -> Vec<String> {
        self.cache.pipelines.values_mut().filter_map(|(m, _)| m.overdue(after)).collect()
    }

    /// The guest memory each uploaded texture came from, as GPU address ranges
    /// (start, end): the HLE forgets those the CPU has written since.
    pub fn texture_ranges(&self) -> Vec<(u64, u64)> {
        let written = merged(&self.cache.pending);
        self.cache.textures.iter().map(|((addr, _), (_, end))| (*addr, *end)).filter(|&(a, b)| !overlaps(&written, a, b)).collect()
    }

    /// What the host keeps, by kind, for the summary line (growth shows a leak).
    pub fn cache_sizes(&self) -> String {
        let c = &self.cache;
        let written = merged(&c.pending);
        format!(
            "targets {}, textures {}, samplers {}, programs {}, modules {}, pipelines {}, compute pipelines {}, images {}",
            self.map.len(),
            c.textures.keys().filter(|(addr, header)| !overlaps(&written, *addr, c.textures[&(*addr, *header)].1)).count(),
            c.samplers.len(),
            c.code.len(),
            c.modules.len(),
            c.pipelines.len(),
            c.compute_pipelines.len(),
            c.images.len()
        )
    }

    /// Runs a draw; one that cannot be done is counted in `skipped` by
    /// reason, and the reason returned.
    pub fn draw(&mut self, d: &Draw, mem: Mem) -> Result<(), String> {
        self.textures_read.clear();
        self.drain_invalidations();
        let result = self.try_draw(d, mem);
        match &result {
            Ok(()) => self.drawn += 1,
            Err(why) => self.skip(why.clone()),
        }
        result
    }

    fn try_draw(&mut self, d: &Draw, mem: Mem) -> Result<(), String> {
        use wgpu::PrimitiveTopology as P;
        let topology = match d.topology {
            0 => P::PointList,
            1 => P::LineList,
            3 => P::LineStrip,
            4 => P::TriangleList,
            5 => P::TriangleStrip,
            t => return Err(format!("topology {t}")),
        };
        if d.count == 0 {
            return Ok(());
        }
        if d.programs[0].is_some() || d.programs[2..5].iter().any(Option::is_some) {
            return Err("vertex-A, tessellation or geometry program".into());
        }
        let ps = match d.programs[5] {
            Some(at) => Some(self.module(mem, at, None, &BTreeMap::new(), &d.cbufs[4], 0)?),
            None => None,
        };
        // Attributes fetched as integers: the vertex program declares them so.
        let ints: BTreeMap<u8, bool> = d.attributes.iter().filter(|a| !a.constant && matches!(a.format.kind, AttrKind::Uint | AttrKind::Sint)).map(|a| (a.location, a.format.kind == AttrKind::Sint)).collect();
        // The user clip planes the draw enables, where the device can clip (else none).
        let clip = if self.clip_distances { d.render.user_clip } else { 0 };
        let vs = self.module(mem, d.programs[1].ok_or("no vertex program")?, ps.as_ref().map(|p| &p.0), &ints, &d.cbufs[0], clip)?;

        // Attachments: the locations the pixel program writes, and depth when tested.
        let none = BTreeMap::new();
        let written = ps.as_ref().map_or(&none, |p| &p.0.interface.targets);
        let mut colors: Vec<Option<(Surface, wgpu::TextureFormat)>> = Vec::new();
        for (location, s) in d.targets.iter().enumerate() {
            let used = written.contains_key(&(location as u8)) && d.render.outputs.get(location).is_some_and(|o| o.write_mask != 0);
            colors.push(match used {
                false => None,
                true => Some((*s, color_format(s.format).filter(|f| self.renderable(*f)).ok_or_else(|| format!("color target format {:#x}", s.format))?)),
            });
        }
        let ds = d.render.depth_stencil;
        let depth = match d.depth {
            Some(s) if ds.depth_test || ds.stencil_test => Some((s, depth_format(s.format).filter(|f| self.renderable(*f)).ok_or_else(|| format!("depth target format {:#x}", s.format))?)),
            _ => None,
        };
        let sizes: BTreeSet<(u32, u32)> = colors.iter().flatten().chain(&depth).map(|(s, _)| (s.width, s.height)).collect();
        let (width, height) = match sizes.len() {
            0 => return Err("no attachment".into()),
            1 => sizes.into_iter().next().unwrap_or_default(),
            _ => return Err("attachments of different sizes".into()),
        };

        let v = d.render.viewport.to_wgpu()?;
        if v.negate_y || v.remap_z {
            return Err("viewport needs a y flip or z remap in the vertex program".into());
        }
        let [x, y, w, h] = d.render.wgpu_scissor()?;
        let (x1, y1) = ((x + w).min(width), (y + h).min(height));
        if x >= x1 || y >= y1 {
            return Ok(());
        }

        // Vertex input, by stream.
        let mut layouts: Vec<(u8, wgpu::VertexStepMode, u64, Vec<wgpu::VertexAttribute>)> = Vec::new();
        for &location in vs.0.interface.attributes.keys() {
            let a = d.attributes.iter().find(|a| a.location == location).ok_or_else(|| format!("vertex attribute {location} not programmed"))?;
            if a.constant {
                return Err("constant vertex attribute".into());
            }
            let format = wgpu_vertex_format(&a.format).ok_or_else(|| format!("vertex attribute format {:#x}", a.raw))?;
            let s = d.streams.iter().find(|s| s.index == a.stream).ok_or("vertex attribute on a disabled stream")?;
            let attribute = wgpu::VertexAttribute { format, offset: u64::from(a.offset), shader_location: u32::from(location) };
            // An instance stream: one element, that of the instance drawn, read by every
            // instance (stride 0); an instanced draw comes as one draw per instance.
            match layouts.iter_mut().find(|l| l.0 == s.index) {
                Some(l) => l.3.push(attribute),
                None if s.per_instance => layouts.push((s.index, wgpu::VertexStepMode::Instance, 0, vec![attribute])),
                None => layouts.push((s.index, wgpu::VertexStepMode::Vertex, u64::from(s.stride), vec![attribute])),
            }
        }

        // Indices (8-bit ones widened: wgpu has none) and the last vertex they reach.
        let (indices, last) = match &d.index {
            Some(ib) => {
                let size = usize::from(ib.bytes_per_index);
                if size == 0 {
                    return Err("index size".into());
                }
                let addr = ib.address + u64::from(ib.first) * size as u64;
                let restart = ib.restart_enable.then_some(ib.restart_index);
                // Staged once a submit: the same indices serve many draws.
                let (slice, format, max) = match self.staged_indices(addr, size, d.count, restart) {
                    Some(found) => found,
                    None => {
                        let len = size * d.count as usize;
                        let found = if let 2 | 4 = size {
                            // Read straight into the arena: wgpu takes these as they are.
                            let slice = self.stage_read(addr, len, mem).ok_or("index buffer not readable")?;
                            let max = index_max(self.arena_bytes(&slice, len), size, restart);
                            (slice, if size == 4 { wgpu::IndexFormat::Uint32 } else { wgpu::IndexFormat::Uint16 }, max)
                        } else {
                            let mut raw = vec![0u8; len];
                            if !mem(addr, &mut raw) {
                                return Err("index buffer not readable".into());
                            }
                            let max = index_max(&raw, size, restart);
                            let bytes: Vec<u8> = if size == 1 { raw.iter().flat_map(|&b| u16::from(b).to_le_bytes()).collect() } else { raw.chunks_exact(size).flat_map(|c| (index_value(c) as u16).to_le_bytes()).collect() };
                            (self.stage(&bytes), wgpu::IndexFormat::Uint16, max)
                        };
                        self.remember_indices(addr, size, d.count, restart, found.clone());
                        found
                    }
                };
                (Some((slice, format)), (i64::from(d.base_vertex) + i64::from(max)).max(0) as u64)
            }
            None => (None, u64::from(d.first) + u64::from(d.count) - 1),
        };
        let mut vertex_buffers = Vec::new();
        for (index, step, stride, attributes) in &layouts {
            let s = d.streams.iter().find(|s| s.index == *index).ok_or("stream")?;
            let per_instance = *step == wgpu::VertexStepMode::Instance;
            // The instance's element (instance / divisor; a divisor of 0 keeps element 0, as
            // Vulkan's, which NVK writes unchanged).
            let element = if per_instance && s.divisor != 0 { u64::from(d.instance / s.divisor) } else { 0 };
            let start = s.address + element * u64::from(s.stride);
            let elements = if per_instance { 1 } else { last + 1 };
            let reach = attributes.iter().map(|a| a.offset + a.format.size()).max().unwrap_or(0);
            let bytes = (elements - 1) * stride + reach;
            if (element > 0 && start > s.end) || s.end.wrapping_sub(start) < bytes - 1 || bytes > 64 << 20 {
                return Err("vertex stream shorter than the draw".into());
            }
            let padded = bytes.next_multiple_of(4) as usize;
            let slice = match self.staged_guest(crate::surfaces::Read::Vertex, start, padded, 0, 0) {
                Some(slice) => slice,
                None => {
                    let slice = self.stage_read(start, bytes as usize, mem).ok_or("vertex stream not readable")?;
                    self.remember_staged(crate::surfaces::Read::Vertex, start, padded, slice.clone());
                    slice
                }
            };
            vertex_buffers.push(slice);
        }

        // Constant buffers and textures, per stage (bind group 0 vertex, 1 pixel).
        let attached: Vec<u64> = colors.iter().flatten().chain(&depth).map(|(s, _)| s.addr).collect();
        let mut stages = vec![self.stage_bindings(d, 0, &vs.0, &attached, mem)?];
        if let Some(p) = &ps {
            stages.push(self.stage_bindings(d, 4, &p.0, &attached, mem)?);
        }

        // The pipeline.
        let r = &d.render;
        let color_states: Vec<Option<wgpu::ColorTargetState>> = colors
            .iter()
            .enumerate()
            .map(|(location, c)| -> Result<_, String> {
                let Some((_, format)) = c else { return Ok(None) };
                let o = &r.outputs[location];
                let blend = match o.blend {
                    Some(b) => Some(wgpu::BlendState { color: b.color.wgpu().ok_or("blend equation")?, alpha: b.wgpu_alpha().wgpu().ok_or("blend equation")? }),
                    None => None,
                };
                Ok(Some(wgpu::ColorTargetState { format: *format, blend, write_mask: wgpu::ColorWrites::from_bits_truncate(u32::from(o.write_mask)) }))
            })
            .collect::<Result<_, _>>()?;
        let depth_state = match &depth {
            Some((_, format)) => {
                let face = |f: crate::state::raster::StencilFace| f.wgpu().ok_or("stencil state");
                let (front, back) = if ds.stencil_test { (face(ds.front)?, face(if ds.two_sided { ds.back } else { ds.front })?) } else { Default::default() };
                let bias = &r.raster.depth_bias;
                Some(wgpu::DepthStencilState {
                    format: *format,
                    depth_write_enabled: Some(ds.depth_test && ds.depth_write),
                    depth_compare: Some(if ds.depth_test { ds.depth_compare.wgpu().ok_or("depth function")? } else { wgpu::CompareFunction::Always }),
                    stencil: wgpu::StencilState { front, back, read_mask: u32::from(ds.front.read_mask), write_mask: u32::from(ds.front.write_mask) },
                    bias: if bias.fill { wgpu::DepthBiasState { constant: bias.constant as i32, slope_scale: bias.slope_scale, clamp: bias.clamp } } else { Default::default() },
                })
            }
            None => None,
        };
        let strip = matches!(topology, P::LineStrip | P::TriangleStrip).then(|| indices.as_ref().map(|i| i.1)).flatten();
        let primitive = wgpu::PrimitiveState {
            topology,
            strip_index_format: strip,
            front_face: r.raster.wgpu_front_face().ok_or("front face")?,
            cull_mode: r.raster.wgpu_cull()?,
            ..Default::default()
        };
        // Bind groups: 0 vertex, 1 pixel; constant banks at their bank, textures at 32 + 2i, samplers at 33 + 2i.
        // A group's uniform buffers take dynamic offsets when the device allows that many: the
        // group then serves every draw with the same layout, textures and samplers, whatever the
        // banks' place in the arena (which moves at every flush). Over the limit for the pair, the
        // pixel group (the textures: dozens of entries) keeps them and the vertex group's are
        // baked into it, so that only that small group is made again.
        let uniforms: Vec<usize> = stages.iter().map(|b| b.cbufs.len() + usize::from(b.swizzles.is_some())).collect();
        let dynamic = dynamic_groups(&uniforms, self.dynamic_max as usize);
        let visibility = [wgpu::ShaderStages::VERTEX, wgpu::ShaderStages::FRAGMENT];
        let mut group_layouts: Vec<Vec<wgpu::BindGroupLayoutEntry>> = stages.iter().zip(visibility).map(|(b, v)| b.layout(v)).collect();
        for (entries, _) in group_layouts.iter_mut().zip(&dynamic).filter(|(_, d)| **d) {
            for e in entries {
                if let wgpu::BindingType::Buffer { has_dynamic_offset, .. } = &mut e.ty {
                    *has_dynamic_offset = true;
                }
            }
        }
        let groups = group_layouts.iter().rposition(|e| !e.is_empty()).map_or(0, |g| g + 1);
        group_layouts.truncate(groups);
        let key: PipelineKey = ((vs.0.key, vs.0.wgsl.len()), ps.as_ref().map(|p| (p.0.key, p.0.wgsl.len())), layouts, primitive, color_states, depth_state, group_layouts);
        // Play mode: a pending pipeline is waited for within the frame's budget.
        let waiting = std::time::Instant::now();
        let until = waiting + if self.background { self.wait_left } else { std::time::Duration::ZERO };
        let found = self.cache.pipelines.get_mut(&key).map(|(made, layouts)| (made.wait(until), layouts.clone()));
        let (pipeline, bind_layouts) = match found {
            Some(found) => found,
            None => {
                let since = std::time::Instant::now();
                let (_, _, layouts, primitive, color_states, depth_state, group_layouts) = &key;
                // Pipelines with the same group layouts share them, and so their bind groups.
                let bind_layouts: Vec<wgpu::BindGroupLayout> = group_layouts
                    .iter()
                    .map(|entries| self.cache.layouts.entry(entries.clone()).or_insert_with(|| self.device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor { label: None, entries })).clone())
                    .collect();
                let refs: Vec<Option<&wgpu::BindGroupLayout>> = bind_layouts.iter().map(Some).collect();
                let layout = self.device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor { label: None, bind_group_layouts: &refs, ..Default::default() });
                // Owned: a worker thread may make it.
                let buffers: Vec<(u64, wgpu::VertexStepMode, Vec<wgpu::VertexAttribute>)> = layouts.iter().map(|(_, step, stride, attributes)| (*stride, *step, attributes.clone())).collect();
                let (device, vertex, pixel) = (self.device.clone(), vs.1.clone(), ps.as_ref().map(|p| p.1.clone()));
                let (primitive, targets, depth_stencil) = (*primitive, color_states.clone(), depth_state.clone());
                let cache = self.disk_cache.as_ref().map(|d| d.cache.clone());
                let make = move || {
                    let buffers: Vec<Option<wgpu::VertexBufferLayout>> = buffers.iter().map(|(stride, step, attributes)| Some(wgpu::VertexBufferLayout { array_stride: *stride, step_mode: *step, attributes })).collect();
                    device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                        label: None,
                        layout: Some(&layout),
                        vertex: wgpu::VertexState { module: &vertex, entry_point: Some("main"), compilation_options: Default::default(), buffers: &buffers },
                        fragment: pixel.as_ref().map(|module| wgpu::FragmentState { module, entry_point: Some("main"), compilation_options: Default::default(), targets: &targets }),
                        primitive,
                        depth_stencil,
                        multisample: Default::default(),
                        multiview_mask: None,
                        cache: cache.as_ref(),
                    })
                };
                let label = format!("p{:016x}+{}", vs.0.key, ps.as_ref().map_or("none".into(), |p| format!("{:016x}", p.0.key)));
                let mut made = self.make(make, label.clone());
                let ready = made.wait(until);
                self.cache.pipelines.insert(key, (made, bind_layouts.clone()));
                self.compile.pipelines += 1;
                self.compile.pipeline_time += since.elapsed();
                self.compile.made.push(label);
                (ready, bind_layouts)
            }
        };
        if self.background {
            self.wait_left = self.wait_left.saturating_sub(waiting.elapsed());
        }
        let Some(pipeline) = pipeline else {
            // Still being made on a worker thread (play mode), the frame's wait spent:
            // the frame goes on without this draw.
            self.compile.waited += 1;
            return Ok(());
        };

        let mut bind_groups = Vec::with_capacity(bind_layouts.len());
        for (g, layout) in bind_layouts.iter().enumerate() {
            let Bindings { cbufs, textures, swizzles, .. } = &stages[g];
            let mut buffers: Vec<(u32, (wgpu::Buffer, u64, u64))> = cbufs
                .iter()
                .map(|(b, (addr, data))| {
                    let (len, zeros) = stages[g].staged(*b, data);
                    (u32::from(*b), self.stage_guest(crate::surfaces::Read::Bank, *addr, &data[..len], zeros, 0x10000))
                })
                .collect();
            if let Some(codes) = swizzles {
                let bytes: Vec<u8> = codes.iter().flat_map(|c| c.to_le_bytes()).collect();
                buffers.push((shader::SWIZZLE_BINDING, self.stage(&bytes)));
            }
            // In binding order, as wgpu takes them: the banks', then the swizzles' (binding 999).
            let offsets: Vec<u32> = if dynamic[g] { buffers.iter().map(|(_, (_, at, _))| u32::try_from(*at).unwrap_or(u32::MAX)).collect() } else { Vec::new() };
            bind_groups.push((self.bind_group(layout, &buffers, textures, dynamic[g]), offsets));
        }
        let index_buffer = indices;

        // The pass: the open one when this draw writes the same targets (every
        // attachment loads and stores, so consecutive draws can share it; one
        // sampling its own attachment is refused by wgpu either way); else a new
        // one. Anything else recorded ends it (`Surfaces::encoder`). Timed draws
        // each keep a pass of their own.
        let key: PassKey = (colors.clone(), depth, ps.is_some());
        if self.timing.is_some() || self.pass.as_ref().is_none_or(|(open, ..)| *open != key) {
            let color_views: Vec<Option<wgpu::TextureView>> = colors.iter().map(|c| c.map(|(s, f)| self.texture(&s, f))).collect();
            let depth_view = depth.map(|(s, f)| (self.texture(&s, f), f));
            let timed = self.timing.as_mut().and_then(|t| t.next(format!("draw vs {:016x} ps {}", vs.0.key, ps.as_ref().map_or("none".into(), |p| format!("{:016x}", p.0.key)))));
            let mut encoder = self.encoder();
            let load = wgpu::Operations { load: wgpu::LoadOp::Load, store: wgpu::StoreOp::Store };
            let attachments: Vec<Option<wgpu::RenderPassColorAttachment>> = color_views.iter().map(|v| v.as_ref().map(|view| wgpu::RenderPassColorAttachment { view, depth_slice: None, resolve_target: None, ops: load })).collect();
            // Without a pixel program the pipeline has no color targets: neither may the pass.
            let attachments = if ps.is_none() { Vec::new() } else { attachments };
            let timestamp_writes = timed.zip(self.timing.as_ref()).map(|(i, t)| wgpu::RenderPassTimestampWrites { query_set: &t.set, beginning_of_pass_write_index: Some(i), end_of_pass_write_index: Some(i + 1) });
            let pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("draw"),
                timestamp_writes,
                color_attachments: &attachments,
                depth_stencil_attachment: depth_view.as_ref().map(|(view, f)| wgpu::RenderPassDepthStencilAttachment {
                    view,
                    depth_ops: Some(wgpu::Operations { load: wgpu::LoadOp::Load, store: wgpu::StoreOp::Store }),
                    stencil_ops: f.has_stencil_aspect().then_some(wgpu::Operations { load: wgpu::LoadOp::Load, store: wgpu::StoreOp::Store }),
                }),
                ..Default::default()
            });
            let pass = pass.forget_lifetime();
            self.record(encoder);
            self.pass = Some((key, pass, PassState::default()));
        }
        {
            let Some((_, pass, last)) = self.pass.as_mut() else { return Err("no open render pass".into()) };
            pass.set_pipeline(&pipeline);
            if changed(&mut last.viewport, [v.x, v.y, v.width, v.height, v.min_depth, v.max_depth]) {
                pass.set_viewport(v.x, v.y, v.width, v.height, v.min_depth, v.max_depth);
            }
            if changed(&mut last.scissor, [x, y, x1 - x, y1 - y]) {
                pass.set_scissor_rect(x, y, x1 - x, y1 - y);
            }
            if changed(&mut last.blend, r.blend_constant) {
                let [cr, cg, cb, ca] = r.blend_constant.map(f64::from);
                pass.set_blend_constant(wgpu::Color { r: cr, g: cg, b: cb, a: ca });
            }
            if changed(&mut last.stencil, u32::from(ds.front.reference)) {
                pass.set_stencil_reference(u32::from(ds.front.reference));
            }
            for (g, (group, offsets)) in bind_groups.iter().enumerate() {
                pass.set_bind_group(g as u32, group, offsets);
            }
            last.vertex.resize(last.vertex.len().max(vertex_buffers.len()), None);
            for (slot, b @ (buffer, at, size)) in vertex_buffers.iter().enumerate() {
                if changed(&mut last.vertex[slot], b.clone()) {
                    pass.set_vertex_buffer(slot as u32, buffer.slice(*at..at + size));
                }
            }
            match &index_buffer {
                Some((b @ (buffer, at, size), format)) => {
                    if changed(&mut last.index, (b.clone(), *format)) {
                        pass.set_index_buffer(buffer.slice(*at..at + size), *format);
                    }
                    pass.draw_indexed(0..d.count, d.base_vertex, d.instance..d.instance + 1);
                }
                None => pass.draw(d.first..d.first + d.count, d.instance..d.instance + 1),
            }
        }
        for (s, _) in colors.iter().flatten() {
            self.wrote((s.addr, s.format));
        }
        if let Some((s, _)) = &depth
            && (ds.depth_write || ds.stencil_test)
        {
            self.mark((s.addr, s.format));
        }
        Ok(())
    }

    /// The translation and WGSL module of the program at `at`; a vertex
    /// program pairs with its pixel program's varyings.
    fn module(&mut self, mem: Mem, at: u64, pixel: Option<&Translation>, ints: &BTreeMap<u8, bool>, banks: &[Option<(u64, u32)>], clip: u8) -> Module {
        // A failed read is not kept: play mode may read while guest threads map memory.
        let key = match self.cache.programs.get(&at) {
            Some(&key) => key,
            None => {
                let (key, code) = read_program(mem, at).ok_or_else(|| format!("program at {at:#x} not readable"))?;
                self.cache.code.insert(key, code);
                self.cache.programs.insert(at, key);
                key
            }
        };
        let pair = pixel.map(|p| p.key);
        let mask = ints.iter().fold(0u64, |m, (&l, &signed)| m | 1 << (l % 32) | u64::from(signed) << (32 + l % 32));
        // Jump tables: the translation without them (variant 0) says where they are;
        // the tables' words give the targets, and the module with exact blocks is
        // keyed by them too. Both are made in one call: variant 0 (every instruction
        // a block, a switch of thousands of cases the driver took minutes over)
        // serves only while a table's bank cannot be read.
        let mut id = (key, pair, mask, 0, clip);
        if !self.cache.modules.contains_key(&id) {
            self.translate_module(id, pixel, ints, None);
        }
        if let Some(Ok(m)) = self.cache.modules.get(&id)
            && let Some(tables) = m.0.interface.jump_tables.clone()
            && let Some(bytes) = jump_banks(&tables, banks, mem)
        {
            match self.cache.jumps.get(&(key, pair, mask, clip)) {
                // The variant these bytes gave: its module is in the cache since.
                Some((seen, variant)) if *seen == bytes => id.3 = *variant,
                _ => {
                    let targets = jump_targets(&tables, &bytes, self.cache.code[&key].len());
                    id.3 = {
                        use std::hash::{Hash, Hasher};
                        let mut h = std::collections::hash_map::DefaultHasher::new();
                        targets.hash(&mut h);
                        h.finish() | 1
                    };
                    self.cache.jumps.insert((key, pair, mask, clip), (bytes, id.3));
                    if !self.cache.modules.contains_key(&id) {
                        self.translate_module(id, pixel, ints, Some(targets));
                    }
                }
            }
        }
        self.cache.modules[&id].clone()
    }

    /// Translates program `id.0` (paired with `pixel`, attributes `ints`, the
    /// jump targets of variant `id.3`) into the module cache.
    fn translate_module(&mut self, id: ModuleId, pixel: Option<&Translation>, ints: &BTreeMap<u8, bool>, jump: Option<BTreeSet<usize>>) {
        let key = id.0;
        let since = std::time::Instant::now();
        let options = Options { varyings: pixel.map(|p| p.interface.varyings.clone()).unwrap_or_default(), int_attributes: ints.clone(), jump_targets: jump, clip_planes: id.4, ..Default::default() };
        let made = shader::translate_with(&self.cache.code[&key], &options).map_err(|e| format!("program {key:016x}: {e}")).and_then(|mut t| {
            // GYROSHIP_WGSL_OVERRIDE=<dir>: <dir>/<key>.wgsl replaces the translation (a
            // debugging aid: a program edited to show its intermediate values).
            if let Some(text) = std::env::var_os("GYROSHIP_WGSL_OVERRIDE").and_then(|d| std::fs::read_to_string(std::path::Path::new(&d).join(format!("{key:016x}.wgsl"))).ok()) {
                t.wgsl = text;
            }
            let m = self.shader_module(&format!("{key:016x}"), &t.wgsl).map_err(|()| format!("program {key:016x}: invalid WGSL"))?;
            Ok(std::sync::Arc::new((t, m)))
        });
        self.cache.modules.insert(id, made);
        self.compile.modules += 1;
        self.compile.module_time += since.elapsed();
        self.compile.made.push(format!("m{key:016x}/{:x}", id.3));
    }

    /// A module from translated WGSL, parsed and validated once, by wgpu; refused
    /// when wgpu reports an error (its message is then among the wgpu errors).
    /// A module is never used in a pipeline before this check: one invalid
    /// pipeline would void the frame's single command encoder.
    pub(crate) fn shader_module(&self, label: &str, wgsl: &str) -> Result<wgpu::ShaderModule, ()> {
        let m = self.device.create_shader_module(wgpu::ShaderModuleDescriptor { label: Some(label), source: wgpu::ShaderSource::Wgsl(wgsl.into()) });
        let info = pollster::block_on(m.get_compilation_info());
        match info.messages.iter().any(|m| m.message_type == wgpu::CompilationMessageType::Error) {
            true => Err(()),
            false => Ok(m),
        }
    }

    /// A stage's constant banks (64 KiB each, as bound) and textures.
    fn stage_bindings<'t>(&mut self, d: &Draw, stage: usize, t: &'t Translation, attached: &[u64], mem: Mem) -> Result<Bindings<'t>, String> {
        let mut banks = BTreeMap::new();
        for b in t.interface.cbufs.iter().chain(&t.interface.handle_banks).copied() {
            let (addr, size) = d.cbufs[stage].get(usize::from(b)).copied().flatten().ok_or_else(|| format!("constant bank {b} not bound"))?;
            // Its bound size only: binding it as 64 KiB is `stage_window`'s.
            let data = self.read_guest(crate::surfaces::Read::Bank, addr, (size as usize).min(CBUF_BYTES), mem).ok_or_else(|| format!("constant bank {b} at {addr:#x} not readable"))?;
            banks.insert(b, (addr, data));
        }
        let (textures, swizzles) = self.texture_slots(t, &banks, d.headers, d.samplers, d.sampler_binding, attached, mem)?;
        // The handle banks were for the textures; the program reads the others.
        banks.retain(|b, _| t.interface.cbufs.contains(b));
        Ok(Bindings { cbufs: banks, textures, swizzles: t.interface.swizzles.then_some(swizzles), sizes: &t.interface.cbuf_sizes })
    }

    /// A program's texture slots (view, sampler) and swizzle codes, from its
    /// handles in `banks` and the header and sampler pools.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn texture_slots(&mut self, t: &Translation, banks: &BTreeMap<u8, Bank>, headers: Pool, samplers: Pool, binding: SamplerBinding, attached: &[u64], mem: Mem) -> Result<(Vec<Slot>, Vec<u32>), String> {
        let mut textures = Vec::new();
        let mut swizzles = vec![0u32; t.interface.textures.len().next_multiple_of(4)];
        for (slot, x) in t.interface.textures.iter().enumerate() {
            let word = |bank: u8, offset: u16| -> Option<u32> {
                let at = usize::from(offset);
                let w = banks.get(&bank)?.1.get(at..at + 4)?;
                Some(u32::from_le_bytes([w[0], w[1], w[2], w[3]]))
            };
            let handle = TexHandle::new(
                match &x.handle {
                    Handle::CBuf { bank, offset } => word(*bank, *offset),
                    Handle::Uniform(e) => e.eval(&word),
                    other => return Err(format!("texture handle {other:?}")),
                }
                .ok_or("texture handle offset")?,
            );
            use wgpu::TextureViewDimension as V;
            let (dimension, types): (V, &[TextureType]) = match x.dim {
                TexDim::D2 => (V::D2, &[TextureType::TwoD, TextureType::TwoDNoMipmap]),
                TexDim::D3 => (V::D3, &[TextureType::ThreeD]),
                TexDim::Array2D => (V::D2Array, &[TextureType::TwoDArray]),
                TexDim::Cube => (V::Cube, &[TextureType::Cubemap]),
                TexDim::ArrayCube => (V::CubeArray, &[TextureType::CubemapArray]),
                other => return Err(format!("texture {other:?} (compare {})", x.compare)),
            };
            let header = self.descriptor(headers.entry(handle.header).ok_or("texture index past the pool")?, mem).ok_or("texture header not readable")?;
            let tic = textures::tic(&header).ok_or("texture header layout")?;
            if attached.contains(&tic.address) {
                return Err("texture is a target of the same draw".into());
            }
            if !types.contains(&tic.texture_type) {
                return Err(format!("{:?} texture read as {:?}", tic.texture_type, x.dim));
            }
            let layers = textures::layers(&tic);
            if matches!(dimension, V::Cube | V::CubeArray) && tic.width != tic.height {
                return Err(format!("cube texture of {}x{}", tic.width, tic.height));
            }
            // A host texture fits the view when it has its dimension and enough layers.
            let fits = |t: &wgpu::Texture| -> Result<(), String> {
                let three_d = t.dimension() == wgpu::TextureDimension::D3;
                if three_d != (dimension == V::D3) || (!three_d && t.depth_or_array_layers() < layers) {
                    return Err(format!("{:?} texture of {} layers read as {:?} of {layers}", t.dimension(), t.depth_or_array_layers(), x.dim));
                }
                Ok(())
            };
            // Every level of the host texture (a texture from memory holds the view's).
            let view_desc = |aspect| wgpu::TextureViewDescriptor { dimension: Some(dimension), aspect, array_layer_count: Some(layers), ..Default::default() };
            // Only queried (TXQ): a stand-in of its size and levels, as the texture itself
            // may be an image the same dispatch writes, which wgpu cannot also bind for reading.
            if x.query {
                let three_d = dimension == V::D3;
                let size = wgpu::Extent3d { width: tic.width.max(1), height: tic.height.max(1), depth_or_array_layers: if three_d { tic.depth.max(1) } else { 1 } };
                let dim = if three_d { wgpu::TextureDimension::D3 } else { wgpu::TextureDimension::D2 };
                let levels = (tic.max_mip_level + 1).min(size.max_mips(dim));
                let device = &self.device;
                let texture = self.cache.queried.entry((size.width, size.height, size.depth_or_array_layers, levels)).or_insert_with(|| {
                    device.create_texture(&wgpu::TextureDescriptor { label: Some("queried"), size, mip_level_count: levels, sample_count: 1, dimension: dim, format: wgpu::TextureFormat::R8Unorm, usage: wgpu::TextureUsages::TEXTURE_BINDING, view_formats: &[] })
                }).clone();
                let dimension = if three_d { V::D3 } else { V::D2 };
                textures.push(Slot { view: self.view(&texture, &Default::default()), sampler: None, depth: false, compare: false, dimension });
                continue;
            }
            // An image a compute program wrote, or a render target at that address: its host texture.
            let (view, depth, source) = match Self::latest_at(&self.map, tic.address) {
                // With the header's format when the image allows it as a view (sRGB over linear storage).
                _ if self.cache.images.contains_key(&tic.address) => {
                    let image = match self.cache.twins.get(&tic.address) {
                        Some(twin) => twin,
                        None => {
                            let twin = self.float_twin(&self.cache.images[&tic.address].clone())?;
                            self.cache.twins.entry(tic.address).or_insert(twin)
                        }
                    }
                    .clone();
                    fits(&image)?;
                    let view = self.view_as(&image, textures::wgpu_texture_format(&tic), wgpu::TextureUsages::TEXTURE_BINDING, view_desc(wgpu::TextureAspect::All));
                    (view, false, "image")
                }
                Some(e) => {
                    fits(&e.texture)?;
                    let depth = e.texture.format().is_depth_stencil_format();
                    let (key, stamp, texture) = ((e.surface.addr, e.surface.format), e.written, e.texture.clone());
                    // A color target read counts as touched: frame dumps show what the frame sampled.
                    if !depth {
                        self.touched.insert(key);
                    }
                    // The game sizes a target to an aligned width and samples it through a
                    // header of its real size: coordinates normalize to the header's size.
                    // ponytail: depth targets stay whole (WebGPU copies only whole depth
                    // subresources); a depth-writing crop pass if one is read so.
                    let (w, h) = (tic.width.min(texture.width()), tic.height.min(texture.height()));
                    let texture = if !depth && w > 0 && h > 0 && (w, h) != (texture.width(), texture.height()) { self.cropped(key, stamp, &texture, w, h) } else { texture };
                    let aspect = if depth { wgpu::TextureAspect::DepthOnly } else { wgpu::TextureAspect::All };
                    (self.view_as(&texture, textures::wgpu_texture_format(&tic), wgpu::TextureUsages::TEXTURE_BINDING, view_desc(aspect)), depth, "target")
                }
                None => match self.upload(&tic, header, mem) {
                    // A color texture read with a compare (the 1x1 stand-in bound when
                    // there is no shadow): the hardware compares with its R channel.
                    Ok(texture) if x.compare && !texture.format().is_depth_stencil_format() && texture.dimension() == wgpu::TextureDimension::D2 => {
                        let depth = match self.cache.compare_depths.get(&(tic.address, header)) {
                            Some(d) => d.clone(),
                            None => {
                                let d = self.as_depth(&texture);
                                self.cache.compare_depths.insert((tic.address, header), d.clone());
                                d
                            }
                        };
                        (self.view(&depth, &view_desc(wgpu::TextureAspect::DepthOnly)), true, "memory")
                    }
                    Ok(texture) => {
                        let depth = texture.format().is_depth_stencil_format();
                        (self.view(&texture, &view_desc(if depth { wgpu::TextureAspect::DepthOnly } else { wgpu::TextureAspect::All })), depth, "memory")
                    }
                    Err(why) if x.compare => {
                        // Where the nearest depth targets are, to tell a missing shadow pass from an offset view.
                        let mut near: Vec<(u64, u32, u32, u32)> = self.map.values().filter(|e| e.texture.format().is_depth_stencil_format()).map(|e| (e.surface.addr, e.surface.width, e.surface.height, e.surface.layers)).collect();
                        near.sort_by_key(|n| n.0.abs_diff(tic.address));
                        near.truncate(3);
                        return Err(format!("depth-compare texture not in a depth target: {:#x} {:?} {:?} {}x{}x{}; nearest depth targets {near:x?}; {why}", tic.address, tic.texture_type, tic.components, tic.width, tic.height, textures::layers(&tic)));
                    }
                    Err(why) => return Err(why),
                },
            };
            if x.compare && !depth {
                return Err(format!("depth-compare texture in a color {source}"));
            }
            swizzles[slot] = textures::swizzle_code(&tic);
            self.textures_read.push((tic.address, source));
            let sampler = match x.fetch {
                true => None,
                false => {
                    let index = match binding {
                        SamplerBinding::Independently => handle.sampler,
                        SamplerBinding::ViaHeaderBinding => handle.header,
                    };
                    let bytes = self.descriptor(samplers.entry(index).ok_or("sampler index past the pool")?, mem).ok_or("sampler header not readable")?;
                    // A comparison sampler filters (PCF) whatever the depth format.
                    Some(self.sampler(bytes, !depth || x.compare, x.compare))
                }
            };
            textures.push(Slot { view, sampler, depth, compare: x.compare, dimension });
        }
        Ok((textures, swizzles))
    }

    /// A 2D, 3D, 2D array or cube texture from guest memory: the level the view
    /// starts at, every layer (block linear: one every `layer_stride`). ZF32 is
    /// uploaded as R32Float then written as depth by a pass (WebGPU copies no
    /// buffer into Depth32Float): a shadow map the game samples but never renders.
    fn upload(&mut self, tic: &textures::Tic, header: [u8; 32], mem: Mem) -> Result<wgpu::Texture, String> {
        if let Some((t, _)) = self.cache.textures.get(&(tic.address, header)) {
            return Ok(t.clone());
        }
        let started = std::time::Instant::now();
        let three_d = tic.texture_type == TextureType::ThreeD;
        let format = textures::wgpu_texture_format(tic).ok_or_else(|| format!("texture format {:?}", tic.components))?;
        let zf32 = format == wgpu::TextureFormat::Depth32Float && !three_d;
        if let Some(note) = textures::format_notes(tic).into_iter().find(|n| !n.starts_with("needs Features::")).filter(|_| !zf32) {
            return Err(format!("texture format {:?}: {note}", tic.components));
        }
        if (format.is_depth_stencil_format() && !zf32) || !self.device.features().contains(format.required_features()) {
            return Err(format!("texture format {format:?}"));
        }
        let format = if zf32 { wgpu::TextureFormat::R32Float } else { float_format(format).unwrap_or(format) };
        let (bpe, bw, bh) = tic.components.element().ok_or("texture element size")?;
        // The view's levels (RES_VIEW_MIN/MAX_MIP_LEVEL); the first is the host texture's
        // level 0. MIN_LOD_CLAMP (from the view's first level, as NIL packs it) hides the
        // levels below it, which a streamed texture may not hold yet: its whole part
        // moves the first level, as an integer clamp would.
        let (view_first, view_last) = tic.view_levels.unwrap_or((0, tic.max_mip_level));
        let level = (view_first + tic.min_lod_clamp.map_or(0, |c| c as u32)).min(tic.max_mip_level);
        let (width, height) = ((tic.width >> level).max(1), (tic.height >> level).max(1));
        let (columns, rows) = (width.div_ceil(bw), height.div_ceil(bh));
        let layers = textures::layers(tic);
        if layers == 0 || layers > self.device.limits().max_texture_array_layers {
            return Err(format!("texture of {layers} layers"));
        }
        let depth = if three_d { (tic.depth >> level).max(1) } else { layers };
        let dimension = if three_d { wgpu::TextureDimension::D3 } else { wgpu::TextureDimension::D2 };
        // Compressed formats need whole blocks: the host texture covers them.
        let size = wgpu::Extent3d { width: columns * bw, height: rows * bh, depth_or_array_layers: depth };
        // Block linear: every level the view reaches, a mip chain wgpu can hold (pitch
        // textures have one level; a ZF32 shadow map's depth copy takes level 0 only).
        let last = if tic.texture_type == TextureType::TwoDNoMipmap { level } else { view_last.min(tic.max_mip_level).max(level) };
        let levels = if tic.layout == Layout::BlockLinear && !zf32 { (last - level + 1).min(size.max_mips(dimension)) } else { 1 };
        // Each level's texels (linear, layer after layer) and its size in elements.
        let mut data: Vec<(Vec<u8>, u32, u32)> = Vec::new();
        let end;
        match tic.layout {
            Layout::BlockLinear => {
                let offsets = textures::level_offsets(tic);
                let stride = textures::layer_stride(tic).ok_or("texture layer stride")?;
                let mut reach = 0;
                for l in level..level + levels {
                    let bl = textures::tic_level(tic, l).ok_or("texture level")?;
                    let offset = *offsets.get(l as usize).ok_or("texture level")?;
                    let mut raw = vec![0u8; bl.size() as usize];
                    let mut texels = Vec::new();
                    for layer in 0..u64::from(layers) {
                        if !mem(tic.address + layer * stride + offset, &mut raw) {
                            return Err("texture not readable".into());
                        }
                        // Every layer's room at once (an 85 MiB pool would otherwise regrow ~10 times).
                        let t = bl.deswizzle(&raw);
                        if texels.is_empty() {
                            texels = t;
                            texels.reserve_exact(texels.len() * (layers as usize - 1));
                        } else {
                            texels.extend_from_slice(&t);
                        }
                    }
                    reach = reach.max(offset + raw.len() as u64);
                    data.push((texels, bl.width, bl.height));
                }
                end = tic.address + u64::from(layers - 1) * stride + reach;
            }
            Layout::Pitch if level == 0 && !three_d && layers == 1 => {
                let row = (columns * bpe) as usize;
                let mut raw = vec![0u8; tic.pitch as usize * (rows as usize - 1) + row];
                if !mem(tic.address, &mut raw) {
                    return Err("texture not readable".into());
                }
                end = tic.address + raw.len() as u64;
                data.push((raw.chunks(tic.pitch as usize).flat_map(|r| &r[..row]).copied().collect(), columns, rows));
            }
            other => return Err(format!("texture layout {other:?}")),
        }
        let texture = self.device.create_texture(&wgpu::TextureDescriptor {
            label: None,
            size,
            mip_level_count: levels,
            sample_count: 1,
            dimension,
            format,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });
        let mut bytes = 0;
        for (i, (texels, guest_columns, guest_rows)) in data.iter().enumerate() {
            let i = i as u32;
            // wgpu halves the block-rounded level 0; the guest halves the real size: on
            // a size that is not a power of two they may differ by a block, then the
            // guest's texels are copied into wgpu's (the rest stays zero).
            let extent = size.mip_level_size(i, dimension).physical_size(format);
            let (host_columns, host_rows) = (extent.width / bw, extent.height / bh);
            let packed;
            let texels = if (host_columns, host_rows) == (*guest_columns, *guest_rows) {
                texels
            } else {
                packed = repack(texels, (*guest_columns, *guest_rows), (host_columns, host_rows), extent.depth_or_array_layers, bpe);
                &packed
            };
            self.queue.write_texture(
                wgpu::TexelCopyTextureInfo { texture: &texture, mip_level: i, origin: wgpu::Origin3d::ZERO, aspect: wgpu::TextureAspect::All },
                texels,
                wgpu::TexelCopyBufferLayout { offset: 0, bytes_per_row: Some(host_columns * bpe), rows_per_image: Some(host_rows) },
                extent,
            );
            bytes += texels.len() as u64;
        }
        let texture = if zf32 { self.as_depth(&texture) } else { texture };
        // A large upload's staging memory is freed once the GPU is past it: at once, so
        // that two such uploads in a frame (a 128 MiB shadow atlas and a 128 MiB light
        // probe texture in the tutorial) do not exhaust it.
        if bytes >= 32 << 20 {
            self.flush();
            let _ = self.device.poll(wgpu::PollType::wait_indefinitely());
        }
        self.cache.textures.insert((tic.address, header), (texture.clone(), end));
        self.cache.spans = None;
        self.staging.uploads[0] += bytes;
        self.staging.uploads[1] += started.elapsed().as_nanos() as u64;
        Ok(texture)
    }

    /// A texture with 32-bit integer channels as a float one with the same bits, level 0:
    /// programs declare every sampled texture as float and bitcast what they load.
    /// ponytail: a copy per image and dispatch; integer texture declarations if it costs.
    fn float_twin(&mut self, t: &wgpu::Texture) -> Result<wgpu::Texture, String> {
        let Some(format) = float_format(t.format()) else {
            return match t.format().sample_type(None, None) {
                Some(wgpu::TextureSampleType::Uint | wgpu::TextureSampleType::Sint) => Err(format!("integer texture {:?} sampled", t.format())),
                _ => Ok(t.clone()),
            };
        };
        let size = t.size();
        let row = (size.width * format.block_copy_size(None).unwrap_or(4)).next_multiple_of(wgpu::COPY_BYTES_PER_ROW_ALIGNMENT);
        let buffer = self.device.create_buffer(&wgpu::BufferDescriptor { label: Some("float twin"), size: u64::from(row * size.height * size.depth_or_array_layers), usage: wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST, mapped_at_creation: false });
        let twin = self.device.create_texture(&wgpu::TextureDescriptor { label: Some("float twin"), size, mip_level_count: 1, sample_count: 1, dimension: t.dimension(), format, usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST, view_formats: &[] });
        let layout = wgpu::TexelCopyBufferLayout { offset: 0, bytes_per_row: Some(row), rows_per_image: Some(size.height) };
        let mut encoder = self.encoder();
        encoder.copy_texture_to_buffer(t.as_image_copy(), wgpu::TexelCopyBufferInfo { buffer: &buffer, layout }, size);
        encoder.copy_buffer_to_texture(wgpu::TexelCopyBufferInfo { buffer: &buffer, layout }, twin.as_image_copy(), size);
        self.record(encoder);
        Ok(twin)
    }

    /// The top-left `w` × `h` of a color target, every layer, as a texture of that
    /// size: copied again only when the target was written since.
    fn cropped(&mut self, key: (u64, u32), stamp: u64, t: &wgpu::Texture, w: u32, h: u32) -> wgpu::Texture {
        let id = (key.0, key.1, w, h);
        if let Some((s, c)) = self.cache.crops.get(&id)
            && *s == stamp
        {
            return c.clone();
        }
        let size = wgpu::Extent3d { width: w, height: h, depth_or_array_layers: t.depth_or_array_layers() };
        // Read through an sRGB or linear view as the target itself may be.
        let f = t.format();
        let other = if f.is_srgb() { f.remove_srgb_suffix() } else { f.add_srgb_suffix() };
        let view_formats = if other == f { Vec::new() } else { vec![other] };
        let crop = self.device.create_texture(&wgpu::TextureDescriptor {
            label: Some("crop"),
            size,
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: f,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &view_formats,
        });
        let mut encoder = self.encoder();
        encoder.copy_texture_to_texture(t.as_image_copy(), crop.as_image_copy(), size);
        self.record(encoder);
        self.cache.crops.insert(id, (stamp, crop.clone()));
        crop
    }

    /// A Depth32Float copy of a float texture's R channel, every layer, by a pass
    /// that writes each texel as fragment depth.
    fn as_depth(&mut self, floats: &wgpu::Texture) -> wgpu::Texture {
        let device = &self.device;
        let pipeline = self.cache.depth_copy.get_or_insert_with(|| {
            let wgsl = "@group(0) @binding(0) var src: texture_2d<f32>;
                @vertex fn vs(@builtin(vertex_index) i: u32) -> @builtin(position) vec4<f32> {
                    return vec4<f32>(f32(i & 1u) * 4.0 - 1.0, f32(i >> 1u) * 4.0 - 1.0, 0.0, 1.0);
                }
                @fragment fn fs(@builtin(position) p: vec4<f32>) -> @builtin(frag_depth) f32 {
                    return textureLoad(src, vec2<i32>(p.xy), 0).r;
                }";
            let module = device.create_shader_module(wgpu::ShaderModuleDescriptor { label: Some("depth copy"), source: wgpu::ShaderSource::Wgsl(wgsl.into()) });
            device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some("depth copy"),
                layout: None,
                vertex: wgpu::VertexState { module: &module, entry_point: Some("vs"), compilation_options: Default::default(), buffers: &[] },
                primitive: Default::default(),
                depth_stencil: Some(wgpu::DepthStencilState { format: wgpu::TextureFormat::Depth32Float, depth_write_enabled: Some(true), depth_compare: Some(wgpu::CompareFunction::Always), stencil: Default::default(), bias: Default::default() }),
                multisample: Default::default(),
                fragment: Some(wgpu::FragmentState { module: &module, entry_point: Some("fs"), compilation_options: Default::default(), targets: &[] }),
                multiview_mask: None,
                cache: None,
            })
        }).clone();
        let depth = self.device.create_texture(&wgpu::TextureDescriptor {
            label: Some("zf32"),
            size: floats.size(),
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Depth32Float,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::RENDER_ATTACHMENT,
            view_formats: &[],
        });
        let mut encoder = self.encoder();
        for layer in 0..floats.size().depth_or_array_layers {
            let one = |t: &wgpu::Texture| t.create_view(&wgpu::TextureViewDescriptor { dimension: Some(wgpu::TextureViewDimension::D2), base_array_layer: layer, array_layer_count: Some(1), ..Default::default() });
            let group = self.device.create_bind_group(&wgpu::BindGroupDescriptor { label: None, layout: &pipeline.get_bind_group_layout(0), entries: &[wgpu::BindGroupEntry { binding: 0, resource: wgpu::BindingResource::TextureView(&one(floats)) }] });
            let view = one(&depth);
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("depth copy"),
                color_attachments: &[],
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment { view: &view, depth_ops: Some(wgpu::Operations { load: wgpu::LoadOp::Clear(0.0), store: wgpu::StoreOp::Store }), stencil_ops: None }),
                ..Default::default()
            });
            pass.set_pipeline(&pipeline);
            pass.set_bind_group(0, &group, &[]);
            pass.draw(0..3, 0..1);
        }
        self.record(encoder);
        depth
    }

    /// The GPU wrote guest memory `[start, start + len)`: the uploaded textures it
    /// overlaps are forgotten before the cache is next read. Recorded only (the GPU
    /// writes a few bytes at a time, thousands of times a frame); contiguous writes
    /// merge.
    pub fn invalidate(&mut self, start: u64, len: u64) {
        self.memo.wrote(start, len);
        let end = start.saturating_add(len);
        match self.cache.pending.last_mut() {
            Some(last) if start <= last.1 && end >= last.0 => *last = (last.0.min(start), last.1.max(end)),
            _ => self.cache.pending.push((start, end)),
        }
    }

    /// Applies the recorded GPU writes: one pass over the uploaded textures drops
    /// those they overlap, then the depth copies made from them.
    pub(crate) fn drain_invalidations(&mut self) {
        if self.cache.pending.is_empty() {
            return;
        }
        let c = &mut self.cache;
        let written = merged(&c.pending);
        c.pending.clear();
        // Most writes (constant-buffer loads, before nearly every draw) overlap no
        // texture: checked against the spans index before passing over them all.
        let spans = c.spans.get_or_insert_with(|| spans(c.textures.iter().map(|((a, _), (_, b))| (*a, *b))));
        if !written.iter().any(|&(s, e)| hits(spans, s, e)) {
            return;
        }
        let before = c.textures.len();
        c.textures.retain(|(addr, _), (_, last)| !overlaps(&written, *addr, *last));
        if c.textures.len() != before {
            c.spans = None;
            let textures = &c.textures;
            c.compare_depths.retain(|k, _| textures.contains_key(k));
        }
    }

    /// The sampler of a header; a non-filtering one (nearest everywhere) for
    /// unfilterable textures; with `compare`, a comparison sampler with the
    /// header's TEXSAMP0_DEPTH_COMPARE_FUNC (taken even when its
    /// TEXSAMP0_DEPTH_COMPARE bit is clear: the program asked for a compare).
    fn sampler(&mut self, bytes: [u8; 32], filtering: bool, compare: bool) -> wgpu::Sampler {
        if let Some(s) = self.cache.samplers.get(&(bytes, filtering, compare)) {
            return s.clone();
        }
        let t = textures::tsc(&bytes);
        let border = self.device.features().contains(wgpu::Features::ADDRESS_MODE_CLAMP_TO_BORDER);
        let mode = |a: textures::Address| match a.wgpu() {
            Some(wgpu::AddressMode::ClampToBorder) if !border => wgpu::AddressMode::ClampToEdge,
            Some(m) => m,
            None => wgpu::AddressMode::ClampToEdge,
        };
        let address = t.address.map(mode);
        let (mag, min, mip) = match filtering {
            true => (t.mag_filter.wgpu().unwrap_or(wgpu::FilterMode::Linear), t.min_filter.wgpu().unwrap_or(wgpu::FilterMode::Linear), t.mip_filter.wgpu().unwrap_or(wgpu::MipmapFilterMode::Nearest)),
            false => (wgpu::FilterMode::Nearest, wgpu::FilterMode::Nearest, wgpu::MipmapFilterMode::Nearest),
        };
        let linear = mag == wgpu::FilterMode::Linear && min == wgpu::FilterMode::Linear && mip == wgpu::MipmapFilterMode::Linear;
        let (lod_min_clamp, lod_max_clamp) = t.wgpu_lod_clamp();
        let s = self.device.create_sampler(&wgpu::SamplerDescriptor {
            label: None,
            address_mode_u: address[0],
            address_mode_v: address[1],
            address_mode_w: address[2],
            mag_filter: mag,
            min_filter: min,
            mipmap_filter: mip,
            lod_min_clamp,
            lod_max_clamp: lod_max_clamp.max(lod_min_clamp),
            compare: compare.then(|| t.compare_func.wgpu()),
            anisotropy_clamp: if linear { t.max_anisotropy.clamp(1, 16) } else { 1 },
            border_color: address.contains(&wgpu::AddressMode::ClampToBorder).then(|| t.wgpu_border_color().unwrap_or(wgpu::SamplerBorderColor::TransparentBlack)),
        });
        self.cache.samplers.insert((bytes, filtering, compare), s.clone());
        s
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pushbuf::Clear;
    use crate::sass::Stage;
    use crate::shader::{Interface, Texture};
    use crate::state::resources;

    const HEADERS: u64 = 0x40_0000;
    const SAMPLERS: u64 = 0x41_0000;

    /// Sets bits hi..=lo of a 32-byte header.
    fn put(h: &mut [u8; 32], fields: &[(u32, u32, u32)]) {
        for &(hi, lo, v) in fields {
            for (i, b) in (lo..=hi).enumerate() {
                let byte = &mut h[(b / 8) as usize];
                *byte = *byte & !(1 << (b % 8)) | ((v >> i & 1) as u8) << (b % 8);
            }
        }
    }

    /// Binds texture slot `x` (handle 0 at c[0][0]: header 0, sampler 0) through
    /// `texture_slots`, from `memory` plus the two headers, and runs `wgsl` (declaring
    /// `out: array<f32>` at binding 1) as one compute thread; `n` floats of `out` back.
    fn run(s: &mut Surfaces, x: Texture, tic: [u8; 32], tsc: [u8; 32], memory: Vec<(u64, Vec<u8>)>, wgsl: &str, n: usize) -> Vec<f32> {
        let t = Translation { key: 0, stage: Stage::Compute, wgsl: wgsl.into(), interface: Interface { textures: vec![x], ..Default::default() } };
        let regions = [memory, vec![(HEADERS, tic.to_vec()), (SAMPLERS, tsc.to_vec())]].concat();
        let mut mem = |addr: u64, out: &mut [u8]| {
            regions.iter().any(|(at, data)| {
                let Some(src) = addr.checked_sub(*at).and_then(|o| data.get(o as usize..o as usize + out.len())) else { return false };
                out.copy_from_slice(src);
                true
            })
        };
        let reg = |m: u32| match m {
            0x1578 => HEADERS as u32,
            0x1560 => SAMPLERS as u32,
            _ => 0,
        };
        let banks = BTreeMap::from([(0u8, (0, std::sync::Arc::new(vec![0u8; 16])))]);
        let (slots, _) = s.texture_slots(&t, &banks, resources::header_pool(&reg), resources::sampler_pool(&reg), resources::sampler_binding(&reg), &[], &mut mem).unwrap_or_else(|e| panic!("{e}"));
        let none = BTreeMap::new();
        let bindings = Bindings { cbufs: BTreeMap::new(), textures: slots, swizzles: None, sizes: &none };
        let mut entries = bindings.layout(wgpu::ShaderStages::COMPUTE);
        let storage = wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: false }, has_dynamic_offset: false, min_binding_size: None };
        entries.push(wgpu::BindGroupLayoutEntry { binding: 1, visibility: wgpu::ShaderStages::COMPUTE, ty: storage, count: None });
        let d = &s.device;
        let layout = d.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor { label: None, entries: &entries });
        let pipeline_layout = d.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor { label: None, bind_group_layouts: &[Some(&layout)], ..Default::default() });
        let module = d.create_shader_module(wgpu::ShaderModuleDescriptor { label: None, source: wgpu::ShaderSource::Wgsl(wgsl.into()) });
        let pipeline = d.create_compute_pipeline(&wgpu::ComputePipelineDescriptor { label: None, layout: Some(&pipeline_layout), module: &module, entry_point: Some("main"), compilation_options: Default::default(), cache: None });
        let size = 4 * n as u64;
        let out = d.create_buffer(&wgpu::BufferDescriptor { label: None, size, usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC, mapped_at_creation: false });
        let read = d.create_buffer(&wgpu::BufferDescriptor { label: None, size, usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ, mapped_at_creation: false });
        let slot = &bindings.textures[0];
        let mut group = vec![wgpu::BindGroupEntry { binding: 32, resource: wgpu::BindingResource::TextureView(&slot.view) }, wgpu::BindGroupEntry { binding: 1, resource: out.as_entire_binding() }];
        if let Some(sampler) = &slot.sampler {
            group.push(wgpu::BindGroupEntry { binding: 33, resource: wgpu::BindingResource::Sampler(sampler) });
        }
        let group = d.create_bind_group(&wgpu::BindGroupDescriptor { label: None, layout: &layout, entries: &group });
        let mut encoder = d.create_command_encoder(&Default::default());
        {
            let mut pass = encoder.begin_compute_pass(&Default::default());
            pass.set_pipeline(&pipeline);
            pass.set_bind_group(0, &group, &[]);
            pass.dispatch_workgroups(1, 1, 1);
        }
        encoder.copy_buffer_to_buffer(&out, 0, &read, 0, size);
        s.flush();
        s.queue.submit([encoder.finish()]);
        read.slice(..).map_async(wgpu::MapMode::Read, |r| r.expect("readback mapping failed"));
        s.device.poll(wgpu::PollType::wait_indefinitely()).expect("GPU poll");
        let bytes = read.slice(..).get_mapped_range().expect("mapped").to_vec();
        assert!(s.errors.lock().unwrap().is_empty(), "{:?}", s.errors);
        bytes.as_chunks::<4>().0.iter().map(|b| f32::from_le_bytes(*b)).collect()
    }

    #[test]
    fn a_block_linear_2d_array_is_uploaded_with_every_layer() {
        let Ok(mut s) = Surfaces::headless(crate::Backend::Auto) else {
            eprintln!("no GPU adapter: test skipped");
            return;
        };
        // 3 layers of 8x8 RGBA8 with 2 levels: level 0 one GOB, level 1 (4x4) another, so
        // layers are 1024 bytes apart. Texel (x, y) of layer l is (40 l + x, y, l, 255) at
        // level 0, (40 l + x, y, 100 + l, 255) at level 1.
        const TEX: u64 = 0x50_0000;
        let mut tic = [0u8; 32];
        put(&mut tic, &[(6, 0, 0x08), (9, 7, 2), (12, 10, 2), (15, 13, 2), (18, 16, 2), (21, 19, 2), (24, 22, 3), (27, 25, 4), (30, 28, 5)]);
        put(&mut tic, &[(63, 41, (TEX >> 9) as u32), (87, 85, 3), (127, 124, 1), (144, 128, 7), (175, 160, 7), (189, 176, 2), (154, 151, 5), (191, 191, 1), (231, 228, 1)]);
        let t = textures::tic(&tic).unwrap();
        assert_eq!(textures::layer_stride(&t), Some(1024));
        let offsets = textures::level_offsets(&t);
        let mut memory = vec![0u8; 3 * 1024];
        for l in 0..3u32 {
            for level in 0..2u32 {
                let bl = textures::tic_level(&t, level).unwrap();
                for (x, y) in (0..8 >> level).flat_map(|x| (0..8 >> level).map(move |y| (x, y))) {
                    let at = (1024 * u64::from(l) + offsets[level as usize] + bl.offset(4 * x, y, 0)) as usize;
                    memory[at..at + 4].copy_from_slice(&[(40 * l + x) as u8, y as u8, (100 * level + l) as u8, 255]);
                }
            }
        }
        let x = Texture { handle: Handle::CBuf { bank: 0, offset: 0 }, dim: TexDim::Array2D, compare: false, fetch: true, query: false };
        let wgsl = "@group(0) @binding(32) var tex0: texture_2d_array<f32>;
            @group(0) @binding(1) var<storage, read_write> out: array<f32>;
            @compute @workgroup_size(1) fn main() {
                for (var l = 0u; l < 3u; l++) {
                    let c = textureLoad(tex0, vec2<i32>(3, 5), l, 0);
                    let m = textureLoad(tex0, vec2<i32>(2, 1), l, 1);
                    out[6u * l] = c.r; out[6u * l + 1u] = c.g; out[6u * l + 2u] = c.b; out[6u * l + 3u] = f32(textureNumLayers(tex0));
                    out[6u * l + 4u] = m.b; out[6u * l + 5u] = f32(textureNumLevels(tex0));
                }
            }";
        let got = run(&mut s, x, tic, [0; 32], vec![(TEX, memory)], wgsl, 18);
        for l in 0..3 {
            let texel: Vec<u32> = got[6 * l..6 * l + 3].iter().map(|v| (v * 255.0).round() as u32).collect();
            assert_eq!(texel, vec![40 * l as u32 + 3, 5, l as u32], "layer {l}");
            assert_eq!(got[6 * l + 3], 3.0);
            // Level 1 is uploaded too: the view holds both levels.
            assert_eq!(((got[6 * l + 4] * 255.0).round() as u32, got[6 * l + 5]), (100 + l as u32, 2.0), "layer {l} level 1");
        }
        // The cached range covers the three layers, both levels of the last one.
        assert_eq!(s.texture_ranges(), vec![(TEX, TEX + 2 * 1024 + 1024)]);
    }

    #[test]
    fn the_pixel_group_keeps_dynamic_offsets_over_the_limit() {
        // Corneria's main pass: 3 vertex banks and 13 pixel banks plus swizzles, over 15.
        assert_eq!(dynamic_groups(&[3, 14], 15), [false, true]);
        assert_eq!(dynamic_groups(&[4, 11], 15), [true, true]);
        assert_eq!(dynamic_groups(&[3, 16], 15), [true, false]);
        assert_eq!(dynamic_groups(&[16, 16], 15), [false, false]);
        assert_eq!(dynamic_groups(&[9], 15), [true]);
    }

    #[test]
    fn repack_copies_what_fits_and_zeroes_the_rest() {
        // Two slices of 3x2 two-byte elements into 2x3: each row's first two elements,
        // both rows, then a zero row.
        let src: Vec<u8> = (0..24).collect();
        let out = repack(&src, (3, 2), (2, 3), 2, 2);
        assert_eq!(out, [vec![0, 1, 2, 3, 6, 7, 8, 9, 0, 0, 0, 0], vec![12, 13, 14, 15, 18, 19, 20, 21, 0, 0, 0, 0]].concat());
    }

    #[test]
    fn a_depth_compare_slot_compares_with_the_samplers_function() {
        let Ok(mut s) = Surfaces::headless(crate::Backend::Auto) else {
            eprintln!("no GPU adapter: test skipped");
            return;
        };
        // A 4x4 Depth32Float target cleared to 0.5, read through a ZF32 header.
        const DEPTH: u64 = 0x60_0000;
        let target = Surface { addr: DEPTH, width: 4, height: 4, format: 0x0a, layers: 1 };
        s.clear(&Clear { layer: 0, color: None, depth: Some((target, Some(0.5f32.to_bits()), None)), scissor: None });
        assert!(s.skipped.is_empty(), "{:?}", s.skipped);
        let mut tic = [0u8; 32];
        put(&mut tic, &[(6, 0, 0x2f), (9, 7, 7), (12, 10, 7), (15, 13, 7), (18, 16, 7), (21, 19, 2), (24, 22, 0), (27, 25, 0), (30, 28, 7)]);
        put(&mut tic, &[(63, 41, (DEPTH >> 9) as u32), (87, 85, 3), (144, 128, 3), (175, 160, 3), (154, 151, 1), (191, 191, 1)]);
        // DEPTH_COMPARE on, ZC_LEQUAL (the reference <= depth passes), point filters.
        let mut tsc = [0u8; 32];
        put(&mut tsc, &[(9, 9, 1), (12, 10, 3), (34, 32, 1), (37, 36, 1), (39, 38, 1)]);
        let x = Texture { handle: Handle::CBuf { bank: 0, offset: 0 }, dim: TexDim::D2, compare: true, fetch: false, query: false };
        let wgsl = "@group(0) @binding(32) var tex0: texture_depth_2d;
            @group(0) @binding(33) var smp0: sampler_comparison;
            @group(0) @binding(1) var<storage, read_write> out: array<f32>;
            @compute @workgroup_size(1) fn main() {
                let refs = array<f32, 3>(0.25, 0.5, 0.75);
                for (var i = 0u; i < 3u; i++) { out[i] = textureSampleCompareLevel(tex0, smp0, vec2<f32>(0.5, 0.5), refs[i]); }
            }";
        assert_eq!(run(&mut s, x.clone(), tic, tsc, Vec::new(), wgsl, 3), vec![1.0, 1.0, 0.0]);
        // ZC_GREATER: only the reference above the depth passes (a second cached sampler).
        put(&mut tsc, &[(12, 10, 4)]);
        assert_eq!(run(&mut s, x.clone(), tic, tsc, Vec::new(), wgsl, 3), vec![0.0, 0.0, 1.0]);
        // No target at another address: ZF32 texels from memory, here 0.5 but for the
        // last texel (0.25, sampled nowhere), give the same compares.
        const ATLAS: u64 = 0x70_0000;
        put(&mut tic, &[(63, 41, (ATLAS >> 9) as u32)]);
        let size = textures::tic_level(&textures::tic(&tic).unwrap(), 0).unwrap().size() as usize;
        let mut memory = 0.5f32.to_le_bytes().repeat(size / 4);
        memory[size - 4..].copy_from_slice(&0.25f32.to_le_bytes());
        assert_eq!(run(&mut s, x, tic, tsc, vec![(ATLAS, memory)], wgsl, 3), vec![0.0, 0.0, 1.0]);
    }

    #[test]
    fn a_module_wgpu_refuses_is_refused_and_its_error_kept() {
        let Ok(s) = Surfaces::headless(crate::Backend::Auto) else {
            eprintln!("no GPU adapter: test skipped");
            return;
        };
        assert!(s.shader_module("ok", "@compute @workgroup_size(1) fn main() {}").is_ok());
        assert!(s.errors.lock().unwrap().is_empty());
        assert!(s.shader_module("bad", "@compute @workgroup_size(1) fn main() { let x: u32 = 1.5; }").is_err());
        assert_eq!(s.errors.lock().unwrap().len(), 1, "{:?}", s.errors);
    }

    #[test]
    fn merged_writes_overlap_what_each_write_overlapped() {
        // Writes, some touching, overlapping, empty or nested; textures as [a, b).
        let writes = [(100u64, 104u64), (104, 108), (300, 300), (50, 60), (55, 58), (1000, 2000), (1500, 1504), (2000, 2004), (7, 9)];
        let one_by_one = |a: u64, b: u64| writes.iter().any(|&(start, end)| b > start && a < end);
        let all = merged(&writes);
        for a in (0..2100).step_by(3) {
            for len in [0u64, 1, 4, 7, 50, 400] {
                assert_eq!(overlaps(&all, a, a + len), one_by_one(a, a + len), "[{a}, {})", a + len);
            }
        }
        assert!(!overlaps(&merged(&[]), 0, u64::MAX));
    }

    #[test]
    fn the_spans_index_finds_a_write_over_any_texture() {
        let mut seed = 0x2545_f491_4f6c_dd1du64;
        let mut next = |n: u64| {
            seed ^= seed << 13;
            seed ^= seed >> 7;
            seed ^= seed << 17;
            seed % n
        };
        for round in 0..2000 {
            // Textures and writes, some empty, nested or touching.
            let (t, w) = (next(12), 1 + next(6));
            let mut ranges = |count: u64, len: u64| -> Vec<(u64, u64)> {
                (0..count)
                    .map(|_| {
                        let a = next(4000);
                        (a, a + next(len))
                    })
                    .collect()
            };
            let textures = ranges(t, 600);
            let writes = merged(&ranges(w, 40));
            let index = spans(textures.iter().copied());
            let fast = writes.iter().any(|&(s, e)| hits(&index, s, e));
            assert_eq!(fast, textures.iter().any(|&(a, b)| overlaps(&writes, a, b)), "round {round}: {textures:?} {writes:?}");
        }
    }

    #[test]
    fn programs_and_banks_are_read_no_further_than_their_readable_pages() {
        // Readable memory: [0x10_0f00, 0x10_1000) only, an SPH, EXIT and the closing self-branch.
        let w = |op: u128, f: u128| (op | 7 << 12 | f).to_le_bytes();
        let mut page = vec![0u8; crate::sass::SPH_SIZE];
        page.extend(w(0x94d, 7 << 87));
        page.extend(w(0x947, 7 << 87 | ((-16i64 as u64 as u128) & ((1 << 50) - 1)) << 32));
        page.resize(0x100, 0xaa);
        let mut mem = |a: u64, buf: &mut [u8]| {
            let Some(at) = a.checked_sub(0x10_0f00).map(|at| at as usize).filter(|at| at + buf.len() <= page.len()) else { return false };
            buf.copy_from_slice(&page[at..at + buf.len()]);
            true
        };
        let (_, code) = read_program(&mut mem, 0x10_0f00).expect("readable up to its page end");
        assert_eq!(code.len(), crate::sass::SPH_SIZE + 0x20);
        let mut bank = vec![0u8; 0x10000];
        assert_eq!(read_pages(&mut mem, 0x10_0f80, &mut bank), 0x80);
        assert_eq!(bank[..0x80], page[0x80..]);
        assert_eq!(read_pages(&mut mem, 0x10_1000, &mut bank), 0);
    }

    #[test]
    fn index_max_matches_the_decode_then_filter_formula() {
        // The former formula: decode every index, drop the restart value, take the max.
        let former = |raw: &[u8], size: usize, restart: Option<u32>| -> u32 {
            let values: Vec<u32> = raw.chunks_exact(size).map(index_value).collect();
            values.iter().filter(|&&i| restart != Some(i)).max().copied().unwrap_or(0)
        };
        let words: Vec<u32> = (0..257u32).map(|i| i.wrapping_mul(2_654_435_761) >> (i % 29)).collect();
        for size in [1usize, 2, 4] {
            let raw: Vec<u8> = words.iter().flat_map(|w| w.to_le_bytes()[..size].to_vec()).collect();
            let top = former(&raw, size, None);
            for restart in [None, Some(0), Some(top), Some(1), Some(u32::MAX)] {
                assert_eq!(index_max(&raw, size, restart), former(&raw, size, restart), "size {size} restart {restart:?}");
            }
            assert_eq!(index_max(&[], size, None), 0);
            // Only restart values: none is left.
            let only = top.to_le_bytes()[..size].repeat(5);
            assert_eq!(index_max(&only, size, Some(top)), 0);
        }
    }

    #[test]
    fn a_read_sees_the_target_written_last_at_its_address() {
        let Ok(mut s) = Surfaces::headless(crate::Backend::Auto) else {
            eprintln!("no GPU adapter: test skipped");
            return;
        };
        // A depth target and an RGBA16F color target share an address, as the game's
        // transient memory does; a fetch there reads whichever was written last.
        const AT: u64 = 0x80_0000;
        let depth = Surface { addr: AT, width: 4, height: 4, format: 0x0a, layers: 1 };
        let color = Surface { addr: AT, width: 4, height: 4, format: 0xca, layers: 1 };
        let mut tic = [0u8; 32];
        put(&mut tic, &[(6, 0, 0x2f), (9, 7, 7), (12, 10, 7), (15, 13, 7), (18, 16, 7), (21, 19, 2), (24, 22, 0), (27, 25, 0), (30, 28, 7)]);
        put(&mut tic, &[(63, 41, (AT >> 9) as u32), (87, 85, 3), (144, 128, 3), (175, 160, 3), (154, 151, 1), (191, 191, 1)]);
        let x = Texture { handle: Handle::CBuf { bank: 0, offset: 0 }, dim: TexDim::D2, compare: false, fetch: true, query: false };
        let wgsl = "@group(0) @binding(32) var tex0: texture_2d<f32>;
            @group(0) @binding(1) var<storage, read_write> out: array<f32>;
            @compute @workgroup_size(1) fn main() { out[0] = textureLoad(tex0, vec2<i32>(1, 1), 0).r; }";
        let clear_depth = |s: &mut Surfaces, z: f32| s.clear(&Clear { layer: 0, color: None, depth: Some((depth, Some(z.to_bits()), None)), scissor: None });
        let clear_color = |s: &mut Surfaces, r: f32| s.clear(&Clear { layer: 0, color: Some((color, [r.to_bits(), 0, 0, 1f32.to_bits()], 0xf)), depth: None, scissor: None });
        clear_depth(&mut s, 0.5);
        clear_color(&mut s, 0.25);
        assert_eq!(run(&mut s, x.clone(), tic, [0; 32], Vec::new(), wgsl, 1), vec![0.25]);
        clear_depth(&mut s, 0.75);
        assert_eq!(run(&mut s, x, tic, [0; 32], Vec::new(), wgsl, 1), vec![0.75]);
        assert!(s.skipped.is_empty(), "{:?}", s.skipped);
    }

    #[test]
    fn a_target_read_through_a_smaller_header_samples_the_header_size() {
        let Ok(mut s) = Surfaces::headless(crate::Backend::Auto) else {
            eprintln!("no GPU adapter: test skipped");
            return;
        };
        // An 8x4 RGBA16F target (an aligned width): columns 0..4 hold 0.25, 4..8 hold 1.0.
        // Read through a 4x4 header, u = 0.9 is texel 3 of 4, not texel 7 of 8.
        const AT: u64 = 0x90_0000;
        let color = Surface { addr: AT, width: 8, height: 4, format: 0xca, layers: 1 };
        let value = |v: f32| [v.to_bits(), 0, 0, 1f32.to_bits()];
        s.clear(&Clear { layer: 0, color: Some((color, value(0.25), 0xf)), depth: None, scissor: None });
        s.clear(&Clear { layer: 0, color: Some((color, value(1.0), 0xf)), depth: None, scissor: Some([4, 0, 4, 4]) });
        let mut tic = [0u8; 32];
        put(&mut tic, &[(6, 0, 0x2f), (9, 7, 7), (12, 10, 7), (15, 13, 7), (18, 16, 7), (21, 19, 2), (24, 22, 0), (27, 25, 0), (30, 28, 7)]);
        put(&mut tic, &[(63, 41, (AT >> 9) as u32), (87, 85, 3), (144, 128, 3), (175, 160, 3), (154, 151, 1), (191, 191, 1)]);
        // Point filters.
        let mut tsc = [0u8; 32];
        put(&mut tsc, &[(34, 32, 1), (37, 36, 1), (39, 38, 1)]);
        let x = Texture { handle: Handle::CBuf { bank: 0, offset: 0 }, dim: TexDim::D2, compare: false, fetch: false, query: false };
        let wgsl = "@group(0) @binding(32) var tex0: texture_2d<f32>;
            @group(0) @binding(33) var smp0: sampler;
            @group(0) @binding(1) var<storage, read_write> out: array<f32>;
            @compute @workgroup_size(1) fn main() {
                out[0] = textureSampleLevel(tex0, smp0, vec2<f32>(0.9, 0.5), 0.0).r;
                out[1] = f32(textureDimensions(tex0).x);
            }";
        assert_eq!(run(&mut s, x, tic, tsc, Vec::new(), wgsl, 2), vec![0.25, 4.0]);
        assert!(s.skipped.is_empty(), "{:?}", s.skipped);
    }

    /// Jump targets come from the table alone: its entries when its length is
    /// known, else its words up to the first that is not an instruction offset.
    #[test]
    fn jump_targets_come_from_the_table_alone() {
        let words = |w: &[u32]| w.iter().flat_map(|x| x.to_le_bytes()).collect::<Vec<u8>>();
        // Base 0x100: 0x10 and 0x30 are instructions 17 and 19 of a 0x400-byte program.
        assert_eq!(jump_targets(&[(1, 0x100, 0, Some(2))], &[words(&[0x10, 0x30])], 0x400), BTreeSet::from([17, 19]));
        // 0x13 is no instruction offset: the table ended before it.
        assert_eq!(jump_targets(&[(1, 0x100, 0, None)], &[words(&[0x10, 0x30, 0x13, 0x50])], 0x400), BTreeSet::from([17, 19]));
    }

    /// D42: the deterministic engine makes a pipeline at once; play mode hands
    /// it to a worker thread, pending until done, and names a slow one once.
    #[test]
    fn play_mode_makes_pipelines_on_worker_threads() {
        let Ok(mut s) = Surfaces::headless(crate::Backend::Auto) else {
            eprintln!("no GPU adapter: test skipped");
            return;
        };
        assert!(matches!(s.make(|| 7u32, "now".into()), Made::Ready(7)));
        s.background = true;
        let (tx, rx) = std::sync::mpsc::channel::<()>();
        let mut slow = s.make(move || rx.recv().map_or(0, |()| 8u32), "slow".into());
        assert_eq!(slow.ready(), None);
        assert_eq!(slow.overdue(std::time::Duration::ZERO).as_deref(), Some("slow"));
        assert_eq!(slow.overdue(std::time::Duration::ZERO), None, "named once");
        // One that never returns holds its own worker only.
        let mut quick = s.make(|| 9u32, "quick".into());
        let since = std::time::Instant::now();
        while quick.ready().is_none() {
            assert!(since.elapsed() < std::time::Duration::from_secs(5), "a stuck worker held the others");
            std::thread::yield_now();
        }
        // A wait gives up at its deadline, and returns a pipeline made in time.
        let waited = std::time::Instant::now();
        assert_eq!(slow.wait(waited + std::time::Duration::from_millis(20)), None);
        assert!(waited.elapsed() >= std::time::Duration::from_millis(20));
        let mut soon = s.make(|| 10u32, "soon".into());
        assert_eq!(soon.wait(std::time::Instant::now() + std::time::Duration::from_secs(5)), Some(10));
        tx.send(()).unwrap();
        while slow.ready().is_none() {
            assert!(since.elapsed() < std::time::Duration::from_secs(5));
            std::thread::yield_now();
        }
        assert_eq!((slow.ready(), quick.ready()), (Some(8), Some(9)));
    }

    /// `--pipeline-cache`: a pipeline made with the cache is written to its file,
    /// which the next run reads back without an error (Vulkan only).
    #[test]
    fn the_pipeline_cache_is_written_and_read_back() {
        // Next to the test binary, in the target folder.
        let exe = std::env::current_exe().unwrap();
        let dir = exe.parent().unwrap().join(format!("pipeline-cache-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        let Ok(mut s) = Surfaces::headless(crate::Backend::Auto) else {
            eprintln!("no GPU adapter: test skipped");
            return;
        };
        let Ok((file, 0)) = s.use_pipeline_cache(&dir) else {
            eprintln!("no pipeline cache on this backend: test skipped");
            return;
        };
        let module = s.device.create_shader_module(wgpu::ShaderModuleDescriptor { label: None, source: wgpu::ShaderSource::Wgsl("@compute @workgroup_size(1) fn main() {}".into()) });
        let cache = s.disk_cache.as_ref().map(|d| d.cache.clone());
        let _ = s.device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor { label: None, layout: None, module: &module, entry_point: Some("main"), compilation_options: Default::default(), cache: cache.as_ref() });
        s.save_pipeline_cache(true).expect("a writing thread").join().unwrap();
        assert!(std::fs::read(&file).unwrap().starts_with(b"WGPUPLCH"));
        let Ok(mut next) = Surfaces::headless(crate::Backend::Auto) else { panic!("the adapter went away") };
        let (_, size) = next.use_pipeline_cache(&dir).unwrap();
        assert!(size > 0);
        assert!(s.errors.lock().unwrap().is_empty() && next.errors.lock().unwrap().is_empty());
        let _ = std::fs::remove_dir_all(&dir);
    }
}
