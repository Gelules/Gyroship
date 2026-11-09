//! Render targets as wgpu textures, keyed by GPU address and format: the
//! host side of the 3D class's color and depth targets. Clears run here;
//! the targets a frame touched can be written out as PNG files.
//!
//! ponytail: whole-target clears only (full channel mask, no scissor smaller
//! than the target); the others are counted in `skipped`. Draws: `draw.rs`.

use crate::pushbuf::{Clear, Surface};
use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::path::Path;

/// The wgpu format of a 3D-class color target format (clc797.h), for the
/// renderable formats.
pub(crate) fn color_format(f: u32) -> Option<wgpu::TextureFormat> {
    use wgpu::TextureFormat as T;
    Some(match f {
        0xc0 => T::Rgba32Float,
        0xc6 => T::Rgba16Unorm,
        0xc7 => T::Rgba16Snorm,
        0xca => T::Rgba16Float,
        0xcb => T::Rg32Float,
        0xcf => T::Bgra8Unorm,
        0xd0 => T::Bgra8UnormSrgb,
        0xd1 => T::Rgb10a2Unorm,
        0xd5 => T::Rgba8Unorm,
        0xd6 => T::Rgba8UnormSrgb,
        0xda => T::Rg16Unorm,
        0xdb => T::Rg16Snorm,
        0xde => T::Rg16Float,
        0xe0 => T::Rg11b10Ufloat,
        0xe5 => T::R32Float,
        0xea => T::Rg8Unorm,
        0xeb => T::Rg8Snorm,
        0xee => T::R16Unorm,
        0xf2 => T::R16Float,
        0xf3 => T::R8Unorm,
        _ => return None,
    })
}

/// A view of `texture` in `format` when it is the texture's sRGB or linear
/// counterpart (one of its view formats), limited then to `usage` (an sRGB view
/// cannot carry the texture's storage usage); else in the texture's format.
pub(crate) fn view_as(texture: &wgpu::Texture, format: Option<wgpu::TextureFormat>, usage: wgpu::TextureUsages, desc: wgpu::TextureViewDescriptor) -> wgpu::TextureView {
    let other = format.filter(|f| *f != texture.format() && f.remove_srgb_suffix() == texture.format().remove_srgb_suffix());
    texture.create_view(&wgpu::TextureViewDescriptor { format: other, usage: other.map(|_| usage), ..desc })
}

/// The wgpu format of a depth target format (SET_ZT_FORMAT).
pub(crate) fn depth_format(f: u32) -> Option<wgpu::TextureFormat> {
    use wgpu::TextureFormat as T;
    Some(match f {
        0x0a => T::Depth32Float,
        0x13 => T::Depth16Unorm,
        0x14 | 0x15 | 0x16 | 0x18 => T::Depth24PlusStencil8,
        0x19 => T::Depth32FloatStencil8,
        _ => return None,
    })
}

/// Copies a target into opaque RGBA8 for a PNG to look at (missing channels
/// read as 0; alpha is dropped, as a cleared alpha of 0 would hide the image).
const BLIT: &str = r#"
@group(0) @binding(0) var src: texture_2d<f32>;

@vertex
fn vs(@builtin(vertex_index) i: u32) -> @builtin(position) vec4<f32> {
    let p = vec2(f32(i & 1u) * 4.0 - 1.0, f32(i >> 1u) * 4.0 - 1.0);
    return vec4(p, 0.0, 1.0);
}

@fragment
fn fs(@builtin(position) p: vec4<f32>) -> @location(0) vec4<f32> {
    return vec4(textureLoad(src, vec2<i32>(p.xy), 0).rgb, 1.0);
}
"#;

/// A clear bounded by a scissor or a channel mask: one triangle over the
/// target at depth `value.depth`, of color `value.color`, drawn with the
/// scissor and write masks.
const FILL: &str = r#"
struct Value { color: vec4<f32>, depth: f32 };
@group(0) @binding(0) var<uniform> value: Value;

@vertex
fn vs(@builtin(vertex_index) i: u32) -> @builtin(position) vec4<f32> {
    let p = vec2(f32(i & 1u) * 4.0 - 1.0, f32(i >> 1u) * 4.0 - 1.0);
    return vec4(p, value.depth, 1.0);
}

@fragment
fn fs() -> @location(0) vec4<f32> {
    return value.color;
}
"#;

/// What a fill writes: a color target's channels, or depth and/or stencil.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
enum Fill {
    Color { mask: u8 },
    Depth { depth: bool, stencil: bool },
}

pub(crate) struct Entry {
    pub(crate) surface: Surface,
    pub(crate) texture: wgpu::Texture,
    /// When it was last written (`Surfaces::stamp`; 0: never). The game puts targets
    /// of different formats at one address (a depth buffer, then a color target): a
    /// read takes the latest, as the shared memory would hold.
    pub(crate) written: u64,
}

pub struct Surfaces {
    pub(crate) adapter: wgpu::Adapter,
    pub(crate) device: wgpu::Device,
    pub(crate) queue: wgpu::Queue,
    pub(crate) map: BTreeMap<(u64, u32), Entry>,
    /// The count of target writes, for `Entry::written`.
    stamp: u64,
    /// Color targets written since the last dump.
    pub(crate) touched: BTreeSet<(u64, u32)>,
    blit: (wgpu::RenderPipeline, wgpu::BindGroupLayout),
    fill: (wgpu::ShaderModule, wgpu::BindGroupLayout, std::collections::HashMap<(wgpu::TextureFormat, Fill), wgpu::RenderPipeline>),
    pub(crate) cache: crate::draw::Cache,
    /// Draws done.
    pub drawn: u64,
    /// The textures the last draw read, slot by slot: address, and where the
    /// texels came from (a render target, a compute image, or guest memory).
    pub textures_read: Vec<(u64, &'static str)>,
    /// Operations not done, by reason.
    pub skipped: BTreeMap<String, u64>,
    /// What the upload arena carried, for `--perf` (host statistics).
    pub staging: Staging,
    /// wgpu validation errors (the first ones), which the HLE takes as anomalies.
    pub errors: Arc<Mutex<Vec<String>>>,
    /// The work recorded since the last [`Surfaces::flush`], in order.
    pending: Option<wgpu::CommandEncoder>,
    /// GPU time per program, when asked for (`timing.rs`).
    pub(crate) timing: Option<crate::timing::Timing>,
    /// GPU time per frame, for `--perf` ([`Surfaces::measure_frames`]).
    clock: Option<crate::timing::FrameClock>,
    /// The present the work recorded now belongs to (set by the caller at each present).
    pub present: u64,
    /// `--hashes` images on their way back from the GPU, oldest first.
    readbacks: std::collections::VecDeque<Readback>,
    /// The thread hashing them, and the channel to it.
    hasher: Option<Hasher>,
    /// This frame's vertex, index and constant data: one buffer, written at flush.
    arena: (wgpu::Buffer, Vec<u8>),
    /// Offsets of the arena's staged blocks by content hash: draws share identical
    /// constant banks (64 KiB each) instead of staging each copy.
    staged: FxMap<u64, Vec<(usize, usize)>>,
    /// The color target written last (the presented image at a present).
    last_color: Option<(u64, u32)>,
    /// The window the presented images go to, if any.
    window: Option<Window>,
    /// This submit's guest reads and staged blocks, for the reads that repeat.
    pub(crate) memo: Memo,
}

/// A presented image copied for hashing: mapped when the GPU gets to it.
struct Readback {
    present: u64,
    target: u64,
    buffer: wgpu::Buffer,
    /// Bytes of a row, its padded stride in the buffer, and the rows.
    row: u32,
    padded: u32,
    height: u32,
    /// 1 mapping, 2 mapped, 3 failed.
    state: Arc<std::sync::atomic::AtomicU8>,
}

/// The channel to the thread hashing read-back images (present, target, rows),
/// and that thread, which returns (present, target, hash) for each.
type Hasher = (std::sync::mpsc::Sender<(u64, u64, Vec<u8>)>, std::thread::JoinHandle<Vec<(u64, u64, u64)>>);

/// 64-bit FNV-1a, the hash of `--hashes`.
fn fnv(bytes: &[u8]) -> u64 {
    bytes.iter().fold(0xcbf2_9ce4_8422_2325, |h, &b| (h ^ u64::from(b)).wrapping_mul(0x100_0000_01b3))
}

/// A window's surface and the pipeline that copies an image into it.
struct Window {
    surface: wgpu::Surface<'static>,
    config: wgpu::SurfaceConfiguration,
    pipeline: wgpu::RenderPipeline,
    layout: wgpu::BindGroupLayout,
}

/// Copies the drawn area of a target into the window, scaled with point
/// sampling: the raw texel values, as the console's display sends them.
const SHOW: &str = r#"
struct Params { area: vec2<u32>, size: vec2<u32> };
@group(0) @binding(0) var src: texture_2d<f32>;
@group(0) @binding(1) var<uniform> params: Params;

@vertex
fn vs(@builtin(vertex_index) i: u32) -> @builtin(position) vec4<f32> {
    let p = vec2(f32(i & 1u) * 4.0 - 1.0, f32(i >> 1u) * 4.0 - 1.0);
    return vec4(p, 0.0, 1.0);
}

@fragment
fn fs(@builtin(position) p: vec4<f32>) -> @location(0) vec4<f32> {
    let at = vec2<u32>(floor(p.xy * vec2<f32>(params.area) / vec2<f32>(params.size)));
    return vec4(textureLoad(src, min(at, params.area - 1u), 0).rgb, 1.0);
}
"#;

impl Window {
    fn new(adapter: &wgpu::Adapter, device: &wgpu::Device, surface: wgpu::Surface<'static>, size: (u32, u32)) -> Window {
        let caps = surface.get_capabilities(adapter);
        // Not sRGB: the image's bytes are already encoded (no second gamma).
        let format = caps.formats.iter().copied().find(|f| !f.is_srgb()).unwrap_or(caps.formats[0]);
        // D27: no tearing and no queue of old frames; virtual time sets the pace.
        let present_mode = if caps.present_modes.contains(&wgpu::PresentMode::Mailbox) { wgpu::PresentMode::Mailbox } else { wgpu::PresentMode::AutoVsync };
        let mut config = surface.get_default_config(adapter, size.0.max(1), size.1.max(1)).expect("a surface the adapter can present to");
        config.format = format;
        config.present_mode = present_mode;
        config.desired_maximum_frame_latency = 1;
        surface.configure(device, &config);
        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor { label: Some("show"), source: wgpu::ShaderSource::Wgsl(SHOW.into()) });
        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: None,
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture { sample_type: wgpu::TextureSampleType::Float { filterable: false }, view_dimension: wgpu::TextureViewDimension::D2, multisampled: false },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry { binding: 1, visibility: wgpu::ShaderStages::FRAGMENT, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Uniform, has_dynamic_offset: false, min_binding_size: None }, count: None },
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor { label: None, bind_group_layouts: &[Some(&layout)], ..Default::default() });
        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("show"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState { module: &module, entry_point: Some("vs"), compilation_options: Default::default(), buffers: &[] },
            fragment: Some(wgpu::FragmentState { module: &module, entry_point: Some("fs"), compilation_options: Default::default(), targets: &[Some(format.into())] }),
            primitive: Default::default(),
            depth_stencil: None,
            multisample: Default::default(),
            multiview_mask: None,
            cache: None,
        });
        Window { surface, config, pipeline, layout }
    }
}

/// What the upload arena carried since the start: staged blocks (and those an
/// identical block already in the frame's arena served), their bytes by kind
/// (constant banks, other data), the zeros completing bank windows, and the
/// bytes and count of the arena uploads. Then the guest reads for the GPU by
/// kind ([`Read`]): reads and bytes, and those the submit's [`Memo`] served.
#[derive(Clone, Copy, Debug, Default)]
pub struct Staging {
    pub calls: u64,
    pub hits: u64,
    pub bank_bytes: u64,
    pub data_bytes: u64,
    pub zero_bytes: u64,
    pub flushed: u64,
    pub flushes: u64,
    pub reads: [u64; 5],
    pub read_bytes: [u64; 5],
    pub memo_hits: [u64; 5],
    pub memo_bytes: [u64; 5],
}

/// What a guest read for the GPU was for, an index into [`Staging`]'s arrays.
#[derive(Clone, Copy)]
pub(crate) enum Read {
    Vertex,
    Index,
    Bank,
    ComputeBank,
    Global,
}

/// A range of the upload arena: its buffer, offset and size.
pub(crate) type Slice = (wgpu::Buffer, u64, u64);

/// Staged indices: the range, their format and the highest index.
pub(crate) type Indices = (Slice, wgpu::IndexFormat, u32);

/// A memo value and the write epoch it was made at.
type Dated<T> = (T, u64);

/// A fast hasher for host maps keyed by addresses and small tuples, where
/// SipHash (the default) costs more than the lookup; fixed, so iteration
/// orders are the same in every run.
#[derive(Default, Clone, Copy)]
pub(crate) struct Fx(u64);

impl Fx {
    fn add(&mut self, w: u64) {
        self.0 = (self.0.rotate_left(5) ^ w).wrapping_mul(0x517c_c1b7_2722_0a95);
    }
}

impl std::hash::Hasher for Fx {
    fn finish(&self) -> u64 {
        self.0
    }
    fn write(&mut self, bytes: &[u8]) {
        let (words, tail) = bytes.as_chunks::<8>();
        for w in words {
            self.add(u64::from_le_bytes(*w));
        }
        for &b in tail {
            self.add(u64::from(b));
        }
    }
    fn write_u8(&mut self, n: u8) {
        self.add(n.into());
    }
    fn write_u32(&mut self, n: u32) {
        self.add(n.into());
    }
    fn write_u64(&mut self, n: u64) {
        self.add(n);
    }
    fn write_usize(&mut self, n: usize) {
        self.add(n as u64);
    }
}

/// A map hashed with [`Fx`].
pub(crate) type FxMap<K, V> = HashMap<K, V, std::hash::BuildHasherDefault<Fx>>;

/// Bytes read from guest memory, shared by the draws that read them.
type Bytes = Arc<Vec<u8>>;

/// What this submit already read from guest memory for the GPU, and staged:
/// most reads repeat (the same constant banks and meshes for many draws), and a
/// repeat skips the read, the hash and the copy. A GPU write stamps the 4 KiB
/// pages it touches with a new epoch; an entry is stale once one of its pages
/// was stamped after it was made. Host state only, emptied at each submit and
/// flush (guest code may have written in between).
#[derive(Default)]
pub(crate) struct Memo {
    /// (kind, address, length) -> the bytes.
    reads: FxMap<(u8, u64, usize), Dated<Bytes>>,
    /// (kind, address, length, filled to, bound as) -> the staged block.
    staged: FxMap<(u8, u64, usize, usize, usize), Dated<Slice>>,
    /// (address, index size, count, restart index) -> the staged indices.
    indices: FxMap<(u64, usize, u32, Option<u32>), Dated<Indices>>,
    /// The epoch of each page written by the GPU in this submit, in page order:
    /// an entry's check visits only the written pages inside it.
    pages: BTreeMap<u64, u64>,
    epoch: u64,
}

impl Memo {
    fn clear(&mut self) {
        self.reads.clear();
        self.staged.clear();
        self.indices.clear();
        self.pages.clear();
    }

    /// The GPU wrote `[start, start + len)`.
    pub(crate) fn wrote(&mut self, start: u64, len: u64) {
        if len == 0 {
            return;
        }
        self.epoch += 1;
        for page in start >> 12..=start.saturating_add(len - 1) >> 12 {
            self.pages.insert(page, self.epoch);
        }
    }

    /// No GPU write on `[addr, addr + len)`'s pages since `epoch`.
    fn fresh(&self, addr: u64, len: usize, epoch: u64) -> bool {
        let (first, last) = (addr >> 12, addr.saturating_add(len.max(1) as u64 - 1) >> 12);
        self.pages.range(first..=last).all(|(_, e)| *e <= epoch)
    }
}

/// Offsets in the upload arena: the uniform binding alignment wgpu allows at most.
const ARENA_ALIGN: usize = 256;

/// What the game's programs cannot run without (D32): subgroup operations,
/// on subgroups of exactly 32 lanes, since the translated warp code (shuffles,
/// the upscaler's matrix fragments) assumes the console's warp size.
const NEEDED: wgpu::Features = wgpu::Features::SUBGROUP;

/// Refuses an adapter that lacks what the game needs, with the reason.
pub fn check_adapter(features: wgpu::Features, subgroup_min: u32, subgroup_max: u32) -> Result<(), String> {
    let missing = NEEDED - features;
    if !missing.is_empty() {
        return Err(format!("this GPU lacks {missing:?}, which the game's shaders need"));
    }
    if (subgroup_min, subgroup_max) != (32, 32) {
        return Err(format!("this GPU runs subgroups of {subgroup_min} to {subgroup_max} lanes; the game's shaders need exactly 32"));
    }
    Ok(())
}

/// A uniform binding of a range of the upload arena.
pub(crate) fn uniform((buffer, at, size): &(wgpu::Buffer, u64, u64)) -> wgpu::BindingResource<'_> {
    wgpu::BindingResource::Buffer(wgpu::BufferBinding { buffer, offset: *at, size: wgpu::BufferSize::new(*size) })
}

fn arena_buffer(device: &wgpu::Device, size: u64) -> wgpu::Buffer {
    let usage = wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::INDEX | wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST;
    device.create_buffer(&wgpu::BufferDescriptor { label: Some("arena"), size, usage, mapped_at_creation: false })
}

use std::sync::{Arc, Mutex};

impl Surfaces {
    /// A headless device with the adapter's own limits (targets reach 11264
    /// pixels) and, when available, the 16-bit normalized, D32S8 and BC
    /// formats, filterable f32 textures, border-color clamping and subgroup
    /// (quad) operations.
    pub fn headless(backend: crate::Backend) -> Result<Surfaces, String> {
        Surfaces::open(&crate::Renderer::instance_for(backend), None)
    }

    /// The same, also presenting into a window's `surface` (made from
    /// `instance` on the window's thread; `size` in pixels): each present
    /// shows the image there ([`Surfaces::show`]).
    pub fn windowed(instance: &wgpu::Instance, surface: wgpu::Surface<'static>, size: (u32, u32)) -> Result<Surfaces, String> {
        Surfaces::open(instance, Some((surface, size)))
    }

    fn open(instance: &wgpu::Instance, window: Option<(wgpu::Surface<'static>, (u32, u32))>) -> Result<Surfaces, String> {
        let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            compatible_surface: window.as_ref().map(|w| &w.0),
            force_fallback_adapter: false,
            apply_limit_buckets: false,
        }))
        .map_err(|e| format!("no suitable GPU adapter: {e}"))?;
        // 16-bit normalized targets are renderable only through the adapter's own
        // format features (wgpu guarantees them as storage only).
        let wanted = wgpu::Features::TEXTURE_FORMAT_16BIT_NORM
            | wgpu::Features::DEPTH32FLOAT_STENCIL8
            | wgpu::Features::TEXTURE_ADAPTER_SPECIFIC_FORMAT_FEATURES
            | wgpu::Features::RG11B10UFLOAT_RENDERABLE
            | wgpu::Features::TEXTURE_COMPRESSION_BC
            | wgpu::Features::FLOAT32_FILTERABLE
            | wgpu::Features::ADDRESS_MODE_CLAMP_TO_BORDER
            | wgpu::Features::SUBGROUP
            | crate::timing::Timing::wanted();
        let info = adapter.get_info();
        check_adapter(adapter.features(), info.subgroup_min_size, info.subgroup_max_size).map_err(|e| format!("{} ({:?}): {e}", info.name, info.backend))?;
        let descriptor = wgpu::DeviceDescriptor { required_features: (adapter.features() & wanted) | NEEDED, required_limits: adapter.limits(), ..Default::default() };
        let (device, queue) = pollster::block_on(adapter.request_device(&descriptor)).map_err(|e| format!("cannot open the GPU device: {e}"))?;
        let errors = Arc::new(Mutex::new(Vec::new()));
        let sink = errors.clone();
        device.on_uncaptured_error(Arc::new(move |e: wgpu::Error| {
            if let Ok(mut v) = sink.lock()
                && v.len() < 16
            {
                // Debug: the Display alone is "Validation Error", without its description.
                v.push(format!("{e:?}"));
            }
        }));
        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor { label: Some("blit"), source: wgpu::ShaderSource::Wgsl(BLIT.into()) });
        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: None,
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Texture { sample_type: wgpu::TextureSampleType::Float { filterable: false }, view_dimension: wgpu::TextureViewDimension::D2, multisampled: false },
                count: None,
            }],
        });
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor { label: None, bind_group_layouts: &[Some(&layout)], ..Default::default() });
        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("blit"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState { module: &module, entry_point: Some("vs"), compilation_options: Default::default(), buffers: &[] },
            fragment: Some(wgpu::FragmentState { module: &module, entry_point: Some("fs"), compilation_options: Default::default(), targets: &[Some(wgpu::TextureFormat::Rgba8Unorm.into())] }),
            primitive: Default::default(),
            depth_stencil: None,
            multisample: Default::default(),
            multiview_mask: None,
            cache: None,
        });
        let fill_module = device.create_shader_module(wgpu::ShaderModuleDescriptor { label: Some("fill"), source: wgpu::ShaderSource::Wgsl(FILL.into()) });
        let fill_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: None,
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::VERTEX_FRAGMENT,
                ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Uniform, has_dynamic_offset: false, min_binding_size: wgpu::BufferSize::new(32) },
                count: None,
            }],
        });
        let fill = (fill_module, fill_layout, Default::default());
        let arena = (arena_buffer(&device, 16 << 20), Vec::new());
        let window = window.map(|(surface, size)| Window::new(&adapter, &device, surface, size));
        let timing = crate::timing::Timing::new(&device, &queue);
        Ok(Surfaces { adapter, device, queue, map: BTreeMap::new(), stamp: 0, touched: BTreeSet::new(), blit: (pipeline, layout), fill, cache: Default::default(), drawn: 0, textures_read: Vec::new(), skipped: BTreeMap::new(), errors, pending: None, timing, clock: None, present: 0, readbacks: Default::default(), hasher: None, arena, staging: Staging::default(), staged: FxMap::default(), last_color: None, window, memo: Memo::default() })
    }

    /// Shows the presented image (the color target written last, its top-left
    /// `area`) in the window, scaled to `size` with point sampling. Nothing
    /// happens without a window, with nothing written, or while minimized.
    pub fn show(&mut self, area: (u32, u32), size: (u32, u32)) {
        self.flush();
        let (Some(w), Some(key)) = (self.window.as_mut(), self.last_color) else { return };
        let Some(e) = self.map.get(&key) else { return };
        if size.0 == 0 || size.1 == 0 || area.0 == 0 || area.1 == 0 {
            return;
        }
        if (w.config.width, w.config.height) != size {
            (w.config.width, w.config.height) = size;
            w.surface.configure(&self.device, &w.config);
        }
        let frame = match w.surface.get_current_texture() {
            wgpu::CurrentSurfaceTexture::Success(t) | wgpu::CurrentSurfaceTexture::Suboptimal(t) => t,
            // Resized or lost under us: configure again, show the next frame.
            _ => {
                w.surface.configure(&self.device, &w.config);
                return;
            }
        };
        let area = (area.0.min(e.surface.width), area.1.min(e.surface.height));
        let words: Vec<u8> = [area.0, area.1, size.0, size.1].iter().flat_map(|v| v.to_le_bytes()).collect();
        let params = wgpu::util::DeviceExt::create_buffer_init(&self.device, &wgpu::util::BufferInitDescriptor { label: Some("show"), contents: &words, usage: wgpu::BufferUsages::UNIFORM });
        let src = e.texture.create_view(&wgpu::TextureViewDescriptor { dimension: Some(wgpu::TextureViewDimension::D2), array_layer_count: Some(1), ..Default::default() });
        let group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &w.layout,
            entries: &[wgpu::BindGroupEntry { binding: 0, resource: wgpu::BindingResource::TextureView(&src) }, wgpu::BindGroupEntry { binding: 1, resource: params.as_entire_binding() }],
        });
        let view = frame.texture.create_view(&Default::default());
        let mut encoder = self.device.create_command_encoder(&Default::default());
        {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("show"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment { view: &view, depth_slice: None, resolve_target: None, ops: wgpu::Operations { load: wgpu::LoadOp::Clear(wgpu::Color::BLACK), store: wgpu::StoreOp::Store } })],
                ..Default::default()
            });
            pass.set_pipeline(&w.pipeline);
            pass.set_bind_group(0, &group, &[]);
            pass.draw(0..3, 0..1);
        }
        self.queue.submit([encoder.finish()]);
        self.queue.present(frame);
    }

    /// The encoder that collects a frame's passes: one submit per frame
    /// instead of one per operation. Give it back with [`Surfaces::record`].
    pub(crate) fn encoder(&mut self) -> wgpu::CommandEncoder {
        self.pending.take().unwrap_or_else(|| {
            let mut encoder = self.device.create_command_encoder(&Default::default());
            if let Some(c) = &self.clock {
                c.open(&mut encoder);
            }
            encoder
        })
    }

    /// Times each frame's GPU work from now on (`--perf`); false if the GPU
    /// has no timestamps inside command encoders.
    pub fn measure_frames(&mut self) -> bool {
        self.clock = crate::timing::FrameClock::new(&self.device, &self.queue);
        self.clock.is_some()
    }

    /// Waits for the GPU; each present's GPU milliseconds (index: present) and
    /// how many flushes had to wait for the GPU to free a timing slot.
    pub fn frame_times(&mut self) -> (Vec<f64>, u64) {
        self.flush();
        self.clock.as_mut().map_or((Vec::new(), 0), |c| (c.finish(&self.device), c.stalls))
    }

    pub(crate) fn record(&mut self, encoder: wgpu::CommandEncoder) {
        self.pending = Some(encoder);
    }

    /// Submits the work recorded so far (at each present, and before any
    /// readback). Uploads (`write_texture`) only ever go to new textures,
    /// which nothing recorded before can use, so they need no flush.
    /// Counts a guest read for the GPU, `memo` if the memo served it (host
    /// statistics, see [`Staging`]).
    pub(crate) fn note_read(&mut self, kind: Read, len: usize, memo: bool) {
        let (k, s) = (kind as usize, &mut self.staging);
        s.reads[k] += 1;
        s.read_bytes[k] += len as u64;
        s.memo_hits[k] += u64::from(memo);
        s.memo_bytes[k] += if memo { len as u64 } else { 0 };
    }

    /// Guest code may have run since the last GPU work (a new submit, or a
    /// wait that ended): what was read from guest memory may have changed.
    pub fn new_submit(&mut self) {
        self.memo.clear();
    }

    /// `len` bytes of guest memory at `addr`, read through `mem` unless this
    /// submit read them already and the GPU has not written them since; None
    /// if they are not readable.
    pub(crate) fn read_guest(&mut self, kind: Read, addr: u64, len: usize, mem: crate::draw::Mem) -> Option<Arc<Vec<u8>>> {
        let key = (kind as u8, addr, len);
        if let Some((bytes, epoch)) = self.memo.reads.get(&key)
            && self.memo.fresh(addr, len, *epoch)
        {
            let bytes = bytes.clone();
            self.note_read(kind, len, true);
            return Some(bytes);
        }
        self.note_read(kind, len, false);
        let mut bytes = vec![0u8; len];
        if !mem(addr, &mut bytes) {
            return None;
        }
        let bytes = Arc::new(bytes);
        self.memo.reads.insert(key, (bytes.clone(), self.memo.epoch));
        Some(bytes)
    }

    /// As [`Surfaces::stage_window`], for `bytes` read from guest memory at
    /// `addr`: staged once a submit while the GPU does not write them.
    pub(crate) fn stage_guest(&mut self, kind: Read, addr: u64, bytes: &[u8], zeros: usize, window: usize) -> Slice {
        let key = (kind as u8, addr, bytes.len(), zeros, window);
        if let Some((slice, epoch)) = self.memo.staged.get(&key)
            && self.memo.fresh(addr, bytes.len(), *epoch)
        {
            return slice.clone();
        }
        let slice = self.stage_window(bytes, zeros, window);
        self.memo.staged.insert(key, (slice.clone(), self.memo.epoch));
        slice
    }

    /// The block [`Surfaces::stage_guest`] staged for these arguments, if this
    /// submit made it and the GPU has not written its bytes since (the read is
    /// counted in [`Staging`]).
    pub(crate) fn staged_guest(&mut self, kind: Read, addr: u64, len: usize, zeros: usize, window: usize) -> Option<Slice> {
        let found = self.memo.staged.get(&(kind as u8, addr, len, zeros, window)).filter(|(_, epoch)| self.memo.fresh(addr, len, *epoch)).map(|(slice, _)| slice.clone());
        self.note_read(kind, len, found.is_some());
        found
    }

    /// The staged indices of `count` indices of `size` bytes at `addr` (with a
    /// restart index), their format and the highest index, if this submit made
    /// them and the GPU has not written them since.
    pub(crate) fn staged_indices(&mut self, addr: u64, size: usize, count: u32, restart: Option<u32>) -> Option<Indices> {
        let len = size * count as usize;
        let found = self.memo.indices.get(&(addr, size, count, restart)).filter(|(_, epoch)| self.memo.fresh(addr, len, *epoch)).map(|(indices, _)| indices.clone());
        self.note_read(Read::Index, len, found.is_some());
        found
    }

    /// Records indices staged by the caller (see [`Surfaces::staged_indices`]).
    pub(crate) fn remember_indices(&mut self, addr: u64, size: usize, count: u32, restart: Option<u32>, staged: Indices) {
        self.memo.indices.insert((addr, size, count, restart), (staged, self.memo.epoch));
    }

    pub fn flush(&mut self) {
        self.memo.clear();
        // The arena's writes land before this submit's commands, and after the
        // previous frame's, which read the same bytes: queue order.
        if !self.arena.1.is_empty() {
            self.staging.flushed += self.arena.1.len() as u64;
            self.staging.flushes += 1;
            self.queue.write_buffer(&self.arena.0, 0, &self.arena.1);
            self.arena.1.clear();
            self.staged.clear();
        }
        if let Some(mut e) = self.pending.take() {
            if let Some(t) = &self.timing {
                t.resolve(&mut e);
            }
            if let Some(c) = &mut self.clock {
                c.close(&self.device, &mut e);
            }
            self.queue.submit([e.finish()]);
            if let Some(t) = &mut self.timing {
                t.collect(&self.device);
            }
            if let Some(c) = &mut self.clock {
                c.submitted(&self.device, self.present);
            }
        }
    }

    /// Copies `bytes` (padded to 4) into this frame's upload arena; returns
    /// the buffer and the range they will occupy when the frame is submitted.
    pub(crate) fn stage(&mut self, bytes: &[u8]) -> (wgpu::Buffer, u64, u64) {
        self.stage_window(bytes, 0, 0)
    }

    /// As [`Surfaces::stage`], the range given being at least `window` bytes: a
    /// constant bank is bound as 64 KiB but only its bound size is hashed and
    /// compared; the rest of its window is zeros, as the console reads past a bank's
    /// size (text vanished when it held the arena's next blocks).
    pub(crate) fn stage_window(&mut self, bytes: &[u8], zeros: usize, window: usize) -> (wgpu::Buffer, u64, u64) {
        let size = bytes.len().next_multiple_of(4).max(4);
        // Filled: the bytes, then zeros up to `zeros`; bound: `window` at least.
        let fill = size.max(zeros);
        let range = fill.max(window);
        // A fast word hash (SipHash cost 5% of a tutorial frame); bytes are compared after.
        let (words, tail) = bytes.as_chunks::<8>();
        let hash = words.iter().map(|w| u64::from_le_bytes(*w)).chain(tail.iter().map(|&b| u64::from(b))).fold(bytes.len() as u64, |h, w| (h.rotate_left(5) ^ w).wrapping_mul(0x517c_c1b7_2722_0a95));
        let capacity = self.arena.0.size();
        self.staging.calls += 1;
        if let Some(&(at, _)) = self.staged.get(&hash).and_then(|v| v.iter().find(|&&(at, filled)| filled >= fill && (at + range) as u64 <= capacity && self.arena.1.get(at..at + bytes.len()) == Some(bytes))) {
            self.staging.hits += 1;
            return (self.arena.0.clone(), at as u64, range as u64);
        }
        let at = self.arena.1.len().next_multiple_of(ARENA_ALIGN);
        if (at + range) as u64 > capacity {
            // Full: its bytes are written now (it is not reused this frame) and a
            // buffer twice as large takes over; recorded work keeps the old one alive.
            self.staging.flushed += self.arena.1.len() as u64;
            self.queue.write_buffer(&self.arena.0, 0, &self.arena.1);
            let capacity = (capacity * 2).max((range as u64).next_power_of_two());
            self.arena = (arena_buffer(&self.device, capacity), Vec::new());
            self.staged.clear();
            // ponytail: staged blocks stay valid in the old buffer; dropped for simplicity (rare).
            self.memo.staged.clear();
            self.memo.indices.clear();
            return self.stage_window(bytes, zeros, window);
        }
        match window {
            0 => self.staging.data_bytes += size as u64,
            _ => self.staging.bank_bytes += size as u64,
        }
        self.staging.zero_bytes += (fill - size) as u64;
        let data = &mut self.arena.1;
        data.resize(at, 0);
        data.extend_from_slice(bytes);
        data.resize(at + fill, 0);
        self.staged.entry(hash).or_default().push((at, fill));
        (self.arena.0.clone(), at as u64, range as u64)
    }

    /// A color target was written.
    pub(crate) fn wrote(&mut self, key: (u64, u32)) {
        self.touched.insert(key);
        self.last_color = Some(key);
        self.mark(key);
    }

    /// A target (color or depth) was written: it is now the latest at its address.
    pub(crate) fn mark(&mut self, key: (u64, u32)) {
        self.stamp += 1;
        if let Some(e) = self.map.get_mut(&key) {
            e.written = self.stamp;
        }
    }

    /// The target a read at `addr` sees in `map`: the one written last there.
    pub(crate) fn latest_at(map: &BTreeMap<(u64, u32), Entry>, addr: u64) -> Option<&Entry> {
        map.range((addr, 0)..=(addr, u32::MAX)).map(|(_, e)| e).max_by_key(|e| e.written)
    }

    /// Starts hashing the image present `present` shows: the color target written
    /// last, its top-left `width` × `height` texels, raw bytes (64-bit FNV-1a).
    /// The copy is read back when the GPU gets to it, and hashed on a helper
    /// thread: the frame never waits ([`Surfaces::presented_hashes`]). False if
    /// nothing was written or the target is smaller.
    pub fn hash_presented(&mut self, present: u64, width: u32, height: u32) -> bool {
        self.flush();
        let Some(e) = self.last_color.and_then(|k| self.map.get(&k)) else { return false };
        let Some(bpp) = e.texture.format().block_copy_size(None) else { return false };
        if width == 0 || height == 0 || width > e.surface.width || height > e.surface.height {
            return false;
        }
        let row = width * bpp;
        let padded = row.next_multiple_of(wgpu::COPY_BYTES_PER_ROW_ALIGNMENT);
        let buffer = self.device.create_buffer(&wgpu::BufferDescriptor { label: Some("hash"), size: u64::from(padded * height), usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ, mapped_at_creation: false });
        let mut encoder = self.device.create_command_encoder(&Default::default());
        encoder.copy_texture_to_buffer(
            e.texture.as_image_copy(),
            wgpu::TexelCopyBufferInfo { buffer: &buffer, layout: wgpu::TexelCopyBufferLayout { offset: 0, bytes_per_row: Some(padded), rows_per_image: Some(height) } },
            wgpu::Extent3d { width, height, depth_or_array_layers: 1 },
        );
        let target = e.surface.addr;
        self.queue.submit([encoder.finish()]);
        let state = Arc::new(std::sync::atomic::AtomicU8::new(1));
        let done = state.clone();
        buffer.slice(..).map_async(wgpu::MapMode::Read, move |r| done.store(if r.is_ok() { 2 } else { 3 }, std::sync::atomic::Ordering::Release));
        self.readbacks.push_back(Readback { present, target, buffer, row, padded, height, state });
        // ponytail: a few images in flight bound the memory; the GPU rarely lags more.
        self.collect_hashes(self.readbacks.len() > 8);
        true
    }

    /// Hands the images back from the GPU to the hasher, oldest first; with
    /// `wait`, waits for the GPU first.
    fn collect_hashes(&mut self, wait: bool) {
        let _ = self.device.poll(if wait { wgpu::PollType::wait_indefinitely() } else { wgpu::PollType::Poll });
        while let Some(r) = self.readbacks.front() {
            match r.state.load(std::sync::atomic::Ordering::Acquire) {
                2 => {
                    let mut bytes = Vec::with_capacity((r.row * r.height) as usize);
                    if let Ok(mapped) = r.buffer.slice(..).get_mapped_range() {
                        for line in mapped.chunks(r.padded as usize) {
                            bytes.extend_from_slice(&line[..r.row as usize]);
                        }
                    }
                    r.buffer.unmap();
                    let (tx, _) = self.hasher.get_or_insert_with(|| {
                        let (tx, rx) = std::sync::mpsc::channel::<(u64, u64, Vec<u8>)>();
                        (tx, std::thread::spawn(move || rx.into_iter().map(|(present, target, bytes)| (present, target, fnv(&bytes))).collect()))
                    });
                    let _ = tx.send((r.present, r.target, bytes));
                }
                3 => eprintln!("gyroship: --hashes: present {}: the image could not be read back", r.present),
                _ => break,
            }
            self.readbacks.pop_front();
        }
    }

    /// Waits for every image in flight; (present, target, hash) of each image
    /// hashed, by present.
    pub fn presented_hashes(&mut self) -> Vec<(u64, u64, u64)> {
        self.collect_hashes(true);
        let Some((tx, hasher)) = self.hasher.take() else { return Vec::new() };
        drop(tx);
        let mut all = hasher.join().unwrap_or_default();
        all.sort_unstable();
        all
    }

    /// The GPU and backend in use, for the journal.
    pub fn describe(&self) -> String {
        let info = self.adapter.get_info();
        format!("{} ({:?}, driver {} {})", info.name, info.backend, info.driver, info.driver_info)
    }

    /// Whether this device can render to `format` (and read it back).
    pub(crate) fn renderable(&self, format: wgpu::TextureFormat) -> bool {
        let features = if self.device.features().contains(wgpu::Features::TEXTURE_ADAPTER_SPECIFIC_FORMAT_FEATURES) {
            self.adapter.get_texture_format_features(format)
        } else {
            format.guaranteed_format_features(self.device.features())
        };
        features.allowed_usages.contains(wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_SRC)
    }

    pub(crate) fn skip(&mut self, why: impl Into<String>) {
        *self.skipped.entry(why.into()).or_default() += 1;
    }

    /// The texture of a target, made (or remade, if its size changed) on demand.
    pub(crate) fn texture(&mut self, s: &Surface, format: wgpu::TextureFormat) -> wgpu::TextureView {
        let key = (s.addr, s.format);
        let stale = self.map.get(&key).is_none_or(|e| e.surface != *s);
        if stale {
            // Also a storage texture where the format allows it: compute programs may work on a
            // target (the game's upscaler writes the layer its interface is then drawn on). An
            // sRGB target is kept in its linear format, rendered through an sRGB view: wgpu has
            // no sRGB storage.
            let usages = |f: wgpu::TextureFormat| self.adapter.get_texture_format_features(f).allowed_usages;
            let linear = format.remove_srgb_suffix();
            let base = if linear != format && usages(linear).contains(wgpu::TextureUsages::STORAGE_BINDING) { linear } else { format };
            let storage = usages(base) & wgpu::TextureUsages::STORAGE_BINDING;
            let texture = self.device.create_texture(&wgpu::TextureDescriptor {
                label: None,
                size: wgpu::Extent3d { width: s.width.max(1), height: s.height.max(1), depth_or_array_layers: s.layers },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: base,
                usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_SRC | storage,
                view_formats: if base == format { &[] } else { std::slice::from_ref(&format) },
            });
            self.map.insert(key, Entry { surface: *s, texture, written: 0 });
        }
        view_as(&self.map[&key].texture, Some(format), wgpu::TextureUsages::RENDER_ATTACHMENT, wgpu::TextureViewDescriptor { dimension: Some(wgpu::TextureViewDimension::D2), array_layer_count: Some(1), ..Default::default() })
    }

    /// Runs a clear on the host textures.
    pub fn clear(&mut self, c: &Clear) {
        let covers = |s: &Surface| c.scissor.is_none_or(|[x, y, w, h]| x == 0 && y == 0 && w >= s.width && h >= s.height);
        if c.layer != 0 {
            return self.skip("clear of a layer other than 0");
        }
        if let Some((s, value, mask)) = &c.color {
            match color_format(s.format) {
                None => self.skip("color clear: unsupported format"),
                Some(f) if !self.renderable(f) => self.skip("color clear: format not renderable here"),
                Some(format) if *mask != 0xf || !covers(s) => {
                    let color = value.map(f32::from_bits);
                    if self.fill(s, format, Fill::Color { mask: *mask }, c.scissor, color, 0.0, 0) {
                        self.wrote((s.addr, s.format));
                    }
                }
                Some(format) => {
                    let view = self.texture(s, format);
                    let [r, g, b, a] = value.map(|v| f64::from(f32::from_bits(v)));
                    let mut encoder = self.encoder();
                    encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                        label: Some("clear"),
                        color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                            view: &view,
                            depth_slice: None,
                            resolve_target: None,
                            ops: wgpu::Operations { load: wgpu::LoadOp::Clear(wgpu::Color { r, g, b, a }), store: wgpu::StoreOp::Store },
                        })],
                        ..Default::default()
                    });
                    self.record(encoder);
                    self.wrote((s.addr, s.format));
                }
            }
        }
        if let Some((s, z, stencil)) = &c.depth {
            match depth_format(s.format) {
                None => self.skip("depth clear: unsupported format"),
                Some(f) if !self.renderable(f) => self.skip("depth clear: format not renderable here"),
                Some(format) if !covers(s) => {
                    let writes = z.is_some() || (stencil.is_some() && format.has_stencil_aspect());
                    let fill = Fill::Depth { depth: z.is_some(), stencil: stencil.is_some() && format.has_stencil_aspect() };
                    if self.fill(s, format, fill, c.scissor, [0.0; 4], z.map_or(0.0, f32::from_bits), stencil.unwrap_or(0)) && writes {
                        self.mark((s.addr, s.format));
                    }
                }
                Some(format) => {
                    let view = self.texture(s, format);
                    let depth_ops = Some(wgpu::Operations { load: z.map_or(wgpu::LoadOp::Load, |z| wgpu::LoadOp::Clear(f32::from_bits(z))), store: wgpu::StoreOp::Store });
                    let stencil_ops = format.has_stencil_aspect().then(|| wgpu::Operations { load: stencil.map_or(wgpu::LoadOp::Load, |v| wgpu::LoadOp::Clear(u32::from(v))), store: wgpu::StoreOp::Store });
                    let mut encoder = self.encoder();
                    encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                        label: Some("clear"),
                        depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment { view: &view, depth_ops, stencil_ops }),
                        ..Default::default()
                    });
                    self.record(encoder);
                    if z.is_some() || (stencil.is_some() && format.has_stencil_aspect()) {
                        self.mark((s.addr, s.format));
                    }
                }
            }
        }
    }

    /// Fills `rect` (all of the target when None) of a target: a color
    /// with a channel mask, or a depth and/or stencil value. False when the
    /// rect clips to nothing (no pass recorded).
    #[allow(clippy::too_many_arguments)]
    fn fill(&mut self, s: &Surface, format: wgpu::TextureFormat, fill: Fill, rect: Option<[u32; 4]>, color: [f32; 4], depth: f32, stencil: u8) -> bool {
        let [x, y, w, h] = rect.unwrap_or([0, 0, s.width, s.height]);
        let (x1, y1) = (x.saturating_add(w).min(s.width), y.saturating_add(h).min(s.height));
        if x >= x1 || y >= y1 {
            return false;
        }
        let pipeline = match self.fill.2.get(&(format, fill)) {
            Some(p) => p.clone(),
            None => {
                let layout = self.device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor { label: None, bind_group_layouts: &[Some(&self.fill.1)], ..Default::default() });
                let (targets, depth_stencil) = match fill {
                    Fill::Color { mask } => (vec![Some(wgpu::ColorTargetState { format, blend: None, write_mask: wgpu::ColorWrites::from_bits_truncate(u32::from(mask)) })], None),
                    Fill::Depth { depth, stencil } => {
                        let face = wgpu::StencilFaceState { compare: wgpu::CompareFunction::Always, fail_op: wgpu::StencilOperation::Replace, depth_fail_op: wgpu::StencilOperation::Replace, pass_op: wgpu::StencilOperation::Replace };
                        let stencil = if stencil { wgpu::StencilState { front: face, back: face, read_mask: 0xff, write_mask: 0xff } } else { Default::default() };
                        (Vec::new(), Some(wgpu::DepthStencilState { format, depth_write_enabled: Some(depth), depth_compare: Some(wgpu::CompareFunction::Always), stencil, bias: Default::default() }))
                    }
                };
                let p = self.device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                    label: Some("fill"),
                    layout: Some(&layout),
                    vertex: wgpu::VertexState { module: &self.fill.0, entry_point: Some("vs"), compilation_options: Default::default(), buffers: &[] },
                    fragment: (!targets.is_empty()).then(|| wgpu::FragmentState { module: &self.fill.0, entry_point: Some("fs"), compilation_options: Default::default(), targets: &targets }),
                    primitive: Default::default(),
                    depth_stencil,
                    multisample: Default::default(),
                    multiview_mask: None,
                    cache: None,
                });
                self.fill.2.insert((format, fill), p.clone());
                p
            }
        };
        let mut bytes: Vec<u8> = color.iter().chain(&[depth, 0.0, 0.0, 0.0]).flat_map(|v| v.to_le_bytes()).collect();
        bytes.truncate(32);
        let u = self.stage(&bytes);
        let group = self.device.create_bind_group(&wgpu::BindGroupDescriptor { label: None, layout: &self.fill.1, entries: &[wgpu::BindGroupEntry { binding: 0, resource: uniform(&u) }] });
        let view = self.texture(s, format);
        let load = wgpu::Operations { load: wgpu::LoadOp::Load, store: wgpu::StoreOp::Store };
        let mut encoder = self.encoder();
        {
            let color_attachment = [Some(wgpu::RenderPassColorAttachment { view: &view, depth_slice: None, resolve_target: None, ops: load })];
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("fill"),
                color_attachments: if matches!(fill, Fill::Color { .. }) { &color_attachment } else { &[] },
                depth_stencil_attachment: matches!(fill, Fill::Depth { .. }).then(|| wgpu::RenderPassDepthStencilAttachment {
                    view: &view,
                    depth_ops: Some(wgpu::Operations { load: wgpu::LoadOp::Load, store: wgpu::StoreOp::Store }),
                    stencil_ops: format.has_stencil_aspect().then_some(wgpu::Operations { load: wgpu::LoadOp::Load, store: wgpu::StoreOp::Store }),
                }),
                ..Default::default()
            });
            pass.set_pipeline(&pipeline);
            pass.set_bind_group(0, &group, &[]);
            pass.set_scissor_rect(x, y, x1 - x, y1 - y);
            pass.set_stencil_reference(u32::from(stencil));
            pass.draw(0..3, 0..1);
        }
        self.record(encoder);
        true
    }

    /// Writes every color target touched since the last call as
    /// `{prefix}-{address}-{format}-{w}x{h}.png` in `dir`; returns how many.
    pub fn dump(&mut self, dir: &Path, prefix: &str) -> Result<usize, String> {
        std::fs::create_dir_all(dir).map_err(|e| format!("{}: {e}", dir.display()))?;
        self.flush();
        let mut keys = std::mem::take(&mut self.touched);
        // GYROSHIP_DUMP_ONLY=<hex address>: that target alone (dumps every few frames).
        if let Some(only) = std::env::var("GYROSHIP_DUMP_ONLY").ok().and_then(|v| u64::from_str_radix(v.trim_start_matches("0x"), 16).ok()) {
            keys.retain(|k| k.0 == only);
        }
        for key in &keys {
            let e = &self.map[key];
            let (w, h) = (e.surface.width.max(1), e.surface.height.max(1));
            let rgba = self.rgba(e)?;
            let name = format!("{prefix}-{:010x}-{:02x}-{w}x{h}.png", e.surface.addr, e.surface.format);
            crate::write_png(&dir.join(name), w, h, &rgba).map_err(|e| e.to_string())?;
        }
        Ok(keys.len())
    }

    /// Forgets which targets were written (a frame not dumped).
    pub fn untouch(&mut self) {
        self.touched.clear();
    }

    /// Layer 0 of a color target as tightly packed RGBA8.
    fn rgba(&self, e: &Entry) -> Result<Vec<u8>, String> {
        let (w, h) = (e.surface.width.max(1), e.surface.height.max(1));
        let out = self.device.create_texture(&wgpu::TextureDescriptor {
            label: Some("blit"),
            size: wgpu::Extent3d { width: w, height: h, depth_or_array_layers: 1 },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8Unorm,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
            view_formats: &[],
        });
        let src = e.texture.create_view(&wgpu::TextureViewDescriptor { dimension: Some(wgpu::TextureViewDimension::D2), array_layer_count: Some(1), ..Default::default() });
        let group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &self.blit.1,
            entries: &[wgpu::BindGroupEntry { binding: 0, resource: wgpu::BindingResource::TextureView(&src) }],
        });
        let row = w * 4;
        let padded = row.div_ceil(wgpu::COPY_BYTES_PER_ROW_ALIGNMENT) * wgpu::COPY_BYTES_PER_ROW_ALIGNMENT;
        let buffer = self.device.create_buffer(&wgpu::BufferDescriptor { label: Some("readback"), size: u64::from(padded * h), usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ, mapped_at_creation: false });
        let mut encoder = self.device.create_command_encoder(&Default::default());
        {
            let view = out.create_view(&Default::default());
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("blit"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment { view: &view, depth_slice: None, resolve_target: None, ops: wgpu::Operations { load: wgpu::LoadOp::Clear(wgpu::Color::BLACK), store: wgpu::StoreOp::Store } })],
                ..Default::default()
            });
            pass.set_pipeline(&self.blit.0);
            pass.set_bind_group(0, &group, &[]);
            pass.draw(0..3, 0..1);
        }
        encoder.copy_texture_to_buffer(
            out.as_image_copy(),
            wgpu::TexelCopyBufferInfo { buffer: &buffer, layout: wgpu::TexelCopyBufferLayout { offset: 0, bytes_per_row: Some(padded), rows_per_image: Some(h) } },
            out.size(),
        );
        self.queue.submit([encoder.finish()]);
        let slice = buffer.slice(..);
        slice.map_async(wgpu::MapMode::Read, |_| ());
        self.device.poll(wgpu::PollType::wait_indefinitely()).map_err(|e| format!("GPU poll failed: {e}"))?;
        let mapped = slice.get_mapped_range().map_err(|e| format!("readback mapping failed: {e:?}"))?;
        let mut pixels = Vec::with_capacity((row * h) as usize);
        for y in 0..h {
            let start = (y * padded) as usize;
            pixels.extend_from_slice(&mapped[start..start + row as usize]);
        }
        Ok(pixels)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Synthetic adapter descriptions: subgroups of 32 pass; none, or other
    /// sizes (AMD's 64, Intel's 8 to 32), are refused with a reason.
    #[test]
    fn adapters_without_32_lane_subgroups_are_refused() {
        let sub = wgpu::Features::SUBGROUP;
        assert_eq!(check_adapter(sub | wgpu::Features::TEXTURE_COMPRESSION_BC, 32, 32), Ok(()));
        assert!(check_adapter(wgpu::Features::empty(), 32, 32).unwrap_err().contains("SUBGROUP"));
        assert!(check_adapter(sub, 64, 64).unwrap_err().contains("64 to 64"));
        assert!(check_adapter(sub, 8, 32).unwrap_err().contains("exactly 32"));
    }

    /// Identical blocks staged in a frame share one range; others, and blocks
    /// after a flush, take new ones. Needs a GPU adapter.
    #[test]
    fn identical_staged_blocks_share_their_range() {
        let Ok(mut s) = Surfaces::headless(crate::Backend::Auto) else {
            eprintln!("no GPU adapter: test skipped");
            return;
        };
        let (a, b) = (vec![1u8; 4096], vec![2u8; 4096]);
        let first = s.stage(&a).1;
        assert_eq!(s.stage(&a).1, first);
        assert_ne!(s.stage(&b).1, first);
        assert_eq!(s.stage(&a[..100]).2, 100);
        s.flush();
        assert_eq!(s.arena.1.len(), 0);
        assert_eq!(s.stage(&b).1, 0);
    }

    /// A clear reaches the texture: 4x2 targets in RF16_GF16_BF16_AF16 and in
    /// R16_G16_B16_A16 (the game's violet target) cleared to (0.5, 0, 0.5, 0)
    /// read back as opaque violet. Needs a GPU adapter.
    #[test]
    fn a_clear_reads_back_as_its_color() {
        let Ok(mut s) = Surfaces::headless(crate::Backend::Auto) else {
            eprintln!("no GPU adapter: test skipped");
            return;
        };
        let half = 0.5f32.to_bits();
        for format in [0xca, 0xc6] {
            let target = Surface { addr: 0x81_0000_2000, width: 4, height: 2, format, layers: 1 };
            s.clear(&Clear { layer: 0, color: Some((target, [half, 0, half, 0], 0xf)), depth: None, scissor: None });
            assert!(s.skipped.is_empty(), "format {format:#x}: {:?}", s.skipped);
            s.flush();
            let rgba = s.rgba(&s.map[&(target.addr, target.format)]).unwrap();
            assert_eq!(rgba.len(), 4 * 2 * 4);
            assert!(rgba.chunks(4).all(|p| p[0].abs_diff(128) <= 1 && p[1] == 0 && p[2].abs_diff(128) <= 1 && p[3] == 255), "format {format:#x}: {rgba:?}");
        }
    }

    /// A clear bounded by a scissor fills only its rectangle, and a channel
    /// mask keeps the other channels. Needs a GPU adapter.
    #[test]
    fn a_scissored_masked_clear_fills_its_rectangle_only() {
        let Ok(mut s) = Surfaces::headless(crate::Backend::Auto) else {
            eprintln!("no GPU adapter: test skipped");
            return;
        };
        let (one, half) = (1f32.to_bits(), 0.5f32.to_bits());
        let target = Surface { addr: 0x81_0000_4000, width: 4, height: 2, format: 0xd5, layers: 1 };
        s.clear(&Clear { layer: 0, color: Some((target, [0, 0, one, one], 0xf)), depth: None, scissor: None });
        // Red only (mask R), on the left half.
        s.clear(&Clear { layer: 0, color: Some((target, [one, half, 0, one], 0x1)), depth: None, scissor: Some([0, 0, 2, 2]) });
        assert!(s.skipped.is_empty(), "{:?}", s.skipped);
        s.flush();
        let rgba = s.rgba(&s.map[&(target.addr, target.format)]).unwrap();
        for (i, p) in rgba.chunks(4).enumerate() {
            let left = i % 4 < 2;
            assert_eq!(p, if left { [255, 0, 255, 255] } else { [0, 0, 255, 255] }, "pixel {i}: {rgba:?}");
        }
    }

    /// A guest range read again in a submit comes from the memo; a GPU write on
    /// one of its pages, or a new submit, sends the next read to memory; a write
    /// elsewhere does not. Needs a GPU adapter.
    #[test]
    fn repeated_guest_reads_skip_memory_until_written() {
        let Ok(mut s) = Surfaces::headless(crate::Backend::Auto) else {
            eprintln!("no GPU adapter: test skipped");
            return;
        };
        let reads = std::cell::Cell::new(0);
        let mut mem = |_: u64, buf: &mut [u8]| {
            reads.set(reads.get() + 1);
            buf.fill(7);
            true
        };
        let mut read = |s: &mut Surfaces| s.read_guest(Read::Bank, 0x1000, 16, &mut mem).map(|b| b.to_vec());
        assert_eq!(read(&mut s), Some(vec![7; 16]));
        assert_eq!(read(&mut s), Some(vec![7; 16]));
        s.invalidate(0x3000, 4);
        read(&mut s);
        assert_eq!(reads.get(), 1);
        s.invalidate(0x1ff0, 0x20);
        read(&mut s);
        assert_eq!(reads.get(), 2);
        s.new_submit();
        read(&mut s);
        assert_eq!(reads.get(), 3);
    }

    /// Presented images are hashed off the frame: each present's hash comes
    /// back in present order, FNV-1a over the image's raw bytes; an area larger
    /// than the target gives none. Needs a GPU adapter.
    #[test]
    fn presented_images_hash_in_present_order() {
        let Ok(mut s) = Surfaces::headless(crate::Backend::Auto) else {
            eprintln!("no GPU adapter: test skipped");
            return;
        };
        let one = 1f32.to_bits();
        let target = Surface { addr: 0x81_0000_6000, width: 4, height: 2, format: 0xd5, layers: 1 };
        s.clear(&Clear { layer: 0, color: Some((target, [0, 0, one, one], 0xf)), depth: None, scissor: None });
        assert!(s.hash_presented(1, 4, 2));
        assert!(s.hash_presented(2, 4, 1));
        assert!(!s.hash_presented(3, 5, 1));
        let blue = |texels: usize| fnv(&[0u8, 0, 255, 255].repeat(texels));
        assert_eq!(s.presented_hashes(), vec![(1, target.addr, blue(8)), (2, target.addr, blue(4))]);
    }
}
