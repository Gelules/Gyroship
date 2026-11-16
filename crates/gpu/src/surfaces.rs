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

/// A cache that forgets what two presents in a row left unused: `now` holds
/// what this frame made or used, `before` the previous frame's.
pub(crate) struct Aged<K, V> {
    now: FxMap<K, V>,
    before: FxMap<K, V>,
}

impl<K, V> Default for Aged<K, V> {
    fn default() -> Self {
        Aged { now: FxMap::default(), before: FxMap::default() }
    }
}

impl<K: std::hash::Hash + Eq, V: Clone> Aged<K, V> {
    fn get_or(&mut self, key: K, make: impl FnOnce(&K) -> V) -> V {
        if let Some(v) = self.now.get(&key) {
            return v.clone();
        }
        let v = self.before.remove(&key).unwrap_or_else(|| make(&key));
        self.now.insert(key, v.clone());
        v
    }

    fn age(&mut self) {
        std::mem::swap(&mut self.now, &mut self.before);
        self.now.clear();
    }
}

/// A bind group's layout and entries, as kept with it.
type GroupKey = (wgpu::BindGroupLayout, Vec<Bound>);

/// Values by a hash of their key, two presents' worth as [`Aged`] keeps them:
/// the caller hashes and compares its key through references, and builds the
/// owned key only for a value made (bind groups: layout and entries).
pub(crate) struct Groups<K, V> {
    now: FxMap<u64, Vec<(K, V)>>,
    before: FxMap<u64, Vec<(K, V)>>,
}

impl<K, V> Default for Groups<K, V> {
    fn default() -> Self {
        Groups { now: FxMap::default(), before: FxMap::default() }
    }
}

impl<K, V: Clone> Groups<K, V> {
    /// The value of hash `hash` whose key `same` recognizes; else `make`'s (key, value).
    fn get_or(&mut self, hash: u64, same: impl Fn(&K) -> bool, make: impl FnOnce() -> (K, V)) -> V {
        if let Some((_, g)) = self.now.get(&hash).and_then(|b| b.iter().find(|(k, _)| same(k))) {
            return g.clone();
        }
        let kept = self.before.get_mut(&hash).and_then(|b| b.iter().position(|(k, _)| same(k)).map(|i| b.swap_remove(i)));
        let entry = kept.unwrap_or_else(make);
        let g = entry.1.clone();
        self.now.entry(hash).or_default().push(entry);
        g
    }

    fn age(&mut self) {
        std::mem::swap(&mut self.now, &mut self.before);
        self.now.clear();
    }
}

/// A texture view's texture and description (all of it but the label).
type ViewKey = (wgpu::Texture, Option<wgpu::TextureFormat>, Option<wgpu::TextureViewDimension>, Option<wgpu::TextureUsages>, wgpu::TextureAspect, u32, Option<u32>, u32, Option<u32>);

/// A bind group's resource at a binding: a buffer range, a view or a sampler.
#[derive(PartialEq, Eq, Hash)]
pub(crate) enum Bound {
    Buffer(u32, wgpu::Buffer, u64, u64),
    View(u32, wgpu::TextureView),
    Sampler(u32, wgpu::Sampler),
}

/// Whether `device` can render to `format` (and read it back).
fn renderable_on(adapter: &wgpu::Adapter, device: &wgpu::Device, format: wgpu::TextureFormat) -> bool {
    let features = if device.features().contains(wgpu::Features::TEXTURE_ADAPTER_SPECIFIC_FORMAT_FEATURES) {
        adapter.get_texture_format_features(format)
    } else {
        format.guaranteed_format_features(device.features())
    };
    features.allowed_usages.contains(wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_SRC)
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
    /// The render pass the last draw recorded into, left open for the next draws
    /// to the same targets (a pass per draw cost a begin and an end each, ~585 a
    /// frame); anything else recorded ends it first ([`Surfaces::encoder`]).
    /// Declared before `pending`, so that it ends before its encoder is dropped.
    pub(crate) pass: Option<(crate::draw::PassKey, wgpu::RenderPass<'static>, crate::draw::PassState)>,
    /// Whether each target format is renderable here, asked once (a driver query per draw otherwise).
    renderable: Vec<(wgpu::TextureFormat, bool)>,
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
    /// The arena's way to the GPU: mapped chunks that come back once the GPU is
    /// done with them (`write_buffer` took fresh staging pages each frame in the
    /// game: 10 MB in 6.6 ms, 1.8 ms into reused ones).
    belt: wgpu::util::StagingBelt,
    /// An upload of 32 MiB or more was made since the last flush (see
    /// `Surfaces::flush_big_uploads`).
    pub(crate) big_upload: bool,
    /// Uploaded texels on their way to their textures (`Surfaces::upload`): a
    /// belt of their own (its chunks as large as the largest level uploaded),
    /// and the copies out of it, which `flush` submits before the frame.
    pub(crate) texture_belt: wgpu::util::StagingBelt,
    pub(crate) uploads: Option<wgpu::CommandEncoder>,
    /// Offsets of the arena's staged blocks by content hash: draws share identical
    /// constant banks (64 KiB each) instead of staging each copy.
    staged: FxMap<u64, Vec<(usize, usize)>>,
    /// The color target written last (the presented image at a present).
    last_color: Option<(u64, u32)>,
    /// The window the presented images go to, if any.
    window: Option<Window>,
    /// Play mode: the thread that finishes, submits and shows each frame
    /// ([`Surfaces::start_submit_thread`]).
    submit: Option<Submit>,
    /// Play mode: the submit thread ended (its panic is on stderr), losing the
    /// frames it held and their uploads: the run must stop.
    pub submit_ended: bool,
    /// This submit's guest reads and staged blocks, for the reads that repeat.
    pub(crate) memo: Memo,
    /// Shader modules and pipelines made since the caller last took them, and
    /// in the whole run (the caller adds what it takes).
    pub compile: Compile,
    pub compile_total: Compile,
    /// Play mode (D42): pipelines are made on worker threads; a draw whose
    /// pipeline is not made yet waits for it within the frame's budget, then is
    /// skipped. Off: made at once.
    pub background: bool,
    /// Play mode: how long this frame may still wait for pipelines being made
    /// before it skips their draws ([`PIPELINE_WAIT`] at each present).
    pub(crate) wait_left: std::time::Duration,
    /// The worker threads' queue, started at the first pipeline they make.
    pub(crate) workers: Option<std::sync::mpsc::Sender<crate::draw::Job>>,
    /// Pipelines the worker threads made since the caller took them: label, time.
    pub finished: Arc<Mutex<Vec<(String, std::time::Duration)>>>,
    /// `--pipeline-cache`: the driver's compiled pipelines, kept on disk.
    pub(crate) disk_cache: Option<DiskCache>,
    /// `--pipeline-cache DIR`: the translated shaders, kept in DIR/shaders (D72).
    pub shader_cache: Option<crate::shader::cache::ShaderCache>,
    /// Texture views and bind groups, made once while the frames use them: a
    /// view per texture slot and a bind group per draw cost wgpu and the
    /// driver more than the rest of the draw.
    views: Aged<ViewKey, wgpu::TextureView>,
    bind_groups: Groups<GroupKey, wgpu::BindGroup>,
    /// The device's limit on uniform buffers bound at dynamic offsets.
    pub(crate) dynamic_max: u32,
    /// Whether the device has Features::CLIP_DISTANCES (user clip planes).
    pub(crate) clip_distances: bool,
}

/// A pipeline cache kept in a file between runs (`--pipeline-cache`): the
/// driver's compiled pipelines, which it then reuses instead of compiling.
pub(crate) struct DiskCache {
    pub(crate) cache: wgpu::PipelineCache,
    file: std::path::PathBuf,
    /// The data's size when last written (it only grows), whether a write
    /// runs, and when the last one started.
    written: Arc<std::sync::atomic::AtomicUsize>,
    busy: Arc<std::sync::atomic::AtomicBool>,
    last: std::time::Instant,
}

/// How often a run writes its pipeline cache: a killed run loses at most that much.
const CACHE_EVERY: std::time::Duration = std::time::Duration::from_secs(30);

/// Play mode: what a frame may spend waiting for its new pipelines. The driver
/// makes one it has cached in 0 to 8 ms (43 at the first image of Corneria's
/// opening scene, run 14): the frame comes later instead of without them.
pub(crate) const PIPELINE_WAIT: std::time::Duration = std::time::Duration::from_millis(50);

/// Shader modules (translation and WGSL check) and pipelines made, and the
/// host time they took: what a frame showing new materials spends before it
/// draws (D42).
#[derive(Default, Clone, Debug)]
pub struct Compile {
    pub modules: u32,
    pub module_time: std::time::Duration,
    /// Modules made from the shader cache's entries (D72), the time that took, and
    /// the time translating them took when they were stored, less that.
    pub loaded: u32,
    pub load_time: std::time::Duration,
    pub saved: std::time::Duration,
    pub pipelines: u32,
    pub pipeline_time: std::time::Duration,
    /// What was made, in order: `m<program>/<variant>` for a module (`l…` when
    /// loaded), `p<vertex or compute program>+<pixel program>` for a pipeline (not summed).
    pub made: Vec<String>,
    /// Draws and dispatches skipped while their pipeline was made (play mode).
    pub waited: u64,
    /// Pipelines the worker threads made, and their time (play mode).
    pub background: u32,
    pub background_time: std::time::Duration,
}

impl std::ops::AddAssign for Compile {
    fn add_assign(&mut self, o: Compile) {
        self.modules += o.modules;
        self.module_time += o.module_time;
        self.loaded += o.loaded;
        self.load_time += o.load_time;
        self.saved += o.saved;
        self.pipelines += o.pipelines;
        self.pipeline_time += o.pipeline_time;
        self.waited += o.waited;
        self.background += o.background;
        self.background_time += o.background_time;
    }
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

/// Text over the presented image (`--hud`), in the window and the dumped frame only, never in
/// a game target: an 8-bit coverage mask of `size` texels, rows packed, stretched over `rect`
/// (x, y, width, height in presented-image pixels), in `color` (alpha: opacity), over a black
/// copy one texel to the lower right (its shadow).
pub struct Mask {
    rect: [f32; 4],
    size: (u32, u32),
    coverage: Vec<u8>,
    color: [f32; 4],
}

impl Mask {
    /// None for an empty mask, or one short of its texels.
    pub fn new(rect: [f32; 4], size: (u32, u32), coverage: Vec<u8>, color: [f32; 4]) -> Option<Mask> {
        (size.0 > 0 && size.1 > 0 && coverage.len() == size.0 as usize * size.1 as usize).then_some(Mask { rect, size, coverage, color })
    }

    /// SHOW's text on the CPU: `rgb` is the image at the pixel whose centre is `q`
    /// (presented-image pixels); the shadow, then the text, are mixed in as the shader does.
    fn over(&self, q: [f32; 2], rgb: &mut [u8]) {
        let [x, y, w, h] = self.rect;
        if !(q[0] >= x && q[1] >= y && q[0] < x + w && q[1] < y + h) {
            return;
        }
        let (mw, mh) = (i64::from(self.size.0), i64::from(self.size.1));
        let t = [((q[0] - x) * mw as f32 / w).floor() as i64, ((q[1] - y) * mh as f32 / h).floor() as i64];
        let cover = |u: i64, v: i64| if (0..mw).contains(&u) && (0..mh).contains(&v) { f32::from(self.coverage[(v * mw + u) as usize]) / 255.0 } else { 0.0 };
        let (shadow, text) = (cover(t[0] - 1, t[1] - 1) * self.color[3], cover(t[0], t[1]) * self.color[3]);
        for (c, k) in rgb.iter_mut().zip(self.color).take(3) {
            let v = f32::from(*c) / 255.0 * (1.0 - shadow);
            *c = ((v * (1.0 - text) + k * text) * 255.0).round() as u8;
        }
    }
}

/// `mask` painted into `rgba`, an image of `w` × `h` pixels shown at its own size.
fn paint(mask: &Mask, rgba: &mut [u8], w: u32, h: u32) {
    let [x, y, mw, mh] = mask.rect;
    for py in y.max(0.0) as u32..((y + mh).ceil() as u32).min(h) {
        for px in x.max(0.0) as u32..((x + mw).ceil() as u32).min(w) {
            let at = (py * w + px) as usize * 4;
            mask.over([px as f32 + 0.5, py as f32 + 0.5], &mut rgba[at..at + 4]);
        }
    }
}

/// Copies the drawn area of a target into the window, scaled with point sampling (the raw
/// texel values, as the console's display sends them), then mixes the text mask over it: its
/// shadow, then its colour. A zero rectangle draws no text.
const SHOW: &str = r#"
struct Params { area: vec2<u32>, size: vec2<u32>, rect: vec4<f32>, color: vec4<f32> };
@group(0) @binding(0) var src: texture_2d<f32>;
@group(0) @binding(1) var<uniform> params: Params;
@group(0) @binding(2) var mask: texture_2d<f32>;

@vertex
fn vs(@builtin(vertex_index) i: u32) -> @builtin(position) vec4<f32> {
    let p = vec2(f32(i & 1u) * 4.0 - 1.0, f32(i >> 1u) * 4.0 - 1.0);
    return vec4(p, 0.0, 1.0);
}

// The mask's coverage at texel t; 0 outside it.
fn cover(t: vec2<i32>) -> f32 {
    if any(t < vec2(0)) || any(t >= vec2<i32>(textureDimensions(mask))) {
        return 0.0;
    }
    return textureLoad(mask, t, 0).r;
}

@fragment
fn fs(@builtin(position) p: vec4<f32>) -> @location(0) vec4<f32> {
    let q = p.xy * vec2<f32>(params.area) / vec2<f32>(params.size);
    var c = textureLoad(src, min(vec2<u32>(floor(q)), params.area - 1u), 0).rgb;
    let r = params.rect;
    if all(q >= r.xy) && all(q < r.xy + r.zw) {
        let t = vec2<i32>(floor((q - r.xy) * vec2<f32>(textureDimensions(mask)) / r.zw));
        c = mix(c, vec3(0.0), cover(t - vec2(1)) * params.color.a);
        c = mix(c, params.color.rgb, cover(t) * params.color.a);
    }
    return vec4(c, 1.0);
}
"#;

/// The SHOW pass, apart from any window: a test draws it offscreen.
struct Show {
    pipeline: wgpu::RenderPipeline,
    layout: wgpu::BindGroupLayout,
    /// The mask's texture (1 × 1 until a mask comes), and the mask it holds.
    mask: (wgpu::Texture, Option<Arc<Mask>>),
}

/// An 8-bit texture for a mask's coverage.
fn mask_texture(device: &wgpu::Device, (width, height): (u32, u32)) -> wgpu::Texture {
    device.create_texture(&wgpu::TextureDescriptor {
        label: Some("show mask"),
        size: wgpu::Extent3d { width, height, depth_or_array_layers: 1 },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::R8Unorm,
        usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
        view_formats: &[],
    })
}

impl Show {
    fn new(device: &wgpu::Device, format: wgpu::TextureFormat) -> Show {
        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor { label: Some("show"), source: wgpu::ShaderSource::Wgsl(SHOW.into()) });
        let texture = |binding| wgpu::BindGroupLayoutEntry {
            binding,
            visibility: wgpu::ShaderStages::FRAGMENT,
            ty: wgpu::BindingType::Texture { sample_type: wgpu::TextureSampleType::Float { filterable: false }, view_dimension: wgpu::TextureViewDimension::D2, multisampled: false },
            count: None,
        };
        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: None,
            entries: &[
                texture(0),
                wgpu::BindGroupLayoutEntry { binding: 1, visibility: wgpu::ShaderStages::FRAGMENT, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Uniform, has_dynamic_offset: false, min_binding_size: None }, count: None },
                texture(2),
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
        Show { pipeline, layout, mask: (mask_texture(device, (1, 1)), None) }
    }

    /// Draws the top-left `area` of `texture` (a target of size `target`) into `view`, of size
    /// `size`, with point sampling, and `mask` over it. The mask is uploaded when it changes.
    #[allow(clippy::too_many_arguments)]
    fn draw(&mut self, device: &wgpu::Device, queue: &wgpu::Queue, texture: &wgpu::Texture, target: (u32, u32), area: (u32, u32), size: (u32, u32), mask: Option<&Arc<Mask>>, view: &wgpu::TextureView) {
        if let Some(m) = mask
            && !self.mask.1.as_ref().is_some_and(|old| Arc::ptr_eq(old, m))
        {
            let extent = wgpu::Extent3d { width: m.size.0, height: m.size.1, depth_or_array_layers: 1 };
            if self.mask.0.size() != extent {
                self.mask.0 = mask_texture(device, m.size);
            }
            queue.write_texture(self.mask.0.as_image_copy(), &m.coverage, wgpu::TexelCopyBufferLayout { offset: 0, bytes_per_row: Some(m.size.0), rows_per_image: None }, extent);
            self.mask.1 = Some(m.clone());
        }
        let area = (area.0.min(target.0), area.1.min(target.1));
        let (rect, color) = mask.map_or(([0.0; 4], [0.0; 4]), |m| (m.rect, m.color));
        let words: Vec<u8> = [area.0, area.1, size.0, size.1].into_iter().chain(rect.map(f32::to_bits)).chain(color.map(f32::to_bits)).flat_map(u32::to_le_bytes).collect();
        let params = wgpu::util::DeviceExt::create_buffer_init(device, &wgpu::util::BufferInitDescriptor { label: Some("show"), contents: &words, usage: wgpu::BufferUsages::UNIFORM });
        let src = texture.create_view(&wgpu::TextureViewDescriptor { dimension: Some(wgpu::TextureViewDimension::D2), array_layer_count: Some(1), ..Default::default() });
        let text = self.mask.0.create_view(&Default::default());
        let group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &self.layout,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: wgpu::BindingResource::TextureView(&src) },
                wgpu::BindGroupEntry { binding: 1, resource: params.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: wgpu::BindingResource::TextureView(&text) },
            ],
        });
        let mut encoder = device.create_command_encoder(&Default::default());
        {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("show"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment { view, depth_slice: None, resolve_target: None, ops: wgpu::Operations { load: wgpu::LoadOp::Clear(wgpu::Color::BLACK), store: wgpu::StoreOp::Store } })],
                ..Default::default()
            });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &group, &[]);
            pass.draw(0..3, 0..1);
        }
        queue.submit([encoder.finish()]);
    }
}

/// A window's surface and the pass that copies an image into it.
struct Window {
    surface: wgpu::Surface<'static>,
    config: wgpu::SurfaceConfiguration,
    show: Show,
}

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
        Window { surface, config, show: Show::new(device, format) }
    }

    /// Shows the top-left `area` of `texture` (a target of size `target`),
    /// scaled to `size` with point sampling, and `mask` over it. Nothing happens
    /// with an empty area or size (minimized).
    #[allow(clippy::too_many_arguments)]
    fn show(&mut self, device: &wgpu::Device, queue: &wgpu::Queue, texture: &wgpu::Texture, target: (u32, u32), area: (u32, u32), size: (u32, u32), mask: Option<&Arc<Mask>>) {
        if size.0 == 0 || size.1 == 0 || area.0 == 0 || area.1 == 0 {
            return;
        }
        if (self.config.width, self.config.height) != size {
            (self.config.width, self.config.height) = size;
            self.surface.configure(device, &self.config);
        }
        let frame = match self.surface.get_current_texture() {
            wgpu::CurrentSurfaceTexture::Success(t) | wgpu::CurrentSurfaceTexture::Suboptimal(t) => t,
            // Resized or lost under us: configure again, show the next frame.
            _ => {
                self.surface.configure(device, &self.config);
                return;
            }
        };
        self.show.draw(device, queue, texture, target, area, size, mask, &frame.texture.create_view(&Default::default()));
        queue.present(frame);
    }
}

/// What a flush submits, in this order: the copies into new textures, the
/// arena's bytes into its buffer, the frame's commands; and the present they
/// belong to (for the frame clock).
struct Frame {
    uploads: Option<wgpu::CommandEncoder>,
    arena: (wgpu::Buffer, Vec<u8>),
    pending: Option<wgpu::CommandEncoder>,
    present: u64,
}

/// The GPU half of a flush: `f`'s command buffers finished and submitted, its
/// arena bytes copied through `belt`. The texture uploads' and the arena's
/// copies run before the frame's commands (their command buffers come first),
/// and after the previous frame's, which read the same arena bytes: queue
/// order. `stamp`: the frame clock's first timestamp is written here, just
/// before the frame's commands (the submit thread's; inline, `encoder` wrote
/// it, so the span then leaves out the barriers and first-use clears wgpu puts
/// at the head of the frame's commands: microseconds). Returns the arena
/// bytes, emptied.
#[allow(clippy::too_many_arguments)]
fn submit_frame(device: &wgpu::Device, queue: &wgpu::Queue, belt: &mut wgpu::util::StagingBelt, mut clock: Option<&mut crate::timing::FrameClock>, timing: Option<&mut crate::timing::Timing>, stamp: bool, f: Frame) -> Vec<u8> {
    let mut buffers = Vec::with_capacity(4);
    if let Some(uploads) = f.uploads {
        buffers.push(uploads.finish());
    }
    let (arena, mut bytes) = f.arena;
    if let Some(size) = wgpu::BufferSize::new(bytes.len() as u64) {
        let mut upload = device.create_command_encoder(&Default::default());
        belt.write_buffer(&mut upload, &arena, 0, size).copy_from_slice(&bytes);
        belt.finish();
        buffers.push(upload.finish());
        bytes.clear();
    }
    if let Some(mut e) = f.pending {
        if let Some(t) = &timing {
            t.resolve(&mut e);
        }
        if let Some(c) = clock.as_deref_mut() {
            if stamp {
                let mut s = device.create_command_encoder(&Default::default());
                c.open(&mut s);
                buffers.push(s.finish());
            }
            c.close(device, &mut e);
        }
        buffers.push(e.finish());
        queue.submit(buffers);
        if let Some(t) = timing {
            t.collect(device);
        }
        if let Some(c) = clock {
            c.submitted(device, f.present);
        }
    } else if !buffers.is_empty() {
        queue.submit(buffers);
    }
    // Chunks whose copies were submitted come back once the GPU is done.
    belt.recall();
    bytes
}

/// A job for the submit thread.
type SubmitJob = Box<dyn FnOnce(&mut Submitter) + Send>;

/// What the submit thread owns: the arena's staging, the frame clock and the window.
struct Submitter {
    device: wgpu::Device,
    queue: wgpu::Queue,
    belt: wgpu::util::StagingBelt,
    clock: Option<crate::timing::FrameClock>,
    window: Option<Window>,
    /// Each frame's arena bytes, emptied, back to the recording thread.
    spare: std::sync::mpsc::Sender<Vec<u8>>,
}

/// The way to the submit thread: its jobs (one running and one waiting at
/// most), the arena bytes it gives back, the nanoseconds its jobs took, and
/// the thread itself (its CPU time, `--perf`).
struct Submit {
    jobs: std::sync::mpsc::SyncSender<SubmitJob>,
    spare: std::sync::mpsc::Receiver<Vec<u8>>,
    busy: Arc<std::sync::atomic::AtomicU64>,
    thread: std::thread::JoinHandle<()>,
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
    /// Texture views and bind groups made, then asked for (the rest came from their caches).
    pub views: [u64; 2],
    pub bind_groups: [u64; 2],
    /// Textures uploaded from guest memory: bytes, and host nanoseconds (read,
    /// deswizzle, copy, and the wait after a large one).
    pub uploads: [u64; 2],
    /// Host nanoseconds in `flush`: the arena's copy, the command buffers
    /// finished, the submit (in play mode, handing the frame to the submit
    /// thread instead).
    pub flush_ns: u64,
    /// Play mode: host nanoseconds the submit thread's jobs took (each
    /// flush's GPU half, each frame shown), as of the last flush.
    pub submit_ns: u64,
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
    /// Address -> a texture or sampler header (two guest reads per texture slot of every draw).
    descriptors: FxMap<u64, Dated<[u8; 32]>>,
    /// Texture slots resolved this submit, by their inputs (`draw::SlotKey`): about
    /// 17 a draw for a few hundred distinct ones. Also emptied when a texture is
    /// dropped, a target made, or a dispatch starts; a target's slot is checked
    /// against the target's last write at each use.
    pub(crate) slots: FxMap<crate::draw::SlotKey, crate::draw::Resolved>,
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
        self.descriptors.clear();
        self.slots.clear();
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
            | wgpu::Features::PIPELINE_CACHE
            | wgpu::Features::TEXTURE_ATOMIC
            | wgpu::Features::CLIP_DISTANCES
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
        let belt = wgpu::util::StagingBelt::new(device.clone(), 16 << 20);
        let texture_belt = wgpu::util::StagingBelt::new(device.clone(), 16 << 20);
        let dynamic_max = device.limits().max_dynamic_uniform_buffers_per_pipeline_layout;
        let clip_distances = device.features().contains(wgpu::Features::CLIP_DISTANCES);
        let window = window.map(|(surface, size)| Window::new(&adapter, &device, surface, size));
        let timing = crate::timing::Timing::new(&device, &queue);
        let renderable = (0..=0xff).filter_map(color_format).chain((0..=0xff).filter_map(depth_format)).map(|f| (f, renderable_on(&adapter, &device, f))).collect();
        Ok(Surfaces { renderable, adapter, device, queue, map: BTreeMap::new(), stamp: 0, touched: BTreeSet::new(), blit: (pipeline, layout), fill, cache: Default::default(), drawn: 0, textures_read: Vec::new(), skipped: BTreeMap::new(), errors, pass: None, pending: None, timing, clock: None, present: 0, readbacks: Default::default(), hasher: None, arena, belt, big_upload: false, texture_belt, uploads: None, staging: Staging::default(), staged: FxMap::default(), last_color: None, window, submit: None, submit_ended: false, memo: Memo::default(), compile: Compile::default(), compile_total: Compile::default(), background: false, wait_left: PIPELINE_WAIT, workers: None, finished: Default::default(), disk_cache: None, shader_cache: None, views: Aged::default(), bind_groups: Groups::default(), dynamic_max, clip_distances })
    }

    /// Shows the presented image (the color target written last, its top-left
    /// `area`) in the window, scaled to `size` with point sampling, with `mask`
    /// over it. Nothing happens without a window, with nothing written, or
    /// while minimized.
    pub fn show(&mut self, area: (u32, u32), size: (u32, u32), mask: Option<Arc<Mask>>) {
        self.flush();
        let Some(e) = self.last_color.and_then(|k| self.map.get(&k)) else { return };
        let (texture, target) = (e.texture.clone(), (e.surface.width, e.surface.height));
        if self.submit.is_some() {
            // Play mode: the submit thread shows it, after the frame it was just given.
            return self.send(Box::new(move |s: &mut Submitter| {
                if let Some(w) = &mut s.window {
                    w.show(&s.device, &s.queue, &texture, target, area, size, mask.as_ref());
                }
            }));
        }
        if let Some(w) = &mut self.window {
            w.show(&self.device, &self.queue, &texture, target, area, size, mask.as_ref());
        }
    }

    /// The encoder that collects a frame's passes: one submit per frame
    /// instead of one per operation. Give it back with [`Surfaces::record`].
    pub(crate) fn encoder(&mut self) -> wgpu::CommandEncoder {
        // A pass left open for the next draw ends before anything else is recorded.
        self.pass = None;
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
    /// how many flushes had to wait for the GPU to free a timing slot. At the
    /// end of a run: in play mode the clock comes back from the submit thread
    /// once it has submitted every frame (none if it is stuck past 3 s).
    pub fn frame_times(&mut self) -> (Vec<f64>, u64) {
        self.flush();
        if self.submit.is_some() {
            let (tx, rx) = std::sync::mpsc::channel();
            self.send(Box::new(move |s: &mut Submitter| {
                let _ = tx.send(s.clock.take());
            }));
            self.clock = rx.recv_timeout(std::time::Duration::from_secs(3)).ok().flatten();
        }
        self.clock.as_mut().map_or((Vec::new(), 0), |c| (c.finish(&self.device), c.stalls))
    }

    /// `--pipeline-cache DIR`: the driver's compiled pipelines are kept in DIR, in
    /// a file named after the adapter (wgpu also checks the driver's version and
    /// starts empty on a mismatch), for the next runs to reuse. Vulkan only.
    /// Returns the file and the size it had.
    pub fn use_pipeline_cache(&mut self, dir: &std::path::Path) -> Result<(std::path::PathBuf, usize), String> {
        if !self.device.features().contains(wgpu::Features::PIPELINE_CACHE) {
            return Err("this GPU backend keeps no pipeline cache (Vulkan only)".into());
        }
        let name = wgpu::util::pipeline_cache_key(&self.adapter.get_info()).ok_or("no pipeline cache for this adapter")?;
        std::fs::create_dir_all(dir).map_err(|e| format!("{}: {e}", dir.display()))?;
        let file = dir.join(name);
        let data = std::fs::read(&file).ok();
        let errors = self.errors.lock().map_or(0, |e| e.len());
        // SAFETY: the data comes from get_data on an adapter of this name (the
        // file's); a stale one (another driver) is dropped, as `fallback` asks.
        let cache = unsafe { self.device.create_pipeline_cache(&wgpu::PipelineCacheDescriptor { label: Some("pipelines"), data: data.as_deref(), fallback: true }) };
        // A cache that failed to be made would void every pipeline made with it.
        if self.errors.lock().map_or(0, |e| e.len()) > errors {
            return Err("the pipeline cache could not be made".into());
        }
        let size = data.map_or(0, |d| d.len());
        let written = Arc::new(std::sync::atomic::AtomicUsize::new(size));
        self.disk_cache = Some(DiskCache { cache, file: file.clone(), written, busy: Default::default(), last: std::time::Instant::now() });
        Ok((file, size))
    }

    /// Writes the pipeline cache to its file if it grew, on a thread of its own
    /// (the driver may hold the cache while it compiles), through a temporary
    /// file and a rename: a run killed meanwhile leaves the last whole file.
    /// Every CACHE_EVERY at most, unless `now`; returns the writing thread.
    pub fn save_pipeline_cache(&mut self, now: bool) -> Option<std::thread::JoinHandle<()>> {
        use std::sync::atomic::Ordering;
        let d = self.disk_cache.as_mut()?;
        if (!now && d.last.elapsed() < CACHE_EVERY) || d.busy.swap(true, Ordering::AcqRel) {
            return None;
        }
        d.last = std::time::Instant::now();
        let (cache, file, written, busy) = (d.cache.clone(), d.file.clone(), d.written.clone(), d.busy.clone());
        std::thread::Builder::new()
            .name("pipeline cache".into())
            .spawn(move || {
                if let Some(data) = cache.get_data()
                    && data.len() > written.load(Ordering::Acquire)
                {
                    let tmp = file.with_extension(format!("{}.tmp", std::process::id()));
                    match std::fs::write(&tmp, &data).and_then(|()| std::fs::rename(&tmp, &file)) {
                        Ok(()) => written.store(data.len(), Ordering::Release),
                        Err(e) => eprintln!("gyroship: --pipeline-cache {}: {e}", file.display()),
                    }
                }
                busy.store(false, Ordering::Release);
            })
            .ok()
    }

    /// The run's last write of the pipeline cache, waited for `patience` at most
    /// (a driver stuck on a compile may hold the cache).
    pub fn finish_pipeline_cache(&mut self, patience: std::time::Duration) {
        let since = std::time::Instant::now();
        while since.elapsed() < patience {
            if self.disk_cache.as_ref().is_none_or(|d| !d.busy.load(std::sync::atomic::Ordering::Acquire)) {
                break;
            }
            std::thread::sleep(std::time::Duration::from_millis(10));
        }
        if let Some(writer) = self.save_pipeline_cache(true) {
            while !writer.is_finished() && since.elapsed() < patience {
                std::thread::sleep(std::time::Duration::from_millis(10));
            }
        }
    }

    pub(crate) fn record(&mut self, encoder: wgpu::CommandEncoder) {
        self.pending = Some(encoder);
    }

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
    pub fn new_submit(&mut self, verified: bool) {
        self.memo.clear();
        self.begin_job(verified);
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
        // ponytail: a bank may be bound at its full size near the end of a mapping
        // (a ring buffer): its readable pages, then zeros, as programs never read
        // past them (the console's GPU would fault).
        if !mem(addr, &mut bytes) {
            let n = crate::draw::read_pages(mem, addr, &mut bytes);
            if n == 0 {
                return None;
            }
            bytes[n..].fill(0);
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

    /// The 32-byte texture or sampler header at `addr`, read once a submit while
    /// the GPU does not write it; None if not readable.
    pub(crate) fn descriptor(&mut self, addr: u64, mem: crate::draw::Mem) -> Option<[u8; 32]> {
        if let Some((d, epoch)) = self.memo.descriptors.get(&addr)
            && self.memo.fresh(addr, 32, *epoch)
        {
            return Some(*d);
        }
        let mut d = [0u8; 32];
        if !mem(addr, &mut d) {
            return None;
        }
        self.memo.descriptors.insert(addr, (d, self.memo.epoch));
        Some(d)
    }

    /// Records a block the caller staged from guest memory at `addr` (see
    /// [`Surfaces::staged_guest`]).
    pub(crate) fn remember_staged(&mut self, kind: Read, addr: u64, len: usize, slice: Slice) {
        self.memo.staged.insert((kind as u8, addr, len, 0, 0), (slice, self.memo.epoch));
    }

    /// Reads `len` guest bytes at `addr` straight into the arena (padded to 4),
    /// neither hashed nor compared with what it holds: vertices and indices,
    /// whose blocks never repeat at another address (the memo finds them at
    /// theirs). None if they are not readable.
    pub(crate) fn stage_read(&mut self, addr: u64, len: usize, mem: crate::draw::Mem) -> Option<Slice> {
        let range = len.next_multiple_of(4).max(4);
        let mut at = self.arena.1.len().next_multiple_of(ARENA_ALIGN);
        if (at + range) as u64 > self.arena.0.size() {
            self.grow_arena(range);
            at = 0;
        }
        let (before, data) = (self.arena.1.len(), &mut self.arena.1);
        data.resize(at + range, 0);
        if !mem(addr, &mut data[at..at + len]) {
            data.truncate(before);
            return None;
        }
        self.staging.calls += 1;
        self.staging.data_bytes += range as u64;
        Some((self.arena.0.clone(), at as u64, range as u64))
    }

    /// The bytes of a block staged in the current arena.
    pub(crate) fn arena_bytes(&self, s: &Slice, len: usize) -> &[u8] {
        &self.arena.1[s.1 as usize..s.1 as usize + len]
    }

    /// Records indices staged by the caller (see [`Surfaces::staged_indices`]).
    pub(crate) fn remember_indices(&mut self, addr: u64, size: usize, count: u32, restart: Option<u32>, staged: Indices) {
        self.memo.indices.insert((addr, size, count, restart), (staged, self.memo.epoch));
    }

    /// Submits the work recorded so far (at each present, and before any
    /// readback). Uploads only ever go to new textures, which nothing
    /// recorded before can use: their copies need no flush of their own. A
    /// refresh in place copies among the frame's commands instead.
    pub fn flush(&mut self) {
        let started = std::time::Instant::now();
        self.pass = None;
        self.memo.clear();
        let whole = self.arena.1.len().next_multiple_of(wgpu::COPY_BUFFER_ALIGNMENT as usize);
        self.arena.1.resize(whole, 0);
        if whole > 0 {
            self.staging.flushed += whole as u64;
            self.staging.flushes += 1;
            self.staged.clear();
        }
        if self.submit.is_some() {
            self.hand_over();
        } else {
            // The texture staging is closed even with no upload: a refresh may have used it.
            self.texture_belt.finish();
            let f = Frame { uploads: self.uploads.take(), arena: (self.arena.0.clone(), std::mem::take(&mut self.arena.1)), pending: self.pending.take(), present: self.present };
            self.arena.1 = submit_frame(&self.device, &self.queue, &mut self.belt, self.clock.as_mut(), self.timing.as_mut(), false, f);
            self.texture_belt.recall();
        }
        if let Some(t) = &self.submit {
            self.staging.submit_ns = t.busy.load(std::sync::atomic::Ordering::Relaxed);
        }
        self.staging.flush_ns += started.elapsed().as_nanos() as u64;
    }

    /// Play mode: the work recorded so far goes to the submit thread, which
    /// finishes and submits it while this thread records the next frame.
    fn hand_over(&mut self) {
        if self.uploads.is_none() && self.pending.is_none() && self.arena.1.is_empty() {
            return;
        }
        // The texture staging is mapped again right after the submit that carries its
        // copies (each staging allocation records its copy at once: `stage_level`).
        match self.pending.as_ref().or(self.uploads.as_ref()) {
            Some(e) => self.texture_belt.finish_and_recall_on_submit(e),
            None => self.texture_belt.finish(),
        }
        let len = self.arena.1.len();
        let spare = self.submit.as_ref().and_then(|t| t.spare.try_recv().ok()).unwrap_or_else(|| Vec::with_capacity(len));
        let f = Frame { uploads: self.uploads.take(), arena: (self.arena.0.clone(), std::mem::replace(&mut self.arena.1, spare)), pending: self.pending.take(), present: self.present };
        self.send(Box::new(move |s: &mut Submitter| {
            let bytes = submit_frame(&s.device, &s.queue, &mut s.belt, s.clock.as_mut(), None, true, f);
            let _ = s.spare.send(bytes);
        }));
    }

    /// Play mode (D65): from now on each frame is finished, submitted and
    /// shown on a thread of its own while this one records the next; the
    /// frame clock and the window go with it. Not with GYROSHIP_GPU_TIMES,
    /// which waits for the GPU at each flush anyway.
    pub fn start_submit_thread(&mut self) -> std::io::Result<()> {
        if self.timing.is_some() {
            return Ok(());
        }
        let (jobs, rx) = std::sync::mpsc::sync_channel::<SubmitJob>(1);
        let (back, spare) = std::sync::mpsc::channel();
        let busy = Arc::new(std::sync::atomic::AtomicU64::new(0));
        // The arena's staging is the thread's now; this one serves only if the thread ends.
        let belt = std::mem::replace(&mut self.belt, wgpu::util::StagingBelt::new(self.device.clone(), 16 << 20));
        let mut s = Submitter { device: self.device.clone(), queue: self.queue.clone(), belt, clock: self.clock.take(), window: self.window.take(), spare: back };
        let took = busy.clone();
        let thread = std::thread::Builder::new().name("gpu submit".into()).spawn(move || {
            for job in rx {
                let started = std::time::Instant::now();
                job(&mut s);
                took.fetch_add(started.elapsed().as_nanos() as u64, std::sync::atomic::Ordering::Relaxed);
            }
        })?;
        self.submit = Some(Submit { jobs, spare, busy, thread });
        Ok(())
    }

    /// Play mode: the submit thread, while it runs (its CPU time, `--perf`).
    pub fn submit_thread(&self) -> Option<&std::thread::JoinHandle<()>> {
        self.submit.as_ref().map(|t| &t.thread)
    }

    /// Play mode: queues a job for the submit thread, waiting while another
    /// one waits. If the thread ended, frames are submitted here until the
    /// run stops (`submit_ended`).
    fn send(&mut self, job: SubmitJob) {
        if let Some(t) = &self.submit
            && t.jobs.send(job).is_err()
        {
            self.submit = None;
            self.submit_ended = true;
        }
    }

    /// Play mode: returns once the submit thread has run every job sent so
    /// far, for work submitted here, or a wait for the GPU, that must follow them.
    pub(crate) fn sync(&mut self) {
        if self.submit.is_some() {
            let (tx, rx) = std::sync::mpsc::sync_channel(1);
            self.send(Box::new(move |_: &mut Submitter| {
                let _ = tx.send(());
            }));
            let _ = rx.recv(); // at once if the thread ended: the job was dropped
        }
    }

    /// As [`Surfaces::sync`], `patience` at most, even with the submit thread
    /// stuck in the driver (the end of a run); false if it is still busy then.
    /// A thread that ended holds nothing more.
    pub fn drain(&self, patience: std::time::Duration) -> bool {
        use std::sync::mpsc::{RecvTimeoutError, TrySendError};
        let Some(t) = &self.submit else { return true };
        let since = std::time::Instant::now();
        let (tx, rx) = std::sync::mpsc::sync_channel(1);
        let mut job: SubmitJob = Box::new(move |_: &mut Submitter| {
            let _ = tx.send(());
        });
        loop {
            match t.jobs.try_send(job) {
                Ok(()) => return !matches!(rx.recv_timeout(patience.saturating_sub(since.elapsed())), Err(RecvTimeoutError::Timeout)),
                Err(TrySendError::Full(j)) if since.elapsed() < patience => job = j,
                Err(TrySendError::Full(_)) => return false,
                Err(TrySendError::Disconnected(_)) => return true,
            }
            std::thread::sleep(std::time::Duration::from_millis(1));
        }
    }

    /// After an upload of 32 MiB or more (`big_upload`), at the start of the next
    /// draw or dispatch: the work so far submitted and waited for, then the
    /// texture belt made anew, which frees its chunks: the largest kept the
    /// largest level uploaded (64 MiB for a streaming pool), and RAM runs short
    /// in Corneria (1.3 GB free, the write watch then over-reports, D44).
    pub(crate) fn flush_big_uploads(&mut self) {
        if std::mem::take(&mut self.big_upload) {
            self.flush();
            self.sync();
            let _ = self.device.poll(wgpu::PollType::wait_indefinitely());
            self.texture_belt = wgpu::util::StagingBelt::new(self.device.clone(), 16 << 20);
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
            self.grow_arena(range);
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

    /// The arena is full: its bytes are written now (it is not reused this frame)
    /// and a buffer twice as large, with room for `range`, takes over; recorded
    /// work keeps the old one alive.
    fn grow_arena(&mut self, range: usize) {
        // Play mode: no frame may wait in the submit thread's queue, or this write
        // (sent with the next submit) would land before that frame's own arena copy.
        self.sync();
        self.staging.flushed += self.arena.1.len() as u64;
        self.queue.write_buffer(&self.arena.0, 0, &self.arena.1);
        let capacity = (self.arena.0.size() * 2).max((range as u64).next_power_of_two());
        self.arena = (arena_buffer(&self.device, capacity), Vec::new());
        self.staged.clear();
        // ponytail: staged blocks stay valid in the old buffer; dropped for simplicity (rare).
        self.memo.staged.clear();
        self.memo.indices.clear();
    }

    /// A view of `texture` as `desc` describes it.
    pub(crate) fn view(&mut self, texture: &wgpu::Texture, desc: &wgpu::TextureViewDescriptor) -> wgpu::TextureView {
        let key = (texture.clone(), desc.format, desc.dimension, desc.usage, desc.aspect, desc.base_mip_level, desc.mip_level_count, desc.base_array_layer, desc.array_layer_count);
        let views = &mut self.staging.views;
        views[1] += 1;
        self.views.get_or(key, |_| {
            views[0] += 1;
            texture.create_view(desc)
        })
    }

    /// A view of `texture` in `format` when it is the texture's sRGB or linear
    /// counterpart (one of its view formats), limited then to `usage` (an sRGB
    /// view cannot carry the texture's storage usage); else in the texture's format.
    pub(crate) fn view_as(&mut self, texture: &wgpu::Texture, format: Option<wgpu::TextureFormat>, usage: wgpu::TextureUsages, desc: wgpu::TextureViewDescriptor) -> wgpu::TextureView {
        let other = format.filter(|f| *f != texture.format() && f.remove_srgb_suffix() == texture.format().remove_srgb_suffix());
        self.view(texture, &wgpu::TextureViewDescriptor { format: other, usage: other.map(|_| usage), ..desc })
    }

    /// A bind group of `layout` with these buffer ranges (bound from 0 when
    /// their offsets are dynamic), then the slots' views and samplers.
    pub(crate) fn bind_group(&mut self, layout: &wgpu::BindGroupLayout, buffers: &[(u32, Slice)], textures: &[crate::draw::Slot], dynamic: bool) -> wgpu::BindGroup {
        use std::hash::{Hash, Hasher};
        // The entries, borrowed: a buffer at offset 0 when the offset is dynamic, then each
        // texture slot's view and sampler. Hashed and compared as they are; cloned into a
        // key only for a group made here.
        let at = |a: u64| if dynamic { 0 } else { a };
        let mut h = Fx::default();
        layout.hash(&mut h);
        for (binding, (buffer, a, size)) in buffers {
            (binding, buffer, at(*a), size).hash(&mut h);
        }
        for s in textures {
            s.view.hash(&mut h);
            s.sampler.hash(&mut h);
        }
        let same = |(l, bound): &GroupKey| {
            let mut it = bound.iter();
            l == layout
                && buffers.iter().all(|(binding, (buffer, a, size))| matches!(it.next(), Some(Bound::Buffer(b, f, o, z)) if b == binding && f == buffer && *o == at(*a) && z == size))
                && textures.iter().enumerate().all(|(slot, s)| {
                    matches!(it.next(), Some(Bound::View(b, v)) if *b == 32 + 2 * slot as u32 && *v == s.view)
                        && s.sampler.as_ref().is_none_or(|sampler| matches!(it.next(), Some(Bound::Sampler(b, x)) if *b == 33 + 2 * slot as u32 && x == sampler))
                })
                && it.next().is_none()
        };
        let (device, made) = (&self.device, &mut self.staging.bind_groups);
        made[1] += 1;
        self.bind_groups.get_or(h.finish(), same, || {
            made[0] += 1;
            let mut bound = Vec::with_capacity(buffers.len() + 2 * textures.len());
            for (binding, (buffer, a, size)) in buffers {
                bound.push(Bound::Buffer(*binding, buffer.clone(), at(*a), *size));
            }
            for (slot, s) in textures.iter().enumerate() {
                bound.push(Bound::View(32 + 2 * slot as u32, s.view.clone()));
                if let Some(sampler) = &s.sampler {
                    bound.push(Bound::Sampler(33 + 2 * slot as u32, sampler.clone()));
                }
            }
            let entries: Vec<wgpu::BindGroupEntry> = bound
                .iter()
                .map(|b| match b {
                    Bound::Buffer(binding, buffer, at, size) => wgpu::BindGroupEntry { binding: *binding, resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding { buffer, offset: *at, size: wgpu::BufferSize::new(*size) }) },
                    Bound::View(binding, view) => wgpu::BindGroupEntry { binding: *binding, resource: wgpu::BindingResource::TextureView(view) },
                    Bound::Sampler(binding, sampler) => wgpu::BindGroupEntry { binding: *binding, resource: wgpu::BindingResource::Sampler(sampler) },
                })
                .collect();
            let group = device.create_bind_group(&wgpu::BindGroupDescriptor { label: None, layout, entries: &entries });
            ((layout.clone(), bound), group)
        })
    }

    /// A frame was presented: the views and bind groups neither it nor the
    /// one before used are dropped.
    pub fn next_frame(&mut self) {
        self.views.age();
        self.bind_groups.age();
        self.wait_left = PIPELINE_WAIT;
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
        // The copy is submitted here: after the frame (in play mode, once the submit thread has).
        self.flush();
        self.sync();
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

    /// The image a present shows (the color target written last, its top-left
    /// `width` × `height` texels) as RGBA rows, waiting for the GPU; None if
    /// nothing was written, the target is smaller, or the copy failed.
    pub fn presented_rgba(&mut self, width: u32, height: u32) -> Option<Vec<u8>> {
        self.flush();
        self.sync();
        let e = self.last_color.and_then(|k| self.map.get(&k))?;
        if width == 0 || height == 0 || width > e.surface.width || height > e.surface.height {
            return None;
        }
        let full = self.rgba(e).ok()?;
        let stride = e.surface.width.max(1) as usize * 4;
        Some(full.chunks(stride).take(height as usize).flat_map(|row| &row[..width as usize * 4]).copied().collect())
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
        self.renderable.iter().find(|(f, _)| *f == format).map_or_else(|| renderable_on(&self.adapter, &self.device, format), |(_, r)| *r)
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
            // A slot that read guest memory, or another target, at this address reads this one now.
            self.memo.slots.clear();
        }
        let texture = self.map[&key].texture.clone();
        self.view_as(&texture, Some(format), wgpu::TextureUsages::RENDER_ATTACHMENT, wgpu::TextureViewDescriptor { dimension: Some(wgpu::TextureViewDimension::D2), array_layer_count: Some(1), ..Default::default() })
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
    /// `mask` is painted into the presented image's file only, as SHOW draws it
    /// at the image's own size.
    pub fn dump(&mut self, dir: &Path, prefix: &str, mask: Option<&Mask>) -> Result<usize, String> {
        std::fs::create_dir_all(dir).map_err(|e| format!("{}: {e}", dir.display()))?;
        self.flush();
        self.sync();
        let mut keys = std::mem::take(&mut self.touched);
        // GYROSHIP_DUMP_ONLY=<hex address>: that target alone (dumps every few frames);
        // `presented`: the presented image, whichever buffer holds it.
        match std::env::var("GYROSHIP_DUMP_ONLY").ok().as_deref() {
            Some("presented") => keys.retain(|k| Some(*k) == self.last_color),
            Some(v) => {
                if let Ok(only) = u64::from_str_radix(v.trim_start_matches("0x"), 16) {
                    keys.retain(|k| k.0 == only);
                }
            }
            None => {}
        }
        for key in &keys {
            let e = &self.map[key];
            let (w, h) = (e.surface.width.max(1), e.surface.height.max(1));
            let mut rgba = self.rgba(e)?;
            if let Some(m) = mask
                && Some(*key) == self.last_color
            {
                paint(m, &mut rgba, w, h);
            }
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

    /// How fast the queue takes a heavy Corneria frame's arena (10 MiB), by
    /// `write_buffer` and by `write_buffer_with`: `cargo test --release -p
    /// gyroship-gpu upload_speed -- --ignored --nocapture`.
    #[test]
    #[ignore]
    fn upload_speed() {
        let Ok(s) = Surfaces::headless(crate::Backend::Auto) else { return };
        let data = vec![7u8; 10 << 20];
        let buffer = arena_buffer(&s.device, 16 << 20);
        let rate = |t: std::time::Duration| data.len() as f64 / t.as_secs_f64() / 1e9;
        for round in 0..6 {
            let t = std::time::Instant::now();
            s.queue.write_buffer(&buffer, 0, &data);
            let plain = t.elapsed();
            let t = std::time::Instant::now();
            if let Some(mut view) = s.queue.write_buffer_with(&buffer, 0, wgpu::BufferSize::new(data.len() as u64).unwrap()) {
                view.copy_from_slice(&data);
            }
            let with = t.elapsed();
            s.queue.submit([]);
            let _ = s.device.poll(wgpu::PollType::wait_indefinitely());
            eprintln!("round {round}: write_buffer {:.2} ms ({:.1} GB/s), write_buffer_with {:.2} ms ({:.1} GB/s)", plain.as_secs_f64() * 1e3, rate(plain), with.as_secs_f64() * 1e3, rate(with));
        }
        // Frames in flight: three uploads and submits before waiting, then the
        // same through a staging belt whose chunks come back after each wait.
        let mut belt = wgpu::util::StagingBelt::new(s.device.clone(), 16 << 20);
        for round in 0..4 {
            let mut plain = Vec::new();
            let mut belted = Vec::new();
            for _ in 0..3 {
                let t = std::time::Instant::now();
                s.queue.write_buffer(&buffer, 0, &data);
                plain.push(t.elapsed().as_secs_f64() * 1e3);
                s.queue.submit([]);
            }
            for _ in 0..3 {
                let mut encoder = s.device.create_command_encoder(&Default::default());
                let t = std::time::Instant::now();
                belt.write_buffer(&mut encoder, &buffer, 0, wgpu::BufferSize::new(data.len() as u64).unwrap()).copy_from_slice(&data);
                belt.finish();
                belted.push(t.elapsed().as_secs_f64() * 1e3);
                s.queue.submit([encoder.finish()]);
                belt.recall();
            }
            let _ = s.device.poll(wgpu::PollType::wait_indefinitely());
            eprintln!("in flight, round {round}: write_buffer {plain:.2?} ms, belt {belted:.2?} ms");
        }
    }

    #[test]
    fn an_aged_cache_forgets_what_two_frames_left_unused() {
        let mut c: Aged<u32, u32> = Aged::default();
        let mut made = 0;
        let mut get = |c: &mut Aged<u32, u32>, k: u32| {
            c.get_or(k, |k| {
                made += 1;
                k * 10
            })
        };
        assert_eq!((get(&mut c, 1), get(&mut c, 1)), (10, 10));
        c.age();
        // Used the frame before: carried over.
        assert_eq!(get(&mut c, 1), 10);
        c.age();
        c.age();
        // Unused for two frames: made again.
        assert_eq!(get(&mut c, 1), 10);
        assert_eq!(made, 2);
    }

    #[test]
    fn groups_are_found_by_key_within_a_hash_and_aged_like_aged() {
        let mut g: Groups<u32, u32> = Groups::default();
        let mut made = 0;
        // Keys 1 and 2 share hash 7: each is told apart by `same`.
        let mut get = |g: &mut Groups<u32, u32>, k: u32| {
            g.get_or(7, |x| *x == k, || {
                made += 1;
                (k, k * 10)
            })
        };
        assert_eq!((get(&mut g, 1), get(&mut g, 2), get(&mut g, 1)), (10, 20, 10));
        g.age();
        // Used the frame before: carried over, not made again.
        assert_eq!(get(&mut g, 2), 20);
        g.age();
        g.age();
        // Unused for two frames: made again.
        assert_eq!(get(&mut g, 2), 20);
        assert_eq!(made, 3);
    }

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

    /// SHOW draws the image and the text as a dump paints them, at the image's size and in a
    /// 1280 × 720 window, within one unit (the GPU's rounding). The numbers keep every floor
    /// at least 0.16 from an integer. Needs a GPU adapter.
    #[test]
    fn show_draws_the_image_and_the_text_as_a_dump_paints_them() {
        let Ok(s) = Surfaces::headless(crate::Backend::Auto) else {
            eprintln!("no GPU adapter: test skipped");
            return;
        };
        let texture = |(width, height), usage| s.device.create_texture(&wgpu::TextureDescriptor { label: None, size: wgpu::Extent3d { width, height, depth_or_array_layers: 1 }, mip_level_count: 1, sample_count: 1, dimension: wgpu::TextureDimension::D2, format: wgpu::TextureFormat::Rgba8Unorm, usage, view_formats: &[] });
        let (iw, ih) = (1920u32, 1080u32);
        let image: Vec<u8> = (0..iw * ih).flat_map(|i| [(i % iw) as u8, (i / iw) as u8, (i % 251) as u8, 255]).collect();
        let source = texture((iw, ih), wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST);
        s.queue.write_texture(source.as_image_copy(), &image, wgpu::TexelCopyBufferLayout { offset: 0, bytes_per_row: Some(iw * 4), rows_per_image: None }, source.size());
        let coverage: Vec<u8> = (0..30 * 12).map(|i| [0, 255, 128][i % 3]).collect();
        let mut show = Show::new(&s.device, wgpu::TextureFormat::Rgba8Unorm);
        for (ww, wh) in [(iw, ih), (1280, 720)] {
            let rect = [40.0, 212.0, 30.0 * iw as f32 / ww as f32, 12.0 * ih as f32 / wh as f32];
            let mask = Arc::new(Mask::new(rect, (30, 12), coverage.clone(), [1.0; 4]).unwrap());
            let out = texture((ww, wh), wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_SRC);
            show.draw(&s.device, &s.queue, &source, (iw, ih), (iw, ih), (ww, wh), Some(&mask), &out.create_view(&Default::default()));
            let got = s.rgba(&Entry { surface: Surface { addr: 0, width: ww, height: wh, format: 0xd5, layers: 1 }, texture: out, written: 0 }).unwrap();
            let mut text = 0;
            for (i, p) in got.chunks(4).enumerate() {
                let (x, y) = (i as u32 % ww, i as u32 / ww);
                let q = [(x as f32 + 0.5) * iw as f32 / ww as f32, (y as f32 + 0.5) * ih as f32 / wh as f32];
                let at = ((q[1] as u32).min(ih - 1) * iw + (q[0] as u32).min(iw - 1)) as usize * 4;
                let mut want = image[at..at + 4].to_vec();
                mask.over(q, &mut want);
                text += usize::from(want != image[at..at + 4]);
                assert!(p.iter().zip(&want).all(|(a, b)| a.abs_diff(*b) <= 1), "window {ww}x{wh}, pixel ({x}, {y}): {p:?}, painted {want:?}");
            }
            assert!(text > 100, "window {ww}x{wh}: the text changed {text} pixels");
        }
        assert!(s.errors.lock().unwrap().is_empty(), "{:?}", s.errors.lock().unwrap());
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
        s.new_submit(true);
        read(&mut s);
        assert_eq!(reads.get(), 3);
        // Texture and sampler headers alike.
        assert_eq!(s.descriptor(0x2000, &mut mem), Some([7; 32]));
        assert_eq!(s.descriptor(0x2000, &mut mem), Some([7; 32]));
        assert_eq!(reads.get(), 4);
        s.invalidate(0x2010, 4);
        s.descriptor(0x2000, &mut mem);
        assert_eq!(reads.get(), 5);
    }

    /// Vertices read straight into the arena: the guest's bytes then zeros to a
    /// multiple of 4, aligned; an unreadable read leaves the arena as it was.
    #[test]
    fn guest_blocks_are_read_straight_into_the_arena() {
        let Ok(mut s) = Surfaces::headless(crate::Backend::Auto) else {
            eprintln!("no GPU adapter: test skipped");
            return;
        };
        let mut mem = |addr: u64, buf: &mut [u8]| {
            buf.iter_mut().enumerate().for_each(|(i, b)| *b = (addr as usize + i) as u8);
            addr != 0xdead
        };
        let a = s.stage_read(0x10, 6, &mut mem).unwrap();
        assert_eq!((a.1, a.2), (0, 8));
        assert_eq!(s.arena_bytes(&a, 8), [0x10, 0x11, 0x12, 0x13, 0x14, 0x15, 0, 0]);
        let len = s.arena.1.len();
        assert!(s.stage_read(0xdead, 64, &mut mem).is_none());
        assert_eq!(s.arena.1.len(), len);
        let b = s.stage_read(0x20, 4, &mut mem).unwrap();
        assert_eq!((b.1, s.arena_bytes(&b, 4)), (ARENA_ALIGN as u64, &[0x20, 0x21, 0x22, 0x23][..]));
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

    /// Frames handed to the submit thread draw what inline frames draw. Five
    /// presents of masked clears (fills whose colour is staged in the arena).
    /// On the thread, a job holds it at presents 2 and 4: frame 2 still waits
    /// in its queue while the third grows the arena (fills before and after
    /// the growth on other channels), and frame 4 while the fifth is recorded
    /// and handed over (red only, over frame 4's green). Each image reads back
    /// as its colours, the frame clock timed every present, no wgpu error
    /// came, and a drain gives up on a held thread. Needs a GPU adapter.
    #[test]
    fn the_submit_thread_draws_the_inline_frames() {
        let target = Surface { addr: 0x81_0000_8000, width: 4, height: 2, format: 0xd5, layers: 1 };
        let fill = |s: &mut Surfaces, mask: u8, v: [u8; 4]| s.clear(&Clear { layer: 0, color: Some((target, v.map(|c| (f32::from(c) / 255.0).to_bits()), mask)), depth: None, scissor: None });
        // A job holding the submit thread until the sender it returns is dropped.
        let hold = |s: &mut Surfaces| {
            let (open, gate) = std::sync::mpsc::channel::<()>();
            s.send(Box::new(move |_: &mut Submitter| {
                let _ = gate.recv();
            }));
            open
        };
        for threaded in [false, true] {
            let Ok(mut s) = Surfaces::headless(crate::Backend::Auto) else {
                eprintln!("no GPU adapter: test skipped");
                return;
            };
            let timed = s.measure_frames();
            if threaded {
                s.start_submit_thread().unwrap();
                assert!(s.submit.is_some(), "no submit thread (GYROSHIP_GPU_TIMES set?)");
            }
            for n in 1..=5u8 {
                s.present = u64::from(n);
                fill(&mut s, if n == 5 { 0x1 } else { 0x3 }, [17 * n, 255 - 17 * n, 0, 0]);
                if n == 3 {
                    s.stage(&vec![1; 17 << 20]);
                    fill(&mut s, 0x4, [0, 0, 64, 0]);
                }
                if threaded && n % 2 == 0 {
                    let open = hold(&mut s);
                    std::thread::spawn(move || {
                        std::thread::sleep(std::time::Duration::from_millis(200));
                        drop(open);
                    });
                }
                s.flush();
                if n % 2 == 1 {
                    assert!(s.hash_presented(u64::from(n), 4, 2));
                }
            }
            assert!(s.arena.0.size() > 16 << 20, "the arena grew");
            let (ms, _) = s.frame_times();
            let rgba = |n: u8| {
                let green = if n == 5 { 4 } else { n };
                [17 * n, 255 - 17 * green, if n < 3 { 0 } else { 64 }, 0]
            };
            let want: Vec<_> = [1u8, 3, 5].iter().map(|&n| (u64::from(n), target.addr, fnv(&rgba(n).repeat(8)))).collect();
            assert_eq!(s.presented_hashes(), want, "threaded {threaded}");
            // Microseconds of work: a slot without its first timestamp would give the GPU clock's value.
            assert!(!timed || (1..=5).all(|n| ms.get(n).is_some_and(|&t| t > 0.0 && t < 100.0)), "threaded {threaded}: GPU times {ms:?}");
            assert!(s.errors.lock().unwrap().is_empty(), "threaded {threaded}: {:?}", s.errors.lock().unwrap());
            if threaded {
                let open = hold(&mut s);
                assert!(!s.drain(std::time::Duration::from_millis(100)), "a drain waits for a held thread");
                drop(open);
                assert!(s.drain(std::time::Duration::from_secs(3)));
            }
        }
    }
}
