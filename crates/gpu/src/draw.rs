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
use crate::surfaces::{Surfaces, color_format, depth_format, uniform};
use std::collections::{BTreeMap, BTreeSet, HashMap};

/// Reads guest memory at a GPU address; false if a byte is not readable.
pub type Mem<'a> = &'a mut dyn FnMut(u64, &mut [u8]) -> bool;

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
        let mut chunk = vec![0u8; 0x1000];
        if !mem(gpu + code.len() as u64, &mut chunk) {
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

type Module = Result<(Translation, wgpu::ShaderModule), String>;

#[derive(Default)]
pub struct Cache {
    /// Program key by GPU address, and code by key.
    programs: HashMap<u64, Option<u64>>,
    pub(crate) code: HashMap<u64, Vec<u8>>,
    /// Compute program keys by GPU address, compiled compute programs, and the
    /// images compute programs wrote, by GPU address (see `compute.rs`).
    pub(crate) compute_programs: HashMap<u64, Option<u64>>,
    pub(crate) compute: HashMap<String, Result<(Translation, wgpu::ShaderModule), String>>,
    pub(crate) compute_pipelines: HashMap<String, (wgpu::ComputePipeline, wgpu::BindGroupLayout)>,
    pub(crate) images: HashMap<u64, wgpu::Texture>,
    /// Translations by (program key, key of the pixel program it pairs with,
    /// mask of the attributes fetched as integers, hash of the jump targets or 0).
    modules: HashMap<(u64, Option<u64>, u64, u64), Module>,
    pipelines: HashMap<String, (wgpu::RenderPipeline, Vec<wgpu::BindGroupLayout>)>,
    /// Uploaded textures by address and header, with the end of the memory they came from.
    textures: HashMap<(u64, [u8; 32]), (wgpu::Texture, u64)>,
    /// Samplers by header, filtering and compare.
    samplers: HashMap<([u8; 32], bool, bool), wgpu::Sampler>,
    /// Stand-ins for textures only queried, by width, height, depth and levels.
    queried: HashMap<(u32, u32, u32, u32), wgpu::Texture>,
    /// The pass that writes R32Float texels as depth (ZF32 uploads).
    depth_copy: Option<wgpu::RenderPipeline>,
    /// Float twins of integer compute images, by address, until the next dispatch.
    pub(crate) twins: HashMap<u64, wgpu::Texture>,
    /// Color textures from memory read with a depth compare: their R channel as
    /// depth, by address and header; forgotten with any invalidation.
    compare_depths: HashMap<(u64, [u8; 32]), wgpu::Texture>,
}

/// The instructions a program's jump tables may send it to: every word of each
/// table's bank (as bound) that, added to the jump's base, is an instruction's
/// byte offset (a 16-byte multiple inside the program). Words that are not table
/// entries only split blocks more. None if a bank is unbound or unreadable.
fn jump_targets(tables: &[(u8, u32)], banks: &[Option<(u64, u32)>], mem: Mem, code_len: usize) -> Option<BTreeSet<usize>> {
    let mut targets = BTreeSet::new();
    for &(bank, base) in tables {
        let (addr, size) = (*banks.get(usize::from(bank))?)?;
        let mut data = vec![0u8; (size as usize).min(CBUF_BYTES)];
        if !mem(addr, &mut data) {
            return None;
        }
        for w in data.as_chunks::<4>().0 {
            let at = u32::from_le_bytes(*w).wrapping_add(base);
            if at % 16 == 0 && (at as usize) < code_len {
                targets.insert(at as usize / 16);
            }
        }
    }
    Some(targets)
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

/// A stage's bindings.
pub(crate) struct Bindings {
    /// Constant buffers by bank.
    pub(crate) cbufs: BTreeMap<u8, Vec<u8>>,
    pub(crate) textures: Vec<Slot>,
    /// Each slot's header swizzle, when the program reads them.
    pub(crate) swizzles: Option<Vec<u32>>,
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

impl Bindings {
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

impl Surfaces {
    /// The guest memory each uploaded texture came from, as GPU address ranges
    /// (start, end): the HLE forgets those the CPU has written since.
    pub fn texture_ranges(&self) -> Vec<(u64, u64)> {
        self.cache.textures.iter().map(|((addr, _), (_, end))| (*addr, *end)).collect()
    }

    /// What the host keeps, by kind, for the summary line (growth shows a leak).
    pub fn cache_sizes(&self) -> String {
        let c = &self.cache;
        format!(
            "targets {}, textures {}, samplers {}, programs {}, modules {}, pipelines {}, compute pipelines {}, images {}",
            self.map.len(),
            c.textures.len(),
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
            Some(at) => Some(self.module(mem, at, None, &BTreeMap::new(), &d.cbufs[4])?),
            None => None,
        };
        // Attributes fetched as integers: the vertex program declares them so.
        let ints: BTreeMap<u8, bool> = d.attributes.iter().filter(|a| !a.constant && matches!(a.format.kind, AttrKind::Uint | AttrKind::Sint)).map(|a| (a.location, a.format.kind == AttrKind::Sint)).collect();
        let vs = self.module(mem, d.programs[1].ok_or("no vertex program")?, ps.as_ref().map(|p| &p.0), &ints, &d.cbufs[0])?;

        // Attachments: the locations the pixel program writes, and depth when tested.
        let written = ps.as_ref().map(|p| p.0.interface.targets.clone()).unwrap_or_default();
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
            match layouts.iter_mut().find(|l| l.0 == s.index) {
                Some(l) => l.3.push(attribute),
                None => layouts.push((s.index, if s.per_instance { wgpu::VertexStepMode::Instance } else { wgpu::VertexStepMode::Vertex }, u64::from(s.stride), vec![attribute])),
            }
        }

        // Indices (8-bit ones widened: wgpu has none) and the last vertex they reach.
        let (indices, last) = match &d.index {
            Some(ib) => {
                let size = usize::from(ib.bytes_per_index);
                if size == 0 {
                    return Err("index size".into());
                }
                let mut raw = vec![0u8; size * d.count as usize];
                if !mem(ib.address + u64::from(ib.first) * size as u64, &mut raw) {
                    return Err("index buffer not readable".into());
                }
                let values: Vec<u32> = raw.chunks_exact(size).map(|c| c.iter().rev().fold(0, |v, b| v << 8 | u32::from(*b))).collect();
                let max = values.iter().filter(|&&i| !(ib.restart_enable && i == ib.restart_index)).max().copied().unwrap_or(0);
                let (bytes, format) = match size {
                    4 => (raw, wgpu::IndexFormat::Uint32),
                    2 => (raw, wgpu::IndexFormat::Uint16),
                    _ => (values.iter().flat_map(|&i| (i as u16).to_le_bytes()).collect(), wgpu::IndexFormat::Uint16),
                };
                (Some((bytes, format)), (i64::from(d.base_vertex) + i64::from(max)).max(0) as u64)
            }
            None => (None, u64::from(d.first) + u64::from(d.count) - 1),
        };
        let mut vertex_buffers = Vec::new();
        for (index, step, stride, attributes) in &layouts {
            let s = d.streams.iter().find(|s| s.index == *index).ok_or("stream")?;
            let elements = if *step == wgpu::VertexStepMode::Instance { 1 } else { last + 1 };
            let reach = attributes.iter().map(|a| a.offset + a.format.size()).max().unwrap_or(0);
            let bytes = (elements - 1) * stride + reach;
            if s.end.wrapping_sub(s.address) < bytes - 1 || bytes > 64 << 20 {
                return Err("vertex stream shorter than the draw".into());
            }
            let mut data = vec![0u8; bytes.next_multiple_of(4) as usize];
            if !mem(s.address, &mut data[..bytes as usize]) {
                return Err("vertex stream not readable".into());
            }
            vertex_buffers.push(self.stage(&data));
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
        let visibility = [wgpu::ShaderStages::VERTEX, wgpu::ShaderStages::FRAGMENT];
        let group_layouts: Vec<Vec<wgpu::BindGroupLayoutEntry>> = stages.iter().zip(visibility).map(|(b, v)| b.layout(v)).collect();
        let groups = group_layouts.iter().rposition(|e| !e.is_empty()).map_or(0, |g| g + 1);
        // The WGSL lengths tell a program's module with exact jump-table blocks from the first one.
        let key = format!("{:x}.{}/{:x?}/{layouts:?}/{primitive:?}/{color_states:?}/{depth_state:?}/{:?}", vs.0.key, vs.0.wgsl.len(), ps.as_ref().map(|p| (p.0.key, p.0.wgsl.len())), &group_layouts[..groups]);
        let (pipeline, bind_layouts) = match self.cache.pipelines.get(&key) {
            Some(p) => p.clone(),
            None => {
                let bind_layouts: Vec<wgpu::BindGroupLayout> = group_layouts[..groups].iter().map(|entries| self.device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor { label: None, entries })).collect();
                let refs: Vec<Option<&wgpu::BindGroupLayout>> = bind_layouts.iter().map(Some).collect();
                let layout = self.device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor { label: None, bind_group_layouts: &refs, ..Default::default() });
                let buffers: Vec<Option<wgpu::VertexBufferLayout>> = layouts.iter().map(|(_, step, stride, attributes)| Some(wgpu::VertexBufferLayout { array_stride: *stride, step_mode: *step, attributes })).collect();
                let p = self.device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                    label: None,
                    layout: Some(&layout),
                    vertex: wgpu::VertexState { module: &vs.1, entry_point: Some("main"), compilation_options: Default::default(), buffers: &buffers },
                    fragment: ps.as_ref().map(|p| wgpu::FragmentState { module: &p.1, entry_point: Some("main"), compilation_options: Default::default(), targets: &color_states }),
                    primitive,
                    depth_stencil: depth_state,
                    multisample: Default::default(),
                    multiview_mask: None,
                    cache: None,
                });
                self.cache.pipelines.insert(key, (p.clone(), bind_layouts.clone()));
                (p, bind_layouts)
            }
        };

        let mut bind_groups = Vec::new();
        for (g, layout) in bind_layouts.iter().enumerate() {
            let Bindings { cbufs, textures, swizzles } = &stages[g];
            let mut buffers: Vec<(u32, (wgpu::Buffer, u64, u64))> = cbufs.iter().map(|(b, data)| (u32::from(*b), self.stage(data))).collect();
            if let Some(codes) = swizzles {
                let bytes: Vec<u8> = codes.iter().flat_map(|c| c.to_le_bytes()).collect();
                buffers.push((shader::SWIZZLE_BINDING, self.stage(&bytes)));
            }
            let mut entries: Vec<wgpu::BindGroupEntry> = buffers.iter().map(|(b, u)| wgpu::BindGroupEntry { binding: *b, resource: uniform(u) }).collect();
            for (slot, Slot { view, sampler, .. }) in textures.iter().enumerate() {
                entries.push(wgpu::BindGroupEntry { binding: 32 + 2 * slot as u32, resource: wgpu::BindingResource::TextureView(view) });
                if let Some(sampler) = sampler {
                    entries.push(wgpu::BindGroupEntry { binding: 33 + 2 * slot as u32, resource: wgpu::BindingResource::Sampler(sampler) });
                }
            }
            bind_groups.push(self.device.create_bind_group(&wgpu::BindGroupDescriptor { label: None, layout, entries: &entries }));
        }
        let index_buffer = indices.as_ref().map(|(bytes, format)| (self.stage(bytes), *format));

        // The pass.
        let color_views: Vec<Option<wgpu::TextureView>> = colors.iter().map(|c| c.map(|(s, f)| self.texture(&s, f))).collect();
        let depth_view = depth.map(|(s, f)| (self.texture(&s, f), f));
        let timed = self.timing.as_mut().and_then(|t| t.next(format!("draw vs {:016x} ps {}", vs.0.key, ps.as_ref().map_or("none".into(), |p| format!("{:016x}", p.0.key)))));
        let mut encoder = self.encoder();
        {
            let load = wgpu::Operations { load: wgpu::LoadOp::Load, store: wgpu::StoreOp::Store };
            let attachments: Vec<Option<wgpu::RenderPassColorAttachment>> = color_views.iter().map(|v| v.as_ref().map(|view| wgpu::RenderPassColorAttachment { view, depth_slice: None, resolve_target: None, ops: load })).collect();
            // Without a pixel program the pipeline has no color targets: neither may the pass.
            let attachments = if ps.is_none() { Vec::new() } else { attachments };
            let timestamp_writes = timed.zip(self.timing.as_ref()).map(|(i, t)| wgpu::RenderPassTimestampWrites { query_set: &t.set, beginning_of_pass_write_index: Some(i), end_of_pass_write_index: Some(i + 1) });
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
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
            pass.set_pipeline(&pipeline);
            pass.set_viewport(v.x, v.y, v.width, v.height, v.min_depth, v.max_depth);
            pass.set_scissor_rect(x, y, x1 - x, y1 - y);
            let [cr, cg, cb, ca] = r.blend_constant.map(f64::from);
            pass.set_blend_constant(wgpu::Color { r: cr, g: cg, b: cb, a: ca });
            pass.set_stencil_reference(u32::from(ds.front.reference));
            for (g, group) in bind_groups.iter().enumerate() {
                pass.set_bind_group(g as u32, group, &[]);
            }
            for (slot, (buffer, at, size)) in vertex_buffers.iter().enumerate() {
                pass.set_vertex_buffer(slot as u32, buffer.slice(*at..at + size));
            }
            match &index_buffer {
                Some(((buffer, at, size), format)) => {
                    pass.set_index_buffer(buffer.slice(*at..at + size), *format);
                    pass.draw_indexed(0..d.count, d.base_vertex, 0..1);
                }
                None => pass.draw(d.first..d.first + d.count, 0..1),
            }
        }
        self.record(encoder);
        for (s, _) in colors.iter().flatten() {
            self.wrote((s.addr, s.format));
        }
        Ok(())
    }

    /// The translation and WGSL module of the program at `at`; a vertex
    /// program pairs with its pixel program's varyings.
    fn module(&mut self, mem: Mem, at: u64, pixel: Option<&Translation>, ints: &BTreeMap<u8, bool>, banks: &[Option<(u64, u32)>]) -> Module {
        let key = match self.cache.programs.get(&at) {
            Some(key) => *key,
            None => {
                let key = read_program(mem, at).map(|(key, code)| {
                    self.cache.code.insert(key, code);
                    key
                });
                self.cache.programs.insert(at, key);
                key
            }
        }
        .ok_or("program not readable")?;
        let pair = pixel.map(|p| p.key);
        let mask = ints.iter().fold(0u64, |m, (&l, &signed)| m | 1 << (l % 32) | u64::from(signed) << (32 + l % 32));
        let mut options = Options { varyings: pixel.map(|p| p.interface.varyings.clone()).unwrap_or_default(), int_attributes: ints.clone(), ..Default::default() };
        // Jump tables: the translation without them says where they are; the bound
        // banks' words give the targets, and the module with exact blocks is keyed
        // by them too (another table content, another module).
        let mut id = (key, pair, mask, 0);
        if let Some(Ok((t, _))) = self.cache.modules.get(&id)
            && let Some(tables) = &t.interface.jump_tables
            && let Some(targets) = jump_targets(tables, banks, mem, self.cache.code[&key].len())
        {
            id.3 = {
                use std::hash::{Hash, Hasher};
                let mut h = std::collections::hash_map::DefaultHasher::new();
                targets.hash(&mut h);
                h.finish() | 1
            };
            options.jump_targets = Some(targets);
        }
        if !self.cache.modules.contains_key(&id) {
            let made = shader::translate_with(&self.cache.code[&key], &options).map_err(|e| format!("program {key:016x}: {e}")).and_then(|t| {
                if let Err(e) = shader::validate(&t.wgsl) {
                    self.error(format!("program {key:016x}: {e}"));
                    return Err(format!("program {key:016x}: invalid WGSL"));
                }
                let m = self.device.create_shader_module(wgpu::ShaderModuleDescriptor { label: Some(&format!("{key:016x}")), source: wgpu::ShaderSource::Wgsl(t.wgsl.as_str().into()) });
                Ok((t, m))
            });
            self.cache.modules.insert(id, made);
        }
        self.cache.modules[&id].clone()
    }

    /// A stage's constant banks (64 KiB each, as bound) and textures.
    fn stage_bindings(&mut self, d: &Draw, stage: usize, t: &Translation, attached: &[u64], mem: Mem) -> Result<Bindings, String> {
        let handle_banks = Self::handle_banks(t);
        let mut banks = BTreeMap::new();
        for b in t.interface.cbufs.iter().copied().chain(handle_banks) {
            let (addr, size) = d.cbufs[stage].get(usize::from(b)).copied().flatten().ok_or_else(|| format!("constant bank {b} not bound"))?;
            let mut data = vec![0u8; CBUF_BYTES];
            let n = (size as usize).min(CBUF_BYTES);
            if !mem(addr, &mut data[..n]) {
                return Err("constant buffer not readable".into());
            }
            banks.insert(b, data);
        }
        let (textures, swizzles) = self.texture_slots(t, &banks, d.headers, d.samplers, d.sampler_binding, attached, mem)?;
        let cbufs = banks.into_iter().filter(|(b, _)| t.interface.cbufs.contains(b)).collect();
        Ok(Bindings { cbufs, textures, swizzles: t.interface.swizzles.then_some(swizzles) })
    }

    /// The constant banks a program's texture handles live in.
    pub(crate) fn handle_banks(t: &Translation) -> BTreeSet<u8> {
        let mut banks = BTreeSet::new();
        for x in &t.interface.textures {
            match &x.handle {
                Handle::CBuf { bank, .. } => {
                    banks.insert(*bank);
                }
                Handle::Uniform(e) => e.banks(&mut banks),
                _ => {}
            }
        }
        banks
    }

    /// A program's texture slots (view, sampler) and swizzle codes, from its
    /// handles in `banks` and the header and sampler pools.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn texture_slots(&mut self, t: &Translation, banks: &BTreeMap<u8, Vec<u8>>, headers: Pool, samplers: Pool, binding: SamplerBinding, attached: &[u64], mem: Mem) -> Result<(Vec<Slot>, Vec<u32>), String> {
        let mut textures = Vec::new();
        let mut swizzles = vec![0u32; t.interface.textures.len().next_multiple_of(4)];
        for (slot, x) in t.interface.textures.iter().enumerate() {
            let word = |bank: u8, offset: u16| -> Option<u32> {
                let at = usize::from(offset);
                let w = banks.get(&bank)?.get(at..at + 4)?;
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
            let mut header = [0u8; 32];
            if !mem(headers.entry(handle.header).ok_or("texture index past the pool")?, &mut header) {
                return Err("texture header not readable".into());
            }
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
            let view_desc = |aspect| wgpu::TextureViewDescriptor { dimension: Some(dimension), aspect, array_layer_count: Some(layers), mip_level_count: Some(1), ..Default::default() };
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
                });
                let dimension = if three_d { V::D3 } else { V::D2 };
                textures.push(Slot { view: texture.create_view(&Default::default()), sampler: None, depth: false, compare: false, dimension });
                continue;
            }
            // An image a compute program wrote, or a render target at that address: its host texture.
            let (view, depth, source) = match self.map.values().find(|e| e.surface.addr == tic.address) {
                // With the header's format when the image allows it as a view (sRGB over linear storage).
                _ if self.cache.images.contains_key(&tic.address) => {
                    let image = match self.cache.twins.get(&tic.address) {
                        Some(twin) => twin,
                        None => {
                            let twin = self.float_twin(&self.cache.images[&tic.address].clone())?;
                            self.cache.twins.entry(tic.address).or_insert(twin)
                        }
                    };
                    fits(image)?;
                    let view = crate::surfaces::view_as(image, textures::wgpu_texture_format(&tic), wgpu::TextureUsages::TEXTURE_BINDING, view_desc(wgpu::TextureAspect::All));
                    (view, false, "image")
                }
                Some(e) => {
                    fits(&e.texture)?;
                    let depth = e.texture.format().is_depth_stencil_format();
                    // A color target read counts as touched: frame dumps show what the frame sampled.
                    if !depth {
                        self.touched.insert((e.surface.addr, e.surface.format));
                    }
                    let aspect = if depth { wgpu::TextureAspect::DepthOnly } else { wgpu::TextureAspect::All };
                    (crate::surfaces::view_as(&e.texture, textures::wgpu_texture_format(&tic), wgpu::TextureUsages::TEXTURE_BINDING, view_desc(aspect)), depth, "target")
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
                        (depth.create_view(&view_desc(wgpu::TextureAspect::DepthOnly)), true, "memory")
                    }
                    Ok(texture) => {
                        let depth = texture.format().is_depth_stencil_format();
                        (texture.create_view(&view_desc(if depth { wgpu::TextureAspect::DepthOnly } else { wgpu::TextureAspect::All })), depth, "memory")
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
                    let mut bytes = [0u8; 32];
                    if !mem(samplers.entry(index).ok_or("sampler index past the pool")?, &mut bytes) {
                        return Err("sampler header not readable".into());
                    }
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
        let level = tic.view_levels.map_or(0, |v| v.0);
        let (width, height) = ((tic.width >> level).max(1), (tic.height >> level).max(1));
        let (columns, rows) = (width.div_ceil(bw), height.div_ceil(bh));
        let layers = textures::layers(tic);
        if layers == 0 || layers > self.device.limits().max_texture_array_layers {
            return Err(format!("texture of {layers} layers"));
        }
        let depth = if three_d { (tic.depth >> level).max(1) } else { layers };
        let end;
        let data = match tic.layout {
            Layout::BlockLinear => {
                let l = textures::tic_level(tic, level).ok_or("texture level")?;
                let offset = *textures::level_offsets(tic).get(level as usize).ok_or("texture level")?;
                let stride = textures::layer_stride(tic).ok_or("texture layer stride")?;
                let mut raw = vec![0u8; l.size() as usize];
                let mut data = Vec::new();
                for layer in 0..u64::from(layers) {
                    if !mem(tic.address + layer * stride + offset, &mut raw) {
                        return Err("texture not readable".into());
                    }
                    data.extend(l.deswizzle(&raw));
                }
                end = tic.address + u64::from(layers - 1) * stride + offset + raw.len() as u64;
                data
            }
            Layout::Pitch if level == 0 && !three_d && layers == 1 => {
                let row = (columns * bpe) as usize;
                let mut raw = vec![0u8; tic.pitch as usize * (rows as usize - 1) + row];
                if !mem(tic.address, &mut raw) {
                    return Err("texture not readable".into());
                }
                end = tic.address + raw.len() as u64;
                raw.chunks(tic.pitch as usize).flat_map(|r| &r[..row]).copied().collect()
            }
            other => return Err(format!("texture layout {other:?}")),
        };
        // Compressed formats need whole blocks: the host texture covers them.
        let texture = self.device.create_texture(&wgpu::TextureDescriptor {
            label: None,
            size: wgpu::Extent3d { width: columns * bw, height: rows * bh, depth_or_array_layers: depth },
            mip_level_count: 1,
            sample_count: 1,
            dimension: if three_d { wgpu::TextureDimension::D3 } else { wgpu::TextureDimension::D2 },
            format,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });
        self.queue.write_texture(
            texture.as_image_copy(),
            &data,
            wgpu::TexelCopyBufferLayout { offset: 0, bytes_per_row: Some(columns * bpe), rows_per_image: Some(rows) },
            texture.size(),
        );
        let texture = if zf32 { self.as_depth(&texture) } else { texture };
        // A large upload's staging memory is freed once the GPU is past it: at once, so
        // that two such uploads in a frame (a 128 MiB shadow atlas and a 128 MiB light
        // probe texture in the tutorial) do not exhaust it.
        if data.len() >= 32 << 20 {
            self.flush();
            let _ = self.device.poll(wgpu::PollType::wait_indefinitely());
        }
        self.cache.textures.insert((tic.address, header), (texture.clone(), end));
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

    /// Forgets the uploaded textures whose memory `[start, start + len)` overlaps:
    /// the GPU wrote there.
    pub fn invalidate(&mut self, start: u64, len: u64) {
        let end = start.saturating_add(len);
        self.cache.textures.retain(|(addr, _), (_, last)| *last <= start || *addr >= end);
        self.cache.compare_depths.clear();
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
        let banks = BTreeMap::from([(0u8, vec![0u8; 16])]);
        let (slots, _) = s.texture_slots(&t, &banks, resources::header_pool(&reg), resources::sampler_pool(&reg), resources::sampler_binding(&reg), &[], &mut mem).unwrap_or_else(|e| panic!("{e}"));
        let bindings = Bindings { cbufs: BTreeMap::new(), textures: slots, swizzles: None };
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
        // layers are 1024 bytes apart. Texel (x, y) of layer l is (40 l + x, y, l, 255).
        const TEX: u64 = 0x50_0000;
        let mut tic = [0u8; 32];
        put(&mut tic, &[(6, 0, 0x08), (9, 7, 2), (12, 10, 2), (15, 13, 2), (18, 16, 2), (21, 19, 2), (24, 22, 3), (27, 25, 4), (30, 28, 5)]);
        put(&mut tic, &[(63, 41, (TEX >> 9) as u32), (87, 85, 3), (127, 124, 1), (144, 128, 7), (175, 160, 7), (189, 176, 2), (154, 151, 5), (191, 191, 1)]);
        let t = textures::tic(&tic).unwrap();
        assert_eq!(textures::layer_stride(&t), Some(1024));
        let level = textures::tic_level(&t, 0).unwrap();
        let mut memory = vec![0u8; 3 * 1024];
        for l in 0..3u32 {
            for (x, y) in (0..8).flat_map(|x| (0..8).map(move |y| (x, y))) {
                let at = 1024 * l as usize + level.offset(4 * x, y, 0) as usize;
                memory[at..at + 4].copy_from_slice(&[(40 * l + x) as u8, y as u8, l as u8, 255]);
            }
        }
        let x = Texture { handle: Handle::CBuf { bank: 0, offset: 0 }, dim: TexDim::Array2D, compare: false, fetch: true, query: false };
        let wgsl = "@group(0) @binding(32) var tex0: texture_2d_array<f32>;
            @group(0) @binding(1) var<storage, read_write> out: array<f32>;
            @compute @workgroup_size(1) fn main() {
                for (var l = 0u; l < 3u; l++) {
                    let c = textureLoad(tex0, vec2<i32>(3, 5), l, 0);
                    out[4u * l] = c.r; out[4u * l + 1u] = c.g; out[4u * l + 2u] = c.b; out[4u * l + 3u] = f32(textureNumLayers(tex0));
                }
            }";
        let got = run(&mut s, x, tic, [0; 32], vec![(TEX, memory)], wgsl, 12);
        for l in 0..3 {
            let texel: Vec<u32> = got[4 * l..4 * l + 3].iter().map(|v| (v * 255.0).round() as u32).collect();
            assert_eq!(texel, vec![40 * l as u32 + 3, 5, l as u32], "layer {l}");
            assert_eq!(got[4 * l + 3], 3.0);
        }
        // The cached range covers the three layers (level 0 of the last one).
        assert_eq!(s.texture_ranges(), vec![(TEX, TEX + 2 * 1024 + 512)]);
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
}
