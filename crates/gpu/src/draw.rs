//! Draws on wgpu: a 3D-class draw (`pushbuf::Draw`) becomes a render pass
//! into its targets. The game's programs are translated to WGSL; vertices,
//! indices and constant buffers are read from guest memory at the draw;
//! textures come from a render target at the same address, else from
//! guest memory.
//!
//! ponytail: one encoder and fresh buffers per draw (ordering for free; a
//! ring buffer when draw counts make it slow); textures cached by address
//! and header, dropped when the GPU writes their memory (not when the CPU
//! does); float vertex attributes; no
//! depth-compare textures;
//! pipelines keyed by a Debug string. Each draw not done counts its reason
//! in `Surfaces::skipped`.

use crate::pushbuf::{Draw, Surface};
use crate::sass::texture::TexDim;
use crate::shader::{self, Handle, Options, Translation};
use crate::state::resources::{SamplerBinding, TexHandle};
use crate::state::textures::{self, Layout, TextureType};
use crate::state::vertex::{AttrKind, wgpu_vertex_format};
use crate::surfaces::{Surfaces, color_format, depth_format};
use std::collections::{BTreeMap, BTreeSet, HashMap};
use wgpu::util::DeviceExt;

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
    code: HashMap<u64, Vec<u8>>,
    /// Translations by (program key, key of the pixel program it pairs with).
    modules: HashMap<(u64, Option<u64>), Module>,
    pipelines: HashMap<String, (wgpu::RenderPipeline, Vec<wgpu::BindGroupLayout>)>,
    /// Uploaded textures by address and header, with the end of the memory they came from.
    textures: HashMap<(u64, [u8; 32]), (wgpu::Texture, u64)>,
    samplers: HashMap<([u8; 32], bool), wgpu::Sampler>,
}

/// A stage's bindings.
struct Bindings {
    /// Constant buffers by bank.
    cbufs: BTreeMap<u8, Vec<u8>>,
    textures: Vec<Slot>,
    /// Each slot's header swizzle, when the program reads them.
    swizzles: Option<Vec<u32>>,
}

/// A texture slot: its view, its sampler unless fetched, and whether it is
/// a depth target (read as unfilterable floats, with a non-filtering sampler).
struct Slot {
    view: wgpu::TextureView,
    sampler: Option<wgpu::Sampler>,
    depth: bool,
    dimension: wgpu::TextureViewDimension,
}

impl Bindings {
    /// The layout these bindings take in a stage.
    fn layout(&self, visibility: wgpu::ShaderStages) -> Vec<wgpu::BindGroupLayoutEntry> {
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
            let sample_type = wgpu::TextureSampleType::Float { filterable };
            entries.push(wgpu::BindGroupLayoutEntry { binding: 32 + 2 * i as u32, visibility, ty: wgpu::BindingType::Texture { sample_type, view_dimension: s.dimension, multisampled: false }, count: None });
            if s.sampler.is_some() {
                let ty = if s.depth { wgpu::SamplerBindingType::NonFiltering } else { wgpu::SamplerBindingType::Filtering };
                entries.push(wgpu::BindGroupLayoutEntry { binding: 33 + 2 * i as u32, visibility, ty: wgpu::BindingType::Sampler(ty), count: None });
            }
        }
        entries
    }
}

impl Surfaces {
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
            Some(at) => Some(self.module(mem, at, None)?),
            None => None,
        };
        let vs = self.module(mem, d.programs[1].ok_or("no vertex program")?, ps.as_ref().map(|p| &p.0))?;

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
            if matches!(a.format.kind, AttrKind::Uint | AttrKind::Sint) {
                return Err("integer vertex attribute".into());
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
            vertex_buffers.push(self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor { label: None, contents: &data, usage: wgpu::BufferUsages::VERTEX }));
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
        let key = format!("{:x}/{:x?}/{layouts:?}/{primitive:?}/{color_states:?}/{depth_state:?}/{:?}", vs.0.key, ps.as_ref().map(|p| p.0.key), &group_layouts[..groups]);
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
            let mut buffers: Vec<(u32, wgpu::Buffer)> = cbufs.iter().map(|(b, data)| (u32::from(*b), self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor { label: None, contents: data, usage: wgpu::BufferUsages::UNIFORM }))).collect();
            if let Some(codes) = swizzles {
                let bytes: Vec<u8> = codes.iter().flat_map(|c| c.to_le_bytes()).collect();
                buffers.push((shader::SWIZZLE_BINDING, self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor { label: None, contents: &bytes, usage: wgpu::BufferUsages::UNIFORM })));
            }
            let mut entries: Vec<wgpu::BindGroupEntry> = buffers.iter().map(|(b, buffer)| wgpu::BindGroupEntry { binding: *b, resource: buffer.as_entire_binding() }).collect();
            for (slot, Slot { view, sampler, .. }) in textures.iter().enumerate() {
                entries.push(wgpu::BindGroupEntry { binding: 32 + 2 * slot as u32, resource: wgpu::BindingResource::TextureView(view) });
                if let Some(sampler) = sampler {
                    entries.push(wgpu::BindGroupEntry { binding: 33 + 2 * slot as u32, resource: wgpu::BindingResource::Sampler(sampler) });
                }
            }
            bind_groups.push(self.device.create_bind_group(&wgpu::BindGroupDescriptor { label: None, layout, entries: &entries }));
        }
        let index_buffer = indices.as_ref().map(|(bytes, format)| {
            let mut padded = bytes.clone();
            padded.resize(bytes.len().next_multiple_of(4), 0);
            (self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor { label: None, contents: &padded, usage: wgpu::BufferUsages::INDEX }), *format)
        });

        // The pass.
        let color_views: Vec<Option<wgpu::TextureView>> = colors.iter().map(|c| c.map(|(s, f)| self.texture(&s, f))).collect();
        let depth_view = depth.map(|(s, f)| (self.texture(&s, f), f));
        let mut encoder = self.device.create_command_encoder(&Default::default());
        {
            let load = wgpu::Operations { load: wgpu::LoadOp::Load, store: wgpu::StoreOp::Store };
            let attachments: Vec<Option<wgpu::RenderPassColorAttachment>> = color_views.iter().map(|v| v.as_ref().map(|view| wgpu::RenderPassColorAttachment { view, depth_slice: None, resolve_target: None, ops: load })).collect();
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("draw"),
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
            for (slot, buffer) in vertex_buffers.iter().enumerate() {
                pass.set_vertex_buffer(slot as u32, buffer.slice(..));
            }
            match &index_buffer {
                Some((buffer, format)) => {
                    pass.set_index_buffer(buffer.slice(..), *format);
                    pass.draw_indexed(0..d.count, d.base_vertex, 0..1);
                }
                None => pass.draw(d.first..d.first + d.count, 0..1),
            }
        }
        self.queue.submit([encoder.finish()]);
        for (s, _) in colors.iter().flatten() {
            self.touched.insert((s.addr, s.format));
        }
        Ok(())
    }

    /// The translation and WGSL module of the program at `at`; a vertex
    /// program pairs with its pixel program's varyings.
    fn module(&mut self, mem: Mem, at: u64, pixel: Option<&Translation>) -> Module {
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
        if !self.cache.modules.contains_key(&(key, pair)) {
            let options = Options { varyings: pixel.map(|p| p.interface.varyings.clone()).unwrap_or_default() };
            let made = shader::translate_with(&self.cache.code[&key], &options).map_err(|e| format!("program {key:016x}: {e}")).and_then(|t| {
                if let Err(e) = shader::validate(&t.wgsl) {
                    self.error(format!("program {key:016x}: {e}"));
                    return Err(format!("program {key:016x}: invalid WGSL"));
                }
                let m = self.device.create_shader_module(wgpu::ShaderModuleDescriptor { label: Some(&format!("{key:016x}")), source: wgpu::ShaderSource::Wgsl(t.wgsl.as_str().into()) });
                Ok((t, m))
            });
            self.cache.modules.insert((key, pair), made);
        }
        self.cache.modules[&(key, pair)].clone()
    }

    /// A stage's constant banks (64 KiB each, as bound) and textures.
    fn stage_bindings(&mut self, d: &Draw, stage: usize, t: &Translation, attached: &[u64], mem: Mem) -> Result<Bindings, String> {
        let mut handle_banks = BTreeSet::new();
        for x in &t.interface.textures {
            match &x.handle {
                Handle::CBuf { bank, .. } => {
                    handle_banks.insert(*bank);
                }
                Handle::Uniform(e) => e.banks(&mut handle_banks),
                _ => {}
            }
        }
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
            if x.compare || !matches!(x.dim, TexDim::D2 | TexDim::D3) {
                return Err(format!("texture {:?} (compare {})", x.dim, x.compare));
            }
            let mut header = [0u8; 32];
            if !mem(d.headers.entry(handle.header).ok_or("texture index past the pool")?, &mut header) {
                return Err("texture header not readable".into());
            }
            let tic = textures::tic(&header).ok_or("texture header layout")?;
            if attached.contains(&tic.address) {
                return Err("texture is a target of the same draw".into());
            }
            let want = if x.dim == TexDim::D3 { TextureType::ThreeD } else { TextureType::TwoD };
            if want != tic.texture_type && !(want == TextureType::TwoD && tic.texture_type == TextureType::TwoDNoMipmap) {
                return Err(format!("{:?} texture read as {:?}", tic.texture_type, x.dim));
            }
            // A render target at that address: its host texture.
            let (view, depth) = match self.map.values().find(|e| e.surface.addr == tic.address) {
                Some(e) => {
                    let depth = e.texture.format().is_depth_stencil_format();
                    let aspect = if depth { wgpu::TextureAspect::DepthOnly } else { wgpu::TextureAspect::All };
                    (e.texture.create_view(&wgpu::TextureViewDescriptor { dimension: Some(wgpu::TextureViewDimension::D2), aspect, array_layer_count: Some(1), mip_level_count: Some(1), ..Default::default() }), depth)
                }
                None => (self.upload(&tic, header, mem)?.create_view(&Default::default()), false),
            };
            swizzles[slot] = textures::swizzle_code(&tic);
            self.textures_read.push(tic.address);
            let sampler = match x.fetch {
                true => None,
                false => {
                    let index = match d.sampler_binding {
                        SamplerBinding::Independently => handle.sampler,
                        SamplerBinding::ViaHeaderBinding => handle.header,
                    };
                    let mut bytes = [0u8; 32];
                    if !mem(d.samplers.entry(index).ok_or("sampler index past the pool")?, &mut bytes) {
                        return Err("sampler header not readable".into());
                    }
                    Some(self.sampler(bytes, !depth))
                }
            };
            let dimension = if x.dim == TexDim::D3 { wgpu::TextureViewDimension::D3 } else { wgpu::TextureViewDimension::D2 };
            textures.push(Slot { view, sampler, depth, dimension });
        }
        let cbufs = banks.into_iter().filter(|(b, _)| t.interface.cbufs.contains(b)).collect();
        Ok(Bindings { cbufs, textures, swizzles: t.interface.swizzles.then_some(swizzles) })
    }

    /// A 2D or 3D texture from guest memory: the level the view starts at, layer 0.
    fn upload(&mut self, tic: &textures::Tic, header: [u8; 32], mem: Mem) -> Result<wgpu::Texture, String> {
        if let Some((t, _)) = self.cache.textures.get(&(tic.address, header)) {
            return Ok(t.clone());
        }
        let three_d = tic.texture_type == TextureType::ThreeD;
        let format = textures::wgpu_texture_format(tic).ok_or_else(|| format!("texture format {:?}", tic.components))?;
        if let Some(note) = textures::format_notes(tic).into_iter().find(|n| !n.starts_with("needs Features::")) {
            return Err(format!("texture format {:?}: {note}", tic.components));
        }
        if format.is_depth_stencil_format() || !self.device.features().contains(format.required_features()) {
            return Err(format!("texture format {format:?}"));
        }
        let (bpe, bw, bh) = tic.components.element().ok_or("texture element size")?;
        let level = tic.view_levels.map_or(0, |v| v.0);
        let (width, height) = ((tic.width >> level).max(1), (tic.height >> level).max(1));
        let (columns, rows) = (width.div_ceil(bw), height.div_ceil(bh));
        let depth = if three_d { (tic.depth >> level).max(1) } else { 1 };
        let end;
        let data = match tic.layout {
            Layout::BlockLinear => {
                let l = textures::tic_level(tic, level).ok_or("texture level")?;
                let offset = *textures::level_offsets(tic).get(level as usize).ok_or("texture level")?;
                let mut raw = vec![0u8; l.size() as usize];
                if !mem(tic.address + offset, &mut raw) {
                    return Err("texture not readable".into());
                }
                end = tic.address + offset + raw.len() as u64;
                l.deswizzle(&raw)
            }
            Layout::Pitch if level == 0 && !three_d => {
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
        self.cache.textures.insert((tic.address, header), (texture.clone(), end));
        Ok(texture)
    }

    /// Forgets the uploaded textures whose memory `[start, start + len)` overlaps:
    /// the GPU wrote there.
    pub fn invalidate(&mut self, start: u64, len: u64) {
        let end = start.saturating_add(len);
        self.cache.textures.retain(|(addr, _), (_, last)| *last <= start || *addr >= end);
    }

    /// The sampler of a header; a non-filtering one (nearest everywhere) for
    /// unfilterable textures.
    fn sampler(&mut self, bytes: [u8; 32], filtering: bool) -> wgpu::Sampler {
        if let Some(s) = self.cache.samplers.get(&(bytes, filtering)) {
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
            compare: None,
            anisotropy_clamp: if linear { t.max_anisotropy.clamp(1, 16) } else { 1 },
            border_color: address.contains(&wgpu::AddressMode::ClampToBorder).then(|| t.wgpu_border_color().unwrap_or(wgpu::SamplerBorderColor::TransparentBlack)),
        });
        self.cache.samplers.insert((bytes, filtering), s.clone());
        s
    }
}
