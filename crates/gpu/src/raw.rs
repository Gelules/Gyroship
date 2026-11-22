//! The raw Vulkan path (D110 step 6): the game's draws recorded with ash beside wgpu, through
//! monolithic pipelines in dynamic rendering, made as wgpu-hal makes its own (wgpu-hal 30.0.1,
//! `Device::create_render_pipeline`, `create_bind_group_layout`, `create_sampler`): the same SPIR-V
//! (`vk::spirv`) and the same states, mapped by wgpu-hal's own `conv`, so that they draw the same
//! pixels (`a_raw_draw_paints_what_wgpu_paints`).

use crate::surfaces::FxMap;
use ash::vk;
use wgpu::hal::{api::Vulkan, vulkan::conv};

/// What a raw render pipeline is made from: the parts of wgpu's descriptor, as draw.rs gives them,
/// with the modules and the layout already raw. Multisampling is wgpu's default, as in draw.rs.
pub(crate) struct PipelineDesc<'a> {
    /// The vertex and pixel modules ([`Raw::module`]), and their entry points (the game's are
    /// `main`; the fill module's `vs` and `fs`).
    pub vertex: vk::ShaderModule,
    pub pixel: Option<vk::ShaderModule>,
    pub entries: [&'a std::ffi::CStr; 2],
    /// The vertex buffers by slot.
    pub buffers: &'a [Option<wgpu::VertexBufferLayout<'a>>],
    pub primitive: wgpu::PrimitiveState,
    pub depth_stencil: Option<&'a wgpu::DepthStencilState>,
    pub targets: &'a [Option<wgpu::ColorTargetState>],
    pub layout: vk::PipelineLayout,
}

/// A binding's resource in a raw descriptor set.
pub(crate) enum Resource {
    /// A buffer, its offset and size.
    Buffer(vk::Buffer, u64, u64),
    /// A texture view, in the layout wgpu-hal gives its use.
    Texture(vk::ImageView, vk::ImageLayout),
    Sampler(vk::Sampler),
}

/// Sets per descriptor pool: a new pool is made when the last is full.
const POOL_SETS: u32 = 1024;

/// The raw objects made on a device, destroyed with it.
pub(crate) struct Raw {
    /// Keeps the VkDevice until the objects below are destroyed (`Drop`).
    device: wgpu::Device,
    adapter: wgpu::Adapter,
    pub(crate) ash: ash::Device,
    /// Each stage may run with varying subgroup sizes, as wgpu-hal asks with Features::SUBGROUP.
    varying: bool,
    set_layouts: FxMap<Vec<wgpu::BindGroupLayoutEntry>, vk::DescriptorSetLayout>,
    pipeline_layouts: FxMap<Vec<vk::DescriptorSetLayout>, vk::PipelineLayout>,
    modules: Vec<vk::ShaderModule>,
    pipelines: Vec<vk::Pipeline>,
    samplers: Vec<vk::Sampler>,
}

/// A raw pipeline as the recorder binds it: the pipeline, its layout, and each group's set
/// layout with the entries its sets are written in.
pub(crate) struct RawPipeline {
    pub(crate) pipeline: vk::Pipeline,
    pub(crate) layout: vk::PipelineLayout,
    pub(crate) sets: smallvec::SmallVec<[(vk::DescriptorSetLayout, std::sync::Arc<[wgpu::BindGroupLayoutEntry]>); 2]>,
}

/// `GYROSHIP_RAW=<program key>` (D110 step 6a): the draws of the pipelines that take this vertex
/// or pixel program are recorded in raw Vulkan; `all`: every draw's. Read at the device's opening.
pub(crate) const RAW: &str = "GYROSHIP_RAW";

/// The raw path when it is on: its objects, the program whose pipelines are raw (None: all),
/// the SPIR-V options of the device, and each module's raw module by serial.
pub(crate) struct RawMode {
    pub(crate) raw: Raw,
    pub(crate) program: Option<u64>,
    pub(crate) options: naga::back::spv::Options<'static>,
    pub(crate) modules: FxMap<u32, vk::ShaderModule>,
    /// The fill module's raw pipelines, by format and kind (every draw raw: step 6b).
    pub(crate) fills: FxMap<(wgpu::TextureFormat, crate::surfaces::Fill), std::sync::Arc<RawPipeline>>,
}

impl RawMode {
    /// The raw path asked for by `value` (`RAW`'s), when the device can: Vulkan with dynamic
    /// rendering; else none, said on stderr.
    pub(crate) fn new(adapter: &wgpu::Adapter, device: &wgpu::Device, extras: crate::vk::Extras, value: &str) -> Option<RawMode> {
        let program = match value {
            "all" => None,
            key => match u64::from_str_radix(key.trim_start_matches("0x"), 16) {
                Ok(key) => Some(key),
                Err(_) => {
                    eprintln!("gyroship: {RAW}={key}: not a program key (hex) nor `all`: the raw path stays off");
                    return None;
                }
            },
        };
        let mode = Raw::new(adapter, device).filter(|_| extras.dynamic_rendering).zip(crate::vk::spirv_options(adapter, device)).map(|(raw, options)| RawMode { raw, program, options, modules: FxMap::default(), fills: FxMap::default() });
        match (&mode, program) {
            (None, _) => eprintln!("gyroship: the raw Vulkan path needs Vulkan with dynamic rendering, which this device lacks: the game draws through wgpu"),
            (Some(_), Some(key)) => eprintln!("gyroship: the draws of program {key:016x} recorded in raw Vulkan"),
            (Some(_), None) => eprintln!("gyroship: every draw recorded in raw Vulkan"),
        }
        mode
    }

    /// Whether the pipeline of these programs is raw.
    pub(crate) fn wants(&self, vertex: u64, pixel: Option<u64>) -> bool {
        self.program.is_none_or(|p| p == vertex || pixel == Some(p))
    }

    /// The fill module's raw pipeline for `targets` and `depth` (made once, kept by `key`): its
    /// entry points `vs` and `fs`, one group with its uniform at binding 0, as `Surfaces::fill`'s.
    pub(crate) fn fill(&mut self, key: (wgpu::TextureFormat, crate::surfaces::Fill), targets: &[Option<wgpu::ColorTargetState>], depth: Option<&wgpu::DepthStencilState>) -> Result<std::sync::Arc<RawPipeline>, String> {
        if let Some(p) = self.fills.get(&key) {
            return Ok(p.clone());
        }
        let module = naga::front::wgsl::parse_str(crate::surfaces::FILL).map_err(|e| e.message().to_string())?;
        let info = naga::valid::Validator::new(naga::valid::ValidationFlags::all(), naga::valid::Capabilities::all()).validate(&module).map_err(|e| e.to_string())?;
        let entries = [crate::surfaces::fill_entry()];
        let map = crate::vk::binding_map(&[&entries]);
        let vertex = self.raw.module(&crate::vk::spirv(&module, &info, naga::ShaderStage::Vertex, "vs", &self.options, &map)?)?;
        let pixel = match targets.is_empty() {
            true => None,
            false => Some(self.raw.module(&crate::vk::spirv(&module, &info, naga::ShaderStage::Fragment, "fs", &self.options, &map)?)?),
        };
        let set = self.raw.set_layout(&entries)?;
        let layout = self.raw.pipeline_layout(&[set])?;
        let pipeline = self.raw.pipeline(&PipelineDesc { vertex, pixel, entries: [c"vs", c"fs"], buffers: &[], primitive: Default::default(), depth_stencil: depth, targets, layout })?;
        let p = std::sync::Arc::new(RawPipeline { pipeline, layout, sets: smallvec::smallvec![(set, std::sync::Arc::from(&entries[..]))] });
        self.fills.insert(key, p.clone());
        Ok(p)
    }
}

/// What wgpu-core refuses in a render pipeline that draw.rs lets through (wgpu 30: `Device::
/// create_render_pipeline`'s checks), refused here before a raw one is made, by name: there, the
/// frame's encoder would be lost; here, the draws of this pipeline only. Blending is checked
/// against the adapter's own format features, which the device asks for on Vulkan.
pub(crate) fn refused(buffers: &[Option<wgpu::VertexBufferLayout>], targets: &[Option<wgpu::ColorTargetState>], depth: Option<&wgpu::DepthStencilState>, adapter: &wgpu::Adapter) -> Option<&'static str> {
    let limits = adapter.limits();
    if targets.iter().flatten().any(|t| t.blend.is_some() && !adapter.get_texture_format_features(t.format).flags.contains(wgpu::TextureFormatFeatureFlags::BLENDABLE)) {
        return Some("raw pipeline: blending on a format that cannot blend");
    }
    for b in buffers.iter().flatten() {
        if b.array_stride > u64::from(limits.max_vertex_buffer_array_stride) || b.array_stride % wgpu::VERTEX_ALIGNMENT != 0 {
            return Some("raw pipeline: vertex stride wgpu refuses");
        }
        let room = if b.array_stride == 0 { u64::from(limits.max_vertex_buffer_array_stride) } else { b.array_stride };
        if b.attributes.iter().any(|a| a.offset + a.format.size() > room || a.offset % a.format.size().min(4) != 0 || a.shader_location >= limits.max_vertex_attributes) {
            return Some("raw pipeline: vertex attribute wgpu refuses");
        }
    }
    use wgpu::BlendFactor as F;
    let blends = || targets.iter().flatten().filter_map(|t| t.blend).flat_map(|b| [b.color, b.alpha]);
    if blends().any(|c| [c.src_factor, c.dst_factor].iter().any(|f| matches!(f, F::Src1 | F::OneMinusSrc1 | F::Src1Alpha | F::OneMinusSrc1Alpha))) {
        return Some("raw pipeline: dual-source blending (not asked of the device)");
    }
    if blends().any(|c| matches!(c.operation, wgpu::BlendOperation::Min | wgpu::BlendOperation::Max) && (c.src_factor != F::One || c.dst_factor != F::One)) {
        return Some("raw pipeline: min or max blending with a factor other than one");
    }
    if depth.is_some_and(|d| d.stencil.is_enabled() && !d.format.has_stencil_aspect()) {
        return Some("raw pipeline: stencil test on a format without stencil");
    }
    None
}

/// A viewport wgpu-core refuses (`RenderPass::set_viewport`'s checks), refused here before a raw
/// draw, by name.
pub(crate) fn viewport_refused([x, y, w, h, near, far]: [f32; 6], limits: &wgpu::Limits) -> Option<&'static str> {
    let max = limits.max_texture_dimension_2d as f32;
    if w < 0.0 || h < 0.0 || w > max || h > max || x < -2.0 * max || y < -2.0 * max || x + w > 2.0 * max - 1.0 || y + h > 2.0 * max - 1.0 || !(0.0..=1.0).contains(&near) || !(0.0..=1.0).contains(&far) || near > far {
        return Some("raw draw: viewport wgpu refuses");
    }
    None
}

/// Clears the whole of a target as wgpu's clear pass does: an instance of rendering that clears
/// its color (made from wgpu's f64 values as wgpu-hal's `make_vk_clear_color` makes it), or its
/// depth and stencil (each kept when None), stores, and ends.
#[allow(clippy::too_many_arguments)]
pub(crate) fn clear(ash: &ash::Device, cb: vk::CommandBuffer, (width, height): (u32, u32), view: vk::ImageView, format: wgpu::TextureFormat, [r, g, b, a]: [f64; 4], depth: Option<f32>, stencil: Option<u32>) {
    let attachment = |layout, clear: bool, value| vk::RenderingAttachmentInfo::default().image_view(view).image_layout(layout).load_op(if clear { vk::AttachmentLoadOp::CLEAR } else { vk::AttachmentLoadOp::LOAD }).store_op(vk::AttachmentStoreOp::STORE).clear_value(value);
    let none = vk::RenderingAttachmentInfo::default();
    let area = vk::Rect2D { offset: vk::Offset2D::default(), extent: vk::Extent2D { width, height } };
    let ds = vk::ClearValue { depth_stencil: vk::ClearDepthStencilValue { depth: depth.unwrap_or(0.0), stencil: stencil.unwrap_or(0) } };
    let (colors, d, s) = if format.has_depth_aspect() || format.has_stencil_aspect() {
        let layout = vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL;
        (Vec::new(), if format.has_depth_aspect() { attachment(layout, depth.is_some(), ds) } else { none }, if format.has_stencil_aspect() { attachment(layout, stencil.is_some(), ds) } else { none })
    } else {
        let color = match format.sample_type(None, None) {
            Some(wgpu::TextureSampleType::Sint) => vk::ClearColorValue { int32: [r as i32, g as i32, b as i32, a as i32] },
            Some(wgpu::TextureSampleType::Uint) => vk::ClearColorValue { uint32: [r as u32, g as u32, b as u32, a as u32] },
            _ => vk::ClearColorValue { float32: [r as f32, g as f32, b as f32, a as f32] },
        };
        (vec![attachment(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL, true, vk::ClearValue { color })], none, none)
    };
    let info = vk::RenderingInfo::default().render_area(area).layer_count(1).color_attachments(&colors).depth_attachment(&d).stencil_attachment(&s);
    unsafe {
        ash.cmd_begin_rendering(cb, &info);
        ash.cmd_end_rendering(cb);
    }
}

impl crate::surfaces::Surfaces {
    /// The raw pipeline of these modules and states: each module's SPIR-V written once (by
    /// serial), its bindings numbered for these groups (which a module's own interface fixes), a
    /// set layout per group, the pipeline layout, then `Raw::pipeline`.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn raw_pipeline(
        &mut self,
        vs: &crate::draw::Built,
        ps: Option<&crate::draw::Built>,
        buffers: &[Option<wgpu::VertexBufferLayout>],
        primitive: wgpu::PrimitiveState,
        targets: &[Option<wgpu::ColorTargetState>],
        depth: Option<&wgpu::DepthStencilState>,
        groups: &[&[wgpu::BindGroupLayoutEntry]],
    ) -> Result<RawPipeline, String> {
        if let Some(why) = refused(buffers, targets, depth, &self.adapter) {
            return Err(why.into());
        }
        let (Some(m), ir) = (self.raw.as_mut(), &self.cache.ir) else { return Err("the raw path is off".into()) };
        let map = crate::vk::binding_map(groups);
        let mut module = |b: &crate::draw::Built| -> Result<vk::ShaderModule, String> {
            if let Some(&made) = m.modules.get(&b.2) {
                return Ok(made);
            }
            let (module, info) = ir.get(&b.2).ok_or_else(|| format!("program {:016x}: its naga module was not kept", b.0.key))?;
            let entry = module.entry_points.first().ok_or("no entry point")?;
            let made = m.raw.module(&crate::vk::spirv(module, info, entry.stage, &entry.name, &m.options, &map)?)?;
            m.modules.insert(b.2, made);
            Ok(made)
        };
        let vertex = module(vs)?;
        let pixel = ps.map(&mut module).transpose()?;
        let sets = groups.iter().map(|e| Ok((m.raw.set_layout(e)?, std::sync::Arc::from(*e)))).collect::<Result<smallvec::SmallVec<[_; 2]>, String>>()?;
        let layout = m.raw.pipeline_layout(&sets.iter().map(|s| s.0).collect::<Vec<_>>())?;
        let pipeline = m.raw.pipeline(&PipelineDesc { vertex, pixel, entries: [c"main", c"main"], buffers, primitive, depth_stencil: depth, targets, layout })?;
        Ok(RawPipeline { pipeline, layout, sets })
    }
}

impl Raw {
    /// The raw path of `device`, off Vulkan none.
    pub(crate) fn new(adapter: &wgpu::Adapter, device: &wgpu::Device) -> Option<Raw> {
        let ash = unsafe { device.as_hal::<Vulkan>() }?.raw_device().clone();
        let varying = device.features().contains(wgpu::Features::SUBGROUP);
        Some(Raw {
            device: device.clone(),
            adapter: adapter.clone(),
            ash,
            varying,
            set_layouts: FxMap::default(),
            pipeline_layouts: FxMap::default(),
            modules: Vec::new(),
            pipelines: Vec::new(),
            samplers: Vec::new(),
        })
    }

    /// `format` as wgpu-hal makes it (Depth24PlusStencil8 may be D32_SFLOAT_S8_UINT).
    pub(crate) fn format(&self, format: wgpu::TextureFormat) -> vk::Format {
        unsafe { self.adapter.as_hal::<Vulkan>() }.map_or(vk::Format::UNDEFINED, |a| a.texture_format_as_raw(format))
    }

    /// A module from SPIR-V words (`vk::spirv`).
    pub(crate) fn module(&mut self, words: &[u32]) -> Result<vk::ShaderModule, String> {
        let m = unsafe { self.ash.create_shader_module(&vk::ShaderModuleCreateInfo::default().code(words), None) }.map_err(|e| format!("raw shader module: {e}"))?;
        self.modules.push(m);
        Ok(m)
    }

    /// The set layout of a bind group with `entries`: one binding per entry, numbered from 0 in
    /// the order of their numbers (as `vk::binding_map` numbers the shaders' bindings).
    pub(crate) fn set_layout(&mut self, entries: &[wgpu::BindGroupLayoutEntry]) -> Result<vk::DescriptorSetLayout, String> {
        if let Some(&l) = self.set_layouts.get(entries) {
            return Ok(l);
        }
        let mut sorted = entries.to_vec();
        sorted.sort_unstable_by_key(|e| e.binding);
        let bindings: Vec<vk::DescriptorSetLayoutBinding> = (0u32..)
            .zip(&sorted)
            .map(|(at, e)| {
                vk::DescriptorSetLayoutBinding::default()
                    .binding(at)
                    .descriptor_type(conv::map_binding_type(e.ty))
                    .descriptor_count(e.count.map_or(1, std::num::NonZeroU32::get))
                    .stage_flags(conv::map_shader_stage(e.visibility))
            })
            .collect();
        let l = unsafe { self.ash.create_descriptor_set_layout(&vk::DescriptorSetLayoutCreateInfo::default().bindings(&bindings), None) }.map_err(|e| format!("raw set layout: {e}"))?;
        self.set_layouts.insert(entries.to_vec(), l);
        Ok(l)
    }

    /// The pipeline layout of `sets`, in group order.
    pub(crate) fn pipeline_layout(&mut self, sets: &[vk::DescriptorSetLayout]) -> Result<vk::PipelineLayout, String> {
        if let Some(&l) = self.pipeline_layouts.get(sets) {
            return Ok(l);
        }
        let l = unsafe { self.ash.create_pipeline_layout(&vk::PipelineLayoutCreateInfo::default().set_layouts(sets), None) }.map_err(|e| format!("raw pipeline layout: {e}"))?;
        self.pipeline_layouts.insert(sets.to_vec(), l);
        Ok(l)
    }

    /// A render pipeline, made as wgpu-hal makes `d` (its states through wgpu-hal's `conv`), for
    /// dynamic rendering instead of a render pass. No pipeline cache (ours is step 7's).
    pub(crate) fn pipeline(&mut self, d: &PipelineDesc) -> Result<vk::Pipeline, String> {
        let dynamic = [vk::DynamicState::VIEWPORT, vk::DynamicState::SCISSOR, vk::DynamicState::BLEND_CONSTANTS, vk::DynamicState::STENCIL_REFERENCE];
        let (mut bindings, mut attributes) = (Vec::new(), Vec::new());
        for (slot, b) in (0u32..).zip(d.buffers) {
            let Some(b) = b else { continue };
            let rate = match b.step_mode {
                wgpu::VertexStepMode::Vertex => vk::VertexInputRate::VERTEX,
                wgpu::VertexStepMode::Instance => vk::VertexInputRate::INSTANCE,
            };
            bindings.push(vk::VertexInputBindingDescription { binding: slot, stride: b.array_stride as u32, input_rate: rate });
            attributes.extend(b.attributes.iter().map(|a| vk::VertexInputAttributeDescription { location: a.shader_location, binding: slot, format: conv::map_vertex_format(a.format), offset: a.offset as u32 }));
        }
        let vertex_input = vk::PipelineVertexInputStateCreateInfo::default().vertex_binding_descriptions(&bindings).vertex_attribute_descriptions(&attributes);
        let assembly = vk::PipelineInputAssemblyStateCreateInfo::default().topology(conv::map_topology(d.primitive.topology)).primitive_restart_enable(d.primitive.strip_index_format.is_some());
        let flags = if self.varying { vk::PipelineShaderStageCreateFlags::ALLOW_VARYING_SUBGROUP_SIZE } else { vk::PipelineShaderStageCreateFlags::empty() };
        let stage = |stage, module, name| vk::PipelineShaderStageCreateInfo::default().flags(flags).stage(stage).module(module).name(name);
        let stages: Vec<_> = std::iter::once(stage(vk::ShaderStageFlags::VERTEX, d.vertex, d.entries[0])).chain(d.pixel.map(|p| stage(vk::ShaderStageFlags::FRAGMENT, p, d.entries[1]))).collect();
        let mut raster = vk::PipelineRasterizationStateCreateInfo::default()
            .polygon_mode(conv::map_polygon_mode(d.primitive.polygon_mode))
            .front_face(conv::map_front_face(d.primitive.front_face))
            .line_width(1.0)
            .depth_clamp_enable(d.primitive.unclipped_depth);
        if let Some(face) = d.primitive.cull_mode {
            raster = raster.cull_mode(conv::map_cull_face(face));
        }
        if d.primitive.conservative {
            return Err("raw pipeline: conservative rasterization".into());
        }
        let mut depth = vk::PipelineDepthStencilStateCreateInfo::default();
        let (mut depth_format, mut stencil_format) = (vk::Format::UNDEFINED, vk::Format::UNDEFINED);
        if let Some(ds) = d.depth_stencil {
            let raw = self.format(ds.format);
            depth_format = if ds.format.has_depth_aspect() { raw } else { vk::Format::UNDEFINED };
            stencil_format = if ds.format.has_stencil_aspect() { raw } else { vk::Format::UNDEFINED };
            if ds.is_depth_enabled() {
                depth = depth.depth_test_enable(true).depth_write_enable(ds.depth_write_enabled.unwrap_or_default()).depth_compare_op(conv::map_comparison(ds.depth_compare.unwrap_or_default()));
            }
            if ds.stencil.is_enabled() {
                let s = &ds.stencil;
                depth = depth.stencil_test_enable(true).front(conv::map_stencil_face(&s.front, s.read_mask, s.write_mask)).back(conv::map_stencil_face(&s.back, s.read_mask, s.write_mask));
            }
            if ds.bias.is_enabled() {
                raster = raster.depth_bias_enable(true).depth_bias_constant_factor(ds.bias.constant as f32).depth_bias_clamp(ds.bias.clamp).depth_bias_slope_factor(ds.bias.slope_scale);
            }
        }
        let viewport = vk::PipelineViewportStateCreateInfo::default().scissor_count(1).viewport_count(1);
        let ms = wgpu::MultisampleState::default();
        let mask = [ms.mask as u32, (ms.mask >> 32) as u32];
        let multisample = vk::PipelineMultisampleStateCreateInfo::default().rasterization_samples(vk::SampleCountFlags::from_raw(ms.count)).alpha_to_coverage_enable(ms.alpha_to_coverage_enabled).sample_mask(&mask);
        let mut blends = Vec::with_capacity(d.targets.len());
        let mut formats = Vec::with_capacity(d.targets.len());
        for t in d.targets {
            let Some(t) = t else {
                blends.push(vk::PipelineColorBlendAttachmentState::default());
                formats.push(vk::Format::UNDEFINED);
                continue;
            };
            let mut b = vk::PipelineColorBlendAttachmentState::default().color_write_mask(vk::ColorComponentFlags::from_raw(t.write_mask.bits()));
            if let Some(blend) = &t.blend {
                let (color_op, color_src, color_dst) = conv::map_blend_component(&blend.color);
                let (alpha_op, alpha_src, alpha_dst) = conv::map_blend_component(&blend.alpha);
                b = b.blend_enable(true).color_blend_op(color_op).src_color_blend_factor(color_src).dst_color_blend_factor(color_dst).alpha_blend_op(alpha_op).src_alpha_blend_factor(alpha_src).dst_alpha_blend_factor(alpha_dst);
            }
            blends.push(b);
            formats.push(self.format(t.format));
        }
        let blend = vk::PipelineColorBlendStateCreateInfo::default().attachments(&blends);
        let dynamic = vk::PipelineDynamicStateCreateInfo::default().dynamic_states(&dynamic);
        let mut rendering = vk::PipelineRenderingCreateInfo::default().color_attachment_formats(&formats).depth_attachment_format(depth_format).stencil_attachment_format(stencil_format);
        let info = vk::GraphicsPipelineCreateInfo::default()
            .push_next(&mut rendering)
            .layout(d.layout)
            .stages(&stages)
            .vertex_input_state(&vertex_input)
            .input_assembly_state(&assembly)
            .rasterization_state(&raster)
            .viewport_state(&viewport)
            .multisample_state(&multisample)
            .depth_stencil_state(&depth)
            .color_blend_state(&blend)
            .dynamic_state(&dynamic);
        let p = unsafe { self.ash.create_graphics_pipelines(vk::PipelineCache::null(), &[info], None) }.map_err(|(_, e)| format!("raw pipeline: {e}"))?[0];
        self.pipelines.push(p);
        Ok(p)
    }

    /// A sampler made as wgpu-core then wgpu-hal make `d`: the anisotropy wgpu-core passes on
    /// (at most 16, or 1 without anisotropic filtering), the rest field by field.
    pub(crate) fn sampler(&mut self, d: &wgpu::SamplerDescriptor) -> Result<vk::Sampler, String> {
        let anisotropic = self.adapter.get_downlevel_capabilities().flags.contains(wgpu::DownlevelFlags::ANISOTROPIC_FILTERING);
        let anisotropy = if anisotropic { d.anisotropy_clamp.min(16) } else { 1 };
        let mut info = vk::SamplerCreateInfo::default()
            .mag_filter(conv::map_filter_mode(d.mag_filter))
            .min_filter(conv::map_filter_mode(d.min_filter))
            .mipmap_mode(conv::map_mip_filter_mode(d.mipmap_filter))
            .address_mode_u(conv::map_address_mode(d.address_mode_u))
            .address_mode_v(conv::map_address_mode(d.address_mode_v))
            .address_mode_w(conv::map_address_mode(d.address_mode_w))
            .min_lod(d.lod_min_clamp)
            .max_lod(d.lod_max_clamp);
        if let Some(f) = d.compare {
            info = info.compare_enable(true).compare_op(conv::map_comparison(f));
        }
        if anisotropy != 1 {
            info = info.anisotropy_enable(true).max_anisotropy(f32::from(anisotropy));
        }
        if let Some(color) = d.border_color {
            info = info.border_color(conv::map_border_color(color));
        }
        let s = unsafe { self.ash.create_sampler(&info, None) }.map_err(|e| format!("raw sampler: {e}"))?;
        self.samplers.push(s);
        Ok(s)
    }

}

/// The raw path's descriptor sets, made and freed by the recorder alone (a pool is not shared
/// between threads): pools of `POOL_SETS` sets, a new one when the last is full; destroyed once
/// the GPU is done with them, before the device.
pub(crate) struct Pools {
    device: wgpu::Device,
    pub(crate) ash: ash::Device,
    pools: Vec<vk::DescriptorPool>,
}

impl Pools {
    pub(crate) fn new(device: &wgpu::Device) -> Option<Pools> {
        let ash = unsafe { device.as_hal::<Vulkan>() }?.raw_device().clone();
        Some(Pools { device: device.clone(), ash, pools: Vec::new() })
    }

    /// A descriptor set of `layout` (made from `entries`), with `resources` by entry, in the order
    /// of `entries`; and the pool it is from, to free it with.
    pub(crate) fn set(&mut self, layout: vk::DescriptorSetLayout, entries: &[wgpu::BindGroupLayoutEntry], resources: &[Resource]) -> Result<(vk::DescriptorPool, vk::DescriptorSet), String> {
        let (pool, set) = self.allocate(layout)?;
        // The binding numbers of the layout: the entries' rank by number.
        let mut order: Vec<(u32, usize)> = entries.iter().enumerate().map(|(i, e)| (e.binding, i)).collect();
        order.sort_unstable();
        let (mut buffers, mut images) = (Vec::with_capacity(entries.len()), Vec::with_capacity(entries.len()));
        for &(_, i) in &order {
            match resources[i] {
                Resource::Buffer(buffer, offset, range) => buffers.push(vk::DescriptorBufferInfo { buffer, offset, range }),
                Resource::Texture(image_view, image_layout) => images.push(vk::DescriptorImageInfo { sampler: vk::Sampler::null(), image_view, image_layout }),
                Resource::Sampler(sampler) => images.push(vk::DescriptorImageInfo { sampler, ..Default::default() }),
            }
        }
        let (mut b, mut im) = (0, 0);
        let writes: Vec<vk::WriteDescriptorSet> = (0u32..)
            .zip(&order)
            .map(|(at, &(_, i))| {
                let w = vk::WriteDescriptorSet::default().dst_set(set).dst_binding(at).descriptor_type(conv::map_binding_type(entries[i].ty));
                if let Resource::Buffer(..) = resources[i] {
                    b += 1;
                    w.buffer_info(std::slice::from_ref(&buffers[b - 1]))
                } else {
                    im += 1;
                    w.image_info(std::slice::from_ref(&images[im - 1]))
                }
            })
            .collect();
        unsafe { self.ash.update_descriptor_sets(&writes, &[]) };
        Ok((pool, set))
    }

    /// Frees sets the GPU is done with.
    pub(crate) fn free(&mut self, sets: &[(vk::DescriptorPool, vk::DescriptorSet)]) {
        for &(pool, set) in sets {
            // ponytail: one call per set; by pool if freeing ever shows in --perf.
            let _ = unsafe { self.ash.free_descriptor_sets(pool, &[set]) };
        }
    }

    /// A set from the last pool, or from a new one when it is full.
    fn allocate(&mut self, layout: vk::DescriptorSetLayout) -> Result<(vk::DescriptorPool, vk::DescriptorSet), String> {
        for _ in 0..2 {
            if let Some(&pool) = self.pools.last() {
                match unsafe { self.ash.allocate_descriptor_sets(&vk::DescriptorSetAllocateInfo::default().descriptor_pool(pool).set_layouts(&[layout])) } {
                    Ok(sets) => return Ok((pool, sets[0])),
                    Err(vk::Result::ERROR_OUT_OF_POOL_MEMORY | vk::Result::ERROR_FRAGMENTED_POOL) => {}
                    Err(e) => return Err(format!("raw descriptor set: {e}")),
                }
            }
            // ponytail: room for 16 of each kind per set on average; another pool when it fills.
            let sizes = [vk::DescriptorType::UNIFORM_BUFFER, vk::DescriptorType::UNIFORM_BUFFER_DYNAMIC, vk::DescriptorType::SAMPLED_IMAGE, vk::DescriptorType::SAMPLER, vk::DescriptorType::STORAGE_BUFFER, vk::DescriptorType::STORAGE_IMAGE]
                .map(|ty| vk::DescriptorPoolSize { ty, descriptor_count: 16 * POOL_SETS });
            let info = vk::DescriptorPoolCreateInfo::default().flags(vk::DescriptorPoolCreateFlags::FREE_DESCRIPTOR_SET).max_sets(POOL_SETS).pool_sizes(&sizes);
            self.pools.push(unsafe { self.ash.create_descriptor_pool(&info, None) }.map_err(|e| format!("raw descriptor pool: {e}"))?);
        }
        Err("raw descriptor set: a new pool is full".into())
    }
}

impl Drop for Pools {
    /// The pools go once the GPU is done with their sets, before the device (which `device` holds).
    fn drop(&mut self) {
        let _ = self.device.poll(wgpu::PollType::wait_indefinitely());
        for &p in &self.pools {
            unsafe { self.ash.destroy_descriptor_pool(p, None) };
        }
    }
}

/// Starts rendering into `colors` (a hole: None) and `depth`, kept as they are (LOAD, STORE), as
/// wgpu's passes over targets in use do.
pub(crate) fn begin_rendering(ash: &ash::Device, cb: vk::CommandBuffer, (width, height): (u32, u32), colors: &[Option<vk::ImageView>], depth: Option<(vk::ImageView, wgpu::TextureFormat)>) {
    let kept = |view, layout| vk::RenderingAttachmentInfo::default().image_view(view).image_layout(layout).load_op(vk::AttachmentLoadOp::LOAD).store_op(vk::AttachmentStoreOp::STORE);
    let colors: Vec<_> = colors.iter().map(|v| v.map_or_else(vk::RenderingAttachmentInfo::default, |v| kept(v, vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL))).collect();
    let none = vk::RenderingAttachmentInfo::default();
    let (depth_view, stencil_view) = match depth {
        Some((view, format)) => {
            let a = kept(view, vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL);
            (if format.has_depth_aspect() { a } else { none }, if format.has_stencil_aspect() { a } else { none })
        }
        None => (none, none),
    };
    let area = vk::Rect2D { offset: vk::Offset2D::default(), extent: vk::Extent2D { width, height } };
    let info = vk::RenderingInfo::default().render_area(area).layer_count(1).color_attachments(&colors).depth_attachment(&depth_view).stencil_attachment(&stencil_view);
    unsafe { ash.cmd_begin_rendering(cb, &info) };
}

/// The viewport `[x, y, width, height, min depth, max depth]` as wgpu-hal sets it: upside down,
/// so that clip space has y up as in wgpu.
pub(crate) fn viewport(ash: &ash::Device, cb: vk::CommandBuffer, [x, y, w, h, min_depth, max_depth]: [f32; 6]) {
    unsafe { ash.cmd_set_viewport(cb, 0, &[vk::Viewport { x, y: y + h, width: w, height: -h, min_depth, max_depth }]) };
}

/// The scissor `[x, y, width, height]`.
pub(crate) fn scissor(ash: &ash::Device, cb: vk::CommandBuffer, [x, y, width, height]: [u32; 4]) {
    unsafe { ash.cmd_set_scissor(cb, 0, &[vk::Rect2D { offset: vk::Offset2D { x: x as i32, y: y as i32 }, extent: vk::Extent2D { width, height } }]) };
}

/// A whole texture (`image`, of `format`) going from use `from` to use `to`, with the layouts,
/// stages and accesses wgpu-hal gives these uses: what wgpu would put between them.
pub(crate) fn barrier(ash: &ash::Device, cb: vk::CommandBuffer, image: vk::Image, format: wgpu::TextureFormat, from: wgpu::TextureUses, to: wgpu::TextureUses) {
    let ((src_stage, src_access), (dst_stage, dst_access)) = (conv::map_texture_usage_to_barrier(from), conv::map_texture_usage_to_barrier(to));
    let b = vk::ImageMemoryBarrier::default()
        .src_access_mask(src_access)
        .dst_access_mask(dst_access)
        .old_layout(conv::derive_image_layout(from, format))
        .new_layout(conv::derive_image_layout(to, format))
        .image(image)
        .subresource_range(conv::map_subresource_range(&wgpu::ImageSubresourceRange::default(), format));
    unsafe { ash.cmd_pipeline_barrier(cb, src_stage, dst_stage, vk::DependencyFlags::empty(), &[], &[], &[b]) };
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::surfaces::Surfaces;
    use crate::vk::{binding_map, spirv, spirv_options};

    /// Raw draws paint what wgpu paints: two draws (triangles placed by a bank read at a dynamic
    /// offset, a texture read through a filtering sampler, blending with a constant, a viewport
    /// and a scissor inside the target, bindings listed out of order and numbered as the game's)
    /// made once by wgpu and once by the raw path, read back and compared bit for bit. In the
    /// first, the stencil keeps the second triangle, nearer, off the first; in the second, a
    /// triangle just in front of the first is hidden by the depth bias, and shown without the
    /// depth test or the bias.
    #[test]
    fn a_raw_draw_paints_what_wgpu_paints() {
        let Ok(s) = Surfaces::headless(crate::Backend::Auto) else {
            eprintln!("no GPU adapter: test skipped");
            return;
        };
        let (Some(options), Some(mut raw)) = (spirv_options(&s.adapter, &s.device), Raw::new(&s.adapter, &s.device)) else {
            eprintln!("not Vulkan: test skipped");
            return;
        };
        if !s.vulkan.dynamic_rendering {
            eprintln!("no dynamic rendering: test skipped");
            return;
        }
        let (d, q) = (&s.device, &s.queue);
        let vertex = "struct Bank { v: array<vec4<f32>, 8> }
            @group(0) @binding(3) var<uniform> c3: Bank;
            struct Out { @builtin(position) p: vec4<f32>, @location(0) c: vec4<f32>, @location(1) uv: vec2<f32> }
            @vertex fn main(@builtin(vertex_index) i: u32) -> Out {
                let corners = array<vec2<f32>, 3>(vec2(-1.0, -1.0), vec2(1.0, -1.0), vec2(-1.0, 1.0));
                let t = c3.v[i / 3u];
                let xy = corners[i % 3u] * t.w + t.xy;
                var o: Out;
                o.p = vec4<f32>(xy, t.z, 1.0);
                o.c = c3.v[3u + i / 3u];
                o.uv = xy * 0.5 + 0.5;
                return o;
            }";
        let pixel = "struct Tint { t: vec4<f32> }
            @group(1) @binding(0) var<uniform> c0: Tint;
            @group(1) @binding(32) var t0: texture_2d<f32>;
            @group(1) @binding(33) var s0: sampler;
            struct In { @location(0) c: vec4<f32>, @location(1) uv: vec2<f32> }
            @fragment fn main(i: In) -> @location(0) vec4<f32> {
                return i.c * c0.t + textureSample(t0, s0, i.uv);
            }";
        let ir: Vec<(naga::Module, naga::valid::ModuleInfo)> = [vertex, pixel]
            .iter()
            .map(|wgsl| {
                let module = naga::front::wgsl::parse_str(wgsl).unwrap();
                let info = naga::valid::Validator::new(naga::valid::ValidationFlags::all(), naga::valid::Capabilities::all()).validate(&module).unwrap();
                (module, info)
            })
            .collect();
        let (v, f) = (wgpu::ShaderStages::VERTEX, wgpu::ShaderStages::FRAGMENT);
        let uniform = |binding, size, dynamic, visibility| wgpu::BindGroupLayoutEntry { binding, visibility, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Uniform, has_dynamic_offset: dynamic, min_binding_size: wgpu::BufferSize::new(size) }, count: None };
        let groups = [
            vec![uniform(3, 128, true, v)],
            vec![
                wgpu::BindGroupLayoutEntry { binding: 33, visibility: f, ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering), count: None },
                uniform(0, 16, false, f),
                wgpu::BindGroupLayoutEntry { binding: 32, visibility: f, ty: wgpu::BindingType::Texture { sample_type: wgpu::TextureSampleType::Float { filterable: true }, view_dimension: wgpu::TextureViewDimension::D2, multisampled: false }, count: None },
            ],
        ];

        // The resources both draws read.
        let floats = |v: &[f32]| v.iter().flat_map(|x| x.to_le_bytes()).collect::<Vec<u8>>();
        // Three triangles (offset x, y, depth, size), then their colours: the first two overlap
        // in a band; the third lies inside the first, 0.001 in front of it.
        let mut bank = vec![0u8; 256];
        bank.extend(floats(&[-0.1, -0.1, 0.75, 0.9, -0.4, -0.4, 0.25, -0.9, -0.7, -0.7, 0.749, 0.4]));
        bank.extend(floats(&[1.0, 0.2, 0.1, 1.0, 0.1, 0.9, 0.3, 0.6, 0.2, 0.3, 1.0, 0.8]));
        bank.resize(256 + 128, 0);
        let buffer = |bytes: &[u8]| {
            let b = d.create_buffer(&wgpu::BufferDescriptor { label: None, size: bytes.len() as u64, usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST, mapped_at_creation: false });
            q.write_buffer(&b, 0, bytes);
            b
        };
        let (bank, tint) = (buffer(&bank), buffer(&floats(&[0.5, 0.75, 1.0, 0.5])));
        let texture = |format, usage, size| d.create_texture(&wgpu::TextureDescriptor { label: None, size: wgpu::Extent3d { width: size, height: size, depth_or_array_layers: 1 }, mip_level_count: 1, sample_count: 1, dimension: wgpu::TextureDimension::D2, format, usage, view_formats: &[] });
        let t = texture(wgpu::TextureFormat::Rgba8Unorm, wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST, 8);
        let texels: Vec<u8> = (0..8 * 8 * 4).map(|i| (i * 37 % 251) as u8).collect();
        q.write_texture(t.as_image_copy(), &texels, wgpu::TexelCopyBufferLayout { offset: 0, bytes_per_row: Some(32), rows_per_image: None }, wgpu::Extent3d { width: 8, height: 8, depth_or_array_layers: 1 });
        let sampling = wgpu::SamplerDescriptor { mag_filter: wgpu::FilterMode::Linear, min_filter: wgpu::FilterMode::Linear, ..Default::default() };
        let sampler = d.create_sampler(&sampling);
        let target = || texture(wgpu::TextureFormat::Rgba8Unorm, wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC, 64);
        let depth = || texture(wgpu::TextureFormat::Depth24PlusStencil8, wgpu::TextureUsages::RENDER_ATTACHMENT, 64);
        let (a, da, b, db) = (target(), depth(), target(), depth());
        let view = |t: &wgpu::Texture| t.create_view(&Default::default());
        let (av, dav, bv, dbv, tv) = (view(&a), view(&da), view(&b), view(&db), view(&t));

        // The states both pipelines have.
        let face = wgpu::StencilFaceState { compare: wgpu::CompareFunction::NotEqual, fail_op: wgpu::StencilOperation::Keep, depth_fail_op: wgpu::StencilOperation::Keep, pass_op: wgpu::StencilOperation::Replace };
        // The first draw's stencil, the second draw's bias (0.006 at these depths).
        let stenciled = wgpu::DepthStencilState {
            format: wgpu::TextureFormat::Depth24PlusStencil8,
            depth_write_enabled: Some(true),
            depth_compare: Some(wgpu::CompareFunction::Less),
            stencil: wgpu::StencilState { front: face, back: face, read_mask: 0xff, write_mask: 0xff },
            bias: Default::default(),
        };
        let biased = wgpu::DepthStencilState { stencil: Default::default(), bias: wgpu::DepthBiasState { constant: 100_000, slope_scale: 0.0, clamp: 0.0 }, ..stenciled.clone() };
        let component = |src, dst| wgpu::BlendComponent { src_factor: src, dst_factor: dst, operation: wgpu::BlendOperation::Add };
        let blend = wgpu::BlendState { color: component(wgpu::BlendFactor::Constant, wgpu::BlendFactor::OneMinusConstant), alpha: component(wgpu::BlendFactor::One, wgpu::BlendFactor::Zero) };
        let targets = [Some(wgpu::ColorTargetState { format: wgpu::TextureFormat::Rgba8Unorm, blend: Some(blend), write_mask: wgpu::ColorWrites::ALL })];
        let (vp, sc, constant, reference) = ([4.0, 6.0, 52.0, 50.0, 0.0, 1.0], [2, 3, 58, 55], [0.25, 0.5, 0.75, 1.0], 0x5a);

        // wgpu's draw into `a`.
        let layouts: Vec<wgpu::BindGroupLayout> = groups.iter().map(|entries| d.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor { label: None, entries })).collect();
        let layout = d.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor { label: None, bind_group_layouts: &[Some(&layouts[0]), Some(&layouts[1])], ..Default::default() });
        let modules: Vec<wgpu::ShaderModule> = ir.iter().map(|(m, _)| d.create_shader_module(wgpu::ShaderModuleDescriptor { label: None, source: wgpu::ShaderSource::Naga(std::borrow::Cow::Owned(m.clone())) })).collect();
        let pipeline = |depth_stencil: &wgpu::DepthStencilState| {
            d.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: None,
                layout: Some(&layout),
                vertex: wgpu::VertexState { module: &modules[0], entry_point: Some("main"), compilation_options: Default::default(), buffers: &[] },
                fragment: Some(wgpu::FragmentState { module: &modules[1], entry_point: Some("main"), compilation_options: Default::default(), targets: &targets }),
                primitive: Default::default(),
                depth_stencil: Some(depth_stencil.clone()),
                multisample: Default::default(),
                multiview_mask: None,
                cache: None,
            })
        };
        let pipelines = [pipeline(&stenciled), pipeline(&biased)];
        let group = |l, resources: &[(u32, wgpu::BindingResource)]| d.create_bind_group(&wgpu::BindGroupDescriptor { label: None, layout: l, entries: &resources.iter().map(|(binding, resource)| wgpu::BindGroupEntry { binding: *binding, resource: resource.clone() }).collect::<Vec<_>>() });
        let g0 = group(&layouts[0], &[(3, wgpu::BindingResource::Buffer(wgpu::BufferBinding { buffer: &bank, offset: 0, size: wgpu::BufferSize::new(128) }))]);
        let g1 = group(&layouts[1], &[(0, tint.as_entire_binding()), (32, wgpu::BindingResource::TextureView(&tv)), (33, wgpu::BindingResource::Sampler(&sampler))]);
        /// A pass over `view` and `depth`, both cleared first.
        fn cleared<'e>(e: &'e mut wgpu::CommandEncoder, view: &wgpu::TextureView, depth: &wgpu::TextureView) -> wgpu::RenderPass<'e> {
            e.begin_render_pass(&wgpu::RenderPassDescriptor {
                color_attachments: &[Some(wgpu::RenderPassColorAttachment { view, depth_slice: None, resolve_target: None, ops: wgpu::Operations { load: wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT), store: wgpu::StoreOp::Store } })],
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment { view: depth, depth_ops: Some(wgpu::Operations { load: wgpu::LoadOp::Clear(1.0), store: wgpu::StoreOp::Store }), stencil_ops: Some(wgpu::Operations { load: wgpu::LoadOp::Clear(0), store: wgpu::StoreOp::Store }) }),
                ..Default::default()
            })
        }
        let mut e1 = d.create_command_encoder(&Default::default());
        {
            let mut pass = cleared(&mut e1, &av, &dav);
            pass.set_pipeline(&pipelines[0]);
            pass.set_bind_group(0, &g0, &[256]);
            pass.set_bind_group(1, &g1, &[]);
            pass.set_viewport(vp[0], vp[1], vp[2], vp[3], vp[4], vp[5]);
            pass.set_scissor_rect(sc[0], sc[1], sc[2], sc[3]);
            pass.set_blend_constant(wgpu::Color { r: constant[0], g: constant[1], b: constant[2], a: constant[3] });
            pass.set_stencil_reference(reference);
            pass.draw(0..6, 0..1);
            pass.set_pipeline(&pipelines[1]);
            pass.draw(6..9, 0..1);
        }
        // `b` cleared as `a` was (by wgpu: the raw draw keeps what it finds), then set in the
        // states the raw draw uses: the boundary before a raw segment.
        drop(cleared(&mut e1, &bv, &dbv));
        let rt = |texture, state| wgpu::TextureTransition { texture, selector: None, state };
        e1.transition_resources(
            [&bank, &tint].into_iter().map(|buffer| wgpu::BufferTransition { buffer, state: wgpu::BufferUses::UNIFORM }),
            [rt(&b, wgpu::TextureUses::COLOR_TARGET), rt(&db, wgpu::TextureUses::DEPTH_STENCIL_WRITE), rt(&t, wgpu::TextureUses::RESOURCE)].into_iter(),
        );

        // The raw draw into `b`, from the same modules, states and resources.
        let map = binding_map(&groups.iter().map(Vec::as_slice).collect::<Vec<_>>());
        let mut module = |(m, i): &(naga::Module, naga::valid::ModuleInfo)| raw.module(&spirv(m, i, m.entry_points[0].stage, "main", &options, &map).unwrap()).unwrap();
        let (vm, pm) = (module(&ir[0]), module(&ir[1]));
        let sets = [raw.set_layout(&groups[0]).unwrap(), raw.set_layout(&groups[1]).unwrap()];
        let raw_layout = raw.pipeline_layout(&sets).unwrap();
        let mut raw_pipeline = |depth_stencil| raw.pipeline(&PipelineDesc { vertex: vm, pixel: Some(pm), entries: [c"main", c"main"], buffers: &[], primitive: Default::default(), depth_stencil: Some(depth_stencil), targets: &targets, layout: raw_layout }).unwrap();
        let raw_pipelines = [raw_pipeline(&stenciled), raw_pipeline(&biased)];
        let image = |v: &wgpu::TextureView| unsafe { v.as_hal::<Vulkan>().map(|h| h.raw_handle()) }.unwrap();
        let handle = |b: &wgpu::Buffer| unsafe { b.as_hal::<Vulkan>().map(|h| h.raw_handle()) }.unwrap();
        let raw_sampler = raw.sampler(&sampling).unwrap();
        let read = conv::derive_image_layout(wgpu::TextureUses::RESOURCE, wgpu::TextureFormat::Rgba8Unorm);
        let mut pools = Pools::new(d).unwrap();
        let (_, s0) = pools.set(sets[0], &groups[0], &[Resource::Buffer(handle(&bank), 0, 128)]).unwrap();
        let (_, s1) = pools.set(sets[1], &groups[1], &[Resource::Sampler(raw_sampler), Resource::Buffer(handle(&tint), 0, 16), Resource::Texture(image(&tv), read)]).unwrap();
        let (bh, dbh) = (image(&bv), image(&dbv));
        let mut e2 = d.create_command_encoder(&Default::default());
        unsafe {
            e2.as_hal_mut::<Vulkan, _, _>(|e| {
                let cb = e.expect("a raw encoder").raw_handle();
                begin_rendering(&raw.ash, cb, (64, 64), &[Some(bh)], Some((dbh, wgpu::TextureFormat::Depth24PlusStencil8)));
                viewport(&raw.ash, cb, vp.map(|x| x as f32));
                scissor(&raw.ash, cb, sc);
                raw.ash.cmd_set_blend_constants(cb, &constant.map(|x| x as f32));
                raw.ash.cmd_set_stencil_reference(cb, vk::StencilFaceFlags::FRONT_AND_BACK, reference);
                raw.ash.cmd_bind_pipeline(cb, vk::PipelineBindPoint::GRAPHICS, raw_pipelines[0]);
                raw.ash.cmd_bind_descriptor_sets(cb, vk::PipelineBindPoint::GRAPHICS, raw_layout, 0, &[s0, s1], &[256]);
                raw.ash.cmd_draw(cb, 6, 1, 0, 0);
                raw.ash.cmd_bind_pipeline(cb, vk::PipelineBindPoint::GRAPHICS, raw_pipelines[1]);
                raw.ash.cmd_draw(cb, 3, 1, 6, 0);
                raw.ash.cmd_end_rendering(cb);
            })
        };

        // Both targets read back.
        let mut e3 = d.create_command_encoder(&Default::default());
        let readbacks: Vec<wgpu::Buffer> = [&a, &b]
            .into_iter()
            .map(|texture| {
                let out = d.create_buffer(&wgpu::BufferDescriptor { label: None, size: 64 * 64 * 4, usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ, mapped_at_creation: false });
                e3.copy_texture_to_buffer(texture.as_image_copy(), wgpu::TexelCopyBufferInfo { buffer: &out, layout: wgpu::TexelCopyBufferLayout { offset: 0, bytes_per_row: Some(256), rows_per_image: None } }, wgpu::Extent3d { width: 64, height: 64, depth_or_array_layers: 1 });
                out
            })
            .collect();
        q.submit([e1.finish(), e2.finish(), e3.finish()]);
        let pixels: Vec<Vec<u8>> = readbacks
            .iter()
            .map(|out| {
                out.slice(..).map_async(wgpu::MapMode::Read, |_| ());
                d.poll(wgpu::PollType::wait_indefinitely()).unwrap();
                out.slice(..).get_mapped_range().unwrap().to_vec()
            })
            .collect();
        let shades = pixels[0].chunks(4).collect::<std::collections::BTreeSet<_>>().len();
        assert!(shades > 20, "wgpu's draw shows {shades} shades only");
        let differ = pixels[0].chunks(4).zip(pixels[1].chunks(4)).filter(|(x, y)| x != y).count();
        assert_eq!(differ, 0, "{differ} of 4096 pixels differ");
    }
}

impl Drop for Raw {
    /// The objects go once the GPU is done with them, before the device (which `device` holds).
    fn drop(&mut self) {
        let _ = self.device.poll(wgpu::PollType::wait_indefinitely());
        unsafe {
            for &p in &self.pipelines {
                self.ash.destroy_pipeline(p, None);
            }
            for &l in self.pipeline_layouts.values() {
                self.ash.destroy_pipeline_layout(l, None);
            }
            for &l in self.set_layouts.values() {
                self.ash.destroy_descriptor_set_layout(l, None);
            }
            for &m in &self.modules {
                self.ash.destroy_shader_module(m, None);
            }
            for &s in &self.samplers {
                self.ash.destroy_sampler(s, None);
            }
        }
    }
}
