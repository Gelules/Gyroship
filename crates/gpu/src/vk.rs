//! The GPU device on Vulkan, opened through wgpu-hal (D110): the device `request_device`
//! would open, plus what the raw Vulkan path needs, when the GPU has it.

use ash::vk;
use naga::back::spv;
use wgpu::hal::{api::Vulkan, vulkan};

/// What the device has beyond wgpu's own extensions and features.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Extras {
    /// VK_EXT_graphics_pipeline_library, with VK_KHR_pipeline_library.
    pub pipeline_library: bool,
    /// Whether the GPU links pipeline libraries fast when not asked to optimize.
    pub fast_linking: bool,
    /// Vulkan 1.3's dynamic rendering.
    pub dynamic_rendering: bool,
    /// Vulkan 1.3's pipeline creation cache control: a creation may fail rather than compile
    /// (FAIL_ON_PIPELINE_COMPILE_REQUIRED), which counts the compiles a run makes (D110 step 7b).
    pub cache_control: bool,
}

/// Opens `adapter`'s device with `descriptor` as `request_device` does, after its checks
/// (`create_device_from_hal` skips them); on Vulkan, with the extras the GPU has.
pub(crate) fn open(adapter: &wgpu::Adapter, descriptor: &wgpu::DeviceDescriptor<'_>) -> Result<(wgpu::Device, wgpu::Queue, Extras), String> {
    let missing = descriptor.required_features - adapter.features();
    if !missing.is_empty() {
        return Err(format!("the adapter lacks {missing:?}"));
    }
    let experimental = descriptor.required_features & wgpu::Features::all_experimental_mask();
    if !experimental.is_empty() && !descriptor.experimental_features.is_enabled() {
        return Err(format!("{experimental:?} are experimental and not enabled"));
    }
    let mut over = None;
    descriptor.required_limits.check_limits_with_fail_fn(&adapter.limits(), true, |name, asked, allowed| over = Some(format!("{name} {asked} is past the adapter's {allowed}")));
    if let Some(e) = over {
        return Err(e);
    }
    let (open, extras) = {
        let Some(hal) = (unsafe { adapter.as_hal::<Vulkan>() }) else {
            let (device, queue) = pollster::block_on(adapter.request_device(descriptor)).map_err(|e| e.to_string())?;
            return Ok((device, queue, Extras::default()));
        };
        let extras = supported(&hal);
        let mut library = vk::PhysicalDeviceGraphicsPipelineLibraryFeaturesEXT::default().graphics_pipeline_library(true);
        let mut rendering = vk::PhysicalDeviceDynamicRenderingFeatures::default().dynamic_rendering(true);
        // Each feature in a structure of its own: wgpu-hal chains other promoted ones, which
        // forbids VkPhysicalDeviceVulkan13Features.
        let mut control = vk::PhysicalDevicePipelineCreationCacheControlFeatures::default().pipeline_creation_cache_control(true);
        let add = callback(|args| {
            if extras.pipeline_library {
                args.extensions.extend([ash::khr::pipeline_library::NAME, ash::ext::graphics_pipeline_library::NAME]);
                *args.create_info = args.create_info.push_next(&mut library);
            }
            if extras.dynamic_rendering {
                *args.create_info = args.create_info.push_next(&mut rendering);
            }
            if extras.cache_control {
                *args.create_info = args.create_info.push_next(&mut control);
            }
        });
        let open = unsafe { hal.open_with_callback(descriptor.required_features, &descriptor.required_limits, &descriptor.memory_hints, Some(add)) }.map_err(|e| e.to_string())?;
        (open, extras)
    };
    let (device, queue) = unsafe { adapter.create_device_from_hal(open, descriptor) }.map_err(|e| e.to_string())?;
    eprintln!("gyroship: vulkan {extras:?}");
    Ok((device, queue, extras))
}

/// The callback as wgpu-hal takes it: a closure handed straight to `Box::new` is not
/// inferred general enough over its argument's lifetimes.
fn callback<'a>(f: impl for<'arg, 'pnext> FnOnce(vulkan::CreateDeviceCallbackArgs<'arg, 'pnext, 'a>) + 'a) -> Box<vulkan::CreateDeviceCallback<'a>> {
    Box::new(f)
}

/// The extras `hal` can have: the extensions it lists, the features it reports.
fn supported(hal: &vulkan::Adapter) -> Extras {
    let caps = hal.physical_device_capabilities();
    let shared = hal.shared_instance();
    let library = caps.supports_extension(ash::khr::pipeline_library::NAME) && caps.supports_extension(ash::ext::graphics_pipeline_library::NAME);
    // ponytail: dynamic rendering on Vulkan 1.3 devices only (wgpu-hal asks for 1.3 from a 1.1
    // loader on); VK_KHR_dynamic_rendering on older ones if a GPU ever needs it.
    let core13 = caps.properties().api_version >= vk::API_VERSION_1_3 && shared.instance_api_version() >= vk::API_VERSION_1_1;
    let (instance, device) = (shared.raw_instance(), hal.raw_physical_device());
    let mut gpl = vk::PhysicalDeviceGraphicsPipelineLibraryFeaturesEXT::default();
    let mut rendering = vk::PhysicalDeviceDynamicRenderingFeatures::default();
    let mut control = vk::PhysicalDevicePipelineCreationCacheControlFeatures::default();
    let mut fast = vk::PhysicalDeviceGraphicsPipelineLibraryPropertiesEXT::default();
    let mut features = vk::PhysicalDeviceFeatures2::default();
    if library {
        features = features.push_next(&mut gpl);
    }
    if core13 {
        features = features.push_next(&mut rendering).push_next(&mut control);
    }
    unsafe { instance.get_physical_device_features2(device, &mut features) };
    if library {
        unsafe { instance.get_physical_device_properties2(device, &mut vk::PhysicalDeviceProperties2::default().push_next(&mut fast)) };
    }
    Extras {
        pipeline_library: gpl.graphics_pipeline_library == vk::TRUE,
        fast_linking: fast.graphics_pipeline_library_fast_linking == vk::TRUE,
        dynamic_rendering: rendering.dynamic_rendering == vk::TRUE,
        cache_control: control.pipeline_creation_cache_control == vk::TRUE,
    }
}

/// The SPIR-V options wgpu-hal writes `device`'s modules with (wgpu-hal 30.0.1,
/// `Adapter::device_from_raw`), worked out from the same queries; None off Vulkan. With them,
/// [`spirv`] writes the bytes wgpu-hal writes (`spirv_is_the_one_wgpu_writes`).
pub(crate) fn spirv_options(adapter: &wgpu::Adapter, device: &wgpu::Device) -> Option<spv::Options<'static>> {
    use naga::proc::BoundsCheckPolicy::{Restrict, Unchecked};
    use spv::Capability as C;
    use wgpu::Features as F;
    let hal = unsafe { adapter.as_hal::<Vulkan>() }?;
    let non_semantic = unsafe { device.as_hal::<Vulkan>() }?.enabled_device_extensions().contains(&ash::khr::shader_non_semantic_info::NAME);
    let (caps, shared) = (hal.physical_device_capabilities(), hal.shared_instance());
    let version = caps.properties().api_version;
    let has = |e: &std::ffi::CStr| caps.supports_extension(e);
    // wgpu-hal's private capabilities, from the features it queries, when it queries them.
    let queried = [
        has(ash::ext::robustness2::NAME),
        has(ash::ext::image_robustness::NAME),
        version >= vk::API_VERSION_1_3 || has(ash::khr::zero_initialize_workgroup_memory::NAME),
        version >= vk::API_VERSION_1_3 || has(ash::khr::shader_integer_dot_product::NAME),
        version >= vk::API_VERSION_1_2 || has(ash::khr::shader_float16_int8::NAME),
    ];
    let mut robustness2 = vk::PhysicalDeviceRobustness2FeaturesEXT::default();
    let mut image = vk::PhysicalDeviceImageRobustnessFeatures::default();
    let mut zero = vk::PhysicalDeviceZeroInitializeWorkgroupMemoryFeatures::default();
    let mut dot = vk::PhysicalDeviceShaderIntegerDotProductFeatures::default();
    let mut int8 = vk::PhysicalDeviceShaderFloat16Int8Features::default();
    let mut features = vk::PhysicalDeviceFeatures2::default();
    if queried[0] {
        features = features.push_next(&mut robustness2);
    }
    if queried[1] {
        features = features.push_next(&mut image);
    }
    if queried[2] {
        features = features.push_next(&mut zero);
    }
    if queried[3] {
        features = features.push_next(&mut dot);
    }
    if queried[4] {
        features = features.push_next(&mut int8);
    }
    unsafe { shared.raw_instance().get_physical_device_features2(hal.raw_physical_device(), &mut features) };
    let robust_image = if queried[0] { robustness2.robust_image_access2 != 0 } else { image.robust_image_access != 0 };

    let (features, downlevel) = (device.features(), adapter.get_downlevel_capabilities().flags);
    let mut capabilities = vec![C::Shader, C::Matrix, C::Sampled1D, C::Image1D, C::ImageQuery, C::DerivativeControl, C::StorageImageExtendedFormats];
    let mut add = |on: bool, more: &[C]| {
        if on {
            capabilities.extend_from_slice(more);
        }
    };
    add(downlevel.contains(wgpu::DownlevelFlags::CUBE_ARRAY_TEXTURES), &[C::SampledCubeArray]);
    add(downlevel.contains(wgpu::DownlevelFlags::MULTISAMPLED_SHADING), &[C::SampleRateShading]);
    add(features.contains(F::MULTIVIEW), &[C::MultiView]);
    add(features.contains(F::PRIMITIVE_INDEX), &[C::Geometry]);
    add(
        features.intersects(F::SUBGROUP | F::SUBGROUP_VERTEX),
        &[C::GroupNonUniform, C::GroupNonUniformVote, C::GroupNonUniformArithmetic, C::GroupNonUniformBallot, C::GroupNonUniformShuffle, C::GroupNonUniformShuffleRelative, C::GroupNonUniformQuad],
    );
    add(features.intersects(F::SAMPLED_TEXTURE_AND_STORAGE_BUFFER_ARRAY_NON_UNIFORM_INDEXING | F::STORAGE_TEXTURE_ARRAY_NON_UNIFORM_INDEXING | F::UNIFORM_BUFFER_BINDING_ARRAYS), &[C::ShaderNonUniform]);
    add(features.contains(F::BGRA8UNORM_STORAGE), &[C::StorageImageWriteWithoutFormat]);
    add(features.contains(F::SHADER_INT64), &[C::Int64]);
    add(features.contains(F::SHADER_F16), &[C::Float16]);
    add(features.contains(F::SHADER_I16), &[C::Int16]);
    add(features.intersects(F::SHADER_INT64_ATOMIC_ALL_OPS | F::SHADER_INT64_ATOMIC_MIN_MAX | F::TEXTURE_INT64_ATOMIC), &[C::Int64Atomics]);
    add(features.contains(F::TEXTURE_INT64_ATOMIC), &[C::Int64ImageEXT]);
    add(features.contains(F::SHADER_FLOAT32_ATOMIC), &[C::AtomicFloat32AddEXT]);
    add(features.contains(F::CLIP_DISTANCES), &[C::ClipDistance]);
    add(features.intersects(F::SHADER_BARYCENTRICS | F::SHADER_PER_VERTEX), &[C::FragmentBarycentricKHR]);
    add(features.contains(F::SHADER_DRAW_INDEX), &[C::DrawParameters]);
    // ponytail: ray tracing, mesh shading and cooperative matrices are left out: they are
    // experimental features, which `open` never enables.
    add(dot.shader_integer_dot_product != 0, &[C::DotProductInputAllKHR, C::DotProductInput4x8BitKHR, C::DotProductInput4x8BitPackedKHR, C::DotProductKHR]);
    add(int8.shader_int8 != 0, &[C::Int8]);

    // The instance's flags are its defaults (`Renderer::instance_for`).
    let debug = wgpu::InstanceFlags::default().contains(wgpu::InstanceFlags::DEBUG);
    let mut flags = spv::WriterFlags::FORCE_POINT_SIZE;
    flags.set(spv::WriterFlags::DEBUG, debug);
    // Qualcomm's vendor id.
    flags.set(spv::WriterFlags::LABEL_VARYINGS, caps.properties().vendor_id != 0x5143);
    flags.set(spv::WriterFlags::PRINT_ON_RAY_QUERY_INITIALIZATION_FAIL | spv::WriterFlags::PRINT_ON_TRACE_RAYS_FAIL, debug && (shared.instance_api_version() >= vk::API_VERSION_1_3 || non_semantic));
    let unless = |robust: bool| if robust { Unchecked } else { Restrict };
    let limits = device.limits();
    Some(spv::Options {
        lang_version: match version {
            v if v >= vk::API_VERSION_1_3 => (1, 6),
            v if v >= vk::API_VERSION_1_2 => (1, 5),
            v if v >= vk::API_VERSION_1_1 => (1, 3),
            _ => (1, 0),
        },
        flags,
        capabilities: Some(capabilities.into_iter().collect()),
        bounds_check_policies: naga::proc::BoundsCheckPolicies { index: Restrict, buffer: unless(robustness2.robust_buffer_access2 == vk::TRUE), image_load: unless(robust_image), binding_array: Unchecked },
        zero_initialize_workgroup_memory: if zero.shader_zero_initialize_workgroup_memory == vk::TRUE { spv::ZeroInitializeWorkgroupMemoryMode::Native } else { spv::ZeroInitializeWorkgroupMemoryMode::Polyfill },
        force_loop_bounding: true,
        ray_query_initialization_tracking: true,
        // ponytail: wgpu-hal sets it only with SHADER_F16, which `surfaces::descriptor` never asks for.
        use_storage_input_output_16: false,
        fake_missing_bindings: false,
        binding_map: Default::default(),
        debug_info: None,
        task_dispatch_limits: Some(naga::back::TaskDispatchLimits { max_mesh_workgroups_per_dim: limits.max_mesh_workgroups_per_dimension, max_mesh_workgroups_total: limits.max_mesh_workgroup_total_count }),
        mesh_shader_primitive_indices_clamp: true,
        trace_ray_argument_validation: true,
        emit_int_div_checks: true,
    })
}

/// Entry point `entry` (of `stage`) of `module` as SPIR-V, written as wgpu-hal writes it
/// for a pipeline (`Device::compile_stage`): overrides processed for that entry point, the
/// bindings renumbered by `map` ([`binding_map`]).
pub(crate) fn spirv(module: &naga::Module, info: &naga::valid::ModuleInfo, stage: naga::ShaderStage, entry: &str, options: &spv::Options, map: &spv::BindingMap) -> Result<Vec<u32>, String> {
    let (module, info) = naga::back::pipeline_constants::process_overrides(module, info, Some((stage, entry)), &Default::default()).map_err(|e| e.to_string())?;
    let mut options = options.clone();
    options.binding_map = map.clone();
    spv::write_vec(&module, &info, &options, Some(&spv::PipelineOptions { shader_stage: stage, entry_point: entry.to_string() })).map_err(|e| e.to_string())
}

/// The bindings of a pipeline whose groups have `groups` as layout entries, renumbered as
/// wgpu-hal numbers them (`create_bind_group_layout`, `create_pipeline_layout`): in each
/// group, from 0 in the order of their numbers (wgpu-core sorts the entries).
pub(crate) fn binding_map(groups: &[&[wgpu::BindGroupLayoutEntry]]) -> spv::BindingMap {
    let mut map = spv::BindingMap::default();
    for (group, entries) in (0u32..).zip(groups) {
        let mut entries = entries.to_vec();
        entries.sort_unstable_by_key(|e| e.binding);
        for (at, e) in (0u32..).zip(&entries) {
            map.insert(naga::ResourceBinding { group, binding: e.binding }, spv::BindingInfo { descriptor_set: group, binding: at, binding_array_size: e.count.map(std::num::NonZeroU32::get) });
        }
    }
    map
}

/// `GYROSHIP_PASSTHROUGH` (the proof of D110 step 5): pipelines take modules made from our
/// SPIR-V (`create_shader_module_passthrough`) instead of wgpu's. Read at the device's opening.
pub(crate) const PASSTHROUGH: &str = "GYROSHIP_PASSTHROUGH";

impl crate::surfaces::Surfaces {
    /// Under [`PASSTHROUGH`]: `made`'s naga module and its validation, kept for its pipelines'
    /// SPIR-V, parsed from its WGSL (as wgpu and the disk cache parse it).
    pub(crate) fn keep_ir(&mut self, made: &crate::draw::Built) {
        // The raw path writes its modules from it too.
        if self.passthrough.is_none() && self.raw.is_none() {
            return;
        }
        let parsed = naga::front::wgsl::parse_str(&made.0.wgsl).map_err(|e| e.message().to_string()).and_then(|module| {
            let info = naga::valid::Validator::new(naga::valid::ValidationFlags::all(), naga::valid::Capabilities::all()).validate(&module).map_err(|e| e.to_string())?;
            Ok((module, info))
        });
        match parsed {
            Ok(ir) => {
                self.cache.ir.insert(made.2, ir);
            }
            Err(e) => eprintln!("gyroship: {PASSTHROUGH}: program {:016x}: {e}", made.0.key),
        }
    }

    /// Under [`PASSTHROUGH`]: `made`'s module from our SPIR-V, for a pipeline whose groups have
    /// `groups` as layout entries. None without it, or when the SPIR-V cannot be written (said
    /// on stderr): the pipeline then takes wgpu's module.
    pub(crate) fn passthrough(&self, made: &crate::draw::Built, groups: &[&[wgpu::BindGroupLayoutEntry]]) -> Option<wgpu::ShaderModule> {
        let options = self.passthrough.as_ref()?;
        let written = self.cache.ir.get(&made.2).ok_or_else(|| "its naga module was not kept".to_string()).and_then(|(module, info)| {
            let entry = module.entry_points.first().ok_or("no entry point")?;
            Ok((spirv(module, info, entry.stage, &entry.name, options, &binding_map(groups))?, entry.name.clone()))
        });
        match written {
            Ok((words, name)) => Some(unsafe {
                self.device.create_shader_module_passthrough(wgpu::ShaderModuleDescriptorPassthrough {
                    label: Some(&format!("{:016x}", made.0.key)),
                    // The workgroup size is not used with SPIR-V.
                    entry_points: std::borrow::Cow::Owned(vec![wgpu::PassthroughShaderEntryPoint { name: name.into(), workgroup_size: (0, 0, 0) }]),
                    spirv: Some(std::borrow::Cow::Owned(words)),
                    ..Default::default()
                })
            }),
            Err(e) => {
                eprintln!("gyroship: {PASSTHROUGH}: program {:016x}: {e}", made.0.key);
                None
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::surfaces::{Surfaces, descriptor};

    /// The game's device has the features and limits `request_device` gives with the same
    /// descriptor, wgpu's extensions plus the extras only, and limits past the adapter's are
    /// refused, as `request_device` refuses them.
    #[test]
    fn the_device_is_the_one_request_device_opens() {
        let Ok(s) = Surfaces::headless(crate::Backend::Auto) else {
            eprintln!("no GPU adapter: test skipped");
            return;
        };
        let asked = descriptor(&s.adapter);
        let (theirs, _) = pollster::block_on(s.adapter.request_device(&asked)).unwrap();
        assert_eq!(s.device.features(), theirs.features());
        assert_eq!(s.device.limits(), theirs.limits());
        if let (Some(ours), Some(theirs)) = unsafe { (s.device.as_hal::<Vulkan>(), theirs.as_hal::<Vulkan>()) } {
            let (ours, theirs) = (ours.enabled_device_extensions(), theirs.enabled_device_extensions());
            assert!(theirs.iter().all(|e| ours.contains(e)));
            let added: Vec<_> = ours.iter().filter(|e| !theirs.contains(e)).copied().collect();
            let extras: &[&std::ffi::CStr] = if s.vulkan.pipeline_library { &[ash::khr::pipeline_library::NAME, ash::ext::graphics_pipeline_library::NAME] } else { &[] };
            assert_eq!(added, extras);
        }
        let mut past = asked.clone();
        past.required_limits.max_bind_groups += 1;
        assert!(open(&s.adapter, &past).is_err());
    }

    /// Our SPIR-V is the bytes wgpu-hal writes: a pipeline made from our modules, with
    /// bindings numbered as ours are (banks at their number, swizzle codes at 999, a texture
    /// and its sampler at 32 and 33, listed out of order), finds in the pipeline cache the
    /// one wgpu made from the same naga modules, and the cache does not grow. Written with
    /// another bounds policy, the same modules miss it.
    #[test]
    fn spirv_is_the_one_wgpu_writes() {
        let Ok(s) = Surfaces::headless(crate::Backend::Auto) else {
            eprintln!("no GPU adapter: test skipped");
            return;
        };
        let wanted = wgpu::Features::PIPELINE_CACHE | wgpu::Features::PASSTHROUGH_SHADERS;
        let Some(options) = spirv_options(&s.adapter, &s.device).filter(|_| s.device.features().contains(wanted)) else {
            eprintln!("no Vulkan pipeline cache or passthrough shaders: test skipped");
            return;
        };
        let vertex = "struct Bank { v: array<vec4<u32>, 4096> }
            @group(0) @binding(3) var<uniform> c3: Bank;
            @group(0) @binding(999) var<uniform> codes: array<vec4<u32>, 4>;
            @vertex fn main(@builtin(vertex_index) i: u32) -> @builtin(position) vec4<f32> {
                return vec4<f32>(bitcast<f32>(c3.v[i].x), f32(codes[i & 3u].y), 0.0, 1.0);
            }";
        let pixel = "struct Bank { v: array<vec4<u32>, 4096> }
            @group(1) @binding(0) var<uniform> c0: Bank;
            @group(1) @binding(32) var t0: texture_2d<f32>;
            @group(1) @binding(33) var s0: sampler;
            @fragment fn main(@builtin(position) p: vec4<f32>) -> @location(0) vec4<f32> {
                let i = vec2<i32>(p.xy);
                return textureLoad(t0, i, i32(c0.v[u32(i.x)].x)) + textureSample(t0, s0, p.xy);
            }";
        let ir: Vec<(naga::Module, naga::valid::ModuleInfo)> = [vertex, pixel]
            .iter()
            .map(|wgsl| {
                let module = naga::front::wgsl::parse_str(wgsl).unwrap();
                let info = naga::valid::Validator::new(naga::valid::ValidationFlags::all(), naga::valid::Capabilities::all()).validate(&module).unwrap();
                (module, info)
            })
            .collect();
        let uniform = |binding, size, visibility| wgpu::BindGroupLayoutEntry { binding, visibility, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Uniform, has_dynamic_offset: false, min_binding_size: wgpu::BufferSize::new(size) }, count: None };
        let (v, f) = (wgpu::ShaderStages::VERTEX, wgpu::ShaderStages::FRAGMENT);
        let groups = [
            vec![uniform(999, 64, v), uniform(3, 65536, v)],
            vec![
                wgpu::BindGroupLayoutEntry { binding: 33, visibility: f, ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering), count: None },
                uniform(0, 65536, f),
                wgpu::BindGroupLayoutEntry { binding: 32, visibility: f, ty: wgpu::BindingType::Texture { sample_type: wgpu::TextureSampleType::Float { filterable: true }, view_dimension: wgpu::TextureViewDimension::D2, multisampled: false }, count: None },
            ],
        ];
        let layouts: Vec<wgpu::BindGroupLayout> = groups.iter().map(|entries| s.device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor { label: None, entries })).collect();
        let layout = s.device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor { label: None, bind_group_layouts: &[Some(&layouts[0]), Some(&layouts[1])], ..Default::default() });
        let cache = unsafe { s.device.create_pipeline_cache(&wgpu::PipelineCacheDescriptor { label: None, data: None, fallback: true }) };
        // The cache's size once a pipeline of `modules` is made with it.
        let cached = |modules: &[wgpu::ShaderModule]| {
            let _ = s.device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: None,
                layout: Some(&layout),
                vertex: wgpu::VertexState { module: &modules[0], entry_point: Some("main"), compilation_options: Default::default(), buffers: &[] },
                fragment: Some(wgpu::FragmentState { module: &modules[1], entry_point: Some("main"), compilation_options: Default::default(), targets: &[Some(wgpu::TextureFormat::Rgba8Unorm.into())] }),
                primitive: Default::default(),
                depth_stencil: None,
                multisample: Default::default(),
                multiview_mask: None,
                cache: Some(&cache),
            });
            cache.get_data().map_or(0, |d| d.len())
        };
        // wgpu's: from the naga modules (as the disk cache gives them: no WGSL to debug with).
        let theirs: Vec<wgpu::ShaderModule> = ir.iter().map(|(m, _)| s.device.create_shader_module(wgpu::ShaderModuleDescriptor { label: None, source: wgpu::ShaderSource::Naga(std::borrow::Cow::Owned(m.clone())) })).collect();
        let first = cached(&theirs);
        assert!(first > 0, "the pipeline cache kept nothing");
        let entries: Vec<&[wgpu::BindGroupLayoutEntry]> = groups.iter().map(Vec::as_slice).collect();
        let ours = |options: &spv::Options| -> Vec<wgpu::ShaderModule> {
            ir.iter()
                .map(|(module, info)| {
                    let words = spirv(module, info, module.entry_points[0].stage, "main", options, &binding_map(&entries)).unwrap();
                    let entry_points = std::borrow::Cow::Owned(vec![wgpu::PassthroughShaderEntryPoint { name: "main".into(), workgroup_size: (0, 0, 0) }]);
                    unsafe { s.device.create_shader_module_passthrough(wgpu::ShaderModuleDescriptorPassthrough { label: None, entry_points, spirv: Some(std::borrow::Cow::Owned(words)), ..Default::default() }) }
                })
                .collect()
        };
        assert_eq!(cached(&ours(&options)), first, "our SPIR-V missed the pipeline wgpu made");
        let mut other = options.clone();
        other.bounds_check_policies.buffer = match other.bounds_check_policies.buffer {
            naga::proc::BoundsCheckPolicy::Unchecked => naga::proc::BoundsCheckPolicy::Restrict,
            _ => naga::proc::BoundsCheckPolicy::Unchecked,
        };
        assert!(cached(&ours(&other)) > first, "another bounds policy found the pipeline: the cache does not tell SPIR-V apart");
    }
}
