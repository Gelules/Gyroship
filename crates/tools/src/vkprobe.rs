//! `vk-probe` (D110, step 2): what Vulkan pipeline libraries (VK_EXT_graphics_pipeline_library)
//! and shader objects (VK_EXT_shader_object) cost and give on this GPU, for the game's own
//! programs, before any of them enters the renderer. A harness of its own on ash, to be thrown
//! away once the measures are in.
//! - `compile`: every translatable vertex and pixel program of data.zip as a library (pixel
//!   programs with the jump targets their blob holds, vertex programs unpaired), half on one
//!   thread and half on several: the time per library and the process's committed memory; then a
//!   sample again from the pipeline cache, which must compile nothing
//!   (FAIL_ON_PIPELINE_COMPILE_REQUIRED), and with an empty cache (the driver's own disk cache).
//! - `link`: pairs play made (the shader cache's vertex variants, each with a pixel program of
//!   the same varyings): their libraries, then linked fast, linked with link-time optimisation,
//!   and made whole.
//! - `pixels`: loop-free pixel programs with plain bindings, over a full-screen triangle with
//!   pseudo-random inputs of a fixed seed: whole, fast-linked, optimised and as shader objects,
//!   their pixels compared bit for bit; then the first fast-linked ones again after the driver had
//!   30 s to itself, while the process's CPU time, threads and committed memory are watched.
//! - `vertex-input`: an integer vertex attribute read by a float input, through dynamic vertex
//!   input with VK_EXT_legacy_vertex_attributes: what the shader gets.
//!
//! Each run renames the entry points (unless `--warm`), so that the driver's disk cache holds none
//! of them: the times are a first run's. The SPIR-V comes from naga's writer. The output names
//! programs by their keys only; keep it under Data all the same.

use ash::vk;
use gyroship_gpu::sass::Stage;
use gyroship_gpu::shader::sph::Interp;
use gyroship_gpu::shader::{self, Options, Translation, cache};
use std::collections::{BTreeMap, HashMap, HashSet};
use std::ffi::{CStr, CString, c_void};
use std::path::Path;
use std::sync::atomic::{AtomicBool, AtomicU32, AtomicU64, AtomicUsize, Ordering::Relaxed};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

type R<T> = Result<T, String>;

/// The probe's parts and settings, from the command line.
pub struct Probe<'a> {
    pub zip: &'a Path,
    pub cache: Option<&'a Path>,
    pub parts: &'a [String],
    pub threads: usize,
    pub limit: usize,
    pub validate: bool,
    pub warm: bool,
}

/// The side of the targets and textures the pixel part draws, and a target's bytes (RGBA32F).
const SIDE: u32 = 64;
const TARGET_BYTES: u64 = (SIDE * SIDE * 16) as u64;
/// The pixel part's programs at most, and those it draws again after the driver's 30 s.
const DRAWN: usize = 200;
const KEPT: usize = 20;

/// The validation layer's errors, and the messages printed (the first 40). Naga's SPIR-V writer
/// decorates struct types it also uses for function variables with offsets (an explicit layout in
/// the Function storage class), which spirv-val refuses since VUID-StandaloneSpirv-None-10684;
/// wgpu hands the game the same SPIR-V, which the driver takes: counted apart.
static ERRORS: AtomicU32 = AtomicU32::new(0);
static NAGA_LAYOUT: AtomicU32 = AtomicU32::new(0);
static PRINTED: AtomicU32 = AtomicU32::new(0);
const NAGA_LAYOUT_VUID: &str = "VUID-StandaloneSpirv-None-10684";

pub fn run(a: &Probe) -> super::Result<bool> {
    const PARTS: [&str; 4] = ["compile", "link", "pixels", "vertex-input"];
    let parts: Vec<&str> = if a.parts.is_empty() { PARTS.to_vec() } else { a.parts.iter().map(String::as_str).collect() };
    if let Some(p) = parts.iter().find(|p| !PARTS.contains(p)) {
        return Err(format!("{p}: no such part ({})", PARTS.join(", ")).into());
    }
    let gpu = Gpu::open(a.validate)?;
    println!("device       {}, driver {}, validation {}, legacy vertex attributes {}", gpu.name, gpu.driver, if a.validate { "on" } else { "off" }, gpu.legacy);
    // A name for this run's entry points: the driver's disk cache has none of them.
    let nonce = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map_or(1, |d| d.as_nanos() as u64);
    let entry = if a.warm { c"main".to_owned() } else { CString::new(format!("p{nonce:x}")).expect("no NUL") };
    let limit = if a.limit == 0 { usize::MAX } else { a.limit };
    let translations = if parts.iter().any(|p| ["compile", "pixels"].contains(p)) { programs(a.zip, limit)? } else { Vec::new() };
    let mut ok = true;
    for part in parts {
        let since = Instant::now();
        let done = match part {
            "compile" => compile(&gpu, &translations, &entry, a.threads),
            "link" => a.cache.ok_or_else(|| "the link part needs --cache".to_string()).and_then(|root| link(&gpu, root, &entry, limit)),
            "pixels" => pixels(&gpu, &translations, &entry, limit),
            _ => vertex_input(&gpu),
        };
        if let Err(e) = done {
            ok = false;
            println!("{part}: {e}");
        }
        println!("{part}: done in {:.1} s", since.elapsed().as_secs_f64());
    }
    let errors = ERRORS.load(Relaxed);
    println!("validation   {errors} errors; {} reports of naga's function-variable layouts ({NAGA_LAYOUT_VUID}, the game's SPIR-V too)", NAGA_LAYOUT.load(Relaxed));
    Ok(ok && errors == 0)
}

/// data.zip's vertex and pixel programs, translated as Setup would: each pixel program with the
/// jump targets its blob holds, each vertex program unpaired (it declares what it writes); at
/// most `limit` of each stage. With each, whether its SASS only jumps forward (`forward_only`).
fn programs(zip: &Path, limit: usize) -> R<Vec<(Translation, bool)>> {
    let mut out = Vec::new();
    let mut counts: HashMap<Stage, usize> = HashMap::new();
    for (_, p) in gyroship_gpu::precompile::inventory(zip).map_err(|e| e.to_string())? {
        if !matches!(p.stage, Stage::Vertex | Stage::Pixel) || counts.get(&p.stage).is_some_and(|n| *n >= limit) {
            continue;
        }
        let Ok(t) = shader::translate_with(&p.code, &Options::default()) else { continue };
        let (t, targets) = match t.interface.jump_tables.clone().filter(|t| !t.is_empty()) {
            Some(tables) => {
                let targets = p.blob_targets(&tables);
                match shader::translate_with(&p.code, &Options { jump_targets: Some(targets.clone()), ..Default::default() }) {
                    Ok(t) => (t, Some(targets)),
                    Err(_) => continue,
                }
            }
            None => (t, None),
        };
        *counts.entry(p.stage).or_default() += 1;
        let forward = forward_only(p.code.get(gyroship_gpu::sass::SPH_SIZE..).unwrap_or_default(), targets.as_ref());
        out.push((t, forward));
    }
    Ok(out)
}

/// Whether a program only jumps forward, so that it ends whatever its inputs: every branch goes
/// past itself (but the closing self-branch), and an indirect jump only to known targets past it.
/// Calls are let through: each returns once (the game's shaders do not recurse).
fn forward_only(sass: &[u8], targets: Option<&std::collections::BTreeSet<usize>>) -> bool {
    use gyroship_gpu::sass::{self, Op, control::ControlOp};
    sass::words(sass).enumerate().all(|(i, w)| match sass::decode(w) {
        Some(Op::Control(ControlOp::Bra { offset, .. })) => offset == -16 || (16 * (i as i64 + 1) + offset) / 16 > i as i64,
        Some(Op::Control(ControlOp::Brx { .. })) => targets.is_some_and(|t| !t.is_empty() && t.iter().all(|&t| t > i)),
        _ => true,
    })
}

/// What a binding of a translated program is, for Vulkan.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Kind {
    Uniform,
    Storage,
    Texture2D,
    Texture,
    Depth,
    Image,
    Sampler,
    CompareSampler,
}

impl Kind {
    fn descriptor(self) -> vk::DescriptorType {
        match self {
            Kind::Uniform => vk::DescriptorType::UNIFORM_BUFFER,
            Kind::Storage => vk::DescriptorType::STORAGE_BUFFER,
            Kind::Texture2D | Kind::Texture | Kind::Depth => vk::DescriptorType::SAMPLED_IMAGE,
            Kind::Image => vk::DescriptorType::STORAGE_IMAGE,
            Kind::Sampler | Kind::CompareSampler => vk::DescriptorType::SAMPLER,
        }
    }
}

/// A translated program as Vulkan takes it: its SPIR-V (entry point renamed), its descriptor
/// bindings (in one set: 0 for a vertex program, 1 for a pixel program, as the translator groups
/// them), and whether the pixel part can draw it: no loop (random inputs cannot hang it), and
/// only banks, 2D float textures and plain samplers.
struct Shader {
    key: u64,
    stage: vk::ShaderStageFlags,
    words: Vec<u32>,
    bindings: Vec<(u32, Kind)>,
    /// Why the pixel part cannot draw it, if it cannot.
    why: Option<String>,
    interface: shader::Interface,
}

fn prepare(t: &Translation, entry: &CStr, forward: bool) -> R<Shader> {
    let name = entry.to_str().map_err(|e| e.to_string())?;
    let wgsl = if name == "main" { t.wgsl.clone() } else { t.wgsl.replacen("fn main(", &format!("fn {name}("), 1) };
    let module = naga::front::wgsl::parse_str(&wgsl).map_err(|e| format!("{:016x}: {}", t.key, e.message()))?;
    let info = naga::valid::Validator::new(naga::valid::ValidationFlags::all(), naga::valid::Capabilities::all()).validate(&module).map_err(|e| format!("{:016x}: {e:?}", t.key))?;
    let (stage, naga_stage) = if t.stage == Stage::Vertex { (vk::ShaderStageFlags::VERTEX, naga::ShaderStage::Vertex) } else { (vk::ShaderStageFlags::FRAGMENT, naga::ShaderStage::Fragment) };
    // Indices clamped: random banks read out of range get a value of the range, the same in every
    // pipeline compared.
    let policy = naga::proc::BoundsCheckPolicy::Restrict;
    let flags = naga::back::spv::WriterFlags::ADJUST_COORDINATE_SPACE | naga::back::spv::WriterFlags::LABEL_VARYINGS | naga::back::spv::WriterFlags::CLAMP_FRAG_DEPTH;
    let options = naga::back::spv::Options {
        lang_version: (1, 6),
        flags,
        bounds_check_policies: naga::proc::BoundsCheckPolicies { index: policy, buffer: policy, image_load: policy, binding_array: policy },
        zero_initialize_workgroup_memory: naga::back::spv::ZeroInitializeWorkgroupMemoryMode::Native,
        ..Default::default()
    };
    let pipeline = naga::back::spv::PipelineOptions { shader_stage: naga_stage, entry_point: name.into() };
    let words = naga::back::spv::write_vec(&module, &info, &options, Some(&pipeline)).map_err(|e| format!("{:016x}: {e}", t.key))?;
    let bindings: Vec<(u32, Kind)> = module
        .global_variables
        .iter()
        .filter_map(|(_, g)| {
            let b = g.binding.as_ref()?;
            let kind = match g.space {
                naga::AddressSpace::Uniform => Kind::Uniform,
                naga::AddressSpace::Storage { .. } => Kind::Storage,
                naga::AddressSpace::Handle => match module.types[g.ty].inner {
                    naga::TypeInner::Image { dim: naga::ImageDimension::D2, arrayed: false, class: naga::ImageClass::Sampled { kind: naga::ScalarKind::Float, multi: false } } => Kind::Texture2D,
                    naga::TypeInner::Image { class: naga::ImageClass::Depth { .. }, .. } => Kind::Depth,
                    naga::TypeInner::Image { class: naga::ImageClass::Storage { .. }, .. } => Kind::Image,
                    naga::TypeInner::Image { .. } => Kind::Texture,
                    naga::TypeInner::Sampler { comparison } => {
                        if comparison {
                            Kind::CompareSampler
                        } else {
                            Kind::Sampler
                        }
                    }
                    _ => return None,
                },
                _ => return None,
            };
            Some((b.binding, kind))
        })
        .collect();
    let other = bindings.iter().map(|(_, k)| *k).find(|k| !matches!(k, Kind::Uniform | Kind::Texture2D | Kind::Sampler));
    let why = if forward { other.map(|k| format!("{k:?} bindings")) } else { Some("a backward jump".to_string()) };
    Ok(Shader { key: t.key, stage, words, bindings, why, interface: t.interface.clone() })
}

/// A vertex program of the probe's own for a pixel program: a full-screen triangle, and each
/// varying the pixel program reads, with its interpolation, made of the position.
fn synthetic_vertex(varyings: &BTreeMap<u8, Interp>, entry: &str) -> String {
    let (mut fields, mut sets) = (String::new(), String::new());
    for (&v, &mode) in varyings {
        let interp = match mode {
            Interp::Constant => "flat",
            Interp::ScreenLinear => "linear",
            _ => "perspective",
        };
        fields += &format!("@location({v}) @interpolate({interp}) v{v}: vec4<f32>,\n");
        let k = f32::from(v) + 1.0;
        sets += &format!("o.v{v} = vec4<f32>(uv.x * {:.3}, uv.y * {:.3} - 0.5, (uv.x + uv.y) * {:.3}, {:.3});\n", 0.5 * k, 1.0 / k, 0.25 * k, 0.1 * k);
    }
    format!(
        "struct Out {{\n@builtin(position) pos: vec4<f32>,\n{fields}}}\n@vertex fn {entry}(@builtin(vertex_index) i: u32) -> Out {{\nvar o: Out;\nlet uv = vec2<f32>(f32((i << 1u) & 2u), f32(i & 2u));\no.pos = vec4<f32>(uv * 2.0 - 1.0, 0.5, 1.0);\n{sets}return o;\n}}\n"
    )
}

/// A program of the probe's own, from its WGSL (without a backward jump).
fn own(stage: Stage, wgsl: String, entry: &CStr) -> R<Shader> {
    prepare(&Translation { key: 0, stage, wgsl, interface: Default::default() }, entry, true)
}

/// Words of a fixed seed (xorshift).
fn random_words(seed: u64, n: usize) -> Vec<u32> {
    let mut x = seed.wrapping_mul(0x9e37_79b9_7f4a_7c15) | 1;
    (0..n)
        .map(|_| {
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            x as u32
        })
        .collect()
}

/// The Vulkan side: an instance (with Khronos' validation layer when asked), the GPU's device with
/// every feature it has, its queue and a command pool.
struct Gpu {
    _entry: ash::Entry,
    instance: ash::Instance,
    debug: Option<(ash::ext::debug_utils::Instance, vk::DebugUtilsMessengerEXT)>,
    device: ash::Device,
    objects: ash::ext::shader_object::Device,
    queue: vk::Queue,
    pool: vk::CommandPool,
    memory: vk::PhysicalDeviceMemoryProperties,
    legacy: bool,
    name: String,
    driver: String,
}

/// VkPhysicalDeviceLegacyVertexAttributesFeaturesEXT, which ash 0.38 predates.
#[repr(C)]
struct Legacy {
    s_type: vk::StructureType,
    p_next: *mut c_void,
    legacy_vertex_attributes: vk::Bool32,
}
const LEGACY_FEATURES: vk::StructureType = vk::StructureType::from_raw(1_000_495_000);

unsafe extern "system" fn report(severity: vk::DebugUtilsMessageSeverityFlagsEXT, _: vk::DebugUtilsMessageTypeFlagsEXT, data: *const vk::DebugUtilsMessengerCallbackDataEXT<'_>, _: *mut c_void) -> vk::Bool32 {
    // SAFETY: the layer hands valid callback data for the call's duration.
    let message = unsafe { data.as_ref().and_then(|d| d.message_as_c_str()) }.map_or_else(String::new, |m| m.to_string_lossy().into_owned());
    if message.contains(NAGA_LAYOUT_VUID) {
        NAGA_LAYOUT.fetch_add(1, Relaxed);
        return vk::FALSE;
    }
    if severity.contains(vk::DebugUtilsMessageSeverityFlagsEXT::ERROR) {
        ERRORS.fetch_add(1, Relaxed);
    }
    if PRINTED.fetch_add(1, Relaxed) < 40 {
        eprintln!("validation {severity:?}: {message}");
    }
    vk::FALSE
}

impl Gpu {
    fn open(validate: bool) -> R<Gpu> {
        let fail = |what: &'static str| move |e: vk::Result| format!("{what}: {e}");
        // SAFETY: the system's Vulkan loader, its functions called as Vulkan defines them; every
        // object made here is destroyed in Drop, in reverse order.
        unsafe {
            let entry = ash::Entry::load().map_err(|e| format!("Vulkan loader: {e}"))?;
            let app = vk::ApplicationInfo::default().application_name(c"gyroship vk-probe").api_version(vk::API_VERSION_1_3);
            let layers = if validate { vec![c"VK_LAYER_KHRONOS_validation".as_ptr()] } else { Vec::new() };
            let extensions = if validate { vec![c"VK_EXT_debug_utils".as_ptr(), c"VK_EXT_validation_features".as_ptr()] } else { Vec::new() };
            let enables = [vk::ValidationFeatureEnableEXT::SYNCHRONIZATION_VALIDATION];
            let mut checks = vk::ValidationFeaturesEXT::default().enabled_validation_features(&enables);
            let mut info = vk::InstanceCreateInfo::default().application_info(&app).enabled_layer_names(&layers).enabled_extension_names(&extensions);
            if validate {
                info = info.push_next(&mut checks);
            }
            let instance = entry.create_instance(&info, None).map_err(fail("instance"))?;
            let debug = if validate {
                let utils = ash::ext::debug_utils::Instance::new(&entry, &instance);
                let info = vk::DebugUtilsMessengerCreateInfoEXT::default()
                    .message_severity(vk::DebugUtilsMessageSeverityFlagsEXT::ERROR | vk::DebugUtilsMessageSeverityFlagsEXT::WARNING)
                    .message_type(vk::DebugUtilsMessageTypeFlagsEXT::GENERAL | vk::DebugUtilsMessageTypeFlagsEXT::VALIDATION | vk::DebugUtilsMessageTypeFlagsEXT::PERFORMANCE)
                    .pfn_user_callback(Some(report));
                let messenger = utils.create_debug_utils_messenger(&info, None).map_err(fail("debug messenger"))?;
                Some((utils, messenger))
            } else {
                None
            };
            let devices = instance.enumerate_physical_devices().map_err(fail("devices"))?;
            let pd = *devices.iter().find(|&&d| instance.get_physical_device_properties(d).vendor_id == 0x10de).or(devices.first()).ok_or("no Vulkan device")?;
            let props = instance.get_physical_device_properties(pd);
            let name = props.device_name_as_c_str().map_or_else(|_| "?".to_string(), |n| n.to_string_lossy().into_owned());
            let driver = format!("{}.{}", props.driver_version >> 22, (props.driver_version >> 14) & 0xff);
            let available: Vec<CString> = instance.enumerate_device_extension_properties(pd).map_err(fail("extensions"))?.iter().filter_map(|e| e.extension_name_as_c_str().ok().map(CStr::to_owned)).collect();
            let has = |n: &CStr| available.iter().any(|a| a.as_c_str() == n);
            let wanted = [c"VK_KHR_pipeline_library", c"VK_EXT_graphics_pipeline_library", c"VK_EXT_shader_object", c"VK_EXT_vertex_input_dynamic_state"];
            if let Some(missing) = wanted.iter().find(|n| !has(n)) {
                return Err(format!("{name}: no {}", missing.to_string_lossy()));
            }
            let legacy = has(c"VK_EXT_legacy_vertex_attributes");
            let mut names: Vec<*const std::ffi::c_char> = wanted.iter().map(|n| n.as_ptr()).collect();
            if legacy {
                names.push(c"VK_EXT_legacy_vertex_attributes".as_ptr());
            }
            // Every feature the device has, enabled: as close to the game's device as a probe needs.
            let (mut v11, mut v12, mut v13) = (vk::PhysicalDeviceVulkan11Features::default(), vk::PhysicalDeviceVulkan12Features::default(), vk::PhysicalDeviceVulkan13Features::default());
            let (mut gpl, mut so, mut vi) = (vk::PhysicalDeviceGraphicsPipelineLibraryFeaturesEXT::default(), vk::PhysicalDeviceShaderObjectFeaturesEXT::default(), vk::PhysicalDeviceVertexInputDynamicStateFeaturesEXT::default());
            let mut old = Legacy { s_type: LEGACY_FEATURES, p_next: std::ptr::null_mut(), legacy_vertex_attributes: vk::FALSE };
            let core = {
                let mut f2 = vk::PhysicalDeviceFeatures2::default().push_next(&mut v11).push_next(&mut v12).push_next(&mut v13).push_next(&mut gpl).push_next(&mut so).push_next(&mut vi);
                if legacy {
                    old.p_next = f2.p_next;
                    f2.p_next = (&raw mut old).cast();
                }
                instance.get_physical_device_features2(pd, &mut f2);
                f2.features
            };
            if gpl.graphics_pipeline_library == vk::FALSE || so.shader_object == vk::FALSE || vi.vertex_input_dynamic_state == vk::FALSE || v13.dynamic_rendering == vk::FALSE {
                return Err(format!("{name}: pipeline libraries, shader objects, dynamic vertex input or dynamic rendering missing"));
            }
            let mut fast = vk::PhysicalDeviceGraphicsPipelineLibraryPropertiesEXT::default();
            {
                let mut p2 = vk::PhysicalDeviceProperties2::default().push_next(&mut fast);
                instance.get_physical_device_properties2(pd, &mut p2);
            }
            println!("libraries    fast linking {}, independent interpolation {}", fast.graphics_pipeline_library_fast_linking, fast.graphics_pipeline_library_independent_interpolation_decoration);
            // The queried structs chain again: each one's link goes first.
            (v11.p_next, v12.p_next, v13.p_next) = (std::ptr::null_mut(), std::ptr::null_mut(), std::ptr::null_mut());
            (gpl.p_next, so.p_next, vi.p_next, old.p_next) = (std::ptr::null_mut(), std::ptr::null_mut(), std::ptr::null_mut(), std::ptr::null_mut());
            let family = instance.get_physical_device_queue_family_properties(pd).iter().position(|f| f.queue_flags.contains(vk::QueueFlags::GRAPHICS)).ok_or("no graphics queue")? as u32;
            let priorities = [1.0f32];
            let queues = [vk::DeviceQueueCreateInfo::default().queue_family_index(family).queue_priorities(&priorities)];
            let mut f2 = vk::PhysicalDeviceFeatures2::default().features(core).push_next(&mut v11).push_next(&mut v12).push_next(&mut v13).push_next(&mut gpl).push_next(&mut so).push_next(&mut vi);
            if legacy {
                old.p_next = f2.p_next;
                f2.p_next = (&raw mut old).cast();
            }
            let info = vk::DeviceCreateInfo::default().queue_create_infos(&queues).enabled_extension_names(&names).push_next(&mut f2);
            let device = instance.create_device(pd, &info, None).map_err(fail("device"))?;
            let objects = ash::ext::shader_object::Device::new(&instance, &device);
            let queue = device.get_device_queue(family, 0);
            let pool = device.create_command_pool(&vk::CommandPoolCreateInfo::default().queue_family_index(family).flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER), None).map_err(fail("command pool"))?;
            let memory = instance.get_physical_device_memory_properties(pd);
            Ok(Gpu { _entry: entry, instance, debug, device, objects, queue, pool, memory, legacy: legacy && old.legacy_vertex_attributes == vk::TRUE, name, driver })
        }
    }

    fn pipeline_cache(&self) -> R<vk::PipelineCache> {
        // SAFETY: an empty cache, destroyed by its user.
        unsafe { self.device.create_pipeline_cache(&vk::PipelineCacheCreateInfo::default(), None) }.map_err(|e| format!("pipeline cache: {e}"))
    }

    /// The set layout of a program's bindings (an empty one without a program).
    fn set_layout(&self, s: Option<&Shader>) -> R<vk::DescriptorSetLayout> {
        let bindings: Vec<vk::DescriptorSetLayoutBinding> = s.map_or_else(Vec::new, |s| s.bindings.iter().map(|&(b, k)| vk::DescriptorSetLayoutBinding::default().binding(b).descriptor_type(k.descriptor()).descriptor_count(1).stage_flags(s.stage)).collect());
        // SAFETY: valid bindings; the layout is destroyed by its user.
        unsafe { self.device.create_descriptor_set_layout(&vk::DescriptorSetLayoutCreateInfo::default().bindings(&bindings), None) }.map_err(|e| format!("set layout: {e}"))
    }

    /// A pipeline layout of independent sets (a library's may leave the other stage's null), or of
    /// plain ones (the shader objects': they are made without independent sets).
    fn layout(&self, sets: &[vk::DescriptorSetLayout], independent: bool) -> R<vk::PipelineLayout> {
        let flags = if independent { vk::PipelineLayoutCreateFlags::INDEPENDENT_SETS_EXT } else { vk::PipelineLayoutCreateFlags::empty() };
        let info = vk::PipelineLayoutCreateInfo::default().flags(flags).set_layouts(sets);
        // SAFETY: valid set layouts or null ones (graphicsPipelineLibrary is enabled).
        unsafe { self.device.create_pipeline_layout(&info, None) }.map_err(|e| format!("pipeline layout: {e}"))
    }

    /// A pipeline, or the library of one `part`: the programs' stages, `targets` RGBA32F color
    /// targets, dynamic viewport, scissor and vertex input.
    fn pipeline(&self, part: Option<vk::GraphicsPipelineLibraryFlagsEXT>, stages: &[&Shader], entry: &CStr, layout: vk::PipelineLayout, targets: usize, flags: vk::PipelineCreateFlags, cache: vk::PipelineCache) -> Result<vk::Pipeline, vk::Result> {
        // SAFETY: SPIR-V from naga's writer, states valid for every part; the modules go once the
        // pipeline is made, the pipeline is its caller's.
        unsafe {
            let mut modules = Vec::new();
            for s in stages {
                modules.push(self.device.create_shader_module(&vk::ShaderModuleCreateInfo::default().code(&s.words), None)?);
            }
            let infos: Vec<vk::PipelineShaderStageCreateInfo> = stages.iter().zip(&modules).map(|(s, &m)| vk::PipelineShaderStageCreateInfo::default().stage(s.stage).module(m).name(entry)).collect();
            let vertex_input = vk::PipelineVertexInputStateCreateInfo::default();
            let assembly = vk::PipelineInputAssemblyStateCreateInfo::default().topology(vk::PrimitiveTopology::TRIANGLE_LIST);
            let viewport = vk::PipelineViewportStateCreateInfo::default().viewport_count(1).scissor_count(1);
            let raster = vk::PipelineRasterizationStateCreateInfo::default().polygon_mode(vk::PolygonMode::FILL).cull_mode(vk::CullModeFlags::NONE).front_face(vk::FrontFace::COUNTER_CLOCKWISE).line_width(1.0);
            let multisample = vk::PipelineMultisampleStateCreateInfo::default().rasterization_samples(vk::SampleCountFlags::TYPE_1);
            let depth = vk::PipelineDepthStencilStateCreateInfo::default();
            let attachments = vec![vk::PipelineColorBlendAttachmentState::default().color_write_mask(vk::ColorComponentFlags::RGBA); targets];
            let blend = vk::PipelineColorBlendStateCreateInfo::default().attachments(&attachments);
            // Each part's own dynamic states: vertex input for the input interface, viewport and
            // scissor for the shaders before rasterization; a whole pipeline has them all.
            let all = vk::GraphicsPipelineLibraryFlagsEXT::VERTEX_INPUT_INTERFACE | vk::GraphicsPipelineLibraryFlagsEXT::PRE_RASTERIZATION_SHADERS;
            let p = part.unwrap_or(all);
            let mut dynamic_states = Vec::new();
            if p.contains(vk::GraphicsPipelineLibraryFlagsEXT::VERTEX_INPUT_INTERFACE) {
                dynamic_states.push(vk::DynamicState::VERTEX_INPUT_EXT);
            }
            if p.contains(vk::GraphicsPipelineLibraryFlagsEXT::PRE_RASTERIZATION_SHADERS) {
                dynamic_states.extend([vk::DynamicState::VIEWPORT, vk::DynamicState::SCISSOR]);
            }
            let dynamic = vk::PipelineDynamicStateCreateInfo::default().dynamic_states(&dynamic_states);
            let formats = vec![vk::Format::R32G32B32A32_SFLOAT; targets];
            let mut rendering = vk::PipelineRenderingCreateInfo::default().color_attachment_formats(&formats);
            let mut library = vk::GraphicsPipelineLibraryCreateInfoEXT::default().flags(part.unwrap_or_default());
            let mut info = vk::GraphicsPipelineCreateInfo::default()
                .flags(flags)
                .stages(&infos)
                .vertex_input_state(&vertex_input)
                .input_assembly_state(&assembly)
                .viewport_state(&viewport)
                .rasterization_state(&raster)
                .multisample_state(&multisample)
                .depth_stencil_state(&depth)
                .color_blend_state(&blend)
                .dynamic_state(&dynamic)
                .layout(layout)
                .push_next(&mut rendering);
            if part.is_some() {
                info = info.push_next(&mut library);
            }
            let made = self.device.create_graphics_pipelines(cache, &[info], None);
            for m in modules {
                self.device.destroy_shader_module(m, None);
            }
            made.map(|p| p[0]).map_err(|(_, e)| e)
        }
    }

    /// A pipeline linked from its four libraries, fast or with link-time optimisation.
    fn link(&self, libraries: &[vk::Pipeline], layout: vk::PipelineLayout, lto: bool, cache: vk::PipelineCache) -> Result<vk::Pipeline, vk::Result> {
        let mut library = vk::PipelineLibraryCreateInfoKHR::default().libraries(libraries);
        let flags = if lto { vk::PipelineCreateFlags::LINK_TIME_OPTIMIZATION_EXT } else { vk::PipelineCreateFlags::empty() };
        let info = vk::GraphicsPipelineCreateInfo::default().flags(flags).layout(layout).push_next(&mut library);
        // SAFETY: libraries made with compatible layouts, retained for link-time optimisation.
        unsafe { self.device.create_graphics_pipelines(cache, &[info], None) }.map(|p| p[0]).map_err(|(_, e)| e)
    }

    /// The library of one program: its time (ms), everything destroyed after.
    fn library(&self, s: &Shader, entry: &CStr, cache: vk::PipelineCache, flags: vk::PipelineCreateFlags) -> R<f64> {
        let set = self.set_layout(Some(s))?;
        let vertex = s.stage == vk::ShaderStageFlags::VERTEX;
        let null = vk::DescriptorSetLayout::null();
        let layout = self.layout(&if vertex { [set, null] } else { [null, set] }, true)?;
        let part = if vertex { vk::GraphicsPipelineLibraryFlagsEXT::PRE_RASTERIZATION_SHADERS } else { vk::GraphicsPipelineLibraryFlagsEXT::FRAGMENT_SHADER };
        let since = Instant::now();
        let made = self.pipeline(Some(part), &[s], entry, layout, 1, flags, cache);
        let ms = since.elapsed().as_secs_f64() * 1e3;
        // SAFETY: objects of this call only.
        unsafe {
            if let Ok(p) = made {
                self.device.destroy_pipeline(p, None);
            }
            self.device.destroy_pipeline_layout(layout, None);
            self.device.destroy_descriptor_set_layout(set, None);
        }
        made.map(|_| ms).map_err(|e| format!("{:016x}: {e}", s.key))
    }

    fn memory_type(&self, bits: u32, flags: vk::MemoryPropertyFlags) -> R<u32> {
        (0..self.memory.memory_type_count).find(|&i| bits & (1 << i) != 0 && self.memory.memory_types[i as usize].property_flags.contains(flags)).ok_or_else(|| format!("no memory type {flags:?}"))
    }

    /// A host-visible buffer, mapped for its life.
    fn buffer(&self, size: u64, usage: vk::BufferUsageFlags) -> R<Buffer> {
        // SAFETY: a buffer, its memory bound and mapped once; freed by `free_buffer`.
        unsafe {
            let buffer = self.device.create_buffer(&vk::BufferCreateInfo::default().size(size).usage(usage), None).map_err(|e| format!("buffer: {e}"))?;
            let need = self.device.get_buffer_memory_requirements(buffer);
            let kind = self.memory_type(need.memory_type_bits, vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT)?;
            let memory = self.device.allocate_memory(&vk::MemoryAllocateInfo::default().allocation_size(need.size).memory_type_index(kind), None).map_err(|e| format!("memory: {e}"))?;
            self.device.bind_buffer_memory(buffer, memory, 0).map_err(|e| format!("bind: {e}"))?;
            let at = self.device.map_memory(memory, 0, size, vk::MemoryMapFlags::empty()).map_err(|e| format!("map: {e}"))?.cast::<u8>();
            Ok(Buffer { buffer, memory, at, size })
        }
    }

    /// A buffer holding `words`.
    fn filled(&self, words: &[u32], usage: vk::BufferUsageFlags) -> R<Buffer> {
        let b = self.buffer(4 * words.len() as u64, usage)?;
        // SAFETY: the mapping holds the buffer's size, which is the words'.
        unsafe { std::ptr::copy_nonoverlapping(words.as_ptr().cast::<u8>(), b.at, 4 * words.len()) };
        Ok(b)
    }

    fn free_buffer(&self, b: &Buffer) {
        // SAFETY: the buffer is no longer in use (every submit is waited for).
        unsafe {
            self.device.destroy_buffer(b.buffer, None);
            self.device.free_memory(b.memory, None);
        }
    }

    /// A SIDE×SIDE image of the device's memory, with a view.
    fn image(&self, format: vk::Format, usage: vk::ImageUsageFlags) -> R<Image> {
        // SAFETY: an image, its memory and view; freed by `free_image`.
        unsafe {
            let info = vk::ImageCreateInfo::default()
                .image_type(vk::ImageType::TYPE_2D)
                .format(format)
                .extent(vk::Extent3D { width: SIDE, height: SIDE, depth: 1 })
                .mip_levels(1)
                .array_layers(1)
                .samples(vk::SampleCountFlags::TYPE_1)
                .tiling(vk::ImageTiling::OPTIMAL)
                .usage(usage)
                .initial_layout(vk::ImageLayout::UNDEFINED);
            let image = self.device.create_image(&info, None).map_err(|e| format!("image: {e}"))?;
            let need = self.device.get_image_memory_requirements(image);
            let kind = self.memory_type(need.memory_type_bits, vk::MemoryPropertyFlags::DEVICE_LOCAL)?;
            let memory = self.device.allocate_memory(&vk::MemoryAllocateInfo::default().allocation_size(need.size).memory_type_index(kind), None).map_err(|e| format!("memory: {e}"))?;
            self.device.bind_image_memory(image, memory, 0).map_err(|e| format!("bind: {e}"))?;
            let view = self.device.create_image_view(&vk::ImageViewCreateInfo::default().image(image).view_type(vk::ImageViewType::TYPE_2D).format(format).subresource_range(COLOR), None).map_err(|e| format!("view: {e}"))?;
            Ok(Image { image, memory, view })
        }
    }

    fn free_image(&self, i: &Image) {
        // SAFETY: the image is no longer in use (every submit is waited for).
        unsafe {
            self.device.destroy_image_view(i.view, None);
            self.device.destroy_image(i.image, None);
            self.device.free_memory(i.memory, None);
        }
    }

    /// Records one command buffer, submits it and waits for its end.
    fn once(&self, record: impl FnOnce(vk::CommandBuffer)) -> R<()> {
        // SAFETY: a command buffer of this pool, recorded, submitted, waited for, then freed.
        unsafe {
            let cb = self.device.allocate_command_buffers(&vk::CommandBufferAllocateInfo::default().command_pool(self.pool).level(vk::CommandBufferLevel::PRIMARY).command_buffer_count(1)).map_err(|e| format!("command buffer: {e}"))?[0];
            self.device.begin_command_buffer(cb, &vk::CommandBufferBeginInfo::default().flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT)).map_err(|e| format!("begin: {e}"))?;
            record(cb);
            self.device.end_command_buffer(cb).map_err(|e| format!("end: {e}"))?;
            let fence = self.device.create_fence(&vk::FenceCreateInfo::default(), None).map_err(|e| format!("fence: {e}"))?;
            let buffers = [cb];
            let done = self.device.queue_submit(self.queue, &[vk::SubmitInfo::default().command_buffers(&buffers)], fence).and_then(|()| self.device.wait_for_fences(&[fence], true, u64::MAX));
            self.device.destroy_fence(fence, None);
            self.device.free_command_buffers(self.pool, &buffers);
            done.map_err(|e| format!("submit: {e}"))
        }
    }

    /// An image's layout changed, every access before it done.
    fn barrier(&self, cb: vk::CommandBuffer, image: vk::Image, from: vk::ImageLayout, to: vk::ImageLayout) {
        let b = vk::ImageMemoryBarrier::default()
            .src_access_mask(vk::AccessFlags::MEMORY_WRITE)
            .dst_access_mask(vk::AccessFlags::MEMORY_READ | vk::AccessFlags::MEMORY_WRITE)
            .old_layout(from)
            .new_layout(to)
            .image(image)
            .subresource_range(COLOR);
        // SAFETY: a barrier on a valid image, recorded in a command buffer being recorded.
        unsafe { self.device.cmd_pipeline_barrier(cb, vk::PipelineStageFlags::ALL_COMMANDS, vk::PipelineStageFlags::ALL_COMMANDS, vk::DependencyFlags::empty(), &[], &[], &[b]) };
    }

    /// A texture of random RGBA8 texels, ready to be sampled.
    fn texture(&self, seed: u64) -> R<Image> {
        let image = self.image(vk::Format::R8G8B8A8_UNORM, vk::ImageUsageFlags::SAMPLED | vk::ImageUsageFlags::TRANSFER_DST)?;
        let staging = self.filled(&random_words(seed, (SIDE * SIDE) as usize), vk::BufferUsageFlags::TRANSFER_SRC)?;
        let region = vk::BufferImageCopy::default().image_subresource(LAYERS).image_extent(vk::Extent3D { width: SIDE, height: SIDE, depth: 1 });
        let done = self.once(|cb| {
            self.barrier(cb, image.image, vk::ImageLayout::UNDEFINED, vk::ImageLayout::TRANSFER_DST_OPTIMAL);
            // SAFETY: a copy into the image just made, in its transfer layout.
            unsafe { self.device.cmd_copy_buffer_to_image(cb, staging.buffer, image.image, vk::ImageLayout::TRANSFER_DST_OPTIMAL, &[region]) };
            self.barrier(cb, image.image, vk::ImageLayout::TRANSFER_DST_OPTIMAL, vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL);
        });
        self.free_buffer(&staging);
        done.map(|()| image)
    }

    /// Draws a full-screen triangle into `targets` RGBA32F targets cleared to zero, with a
    /// pipeline or shader objects, the pixel program's set (set 1) bound, and the vertex input of
    /// `vertex` (a buffer and its attribute's format) or none: the targets' bytes, in order.
    fn draw(&self, with: With, layout: vk::PipelineLayout, set: Option<vk::DescriptorSet>, targets: usize, vertex: Option<(vk::Buffer, vk::Format)>) -> R<Vec<u8>> {
        let mut images = Vec::new();
        for _ in 0..targets {
            images.push(self.image(vk::Format::R32G32B32A32_SFLOAT, vk::ImageUsageFlags::COLOR_ATTACHMENT | vk::ImageUsageFlags::TRANSFER_SRC)?);
        }
        let readback = self.buffer(TARGET_BYTES * targets as u64, vk::BufferUsageFlags::TRANSFER_DST)?;
        let done = self.once(|cb| {
            for i in &images {
                self.barrier(cb, i.image, vk::ImageLayout::UNDEFINED, vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL);
            }
            let attachments: Vec<vk::RenderingAttachmentInfo> = images
                .iter()
                .map(|i| vk::RenderingAttachmentInfo::default().image_view(i.view).image_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL).load_op(vk::AttachmentLoadOp::CLEAR).store_op(vk::AttachmentStoreOp::STORE).clear_value(vk::ClearValue { color: vk::ClearColorValue { float32: [0.0; 4] } }))
                .collect();
            let area = vk::Rect2D { offset: vk::Offset2D::default(), extent: vk::Extent2D { width: SIDE, height: SIDE } };
            let viewport = vk::Viewport { x: 0.0, y: 0.0, width: SIDE as f32, height: SIDE as f32, min_depth: 0.0, max_depth: 1.0 };
            let bindings: Vec<vk::VertexInputBindingDescription2EXT> = vertex.iter().map(|_| vk::VertexInputBindingDescription2EXT::default().binding(0).stride(16).input_rate(vk::VertexInputRate::VERTEX).divisor(1)).collect();
            let attributes: Vec<vk::VertexInputAttributeDescription2EXT> = vertex.iter().map(|(_, format)| vk::VertexInputAttributeDescription2EXT::default().location(0).binding(0).format(*format).offset(0)).collect();
            // SAFETY: commands on valid objects, in a command buffer being recorded; every state
            // the draw needs is set.
            unsafe {
                self.device.cmd_begin_rendering(cb, &vk::RenderingInfo::default().render_area(area).layer_count(1).color_attachments(&attachments));
                match with {
                    With::Pipeline(p) => {
                        self.device.cmd_bind_pipeline(cb, vk::PipelineBindPoint::GRAPHICS, p);
                        self.device.cmd_set_viewport(cb, 0, &[viewport]);
                        self.device.cmd_set_scissor(cb, 0, &[area]);
                    }
                    With::Objects(vs, fs) => self.object_states(cb, vs, fs, targets, viewport, area),
                }
                self.objects.cmd_set_vertex_input(cb, &bindings, &attributes);
                if let Some((buffer, _)) = vertex {
                    self.device.cmd_bind_vertex_buffers(cb, 0, &[buffer], &[0]);
                }
                if let Some(set) = set {
                    self.device.cmd_bind_descriptor_sets(cb, vk::PipelineBindPoint::GRAPHICS, layout, 1, &[set], &[]);
                }
                self.device.cmd_draw(cb, 3, 1, 0, 0);
                self.device.cmd_end_rendering(cb);
            }
            for (n, i) in images.iter().enumerate() {
                self.barrier(cb, i.image, vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL, vk::ImageLayout::TRANSFER_SRC_OPTIMAL);
                let region = vk::BufferImageCopy::default().buffer_offset(TARGET_BYTES * n as u64).image_subresource(LAYERS).image_extent(vk::Extent3D { width: SIDE, height: SIDE, depth: 1 });
                // SAFETY: a copy out of an image in its transfer layout, into the readback buffer.
                unsafe { self.device.cmd_copy_image_to_buffer(cb, i.image, vk::ImageLayout::TRANSFER_SRC_OPTIMAL, readback.buffer, &[region]) };
            }
            let host = vk::MemoryBarrier::default().src_access_mask(vk::AccessFlags::TRANSFER_WRITE).dst_access_mask(vk::AccessFlags::HOST_READ);
            // SAFETY: the copies' writes made visible to the host.
            unsafe { self.device.cmd_pipeline_barrier(cb, vk::PipelineStageFlags::TRANSFER, vk::PipelineStageFlags::HOST, vk::DependencyFlags::empty(), &[host], &[], &[]) };
        });
        // SAFETY: the mapping holds the readback's size; the submit is over.
        let bytes = done.map(|()| unsafe { std::slice::from_raw_parts(readback.at, readback.size as usize) }.to_vec());
        self.free_buffer(&readback);
        for i in &images {
            self.free_image(i);
        }
        bytes
    }

    /// Shader objects bound, with every state a draw with them needs.
    ///
    /// # Safety
    /// `cb` is being recorded, inside a rendering of `targets` color targets.
    unsafe fn object_states(&self, cb: vk::CommandBuffer, vs: vk::ShaderEXT, fs: vk::ShaderEXT, targets: usize, viewport: vk::Viewport, area: vk::Rect2D) {
        let o = &self.objects;
        // SAFETY: the caller's.
        unsafe {
            let stages = [vk::ShaderStageFlags::VERTEX, vk::ShaderStageFlags::TESSELLATION_CONTROL, vk::ShaderStageFlags::TESSELLATION_EVALUATION, vk::ShaderStageFlags::GEOMETRY, vk::ShaderStageFlags::FRAGMENT];
            o.cmd_bind_shaders(cb, &stages, &[vs, vk::ShaderEXT::null(), vk::ShaderEXT::null(), vk::ShaderEXT::null(), fs]);
            o.cmd_set_viewport_with_count(cb, &[viewport]);
            o.cmd_set_scissor_with_count(cb, &[area]);
            o.cmd_set_rasterizer_discard_enable(cb, false);
            o.cmd_set_cull_mode(cb, vk::CullModeFlags::NONE);
            o.cmd_set_front_face(cb, vk::FrontFace::COUNTER_CLOCKWISE);
            o.cmd_set_depth_test_enable(cb, false);
            o.cmd_set_depth_write_enable(cb, false);
            o.cmd_set_depth_bounds_test_enable(cb, false);
            o.cmd_set_depth_bias_enable(cb, false);
            o.cmd_set_depth_clamp_enable(cb, false);
            o.cmd_set_stencil_test_enable(cb, false);
            o.cmd_set_primitive_topology(cb, vk::PrimitiveTopology::TRIANGLE_LIST);
            o.cmd_set_primitive_restart_enable(cb, false);
            o.cmd_set_polygon_mode(cb, vk::PolygonMode::FILL);
            o.cmd_set_rasterization_samples(cb, vk::SampleCountFlags::TYPE_1);
            o.cmd_set_sample_mask(cb, vk::SampleCountFlags::TYPE_1, &[u32::MAX]);
            o.cmd_set_alpha_to_coverage_enable(cb, false);
            o.cmd_set_alpha_to_one_enable(cb, false);
            o.cmd_set_logic_op_enable(cb, false);
            if targets > 0 {
                o.cmd_set_color_blend_enable(cb, 0, &vec![vk::FALSE; targets]);
                o.cmd_set_color_write_mask(cb, 0, &vec![vk::ColorComponentFlags::RGBA; targets]);
            }
        }
    }
}

impl Drop for Gpu {
    fn drop(&mut self) {
        // SAFETY: the objects made in `open`, in reverse order, the device idle first.
        unsafe {
            let _ = self.device.device_wait_idle();
            self.device.destroy_command_pool(self.pool, None);
            self.device.destroy_device(None);
            if let Some((utils, messenger)) = &self.debug {
                utils.destroy_debug_utils_messenger(*messenger, None);
            }
            self.instance.destroy_instance(None);
        }
    }
}

const COLOR: vk::ImageSubresourceRange = vk::ImageSubresourceRange { aspect_mask: vk::ImageAspectFlags::COLOR, base_mip_level: 0, level_count: 1, base_array_layer: 0, layer_count: 1 };
const LAYERS: vk::ImageSubresourceLayers = vk::ImageSubresourceLayers { aspect_mask: vk::ImageAspectFlags::COLOR, mip_level: 0, base_array_layer: 0, layer_count: 1 };

/// A host-visible buffer, mapped for its life.
struct Buffer {
    buffer: vk::Buffer,
    memory: vk::DeviceMemory,
    at: *mut u8,
    size: u64,
}

/// A SIDE×SIDE image of the device's memory, with its view.
struct Image {
    image: vk::Image,
    memory: vk::DeviceMemory,
    view: vk::ImageView,
}

/// What a draw is made with: a pipeline, or a vertex and a pixel shader object.
#[derive(Clone, Copy)]
enum With {
    Pipeline(vk::Pipeline),
    Objects(vk::ShaderEXT, vk::ShaderEXT),
}

/// Times (ms) as a line: count, mean, median, 95th percentile, maximum, total.
fn summary(label: &str, ms: &[f64]) -> String {
    if ms.is_empty() {
        return format!("{label}: none");
    }
    let mut v = ms.to_vec();
    v.sort_by(f64::total_cmp);
    let at = |q: f64| v[((v.len() - 1) as f64 * q) as usize];
    let total: f64 = v.iter().sum();
    format!("{label}: {} made, mean {:.2} ms, median {:.2}, p95 {:.2}, max {:.2}, total {:.1} s", v.len(), total / v.len() as f64, at(0.5), at(0.95), v[v.len() - 1], total / 1e3)
}

/// The process's committed memory, sampled every 20 ms on a thread of its own, at its peak.
struct Watch {
    stop: Arc<AtomicBool>,
    peak: Arc<AtomicU64>,
    base: u64,
    thread: std::thread::JoinHandle<()>,
}

impl Watch {
    fn start() -> Watch {
        let (stop, peak, base) = (Arc::new(AtomicBool::new(false)), Arc::new(AtomicU64::new(0)), host::committed());
        let (s, p) = (stop.clone(), peak.clone());
        let thread = std::thread::spawn(move || {
            while !s.load(Relaxed) {
                p.fetch_max(host::committed(), Relaxed);
                std::thread::sleep(Duration::from_millis(20));
            }
        });
        Watch { stop, peak, base, thread }
    }

    /// Its peak above where it started, in MB.
    fn stop(self) -> f64 {
        self.stop.store(true, Relaxed);
        let _ = self.thread.join();
        self.peak.load(Relaxed).saturating_sub(self.base) as f64 / 1e6
    }
}

/// `compile`: each program's library, half on one thread and half on `threads`.
fn compile(gpu: &Gpu, translations: &[(Translation, bool)], entry: &CStr, threads: usize) -> R<()> {
    let shaders: Vec<Shader> = translations.iter().filter_map(|(t, forward)| prepare(t, entry, *forward).map_err(|e| eprintln!("{e}")).ok()).collect();
    let vertex = shaders.iter().filter(|s| s.stage == vk::ShaderStageFlags::VERTEX).count();
    println!("compile      {} programs: {vertex} vertex, {} pixel", shaders.len(), shaders.len() - vertex);
    let (one, many): (Vec<&Shader>, Vec<&Shader>) = shaders.iter().enumerate().fold((Vec::new(), Vec::new()), |(mut a, mut b), (i, s)| {
        if i % 2 == 0 { a.push(s) } else { b.push(s) }
        (a, b)
    });
    let retain = vk::PipelineCreateFlags::LIBRARY_KHR | vk::PipelineCreateFlags::RETAIN_LINK_TIME_OPTIMIZATION_INFO_EXT;
    let first = gpu.pipeline_cache()?;
    libraries(gpu, &one, entry, 1, first, retain, "1 thread")?;
    let second = gpu.pipeline_cache()?;
    libraries(gpu, &many, entry, threads, second, retain, &format!("{threads} threads"))?;
    // Again from the first half's cache: nothing should compile.
    let sample = &one[..one.len().min(200)];
    libraries(gpu, sample, entry, 1, first, retain | vk::PipelineCreateFlags::FAIL_ON_PIPELINE_COMPILE_REQUIRED, "again, from the pipeline cache, failing on a compile")?;
    // Again with an empty cache: what the driver's own disk cache gives a second launch.
    let empty = gpu.pipeline_cache()?;
    libraries(gpu, sample, entry, 1, empty, retain, "again, with an empty pipeline cache")?;
    // SAFETY: caches no pipeline creation uses any more.
    unsafe {
        for c in [first, second, empty] {
            gpu.device.destroy_pipeline_cache(c, None);
        }
    }
    Ok(())
}

/// Each shader's library on `threads` threads, each destroyed once made: the times by stage,
/// the failures, and the process's committed memory at its peak.
fn libraries(gpu: &Gpu, shaders: &[&Shader], entry: &CStr, threads: usize, cache: vk::PipelineCache, flags: vk::PipelineCreateFlags, label: &str) -> R<()> {
    let next = AtomicUsize::new(0);
    let times: Mutex<Vec<(vk::ShaderStageFlags, f64)>> = Mutex::new(Vec::new());
    let failed: Mutex<Vec<String>> = Mutex::new(Vec::new());
    let watch = Watch::start();
    let since = Instant::now();
    std::thread::scope(|scope| {
        for _ in 0..threads.max(1) {
            scope.spawn(|| {
                while let Some(s) = shaders.get(next.fetch_add(1, Relaxed)) {
                    match gpu.library(s, entry, cache, flags) {
                        Ok(ms) => times.lock().expect("times").push((s.stage, ms)),
                        Err(e) => failed.lock().expect("failed").push(e),
                    }
                }
            });
        }
    });
    let wall = since.elapsed().as_secs_f64();
    let peak = watch.stop();
    let times = times.into_inner().expect("times");
    let failed = failed.into_inner().expect("failed");
    for (name, stage) in [("vertex", vk::ShaderStageFlags::VERTEX), ("pixel", vk::ShaderStageFlags::FRAGMENT)] {
        let ms: Vec<f64> = times.iter().filter(|t| t.0 == stage).map(|t| t.1).collect();
        println!("  {label}, {name} libraries: {}", summary("", &ms).trim_start_matches(": "));
    }
    let compile_required = failed.iter().filter(|e| e.contains("PIPELINE_COMPILE_REQUIRED")).count();
    println!("  {label}: {wall:.1} s in all, {} failed ({compile_required} asked for a compile), committed memory peak +{peak:.0} MB", failed.len());
    for e in failed.iter().filter(|e| !e.contains("PIPELINE_COMPILE_REQUIRED")).take(5) {
        println!("    {e}");
    }
    Ok(())
}

/// `link`: pairs play made, their libraries linked fast, with link-time optimisation, and made
/// whole (from another entry name: no stage the libraries compiled serves it).
fn link(gpu: &Gpu, root: &Path, entry: &CStr, limit: usize) -> R<()> {
    let (stored, _) = cache::entries(root);
    let mut pixels: HashMap<Vec<(u8, Interp)>, &cache::Stored> = HashMap::new();
    for s in stored.iter().filter(|s| s.stage == Stage::Pixel) {
        pixels.entry(s.interface.varyings.iter().map(|(l, m)| (*l, *m)).collect()).or_insert(s);
    }
    let mut seen = HashSet::new();
    let mut pairs = Vec::new();
    for v in stored.iter().filter(|s| s.stage == Stage::Vertex) {
        let want: Vec<(u8, Interp)> = v.options.varyings.iter().map(|(l, m)| (*l, *m)).collect();
        if let Some(p) = pixels.get(&want)
            && seen.insert((v.key, p.key))
            && pairs.len() < limit.min(DRAWN)
        {
            pairs.push((v, *p));
        }
    }
    let whole_entry = CString::new(format!("{}w", entry.to_str().unwrap_or("main"))).expect("no NUL");
    let mut prepared = Vec::new();
    for (v, p) in &pairs {
        let (vt, pt) = (shader::translate_with(&v.code, &v.options).map_err(|e| e.to_string())?, shader::translate_with(&p.code, &p.options).map_err(|e| e.to_string())?);
        prepared.push(((prepare(&vt, entry, true)?, prepare(&pt, entry, true)?), (prepare(&vt, &whole_entry, true)?, prepare(&pt, &whole_entry, true)?)));
    }
    println!("link         {} pairs play made", prepared.len());
    let cache = gpu.pipeline_cache()?;
    let retain = vk::PipelineCreateFlags::LIBRARY_KHR | vk::PipelineCreateFlags::RETAIN_LINK_TIME_OPTIMIZATION_INFO_EXT;
    let null = vk::PipelineLayout::null();
    let input = gpu.pipeline(Some(vk::GraphicsPipelineLibraryFlagsEXT::VERTEX_INPUT_INTERFACE), &[], entry, null, 0, retain, cache).map_err(|e| format!("vertex input library: {e}"))?;
    let mut outputs: HashMap<usize, vk::Pipeline> = HashMap::new();
    let (mut libs, mut fast, mut lto, mut whole, mut failed) = (Vec::new(), Vec::new(), Vec::new(), Vec::new(), Vec::new());
    // Everything stays alive until the driver has had its 30 s: work it does in the background
    // for a pipeline would stop with it.
    let (mut alive, mut layouts, mut sets) = (Vec::new(), Vec::new(), Vec::new());
    for ((vs, ps), (wvs, wps)) in &prepared {
        let targets = ps.interface.targets.keys().max().map_or(0, |t| usize::from(*t) + 1);
        let output = match outputs.get(&targets) {
            Some(o) => *o,
            None => {
                let o = gpu.pipeline(Some(vk::GraphicsPipelineLibraryFlagsEXT::FRAGMENT_OUTPUT_INTERFACE), &[], entry, null, targets, retain, cache).map_err(|e| format!("output library: {e}"))?;
                outputs.insert(targets, o);
                o
            }
        };
        let (vset, pset) = (gpu.set_layout(Some(vs))?, gpu.set_layout(Some(ps))?);
        let nullset = vk::DescriptorSetLayout::null();
        let (vl, pl, full) = (gpu.layout(&[vset, nullset], true)?, gpu.layout(&[nullset, pset], true)?, gpu.layout(&[vset, pset], true)?);
        let mut made = Vec::new();
        let mut time = |what: &mut Vec<f64>, f: &dyn Fn() -> Result<vk::Pipeline, vk::Result>| -> Option<vk::Pipeline> {
            let since = Instant::now();
            match f() {
                Ok(p) => {
                    what.push(since.elapsed().as_secs_f64() * 1e3);
                    made.push(p);
                    Some(p)
                }
                Err(e) => {
                    failed.push(format!("{:016x}+{:016x}: {e}", vs.key, ps.key));
                    None
                }
            }
        };
        let vlib = time(&mut libs, &|| gpu.pipeline(Some(vk::GraphicsPipelineLibraryFlagsEXT::PRE_RASTERIZATION_SHADERS), &[vs], entry, vl, targets, retain, cache));
        let plib = time(&mut libs, &|| gpu.pipeline(Some(vk::GraphicsPipelineLibraryFlagsEXT::FRAGMENT_SHADER), &[ps], entry, pl, targets, retain, cache));
        if let (Some(vlib), Some(plib)) = (vlib, plib) {
            let parts = [input, vlib, plib, output];
            time(&mut fast, &|| gpu.link(&parts, full, false, cache));
            time(&mut lto, &|| gpu.link(&parts, full, true, cache));
        }
        time(&mut whole, &|| gpu.pipeline(None, &[wvs, wps], &whole_entry, full, targets, vk::PipelineCreateFlags::empty(), cache));
        alive.extend(made);
        layouts.extend([vl, pl, full]);
        sets.extend([vset, pset]);
    }
    println!("  {}", summary("libraries of the pairs", &libs));
    println!("  {}", summary("fast links", &fast));
    println!("  {}", summary("links with link-time optimisation", &lto));
    println!("  {}", summary("whole pipelines", &whole));
    println!("  {} failed", failed.len());
    for f in failed.iter().take(5) {
        println!("    {f}");
    }
    println!("  then 30 s with nothing to do, every pipeline alive:");
    watch_driver();
    // SAFETY: the pairs' objects, the shared libraries and the cache, no longer used.
    unsafe {
        for p in alive {
            gpu.device.destroy_pipeline(p, None);
        }
        for l in layouts {
            gpu.device.destroy_pipeline_layout(l, None);
        }
        for s in sets {
            gpu.device.destroy_descriptor_set_layout(s, None);
        }
        gpu.device.destroy_pipeline(input, None);
        for o in outputs.into_values() {
            gpu.device.destroy_pipeline(o, None);
        }
        gpu.device.destroy_pipeline_cache(cache, None);
    }
    Ok(())
}

/// `pixels`: whole, fast-linked, optimised and shader objects compared bit for bit, then the
/// driver's 30 s.
fn pixels(gpu: &Gpu, translations: &[(Translation, bool)], entry: &CStr, limit: usize) -> R<()> {
    let candidates: Vec<Shader> = translations.iter().filter(|(t, _)| t.stage == Stage::Pixel).filter_map(|(t, forward)| prepare(t, entry, *forward).ok()).collect();
    let mut reasons: BTreeMap<String, usize> = BTreeMap::new();
    for s in &candidates {
        let why = s.why.clone().or_else(|| s.interface.targets.is_empty().then(|| "no target".to_string()));
        *reasons.entry(why.unwrap_or_else(|| "drawable".to_string())).or_default() += 1;
    }
    let shaders: Vec<&Shader> = candidates.iter().filter(|s| s.why.is_none() && !s.interface.targets.is_empty()).take(limit.min(DRAWN)).collect();
    println!("pixels       {} pixel programs drawn, of {}: {}", shaders.len(), candidates.len(), reasons.iter().map(|(r, n)| format!("{r} {n}")).collect::<Vec<_>>().join(", "));
    let usage = vk::BufferUsageFlags::UNIFORM_BUFFER;
    let mut banks = Vec::new();
    for i in 0..16 {
        banks.push(gpu.filled(&random_words(0x100 + i, 0x4000), usage)?);
    }
    let swizzles = gpu.filled(&random_words(0x999, 0x4000), usage)?;
    let mut textures = Vec::new();
    for i in 0..8 {
        textures.push(gpu.texture(0x200 + i)?);
    }
    // SAFETY: a sampler and a descriptor pool, destroyed at the end.
    let (sampler, pool) = unsafe {
        let sampler = gpu.device.create_sampler(&vk::SamplerCreateInfo::default().mag_filter(vk::Filter::LINEAR).min_filter(vk::Filter::LINEAR).address_mode_u(vk::SamplerAddressMode::REPEAT).address_mode_v(vk::SamplerAddressMode::REPEAT).address_mode_w(vk::SamplerAddressMode::REPEAT).max_lod(0.0), None).map_err(|e| format!("sampler: {e}"))?;
        let sizes = [vk::DescriptorPoolSize { ty: vk::DescriptorType::UNIFORM_BUFFER, descriptor_count: 8192 }, vk::DescriptorPoolSize { ty: vk::DescriptorType::SAMPLED_IMAGE, descriptor_count: 16384 }, vk::DescriptorPoolSize { ty: vk::DescriptorType::SAMPLER, descriptor_count: 16384 }];
        let pool = gpu.device.create_descriptor_pool(&vk::DescriptorPoolCreateInfo::default().max_sets(1024).pool_sizes(&sizes), None).map_err(|e| format!("descriptor pool: {e}"))?;
        (sampler, pool)
    };
    let cache = gpu.pipeline_cache()?;
    let retain = vk::PipelineCreateFlags::LIBRARY_KHR | vk::PipelineCreateFlags::RETAIN_LINK_TIME_OPTIMIZATION_INFO_EXT;
    let null = vk::PipelineLayout::null();
    let input = gpu.pipeline(Some(vk::GraphicsPipelineLibraryFlagsEXT::VERTEX_INPUT_INTERFACE), &[], entry, null, 0, retain, cache).map_err(|e| format!("vertex input library: {e}"))?;
    let mut outputs: HashMap<usize, vk::Pipeline> = HashMap::new();
    let (mut same, mut differ, mut objects_ms) = ([0usize; 3], Vec::new(), Vec::new());
    let mut kept: Vec<Kept> = Vec::new();
    let mut destroyed: Vec<vk::Pipeline> = Vec::new();
    let mut layouts: Vec<(vk::PipelineLayout, [vk::DescriptorSetLayout; 2])> = Vec::new();
    for &ps in &shaders {
        let targets = ps.interface.targets.keys().max().map_or(0, |t| usize::from(*t) + 1);
        let output = match outputs.get(&targets) {
            Some(o) => *o,
            None => {
                let o = gpu.pipeline(Some(vk::GraphicsPipelineLibraryFlagsEXT::FRAGMENT_OUTPUT_INTERFACE), &[], entry, null, targets, retain, cache).map_err(|e| format!("output library: {e}"))?;
                outputs.insert(targets, o);
                o
            }
        };
        let vs = own(Stage::Vertex, synthetic_vertex(&ps.interface.varyings, entry.to_str().unwrap_or("main")), entry)?;
        let (vset, pset) = (gpu.set_layout(None)?, gpu.set_layout(Some(ps))?);
        let nullset = vk::DescriptorSetLayout::null();
        let (vl, pl, full) = (gpu.layout(&[vset, nullset], true)?, gpu.layout(&[nullset, pset], true)?, gpu.layout(&[vset, pset], true)?);
        let plain = gpu.layout(&[vset, pset], false)?;
        layouts.push((full, [vset, pset]));
        // The pixel program's set: banks, swizzles, the textures and the sampler.
        let set = {
            let layouts = [pset];
            // SAFETY: a set of a pool with room for it.
            let set = unsafe { gpu.device.allocate_descriptor_sets(&vk::DescriptorSetAllocateInfo::default().descriptor_pool(pool).set_layouts(&layouts)) }.map_err(|e| format!("descriptor set: {e}"))?[0];
            let buffers: Vec<vk::DescriptorBufferInfo> = ps.bindings.iter().map(|&(b, _)| vk::DescriptorBufferInfo { buffer: if b == shader::SWIZZLE_BINDING { swizzles.buffer } else { banks[b as usize % 16].buffer }, offset: 0, range: 0x10000 }).collect();
            let images: Vec<vk::DescriptorImageInfo> = ps.bindings.iter().map(|&(b, _)| vk::DescriptorImageInfo { sampler, image_view: textures[(b as usize / 2) % 8].view, image_layout: vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL }).collect();
            let writes: Vec<vk::WriteDescriptorSet> = ps
                .bindings
                .iter()
                .enumerate()
                .map(|(i, &(b, k))| {
                    let w = vk::WriteDescriptorSet::default().dst_set(set).dst_binding(b).descriptor_type(k.descriptor());
                    if k == Kind::Uniform { w.buffer_info(std::slice::from_ref(&buffers[i])) } else { w.image_info(std::slice::from_ref(&images[i])) }
                })
                .collect();
            // SAFETY: writes of valid descriptors into a set no command uses yet.
            unsafe { gpu.device.update_descriptor_sets(&writes, &[]) };
            set
        };
        let whole = gpu.pipeline(None, &[&vs, ps], entry, full, targets, vk::PipelineCreateFlags::empty(), cache);
        let vlib = gpu.pipeline(Some(vk::GraphicsPipelineLibraryFlagsEXT::PRE_RASTERIZATION_SHADERS), &[&vs], entry, vl, targets, retain, cache);
        let plib = gpu.pipeline(Some(vk::GraphicsPipelineLibraryFlagsEXT::FRAGMENT_SHADER), &[ps], entry, pl, targets, retain, cache);
        let (Ok(whole), Ok(vlib), Ok(plib)) = (whole, vlib, plib) else {
            differ.push(format!("{:016x}: not made", ps.key));
            continue;
        };
        let parts = [input, vlib, plib, output];
        let (Ok(fast), Ok(lto)) = (gpu.link(&parts, full, false, cache), gpu.link(&parts, full, true, cache)) else {
            differ.push(format!("{:016x}: not linked", ps.key));
            continue;
        };
        // Shader objects, unlinked, with the pipeline's set layouts.
        let sets = [vset, pset];
        let code = |s: &Shader| -> Vec<u8> { s.words.iter().flat_map(|w| w.to_le_bytes()).collect() };
        let (vcode, pcode) = (code(&vs), code(ps));
        let infos = [
            vk::ShaderCreateInfoEXT::default().stage(vk::ShaderStageFlags::VERTEX).next_stage(vk::ShaderStageFlags::FRAGMENT).code_type(vk::ShaderCodeTypeEXT::SPIRV).code(&vcode).name(entry).set_layouts(&sets),
            vk::ShaderCreateInfoEXT::default().stage(vk::ShaderStageFlags::FRAGMENT).code_type(vk::ShaderCodeTypeEXT::SPIRV).code(&pcode).name(entry).set_layouts(&sets),
        ];
        let since = Instant::now();
        // SAFETY: SPIR-V from naga's writer and valid set layouts; the objects are destroyed below.
        let objects = unsafe { gpu.objects.create_shaders(&infos, None) };
        objects_ms.push(since.elapsed().as_secs_f64() * 1e3);
        let reference = gpu.draw(With::Pipeline(whole), full, Some(set), targets, None)?;
        let mut drawn = vec![gpu.draw(With::Pipeline(fast), full, Some(set), targets, None)?, gpu.draw(With::Pipeline(lto), full, Some(set), targets, None)?];
        match &objects {
            Ok(o) => drawn.push(gpu.draw(With::Objects(o[0], o[1]), plain, Some(set), targets, None)?),
            Err((_, e)) => differ.push(format!("{:016x}: no shader objects: {e}", ps.key)),
        }
        for (i, bytes) in drawn.iter().enumerate() {
            if *bytes == reference {
                same[i] += 1;
            } else {
                let n = bytes.iter().zip(&reference).filter(|(a, b)| a != b).count();
                differ.push(format!("{:016x}: {} differs from whole in {n} bytes", ps.key, ["fast link", "link-time optimisation", "shader objects"][i]));
            }
        }
        // The first programs' libraries, linked again for the driver's 30 s.
        if kept.len() < KEPT {
            kept.push(Kept { vlib, plib, output, layout: full, set, targets, earlier: drawn[0].clone(), key: ps.key });
            destroyed.extend([whole, fast, lto]);
        } else {
            destroyed.extend([whole, fast, lto, vlib, plib]);
        }
        // SAFETY: shader objects no submitted work uses any more.
        unsafe {
            let made = match objects {
                Ok(o) => o,
                Err((o, _)) => o,
            };
            for o in made.into_iter().filter(|o| *o != vk::ShaderEXT::null()) {
                gpu.objects.destroy_shader(o, None);
            }
            gpu.device.destroy_pipeline_layout(vl, None);
            gpu.device.destroy_pipeline_layout(pl, None);
            gpu.device.destroy_pipeline_layout(plain, None);
        }
    }
    let n = shaders.len();
    println!("  same pixels as the whole pipeline: fast link {}/{n}, link-time optimisation {}/{n}, shader objects {}/{n}", same[0], same[1], same[2]);
    println!("  {}", summary("shader objects (vertex and pixel, unlinked)", &objects_ms));
    for d in differ.iter().take(10) {
        println!("    {d}");
    }
    background(gpu, &kept, input, cache)?;
    // SAFETY: everything this part made, no submitted work using it any more.
    unsafe {
        for p in destroyed.into_iter().chain(kept.iter().flat_map(|k| [k.vlib, k.plib])).chain([input]).chain(outputs.into_values()) {
            gpu.device.destroy_pipeline(p, None);
        }
        for (l, sets) in layouts {
            gpu.device.destroy_pipeline_layout(l, None);
            for s in sets {
                gpu.device.destroy_descriptor_set_layout(s, None);
            }
        }
        gpu.device.destroy_descriptor_pool(pool, None);
        gpu.device.destroy_sampler(sampler, None);
        gpu.device.destroy_pipeline_cache(cache, None);
    }
    for b in banks.iter().chain([&swizzles]) {
        gpu.free_buffer(b);
    }
    for t in &textures {
        gpu.free_image(t);
    }
    Ok(())
}

/// 30 s with nothing to do: the process's CPU time (the probe's threads sleep, so it is the
/// driver's), threads and committed memory, every 5 s.
fn watch_driver() {
    let (mut cpu, start) = (host::cpu(), Instant::now());
    println!("  0 s: {} threads, {:.0} MB committed", host::threads(), host::committed() as f64 / 1e6);
    for _ in 0..6 {
        std::thread::sleep(Duration::from_secs(5));
        let now = host::cpu();
        println!("  {} s: {:.0} ms of CPU in the last 5 s, {} threads, {:.0} MB committed", start.elapsed().as_secs(), (now - cpu).as_secs_f64() * 1e3, host::threads(), host::committed() as f64 / 1e6);
        cpu = now;
    }
}

/// A program the pixel part keeps for the driver's 30 s: its libraries, the output library and
/// layout it links with, its set and targets, what its first fast link drew, and its key.
struct Kept {
    vlib: vk::Pipeline,
    plib: vk::Pipeline,
    output: vk::Pipeline,
    layout: vk::PipelineLayout,
    set: vk::DescriptorSet,
    targets: usize,
    earlier: Vec<u8>,
    key: u64,
}

/// The kept programs linked fast again and drawn at once, then drawn over and over for 30 s, as
/// a game uses its pipelines (a driver may optimise a fast-linked pipeline only once it is used):
/// each draw compared with the first; every 5 s, the CPU time the process spent beside this
/// thread's own (the driver's), its threads and its committed memory.
fn background(gpu: &Gpu, kept: &[Kept], input: vk::Pipeline, cache: vk::PipelineCache) -> R<()> {
    let mut linked = Vec::new();
    for k in kept {
        let p = gpu.link(&[input, k.vlib, k.plib, k.output], k.layout, false, cache).map_err(|e| format!("{:016x}: link: {e}", k.key))?;
        let now = gpu.draw(With::Pipeline(p), k.layout, Some(k.set), k.targets, None)?;
        linked.push((p, now));
    }
    println!("background   {} pipelines linked fast again, then drawn over and over for 30 s", linked.len());
    let own = || gyroship_runtime::profile::own_cpu_time().unwrap_or_default();
    let (start, mut cpu, mut mine, mut report) = (Instant::now(), host::cpu(), own(), 5);
    println!("  0 s: {} threads, {:.0} MB committed", host::threads(), host::committed() as f64 / 1e6);
    let (mut draws, mut changed, mut first) = (0usize, BTreeMap::new(), None);
    while start.elapsed() < Duration::from_secs(30) {
        for ((p, now), k) in linked.iter().zip(kept) {
            draws += 1;
            if gpu.draw(With::Pipeline(*p), k.layout, Some(k.set), k.targets, None)? != *now {
                *changed.entry(k.key).or_insert(0usize) += 1;
                first.get_or_insert(start.elapsed());
            }
        }
        if start.elapsed().as_secs() >= report {
            let (all, me) = (host::cpu(), own());
            println!("  {report} s: {:.0} ms of CPU beside the drawing thread in the last 5 s, {} threads, {:.0} MB committed", ((all - cpu).saturating_sub(me - mine)).as_secs_f64() * 1e3, host::threads(), host::committed() as f64 / 1e6);
            (cpu, mine, report) = (all, me, report + 5);
        }
    }
    let relinked: Vec<String> = linked.iter().zip(kept).filter(|((_, now), k)| *now != k.earlier).map(|(_, k)| format!("{:016x}", k.key)).collect();
    let list = |v: &[String]| if v.is_empty() { String::new() } else { format!(" ({})", v.join(" ")) };
    let differed: Vec<String> = changed.iter().map(|(k, n)| format!("{k:016x} x{n}")).collect();
    println!("  {draws} draws in 30 s: {} of {} pipelines gave other pixels than when just linked{}{}", changed.len(), linked.len(), list(&differed), first.map_or_else(String::new, |t| format!(", first after {:.1} s", t.as_secs_f64())));
    println!("  just linked again, against the first link: {} give other pixels{}", relinked.len(), list(&relinked));
    // SAFETY: pipelines no submitted work uses any more.
    unsafe {
        for (p, _) in linked {
            gpu.device.destroy_pipeline(p, None);
        }
    }
    Ok(())
}

/// `vertex-input`: an R32G32B32A32_UINT attribute read by a vec4<f32> input, through dynamic
/// vertex input: the bits the shader gets.
fn vertex_input(gpu: &Gpu) -> R<()> {
    let entry = c"main";
    let vs = own(Stage::Vertex, "struct Out {\n@builtin(position) pos: vec4<f32>,\n@location(0) @interpolate(flat) v: vec4<f32>,\n}\n@vertex fn main(@builtin(vertex_index) i: u32, @location(0) a: vec4<f32>) -> Out {\nvar o: Out;\nlet uv = vec2<f32>(f32((i << 1u) & 2u), f32(i & 2u));\no.pos = vec4<f32>(uv * 2.0 - 1.0, 0.5, 1.0);\no.v = a;\nreturn o;\n}\n".into(), entry)?;
    let ps = own(Stage::Pixel, "@fragment fn main(@location(0) @interpolate(flat) v: vec4<f32>) -> @location(0) vec4<f32> {\nreturn v;\n}\n".into(), entry)?;
    let (vset, pset) = (gpu.set_layout(None)?, gpu.set_layout(None)?);
    let layout = gpu.layout(&[vset, pset], true)?;
    let cache = gpu.pipeline_cache()?;
    let pipeline = gpu.pipeline(None, &[&vs, &ps], entry, layout, 1, vk::PipelineCreateFlags::empty(), cache).map_err(|e| format!("pipeline: {e}"))?;
    let input = [0x3f80_0000u32, 7, 0x4049_0fdb, u32::MAX];
    let buffer = gpu.filled(&[input; 3].concat(), vk::BufferUsageFlags::VERTEX_BUFFER)?;
    let bytes = gpu.draw(With::Pipeline(pipeline), layout, None, 1, Some((buffer.buffer, vk::Format::R32G32B32A32_UINT)));
    gpu.free_buffer(&buffer);
    // SAFETY: objects of this part, no submitted work using them any more.
    unsafe {
        gpu.device.destroy_pipeline(pipeline, None);
        gpu.device.destroy_pipeline_layout(layout, None);
        gpu.device.destroy_descriptor_set_layout(vset, None);
        gpu.device.destroy_descriptor_set_layout(pset, None);
        gpu.device.destroy_pipeline_cache(cache, None);
    }
    let bytes = bytes?;
    // The texel at the target's center.
    let at = ((SIDE / 2 * SIDE + SIDE / 2) * 16) as usize;
    let got: Vec<u32> = bytes[at..at + 16].as_chunks::<4>().0.iter().map(|b| u32::from_le_bytes(*b)).collect();
    let converted: Vec<u32> = input.iter().map(|&v| (v as f32).to_bits()).collect();
    let how = if got == input { "the bits as they are" } else if got == converted { "the integers converted to floats" } else { "neither the bits nor the integers converted" };
    println!("vertex-input attribute {input:08x?} read as vec4<f32>: {got:08x?} ({how}; legacy vertex attributes {})", gpu.legacy);
    Ok(())
}

/// The process's committed memory, CPU time and threads, from Windows.
#[cfg(windows)]
mod host {
    use std::ffi::c_void;
    use std::time::Duration;

    #[repr(C)]
    #[derive(Default)]
    struct Counters {
        cb: u32,
        faults: u32,
        peak_working_set: usize,
        working_set: usize,
        peak_paged: usize,
        paged: usize,
        peak_nonpaged: usize,
        nonpaged: usize,
        pagefile: usize,
        peak_pagefile: usize,
        private: usize,
    }

    #[repr(C)]
    #[derive(Clone, Copy, Default)]
    struct FileTime {
        low: u32,
        high: u32,
    }

    #[repr(C)]
    #[derive(Default)]
    struct ThreadEntry {
        size: u32,
        usage: u32,
        thread: u32,
        owner: u32,
        base_priority: i32,
        delta_priority: i32,
        flags: u32,
    }

    #[link(name = "kernel32")]
    unsafe extern "system" {
        fn GetCurrentProcess() -> *mut c_void;
        fn GetCurrentProcessId() -> u32;
        fn K32GetProcessMemoryInfo(process: *mut c_void, counters: *mut Counters, cb: u32) -> i32;
        fn GetProcessTimes(process: *mut c_void, created: *mut FileTime, exited: *mut FileTime, kernel: *mut FileTime, user: *mut FileTime) -> i32;
        fn CreateToolhelp32Snapshot(flags: u32, process: u32) -> *mut c_void;
        fn Thread32First(snapshot: *mut c_void, entry: *mut ThreadEntry) -> i32;
        fn Thread32Next(snapshot: *mut c_void, entry: *mut ThreadEntry) -> i32;
        fn CloseHandle(handle: *mut c_void) -> i32;
    }

    /// The process's committed (private) bytes.
    pub fn committed() -> u64 {
        let mut c = Counters { cb: size_of::<Counters>() as u32, ..Default::default() };
        // SAFETY: the process's pseudo-handle and a counters struct of the size given.
        if unsafe { K32GetProcessMemoryInfo(GetCurrentProcess(), &mut c, c.cb) } == 0 { 0 } else { c.private as u64 }
    }

    /// The process's CPU time, all its threads', user and kernel.
    pub fn cpu() -> Duration {
        let (mut created, mut exited, mut kernel, mut user) = Default::default();
        // SAFETY: the process's pseudo-handle and four FILETIMEs to fill.
        if unsafe { GetProcessTimes(GetCurrentProcess(), &mut created, &mut exited, &mut kernel, &mut user) } == 0 {
            return Duration::ZERO;
        }
        let ticks = |f: FileTime| u64::from(f.high) << 32 | u64::from(f.low);
        Duration::from_nanos((ticks(kernel) + ticks(user)) * 100)
    }

    /// The process's threads.
    pub fn threads() -> usize {
        // SAFETY: a snapshot of the system's threads (TH32CS_SNAPTHREAD), walked then closed.
        unsafe {
            let snapshot = CreateToolhelp32Snapshot(0x4, 0);
            if snapshot as isize == -1 {
                return 0;
            }
            let me = GetCurrentProcessId();
            let mut e = ThreadEntry { size: size_of::<ThreadEntry>() as u32, ..Default::default() };
            let mut n = 0;
            let mut more = Thread32First(snapshot, &mut e) != 0;
            while more {
                n += usize::from(e.owner == me);
                more = Thread32Next(snapshot, &mut e) != 0;
            }
            CloseHandle(snapshot);
            n
        }
    }
}

#[cfg(not(windows))]
mod host {
    pub fn committed() -> u64 {
        0
    }
    pub fn cpu() -> std::time::Duration {
        std::time::Duration::ZERO
    }
    pub fn threads() -> usize {
        0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The probe's own vertex program is valid WGSL and writes each varying with its pixel
    /// program's interpolation.
    #[test]
    fn a_synthetic_vertex_program_writes_what_the_pixel_program_reads() {
        let varyings = BTreeMap::from([(0, Interp::Perspective), (3, Interp::Constant), (5, Interp::ScreenLinear)]);
        let wgsl = synthetic_vertex(&varyings, "p1");
        let module = naga::front::wgsl::parse_str(&wgsl).unwrap();
        naga::valid::Validator::new(naga::valid::ValidationFlags::all(), naga::valid::Capabilities::all()).validate(&module).unwrap();
        for want in ["@location(0) @interpolate(perspective)", "@location(3) @interpolate(flat)", "@location(5) @interpolate(linear)", "fn p1("] {
            assert!(wgsl.contains(want), "{want}\n{wgsl}");
        }
    }

    /// A branch back, or an indirect jump to an unknown or earlier place, may loop; branches
    /// forward and the closing self-branch end.
    #[test]
    fn only_forward_jumps_end_whatever_the_inputs() {
        // BRA (0x947) under PT, its offset (bytes) at bits 34..82 in words, its predicate PT.
        let bra = |offset: i64| -> [u8; 16] { (0x947u128 | 7 << 12 | ((offset / 4) as u128 & ((1 << 48) - 1)) << 34 | 7 << 87).to_le_bytes() };
        let nop = [0u8; 16];
        let program = |words: &[[u8; 16]]| words.concat();
        assert!(forward_only(&program(&[bra(16), nop, bra(-16)]), None), "forward, then the closing self-branch");
        assert!(!forward_only(&program(&[nop, bra(-32), bra(-16)]), None), "back to the start");
        // BRX (0x949): its targets, from the blob, must lie past it.
        let brx = (0x949u128 | 7 << 12 | 7 << 87).to_le_bytes();
        let later = std::collections::BTreeSet::from([2]);
        assert!(forward_only(&program(&[brx, nop, nop, bra(-16)]), Some(&later)));
        assert!(!forward_only(&program(&[nop, brx, bra(-16)]), Some(&std::collections::BTreeSet::from([0]))));
        assert!(!forward_only(&program(&[brx, nop, bra(-16)]), None), "targets unknown");
    }
}
