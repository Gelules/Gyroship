//! The GPU device on Vulkan, opened through wgpu-hal (D110): the device `request_device`
//! would open, plus what the raw Vulkan path needs, when the GPU has it.

use ash::vk;
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
        let add = callback(|args| {
            if extras.pipeline_library {
                args.extensions.extend([ash::khr::pipeline_library::NAME, ash::ext::graphics_pipeline_library::NAME]);
                *args.create_info = args.create_info.push_next(&mut library);
            }
            if extras.dynamic_rendering {
                *args.create_info = args.create_info.push_next(&mut rendering);
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
    let mut fast = vk::PhysicalDeviceGraphicsPipelineLibraryPropertiesEXT::default();
    let mut features = vk::PhysicalDeviceFeatures2::default();
    if library {
        features = features.push_next(&mut gpl);
    }
    if core13 {
        features = features.push_next(&mut rendering);
    }
    unsafe { instance.get_physical_device_features2(device, &mut features) };
    if library {
        unsafe { instance.get_physical_device_properties2(device, &mut vk::PhysicalDeviceProperties2::default().push_next(&mut fast)) };
    }
    Extras {
        pipeline_library: gpl.graphics_pipeline_library == vk::TRUE,
        fast_linking: fast.graphics_pipeline_library_fast_linking == vk::TRUE,
        dynamic_rendering: rendering.dynamic_rendering == vk::TRUE,
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
}
