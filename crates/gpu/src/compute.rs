//! Compute dispatches on wgpu: a QMD's program, translated to a WGSL compute
//! shader, runs over its grid with the QMD's constant banks and the compute
//! class's texture pools; the images it writes are host storage textures kept
//! by GPU address, which later draws sample instead of guest memory.
//!
//! ponytail: images written whole (SUST with all four channels), float and
//! normalized storage formats only; image contents stay on the host (never
//! written back to guest memory).

use crate::draw::{Bindings, Mem};
use crate::pushbuf::Pools;
use crate::shader::{self, Handle, IMAGE_BINDING, Options, Translation};
use crate::sass::texture::TexDim;
use crate::state::qmd::Launch;
use crate::state::resources::TexHandle;
use crate::state::textures::{self, TextureType};
use crate::surfaces::Surfaces;
use std::collections::BTreeMap;

/// The bytes of a constant bank as the translated programs declare it.
const CBUF_BYTES: usize = 0x10000;

/// The WGSL storage format of a wgpu format a compute program may write
/// (float and normalized ones: SUST stores floats). Beyond WebGPU's own,
/// they need the adapter's support (TEXTURE_ADAPTER_SPECIFIC_FORMAT_FEATURES).
fn storage_format(f: wgpu::TextureFormat) -> Option<&'static str> {
    use wgpu::TextureFormat as T;
    Some(match f {
        T::Rgba16Float => "rgba16float",
        T::Rgba32Float => "rgba32float",
        T::R32Float => "r32float",
        T::Rg32Float => "rg32float",
        T::Rgba8Unorm => "rgba8unorm",
        T::Rgba8Snorm => "rgba8snorm",
        T::R8Unorm => "r8unorm",
        T::Rg8Unorm => "rg8unorm",
        T::R16Float => "r16float",
        T::Rg16Float => "rg16float",
        T::R16Unorm => "r16unorm",
        T::Rg16Unorm => "rg16unorm",
        T::Rgba16Unorm => "rgba16unorm",
        T::Rgb10a2Unorm => "rgb10a2unorm",
        T::Rg11b10Ufloat => "rg11b10ufloat",
        _ => return None,
    })
}

/// A translated compute program for one block size and set of image formats.
pub(crate) type Compiled = Result<(Translation, wgpu::ShaderModule), String>;

impl Surfaces {
    /// Runs a dispatch; an empty grid does nothing.
    pub fn dispatch(&mut self, l: &Launch, pools: Pools, mem: Mem) -> Result<(), String> {
        if l.grid.contains(&0) || l.program == 0 {
            return Ok(());
        }
        let key = match self.cache.compute_programs.get(&l.program) {
            Some(k) => *k,
            None => {
                let k = crate::draw::read_compute_program(mem, l.program).map(|(key, code)| {
                    self.cache.code.insert(key, code);
                    key
                });
                self.cache.compute_programs.insert(l.program, k);
                k
            }
        }
        .ok_or("compute program not readable")?;
        if l.shared > self.device.limits().max_compute_workgroup_storage_size {
            return Err(format!("{:#x} bytes of shared memory", l.shared));
        }
        let mut banks = BTreeMap::new();
        for &(bank, addr, size) in &l.cbufs {
            let mut data = vec![0u8; CBUF_BYTES];
            let n = (size as usize).min(CBUF_BYTES);
            if !mem(addr, &mut data[..n]) {
                return Err("compute constant buffer not readable".into());
            }
            banks.insert(bank, data);
        }
        let word = |banks: &BTreeMap<u8, Vec<u8>>, h: &Handle| -> Option<u32> {
            let Handle::CBuf { bank, offset } = h else { return None };
            let at = usize::from(*offset);
            let w = banks.get(bank)?.get(at..at + 4)?;
            Some(u32::from_le_bytes([w[0], w[1], w[2], w[3]]))
        };
        let mut tic_of = |h: &Handle| -> Result<textures::Tic, String> {
            let handle = TexHandle::new(word(&banks, h).ok_or_else(|| format!("image handle {h:?}"))?);
            let mut header = [0u8; 32];
            if !mem(pools.headers.entry(handle.header).ok_or("image index past the pool")?, &mut header) {
                return Err("image header not readable".into());
            }
            textures::tic(&header).ok_or_else(|| "image header layout".into())
        };
        // The images and queried textures: their headers give the formats and types the translation needs.
        let first = self.compute_program(key, l, Vec::new(), BTreeMap::new())?.0;
        let mut images = Vec::new();
        for img in &first.interface.images {
            let tic = tic_of(&img.handle)?;
            let format = textures::wgpu_texture_format(&tic).ok_or_else(|| format!("image format {:?}", tic.components))?;
            let name = storage_format(format).ok_or_else(|| format!("image format {format:?} as storage"))?;
            if img.read && !self.adapter.get_texture_format_features(format).flags.contains(wgpu::TextureFormatFeatureFlags::STORAGE_READ_WRITE) {
                return Err(format!("image format {format:?} both read and written"));
            }
            images.push((tic, format, name.to_string()));
        }
        let mut queried = BTreeMap::new();
        for (slot, x) in first.interface.textures.iter().enumerate().filter(|(_, x)| x.query) {
            if tic_of(&x.handle).is_ok_and(|t| t.texture_type == TextureType::ThreeD) {
                queried.insert(slot, TexDim::D3);
            }
        }
        let (t, module) = self.compute_program(key, l, images.iter().map(|i| i.2.clone()).collect(), queried)?;
        if let Some(b) = t.interface.cbufs.iter().chain(&Self::handle_banks(&t)).find(|b| !banks.contains_key(b)) {
            return Err(format!("compute constant bank {b} not bound"));
        }

        // Host storage textures, by address: a render target's own texture when the image is
        // one (the game draws glyphs, then filters them in place), else one kept for images.
        let mut views = Vec::new();
        for (tic, format, _) in &images {
            let three_d = tic.texture_type == TextureType::ThreeD;
            let size = wgpu::Extent3d { width: tic.width, height: tic.height, depth_or_array_layers: if three_d { tic.depth } else { 1 } };
            let target = self.map.values().find(|e| {
                let t = &e.texture;
                e.surface.addr == tic.address && t.format() == *format && t.width() == size.width && t.height() == size.height && !three_d && t.usage().contains(wgpu::TextureUsages::STORAGE_BINDING)
            });
            if let Some(e) = target {
                views.push(e.texture.create_view(&wgpu::TextureViewDescriptor { dimension: Some(wgpu::TextureViewDimension::D2), array_layer_count: Some(1), ..Default::default() }));
                self.cache.images.remove(&tic.address);
                continue;
            }
            let fresh = self.cache.images.get(&tic.address).is_none_or(|x| x.size() != size || x.format() != *format);
            if fresh {
                let texture = self.device.create_texture(&wgpu::TextureDescriptor {
                    label: Some("image"),
                    size,
                    mip_level_count: 1,
                    sample_count: 1,
                    dimension: if three_d { wgpu::TextureDimension::D3 } else { wgpu::TextureDimension::D2 },
                    format: *format,
                    usage: wgpu::TextureUsages::STORAGE_BINDING | wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_SRC | wgpu::TextureUsages::COPY_DST,
                    view_formats: &[],
                });
                self.cache.images.insert(tic.address, texture);
            }
            views.push(self.cache.images[&tic.address].create_view(&Default::default()));
        }

        // Bindings: constant banks, textures and swizzles as for draws, then the images.
        let (textures, swizzles) = self.texture_slots(&t, &banks, pools.headers, pools.samplers, pools.sampler_binding, &[], mem)?;
        let cbufs: BTreeMap<u8, Vec<u8>> = banks.into_iter().filter(|(b, _)| t.interface.cbufs.contains(b)).collect();
        let bindings = Bindings { cbufs, textures, swizzles: t.interface.swizzles.then_some(swizzles) };
        let mut layout_entries = bindings.layout(wgpu::ShaderStages::COMPUTE);
        for (i, (img, (_, format, _))) in t.interface.images.iter().zip(&images).enumerate() {
            let view_dimension = match img.dim {
                crate::sass::memory::ImageDim::D3 => wgpu::TextureViewDimension::D3,
                crate::sass::memory::ImageDim::D1 => wgpu::TextureViewDimension::D1,
                _ => wgpu::TextureViewDimension::D2,
            };
            layout_entries.push(wgpu::BindGroupLayoutEntry {
                binding: IMAGE_BINDING + i as u32,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::StorageTexture { access: if img.read { wgpu::StorageTextureAccess::ReadWrite } else { wgpu::StorageTextureAccess::WriteOnly }, format: *format, view_dimension },
                count: None,
            });
        }
        let id = format!("{key:x}/{:?}/{:#x}/{layout_entries:?}", l.block, l.shared);
        let (pipeline, layout) = match self.cache.compute_pipelines.get(&id) {
            Some(p) => p.clone(),
            None => {
                let layout = self.device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor { label: None, entries: &layout_entries });
                let pipeline_layout = self.device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor { label: None, bind_group_layouts: &[Some(&layout)], ..Default::default() });
                let pipeline = self.device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                    label: Some(&format!("{key:016x}")),
                    layout: Some(&pipeline_layout),
                    module: &module,
                    entry_point: Some("main"),
                    compilation_options: Default::default(),
                    cache: None,
                });
                self.cache.compute_pipelines.insert(id, (pipeline.clone(), layout.clone()));
                (pipeline, layout)
            }
        };
        let uniform = |device: &wgpu::Device, contents: &[u8]| wgpu::util::DeviceExt::create_buffer_init(device, &wgpu::util::BufferInitDescriptor { label: None, contents, usage: wgpu::BufferUsages::UNIFORM });
        let mut buffers: Vec<(u32, wgpu::Buffer)> = bindings.cbufs.iter().map(|(b, data)| (u32::from(*b), uniform(&self.device, data))).collect();
        if let Some(codes) = &bindings.swizzles {
            let bytes: Vec<u8> = codes.iter().flat_map(|c| c.to_le_bytes()).collect();
            buffers.push((shader::SWIZZLE_BINDING, uniform(&self.device, &bytes)));
        }
        let mut entries: Vec<wgpu::BindGroupEntry> = buffers.iter().map(|(b, buffer)| wgpu::BindGroupEntry { binding: *b, resource: buffer.as_entire_binding() }).collect();
        for (slot, s) in bindings.textures.iter().enumerate() {
            entries.push(wgpu::BindGroupEntry { binding: 32 + 2 * slot as u32, resource: wgpu::BindingResource::TextureView(&s.view) });
            if let Some(sampler) = &s.sampler {
                entries.push(wgpu::BindGroupEntry { binding: 33 + 2 * slot as u32, resource: wgpu::BindingResource::Sampler(sampler) });
            }
        }
        for (i, view) in views.iter().enumerate() {
            entries.push(wgpu::BindGroupEntry { binding: IMAGE_BINDING + i as u32, resource: wgpu::BindingResource::TextureView(view) });
        }
        let group = self.device.create_bind_group(&wgpu::BindGroupDescriptor { label: None, layout: &layout, entries: &entries });
        let mut encoder = self.device.create_command_encoder(&Default::default());
        {
            let mut pass = encoder.begin_compute_pass(&Default::default());
            pass.set_pipeline(&pipeline);
            pass.set_bind_group(0, &group, &[]);
            pass.dispatch_workgroups(l.grid[0], l.grid[1], l.grid[2]);
        }
        self.queue.submit([encoder.finish()]);
        Ok(())
    }

    /// The translation and module of a compute program for a launch's block
    /// and shared memory sizes, image formats and queried texture types.
    fn compute_program(&mut self, key: u64, l: &Launch, images: Vec<String>, queried: BTreeMap<usize, TexDim>) -> Compiled {
        let id = format!("{key:x}/{:?}/{:#x}/{images:?}/{queried:?}", l.block, l.shared);
        if let Some(c) = self.cache.compute.get(&id) {
            return c.clone();
        }
        let options = Options { workgroup: l.block, shared: l.shared, images, queried, ..Default::default() };
        let made = shader::translate_with(&self.cache.code[&key], &options).map_err(|e| format!("compute program {key:016x}: {e}")).and_then(|t| {
            if let Err(e) = shader::validate(&t.wgsl) {
                self.error(format!("compute program {key:016x}: {e}"));
                return Err(format!("compute program {key:016x}: invalid WGSL"));
            }
            let module = self.device.create_shader_module(wgpu::ShaderModuleDescriptor { label: Some(&format!("{key:016x}")), source: wgpu::ShaderSource::Wgsl(t.wgsl.as_str().into()) });
            Ok((t, module))
        });
        self.cache.compute.insert(id, made.clone());
        made
    }
}
