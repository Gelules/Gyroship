//! Compute dispatches on wgpu: a QMD's program, translated to a WGSL compute
//! shader, runs over its grid with the QMD's constant banks; the images it
//! writes are host storage textures kept by GPU address, which later draws
//! sample instead of guest memory.
//!
//! ponytail: images written whole (SUST with all four channels), float
//! storage formats only, no sampled textures or shared memory yet; image
//! contents stay on the host (never written back to guest memory).

use crate::draw::Mem;
use crate::shader::{self, Handle, IMAGE_BINDING, Options, Translation};
use crate::state::qmd::Launch;
use crate::state::resources::{Pool, TexHandle};
use crate::state::textures::{self, TextureType};
use crate::surfaces::Surfaces;
use std::collections::BTreeMap;

/// The bytes of a constant bank as the translated programs declare it.
const CBUF_BYTES: usize = 0x10000;

/// The WGSL storage format of a wgpu format a compute program may write.
fn storage_format(f: wgpu::TextureFormat) -> Option<&'static str> {
    use wgpu::TextureFormat as T;
    Some(match f {
        T::Rgba16Float => "rgba16float",
        T::Rgba32Float => "rgba32float",
        T::R32Float => "r32float",
        T::Rg32Float => "rg32float",
        T::Rgba8Unorm => "rgba8unorm",
        T::Rgba8Snorm => "rgba8snorm",
        _ => return None,
    })
}

/// A compiled compute program for one block size and set of image formats.
type Compiled = Result<(Translation, wgpu::ComputePipeline, wgpu::BindGroupLayout), String>;

impl Surfaces {
    /// Runs a dispatch; an empty grid does nothing.
    pub fn dispatch(&mut self, l: &Launch, headers: Pool, mem: Mem) -> Result<(), String> {
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
        if l.shared != 0 {
            return Err("compute program with shared memory".into());
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
        // The images: their headers give the formats the translation needs.
        let first = self.compute_program(key, l.block, Vec::new())?.0;
        let mut images = Vec::new();
        for img in &first.interface.images {
            let handle = TexHandle::new(word(&banks, &img.handle).ok_or_else(|| format!("image handle {:?}", img.handle))?);
            let mut header = [0u8; 32];
            if !mem(headers.entry(handle.header).ok_or("image index past the pool")?, &mut header) {
                return Err("image header not readable".into());
            }
            let tic = textures::tic(&header).ok_or("image header layout")?;
            let format = textures::wgpu_texture_format(&tic).ok_or_else(|| format!("image format {:?}", tic.components))?;
            let name = storage_format(format).ok_or_else(|| format!("image format {format:?} as storage"))?;
            images.push((tic, format, name.to_string()));
        }
        let (t, pipeline, layout) = self.compute_program(key, l.block, images.iter().map(|i| i.2.clone()).collect())?;

        // Host storage textures, by address.
        let mut views = Vec::new();
        for (tic, format, _) in &images {
            let three_d = tic.texture_type == TextureType::ThreeD;
            let size = wgpu::Extent3d { width: tic.width, height: tic.height, depth_or_array_layers: if three_d { tic.depth } else { 1 } };
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
        let buffers: Vec<(u32, wgpu::Buffer)> = banks
            .iter()
            .filter(|(b, _)| t.interface.cbufs.contains(b))
            .map(|(b, data)| (u32::from(*b), wgpu::util::DeviceExt::create_buffer_init(&self.device, &wgpu::util::BufferInitDescriptor { label: None, contents: data, usage: wgpu::BufferUsages::UNIFORM })))
            .collect();
        if let Some(b) = t.interface.cbufs.iter().find(|b| !banks.contains_key(b)) {
            return Err(format!("compute constant bank {b} not bound"));
        }
        let mut entries: Vec<wgpu::BindGroupEntry> = buffers.iter().map(|(b, buffer)| wgpu::BindGroupEntry { binding: *b, resource: buffer.as_entire_binding() }).collect();
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

    /// The translation, pipeline and bind group layout of a compute program
    /// for a block size and image formats.
    fn compute_program(&mut self, key: u64, block: [u32; 3], images: Vec<String>) -> Compiled {
        let id = format!("{key:x}/{block:?}/{images:?}");
        if let Some(c) = self.cache.compute.get(&id) {
            return c.clone();
        }
        let options = Options { workgroup: block, images: images.clone(), ..Default::default() };
        let made = shader::translate_with(&self.cache.code[&key], &options).map_err(|e| format!("compute program {key:016x}: {e}")).and_then(|t| {
            if let Err(e) = shader::validate(&t.wgsl) {
                self.error(format!("compute program {key:016x}: {e}"));
                return Err(format!("compute program {key:016x}: invalid WGSL"));
            }
            if !t.interface.textures.is_empty() {
                return Err("compute program sampling textures".into());
            }
            let module = self.device.create_shader_module(wgpu::ShaderModuleDescriptor { label: Some(&format!("{key:016x}")), source: wgpu::ShaderSource::Wgsl(t.wgsl.as_str().into()) });
            let visibility = wgpu::ShaderStages::COMPUTE;
            let mut entries: Vec<wgpu::BindGroupLayoutEntry> = t
                .interface
                .cbufs
                .iter()
                .map(|b| wgpu::BindGroupLayoutEntry {
                    binding: u32::from(*b),
                    visibility,
                    ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Uniform, has_dynamic_offset: false, min_binding_size: wgpu::BufferSize::new(CBUF_BYTES as u64) },
                    count: None,
                })
                .collect();
            for (i, img) in t.interface.images.iter().enumerate() {
                let format = match images.get(i).map(String::as_str) {
                    Some("rgba32float") => wgpu::TextureFormat::Rgba32Float,
                    Some("r32float") => wgpu::TextureFormat::R32Float,
                    Some("rg32float") => wgpu::TextureFormat::Rg32Float,
                    Some("rgba8unorm") => wgpu::TextureFormat::Rgba8Unorm,
                    Some("rgba8snorm") => wgpu::TextureFormat::Rgba8Snorm,
                    _ => wgpu::TextureFormat::Rgba16Float,
                };
                let view_dimension = match img.dim {
                    crate::sass::memory::ImageDim::D3 => wgpu::TextureViewDimension::D3,
                    crate::sass::memory::ImageDim::D1 => wgpu::TextureViewDimension::D1,
                    _ => wgpu::TextureViewDimension::D2,
                };
                entries.push(wgpu::BindGroupLayoutEntry { binding: IMAGE_BINDING + i as u32, visibility, ty: wgpu::BindingType::StorageTexture { access: wgpu::StorageTextureAccess::WriteOnly, format, view_dimension }, count: None });
            }
            let layout = self.device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor { label: None, entries: &entries });
            let pipeline_layout = self.device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor { label: None, bind_group_layouts: &[Some(&layout)], ..Default::default() });
            let pipeline = self.device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some(&format!("{key:016x}")),
                layout: Some(&pipeline_layout),
                module: &module,
                entry_point: Some("main"),
                compilation_options: Default::default(),
                cache: None,
            });
            Ok((t, pipeline, layout))
        });
        self.cache.compute.insert(id, made.clone());
        made
    }
}
