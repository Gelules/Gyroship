//! Compute dispatches on wgpu: a QMD's program, translated to a WGSL compute
//! shader, runs over its grid with the QMD's constant banks and the compute
//! class's texture pools; the images it writes are host storage textures kept
//! by GPU address, which later draws sample instead of guest memory.
//!
//! ponytail: images written whole (SUST with all four channels), float and
//! normalized storage formats only; image contents stay on the host (never
//! written back to guest memory).

use crate::draw::{Bank, Banks, Bindings, Mem};
use crate::pushbuf::Pools;
use crate::shader::{self, Handle, IMAGE_BINDING, Options, Translation};
use crate::sass::memory::ImageDim;
use crate::sass::texture::TexDim;
use crate::state::qmd::Launch;
use crate::state::resources::TexHandle;
use crate::state::textures::{self, TextureType};
use crate::surfaces::Surfaces;
use std::collections::BTreeMap;

/// The bytes of a constant bank as the translated programs declare it.
const CBUF_BYTES: usize = 0x10000;

/// The storage format and its WGSL name for an image of format `f` (float and
/// normalized ones: SUST stores floats). Beyond WebGPU's own, they need the
/// adapter's support (TEXTURE_ADAPTER_SPECIFIC_FORMAT_FEATURES). An sRGB image
/// is stored as its linear format, texels as they are (surface stores do not
/// convert); views that sample it with its header still decode sRGB.
fn storage_format(f: wgpu::TextureFormat) -> Option<(wgpu::TextureFormat, &'static str)> {
    use wgpu::TextureFormat as T;
    if f == T::Rgba8UnormSrgb {
        return Some((T::Rgba8Unorm, "rgba8unorm"));
    }
    Some((f, match f {
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
        T::R32Uint => "r32uint",
        T::R32Sint => "r32sint",
        T::Rg32Uint => "rg32uint",
        T::Rg32Sint => "rg32sint",
        T::Rgba32Uint => "rgba32uint",
        T::Rgba32Sint => "rgba32sint",
        _ => return None,
    }))
}

/// A translated compute program for one block size and set of image formats.
pub(crate) type Compiled = Result<std::sync::Arc<(Translation, wgpu::ShaderModule)>, String>;

/// What an image slot is bound to: a storage view of its texture, or (`Image::buffer`)
/// a buffer the texture is copied into before the dispatch, and back out of after it.
enum ImageBinding {
    View(wgpu::TextureView),
    Buffer(wgpu::Buffer, wgpu::Texture, wgpu::TexelCopyBufferLayout),
}

impl Surfaces {
    /// Runs a dispatch; an empty grid does nothing.
    pub fn dispatch(&mut self, l: &Launch, pools: Pools, mem: Mem) -> Result<(), String> {
        self.flush_big_uploads();
        self.textures_read.clear();
        self.drain_invalidations();
        // A dispatch may write the images: their float twins are stale.
        self.cache.twins.clear();
        // The dispatch may make, drop or write images: slots resolved to one are stale.
        self.memo.forget_slots();
        if l.grid.contains(&0) || l.program == 0 {
            return Ok(());
        }
        let key = match self.cache.compute_programs.get(&l.program) {
            Some(&k) => k,
            None => {
                let (k, code) = crate::draw::read_compute_program(mem, l.program).ok_or_else(|| format!("compute program at {:#x} not readable", l.program))?;
                self.cache.code.insert(k, code);
                self.cache.compute_programs.insert(l.program, k);
                k
            }
        };
        if l.shared > self.device.limits().max_compute_workgroup_storage_size {
            return Err(format!("{:#x} bytes of shared memory", l.shared));
        }
        let mut banks = Banks::new();
        for &(bank, addr, size) in &l.cbufs {
            let read = self.read_guest(crate::surfaces::Read::ComputeBank, addr, (size as usize).min(CBUF_BYTES), mem).ok_or_else(|| format!("compute constant bank {bank} at {addr:#x} not readable"))?;
            // In bank order; a bank given twice keeps its last binding.
            match banks.binary_search_by_key(&bank, |(k, _)| *k) {
                Ok(i) => banks[i].1 = read,
                Err(i) => banks.insert(i, (bank, read)),
            }
        }
        let word = |banks: &[(u8, Bank)], h: &Handle| -> Option<u32> {
            let Handle::CBuf { bank, offset } = h else { return None };
            let at = usize::from(*offset);
            let w = crate::draw::bank(banks, *bank)?.1.get(at..at + 4)?;
            Some(u32::from_le_bytes([w[0], w[1], w[2], w[3]]))
        };
        let mut tic_of = |h: &Handle| -> Result<textures::Tic, String> {
            let handle = TexHandle::new(word(&banks[..], h).ok_or_else(|| format!("image handle {h:?}"))?);
            let mut header = [0u8; 32];
            if !mem(pools.headers.entry(handle.header).ok_or("image index past the pool")?, &mut header) {
                return Err("image header not readable".into());
            }
            textures::tic(&header).ok_or_else(|| "image header layout".into())
        };
        // The images and queried textures: their headers give the formats and types the translation needs.
        let first = self.compute_program(key, l, Vec::new(), BTreeMap::new())?;
        let mut images = Vec::new();
        for img in &first.0.interface.images {
            let tic = tic_of(&img.handle)?;
            let header = textures::wgpu_texture_format(&tic).ok_or_else(|| format!("image format {:?}", tic.components))?;
            let (format, name) = storage_format(header).ok_or_else(|| format!("image format {header:?} as storage, at {:#x} ({:?} {}x{}x{})", tic.address, tic.texture_type, tic.width, tic.height, textures::layers(&tic)))?;
            if img.read && !self.adapter.get_texture_format_features(format).flags.contains(wgpu::TextureFormatFeatureFlags::STORAGE_READ_WRITE) {
                return Err(format!("image format {format:?} both read and written"));
            }
            if img.atomic && !img.buffer && !self.device.features().contains(wgpu::Features::TEXTURE_ATOMIC) {
                return Err("image atomics need TEXTURE_ATOMIC".into());
            }
            images.push((tic, format, name.to_string(), header));
        }
        let mut queried = BTreeMap::new();
        for (slot, x) in first.0.interface.textures.iter().enumerate().filter(|(_, x)| x.query) {
            if tic_of(&x.handle).is_ok_and(|t| t.texture_type == TextureType::ThreeD) {
                queried.insert(slot, TexDim::D3);
            }
        }
        let compiled = self.compute_program(key, l, images.iter().map(|i| i.2.clone()).collect(), queried)?;
        let (t, module) = (&compiled.0, &compiled.1);
        if let Some(b) = t.interface.cbufs.iter().chain(&t.interface.handle_banks).find(|b| crate::draw::bank(&banks, **b).is_none()) {
            return Err(format!("compute constant bank {b} not bound"));
        }

        // Host storage textures, by address: a render target's own texture when the image is
        // one (the game draws glyphs, then filters them in place), else one kept for images.
        // An array image (SULD/SUST .D2Array) is a 2D texture of the header's layers.
        let mut views = Vec::new();
        let mut written = Vec::new();
        for ((tic, format, _, header), img) in images.iter().zip(&t.interface.images) {
            let three_d = tic.texture_type == TextureType::ThreeD;
            let array = img.dim == ImageDim::D2Array;
            let size = wgpu::Extent3d { width: tic.width, height: tic.height, depth_or_array_layers: if three_d || array { tic.depth.max(1) } else { 1 } };
            let target = self.map.values().find(|e| {
                let t = &e.texture;
                e.surface.addr == tic.address && t.format() == *format && t.width() == size.width && t.height() == size.height && !three_d && !array && !img.atomic && t.usage().contains(wgpu::TextureUsages::STORAGE_BINDING)
            });
            if let Some(e) = target {
                views.push(ImageBinding::View(e.texture.create_view(&wgpu::TextureViewDescriptor { dimension: Some(wgpu::TextureViewDimension::D2), array_layer_count: Some(1), ..Default::default() })));
                // Stamped once the dispatch is recorded: its own texture reads (and a
                // dispatch that fails before) still see the targets as they were.
                written.push((e.surface.addr, e.surface.format));
                self.cache.images.remove(&tic.address);
                continue;
            }
            let fresh = self.cache.images.get(&tic.address).is_none_or(|x| x.size() != size || x.format() != *format);
            if fresh {
                // r32 images may be reduced into: every one can be bound for atomics.
                let r32 = matches!(format, wgpu::TextureFormat::R32Uint | wgpu::TextureFormat::R32Sint) && self.device.features().contains(wgpu::Features::TEXTURE_ATOMIC);
                let atomic = if r32 { wgpu::TextureUsages::STORAGE_ATOMIC } else { wgpu::TextureUsages::empty() };
                let texture = self.device.create_texture(&wgpu::TextureDescriptor {
                    label: Some("image"),
                    size,
                    mip_level_count: 1,
                    sample_count: 1,
                    dimension: if three_d { wgpu::TextureDimension::D3 } else { wgpu::TextureDimension::D2 },
                    format: *format,
                    usage: wgpu::TextureUsages::STORAGE_BINDING | wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_SRC | wgpu::TextureUsages::COPY_DST | atomic,
                    view_formats: if header == format { &[] } else { std::slice::from_ref(header) },
                });
                self.cache.images.insert(tic.address, texture);
            }
            let texture = &self.cache.images[&tic.address];
            if img.buffer {
                // After the 256-byte header (`shader::IMAGE_BUFFER`), rows padded as a copy pads them.
                let row = (size.width * 4).next_multiple_of(wgpu::COPY_BYTES_PER_ROW_ALIGNMENT);
                let bytes = 256 + u64::from(row) * u64::from(size.height) * u64::from(size.depth_or_array_layers);
                if bytes > self.device.limits().max_storage_buffer_binding_size {
                    return Err(format!("image atomics on {bytes} bytes: past the storage buffer limit"));
                }
                let buffer = self.device.create_buffer(&wgpu::BufferDescriptor { label: Some("image atomics"), size: bytes, usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST, mapped_at_creation: false });
                // A new buffer: its write lands before any command that uses it.
                let header: Vec<u8> = [size.width, size.height, size.depth_or_array_layers, row / 4].iter().flat_map(|v| v.to_le_bytes()).collect();
                self.queue.write_buffer(&buffer, 0, &header);
                views.push(ImageBinding::Buffer(buffer, texture.clone(), wgpu::TexelCopyBufferLayout { offset: 256, bytes_per_row: Some(row), rows_per_image: Some(size.height) }));
                continue;
            }
            let dimension = array.then_some(wgpu::TextureViewDimension::D2Array);
            views.push(ImageBinding::View(texture.create_view(&wgpu::TextureViewDescriptor { dimension, ..Default::default() })));
        }

        // Global memory (LDG): each pointer's buffer, read from guest memory. The constant
        // bank holds a storage-buffer descriptor there: the 64-bit address, then the size.
        // ponytail: read again at every dispatch; cache by address if it shows in profiles.
        let mut globals = Vec::new();
        for &(bank, at) in &t.interface.buffers {
            let word = |o: usize| crate::draw::bank(&banks, bank).and_then(|b| b.1.get(o..o + 4)).map(|w| u32::from_le_bytes([w[0], w[1], w[2], w[3]]));
            let at = usize::from(at);
            let (Some(lo), Some(hi), Some(size)) = (word(at), word(at + 4), word(at + 8)) else { return Err("global memory descriptor out of its bank".into()) };
            // The word after the pointer is not proved to be a size: at least 1 MiB is read
            // when memory allows, else that many bytes.
            // ponytail: a window, not the buffer's real extent; bound it by the program's
            // offsets if a buffer ever runs past it.
            let address = u64::from(hi) << 32 | u64::from(lo);
            let mut data = Vec::new();
            // The device's own bounds (DX12's are lower than Vulkan's here).
            let limits = self.device.limits();
            let max = (1u64 << 30).min(limits.max_storage_buffer_binding_size).min(limits.max_buffer_size);
            for bytes in [size.max(1 << 20), size] {
                if bytes == 0 || u64::from(bytes.next_multiple_of(4)) > max {
                    continue;
                }
                data = vec![0u8; bytes.next_multiple_of(4) as usize];
                if mem(address, &mut data) {
                    self.note_read(crate::surfaces::Read::Global, data.len(), false);
                    break;
                }
                data.clear();
            }
            if data.is_empty() {
                return Err(format!("global memory at c[{bank}][{at:#x}] ({address:#x}, size word {size:#x}) not readable"));
            }
            // Not create_buffer_init: an invalid buffer must come out as a wgpu error, not a panic.
            let buffer = self.device.create_buffer(&wgpu::BufferDescriptor { label: Some("global"), size: data.len() as u64, usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST, mapped_at_creation: false });
            self.queue.write_buffer(&buffer, 0, &data);
            globals.push(buffer);
        }

        // Bindings: constant banks, textures and swizzles as for draws, then the images.
        let (textures, swizzles, set) = self.texture_slots(t, &banks, pools.headers, pools.samplers, pools.sampler_binding, &[], mem)?;
        let cbufs = banks.iter().filter(|(b, _)| t.interface.cbufs.contains(b)).map(|&(b, (id, _))| (b, id)).collect();
        let bindings = Bindings { cbufs, textures, swizzles: t.interface.swizzles.then_some(swizzles), set, sizes: &t.interface.cbuf_sizes };
        let mut layout_entries = bindings.layout(wgpu::ShaderStages::COMPUTE);
        for (i, (img, (_, format, _, _))) in t.interface.images.iter().zip(&images).enumerate() {
            let view_dimension = match img.dim {
                ImageDim::D3 => wgpu::TextureViewDimension::D3,
                ImageDim::D1 => wgpu::TextureViewDimension::D1,
                ImageDim::D2Array => wgpu::TextureViewDimension::D2Array,
                _ => wgpu::TextureViewDimension::D2,
            };
            let access = if img.atomic {
                wgpu::StorageTextureAccess::Atomic
            } else if img.read {
                wgpu::StorageTextureAccess::ReadWrite
            } else {
                wgpu::StorageTextureAccess::WriteOnly
            };
            let ty = if img.buffer {
                wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: false }, has_dynamic_offset: false, min_binding_size: None }
            } else {
                wgpu::BindingType::StorageTexture { access, format: *format, view_dimension }
            };
            layout_entries.push(wgpu::BindGroupLayoutEntry { binding: IMAGE_BINDING + i as u32, visibility: wgpu::ShaderStages::COMPUTE, ty, count: None });
        }
        for i in 0..globals.len() as u32 {
            layout_entries.push(wgpu::BindGroupLayoutEntry {
                binding: shader::BUFFER_BINDING + i,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: true }, has_dynamic_offset: false, min_binding_size: None },
                count: None,
            });
        }
        let id = format!("{key:x}/{:?}/{:#x}/{layout_entries:?}", l.block, l.shared);
        // Made at once, even in play mode (D42 skips draws only): a dispatch skipped while
        // its pipeline compiles can lose state the game computes once (run 45c: every
        // presented image black for good).
        if !self.cache.compute_pipelines.contains_key(&id) {
            let since = std::time::Instant::now();
            let layout = self.device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor { label: None, entries: &layout_entries });
            let pipeline_layout = self.device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor { label: None, bind_group_layouts: &[Some(&layout)], ..Default::default() });
            let pipeline = self.device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some(&format!("{key:016x}")),
                layout: Some(&pipeline_layout),
                module,
                entry_point: Some("main"),
                compilation_options: Default::default(),
                cache: self.disk_cache.as_ref().map(|d| &d.cache),
            });
            self.cache.compute_pipelines.insert(id.clone(), (pipeline, layout));
            self.compile.pipelines += 1;
            self.compile.pipeline_time += since.elapsed();
            self.compile.made.push(format!("p{key:016x}+compute"));
        }
        let (pipeline, layout) = self.cache.compute_pipelines[&id].clone();
        // Staged, bound and recorded in order with the draws, by the recorder.
        let banks: Vec<(u8, u32, usize)> = bindings.cbufs.iter().map(|&(b, id)| (b, id, bindings.window(b))).collect();
        let (slots, swizzles, grid) = (bindings.textures.clone(), bindings.swizzles.clone(), l.grid);
        self.run(move |rec| {
            let mut buffers: Vec<(u32, (wgpu::Buffer, u64, u64))> = banks.iter().map(|&(b, id, window)| (u32::from(b), rec.stage_bank(id, window))).collect();
            if let Some(codes) = &swizzles {
                let bytes: Vec<u8> = codes.iter().flat_map(|c| c.to_le_bytes()).collect();
                buffers.push((shader::SWIZZLE_BINDING, rec.stage(&bytes)));
            }
            let mut entries: Vec<wgpu::BindGroupEntry> = buffers.iter().map(|(b, u)| wgpu::BindGroupEntry { binding: *b, resource: crate::surfaces::uniform(u) }).collect();
            for (slot, s) in slots.iter().enumerate() {
                entries.push(wgpu::BindGroupEntry { binding: 32 + 2 * slot as u32, resource: wgpu::BindingResource::TextureView(&s.view) });
                if let Some(sampler) = &s.sampler {
                    entries.push(wgpu::BindGroupEntry { binding: 33 + 2 * slot as u32, resource: wgpu::BindingResource::Sampler(sampler) });
                }
            }
            for (i, view) in views.iter().enumerate() {
                let resource = match view {
                    ImageBinding::View(v) => wgpu::BindingResource::TextureView(v),
                    ImageBinding::Buffer(b, ..) => b.as_entire_binding(),
                };
                entries.push(wgpu::BindGroupEntry { binding: IMAGE_BINDING + i as u32, resource });
            }
            for (i, buffer) in globals.iter().enumerate() {
                entries.push(wgpu::BindGroupEntry { binding: shader::BUFFER_BINDING + i as u32, resource: buffer.as_entire_binding() });
            }
            let group = rec.device.create_bind_group(&wgpu::BindGroupDescriptor { label: None, layout: &layout, entries: &entries });
            let timed = rec.timing.as_mut().and_then(|t| t.next(format!("compute {key:016x}")));
            let mut encoder = rec.encoder();
            // Images bound as buffers: their texels in before the dispatch, back out after it.
            for v in &views {
                if let ImageBinding::Buffer(b, t, layout) = v {
                    encoder.copy_texture_to_buffer(t.as_image_copy(), wgpu::TexelCopyBufferInfo { buffer: b, layout: *layout }, t.size());
                }
            }
            {
                let timestamp_writes = timed.zip(rec.timing.as_ref()).map(|(i, t)| wgpu::ComputePassTimestampWrites { query_set: &t.set, beginning_of_pass_write_index: Some(i), end_of_pass_write_index: Some(i + 1) });
                let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor { label: Some("dispatch"), timestamp_writes });
                pass.set_pipeline(&pipeline);
                pass.set_bind_group(0, &group, &[]);
                pass.dispatch_workgroups(grid[0], grid[1], grid[2]);
            }
            for v in &views {
                if let ImageBinding::Buffer(b, t, layout) = v {
                    encoder.copy_buffer_to_texture(wgpu::TexelCopyBufferInfo { buffer: b, layout: *layout }, t.as_image_copy(), t.size());
                }
            }
            rec.record(encoder);
        });
        for key in written {
            self.mark(key);
        }
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
        let made = self.translated(key, options, "compute program", format!("{key:016x}/compute"), |_| {});
        self.cache.compute.insert(id, made.clone());
        made
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::state::resources;

    /// A SASS word from (lo, value) fields, guard PT.
    fn w(op: u16, fields: &[(u32, u128)]) -> [u8; 16] {
        fields.iter().fold(u128::from(op) | 7 << 12, |v, (lo, x)| v | x << lo).to_le_bytes()
    }

    /// An exact half for the small integers used here.
    fn half(x: f32) -> u32 {
        if x == 0.0 {
            return 0;
        }
        let b = x.to_bits();
        let exp = ((b >> 23) & 0xff) as i32 - 127 + 15;
        (b >> 16) & 0x8000 | (exp as u32) << 10 | (b >> 13) & 0x3ff
    }

    fn pair(lo: f32, hi: f32) -> u32 {
        half(lo) | half(hi) << 16
    }

    /// Sets bits hi..=lo of a 32-byte texture header.
    fn put(h: &mut [u8; 32], hi: u32, lo: u32, v: u32) {
        for (i, b) in (lo..=hi).enumerate() {
            let byte = &mut h[(b / 8) as usize];
            *byte = *byte & !(1 << (b % 8)) | ((v >> i & 1) as u8) << (b % 8);
        }
    }

    /// Runs `code` (closed by EXIT and the self-branch) as one 32-lane warp with
    /// `shared` bytes of shared memory. c[0][0x80] points at `global` (its size at
    /// 0x88); c[0][0x20] is the handle of a 32 × `rows` RGBA32F image, read back as
    /// words, texel after texel. None without a GPU. `consts` fill c[0] from 0x100.
    fn run_warp(code: &[[u8; 16]], global: Vec<u8>, rows: u32, shared: u32) -> Option<Vec<u32>> {
        run_warp_with(code, global, rows, shared, &[])
    }

    fn run_warp_with(code: &[[u8; 16]], global: Vec<u8>, rows: u32, shared: u32, consts: &[u8]) -> Option<Vec<u32>> {
        let Ok(mut s) = Surfaces::headless(crate::Backend::Auto) else {
            eprintln!("no GPU adapter: test skipped");
            return None;
        };
        // Play mode's worker threads too: a dispatch never waits for its pipeline
        // (skipped, it gave black images for good), so every warp test runs whole.
        s.background = true;
        const PROGRAM: u64 = 0x10_0000;
        const CBUF: u64 = 0x20_0000;
        const GLOBAL: u64 = 0x30_0000;
        const HEADERS: u64 = 0x40_0000;
        const IMAGE: u64 = 0x50_0000;
        let closing = [w(0x94d, &[(87, 7)]), w(0x947, &[(87, 7), (32, (-16i64 as u64 as u128) & ((1 << 50) - 1))])];
        let mut program = [code, &closing[..]].concat().concat();
        program.resize(0x1000, 0); // programs are read in 4 KiB chunks
        let mut cbuf = vec![0u8; 0x10000];
        cbuf[0x80..0x88].copy_from_slice(&GLOBAL.to_le_bytes());
        cbuf[0x88..0x8c].copy_from_slice(&(global.len() as u32).to_le_bytes());
        cbuf[0x100..0x100 + consts.len()].copy_from_slice(consts);
        // Header 0: a 32 × rows pitch RGBA32F image at IMAGE.
        let mut header = [0u8; 32];
        for (hi, lo, v) in [(6, 0, 0x01), (9, 7, 7), (12, 10, 7), (15, 13, 7), (18, 16, 7), (21, 19, 2), (24, 22, 3), (27, 25, 4), (30, 28, 5)] {
            put(&mut header, hi, lo, v);
        }
        for (hi, lo, v) in [(63, 37, (IMAGE >> 5) as u32), (87, 85, 2), (111, 96, 512 >> 5), (144, 128, 31), (154, 151, 1), (175, 160, rows - 1), (191, 191, 1)] {
            put(&mut header, hi, lo, v);
        }
        let regions: Vec<(u64, Vec<u8>)> = vec![(PROGRAM, program), (CBUF, cbuf), (GLOBAL, global), (HEADERS, header.to_vec())];
        let mut mem = |addr: u64, out: &mut [u8]| {
            regions.iter().any(|(at, data)| {
                let Some(start) = addr.checked_sub(*at).map(|o| o as usize) else { return false };
                let Some(src) = data.get(start..start + out.len()) else { return false };
                out.copy_from_slice(src);
                true
            })
        };
        let reg = |m: u32| match m {
            0x1578 => HEADERS as u32,
            _ => 0,
        };
        let pools = Pools { headers: resources::header_pool(&reg), samplers: resources::sampler_pool(&reg), sampler_binding: resources::sampler_binding(&reg) };
        let launch = Launch { program: PROGRAM, grid: [1, 1, 1], block: [32, 1, 1], shared, cbufs: vec![(0, CBUF, 0x10000)] };
        s.dispatch(&launch, pools, &mut mem).unwrap_or_else(|e| panic!("{e}"));
        s.flush();
        let image = &s.cache.images[&IMAGE];
        let size = u64::from(512 * rows);
        let buffer = s.device.create_buffer(&wgpu::BufferDescriptor { label: None, size, usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ, mapped_at_creation: false });
        let mut encoder = s.device.create_command_encoder(&Default::default());
        let layout = wgpu::TexelCopyBufferLayout { offset: 0, bytes_per_row: Some(512), rows_per_image: Some(rows) };
        encoder.copy_texture_to_buffer(image.as_image_copy(), wgpu::TexelCopyBufferInfo { buffer: &buffer, layout }, image.size());
        s.queue.submit([encoder.finish()]);
        buffer.slice(..).map_async(wgpu::MapMode::Read, |r| r.expect("readback mapping failed"));
        s.device.poll(wgpu::PollType::wait_indefinitely()).expect("GPU poll");
        let bytes = buffer.slice(..).get_mapped_range().expect("mapped").to_vec();
        Some(bytes.as_chunks::<4>().0.iter().map(|b| u32::from_le_bytes(*b)).collect())
    }

    #[test]
    fn hmma_16816_matches_a_matrix_product_over_ptx_fragments() {
        // A 16x16, B 16x8, C 16x8, small integers: every product is exact in halves, and
        // every result is non-zero (its bits go through an f32 store).
        let a = |r: usize, k: usize| ((r + 2 * k) % 5) as f32 - 2.0;
        let b = |k: usize, c: usize| ((3 * k + c) % 3) as f32 - 1.0;
        let c = |r: usize, col: usize| (100 + 8 * r + col) as f32;
        let d = |r: usize, col: usize| c(r, col) + (0..16).map(|k| a(r, k) * b(k, col)).sum::<f32>();
        // Lane l (g = l / 4, q = l % 4): A rows g, g + 8 at k = 2q, 2q + 1 and + 8; B column
        // g at k = 2q, 2q + 1 and + 8; C/D rows g, g + 8 at columns 2q, 2q + 1 (PTX mma).
        let mut global = Vec::new();
        for l in 0..32 {
            let (g, q) = (l / 4, l % 4);
            let k = 2 * q;
            for word in [
                pair(a(g, k), a(g, k + 1)),
                pair(a(g + 8, k), a(g + 8, k + 1)),
                pair(a(g, k + 8), a(g, k + 9)),
                pair(a(g + 8, k + 8), a(g + 8, k + 9)),
                pair(b(k, g), b(k + 1, g)),
                pair(b(k + 8, g), b(k + 9, g)),
                pair(c(g, k), c(g, k + 1)),
                pair(c(g + 8, k), c(g + 8, k + 1)),
            ] {
                global.extend(word.to_le_bytes());
            }
        }
        // S2R R0 = lane; S2R R1 = tid.y (0); MOV R5 = 32; IMAD R2 = R0 * R5 + c[0][0x80];
        // LDG.128 R4, LDG.64 R8, LDG.64 R10 from [R2.64]; HMMA.16816.F16 R12 = R4, R8, R10;
        // SUST.D2 [R0, R1] = R12..R15.
        let code = [
            w(0x919, &[(16, 0), (72, 0)]),
            w(0x919, &[(16, 1), (72, 0x22)]),
            w(0x802, &[(16, 5), (32, 32), (72, 0xf)]),
            w(0x624, &[(16, 2), (24, 0), (64, 5), (38, 0x80), (54, 0), (81, 7), (87, 7)]),
            w(0x381, &[(16, 4), (24, 2), (72, 1), (40, 0), (73, 6), (81, 7)]),
            w(0x381, &[(16, 8), (24, 2), (72, 1), (40, 16), (73, 5), (81, 7)]),
            w(0x381, &[(16, 10), (24, 2), (72, 1), (40, 24), (73, 5), (81, 7)]),
            w(0x23c, &[(16, 12), (24, 4), (32, 8), (64, 10), (75, 1)]),
            w(0xb9b, &[(24, 0), (32, 12), (40, 0x20 / 4), (54, 0), (61, 3), (72, 0xf)]),
        ];
        let Some(words) = run_warp(&code, global, 1, 0) else { return };
        for l in 0..32 {
            let (g, q) = (l / 4, l % 4);
            let want = [pair(d(g, 2 * q), d(g, 2 * q + 1)), pair(d(g + 8, 2 * q), d(g + 8, 2 * q + 1))];
            assert_eq!([words[4 * l], words[4 * l + 1]], want, "lane {l}");
        }
    }

    /// FLO, as PTX's bfind: per lane, plain, signed, .SH and inverted forms on
    /// edge values, against a reference written from the definition.
    #[test]
    fn flo_finds_the_leading_bit_as_ptx_bfind() {
        let values: Vec<u32> = [0, 1, 2, 3, 0x8000_0000, 0xffff_ffff, 0x7fff_ffff, 0xffff_fffe, 0x0001_0000, 0x4000_0001, 0xc000_0000, 0x0000_ffff, 0xfffe_ffff, 0x10, 0x8000_0001, 0x2468_ace0]
            .into_iter()
            .chain((0..16).map(|i| 0x9e37_79b9u32.wrapping_mul(i + 1)))
            .collect();
        let unsigned = |x: u32| if x == 0 { u32::MAX } else { 31 - x.leading_zeros() };
        let signed = |x: u32| if (x as i32) < 0 { unsigned(!x) } else { unsigned(x) };
        let sh = |r: u32| if r == u32::MAX { r } else { 31 - r };
        let global: Vec<u8> = values.iter().flat_map(|v| v.to_le_bytes()).collect();
        // S2R R0 = lane; S2R R1 = tid.y; MOV R5 = 4; IMAD R2 = R0 * R5 + c[0][0x80]; LDG.32 R4;
        // FLO R12 = R4; FLO.S32 R13 = R4; FLO.SH R14 = R4; FLO R15 = ~R4; SUST [R0, R1] = R12..R15.
        let flo = |dst: u128, extra: &[(u32, u128)]| w(0x300, &[&[(16, dst), (32, 4), (81, 7)][..], extra].concat());
        let code = [
            w(0x919, &[(16, 0), (72, 0)]),
            w(0x919, &[(16, 1), (72, 0x22)]),
            w(0x802, &[(16, 5), (32, 4), (72, 0xf)]),
            w(0x624, &[(16, 2), (24, 0), (64, 5), (38, 0x80), (54, 0), (81, 7), (87, 7)]),
            w(0x381, &[(16, 4), (24, 2), (72, 1), (40, 0), (73, 4), (81, 7)]),
            flo(12, &[]),
            flo(13, &[(73, 1)]),
            flo(14, &[(74, 1)]),
            flo(15, &[(63, 1)]),
            w(0xb9b, &[(24, 0), (32, 12), (40, 0x20 / 4), (54, 0), (61, 3), (72, 0xf)]),
        ];
        let Some(words) = run_warp(&code, global, 1, 0) else { return };
        for (l, &x) in values.iter().enumerate() {
            assert_eq!(words[4 * l..4 * l + 4], [unsigned(x), signed(x), sh(unsigned(x)), unsigned(!x)], "lane {l}: {x:#x}");
        }
    }

    #[test]
    fn iabs_and_brev_follow_ptx_abs_and_brev() {
        let values: Vec<u32> = [0, 1, 0xffff_ffff, 0x8000_0000, 0x7fff_ffff, 0x8000_0001, 0x1234_5678, 0xfedc_ba98].into_iter().chain((0..24).map(|i| 0x9e37_79b9u32.wrapping_mul(i + 1))).collect();
        let global: Vec<u8> = values.iter().flat_map(|v| v.to_le_bytes()).collect();
        // R4 = lane's value; IABS R12 = |R4|; BREV R13 = R4 bit-reversed; SUST R12..R15 to (lane, 0).
        let mut code = load_per_lane(4, 4);
        code.extend([w(0x213, &[(16, 12), (32, 4)]), w(0x301, &[(16, 13), (32, 4)])]);
        code.extend(store_row(0, 12));
        let Some(words) = run_warp(&code, global, 1, 0) else { return };
        for (l, &x) in values.iter().enumerate() {
            assert_eq!(words[4 * l..4 * l + 2], [(x as i32).wrapping_abs() as u32, x.reverse_bits()], "lane {l}: {x:#x}");
        }
    }

    /// S2R R0 = lane; S2R R1 = tid.y (0); MOV R5 = `stride`; IMAD R2 = R0 * R5 +
    /// c[0][0x80]; LDG R4.. (`ty`: 4 = 32 bits, 5 = 64) = [R2.64]: lane l's data.
    fn load_per_lane(stride: u128, ty: u128) -> Vec<[u8; 16]> {
        vec![
            w(0x919, &[(16, 0), (72, 0)]),
            w(0x919, &[(16, 1), (72, 0x22)]),
            w(0x802, &[(16, 5), (32, stride), (72, 0xf)]),
            w(0x624, &[(16, 2), (24, 0), (64, 5), (38, 0x80), (54, 0), (81, 7), (87, 7)]),
            w(0x381, &[(16, 4), (24, 2), (72, 1), (40, 0), (73, ty), (81, 7)]),
        ]
    }

    /// SUST of R`data`..+3 to texel (lane, `row`) (coordinates in R24, R25).
    fn store_row(row: u128, data: u128) -> Vec<[u8; 16]> {
        vec![
            w(0x202, &[(16, 24), (32, 0), (72, 0xf)]),
            w(0x802, &[(16, 25), (32, row), (72, 0xf)]),
            w(0xb9b, &[(24, 24), (32, data), (40, 0x20 / 4), (54, 0), (61, 3), (72, 0xf)]),
        ]
    }

    /// SEL R`dst` = P`p` ? 1 : 0 (R30 holds 1).
    fn sel(dst: u128, p: u128, not: bool) -> [u8; 16] {
        w(0x807, &[(16, dst), (24, 30), (32, 0), (87, p), (90, u128::from(not))])
    }

    /// A word under guard @P`p`.
    fn guarded(word: [u8; 16], p: u128) -> [u8; 16] {
        (u128::from_le_bytes(word) & !(0xf << 12) | p << 12).to_le_bytes()
    }

    /// F2I to 8, 16 and 32 bits, signed and not, in the four rounding modes, on
    /// edge values (NaN, infinities, ties, out of range), as PTX cvt.rni etc.
    #[test]
    fn f2i_saturates_to_small_widths_as_ptx_cvt() {
        let values: [f32; 32] = [
            f32::NAN, f32::INFINITY, f32::NEG_INFINITY, 0.0, -0.0, 0.5, 1.5, 2.5, -0.5, -1.5, -2.5, 65535.4, 65535.5, 65536.0, 70000.0, -1.0,
            -32768.4, -32768.6, 32767.5, 127.5, 128.0, -128.5, -129.0, 255.9, 256.0, 1e10, -1e10, 3e9, -3e9, 2147483520.0, 4294967040.0, 1.0000001,
        ];
        // (bits, signed, rounding: 0 nearest even, 1 down, 2 up, 3 toward zero)
        let forms: [(u32, bool, u8); 8] = [(16, false, 0), (16, true, 0), (8, false, 3), (8, true, 1), (32, false, 2), (32, true, 0), (16, true, 3), (16, false, 2)];
        let reference = |x: f32, (bits, signed, rnd): (u32, bool, u8)| -> u32 {
            if x.is_nan() {
                return 0;
            }
            let r = f64::from(match rnd {
                0 => x.round_ties_even(),
                1 => x.floor(),
                2 => x.ceil(),
                _ => x.trunc(),
            });
            let (lo, hi) = if signed { (-(2f64.powi(bits as i32 - 1)), 2f64.powi(bits as i32 - 1) - 1.0) } else { (0.0, 2f64.powi(bits as i32) - 1.0) };
            r.clamp(lo, hi) as i64 as u32
        };
        let f2i = |dst: u128, (bits, signed, rnd): (u32, bool, u8)| {
            w(0x305, &[(16, dst), (32, 4), (72, u128::from(signed)), (75, u128::from((bits / 8).ilog2())), (77, 1), (78, u128::from(rnd)), (80, 1), (84, 2)])
        };
        let mut code = load_per_lane(4, 4);
        code.extend(forms.iter().enumerate().map(|(k, f)| f2i(12 + k as u128, *f)));
        code.extend(store_row(0, 12));
        code.extend(store_row(1, 16));
        let global = values.iter().flat_map(|v| v.to_le_bytes()).collect();
        let Some(words) = run_warp(&code, global, 2, 0) else { return };
        for (l, &x) in values.iter().enumerate() {
            for (k, f) in forms.iter().enumerate() {
                assert_eq!(words[(k / 4) * 128 + 4 * l + k % 4], reference(x, *f), "{x} as {f:?}");
            }
        }
    }

    /// An f16 to f32, for the reference.
    fn f16(h: u16) -> f32 {
        let (sign, exp, frac) = (if h >> 15 != 0 { -1.0 } else { 1.0 }, i32::from(h >> 10 & 0x1f), f32::from(h & 0x3ff));
        sign * match exp {
            0 => frac * 2f32.powi(-24),
            31 if frac == 0.0 => f32::INFINITY,
            31 => f32::NAN,
            e => (1.0 + frac / 1024.0) * 2f32.powi(e - 15),
        }
    }

    /// HSETP2 per half, with .H_AND and lane selectors, combined with
    /// accumulators, on halves including NaN, infinities, zeros and denormals.
    #[test]
    fn hsetp2_compares_each_half_and_combines_them() {
        let halves: [u16; 12] = [0x0000, 0x8000, 0x3c00, 0xbc00, 0x7c00, 0xfc00, 0x7e00, 0x3555, 0x0001, 0x7bff, 0x4000, 0xc000];
        let pair = |l: usize| -> (u32, u32) {
            let h = |k: usize| u32::from(halves[k % 12]);
            (h(5 * l + 1) | h(7 * l + 3) << 16, h(3 * l) | h(11 * l + 2) << 16)
        };
        let global = (0..32).flat_map(|l| [pair(l).0, pair(l).1]).flat_map(|v| v.to_le_bytes()).collect();
        // R4 = a, R5 = b. P0, P1 = a > b per half; P2, P3 = .H_AND of a <u b (unordered),
        // or'ed with P0; P4 = (a.H0_H0 != b.H1_H1) xor !P1.
        let hsetp2 = |fields: &[(u32, u128)]| w(0x234, &[&[(24, 4), (32, 5)][..], fields].concat());
        let mut code = load_per_lane(8, 5);
        code.extend([
            w(0x802, &[(16, 30), (32, 1), (72, 0xf)]),
            hsetp2(&[(76, 4), (81, 0), (84, 1), (87, 7)]),
            hsetp2(&[(69, 1), (71, 1), (76, 9), (81, 2), (84, 3), (87, 0)]),
            hsetp2(&[(74, 2), (60, 3), (69, 2), (76, 5), (81, 4), (84, 7), (87, 1), (90, 1)]),
        ]);
        code.extend((0..4).map(|p| sel(12 + p, p, false)));
        code.extend((0..4).map(|k| sel(16 + k, 4, false)));
        code.extend(store_row(0, 12));
        code.extend(store_row(1, 16));
        let Some(words) = run_warp(&code, global, 2, 0) else { return };
        for l in 0..32 {
            let (a, b) = pair(l);
            let (alo, ahi, blo, bhi) = (f16(a as u16), f16((a >> 16) as u16), f16(b as u16), f16((b >> 16) as u16));
            let (p0, p1) = (alo > blo, ahi > bhi);
            // Unordered less-than (true on NaN) and ordered not-equal (false on NaN).
            let ltu = |x: f32, y: f32| x.partial_cmp(&y).is_none_or(|o| o.is_lt());
            let both = ltu(alo, blo) && ltu(ahi, bhi);
            let ne = alo.partial_cmp(&bhi).is_some_and(|o| o.is_ne());
            let want = [p0, p1, both || p0, !both || p0, ne != !p1].map(u32::from);
            assert_eq!([words[4 * l], words[4 * l + 1], words[4 * l + 2], words[4 * l + 3], words[128 + 4 * l]], want, "lane {l}: {a:#x} {b:#x}");
        }
    }

    /// LDC of U16, S16, U8 and S8 at a register index, from bytes with their sign bits set or not.
    #[test]
    fn ldc_of_small_types_at_a_register_index_extends_them() {
        let consts: Vec<u8> = (0..64u32).map(|i| (i * 73 + 41) as u8).collect();
        let ldc = |dst: u128, index: u128, offset: u128, ty: u128| w(0xb82, &[(16, dst), (24, index), (38, offset), (54, 0), (73, ty)]);
        // R3 = lane × 2; R12 = U16 c[0][R3 + 0x100]; R13 = S16 there; R14 = U8 c[0][R0 + 0x101];
        // R15 = S8 c[0][R0 + 0x120].
        let code = [
            w(0x919, &[(16, 0), (72, 0)]),
            w(0x919, &[(16, 1), (72, 0x22)]),
            w(0x802, &[(16, 5), (32, 2), (72, 0xf)]),
            w(0x224, &[(16, 3), (24, 0), (32, 5), (64, 255), (81, 7), (87, 7)]),
            ldc(12, 3, 0x100, 2),
            ldc(13, 3, 0x100, 3),
            ldc(14, 0, 0x101, 0),
            ldc(15, 0, 0x120, 1),
            w(0xb9b, &[(24, 0), (32, 12), (40, 0x20 / 4), (54, 0), (61, 3), (72, 0xf)]),
        ];
        let Some(words) = run_warp_with(&code, Vec::new(), 1, 0, &consts) else { return };
        for l in 0..32 {
            let h = u16::from_le_bytes([consts[2 * l], consts[2 * l + 1]]);
            let want = [u32::from(h), h as i16 as i32 as u32, u32::from(consts[l + 1]), consts[l + 0x20] as i8 as i32 as u32];
            assert_eq!(words[4 * l..4 * l + 4], want, "lane {l}");
        }
    }

    /// VOTE all, any and eq into predicates and ballots, VOTEU into a uniform
    /// register and predicate, and votes under a guard counting only its lanes.
    #[test]
    fn vote_counts_the_active_lanes_as_ptx_vote() {
        // Lane l: P0 = l % 3 == 0, P1 = true, P2 = false, P3 = l < 16, P4 = l >= 16.
        let bits = |l: u32| u32::from(l.is_multiple_of(3)) | 2 | u32::from(l < 16) << 3 | u32::from(l >= 16) << 4;
        let global = (0..32).flat_map(|l| bits(l).to_le_bytes()).collect();
        let vote = |dst: u128, mode: u128, p: u128, not: bool| w(0x806, &[(16, dst), (72, mode), (81, 5), (87, p), (90, u128::from(not))]);
        let x = 0x1234_5678u32;
        let mov = |dst: u128, v: u32| w(0x802, &[(16, dst), (32, u128::from(v)), (72, 0xf)]);
        let mut code = load_per_lane(4, 4);
        code.extend([
            w(0x804, &[(24, 4), (32, 0x1f)]),
            mov(30, 1),
            mov(13, x),
            mov(14, x),
            mov(22, x),
            mov(23, x),
            vote(12, 0, 0, false), // ballot P0; all P0
            sel(16, 5, false),
            vote(255, 1, 2, false), // any P2
            sel(17, 5, false),
            vote(255, 2, 0, false), // eq P0
            sel(18, 5, false),
            vote(255, 2, 2, true), // eq !P2
            sel(19, 5, false),
            vote(255, 1, 0, false), // any P0
            sel(20, 5, false),
            // VOTEU.ALL UR5, UP5 = P0; PLOP3 P5 = UP5; MOV R15 = UR5.
            w(0x886, &[(16, 5), (72, 0), (81, 5), (87, 0)]),
            w(0x81c, &[(81, 5), (84, 7), (87, 7), (77, 7), (68, 5), (67, 1), (64, 2), (72, 0x15)]),
            sel(21, 5, false),
            w(0xc02, &[(16, 15), (32, 5), (72, 0xf)]),
            // Lanes < 16 only: ballot and all of P1; eq of P4 (false on all of them).
            guarded(vote(13, 0, 1, false), 3),
            guarded(sel(14, 5, false), 3),
            guarded(vote(255, 2, 4, false), 3),
            guarded(sel(22, 5, false), 3),
        ]);
        code.extend(store_row(0, 12));
        code.extend(store_row(1, 16));
        code.extend(store_row(2, 20));
        let Some(words) = run_warp(&code, global, 3, 0) else { return };
        let ballot = (0..32).filter(|l| l % 3 == 0).fold(0u32, |b, l| b | 1 << l);
        for l in 0..32 {
            let low = l < 16;
            let want = [
                [ballot, if low { 0xffff } else { x }, if low { 1 } else { x }, ballot],
                [0, 0, 0, 1],
                [1, 0, if low { 1 } else { x }, x],
            ];
            for (row, want) in want.iter().enumerate() {
                assert_eq!(words[row * 128 + 4 * l..row * 128 + 4 * l + 4], *want, "lane {l}, row {row}");
            }
        }
    }

    /// P2R: predicates packed under a mask over a register's other bits.
    #[test]
    fn p2r_packs_predicates_under_its_mask() {
        let values: Vec<u32> = (0..32u32).map(|i| 0x9e37_79b9u32.wrapping_mul(i + 1)).collect();
        let p2r = |dst: u128, a: u128, mask: u128| w(0x803, &[(16, dst), (24, a), (32, mask)]);
        let mut code = load_per_lane(4, 4);
        code.extend([w(0x804, &[(24, 4), (32, 0x7f)]), w(0x802, &[(16, 6), (32, 0xffff_ffff), (72, 0xf)]), p2r(12, 255, 0x40), p2r(13, 6, 0x2a), p2r(14, 255, 0x7f), p2r(15, 4, 0)]);
        code.extend(store_row(0, 12));
        let global = values.iter().flat_map(|v| v.to_le_bytes()).collect();
        let Some(words) = run_warp(&code, global, 1, 0) else { return };
        for (l, &v) in values.iter().enumerate() {
            assert_eq!(words[4 * l..4 * l + 4], [v & 0x40, !0x2a | v & 0x2a, v & 0x7f, v], "lane {l}: {v:#x}");
        }
    }

    /// LDG from a uniform register pair (ULDC.64 of the pointer) plus a 32-bit register.
    #[test]
    fn ldg_from_a_uniform_base_reads_the_pointers_buffer() {
        let values: Vec<u32> = (0..35u32).map(|i| 0x4000_0000 | i.wrapping_mul(0x0101_0101)).collect();
        let ldg = |dst: u128, offset: u128, ty: u128| w(0x981, &[(16, dst), (24, 2), (32, 4), (91, 1), (72, 1), (40, offset), (73, ty), (81, 7)]);
        // ULDC.64 UR4 = c[0][0x80]; S2R R0 = lane; MOV R5 = 4; IMAD R2 = R0 * R5 + RZ;
        // LDG R12 = [R2 + UR4]; LDG R13 = [R2 + UR4 + 4]; LDG.64 R14 = [R2 + UR4 + 8].
        let code = [
            w(0xab9, &[(16, 4), (38, 0x80), (54, 0), (73, 5)]),
            w(0x919, &[(16, 0), (72, 0)]),
            w(0x919, &[(16, 1), (72, 0x22)]),
            w(0x802, &[(16, 5), (32, 4), (72, 0xf)]),
            w(0x224, &[(16, 2), (24, 0), (32, 5), (64, 255), (81, 7), (87, 7)]),
            ldg(12, 0, 4),
            ldg(13, 4, 4),
            ldg(14, 8, 5),
            w(0xb9b, &[(24, 0), (32, 12), (40, 0x20 / 4), (54, 0), (61, 3), (72, 0xf)]),
        ];
        let global = values.iter().flat_map(|v| v.to_le_bytes()).collect();
        let Some(words) = run_warp(&code, global, 1, 0) else { return };
        for l in 0..32 {
            assert_eq!(words[4 * l..4 * l + 4], values[l..l + 4], "lane {l}");
        }
    }

    #[test]
    fn an_image_store_outside_the_image_is_dropped() {
        // Lane l stores 1.0 at (l, 0), then 2.0 at (l, 1), past the single row: row 0
        // must keep 1.0 (the game's upscaler stores its last tile's extra row this way).
        let imm = |dst: u128, v: u128| w(0x802, &[(16, dst), (32, v), (72, 0xf)]);
        let sust = |coord: u128, data: u128| w(0xb9b, &[(24, coord), (32, data), (40, 0x20 / 4), (54, 0), (61, 3), (72, 0xf)]);
        let mut code = vec![w(0x919, &[(16, 10), (72, 0)]), imm(11, 0), w(0x202, &[(16, 20), (32, 10), (72, 0xf)]), imm(21, 1)];
        for k in 0..4 {
            code.push(imm(12 + k, 0x3f80_0000));
            code.push(imm(16 + k, 0x4000_0000));
        }
        code.extend([sust(10, 12), sust(20, 16)]);
        let Some(words) = run_warp(&code, Vec::new(), 1, 0) else { return };
        assert!(words.iter().all(|&x| x == 0x3f80_0000), "{words:x?}");
    }

    #[test]
    fn an_image_reduction_adds_atomically_on_the_gpu() {
        let Ok(mut s) = Surfaces::headless(crate::Backend::Auto) else {
            eprintln!("no GPU adapter: test skipped");
            return;
        };
        if !s.device.features().contains(wgpu::Features::TEXTURE_ATOMIC) {
            eprintln!("no texture atomics: test skipped");
            return;
        }
        s.background = true;
        const PROGRAM: u64 = 0x10_0000;
        const CBUF: u64 = 0x20_0000;
        const HEADERS: u64 = 0x40_0000;
        const IMAGE: u64 = 0x50_0000;
        // R0 = R1 = 0, R3 = 1; 32 lanes SURED.ADD.U32 R3 at (R0, R1) of the 2D image c[0][0x20].
        let mov = |d: u128, v: u128| w(0x802, &[(16, d), (32, v), (72, 0xf)]);
        let sured = w(0x59f, &[(24, 0), (32, 3), (40, 0x20 / 4), (54, 0), (61, 3), (77, 7), (84, 1)]);
        let closing = [w(0x94d, &[(87, 7)]), w(0x947, &[(87, 7), (32, (-16i64 as u64 as u128) & ((1 << 50) - 1))])];
        let mut program = [mov(0, 0), mov(1, 0), mov(3, 1), sured, closing[0], closing[1]].concat();
        program.resize(0x1000, 0);
        // Header 0: a 64 × 1 pitch R32 UINT image at IMAGE.
        let mut header = [0u8; 32];
        for (hi, lo, v) in [(6, 0, 0x0f), (9, 7, 4), (12, 10, 4), (15, 13, 4), (18, 16, 4), (21, 19, 2), (24, 22, 3), (27, 25, 4), (30, 28, 5), (63, 37, (IMAGE >> 5) as u32), (87, 85, 2), (111, 96, 256 >> 5), (144, 128, 63), (154, 151, 1), (191, 191, 1)] {
            put(&mut header, hi, lo, v);
        }
        let regions: Vec<(u64, Vec<u8>)> = vec![(PROGRAM, program), (CBUF, vec![0u8; 0x10000]), (HEADERS, header.to_vec())];
        let mut mem = |addr: u64, out: &mut [u8]| {
            regions.iter().any(|(at, data)| {
                let Some(src) = addr.checked_sub(*at).and_then(|o| data.get(o as usize..o as usize + out.len())) else { return false };
                out.copy_from_slice(src);
                true
            })
        };
        let reg = |m: u32| if m == 0x1578 { HEADERS as u32 } else { 0 };
        let pools = Pools { headers: resources::header_pool(&reg), samplers: resources::sampler_pool(&reg), sampler_binding: resources::sampler_binding(&reg) };
        let launch = Launch { program: PROGRAM, grid: [1, 1, 1], block: [32, 1, 1], shared: 0, cbufs: vec![(0, CBUF, 0x10000)] };
        s.dispatch(&launch, pools, &mut mem).unwrap_or_else(|e| panic!("{e}"));
        s.flush();
        let buffer = s.device.create_buffer(&wgpu::BufferDescriptor { label: None, size: 4, usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ, mapped_at_creation: false });
        let mut encoder = s.device.create_command_encoder(&Default::default());
        let layout = wgpu::TexelCopyBufferLayout { offset: 0, bytes_per_row: None, rows_per_image: None };
        encoder.copy_texture_to_buffer(s.cache.images[&IMAGE].as_image_copy(), wgpu::TexelCopyBufferInfo { buffer: &buffer, layout }, wgpu::Extent3d { width: 1, height: 1, depth_or_array_layers: 1 });
        s.queue.submit([encoder.finish()]);
        buffer.slice(..).map_async(wgpu::MapMode::Read, |r| r.expect("readback mapping failed"));
        s.device.poll(wgpu::PollType::wait_indefinitely()).expect("GPU poll");
        let texel = buffer.slice(..).get_mapped_range().expect("mapped").to_vec();
        assert_eq!(texel, 32u32.to_le_bytes(), "{:?}", s.errors.lock().unwrap());
    }

    #[test]
    fn an_image_atomic_gives_each_lane_its_old_value_on_the_gpu() {
        let Ok(mut s) = Surfaces::headless(crate::Backend::Auto) else {
            eprintln!("no GPU adapter: test skipped");
            return;
        };
        s.background = true;
        const PROGRAM: u64 = 0x10_0000;
        const CBUF: u64 = 0x20_0000;
        const HEADERS: u64 = 0x40_0000;
        const IMAGE: u64 = 0x50_0000;
        // 32 lanes: SUATOM.ADD.U32 R4 = texel (0, 0) of the 2D image c[0][0x20], plus R3 = 1;
        // then SURED.ADD R3 at (R4, R5 = 1). The old values are 0..31, one per lane: row 1
        // holds 1 in texels 0..31, and texel (0, 0) ends at 32.
        let mov = |d: u128, v: u128| w(0x802, &[(16, d), (32, v), (72, 0xf)]);
        let suatom = w(0x593, &[(16, 4), (24, 0), (32, 3), (40, 0x20 / 4), (54, 0), (61, 3), (77, 7), (81, 7), (84, 1)]);
        let sured = w(0x59f, &[(24, 4), (32, 3), (40, 0x20 / 4), (54, 0), (61, 3), (77, 7), (84, 1)]);
        let closing = [w(0x94d, &[(87, 7)]), w(0x947, &[(87, 7), (32, (-16i64 as u64 as u128) & ((1 << 50) - 1))])];
        let mut program = [mov(0, 0), mov(1, 0), mov(3, 1), mov(5, 1), suatom, sured, closing[0], closing[1]].concat();
        program.resize(0x1000, 0);
        // Header 0: a 64 × 2 pitch R32 UINT image at IMAGE.
        let mut header = [0u8; 32];
        for (hi, lo, v) in [(6, 0, 0x0f), (9, 7, 4), (12, 10, 4), (15, 13, 4), (18, 16, 4), (21, 19, 2), (24, 22, 3), (27, 25, 4), (30, 28, 5), (63, 37, (IMAGE >> 5) as u32), (87, 85, 2), (111, 96, 256 >> 5), (144, 128, 63), (154, 151, 1), (175, 160, 1), (191, 191, 1)] {
            put(&mut header, hi, lo, v);
        }
        let regions: Vec<(u64, Vec<u8>)> = vec![(PROGRAM, program), (CBUF, vec![0u8; 0x10000]), (HEADERS, header.to_vec())];
        let mut mem = |addr: u64, out: &mut [u8]| {
            regions.iter().any(|(at, data)| {
                let Some(src) = addr.checked_sub(*at).and_then(|o| data.get(o as usize..o as usize + out.len())) else { return false };
                out.copy_from_slice(src);
                true
            })
        };
        let reg = |m: u32| if m == 0x1578 { HEADERS as u32 } else { 0 };
        let pools = Pools { headers: resources::header_pool(&reg), samplers: resources::sampler_pool(&reg), sampler_binding: resources::sampler_binding(&reg) };
        let launch = Launch { program: PROGRAM, grid: [1, 1, 1], block: [32, 1, 1], shared: 0, cbufs: vec![(0, CBUF, 0x10000)] };
        s.dispatch(&launch, pools, &mut mem).unwrap_or_else(|e| panic!("{e}"));
        s.flush();
        let buffer = s.device.create_buffer(&wgpu::BufferDescriptor { label: None, size: 512, usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ, mapped_at_creation: false });
        let mut encoder = s.device.create_command_encoder(&Default::default());
        let layout = wgpu::TexelCopyBufferLayout { offset: 0, bytes_per_row: Some(256), rows_per_image: Some(2) };
        let image = &s.cache.images[&IMAGE];
        encoder.copy_texture_to_buffer(image.as_image_copy(), wgpu::TexelCopyBufferInfo { buffer: &buffer, layout }, image.size());
        s.queue.submit([encoder.finish()]);
        buffer.slice(..).map_async(wgpu::MapMode::Read, |r| r.expect("readback mapping failed"));
        s.device.poll(wgpu::PollType::wait_indefinitely()).expect("GPU poll");
        let bytes = buffer.slice(..).get_mapped_range().expect("mapped").to_vec();
        let texels: Vec<u32> = bytes.as_chunks::<4>().0.iter().map(|b| u32::from_le_bytes(*b)).collect();
        let want: Vec<u32> = (0..128).map(|i| match i {
            0 => 32,
            64..96 => 1,
            _ => 0,
        }).collect();
        assert_eq!(texels, want, "{:?}", s.errors.lock().unwrap());
    }

    #[test]
    fn ldsm_and_movm_follow_ptx_ldmatrix_and_movmatrix() {
        // Lane l stores row l of four stacked 8x8 matrices of 16-bit values in shared
        // memory (16 bytes at l * 16); element (row, col) = 0x4000 + row * 8 + col, which
        // keeps every packed word a normal f32 through the image store.
        let elem = |row: usize, col: usize| (0x4000 + row * 8 + col) as u32;
        let mut global = Vec::new();
        for l in 0..32 {
            for j in 0..4 {
                global.extend((elem(l, 2 * j) | elem(l, 2 * j + 1) << 16).to_le_bytes());
            }
        }
        // S2R R0 = lane; MOV R5 = 16; IMAD R1 = R0 * R5 + RZ (the row address);
        // IMAD R2 = R0 * R5 + c[0][0x80]; LDG.128 R8 = [R2.64]; STS.128 [R1] = R8;
        // BAR.SYNC; LDSM.x4 R12 = [R1]; LDSM.x4.MT88 R16 = [R1]; MOVM.MT88 R20 = R12;
        // texel (lane, 0) = R12..R15, (lane, 1) = R16..R19, (lane, 2) = R20..R23.
        let mov = |dst: u128, src: u128| w(0x202, &[(16, dst), (32, src), (72, 0xf)]);
        let imm = |dst: u128, v: u128| w(0x802, &[(16, dst), (32, v), (72, 0xf)]);
        let sust = |coord: u128, data: u128| w(0xb9b, &[(24, coord), (32, data), (40, 0x20 / 4), (54, 0), (61, 3), (72, 0xf)]);
        let code = [
            w(0x919, &[(16, 0), (72, 0)]),
            imm(5, 16),
            w(0x224, &[(16, 1), (24, 0), (32, 5), (64, 255), (81, 7), (87, 7)]),
            w(0x624, &[(16, 2), (24, 0), (64, 5), (38, 0x80), (54, 0), (81, 7), (87, 7)]),
            w(0x381, &[(16, 8), (24, 2), (72, 1), (40, 0), (73, 6), (81, 7)]),
            w(0x388, &[(24, 1), (32, 8), (73, 6)]),
            w(0xb1d, &[]),
            w(0x83b, &[(16, 12), (24, 1), (72, 2)]),
            w(0x83b, &[(16, 16), (24, 1), (72, 2), (78, 1)]),
            w(0x23a, &[(16, 20), (24, 12)]),
            // Coordinates (x, y) in R24..R29: (lane, 0), (lane, 1), (lane, 2).
            mov(24, 0),
            imm(25, 0),
            mov(26, 0),
            imm(27, 1),
            mov(28, 0),
            imm(29, 2),
            sust(24, 12),
            sust(26, 16),
            sust(28, 20),
        ];
        let Some(words) = run_warp(&code, global, 3, 0x200) else { return };
        let texel = |row: usize, l: usize| [0, 1, 2, 3].map(|k| words[row * 128 + 4 * l + k]);
        for l in 0..32 {
            let (g, q) = (l / 4, l % 4);
            // ldmatrix: register i = row g of matrix i, columns 2q and 2q + 1.
            let plain = [0, 1, 2, 3].map(|i| elem(8 * i + g, 2 * q) | elem(8 * i + g, 2 * q + 1) << 16);
            // .trans: column g of matrix i, rows 2q and 2q + 1.
            let trans = [0, 1, 2, 3].map(|i| elem(8 * i + 2 * q, g) | elem(8 * i + 2 * q + 1, g) << 16);
            assert_eq!(texel(0, l), plain, "LDSM, lane {l}");
            assert_eq!(texel(1, l), trans, "LDSM.MT88, lane {l}");
            // movmatrix.trans of matrix 0 is what the transposed load gives for it.
            assert_eq!(texel(2, l)[0], trans[0], "MOVM, lane {l}");
        }
    }
}
