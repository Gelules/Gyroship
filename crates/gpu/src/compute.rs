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
        _ => return None,
    }))
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
            let header = textures::wgpu_texture_format(&tic).ok_or_else(|| format!("image format {:?}", tic.components))?;
            let (format, name) = storage_format(header).ok_or_else(|| format!("image format {header:?} as storage"))?;
            if img.read && !self.adapter.get_texture_format_features(format).flags.contains(wgpu::TextureFormatFeatureFlags::STORAGE_READ_WRITE) {
                return Err(format!("image format {format:?} both read and written"));
            }
            images.push((tic, format, name.to_string(), header));
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
        for (tic, format, _, header) in &images {
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
                    view_formats: if header == format { &[] } else { std::slice::from_ref(header) },
                });
                self.cache.images.insert(tic.address, texture);
            }
            views.push(self.cache.images[&tic.address].create_view(&Default::default()));
        }

        // Global memory (LDG): each pointer's buffer, read from guest memory. The constant
        // bank holds a storage-buffer descriptor there: the 64-bit address, then the size.
        // ponytail: read again at every dispatch; cache by address if it shows in profiles.
        let mut globals = Vec::new();
        for &(bank, at) in &t.interface.buffers {
            let word = |o: usize| banks.get(&bank).and_then(|b| b.get(o..o + 4)).map(|w| u32::from_le_bytes([w[0], w[1], w[2], w[3]]));
            let at = usize::from(at);
            let (Some(lo), Some(hi), Some(size)) = (word(at), word(at + 4), word(at + 8)) else { return Err("global memory descriptor out of its bank".into()) };
            // The word after the pointer is not proved to be a size: at least 1 MiB is read
            // when memory allows, else that many bytes.
            // ponytail: a window, not the buffer's real extent; bound it by the program's
            // offsets if a buffer ever runs past it.
            let address = u64::from(hi) << 32 | u64::from(lo);
            let mut data = Vec::new();
            for bytes in [size.max(1 << 20), size] {
                if bytes == 0 || bytes > 1 << 30 {
                    continue;
                }
                data = vec![0u8; bytes.next_multiple_of(4) as usize];
                if mem(address, &mut data) {
                    break;
                }
                data.clear();
            }
            if data.is_empty() {
                return Err(format!("global memory at c[{bank}][{at:#x}] ({address:#x}, size word {size:#x}) not readable"));
            }
            globals.push(wgpu::util::DeviceExt::create_buffer_init(&self.device, &wgpu::util::BufferInitDescriptor { label: Some("global"), contents: &data, usage: wgpu::BufferUsages::STORAGE }));
        }

        // Bindings: constant banks, textures and swizzles as for draws, then the images.
        let (textures, swizzles) = self.texture_slots(&t, &banks, pools.headers, pools.samplers, pools.sampler_binding, &[], mem)?;
        let cbufs: BTreeMap<u8, Vec<u8>> = banks.into_iter().filter(|(b, _)| t.interface.cbufs.contains(b)).collect();
        let bindings = Bindings { cbufs, textures, swizzles: t.interface.swizzles.then_some(swizzles) };
        let mut layout_entries = bindings.layout(wgpu::ShaderStages::COMPUTE);
        for (i, (img, (_, format, _, _))) in t.interface.images.iter().zip(&images).enumerate() {
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
        for i in 0..globals.len() as u32 {
            layout_entries.push(wgpu::BindGroupLayoutEntry {
                binding: shader::BUFFER_BINDING + i,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: true }, has_dynamic_offset: false, min_binding_size: None },
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
        for (i, buffer) in globals.iter().enumerate() {
            entries.push(wgpu::BindGroupEntry { binding: shader::BUFFER_BINDING + i as u32, resource: buffer.as_entire_binding() });
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

    #[test]
    fn hmma_16816_matches_a_matrix_product_over_ptx_fragments() {
        let Ok(mut s) = Surfaces::headless(crate::Backend::Auto) else {
            eprintln!("no GPU adapter: test skipped");
            return;
        };
        const PROGRAM: u64 = 0x10_0000;
        const CBUF: u64 = 0x20_0000;
        const GLOBAL: u64 = 0x30_0000;
        const HEADERS: u64 = 0x40_0000;
        const IMAGE: u64 = 0x50_0000;
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
        // SUST.D2 [R0, R1] = R12..R15 to the image of c[0][0x20]; EXIT.
        let code: Vec<u8> = [
            w(0x919, &[(16, 0), (72, 0)]),
            w(0x919, &[(16, 1), (72, 0x22)]),
            w(0x802, &[(16, 5), (32, 32), (72, 0xf)]),
            w(0x624, &[(16, 2), (24, 0), (64, 5), (38, 0x80), (54, 0), (81, 7), (87, 7)]),
            w(0x381, &[(16, 4), (24, 2), (72, 1), (40, 0), (73, 6), (81, 7)]),
            w(0x381, &[(16, 8), (24, 2), (72, 1), (40, 16), (73, 5), (81, 7)]),
            w(0x381, &[(16, 10), (24, 2), (72, 1), (40, 24), (73, 5), (81, 7)]),
            w(0x23c, &[(16, 12), (24, 4), (32, 8), (64, 10), (75, 1)]),
            w(0xb9b, &[(24, 0), (32, 12), (40, 0x20 / 4), (54, 0), (61, 3), (72, 0xf)]),
            w(0x94d, &[(87, 7)]),
            w(0x947, &[(87, 7), (32, (-16i64 as u64 as u128) & ((1 << 50) - 1))]),
        ]
        .concat();
        let mut code = code;
        code.resize(0x1000, 0); // programs are read in 4 KiB chunks
        let mut cbuf = vec![0u8; 0x10000];
        cbuf[0x80..0x88].copy_from_slice(&GLOBAL.to_le_bytes());
        cbuf[0x88..0x8c].copy_from_slice(&(global.len() as u32).to_le_bytes());
        // Header 0: a 32x1 pitch RGBA32F image at IMAGE.
        let mut header = [0u8; 32];
        for (hi, lo, v) in [(6, 0, 0x01), (9, 7, 7), (12, 10, 7), (15, 13, 7), (18, 16, 7), (21, 19, 2), (24, 22, 3), (27, 25, 4), (30, 28, 5)] {
            put(&mut header, hi, lo, v);
        }
        for (hi, lo, v) in [(63, 37, (IMAGE >> 5) as u32), (87, 85, 2), (111, 96, 512 >> 5), (144, 128, 31), (154, 151, 1), (191, 191, 1)] {
            put(&mut header, hi, lo, v);
        }
        let regions: Vec<(u64, Vec<u8>)> = vec![(PROGRAM, code), (CBUF, cbuf), (GLOBAL, global), (HEADERS, header.to_vec())];
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
        let launch = Launch { program: PROGRAM, grid: [1, 1, 1], block: [32, 1, 1], shared: 0, cbufs: vec![(0, CBUF, 0x10000)] };
        s.dispatch(&launch, pools, &mut mem).unwrap_or_else(|e| panic!("{e}"));
        // Read the image back: texel l holds lane l's two D registers as f32 bits.
        let image = &s.cache.images[&IMAGE];
        let buffer = s.device.create_buffer(&wgpu::BufferDescriptor { label: None, size: 512, usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ, mapped_at_creation: false });
        let mut encoder = s.device.create_command_encoder(&Default::default());
        encoder.copy_texture_to_buffer(image.as_image_copy(), wgpu::TexelCopyBufferInfo { buffer: &buffer, layout: wgpu::TexelCopyBufferLayout { offset: 0, bytes_per_row: Some(512), rows_per_image: None } }, image.size());
        s.queue.submit([encoder.finish()]);
        buffer.slice(..).map_async(wgpu::MapMode::Read, |r| r.expect("readback mapping failed"));
        s.device.poll(wgpu::PollType::wait_indefinitely()).expect("GPU poll");
        let bytes = buffer.slice(..).get_mapped_range().expect("mapped").to_vec();
        let words: Vec<u32> = bytes.as_chunks::<4>().0.iter().map(|b| u32::from_le_bytes(*b)).collect();
        for l in 0..32 {
            let (g, q) = (l / 4, l % 4);
            let want = [pair(d(g, 2 * q), d(g, 2 * q + 1)), pair(d(g + 8, 2 * q), d(g + 8, 2 * q + 1))];
            assert_eq!([words[4 * l], words[4 * l + 1]], want, "lane {l}");
        }
    }
}
