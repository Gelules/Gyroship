//! Render targets as wgpu textures, keyed by GPU address and format: the
//! host side of the 3D class's color and depth targets. Clears run here;
//! the targets a frame touched can be written out as PNG files.
//!
//! ponytail: whole-target clears only (full channel mask, no scissor smaller
//! than the target); the others are counted in `skipped`. Draws: `draw.rs`.

use crate::pushbuf::{Clear, Surface};
use std::collections::{BTreeMap, BTreeSet};
use std::path::Path;

/// The wgpu format of a 3D-class color target format (clc797.h), for the
/// renderable formats.
pub(crate) fn color_format(f: u32) -> Option<wgpu::TextureFormat> {
    use wgpu::TextureFormat as T;
    Some(match f {
        0xc0 => T::Rgba32Float,
        0xc6 => T::Rgba16Unorm,
        0xc7 => T::Rgba16Snorm,
        0xca => T::Rgba16Float,
        0xcb => T::Rg32Float,
        0xcf => T::Bgra8Unorm,
        0xd0 => T::Bgra8UnormSrgb,
        0xd1 => T::Rgb10a2Unorm,
        0xd5 => T::Rgba8Unorm,
        0xd6 => T::Rgba8UnormSrgb,
        0xda => T::Rg16Unorm,
        0xdb => T::Rg16Snorm,
        0xde => T::Rg16Float,
        0xe0 => T::Rg11b10Ufloat,
        0xe5 => T::R32Float,
        0xea => T::Rg8Unorm,
        0xee => T::R16Unorm,
        0xf2 => T::R16Float,
        0xf3 => T::R8Unorm,
        _ => return None,
    })
}

/// The wgpu format of a depth target format (SET_ZT_FORMAT).
pub(crate) fn depth_format(f: u32) -> Option<wgpu::TextureFormat> {
    use wgpu::TextureFormat as T;
    Some(match f {
        0x0a => T::Depth32Float,
        0x13 => T::Depth16Unorm,
        0x14 | 0x15 | 0x16 | 0x18 => T::Depth24PlusStencil8,
        0x19 => T::Depth32FloatStencil8,
        _ => return None,
    })
}

/// Copies a target into opaque RGBA8 for a PNG to look at (missing channels
/// read as 0; alpha is dropped, as a cleared alpha of 0 would hide the image).
const BLIT: &str = r#"
@group(0) @binding(0) var src: texture_2d<f32>;

@vertex
fn vs(@builtin(vertex_index) i: u32) -> @builtin(position) vec4<f32> {
    let p = vec2(f32(i & 1u) * 4.0 - 1.0, f32(i >> 1u) * 4.0 - 1.0);
    return vec4(p, 0.0, 1.0);
}

@fragment
fn fs(@builtin(position) p: vec4<f32>) -> @location(0) vec4<f32> {
    return vec4(textureLoad(src, vec2<i32>(p.xy), 0).rgb, 1.0);
}
"#;

pub(crate) struct Entry {
    pub(crate) surface: Surface,
    pub(crate) texture: wgpu::Texture,
}

pub struct Surfaces {
    adapter: wgpu::Adapter,
    pub(crate) device: wgpu::Device,
    pub(crate) queue: wgpu::Queue,
    pub(crate) map: BTreeMap<(u64, u32), Entry>,
    /// Color targets written since the last dump.
    pub(crate) touched: BTreeSet<(u64, u32)>,
    blit: (wgpu::RenderPipeline, wgpu::BindGroupLayout),
    pub(crate) cache: crate::draw::Cache,
    /// Draws done.
    pub drawn: u64,
    /// Operations not done, by reason.
    pub skipped: BTreeMap<String, u64>,
    /// wgpu validation errors (the first ones): reported, never fatal to the game.
    pub errors: Arc<Mutex<Vec<String>>>,
}

use std::sync::{Arc, Mutex};

impl Surfaces {
    /// A headless device with the adapter's own limits (targets reach 11264
    /// pixels) and, when available, the 16-bit normalized, D32S8 and BC
    /// formats, filterable f32 textures and border-color clamping.
    pub fn headless(backend: crate::Backend) -> Result<Surfaces, String> {
        let instance = crate::Renderer::instance_for(backend);
        let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            compatible_surface: None,
            force_fallback_adapter: false,
            apply_limit_buckets: false,
        }))
        .map_err(|e| format!("no suitable GPU adapter: {e}"))?;
        // 16-bit normalized targets are renderable only through the adapter's own
        // format features (wgpu guarantees them as storage only).
        let wanted = wgpu::Features::TEXTURE_FORMAT_16BIT_NORM
            | wgpu::Features::DEPTH32FLOAT_STENCIL8
            | wgpu::Features::TEXTURE_ADAPTER_SPECIFIC_FORMAT_FEATURES
            | wgpu::Features::RG11B10UFLOAT_RENDERABLE
            | wgpu::Features::TEXTURE_COMPRESSION_BC
            | wgpu::Features::FLOAT32_FILTERABLE
            | wgpu::Features::ADDRESS_MODE_CLAMP_TO_BORDER;
        let descriptor = wgpu::DeviceDescriptor { required_features: adapter.features() & wanted, required_limits: adapter.limits(), ..Default::default() };
        let (device, queue) = pollster::block_on(adapter.request_device(&descriptor)).map_err(|e| format!("cannot open the GPU device: {e}"))?;
        let errors = Arc::new(Mutex::new(Vec::new()));
        let sink = errors.clone();
        device.on_uncaptured_error(Arc::new(move |e: wgpu::Error| {
            if let Ok(mut v) = sink.lock()
                && v.len() < 16
            {
                v.push(e.to_string());
            }
        }));
        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor { label: Some("blit"), source: wgpu::ShaderSource::Wgsl(BLIT.into()) });
        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: None,
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Texture { sample_type: wgpu::TextureSampleType::Float { filterable: false }, view_dimension: wgpu::TextureViewDimension::D2, multisampled: false },
                count: None,
            }],
        });
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor { label: None, bind_group_layouts: &[Some(&layout)], ..Default::default() });
        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("blit"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState { module: &module, entry_point: Some("vs"), compilation_options: Default::default(), buffers: &[] },
            fragment: Some(wgpu::FragmentState { module: &module, entry_point: Some("fs"), compilation_options: Default::default(), targets: &[Some(wgpu::TextureFormat::Rgba8Unorm.into())] }),
            primitive: Default::default(),
            depth_stencil: None,
            multisample: Default::default(),
            multiview_mask: None,
            cache: None,
        });
        Ok(Surfaces { adapter, device, queue, map: BTreeMap::new(), touched: BTreeSet::new(), blit: (pipeline, layout), cache: Default::default(), drawn: 0, skipped: BTreeMap::new(), errors })
    }

    /// Whether this device can render to `format` (and read it back).
    pub(crate) fn renderable(&self, format: wgpu::TextureFormat) -> bool {
        let features = if self.device.features().contains(wgpu::Features::TEXTURE_ADAPTER_SPECIFIC_FORMAT_FEATURES) {
            self.adapter.get_texture_format_features(format)
        } else {
            format.guaranteed_format_features(self.device.features())
        };
        features.allowed_usages.contains(wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_SRC)
    }

    pub(crate) fn skip(&mut self, why: impl Into<String>) {
        *self.skipped.entry(why.into()).or_default() += 1;
    }

    /// Keeps a message with the wgpu errors (the first ones).
    pub(crate) fn error(&self, what: String) {
        if let Ok(mut v) = self.errors.lock()
            && v.len() < 16
        {
            v.push(what);
        }
    }

    /// The texture of a target, made (or remade, if its size changed) on demand.
    pub(crate) fn texture(&mut self, s: &Surface, format: wgpu::TextureFormat) -> wgpu::TextureView {
        let key = (s.addr, s.format);
        let stale = self.map.get(&key).is_none_or(|e| e.surface != *s);
        if stale {
            let texture = self.device.create_texture(&wgpu::TextureDescriptor {
                label: None,
                size: wgpu::Extent3d { width: s.width.max(1), height: s.height.max(1), depth_or_array_layers: s.layers },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format,
                usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_SRC,
                view_formats: &[],
            });
            self.map.insert(key, Entry { surface: *s, texture });
        }
        self.map[&key].texture.create_view(&wgpu::TextureViewDescriptor { dimension: Some(wgpu::TextureViewDimension::D2), array_layer_count: Some(1), ..Default::default() })
    }

    /// Runs a clear on the host textures.
    pub fn clear(&mut self, c: &Clear) {
        let covers = |s: &Surface| c.scissor.is_none_or(|[x, y, w, h]| x == 0 && y == 0 && w >= s.width && h >= s.height);
        if c.layer != 0 {
            return self.skip("clear of a layer other than 0");
        }
        let mut encoder = self.device.create_command_encoder(&Default::default());
        if let Some((s, value, mask)) = &c.color {
            match color_format(s.format) {
                None => self.skip("color clear: unsupported format"),
                Some(f) if !self.renderable(f) => self.skip("color clear: format not renderable here"),
                Some(_) if *mask != 0xf => self.skip("color clear: partial channel mask"),
                Some(_) if !covers(s) => self.skip("color clear: scissor smaller than the target"),
                Some(format) => {
                    let view = self.texture(s, format);
                    let [r, g, b, a] = value.map(|v| f64::from(f32::from_bits(v)));
                    encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                        label: Some("clear"),
                        color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                            view: &view,
                            depth_slice: None,
                            resolve_target: None,
                            ops: wgpu::Operations { load: wgpu::LoadOp::Clear(wgpu::Color { r, g, b, a }), store: wgpu::StoreOp::Store },
                        })],
                        ..Default::default()
                    });
                    self.touched.insert((s.addr, s.format));
                }
            }
        }
        if let Some((s, z, stencil)) = &c.depth {
            match depth_format(s.format) {
                None => self.skip("depth clear: unsupported format"),
                Some(f) if !self.renderable(f) => self.skip("depth clear: format not renderable here"),
                Some(_) if !covers(s) => self.skip("depth clear: scissor smaller than the target"),
                Some(format) => {
                    let view = self.texture(s, format);
                    let depth_ops = Some(wgpu::Operations { load: z.map_or(wgpu::LoadOp::Load, |z| wgpu::LoadOp::Clear(f32::from_bits(z))), store: wgpu::StoreOp::Store });
                    let stencil_ops = format.has_stencil_aspect().then(|| wgpu::Operations { load: stencil.map_or(wgpu::LoadOp::Load, |v| wgpu::LoadOp::Clear(u32::from(v))), store: wgpu::StoreOp::Store });
                    encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                        label: Some("clear"),
                        depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment { view: &view, depth_ops, stencil_ops }),
                        ..Default::default()
                    });
                }
            }
        }
        self.queue.submit([encoder.finish()]);
    }

    /// Writes every color target touched since the last call as
    /// `{prefix}-{address}-{format}-{w}x{h}.png` in `dir`; returns how many.
    pub fn dump(&mut self, dir: &Path, prefix: &str) -> Result<usize, String> {
        std::fs::create_dir_all(dir).map_err(|e| format!("{}: {e}", dir.display()))?;
        let keys = std::mem::take(&mut self.touched);
        for key in &keys {
            let e = &self.map[key];
            let (w, h) = (e.surface.width.max(1), e.surface.height.max(1));
            let rgba = self.rgba(e)?;
            let name = format!("{prefix}-{:010x}-{:02x}-{w}x{h}.png", e.surface.addr, e.surface.format);
            crate::write_png(&dir.join(name), w, h, &rgba).map_err(|e| e.to_string())?;
        }
        Ok(keys.len())
    }

    /// Forgets which targets were written (a frame not dumped).
    pub fn untouch(&mut self) {
        self.touched.clear();
    }

    /// Layer 0 of a color target as tightly packed RGBA8.
    fn rgba(&self, e: &Entry) -> Result<Vec<u8>, String> {
        let (w, h) = (e.surface.width.max(1), e.surface.height.max(1));
        let out = self.device.create_texture(&wgpu::TextureDescriptor {
            label: Some("blit"),
            size: wgpu::Extent3d { width: w, height: h, depth_or_array_layers: 1 },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8Unorm,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
            view_formats: &[],
        });
        let src = e.texture.create_view(&wgpu::TextureViewDescriptor { dimension: Some(wgpu::TextureViewDimension::D2), array_layer_count: Some(1), ..Default::default() });
        let group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &self.blit.1,
            entries: &[wgpu::BindGroupEntry { binding: 0, resource: wgpu::BindingResource::TextureView(&src) }],
        });
        let row = w * 4;
        let padded = row.div_ceil(wgpu::COPY_BYTES_PER_ROW_ALIGNMENT) * wgpu::COPY_BYTES_PER_ROW_ALIGNMENT;
        let buffer = self.device.create_buffer(&wgpu::BufferDescriptor { label: Some("readback"), size: u64::from(padded * h), usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ, mapped_at_creation: false });
        let mut encoder = self.device.create_command_encoder(&Default::default());
        {
            let view = out.create_view(&Default::default());
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("blit"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment { view: &view, depth_slice: None, resolve_target: None, ops: wgpu::Operations { load: wgpu::LoadOp::Clear(wgpu::Color::BLACK), store: wgpu::StoreOp::Store } })],
                ..Default::default()
            });
            pass.set_pipeline(&self.blit.0);
            pass.set_bind_group(0, &group, &[]);
            pass.draw(0..3, 0..1);
        }
        encoder.copy_texture_to_buffer(
            out.as_image_copy(),
            wgpu::TexelCopyBufferInfo { buffer: &buffer, layout: wgpu::TexelCopyBufferLayout { offset: 0, bytes_per_row: Some(padded), rows_per_image: Some(h) } },
            out.size(),
        );
        self.queue.submit([encoder.finish()]);
        let slice = buffer.slice(..);
        slice.map_async(wgpu::MapMode::Read, |r| r.expect("readback mapping failed"));
        self.device.poll(wgpu::PollType::wait_indefinitely()).map_err(|e| format!("GPU poll failed: {e}"))?;
        let mapped = slice.get_mapped_range().map_err(|e| format!("readback mapping failed: {e:?}"))?;
        let mut pixels = Vec::with_capacity((row * h) as usize);
        for y in 0..h {
            let start = (y * padded) as usize;
            pixels.extend_from_slice(&mapped[start..start + row as usize]);
        }
        Ok(pixels)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A clear reaches the texture: 4x2 targets in RF16_GF16_BF16_AF16 and in
    /// R16_G16_B16_A16 (the game's violet target) cleared to (0.5, 0, 0.5, 0)
    /// read back as opaque violet. Needs a GPU adapter.
    #[test]
    fn a_clear_reads_back_as_its_color() {
        let Ok(mut s) = Surfaces::headless(crate::Backend::Auto) else {
            eprintln!("no GPU adapter: test skipped");
            return;
        };
        let half = 0.5f32.to_bits();
        for format in [0xca, 0xc6] {
            let target = Surface { addr: 0x81_0000_2000, width: 4, height: 2, format, layers: 1 };
            s.clear(&Clear { layer: 0, color: Some((target, [half, 0, half, 0], 0xf)), depth: None, scissor: None });
            assert!(s.skipped.is_empty(), "format {format:#x}: {:?}", s.skipped);
            let rgba = s.rgba(&s.map[&(target.addr, target.format)]).unwrap();
            assert_eq!(rgba.len(), 4 * 2 * 4);
            assert!(rgba.chunks(4).all(|p| p[0].abs_diff(128) <= 1 && p[1] == 0 && p[2].abs_diff(128) <= 1 && p[3] == 255), "format {format:#x}: {rgba:?}");
        }
    }
}
