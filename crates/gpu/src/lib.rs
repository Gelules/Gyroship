//! Rendering on wgpu. For M0 it draws a placeholder test scene (a spinning
//! triangle whose angle and colours depend only on the frame number), so the
//! visual verification loop can be proven before the game renders anything.

use std::path::Path;

pub mod datazip;
pub mod mme;
pub mod pushbuf;
pub mod sass;
pub mod shader;
pub mod state;
pub mod surfaces;

pub const WIDTH: u32 = 640;
pub const HEIGHT: u32 = 360;

const SHADER: &str = r#"
struct Scene { angle: f32, tint: f32, _pad: vec2<f32> };
@group(0) @binding(0) var<uniform> scene: Scene;

struct Out { @builtin(position) pos: vec4<f32>, @location(0) color: vec3<f32> };

@vertex
fn vs(@builtin(vertex_index) i: u32) -> Out {
    let corners = array<vec2<f32>, 3>(vec2(0.0, 0.7), vec2(-0.6, -0.5), vec2(0.6, -0.5));
    let colors = array<vec3<f32>, 3>(vec3(1.0, 0.3, 0.2), vec3(0.2, 0.9, 0.4), vec3(0.3, 0.4, 1.0));
    let c = cos(scene.angle);
    let s = sin(scene.angle);
    let p = corners[i];
    var out: Out;
    out.pos = vec4(vec2(p.x * c - p.y * s, p.x * s + p.y * c) * vec2(0.5625, 1.0), 0.0, 1.0);
    out.color = mix(colors[i], vec3(1.0, 1.0, 0.6), scene.tint);
    return out;
}

@fragment
fn fs(in: Out) -> @location(0) vec4<f32> {
    return vec4(in.color, 1.0);
}
"#;

/// Which wgpu backend to use; Auto tries Vulkan then DX12.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Backend {
    Auto,
    Vulkan,
    Dx12,
}

impl Backend {
    fn backends(self) -> wgpu::Backends {
        match self {
            Backend::Auto => wgpu::Backends::VULKAN | wgpu::Backends::DX12,
            Backend::Vulkan => wgpu::Backends::VULKAN,
            Backend::Dx12 => wgpu::Backends::DX12,
        }
    }
}

pub struct Renderer {
    pub instance: wgpu::Instance,
    pub adapter: wgpu::Adapter,
    pub device: wgpu::Device,
    pub queue: wgpu::Queue,
    format: wgpu::TextureFormat,
    pipeline: wgpu::RenderPipeline,
    uniform: wgpu::Buffer,
    bind_group: wgpu::BindGroup,
}

impl Renderer {
    /// Creates a renderer whose output format is `format`; `surface`, when
    /// given, constrains the adapter to one that can present to it.
    pub fn new(instance: wgpu::Instance, surface: Option<&wgpu::Surface<'_>>, format: wgpu::TextureFormat) -> Result<Renderer, String> {
        let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            compatible_surface: surface,
            force_fallback_adapter: false,
            apply_limit_buckets: false,
        }))
        .map_err(|e| format!("no suitable GPU adapter: {e}"))?;
        let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor::default()))
            .map_err(|e| format!("cannot open the GPU device: {e}"))?;
        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("test scene"),
            source: wgpu::ShaderSource::Wgsl(SHADER.into()),
        });
        let uniform = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("scene"),
            size: 16,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: None,
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::VERTEX,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            }],
        });
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &layout,
            entries: &[wgpu::BindGroupEntry { binding: 0, resource: uniform.as_entire_binding() }],
        });
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: None,
            bind_group_layouts: &[Some(&layout)],
            ..Default::default()
        });
        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("test scene"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState { module: &module, entry_point: Some("vs"), compilation_options: Default::default(), buffers: &[] },
            fragment: Some(wgpu::FragmentState {
                module: &module,
                entry_point: Some("fs"),
                compilation_options: Default::default(),
                targets: &[Some(format.into())],
            }),
            primitive: Default::default(),
            depth_stencil: None,
            multisample: Default::default(),
            multiview_mask: None,
            cache: None,
        });
        Ok(Renderer { instance, adapter, device, queue, format, pipeline, uniform, bind_group })
    }

    /// Headless renderer drawing into an RGBA8 texture.
    pub fn headless(backend: Backend) -> Result<Renderer, String> {
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor { backends: backend.backends(), ..wgpu::InstanceDescriptor::new_without_display_handle() });
        Renderer::new(instance, None, wgpu::TextureFormat::Rgba8UnormSrgb)
    }

    pub fn instance_for(backend: Backend) -> wgpu::Instance {
        wgpu::Instance::new(wgpu::InstanceDescriptor { backends: backend.backends(), ..wgpu::InstanceDescriptor::new_without_display_handle() })
    }

    /// One-line description of the GPU and backend in use, for logs.
    pub fn describe(&self) -> String {
        let info = self.adapter.get_info();
        format!("{} ({:?}, driver {} {})", info.name, info.backend, info.driver, info.driver_info)
    }

    /// Draws frame `frame` of the test scene into `view`.
    pub fn draw(&self, view: &wgpu::TextureView, frame: u64) {
        let angle = frame as f32 * 0.05;
        let tint = (frame % 120) as f32 / 240.0;
        let mut bytes = [0u8; 16];
        bytes[0..4].copy_from_slice(&angle.to_le_bytes());
        bytes[4..8].copy_from_slice(&tint.to_le_bytes());
        self.queue.write_buffer(&self.uniform, 0, &bytes);
        let mut encoder = self.device.create_command_encoder(&Default::default());
        {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("test scene"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view,
                    depth_slice: None,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color { r: 0.02, g: 0.03, b: 0.08 + 0.1 * (frame % 60) as f64 / 60.0, a: 1.0 }),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                ..Default::default()
            });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &self.bind_group, &[]);
            pass.draw(0..3, 0..1);
        }
        self.queue.submit([encoder.finish()]);
    }

    /// Renders frame `frame` off screen and returns tightly packed RGBA8 pixels.
    pub fn render_rgba(&self, frame: u64) -> Result<Vec<u8>, String> {
        let texture = self.device.create_texture(&wgpu::TextureDescriptor {
            label: Some("offscreen"),
            size: wgpu::Extent3d { width: WIDTH, height: HEIGHT, depth_or_array_layers: 1 },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: self.format,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
            view_formats: &[],
        });
        self.draw(&texture.create_view(&Default::default()), frame);
        let row = WIDTH * 4;
        let padded = row.div_ceil(wgpu::COPY_BYTES_PER_ROW_ALIGNMENT) * wgpu::COPY_BYTES_PER_ROW_ALIGNMENT;
        let buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("readback"),
            size: u64::from(padded * HEIGHT),
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        let mut encoder = self.device.create_command_encoder(&Default::default());
        encoder.copy_texture_to_buffer(
            texture.as_image_copy(),
            wgpu::TexelCopyBufferInfo {
                buffer: &buffer,
                layout: wgpu::TexelCopyBufferLayout { offset: 0, bytes_per_row: Some(padded), rows_per_image: Some(HEIGHT) },
            },
            texture.size(),
        );
        self.queue.submit([encoder.finish()]);
        let slice = buffer.slice(..);
        slice.map_async(wgpu::MapMode::Read, |r| r.expect("readback mapping failed"));
        self.device.poll(wgpu::PollType::wait_indefinitely()).map_err(|e| format!("GPU poll failed: {e}"))?;
        let mapped = slice.get_mapped_range().map_err(|e| format!("readback mapping failed: {e:?}"))?;
        let mut pixels = Vec::with_capacity((row * HEIGHT) as usize);
        for y in 0..HEIGHT {
            let start = (y * padded) as usize;
            pixels.extend_from_slice(&mapped[start..start + row as usize]);
        }
        Ok(pixels)
    }
}

/// Writes RGBA8 pixels as a PNG file.
pub fn write_png(path: &Path, width: u32, height: u32, rgba: &[u8]) -> std::io::Result<()> {
    let file = std::io::BufWriter::new(std::fs::File::create(path)?);
    let mut encoder = png::Encoder::new(file, width, height);
    encoder.set_color(png::ColorType::Rgba);
    encoder.set_depth(png::BitDepth::Eight);
    let mut writer = encoder.write_header().map_err(std::io::Error::other)?;
    writer.write_image_data(rgba).map_err(std::io::Error::other)?;
    writer.finish().map_err(std::io::Error::other)
}
