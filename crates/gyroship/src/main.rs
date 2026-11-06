//! Gyroship executable. In M0 it runs a placeholder test scene to prove the
//! visual verification loop: headless frame dumps, a real window that can be
//! captured, fixed time step, input replay and a log file.

use clap::Parser;
use gyroship_gpu::{Backend, HEIGHT, Renderer, WIDTH, write_png};
use std::fs::File;
use std::io::Write;
use std::path::PathBuf;
use std::sync::Arc;
use winit::application::ApplicationHandler;
use winit::event::WindowEvent;
use winit::event_loop::{ActiveEventLoop, ControlFlow, EventLoop};
use winit::window::{Window, WindowId};

#[derive(Parser)]
#[command(name = "gyroship", version, about = "Gyroship runtime (M0: placeholder test scene)")]
struct Args {
    /// Render off screen, without a window
    #[arg(long)]
    headless: bool,
    /// Stop after this many frames (default: run until the window closes; 60 when headless)
    #[arg(long)]
    frames: Option<u64>,
    /// Write rendered frames as PNG files into this folder (headless only)
    #[arg(long, value_name = "DIR")]
    dump_frames: Option<PathBuf>,
    /// Dump one frame out of K
    #[arg(long, value_name = "K", default_value_t = 1)]
    dump_every: u64,
    /// Recorded inputs, one line per frame; the time step is fixed either way
    #[arg(long, value_name = "FILE")]
    replay: Option<PathBuf>,
    /// Also write the log to this file
    #[arg(long, value_name = "FILE")]
    log: Option<PathBuf>,
    /// GPU backend
    #[arg(long, value_enum, default_value_t = BackendArg::Auto)]
    backend: BackendArg,
}

#[derive(Clone, Copy, clap::ValueEnum)]
enum BackendArg {
    Auto,
    Vulkan,
    Dx12,
}

impl From<BackendArg> for Backend {
    fn from(b: BackendArg) -> Backend {
        match b {
            BackendArg::Auto => Backend::Auto,
            BackendArg::Vulkan => Backend::Vulkan,
            BackendArg::Dx12 => Backend::Dx12,
        }
    }
}

struct Log(Option<File>);

impl Log {
    fn line(&mut self, msg: &str) {
        eprintln!("{msg}");
        if let Some(f) = &mut self.0 {
            let _ = writeln!(f, "{msg}");
        }
    }
}

/// Per-frame inputs from a replay file (blank lines and `#` comments skipped).
fn load_replay(path: &PathBuf) -> Result<Vec<String>, String> {
    let text = std::fs::read_to_string(path).map_err(|e| format!("{}: {e}", path.display()))?;
    Ok(text.lines().map(str::trim).filter(|l| !l.is_empty() && !l.starts_with('#')).map(String::from).collect())
}

fn main() -> std::process::ExitCode {
    let args = Args::parse();
    let mut log = Log(None);
    if let Some(path) = &args.log {
        match File::create(path) {
            Ok(f) => log.0 = Some(f),
            Err(e) => {
                eprintln!("error: cannot create log {}: {e}", path.display());
                return std::process::ExitCode::from(2);
            }
        }
    }
    let result = if args.headless { run_headless(&args, &mut log) } else { run_window(&args, &mut log) };
    match result {
        Ok(()) => std::process::ExitCode::SUCCESS,
        Err(e) => {
            log.line(&format!("error: {e}"));
            std::process::ExitCode::from(1)
        }
    }
}

fn run_headless(args: &Args, log: &mut Log) -> Result<(), String> {
    let replay = args.replay.as_ref().map(load_replay).transpose()?;
    let frames = args.frames.or(replay.as_ref().map(|r| r.len() as u64)).unwrap_or(60);
    let renderer = Renderer::headless(args.backend.into())?;
    log.line(&format!("headless: {} frames on {}", frames, renderer.describe()));
    if let Some(dir) = &args.dump_frames {
        std::fs::create_dir_all(dir).map_err(|e| format!("{}: {e}", dir.display()))?;
    }
    for frame in 0..frames {
        if let Some(input) = replay.as_ref().and_then(|r| r.get(frame as usize)) {
            log.line(&format!("frame {frame}: input {input}"));
        }
        let pixels = renderer.render_rgba(frame)?;
        if let Some(dir) = &args.dump_frames
            && frame % args.dump_every.max(1) == 0
        {
            let path = dir.join(format!("frame_{frame:06}.png"));
            write_png(&path, WIDTH, HEIGHT, &pixels).map_err(|e| format!("{}: {e}", path.display()))?;
            log.line(&format!("frame {frame}: wrote {}", path.display()));
        }
    }
    log.line("headless: done");
    Ok(())
}

struct App<'a> {
    args: &'a Args,
    log: &'a mut Log,
    state: Option<WindowState>,
    frame: u64,
    error: Option<String>,
}

struct WindowState {
    window: Arc<Window>,
    surface: wgpu::Surface<'static>,
    config: wgpu::SurfaceConfiguration,
    renderer: Renderer,
}

impl App<'_> {
    fn open(&mut self, event_loop: &ActiveEventLoop) -> Result<(), String> {
        let attributes = Window::default_attributes()
            .with_title("Gyroship")
            .with_inner_size(winit::dpi::PhysicalSize::new(WIDTH, HEIGHT));
        let window = Arc::new(event_loop.create_window(attributes).map_err(|e| e.to_string())?);
        let instance = Renderer::instance_for(self.args.backend.into());
        let surface = instance.create_surface(window.clone()).map_err(|e| e.to_string())?;
        let size = window.inner_size();
        // Probe the surface's preferred format with a throwaway adapter request.
        let probe = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            compatible_surface: Some(&surface),
            ..Default::default()
        }))
        .map_err(|e| e.to_string())?;
        let format = surface.get_capabilities(&probe).formats[0];
        let renderer = Renderer::new(instance, Some(&surface), format)?;
        let config = surface
            .get_default_config(&renderer.adapter, size.width.max(1), size.height.max(1))
            .ok_or("surface not supported by the adapter")?;
        surface.configure(&renderer.device, &config);
        self.log.line(&format!("window: {}x{} {:?} on {}", config.width, config.height, config.format, renderer.describe()));
        window.request_redraw();
        self.state = Some(WindowState { window, surface, config, renderer });
        Ok(())
    }

    fn redraw(&mut self, event_loop: &ActiveEventLoop) {
        let Some(s) = &mut self.state else { return };
        let texture = match s.surface.get_current_texture() {
            wgpu::CurrentSurfaceTexture::Success(t) | wgpu::CurrentSurfaceTexture::Suboptimal(t) => t,
            other => {
                self.log.line(&format!("frame {}: surface unavailable ({other:?}), reconfiguring", self.frame));
                s.surface.configure(&s.renderer.device, &s.config);
                s.window.request_redraw();
                return;
            }
        };
        s.renderer.draw(&texture.texture.create_view(&Default::default()), self.frame);
        s.window.pre_present_notify();
        s.renderer.queue.present(texture);
        self.frame += 1;
        if self.args.frames.is_some_and(|n| self.frame >= n) {
            self.log.line(&format!("window: {} frames shown, exiting", self.frame));
            event_loop.exit();
        } else {
            s.window.request_redraw();
        }
    }
}

impl ApplicationHandler for App<'_> {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if self.state.is_none()
            && let Err(e) = self.open(event_loop)
        {
            self.error = Some(e);
            event_loop.exit();
        }
    }

    fn window_event(&mut self, event_loop: &ActiveEventLoop, _id: WindowId, event: WindowEvent) {
        match event {
            WindowEvent::CloseRequested => event_loop.exit(),
            WindowEvent::Resized(size) => {
                if let Some(s) = &mut self.state
                    && size.width > 0
                    && size.height > 0
                {
                    s.config.width = size.width;
                    s.config.height = size.height;
                    s.surface.configure(&s.renderer.device, &s.config);
                }
            }
            WindowEvent::RedrawRequested => self.redraw(event_loop),
            _ => {}
        }
    }
}

fn run_window(args: &Args, log: &mut Log) -> Result<(), String> {
    if args.dump_frames.is_some() {
        return Err("--dump-frames needs --headless; capture a window with gyroship-tools capture".into());
    }
    let event_loop = EventLoop::new().map_err(|e| e.to_string())?;
    event_loop.set_control_flow(ControlFlow::Poll);
    let mut app = App { args, log, state: None, frame: 0, error: None };
    event_loop.run_app(&mut app).map_err(|e| e.to_string())?;
    app.error.map_or(Ok(()), Err)
}
