//! Gyroship's host side (D28): the game's window and its pace; input and
//! sound come next. The generated harness calls [`main`]. `boot --headless`
//! (tests, reference runs) runs without a window, as before.

use gyroship_hle::boot::HostWindow;
use gyroship_runtime::Entry;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::time::{Duration, Instant};
use winit::application::ApplicationHandler;
use winit::event::WindowEvent;
use winit::event_loop::{ActiveEventLoop, ControlFlow, EventLoop};
use winit::window::{Window, WindowId};

/// The window's client size at start: the console's 1080p output.
const WIDTH: u32 = 1920;
const HEIGHT: u32 = 1080;

/// The harness's entry point: `boot <exefs> ...` in a window, unless
/// `--headless`; the runtime's own modes otherwise.
pub fn main(table: Vec<Entry>) {
    let args: Vec<String> = std::env::args().collect();
    if args.get(1).map(String::as_str) != Some("boot") || args.iter().any(|a| a == "--headless") {
        return gyroship_hle::boot::main(table);
    }
    let event_loop = EventLoop::new().unwrap_or_else(|e| fail(&format!("cannot open a window: {e}")));
    event_loop.set_control_flow(ControlFlow::Wait);
    let mut app = App { table: Some(table), shared: Arc::new(Shared::default()), window: None };
    if let Err(e) = event_loop.run_app(&mut app) {
        fail(&format!("window: {e}"));
    }
}

/// `--backend auto|vulkan|dx12` (the HLE checks the spelling).
fn backend() -> gyroship_gpu::Backend {
    let args: Vec<String> = std::env::args().collect();
    match args.iter().position(|a| a == "--backend").and_then(|i| args.get(i + 1)).map(String::as_str) {
        Some("vulkan") => gyroship_gpu::Backend::Vulkan,
        Some("dx12") => gyroship_gpu::Backend::Dx12,
        _ => gyroship_gpu::Backend::Auto,
    }
}

fn fail(why: &str) -> ! {
    eprintln!("gyroship: {why}");
    std::process::exit(2);
}

/// What the window thread tells the guest thread.
#[derive(Default)]
struct Shared {
    closed: AtomicBool,
    /// Client size, width << 32 | height.
    size: AtomicU64,
}

impl Shared {
    fn size(&self) -> (u32, u32) {
        let s = self.size.load(Ordering::Relaxed);
        ((s >> 32) as u32, s as u32)
    }
}

struct App {
    table: Option<Vec<Entry>>,
    shared: Arc<Shared>,
    window: Option<Arc<Window>>,
}

impl ApplicationHandler for App {
    /// The window opens, then the guest starts on a thread of its own; the
    /// window thread only relays events.
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        let Some(table) = self.table.take() else { return };
        let attributes = Window::default_attributes().with_title("Gyroship").with_inner_size(winit::dpi::PhysicalSize::new(WIDTH, HEIGHT));
        let window = Arc::new(event_loop.create_window(attributes).unwrap_or_else(|e| fail(&format!("cannot open a window: {e}"))));
        let size = window.inner_size();
        self.shared.size.store(u64::from(size.width) << 32 | u64::from(size.height), Ordering::Relaxed);
        // The surface comes from this thread: winit gives window handles only here.
        let instance = gyroship_gpu::Renderer::instance_for(backend());
        let surface = instance.create_surface(window.clone()).unwrap_or_else(|e| fail(&format!("cannot draw in the window: {e}")));
        let host = HostWindow { instance, surface, size: (size.width, size.height), pace: pace(self.shared.clone()) };
        // The guest's own fibers carry its code; this thread only needs room for the HLE.
        std::thread::Builder::new().name("guest".into()).stack_size(64 << 20).spawn(move || gyroship_hle::boot::run(table, Some(host))).unwrap_or_else(|e| fail(&format!("cannot start the guest: {e}")));
        self.window = Some(window);
    }

    fn window_event(&mut self, _event_loop: &ActiveEventLoop, _id: WindowId, event: WindowEvent) {
        match event {
            // The guest ends the run at its next frame (its journal flushed); one
            // busy loading ends here 2 s later.
            WindowEvent::CloseRequested => {
                self.shared.closed.store(true, Ordering::Relaxed);
                std::thread::spawn(|| {
                    std::thread::sleep(Duration::from_secs(2));
                    eprintln!("gyroship: window closed while no frame came");
                    std::process::exit(0);
                });
            }
            WindowEvent::Resized(s) => self.shared.size.store(u64::from(s.width) << 32 | u64::from(s.height), Ordering::Relaxed),
            _ => {}
        }
    }
}

/// D27: virtual time leads, the host only waits. Each frame waits until its
/// virtual time on the wall clock; more than a frame late, the origin moves
/// to now (the game slows down, it never skips or catches up).
fn pace(shared: Arc<Shared>) -> gyroship_hle::Pace {
    let mut origin: Option<(Instant, u64)> = None;
    Box::new(move |ticks| {
        if shared.closed.load(Ordering::Relaxed) {
            return None;
        }
        let now = Instant::now();
        match origin {
            Some((t0, k0)) if ticks >= k0 => {
                // A tick is 32 ns (the 31.25 MHz counter).
                let target = t0 + Duration::from_nanos((ticks - k0) * 32);
                if now < target {
                    std::thread::sleep(target - now);
                } else if now - target > Duration::from_micros(16_667) {
                    origin = Some((now, ticks));
                }
            }
            _ => origin = Some((now, ticks)),
        }
        Some(shared.size())
    })
}
