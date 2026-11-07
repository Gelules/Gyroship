//! Gyroship's host side (D28): the game's window and its pace; input and
//! sound come next. The generated harness calls [`main`]. `boot --headless`
//! (tests, reference runs) runs without a window, as before.

pub mod pads;
pub mod sdl;

use gyroship_hle::boot::HostWindow;
use gyroship_runtime::Entry;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::time::{Duration, Instant};
use winit::application::ApplicationHandler;
use winit::event::{ElementState, WindowEvent};
use winit::keyboard::{KeyCode, PhysicalKey};
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
    let mut app = App { table: Some(table), shared: Arc::new(Shared::default()), window: None, pads: None };
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
    /// Keys held, and keys pressed since the last hid sample (a short press
    /// between two samples still counts once), as `key_bit` bits.
    held: AtomicU64,
    pressed: AtomicU64,
    /// The gamepads' state, from SDL.
    pads: pads::Shared,
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
    /// SDL's gamepads, polled from this thread (SDL's rule), if SDL3.dll loads.
    pads: Option<pads::Pads>,
}

/// `--sdl FILE`, else SDL3.dll next to the executable.
fn sdl_path() -> std::path::PathBuf {
    let args: Vec<String> = std::env::args().collect();
    match args.iter().position(|a| a == "--sdl").and_then(|i| args.get(i + 1)) {
        Some(p) => p.into(),
        None => std::env::current_exe().ok().and_then(|e| Some(e.parent()?.join("SDL3.dll"))).unwrap_or_else(|| "SDL3.dll".into()),
    }
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
        // Controllers and sound: without SDL3, the keyboard alone and silence.
        let sdl = match sdl::Sdl::load(&sdl_path()) {
            Ok(s) => Some(Arc::new(s)),
            Err(e) => {
                eprintln!("gyroship: no SDL3 ({e}): keyboard only, no sound");
                None
            }
        };
        let audio = sdl.clone().and_then(|s| pads::audio(s, 2, 48_000));
        self.pads = sdl.map(pads::Pads::new);
        let host = HostWindow { instance, surface, size: (size.width, size.height), pace: pace(self.shared.clone()), input: input(self.shared.clone()), audio };
        // The guest's own fibers carry its code; this thread only needs room for the HLE.
        std::thread::Builder::new().name("guest".into()).stack_size(64 << 20).spawn(move || gyroship_hle::boot::run(table, Some(host))).unwrap_or_else(|e| fail(&format!("cannot start the guest: {e}")));
        self.window = Some(window);
    }

    /// SDL is polled every 4 ms (its events and gamepad states).
    fn about_to_wait(&mut self, event_loop: &ActiveEventLoop) {
        if let Some(p) = self.pads.as_mut() {
            p.poll(&self.shared.pads);
            event_loop.set_control_flow(ControlFlow::WaitUntil(Instant::now() + Duration::from_millis(4)));
        }
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
            WindowEvent::KeyboardInput { event, .. } if !event.repeat => {
                if let PhysicalKey::Code(code) = event.physical_key
                    && let Some(bit) = key_bit(code)
                {
                    match event.state {
                        ElementState::Pressed => {
                            self.shared.held.fetch_or(1 << bit, Ordering::Relaxed);
                            self.shared.pressed.fetch_or(1 << bit, Ordering::Relaxed);
                        }
                        ElementState::Released => {
                            self.shared.held.fetch_and(!(1 << bit), Ordering::Relaxed);
                        }
                    }
                }
            }
            // Keys released while another window has the focus never come: let go of all.
            WindowEvent::Focused(false) => self.shared.held.store(0, Ordering::Relaxed),
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

/// Bits of the left stick's keys, above the Npad buttons.
const STICK_UP: u32 = 48;
const STICK_LEFT: u32 = 49;
const STICK_DOWN: u32 = 50;
const STICK_RIGHT: u32 = 51;

/// The default keyboard layout, by physical key (US labels, so the same
/// places on any layout): the Npad button bit, or a stick key.
fn key_bit(code: KeyCode) -> Option<u32> {
    Some(match code {
        KeyCode::Enter => 0,      // A
        KeyCode::Backspace => 1,  // B
        KeyCode::KeyX => 2,       // X
        KeyCode::KeyY => 3,       // Y
        KeyCode::KeyQ => 6,       // L
        KeyCode::KeyE => 7,       // R
        KeyCode::Digit1 => 8,     // ZL
        KeyCode::Digit3 => 9,     // ZR
        KeyCode::KeyP => 10,      // Plus
        KeyCode::KeyM => 11,      // Minus
        KeyCode::ArrowLeft => 12,
        KeyCode::ArrowUp => 13,
        KeyCode::ArrowRight => 14,
        KeyCode::ArrowDown => 15,
        KeyCode::KeyW => STICK_UP,
        KeyCode::KeyA => STICK_LEFT,
        KeyCode::KeyS => STICK_DOWN,
        KeyCode::KeyD => STICK_RIGHT,
        _ => return None,
    })
}

/// The controller state of a set of `key_bit` bits: buttons, the left stick
/// at full deflection, and its direction bits (LSLeft..LSDown, 16..19).
fn pad_of(bits: u64) -> gyroship_hle::hid::Pad {
    let on = |b: u32| i32::from(bits >> b & 1 != 0);
    let (x, y) = ((on(STICK_RIGHT) - on(STICK_LEFT)) * 0x7fff, (on(STICK_UP) - on(STICK_DOWN)) * 0x7fff);
    let mut buttons = bits & 0xffff_ffff;
    for (moved, bit) in [(x < 0, 16), (y > 0, 17), (x > 0, 18), (y < 0, 19)] {
        if moved {
            buttons |= 1 << bit;
        }
    }
    gyroship_hle::hid::Pad { buttons, l: [x, y], r: [0, 0] }
}

/// The live state for each hid sample: what is held, plus what was pressed
/// since the last sample.
fn input(shared: Arc<Shared>) -> gyroship_hle::LiveInput {
    Box::new(move || {
        let keys = pad_of(shared.held.load(Ordering::Relaxed) | shared.pressed.swap(0, Ordering::Relaxed));
        let p = &shared.pads;
        let (l, r) = p.sticks();
        pads::merge(keys, p.held.load(Ordering::Relaxed) | p.pressed.swap(0, Ordering::Relaxed), l, r)
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn keys_make_buttons_and_a_full_stick() {
        let bit = |c| 1u64 << key_bit(c).unwrap();
        let p = pad_of(bit(KeyCode::Enter) | bit(KeyCode::ArrowUp) | bit(KeyCode::KeyW) | bit(KeyCode::KeyD));
        assert_eq!(p.buttons, 1 | 1 << 13 | 1 << 17 | 1 << 18);
        assert_eq!(p.l, [0x7fff, 0x7fff]);
        // Opposite keys cancel.
        assert_eq!(pad_of(bit(KeyCode::KeyA) | bit(KeyCode::KeyD)), gyroship_hle::hid::Pad::default());
    }

    /// A key pressed and released between two samples is seen by the next one, once.
    #[test]
    fn a_short_press_between_samples_counts_once() {
        let shared = Arc::new(Shared::default());
        let mut sample = input(shared.clone());
        shared.held.fetch_or(1, Ordering::Relaxed);
        shared.pressed.fetch_or(1, Ordering::Relaxed);
        shared.held.fetch_and(!1, Ordering::Relaxed);
        assert_eq!(sample().buttons, 1);
        assert_eq!(sample().buttons, 0);
    }
}
