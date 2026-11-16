//! Gyroship's host side (D28): the game's window and its pace; input and
//! sound come next. The generated harness calls [`main`]. `boot --headless`
//! (tests, reference runs) runs without a window, as before.

#[cfg(feature = "mimalloc")]
#[global_allocator]
static ALLOCATOR: mimalloc::MiMalloc = mimalloc::MiMalloc;

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
use winit::window::{Fullscreen, Window, WindowId};

/// The window's client size at start: the console's 1080p output.
const WIDTH: u32 = 1920;
const HEIGHT: u32 = 1080;

/// The harness's entry point: `boot <exefs> ...` in a window, unless
/// `--headless`; the runtime's own modes otherwise.
pub fn main(table: Vec<Entry>) {
    let args: Vec<String> = std::env::args().collect();
    if args.get(1).map(String::as_str) == Some("boot") {
        if args.iter().any(|a| a == "--help") {
            return print!("{}", gyroship_hle::boot::usage());
        }
        if let Err(e) = gyroship_hle::boot::check_options(&args[1..]) {
            fail(&e);
        }
    }
    if args.get(1).map(String::as_str) != Some("boot") || args.iter().any(|a| a == "--headless") {
        return gyroship_hle::boot::main(table);
    }
    let event_loop = EventLoop::new().unwrap_or_else(|e| fail(&format!("cannot open a window: {e}")));
    event_loop.set_control_flow(ControlFlow::Wait);
    let mut app = App { table: Some(table), shared: Arc::new(Shared::default()), window: None, pads: None, alt: false };
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
    /// The same for the game's keyboard: one bit per USB HID usage.
    keys_held: [AtomicU64; 4],
    keys_pressed: [AtomicU64; 4],
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
    /// Alt is held: Alt+Enter switches between the window and fullscreen.
    alt: bool,
}

/// Borderless fullscreen on the window's monitor, or back to the window; the mouse
/// cursor shows only in the window.
fn set_fullscreen(window: &Window, on: bool) {
    window.set_fullscreen(on.then_some(Fullscreen::Borderless(None)));
    window.set_cursor_visible(!on);
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
        if std::env::args().any(|a| a == "--fullscreen") {
            set_fullscreen(&window, true);
        }
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
            // The guest ends the run at its next frame (its journal flushed). With no
            // frame 2 s later (a long loading, a frozen game), it stops from here,
            // threads' states, journal, reports and --record written, unless the
            // kernel stays busy. Clicks after the first change nothing.
            WindowEvent::CloseRequested => {
                if self.shared.closed.swap(true, Ordering::Relaxed) {
                    return;
                }
                eprintln!("gyroship: window closed: the run stops at its next frame, or within 5 s");
                std::thread::spawn(|| {
                    std::thread::sleep(Duration::from_secs(2));
                    gyroship_hle::stop_from_outside("window closed while no frame came", gyroship_hle::CLOSE_WAIT);
                    eprintln!("gyroship: window closed while no frame came, and the kernel stayed busy: nothing written");
                    std::process::exit(0);
                });
            }
            WindowEvent::Resized(s) => self.shared.size.store(u64::from(s.width) << 32 | u64::from(s.height), Ordering::Relaxed),
            // A key drives the controller (`key_bit`) and, for the game's debug
            // tools, the keyboard (`hid_usage`), which the game only reads when
            // its keyboard setting is on.
            WindowEvent::ModifiersChanged(m) => self.alt = m.state().alt_key(),
            WindowEvent::KeyboardInput { event, .. } if !event.repeat => {
                let PhysicalKey::Code(code) = event.physical_key else { return };
                let (s, down) = (&self.shared, event.state == ElementState::Pressed);
                // Alt+Enter: fullscreen or the window. Enter is the controller's A: the
                // game never sees this one.
                if code == KeyCode::Enter && self.alt {
                    if let (true, Some(w)) = (down, &self.window) {
                        set_fullscreen(w, w.fullscreen().is_none());
                    }
                    return;
                }
                // G, as the controller's Guide: --boost perfect off for full boosts, and on
                // again. The game never sees it.
                if code == KeyCode::KeyG {
                    if down {
                        gyroship_hle::practice::toggle_perfect();
                    }
                    return;
                }
                if let Some(bit) = key_bit(code) {
                    set(&s.held, &s.pressed, bit, down);
                }
                if let Some(u) = hid_usage(code) {
                    let w = usize::from(u / 64);
                    set(&s.keys_held[w], &s.keys_pressed[w], u32::from(u % 64), down);
                }
            }
            // Keys released while another window has the focus never come: let go of all.
            WindowEvent::Focused(false) => {
                self.shared.held.store(0, Ordering::Relaxed);
                self.shared.keys_held.iter().for_each(|w| w.store(0, Ordering::Relaxed));
            }
            _ => {}
        }
    }
}

/// D27: virtual time leads, the host only waits. Each frame waits until its
/// virtual time on the wall clock; more than a frame late, the origin moves
/// to now (the game slows down, it never skips or catches up). With `--play`
/// guest time is the wall clock already: nothing to wait for, and this runs
/// under the GPU side's lock, which must never sleep.
fn pace(shared: Arc<Shared>) -> gyroship_hle::Pace {
    let play = std::env::args().any(|a| a == "--play");
    let mut origin: Option<(Instant, u64)> = None;
    Box::new(move |ticks| {
        if shared.closed.load(Ordering::Relaxed) {
            return None;
        }
        if play {
            return Some(shared.size());
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

/// A key going down (held, and pressed since the last sample) or up.
fn set(held: &AtomicU64, pressed: &AtomicU64, bit: u32, down: bool) {
    if down {
        held.fetch_or(1 << bit, Ordering::Relaxed);
        pressed.fetch_or(1 << bit, Ordering::Relaxed);
    } else {
        held.fetch_and(!(1 << bit), Ordering::Relaxed);
    }
}

/// The USB HID usage of a key (keyboard page), for the game's keyboard,
/// which only its debug tools read: the console (backquote, F1 to F12, Ctrl,
/// Shift) and its text fields.
// ponytail: no numpad nor rare keys; add them if a debug window needs them.
fn hid_usage(code: KeyCode) -> Option<u8> {
    use KeyCode::*;
    const LETTERS: [KeyCode; 26] = [KeyA, KeyB, KeyC, KeyD, KeyE, KeyF, KeyG, KeyH, KeyI, KeyJ, KeyK, KeyL, KeyM, KeyN, KeyO, KeyP, KeyQ, KeyR, KeyS, KeyT, KeyU, KeyV, KeyW, KeyX, KeyY, KeyZ];
    const DIGITS: [KeyCode; 10] = [Digit1, Digit2, Digit3, Digit4, Digit5, Digit6, Digit7, Digit8, Digit9, Digit0];
    const FUNCTIONS: [KeyCode; 12] = [F1, F2, F3, F4, F5, F6, F7, F8, F9, F10, F11, F12];
    for (keys, first) in [(&LETTERS[..], 0x04), (&DIGITS[..], 0x1e), (&FUNCTIONS[..], 0x3a)] {
        if let Some(i) = keys.iter().position(|&k| k == code) {
            return Some(first + i as u8);
        }
    }
    Some(match code {
        Enter => 0x28,
        Escape => 0x29,
        Backspace => 0x2a,
        Tab => 0x2b,
        Space => 0x2c,
        Minus => 0x2d,
        Equal => 0x2e,
        BracketLeft => 0x2f,
        BracketRight => 0x30,
        Backslash => 0x31,
        Semicolon => 0x33,
        Quote => 0x34,
        Backquote => 0x35,
        Comma => 0x36,
        Period => 0x37,
        Slash => 0x38,
        CapsLock => 0x39,
        Insert => 0x49,
        Home => 0x4a,
        PageUp => 0x4b,
        Delete => 0x4c,
        End => 0x4d,
        PageDown => 0x4e,
        ArrowRight => 0x4f,
        ArrowLeft => 0x50,
        ArrowDown => 0x51,
        ArrowUp => 0x52,
        IntlBackslash => 0x64,
        ControlLeft => 0xe0,
        ShiftLeft => 0xe1,
        AltLeft => 0xe2,
        SuperLeft => 0xe3,
        ControlRight => 0xe4,
        ShiftRight => 0xe5,
        AltRight => 0xe6,
        SuperRight => 0xe7,
        _ => return None,
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
    gyroship_hle::hid::Pad { buttons, l: [x, y], ..Default::default() }
}

/// The live state for each hid sample: what is held, plus what was pressed
/// since the last sample.
fn input(shared: Arc<Shared>) -> gyroship_hle::LiveInput {
    Box::new(move || {
        let mut keys = pad_of(shared.held.load(Ordering::Relaxed) | shared.pressed.swap(0, Ordering::Relaxed));
        for (w, (held, pressed)) in shared.keys_held.iter().zip(&shared.keys_pressed).enumerate() {
            keys.keys[w] = held.load(Ordering::Relaxed) | pressed.swap(0, Ordering::Relaxed);
        }
        let p = &shared.pads;
        let (l, r) = p.sticks();
        pads::merge(keys, p.held.load(Ordering::Relaxed) | p.pressed.swap(0, Ordering::Relaxed), l, r)
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_option_the_host_reads_is_listed() {
        let code = include_str!("lib.rs").split("#[cfg(test)]").next().unwrap();
        let read = gyroship_hle::boot::literals(code);
        assert!(read.contains(&"--fullscreen") && read.contains(&"--sdl"), "{read:?}");
        for o in read {
            assert!(gyroship_hle::boot::OPTIONS.iter().any(|(n, _)| *n == o), "{o} is read but not listed");
        }
    }

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
        set(&shared.held, &shared.pressed, 0, true);
        set(&shared.held, &shared.pressed, 0, false);
        set(&shared.keys_held[0], &shared.keys_pressed[0], 0x35, true);
        set(&shared.keys_held[0], &shared.keys_pressed[0], 0x35, false);
        let p = sample();
        assert_eq!((p.buttons, p.keys), (1, [1 << 0x35, 0, 0, 0]));
        let p = sample();
        assert_eq!((p.buttons, p.keys), (0, [0; 4]));
    }

    /// The debug console's keys, and a few others, by their USB HID usages.
    #[test]
    fn keys_have_their_usb_usages() {
        let u = |c| hid_usage(c).unwrap();
        assert_eq!([u(KeyCode::Backquote), u(KeyCode::F1), u(KeyCode::F11), u(KeyCode::F12)], [0x35, 0x3a, 0x44, 0x45]);
        assert_eq!([u(KeyCode::ControlLeft), u(KeyCode::ShiftLeft), u(KeyCode::ControlRight), u(KeyCode::ShiftRight)], [0xe0, 0xe1, 0xe4, 0xe5]);
        assert_eq!([u(KeyCode::KeyA), u(KeyCode::KeyZ), u(KeyCode::Digit1), u(KeyCode::Digit0), u(KeyCode::Enter)], [0x04, 0x1d, 0x1e, 0x27, 0x28]);
        assert_eq!(hid_usage(KeyCode::Numpad5), None);
    }
}
