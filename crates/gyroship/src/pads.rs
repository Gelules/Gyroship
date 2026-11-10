//! Controllers and sound through SDL3 (D30), on the window's thread: the
//! open gamepads, their state as Npad bits and sticks, and the game's
//! stream played on the default device.

use crate::sdl::{self, Gamepad, Sdl, Stream, button as b};
use gyroship_hle::hid::Pad;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

/// Buttons by position, as the console's (D30): A east, B south, X north, Y west.
const BUTTONS: [(i32, u32); 14] = [
    (b::EAST, 0),
    (b::SOUTH, 1),
    (b::NORTH, 2),
    (b::WEST, 3),
    (b::LEFT_STICK, 4),
    (b::RIGHT_STICK, 5),
    (b::LEFT_SHOULDER, 6),
    (b::RIGHT_SHOULDER, 7),
    (b::START, 10),
    (b::BACK, 11),
    (b::DPAD_LEFT, 12),
    (b::DPAD_UP, 13),
    (b::DPAD_RIGHT, 14),
    (b::DPAD_DOWN, 15),
];
/// ZL and ZR: a trigger past half its travel.
const TRIGGERS: [(i32, u32); 2] = [(sdl::axis::LEFT_TRIGGER, 8), (sdl::axis::RIGHT_TRIGGER, 9)];
/// XInput's own dead zone for the left stick, used for both.
const DEAD_ZONE: f32 = 7849.0;

/// The Npad bit of an SDL gamepad button.
pub fn bit_of(button: i32) -> Option<u32> {
    BUTTONS.iter().find(|(sdl, _)| *sdl == button).map(|(_, bit)| *bit)
}

/// A stick, its Y up, with a radial dead zone rescaled to the full range.
pub fn stick(x: i16, y: i16) -> [i32; 2] {
    let (x, y) = (f32::from(x), -f32::from(y));
    let magnitude = x.hypot(y);
    if magnitude <= DEAD_ZONE {
        return [0, 0];
    }
    let scale = ((magnitude - DEAD_ZONE) / (32767.0 - DEAD_ZONE)).min(1.0) * 32767.0 / magnitude;
    [((x * scale).round() as i32).clamp(-32767, 32767), ((y * scale).round() as i32).clamp(-32767, 32767)]
}

/// A stick's direction bits from `first` (left, up, right, down), past half deflection.
fn directions([x, y]: [i32; 2], first: u32) -> u64 {
    [(x < -16384, 0), (y > 16384, 1), (x > 16384, 2), (y < -16384, 3)].iter().filter(|(on, _)| *on).fold(0, |v, (_, i)| v | 1 << (first + i))
}

/// The keyboard's state merged with the gamepads': buttons or'ed, a stick
/// from the gamepads when they move it, with the direction bits of both.
pub fn merge(keys: Pad, buttons: u64, l: [i32; 2], r: [i32; 2]) -> Pad {
    let l = if l != [0, 0] { l } else { keys.l };
    let r = if r != [0, 0] { r } else { keys.r };
    Pad { buttons: keys.buttons | buttons | directions(l, 16) | directions(r, 20), l, r }
}

/// What the gamepads tell the guest thread: buttons held, buttons pressed
/// since the last sample, and both sticks (4 × i16).
#[derive(Default)]
pub struct Shared {
    pub held: AtomicU64,
    pub pressed: AtomicU64,
    pub sticks: AtomicU64,
}

impl Shared {
    pub fn sticks(&self) -> ([i32; 2], [i32; 2]) {
        let s = self.sticks.load(Ordering::Relaxed);
        let v = |i: u32| i32::from((s >> (16 * i)) as u16 as i16);
        ([v(0), v(1)], [v(2), v(3)])
    }
}

/// SDL and the gamepads it opened.
pub struct Pads {
    pub sdl: Arc<Sdl>,
    open: Vec<(u32, Gamepad)>,
}

impl Pads {
    pub fn new(sdl: Arc<Sdl>) -> Pads {
        Pads { sdl, open: Vec::new() }
    }

    /// Handles SDL's events (plugging, presses), then stores every open
    /// gamepad's state, combined, in `shared`.
    pub fn poll(&mut self, shared: &Shared) {
        while let Some(e) = self.sdl.poll() {
            match e.kind() {
                sdl::GAMEPAD_ADDED => {
                    if let Some(g) = self.sdl.open_gamepad(e.which()) {
                        eprintln!("gyroship: controller {} plugged: {}", e.which(), self.sdl.gamepad_name(g));
                        self.open.push((e.which(), g));
                    }
                }
                sdl::GAMEPAD_REMOVED => {
                    if let Some(i) = self.open.iter().position(|(id, _)| *id == e.which()) {
                        eprintln!("gyroship: controller {} unplugged", e.which());
                        self.sdl.close_gamepad(self.open.swap_remove(i).1);
                    }
                }
                // A press between two polls still reaches the next sample.
                sdl::GAMEPAD_BUTTON_DOWN => {
                    if let Some(bit) = bit_of(i32::from(e.button())) {
                        shared.pressed.fetch_or(1 << bit, Ordering::Relaxed);
                    }
                }
                _ => {}
            }
        }
        let (mut held, mut l, mut r) = (0u64, [0, 0], [0, 0]);
        for &(_, g) in &self.open {
            for (button, bit) in BUTTONS {
                if self.sdl.button(g, button) {
                    held |= 1 << bit;
                }
            }
            for (axis, bit) in TRIGGERS {
                if self.sdl.axis(g, axis) > 16384 {
                    held |= 1 << bit;
                }
            }
            let gl = stick(self.sdl.axis(g, sdl::axis::LEFT_X), self.sdl.axis(g, sdl::axis::LEFT_Y));
            let gr = stick(self.sdl.axis(g, sdl::axis::RIGHT_X), self.sdl.axis(g, sdl::axis::RIGHT_Y));
            if l == [0, 0] {
                l = gl;
            }
            if r == [0, 0] {
                r = gr;
            }
        }
        shared.held.store(held, Ordering::Relaxed);
        let pack = |v: [i32; 2], i: u32| u64::from(v[0] as i16 as u16) << (32 * i) | u64::from(v[1] as i16 as u16) << (32 * i + 16);
        shared.sticks.store(pack(l, 0) | pack(r, 1), Ordering::Relaxed);
    }
}

/// The stream handle may move to the guest thread: SDL3's audio streams lock
/// themselves.
struct Output(Stream);
// SAFETY: SDL3 audio streams are safe to use from any thread (SDL_audio.h).
unsafe impl Send for Output {}

/// About 4 sink ticks (2048 frames each): past it, a tick is dropped (the
/// host clock runs slow), and the count says how often.
const MAX_QUEUED: i32 = 4 * 2048 * 8;

/// The game's sound to the default device, gain applied; None if SDL has no
/// device for it.
pub fn audio(sdl: Arc<Sdl>, channels: i32, rate: i32) -> Option<gyroship_hle::AudioOut> {
    let stream = match sdl.open_stream(channels, rate) {
        Ok(s) => Output(s),
        Err(e) => {
            eprintln!("gyroship: no sound: {e}");
            return None;
        }
    };
    let (mut dropped, mut starved, mut first) = (0u64, 0u64, true);
    let start = std::time::Instant::now();
    let frame_bytes = channels.max(1) as usize * 4;
    Some(Box::new(move |frames: &[f32], gain: f32| {
        let s = &stream;
        let queued = sdl.queued(s.0);
        // An empty queue: the device ran dry (a gap, but at the start). A tick of
        // silence goes first, so the queue holds one to two ticks (D30: about 85 ms)
        // and the ticks' jitter (they come at frames, ±16 ms) no longer reaches it.
        // A tick without frames is no gap: the next one with frames counts it.
        if queued == 0 && !frames.is_empty() {
            if !std::mem::take(&mut first) {
                starved += 1;
                if starved.is_power_of_two() {
                    eprintln!("gyroship: sound: {starved} gaps, the last at {:.1} s (the game ran slower than real time)", start.elapsed().as_secs_f32());
                }
            }
            sdl.put(s.0, &vec![0.0; frames.len()]);
        }
        if queued > MAX_QUEUED {
            dropped += 1;
            if dropped.is_power_of_two() {
                eprintln!("gyroship: sound: {dropped} ticks dropped (the host is ahead)");
            }
        } else if !frames.is_empty() {
            let scaled: Vec<f32> = frames.iter().map(|v| v * gain).collect();
            sdl.put(s.0, &scaled);
        }
        // The frames the device still holds: the play engine's sink paces on it.
        sdl.queued(s.0).max(0) as usize / frame_bytes
    }))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn buttons_go_by_position() {
        assert_eq!([bit_of(b::EAST), bit_of(b::SOUTH), bit_of(b::NORTH), bit_of(b::WEST)], [Some(0), Some(1), Some(2), Some(3)]);
        assert_eq!(bit_of(5), None, "Guide stays for the practice menu");
    }

    #[test]
    fn sticks_have_a_dead_zone_and_reach_full_range() {
        assert_eq!(stick(-4605, -4238), [0, 0], "an Xbox 360 stick at rest");
        assert_eq!(stick(32767, 0), [32767, 0]);
        assert_eq!(stick(0, -32768), [0, 32767], "SDL's Y grows downwards");
        let [x, _] = stick(20000, 0);
        assert!(0 < x && x < 20000);
    }

    #[test]
    fn gamepads_and_keys_merge() {
        let keys = Pad { buttons: 1, l: [32767, 0], r: [0, 0] };
        let p = merge(keys, 1 << 1, [0, 0], [0, -32767]);
        assert_eq!(p.l, [32767, 0], "the keyboard's stick when the gamepad's rests");
        assert_eq!(p.buttons, 1 | 1 << 1 | 1 << 18 | 1 << 23);
    }
}
