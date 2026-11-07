//! The SDL3 trial of the M4 plan (step 10): lists the controllers, reports
//! plugging and buttons for a few seconds, and plays one second of 440 Hz.
//! `cargo run -p gyroship --example sdl_trial -- <SDL3.dll> [seconds]`

use gyroship::sdl::{self, Sdl};
use std::time::{Duration, Instant};

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let Some(path) = args.get(1) else {
        eprintln!("usage: sdl_trial <SDL3.dll> [seconds]");
        std::process::exit(2);
    };
    let seconds: u64 = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(10);
    let sdl = Sdl::load(path.as_ref()).unwrap_or_else(|e| {
        eprintln!("{e}");
        std::process::exit(1);
    });
    // One second of A4, stereo, 48 kHz.
    let tone: Vec<f32> = (0..48_000).flat_map(|i| {
        let v = 0.2 * (i as f32 * 440.0 * std::f32::consts::TAU / 48_000.0).sin();
        [v, v]
    }).collect();
    match sdl.open_stream(2, 48_000) {
        Ok(s) => println!("audio: queued {} bytes: {}", std::mem::size_of_val(&tone[..]), sdl.put(s, &tone)),
        Err(e) => println!("{e}"),
    }
    let mut pads = Vec::new();
    let end = Instant::now() + Duration::from_secs(seconds);
    while Instant::now() < end {
        while let Some(e) = sdl.poll() {
            match e.kind() {
                sdl::GAMEPAD_ADDED => {
                    if let Some(g) = sdl.open_gamepad(e.which()) {
                        println!("gamepad {} added: {}", e.which(), sdl.gamepad_name(g));
                        pads.push(g);
                    }
                }
                sdl::GAMEPAD_REMOVED => println!("gamepad {} removed", e.which()),
                sdl::GAMEPAD_BUTTON_DOWN => println!("gamepad {} button {} down", e.which(), e.button()),
                _ => {}
            }
        }
        std::thread::sleep(Duration::from_millis(5));
    }
    for g in pads {
        println!("left stick at the end: {}, {}", sdl.axis(g, sdl::axis::LEFT_X), sdl.axis(g, sdl::axis::LEFT_Y));
        sdl.close_gamepad(g);
    }
}
