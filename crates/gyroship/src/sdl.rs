//! SDL3, loaded at run time (D30): controllers and sound. Without the DLL
//! the window still runs, with the keyboard only and no sound. Only the
//! functions Gyroship calls are declared, with plain arguments.

use std::ffi::{CStr, c_char, c_void};

/// SDL_INIT_AUDIO | SDL_INIT_GAMEPAD (which brings the joystick subsystem).
const INIT: u32 = 0x10 | 0x2000;
/// SDL_EVENT_GAMEPAD_* event types.
pub const GAMEPAD_BUTTON_DOWN: u32 = 0x651;
pub const GAMEPAD_ADDED: u32 = 0x653;
pub const GAMEPAD_REMOVED: u32 = 0x654;
/// SDL_AUDIO_F32 (little endian) and the default playback device.
const AUDIO_F32: i32 = 0x8120;
const DEFAULT_PLAYBACK: u32 = 0xFFFF_FFFF;

#[repr(C)]
struct AudioSpec {
    format: i32,
    channels: i32,
    freq: i32,
}

/// An SDL_Event: 128 bytes, its type first. Gamepad events carry the
/// joystick id at byte 16; button events the button at 20, down at 21.
#[repr(C, align(8))]
pub struct Event(pub [u8; 128]);

impl Event {
    pub fn kind(&self) -> u32 {
        u32::from_le_bytes(self.0[0..4].try_into().unwrap())
    }
    pub fn which(&self) -> u32 {
        u32::from_le_bytes(self.0[16..20].try_into().unwrap())
    }
    pub fn button(&self) -> u8 {
        self.0[20]
    }
}

/// Gamepad buttons by position (SDL_GamepadButton) and axes (SDL_GamepadAxis).
pub mod button {
    pub const SOUTH: i32 = 0;
    pub const EAST: i32 = 1;
    pub const WEST: i32 = 2;
    pub const NORTH: i32 = 3;
    pub const BACK: i32 = 4;
    pub const START: i32 = 6;
    pub const LEFT_STICK: i32 = 7;
    pub const RIGHT_STICK: i32 = 8;
    pub const LEFT_SHOULDER: i32 = 9;
    pub const RIGHT_SHOULDER: i32 = 10;
    pub const DPAD_UP: i32 = 11;
    pub const DPAD_DOWN: i32 = 12;
    pub const DPAD_LEFT: i32 = 13;
    pub const DPAD_RIGHT: i32 = 14;
}
pub mod axis {
    pub const LEFT_X: i32 = 0;
    pub const LEFT_Y: i32 = 1;
    pub const RIGHT_X: i32 = 2;
    pub const RIGHT_Y: i32 = 3;
    pub const LEFT_TRIGGER: i32 = 4;
    pub const RIGHT_TRIGGER: i32 = 5;
}

/// An open gamepad; only [`Sdl`] makes one.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Gamepad(*mut c_void);
/// An open playback stream; only [`Sdl`] makes one.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Stream(*mut c_void);

/// The loaded library and the entry points Gyroship uses.
pub struct Sdl {
    _lib: libloading::Library,
    get_error: unsafe extern "C" fn() -> *const c_char,
    poll_event: unsafe extern "C" fn(*mut Event) -> bool,
    open_gamepad: unsafe extern "C" fn(u32) -> *mut c_void,
    close_gamepad: unsafe extern "C" fn(*mut c_void),
    gamepad_name: unsafe extern "C" fn(*mut c_void) -> *const c_char,
    gamepad_button: unsafe extern "C" fn(*mut c_void, i32) -> bool,
    gamepad_axis: unsafe extern "C" fn(*mut c_void, i32) -> i16,
    open_stream: unsafe extern "C" fn(u32, *const AudioSpec, *const c_void, *mut c_void) -> *mut c_void,
    resume_stream: unsafe extern "C" fn(*mut c_void) -> bool,
    put_stream: unsafe extern "C" fn(*mut c_void, *const c_void, i32) -> bool,
    queued: unsafe extern "C" fn(*mut c_void) -> i32,
    quit: unsafe extern "C" fn(),
}

impl Sdl {
    /// Loads `path` and starts its audio and gamepad subsystems. Call on the
    /// thread that runs the window's event loop (SDL's rule for events).
    pub fn load(path: &std::path::Path) -> Result<Sdl, String> {
        // SAFETY: SDL3.dll's initializers have no preconditions.
        let lib = unsafe { libloading::Library::new(path) }.map_err(|e| format!("{}: {e}", path.display()))?;
        macro_rules! sym {
            ($name:literal) => {
                // SAFETY: the declared signature is SDL3's (SDL3 headers, 3.4).
                *unsafe { lib.get(concat!($name, "\0").as_bytes()) }.map_err(|e| format!("{}: {e}", $name))?
            };
        }
        let set_hint: unsafe extern "C" fn(*const c_char, *const c_char) -> bool = sym!("SDL_SetHint");
        let init: unsafe extern "C" fn(u32) -> bool = sym!("SDL_Init");
        let sdl = Sdl {
            get_error: sym!("SDL_GetError"),
            poll_event: sym!("SDL_PollEvent"),
            open_gamepad: sym!("SDL_OpenGamepad"),
            close_gamepad: sym!("SDL_CloseGamepad"),
            gamepad_name: sym!("SDL_GetGamepadName"),
            gamepad_button: sym!("SDL_GetGamepadButton"),
            gamepad_axis: sym!("SDL_GetGamepadAxis"),
            open_stream: sym!("SDL_OpenAudioDeviceStream"),
            resume_stream: sym!("SDL_ResumeAudioStreamDevice"),
            put_stream: sym!("SDL_PutAudioStreamData"),
            queued: sym!("SDL_GetAudioStreamQueued"),
            quit: sym!("SDL_Quit"),
            _lib: lib,
        };
        // SAFETY: valid C strings; Init once, before any other call.
        unsafe {
            // Controllers keep working while another window has the focus (Gyroship's
            // own window does not see their events: SDL reads them).
            set_hint(c"SDL_JOYSTICK_ALLOW_BACKGROUND_EVENTS".as_ptr(), c"1".as_ptr());
            if !init(INIT) {
                return Err(format!("SDL_Init: {}", sdl.error()));
            }
        }
        Ok(sdl)
    }

    pub fn error(&self) -> String {
        // SAFETY: SDL_GetError returns a valid C string.
        unsafe { CStr::from_ptr((self.get_error)()) }.to_string_lossy().into_owned()
    }

    /// The next pending event (also updates gamepad states).
    pub fn poll(&self) -> Option<Event> {
        let mut e = Event([0; 128]);
        // SAFETY: an SDL_Event-sized, aligned buffer.
        unsafe { (self.poll_event)(&mut e) }.then_some(e)
    }

    pub fn open_gamepad(&self, id: u32) -> Option<Gamepad> {
        // SAFETY: any id; null on failure.
        let g = unsafe { (self.open_gamepad)(id) };
        (!g.is_null()).then_some(Gamepad(g))
    }

    pub fn close_gamepad(&self, g: Gamepad) {
        // SAFETY: a gamepad this library opened.
        unsafe { (self.close_gamepad)(g.0) }
    }

    pub fn gamepad_name(&self, g: Gamepad) -> String {
        // SAFETY: an open gamepad; the name may be null.
        let p = unsafe { (self.gamepad_name)(g.0) };
        if p.is_null() { "?".into() } else { unsafe { CStr::from_ptr(p) }.to_string_lossy().into_owned() }
    }

    pub fn button(&self, g: Gamepad, b: i32) -> bool {
        // SAFETY: an open gamepad.
        unsafe { (self.gamepad_button)(g.0, b) }
    }

    pub fn axis(&self, g: Gamepad, a: i32) -> i16 {
        // SAFETY: an open gamepad.
        unsafe { (self.gamepad_axis)(g.0, a) }
    }

    /// A playback stream of interleaved f32 samples on the default device,
    /// converted and resampled by SDL, already playing.
    pub fn open_stream(&self, channels: i32, rate: i32) -> Result<Stream, String> {
        let spec = AudioSpec { format: AUDIO_F32, channels, freq: rate };
        // SAFETY: a valid spec; no callback.
        let s = unsafe { (self.open_stream)(DEFAULT_PLAYBACK, &spec, std::ptr::null(), std::ptr::null_mut()) };
        // SAFETY: the stream just opened.
        if s.is_null() || !unsafe { (self.resume_stream)(s) } {
            return Err(format!("SDL audio: {}", self.error()));
        }
        Ok(Stream(s))
    }

    pub fn put(&self, s: Stream, samples: &[f32]) -> bool {
        // SAFETY: an open stream and a buffer of that many bytes.
        unsafe { (self.put_stream)(s.0, samples.as_ptr().cast(), std::mem::size_of_val(samples) as i32) }
    }

    /// Bytes waiting in the stream.
    pub fn queued(&self, s: Stream) -> i32 {
        // SAFETY: an open stream.
        unsafe { (self.queued)(s.0) }
    }
}

impl Drop for Sdl {
    fn drop(&mut self) {
        // SAFETY: SDL was initialized by `load`.
        unsafe { (self.quit)() }
    }
}
