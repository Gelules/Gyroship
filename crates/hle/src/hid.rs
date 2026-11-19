//! `hid`: one controller in slot 0 (FullKey; or, while a Joy-Con mouse is there,
//! a Joy-Con pair, used as mice while the game asks for that) and a keyboard
//! plugged in, in the shared memory layout the SDK reads (the Switch 1 one,
//! our reading of the Switch 2 SDK's readers). Every lifo starts with one null entry: its
//! sequence is even (no write in progress), so the readers never spin. A
//! virtual timer samples the controller and the keyboard once per frame, 1 ms
//! before the vsync (D29), from the `--replay` script, else from the host's live
//! input (keyboard, controllers, Joy-Con), else at rest. The game
//! only reads the keyboard when its own keyboard setting is on (debug tools).

use crate::Kernel;
use crate::ipc::{Reply, Request};
use crate::service::Service;
use crate::thread::{Object, Timer, TimerAction};

pub const SHARED_SIZE: u64 = 0x40000;
/// Touch screen, mouse and keyboard lifos.
const BASIC_LIFOS: [usize; 3] = [0x400, 0x3400, KEYBOARD];
const KEYBOARD: usize = 0x3800;
/// A keyboard lifo entry: its sequence, then the state: sampling number,
/// modifiers, attributes, 256 key bits (one per USB HID usage).
const KEY_ENTRY: usize = 0x38;
/// Npad slots: 0x9a00 + slot × 0x5000, slots 0..=9.
const NPAD: usize = 0x9a00;
const NPAD_SLOT: usize = 0x5000;
/// Button lifos (FullKey, Handheld, JoyDual, JoyLeft, JoyRight) and six-axis lifos in a slot.
const SLOT_LIFOS: [usize; 11] = [0x28, 0x378, 0x6c8, 0xa18, 0xd68, 0x1758, 0x1e60, 0x2568, 0x2c70, 0x3378, 0x3a80];
/// The Npad condition block: hold type at +8, "initialized" at +0xc.
const CONDITION: usize = 0x3e200;
/// FullKey style, and the attribute "connected" of a button state.
const FULL_KEY: u32 = 1;
const CONNECTED: u32 = 1;
/// The style of a Joy-Con pair (JoyDual, bit 2), read through the JoyDual lifo.
const JOY_DUAL: u32 = 4;
/// The style of a Joy-Con pair used as mice (bit 12; the game's "joycon-dual-ms"), read
/// through the JoyDual lifo too, whose attributes must then say both Joy-Con are connected
/// (Gyroship writes them so for both pair styles).
pub const DUAL_MOUSE: u32 = 0x1000;
const JOY_DUAL_LIFO: usize = 0x6c8;
const BOTH_CONNECTED: u32 = CONNECTED | 1 << 2 | 1 << 4;
/// The six-axis sensor lifos of a Joy-Con pair, left then right (the SDK gives the pair's
/// style two handles, device 0 and 1). An entry: its sequence, then the state: time since
/// the previous one (ns), sampling number, acceleration (g), angular velocity, angle,
/// direction matrix (f32 each), attributes (bit 0: connected).
const DUAL_MOTION_LIFOS: [usize; 2] = [0x2568, 0x2c70];
const MOTION_ENTRY: usize = 0x68;
const SAMPLE_NS: u64 = 1_000_000_000 / 60;
/// A button lifo: tail at +0x10, count at +0x18, 17 entries of 0x30 from
/// +0x20; an entry is its sequence, then the state: sampling number,
/// buttons, stick L, stick R, attributes.
const LIFO_TAIL: usize = 0x10;
const LIFO_COUNT: usize = 0x18;
const LIFO_ENTRIES: usize = 0x20;
const LIFO_LEN: usize = 17;
const ENTRY: usize = 0x30;
/// The Joy-Con mouse lifos, in the buffer the game lends (InitializeJcms): one per Npad id
/// (0..8) and side, at (2 × id + side) × 0x2c8, read by the SDK's GetJcmsStates. Same
/// header as a button lifo, 17 entries of 0x28: the sequence, then 8 bytes nothing reads,
/// the sampling number, the X and Y counts since the previous state, attributes.
const JCMS_LIFO: u64 = 0x2c8;
const MOUSE_ENTRY: usize = 0x28;
#[cfg(test)]
const LIFO_SIZE: usize = LIFO_ENTRIES + LIFO_LEN * ENTRY;
/// The sample's lead on the vsync, in ticks (1 ms): the console's phase is
/// unknown; any fixed lead keeps runs deterministic.
const SAMPLE_LEAD: u64 = 31_250;
const TICKS_PER_SECOND: u64 = 31_250_000;

/// Button names in bit order, as `--replay` writes them (the Switch's
/// public Npad button numbering; "A = bit 0" is checked by a run).
pub const BUTTONS: [&str; 28] = [
    "A", "B", "X", "Y", "LS", "RS", "L", "R", "ZL", "ZR", "Plus", "Minus", "Left", "Up", "Right", "Down", "LSLeft", "LSUp", "LSRight", "LSDown", "RSLeft", "RSUp", "RSRight", "RSDown", "LeftSL", "LeftSR", "RightSL", "RightSR",
];

/// A Joy-Con's mouse at one sample: the X and Y counts since the previous sample, and
/// whether its sensor rests on a surface.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Mouse {
    pub dx: i32,
    pub dy: i32,
    pub contact: bool,
}

/// A controller's state: buttons and sticks (±0x7fff); the keyboard's: the keys held, one
/// bit per USB HID usage (0x35 the backquote key, 0x3a F1, 0xe0 left Ctrl); the mouse of
/// its left and right Joy-Con, when it has one; and their acceleration, in thousandths of g
/// on the SDK's axes, when it is known.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Pad {
    pub buttons: u64,
    pub l: [i32; 2],
    pub r: [i32; 2],
    pub keys: [u64; 4],
    pub mouse: [Option<Mouse>; 2],
    pub accel: [Option<[i32; 3]>; 2],
}

/// The usages of the keys held, in increasing order.
fn usages(keys: &[u64; 4]) -> impl Iterator<Item = usize> + '_ {
    (0..256).filter(|u| keys[u / 64] >> (u % 64) & 1 != 0)
}

/// One line of a `--replay` script: from this sample on, this state; or
/// the end of the recording.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Step {
    Pad(Pad),
    End,
}

/// Parses a `--replay` script: `<sample> <buttons|-> [L=x,y] [R=x,y]
/// [K=usage,...] [ML=dx,dy,c] [MR=dx,dy,c] [AL=x,y,z] [AR=x,y,z]` or `<sample> end`,
/// buttons joined by `+`, keyboard keys as hex USB HID usages, a Joy-Con mouse as its
/// counts per sample and 1 when it touches a surface, a Joy-Con's acceleration in
/// thousandths of g, samples increasing; `#` starts a comment.
pub fn parse_script(text: &str) -> Result<Vec<(u64, Step)>, String> {
    let mut script: Vec<(u64, Step)> = Vec::new();
    for (n, line) in text.lines().enumerate() {
        let bad = |what: &str| format!("line {}: {what}: {line}", n + 1);
        let mut words = line.split('#').next().unwrap_or("").split_whitespace();
        let Some(sample) = words.next() else { continue };
        let sample: u64 = sample.parse().map_err(|_| bad("not a sample number"))?;
        if script.last().is_some_and(|&(s, _)| s >= sample) {
            return Err(bad("samples must increase"));
        }
        let step = match words.next().ok_or_else(|| bad("no state"))? {
            "end" => Step::End,
            buttons => {
                let mut pad = Pad::default();
                if buttons != "-" {
                    for b in buttons.split('+') {
                        let bit = BUTTONS.iter().position(|&n| n == b).ok_or_else(|| bad(&format!("unknown button {b}")))?;
                        pad.buttons |= 1 << bit;
                    }
                }
                for w in words.by_ref() {
                    let (stick, xy) = w.split_once('=').ok_or_else(|| bad("expected L=x,y, R=x,y, K=keys, ML=dx,dy,c, MR=dx,dy,c, AL=x,y,z or AR=x,y,z"))?;
                    if let Some(side) = ["ML", "MR"].iter().position(|&n| n == stick) {
                        let v = xy.split(',').map(str::parse::<i32>).collect::<Result<Vec<_>, _>>().map_err(|_| bad("expected dx,dy,c"))?;
                        let &[dx, dy, c @ (0 | 1)] = v.as_slice() else { return Err(bad("expected dx,dy,c with c 0 or 1")) };
                        pad.mouse[side] = Some(Mouse { dx, dy, contact: c == 1 });
                        continue;
                    }
                    if let Some(side) = ["AL", "AR"].iter().position(|&n| n == stick) {
                        let v = xy.split(',').map(str::parse::<i32>).collect::<Result<Vec<_>, _>>().map_err(|_| bad("expected x,y,z"))?;
                        let &[x, y, z] = v.as_slice() else { return Err(bad("expected x,y,z")) };
                        pad.accel[side] = Some([x, y, z]);
                        continue;
                    }
                    if stick == "K" {
                        for u in xy.split(',') {
                            let u = usize::from_str_radix(u, 16).ok().filter(|&u| u < 256).ok_or_else(|| bad("expected keys as hex USB HID usages below 0x100"))?;
                            pad.keys[u / 64] |= 1 << (u % 64);
                        }
                        continue;
                    }
                    let (x, y) = xy.split_once(',').ok_or_else(|| bad("expected x,y"))?;
                    let v = [x.parse().map_err(|_| bad("bad x"))?, y.parse().map_err(|_| bad("bad y"))?];
                    match stick {
                        "L" => pad.l = v,
                        "R" => pad.r = v,
                        _ => return Err(bad("expected L= or R=")),
                    }
                }
                Step::Pad(pad)
            }
        };
        if words.next().is_some() {
            return Err(bad("trailing words"));
        }
        script.push((sample, step));
    }
    Ok(script)
}

/// Appends an entry of `size` bytes to a lifo (the bytes from its header on),
/// as the SDK's reader expects: an even sequence, a sampling number one past
/// the previous entry's, the rest of the state (`state`), then the tail, then
/// the count.
fn push_entry(lifo: &mut [u8], size: usize, state: &[u8]) {
    let e = next_entry(lifo, size, 8);
    lifo[e + 0x10..e + size].copy_from_slice(state);
}

/// Makes the next entry of a lifo of `size`-byte entries, whose sampling number sits
/// `number` bytes into an entry: zeroed but for its even sequence and its sampling number,
/// one past the previous entry's; the tail and the count follow. Returns its offset.
fn next_entry(lifo: &mut [u8], size: usize, number: usize) -> usize {
    let u64_at = |b: &[u8], o: usize| u64::from_le_bytes(b[o..o + 8].try_into().unwrap());
    let tail = u64_at(lifo, LIFO_TAIL) as usize % LIFO_LEN;
    let sample = u64_at(lifo, LIFO_ENTRIES + tail * size + number) + 1;
    let next = (tail + 1) % LIFO_LEN;
    let e = LIFO_ENTRIES + next * size;
    lifo[e..e + size].fill(0);
    lifo[e..e + 8].copy_from_slice(&(sample << 1).to_le_bytes());
    lifo[e + number..e + number + 8].copy_from_slice(&sample.to_le_bytes());
    lifo[LIFO_TAIL..LIFO_TAIL + 8].copy_from_slice(&(next as u64).to_le_bytes());
    let count = (u64_at(lifo, LIFO_COUNT) + 1).min(LIFO_LEN as u64);
    lifo[LIFO_COUNT..LIFO_COUNT + 8].copy_from_slice(&count.to_le_bytes());
    e
}

/// Appends a state to a Joy-Con mouse lifo (the bytes from its header on): the counts
/// since the previous state and the attributes (bit 0: a mouse is there, which the game
/// reads as its controller supporting a mouse; its options offer mouse aiming only with
/// that and a Joy-Con pair shown, see `style_to_show`; bit 1: it touches a surface).
pub fn push_mouse(lifo: &mut [u8], mouse: Option<Mouse>) {
    let e = next_entry(lifo, MOUSE_ENTRY, 0x10);
    if let Some(m) = mouse {
        lifo[e + 0x18..e + 0x1c].copy_from_slice(&m.dx.to_le_bytes());
        lifo[e + 0x1c..e + 0x20].copy_from_slice(&m.dy.to_le_bytes());
        lifo[e + 0x20] = 1 | u8::from(m.contact) << 1;
    }
}

/// Appends a state to a six-axis sensor lifo (the bytes from its header on): one sample's
/// time, then, given an acceleration in thousandths of g, that acceleration, no rotation,
/// the direction matrix at rest (identity), connected; without one, a null state (all
/// zero, not connected), for a replay that stops giving it.
pub fn push_motion(lifo: &mut [u8], accel: Option<[i32; 3]>) {
    let e = next_entry(lifo, MOTION_ENTRY, 0x10);
    lifo[e + 8..e + 0x10].copy_from_slice(&SAMPLE_NS.to_le_bytes());
    // Acceleration, angular velocity, angle (3 each), then the 3×3 direction matrix.
    let mut floats = [0f32; 18];
    if let Some(accel) = accel {
        for (i, a) in accel.iter().enumerate() {
            floats[i] = *a as f32 / 1000.0;
        }
        for i in 0..3 {
            floats[9 + 4 * i] = 1.0;
        }
    }
    for (i, f) in floats.iter().enumerate() {
        lifo[e + 0x18 + 4 * i..e + 0x1c + 4 * i].copy_from_slice(&f.to_le_bytes());
    }
    let attributes = if accel.is_some() { CONNECTED } else { 0 };
    lifo[e + 0x60..e + 0x64].copy_from_slice(&attributes.to_le_bytes());
}

/// The words a push changed in a lifo of `size`-byte entries, in the order the SDK's
/// readers need them published: the new entry's sequence made odd ("write in progress"),
/// its state, its even sequence, then the tail, then the count.
fn published(lifo: &[u8], size: usize) -> Vec<(usize, u64)> {
    let word = |o: usize| u64::from_le_bytes(lifo[o..o + 8].try_into().unwrap());
    let e = LIFO_ENTRIES + word(LIFO_TAIL) as usize * size;
    [(e, word(e) | 1)].into_iter().chain((e + 8..e + size).step_by(8).chain([e, LIFO_TAIL, LIFO_COUNT]).map(|o| (o, word(o)))).collect()
}

/// Appends `pad` to a button lifo (the bytes from its header on): buttons,
/// sticks, attributes.
pub fn push(lifo: &mut [u8], pad: &Pad, attributes: u32) {
    let mut state = [0u8; ENTRY - 0x10];
    state[0..8].copy_from_slice(&pad.buttons.to_le_bytes());
    for (i, v) in pad.l.iter().chain(&pad.r).enumerate() {
        state[8 + 4 * i..0xc + 4 * i].copy_from_slice(&v.to_le_bytes());
    }
    state[0x18..0x1c].copy_from_slice(&attributes.to_le_bytes());
    push_entry(lifo, ENTRY, &state);
}

/// The style Npad 0 shows. With a mouse on either side (Joy-Con under `--joycon-mouse`, or
/// a replay's mouse words): a Joy-Con pair used as mice while the game supports that style
/// but not FullKey's (it asked for mouse controls), else a Joy-Con pair while the game
/// supports that style, as the console shows a pair held apart. Otherwise, and without a
/// mouse, the FullKey controller.
pub fn style_to_show(supported: u32, pad: &Pad) -> u32 {
    let mice = pad.mouse.iter().any(Option::is_some);
    if mice && supported & DUAL_MOUSE != 0 && supported & FULL_KEY == 0 {
        DUAL_MOUSE
    } else if mice && supported & JOY_DUAL != 0 {
        JOY_DUAL
    } else {
        FULL_KEY
    }
}

/// Appends the keys of `pad` to the keyboard lifo (the bytes from its header
/// on): the modifiers they make (public format: Ctrl bit 0, Shift bit 1, left
/// and right Alt bits 2 and 3, Gui bit 4), connected, the 256 key bits.
pub fn push_keys(lifo: &mut [u8], pad: &Pad) {
    let held = |u: usize| pad.keys[u / 64] >> (u % 64) & 1 != 0;
    let modifiers = [(0xe0, 0xe4), (0xe1, 0xe5), (0xe2, 0xe2), (0xe6, 0xe6), (0xe3, 0xe7)].iter().enumerate().filter(|(_, (a, b))| held(*a) || held(*b)).fold(0u32, |m, (bit, _)| m | 1 << bit);
    let mut state = [0u8; KEY_ENTRY - 0x10];
    state[0..4].copy_from_slice(&modifiers.to_le_bytes());
    state[4..8].copy_from_slice(&CONNECTED.to_le_bytes());
    for (i, w) in pad.keys.iter().enumerate() {
        state[8 + 8 * i..0x10 + 8 * i].copy_from_slice(&w.to_le_bytes());
    }
    push_entry(lifo, KEY_ENTRY, &state);
}

/// Where the replay script stands (plain data, for save states).
#[derive(Clone, Copy, Debug, Default)]
pub struct Input {
    /// The script's next line.
    pub next: usize,
    pub current: Pad,
    /// The last sample taken.
    pub last: u64,
    /// The buffer of Joy-Con mouse lifos the game lent: address and size.
    pub jcms: Option<(u64, u64)>,
    /// The style set the game supports (SetSupportedNpadStyleSet), and the style Npad 0
    /// shows (0 until hid's shared memory exists).
    pub supported: u32,
    pub style: u32,
}

/// One line of a `--replay` script, as `--record` writes it.
pub fn script_line(sample: u64, p: &Pad) -> String {
    let names: Vec<&str> = BUTTONS.iter().enumerate().filter(|(bit, _)| p.buttons >> bit & 1 != 0).map(|(_, n)| *n).collect();
    let mut line = format!("{sample} {}", if names.is_empty() { "-".to_string() } else { names.join("+") });
    for (name, [x, y]) in [("L", p.l), ("R", p.r)] {
        if (x, y) != (0, 0) {
            line += &format!(" {name}={x},{y}");
        }
    }
    if p.keys != [0; 4] {
        line += &format!(" K={}", usages(&p.keys).map(|u| format!("{u:x}")).collect::<Vec<_>>().join(","));
    }
    for (name, mouse) in [("ML", p.mouse[0]), ("MR", p.mouse[1])] {
        if let Some(m) = mouse {
            line += &format!(" {name}={},{},{}", m.dx, m.dy, u8::from(m.contact));
        }
    }
    for (name, accel) in [("AL", p.accel[0]), ("AR", p.accel[1])] {
        if let Some([x, y, z]) = accel {
            line += &format!(" {name}={x},{y},{z}");
        }
    }
    line
}

#[derive(Debug)]
pub enum Hid {
    /// `hid` (IHidServer): the style set and ids the game supports.
    Service { styles: u32, ids: Vec<u32> },
    AppletResource,
    VibrationList,
}

impl Hid {
    pub fn label(&self) -> String {
        match self {
            Hid::Service { .. } => "hid".into(),
            Hid::AppletResource => "hid IAppletResource".into(),
            Hid::VibrationList => "hid IActiveVibrationDeviceList".into(),
        }
    }

    pub fn request(&mut self, k: &mut Kernel, req: &Request) -> Option<Reply> {
        Some(match (self, req.command) {
            (Hid::Service { .. }, 0) => return Some(Reply::object(Service::Hid(Hid::AppletResource))),
            // GetSharedMemoryHandle.
            (Hid::AppletResource, 0) => {
                let m = k.hid_memory();
                Reply { copy_handles: vec![k.handles.add(Object::SharedMemory(m))?], ..Reply::ok() }
            }
            // Activations, touch screen size, six-axis start.
            (Hid::Service { .. }, 11 | 21 | 31 | 66 | 1004 | 3150) => Reply::ok(),
            // FinalizeJcms, when the game exits: the buffer is the game's again.
            (Hid::Service { .. }, 3111) => {
                k.input.jcms = None;
                Reply::ok()
            }
            // StartJcms(handle: an Npad id, then a side): that mouse's lifo gets a first, null
            // state (no mouse there, nothing moved). The game reads the newest state without
            // looking at how many came: from an empty lifo it read its own stack instead, and
            // took its controller for one with a mouse or not by what lay there.
            (Hid::Service { .. }, 3112) => {
                let h = u64::from(req.u32_at(0)?);
                let lifo = (2 * (h & 0xFF) + (h >> 8 & 0xFF)) * JCMS_LIFO;
                let mut count = [0; 8];
                if let Some((addr, size)) = k.input.jcms
                    && h & 0xFF < 8
                    && h >> 8 & 0xFF < 2
                    && lifo + JCMS_LIFO <= size
                    && k.mem.read(addr + lifo + LIFO_COUNT as u64, &mut count).is_ok()
                    && count == [0; 8]
                {
                    let _ = k.mem.write_object(addr + lifo + LIFO_COUNT as u64, &1u64.to_le_bytes());
                }
                Reply::ok()
            }
            // ActivateNpadWithRevision: the condition becomes "initialized".
            (Hid::Service { .. }, 109) => {
                k.npad_condition(1);
                Reply::ok()
            }
            // InitializeJcms: the client lends a buffer the server fills; it starts empty.
            (Hid::Service { .. }, 3110) => {
                if let Some(&h) = req.copy_handles.first()
                    && let Some((addr, size)) = k.transfer_range(h)
                {
                    k.retain_transfer(h);
                    let _ = k.mem.write_object(addr, &vec![0; size as usize]);
                    k.input.jcms = Some((addr, size));
                    k.log(&format!("hid 3110: jcms buffer {addr:#x}+{size:#x} zeroed"));
                }
                Reply::ok()
            }
            // SetSupportedNpadStyleSet / GetSupportedNpadStyleSet. Npad 0's style follows at
            // once: the game asks right after for the controller screen, or checks the style.
            (Hid::Service { styles, .. }, 100) => {
                let s = req.u32_at(0)?;
                if *styles != s {
                    k.log(&format!("hid: supported style set {s:#x}"));
                }
                *styles = s;
                k.input.supported = s;
                k.restyle();
                Reply::ok()
            }
            (Hid::Service { styles, .. }, 101) => Reply::with(&styles.to_le_bytes()),
            // SetSupportedNpadIdType: the ids, in the pointer buffer.
            (Hid::Service { ids, .. }, 102) => {
                if let Some(b) = req.send_buffer(0) {
                    let mut raw = vec![0; b.size.min(0x100) as usize];
                    let _ = k.mem.read(b.addr, &mut raw);
                    *ids = raw.chunks(4).filter(|c| c.len() == 4).map(|c| u32::from_le_bytes(c.try_into().unwrap())).collect();
                    k.log(&format!("hid: supported npad ids {ids:x?}"));
                }
                Reply::ok()
            }
            // SetNpadJoyHoldType.
            (Hid::Service { .. }, 120) => {
                k.npad_condition(req.u64_at(8)? as u32);
                Reply::ok()
            }
            // CreateActiveVibrationDeviceList; ActivateVibrationDevice; vibration values.
            (Hid::Service { .. }, 203) => return Some(Reply::object(Service::Hid(Hid::VibrationList))),
            (Hid::VibrationList, 0) => Reply::ok(),
            (Hid::Service { .. }, 201 | 213) => Reply::ok(),
            _ => return None,
        })
    }
}

impl Kernel {
    /// hid's shared memory, created and filled at the first request.
    fn hid_memory(&mut self) -> usize {
        if let Some(m) = self.hid_shared {
            return m;
        }
        let m = self.create_shared_memory(SHARED_SIZE);
        self.hid_shared = Some(m);
        // One null entry in every lifo: tail 0, count 1, entry 0 all zero.
        let one = 1u64.to_le_bytes();
        for l in BASIC_LIFOS {
            self.write_shared(m, l + 0x18, &one);
        }
        for s in 0..10 {
            for l in SLOT_LIFOS {
                self.write_shared(m, NPAD + s * NPAD_SLOT + l + 0x18, &one);
            }
        }
        // Slot 0: a FullKey controller, connected, nothing pressed.
        self.write_shared(m, NPAD, &FULL_KEY.to_le_bytes());
        self.write_shared(m, NPAD + 0x28 + 0x20 + 0x28, &CONNECTED.to_le_bytes());
        self.input.style = FULL_KEY;
        self.npad_condition(1);
        // Sample k comes SAMPLE_LEAD before vsync k + 1 (to a tick, on the 60 Hz grid).
        let period = TICKS_PER_SECOND / 60;
        let k = (self.now() * 60).div_ceil(TICKS_PER_SECOND);
        self.timers.push(Timer { start: period - SAMPLE_LEAD, k, num: TICKS_PER_SECOND, den: 60, action: TimerAction::HidSample { shared: m } });
        m
    }

    /// Timer `HidSample`: the state of sample `k` (from the replay script, else
    /// at rest) sets the style slot 0 shows (`restyle`), then goes into slot 0's
    /// lifos (`push_sample`) and the Joy-Con mouse lifos (`push_mice`). Logged
    /// when it changes; the script's end stops the run.
    pub(crate) fn hid_sample(&mut self, m: usize, k: u64) {
        let mut input = self.input;
        while let Some(&(at, step)) = self.config.replay.as_ref().and_then(|s| s.get(input.next))
            && at <= k
        {
            input.next += 1;
            match step {
                Step::Pad(p) => input.current = p,
                Step::End => self.stop(0, &format!("hid: replay ended at sample {k}")),
            }
        }
        // Without a script, the host's live state (keyboard, controllers), read only here.
        if self.config.replay.is_none()
            && let Some(live) = self.live_input.as_mut()
        {
            input.current = live();
        }
        let first = self.journal.first("hid sampling");
        let before = self.input.current;
        if first || input.current != before {
            let p = input.current;
            let mut more = if p.keys == [0; 4] { String::new() } else { format!(" keys {:x?}", usages(&p.keys).collect::<Vec<_>>()) };
            for (name, mouse) in [("ML", p.mouse[0]), ("MR", p.mouse[1])] {
                if let Some(m) = mouse {
                    more += &format!(" {name} [{}, {}, {}]", m.dx, m.dy, u8::from(m.contact));
                }
            }
            for (name, accel) in [("AL", p.accel[0]), ("AR", p.accel[1])] {
                if let Some(a) = accel {
                    more += &format!(" {name} {a:?}");
                }
            }
            self.log(&format!("hid: sample {k} (present {}): buttons {:#x} L {:?} R {:?}{more}", self.gs.presents, p.buttons, p.l, p.r));
            if let Some(lines) = self.recorded.as_mut() {
                lines.push(script_line(k, &p));
            }
        }
        input.last = k;
        self.input = input;
        self.restyle();
        self.push_sample(m, &input.current, &before);
        self.push_mice(&input.current, &before);
    }

    /// Appends `pad` to slot 0's FullKey lifo, to its JoyDual lifo while it shows a Joy-Con
    /// pair, each Joy-Con's acceleration to the pair's six-axis lifos, and its
    /// keys to the keyboard lifo, in the guest's copy too. A side without acceleration writes
    /// nothing, but one null state when `before` (the sample before) had one: it is gone.
    fn push_sample(&mut self, m: usize, pad: &Pad, before: &Pad) {
        self.push_lifo(m, NPAD + 0x28, ENTRY, |l| push(l, pad, CONNECTED));
        if matches!(self.input.style, JOY_DUAL | DUAL_MOUSE) {
            self.push_lifo(m, NPAD + JOY_DUAL_LIFO, ENTRY, |l| push(l, pad, BOTH_CONNECTED));
        }
        for (side, at) in DUAL_MOTION_LIFOS.iter().enumerate() {
            if pad.accel[side].is_some() || before.accel[side].is_some() {
                self.push_lifo(m, NPAD + at, MOTION_ENTRY, |l| push_motion(l, pad.accel[side]));
            }
        }
        self.push_lifo(m, KEYBOARD, KEY_ENTRY, |l| push_keys(l, pad));
    }

    /// Makes Npad 0 show the style `style_to_show` gives. To a Joy-Con pair, its JoyDual
    /// lifo first gets the controller's state with both Joy-Con connected, then the style
    /// word changes: whoever reads the new style finds that state.
    fn restyle(&mut self) {
        let Some(m) = self.hid_shared else { return };
        let style = style_to_show(self.input.supported, &self.input.current);
        if style == self.input.style {
            return;
        }
        if matches!(style, JOY_DUAL | DUAL_MOUSE) {
            let pad = self.input.current;
            self.push_lifo(m, NPAD + JOY_DUAL_LIFO, ENTRY, |l| push(l, &pad, BOTH_CONNECTED));
        }
        // The style word is the slot's first u32; the next one is left as it is.
        let next = u32::from_le_bytes(self.shared[m].data[NPAD + 4..NPAD + 8].try_into().unwrap());
        self.store_shared(m, NPAD, u64::from(style) | u64::from(next) << 32);
        self.input.style = style;
        let what = match style {
            DUAL_MOUSE => "a Joy-Con pair used as mice",
            JOY_DUAL => "a Joy-Con pair",
            _ => "the FullKey controller",
        };
        self.log(&format!("hid: Npad 0 shows {what} (style {style:#x})"));
    }

    /// Appends each Joy-Con mouse of `pad` to its lifo of Npad 0 in the buffer the game
    /// lent, with the ordered writes of `push_lifo`. A side without a mouse writes nothing,
    /// but one null state when the sample before had one: the mouse is gone.
    fn push_mice(&mut self, pad: &Pad, before: &Pad) {
        let Some((addr, size)) = self.input.jcms else { return };
        for side in 0..2 {
            let at = addr + side as u64 * JCMS_LIFO;
            let mut lifo = [0; JCMS_LIFO as usize];
            if (pad.mouse[side].is_none() && before.mouse[side].is_none()) || at + JCMS_LIFO > addr + size || self.mem.read(at, &mut lifo).is_err() {
                continue;
            }
            push_mouse(&mut lifo, pad.mouse[side]);
            for (o, v) in published(&lifo, MOUSE_ENTRY) {
                let _ = self.mem.publish_object(at + o as u64, 8, &[], |_| v);
            }
        }
    }

    /// Runs `push` on a copy of the lifo at `at` (entries of `size` bytes).
    /// Guest threads may read the lifo meanwhile, so only the words `push`
    /// changes are written, each one published last of all so far, in the
    /// order the SDK's reader needs: the new entry's sequence made odd ("write
    /// in progress"), its state, its even sequence, then the tail, then the
    /// count. The guest maps this memory read-only, so its other bytes already
    /// equal ours.
    fn push_lifo(&mut self, m: usize, at: usize, size: usize, push: impl FnOnce(&mut [u8])) {
        let mut lifo = self.shared[m].data[at..at + LIFO_ENTRIES + LIFO_LEN * size].to_vec();
        push(&mut lifo);
        for (o, v) in published(&lifo, size) {
            self.store_shared(m, at + o, v);
        }
    }

    /// The Npad condition: hold type and "initialized".
    fn npad_condition(&mut self, hold: u32) {
        let m = self.hid_memory_index();
        if let Some(m) = m {
            self.write_shared(m, CONDITION + 8, &hold.to_le_bytes());
            self.write_shared(m, CONDITION + 0xc, &[1]);
        }
    }

    fn hid_memory_index(&self) -> Option<usize> {
        self.hid_shared
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The SDK reader's checks (our reading of it): count ≤ 17, every entry's
    /// sequence even, each newer entry's sampling number one past the older
    /// one's; returns the newest state.
    fn read(lifo: &[u8]) -> Pad {
        let u64_at = |o: usize| u64::from_le_bytes(lifo[o..o + 8].try_into().unwrap());
        let i32_at = |o: usize| i32::from_le_bytes(lifo[o..o + 4].try_into().unwrap());
        let (tail, count) = (u64_at(LIFO_TAIL) as usize, u64_at(LIFO_COUNT) as usize);
        assert!(tail < LIFO_LEN && (1..=LIFO_LEN).contains(&count));
        let entry = |back: usize| LIFO_ENTRIES + (tail + LIFO_LEN - back) % LIFO_LEN * ENTRY;
        for back in 0..count {
            assert_eq!(u64_at(entry(back)) % 2, 0, "sequence even");
            if back + 1 < count {
                assert_eq!(u64_at(entry(back) + 8), u64_at(entry(back + 1) + 8) + 1, "sampling numbers follow");
            }
        }
        let e = entry(0);
        assert_eq!(u64_at(e + 0x28) as u32, CONNECTED);
        Pad { buttons: u64_at(e + 0x10), l: [i32_at(e + 0x18), i32_at(e + 0x1c)], r: [i32_at(e + 0x20), i32_at(e + 0x24)], ..Pad::default() }
    }

    #[test]
    fn forty_samples_read_back_as_the_last_one() {
        // As hid_memory leaves it: one null entry, count 1.
        let mut lifo = vec![0u8; LIFO_SIZE];
        lifo[LIFO_COUNT] = 1;
        let mut last = Pad::default();
        for n in 1..=40 {
            last = Pad { buttons: n % 3, l: [n as i32, -(n as i32)], r: [0x7fff, -0x7fff], ..Pad::default() };
            push(&mut lifo, &last, CONNECTED);
            assert_eq!(read(&lifo), last, "sample {n}");
        }
        assert_eq!(lifo[LIFO_COUNT], 17);
        assert_eq!(u64::from_le_bytes(lifo[LIFO_ENTRIES + 6 * ENTRY + 8..][..8].try_into().unwrap()), 40, "sampling number of the newest entry (tail 40 mod 17)");
        assert_eq!(read(&lifo), last);
    }

    /// The keyboard lifo, read as the SDK's reader does: newest entry's
    /// sampling number, modifiers, attributes and keys.
    #[test]
    fn keys_read_back_with_their_modifiers() {
        let mut lifo = vec![0u8; LIFO_ENTRIES + LIFO_LEN * KEY_ENTRY];
        lifo[LIFO_COUNT] = 1;
        let u64_at = |l: &[u8], o: usize| u64::from_le_bytes(l[o..o + 8].try_into().unwrap());
        for n in 1..=40u64 {
            // Backquote, F1, then left Ctrl with right Shift on even samples.
            let mut keys = [1 << 0x35 | 1 << 0x3a, 0, 0, 0];
            if n % 2 == 0 {
                keys[3] = 1 << (0xe0 - 0xc0) | 1 << (0xe5 - 0xc0);
            }
            push_keys(&mut lifo, &Pad { keys, ..Pad::default() });
            let e = LIFO_ENTRIES + u64_at(&lifo, LIFO_TAIL) as usize * KEY_ENTRY;
            assert_eq!(u64_at(&lifo, e) % 2, 0, "sequence even");
            assert_eq!(u64_at(&lifo, e + 8), n, "sampling number");
            let modifiers = u32::from_le_bytes(lifo[e + 0x10..e + 0x14].try_into().unwrap());
            assert_eq!(modifiers, if n % 2 == 0 { 0b11 } else { 0 }, "Ctrl and Shift, sample {n}");
            assert_eq!(u32::from_le_bytes(lifo[e + 0x14..e + 0x18].try_into().unwrap()), CONNECTED);
            assert_eq!([0, 1, 2, 3].map(|i| u64_at(&lifo, e + 0x18 + 8 * i)), keys);
        }
        assert_eq!(lifo[LIFO_COUNT], 17);
    }

    /// What `--record` writes, `--replay` reads back unchanged.
    #[test]
    fn recorded_lines_replay_as_written() {
        let pads = [
            Pad::default(),
            Pad { buttons: 1 | 1 << 13 | 1 << 27, l: [-0x7fff, 0x7fff], ..Pad::default() },
            Pad { r: [3, -4], keys: [1 << 0x35, 0, 0, 1 << (0xe1 - 0xc0)], ..Pad::default() },
            Pad { buttons: 4, mouse: [Some(Mouse { dx: -12, dy: 7, contact: true }), None], ..Pad::default() },
            Pad { mouse: [Some(Mouse::default()), Some(Mouse { dx: 3, dy: 0, contact: false })], ..Pad::default() },
            Pad { accel: [Some([1000, -20, 3]), Some([-1000, 0, 0])], ..Pad::default() },
        ];
        let text: String = pads.iter().enumerate().map(|(i, p)| script_line(10 * i as u64, p) + "\n").collect::<String>() + "70 end\n";
        let expected: Vec<(u64, Step)> = pads.iter().enumerate().map(|(i, p)| (10 * i as u64, Step::Pad(*p))).chain([(70, Step::End)]).collect();
        assert_eq!(parse_script(&text).unwrap(), expected);
        assert_eq!(script_line(7, &pads[1]), "7 A+Up+RightSR L=-32767,32767");
        assert_eq!(script_line(9, &pads[2]), "9 - R=3,-4 K=35,e1");
        // A Joy-Con mouse: its counts and whether it touches a surface, each side by itself.
        assert_eq!(script_line(11, &pads[3]), "11 X ML=-12,7,1");
        assert_eq!(script_line(13, &pads[4]), "13 - ML=0,0,0 MR=3,0,0");
        // A Joy-Con's acceleration, each side by itself.
        assert_eq!(script_line(15, &pads[5]), "15 - AL=1000,-20,3 AR=-1000,0,0");
    }

    #[test]
    fn replay_scripts_parse() {
        let s = parse_script("# title screen\n10 A+Up L=0,32767\n16 -\n20 end\n").unwrap();
        assert_eq!(s, vec![(10, Step::Pad(Pad { buttons: 1 | 1 << 13, l: [0, 32767], ..Pad::default() })), (16, Step::Pad(Pad::default())), (20, Step::End)]);
        assert!(parse_script("5 A\n5 B").unwrap_err().contains("increase"));
        assert!(parse_script("5 Q").unwrap_err().contains("unknown button Q"));
        assert!(parse_script("5 A R=1").is_err());
        let k = parse_script("5 - K=3a,E0\n").unwrap();
        assert_eq!(k, vec![(5, Step::Pad(Pad { keys: [1 << 0x3a, 0, 0, 1 << (0xe0 - 0xc0)], ..Pad::default() }))]);
        assert!(parse_script("5 - K=100").unwrap_err().contains("usages"));
        assert!(parse_script("5 - K=").is_err());
        // A line without mouse words has no mouse; a mouse word wants two counts and 0 or 1.
        assert_eq!(parse_script("5 A L=1,2\n").unwrap()[0].1, Step::Pad(Pad { buttons: 1, l: [1, 2], ..Pad::default() }));
        let m = parse_script("5 - MR=-3,40,1\n").unwrap();
        assert_eq!(m, vec![(5, Step::Pad(Pad { mouse: [None, Some(Mouse { dx: -3, dy: 40, contact: true })], ..Pad::default() }))]);
        for bad in ["5 - ML=1,2", "5 - ML=1,2,2", "5 - ML=1,2,1,0", "5 - ML=a,2,1", "5 - MX=1,2,1"] {
            assert!(parse_script(bad).is_err(), "{bad}");
        }
        // An acceleration word wants three thousandths of g.
        let a = parse_script("5 - AR=-1000,2,-3\n").unwrap();
        assert_eq!(a, vec![(5, Step::Pad(Pad { accel: [None, Some([-1000, 2, -3])], ..Pad::default() }))]);
        for bad in ["5 - AL=1,2", "5 - AL=1,2,3,4", "5 - AL=1.5,2,3", "5 - AX=1,2,3"] {
            assert!(parse_script(bad).is_err(), "{bad}");
        }
    }

    use crate::Config;
    use crate::journal::Journal;
    use crate::result::SUCCESS;
    use gyroship_runtime::mem::{Memory, perm};
    use std::sync::atomic::{AtomicBool, AtomicU64, Ordering::{Acquire, Relaxed, Release}, fence};
    use std::time::{Duration, Instant};

    /// A kernel with hid's shared memory mapped read-only, as the SDK maps it:
    /// the kernel, the shared memory, its guest address.
    fn mapped() -> (Kernel, usize, u64) {
        let mut k = Kernel::new(Memory::new().unwrap(), Journal::memory(), Config::default());
        let m = k.hid_memory();
        let h = k.handles.add(Object::SharedMemory(m)).unwrap();
        let addr = crate::layout::ASLR.0;
        assert_eq!(k.map_shared(h, addr, SHARED_SIZE, perm::R), SUCCESS);
        (k, m, addr)
    }

    /// Sample n: buttons and sticks that say n, so that a torn entry shows.
    fn pad(n: u64) -> Pad {
        Pad { buttons: n, l: [n as i32, -(n as i32)], keys: [n, 0, 0, n << 32], ..Pad::default() }
    }

    /// The ordered writes leave the bytes of the whole copies they replace
    /// (controller, the pair's six-axis lifos and keyboard), in the guest and in
    /// the kernel's copy.
    #[test]
    fn samples_leave_the_bytes_of_a_whole_copy() {
        let (mut k, m, addr) = mapped();
        let at = NPAD + 0x28;
        let mut want = k.shared[m].data.clone();
        let mut guest = vec![0u8; SHARED_SIZE as usize];
        // Acceleration on some samples only: the left side two in three, the right one in two.
        let sample = |n: u64| Pad {
            accel: [(n % 3 != 0).then_some([10 * n as i32, -1, 2]), (n % 2 == 0).then_some([-1000, 0, n as i32])],
            ..pad(n)
        };
        let mut before = Pad::default();
        for n in 1..=40 {
            let p = sample(n);
            push(&mut want[at..at + LIFO_SIZE], &p, CONNECTED);
            for (side, lifo) in DUAL_MOTION_LIFOS.iter().enumerate() {
                if p.accel[side].is_some() || before.accel[side].is_some() {
                    push_motion(&mut want[NPAD + lifo..NPAD + lifo + LIFO_ENTRIES + LIFO_LEN * MOTION_ENTRY], p.accel[side]);
                }
            }
            push_keys(&mut want[KEYBOARD..KEYBOARD + LIFO_ENTRIES + LIFO_LEN * KEY_ENTRY], &p);
            k.push_sample(m, &p, &before);
            k.mem.read_object(addr, &mut guest).unwrap();
            assert!(guest == want && k.shared[m].data == want, "sample {n}");
            before = p;
        }
    }

    /// A six-axis lifo, read as the SDK's GetSixAxisSensorStates does: entries of 0x68, each
    /// sequence even, sampling numbers (the state's second field) one apart; the newest state
    /// gives one sample's time, the acceleration in g, no rotation, the direction matrix at
    /// rest, connected; without acceleration, a null state. A sample writes the acceleration
    /// of each side that has one, one null state on a side whose acceleration goes, and
    /// nothing else.
    #[test]
    fn motion_states_read_back_as_the_sdk_reads_them() {
        let mut lifo = vec![0u8; LIFO_ENTRIES + LIFO_LEN * MOTION_ENTRY];
        lifo[LIFO_COUNT] = 1;
        let u64_at = |l: &[u8], o: usize| u64::from_le_bytes(l[o..o + 8].try_into().unwrap());
        let floats = |l: &[u8], e: usize| (0..18).map(|i| f32::from_le_bytes(l[e + 0x18 + 4 * i..e + 0x1c + 4 * i].try_into().unwrap())).collect::<Vec<_>>();
        let attributes = |l: &[u8], e: usize| u32::from_le_bytes(l[e + 0x60..e + 0x64].try_into().unwrap());
        for n in 1..=40 {
            push_motion(&mut lifo, Some([1000 * n, -500, 250]));
        }
        let (tail, count) = (u64_at(&lifo, LIFO_TAIL) as usize, u64_at(&lifo, LIFO_COUNT) as usize);
        assert_eq!(count, LIFO_LEN);
        let entry = |back: usize| LIFO_ENTRIES + (tail + LIFO_LEN - back) % LIFO_LEN * MOTION_ENTRY;
        for back in 0..count {
            assert_eq!(u64_at(&lifo, entry(back)) % 2, 0, "sequence even");
            if back + 1 < count {
                assert_eq!(u64_at(&lifo, entry(back) + 0x10), u64_at(&lifo, entry(back + 1) + 0x10) + 1, "sampling numbers follow");
            }
        }
        let e = entry(0);
        assert_eq!((u64_at(&lifo, e + 8), u64_at(&lifo, e + 0x10)), (16_666_666, 40), "one sample's time in ns, sampling number");
        let f = floats(&lifo, e);
        assert_eq!(f[..3], [40.0, -0.5, 0.25], "acceleration in g");
        assert_eq!(f[3..9], [0.0; 6], "no angular velocity, no angle");
        assert_eq!(f[9..], [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0], "direction at rest");
        assert_eq!(attributes(&lifo, e), CONNECTED);
        push_motion(&mut lifo, None);
        let e = LIFO_ENTRIES + u64_at(&lifo, LIFO_TAIL) as usize * MOTION_ENTRY;
        assert_eq!((u64_at(&lifo, e + 0x10), floats(&lifo, e), attributes(&lifo, e)), (41, vec![0.0; 18], 0), "a null state");
        // In hid's shared memory: the left side only; when its word goes, one null state.
        let (mut k, m, _) = mapped();
        let count = |k: &Kernel, side: usize| u64_at(&k.shared[m].data, NPAD + DUAL_MOTION_LIFOS[side] + LIFO_COUNT);
        let left = Pad { accel: [Some([1000, 0, 0]), None], ..Pad::default() };
        k.push_sample(m, &left, &Pad::default());
        assert_eq!((count(&k, 0), count(&k, 1)), (2, 1));
        k.push_sample(m, &Pad::default(), &left);
        assert_eq!((count(&k, 0), count(&k, 1)), (3, 1), "a null state");
        k.push_sample(m, &Pad::default(), &Pad::default());
        assert_eq!((count(&k, 0), count(&k, 1)), (3, 1), "then nothing");
    }

    /// With a mouse on either side, Npad 0 shows a Joy-Con pair used as mice while the game
    /// supports that style and not FullKey's, else a Joy-Con pair while it supports that
    /// style; without a mouse, FullKey. Once the style word says a pair, the pair's JoyDual
    /// lifo holds a state that says both Joy-Con are connected (`restyle` publishes it first,
    /// for readers on other threads: this test sees only the end state), and it gets every
    /// sample while a pair is shown; the guest's copy stays equal to the kernel's.
    #[test]
    fn npad_0_shows_a_joy_con_pair_while_there_is_a_mouse_used_as_mice_when_the_game_asks() {
        let (mut k, m, addr) = mapped();
        let word = |k: &Kernel, o: usize| u32::from_le_bytes(k.shared[m].data[o..o + 4].try_into().unwrap());
        let mouse = Pad { buttons: 1, mouse: [Some(Mouse::default()), Some(Mouse::default())], ..Pad::default() };
        // No mouse, or no Joy-Con style supported: FullKey.
        for (supported, pad) in [(0x407, Pad::default()), (DUAL_MOUSE, Pad::default()), (FULL_KEY, mouse)] {
            (k.input.supported, k.input.current) = (supported, pad);
            k.restyle();
            assert_eq!(word(&k, NPAD), FULL_KEY, "{supported:#x}");
        }
        // A mouse on either side is enough, for either pair.
        for side in 0..2 {
            let mut one = Pad::default();
            one.mouse[side] = Some(Mouse::default());
            assert_eq!([style_to_show(DUAL_MOUSE, &one), style_to_show(0x407, &one), style_to_show(0x741f, &one)], [DUAL_MOUSE, JOY_DUAL, JOY_DUAL], "side {side}");
        }
        // Both: the pair, with its JoyDual state.
        let dual = NPAD + JOY_DUAL_LIFO;
        let newest = |k: &Kernel| dual + LIFO_ENTRIES + word(k, dual + LIFO_TAIL) as usize * ENTRY;
        (k.input.supported, k.input.current) = (DUAL_MOUSE, mouse);
        k.restyle();
        assert_eq!(word(&k, NPAD), DUAL_MOUSE);
        assert_eq!(word(&k, dual + LIFO_COUNT), 2, "the null entry, then the pair's state");
        // Attributes: connected, left connected (bit 2), right connected (bit 4); the game
        // takes the pair for one used as mice only with bits 2 and 4.
        assert_eq!((word(&k, newest(&k) + 0x10), word(&k, newest(&k) + 0x28)), (1, 0x15), "buttons, attributes");
        // A sample feeds the pair: its buttons and sticks, both Joy-Con connected.
        let moved = Pad { buttons: 2, l: [5, -6], ..mouse };
        k.push_sample(m, &moved, &mouse);
        assert_eq!(word(&k, dual + LIFO_COUNT), 3, "a sample feeds the pair");
        let e = newest(&k);
        assert_eq!([word(&k, e + 0x10), word(&k, e + 0x18), word(&k, e + 0x1c), word(&k, e + 0x28)], [2, 5, -6i32 as u32, 0x15], "buttons, stick, attributes");
        let mut guest = vec![0u8; SHARED_SIZE as usize];
        k.mem.read_object(addr, &mut guest).unwrap();
        assert!(guest == k.shared[m].data);
        // The game's default styles again: the pair, no longer as mice, still fed.
        k.input.supported = 0x407;
        k.restyle();
        assert_eq!(word(&k, NPAD), JOY_DUAL);
        k.push_sample(m, &mouse, &moved);
        assert_eq!(word(&k, dual + LIFO_COUNT), 5, "the new style's state, then the sample's");
        // The mice gone: FullKey, and the pair's lifo is left alone.
        k.input.current = Pad::default();
        k.restyle();
        assert_eq!(word(&k, NPAD), FULL_KEY);
        k.push_sample(m, &Pad::default(), &mouse);
        assert_eq!(word(&k, dual + LIFO_COUNT), 5);
        let text = k.journal.text();
        for shown in ["a Joy-Con pair used as mice (style 0x1000)", "a Joy-Con pair (style 0x4)", "the FullKey controller (style 0x1)"] {
            assert!(text.contains(&format!("hid: Npad 0 shows {shown}")), "{text}");
        }
    }

    /// Each sample sets the style first: Npad 0 shows the pair (used as mice when the game
    /// supports that style alone) from the first sample whose replay line has a mouse, and
    /// FullKey again from the first one without; each change is logged once.
    #[test]
    fn each_sample_shows_the_pair_while_the_replay_has_a_mouse() {
        for (supported, pair) in [(DUAL_MOUSE, DUAL_MOUSE), (0x407, JOY_DUAL)] {
            let (mut k, m, _) = mapped();
            k.config.replay = Some(parse_script("0 -\n2 - MR=1,0,1\n4 -\n").unwrap());
            k.input.supported = supported;
            let styles: Vec<u32> = (0..6)
                .map(|n| {
                    k.hid_sample(m, n);
                    u32::from_le_bytes(k.shared[m].data[NPAD..NPAD + 4].try_into().unwrap())
                })
                .collect();
            assert_eq!(styles, [FULL_KEY, FULL_KEY, pair, pair, FULL_KEY, FULL_KEY], "{supported:#x}");
            assert_eq!(k.journal.text().matches("hid: Npad 0 shows").count(), 2);
        }
    }

    /// A Joy-Con mouse lifo, read as the SDK's GetJcmsStates does and used as the game
    /// uses what comes back: states newest first, each under an even sequence, sampling
    /// numbers one apart; the game adds the counts of the states newer than the last one
    /// it saw, keeps the contact bit of the newest, and takes bit 0 of the newest state's
    /// attributes for its controller having a mouse.
    #[test]
    fn mouse_states_read_back_as_the_game_adds_them() {
        assert_eq!(JCMS_LIFO as usize, LIFO_ENTRIES + LIFO_LEN * MOUSE_ENTRY);
        let u64_at = |l: &[u8], o: usize| u64::from_le_bytes(l[o..o + 8].try_into().unwrap());
        let i32_at = |l: &[u8], o: usize| i32::from_le_bytes(l[o..o + 4].try_into().unwrap());
        // (sampling number, dx, dy, attributes) of up to 16 states, newest first.
        let read = |l: &[u8]| -> Vec<(u64, i32, i32, u8)> {
            let (tail, count) = (u64_at(l, LIFO_TAIL) as usize, u64_at(l, LIFO_COUNT) as usize);
            assert!(tail < LIFO_LEN && count <= LIFO_LEN);
            let states: Vec<_> = (0..count.min(16))
                .map(|back| {
                    let e = LIFO_ENTRIES + (tail + LIFO_LEN - back) % LIFO_LEN * MOUSE_ENTRY;
                    assert_eq!(u64_at(l, e) % 2, 0, "sequence even");
                    (u64_at(l, e + 0x10), i32_at(l, e + 0x18), i32_at(l, e + 0x1c), l[e + 0x20])
                })
                .collect();
            assert!(states.windows(2).all(|w| w[0].0 == w[1].0 + 1), "sampling numbers follow: {states:?}");
            states
        };
        // As StartJcms leaves it: one null state, which says "no mouse".
        let mut lifo = vec![0u8; JCMS_LIFO as usize];
        lifo[LIFO_COUNT] = 1;
        assert_eq!(read(&lifo), [(0, 0, 0, 0)]);
        // The game's side: the last sampling number seen, the totals, the contact bit.
        let (mut seen, mut total, mut contact) = (0, [0i64; 2], false);
        let mut poll = |l: &[u8]| {
            for &(n, dx, dy, attributes) in read(l).iter().rev() {
                if n > seen {
                    (seen, contact) = (n, attributes >> 1 & 1 != 0);
                    total = [total[0] + i64::from(dx), total[1] + i64::from(dy)];
                }
            }
            (total, contact, read(l)[0].3 & 1 != 0)
        };
        assert_eq!(poll(&lifo), ([0, 0], false, false));
        // Forty samples, polled every third one: nothing is lost or counted twice.
        let mut want = [0i64; 2];
        for n in 1..=40i32 {
            push_mouse(&mut lifo, Some(Mouse { dx: n, dy: -2 * n, contact: n % 2 == 1 }));
            want = [want[0] + i64::from(n), want[1] - 2 * i64::from(n)];
            if n % 3 == 0 {
                assert_eq!(poll(&lifo), (want, n % 2 == 1, true), "sample {n}");
            }
        }
        assert_eq!(lifo[LIFO_COUNT], 17);
        assert_eq!(read(&lifo)[0], (40, 40, -80, 1), "the newest state: its number, its counts, a mouse, in the air");
        // The mouse goes: one null state, and the game no longer sees a mouse.
        push_mouse(&mut lifo, None);
        assert_eq!(read(&lifo)[0], (41, 0, 0, 0));
        assert!(!poll(&lifo).2);
    }

    /// Samples with a mouse reach the buffer the game lent (lifos 0 and 1: Npad 0, left
    /// and right), a sample without one writes nothing, and a mouse that goes leaves one
    /// null state behind it.
    #[test]
    fn a_mouse_is_written_where_the_game_reads_it() {
        let mut k = Kernel::new(Memory::new().unwrap(), Journal::memory(), Config::default());
        // The game's own memory: once lent it is read-only for the game, never for the host.
        let addr = crate::layout::ALIAS.0;
        k.mem.map(addr, 0x1000, gyroship_runtime::mem::Block::new(gyroship_runtime::mem::state::NORMAL, perm::RW)).unwrap();
        k.input.jcms = Some((addr, 0x1000));
        let lifo = |k: &Kernel, side: u64| {
            let mut l = vec![0u8; JCMS_LIFO as usize];
            k.mem.read(addr + side * JCMS_LIFO, &mut l).unwrap();
            l
        };
        let left = Mouse { dx: 5, dy: -6, contact: true };
        let (none, with) = (Pad::default(), Pad { mouse: [Some(left), None], ..Pad::default() });
        k.push_mice(&none, &none);
        assert_eq!(lifo(&k, 0), vec![0; JCMS_LIFO as usize], "no mouse, nothing written");
        k.push_mice(&with, &none);
        k.push_mice(&with, &with);
        let mut want = vec![0u8; JCMS_LIFO as usize];
        push_mouse(&mut want, Some(left));
        push_mouse(&mut want, Some(left));
        assert_eq!(lifo(&k, 0), want);
        assert_eq!(lifo(&k, 1), vec![0; JCMS_LIFO as usize], "the right side has none");
        k.push_mice(&none, &with);
        k.push_mice(&none, &none);
        push_mouse(&mut want, None);
        assert_eq!(lifo(&k, 0), want, "one null state when the mouse goes, then nothing");
        // A buffer too small for the right side's lifo: left alone.
        k.input.jcms = Some((addr, JCMS_LIFO + 8));
        k.push_mice(&Pad { mouse: [None, Some(left)], ..Pad::default() }, &none);
        assert_eq!(lifo(&k, 1), vec![0; JCMS_LIFO as usize]);
    }

    /// GetNpadStates as another core runs it, on the guest's lifo at host
    /// address `lifo`: the `n` newest entries, oldest first, each under a stable
    /// even sequence, sampling numbers consecutive, else start over (the SDK
    /// spins forever; here, a million tries). (sampling number, buttons), newest last.
    fn read_racing(lifo: usize, n: usize) -> Vec<(u64, u64)> {
        // SAFETY: aligned words of a lifo that stays mapped while the readers run.
        let word = |o: usize| unsafe { AtomicU64::from_ptr((lifo + o) as *mut u64) };
        for _ in 0..1_000_000 {
            let tail = word(LIFO_TAIL).load(Acquire) as usize;
            let n = n.min(word(LIFO_COUNT).load(Acquire) as usize);
            let mut out: Vec<(u64, u64)> = Vec::with_capacity(n);
            for i in 0..n {
                let e = LIFO_ENTRIES + (tail + LIFO_LEN + 1 - n + i) % LIFO_LEN * ENTRY;
                let seq = word(e).load(Acquire);
                let state = (word(e + 8).load(Relaxed), word(e + 0x10).load(Relaxed));
                fence(Acquire); // the reader's DMB ISHLD
                if seq & 1 != 0 || word(e).load(Relaxed) != seq || out.last().is_some_and(|p| state.0 != p.0 + 1) {
                    break;
                }
                out.push(state);
            }
            if out.len() == n {
                return out;
            }
        }
        panic!("the reader spins: the lifo stays inconsistent");
    }

    /// While the sampler pushes, readers on other threads (one state, as the
    /// game asks, and all 17) only ever accept whole entries, and never an
    /// older newest entry: no new tail shows before its entry.
    #[test]
    fn readers_on_other_threads_see_whole_samples_in_order() {
        const SAMPLES: u64 = 100_000;
        let (mut k, m, addr) = mapped();
        let lifo = k.mem.base() as usize + addr as usize + NPAD + 0x28;
        let (done, deadline) = (AtomicBool::new(false), Instant::now() + Duration::from_secs(60));
        std::thread::scope(|s| {
            let readers = [1, LIFO_LEN].map(|n| {
                let done = &done;
                s.spawn(move || {
                    let mut newest = 0;
                    loop {
                        let stop = done.load(Acquire) || Instant::now() > deadline;
                        let got = read_racing(lifo, n);
                        assert!(got.iter().all(|&(sample, buttons)| buttons == sample), "a torn entry: {got:?}");
                        let latest = got.last().unwrap().0;
                        assert!(latest >= newest, "sample {latest} after {newest}: a stale entry");
                        newest = latest;
                        if stop {
                            return newest;
                        }
                    }
                })
            });
            for n in 1..=SAMPLES {
                k.push_sample(m, &pad(n), &pad(n - 1));
            }
            done.store(true, Release);
            for r in readers {
                assert_eq!(r.join().unwrap(), SAMPLES, "the last sample, read once the writer stopped");
            }
        });
    }
}
