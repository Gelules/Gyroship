//! `hid`: one controller (FullKey, slot 0) plugged in, in the shared memory
//! layout the SDK reads (the Switch 1 one, our reading of the Switch 2 SDK's
//! readers). Every lifo starts with one null entry: its sequence is even (no
//! write in progress), so the readers never spin. A virtual timer samples
//! the controller once per frame, 1 ms before the vsync (D29), from the
//! `--replay` script or at rest.

use crate::Kernel;
use crate::ipc::{Reply, Request};
use crate::service::Service;
use crate::thread::{Object, Timer, TimerAction};

pub const SHARED_SIZE: u64 = 0x40000;
/// Touch screen, mouse and keyboard lifos.
const BASIC_LIFOS: [usize; 3] = [0x400, 0x3400, 0x3800];
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
/// A button lifo: tail at +0x10, count at +0x18, 17 entries of 0x30 from
/// +0x20; an entry is its sequence, then the state: sampling number,
/// buttons, stick L, stick R, attributes.
const LIFO_TAIL: usize = 0x10;
const LIFO_COUNT: usize = 0x18;
const LIFO_ENTRIES: usize = 0x20;
const LIFO_LEN: usize = 17;
const ENTRY: usize = 0x30;
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

/// A controller's state: buttons and sticks (±0x7fff).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Pad {
    pub buttons: u64,
    pub l: [i32; 2],
    pub r: [i32; 2],
}

/// One line of a `--replay` script: from this sample on, this state; or
/// the end of the recording.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Step {
    Pad(Pad),
    End,
}

/// Parses a `--replay` script: `<sample> <buttons|-> [L=x,y] [R=x,y]` or
/// `<sample> end`, buttons joined by `+`, samples increasing; `#` starts a
/// comment.
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
                    let (stick, xy) = w.split_once('=').ok_or_else(|| bad("expected L=x,y or R=x,y"))?;
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

/// Appends `pad` to a button lifo (the bytes from its header on), as the
/// SDK's reader expects: an even sequence, a sampling number one past the
/// previous entry's, then the tail, then the count.
pub fn push(lifo: &mut [u8], pad: &Pad) {
    let u64_at = |b: &[u8], o: usize| u64::from_le_bytes(b[o..o + 8].try_into().unwrap());
    let tail = u64_at(lifo, LIFO_TAIL) as usize % LIFO_LEN;
    let sample = u64_at(lifo, LIFO_ENTRIES + tail * ENTRY + 8) + 1;
    let next = (tail + 1) % LIFO_LEN;
    let mut entry = [0u8; ENTRY];
    entry[0..8].copy_from_slice(&(sample << 1).to_le_bytes());
    entry[8..0x10].copy_from_slice(&sample.to_le_bytes());
    entry[0x10..0x18].copy_from_slice(&pad.buttons.to_le_bytes());
    for (i, v) in pad.l.iter().chain(&pad.r).enumerate() {
        entry[0x18 + 4 * i..0x1c + 4 * i].copy_from_slice(&v.to_le_bytes());
    }
    entry[0x28..0x2c].copy_from_slice(&CONNECTED.to_le_bytes());
    lifo[LIFO_ENTRIES + next * ENTRY..][..ENTRY].copy_from_slice(&entry);
    lifo[LIFO_TAIL..LIFO_TAIL + 8].copy_from_slice(&(next as u64).to_le_bytes());
    let count = (u64_at(lifo, LIFO_COUNT) + 1).min(LIFO_LEN as u64);
    lifo[LIFO_COUNT..LIFO_COUNT + 8].copy_from_slice(&count.to_le_bytes());
}

/// Where the replay script stands (plain data, for save states).
#[derive(Clone, Copy, Debug, Default)]
pub struct Input {
    /// The script's next line.
    pub next: usize,
    pub current: Pad,
    /// The last sample taken.
    pub last: u64,
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
            // Activations, touch screen size, six-axis and Jcms start.
            (Hid::Service { .. }, 11 | 21 | 31 | 66 | 1004 | 3112 | 3150) => Reply::ok(),
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
                    k.log(&format!("hid 3110: jcms buffer {addr:#x}+{size:#x} zeroed"));
                }
                Reply::ok()
            }
            // SetSupportedNpadStyleSet / GetSupportedNpadStyleSet.
            (Hid::Service { styles, .. }, 100) => {
                let s = req.u32_at(0)?;
                if *styles != s {
                    k.log(&format!("hid: supported style set {s:#x}"));
                }
                *styles = s;
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
        self.npad_condition(1);
        // Sample k comes SAMPLE_LEAD before vsync k + 1 (to a tick, on the 60 Hz grid).
        let period = TICKS_PER_SECOND / 60;
        let k = (self.now() * 60).div_ceil(TICKS_PER_SECOND);
        self.timers.push(Timer { start: period - SAMPLE_LEAD, k, num: TICKS_PER_SECOND, den: 60, action: TimerAction::HidSample { shared: m } });
        m
    }

    /// Timer `HidSample`: the state of sample `k` (from the replay script, else
    /// at rest) goes into slot 0's FullKey lifo. Logged when it changes; the
    /// script's end stops the run.
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
        if first || input.current != self.input.current {
            let p = input.current;
            self.log(&format!("hid: sample {k} (present {}): buttons {:#x} L {:?} R {:?}", self.gs.presents, p.buttons, p.l, p.r));
            if let Some(lines) = self.recorded.as_mut() {
                lines.push(script_line(k, &p));
            }
        }
        input.last = k;
        self.input = input;
        let at = NPAD + 0x28;
        let mut lifo = self.shared[m].data[at..at + LIFO_SIZE].to_vec();
        push(&mut lifo, &input.current);
        self.write_shared(m, at, &lifo);
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
        Pad { buttons: u64_at(e + 0x10), l: [i32_at(e + 0x18), i32_at(e + 0x1c)], r: [i32_at(e + 0x20), i32_at(e + 0x24)] }
    }

    #[test]
    fn forty_samples_read_back_as_the_last_one() {
        // As hid_memory leaves it: one null entry, count 1.
        let mut lifo = vec![0u8; LIFO_SIZE];
        lifo[LIFO_COUNT] = 1;
        let mut last = Pad::default();
        for n in 1..=40 {
            last = Pad { buttons: n % 3, l: [n as i32, -(n as i32)], r: [0x7fff, -0x7fff] };
            push(&mut lifo, &last);
            assert_eq!(read(&lifo), last, "sample {n}");
        }
        assert_eq!(lifo[LIFO_COUNT], 17);
        assert_eq!(u64::from_le_bytes(lifo[LIFO_ENTRIES + 6 * ENTRY + 8..][..8].try_into().unwrap()), 40, "sampling number of the newest entry (tail 40 mod 17)");
        assert_eq!(read(&lifo), last);
    }

    /// What `--record` writes, `--replay` reads back unchanged.
    #[test]
    fn recorded_lines_replay_as_written() {
        let pads = [Pad::default(), Pad { buttons: 1 | 1 << 13 | 1 << 27, l: [-0x7fff, 0x7fff], r: [0, 0] }, Pad { buttons: 0, l: [0, 0], r: [3, -4] }];
        let text: String = pads.iter().enumerate().map(|(i, p)| script_line(10 * i as u64, p) + "\n").collect::<String>() + "40 end\n";
        let expected: Vec<(u64, Step)> = pads.iter().enumerate().map(|(i, p)| (10 * i as u64, Step::Pad(*p))).chain([(40, Step::End)]).collect();
        assert_eq!(parse_script(&text).unwrap(), expected);
        assert_eq!(script_line(7, &pads[1]), "7 A+Up+RightSR L=-32767,32767");
    }

    #[test]
    fn replay_scripts_parse() {
        let s = parse_script("# title screen\n10 A+Up L=0,32767\n16 -\n20 end\n").unwrap();
        assert_eq!(s, vec![(10, Step::Pad(Pad { buttons: 1 | 1 << 13, l: [0, 32767], r: [0, 0] })), (16, Step::Pad(Pad::default())), (20, Step::End)]);
        assert!(parse_script("5 A\n5 B").unwrap_err().contains("increase"));
        assert!(parse_script("5 Q").unwrap_err().contains("unknown button Q"));
        assert!(parse_script("5 A R=1").is_err());
    }
}
