//! Practice actions (M6, D73): with `--practice FILE`, a `trace:` probe's pass may also write
//! game state, or read it for Gyroship's overlay. Lines name a probe's pc and the fields they
//! use, as items of the trace grammar, so the game's offsets stay in the data file (D21):
//!
//! ```text
//! boost <pc> <vehicle> gauge=f32@EXPR lock=u8@EXPR
//! hud-at <x> <y>
//! hud-speed <pc> s=ITEM m=ITEM b=ITEM
//! font <region tag | latin> <data.zip entry>
//! ```
//!
//! `--boost infinite`: as each boost update starts, the gauge (a heat that boosting and braking
//! raise: 0 is a full bar) goes back to 0 and the boost-and-brake lock is lifted, so Boost held
//! never runs out, a release never locks, and a big hit's lock is gone by the next update. A
//! run with practice actions is never a reference: the journal says so at boot.
//!
//! `--hud`: at a `hud-speed` line's pc, the vehicle's speed m·(s + b) (multiplier, base speed,
//! boost speed; each `sN` or `f32@EXPR`) is read and published with the present it was read
//! in, for the speed line drawn from `hud-at` (presented-image pixels). No guest write. `font`
//! lines are `gyroship_practice::font`'s.

use crate::trace::{self, Item, Ty};
use gyroship_gpu::datazip::DataZip;
use gyroship_gpu::surfaces::Mask;
use gyroship_practice::{font, texts};
use gyroship_runtime::Cpu;
use gyroship_runtime::mem::Memory;
use std::collections::VecDeque;
use std::sync::atomic::{AtomicU64, Ordering::Relaxed};
use std::sync::{Arc, OnceLock};

/// The run's actions: set at boot with `--practice`, never otherwise.
static ACTIONS: OnceLock<Practice> = OnceLock::new();

/// `gs.presents`, for code without the kernel lock: one store in `Kernel::present`.
pub(crate) static PRESENTS: AtomicU64 = AtomicU64::new(0);

/// What `--boost` asks.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Boost {
    Infinite,
}

impl Boost {
    pub fn parse(s: &str) -> Option<Boost> {
        (s == "infinite").then_some(Boost::Infinite)
    }

    pub fn name(self) -> &'static str {
        match self {
            Boost::Infinite => "infinite",
        }
    }
}

/// A boost update: its probe's pc, the vehicle, and its fields by name.
type Line = (u64, String, Vec<(String, Item)>);

pub struct Practice {
    mem: Arc<Memory>,
    boost: Option<Boost>,
    /// By pc.
    lines: Vec<Line>,
    /// `--hud`: the speed is read.
    hud: bool,
    /// `hud-speed` lines: the pc, then s, m and b.
    speeds: Vec<(u64, [Item; 3])>,
    /// `hud-at`: the speed line's origin, in presented-image pixels.
    hud_at: Option<(u32, u32)>,
    /// The last speed read: its f32 bits << 32 | the low 32 bits of its present; 0: none yet.
    speed: AtomicU64,
}

impl Practice {
    /// The file's lines, checked against this build's probes (`probes`, as
    /// `gyroship_runtime::probe_counts` gives them): each probe line's pc is a `trace:` probe,
    /// and carries the fields its mode needs.
    pub fn parse(text: &str, boost: Option<Boost>, hud: bool, mem: Arc<Memory>, probes: &[(u64, &str, u64)]) -> Result<Practice, String> {
        let mut p = Practice { mem, boost, lines: Vec::new(), hud, speeds: Vec::new(), hud_at: None, speed: AtomicU64::new(0) };
        for (n, line) in text.lines().enumerate().filter(|(_, l)| !l.trim().is_empty() && !l.trim_start().starts_with('#')) {
            let at = || format!("line {}", n + 1);
            let mut words = line.split_whitespace();
            let kind = words.next().unwrap_or_default();
            match kind {
                "boost" | "hud-speed" => {}
                "hud-at" => {
                    let xy: Option<Vec<u32>> = words.map(|w| w.parse().ok()).collect();
                    let Some(&[x, y]) = xy.as_deref() else { return Err(format!("{}: hud-at takes X and Y, in pixels", at())) };
                    if p.hud_at.replace((x, y)).is_some() {
                        return Err(format!("{}: a second hud-at", at()));
                    }
                    continue;
                }
                "font" => continue,
                w => return Err(format!("{}: unknown kind {w}", at())),
            }
            let pc = words.next().and_then(|w| u64::from_str_radix(w, 16).ok()).ok_or_else(|| format!("{}: no pc", at()))?;
            if !probes.iter().any(|(a, name, _)| *a == pc && name.starts_with("trace:")) {
                return Err(format!("{}: {pc:#x} is not a trace: probe of this build", at()));
            }
            if kind == "hud-speed" {
                let mut items = [None, None, None];
                for w in words {
                    let (name, item) = w.split_once('=').ok_or_else(|| format!("{}: {w} is not name=item", at()))?;
                    let i = ["s", "m", "b"].iter().position(|n| *n == name).ok_or_else(|| format!("{}: hud-speed takes s, m and b, not {name}", at()))?;
                    items[i] = Some(trace::item(item).filter(|it| matches!(it, Item::S(_) | Item::Mem(Ty::F32, _))).ok_or_else(|| format!("{}: {item} is not sN or f32@EXPR", at()))?);
                }
                let [Some(s), Some(m), Some(b)] = items else { return Err(format!("{}: hud-speed needs s, m and b", at())) };
                p.speeds.push((pc, [s, m, b]));
                continue;
            }
            let vehicle = words.next().ok_or_else(|| format!("{}: no vehicle", at()))?.to_string();
            let mut fields = Vec::new();
            for w in words {
                let (name, item) = w.split_once('=').ok_or_else(|| format!("{}: {w} is not name=item", at()))?;
                let item = trace::item(item).filter(|i| matches!(i, Item::Mem(..))).ok_or_else(|| format!("{}: {item} is not TYPE@EXPR", at()))?;
                fields.push((name.to_string(), item));
            }
            for need in boost.map_or(&[][..], |_| &["gauge", "lock"][..]) {
                if !fields.iter().any(|(f, _)| f == need) {
                    return Err(format!("{}: {vehicle} has no {need} field", at()));
                }
            }
            p.lines.push((pc, vehicle, fields));
        }
        if hud && !p.speeds.is_empty() && p.hud_at.is_none() {
            return Err("hud-speed lines need a hud-at line".into());
        }
        p.lines.sort_by_key(|l| l.0);
        Ok(p)
    }

    /// A pass of a `trace:` probe at `pc`: its line's actions. A field whose pointers cannot
    /// be read is left alone.
    pub(crate) fn act(&self, pc: u64, cpu: &Cpu) {
        let Some(Boost::Infinite) = self.boost else { return };
        let Ok(i) = self.lines.binary_search_by_key(&pc, |l| l.0) else { return };
        for (name, item) in &self.lines[i].2 {
            if matches!(name.as_str(), "gauge" | "lock")
                && let Some((addr, size)) = trace::place(item, cpu, &self.mem)
            {
                let _ = self.mem.write(addr, &[0u8; 8][..size]);
            }
        }
    }

    /// With `--hud`, a pass at a `hud-speed` line's pc: s, m and b read where the line says,
    /// and m·(s + b) published with the present it was read in. No guest write. None, and
    /// nothing published, when a read fails; else `[s, m, b, m·(s + b)]`.
    pub(crate) fn read_speed(&self, pc: u64, cpu: &Cpu) -> Option<[f32; 4]> {
        if !self.hud {
            return None;
        }
        let (_, items) = self.speeds.iter().find(|l| l.0 == pc)?;
        let [Some(s), Some(m), Some(b)] = items.each_ref().map(|i| trace::f32_of(i, cpu, &self.mem)) else { return None };
        let v = m * (s + b);
        if !v.is_finite() {
            return None;
        }
        // ponytail: the stamp keeps the present's low 32 bits, 2.27 years at 60 Hz.
        self.speed.store(u64::from(v.to_bits()) << 32 | u64::from(PRESENTS.load(Relaxed) as u32), Relaxed);
        Some([s, m, b, v])
    }

    /// The speed to show with present `shown`: the last one read, while its present is at
    /// least `shown` − 6. A later present counts as fresh: in play mode the GPU thread shows a
    /// present the guest has already passed.
    pub(crate) fn speed_at(&self, shown: u64) -> Option<f32> {
        let w = self.speed.load(Relaxed);
        let stamp = w & 0xffff_ffff;
        (stamp != 0 && stamp + 6 >= shown).then(|| f32::from_bits((w >> 32) as u32))
    }

    /// `hud-at`: the speed line's origin, in presented-image pixels.
    pub(crate) fn hud_at(&self) -> Option<(u32, u32)> {
        self.hud_at
    }
}

/// A mask made, with what it was made for: the text, the image size and the output size.
type Made = (String, (u32, u32), (u32, u32), Arc<Mask>);

/// `--hud`'s text: the speed line, then the frame rate. Their labels in the run's language,
/// the origin (presented-image pixels), the font that draws them, the presents of the last
/// second, and the last mask made.
pub struct Hud {
    label: &'static str,
    fps_label: &'static str,
    origin: (u32, u32),
    face: font::Face,
    presents: VecDeque<u64>,
    /// When the text was loaded: the wall clock's origin.
    born: std::time::Instant,
    last: Option<Made>,
}

/// Presents per second at `now` (ticks of 32 ns): `now` joins `window`, which keeps the
/// presents of the last second, and the rate is their intervals over their span, so that a
/// present handed over a little later after its vsync than the one a second before cannot
/// add one.
fn rate(window: &mut VecDeque<u64>, now: u64) -> u32 {
    window.push_back(now);
    while window.front().is_some_and(|&t| t + crate::gs::TICKS_PER_SECOND <= now) {
        window.pop_front();
    }
    match now.saturating_sub(window[0]) {
        0 => 0,
        span => ((window.len() - 1) as f64 * crate::gs::TICKS_PER_SECOND as f64 / span as f64).round() as u32,
    }
}

impl Hud {
    /// The speed line's font: the `font` line of the run's region (`language`, else en-US),
    /// read from the dump's `data.zip`. Says on stderr, never in the journal, which font it
    /// uses and what it cannot draw.
    pub fn load(practice: &str, origin: Option<(u32, u32)>, zip: &std::path::Path, language: Option<&str>) -> Result<Hud, String> {
        let region = texts::region(language.unwrap_or("en-US"));
        let id = font::entry(practice, region)?;
        let mut z = DataZip::open(zip).map_err(|e| format!("{}: {e}", zip.display()))?;
        let i = z.entries.iter().position(|e| u64::from_str_radix(&e.name, 16) == Ok(id)).ok_or_else(|| format!("{}: no entry {id:x}", zip.display()))?;
        let entry = z.entries.swap_remove(i);
        let face = font::Face::load(&z.read(&entry).map_err(|e| format!("{}: entry {id:x}: {e}", zip.display()))?)?;
        let missing: Vec<String> = face.missing(&font::needed(region)).iter().map(|c| format!("U+{:04X}", u32::from(*c))).collect();
        if !missing.is_empty() {
            eprintln!("gyroship: --hud: {} cannot draw {}", face.source, missing.join(" "));
        }
        eprintln!("gyroship: --hud: {} texts in {}", texts::tag(region), face.source);
        Ok(Hud { label: texts::text(texts::Text::Speed, region), fps_label: texts::text(texts::Text::Fps, region), origin: origin.unwrap_or_default(), face, presents: VecDeque::new(), born: std::time::Instant::now(), last: None })
    }

    /// The text to draw with present `n`, handed over at `now` (virtual ticks): the speed line
    /// while a fresh speed was read (else an empty line, so the counter keeps its place), then
    /// the frame rate. With a window, on the wall clock: the rate it shows. Without, on the
    /// virtual clock: the game's own rate, the same in every run, so dumps stay reproducible.
    pub(crate) fn text(&mut self, n: u64, now: u64, windowed: bool) -> String {
        let at = if windowed { (self.born.elapsed().as_nanos() / 32) as u64 } else { now };
        let speed = actions().and_then(|p| p.speed_at(n)).map_or(String::new(), |v| format!("{} {}", self.label, v.round() as i64));
        format!("{speed}\n{} {}", self.fps_label, rate(&mut self.presents, at))
    }

    /// The mask of `text` for an output of `size` pixels showing an image of `area`: the
    /// glyphs at the output's scale, stretched back over the image, so that each output pixel
    /// shows one mask texel. The last one again when nothing changed.
    pub(crate) fn mask(&mut self, text: &str, area: (u32, u32), size: (u32, u32)) -> Option<Arc<Mask>> {
        if size.0 == 0 || size.1 == 0 {
            return None;
        }
        if let Some((t, a, s, m)) = &self.last
            && (t.as_str(), *a, *s) == (text, area, size)
        {
            return Some(m.clone());
        }
        let m = self.face.mask(text, font::px(size.1));
        let rect = [self.origin.0 as f32, self.origin.1 as f32, m.width as f32 * area.0 as f32 / size.0 as f32, m.height as f32 * area.1 as f32 / size.1 as f32];
        let mask = Arc::new(Mask::new(rect, (m.width, m.height), m.bytes, [1.0; 4])?);
        self.last = Some((text.to_string(), area, size, mask.clone()));
        Some(mask)
    }
}

/// Installs the run's actions (at boot, once).
pub fn install(p: Practice) {
    let _ = ACTIONS.set(p);
}

/// The run's actions, if `--practice` gave any.
pub(crate) fn actions() -> Option<&'static Practice> {
    ACTIONS.get()
}

#[cfg(test)]
mod tests {
    use super::*;
    use gyroship_runtime::mem::{Block, perm, state};

    const AT: u64 = 0x8000_0000;
    const PC: u64 = 0x7100_1000;

    /// A vehicle at [x0] + 0x100: a gauge 1.5 at +0x10, a lock 1 at +0x14, a neighbour 9 at +0x15.
    fn vehicle() -> (Arc<Memory>, Cpu) {
        let mem = Arc::new(Memory::new().unwrap());
        mem.map(AT, 0x2000, Block::new(state::NORMAL, perm::RW)).unwrap();
        mem.write(AT, &(AT + 0x100).to_le_bytes()).unwrap();
        mem.write(AT + 0x110, &1.5f32.to_le_bytes()).unwrap();
        mem.write(AT + 0x114, &[1, 9]).unwrap();
        let mut cpu = Cpu::new(std::ptr::null_mut());
        cpu.x[0] = AT;
        (mem, cpu)
    }

    const PROBES: [(u64, &str, u64); 2] = [(PC, "trace:aw-update x0", 0), (PC + 4, "nnMain", 0)];
    const FILE: &str = "# synthetic\nboost 71001000 arwing gauge=f32@[x0]+0x10 lock=u8@[x0]+0x14\n";

    fn state(mem: &Memory) -> (f32, [u8; 2]) {
        let (mut g, mut l) = ([0u8; 4], [0u8; 2]);
        mem.read(AT + 0x110, &mut g).unwrap();
        mem.read(AT + 0x114, &mut l).unwrap();
        (f32::from_le_bytes(g), l)
    }

    #[test]
    fn infinite_boost_empties_the_gauge_and_lifts_the_lock_where_the_file_says() {
        let (mem, cpu) = vehicle();
        let p = Practice::parse(FILE, Some(Boost::Infinite), false, mem.clone(), &PROBES).unwrap();
        p.act(PC + 4, &cpu);
        assert_eq!(state(&mem), (1.5, [1, 9]), "another pc");
        p.act(PC, &cpu);
        assert_eq!(state(&mem), (0.0, [0, 9]));
        // Off: the same line writes nothing.
        let (mem, cpu) = vehicle();
        Practice::parse(FILE, None, false, mem.clone(), &PROBES).unwrap().act(PC, &cpu);
        assert_eq!(state(&mem), (1.5, [1, 9]));
        // A pointer that cannot be read: nothing written, no fault.
        let (mem, mut cpu) = vehicle();
        cpu.x[0] = 0x10;
        Practice::parse(FILE, Some(Boost::Infinite), false, mem.clone(), &PROBES).unwrap().act(PC, &cpu);
        assert_eq!(state(&mem), (1.5, [1, 9]));
    }

    #[test]
    fn a_line_must_name_a_trace_probe_and_its_fields() {
        let mem = vehicle().0;
        let parse = |text: &str| Practice::parse(text, Some(Boost::Infinite), false, mem.clone(), &PROBES).err().unwrap_or_default();
        assert!(parse("boost 71001004 arwing gauge=f32@x0 lock=u8@x0").contains("not a trace: probe"));
        assert!(parse("boost 71002000 arwing gauge=f32@x0 lock=u8@x0").contains("not a trace: probe"));
        assert!(parse("boost 71001000 arwing gauge=f32@x0").contains("no lock field"));
        assert!(parse("boost 71001000 arwing gauge=x0 lock=u8@x0").contains("not TYPE@EXPR"));
        assert!(parse("speed 71001000 arwing").contains("unknown kind"));
        assert_eq!(parse(FILE), "");
    }

    /// s in s3, m at [x0] + 0x18, b at [[x0] + 0x8] + 0x4: a synthetic layout.
    const HUD: &str = "hud-at 7 9\nhud-speed 71001000 s=s3 m=f32@[x0]+0x18 b=f32@[[x0]+0x8]+0x4\n";

    fn speed_state() -> (Arc<Memory>, Cpu) {
        let (mem, mut cpu) = vehicle();
        mem.write(AT + 0x118, &0.5f32.to_le_bytes()).unwrap();
        mem.write(AT + 0x108, &(AT + 0x200).to_le_bytes()).unwrap();
        mem.write(AT + 0x204, &30.000002f32.to_le_bytes()).unwrap();
        cpu.vr[3] = u128::from(1200f32.to_bits());
        (mem, cpu)
    }

    fn block(mem: &Memory) -> Vec<u8> {
        let mut b = vec![0u8; 0x2000];
        mem.read(AT, &mut b).unwrap();
        b
    }

    /// The only test that writes `PRESENTS`.
    #[test]
    fn the_speed_line_reads_s_m_and_b_and_a_bad_pointer_publishes_nothing() {
        let (mem, mut cpu) = speed_state();
        let before = block(&mem);
        PRESENTS.store(41, Relaxed);
        let p = Practice::parse(HUD, None, true, mem.clone(), &PROBES).unwrap();
        assert_eq!(p.hud_at(), Some((7, 9)));
        assert_eq!(p.read_speed(PC + 4, &cpu), None, "another pc");
        assert_eq!(p.speed_at(41), None, "nothing read yet");
        assert_eq!(p.read_speed(PC, &cpu), Some([1200.0, 0.5, 30.000002, 615.0]));
        for (shown, fresh) in [(41, true), (47, true), (48, false), (40, true)] {
            assert_eq!(p.speed_at(shown), fresh.then_some(615.0), "shown {shown}");
        }
        assert_eq!(block(&mem), before, "no guest write");
        // b's pointer alone unreadable, then x0 alone: nothing read, the last value stays.
        mem.write(AT + 0x108, &0x10u64.to_le_bytes()).unwrap();
        assert_eq!(p.read_speed(PC, &cpu), None);
        mem.write(AT + 0x108, &(AT + 0x200).to_le_bytes()).unwrap();
        cpu.x[0] = 0x10;
        assert_eq!(p.read_speed(PC, &cpu), None);
        assert_eq!(p.speed_at(41), Some(615.0));
        // Without --hud, the same file reads nothing.
        cpu.x[0] = AT;
        assert_eq!(Practice::parse(HUD, None, false, mem, &PROBES).unwrap().read_speed(PC, &cpu), None);
    }

    #[test]
    fn the_frame_rate_counts_the_presents_of_the_last_second() {
        let second = crate::gs::TICKS_PER_SECOND;
        let mut window = VecDeque::new();
        // 60 Hz: right from the second present, and on.
        let counts: Vec<u32> = (0..180).map(|k| rate(&mut window, k * second / 60)).collect();
        assert_eq!((counts[0], counts[1], counts[59], counts[60], counts[179]), (0, 60, 60, 60, 60));
        // Then 30 Hz: down to 30 within a second.
        let start = 180 * second / 60;
        let counts: Vec<u32> = (1..=60).map(|k| rate(&mut window, start + k * second / 30)).collect();
        assert_eq!((counts[0] > 30, counts[29], counts[59]), (true, 30, 30));
        // 30 Hz with each present handed over some time after its vsync (its frame's work
        // varies): still 30, never 31.
        let mut window = VecDeque::new();
        let late = [0, 12_000, 5_000, 9_000];
        let counts: Vec<u32> = (0..120u64).map(|k| rate(&mut window, k * second / 30 + late[k as usize % 4])).collect();
        assert!(counts[1..].iter().all(|&c| c == 30), "{counts:?}");
    }

    #[test]
    fn hud_lines_are_checked() {
        let mem = vehicle().0;
        let parse = |text: &str| Practice::parse(text, None, true, mem.clone(), &PROBES).err().unwrap_or_default();
        assert!(parse("hud-at 7 9\nhud-speed 71001004 s=s3 m=s4 b=s5").contains("not a trace: probe"));
        assert!(parse("hud-at 7 9\nhud-speed 71001000 s=s3 m=s4").contains("needs s, m and b"));
        assert!(parse("hud-at 7 9\nhud-speed 71001000 s=x3 m=s4 b=s5").contains("not sN or f32@EXPR"));
        assert!(parse("hud-at 7 9\nhud-speed 71001000 s=u32@x0 m=s4 b=s5").contains("not sN or f32@EXPR"));
        assert!(parse("hud-at 7 9\nhud-speed 71001000 s=s3 m=s4 b=s5 v=s6").contains("takes s, m and b"));
        assert!(parse("hud-speed 71001000 s=s3 m=s4 b=s5").contains("need a hud-at"));
        assert!(parse("hud-at 7").contains("takes X and Y"));
        assert!(parse("hud-at 7 9\nhud-at 1 2").contains("a second hud-at"));
        assert_eq!(parse("hud-at 7 9\nfont latin 12ab\nfont ja-JP 34cd\nhud-speed 71001000 s=s3 m=s4 b=f32@[x0]+0x4"), "");
    }
}
