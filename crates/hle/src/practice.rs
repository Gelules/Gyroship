//! Practice actions (M6, D73): with `--practice FILE`, a `trace:` probe's pass may also write
//! game state, or read it for Gyroship's overlay. Lines name a probe's pc and the fields they
//! use, as items of the trace grammar, so the game's offsets stay in the data file (D21):
//!
//! ```text
//! boost <pc> <vehicle> gauge=f32@EXPR lock=u8@EXPR
//! hud-at <x> <y>
//! hud-speed <pc> s=ITEM m=ITEM b=ITEM
//! font <region tag | latin> <data.zip entry>
//! ms-keep <pc> ms=wN
//! ms <pc> out=xN
//! ```
//!
//! The level time's milliseconds (M6 steps 6 and 6b, D83), whenever the file has these lines:
//! an `ms-keep` pass keeps the ms in `wN` for its guest thread, where the game is about to
//! format a level time; the thread's next `ms` pass takes it, where the game has just built that
//! time's two-digit seconds, and appends "." and the 3 digits of its milliseconds in place: "SS"
//! becomes "SS.mmm", a point in every language (D74). Only when those digits are the kept ms's
//! seconds, and the variant at `out` is a string; else the game's text stays as it is. `out` must
//! hold a variant that `string_utils.format_int` has just returned: its text sits at the start of
//! a 0x80-byte block of its own, of which the 7 bytes written are the data file's guarantee. The
//! game then places the seconds in its own strings, "MM:SS.mmm" on the results and in the HUD
//! timer, sized by the game.
//!
//! `--boost infinite`: as each boost update starts, the gauge (a heat that boosting and braking
//! raise: 0 is a full bar) goes back to 0 and the boost-and-brake lock is lifted, so Boost held
//! never runs out, a release never locks, and a big hit's lock is gone by the next update. A
//! run with practice actions is never a reference: the journal says so at boot.
//!
//! With an overlay option (`--hud-speed`, `--hud-fps`, `--hud-markers`, or `--hud` for all
//! three): at a `hud-speed` line's pc, the vehicle's speed m·(s + b) (multiplier, base speed,
//! boost speed; each `sN` or `f32@EXPR`) is read and published with the present it was read
//! in, for the speed line drawn from `hud-at` (presented-image pixels). No guest write. `font`
//! lines are `gyroship_practice::font`'s.

use crate::trace::{self, Item, Ty};
use gyroship_gpu::datazip::DataZip;
use gyroship_gpu::surfaces::Mask;
use gyroship_practice::{font, texts};
use gyroship_runtime::Cpu;
use gyroship_runtime::mem::Memory;
use gyroship_research::boost as rule;
use std::collections::{HashMap, VecDeque};
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering::Relaxed};
use std::sync::{Arc, Mutex, OnceLock, PoisonError};

/// The run's actions: set at boot with `--practice`, never otherwise.
static ACTIONS: OnceLock<Practice> = OnceLock::new();

/// `gs.presents`, for code without the kernel lock: one store in `Kernel::present`.
pub(crate) static PRESENTS: AtomicU64 = AtomicU64::new(0);

/// What `--boost` asks.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Boost {
    Infinite,
    /// The boost ends where `gyroship_research::boost::release_now` says, boost held.
    Perfect,
}

impl Boost {
    pub fn parse(s: &str) -> Option<Boost> {
        match s {
            "infinite" => Some(Boost::Infinite),
            "perfect" => Some(Boost::Perfect),
            _ => None,
        }
    }

    pub fn name(self) -> &'static str {
        match self {
            Boost::Infinite => "infinite",
            Boost::Perfect => "perfect",
        }
    }
}

/// The fields `--boost perfect` reads on an Arwing or Blue-Marine line: the state at the
/// update's entry, then the ten settings of `rule::Linear`, by its names.
const PERFECT: [&str; 14] = ["gauge", "speed", "lock", "boosting", "warp", "accel", "decel", "max", "brake", "brake_recover", "brake_floor", "gauge_max", "fill", "drain"];

/// A boost update: its probe's pc, the vehicle, its fields by name, and the perfect release's
/// state: the releases since rest, and whether its guard has spoken.
struct Line {
    pc: u64,
    vehicle: String,
    fields: Vec<(String, Item)>,
    released: AtomicUsize,
    warned: AtomicBool,
}

impl Line {
    fn field(&self, name: &str) -> Option<&Item> {
        self.fields.iter().find(|(n, _)| n == name).map(|(_, i)| i)
    }
}

pub struct Practice {
    mem: Arc<Memory>,
    boost: Option<Boost>,
    /// By pc.
    lines: Vec<Line>,
    /// Any overlay option: the speed is read and the markers' vehicle followed.
    hud: bool,
    /// `hud-speed` lines: the pc, then s, m and b.
    speeds: Vec<(u64, [Item; 3])>,
    /// `hud-at`: the texts' origin, in presented-image pixels.
    hud_at: Option<(u32, u32)>,
    /// The last speed read: its f32 bits << 32 | the low 32 bits of its present; 0: none yet.
    speed: AtomicU64,
    /// `marker <vehicle> x0 y0 … x3 y3`: the release markers in the boost bar, a quad each.
    markers: Vec<(String, [f32; 8])>,
    /// The vehicle whose boost update passed last: its present << 8 | its line's index << 1 |
    /// whether its settings are those the markers were made for; 0: none yet.
    active: AtomicU64,
    /// `--boost perfect` switched off ([`toggle_perfect`]): the game's own, full boosts.
    perfect_off: AtomicBool,
    /// `ms` lines: the pc and the register holding the string variant's address.
    times: Vec<(u64, usize)>,
    /// `ms-keep` lines: the pc and the register (`wN`) holding the ms.
    keeps: Vec<(u64, usize)>,
    /// The ms kept by each guest thread (its TPIDRRO_EL0), until its next `ms` pass: guest
    /// threads interleave between the two passes, as fibers or in play mode.
    kept: Mutex<HashMap<u64, i32>>,
}

/// The VM's string variant: the text's address at +0, its type byte at +0x10 (a string).
const STRING_TYPE: (usize, u8) = (0x10, 0x0f);

impl Practice {
    /// The file's lines, checked against this build's probes (`probes`, as
    /// `gyroship_runtime::probe_counts` gives them): each probe line's pc is a `trace:` probe,
    /// and carries the fields its mode needs.
    pub fn parse(text: &str, boost: Option<Boost>, hud: bool, mem: Arc<Memory>, probes: &[(u64, &str, u64)]) -> Result<Practice, String> {
        let mut p = Practice {
            mem,
            boost,
            lines: Vec::new(),
            hud,
            speeds: Vec::new(),
            hud_at: None,
            speed: AtomicU64::new(0),
            markers: Vec::new(),
            active: AtomicU64::new(0),
            perfect_off: AtomicBool::new(false),
            times: Vec::new(),
            keeps: Vec::new(),
            kept: Mutex::new(HashMap::new()),
        };
        for (n, line) in text.lines().enumerate().filter(|(_, l)| !l.trim().is_empty() && !l.trim_start().starts_with('#')) {
            let at = || format!("line {}", n + 1);
            let mut words = line.split_whitespace();
            let kind = words.next().unwrap_or_default();
            match kind {
                "boost" | "hud-speed" | "ms" | "ms-keep" => {}
                "hud-at" => {
                    let xy: Option<Vec<u32>> = words.map(|w| w.parse().ok()).collect();
                    let Some(&[x, y]) = xy.as_deref() else { return Err(format!("{}: hud-at takes X and Y, in pixels", at())) };
                    if p.hud_at.replace((x, y)).is_some() {
                        return Err(format!("{}: a second hud-at", at()));
                    }
                    continue;
                }
                "font" => continue,
                "marker" => {
                    let vehicle = words.next().unwrap_or_default();
                    if !["arwing", "blue-marine", "landmaster"].contains(&vehicle) {
                        return Err(format!("{}: unknown vehicle {vehicle}", at()));
                    }
                    let xy: Option<Vec<f32>> = words.map(|w| w.parse().ok()).collect();
                    let Some(quad) = xy.and_then(|v| <[f32; 8]>::try_from(v).ok()) else { return Err(format!("{}: marker takes a vehicle and the x y of 4 corners", at())) };
                    p.markers.push((vehicle.to_string(), quad));
                    continue;
                }
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
            if kind == "ms" || kind == "ms-keep" {
                let (name, wanted) = if kind == "ms" { ("out", "out=xN") } else { ("ms", "ms=wN") };
                let item = match (words.next().and_then(|w| w.split_once('=')), words.next()) {
                    (Some((n, item)), None) if n == name => trace::item(item),
                    _ => None,
                };
                let r = match (kind, item) {
                    ("ms", Some(Item::X(r))) | ("ms-keep", Some(Item::W(r))) => r,
                    _ => return Err(format!("{}: {kind} takes {wanted} alone", at())),
                };
                if kind == "ms" { p.times.push((pc, r)) } else { p.keeps.push((pc, r)) }
                continue;
            }
            let vehicle = words.next().ok_or_else(|| format!("{}: no vehicle", at()))?.to_string();
            if !["arwing", "blue-marine", "landmaster"].contains(&vehicle.as_str()) {
                return Err(format!("{}: unknown vehicle {vehicle}", at()));
            }
            let mut fields = Vec::new();
            for w in words {
                let (name, item) = w.split_once('=').ok_or_else(|| format!("{}: {w} is not name=item", at()))?;
                // The lock and the boosting flag are bytes; the gauge, the speed and the settings f32.
                let ty = if matches!(name, "lock" | "boosting") { Ty::U8 } else { Ty::F32 };
                let item = trace::item(item).filter(|i| matches!(i, Item::Mem(t, _) if *t == ty || !PERFECT.contains(&name))).ok_or_else(|| format!("{}: {name}={item} is not {}@EXPR", at(), if ty == Ty::U8 { "u8" } else { "f32" }))?;
                fields.push((name.to_string(), item));
            }
            let need: &[&str] = match (boost, vehicle.as_str()) {
                (Some(Boost::Infinite), _) => &["gauge", "lock"],
                (Some(Boost::Perfect), "arwing" | "blue-marine") => &PERFECT,
                _ => &[],
            };
            for need in need {
                if !fields.iter().any(|(f, _)| f == need) {
                    return Err(format!("{}: {vehicle} has no {need} field", at()));
                }
            }
            p.lines.push(Line { pc, vehicle, fields, released: AtomicUsize::new(0), warned: AtomicBool::new(false) });
        }
        if hud && !p.speeds.is_empty() && p.hud_at.is_none() {
            return Err("hud-speed lines need a hud-at line".into());
        }
        p.lines.sort_by_key(|l| l.pc);
        Ok(p)
    }

    /// A pass of a `trace:` probe at `pc`: its line's actions. A field whose pointers cannot
    /// be read is left alone. Returns the perfect release's guard line, once per line.
    pub(crate) fn act(&self, pc: u64, cpu: &Cpu) -> Option<String> {
        let boost = self.boost?;
        let line = &self.lines[self.lines.binary_search_by_key(&pc, |l| l.pc).ok()?];
        if boost == Boost::Perfect {
            return self.perfect(line, cpu);
        }
        for (name, item) in &line.fields {
            if matches!(name.as_str(), "gauge" | "lock")
                && let Some((addr, size)) = trace::place(item, cpu, &self.mem)
            {
                let _ = self.mem.write(addr, &[0u8; 8][..size]);
            }
        }
        None
    }

    /// `--boost perfect` at a boost update's entry: the lock set when the rule ends the boost
    /// under way, and no other byte written. The Landmaster holds boost to the end: nothing to
    /// do. The guard's line, the first time a line's settings are not those the rule was made
    /// for; no release while they differ.
    fn perfect(&self, line: &Line, cpu: &Cpu) -> Option<String> {
        if line.vehicle == "landmaster" {
            return None;
        }
        let f = |n: &str| trace::f32_of(line.field(n)?, cpu, &self.mem);
        let p = rule::Linear { warp: f("warp")?, accel: f("accel")?, decel: f("decel")?, max: f("max")?, brake: f("brake")?, brake_recover: f("brake_recover")?, brake_floor: f("brake_floor")?, gauge_max: f("gauge_max")?, fill: f("fill")?, drain: f("drain")? };
        let byte = |n: &str| {
            let (at, _) = trace::place(line.field(n)?, cpu, &self.mem)?;
            let mut b = [0u8];
            self.mem.read(at, &mut b).ok()?;
            Some((at, b[0] != 0))
        };
        let ((lock_at, locked), (_, boosting)) = (byte("lock")?, byte("boosting")?);
        let s = rule::State { gauge: f("gauge")?, speed: f("speed")?, locked, boosting, ..rule::State::default() };
        let mut released = line.released.load(Relaxed);
        let release = rule::release_now(&s, &p, &mut released);
        if self.perfect_off.load(Relaxed) {
            // Switched off: the game's own boost runs to the end; only a rest still brings
            // the long start back.
            if released == 0 {
                line.released.store(0, Relaxed);
            }
            return None;
        }
        line.released.store(released, Relaxed);
        match release {
            Some(true) => {
                let _ = self.mem.write(lock_at, &[1]);
                None
            }
            Some(false) => None,
            None => (!line.warned.swap(true, Relaxed)).then(|| format!("practice: boost perfect: the {} line's settings are not those the rule was made for ({p:?}): no release while they differ", line.vehicle)),
        }
    }

    /// With an overlay option, a pass at a `hud-speed` line's pc: s, m and b read where the line says,
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

    /// `hud-at`: the texts' origin, in presented-image pixels.
    pub(crate) fn hud_at(&self) -> Option<(u32, u32)> {
        self.hud_at
    }

    /// With an overlay option, a pass at a boost line's pc: that vehicle is the one the markers are
    /// for, from this present on, and its line's settings say whether they apply (the rule's
    /// own guard: an Arwing or a Blue-Marine with other settings gets none). No guest write.
    pub(crate) fn mark_pass(&self, pc: u64, cpu: &Cpu) {
        if !self.hud || self.markers.is_empty() {
            return;
        }
        let Ok(i) = self.lines.binary_search_by_key(&pc, |l| l.pc) else { return };
        let line = &self.lines[i];
        let f = |n: &str| trace::f32_of(line.field(n)?, cpu, &self.mem);
        let settings = || Some(rule::Linear { warp: f("warp")?, accel: f("accel")?, decel: f("decel")?, max: f("max")?, brake: f("brake")?, brake_recover: f("brake_recover")?, brake_floor: f("brake_floor")?, gauge_max: f("gauge_max")?, fill: f("fill")?, drain: f("drain")? });
        let ok = match line.vehicle.as_str() {
            "arwing" => settings() == Some(rule::ARWING),
            "blue-marine" => settings() == Some(rule::BLUE_MARINE),
            _ => true,
        };
        self.active.store(word(PRESENTS.load(Relaxed), i, ok), Relaxed);
    }

    /// The markers to draw with present `shown`: those of the vehicle whose boost update
    /// passed last, while [`gate`] lets them through.
    pub(crate) fn marks_at(&self, shown: u64) -> Vec<[f32; 8]> {
        // While --boost perfect is switched off for a full boost, the markers hide.
        if self.boost == Some(Boost::Perfect) && self.perfect_off.load(Relaxed) {
            return Vec::new();
        }
        let Some(i) = gate(self.active.load(Relaxed), shown) else { return Vec::new() };
        let vehicle = &self.lines[i].vehicle;
        self.markers.iter().filter(|(v, _)| v == vehicle).map(|(_, q)| *q).collect()
    }

    /// Whether the file has `ms` lines, which write guest memory.
    pub fn writes_ms(&self) -> bool {
        !self.times.is_empty()
    }

    /// A pass at an `ms-keep` line's pc (the ms kept for this guest thread) or at an `ms` line's
    /// (the thread's kept ms taken, its milliseconds after the game's seconds; see the module's
    /// doc). Returns the new text when it wrote one.
    pub(crate) fn time_ms(&self, pc: u64, cpu: &Cpu) -> Option<String> {
        let kept = || self.kept.lock().unwrap_or_else(PoisonError::into_inner);
        if let Some(&(_, w)) = self.keeps.iter().find(|k| k.0 == pc) {
            kept().insert(cpu.tpidrro, cpu.x[w] as u32 as i32);
            return None;
        }
        let &(_, out) = self.times.iter().find(|t| t.0 == pc)?;
        let ms = kept().remove(&cpu.tpidrro)?;
        let mut v = [0u8; 0x11];
        self.mem.read(cpu.x[out], &mut v).ok()?;
        let at = u64::from_le_bytes(v[..8].try_into().unwrap());
        // The text: two digits, then their NUL.
        let mut text = [0u8; 3];
        if v[STRING_TYPE.0] != STRING_TYPE.1 || at == 0 || self.mem.read(at, &mut text).is_err() || text[2] != 0 {
            return None;
        }
        let new = with_ms(&text[..2], ms)?;
        self.mem.write(at, &[new.as_bytes(), &[0]].concat()).ok()?;
        Some(new)
    }
}

/// The game's two-digit seconds `text`, with "." and the 3 digits of `ms`'s milliseconds after
/// them, when they are `ms`'s seconds; None otherwise.
fn with_ms(text: &[u8], ms: i32) -> Option<String> {
    let ms = u32::try_from(ms).ok()?;
    let seconds = std::str::from_utf8(text).ok().filter(|s| s.len() == 2 && s.bytes().all(|b| b.is_ascii_digit()))?;
    (seconds.parse() == Ok(ms / 1000 % 60)).then(|| format!("{seconds}.{:03}", ms % 1000))
}

/// A mask made, with what it was made for: the text, the markers, the image size and the
/// output size.
type Made = (String, Vec<[f32; 8]>, (u32, u32), (u32, u32), Arc<Mask>);

/// The release markers' colour: a magenta that neither the bar's fill (white, orange) nor its
/// frame (cyan) uses.
const MARK: [f32; 4] = [1.0, 0.2, 0.8, 1.0];

/// The overlays' text: the speed line, then the frame rate, each when chosen. Their labels in
/// the run's language, the origin (presented-image pixels), the font that draws them, the
/// presents of the last second, and the last mask made.
pub struct Hud {
    label: &'static str,
    fps_label: &'static str,
    origin: (u32, u32),
    face: font::Face,
    presents: VecDeque<u64>,
    /// When the text was loaded: the wall clock's origin.
    born: std::time::Instant,
    last: Option<Made>,
    /// The lines drawn: the speed's (`--hud-speed`), the frame rate's (`--hud-fps`).
    speed: bool,
    fps: bool,
}

/// The overlays' text: the speed line when chosen (empty while no fresh speed was read, so the
/// counter below keeps its place), then the frame rate's when chosen.
fn lines(speed: Option<String>, fps: Option<String>) -> String {
    [speed, fps].into_iter().flatten().collect::<Vec<_>>().join("\n")
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
    /// The overlays' font: the `font` line of the run's region (`language`, else en-US), read
    /// from the dump's `data.zip`; every overlay option needs it. When there is text to draw,
    /// says on stderr, never in the journal, which font it uses and what it cannot draw.
    pub fn load(practice: &str, origin: Option<(u32, u32)>, zip: &std::path::Path, language: Option<&str>, overlays: crate::Overlays) -> Result<Hud, String> {
        let region = texts::region(language.unwrap_or("en-US"));
        let id = font::entry(practice, region)?;
        let mut z = DataZip::open(zip).map_err(|e| format!("{}: {e}", zip.display()))?;
        let i = z.entries.iter().position(|e| u64::from_str_radix(&e.name, 16) == Ok(id)).ok_or_else(|| format!("{}: no entry {id:x}", zip.display()))?;
        let entry = z.entries.swap_remove(i);
        let face = font::Face::load(&z.read(&entry).map_err(|e| format!("{}: entry {id:x}: {e}", zip.display()))?)?;
        if overlays.speed || overlays.fps {
            let missing: Vec<String> = face.missing(&font::needed(region)).iter().map(|c| format!("U+{:04X}", u32::from(*c))).collect();
            if !missing.is_empty() {
                eprintln!("gyroship: the --hud options: {} cannot draw {}", face.source, missing.join(" "));
            }
            eprintln!("gyroship: the --hud options: {} texts in {}", texts::tag(region), face.source);
        }
        Ok(Hud { label: texts::text(texts::Text::Speed, region), fps_label: texts::text(texts::Text::Fps, region), origin: origin.unwrap_or_default(), face, presents: VecDeque::new(), born: std::time::Instant::now(), last: None, speed: overlays.speed, fps: overlays.fps })
    }

    /// The text to draw with present `n`, handed over at `now` (virtual ticks), as [`lines`]
    /// lays it out. The frame rate: with a window, on the wall clock (the rate it shows);
    /// without, on the virtual clock (the game's own rate, the same in every run, so dumps stay
    /// reproducible).
    pub(crate) fn text(&mut self, n: u64, now: u64, windowed: bool) -> String {
        let at = if windowed { (self.born.elapsed().as_nanos() / 32) as u64 } else { now };
        let speed = self.speed.then(|| actions().and_then(|p| p.speed_at(n)).map_or(String::new(), |v| format!("{} {}", self.label, v.round() as i64)));
        let fps = self.fps.then(|| format!("{} {}", self.fps_label, rate(&mut self.presents, at)));
        lines(speed, fps)
    }

    /// The mask of `text` for an output of `size` pixels showing an image of `area`: the
    /// glyphs at the output's scale, stretched back over the image, so that each output pixel
    /// shows one mask texel. The last one again when nothing changed.
    pub(crate) fn mask(&mut self, text: &str, marks: &[[f32; 8]], area: (u32, u32), size: (u32, u32)) -> Option<Arc<Mask>> {
        if size.0 == 0 || size.1 == 0 {
            return None;
        }
        if let Some((t, k, a, s, m)) = &self.last
            && (t.as_str(), &k[..], *a, *s) == (text, marks, area, size)
        {
            return Some(m.clone());
        }
        // No text (only the markers, or a speed not read yet): one clear texel carries the markers.
        let m = if text.is_empty() { font::Mask { width: 1, height: 1, bytes: vec![0] } } else { self.face.mask(text, font::px(size.1)) };
        let rect = [self.origin.0 as f32, self.origin.1 as f32, m.width as f32 * area.0 as f32 / size.0 as f32, m.height as f32 * area.1 as f32 / size.1 as f32];
        let mask = Arc::new(Mask::new(rect, (m.width, m.height), m.bytes, [1.0; 4])?.with_marks(marks, MARK));
        self.last = Some((text.to_string(), marks.to_vec(), area, size, mask.clone()));
        Some(mask)
    }
}

/// What a boost pass tells the markers: its present << 8 | its line's index << 1 | whether
/// its settings hold.
fn word(present: u64, line: usize, ok: bool) -> u64 {
    (present & 0xff_ffff_ffff_ffff) << 8 | (line as u64 & 0x7f) << 1 | u64::from(ok)
}

/// The markers' gate at present `shown`: the line of the last boost pass, while it passed at
/// least at `shown` − 2 (a later present counts: in play mode the GPU thread shows a present
/// the guest has passed) and its settings hold.
fn gate(word: u64, shown: u64) -> Option<usize> {
    let (stamp, line, ok) = (word >> 8, (word >> 1 & 0x7f) as usize, word & 1 == 1);
    (stamp != 0 && stamp + 2 >= shown && ok).then_some(line)
}

/// `--boost perfect`'s switch, the controller's Guide button or the G key (D82): off, each
/// boost runs as the game has it, to the end of the meter, and the markers hide; on again, the
/// rule resumes. Nothing in the other modes.
pub fn toggle_perfect() {
    if let Some(p) = actions().filter(|p| p.boost == Some(Boost::Perfect)) {
        let off = !p.perfect_off.fetch_xor(true, Relaxed);
        eprintln!("gyroship: practice: boost perfect {}", if off { "off: full boosts (Guide or G: on again)" } else { "on" });
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

    const PROBES: [(u64, &str, u64); 3] = [(PC, "trace:aw-update x0", 0), (PC + 4, "nnMain", 0), (PC + 8, "trace:lm-tick x0", 0)];
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
        assert!(parse("boost 71001000 arwing gauge=x0 lock=u8@x0").contains("gauge=x0 is not f32@EXPR"));
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
        // Without an overlay option, the same file reads nothing.
        cpu.x[0] = AT;
        assert_eq!(Practice::parse(HUD, None, false, mem, &PROBES).unwrap().read_speed(PC, &cpu), None);
    }

    /// A perfect line on a synthetic layout: the state at [x0] (gauge +0x10, lock +0x14,
    /// boosting +0x16, speed +0x18), the ten settings at [[x0] + 0x8] + 8i.
    fn perfect_file(vehicle: &str, pc: u64) -> String {
        let settings: Vec<String> = PERFECT[4..].iter().enumerate().map(|(i, n)| format!("{n}=f32@[[x0]+0x8]+{:#x}", 8 * i)).collect();
        format!("boost {pc:x} {vehicle} gauge=f32@[x0]+0x10 lock=u8@[x0]+0x14 boosting=u8@[x0]+0x16 speed=f32@[x0]+0x18 {}\n", settings.join(" "))
    }

    /// A state and settings written into the synthetic vehicle, with a neighbour byte 0x5a
    /// between the lock and the boosting flag.
    fn put(mem: &Memory, s: &rule::State, p: &rule::Linear) {
        mem.write(AT + 0x108, &(AT + 0x400).to_le_bytes()).unwrap();
        for (i, x) in [p.warp, p.accel, p.decel, p.max, p.brake, p.brake_recover, p.brake_floor, p.gauge_max, p.fill, p.drain].iter().enumerate() {
            mem.write(AT + 0x400 + 8 * i as u64, &x.to_le_bytes()).unwrap();
        }
        mem.write(AT + 0x110, &s.gauge.to_le_bytes()).unwrap();
        mem.write(AT + 0x114, &[u8::from(s.locked), 0x5a, u8::from(s.boosting)]).unwrap();
        mem.write(AT + 0x118, &s.speed.to_le_bytes()).unwrap();
    }

    const TOP: rule::State = rule::State { gauge: 30.0, speed: 900.0, locked: false, boosting: true, braking: false, gauge_state: 0 };

    #[test]
    fn the_perfect_release_writes_the_lock_byte_alone() {
        let (mem, cpu) = vehicle();
        let p = Practice::parse(&perfect_file("arwing", PC), Some(Boost::Perfect), false, mem.clone(), &PROBES).unwrap();
        // A frame before the cap, a lock already set, no boost: nothing written.
        for s in [rule::State { speed: 870.00006, ..TOP }, rule::State { locked: true, ..TOP }, rule::State { boosting: false, ..TOP }] {
            put(&mem, &s, &rule::ARWING);
            let before = block(&mem);
            assert_eq!(p.act(PC, &cpu), None);
            assert_eq!(block(&mem), before, "{s:?}");
        }
        put(&mem, &TOP, &rule::ARWING);
        let before = block(&mem);
        assert_eq!(p.act(PC + 4, &cpu), None, "another pc");
        assert_eq!(block(&mem), before);
        assert_eq!(p.act(PC, &cpu), None);
        let after = block(&mem);
        assert_eq!((0..before.len()).filter(|&i| before[i] != after[i]).collect::<Vec<_>>(), [0x114]);
        assert_eq!(after[0x114], 1);
    }

    #[test]
    fn the_perfect_release_keeps_off_other_settings_and_says_so_once() {
        let (mem, cpu) = vehicle();
        let file = perfect_file("arwing", PC) + &perfect_file("landmaster", PC + 8);
        let p = Practice::parse(&file, Some(Boost::Perfect), false, mem.clone(), &PROBES).unwrap();
        put(&mem, &TOP, &rule::Linear { accel: 1801.0, ..rule::ARWING });
        let before = block(&mem);
        assert!(p.act(PC, &cpu).is_some_and(|l| l.contains("arwing")));
        assert_eq!(p.act(PC, &cpu), None, "said once");
        assert_eq!(block(&mem), before, "no write");
        // The Landmaster holds boost to the end: nothing, even at the cap.
        put(&mem, &TOP, &rule::ARWING);
        let before = block(&mem);
        assert_eq!(p.act(PC + 8, &cpu), None);
        assert_eq!(block(&mem), before);
    }

    #[test]
    fn the_perfect_blue_marine_plays_its_sequence_through_the_action() {
        let (mem, cpu) = vehicle();
        let p = Practice::parse(&perfect_file("blue-marine", PC), Some(Boost::Perfect), false, mem.clone(), &PROBES).unwrap();
        let (mut s, mut lengths, mut was) = (rule::State::default(), Vec::new(), false);
        for _ in 0..2000 {
            put(&mem, &s, &rule::BLUE_MARINE);
            p.act(PC, &cpu);
            let mut lock = [0u8];
            mem.read(AT + 0x114, &mut lock).unwrap();
            s.locked = lock[0] != 0;
            s.linear(&rule::BLUE_MARINE, rule::DT, true, false);
            match (was, s.boosting) {
                (false, true) => lengths.push(1),
                (true, true) => *lengths.last_mut().unwrap() += 1,
                _ => {}
            }
            was = s.boosting;
        }
        assert_eq!(&lengths[..6], &[68, 45, 34, 27, 27, 27]);
    }

    #[test]
    fn switched_off_the_perfect_release_lets_the_boost_run_and_hides_the_markers() {
        let (mem, cpu) = vehicle();
        let file = perfect_file("arwing", PC) + "marker arwing 1 2 3 4 5 6 7 8\n";
        let p = Practice::parse(&file, Some(Boost::Perfect), true, mem.clone(), &PROBES).unwrap();
        p.perfect_off.store(true, Relaxed);
        put(&mem, &TOP, &rule::ARWING);
        let before = block(&mem);
        assert_eq!(p.act(PC, &cpu), None);
        assert_eq!(block(&mem), before, "no release while off");
        p.active.store(word(100, 0, true), Relaxed);
        assert!(p.marks_at(100).is_empty(), "the markers hide");
        // On again: the release, and the markers.
        p.perfect_off.store(false, Relaxed);
        p.act(PC, &cpu);
        assert_eq!(block(&mem)[0x114], 1);
        assert_eq!(p.marks_at(100).len(), 1);
    }

    #[test]
    fn perfect_lines_carry_the_fields_the_rule_reads() {
        let mem = vehicle().0;
        let parse = |text: &str| Practice::parse(text, Some(Boost::Perfect), false, mem.clone(), &PROBES).err().unwrap_or_default();
        let full = perfect_file("arwing", PC);
        assert_eq!(parse(&full), "");
        assert!(parse(&full.replace(" boosting=u8@[x0]+0x16", "")).contains("no boosting field"));
        assert!(parse(&full.replace("lock=u8@", "lock=f32@")).contains("lock=f32@[x0]+0x14 is not u8@EXPR"));
        assert!(parse(&full.replace("arwing", "tank")).contains("unknown vehicle tank"));
        assert_eq!(parse("boost 71001008 landmaster gauge=f32@[x0]+0x10 lock=u8@[x0]+0x14"), "");
    }

    #[test]
    fn markers_show_with_the_vehicle_in_flight_while_its_settings_hold() {
        // The gate: fresh at n and n + 2, gone at n + 3; a later stamp counts; never without a
        // stamp or with other settings.
        let w = word(100, 0, true);
        assert_eq!([100, 102, 103, 99].map(|n| gate(w, n)), [Some(0), Some(0), None, Some(0)]);
        assert_eq!((gate(word(100, 0, false), 100), gate(word(0, 0, true), 0)), (None, None));
        // A pass with the measured settings says they hold, with others not.
        let (mem, cpu) = vehicle();
        let file = perfect_file("arwing", PC) + "marker arwing 1 2 3 4 5 6 7 8\nmarker landmaster 9 9 9 9 9 9 9 9\n";
        let p = Practice::parse(&file, None, true, mem.clone(), &PROBES).unwrap();
        put(&mem, &TOP, &rule::ARWING);
        p.mark_pass(PC, &cpu);
        assert_eq!(p.active.load(Relaxed) & 0xff, 0b1);
        put(&mem, &TOP, &rule::Linear { accel: 1801.0, ..rule::ARWING });
        p.mark_pass(PC, &cpu);
        assert_eq!(p.active.load(Relaxed) & 0xff, 0b0);
        // The vehicle's own markers, and only while the gate lets them through.
        p.active.store(word(100, 0, true), Relaxed);
        assert_eq!(p.marks_at(100), [[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]]);
        assert!(p.marks_at(103).is_empty());
        // Marker lines are checked.
        let parse = |text: &str| Practice::parse(text, None, true, mem.clone(), &PROBES).err().unwrap_or_default();
        assert!(parse("marker tank 1 2 3 4 5 6 7 8").contains("unknown vehicle tank"));
        assert!(parse("marker arwing 1 2 3").contains("the x y of 4 corners"));
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
    fn the_overlays_draw_the_lines_chosen() {
        let (speed, fps) = (Some("SPEED 1200".to_string()), Some("FPS 60".to_string()));
        assert_eq!(lines(speed.clone(), fps.clone()), "SPEED 1200\nFPS 60", "both, as --hud draws them");
        assert_eq!(lines(Some(String::new()), fps.clone()), "\nFPS 60", "no fresh speed: the counter keeps its place");
        assert_eq!(lines(None, fps), "FPS 60", "the frame rate alone, on the first line");
        assert_eq!(lines(speed, None), "SPEED 1200", "the speed alone");
        assert_eq!(lines(None, None), "", "the markers alone: no text");
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

    #[test]
    fn the_milliseconds_follow_the_games_seconds_only_when_they_are_its_time() {
        for (ms, ss, with) in [(0, "00", "00.000"), (999, "00", "00.999"), (59_999, "59", "59.999"), (60_000, "00", "00.000"), (3_599_999, "59", "59.999")] {
            assert_eq!(with_ms(ss.as_bytes(), ms).as_deref(), Some(with), "{ms}");
        }
        // Other seconds, one or three digits, not digits, a whole time, a negative time.
        for (text, ms) in [("01", 0), ("5", 5_000), ("005", 5_000), ("0a", 0), ("", 0), ("00:00", 0), ("00", -1)] {
            assert_eq!(with_ms(text.as_bytes(), ms), None, "{text} at {ms}");
        }
    }

    /// `ms-keep` at PC + 0x10 (w2), `ms` at PC + 0x14 (x20); PC + 4 is not a trace: probe.
    const MS_PROBES: [(u64, &str, u64); 3] = [(PC + 4, "nnMain", 0), (PC + 0x10, "trace:results-ms", 0), (PC + 0x14, "trace:results-ss", 0)];
    const MS_FILE: &str = "ms-keep 71001010 ms=w2\nms 71001014 out=x20\n";

    /// A string variant at AT (its text at AT + 0x200, type 0x0f at +0x10), `text` in a block
    /// of 0x80 bytes of 0xee; `ms` in w2, on guest thread 7 (its TPIDRRO_EL0).
    fn string_state(text: &[u8], ms: i32) -> (Arc<Memory>, Cpu) {
        let mem = Arc::new(Memory::new().unwrap());
        mem.map(AT, 0x2000, Block::new(state::NORMAL, perm::RW)).unwrap();
        mem.write(AT, &(AT + 0x200).to_le_bytes()).unwrap();
        mem.write(AT + 0x10, &[0x0f]).unwrap();
        mem.write(AT + 0x200, &[0xee; 0x80]).unwrap();
        mem.write(AT + 0x200, &[text, &[0]].concat()).unwrap();
        let mut cpu = Cpu::new(std::ptr::null_mut());
        (cpu.x[20], cpu.x[2], cpu.tpidrro) = (AT, u64::from(ms as u32), 7);
        (mem, cpu)
    }

    #[test]
    fn the_milliseconds_are_written_in_place_after_the_games_seconds() {
        let (mem, mut cpu) = string_state(b"07", 7_654);
        let p = Practice::parse(MS_FILE, None, false, mem.clone(), &MS_PROBES).unwrap();
        assert!(p.writes_ms());
        let before = block(&mem);
        assert_eq!(p.time_ms(PC + 0x14, &cpu), None, "nothing kept yet");
        assert_eq!(p.time_ms(PC + 4, &cpu), None, "another pc");
        assert_eq!(p.time_ms(PC + 0x10, &cpu), None);
        // Another guest thread's pass neither takes nor uses this one's ms.
        cpu.tpidrro = 8;
        assert_eq!(p.time_ms(PC + 0x14, &cpu), None);
        assert_eq!(block(&mem), before);
        cpu.tpidrro = 7;
        assert_eq!(p.time_ms(PC + 0x14, &cpu).as_deref(), Some("07.654"));
        let after = block(&mem);
        assert_eq!((0..before.len()).filter(|&i| before[i] != after[i]).collect::<Vec<_>>(), (0x202..0x207).collect::<Vec<_>>());
        assert_eq!(&after[0x200..0x208], b"07.654\0\xee");
        // Taken once.
        mem.write(AT + 0x200, b"07\0").unwrap();
        assert_eq!(p.time_ms(PC + 0x14, &cpu), None, "the kept ms was taken");
        // Without ms lines, nothing.
        let p = Practice::parse("", None, false, mem.clone(), &MS_PROBES).unwrap();
        assert!(!p.writes_ms());
        let before = block(&mem);
        assert_eq!((p.time_ms(PC + 0x10, &cpu), p.time_ms(PC + 0x14, &cpu)), (None, None));
        assert_eq!(block(&mem), before);
    }

    #[test]
    fn the_game_text_stays_as_it_is_unless_it_is_two_digits_of_the_kept_seconds() {
        let practice = |mem: &Arc<Memory>| Practice::parse(MS_FILE, None, false, mem.clone(), &MS_PROBES).unwrap();
        type Change = dyn Fn(&Memory, &mut Cpu);
        let cases: [(&str, &Change); 5] = [
            ("other seconds", &|m, _| m.write(AT + 0x200, b"08\0").unwrap()),
            ("three digits", &|m, _| m.write(AT + 0x200, b"071\0").unwrap()),
            ("a NULL text", &|m, _| m.write(AT, &0u64.to_le_bytes()).unwrap()),
            ("another type", &|m, _| m.write(AT + 0x10, &[0x0e]).unwrap()),
            ("an unreadable variant", &|_, c| c.x[20] = 0x10),
        ];
        for (why, change) in cases {
            let (mem, mut cpu) = string_state(b"07", 7_654);
            let p = practice(&mem);
            p.time_ms(PC + 0x10, &cpu);
            change(&mem, &mut cpu);
            let before = block(&mem);
            assert_eq!(p.time_ms(PC + 0x14, &cpu), None, "{why}");
            assert_eq!(block(&mem), before, "{why}");
        }
        // The lines are checked.
        let mem = string_state(b"07", 0).0;
        let parse = |text: &str| Practice::parse(text, None, false, mem.clone(), &MS_PROBES).err().unwrap_or_default();
        for (line, error) in [
            ("ms 71001014 out=w20", "ms takes out=xN alone"),
            ("ms 71001014 out=x20 ms=w2", "ms takes out=xN alone"),
            ("ms 71001014", "ms takes out=xN alone"),
            ("ms-keep 71001010 ms=x2", "ms-keep takes ms=wN alone"),
            ("ms-keep 71001010 ms=i32@x19", "ms-keep takes ms=wN alone"),
            ("ms 71001004 out=x20", "not a trace: probe"),
        ] {
            assert!(parse(line).contains(error), "{line}: {}", parse(line));
        }
        assert_eq!(parse(MS_FILE), "");
    }
}
