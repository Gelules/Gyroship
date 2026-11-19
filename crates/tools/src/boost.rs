//! `boost-check`: replays `--trace` files of the boost probes through
//! gyroship-research's reimplementation, frame by frame (with the hits the
//! shields probes saw), and reports every value that differs from the game's,
//! bit for bit.

use gyroship_research::boost::{ARWING, BLUE_MARINE, LANDMASTER, Linear, State, add_calc, release_now};
use std::collections::HashMap;
use std::path::PathBuf;

/// Differences printed per trace.
const SHOWN: usize = 12;
/// The Arwing's and the Blue-Marine's state and input, seen from their boost script.
const ST: &str = "[[[[x0]+0x18]+0x38]]";
const IN: &str = "[[[[x0]+0x18]+0x50]]";
/// The Landmaster's vehicle_state, seen from its gauge function.
const VS: &str = "[[[[x0]+0x18]+0x68]]";

struct Pass<'a> {
    present: u64,
    thread: &'a str,
    label: &'a str,
    items: HashMap<&'a str, &'a str>,
}

fn parse(line: &str) -> Option<Pass<'_>> {
    let mut w = line.split_whitespace();
    w.next()?;
    let thread = w.next()?;
    let present = w.next()?.strip_prefix('p')?.parse().ok()?;
    let label = w.next()?;
    Some(Pass { present, thread, label, items: w.filter_map(|t| t.rsplit_once('=')).collect() })
}

impl Pass<'_> {
    fn get(&self, key: &str) -> Result<&str, String> {
        self.items.get(key).copied().filter(|v| *v != "?").ok_or_else(|| format!("p{} {}: no value for {key}", self.present, self.label))
    }
    fn f32(&self, key: &str) -> Result<f32, String> {
        let v = self.get(key)?;
        v.parse().map_err(|_| format!("p{} {}: {key}={v} is not a float", self.present, self.label))
    }
    fn int(&self, key: &str) -> Result<u64, String> {
        let v = self.get(key)?;
        v.strip_prefix("0x").and_then(|h| u64::from_str_radix(h, 16).ok()).ok_or_else(|| format!("p{} {}: {key}={v} is not hex", self.present, self.label))
    }
    fn flag(&self, key: &str) -> Result<bool, String> {
        Ok(self.int(key)? != 0)
    }
}

fn show(s: &State) -> String {
    format!("g={} b={} lock={} boosting={} braking={} state={}", s.gauge, s.speed, s.locked as u8, s.boosting as u8, s.braking as u8, s.gauge_state)
}

/// A count and its noun, singular for one: "1 pause", "2 pauses".
fn count(n: usize, noun: &str) -> String {
    format!("{n} {noun}{}", if n == 1 { "" } else { "s" })
}

/// Presents between two updates that still follow each other (2 at 30 Hz);
/// a longer gap is a pause (cutscene, transition) where other scripts may
/// reset the boost, so the chain restarts after it.
const GAP: u64 = 2;

/// The Arwing's or the Blue-Marine's boost script: its settings, the
/// present, state and inputs of its last update, the hits taken since, and the tally.
struct Run {
    name: &'static str,
    params: Linear,
    last: Option<(u64, State, bool, bool, f32)>,
    hits: Vec<f32>,
    frames: usize,
    exact: usize,
    pauses: usize,
    hits_seen: usize,
    forced: usize,
    /// `--apply perfect`: the rule's count of releases since rest, and the releases made.
    released: usize,
    fired: usize,
}

/// Applies the hits taken after a frame's update; true if one forced the cooldown.
fn take_hits(e: &mut State, hits: &[f32]) -> bool {
    let was = e.locked;
    hits.iter().for_each(|&d| e.hit(d));
    e.locked && !was
}

/// One frame of the Landmaster, from its probes in that frame's order.
#[derive(Default)]
struct LmFrame {
    present: u64,
    settings: bool,
    calc_in: Option<[f32; 5]>,
    step: Option<(f32, f32)>,
    out: Option<f32>,
    gauge: Option<(u32, State)>,
    /// boost_active and brake_active as its boost script read them (trace:in-read).
    input: Option<(bool, bool)>,
    /// Hits taken after this frame's meter update.
    hits: Vec<f32>,
}

/// The Landmaster's boost and brake function (main+0x9bac300), which reads its inputs.
const LM_BOOST: std::ops::Range<u64> = 0x71_09ba_c300..0x71_09ba_dca0;

/// The `--boost` practice action a traced run had on, which the model then applies too: its
/// writes happen after the trace line, at each update's entry.
#[derive(Clone, Copy, PartialEq, clap::ValueEnum)]
pub enum Apply {
    Infinite,
    Perfect,
}

pub fn check(traces: &[PathBuf], apply: Option<Apply>) -> crate::Result<bool> {
    let mut ok = true;
    for path in traces {
        println!("{}", path.display());
        ok &= check_text(&std::fs::read_to_string(path)?, apply)?;
    }
    Ok(ok)
}

/// The probes see one vehicle, the player's, whose script process (S, at
/// [x0]+0x18) moves from frame to frame: frames chain by probe, in order.
fn check_text(text: &str, apply: Option<Apply>) -> crate::Result<bool> {
    let mut runs: Vec<(&str, Run)> = Vec::new();
    let mut open: HashMap<&str, LmFrame> = HashMap::new();
    let mut lm: Vec<LmFrame> = Vec::new();
    let mut bad = Vec::new();
    for line in text.lines() {
        let Some(p) = parse(line) else { continue };
        match p.label {
            "aw-update" | "bm-update" => {
                let s = p.label;
                let state = State {
                    gauge: p.f32(&format!("f32@{ST}+0xa0"))?,
                    speed: p.f32(&format!("f32@{ST}+0x9c"))?,
                    locked: p.flag(&format!("u8@{ST}+0xd3"))?,
                    boosting: p.flag(&format!("u8@{ST}+0xd0"))?,
                    braking: p.flag(&format!("u8@{ST}+0xd1"))?,
                    gauge_state: p.int(&format!("u32@{ST}+0x4c"))? as u32,
                };
                let input = (p.flag(&format!("u8@{IN}+0x86"))?, p.flag(&format!("u8@{IN}+0x87"))?);
                let d = |k: u32| p.f32(&format!("f32@[[[x0]+0x18]+0x30]+{k:#x}"));
                let live = Linear { warp: d(0)?, accel: d(0x18)?, decel: d(0x30)?, max: d(0x48)?, brake: d(0x60)?, brake_recover: d(0x78)?, brake_floor: d(0x90)?, gauge_max: d(0xa8)?, fill: d(0xc0)?, drain: d(0xd8)? };
                let i = match runs.iter().position(|(k, _)| *k == s) {
                    Some(i) => i,
                    None => {
                        let name = match p.label {
                            "aw-update" if live == ARWING => "arwing",
                            "aw-update" => "arwing (other settings)",
                            _ if live == BLUE_MARINE => "blue-marine",
                            _ => "blue-marine (other settings)",
                        };
                        runs.push((s, Run { name, params: live, last: None, hits: Vec::new(), frames: 0, exact: 0, pauses: 0, hits_seen: 0, forced: 0, released: 0, fired: 0 }));
                        runs.len() - 1
                    }
                };
                let r = &mut runs[i].1;
                if let Some((at, ..)) = r.last
                    && p.present > at + GAP
                {
                    r.pauses += 1;
                } else if let Some((_, mut e, boost, brake, dt)) = r.last {
                    e.linear(&r.params, dt, boost, brake);
                    r.forced += take_hits(&mut e, &r.hits) as usize;
                    r.hits_seen += r.hits.len();
                    r.frames += 1;
                    if e == state && e.gauge.to_bits() == state.gauge.to_bits() && e.speed.to_bits() == state.speed.to_bits() {
                        r.exact += 1;
                    } else {
                        bad.push(format!("p{} {}, boost={} brake={} hits {:?} dt={dt}: model {}, game {}", p.present, r.name, boost as u8, brake as u8, r.hits, show(&e), show(&state)));
                    }
                }
                // The update runs from the entry as the practice action leaves it.
                let mut entry = state;
                match apply {
                    Some(Apply::Infinite) => (entry.gauge, entry.locked) = (0.0, false),
                    Some(Apply::Perfect) if release_now(&state, &live, &mut r.released) == Some(true) => {
                        entry.locked = true;
                        r.fired += 1;
                    }
                    _ => {}
                }
                r.last = Some((p.present, entry, input.0, input.1, p.f32("s0")?));
                r.hits.clear();
            }
            // The Arwing's hits go through arwing_shields (aw-hit) or vehicle_shields (aw-damage).
            "aw-hit" | "aw-damage" | "bm-damage" => {
                let script = if p.label.starts_with("aw-") { "aw-update" } else { "bm-update" };
                if let Some((_, r)) = runs.iter_mut().find(|(k, _)| *k == script) {
                    r.hits.push(p.f32("s0")?);
                }
            }
            // After the frame's meter update (the order the Landmaster's probes show).
            "lm-damage" => {
                if let Some(f) = lm.last_mut() {
                    f.hits.push(p.f32("s0")?);
                }
            }
            "lm-tick" => {
                let d = |k: u32| p.f32(&format!("f32@[[[x0]+0x18]+0x30]+{k:#x}"));
                let settings = [d(0x60)?, d(0x78)?, d(0x90)?, d(0xa8)?] == [LANDMASTER.boost_target, LANDMASTER.brake_target, LANDMASTER.gauge_max, LANDMASTER.rate];
                open.insert(p.thread, LmFrame { present: p.present, settings, ..Default::default() });
            }
            "lm-bs-in" | "lm-dt" | "lm-bs-out" => {
                if let Some(f) = open.get_mut(p.thread) {
                    match p.label {
                        "lm-bs-in" => f.calc_in = Some([p.f32("s0")?, p.f32("s1")?, p.f32("s2")?, p.f32("s3")?, p.f32("s4")?]),
                        "lm-dt" => f.step = Some((p.f32("s0")?, p.f32("s8")?)),
                        _ => f.out = Some(p.f32("f32@x3")?),
                    }
                }
            }
            "in-read" => {
                if let Some(f) = open.get_mut(p.thread)
                    && f.input.is_none()
                    && LM_BOOST.contains(&p.int("x30")?)
                {
                    f.input = Some((p.flag("u8@[[x1]]+0x86")?, p.flag("u8@[[x1]]+0x87")?));
                }
            }
            "lm-gauge" => {
                if let Some(mut f) = open.remove(p.thread) {
                    let g = State { gauge: p.f32(&format!("f32@{VS}+0xa0"))?, locked: p.flag(&format!("u8@{VS}+0xd3"))?, gauge_state: p.int(&format!("u32@{VS}+0x4c"))? as u32, ..Default::default() };
                    f.gauge = Some((p.int("u32@[[x0]+0x18]+0x38")? as u32, g));
                    lm.push(f);
                }
            }
            _ => {}
        }
    }
    for (_, r) in &runs {
        println!("  {:12} {}, {} exact; {}, {} forced a cooldown; {}; {}", r.name, count(r.frames, "frame"), r.exact, count(r.hits_seen, "hit"), r.forced, count(r.pauses, "pause"), count(r.fired, "perfect release"));
    }
    if !lm.is_empty() {
        let (mut n, mut settings, mut calc, mut speed, mut carried, mut gauge, mut gating, mut pauses, mut inputs, mut decided, mut hits, mut forced) = (0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0);
        for w in lm.windows(2) {
            let (f, next) = (&w[0], &w[1]);
            if next.present > f.present + GAP {
                pauses += 1;
                continue;
            }
            let (Some(i), Some((dt, r)), Some(out), Some((actions, g)), Some((_, g2))) = (f.calc_in, f.step, f.out, f.gauge, next.gauge) else { continue };
            n += 1;
            settings += f.settings as usize;
            let at = format!("p{} landmaster", f.present);
            if add_calc(i[0], i[1], i[2], i[3], i[4]).to_bits() == r.to_bits() {
                calc += 1;
            } else {
                bad.push(format!("{at}: add_calc{i:?} model {}, game {r}", add_calc(i[0], i[1], i[2], i[3], i[4])));
            }
            // The speed step from the inputs the script read when they are
            // traced, else on the branch the game took; refused while locked.
            let (boost, brake) = f.input.unwrap_or((i[1] == LANDMASTER.boost_target, i[1] == LANDMASTER.brake_target));
            let mut e = State { speed: i[0], locked: g.locked, ..Default::default() };
            e.landmaster_speed(&LANDMASTER, dt, boost, brake);
            if e.speed.to_bits() == out.to_bits() {
                speed += 1;
            } else {
                bad.push(format!("{at}: speed from {} (boost={} brake={} lock={}): model {}, game {out} toward {}", i[0], boost as u8, brake as u8, g.locked as u8, e.speed, i[1]));
            }
            // Each accepted boost or brake is one action of the meter (no leans in our runs).
            if f.input.is_some() {
                inputs += 1;
                if e.boosting as u32 + e.braking as u32 == actions {
                    decided += 1;
                } else {
                    bad.push(format!("{at}: actions from boost={} brake={} lock={}: model {}, game {actions}", boost as u8, brake as u8, g.locked as u8, e.boosting as u32 + e.braking as u32));
                }
            }
            if next.calc_in.is_some_and(|i2| i2[0].to_bits() == out.to_bits()) {
                carried += 1;
            } else {
                bad.push(format!("{at}: speed written {out}, read back {:?} next frame", next.calc_in.map(|i2| i2[0])));
            }
            // The frame's hits come after its meter update and see its is_boosting and is_braking.
            let mut m = State { boosting: e.boosting, braking: e.braking, ..g };
            m.landmaster_gauge(&LANDMASTER, dt, actions);
            forced += take_hits(&mut m, &f.hits) as usize;
            hits += f.hits.len();
            // The next frame's lm-tick empties the meter before its lm-gauge reads it.
            if apply == Some(Apply::Infinite) {
                (m.gauge, m.locked) = (0.0, false);
            }
            if (m.gauge.to_bits(), m.locked, m.gauge_state) == (g2.gauge.to_bits(), g2.locked, g2.gauge_state) {
                gauge += 1;
            } else {
                bad.push(format!("{at}, actions={actions}, hits {:?}, dt={dt}: gauge from {}: model {}, game {}", f.hits, show(&g), show(&m), show(&g2)));
            }
            // Boost and brake are refused while locked, and each counts as an action.
            if if g.locked { actions == 0 && i[1] == 0.0 } else { i[1] == 0.0 || actions >= 1 } {
                gating += 1;
            } else {
                bad.push(format!("{at}: lock={} target={} actions={actions}", g.locked as u8, i[1]));
            }
        }
        println!("  landmaster   {}: settings {settings}, add_calc {calc}, speed {speed}, carried {carried}, gauge {gauge}, gating {gating} exact; inputs traced on {inputs}, {} exact; {}, {forced} forced a cooldown; {}", count(n, "frame"), count(decided, "action"), count(hits, "hit"), count(pauses, "pause"));
    }
    if bad.is_empty() {
        println!("  every frame matches");
    } else {
        println!("  {}, the first {}:", count(bad.len(), "difference"), bad.len().min(SHOWN));
        for b in bad.iter().take(SHOWN) {
            println!("    {b}");
        }
    }
    Ok(bad.is_empty())
}

#[cfg(test)]
mod tests {
    #[test]
    fn a_count_agrees_with_its_number() {
        assert_eq!(super::count(0, "pause"), "0 pauses");
        assert_eq!(super::count(1, "pause"), "1 pause");
        assert_eq!(super::count(2, "perfect release"), "2 perfect releases");
    }
}
