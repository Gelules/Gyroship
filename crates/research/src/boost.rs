//! The boost of the three player vehicles, frame by frame, in the game's own
//! 32-bit float arithmetic, and the search for the best way to use it.
//!
//! Reimplemented from the game's boost scripts (`arwing_boost`,
//! `bluemarine_boost`, `landmaster_boost`) and checked frame by frame against
//! the recompiled game with `gyroship-tools boost-check`. Rates are per second
//! and `dt` is the frame's time step. Every product and sum is rounded to f32
//! on its own, as the game's script operators do (Rust never fuses them).

/// One frame at 60 Hz, as the game stores it (0x3c888889).
pub const DT: f32 = 1.0 / 60.0;

/// Settings of the Arwing's and the Blue-Marine's boost: the ten variables of
/// their boost script, in its order.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Linear {
    /// Boost speed gained per second in a boost gate, without cap (not modeled).
    pub warp: f32,
    /// Boost speed gained per second while boosting, up to `max`.
    pub accel: f32,
    /// Boost speed lost per second while neither boosting nor braking, down to 0.
    pub decel: f32,
    /// Boost speed cap while boosting.
    pub max: f32,
    /// Boost speed lost per second while braking, down to `-brake_floor`.
    pub brake: f32,
    /// Boost speed regained per second after braking, up to 0.
    pub brake_recover: f32,
    /// Lowest boost speed while braking, as a positive number.
    pub brake_floor: f32,
    /// Meter size.
    pub gauge_max: f32,
    /// Meter filled per second while boosting or braking.
    pub fill: f32,
    /// Meter drained per second otherwise.
    pub drain: f32,
}

/// Settings of the Landmaster's boost (entries 4 to 7 of its boost script).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Landmaster {
    /// Boost speed the boost eases toward.
    pub boost_target: f32,
    /// Boost speed the brake eases toward.
    pub brake_target: f32,
    /// Meter size.
    pub gauge_max: f32,
    /// Meter filled per second and per action, and drained per second.
    pub rate: f32,
}

/// The player's Arwing (boost script defaults overridden by `player_arwing`).
pub const ARWING: Linear = Linear { warp: 270.0, accel: 1800.0, decel: 900.0, max: 900.0, brake: 900.0, brake_recover: 450.0, brake_floor: 600.0, gauge_max: 90.0, fill: 60.0, drain: 30.0 };
/// The player's Blue-Marine (defaults overridden by `player_bluemarine`).
pub const BLUE_MARINE: Linear = Linear { warp: 270.0, accel: 250.0, decel: 100.0, max: 300.0, brake: 125.0, brake_recover: 100.0, brake_floor: 300.0, gauge_max: 90.0, fill: 40.0, drain: 20.0 };
/// The Landmaster (its boost script's defaults).
pub const LANDMASTER: Landmaster = Landmaster { boost_target: 300.0, brake_target: -300.0, gauge_max: 90.0, rate: 30.0 };

/// A vehicle's boost state, as the game keeps it between frames.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct State {
    /// The meter (`boost_gauge`), a heat meter: 0 means boost is available.
    pub gauge: f32,
    /// The boost speed (`boost_speed`), added to the vehicle's base speed.
    pub speed: f32,
    /// `boost_gauge_decreasing`: boost and brake are refused until the meter is back to 0.
    pub locked: bool,
    pub boosting: bool,
    pub braking: bool,
    /// `boost_state`: 0 while filling; 1 while draining, or 2 if a button is
    /// held meanwhile (the Landmaster always writes 1).
    pub gauge_state: u32,
}

/// `eclipse.add_calc` (main+0x85a960): 30 times a step toward `target`,
/// `rate` of the gap but at most `cap` either way; a step smaller than `min`
/// becomes `+min`, even when the target is below; 0 at the target.
pub fn add_calc(value: f32, target: f32, rate: f32, cap: f32, min: f32) -> f32 {
    if value == target {
        return 0.0;
    }
    let step = rate * (target - value);
    30.0 * if step.abs() >= min { step.clamp(-cap, cap) } else { min }
}

impl State {
    /// One frame of the Arwing's or the Blue-Marine's `boost_brake_update`
    /// (main+0x8e5d8f0, main+0x94d1290): boost wins over brake; both fill the
    /// meter and are refused while it is locked; letting go drains it from
    /// the same frame. Boost gates, loops and U-turns are not modeled; hits are [`State::hit`].
    pub fn linear(&mut self, p: &Linear, dt: f32, boost: bool, brake: bool) {
        self.boosting = boost && !self.locked;
        self.braking = brake && !self.locked && !self.boosting;
        if self.boosting || self.braking {
            self.gauge_state = 0;
            self.gauge += p.fill * dt;
            if self.gauge > p.gauge_max {
                self.gauge = p.gauge_max;
                self.locked = true;
            }
        } else {
            self.gauge_state = if boost || brake { 2 } else { 1 };
            self.gauge -= p.drain * dt;
            self.locked = self.gauge > 0.0;
            if !self.locked {
                self.gauge = 0.0;
            }
        }
        self.speed = if self.boosting {
            (self.speed + p.accel * dt).min(p.max)
        } else if self.braking {
            (self.speed - p.brake * dt).max(-p.brake_floor)
        } else if self.speed > 0.0 {
            (self.speed - p.decel * dt).max(0.0)
        } else if self.speed < 0.0 {
            (self.speed + p.brake_recover * dt).min(0.0)
        } else {
            self.speed
        };
    }

    /// The Landmaster's boost speed for one frame (main+0x9bac300): it eases
    /// toward the boost target, or the brake target (the brake wins), or 0.
    pub fn landmaster_speed(&mut self, p: &Landmaster, dt: f32, boost: bool, brake: bool) {
        self.boosting = boost && !self.locked;
        self.braking = brake && !self.locked;
        let (target, rate, cap) = match (self.boosting, self.braking) {
            (_, true) => (p.brake_target, 0.2, 6.0),
            (true, false) => (p.boost_target, 0.2, 6.0),
            _ => (0.0, 0.1, 100.0),
        };
        self.speed += dt * add_calc(self.speed, target, rate, cap, 0.001);
    }

    /// The Landmaster's meter for one frame (main+0x9badca0), after its
    /// `actions` (boost, brake and leans, counted twice at most). The first
    /// frame without action locks the meter; it drains from the next one.
    pub fn landmaster_gauge(&mut self, p: &Landmaster, dt: f32, actions: u32) {
        if !self.locked {
            if actions > 0 {
                self.gauge += actions.min(2) as f32 * p.rate * dt;
                self.gauge_state = 0;
                if self.gauge > p.gauge_max {
                    self.gauge = p.gauge_max;
                    self.locked = true;
                }
            } else if self.gauge > 0.0 {
                self.locked = true;
            }
        } else if self.gauge > 0.0 {
            self.gauge -= dt * p.rate;
            self.gauge_state = 1;
            if self.gauge < 0.0 || self.gauge.abs() <= 1.1920929e-4 {
                self.gauge = 0.0;
                self.locked = false;
            }
        }
    }

    /// One frame of the Landmaster with boost and brake only (no leans).
    pub fn landmaster(&mut self, p: &Landmaster, dt: f32, boost: bool, brake: bool) {
        self.landmaster_speed(p, dt, boost, brake);
        self.landmaster_gauge(p, dt, self.boosting as u32 + self.braking as u32);
    }

    /// A hit taken after the frame's boost update, as the shields scripts
    /// handle it (`arwing_shields` main+0x9234ad0, `vehicle_shields`
    /// main+0x94668a0, `bluemarine_shields` main+0x9542d30,
    /// `landmaster_shields` main+0x9c95b90): one of more than 40, or about 40
    /// or 38 (the scripts' float equality), forces the cooldown while
    /// boosting or braking.
    pub fn hit(&mut self, damage: f32) {
        let about = |v: f32| (damage - v).abs() <= 1.1920929e-4 || (damage - v).abs() <= damage.abs() * 4.7683716e-7;
        if (self.boosting || self.braking) && (damage > 40.0 || about(40.0) || about(38.0)) {
            self.locked = true;
        }
    }
}

/// `frames` frames from a standstill: `step` advances the vehicle by one frame
/// with boost held or not, as `pattern` decides from the state so far.
pub fn simulate(frames: usize, mut step: impl FnMut(&mut State, bool), mut pattern: impl FnMut(&State) -> bool) -> Vec<State> {
    let mut s = State::default();
    (0..frames)
        .map(|_| {
            let held = pattern(&s);
            step(&mut s, held);
            s
        })
        .collect()
}

/// Boost for `n` frames, let go until boost is back, and again. Holding past
/// a full meter just boosts until the meter locks.
pub fn boost_for(n: u32) -> impl FnMut(&State) -> bool {
    let mut left = 0;
    move |s| {
        if s.locked {
            left = 0;
            return false;
        }
        if left == 0 {
            if s.gauge > 0.0 {
                return false;
            }
            left = n;
        }
        left -= 1;
        true
    }
}

/// Boost held, let go for `release` frames every `period` frames.
pub fn hold_and_release(period: u32, release: u32) -> impl FnMut(&State) -> bool {
    let mut i = 0;
    move |_| {
        i += 1;
        (i - 1) % period >= release
    }
}

/// Average boost speed over the whole cycles of the run's second half, once
/// they have settled (a cycle starts at the first frame of a boost).
pub fn average(run: &[State]) -> f64 {
    let half = run.len() / 2;
    let starts: Vec<usize> = (half.max(1)..run.len()).filter(|&i| run[i].boosting && !run[i - 1].boosting).collect();
    let (lo, hi) = match (starts.first(), starts.last()) {
        (Some(&a), Some(&b)) if a < b => (a, b),
        _ => (half, run.len()),
    };
    run[lo..hi].iter().map(|s| f64::from(s.speed)).sum::<f64>() / (hi - lo) as f64
}

/// The boost length of 1 to `max_frames` frames per cycle with the highest
/// average boost speed over a minute at 60 Hz, and that average.
pub fn best_cycle(max_frames: u32, step: impl FnMut(&mut State, bool) + Copy) -> (u32, f64) {
    (1..=max_frames).map(|n| (n, average(&simulate(3600, step, boost_for(n))))).fold((0, f64::MIN), |best, c| if c.1 > best.1 { c } else { best })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn arwing(s: &mut State, held: bool) {
        s.linear(&ARWING, DT, held, false);
    }
    fn blue_marine(s: &mut State, held: bool) {
        s.linear(&BLUE_MARINE, DT, held, false);
    }
    fn landmaster(s: &mut State, held: bool) {
        s.landmaster(&LANDMASTER, DT, held, false);
    }

    /// Frames of each run of equal `locked` values, in order.
    fn lock_runs(run: &[State]) -> Vec<(bool, usize)> {
        let mut out: Vec<(bool, usize)> = Vec::new();
        for s in run {
            match out.last_mut() {
                Some((l, n)) if *l == s.locked => *n += 1,
                _ => out.push((s.locked, 1)),
            }
        }
        out
    }

    #[test]
    fn steps_round_like_the_game() {
        assert_eq!(DT.to_bits(), 0x3c88_8889);
        // A first boost frame: the Arwing gains 1800 * dt = 30.000002.
        let run = simulate(1, arwing, |_| true);
        assert_eq!((run[0].gauge, run[0].speed), (1.0, 30.000002));
        // The Landmaster gains dt * (30 * 6) = 3.0000002, not 3.
        assert_eq!(add_calc(0.0, 300.0, 0.2, 6.0, 0.001), 180.0);
        assert_eq!(simulate(1, landmaster, |_| true)[0].speed, 3.0000002);
        // Below 0.001 the step turns upward; at the target it is 0.
        assert_eq!(add_calc(0.005, 0.0, 0.1, 100.0, 0.001), 30.0 * 0.001);
        assert_eq!(add_calc(0.0, 0.0, 0.1, 100.0, 0.001), 0.0);
    }

    #[test]
    fn cooldowns_follow_the_boost_length() {
        // Arwing: n frames of boost lock for 2n frames, from the next frame.
        let runs = lock_runs(&simulate(400, arwing, boost_for(61)));
        assert_eq!(&runs[..3], &[(false, 61), (true, 121), (false, 62)]);
        // (121 locked frames, then the frame that empties the meter unlocks it: 122 without boost.)
        // Blue-Marine: rounding makes 21 frames of boost (meter 14.000002) cost 43 frames.
        let run = simulate(100, blue_marine, boost_for(21));
        assert_eq!(run[20].gauge, 14.000002);
        assert_eq!(lock_runs(&run)[1], (true, 42));
        // Landmaster: a partial boost of n frames locks for n + 1 frames
        // (the release frame, then n frames of drain); a full one for 180.
        let runs = lock_runs(&simulate(1000, landmaster, boost_for(60)));
        assert_eq!(&runs[..2], &[(false, 60), (true, 60)]);
        let runs = lock_runs(&simulate(1000, landmaster, |_| true));
        assert_eq!(&runs[..2], &[(false, 180), (true, 180)]);
    }

    #[test]
    fn big_hits_force_the_cooldown() {
        let boosting = State { gauge: 6.0, boosting: true, ..Default::default() };
        let locks = |damage: f32, mut s: State| {
            s.hit(damage);
            s.locked
        };
        assert!(locks(40.0, boosting) && locks(41.0, boosting) && locks(38.00001, boosting));
        assert!(!locks(20.0, boosting) && !locks(39.0, boosting) && !locks(37.0, boosting));
        assert!(locks(40.0, State { braking: true, ..Default::default() }));
        // Idle, or already in the cooldown: nothing changes.
        assert!(!locks(40.0, State::default()));
        // The Landmaster then drains from the next frame, like after a release.
        let mut s = boosting;
        s.hit(40.0);
        s.landmaster(&LANDMASTER, DT, true, false);
        assert!(!s.boosting && s.locked && s.gauge == 5.5);
    }

    /// The averages also match, to the last bit, those of the write-up's
    /// script (research/boost/boost_math.py), which follows the same rules.
    #[test]
    fn best_cycles() {
        let near = |a: f64, b: f64| (a - b).abs() < 1e-9;
        // Arwing: let go when the boost speed peaks, after 30 frames.
        let (n, avg) = best_cycle(91, arwing);
        assert!(n == 30 && near(avg, 450.00001209047105), "{n} {avg}");
        // Held until it locks: 91 frames of boost (the frame that fills the
        // meter still boosts), 180 of cooldown: 95400 / 271.
        assert!(near(average(&simulate(3600, arwing, |_| true)), 352.02952431048857));
        // Landmaster: hold until the meter locks (181 frames).
        let (n, avg) = best_cycle(181, landmaster);
        assert!(n == 181 && near(avg, 124.70180623397913), "{n} {avg}");
        assert!(near(average(&simulate(3600, landmaster, boost_for(60))), 77.72157222968488));
        // Blue-Marine: shorter is better; held with quick releases is near the cap.
        let quick = average(&simulate(3600, blue_marine, hold_and_release(9, 2)));
        assert!(near(quick, 295.27781168619794) && average(&simulate(3600, blue_marine, boost_for(45))) < quick, "{quick}");
    }
}
