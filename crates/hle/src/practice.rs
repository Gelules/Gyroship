//! Practice actions (M6, D73): with `--practice FILE`, a `trace:` probe's pass may also write
//! game state. Each line of the file names a probe's pc, a vehicle and the fields the actions
//! use, as `TYPE@EXPR` items of the trace grammar, so the game's offsets stay in the data file
//! (D21):
//!
//! ```text
//! boost <pc> <vehicle> gauge=f32@EXPR lock=u8@EXPR
//! ```
//!
//! `--boost infinite`: as each boost update starts, the gauge (a heat that boosting and braking
//! raise: 0 is a full bar) goes back to 0 and the boost-and-brake lock is lifted, so Boost held
//! never runs out, a release never locks, and a big hit's lock is gone by the next update. A
//! run with practice actions is never a reference: the journal says so at boot.

use crate::trace::{self, Item};
use gyroship_runtime::Cpu;
use gyroship_runtime::mem::Memory;
use std::sync::{Arc, OnceLock};

/// The run's actions: set at boot with `--practice`, never otherwise.
static ACTIONS: OnceLock<Practice> = OnceLock::new();

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
}

impl Practice {
    /// The file's lines, checked against this build's probes (`probes`, as
    /// `gyroship_runtime::probe_counts` gives them): each line's pc is a `trace:` probe, and
    /// carries the fields its mode needs.
    pub fn parse(text: &str, boost: Option<Boost>, mem: Arc<Memory>, probes: &[(u64, &str, u64)]) -> Result<Practice, String> {
        let mut lines = Vec::new();
        for (n, line) in text.lines().enumerate().filter(|(_, l)| !l.trim().is_empty() && !l.trim_start().starts_with('#')) {
            let at = || format!("line {}", n + 1);
            let mut words = line.split_whitespace();
            match words.next() {
                Some("boost") => {}
                Some(w) => return Err(format!("{}: unknown kind {w}", at())),
                None => unreachable!(),
            }
            let pc = words.next().and_then(|w| u64::from_str_radix(w, 16).ok()).ok_or_else(|| format!("{}: no pc", at()))?;
            if !probes.iter().any(|(a, name, _)| *a == pc && name.starts_with("trace:")) {
                return Err(format!("{}: {pc:#x} is not a trace: probe of this build", at()));
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
            lines.push((pc, vehicle, fields));
        }
        lines.sort_by_key(|l| l.0);
        Ok(Practice { mem, boost, lines })
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
        let p = Practice::parse(FILE, Some(Boost::Infinite), mem.clone(), &PROBES).unwrap();
        p.act(PC + 4, &cpu);
        assert_eq!(state(&mem), (1.5, [1, 9]), "another pc");
        p.act(PC, &cpu);
        assert_eq!(state(&mem), (0.0, [0, 9]));
        // Off: the same line writes nothing.
        let (mem, cpu) = vehicle();
        Practice::parse(FILE, None, mem.clone(), &PROBES).unwrap().act(PC, &cpu);
        assert_eq!(state(&mem), (1.5, [1, 9]));
        // A pointer that cannot be read: nothing written, no fault.
        let (mem, mut cpu) = vehicle();
        cpu.x[0] = 0x10;
        Practice::parse(FILE, Some(Boost::Infinite), mem.clone(), &PROBES).unwrap().act(PC, &cpu);
        assert_eq!(state(&mem), (1.5, [1, 9]));
    }

    #[test]
    fn a_line_must_name_a_trace_probe_and_its_fields() {
        let mem = vehicle().0;
        let parse = |text: &str| Practice::parse(text, Some(Boost::Infinite), mem.clone(), &PROBES).err().unwrap_or_default();
        assert!(parse("boost 71001004 arwing gauge=f32@x0 lock=u8@x0").contains("not a trace: probe"));
        assert!(parse("boost 71002000 arwing gauge=f32@x0 lock=u8@x0").contains("not a trace: probe"));
        assert!(parse("boost 71001000 arwing gauge=f32@x0").contains("no lock field"));
        assert!(parse("boost 71001000 arwing gauge=x0 lock=u8@x0").contains("not TYPE@EXPR"));
        assert!(parse("speed 71001000 arwing").contains("unknown kind"));
        assert_eq!(parse(FILE), "");
    }
}
