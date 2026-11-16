//! `trace:` probes, for the research on the game's mechanics (D59): every pass
//! of such a probe writes one line to the `--trace` file, stamped like the
//! journal (virtual time, guest thread) and with the present, then the probe's
//! label and the values its name asks for. The deterministic journal never
//! sees them, so a run with trace probes still compares with a reference.
//!
//! A probe's name is `trace:<label> <item>... [if <reg>[&<mask>]=<value>,...]`.
//! An item is `xN` (a 64-bit register, x31 being SP), `wN` (its low 32 bits),
//! `sN` or `dN` (the low float or double of vN), or `TYPE@EXPR`: TYPE is u8,
//! u16, u32, u64, i32, f32 or f64, and EXPR a register `xN`, a hex number or
//! `[EXPR]` (the u64 at that guest address), followed by any `+0xHEX` or
//! `-0xHEX` offsets. A value that cannot be read is `?`. The condition keeps
//! the passes whose register (masked) is one of the values; numbers are hex.

use crate::journal::Journal;
use gyroship_runtime::Cpu;
use gyroship_runtime::mem::Memory;

#[derive(Debug, Clone, PartialEq)]
pub(crate) enum Expr {
    Reg(usize),
    Const(u64),
    Deref(Box<Expr>),
    Offset(Box<Expr>, i64),
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) enum Ty {
    U8,
    U16,
    U32,
    U64,
    I32,
    F32,
    F64,
}

#[derive(Debug, Clone, PartialEq)]
pub(crate) enum Item {
    X(usize),
    W(usize),
    S(usize),
    D(usize),
    Mem(Ty, Expr),
}

#[derive(Debug, Clone, PartialEq)]
struct Filter {
    reg: usize,
    mask: u64,
    values: Vec<u64>,
}

/// A `trace:` probe's name, read.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct Spec {
    label: String,
    items: Vec<(String, Item)>,
    filter: Option<Filter>,
}

fn reg(s: &str, prefix: char) -> Option<usize> {
    let n: usize = s.strip_prefix(prefix)?.parse().ok()?;
    (n < 32).then_some(n)
}

fn hex(s: &str) -> Option<u64> {
    u64::from_str_radix(s.strip_prefix("0x")?, 16).ok()
}

/// An expression at the start of `s`, and the text after it.
fn expr(s: &str) -> Option<(Expr, &str)> {
    let (mut e, mut rest) = match s.strip_prefix('[') {
        Some(inner) => {
            let (e, rest) = expr(inner)?;
            (Expr::Deref(Box::new(e)), rest.strip_prefix(']')?)
        }
        None => {
            let (token, rest) = s.split_at(s.find(['+', '-', ']']).unwrap_or(s.len()));
            (reg(token, 'x').map_or_else(|| hex(token).map(Expr::Const), |n| Some(Expr::Reg(n)))?, rest)
        }
    };
    loop {
        let sign = match rest.as_bytes().first() {
            Some(b'+') => 1,
            Some(b'-') => -1,
            _ => return Some((e, rest)),
        };
        let body = &rest[1..];
        let end = body.find(['+', '-', ']']).unwrap_or(body.len());
        e = Expr::Offset(Box::new(e), sign * hex(&body[..end])? as i64);
        rest = &body[end..];
    }
}

pub(crate) fn item(token: &str) -> Option<Item> {
    if let Some((ty, e)) = token.split_once('@') {
        let ty = match ty {
            "u8" => Ty::U8,
            "u16" => Ty::U16,
            "u32" => Ty::U32,
            "u64" => Ty::U64,
            "i32" => Ty::I32,
            "f32" => Ty::F32,
            "f64" => Ty::F64,
            _ => return None,
        };
        let (e, rest) = expr(e)?;
        return rest.is_empty().then_some(Item::Mem(ty, e));
    }
    reg(token, 'x').map(Item::X).or_else(|| reg(token, 'w').map(Item::W)).or_else(|| reg(token, 's').map(Item::S)).or_else(|| reg(token, 'd').map(Item::D))
}

fn filter(token: &str) -> Option<Filter> {
    let (lhs, values) = token.split_once('=')?;
    let (r, mask) = match lhs.split_once('&') {
        Some((r, m)) => (r, hex(m)?),
        None => (lhs, u64::MAX),
    };
    let (reg, mask) = match reg(r, 'w') {
        Some(n) => (n, mask & 0xffff_ffff),
        None => (reg(r, 'x')?, mask),
    };
    Some(Filter { reg, mask, values: values.split(',').map(hex).collect::<Option<_>>()? })
}

impl Ty {
    fn size(self) -> usize {
        match self {
            Ty::U8 => 1,
            Ty::U16 => 2,
            Ty::U32 | Ty::I32 | Ty::F32 => 4,
            Ty::U64 | Ty::F64 => 8,
        }
    }
}

/// Where a memory item lies and its size; None for a register, or a pointer that cannot be
/// read.
pub(crate) fn place(item: &Item, cpu: &Cpu, mem: &Memory) -> Option<(u64, usize)> {
    let Item::Mem(ty, e) = item else { return None };
    Some((eval(e, cpu, mem)?, ty.size()))
}

/// An `sN` or `f32@EXPR` item's value at this pass; None for another item, or for an address
/// that cannot be read.
pub(crate) fn f32_of(item: &Item, cpu: &Cpu, mem: &Memory) -> Option<f32> {
    match item {
        Item::S(n) => Some(f32::from_bits(cpu.vr[*n] as u32)),
        Item::Mem(Ty::F32, e) => {
            let mut b = [0u8; 4];
            mem.read(eval(e, cpu, mem)?, &mut b).ok()?;
            Some(f32::from_le_bytes(b))
        }
        _ => None,
    }
}

fn eval(e: &Expr, cpu: &Cpu, mem: &Memory) -> Option<u64> {
    Some(match e {
        Expr::Reg(n) => cpu.x[*n],
        Expr::Const(v) => *v,
        Expr::Deref(inner) => {
            let mut b = [0u8; 8];
            mem.read(eval(inner, cpu, mem)?, &mut b).ok()?;
            u64::from_le_bytes(b)
        }
        Expr::Offset(inner, d) => eval(inner, cpu, mem)?.wrapping_add_signed(*d),
    })
}

fn value(item: &Item, cpu: &Cpu, mem: &Memory) -> String {
    match item {
        Item::X(n) => format!("{:#x}", cpu.x[*n]),
        Item::W(n) => format!("{:#x}", cpu.x[*n] as u32),
        Item::S(n) => f32::from_bits(cpu.vr[*n] as u32).to_string(),
        Item::D(n) => f64::from_bits(cpu.vr[*n] as u64).to_string(),
        Item::Mem(ty, e) => {
            let size = ty.size();
            let mut b = [0u8; 8];
            if eval(e, cpu, mem).is_none_or(|at| mem.read(at, &mut b[..size]).is_err()) {
                return "?".into();
            }
            let v = u64::from_le_bytes(b);
            match ty {
                Ty::I32 => (v as u32 as i32).to_string(),
                Ty::F32 => f32::from_bits(v as u32).to_string(),
                Ty::F64 => f64::from_bits(v).to_string(),
                _ => format!("{v:#x}"),
            }
        }
    }
}

impl Spec {
    /// The probe named `name` (`trace:...`), or why it cannot be read.
    pub(crate) fn parse(name: &str) -> Result<Spec, String> {
        let mut words = name.strip_prefix("trace:").ok_or("not a trace: probe")?.split_whitespace();
        let label = words.next().ok_or("no label")?.to_string();
        let (mut items, mut condition) = (Vec::new(), None);
        while let Some(w) = words.next() {
            if w == "if" {
                let c = words.next().ok_or("if without a condition")?;
                condition = Some(filter(c).ok_or_else(|| format!("bad condition {c}"))?);
            } else {
                items.push((w.to_string(), item(w).ok_or_else(|| format!("bad item {w}"))?));
            }
        }
        Ok(Spec { label, items, filter: condition })
    }

    /// The line of a pass (its label and `item=value` words), unless the
    /// condition leaves it out.
    pub(crate) fn line(&self, cpu: &Cpu, mem: &Memory) -> Option<String> {
        if let Some(f) = &self.filter
            && !f.values.contains(&(cpu.x[f.reg] & f.mask))
        {
            return None;
        }
        let mut out = self.label.clone();
        for (name, item) in &self.items {
            out += &format!(" {name}={}", value(item, cpu, mem));
        }
        Some(out)
    }
}

/// The `--trace` file and the probes' names, read at their first pass.
pub struct Trace {
    journal: Journal,
    specs: std::collections::HashMap<u64, Option<Spec>>,
}

impl Trace {
    pub fn open(path: &std::path::Path) -> std::io::Result<Trace> {
        Ok(Trace { journal: Journal::file(path)?, specs: Default::default() })
    }

    /// A pass of the `trace:` probe `name` at `pc`, at virtual time `time` on
    /// guest thread `thread`, during present `present`.
    pub(crate) fn pass(&mut self, pc: u64, name: &str, cpu: &Cpu, mem: &Memory, (time, thread, present): (u64, u64, u64)) {
        let spec = self.specs.entry(pc).or_insert_with(|| Spec::parse(name).map_err(|e| eprintln!("gyroship: probe {name}: {e}; it traces nothing")).ok());
        if let Some(line) = spec.as_ref().and_then(|s| s.line(cpu, mem)) {
            self.journal.line(time, thread, &format!("p{present} {line}"));
        }
    }

    /// A line of the HLE's own, stamped as a pass.
    pub(crate) fn note(&mut self, (time, thread, present): (u64, u64, u64), text: &str) {
        self.journal.line(time, thread, &format!("p{present} {text}"));
    }

    pub fn flush(&self) {
        self.journal.flush();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use gyroship_runtime::mem::{Block, perm, state};

    #[test]
    fn a_trace_probe_reads_registers_and_memory_through_pointers() {
        const AT: u64 = 0x8000_0000;
        let mem = Memory::new().unwrap();
        mem.map(AT, 0x2000, Block::new(state::NORMAL, perm::RW)).unwrap();
        // [AT] points to AT + 0x100, where a float 1.5 sits at +0x18 and a byte 7 at +0x20.
        mem.write(AT, &(AT + 0x100).to_le_bytes()).unwrap();
        mem.write(AT + 0x118, &1.5f32.to_le_bytes()).unwrap();
        mem.write(AT + 0x120, &[7]).unwrap();
        let mut cpu = Cpu::new(std::ptr::null_mut());
        cpu.x[0] = AT;
        cpu.x[2] = 0x1_0080_681a;
        cpu.vr[0] = u128::from(0.25f32.to_bits());
        let spec = Spec::parse("trace:aw x0 w2 s0 f32@[x0]+0x18 u8@[x0+0x0]+0x20 u64@x0-0x10 if w2&0x3ffffff=0x80681a,0x806018").unwrap();
        assert_eq!(spec.line(&cpu, &mem).unwrap(), "aw x0=0x80000000 w2=0x80681a s0=0.25 f32@[x0]+0x18=1.5 u8@[x0+0x0]+0x20=0x7 u64@x0-0x10=?");
        // The condition leaves other passes out.
        cpu.x[2] = 0x806419;
        assert_eq!(spec.line(&cpu, &mem), None);
        assert!(Spec::parse("trace:bad f32@[x0").is_err());
        assert!(Spec::parse("trace:bad q7").is_err());
    }
}
