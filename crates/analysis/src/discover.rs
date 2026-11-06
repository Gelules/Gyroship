//! Function discovery over a linked program.
//!
//! Every FDE is a function. The words between FDEs are padding, PLT stubs or
//! hand-written assembly; assembly is found only by following references into
//! those gaps (symbols, relocated pointers, direct branches), never by
//! decoding blindly, since almost any word decodes.

use crate::decode::decode;
use crate::eh_frame::{self, Range};
use crate::insn::{self, Flow};
use gyroship_loader::link::{Placed, Program};
use std::collections::{BTreeMap, BTreeSet};

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Function {
    /// Guest addresses [start, start + len).
    pub start: u64,
    pub len: u64,
    /// Described by an FDE. Assembly functions have no unwind info, and only
    /// their reached instructions are code.
    pub unwind: bool,
    /// Where execution can enter, sorted: the start of an FDE function, its
    /// landing pads and code pointers into its body; the referenced entries
    /// of an assembly function.
    pub entries: Vec<u64>,
}

impl Function {
    pub fn end(&self) -> u64 {
        self.start + self.len
    }
}

/// A PLT stub: `adrp x16; ldr x17, [x16, #slot]; add x16, x16, #slot; br x17`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Stub {
    pub addr: u64,
    /// The GOT slot it jumps through.
    pub slot: u64,
    /// The slot's value after linking: the import, or 0 if unresolved.
    pub target: u64,
}

#[derive(Debug, Clone, Default)]
pub struct Counts {
    pub landing_pads: usize,
    /// Relocated pointers into the middle of an FDE function.
    pub inner_pointers: usize,
    /// References into gaps that land on padding or data: not code.
    pub dropped: usize,
}

#[derive(Debug, Clone, Default)]
pub struct Discovery {
    /// Sorted by start, disjoint.
    pub functions: Vec<Function>,
    /// Sorted by address.
    pub stubs: Vec<Stub>,
    pub counts: Counts,
}

impl Discovery {
    /// The function whose range contains `addr`.
    pub fn function_at(&self, addr: u64) -> Option<&Function> {
        let i = self
            .functions
            .partition_point(|f| f.start <= addr)
            .checked_sub(1)?;
        Some(&self.functions[i]).filter(|f| addr < f.end())
    }
}

#[derive(Debug)]
pub struct Error(pub String);

impl std::fmt::Display for Error {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}

impl std::error::Error for Error {}

impl From<eh_frame::Error> for Error {
    fn from(e: eh_frame::Error) -> Self {
        Error(e.0)
    }
}

/// One module's view while discovering: its text and what covers it.
struct Text<'a> {
    p: &'a Placed,
    start: u64,
    end: u64,
    /// FDE ranges, sorted; `start` is a guest address, `lsda` stays module-relative.
    fdes: Vec<Range>,
    /// Addresses of the module's PLT stubs.
    stubs: BTreeSet<u64>,
}

impl Text<'_> {
    fn word(&self, addr: u64) -> Option<u32> {
        let at = usize::try_from(addr.checked_sub(self.p.base)?).ok()?;
        Some(u32::from_le_bytes(
            self.p.module.image.get(at..at + 4)?.try_into().ok()?,
        ))
    }
    fn fde_at(&self, addr: u64) -> Option<usize> {
        let i = self
            .fdes
            .partition_point(|r| r.start <= addr)
            .checked_sub(1)?;
        (addr < self.fdes[i].start + self.fdes[i].len).then_some(i)
    }
    /// In text, outside every FDE and stub.
    fn in_gap(&self, addr: u64) -> bool {
        (self.start..self.end).contains(&addr) && self.fde_at(addr).is_none() && !self.in_stub(addr)
    }
    fn in_stub(&self, addr: u64) -> bool {
        self.stubs
            .range(..=addr)
            .next_back()
            .is_some_and(|&s| addr < s + 16)
    }
}

pub fn discover(prog: &Program) -> Result<Discovery, Error> {
    let mut out = Discovery::default();
    let mut texts = Vec::new();
    for p in &prog.modules {
        let m = &p.module;
        let (start, end) = (p.base + m.text.addr, p.base + m.text.end());
        let mut fdes = eh_frame::function_ranges(&m.image, m.mod0.eh_frame_hdr_start)?;
        for r in &mut fdes {
            r.start += p.base;
        }
        for pair in fdes.windows(2) {
            if pair[0].start + pair[0].len > pair[1].start {
                return Err(Error(format!(
                    "FDEs at {:#x} and {:#x} overlap",
                    pair[0].start, pair[1].start
                )));
            }
        }
        if let Some(r) = fdes
            .iter()
            .find(|r| r.start < start || r.start + r.len > end)
        {
            return Err(Error(format!(
                "FDE at {:#x} lies outside the text of {}",
                r.start, p.name
            )));
        }
        let mut t = Text {
            p,
            start,
            end,
            fdes,
            stubs: BTreeSet::new(),
        };
        // PLT stubs live in gaps; look at every gap word.
        let mut a = start;
        while a + 16 <= end {
            if let Some(i) = t.fde_at(a) {
                a = t.fdes[i].start + t.fdes[i].len;
                continue;
            }
            // The slot must be relocated: PLT0 has the same shape but jumps through the lazy resolver.
            if let Some(slot) = stub_slot(&t, a).filter(|s| prog.slots.binary_search(s).is_ok()) {
                let target = prog
                    .read(slot, 8)
                    .map_or(0, |b| u64::from_le_bytes(b.try_into().expect("8 bytes")));
                out.stubs.push(Stub {
                    addr: a,
                    slot,
                    target,
                });
                t.stubs.insert(a);
                a += 16;
            } else {
                a += 4;
            }
        }
        texts.push(t);
    }

    // FDE functions and their extra entries.
    let mut entries: Vec<Vec<BTreeSet<u64>>> = texts
        .iter()
        .map(|t| vec![BTreeSet::new(); t.fdes.len()])
        .collect();
    for (ti, t) in texts.iter().enumerate() {
        for (i, r) in t.fdes.iter().enumerate() {
            entries[ti][i].insert(r.start);
            if let Some(lsda) = r.lsda {
                let rel = r.start - t.p.base;
                for pad in eh_frame::landing_pads(&t.p.module.image, rel, lsda)? {
                    let pad = pad + t.p.base;
                    if !(r.start..r.start + r.len).contains(&pad) {
                        return Err(Error(format!(
                            "landing pad {pad:#x} lies outside its function {:#x}",
                            r.start
                        )));
                    }
                    entries[ti][i].insert(pad);
                    out.counts.landing_pads += 1;
                }
            }
        }
    }

    // References into the gaps, per module: symbols, relocated pointers, and
    // direct branches out of FDE code.
    let mut seeds: Vec<BTreeSet<u64>> = vec![BTreeSet::new(); texts.len()];
    let text_of = |addr: u64| texts.iter().position(|t| (t.start..t.end).contains(&addr));
    for (ti, t) in texts.iter().enumerate() {
        for s in &t.p.module.symbols {
            let a = t.p.base + s.value;
            if s.is_defined() && t.in_gap(a) {
                seeds[ti].insert(a);
            }
        }
    }
    for &slot in &prog.slots {
        let Some(v) = prog
            .read(slot, 8)
            .map(|b| u64::from_le_bytes(b.try_into().expect("8 bytes")))
        else {
            continue;
        };
        let Some(ti) = text_of(v) else { continue };
        let t = &texts[ti];
        match t.fde_at(v) {
            Some(i) if t.fdes[i].start != v => {
                if entries[ti][i].insert(v) {
                    out.counts.inner_pointers += 1;
                }
            }
            Some(_) => {}
            None if t.in_gap(v) => {
                seeds[ti].insert(v);
            }
            None => {}
        }
    }
    for (ti, t) in texts.iter().enumerate() {
        for r in &t.fdes {
            for a in (r.start..r.start + r.len).step_by(4) {
                let Some(w) = t.word(a) else { continue };
                if let Flow::Jump(x) | Flow::Branch(x) | Flow::Call(x) = insn::flow(w, a)
                    && t.in_gap(x)
                {
                    seeds[ti].insert(x);
                }
            }
        }
    }

    // Assembly: walk from each seed; walks that meet become one function.
    let mut asm: Vec<Vec<Function>> = vec![Vec::new(); texts.len()];
    for (ti, t) in texts.iter().enumerate() {
        let mut owner: BTreeMap<u64, usize> = BTreeMap::new();
        let mut parent: Vec<usize> = Vec::new();
        let mut group_entries: Vec<u64> = Vec::new();
        let mut queue: Vec<u64> = seeds[ti].iter().copied().collect();
        let mut done = BTreeSet::new();
        while let Some(seed) = queue.pop() {
            if !done.insert(seed) {
                continue;
            }
            if owner.contains_key(&seed) {
                group_entries.push(seed); // inside code another walk reached
                continue;
            }
            if !is_code(t, seed) {
                out.counts.dropped += 1;
                continue;
            }
            let g = parent.len();
            parent.push(g);
            group_entries.push(seed);
            let mut work = vec![seed];
            while let Some(a) = work.pop() {
                if !t.in_gap(a) {
                    continue; // into an FDE or a stub: a call or tail call
                }
                if let Some(&o) = owner.get(&a) {
                    let (ro, rg) = (find(&mut parent, o), find(&mut parent, g));
                    parent[ro] = rg;
                    continue;
                }
                let Some(w) = t
                    .word(a)
                    .filter(|&w| decode(w).is_some() || insn::flow(w, a) == Flow::Trap)
                else {
                    return Err(Error(format!(
                        "assembly reached from {seed:#x} runs into undecodable word at {a:#x}"
                    )));
                };
                owner.insert(a, g);
                let f = insn::flow(w, a);
                match f {
                    Flow::Jump(x) | Flow::Branch(x) => work.push(x),
                    Flow::Call(x) if t.in_gap(x) => queue.push(x),
                    _ => {}
                }
                if let Some((_, x)) = insn::adr(w, a).filter(|&(_, x)| t.in_gap(x)) {
                    queue.push(x);
                }
                if !f.ends() {
                    work.push(a + 4);
                }
            }
        }
        let mut groups: BTreeMap<usize, (u64, u64, Vec<u64>)> = BTreeMap::new();
        for (&a, &g) in &owner {
            let root = find(&mut parent, g);
            let e = groups.entry(root).or_insert((a, a, Vec::new()));
            e.0 = e.0.min(a);
            e.1 = e.1.max(a);
        }
        for e in group_entries {
            let root = find(&mut parent, owner[&e]);
            groups.get_mut(&root).expect("entry is owned").2.push(e);
        }
        for (_, (lo, hi, mut es)) in groups {
            es.sort_unstable();
            es.dedup();
            if let Some(a) = (lo..hi + 4).step_by(4).find(|&a| !t.in_gap(a)) {
                return Err(Error(format!(
                    "assembly at {lo:#x}..{:#x} spans covered code at {a:#x}",
                    hi + 4
                )));
            }
            asm[ti].push(Function {
                start: lo,
                len: hi + 4 - lo,
                unwind: false,
                entries: es,
            });
        }
    }

    for (ti, t) in texts.iter().enumerate() {
        for (i, r) in t.fdes.iter().enumerate() {
            let es = std::mem::take(&mut entries[ti][i]).into_iter().collect();
            out.functions.push(Function {
                start: r.start,
                len: r.len,
                unwind: true,
                entries: es,
            });
        }
        out.functions.append(&mut asm[ti]);
    }
    out.functions.sort_by_key(|f| f.start);
    if let Some(pair) = out.functions.windows(2).find(|p| p[0].end() > p[1].start) {
        return Err(Error(format!(
            "functions at {:#x} and {:#x} overlap",
            pair[0].start, pair[1].start
        )));
    }
    Ok(out)
}

fn find(parent: &mut [usize], mut g: usize) -> usize {
    while parent[g] != g {
        parent[g] = parent[parent[g]];
        g = parent[g];
    }
    g
}

/// A referenced gap address holds code, not padding or data.
fn is_code(t: &Text, a: u64) -> bool {
    a.is_multiple_of(4)
        && t.word(a)
            .is_some_and(|w| w != 0 && w != 0xD503_201F && decode(w).is_some())
}

/// The GOT slot of a PLT stub at `a`, if the four words there are one.
fn stub_slot(t: &Text, a: u64) -> Option<u64> {
    let w: Vec<u32> = (0..4).map(|i| t.word(a + 4 * i)).collect::<Option<_>>()?;
    let (_, page) = insn::adrp(w[0], a).filter(|&(r, _)| r == 16)?;
    if w[1] & 0xFFC0_03FF != 0xF940_0211 || w[2] & 0xFFC0_03FF != 0x9100_0210 || w[3] != 0xD61F_0220
    {
        return None;
    }
    let ldr = page + u64::from((w[1] >> 10) & 0xFFF) * 8;
    let add = page + u64::from((w[2] >> 10) & 0xFFF);
    (ldr == add).then_some(ldr)
}
