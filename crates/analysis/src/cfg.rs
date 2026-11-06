//! Control-flow graph of one function: the basic blocks reached from its
//! entries, and the targets of its jump tables.
//!
//! A jump table is recognised from its value flow, not from a fixed
//! instruction sequence: `br xB` where `xB = xA + extend(xE)`, `xE` is loaded
//! from `[xT, xI]`, and `xA` and `xT` are constants (ADRP/ADD/ADR/MOV) on every
//! path. The entry count comes from the `cmp xI, #n; b.hi` guard, or from a
//! narrowing of the index. Every target must be an instruction of the function.

use crate::decode::decode;
use crate::discover::Function;
use crate::insn::{self, Extend, Flow};
use std::collections::{BTreeSet, HashMap};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Block {
    pub start: u64,
    pub len: u64,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct JumpTable {
    /// The BR that jumps through it.
    pub branch: u64,
    /// Guest address of its first entry.
    pub table: u64,
    pub entry_size: u8,
    /// Case targets in table order, repeats kept.
    pub targets: Vec<u64>,
    /// No compare bounds the index: entries were read up to the first one
    /// that is not a target, so the list may include extra valid targets.
    pub capped: bool,
}

#[derive(Debug, Clone, Default)]
pub struct Cfg {
    /// Sorted, disjoint.
    pub blocks: Vec<Block>,
    pub tables: Vec<JumpTable>,
    /// BR sites that are not jump tables (tail calls through a pointer).
    pub indirect: Vec<u64>,
    /// Reached words that do not decode.
    pub undecodable: Vec<u64>,
}

impl Cfg {
    pub fn instructions(&self) -> u64 {
        self.blocks.iter().map(|b| b.len / 4).sum()
    }
}

/// Builds the CFG of `f`; `image` is the image of its module, placed at `base`.
pub fn build(image: &[u8], base: u64, f: &Function) -> Cfg {
    let mut body = Body {
        image,
        base,
        f,
        preds: HashMap::new(),
    };
    let mut cfg = Cfg::default();

    // Jump tables first, since their targets are edges of the traversal.
    let branches: Vec<(u64, u8)> = (f.start..f.end())
        .step_by(4)
        .filter_map(|a| match insn::flow(body.word(a), a) {
            Flow::JumpIndirect(r) => Some((a, r)),
            _ => None,
        })
        .collect();
    let mut tables: HashMap<u64, JumpTable> = HashMap::new();
    if !branches.is_empty() {
        for a in (f.start..f.end()).step_by(4) {
            if let Flow::Jump(x) | Flow::Branch(x) = insn::flow(body.word(a), a)
                && body.contains(x)
            {
                body.preds.entry(x).or_default().push(a);
            }
        }
        // A table may sit in a case of another table; retry until nothing new.
        loop {
            let found: Vec<JumpTable> = branches
                .iter()
                .filter(|(a, _)| !tables.contains_key(a))
                .filter_map(|&(a, r)| body.jump_table(a, r))
                .collect();
            if found.is_empty() {
                break;
            }
            for t in found {
                for &x in &t.targets {
                    body.preds.entry(x).or_default().push(t.branch);
                }
                tables.insert(t.branch, t);
            }
        }
    }

    // Traversal from the entries.
    let n = (f.len / 4) as usize;
    let mut reached = vec![false; n];
    let mut leaders: BTreeSet<u64> = f.entries.iter().copied().collect();
    let mut work: Vec<u64> = f.entries.clone();
    let mut undecodable = BTreeSet::new();
    while let Some(a) = work.pop() {
        if !body.contains(a) || reached[body.index(a)] {
            continue;
        }
        let w = body.word(a);
        let flow = insn::flow(w, a);
        if decode(w).is_none() && flow != Flow::Trap {
            undecodable.insert(a);
            continue;
        }
        reached[body.index(a)] = true;
        let edge = |x: u64, leaders: &mut BTreeSet<u64>, work: &mut Vec<u64>| {
            if body.contains(x) {
                leaders.insert(x);
                work.push(x);
            }
        };
        match flow {
            Flow::Jump(x) => edge(x, &mut leaders, &mut work),
            Flow::Branch(x) => {
                edge(x, &mut leaders, &mut work);
                edge(a + 4, &mut leaders, &mut work);
            }
            Flow::JumpIndirect(_) => match tables.get(&a) {
                Some(t) => t
                    .targets
                    .iter()
                    .for_each(|&x| edge(x, &mut leaders, &mut work)),
                None => cfg.indirect.push(a),
            },
            Flow::Return | Flow::Trap => {}
            Flow::Next | Flow::Call(_) | Flow::CallIndirect => work.push(a + 4),
        }
    }

    let mut open: Option<Block> = None;
    for (i, &here) in reached.iter().enumerate() {
        let a = f.start + 4 * i as u64;
        if !here {
            cfg.blocks.extend(open.take());
            continue;
        }
        if leaders.contains(&a) {
            cfg.blocks.extend(open.take());
        }
        let b = open.get_or_insert(Block { start: a, len: 0 });
        b.len += 4;
        if !matches!(
            insn::flow(body.word(a), a),
            Flow::Next | Flow::Call(_) | Flow::CallIndirect
        ) {
            cfg.blocks.extend(open.take());
        }
    }
    cfg.blocks.extend(open);
    cfg.indirect.sort_unstable();
    cfg.tables = tables.into_values().collect();
    cfg.tables.sort_by_key(|t| t.branch);
    cfg.undecodable = undecodable.into_iter().collect();
    cfg
}

/// ponytail: def searches give up past this many visited instructions; raise it if a table is missed for it.
const SEARCH_LIMIT: usize = 50_000;
/// How far back the bound search follows each path.
const BOUND_PATH: usize = 96;
/// ponytail: more paths than this into one table load and the index counts as unbounded.
const MAX_PATHS: usize = 64;

struct Body<'a> {
    image: &'a [u8],
    base: u64,
    f: &'a Function,
    /// Direct-branch and jump-table predecessors inside the function.
    preds: HashMap<u64, Vec<u64>>,
}

impl Body<'_> {
    fn contains(&self, a: u64) -> bool {
        (self.f.start..self.f.end()).contains(&a) && a.is_multiple_of(4)
    }
    fn index(&self, a: u64) -> usize {
        ((a - self.f.start) / 4) as usize
    }
    /// The word at a guest address of the module; 0 (a trap) when unreadable.
    fn word(&self, a: u64) -> u32 {
        self.bytes(a, 4)
            .map_or(0, |b| u32::from_le_bytes(b.try_into().expect("4 bytes")))
    }
    fn bytes(&self, a: u64, len: usize) -> Option<&[u8]> {
        let at = usize::try_from(a.checked_sub(self.base)?).ok()?;
        self.image.get(at..at.checked_add(len)?)
    }
    fn is_entry(&self, a: u64) -> bool {
        self.f.entries.binary_search(&a).is_ok()
    }
    /// Instructions that can run just before `x`; None when `x` is an entry,
    /// so values flow in from outside.
    fn preds_of(&self, x: u64) -> Option<Vec<u64>> {
        if self.is_entry(x) {
            return None;
        }
        let mut ps = self.preds.get(&x).cloned().unwrap_or_default();
        if x > self.f.start && !insn::flow(self.word(x - 4), x - 4).ends() {
            ps.push(x - 4);
        }
        Some(ps)
    }

    /// Every definition of `reg` that reaches the point just before `at`.
    /// None when some path brings the value in from an entry, or the search
    /// is too large. Blocks without known predecessors are dead ends.
    fn reaching_defs(&self, at: u64, reg: u8) -> Option<Vec<u64>> {
        let mut defs = Vec::new();
        let mut seen = BTreeSet::new();
        let mut stack = vec![at];
        while let Some(x) = stack.pop() {
            for y in self.preds_of(x)? {
                if !seen.insert(y) {
                    continue;
                }
                if seen.len() > SEARCH_LIMIT {
                    return None;
                }
                if insn::writes(self.word(y)) & insn::bit(reg) != 0 {
                    defs.push(y);
                } else {
                    stack.push(y);
                }
            }
        }
        Some(defs)
    }

    fn only_def(&self, at: u64, reg: u8) -> Option<u64> {
        match self.reaching_defs(at, reg)?[..] {
            [d] => Some(d),
            _ => None,
        }
    }

    /// The value of `reg` just before `at` when every reaching definition
    /// computes the same constant, through 64-bit MOV and ADD-immediate
    /// chains. A copy around a loop adds nothing; an increment fails.
    fn constant(&self, at: u64, reg: u8) -> Option<u64> {
        let mut values = BTreeSet::new();
        self.constants(at, reg, 0, &mut HashMap::new(), &mut values)?;
        match values.len() {
            1 => values.pop_first(),
            _ => None,
        }
    }

    fn constants(
        &self,
        at: u64,
        reg: u8,
        offset: i64,
        seen: &mut HashMap<(u64, u8), i64>,
        out: &mut BTreeSet<u64>,
    ) -> Option<()> {
        match seen.insert((at, reg), offset) {
            Some(o) if o == offset => return Some(()),
            Some(_) => return None,
            None if seen.len() > 64 => return None,
            None => {}
        }
        for d in self.reaching_defs(at, reg)? {
            let w = self.word(d);
            if let Some((r, v)) = insn::adrp(w, d)
                .or_else(|| insn::adr(w, d))
                .or_else(|| insn::movz(w))
            {
                if r != reg {
                    return None;
                }
                out.insert(v.wrapping_add(offset as u64));
            } else if let Some((_, n, imm, true)) =
                insn::add_imm(w).filter(|&(r, n, _, _)| r == reg && n != 31)
            {
                self.constants(d, n, offset.wrapping_add(imm), seen, out)?;
            } else if let Some((_, m, true)) = insn::mov_reg(w).filter(|&(r, _, _)| r == reg) {
                self.constants(d, m, offset, seen, out)?;
            } else {
                return None;
            }
        }
        Some(())
    }

    fn jump_table(&self, branch: u64, reg: u8) -> Option<JumpTable> {
        let d = self.only_def(branch, reg)?;
        let (rd, rn, rm, ext) = insn::add_reg(self.word(d))?;
        if rd != reg {
            return None;
        }
        let mut options = vec![(rn, rm, ext)];
        if ext == Extend::Lsl(0) {
            options.push((rm, rn, ext));
        }
        for (base_reg, entry_reg, ext) in options {
            let Some(q) = self.only_def(d, entry_reg) else {
                continue;
            };
            let Some(ld) = insn::load_indexed(self.word(q)) else {
                continue;
            };
            if ld.rt != entry_reg || (!ld.scaled && ld.size > 0) {
                continue;
            }
            let (Some(table), Some(base)) = (self.constant(q, ld.rn), self.constant(d, base_reg))
            else {
                continue;
            };
            let size = 1u8 << ld.size;
            let target = |i: u64| -> Option<u64> {
                let b = self.bytes(table.checked_add(i * u64::from(size))?, size.into())?;
                let raw = b.iter().rev().fold(0u64, |v, &x| v << 8 | u64::from(x));
                let mut v = if ld.signed {
                    insn::sext(raw, 8 * u32::from(size)) as u64
                } else {
                    raw
                };
                if !ld.wide {
                    v &= 0xFFFF_FFFF;
                }
                let x = base.wrapping_add(ext.apply(v));
                (self.contains(x) && decode(self.word(x)).is_some()).then_some(x)
            };
            // ponytail: with no bound at all, read as many entries as one index byte or halfword can select.
            let (lo, hi, capped) = self.bound(q, ld.rm).unwrap_or((
                0,
                if ld.size < 2 {
                    1 << (8 << ld.size)
                } else {
                    4096
                },
                true,
            ));
            let targets: Vec<u64> = if capped {
                (lo..hi).map_while(target).collect()
            } else {
                (lo..hi).map(target).collect::<Option<_>>()?
            };
            if targets.is_empty() {
                return None;
            }
            return Some(JumpTable {
                branch,
                table,
                entry_size: size,
                targets,
                capped,
            });
        }
        None
    }

    /// Index range [lo, hi) of a table indexed by `idx` at `load`: the union
    /// over every path that leads to the load. On each path, compares and
    /// conditional branches on the index's value bound it (each branch in the
    /// direction the path takes), and so does any narrowing of the index.
    /// The flag is set when some path is bounded above only by a narrowing.
    fn bound(&self, load: u64, idx: u8) -> Option<(u64, u64, bool)> {
        let (mut lo, mut hi, mut capped) = (u64::MAX, 0, false);
        let mut paths = 0;
        // Partial paths, backwards from the load.
        let mut stack: Vec<Vec<u64>> = vec![Vec::new()];
        while let Some(back) = stack.pop() {
            let x = back.last().copied().unwrap_or(load);
            let found = insn::cmp_imm(self.word(x)).is_some()
                && !back.is_empty()
                && self.path_bound(&back, load, idx).is_some_and(|b| !b.2);
            let ps: Vec<u64> = if found || back.len() >= BOUND_PATH {
                Vec::new()
            } else {
                self.preds_of(x)
                    .unwrap_or_default()
                    .into_iter()
                    .filter(|p| !back.contains(p))
                    .collect()
            };
            if ps.is_empty() {
                paths += 1;
                if paths > MAX_PATHS {
                    return None;
                }
                let (l, h, c) = self.path_bound(&back, load, idx)?;
                (lo, hi, capped) = (lo.min(l), hi.max(h), capped || c);
                continue;
            }
            for p in ps {
                let mut next = back.clone();
                next.push(p);
                stack.push(next);
            }
        }
        Some((lo, hi, capped))
    }

    /// The bound of `bound` along one path, given backwards from the load.
    fn path_bound(&self, back: &[u64], load: u64, idx: u8) -> Option<(u64, u64, bool)> {
        let path: Vec<u64> = back.iter().rev().copied().collect();
        // Value numbering along the path.
        let mut vn: [u32; 32] = std::array::from_fn(|r| r as u32);
        let mut fresh = 32u32;
        let mut caps: HashMap<u32, u64> = HashMap::new();
        // The low bits of one value get one number, however often they are masked.
        let mut masked: HashMap<(u32, u64), u32> = HashMap::new();
        let mut cmps: Vec<(u32, u64, usize)> = Vec::new();
        for (k, &x) in path.iter().enumerate() {
            let w = self.word(x);
            if let Some((r, imm)) = insn::cmp_imm(w) {
                cmps.push((vn[usize::from(r)], imm, k));
                continue;
            }
            let written = insn::writes(w);
            if written == 0 {
                continue;
            }
            let copy = insn::mov_reg(w)
                .map(|(d, m, _)| (d, vn[usize::from(m)]))
                .or_else(|| insn::extend_word(w).map(|(d, n)| (d, vn[usize::from(n)])));
            let low = insn::narrows(w).filter(|n| n.3).map(|(d, n, c, _)| {
                let id = *masked.entry((vn[usize::from(n)], c)).or_insert_with(|| {
                    fresh += 1;
                    fresh
                });
                (d, id, c)
            });
            let cap = insn::narrows(w)
                .map(|(d, _, c, _)| (d, c))
                .or_else(|| insn::load_narrow(w))
                .or_else(|| {
                    insn::load_indexed(w)
                        .filter(|l| l.size < 2 && !l.signed)
                        .map(|l| (l.rt, 1u64 << (8 << l.size)))
                });
            for r in 0..31u8 {
                if written & insn::bit(r) != 0 {
                    fresh += 1;
                    vn[usize::from(r)] = fresh;
                }
            }
            if let Some((d, v)) = copy.filter(|&(d, _)| d < 31) {
                vn[usize::from(d)] = v;
            } else if let Some((d, id, c)) = low.filter(|&(d, _, _)| d < 31) {
                vn[usize::from(d)] = id;
                caps.insert(id, c);
            } else if let Some((d, c)) = cap.filter(|&(d, _)| d < 31) {
                caps.insert(vn[usize::from(d)], c);
            }
        }
        let v = vn[usize::from(idx)];
        let cap = caps.get(&v).copied();
        let (mut lo, mut hi) = (0u64, None::<u64>);
        let mut upper = |n: u64| hi = Some(hi.map_or(n, |h| h.min(n)));
        for &(cv, imm, k) in &cmps {
            if cv != v {
                continue;
            }
            for j in k + 1..path.len() {
                let (y, w) = (path[j], self.word(path[j]));
                if let Some(c) = insn::condition(w).filter(|_| w & 0xFF00_0010 == 0x5400_0000) {
                    let next = path.get(j + 1).copied().unwrap_or(load);
                    // The condition that holds on the path: the branch's, or its inverse if not taken.
                    match if next == y + 4 { c ^ 1 } else { c } {
                        9 | 13 => upper(imm + 1),       // ls, le
                        3 | 11 => upper(imm),           // lo, lt
                        8 | 12 => lo = lo.max(imm + 1), // hi, gt
                        2 | 10 => lo = lo.max(imm),     // hs, ge
                        0 => {
                            lo = lo.max(imm);
                            upper(imm + 1);
                        }
                        _ => {}
                    }
                    break;
                }
                if insn::condition(w).is_some() || insn::touches_flags(w) {
                    break;
                }
            }
        }
        match (hi, cap) {
            (Some(h), c) => Some((lo, c.map_or(h, |c| h.min(c)), false)),
            (None, Some(c)) => Some((lo, c, true)),
            (None, None) => None,
        }
    }
}
