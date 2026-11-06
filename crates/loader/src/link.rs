//! Places the modules of one exefs in a single guest address space, applies
//! their relocations and resolves imports by name, as the OS loader does.

use crate::{bad, dt, u64_at, Error, Module, Result};
use std::collections::HashMap;

pub const R_AARCH64_ABS64: u32 = 257;
pub const R_AARCH64_GLOB_DAT: u32 = 1025;
pub const R_AARCH64_JUMP_SLOT: u32 = 1026;
pub const R_AARCH64_RELATIVE: u32 = 1027;
pub const R_AARCH64_TLS_TPREL: u32 = 1031;

/// Modules in the order the OS loads them; symbol lookup follows this order.
pub const LOAD_ORDER: [&str; 7] = ["rtld", "main", "subsdk0", "subsdk1", "subsdk2", "subsdk3", "sdk"];

/// Guest base of each module: the Ghidra project's bases, so a guest address
/// is also a Ghidra address. All of them fit the 39-bit address space.
pub fn base_of(name: &str) -> Option<u64> {
    Some(match name {
        "rtld" => 0x70_0000_0000,
        "main" => 0x71_0000_0000,
        "subsdk0" => 0x72_0000_0000,
        "subsdk1" => 0x72_1000_0000,
        "subsdk2" => 0x72_2000_0000,
        "subsdk3" => 0x72_3000_0000,
        "sdk" => 0x72_4000_0000,
        _ => return None,
    })
}

#[derive(Debug, Clone)]
pub struct Placed {
    pub name: String,
    pub base: u64,
    /// Its image is relocated for `base` and extended with zeroed bss.
    pub module: Module,
}

impl Placed {
    pub fn end(&self) -> u64 {
        self.base + self.module.image.len() as u64
    }
}

/// A relocation that refers to a symbol by name, or any TLS relocation.
#[derive(Debug, Clone)]
pub struct Import {
    pub module: usize,
    /// Guest address of the patched pointer.
    pub slot: u64,
    pub name: String,
    pub kind: u32,
    /// Resolved guest address; `None` when no module defines the symbol, or
    /// for TLS relocations, which the runtime resolves (M2).
    pub target: Option<u64>,
}

#[derive(Debug, Clone, Default)]
pub struct Stats {
    pub relr: u64,
    pub relative: u64,
    pub symbolic: u64,
}

#[derive(Debug, Clone)]
pub struct Program {
    /// In load order.
    pub modules: Vec<Placed>,
    pub stats: Vec<Stats>,
    pub imports: Vec<Import>,
}

impl Program {
    /// Places each named module at its base, then relocates and links all of them.
    pub fn link(modules: Vec<(String, Module)>) -> Result<Program> {
        let mut placed = Vec::new();
        for (name, mut module) in modules {
            let Some(base) = base_of(&name) else { return bad(format!("no guest base for module {name:?}")) };
            module.image.resize(module.mem_end() as usize, 0);
            placed.push(Placed { name, base, module });
        }
        let order = |p: &Placed| LOAD_ORDER.iter().position(|n| *n == p.name);
        placed.sort_by_key(order);
        for pair in placed.windows(2) {
            if pair[0].name == pair[1].name {
                return bad(format!("module {} given twice", pair[0].name));
            }
            if pair[0].end() > pair[1].base {
                return bad(format!("module {} overlaps {}", pair[0].name, pair[1].name));
            }
        }

        // Global scope: first definition in load order wins, weak or not.
        let mut scope: HashMap<&str, u64> = HashMap::new();
        for p in &placed {
            for s in &p.module.symbols {
                if s.is_defined() && matches!(s.info >> 4, 1 | 2 | 10) {
                    scope.entry(s.name.as_str()).or_insert(p.base + s.value);
                }
            }
        }
        let scope: HashMap<String, u64> = scope.into_iter().map(|(k, v)| (k.to_string(), v)).collect();

        let mut stats = Vec::new();
        let mut imports = Vec::new();
        for (index, p) in placed.iter_mut().enumerate() {
            stats.push(relocate(index, p, &scope, &mut imports)?);
        }
        Ok(Program { modules: placed, stats, imports })
    }

    pub fn unresolved(&self) -> impl Iterator<Item = &Import> {
        self.imports.iter().filter(|i| i.target.is_none())
    }
}

fn relocate(index: usize, p: &mut Placed, scope: &HashMap<String, u64>, imports: &mut Vec<Import>) -> Result<Stats> {
    let base = p.base;
    let m = &mut p.module;
    let mut stats = Stats::default();

    // DT_RELR: an even word is an address to fix, then each odd word is a
    // bitmap of the 63 words that follow it.
    if let (Some(relr), Some(size)) = (m.dyn_value(dt::RELR), m.dyn_value(dt::RELRSZ)) {
        let mut next = 0u64;
        for i in 0..size / 8 {
            let word = u64_at(&m.image, relr + 8 * i)?;
            let mut fix = |at: u64| -> Result<()> {
                let v = u64_at(&m.image, at)?;
                write(&mut m.image, at, v.wrapping_add(base))
            };
            if word & 1 == 0 {
                fix(word)?;
                stats.relr += 1;
                next = word + 8;
            } else {
                for bit in 0..63 {
                    if word >> (bit + 1) & 1 != 0 {
                        fix(next + 8 * bit)?;
                        stats.relr += 1;
                    }
                }
                next += 8 * 63;
            }
        }
    }

    for r in m.relocations.clone() {
        let value = match r.kind {
            R_AARCH64_RELATIVE => {
                stats.relative += 1;
                base.wrapping_add(r.addend as u64)
            }
            R_AARCH64_ABS64 | R_AARCH64_GLOB_DAT | R_AARCH64_JUMP_SLOT | R_AARCH64_TLS_TPREL => {
                stats.symbolic += 1;
                let Some(sym) = m.symbols.get(r.symbol as usize) else {
                    return bad(format!("relocation at {:#x} names symbol {} of {}", r.offset, r.symbol, m.symbols.len()));
                };
                let import = |target| Import { module: index, slot: base + r.offset, name: sym.name.clone(), kind: r.kind, target };
                if r.kind == R_AARCH64_TLS_TPREL {
                    // The offset depends on the thread-pointer layout the runtime picks.
                    imports.push(import(None));
                    continue;
                }
                let target = if r.symbol == 0 {
                    Some(0)
                } else if sym.is_defined() && sym.info >> 4 == 0 {
                    Some(base + sym.value)
                } else {
                    scope.get(&sym.name).copied().or(if sym.info >> 4 == 2 { Some(0) } else { None })
                };
                if r.symbol != 0 {
                    imports.push(import(target));
                }
                // ponytail: unresolved pointers become 0, so a call through them faults at once.
                target.unwrap_or(0).wrapping_add(if target.is_some() { r.addend as u64 } else { 0 })
            }
            other => return bad(format!("unsupported relocation type {other} at {:#x}", r.offset)),
        };
        write(&mut m.image, r.offset, value)?;
    }
    Ok(stats)
}

fn write(image: &mut [u8], at: u64, value: u64) -> Result<()> {
    usize::try_from(at)
        .ok()
        .and_then(|o| image.get_mut(o..o.checked_add(8)?))
        .map(|b| b.copy_from_slice(&value.to_le_bytes()))
        .ok_or_else(|| Error(format!("relocation at {at:#x} is outside the module")))
}
