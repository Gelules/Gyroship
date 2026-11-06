//! Tests on synthetic NSOs built here; no game data is involved.

use gyroship_loader::link::{self, Program};
use gyroship_loader::Module;
use sha2::{Digest, Sha256};

fn put(buf: &mut [u8], off: usize, v: &[u8]) {
    buf[off..off + v.len()].copy_from_slice(v);
}

/// (offset, type, symbol index, addend)
type Rel = (u64, u32, u32, i64);

#[derive(Default)]
struct Spec<'a> {
    /// (name, st_info, value if defined); symbol 0 is the null symbol.
    syms: &'a [(&'a str, u8, Option<u64>)],
    rela: &'a [Rel],
    jmprel: &'a [Rel],
    relr: &'a [u64],
    /// Initial 64-bit words in the data segment, by module address.
    data: &'a [(u64, u64)],
}

const GLOBAL_FUNC: u8 = 0x12;
const GLOBAL_OBJECT: u8 = 0x11;
const WEAK_FUNC: u8 = 0x22;

/// text @0 (0x100), ro @0x1000 (0x200), data @0x2000 (0x200), bss 0x100.
/// ro: symtab @0x1000, strtab @0x1060, DT_HASH @0x10C0, rela @0x10D0,
/// jmprel @0x1130, relr @0x1160. data: dynamic @0x2000, free from 0x2100.
fn nso(spec: &Spec) -> Vec<u8> {
    assert!(spec.syms.len() <= 3 && spec.rela.len() <= 4 && spec.jmprel.len() <= 2 && spec.relr.len() <= 4);
    let mut text = vec![0u8; 0x100];
    put(&mut text, 4, &0x10u32.to_le_bytes());
    put(&mut text, 0x10, b"MOD0");
    for (i, target) in [0x2000i64, 0x2200, 0x2300, 0, 0, 0x2200].iter().enumerate() {
        let rel = if *target == 0 { 0 } else { (*target - 0x10) as i32 };
        put(&mut text, 0x14 + 4 * i, &rel.to_le_bytes());
    }

    let mut ro = vec![0u8; 0x200];
    let mut strtab = vec![0u8];
    for (i, (name, info, value)) in spec.syms.iter().enumerate() {
        let s = 24 * (i + 1);
        put(&mut ro, s, &(strtab.len() as u32).to_le_bytes());
        ro[s + 4] = *info;
        put(&mut ro, s + 6, &u16::from(value.is_some()).to_le_bytes());
        put(&mut ro, s + 8, &value.unwrap_or(0).to_le_bytes());
        put(&mut ro, s + 16, &0x20u64.to_le_bytes());
        strtab.extend_from_slice(name.as_bytes());
        strtab.push(0);
    }
    let needed = strtab.len() as u64;
    strtab.extend_from_slice(b"libfoo.so\0");
    assert!(strtab.len() <= 0x60);
    put(&mut ro, 0x60, &strtab);
    put(&mut ro, 0xC0, &1u32.to_le_bytes());
    put(&mut ro, 0xC4, &(spec.syms.len() as u32 + 1).to_le_bytes());
    for (table, start) in [(spec.rela, 0xD0), (spec.jmprel, 0x130)] {
        for (i, (offset, kind, sym, addend)) in table.iter().enumerate() {
            let r = start + 24 * i;
            put(&mut ro, r, &offset.to_le_bytes());
            put(&mut ro, r + 8, &((u64::from(*sym) << 32) | u64::from(*kind)).to_le_bytes());
            put(&mut ro, r + 16, &addend.to_le_bytes());
        }
    }
    for (i, word) in spec.relr.iter().enumerate() {
        put(&mut ro, 0x160 + 8 * i, &word.to_le_bytes());
    }

    let mut data = vec![0u8; 0x200];
    let mut dynamic: Vec<(i64, u64)> = vec![
        (5, 0x1060),
        (10, strtab.len() as u64),
        (6, 0x1000),
        (11, 24),
        (4, 0x10C0),
        (7, 0x10D0),
        (8, 24 * spec.rela.len() as u64),
        (23, 0x1130),
        (2, 24 * spec.jmprel.len() as u64),
        (1, needed),
    ];
    if !spec.relr.is_empty() {
        dynamic.extend([(36, 0x1160), (35, 8 * spec.relr.len() as u64)]);
    }
    dynamic.push((0, 0));
    for (i, (tag, val)) in dynamic.iter().enumerate() {
        put(&mut data, 16 * i, &tag.to_le_bytes());
        put(&mut data, 16 * i + 8, &val.to_le_bytes());
    }
    for (addr, value) in spec.data {
        put(&mut data, *addr as usize - 0x2000, &value.to_le_bytes());
    }

    let segs = [(&text, 0u32, true), (&ro, 0x1000, true), (&data, 0x2000, false)];
    let mut file = vec![0u8; 0x100];
    put(&mut file, 0, b"NSO0");
    let mut flags = 0u32;
    for (i, (content, addr, compress)) in segs.iter().enumerate() {
        let stored = if *compress { lz4_flex::block::compress(content) } else { content.to_vec() };
        flags |= (u32::from(*compress) << i) | (1 << (i + 3));
        let h = 0x10 + 0x10 * i;
        let file_off = file.len() as u32;
        put(&mut file, h, &file_off.to_le_bytes());
        put(&mut file, h + 4, &addr.to_le_bytes());
        put(&mut file, h + 8, &(content.len() as u32).to_le_bytes());
        put(&mut file, 0x60 + 4 * i, &(stored.len() as u32).to_le_bytes());
        put(&mut file, 0xA0 + 0x20 * i, &Sha256::digest(content));
        file.extend_from_slice(&stored);
    }
    put(&mut file, 0x0C, &flags.to_le_bytes());
    put(&mut file, 0x3C, &0x100u32.to_le_bytes());
    put(&mut file, 0x40, &[0x5A; 32]);
    file
}

fn module(spec: &Spec) -> Module {
    Module::parse_nso(&nso(spec)).unwrap()
}

fn word(p: &Program, module: usize, addr: u64) -> u64 {
    u64::from_le_bytes(p.modules[module].module.image[addr as usize..][..8].try_into().unwrap())
}

const SIMPLE: Spec = Spec {
    syms: &[("main_func", GLOBAL_FUNC, Some(0x40)), ("imported", GLOBAL_FUNC, None)],
    rela: &[(0x2100, 1027, 0, 0x40)],
    jmprel: &[(0x2108, 1026, 2, 0)],
    relr: &[],
    data: &[],
};

#[test]
fn parses_segments_dynamic_symbols_and_relocations() {
    let m = module(&SIMPLE);
    assert_eq!(m.build_id, [0x5A; 32]);
    assert_eq!((m.ro.addr, m.data.addr, m.data.size), (0x1000, 0x2000, 0x200));
    assert_eq!(&m.image[0x10..0x14], b"MOD0");
    assert_eq!(m.mod0.dynamic, 0x2000);
    assert_eq!(m.mod0.bss_end, 0x2300);
    assert_eq!(m.mem_end(), 0x2300);
    assert_eq!(m.needed, vec!["libfoo.so".to_string()]);
    assert_eq!(m.symbols.len(), 3);
    assert_eq!(m.symbols[1].name, "main_func");
    assert!(m.symbols[1].is_defined() && m.symbols[1].is_function());
    assert_eq!(m.symbols[2].name, "imported");
    assert!(!m.symbols[2].is_defined());
    assert_eq!(m.relocations.len(), 2);
    assert_eq!((m.relocations[0].kind, m.relocations[0].addend), (1027, 0x40));
    assert_eq!((m.relocations[1].kind, m.relocations[1].symbol), (1026, 2));
}

#[test]
fn rejects_corrupt_segments() {
    let mut nso = nso(&SIMPLE);
    let ro_hash = 0xA0 + 0x20;
    nso[ro_hash] ^= 1;
    let err = Module::parse_nso(&nso).unwrap_err();
    assert!(err.0.contains("SHA-256"), "{err}");
    assert!(Module::parse_nso(&nso[..0x80]).is_err());
}

#[test]
fn elf_export_maps_segments_at_their_addresses() {
    let m = module(&SIMPLE);
    let elf = m.to_elf();
    assert_eq!(&elf[0..4], b"\x7fELF");
    assert_eq!(u16::from_le_bytes([elf[16], elf[17]]), 3); // ET_DYN
    assert_eq!(u16::from_le_bytes([elf[18], elf[19]]), 183); // AArch64
    let phnum = u16::from_le_bytes([elf[56], elf[57]]) as usize;
    assert_eq!(phnum, 4);
    let ph = |i: usize, f: usize| u64::from_le_bytes(elf[0x40 + 56 * i + f..][..8].try_into().unwrap());
    for i in 0..3 {
        let (offset, vaddr, filesz) = (ph(i, 8) as usize, ph(i, 16) as usize, ph(i, 32) as usize);
        assert_eq!(&elf[offset..offset + filesz], &m.image[vaddr..vaddr + filesz], "segment {i}");
        assert_eq!(offset % 0x1000, vaddr % 0x1000);
    }
    assert_eq!(ph(2, 40), 0x300); // data memsz includes bss
    assert_eq!(u32::from_le_bytes(elf[0x40 + 56 * 3..][..4].try_into().unwrap()), 2); // PT_DYNAMIC
    assert_eq!(ph(3, 16), 0x2000);
}

#[test]
fn links_imports_and_applies_every_relocation_kind() {
    let main = module(&Spec {
        syms: &[("main_func", GLOBAL_FUNC, Some(0x40)), ("imported", GLOBAL_FUNC, None), ("table", GLOBAL_OBJECT, None)],
        rela: &[(0x2100, 1027, 0, 0x40), (0x2110, 257, 3, 8)],
        jmprel: &[(0x2108, 1026, 2, 0)],
        // 0x2118, then a bitmap whose first bit covers 0x2120 and third covers 0x2130.
        relr: &[0x2118, 0b101 << 1 | 1],
        data: &[(0x2118, 0x10), (0x2120, 0x20), (0x2128, 0x77), (0x2130, 0x30)],
    });
    let sdk = module(&Spec {
        syms: &[("imported", GLOBAL_FUNC, Some(0x60)), ("table", GLOBAL_OBJECT, Some(0x1800))],
        ..Default::default()
    });
    // Given out of load order on purpose.
    let p = Program::link(vec![("sdk".into(), sdk), ("main".into(), main)]).unwrap();
    let (main_base, sdk_base) = (link::base_of("main").unwrap(), link::base_of("sdk").unwrap());
    assert_eq!(p.modules[0].name, "main");
    assert_eq!(p.modules[0].base, main_base);
    assert_eq!(p.modules[0].module.image.len(), 0x2300, "bss is materialized");
    assert_eq!(word(&p, 0, 0x2100), main_base + 0x40); // RELATIVE
    assert_eq!(word(&p, 0, 0x2108), sdk_base + 0x60); // JUMP_SLOT
    assert_eq!(word(&p, 0, 0x2110), sdk_base + 0x1808); // ABS64 + addend
    assert_eq!(word(&p, 0, 0x2118), main_base + 0x10); // RELR address word
    assert_eq!(word(&p, 0, 0x2120), main_base + 0x20); // RELR bitmap bit 0
    assert_eq!(word(&p, 0, 0x2128), 0x77); // bit 1 clear: untouched
    assert_eq!(word(&p, 0, 0x2130), main_base + 0x30); // RELR bitmap bit 2
    let s = &p.stats[0];
    assert_eq!((s.relr, s.relative, s.symbolic), (3, 1, 2));
    assert_eq!(p.unresolved().count(), 0);
    let import = p.imports.iter().find(|i| i.name == "imported").unwrap();
    assert_eq!((import.slot, import.target), (main_base + 0x2108, Some(sdk_base + 0x60)));
}

#[test]
fn first_definition_in_load_order_wins() {
    let main = module(&Spec { syms: &[("operator_new", GLOBAL_FUNC, Some(0x50))], ..Default::default() });
    let sdk = module(&Spec {
        syms: &[("operator_new", GLOBAL_FUNC, Some(0x90))],
        jmprel: &[(0x2100, 1026, 1, 0)],
        ..Default::default()
    });
    let p = Program::link(vec![("main".into(), main), ("sdk".into(), sdk)]).unwrap();
    assert_eq!(word(&p, 1, 0x2100), link::base_of("main").unwrap() + 0x50);
}

#[test]
fn reports_unresolved_imports_and_zeroes_weak_ones() {
    let main = module(&Spec {
        syms: &[("missing", GLOBAL_FUNC, None), ("weak_missing", WEAK_FUNC, None)],
        rela: &[(0x2110, 1031, 0, 0x10)],
        jmprel: &[(0x2100, 1026, 1, 0), (0x2108, 1026, 2, 0)],
        data: &[(0x2100, 0xAAAA), (0x2108, 0xBBBB), (0x2110, 0xCCCC)],
        ..Default::default()
    });
    let p = Program::link(vec![("main".into(), main)]).unwrap();
    let unresolved: Vec<_> = p.unresolved().map(|i| (i.kind, i.name.as_str())).collect();
    assert_eq!(unresolved, [(1031, ""), (1026, "missing")]);
    assert_eq!((word(&p, 0, 0x2100), word(&p, 0, 0x2108)), (0, 0));
    assert_eq!(word(&p, 0, 0x2110), 0xCCCC, "TLS slots are left to the runtime");
}

#[test]
fn rejects_unknown_or_duplicate_modules() {
    let err = Program::link(vec![("foo".into(), module(&SIMPLE))]).unwrap_err();
    assert!(err.0.contains("foo"), "{err}");
    let err = Program::link(vec![("main".into(), module(&SIMPLE)), ("main".into(), module(&SIMPLE))]).unwrap_err();
    assert!(err.0.contains("twice"), "{err}");
}
