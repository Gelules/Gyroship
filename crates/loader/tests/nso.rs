//! Tests on a synthetic NSO built here; no game data is involved.

use gyroship_loader::Module;
use sha2::{Digest, Sha256};

fn put(buf: &mut [u8], off: usize, v: &[u8]) {
    buf[off..off + v.len()].copy_from_slice(v);
}

/// text @0 (0x100), ro @0x1000 (0x200), data @0x2000 (0x100), bss 0x100.
/// MOD0 at 0x10; dynamic in data; one defined and one imported symbol;
/// one relative and one jump-slot relocation.
fn synthetic_nso() -> Vec<u8> {
    let mut text = vec![0u8; 0x100];
    put(&mut text, 4, &0x10u32.to_le_bytes());
    put(&mut text, 0x10, b"MOD0");
    for (i, target) in [0x2000i64, 0x2100, 0x2200, 0, 0, 0x2100].iter().enumerate() {
        let rel = if *target == 0 { 0 } else { (*target - 0x10) as i32 };
        put(&mut text, 0x14 + 4 * i, &rel.to_le_bytes());
    }

    let mut ro = vec![0u8; 0x200];
    // symtab @0x1000: null, "main_func" defined in text, "imported" undefined.
    put(&mut ro, 24 + 0, &1u32.to_le_bytes());
    ro[24 + 4] = 0x12; // global function
    put(&mut ro, 24 + 6, &1u16.to_le_bytes());
    put(&mut ro, 24 + 8, &0x40u64.to_le_bytes());
    put(&mut ro, 24 + 16, &0x20u64.to_le_bytes());
    put(&mut ro, 48, &11u32.to_le_bytes());
    ro[48 + 4] = 0x12;
    let strtab = b"\0main_func\0imported\0libfoo.so\0";
    put(&mut ro, 0x40, strtab);
    // DT_HASH @0x1080: nbucket 1, nchain 3.
    put(&mut ro, 0x80, &1u32.to_le_bytes());
    put(&mut ro, 0x84, &3u32.to_le_bytes());
    // rela @0x10C0: R_AARCH64_RELATIVE (1027); jmprel @0x10D8: JUMP_SLOT (1026) to symbol 2.
    put(&mut ro, 0xC0, &0x2080u64.to_le_bytes());
    put(&mut ro, 0xC8, &1027u64.to_le_bytes());
    put(&mut ro, 0xD0, &0x40i64.to_le_bytes());
    put(&mut ro, 0xD8, &0x2088u64.to_le_bytes());
    put(&mut ro, 0xE0, &((2u64 << 32) | 1026).to_le_bytes());

    let mut data = vec![0u8; 0x100];
    let dynamic: [(i64, u64); 11] = [
        (5, 0x1040),
        (10, strtab.len() as u64),
        (6, 0x1000),
        (11, 24),
        (4, 0x1080),
        (7, 0x10C0),
        (8, 24),
        (23, 0x10D8),
        (2, 24),
        (1, 20),
        (0, 0),
    ];
    for (i, (tag, val)) in dynamic.iter().enumerate() {
        put(&mut data, 16 * i, &tag.to_le_bytes());
        put(&mut data, 16 * i + 8, &val.to_le_bytes());
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

#[test]
fn parses_segments_dynamic_symbols_and_relocations() {
    let m = Module::parse_nso(&synthetic_nso()).unwrap();
    assert_eq!(m.build_id, [0x5A; 32]);
    assert_eq!((m.ro.addr, m.data.addr, m.data.size), (0x1000, 0x2000, 0x100));
    assert_eq!(&m.image[0x10..0x14], b"MOD0");
    assert_eq!(m.mod0.dynamic, 0x2000);
    assert_eq!(m.mod0.bss_end, 0x2200);
    assert_eq!(m.mem_end(), 0x2200);
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
    let mut nso = synthetic_nso();
    let ro_hash = 0xA0 + 0x20;
    nso[ro_hash] ^= 1;
    let err = Module::parse_nso(&nso).unwrap_err();
    assert!(err.0.contains("SHA-256"), "{err}");
    assert!(Module::parse_nso(&nso[..0x80]).is_err());
}

#[test]
fn elf_export_maps_segments_at_their_addresses() {
    let m = Module::parse_nso(&synthetic_nso()).unwrap();
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
    assert_eq!(ph(2, 40), 0x200); // data memsz includes bss
    assert_eq!(u32::from_le_bytes(elf[0x40 + 56 * 3..][..4].try_into().unwrap()), 2); // PT_DYNAMIC
    assert_eq!(ph(3, 16), 0x2000);
}
