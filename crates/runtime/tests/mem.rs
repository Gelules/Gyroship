//! Guest memory on synthetic layouts.

use gyroship_runtime::mem::{self, Block, Error, Info, Memory, attr, perm, state};
use gyroship_runtime::{Cpu, ld64, st64};

const TEXT: u64 = 0x70_0000_0000;
const STACK: u64 = 0x20_0000_0000;
const HEAP: u64 = 0x10_0000_0000;

/// Walks the address space the way rtld does: query, jump to base + size,
/// stop when the address wraps to zero.
fn walk(m: &Memory) -> Vec<Info> {
    let mut out = Vec::new();
    let mut addr = 0u64;
    loop {
        let info = m.query(addr);
        out.push(info);
        addr = info.base.wrapping_add(info.size);
        if addr == 0 || out.len() > 100 {
            return out;
        }
    }
}

#[test]
fn walk_sees_free_gaps_and_the_inaccessible_tail() {
    let mut m = Memory::new().unwrap();
    m.map(TEXT, 0x4000, Block::new(state::CODE, perm::RX)).unwrap();
    m.map(TEXT + 0x4000, 0x1000, Block::new(state::CODE, perm::R)).unwrap();
    m.map(TEXT + 0x5000, 0x1000, Block::new(state::CODE_DATA, perm::RW)).unwrap();
    m.map(TEXT + 0x6000, 0x2000, Block::new(state::CODE_DATA, perm::RW)).unwrap();
    m.map(STACK, 0x10000, Block::new(state::STACK, perm::RW)).unwrap();
    let free = Block::new(state::FREE, 0);
    let expected = [
        Info { base: 0, size: STACK, block: free },
        Info { base: STACK, size: 0x10000, block: Block::new(state::STACK, perm::RW) },
        Info { base: STACK + 0x10000, size: TEXT - STACK - 0x10000, block: free },
        Info { base: TEXT, size: 0x4000, block: Block::new(state::CODE, perm::RX) },
        Info { base: TEXT + 0x4000, size: 0x1000, block: Block::new(state::CODE, perm::R) },
        // data and bss merge: same state, permission and attribute.
        Info { base: TEXT + 0x5000, size: 0x3000, block: Block::new(state::CODE_DATA, perm::RW) },
        Info { base: TEXT + 0x8000, size: mem::SPACE - TEXT - 0x8000, block: free },
        Info { base: mem::SPACE, size: mem::SPACE.wrapping_neg(), block: Block::new(state::INACCESSIBLE, 0) },
    ];
    assert_eq!(walk(&m), expected);
    assert_eq!(m.query(TEXT + 0x6800), expected[5]);
}

#[test]
fn memory_info_layout() {
    let info = Info { base: 0x1122_3344_5566_7788, size: 0x4000, block: Block { state: state::CODE, perm: perm::RX, attr: attr::LOCKED } };
    let b = info.to_bytes();
    assert_eq!(&b[0..8], &0x1122_3344_5566_7788u64.to_le_bytes());
    assert_eq!(&b[8..16], &0x4000u64.to_le_bytes());
    assert_eq!(&b[16..28], &[3, 0, 0, 0, 1, 0, 0, 0, 5, 0, 0, 0]);
    assert_eq!(&b[28..], &[0; 12]);
}

#[test]
fn bad_ranges_are_refused() {
    let mut m = Memory::new().unwrap();
    let rw = Block::new(state::NORMAL, perm::RW);
    assert_eq!(m.map(HEAP + 1, 0x1000, rw), Err(Error::InvalidRange));
    assert_eq!(m.map(HEAP, 0, rw), Err(Error::InvalidRange));
    assert_eq!(m.map(mem::SPACE - 0x1000, 0x2000, rw), Err(Error::InvalidRange));
    m.map(HEAP, 0x3000, rw).unwrap();
    assert_eq!(m.map(HEAP + 0x2000, 0x2000, rw), Err(Error::NotFree));
    assert_eq!(m.unmap(HEAP + 0x2000, 0x2000), Err(Error::InvalidState));
    assert_eq!(m.protect(HEAP - 0x1000, 0x2000, perm::R), Err(Error::InvalidState));
}

#[test]
fn protect_splits_then_merges() {
    let mut m = Memory::new().unwrap();
    m.map(HEAP, 0x3000, Block::new(state::NORMAL, perm::RW)).unwrap();
    m.protect(HEAP + 0x1000, 0x1000, perm::R).unwrap();
    let perms: Vec<_> = m.blocks().map(|i| (i.base, i.size, i.block.perm)).collect();
    assert_eq!(perms, [(HEAP, 0x1000, perm::RW), (HEAP + 0x1000, 0x1000, perm::R), (HEAP + 0x2000, 0x1000, perm::RW)]);
    assert_eq!(m.write(HEAP + 0xff8, &[0; 16]), Err(Error::InvalidState));
    m.protect(HEAP + 0x1000, 0x1000, perm::RW).unwrap();
    assert_eq!(m.blocks().count(), 1);
    m.unmap(HEAP + 0x1000, 0x1000).unwrap();
    assert_eq!(m.blocks().count(), 2);
    assert_eq!(m.query(HEAP + 0x1000).block.state, state::FREE);
}

#[test]
fn map_move_locks_the_source_and_unmap_copies_back() {
    let mut m = Memory::new().unwrap();
    m.map(HEAP, 0x2000, Block::new(state::NORMAL, perm::RW)).unwrap();
    m.write(HEAP + 0x10, b"stack bytes").unwrap();
    m.map_move(STACK, HEAP, 0x2000, state::STACK).unwrap();
    let src = m.query(HEAP).block;
    assert_eq!((src.perm, src.attr), (0, attr::LOCKED));
    let mut buf = [0; 11];
    assert_eq!(m.read(HEAP + 0x10, &mut buf), Err(Error::InvalidState));
    m.read(STACK + 0x10, &mut buf).unwrap();
    assert_eq!(&buf, b"stack bytes");
    m.write(STACK + 0x10, b"STACK").unwrap();
    // A second move of a locked source is refused.
    assert_eq!(m.map_move(STACK + 0x10000, HEAP, 0x2000, state::STACK), Err(Error::InvalidState));
    // The destination must be the moved range: the locked source itself is refused.
    assert_eq!(m.unmap_move(HEAP, HEAP, 0x2000, state::STACK), Err(Error::InvalidState));
    m.unmap_move(STACK, HEAP, 0x2000, state::STACK).unwrap();
    assert_eq!(m.query(STACK).block.state, state::FREE);
    m.read(HEAP + 0x10, &mut buf).unwrap();
    assert_eq!(&buf, b"STACK bytes");
    assert_eq!(m.query(HEAP).block, Block::new(state::NORMAL, perm::RW));
}

/// Runs this test binary again in fault mode `mode`; returns its exit code and stderr.
fn run_child(mode: &str) -> (Option<i32>, String) {
    let out = std::process::Command::new(std::env::current_exe().unwrap())
        .args(["--exact", "fault_child", "--nocapture", "--test-threads=1"])
        .env("GYROSHIP_FAULT_CHILD", mode)
        .output()
        .unwrap();
    (out.status.code(), String::from_utf8_lossy(&out.stderr).into_owned())
}

/// Only does something when started by `faults_are_logged_with_the_guest_address`.
#[test]
fn fault_child() {
    let Ok(mode) = std::env::var("GYROSHIP_FAULT_CHILD") else { return };
    let mut m = Memory::new().unwrap();
    m.map(HEAP, 0x1000, Block::new(state::CODE, perm::R)).unwrap();
    mem::install_fault_handler();
    let cpu = Cpu::new(m.base());
    match mode.as_str() {
        "hole" => drop(ld64(&cpu, 0x1234)),
        "wild" => drop(ld64(&cpu, mem::SPACE + 0x5678)),
        "readonly" => st64(&cpu, HEAP + 8, 1),
        _ => unreachable!(),
    }
    eprintln!("no fault");
}

#[test]
fn faults_are_logged_with_the_guest_address() {
    for (mode, line) in [
        ("hole", "guest fault: read at guest address 0x1234 "),
        // The 39-bit mask keeps a wild address inside the reservation.
        ("wild", "guest fault: read at guest address 0x5678 "),
        ("readonly", "guest fault: write at guest address 0x1000000008 "),
    ] {
        let (code, stderr) = run_child(mode);
        assert_eq!(code, Some(mem::FAULT_EXIT), "{mode}: {stderr}");
        assert!(stderr.contains(line), "{mode}: {stderr}");
    }
}
