//! Synthetic functions assembled by hand; no game data is involved.

use gyroship_analysis::cfg::{self, Block};
use gyroship_analysis::discover::Function;
use gyroship_analysis::insn;

const BASE: u64 = 0x71_0000_0000;
const TABLE: u64 = 0x800;

fn cmp(n: u32, imm: u32) -> u32 {
    0x7100_001F | imm << 10 | n << 5
}
fn b_cond(cond: u32, from: u64, to: u64) -> u32 {
    0x5400_0000 | ((((to as i64 - from as i64) / 4) as u32) & 0x7FFFF) << 5 | cond
}
fn b(from: u64, to: u64) -> u32 {
    0x1400_0000 | (((to as i64 - from as i64) / 4) as u32 & 0x03FF_FFFF)
}
fn adrp(d: u32, from: u64, to: u64) -> u32 {
    let pages = ((to & !0xFFF) as i64 - (from & !0xFFF) as i64) >> 12;
    0x9000_0000 | ((pages as u32) & 3) << 29 | (((pages >> 2) as u32) & 0x7FFFF) << 5 | d
}
fn add_imm(d: u32, n: u32, imm: u32) -> u32 {
    0x9100_0000 | imm << 10 | n << 5 | d
}
fn adr(d: u32, from: u64, to: u64) -> u32 {
    let off = (to as i64 - from as i64) as u32;
    0x1000_0000 | (off & 3) << 29 | ((off >> 2) & 0x7FFFF) << 5 | d
}
fn ldrb(t: u32, n: u32, m: u32) -> u32 {
    0x3860_6800 | m << 16 | n << 5 | t
}
fn add_lsl(d: u32, n: u32, m: u32, s: u32) -> u32 {
    0x8B00_0000 | m << 16 | s << 10 | n << 5 | d
}
fn br(n: u32) -> u32 {
    0xD61F_0000 | n << 5
}
const RET: u32 = 0xD65F_03C0;
const NOP: u32 = 0xD503_201F;
const HI: u32 = 8;

/// Places `code` at offset 0 and byte `entries` at TABLE, in a module at BASE.
fn image(code: &[u32], entries: &[u8]) -> (Vec<u8>, Function) {
    let mut img = vec![0u8; 0x1000];
    for (i, w) in code.iter().enumerate() {
        img[4 * i..4 * i + 4].copy_from_slice(&w.to_le_bytes());
    }
    img[TABLE as usize..TABLE as usize + entries.len()].copy_from_slice(entries);
    let f = Function {
        start: BASE,
        len: 4 * code.len() as u64,
        unwind: true,
        entries: vec![BASE],
    };
    (img, f)
}

/// `cmp w0, #N; b.hi default` guards `ldrb w11, [x9, x0]` from a table of
/// word offsets from the label after the BR.
fn switch(code: &mut Vec<u32>, at: u64) {
    code.extend([
        adrp(9, at, BASE + TABLE),
        add_imm(9, 9, TABLE as u32),
        adr(10, at + 8, at + 0x18),
        ldrb(11, 9, 0),
        add_lsl(10, 10, 11, 2),
        br(10),
    ]);
}

#[test]
fn recovers_a_guarded_jump_table() {
    let mut code = vec![cmp(0, 2), b_cond(HI, BASE + 4, BASE + 0x28)];
    switch(&mut code, BASE + 8);
    code.extend([RET, NOP, RET]); // cases at +0x20, +0x24, +0x28
    let (img, f) = image(&code, &[0, 1, 2, 0x7F]);
    let g = cfg::build(&img, BASE, &f);
    assert_eq!(g.tables.len(), 1);
    let t = &g.tables[0];
    assert_eq!(
        (t.branch, t.table, t.entry_size, t.capped),
        (BASE + 0x1C, BASE + TABLE, 1, false)
    );
    assert_eq!(
        t.targets,
        [BASE + 0x20, BASE + 0x24, BASE + 0x28],
        "three entries, not the fourth"
    );
    assert!(g.indirect.is_empty() && g.undecodable.is_empty());
    let starts: Vec<u64> = g.blocks.iter().map(|b| b.start - BASE).collect();
    assert_eq!(starts, [0, 8, 0x20, 0x24, 0x28]);
    assert_eq!(
        g.blocks[3],
        Block {
            start: BASE + 0x24,
            len: 4
        },
        "the NOP case ends where the next case starts"
    );
}

#[test]
fn bounds_the_index_on_every_path_to_the_table() {
    // Index <= 2 falls into the table; index > 2 is checked again (<= 4) and jumps back.
    let mut code = vec![cmp(0, 2), b_cond(HI, BASE + 4, BASE + 0x34)];
    switch(&mut code, BASE + 8);
    code.extend([RET, RET, RET, RET, RET]); // cases at +0x20..+0x30
    code.extend([
        cmp(0, 4),
        b_cond(HI, BASE + 0x38, BASE + 0x40),
        b(BASE + 0x3C, BASE + 8),
        RET,
    ]);
    let (img, f) = image(&code, &[0, 1, 2, 3, 4, 0x7F]);
    let g = cfg::build(&img, BASE, &f);
    let t = &g.tables[0];
    assert!(!t.capped);
    assert_eq!(
        t.targets.len(),
        5,
        "the taken b.hi only bounds the index from below"
    );
    assert!(
        g.blocks.iter().any(|b| b.start == BASE + 0x30),
        "case 4 is reached"
    );
}

#[test]
fn an_unbounded_index_reads_up_to_the_first_invalid_entry() {
    let mut code = Vec::new();
    switch(&mut code, BASE);
    code.extend([RET, RET]);
    let (img, f) = image(&code, &[0, 1, 0x40]);
    let g = cfg::build(&img, BASE, &f);
    let t = &g.tables[0];
    assert!(t.capped);
    assert_eq!(t.targets, [BASE + 0x18, BASE + 0x1C]);
}

#[test]
fn a_branch_through_a_loaded_pointer_stays_indirect() {
    // ldr x8, [x0]; br x8
    let (img, f) = image(&[0xF940_0008, br(8)], &[]);
    let g = cfg::build(&img, BASE, &f);
    assert!(g.tables.is_empty());
    assert_eq!(g.indirect, [BASE + 4]);
}

#[test]
fn register_writes_and_flow() {
    assert_eq!(
        insn::writes(0xA9C27BFD),
        1 << 29 | 1 << 30,
        "ldp x29, x30, [sp, #0x20]! (sp is not tracked)"
    );
    assert_eq!(insn::writes(0xF9000109), 0, "str x9, [x8]");
    assert_eq!(
        insn::writes(0x38401409),
        1 << 9 | 1 << 0,
        "ldrb w9, [x0], #1 writes back x0"
    );
    assert_eq!(insn::writes(cmp(9, 1)), 0);
    assert_eq!(
        insn::writes(0x9400_0010),
        0x4007_FFFF,
        "bl clobbers x0-x18 and x30"
    );
    assert_eq!(insn::writes(0xB8AA0128), 1 << 8, "ldadda w10, w8, [x9]");
    assert_eq!(insn::flow(0x9400_0010, 0x1000), insn::Flow::Call(0x1040));
    assert_eq!(
        insn::flow(b_cond(HI, 0x1000, 0xF00), 0x1000),
        insn::Flow::Branch(0xF00)
    );
    assert_eq!(insn::flow(0xD65F_0FFF, 0), insn::Flow::Return, "retab");
    assert_eq!(insn::flow(0x0000_8002, 0), insn::Flow::Trap, "udf");
    assert_eq!(
        insn::adrp(adrp(16, 0x7100_1234, 0x7123_4000), 0x7100_1234),
        Some((16, 0x7123_4000))
    );
}
