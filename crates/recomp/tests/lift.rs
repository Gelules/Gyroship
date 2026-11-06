//! Hand-assembled words and functions; no game data is involved.

use gyroship_analysis::cfg::{Block, Cfg};
use gyroship_analysis::discover::Function;
use gyroship_recomp::ir::{Addr, Alu, Cond, Op, Operand, Reg, Shift, W, Writeback};
use gyroship_recomp::{Callee, Input, clif, lift::lift, rust};

#[test]
fn lifts_the_common_integer_forms() {
    let pc = 0x71_0000_1000;
    assert_eq!(lift(0x91076100, pc), Op::Alu { op: Alu::Add, w: W::W64, dst: Reg::X(0), a: Reg::X(8), b: Operand::Imm(0x1d8), flags: false });
    assert_eq!(
        lift(0x7100e77f, pc),
        Op::Alu { op: Alu::Sub, w: W::W32, dst: Reg::Zr, a: Reg::X(27), b: Operand::Imm(0x39), flags: true },
        "cmp w27, #0x39"
    );
    assert_eq!(
        lift(0x8b0a0929, pc),
        Op::Alu { op: Alu::Add, w: W::W64, dst: Reg::X(9), a: Reg::X(9), b: Operand::Shifted(Reg::X(10), Shift::Lsl, 2), flags: false }
    );
    assert_eq!(
        lift(0x8b3a5308, pc),
        Op::Alu {
            op: Alu::Add,
            w: W::W64,
            dst: Reg::X(8),
            a: Reg::X(24),
            b: Operand::Extended { reg: Reg::X(26), signed: false, bits: 32, lsl: 4 },
            flags: false
        },
        "add x8, x24, w26, uxtw #4"
    );
    assert_eq!(
        lift(0xa9bf7bfd, pc),
        Op::StorePair { a: Reg::X(29), b: Reg::X(30), size: 3, addr: Addr { base: Reg::Sp, offset: -16, index: None, writeback: Writeback::Pre } }
    );
    assert_eq!(
        lift(0x38401409, pc),
        Op::Load { dst: Reg::X(9), w: W::W32, size: 0, signed: false, addr: Addr { base: Reg::X(0), offset: 1, index: None, writeback: Writeback::Post } }
    );
    assert_eq!(
        lift(0x386869ea, pc),
        Op::Load {
            dst: Reg::X(10),
            w: W::W32,
            size: 0,
            signed: false,
            addr: Addr { base: Reg::X(15), offset: 0, index: Some((Reg::X(8), false, 64, 0)), writeback: Writeback::None }
        }
    );
    assert_eq!(lift(0xb005f8af, pc), Op::Const { dst: Reg::X(15), value: 0x71_0bf1_6000 }, "adrp");
    assert_eq!(lift(0x54001108, pc), Op::Branch(Cond::Flags(8), pc + 0x220), "b.hi");
    assert_eq!(lift(0x37180a89, pc), Op::Branch(Cond::Bit { reg: Reg::X(9), bit: 3, set: true }, pc + 0x150), "tbnz");
    assert_eq!(lift(0x940015dd, pc), Op::Call(pc + 0x5774));
    assert_eq!(lift(0xd65f0fff, pc), Op::Return, "retab");
    assert_eq!(lift(0xd503237f, pc), Op::Nop, "pacibsp");
    assert_eq!(lift(0xd53bd040, pc), Op::ReadThread { dst: Reg::X(0), ro: false }, "mrs x0, tpidr_el0");
    assert_eq!(
        lift(0xbd400040, pc),
        Op::LoadV { dst: 0, size: 2, addr: Addr { base: Reg::X(2), offset: 0, index: None, writeback: Writeback::None } },
        "ldr s0, [x2]"
    );
    assert_eq!(lift(0x1f0218a5, pc), Op::FMulAdd { f64: false, dst: 5, a: 5, b: 2, acc: 6, neg_acc: false, neg_product: false }, "fmadd");
    assert_eq!(lift(0x1e212000, pc), Op::FCmp { f64: false, a: 0, b: Some(1) }, "fcmp s0, s1");
    assert_eq!(lift(0x2f07e600, pc), Op::VConst { dst: 0, value: 0xffff_ffff_0000_0000 }, "movi d0");
    assert_eq!(lift(0x0f0007e1, pc), Op::VConst { dst: 1, value: 0x1f_0000_001f }, "movi v1.2s, #0x1f");
    assert_eq!(lift(0x1e2e1000, pc), Op::VConst { dst: 0, value: u128::from(1.0f32.to_bits()) }, "fmov s0, #1.0");
    assert_eq!(lift(0x4ea01c01, pc), Op::VMov { dst: 1, src: 0, size: 4 }, "mov v1.16b, v0.16b");
    assert_eq!(lift(0x6e042486, pc), Op::Unhandled, "ins: vector lanes are not translated yet");
}

/// `loop: subs x1, x1, #1; add x0, x0, x2; b.ne loop; bl 0x2000; ret` at 0x1000.
fn counted_loop() -> (Vec<u32>, Function, Cfg) {
    let words = vec![0xf1000421, 0x8b020000, 0x54ffffc1, 0x940003fd, 0xd65f03c0];
    let f = Function { start: 0x1000, len: 20, unwind: true, entries: vec![0x1000] };
    let cfg = Cfg { blocks: vec![Block { start: 0x1000, len: 12 }, Block { start: 0x100c, len: 8 }], ..Cfg::default() };
    (words, f, cfg)
}

#[test]
fn both_backends_emit_a_loop_with_a_call() {
    let (words, f, cfg) = counted_loop();
    let word = |a: u64| words[((a - 0x1000) / 4) as usize];
    let callee = |t: u64| if t == 0x2000 { Callee::Direct(t) } else { Callee::Dispatch(t) };
    let input = Input { function: &f, cfg: &cfg, word: &word, callee: &callee, probe: &|_| false };
    let mut src = String::new();
    rust::function(&mut src, &input);
    assert!(src.contains("pub unsafe extern \"C\" fn f_1000(c: *mut Cpu, resume: u64)"));
    assert!(src.contains("rt::subs64(c, a as _, b as _)"));
    assert!(src.contains("if rt::cond(c, 1) { b = 0; } else { b = 1; }"), "b.ne back to the loop head:\n{src}");
    assert!(src.contains("c.x[30] = 0x1010; unsafe { f_2000(c, 0) };"));
    // Cranelift, with its verifier on; the callee goes through the dispatch table here.
    let dispatch = |t: u64| Callee::Dispatch(t);
    let mut obj = clif::Object::new(&[(0x1000, 0x100)], true).unwrap();
    obj.function(&Input { callee: &dispatch, ..input }).unwrap();
    assert!(!obj.finish().unwrap().is_empty());
}
