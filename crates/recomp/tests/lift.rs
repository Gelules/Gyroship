//! Hand-assembled words and functions; no game data is involved.

use gyroship_analysis::cfg::{Block, Cfg};
use gyroship_analysis::discover::Function;
use gyroship_recomp::ir::{Addr, Alu, Cond, Op, Operand, Reg, Rmw, Shift, VCmp, VFp, VInt, W, Writeback, ZERO};
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
        Op::Load { dst: Reg::X(9), w: W::W32, size: 0, signed: false, ordered: false, addr: Addr { base: Reg::X(0), offset: 1, index: None, writeback: Writeback::Post } }
    );
    assert_eq!(
        lift(0x386869ea, pc),
        Op::Load {
            dst: Reg::X(10),
            w: W::W32,
            size: 0,
            signed: false,
            ordered: false,
            addr: Addr { base: Reg::X(15), offset: 0, index: Some((Reg::X(8), false, 64, 0)), writeback: Writeback::None }
        }
    );
    let at = |base| Addr { base: Reg::X(base), offset: 0, index: None, writeback: Writeback::None };
    assert_eq!(lift(0x88dffc20, pc), Op::Load { dst: Reg::X(0), w: W::W32, size: 2, signed: false, ordered: true, addr: at(1) }, "ldar w0, [x1]");
    assert_eq!(lift(0xc89ffc20, pc), Op::Store { src: Reg::X(0), size: 3, ordered: true, addr: at(1) }, "stlr x0, [x1]");
    assert_eq!(lift(0xd5033bbf, pc), Op::Fence, "dmb ish");
    assert_eq!(lift(0xd50339bf, pc), Op::Nop, "dmb ishld");
    assert_eq!(lift(0xd5033abf, pc), Op::Nop, "dmb ishst");
    assert_eq!(lift(0xd5033f9f, pc), Op::Fence, "dsb sy");
    assert_eq!(lift(0xf8e90108, pc), Op::AtomicRmw { op: Rmw::Add, size: 3, src: Reg::X(9), dst: Reg::X(8), addr: at(8) }, "ldaddal x9, x8, [x8]");
    assert_eq!(lift(0x3821105f, pc), Op::AtomicRmw { op: Rmw::Clr, size: 0, src: Reg::X(1), dst: Reg::Zr, addr: at(2) }, "stclrb w1, [x2]");
    assert_eq!(lift(0x88e9fe8a, pc), Op::AtomicCas { size: 2, cmp: Reg::X(9), new: Reg::X(10), addr: at(20) }, "casal w9, w10, [x20]");
    assert_eq!(lift(0xd50b7e2a, pc), Op::Nop, "dc civac, x10");
    assert_eq!(lift(0xd50b7420, pc), Op::Unhandled, "dc zva, x0: the interpreter writes the zeros");
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
    assert_eq!(lift(0x0e208400 | 3 << 22, pc), Op::Unhandled, "add v0.1d is reserved");
}

#[test]
fn lifts_the_vector_lane_forms() {
    let pc = 0x71_0000_1000;
    let at = |n| Addr { base: Reg::X(n), offset: 0, index: None, writeback: Writeback::None };
    assert_eq!(lift(0x0d409000, pc), Op::LoadLane { dst: 0, size: 2, lane: Some(1), q: true, addr: at(0) }, "ld1 {{v0.s}}[1], [x0]");
    assert_eq!(lift(0x4d40c8e8, pc), Op::LoadLane { dst: 8, size: 2, lane: None, q: true, addr: at(7) }, "ld1r {{v8.4s}}, [x7]");
    assert_eq!(lift(0x0d009000, pc), Op::StoreLane { src: 0, size: 2, lane: 1, addr: at(0) }, "st1 {{v0.s}}[1], [x0]");
    assert_eq!(lift(0x4e0c1d00, pc), Op::ToLane { dst: 0, src: Reg::X(8), size: 2, lane: Some(1), q: true }, "mov v0.s[1], w8");
    assert_eq!(
        lift(0x4f831368, pc),
        Op::VFp { op: VFp::Mla, f64: false, q: true, dst: 8, a: 27, b: 3, lane: Some(0) },
        "fmla v8.4s, v27.4s, v3.s[0]"
    );
    assert_eq!(
        lift(0x2eaaad08, pc),
        Op::VInt { op: VInt::UMin, size: 2, q: false, pairwise: true, dst: 8, a: 8, b: 10 },
        "uminp v8.2s, v8.2s, v10.2s"
    );
    assert_eq!(lift(0x4ea0c820, pc), Op::VFCmp { cmp: VCmp::Gt, f64: false, q: true, dst: 0, a: 1, b: None }, "fcmgt v0.4s, v1.4s, #0.0");
    let mask: [u8; 16] = std::array::from_fn(|i| i as u8 + 8);
    assert_eq!(lift(0x6e08410a, pc), Op::VShuffle { dst: 10, a: 8, b: 8, mask }, "ext v10.16b, v8.16b, v8.16b, #8");
    let mask: [u8; 16] = std::array::from_fn(|i| if i < 4 { 20 + i as u8 } else { i as u8 });
    assert_eq!(lift(0x6e042486, pc), Op::VShuffle { dst: 6, a: 6, b: 4, mask }, "mov v6.s[0], v4.s[1]");
    let mask: [u8; 16] = std::array::from_fn(|i| if i < 8 { 4 + i as u8 % 4 } else { ZERO });
    assert_eq!(lift(0x0e0c0441, pc), Op::VShuffle { dst: 1, a: 2, b: 2, mask }, "dup v1.2s, v2.s[1]");
    // Neighbours left to the interpreter.
    for (word, what) in [
        (0x0dc29000, "ld1 {v0.s}[1], [x0], x2"),
        (0x4c40a800, "ld1 {v0.4s, v1.4s}, [x0]"),
        (0x0d609000, "ld2 {v0.s, v1.s}[1], [x0]"),
        (0x0d829000, "st1 {v0.s}[1], [x0], x2"),
        (0x4e421420, "fadd v0.8h, v1.8h, v2.8h"),
        (0x4f3de420, "scvtf v0.4s, v1.4s, #0x3"),
        (0x5fa21020, "fmla s0, s1, v2.s[1]"),
        (0x4e22dc20, "fmulx v0.4s, v1.4s, v2.4s"),
        (0x6ea2d420, "fabd v0.4s, v1.4s, v2.4s"),
    ] {
        assert_eq!(lift(word, pc), Op::Unhandled, "{what}");
        assert_eq!(gyroship_runtime::interp::disassemble(word), what);
    }
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
