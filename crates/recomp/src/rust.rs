//! Backend A: Rust source. Each guest function becomes
//! `pub unsafe extern "C" fn f_<addr>(cpu: *mut Cpu)`; its blocks are the
//! arms of a `loop { match b { .. } }`.

use crate::ir::{Addr, Alu, Cond, Op, Operand, Reg, Shift, Unary, Writeback, W};
use crate::{Callee, Input};
use std::fmt::Write as _;

pub fn name(addr: u64) -> String {
    format!("f_{addr:x}")
}

fn rd(r: Reg, w: W) -> String {
    let v = match r {
        Reg::X(n) => format!("c.x[{n}]"),
        Reg::Sp => "c.x[31]".to_string(),
        Reg::Zr => return "0u64".to_string(),
    };
    match w {
        W::W64 => v,
        W::W32 => format!("({v} as u32 as u64)"),
    }
}

fn wr(r: Reg, w: W, v: &str) -> String {
    let v = match w {
        W::W64 => v.to_string(),
        W::W32 => format!("({v}) as u32 as u64"),
    };
    match r {
        Reg::X(n) => format!("c.x[{n}] = {v};"),
        Reg::Sp => format!("c.x[31] = {v};"),
        Reg::Zr => format!("let _ = {v};"),
    }
}

fn extend(v: &str, signed: bool, bits: u8) -> String {
    match (bits, signed) {
        (64, _) => v.to_string(),
        (b, false) => format!("({v} as u{b} as u64)"),
        (b, true) => format!("({v} as u{b} as i{b} as i64 as u64)"),
    }
}

fn shifted(v: &str, sh: Shift, amount: u8, w: W) -> String {
    if amount == 0 && sh != Shift::Ror {
        return v.to_string();
    }
    match (sh, w) {
        (Shift::Lsl, _) => format!("({v} << {amount})"),
        (Shift::Lsr, _) => format!("({v} >> {amount})"),
        (Shift::Asr, W::W64) => format!("(({v} as i64) >> {amount}) as u64"),
        (Shift::Asr, W::W32) => format!("(({v} as u32 as i32) >> {amount}) as u32 as u64"),
        (Shift::Ror, W::W64) => format!("{v}.rotate_right({amount})"),
        (Shift::Ror, W::W32) => format!("(({v} as u32).rotate_right({amount}) as u64)"),
    }
}

fn operand(b: Operand, w: W) -> String {
    match b {
        Operand::Imm(v) => format!("{:#x}u64", if w == W::W32 { v & 0xFFFF_FFFF } else { v }),
        Operand::Shifted(r, sh, amount) => shifted(&rd(r, w), sh, amount, w),
        Operand::Extended { reg, signed, bits, lsl } => {
            let e = extend(&rd(reg, W::W64), signed, bits);
            if lsl == 0 { e } else { format!("({e} << {lsl})") }
        }
    }
}

fn addr(a: &Addr) -> (String, String) {
    let base = rd(a.base, W::W64);
    let at = match a.index {
        Some((r, signed, bits, sh)) => format!("{base}.wrapping_add({} << {sh})", extend(&rd(r, W::W64), signed, bits)),
        None if a.writeback == Writeback::Post || a.offset == 0 => base.clone(),
        None => format!("{base}.wrapping_add({:#x}u64)", a.offset as u64),
    };
    let wb = match a.writeback {
        Writeback::None => String::new(),
        Writeback::Pre => wr(a.base, W::W64, "a"),
        Writeback::Post => wr(a.base, W::W64, &format!("a.wrapping_add({:#x}u64)", a.offset as u64)),
    };
    (at, wb)
}

fn loaded(v: &str, w: W, size: u8, signed: bool) -> String {
    let b = 8u32 << size;
    match (signed, w) {
        (false, _) => v.to_string(),
        (true, W::W64) => format!("({v} as u{b} as i{b} as i64 as u64)"),
        (true, W::W32) => format!("({v} as u{b} as i{b} as i32 as u32 as u64)"),
    }
}

fn condition(c: Cond) -> String {
    match c {
        Cond::Flags(code) => format!("rt::cond(c, {code})"),
        Cond::Zero { reg, w, nonzero } => format!("{} {} 0", rd(reg, w), if nonzero { "!=" } else { "==" }),
        Cond::Bit { reg, bit, set } => format!("({} >> {bit}) & 1 {} 0", rd(reg, W::W64), if set { "!=" } else { "==" }),
    }
}

/// A data operation as one statement block.
fn data(op: &Op) -> String {
    match *op {
        Op::Alu { op, w, dst, a, b, flags } => {
            let (x, y) = (rd(a, w), operand(b, w));
            let bits = w.bits();
            let e = match (op, flags) {
                (Alu::Add, false) => "a.wrapping_add(b)".to_string(),
                (Alu::Sub, false) => "a.wrapping_sub(b)".to_string(),
                (Alu::Add, true) => format!("rt::adds{bits}(c, a as _, b as _) as u64"),
                (Alu::Sub, true) => format!("rt::subs{bits}(c, a as _, b as _) as u64"),
                (op, flags) => {
                    let e = match op {
                        Alu::And => "a & b",
                        Alu::Orr => "a | b",
                        Alu::Eor => "a ^ b",
                        Alu::Bic => "a & !b",
                        Alu::Orn => "a | !b",
                        _ => "a ^ !b",
                    };
                    if flags { format!("rt::logic{bits}(c, ({e}) as _) as u64") } else { format!("({e})") }
                }
            };
            format!("{{ let a = {x}; let b = {y}; {} }}", wr(dst, w, &e))
        }
        Op::MovWide { w, dst, imm, shift, invert, keep } => {
            let v = u64::from(imm) << shift;
            let e = if keep {
                format!("({} & !{:#x}u64) | {v:#x}u64", rd(dst, w), 0xFFFFu64 << shift)
            } else if invert {
                format!("{:#x}u64", !v)
            } else {
                format!("{v:#x}u64")
            };
            wr(dst, w, &e)
        }
        Op::Bitfield { w, dst, src, immr, imms, signed, insert } => {
            let bits = w.bits();
            let e = if insert {
                format!("rt::bfm({bits}, {}, {}, {immr}, {imms})", rd(dst, w), rd(src, w))
            } else {
                format!("rt::{}({bits}, {}, {immr}, {imms})", if signed { "sbfm" } else { "ubfm" }, rd(src, w))
            };
            wr(dst, w, &e)
        }
        Op::Extr { w, dst, a, b, lsb } => {
            let e = if lsb == 0 { rd(b, w) } else { format!("({} >> {lsb}) | ({} << {})", rd(b, w), rd(a, w), w.bits() - u32::from(lsb)) };
            wr(dst, w, &e)
        }
        Op::ShiftVar { op, w, dst, a, b } => {
            let m = w.bits() - 1;
            let e = match (op, w) {
                (Shift::Lsl, _) => format!("a << (b & {m})"),
                (Shift::Lsr, _) => format!("a >> (b & {m})"),
                (Shift::Asr, W::W64) => format!("((a as i64) >> (b & {m})) as u64"),
                (Shift::Asr, W::W32) => format!("((a as u32 as i32) >> (b & {m})) as u32 as u64"),
                (Shift::Ror, W::W64) => format!("a.rotate_right((b & {m}) as u32)"),
                (Shift::Ror, W::W32) => format!("(a as u32).rotate_right((b & {m}) as u32) as u64"),
            };
            format!("{{ let a = {}; let b = {}; {} }}", rd(a, w), rd(b, w), wr(dst, w, &e))
        }
        Op::MulAdd { w, dst, a, b, acc, sub } => {
            let f = if sub { "wrapping_sub" } else { "wrapping_add" };
            wr(dst, w, &format!("{}.{f}({}.wrapping_mul({}))", rd(acc, w), rd(a, w), rd(b, w)))
        }
        Op::MulAddLong { dst, a, b, acc, signed, sub } => {
            let ext = |r| if signed { format!("({} as u32 as i32 as i64 as u64)", rd(r, W::W32)) } else { rd(r, W::W32) };
            let f = if sub { "wrapping_sub" } else { "wrapping_add" };
            wr(dst, W::W64, &format!("{}.{f}({}.wrapping_mul({}))", rd(acc, W::W64), ext(a), ext(b)))
        }
        Op::MulHigh { dst, a, b, signed } => {
            let e = if signed {
                format!("((({} as i64 as i128) * ({} as i64 as i128)) >> 64) as u64", rd(a, W::W64), rd(b, W::W64))
            } else {
                format!("((({} as u128) * ({} as u128)) >> 64) as u64", rd(a, W::W64), rd(b, W::W64))
            };
            wr(dst, W::W64, &e)
        }
        Op::Div { w, dst, a, b, signed } => {
            let e = match (signed, w) {
                (false, _) => "if b == 0 { 0 } else { a / b }".to_string(),
                (true, W::W64) => "if b == 0 { 0 } else { (a as i64).wrapping_div(b as i64) as u64 }".to_string(),
                (true, W::W32) => "if b == 0 { 0 } else { (a as u32 as i32).wrapping_div(b as u32 as i32) as u32 as u64 }".to_string(),
            };
            format!("{{ let a = {}; let b = {}; {} }}", rd(a, w), rd(b, w), wr(dst, w, &e))
        }
        Op::CondSel { w, dst, a, b, cond, inc, inv } => {
            let other = match (inc, inv) {
                (false, false) => rd(b, w),
                (true, false) => format!("{}.wrapping_add(1)", rd(b, w)),
                (false, true) => format!("!{}", rd(b, w)),
                (true, true) => format!("{}.wrapping_neg()", rd(b, w)),
            };
            wr(dst, w, &format!("if rt::cond(c, {cond}) {{ {} }} else {{ {other} }}", rd(a, w)))
        }
        Op::CondCmp { w, a, b, nzcv, cond, add } => {
            let f = if add { "adds" } else { "subs" };
            format!(
                "if rt::cond(c, {cond}) {{ let a = {}; let b = {}; rt::{f}{}(c, a as _, b as _); }} else {{ rt::set_nzcv(c, {nzcv}); }}",
                rd(a, w),
                operand(b, w),
                w.bits()
            )
        }
        Op::Unary { op, w, dst, src } => {
            let v = rd(src, w);
            let e = match (op, w) {
                (Unary::Clz, W::W64) => format!("{v}.leading_zeros() as u64"),
                (Unary::Clz, W::W32) => format!("({v} as u32).leading_zeros() as u64"),
                (Unary::Rbit, W::W64) => format!("{v}.reverse_bits()"),
                (Unary::Rbit, W::W32) => format!("({v} as u32).reverse_bits() as u64"),
                (Unary::Rev, W::W64) => format!("{v}.swap_bytes()"),
                (Unary::Rev, W::W32) => format!("({v} as u32).swap_bytes() as u64"),
                (Unary::Rev16, _) => format!("rt::rev16({}, {v})", w.bits()),
                (Unary::Rev32, _) => format!("rt::rev32({v})"),
            };
            wr(dst, w, &e)
        }
        Op::Const { dst, value } => wr(dst, W::W64, &format!("{value:#x}u64")),
        Op::Load { dst, w, size, signed, addr: a } => {
            let (at, wb) = addr(&a);
            let v = loaded(&format!("rt::ld{}(c, a)", 8 << size), w, size, signed);
            format!("{{ let a = {at}; let v = {v}; {wb} {} }}", wr(dst, w, "v"))
        }
        Op::Store { src, size, addr: a } => {
            let (at, wb) = addr(&a);
            format!("{{ let a = {at}; rt::st{}(c, a, {}); {wb} }}", 8 << size, rd(src, W::W64))
        }
        Op::LoadPair { a: r1, b: r2, w, size, signed, addr: a } => {
            let (at, wb) = addr(&a);
            let ld = |off: u32| loaded(&format!("rt::ld{}(c, a.wrapping_add({off}))", 8 << size), w, size, signed);
            format!("{{ let a = {at}; let v1 = {}; let v2 = {}; {wb} {} {} }}", ld(0), ld(1 << size), wr(r1, w, "v1"), wr(r2, w, "v2"))
        }
        Op::StorePair { a: r1, b: r2, size, addr: a } => {
            let (at, wb) = addr(&a);
            let n = 8 << size;
            format!(
                "{{ let a = {at}; let (v1, v2) = ({}, {}); rt::st{n}(c, a, v1); rt::st{n}(c, a.wrapping_add({}), v2); {wb} }}",
                rd(r1, W::W64),
                rd(r2, W::W64),
                1 << size
            )
        }
        Op::ReadThread { dst, ro } => wr(dst, W::W64, if ro { "c.tpidrro" } else { "c.tpidr" }),
        Op::Nop => String::new(),
        _ => unreachable!("control flow is emitted by the block"),
    }
}

/// Emits one function into `out`.
pub fn function(out: &mut String, input: &Input) {
    let f = input.function;
    let blocks = &input.cfg.blocks;
    let index = |a: u64| blocks.binary_search_by_key(&a, |b| b.start).ok();
    let call = |target: u64| -> String {
        match (input.callee)(target) {
            Callee::Direct(t) => format!("unsafe {{ {}(c) }}", name(t)),
            Callee::Dispatch(t) => format!("unsafe {{ rt::gyroship_call(c, {t:#x}) }}"),
        }
    };
    // A jump to `t`: to a block of this function, or a tail call.
    let goto = |t: u64| -> String {
        match index(t) {
            Some(i) => format!("b = {i};"),
            None => format!("{}; return;", call(t)),
        }
    };
    let _ = writeln!(out, "#[unsafe(no_mangle)]\npub unsafe extern \"C\" fn {}(c: *mut Cpu) {{", name(f.start));
    out.push_str("    let c = unsafe { &mut *c };\n");
    // One block that never jumps back into itself needs no state machine.
    let single = blocks.len() == 1
        && input.cfg.tables.is_empty()
        && (blocks[0].start..blocks[0].start + blocks[0].len).step_by(4).all(|pc| match crate::lift::lift((input.word)(pc), pc) {
            Op::Jump(t) | Op::Branch(_, t) => index(t).is_none(),
            _ => true,
        });
    if !single {
        let _ = writeln!(out, "    let mut b: u32 = {};\n    loop {{\n        match b {{", index(f.start).unwrap_or(0));
    }
    for (bi, block) in blocks.iter().enumerate() {
        if !single {
            let _ = writeln!(out, "            {bi} => {{");
        }
        let end = block.start + block.len;
        let mut ended = false;
        for pc in (block.start..end).step_by(4) {
            let word = (input.word)(pc);
            let op = crate::lift::lift(word, pc);
            let s = match op {
                Op::Jump(t) => {
                    ended = true;
                    goto(t)
                }
                Op::Branch(cnd, t) => {
                    ended = true;
                    format!("if {} {{ {} }} else {{ {} }}", condition(cnd), goto(t), goto(pc + 4))
                }
                Op::Call(t) => format!("c.x[30] = {:#x}; {};", pc + 4, call(t)),
                Op::CallIndirect(r) => format!("let t = {}; c.x[30] = {:#x}; unsafe {{ rt::gyroship_call(c, t) }};", rd(r, W::W64), pc + 4),
                Op::JumpIndirect(r) => {
                    ended = true;
                    match input.cfg.tables.iter().find(|t| t.branch == pc) {
                        Some(table) => {
                            let mut targets: Vec<u64> = table.targets.clone();
                            targets.sort_unstable();
                            targets.dedup();
                            let arms: String = targets.iter().filter_map(|&t| Some(format!("{t:#x} => {}, ", index(t)?))).collect();
                            format!("b = match {} {{ {arms}t => {{ rt::gyroship_miss(c, t); return; }} }};", rd(r, W::W64))
                        }
                        None => format!("let t = {}; unsafe {{ rt::gyroship_call(c, t) }}; return;", rd(r, W::W64)),
                    }
                }
                Op::Return => {
                    ended = true;
                    "return;".to_string()
                }
                Op::Trap(imm) => {
                    ended = true;
                    format!("rt::gyroship_trap(c, {pc:#x}, {imm}); return;")
                }
                Op::Svc(imm) => format!("rt::gyroship_svc(c, {pc:#x}, {imm});"),
                Op::Unhandled => format!("rt::gyroship_unhandled(c, {pc:#x}, {word:#x});"),
                ref op => data(op),
            };
            if !s.is_empty() {
                let _ = writeln!(out, "                {s}");
            }
        }
        if !ended {
            let _ = writeln!(out, "                {}", goto(end));
        }
        if single {
            out.push_str("}\n");
        } else {
            out.push_str("            }\n");
        }
    }
    if !single {
        out.push_str("            _ => unreachable!(),\n        }\n    }\n}\n");
    }
}
