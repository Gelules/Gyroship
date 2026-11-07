//! Backend A: Rust source. Each guest function becomes
//! `pub unsafe extern "C" fn f_<addr>(cpu: *mut Cpu, resume: u64)`; its blocks are the
//! arms of a `loop { match b { .. } }`. `RET` writes `cpu.pc`, but calls do
//! not check it and `resume` must be 0: this backend is the oracle for code
//! without non-local jumps (D15, D20).

use crate::ir::{Addr, Alu, Cond, FBin, FUn, Op, Operand, Reg, Select, Shift, Unary, Writeback, W};
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

/// `& mask` keeping the low 1 << size bytes of a u128 (nothing for 16 bytes).
fn low_mask(size: u8) -> String {
    if size >= 4 { String::new() } else { format!(" & {:#x}u128", (1u128 << (8u32 << size)) - 1) }
}

fn vload(size: u8, at: &str) -> String {
    match size {
        4 => format!("rt::ld128(c, {at})"),
        s => format!("rt::ld{}(c, {at}) as u128", 8u32 << s),
    }
}

fn vstore(size: u8, at: &str, src: u8) -> String {
    match size {
        4 => format!("rt::st128(c, {at}, c.vr[{src}])"),
        s => format!("rt::st{}(c, {at}, c.vr[{src}] as u64)", 8u32 << s),
    }
}

fn fread(n: u8, f64: bool) -> String {
    if f64 { format!("f64::from_bits(c.vr[{n}] as u64)") } else { format!("f32::from_bits(c.vr[{n}] as u32)") }
}

/// Writes a scalar float; the rest of the register becomes zero.
fn fwrite(n: u8, v: &str) -> String {
    format!("c.vr[{n}] = {v}.to_bits() as u128;")
}

/// A data operation as one statement block.
fn data(op: &Op, pc: u64, word: u32) -> String {
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
        Op::LoadV { dst, size, addr: a } => {
            let (at, wb) = addr(&a);
            format!("{{ let a = {at}; let v = {}; {wb} c.vr[{dst}] = v; }}", vload(size, "a"))
        }
        Op::StoreV { src, size, addr: a } => {
            let (at, wb) = addr(&a);
            format!("{{ let a = {at}; {}; {wb} }}", vstore(size, "a", src))
        }
        Op::LoadPairV { a: r1, b: r2, size, addr: a } => {
            let (at, wb) = addr(&a);
            let second = format!("a.wrapping_add({})", 1u32 << size);
            format!("{{ let a = {at}; let v1 = {}; let v2 = {}; {wb} c.vr[{r1}] = v1; c.vr[{r2}] = v2; }}", vload(size, "a"), vload(size, &second))
        }
        Op::StorePairV { a: r1, b: r2, size, addr: a } => {
            let (at, wb) = addr(&a);
            let second = format!("a.wrapping_add({})", 1u32 << size);
            format!("{{ let a = {at}; {}; {}; {wb} }}", vstore(size, "a", r1), vstore(size, &second, r2))
        }
        Op::VConst { dst, value } => format!("c.vr[{dst}] = {value:#x}u128;"),
        Op::VMov { dst, src, size } => format!("c.vr[{dst}] = c.vr[{src}]{};", low_mask(size)),
        Op::ToV { dst, src, size } => format!("c.vr[{dst}] = {} as u128;", rd(src, if size == 3 { W::W64 } else { W::W32 })),
        Op::FromV { dst, src, size, lane } => {
            let bits = 8u32 << size;
            let w = if size == 3 { W::W64 } else { W::W32 };
            wr(dst, w, &format!("((c.vr[{src}] >> {}){}) as u64", u32::from(lane) * bits, low_mask(size)))
        }
        Op::VLogic { op, q, dst, a, b } => {
            let e = match op {
                Alu::And => "a & b",
                Alu::Orr => "a | b",
                Alu::Eor => "a ^ b",
                Alu::Bic => "a & !b",
                _ => "a | !b",
            };
            format!("{{ let (a, b) = (c.vr[{a}], c.vr[{b}]); c.vr[{dst}] = ({e}){}; }}", low_mask(if q { 4 } else { 3 }))
        }
        Op::VSelect { kind, q, dst, a, b } => {
            let e = match kind {
                Select::Bsl => "(d & a) | (!d & b)",
                Select::Bit => "(d & !b) | (a & b)",
                Select::Bif => "(d & b) | (a & !b)",
            };
            format!("{{ let (d, a, b) = (c.vr[{dst}], c.vr[{a}], c.vr[{b}]); c.vr[{dst}] = ({e}){}; }}", low_mask(if q { 4 } else { 3 }))
        }
        Op::FBin { op, f64, dst, a, b } => {
            let t = if f64 { "64" } else { "32" };
            let e = match op {
                FBin::Add => "a + b".to_string(),
                FBin::Sub => "a - b".to_string(),
                FBin::Mul => "a * b".to_string(),
                FBin::Div => "a / b".to_string(),
                FBin::Max => format!("rt::fmax{t}(a, b)"),
                FBin::Min => format!("rt::fmin{t}(a, b)"),
                FBin::MaxNm => format!("if a.is_nan() {{ b }} else if b.is_nan() {{ a }} else {{ rt::fmax{t}(a, b) }}"),
                FBin::MinNm => format!("if a.is_nan() {{ b }} else if b.is_nan() {{ a }} else {{ rt::fmin{t}(a, b) }}"),
                FBin::Nmul => "-(a * b)".to_string(),
                FBin::Abd => "(a - b).abs()".to_string(),
            };
            // Arm's NaNs are not the host's: as in the Cranelift backend, a NaN result takes the interpreter.
            let store = if matches!(op, FBin::Add | FBin::Sub | FBin::Mul | FBin::Div) {
                format!("if r.is_nan() {{ rt::gyroship_unhandled(c, {pc:#x}, {word:#x}); }} else {{ {} }}", fwrite(dst, "r"))
            } else {
                fwrite(dst, "r")
            };
            format!("{{ let (a, b) = ({}, {}); let r: f{t} = {e}; {store} }}", fread(a, f64), fread(b, f64))
        }
        Op::FUn { op, f64, dst, src } => {
            let e = match op {
                FUn::Neg => "-a",
                FUn::Abs => "a.abs()",
                FUn::Sqrt => "a.sqrt()",
                FUn::Floor => "a.floor()",
                FUn::Ceil => "a.ceil()",
                FUn::Trunc => "a.trunc()",
                FUn::Nearest => "a.round_ties_even()",
                FUn::Round => "a.round()",
            };
            format!("{{ let a = {}; let r = {e}; {} }}", fread(src, f64), fwrite(dst, "r"))
        }
        Op::FMulAdd { f64, dst, a, b, acc, neg_acc, neg_product } => {
            let neg = |x: bool| if x { "-" } else { "" };
            format!(
                "{{ let r = ({}{}).mul_add({}, {}{}); {} }}",
                neg(neg_product),
                fread(a, f64),
                fread(b, f64),
                neg(neg_acc),
                fread(acc, f64),
                fwrite(dst, "r")
            )
        }
        Op::FCmp { f64, a, b } => {
            let y = b.map_or_else(|| "0.0".to_string(), |b| fread(b, f64));
            format!("rt::fcmp(c, {} as f64, {y} as f64);", fread(a, f64))
        }
        Op::FCondCmp { f64, a, b, nzcv, cond } => {
            format!("if rt::cond(c, {cond}) {{ rt::fcmp(c, {} as f64, {} as f64); }} else {{ rt::set_nzcv(c, {nzcv}); }}", fread(a, f64), fread(b, f64))
        }
        Op::FCondSel { f64, dst, a, b, cond } => {
            let m = low_mask(if f64 { 3 } else { 2 });
            format!("c.vr[{dst}] = if rt::cond(c, {cond}) {{ c.vr[{a}]{m} }} else {{ c.vr[{b}]{m} }};")
        }
        Op::FCvt { dst, src, to_f64 } => {
            let e = if to_f64 { format!("{} as f64", fread(src, false)) } else { format!("{} as f32", fread(src, true)) };
            format!("{{ let r = {e}; {} }}", fwrite(dst, "r"))
        }
        Op::IntToF { dst, src, w, f64, signed } => {
            let t = if f64 { "f64" } else { "f32" };
            let v = match (signed, w) {
                (true, W::W32) => format!("({} as u32 as i32)", rd(src, W::W32)),
                (true, W::W64) => format!("({} as i64)", rd(src, W::W64)),
                (false, W::W32) => format!("({} as u32)", rd(src, W::W32)),
                (false, W::W64) => rd(src, W::W64),
            };
            format!("{{ let r = {v} as {t}; {} }}", fwrite(dst, "r"))
        }
        Op::FToInt { dst, w, src, f64, signed } => {
            let t = match (signed, w) {
                (true, W::W32) => "i32",
                (true, W::W64) => "i64",
                (false, W::W32) => "u32",
                (false, W::W64) => "u64",
            };
            wr(dst, w, &format!("({} as {t}) as u64", fread(src, f64)))
        }
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
            Callee::Direct(t) => format!("unsafe {{ {}(c, 0) }}", name(t)),
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
    let _ = writeln!(out, "#[unsafe(no_mangle)]\npub unsafe extern \"C\" fn {}(c: *mut Cpu, resume: u64) {{", name(f.start));
    out.push_str("    let c = unsafe { &mut *c };\n    if resume != 0 {\n        rt::gyroship_miss(c, resume);\n        return;\n    }\n");
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
            if (input.probe)(pc) {
                let _ = writeln!(out, "                unsafe {{ rt::gyroship_probe(c, {pc:#x}) }};");
            }
            let s = match op {
                Op::Jump(t) if t == pc => {
                    ended = true;
                    format!("rt::gyroship_trap(c, {pc:#x}, rt::TRAP_SELF_LOOP); return;")
                }
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
                    "c.pc = c.x[30]; return;".to_string()
                }
                Op::Trap(imm) => {
                    ended = true;
                    format!("rt::gyroship_trap(c, {pc:#x}, {imm}); return;")
                }
                Op::Svc(imm) => format!("unsafe {{ rt::gyroship_svc(c, {pc:#x}, {imm}) }};"),
                Op::Counter(r) => {
                    let n = match r {
                        Reg::X(n) => u32::from(n),
                        Reg::Zr | Reg::Sp => 31,
                    };
                    format!("unsafe {{ rt::gyroship_counter(c, {pc:#x}, {n}) }};")
                }
                Op::Yield(kind) => format!("rt::gyroship_yield(c, {pc:#x}, {kind});"),
                // ponytail: this experimental backend leaves the vector-lane forms and CRC to the interpreter.
                Op::Unhandled
                | Op::Crc { .. }
                | Op::LoadLane { .. }
                | Op::ToLane { .. }
                | Op::VShuffle { .. }
                | Op::VFp { .. }
                | Op::VFUn { .. }
                | Op::VFCmp { .. }
                | Op::VInt { .. }
                | Op::VIntToF { .. } => format!("rt::gyroship_unhandled(c, {pc:#x}, {word:#x});"),
                ref op => data(op, pc, word),
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
