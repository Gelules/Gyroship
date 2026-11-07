//! Differential tests of the vector-lane forms (and CRC32, and the scalar FP
//! forms with Arm's NaN rules) the Cranelift backend translates: each
//! instruction, compiled alone into JIT code, must leave the context exactly as
//! the interpreter does, on random registers full of special values (NaNs with
//! payloads, infinities, signed zeros, subnormals). Known answers worked out
//! from the Arm ARM pseudocode pin both paths down. Encodings are built from
//! the Arm ARM fields; no game data is involved.

use cranelift_jit::{JITBuilder, JITModule};
use gyroship_analysis::cfg;
use gyroship_analysis::discover::Function;
use gyroship_recomp::ir::Op;
use gyroship_recomp::{Callee, Input, clif, lift::lift};
use gyroship_runtime::mem::{Block, Memory, perm, state};
use gyroship_runtime::{self as rt, Cpu, interp};
use std::cell::Cell;
use std::collections::HashMap;
use std::sync::OnceLock;

const RET: u32 = 0xD65F_03C0;
/// The page lane loads read.
const DATA: u64 = 0x10_0000;
/// Where the one-instruction functions start (8 bytes each).
const CODE: u64 = 0x1000;
/// Random contexts per instruction.
const CASES: usize = 1000;

// ---- encoders ----

fn q(q: bool) -> u32 {
    u32::from(q) << 30
}

/// Three-same, two-register misc and permute forms: Q, size and three registers.
fn rrr(base: u32, qv: bool, size: u32, d: u32, n: u32, m: u32) -> u32 {
    base | q(qv) | size << 22 | m << 16 | n << 5 | d
}

const FADD: u32 = 0x0E20_D400;
const FSUB: u32 = 0x0EA0_D400;
const FMUL: u32 = 0x2E20_DC00;
const FMLA: u32 = 0x0E20_CC00;
const FMLS: u32 = 0x0EA0_CC00;
const FADDP: u32 = 0x2E20_D400;
const FCMEQ: u32 = 0x0E20_E400;
const FCMGE: u32 = 0x2E20_E400;
const FCMGT: u32 = 0x2EA0_E400;
/// Two-register misc: compares with zero, FABS/FNEG, SCVTF/UCVTF.
const FP_UNARY: [u32; 9] = [0x0EA0_C800, 0x0EA0_D800, 0x0EA0_E800, 0x2EA0_C800, 0x2EA0_D800, 0x0EA0_F800, 0x2EA0_F800, 0x0E21_D800, 0x2E21_D800];
/// By element: FMLA, FMLS, FMUL.
const FP_ELEM: [u32; 3] = [0x0F80_1000, 0x0F80_5000, 0x0F80_9000];
/// Scalar FADD, FSUB, FMUL, FDIV (ftype in the size field).
const FP_SCALAR: [u32; 4] = [0x1E20_2800, 0x1E20_3800, 0x1E20_0800, 0x1E20_1800];
/// ADD, SUB, ADDP.
const INT_ANY: [u32; 3] = [0x0E20_8400, 0x2E20_8400, 0x0E20_BC00];
/// SMAX, SMIN, UMAX, UMIN, SMAXP, SMINP, UMAXP, UMINP.
const INT_MINMAX: [u32; 8] = [0x0E20_6400, 0x0E20_6C00, 0x2E20_6400, 0x2E20_6C00, 0x0E20_A400, 0x0E20_AC00, 0x2E20_A400, 0x2E20_AC00];
/// UZP1, TRN1, ZIP1, UZP2, TRN2, ZIP2.
const PERMUTE: [u32; 6] = [0x0E00_1800, 0x0E00_2800, 0x0E00_3800, 0x0E00_5800, 0x0E00_6800, 0x0E00_7800];
/// REV64, REV32, REV16, with the largest element size each allows.
const REV: [(u32, u32); 3] = [(0x0E20_0800, 2), (0x2E20_0800, 1), (0x0E20_1800, 0)];

/// By element (S: index H:L, D: index H), register m of 5 bits.
fn by_elem(base: u32, qv: bool, sz: u32, d: u32, n: u32, m: u32, idx: u32) -> u32 {
    let (h, l) = if sz == 1 { (idx, 0) } else { (idx >> 1, idx & 1) };
    base | q(qv) | sz << 22 | l << 21 | m << 16 | h << 11 | n << 5 | d
}

/// imm5 of the copy forms: element size and index.
fn imm5(size: u32, idx: u32) -> u32 {
    (idx << (size + 1) | 1 << size) << 16
}

fn dup_general(qv: bool, size: u32, d: u32, n: u32) -> u32 {
    0x0E00_0C00 | q(qv) | imm5(size, 0) | n << 5 | d
}

fn dup_element(qv: bool, size: u32, idx: u32, d: u32, n: u32) -> u32 {
    0x0E00_0400 | q(qv) | imm5(size, idx) | n << 5 | d
}

/// DUP (scalar): mov <V>d, vn.<T>[idx].
fn dup_scalar(size: u32, idx: u32, d: u32, n: u32) -> u32 {
    0x5E00_0400 | imm5(size, idx) | n << 5 | d
}

fn ins_general(size: u32, idx: u32, d: u32, n: u32) -> u32 {
    0x4E00_1C00 | imm5(size, idx) | n << 5 | d
}

fn ins_element(size: u32, idx: u32, src: u32, d: u32, n: u32) -> u32 {
    0x6E00_0400 | imm5(size, idx) | (src << size) << 11 | n << 5 | d
}

fn ext(qv: bool, imm: u32, d: u32, n: u32, m: u32) -> u32 {
    0x2E00_0000 | q(qv) | m << 16 | imm << 11 | n << 5 | d
}

/// LD1 of one lane; `post` adds the element size to xn.
fn ld1_lane(size: u32, idx: u32, t: u32, n: u32, post: bool) -> u32 {
    // The index spreads over Q:S:size, from the element size up.
    let (opcode, qs_size) = match size {
        0 => (0b000, idx),
        1 => (0b010, idx << 1),
        2 => (0b100, idx << 2),
        _ => (0b100, idx << 3 | 1),
    };
    let (qv, s, sz) = (qs_size >> 3, qs_size >> 2 & 1, qs_size & 3);
    let post = if post { 1 << 23 | 31 << 16 } else { 0 };
    0x0D40_0000 | qv << 30 | post | opcode << 13 | s << 12 | sz << 10 | n << 5 | t
}

fn ld1r(qv: bool, size: u32, t: u32, n: u32, post: bool) -> u32 {
    let post = if post { 1 << 23 | 31 << 16 } else { 0 };
    0x0D40_C000 | q(qv) | post | size << 10 | n << 5 | t
}

/// CRC32B/H/W/X, or CRC32C* (`castagnoli`): wd, wn, wm (xm for X).
fn crc(castagnoli: bool, sz: u32, d: u32, n: u32, m: u32) -> u32 {
    0x1AC0_4000 | u32::from(sz == 3) << 31 | m << 16 | u32::from(castagnoli) << 12 | sz << 10 | n << 5 | d
}

/// One instruction to test, with the register its address comes from.
struct Case {
    word: u32,
    base: Option<usize>,
}

fn cases() -> Vec<Case> {
    let mut c = Vec::new();
    let mut add = |word: u32, base: Option<u32>| c.push(Case { word, base: base.map(|b| b as usize) });
    let fp_regs = [(0, 1, 2), (8, 8, 10), (8, 8, 8), (31, 30, 29)];
    for (qv, sz) in [(false, 0), (true, 0), (true, 1)] {
        for (d, n, m) in fp_regs {
            for base in [FADD, FSUB, FMUL, FMLA, FMLS, FADDP, FCMEQ, FCMGE, FCMGT] {
                add(rrr(base, qv, sz, d, n, m), None);
            }
            for base in FP_UNARY {
                add(rrr(base, qv, sz, d, n, 0), None);
            }
            for base in FP_ELEM {
                for idx in 0..4 >> sz {
                    add(by_elem(base, qv, sz, d, n, m, idx), None);
                }
            }
        }
    }
    for ftype in 0..2 {
        for (d, n, m) in fp_regs {
            for base in FP_SCALAR {
                add(rrr(base, false, ftype, d, n, m), None);
            }
        }
    }
    for qv in [false, true] {
        let width = if qv { 16 } else { 8 };
        for size in 0..4 {
            if size == 3 && !qv {
                continue; // .1d: reserved for these forms
            }
            for (d, n, m) in [(0, 1, 2), (8, 8, 8), (8, 8, 10), (2, 1, 2)] {
                for base in INT_ANY.into_iter().chain(if size < 3 { INT_MINMAX.to_vec() } else { vec![] }).chain(PERMUTE) {
                    add(rrr(base, qv, size, d, n, m), None);
                }
                for (base, largest) in REV {
                    if size <= largest {
                        add(rrr(base, qv, size, d, n, 0), None);
                    }
                }
            }
            add(dup_general(qv, size, 3, 8), None);
            add(dup_general(qv, size, 4, 31), None);
            for idx in 0..16 >> size {
                add(dup_element(qv, size, idx, 2, 7), None);
                add(dup_element(qv, size, idx, 7, 7), None);
            }
        }
        for size in 0..4 {
            for (t, n) in [(8, 7), (0, 31)] {
                for post in [false, true] {
                    add(ld1r(qv, size, t, n, post), Some(n));
                }
            }
        }
        for imm in 0..width {
            add(ext(qv, imm, 10, 8, 8), None);
            add(ext(qv, imm, 1, 2, 3), None);
            add(ext(qv, imm, 8, 8, 10), None);
        }
    }
    for size in 0..4 {
        let lanes = 16 >> size;
        for idx in 0..lanes {
            add(ins_general(size, idx, 3, 8), None);
            add(ins_general(size, idx, 30, 31), None);
            add(ins_element(size, idx, (idx * 7 + 1) % lanes, 6, 4), None);
            add(ins_element(size, idx, idx, 9, 9), None);
            add(dup_scalar(size, idx, 0, 1), None);
            for (t, n) in [(8, 7), (31, 31)] {
                for post in [false, true] {
                    add(ld1_lane(size, idx, t, n, post), Some(n));
                }
            }
        }
    }
    for castagnoli in [false, true] {
        for sz in 0..4 {
            for (d, n, m) in [(0, 1, 2), (5, 5, 5), (3, 31, 4), (31, 6, 31)] {
                add(crc(castagnoli, sz, d, n, m), None);
            }
        }
    }
    c
}

// ---- random contexts ----

/// splitmix64.
struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    fn below(&mut self, n: u64) -> u64 {
        self.next() % n
    }

    /// A float of `exp` exponent bits and `frac` fraction bits, often a special
    /// one; a NaN only with `nan` (random bits aside).
    fn float(&mut self, exp: u32, frac: u32, nan: bool) -> u64 {
        let inf = ((1u64 << exp) - 1) << frac;
        let quiet = 1u64 << (frac - 1);
        let f = self.next() & ((1 << frac) - 1);
        let sign = (self.next() & 1) << (exp + frac);
        let bias = (1u64 << (exp - 1)) - 1;
        sign | match self.below(if nan { 10 } else { 8 }) {
            0 => 0,
            1 => inf,
            2 => f.max(1),     // subnormal
            3 => bias << frac, // 1.0
            4 => inf - 1,      // the largest finite
            5 => self.next() & (inf | (inf - 1)),
            8 => inf | quiet | (f & (quiet - 1)),
            9 => inf | (f & (quiet - 1)).max(1), // signalling
            _ => (bias - 8 + self.below(16)) << frac | f, // near 1
        }
    }

    /// 16 bytes of single or double lanes, or of any bits.
    fn vector(&mut self, nan: bool) -> u128 {
        let mut half = || match self.below(20) {
            0..9 => self.float(8, 23, nan) | self.float(8, 23, nan) << 32,
            9..18 => self.float(11, 52, nan),
            _ => self.next(),
        };
        u128::from(half()) | u128::from(half()) << 64
    }
}

// ---- running ----

/// Every case compiled into one JIT module: word -> host function.
fn compiled() -> &'static HashMap<u32, rt::Guest> {
    static CODE_MAP: OnceLock<HashMap<u32, rt::Guest>> = OnceLock::new();
    CODE_MAP.get_or_init(|| {
        let words: Vec<u32> = cases().iter().map(|c| c.word).chain(known().into_iter().flat_map(|k| k.0)).collect();
        let mut jb = JITBuilder::with_isa(clif::isa(true).unwrap(), cranelift_module::default_libcall_names());
        // What the functions call (every rescue block refers to gyroship_call and gyroship_miss).
        let runtime: [(&str, *const u8); 4] = [
            ("gyroship_call", rt::gyroship_call as *const () as *const u8),
            ("gyroship_crc32", rt::gyroship_crc32 as *const () as *const u8),
            ("gyroship_miss", rt::gyroship_miss as *const () as *const u8),
            ("gyroship_unhandled", unhandled as *const () as *const u8),
        ];
        for (name, f) in runtime {
            jb.symbol(name, f);
        }
        let functions: Vec<Function> =
            (0..words.len()).map(|i| CODE + 8 * i as u64).map(|start| Function { start, len: 8, unwind: true, entries: vec![start] }).collect();
        let defined: Vec<(u64, u64)> = functions.iter().map(|f| (f.start, f.len)).collect();
        let mut obj = clif::Object::with_module(JITModule::new(jb), &defined).unwrap();
        let image: Vec<u8> = words.iter().flat_map(|w| [w.to_le_bytes(), RET.to_le_bytes()].concat()).collect();
        let word = |a: u64| {
            let at = (a - CODE) as usize;
            u32::from_le_bytes(image[at..at + 4].try_into().unwrap())
        };
        for f in &functions {
            let g = cfg::build(&image, CODE, f);
            obj.function(&Input { function: f, cfg: &g, word: &word, callee: &Callee::Dispatch, probe: &|_| false }).unwrap_or_else(|e| panic!("{:#x}: {e}", word(f.start)));
        }
        let (mut module, ids) = obj.into_parts();
        module.finalize_definitions().unwrap();
        let map = words
            .iter()
            .zip(&functions)
            // SAFETY: the JIT compiled this function with the `Guest` signature.
            .map(|(&w, f)| (w, unsafe { std::mem::transmute::<*const u8, rt::Guest>(module.get_finalized_function(ids[&f.start])) }))
            .collect();
        // The code must outlive every test of this process.
        std::mem::forget(module);
        map
    })
}

thread_local! {
    /// Instructions the JIT code of this thread handed to the interpreter.
    static RESCUED: Cell<u64> = const { Cell::new(0) };
}

/// The JIT code's `gyroship_unhandled`: counts, then runs the interpreter.
unsafe extern "C" fn unhandled(cpu: *mut Cpu, pc: u64, word: u32) {
    RESCUED.set(RESCUED.get() + 1);
    // SAFETY: the generated code passes its own context.
    unsafe { rt::gyroship_unhandled(cpu, pc, word) }
}

/// Runs `word` both ways from `cpu`; the first difference, if any.
fn differ(word: u32, cpu: &Cpu) -> Option<String> {
    let (mut jit, mut int) = (clone(cpu), clone(cpu));
    // SAFETY: the context's memory maps the data page every load reads.
    unsafe { compiled()[&word](&mut jit, 0) };
    interp::execute(&mut int, CODE, word).unwrap_or_else(|e| panic!("{word:08x} ({}): {e}", interp::disassemble(word)));
    let mut out = String::new();
    for i in 0..32 {
        if jit.vr[i] != int.vr[i] {
            out += &format!(" v{i}: jit {:032x} interp {:032x} (before {:032x})", jit.vr[i], int.vr[i], cpu.vr[i]);
        }
        if jit.x[i] != int.x[i] {
            out += &format!(" x{i}: jit {:#x} interp {:#x}", jit.x[i], int.x[i]);
        }
    }
    if (jit.n, jit.z, jit.c, jit.v) != (int.n, int.z, int.c, int.v) {
        out += " nzcv";
    }
    (!out.is_empty()).then_some(out)
}

fn clone(c: &Cpu) -> Cpu {
    let mut n = Cpu::new(c.mem);
    (n.x, n.vr, n.n, n.z, n.c, n.v) = (c.x, c.vr, c.n, c.z, c.c, c.v);
    n
}

fn data_page(r: &mut Rng) -> Memory {
    let mut m = Memory::new().unwrap();
    m.map(DATA, 0x1000, Block::new(state::NORMAL, perm::RW)).unwrap();
    let bytes: Vec<u8> = (0..0x100).flat_map(|_| r.vector(true).to_le_bytes()).collect();
    m.write(DATA, &bytes).unwrap();
    m
}

#[test]
fn vector_lane_forms_match_the_interpreter() {
    let mut r = Rng(0x5EED);
    let m = data_page(&mut r);
    let all = cases();
    let mut failures = Vec::new();
    for case in &all {
        let op = lift(case.word, CODE);
        assert_ne!(op, Op::Unhandled, "{:08x} ({}) is not translated", case.word, interp::disassemble(case.word));
        for _ in 0..CASES {
            let mut cpu = Cpu::new(m.base());
            // Half the contexts hold no NaN, so that the native code (not the NaN rescue) runs often.
            let nan = r.next() & 1 == 0;
            cpu.vr = std::array::from_fn(|_| r.vector(nan));
            cpu.x = std::array::from_fn(|_| r.next());
            if let Some(b) = case.base {
                cpu.x[b] = DATA + r.below(0xF00);
            }
            rt::set_nzcv(&mut cpu, r.next() as u8 & 15);
            if let Some(d) = differ(case.word, &cpu) {
                failures.push(format!("{:08x} {}:{d}", case.word, interp::disassemble(case.word)));
                break;
            }
        }
    }
    eprintln!("{} instructions x {CASES} random contexts", all.len());
    assert!(failures.is_empty(), "{} of {} instructions differ:\n{}", failures.len(), all.len(), failures.join("\n"));
}

// ---- known answers ----

fn s4(v: [u32; 4]) -> u128 {
    v.iter().enumerate().fold(0, |acc, (i, &x)| acc | u128::from(x) << (32 * i))
}

fn f4(v: [f32; 4]) -> u128 {
    s4(v.map(f32::to_bits))
}

fn d2(v: [u64; 2]) -> u128 {
    u128::from(v[0]) | u128::from(v[1]) << 64
}

fn f2(v: [f64; 2]) -> u128 {
    d2(v.map(f64::to_bits))
}

/// 16 bytes counting up from `first`.
fn bytes(first: u8) -> u128 {
    u128::from_le_bytes(std::array::from_fn(|i| first + i as u8))
}

/// Register numbers in [`known`]: v0..v31, then x0..x30 from `X`.
const X: usize = 32;
type Regs = Vec<(usize, u128)>;

/// Results worked out by hand from the Arm ARM pseudocode (FPCR.DN = 0): the
/// words, run in order on a context that is zero but for `before`, must leave
/// `after`. Loads read byte k of the data page as k.
fn known() -> Vec<(Vec<u32>, Regs, Regs)> {
    let (one, inf, dnan) = (0x3F80_0000, 0x7F80_0000, 0x7FC0_0000);
    let (one_d, inf_d, dnan_d) = (0x3FF0_0000_0000_0000, 0x7FF0_0000_0000_0000, 0x7FF8_0000_0000_0000);
    let x = |n: usize, v: u64| (X + n, u128::from(v));
    let crc32 = |castagnoli| vec![crc(castagnoli, 3, 0, 0, 1), crc(castagnoli, 0, 0, 0, 2)];
    let digits = vec![x(0, 0xFFFF_FFFF), x(1, u64::from_le_bytes(*b"12345678")), x(2, u64::from(b'9'))];
    vec![
        // FPProcessNaNs: the first signalling NaN, quieted, before the first quiet one; inf - inf is the positive default NaN.
        (
            vec![rrr(FADD, true, 0, 0, 1, 2)],
            vec![(1, s4([0x7FC0_0001, one, inf, 0x7F80_0001])), (2, s4([0xFF80_0002, 0xFFC0_0003, inf | 1 << 31, 0x7FC0_0004]))],
            vec![(0, s4([0xFFC0_0002, 0xFFC0_0003, dnan, 0x7FC0_0001]))],
        ),
        (
            vec![rrr(FSUB, true, 0, 0, 1, 2)],
            vec![(1, s4([0x7FC0_0011, inf, one, 0xFF80_0012])), (2, s4([0x7F80_0013, inf, 0xFFC0_0014, 0x7FC0_0015]))],
            vec![(0, s4([0x7FC0_0013, dnan, 0xFFC0_0014, 0xFFC0_0012]))],
        ),
        // fmul v0.4s, v1.4s, v2.s[2]: every lane times -inf.
        (
            vec![by_elem(FP_ELEM[2], true, 0, 0, 1, 2, 2)],
            vec![(1, s4([0, 0xFFC0_0017, 0x7F80_0018, 0x4000_0000])), (2, s4([0x7F80_0019, one, inf | 1 << 31, one]))],
            vec![(0, s4([dnan, 0xFFC0_0017, 0x7FC0_0018, inf | 1 << 31]))],
        ),
        // FADDP adds the even lane of a:b to the odd one, in that order.
        (
            vec![rrr(FADDP, true, 0, 0, 1, 2)],
            vec![(1, s4([0x7FC0_0021, 0x7F80_0022, 0xFFC0_0023, 0x7FC0_0024])), (2, s4([inf, inf | 1 << 31, one, 0x7F80_0025]))],
            vec![(0, s4([0x7FC0_0022, 0xFFC0_0023, dnan, 0x7FC0_0025]))],
        ),
        // faddp s0, v1.2s (scalar): the rest of v0 is cleared.
        (vec![0x7E30_D820], vec![(0, u128::MAX), (1, s4([0x7FC0_0031, 0x7F80_0032, one, one]))], vec![(0, 0x7FC0_0032)]),
        // FPMulAdd: the addend's NaN comes first; a quiet NaN addend with inf * 0 gives the default NaN.
        (
            vec![rrr(FMLA, true, 0, 0, 1, 2)],
            vec![(0, s4([0x7FC0_0005, 0xFFC0_0006, one, 0x7F80_0007])), (1, s4([inf, one, inf, 0x7FC0_0008])), (2, s4([0, 0x7FC0_0009, 0, one]))],
            vec![(0, s4([dnan, 0xFFC0_0006, dnan, 0x7FC0_0007]))],
        ),
        // FMLS negates the first factor (a NaN too) before FPMulAdd.
        (
            vec![rrr(FMLS, true, 0, 0, 1, 2)],
            vec![(0, s4([one, one, one, 0xFF80_000B])), (1, s4([0x7FC0_000A, 0x7F80_000C, one, one])), (2, s4([one, 0x7FC0_000D, one, one]))],
            vec![(0, s4([0xFFC0_000A, 0xFFC0_000C, 0, 0xFFC0_000B]))],
        ),
        // Doubles.
        (
            vec![rrr(FADD, true, 1, 0, 1, 2)],
            vec![(1, d2([0x7FF8_0000_0000_0041, inf_d | 1 << 63])), (2, d2([0x7FF0_0000_0000_0042, inf_d]))],
            vec![(0, d2([0x7FF8_0000_0000_0042, dnan_d]))],
        ),
        (
            vec![rrr(FMLA, true, 1, 0, 1, 2)],
            vec![(0, d2([0x7FF8_0000_0000_0051, one_d])), (1, d2([inf_d, 0x7FF0_0000_0000_0052])), (2, d2([0, 0xFFF8_0000_0000_0053]))],
            vec![(0, d2([dnan_d, 0x7FF8_0000_0000_0052]))],
        ),
        // Scalars (fsub s0, s1, s2 and fdiv d0, d1, d2) clear the rest of the register.
        (
            vec![rrr(FP_SCALAR[1], false, 0, 0, 1, 2)],
            vec![(0, u128::MAX), (1, s4([0x7FC0_0001, one, one, one])), (2, s4([0x7F80_0002, 0, 0, 0]))],
            vec![(0, 0x7FC0_0002)],
        ),
        (vec![rrr(FP_SCALAR[3], false, 1, 0, 1, 2)], vec![(0, u128::MAX)], vec![(0, u128::from(dnan_d))]),
        // fmla v8.4s, v27.4s, v19.s[3]: Rm = M:Rm, index H:L.
        (vec![0x4FB3_1B68], vec![(8, f4([1.0; 4])), (27, f4([1.0, 2.0, 3.0, 4.0])), (19, f4([10.0, 20.0, 30.0, 0.5]))], vec![(8, f4([1.5, 2.0, 2.5, 3.0]))]),
        // fmls v0.2d, v1.2d, v2.d[1]: index H.
        (vec![0x4FC2_5820], vec![(0, f2([10.0, 20.0])), (1, f2([1.0, 2.0])), (2, f2([3.0, 0.5]))], vec![(0, f2([9.5, 19.0]))]),
        // ld1 {v0.h}[5], [x1], #2: index Q:S:size<1>; the other lanes are kept.
        (
            vec![0x4DDF_4820],
            vec![(0, u128::MAX), x(1, DATA + 0x10)],
            vec![(0, !(0xFFFF << 80) | 0x1110 << 80), x(1, DATA + 0x12)],
        ),
        // ld1 {v0.s}[3], [x2]: index Q:S, no writeback.
        (vec![0x4D40_9040], vec![(0, s4([1, 2, 3, 4])), x(2, DATA + 0x40)], vec![(0, s4([1, 2, 3, 0x4342_4140])), x(2, DATA + 0x40)]),
        // ld1r {v8.2s}, [x7], #4: the upper half is cleared.
        (vec![0x0DDF_C8E8], vec![(8, u128::MAX), x(7, DATA + 0x20)], vec![(8, 0x2322_2120_2322_2120), x(7, DATA + 0x24)]),
        // mov v0.b[13], v1.b[6]: imm5 and imm4.
        (vec![0x6E1B_3420], vec![(0, bytes(0x80)), (1, bytes(0x40))], vec![(0, bytes(0x80) & !(0xFF << 104) | 0x46 << 104)]),
        // ext v1.16b, v2.16b, v3.16b, #13
        (vec![0x6E03_6841], vec![(2, bytes(0)), (3, bytes(16))], vec![(1, bytes(13))]),
        // uminp v8.2s, v8.2s, v10.2s: unsigned, pairs of v8 then of v10.
        (vec![0x2EAA_AD08], vec![(8, s4([5, 3, 1, 1])), (10, s4([0xFFFF_FFFF, 7, 1, 1]))], vec![(8, s4([3, 7, 0, 0]))]),
        // smaxp and sminp v0.4s, v1.4s, v2.4s: signed.
        (
            vec![0x4EA2_A420],
            vec![(1, s4([0xFFFF_FFFF, 2, 0x8000_0000, 0])), (2, s4([5, 0xFFFF_FFFE, 0xFFFF_FFF0, 0xFFFF_FFF1]))],
            vec![(0, s4([2, 0, 5, 0xFFFF_FFF1]))],
        ),
        (
            vec![0x4EA2_AC20],
            vec![(1, s4([0xFFFF_FFFF, 2, 0x8000_0000, 0])), (2, s4([5, 0xFFFF_FFFE, 0xFFFF_FFF0, 0xFFFF_FFF1]))],
            vec![(0, s4([0xFFFF_FFFF, 0x8000_0000, 0xFFFF_FFFE, 0xFFFF_FFF0]))],
        ),
        // uzp1 v0.4s, v1.4s, v2.4s
        (vec![0x4E82_1820], vec![(1, s4([1, 2, 3, 4])), (2, s4([5, 6, 7, 8]))], vec![(0, s4([1, 3, 5, 7]))]),
        // The check values of CRC-32 and CRC-32C: "123456789", with the initial and final inversions.
        (crc32(false), digits.clone(), vec![x(0, u64::from(!0xCBF4_3926u32))]),
        (crc32(true), digits, vec![x(0, u64::from(!0xE306_9283u32))]),
    ]
}

#[test]
fn known_answers_of_the_arm_arm() {
    let mut m = Memory::new().unwrap();
    m.map(DATA, 0x1000, Block::new(state::NORMAL, perm::RW)).unwrap();
    m.write(DATA, &std::array::from_fn::<u8, 256, _>(|k| k as u8)).unwrap();
    for (words, before, after) in known() {
        let mut cpu = Cpu::new(m.base());
        for (r, v) in before {
            if r < X { cpu.vr[r] = v } else { cpu.x[r - X] = v as u64 }
        }
        for &w in &words {
            assert_eq!(differ(w, &cpu), None, "{}", interp::disassemble(w));
            interp::execute(&mut cpu, CODE, w).unwrap();
        }
        let what: Vec<String> = words.iter().map(|&w| interp::disassemble(w)).collect();
        for (r, v) in after {
            let (name, got) = if r < X { (format!("v{r}"), cpu.vr[r]) } else { (format!("x{}", r - X), u128::from(cpu.x[r - X])) };
            assert!(got == v, "{}: {name} = {got:#x}, expected {v:#x}", what.join("; "));
        }
    }
}

/// On NaN-free lanes, the FP forms with Arm's NaN rules must stay native: the
/// interpreter is only their rescue for NaN results.
#[test]
fn nan_free_lanes_stay_native() {
    // Normal lanes; in `low`, the upper half (which 8-byte forms ignore) holds signalling NaNs.
    let (full, low) = (f4([1.5, 2.0, 0.75, 3.0]), s4([1.5f32.to_bits(), 2.0f32.to_bits(), 0x7F80_0001, 0x7F80_0001]));
    for case in cases() {
        let garbage = match lift(case.word, CODE) {
            Op::VFp { q, lane, .. } => !q && lane.is_none_or(|l| l < 2),
            Op::FBin { .. } => false,
            _ => continue,
        };
        let mut cpu = Cpu::new(std::ptr::null_mut());
        cpu.vr = [if garbage { low } else { full }; 32];
        let before = RESCUED.get();
        assert_eq!(differ(case.word, &cpu), None, "{}", interp::disassemble(case.word));
        assert_eq!(RESCUED.get(), before, "{}: NaN-free lanes went to the interpreter", interp::disassemble(case.word));
    }
}
