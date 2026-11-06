//! Non-local control transfers (D20) on hand-assembled guest programs, run
//! in-process: the Cranelift backend compiles them into JIT code, and the
//! runtime's driver executes them on a synthetic guest memory.

use cranelift_jit::{JITBuilder, JITModule};
use gyroship_analysis::cfg;
use gyroship_analysis::discover::Function;
use gyroship_recomp::{Callee, Input, clif};
use gyroship_runtime::mem::{Block, Memory, perm, state};
use gyroship_runtime::{self as rt, Cpu, EXIT};
use std::collections::{BTreeMap, HashSet};
use std::sync::Once;

/// A tiny AArch64 assembler: only what the programs below use.
mod a64 {
    pub const SP: u32 = 31;
    pub const RET: u32 = 0xD65F_03C0;
    /// stp x29, x30, [sp, #-16]!
    pub const PUSH: u32 = 0xA9BF_7BFD;
    /// ldp x29, x30, [sp], #16
    pub const POP: u32 = 0xA8C1_7BFD;
    fn off(from: u64, to: u64) -> u32 {
        ((to as i64 - from as i64) / 4) as u32
    }
    pub fn movz(d: u32, imm: u64) -> u32 {
        assert!(imm < 0x10000);
        0xD280_0000 | (imm as u32) << 5 | d
    }
    /// add xd, xn, #imm (31 is SP on both sides).
    pub fn add(d: u32, n: u32, imm: u32) -> u32 {
        0x9100_0000 | imm << 10 | n << 5 | d
    }
    /// subs xd, xn, #imm (cmp when d is 31).
    pub fn subs(d: u32, n: u32, imm: u32) -> u32 {
        0xF100_0000 | imm << 10 | n << 5 | d
    }
    pub fn str(t: u32, n: u32, imm: u32) -> u32 {
        0xF900_0000 | (imm / 8) << 10 | n << 5 | t
    }
    pub fn ldr(t: u32, n: u32, imm: u32) -> u32 {
        0xF940_0000 | (imm / 8) << 10 | n << 5 | t
    }
    pub fn bl(from: u64, to: u64) -> u32 {
        0x9400_0000 | (off(from, to) & 0x3FF_FFFF)
    }
    pub fn b(from: u64, to: u64) -> u32 {
        0x1400_0000 | (off(from, to) & 0x3FF_FFFF)
    }
    /// b.cond (0 eq, 1 ne).
    pub fn bcond(cond: u32, from: u64, to: u64) -> u32 {
        0x5400_0000 | (off(from, to) & 0x7FFFF) << 5 | cond
    }
    pub fn cbnz(t: u32, from: u64, to: u64) -> u32 {
        0xB500_0000 | (off(from, to) & 0x7FFFF) << 5 | t
    }
    pub fn adr(d: u32, from: u64, to: u64) -> u32 {
        let o = (to - from) as u32;
        0x1000_0000 | (o & 3) << 29 | ((o >> 2) & 0x7FFFF) << 5 | d
    }
    pub fn br(n: u32) -> u32 {
        0xD61F_0000 | n << 5
    }
}
use a64::*;

const DATA: u64 = 0x8000;
const SLOT_M: u64 = DATA + 0x100;
const SLOT_F: u64 = DATA + 0x200;
const COUNT_F: u64 = DATA + 0x300;
const STACKS: u64 = 0x1_0000;
const TOP: u64 = 0x2_0000;
const FIBER_TOP: u64 = 0x3_0000;

// Case 1: longjmp to the continuation of an ancestor two levels up.
const A: u64 = 0x1000;
const SETJMP: u64 = 0x1100;
const LONGJMP: u64 = 0x1200;
const B: u64 = 0x1300;
const C: u64 = 0x1400;
// Case 2: a fake unwinder sets SP and x30 to a landing pad, then returns.
const A2: u64 = 0x2000;
const LANDING: u64 = A2 + 0x20;
const THROWER: u64 = 0x2100;
const JUMPTO: u64 = 0x2200;
// Case 3: two contexts that switch to each other through SWITCH.
const M: u64 = 0x3000;
const F: u64 = 0x3400;
const SWITCH: u64 = 0x3800;
// Case 4: a computed goto to an inner entry.
const G: u64 = 0x4000;
// Case 5: P -> Q -> R -> S, resumed directly three levels down.
const P: u64 = 0x5000;
const Q: u64 = 0x5100;
const R: u64 = 0x5200;
const S: u64 = 0x5300;
// Case 7: a long chain of tail calls between T and U.
const T: u64 = 0x6000;
const U: u64 = 0x6100;
const T2: u64 = 0x6200;
const U2: u64 = 0x6300;
// Case 8: a call as the last word of a function.
const V: u64 = 0x7000;
const W: u64 = 0x7100;

struct Program {
    words: BTreeMap<u64, u32>,
    functions: Vec<Function>,
}

impl Program {
    fn function(&mut self, start: u64, code: &[u32], inner: &[u64]) {
        for (i, &w) in code.iter().enumerate() {
            self.words.insert(start + 4 * i as u64, w);
        }
        let mut entries = vec![start];
        entries.extend_from_slice(inner);
        self.functions.push(Function { start, len: 4 * code.len() as u64, unwind: true, entries });
    }
}

fn program() -> Program {
    let mut p = Program { words: BTreeMap::new(), functions: Vec::new() };
    p.function(A, &[PUSH, movz(0, 0), bl(A + 8, SETJMP), cbnz(0, A + 0xc, A + 0x18), bl(A + 0x10, B), movz(1, 99), movz(2, 7), POP, RET], &[]);
    p.function(SETJMP, &[movz(9, DATA), add(10, SP, 0), str(10, 9, 0), str(30, 9, 8), movz(0, 0), RET], &[]);
    p.function(LONGJMP, &[movz(9, DATA), ldr(10, 9, 0), add(SP, 10, 0), ldr(30, 9, 8), movz(0, 1), br(30)], &[]);
    p.function(B, &[PUSH, bl(B + 4, C), POP, RET], &[]);
    p.function(C, &[PUSH, bl(C + 4, LONGJMP), POP, RET], &[]);

    p.function(A2, &[PUSH, movz(9, DATA), add(10, SP, 0), str(10, 9, 16), bl(A2 + 0x10, THROWER), movz(1, 99), POP, RET, movz(2, 5), POP, RET], &[LANDING]);
    p.function(THROWER, &[PUSH, bl(THROWER + 4, JUMPTO), movz(1, 98), POP, RET], &[]);
    p.function(JUMPTO, &[movz(9, DATA), ldr(10, 9, 16), add(SP, 10, 0), movz(30, LANDING), RET], &[]);

    p.function(M, &[PUSH, movz(19, 0), movz(0, SLOT_M), movz(1, SLOT_F), bl(M + 0x10, SWITCH), add(19, 19, 1), subs(31, 19, 1000), bcond(1, M + 0x1c, M + 8), POP, RET], &[]);
    p.function(F, &[movz(19, 0), add(19, 19, 1), movz(9, COUNT_F), str(19, 9, 0), movz(0, SLOT_F), movz(1, SLOT_M), bl(F + 0x18, SWITCH), b(F + 0x1c, F + 4)], &[]);
    p.function(SWITCH, &[add(9, SP, 0), str(9, 0, 0), str(30, 0, 8), str(19, 0, 16), ldr(9, 1, 0), add(SP, 9, 0), ldr(30, 1, 8), ldr(19, 1, 16), RET], &[]);

    // The target is G + 0x14; the other inner entry, G + 0x1c, gives 98.
    p.function(G, &[movz(0, 0), adr(9, G + 4, G + 0x14), br(9), movz(0, 99), RET, movz(0, 42), RET, movz(0, 98), RET], &[G + 0x14, G + 0x1c]);

    for (f, callee, add_imm) in [(P, Q, 100), (Q, R, 10), (R, S, 1)] {
        p.function(f, &[PUSH, bl(f + 4, callee), add(0, 0, add_imm), POP, RET], &[]);
    }
    p.function(S, &[movz(0, 0), RET], &[]);

    p.function(T, &[subs(0, 0, 1), bcond(0, T + 4, T + 0xc), b(T + 8, U), RET], &[]);
    p.function(U, &[b(U, T)], &[]);
    // The same chain through registers: `br` to a function start.
    p.function(T2, &[subs(0, 0, 1), bcond(0, T2 + 4, T2 + 0x10), movz(9, U2), br(9), RET], &[]);
    p.function(U2, &[movz(9, T2), br(9)], &[]);

    // A function whose last word is a call: execution falls into the next function.
    p.function(V, &[PUSH, bl(V + 4, W)], &[]);
    p.function(V + 8, &[POP, add(0, 0, 1), RET], &[]);
    p.function(W, &[movz(0, 41), RET], &[]);
    p
}

/// Compiles the program once into JIT code and installs its dispatch table.
fn setup() {
    static ONCE: Once = Once::new();
    ONCE.call_once(|| {
        let p = program();
        let mut jb = JITBuilder::with_isa(clif::isa(true).unwrap(), cranelift_module::default_libcall_names());
        let runtime: [(&str, *const u8); 5] = [
            ("gyroship_call", rt::gyroship_call as *const () as *const u8),
            ("gyroship_unhandled", rt::gyroship_unhandled as *const () as *const u8),
            ("gyroship_trap", rt::gyroship_trap as *const () as *const u8),
            ("gyroship_svc", rt::gyroship_svc as *const () as *const u8),
            ("gyroship_miss", rt::gyroship_miss as *const () as *const u8),
        ];
        for (name, f) in runtime {
            jb.symbol(name, f);
        }
        let defined: Vec<(u64, u64)> = p.functions.iter().map(|f| (f.start, f.len)).collect();
        let mut obj = clif::Object::with_module(JITModule::new(jb), &defined).unwrap();
        // One image from the lowest to the highest word; gaps are zero.
        let (lo, hi) = (*p.words.keys().next().unwrap(), *p.words.keys().next_back().unwrap() + 4);
        let mut image = vec![0u8; (hi - lo) as usize];
        for (&a, &w) in &p.words {
            image[(a - lo) as usize..][..4].copy_from_slice(&w.to_le_bytes());
        }
        let starts: HashSet<u64> = p.functions.iter().map(|f| f.start).collect();
        let word = |a: u64| p.words.get(&a).copied().unwrap_or(0);
        let callee = |t: u64| if starts.contains(&t) { Callee::Direct(t) } else { Callee::Dispatch(t) };
        for f in &p.functions {
            let g = cfg::build(&image, lo, f);
            obj.function(&Input { function: f, cfg: &g, word: &word, callee: &callee }).unwrap_or_else(|e| panic!("{:#x}: {e}", f.start));
        }
        let (mut module, ids) = obj.into_parts();
        module.finalize_definitions().unwrap();
        let table = p
            .functions
            .iter()
            .map(|f| {
                let code = module.get_finalized_function(ids[&f.start]);
                // SAFETY: the JIT compiled this function with the `Guest` signature.
                rt::Entry { addr: f.start, len: f.len, function: unsafe { std::mem::transmute::<*const u8, rt::Guest>(code) } }
            })
            .collect();
        rt::install(table);
        // The code must outlive every test of this process.
        std::mem::forget(module);
    });
}

/// A guest memory with a data page and the stacks, and a context that
/// starts at `pc` with SP = `TOP` and x30 = `EXIT`.
fn machine(pc: u64) -> (Memory, Cpu) {
    setup();
    let mut m = Memory::new().unwrap();
    m.map(DATA, 0x1000, Block::new(state::NORMAL, perm::RW)).unwrap();
    m.map(STACKS, FIBER_TOP - STACKS, Block::new(state::STACK, perm::RW)).unwrap();
    let mut cpu = Cpu::new(m.base());
    (cpu.x[31], cpu.x[30], cpu.pc) = (TOP, EXIT, pc);
    (m, cpu)
}

fn run(cpu: &mut Cpu) {
    // SAFETY: the context's memory maps every page the programs touch.
    unsafe { rt::run(cpu) };
}

fn read64(m: &Memory, a: u64) -> u64 {
    let mut b = [0; 8];
    m.read(a, &mut b).unwrap();
    u64::from_le_bytes(b)
}

#[test]
fn longjmp_reaches_an_ancestor_two_levels_up() {
    let (_m, mut cpu) = machine(A);
    run(&mut cpu);
    assert_eq!((cpu.x[0], cpu.x[1], cpu.x[2], cpu.x[31]), (1, 0, 7, TOP));
}

#[test]
fn unwinder_lands_on_a_landing_pad() {
    let (_m, mut cpu) = machine(A2);
    run(&mut cpu);
    assert_eq!((cpu.x[1], cpu.x[2], cpu.x[31]), (0, 5, TOP));
}

#[test]
fn two_contexts_switch_a_thousand_times() {
    let (mut m, mut cpu) = machine(M);
    // The second context starts at F on its own stack.
    let slot: Vec<u8> = [FIBER_TOP, F, 0].iter().flat_map(|v| v.to_le_bytes()).collect();
    m.write(SLOT_F, &slot).unwrap();
    run(&mut cpu);
    assert_eq!((cpu.x[19], read64(&m, COUNT_F), cpu.x[31]), (1000, 1000, TOP));
}

#[test]
fn computed_goto_reaches_an_inner_entry() {
    let (_m, mut cpu) = machine(G);
    run(&mut cpu);
    assert_eq!(cpu.x[0], 42);
}

#[test]
fn resume_three_levels_down() {
    let (_m, mut cpu) = machine(P);
    run(&mut cpu);
    assert_eq!((cpu.x[0], cpu.x[31]), (111, TOP));

    // The same state as just after S returned: frames of R, Q and P on the
    // stack, each with the return address of its caller.
    let (mut m, mut cpu) = machine(R + 8);
    let frames: Vec<u8> = [0, Q + 8, 0, P + 8, 0, EXIT].iter().flat_map(|v| v.to_le_bytes()).collect();
    m.write(TOP - 48, &frames).unwrap();
    (cpu.x[0], cpu.x[31]) = (0, TOP - 48);
    run(&mut cpu);
    assert_eq!((cpu.x[0], cpu.x[31]), (111, TOP));
}

#[test]
fn tail_call_chains_do_not_overflow_the_host_stack() {
    // 400 000 nested tail calls without the stack limit, direct then through registers.
    for start in [T, T2] {
        let (_m, mut cpu) = machine(start);
        cpu.x[0] = 200_000;
        run(&mut cpu);
        assert_eq!(cpu.x[0], 0, "{start:#x}");
    }
}

#[test]
fn a_call_can_end_a_function() {
    let (_m, mut cpu) = machine(V);
    run(&mut cpu);
    assert_eq!((cpu.x[0], cpu.x[31]), (42, TOP));
}

/// Only does something when started by `resume_on_an_unknown_address_is_fatal`.
#[test]
fn unknown_resume_child() {
    if std::env::var("GYROSHIP_RESUME_CHILD").is_err() {
        return;
    }
    // R + 0xc is inside the continuation block of `bl S`, not at its start.
    let (_m, mut cpu) = machine(R + 0xc);
    run(&mut cpu);
}

#[test]
fn resume_on_an_unknown_address_is_fatal() {
    let out = std::process::Command::new(std::env::current_exe().unwrap())
        .args(["--exact", "unknown_resume_child", "--nocapture", "--test-threads=1"])
        .env("GYROSHIP_RESUME_CHILD", "1")
        .output()
        .unwrap();
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(!out.status.success(), "{stderr}");
    assert!(stderr.contains("guest address 0x520c is neither a known block nor a resume point"), "{stderr}");
}
