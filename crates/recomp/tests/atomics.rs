//! Guest atomics on several host threads at once (multi-core plan, stage 1):
//! hand-assembled programs, compiled by the Cranelift backend into JIT code
//! and run by the runtime's driver on a synthetic guest memory.

use cranelift_jit::{JITBuilder, JITModule};
use gyroship_analysis::cfg;
use gyroship_analysis::discover::Function;
use gyroship_recomp::ir::Op;
use gyroship_recomp::lift::lift;
use gyroship_recomp::{Callee, Input, clif};
use gyroship_runtime::mem::{Block, Memory, perm, state};
use gyroship_runtime::{self as rt, Cpu, EXIT, interp};
use std::collections::BTreeMap;
use std::hint::spin_loop;
use std::sync::Once;
use std::sync::atomic::{AtomicU32, AtomicU64, Ordering};

const RET: u32 = 0xd65f_03c0;
/// subs x3, x3, #1
const COUNT_DOWN: u32 = 0xf100_0463;

/// b.ne from `from` back to `to`.
fn bne(from: u64, to: u64) -> u32 {
    0x5400_0001 | ((((to as i64 - from as i64) / 4) as u32) & 0x7ffff) << 5
}

/// stlr w1, [x0]; ldar w2, [x3]
const SB_ORDERED: u64 = 0x1000;
/// str w1, [x0]; ldr w2, [x3]: the same without ordering, for comparison.
const SB_PLAIN: u64 = 0x1100;
/// x3 times: ldxr w1, [x0]; add w1, w1, #1; stxr w2, w1, [x0]; cbnz w2, retry.
const EXCLUSIVE: u64 = 0x1200;
/// x3 times: ldadd w4, w5, [x0].
const LSE: u64 = 0x1300;
/// x3 times: ldr w1, [x0]; add w2, w1, #1; mov w5, w1; cas w1, w2, [x0];
/// cmp w1, w5; b.ne retry.
const CAS: u64 = 0x1400;
/// One function per word of `atomic_forms()`: the word, then RET.
const FORMS: u64 = 0x2000;

const DATA: u64 = 0x8000;
const FLAGS: [u64; 2] = [DATA, DATA + 0x40];
const COUNTER: u64 = DATA + 0x80;
const SLOT: u64 = DATA + 0x100;

/// Every LSE atomic (9 operations) and CAS, in each size and ordering, on
/// Rs = x1, Rt = x2, Rn = x0.
fn atomic_forms() -> Vec<u32> {
    let mut words = Vec::new();
    for size in 0..4 {
        for order in 0..4 {
            for (o3, opc) in (0..8).map(|opc| (0, opc)).chain([(1, 0)]) {
                words.push(0x3820_0000 | size << 30 | order << 22 | 1 << 16 | o3 << 15 | opc << 12 | 2);
            }
            // CAS: L is bit 22, o0 bit 15.
            words.push(0x08a0_7c00 | size << 30 | (order >> 1) << 22 | 1 << 16 | (order & 1) << 15 | 2);
        }
    }
    words
}

fn functions() -> Vec<(u64, Vec<u32>)> {
    let mut f = vec![
        (SB_ORDERED, vec![0x889f_fc01, 0x88df_fc62, RET]),
        (SB_PLAIN, vec![0xb900_0001, 0xb940_0062, RET]),
        (EXCLUSIVE, vec![0x885f_7c01, 0x1100_0421, 0x8802_7c01, 0x35ff_ffa2, COUNT_DOWN, bne(EXCLUSIVE + 0x14, EXCLUSIVE), RET]),
        (LSE, vec![0xb824_0005, COUNT_DOWN, bne(LSE + 8, LSE), RET]),
        (CAS, vec![0xb940_0001, 0x1100_0422, 0x2a01_03e5, 0x88a1_7c02, 0x6b05_003f, bne(CAS + 0x14, CAS), COUNT_DOWN, bne(CAS + 0x1c, CAS), RET]),
    ];
    f.extend(atomic_forms().into_iter().enumerate().map(|(i, w)| (FORMS + 8 * i as u64, vec![w, RET])));
    f
}

/// Compiles the functions once into JIT code and installs their dispatch table.
fn setup() {
    static ONCE: Once = Once::new();
    ONCE.call_once(|| {
        let mut words = BTreeMap::new();
        let mut defined = Vec::new();
        for (start, code) in functions() {
            for (i, &w) in code.iter().enumerate() {
                words.insert(start + 4 * i as u64, w);
            }
            defined.push(Function { start, len: 4 * code.len() as u64, unwind: true, entries: vec![start] });
        }
        let mut jb = JITBuilder::with_isa(clif::isa(true).unwrap(), cranelift_module::default_libcall_names());
        let runtime: [(&str, *const u8); 8] = [
            ("gyroship_counter", rt::gyroship_counter as *const () as *const u8),
            ("gyroship_yield", rt::gyroship_yield as *const () as *const u8),
            ("gyroship_probe", rt::gyroship_probe as *const () as *const u8),
            ("gyroship_call", rt::gyroship_call as *const () as *const u8),
            ("gyroship_unhandled", rt::gyroship_unhandled as *const () as *const u8),
            ("gyroship_trap", rt::gyroship_trap as *const () as *const u8),
            ("gyroship_svc", rt::gyroship_svc as *const () as *const u8),
            ("gyroship_miss", rt::gyroship_miss as *const () as *const u8),
        ];
        for (name, f) in runtime {
            jb.symbol(name, f);
        }
        let spans: Vec<(u64, u64)> = defined.iter().map(|f| (f.start, f.len)).collect();
        let mut obj = clif::Object::with_module(JITModule::new(jb), &spans).unwrap();
        let (lo, hi) = (*words.keys().next().unwrap(), *words.keys().next_back().unwrap() + 4);
        let mut image = vec![0u8; (hi - lo) as usize];
        for (&a, &w) in &words {
            image[(a - lo) as usize..][..4].copy_from_slice(&w.to_le_bytes());
        }
        let word = |a: u64| words.get(&a).copied().unwrap_or(0);
        for f in &defined {
            let g = cfg::build(&image, lo, f);
            obj.function(&Input { function: f, cfg: &g, word: &word, callee: &Callee::Dispatch, probe: &|_| false }).unwrap_or_else(|e| panic!("{:#x}: {e}", f.start));
        }
        let (mut module, ids) = obj.into_parts();
        module.finalize_definitions().unwrap();
        let table = defined
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

/// A guest memory with one data page.
fn machine() -> Memory {
    setup();
    let m = Memory::new().unwrap();
    m.map(DATA, 0x1000, Block::new(state::NORMAL, perm::RW)).unwrap();
    m
}

fn word(base: usize, a: u64) -> &'static AtomicU32 {
    // SAFETY: a mapped, aligned guest word of a memory that outlives its users here.
    unsafe { AtomicU32::from_ptr((base as *mut u8).add(a as usize).cast()) }
}

/// Runs `f` from its start on a fresh context: x0, x1, x3, x4 as given.
fn call(base: usize, f: u64, args: [u64; 4]) -> Cpu {
    let mut cpu = Cpu::new(base as *mut u8);
    (cpu.x[0], cpu.x[1], cpu.x[3], cpu.x[4]) = (args[0], args[1], args[2], args[3]);
    (cpu.pc, cpu.x[30]) = (f, EXIT);
    // SAFETY: the context's memory maps every page the functions touch.
    unsafe { rt::run(&mut cpu) };
    cpu
}

/// Dekker's test, `rounds` times: two host threads leave a spin barrier
/// together; each stores 1 to its flag, then loads the other's. Returns how
/// many rounds both loads saw 0, which Arm forbids for STLR then LDAR (x86
/// lets a load pass an earlier store unless a fence separates them).
fn store_buffering(base: usize, f: u64, rounds: u32) -> u32 {
    let (go, done) = (AtomicU32::new(0), AtomicU32::new(0));
    let seen = [AtomicU64::new(0), AtomicU64::new(0)];
    let mut both_zero = 0;
    std::thread::scope(|s| {
        for t in 0..2 {
            let (go, done, seen) = (&go, &done, &seen);
            s.spawn(move || {
                for round in 1..=rounds {
                    while go.load(Ordering::Acquire) != round {
                        spin_loop();
                    }
                    let cpu = call(base, f, [FLAGS[t], 1, FLAGS[1 - t], 0]);
                    seen[t].store(cpu.x[2], Ordering::Relaxed);
                    done.fetch_add(1, Ordering::Release);
                }
            });
        }
        for round in 1..=rounds {
            for a in FLAGS {
                word(base, a).store(0, Ordering::Relaxed);
            }
            done.store(0, Ordering::Relaxed);
            go.store(round, Ordering::Release);
            while done.load(Ordering::Acquire) != 2 {
                spin_loop();
            }
            if seen.iter().all(|v| v.load(Ordering::Relaxed) == 0) {
                both_zero += 1;
            }
        }
    });
    both_zero
}

#[test]
fn a_release_store_and_an_acquire_load_never_both_miss() {
    let m = machine();
    assert_eq!(store_buffering(m.base() as usize, SB_ORDERED, 100_000), 0);
}

/// The same test on plain STR and LDR, to show that the harness sees the host
/// reorder them: `cargo test -- --ignored` (it depends on timing, so it only reports).
#[test]
#[ignore]
fn plain_stores_and_loads_do_get_reordered() {
    let m = machine();
    let n = store_buffering(m.base() as usize, SB_PLAIN, 100_000);
    eprintln!("plain STR then LDR: both loads saw 0 in {n} of 100000 rounds");
}

/// Each atomic form, compiled and interpreted from the same registers and
/// memory: the memory doubleword and x1, x2 must agree. The doubleword's
/// bytes past the access show a write that is too wide.
#[test]
fn compiled_atomics_match_the_interpreter() {
    let m = machine();
    let base = m.base() as usize;
    let slot = || {
        // SAFETY: a mapped, aligned guest doubleword.
        unsafe { AtomicU64::from_ptr((base as *mut u8).add(SLOT as usize).cast()) }
    };
    let mut seed = 0x5eed_u64;
    let mut next = || {
        seed ^= seed << 13;
        seed ^= seed >> 7;
        seed ^= seed << 17;
        seed
    };
    for (i, word) in atomic_forms().into_iter().enumerate() {
        assert!(matches!(lift(word, 0), Op::AtomicRmw { .. } | Op::AtomicCas { .. }), "{word:08x} ({}) is not compiled", interp::disassemble(word));
        let f = FORMS + 8 * i as u64;
        for trial in 0..200 {
            let (old, a, b) = (next(), next(), next());
            // Half the CAS trials compare equal.
            let a = if trial % 2 == 0 { old } else { a };
            let run = |native: bool| {
                slot().store(old, Ordering::Relaxed);
                let mut cpu = Cpu::new(base as *mut u8);
                (cpu.x[0], cpu.x[1], cpu.x[2]) = (SLOT, a, b);
                if native {
                    (cpu.pc, cpu.x[30]) = (f, EXIT);
                    // SAFETY: the context's memory maps the data page.
                    unsafe { rt::run(&mut cpu) };
                } else {
                    interp::execute(&mut cpu, f, word).unwrap();
                }
                (slot().load(Ordering::Relaxed), cpu.x[1], cpu.x[2])
            };
            let (jit, int) = (run(true), run(false));
            assert_eq!(jit, int, "{word:08x} ({}) from [x0] {old:#x}, x1 {a:#x}, x2 {b:#x}", interp::disassemble(word));
        }
    }
}

#[test]
fn increments_from_four_threads_add_up() {
    const PER_THREAD: u64 = 50_000;
    let m = machine();
    let base = m.base() as usize;
    for f in [EXCLUSIVE, LSE, CAS] {
        word(base, COUNTER).store(0, Ordering::Relaxed);
        std::thread::scope(|s| {
            for _ in 0..4 {
                s.spawn(move || {
                    call(base, f, [COUNTER, 0, PER_THREAD, 1]);
                });
            }
        });
        assert_eq!(u64::from(word(base, COUNTER).load(Ordering::Relaxed)), 4 * PER_THREAD, "function {f:#x}");
    }
}
