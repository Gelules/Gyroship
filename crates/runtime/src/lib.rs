//! Guest CPU context, guest memory, dispatch table and exceptions.
//!
//! Recompiled functions are `extern "C" fn(*mut Cpu, resume: u64)`, whatever
//! backend generated them. They reach the rest of the runtime through the
//! `gyroship_*` symbols below.
//!
//! Control transfers (D20): `RET` writes its target to `Cpu::pc`, and after
//! each call the caller checks that `pc` is its own continuation. When it is
//! not (an exception landing pad, `longjmp`, a fiber switch), the caller
//! jumps there if the address is one of its resume points, or returns, and
//! the mismatch travels down the host stack to the frame that owns the
//! target, or to [`run`], which resumes the function that contains it.

pub mod harness;
pub mod interp;
pub mod mem;
#[cfg(windows)]
pub mod fiber;

use std::sync::OnceLock;
use std::sync::atomic::{AtomicU64, Ordering};

/// The context of one guest thread. The layout is fixed: generated code
/// addresses its fields by offset (see the `OFFSET_*` constants).
#[repr(C)]
pub struct Cpu {
    /// x0..x30, then SP.
    pub x: [u64; 32],
    pub n: bool,
    pub z: bool,
    pub c: bool,
    pub v: bool,
    pub tpidr: u64,
    pub tpidrro: u64,
    /// Host address of guest address 0.
    pub mem: *mut u8,
    /// v0..v31, little-endian lanes.
    pub vr: [u128; 32],
    pub fpcr: u64,
    pub fpsr: u64,
    /// Where guest execution goes on: set by `RET`, by non-local jumps and
    /// before [`run`].
    pub pc: u64,
    /// Host stack address below which generated code stops nesting tail
    /// calls and hands them to [`run`] instead (set by `run`).
    pub stack_limit: u64,
}

pub const OFFSET_X: i32 = 0;
pub const OFFSET_N: i32 = 256;
pub const OFFSET_Z: i32 = 257;
pub const OFFSET_C: i32 = 258;
pub const OFFSET_V: i32 = 259;
pub const OFFSET_TPIDR: i32 = 264;
pub const OFFSET_TPIDRRO: i32 = 272;
pub const OFFSET_MEM: i32 = 280;
pub const OFFSET_VR: i32 = 288;
pub const OFFSET_PC: i32 = 816;
pub const OFFSET_STACK_LIMIT: i32 = 824;

const _: () = {
    assert!(std::mem::offset_of!(Cpu, x) == OFFSET_X as usize);
    assert!(std::mem::offset_of!(Cpu, n) == OFFSET_N as usize);
    assert!(std::mem::offset_of!(Cpu, z) == OFFSET_Z as usize);
    assert!(std::mem::offset_of!(Cpu, c) == OFFSET_C as usize);
    assert!(std::mem::offset_of!(Cpu, v) == OFFSET_V as usize);
    assert!(std::mem::offset_of!(Cpu, tpidr) == OFFSET_TPIDR as usize);
    assert!(std::mem::offset_of!(Cpu, tpidrro) == OFFSET_TPIDRRO as usize);
    assert!(std::mem::offset_of!(Cpu, mem) == OFFSET_MEM as usize);
    assert!(std::mem::offset_of!(Cpu, vr) == OFFSET_VR as usize);
    assert!(std::mem::offset_of!(Cpu, pc) == OFFSET_PC as usize);
    assert!(std::mem::offset_of!(Cpu, stack_limit) == OFFSET_STACK_LIMIT as usize);
};

impl Cpu {
    pub fn new(mem: *mut u8) -> Cpu {
        Cpu { x: [0; 32], n: false, z: false, c: false, v: false, tpidr: 0, tpidrro: 0, mem, vr: [0; 32], fpcr: 0, fpsr: 0, pc: 0, stack_limit: 0 }
    }
}

/// A recompiled guest function. `resume` is 0 to start at the beginning, or
/// one of the function's resume points (an entry, or the continuation of a
/// call or SVC); any other address is a fatal [`gyroship_miss`].
pub type Guest = unsafe extern "C" fn(*mut Cpu, u64);

/// One entry of a dispatch table emitted by the generator: a guest function,
/// its size in bytes, and its host code.
#[repr(C)]
#[derive(Clone, Copy)]
pub struct Entry {
    pub addr: u64,
    pub len: u64,
    pub function: Guest,
}

/// The entries of a generated table.
///
/// # Safety
/// `table` must point to `len` entries that live for the whole program.
pub unsafe fn entries(table: *const Entry, len: u64) -> impl Iterator<Item = Entry> {
    // SAFETY: guaranteed by the caller.
    unsafe { std::slice::from_raw_parts(table, len as usize) }.iter().copied()
}

static DISPATCH: OnceLock<Vec<Entry>> = OnceLock::new();

/// Installs the dispatch table. Functions never overlap (D14).
pub fn install(mut table: Vec<Entry>) {
    table.sort_unstable_by_key(|e| e.addr);
    let _ = DISPATCH.set(table);
}

/// The function that starts at `addr`.
pub fn lookup(addr: u64) -> Option<Guest> {
    containing(addr).filter(|e| e.addr == addr).map(|e| e.function)
}

/// The function whose body contains `pc`.
pub fn containing(pc: u64) -> Option<Entry> {
    let t = DISPATCH.get()?;
    let e = t[..t.partition_point(|e| e.addr <= pc)].last()?;
    (pc - e.addr < e.len).then_some(*e)
}

/// The guest function whose recompiled code starts closest below host address
/// `rip`. Host code sizes are not recorded, so this is only a hint.
pub fn function_at_host(rip: u64) -> Option<u64> {
    let host = |e: &Entry| e.function as usize as u64;
    DISPATCH.get()?.iter().filter(|e| host(e) <= rip).max_by_key(|e| host(e)).map(|e| e.addr)
}

/// The return address of a thread's first function: [`run`] stops there.
pub const EXIT: u64 = 0xFFFF_FFFF_FFFF_FFF0;

/// Host stack that generated code may use below [`run`] before it hands tail
/// calls back to it. The host thread or fiber needs this much plus a margin.
pub const STACK_BUDGET: u64 = 512 << 10;

/// Runs guest code from `cpu.pc` until it returns to [`EXIT`]: the driver of
/// one guest thread. Each step calls the function that contains `cpu.pc`, at
/// its start or at a resume point; a non-local jump unwinds host frames back
/// here, and the loop goes on at the new `cpu.pc`.
///
/// # Safety
/// `cpu` must point to a valid context whose memory covers what guest code touches.
pub unsafe fn run(cpu: *mut Cpu) {
    let here = 0u8;
    // SAFETY: guaranteed by the caller.
    unsafe { (*cpu).stack_limit = (&raw const here as u64).saturating_sub(STACK_BUDGET) };
    loop {
        // SAFETY: as above.
        let pc = unsafe { (*cpu).pc };
        if pc == EXIT {
            return;
        }
        let Some(e) = containing(pc) else { panic!("guest pc {pc:#x} is in no recompiled function") };
        // SAFETY: as above.
        unsafe { (e.function)(cpu, if pc == e.addr { 0 } else { pc }) };
    }
}

/// Untranslated instructions the interpreter executed.
pub static INTERPRETED: AtomicU64 = AtomicU64::new(0);
/// Untranslated instructions the interpreter does not cover either (each logged once).
pub static UNHANDLED: AtomicU64 = AtomicU64::new(0);

/// An indirect call or jump to `target`: calls the function that starts
/// there. An address inside a function is a non-local jump: it becomes
/// `cpu.pc`, and the generated callers hand it down to [`run`].
///
/// # Safety
/// `cpu` must point to a valid context whose memory covers what the callee touches.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn gyroship_call(cpu: *mut Cpu, target: u64) {
    match containing(target) {
        // SAFETY: guaranteed by the caller.
        Some(e) if e.addr == target => unsafe { (e.function)(cpu, 0) },
        // SAFETY: as above.
        Some(_) => unsafe { (*cpu).pc = target },
        None => panic!("no recompiled function contains guest address {target:#x}"),
    }
}

/// An instruction the generator did not translate: the interpreter runs it.
///
/// # Safety
/// `cpu` must point to a valid context.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn gyroship_unhandled(cpu: *mut Cpu, pc: u64, word: u32) {
    static LOGGED: std::sync::Mutex<Vec<u32>> = std::sync::Mutex::new(Vec::new());
    // SAFETY: guaranteed by the caller.
    let c = unsafe { &mut *cpu };
    match interp::execute(c, pc, word) {
        Ok(()) => {
            INTERPRETED.fetch_add(1, Ordering::Relaxed);
        }
        Err(why) => {
            UNHANDLED.fetch_add(1, Ordering::Relaxed);
            let mut logged = LOGGED.lock().unwrap_or_else(|e| e.into_inner());
            if !logged.contains(&word) {
                logged.push(word);
                eprintln!("gyroship: instruction {word:08x} at {pc:#x} is not executed: {why}");
            }
        }
    }
}

/// `imm` of [`gyroship_trap`] for a branch to itself (`b .`), the idiom of
/// a fatal error: the generator stops there instead of spinning.
pub const TRAP_SELF_LOOP: u32 = 0x1_0000;

#[unsafe(no_mangle)]
pub extern "C" fn gyroship_trap(_cpu: *mut Cpu, pc: u64, imm: u32) {
    if imm == TRAP_SELF_LOOP {
        panic!("guest branches to itself at {pc:#x} (a fatal-error loop)");
    }
    panic!("guest trap #{imm:#x} at {pc:#x}");
}

/// What the HLE plugs into generated code (`gyroship-hle` registers it; the
/// runtime does not depend on the HLE, so harnesses without one still link).
/// Every hook is a point where the scheduler may switch guest threads, and
/// writes its results into the context before it does.
#[derive(Clone, Copy)]
pub struct Hooks {
    /// `svc #imm` at `pc`: arguments in x0..x7, Result in w0, outputs in x1 and x2.
    pub svc: unsafe fn(*mut Cpu, u64, u32),
    /// `mrs x<rt>, CNTPCT_EL0`: writes the virtual counter to x<rt> (31: nowhere).
    pub counter: unsafe fn(*mut Cpu, u32),
    /// `yield` (1), `wfe` (2) or `wfi` (3).
    pub wait: unsafe fn(*mut Cpu, u32),
    /// The first pass at a probe (`pc`, its name).
    pub probe: unsafe fn(*mut Cpu, u64, &str),
    /// A guest fault (its message), just before the process exits: a last
    /// chance to write the journal.
    pub fault: fn(&str),
}

static HOOKS: OnceLock<Hooks> = OnceLock::new();

/// Registers the hooks; the first registration wins.
pub fn set_hooks(hooks: Hooks) {
    let _ = HOOKS.set(hooks);
}

/// The virtual counter without an HLE (`CNTPCT_EL0`, in ticks of 31.25 MHz):
/// one global clock that each read advances by [`COUNTER_STEP`].
pub static COUNTER: AtomicU64 = AtomicU64::new(0);
/// Ticks one counter read costs: 1.024 µs.
pub const COUNTER_STEP: u64 = 32;

/// `mrs x<rt>, CNTPCT_EL0`.
///
/// # Safety
/// `cpu` must point to a valid context.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn gyroship_counter(cpu: *mut Cpu, _pc: u64, rt: u32) {
    if let Some(h) = HOOKS.get() {
        // SAFETY: guaranteed by the caller.
        return unsafe { (h.counter)(cpu, rt) };
    }
    let v = COUNTER.fetch_add(COUNTER_STEP, Ordering::Relaxed) + COUNTER_STEP;
    if rt < 31 {
        // SAFETY: guaranteed by the caller.
        unsafe { (*cpu).x[rt as usize] = v };
    }
}

/// `yield` (1), `wfe` (2) or `wfi` (3): nothing to do without a scheduler.
///
/// # Safety
/// `cpu` must point to a valid context.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn gyroship_yield(cpu: *mut Cpu, _pc: u64, kind: u32) {
    if let Some(h) = HOOKS.get() {
        // SAFETY: guaranteed by the caller.
        unsafe { (h.wait)(cpu, kind) }
    }
}

/// Named guest addresses whose passes are counted (D21: sondes).
static PROBES: OnceLock<Vec<(u64, String, AtomicU64)>> = OnceLock::new();

pub fn install_probes(mut probes: Vec<(u64, String)>) {
    probes.sort_unstable_by_key(|p| p.0);
    let _ = PROBES.set(probes.into_iter().map(|(a, n)| (a, n, AtomicU64::new(0))).collect());
}

/// Each probe with the number of times guest code passed it.
pub fn probe_counts() -> Vec<(u64, &'static str, u64)> {
    PROBES.get().map_or_else(Vec::new, |p| p.iter().map(|(a, n, c)| (*a, n.as_str(), c.load(Ordering::Relaxed))).collect())
}

/// Generated code passes a probed address; the first pass goes to the
/// HLE's hook, or to stderr without one.
///
/// # Safety
/// `cpu` must point to a valid context.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn gyroship_probe(cpu: *mut Cpu, pc: u64) {
    let Some(p) = PROBES.get() else { return };
    if let Ok(i) = p.binary_search_by_key(&pc, |e| e.0)
        && p[i].2.fetch_add(1, Ordering::Relaxed) == 0
    {
        match HOOKS.get() {
            // SAFETY: guaranteed by the caller.
            Some(h) => unsafe { (h.probe)(cpu, pc, &p[i].1) },
            None => eprintln!("gyroship: probe {} at {pc:#x}", p[i].1),
        }
    }
}

/// `svc #imm` at `pc`, handled by the HLE's hook.
///
/// # Safety
/// `cpu` must point to a valid context.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn gyroship_svc(cpu: *mut Cpu, pc: u64, imm: u32) {
    match HOOKS.get() {
        // SAFETY: guaranteed by the caller.
        Some(h) => unsafe { (h.svc)(cpu, pc, imm) },
        None => panic!("guest svc #{imm:#x} at {pc:#x} and no HLE"),
    }
}

/// A computed jump or a resume landed on an address its function does not
/// know: an analysis gap, fatal.
#[unsafe(no_mangle)]
pub extern "C" fn gyroship_miss(_cpu: *mut Cpu, pc: u64) {
    panic!("guest address {pc:#x} is neither a known block nor a resume point of its function");
}

// Helpers for generated Rust code; Cranelift code inlines the same semantics.

#[inline(always)]
pub fn cond(c: &Cpu, code: u8) -> bool {
    let r = match code >> 1 {
        0 => c.z,
        1 => c.c,
        2 => c.n,
        3 => c.v,
        4 => c.c && !c.z,
        5 => c.n == c.v,
        6 => !c.z && c.n == c.v,
        _ => true,
    };
    if code & 1 == 1 && code != 15 { !r } else { r }
}

#[inline(always)]
pub fn set_nzcv(c: &mut Cpu, nzcv: u8) {
    (c.n, c.z, c.c, c.v) = (nzcv & 8 != 0, nzcv & 4 != 0, nzcv & 2 != 0, nzcv & 1 != 0);
}

#[inline(always)]
pub fn adds64(c: &mut Cpu, a: u64, b: u64) -> u64 {
    let (r, carry) = a.overflowing_add(b);
    (c.n, c.z, c.c, c.v) = ((r as i64) < 0, r == 0, carry, (a as i64).overflowing_add(b as i64).1);
    r
}

#[inline(always)]
pub fn subs64(c: &mut Cpu, a: u64, b: u64) -> u64 {
    let r = a.wrapping_sub(b);
    (c.n, c.z, c.c, c.v) = ((r as i64) < 0, r == 0, a >= b, (a as i64).overflowing_sub(b as i64).1);
    r
}

#[inline(always)]
pub fn adds32(c: &mut Cpu, a: u32, b: u32) -> u32 {
    let (r, carry) = a.overflowing_add(b);
    (c.n, c.z, c.c, c.v) = ((r as i32) < 0, r == 0, carry, (a as i32).overflowing_add(b as i32).1);
    r
}

#[inline(always)]
pub fn subs32(c: &mut Cpu, a: u32, b: u32) -> u32 {
    let r = a.wrapping_sub(b);
    (c.n, c.z, c.c, c.v) = ((r as i32) < 0, r == 0, a >= b, (a as i32).overflowing_sub(b as i32).1);
    r
}

#[inline(always)]
pub fn logic64(c: &mut Cpu, r: u64) -> u64 {
    (c.n, c.z, c.c, c.v) = ((r as i64) < 0, r == 0, false, false);
    r
}

#[inline(always)]
pub fn logic32(c: &mut Cpu, r: u32) -> u32 {
    (c.n, c.z, c.c, c.v) = ((r as i32) < 0, r == 0, false, false);
    r
}

#[inline(always)]
fn mask(bits: u32) -> u64 {
    if bits >= 64 { !0 } else { (1 << bits) - 1 }
}

/// UBFM on `bits`-wide data (32 or 64).
#[inline(always)]
pub fn ubfm(bits: u32, src: u64, immr: u32, imms: u32) -> u64 {
    let src = src & mask(bits);
    let r = if imms >= immr { (src >> immr) & mask(imms - immr + 1) } else { (src & mask(imms + 1)) << (bits - immr) };
    r & mask(bits)
}

#[inline(always)]
pub fn sbfm(bits: u32, src: u64, immr: u32, imms: u32) -> u64 {
    let r = if imms >= immr {
        let width = imms - immr + 1;
        (((src >> immr) << (64 - width)) as i64 >> (64 - width)) as u64
    } else {
        let width = imms + 1;
        ((((src << (64 - width)) as i64) >> (64 - width)) as u64) << (bits - immr)
    };
    r & mask(bits)
}

#[inline(always)]
pub fn bfm(bits: u32, dst: u64, src: u64, immr: u32, imms: u32) -> u64 {
    let r = if imms >= immr {
        let m = mask(imms - immr + 1);
        (dst & !m) | ((src >> immr) & m)
    } else {
        let lsb = bits - immr;
        let m = mask(imms + 1) << lsb;
        (dst & !m) | ((src << lsb) & m)
    };
    r & mask(bits)
}

#[inline(always)]
pub fn rev16(bits: u32, v: u64) -> u64 {
    (((v >> 8) & 0x00FF_00FF_00FF_00FF) | ((v << 8) & 0xFF00_FF00_FF00_FF00)) & mask(bits)
}

#[inline(always)]
pub fn rev32(v: u64) -> u64 {
    u64::from((v as u32).swap_bytes()) | u64::from(((v >> 32) as u32).swap_bytes()) << 32
}

/// Offset of a guest address in the reservation. The mask keeps a wild
/// address inside the reservation, where it faults instead of touching host
/// memory; it does not make the access correct.
#[inline(always)]
pub(crate) fn host(addr: u64) -> usize {
    (addr & (mem::SPACE - 1)) as usize
}

macro_rules! access {
    ($ld:ident, $st:ident, $t:ty) => {
        #[inline(always)]
        pub fn $ld(c: &Cpu, addr: u64) -> u64 {
            // SAFETY: the guest address space is one reservation that covers every masked address.
            unsafe { (c.mem.add(host(addr)) as *const $t).read_unaligned() as u64 }
        }
        #[inline(always)]
        pub fn $st(c: &Cpu, addr: u64, v: u64) {
            // SAFETY: as above.
            unsafe { (c.mem.add(host(addr)) as *mut $t).write_unaligned(v as $t) }
        }
    };
}
access!(ld8, st8, u8);
access!(ld16, st16, u16);
access!(ld32, st32, u32);
access!(ld64, st64, u64);

#[inline(always)]
pub fn ld128(c: &Cpu, addr: u64) -> u128 {
    // SAFETY: as for the other accesses.
    unsafe { (c.mem.add(host(addr)) as *const u128).read_unaligned() }
}

#[inline(always)]
pub fn st128(c: &Cpu, addr: u64, v: u128) {
    // SAFETY: as for the other accesses.
    unsafe { (c.mem.add(host(addr)) as *mut u128).write_unaligned(v) }
}

/// NZCV of an FCMP: equal 0110, less 1000, greater 0010, unordered 0011.
#[inline(always)]
pub fn fcmp(c: &mut Cpu, a: f64, b: f64) {
    let unordered = a.is_nan() || b.is_nan();
    (c.n, c.z, c.c, c.v) = (a < b, a == b, a >= b || unordered, unordered);
}

/// FMAX: NaN propagates, and +0 is larger than -0.
#[inline(always)]
pub fn fmax64(a: f64, b: f64) -> f64 {
    if a.is_nan() || b.is_nan() {
        a + b
    } else if a == b {
        f64::from_bits(a.to_bits() & b.to_bits())
    } else if a > b {
        a
    } else {
        b
    }
}

/// FMIN: NaN propagates, and -0 is smaller than +0.
#[inline(always)]
pub fn fmin64(a: f64, b: f64) -> f64 {
    if a.is_nan() || b.is_nan() {
        a + b
    } else if a == b {
        f64::from_bits(a.to_bits() | b.to_bits())
    } else if a < b {
        a
    } else {
        b
    }
}

#[inline(always)]
pub fn fmax32(a: f32, b: f32) -> f32 {
    if a.is_nan() || b.is_nan() {
        a + b
    } else if a == b {
        f32::from_bits(a.to_bits() & b.to_bits())
    } else if a > b {
        a
    } else {
        b
    }
}

#[inline(always)]
pub fn fmin32(a: f32, b: f32) -> f32 {
    if a.is_nan() || b.is_nan() {
        a + b
    } else if a == b {
        f32::from_bits(a.to_bits() | b.to_bits())
    } else if a < b {
        a
    } else {
        b
    }
}
