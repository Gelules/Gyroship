//! Guest CPU context, guest memory, dispatch table and exceptions.
//!
//! Recompiled functions are `extern "C" fn(*mut Cpu)`, whatever backend
//! generated them. They reach the rest of the runtime through the
//! `gyroship_*` symbols below.

pub mod harness;
pub mod interp;

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
};

impl Cpu {
    pub fn new(mem: *mut u8) -> Cpu {
        Cpu { x: [0; 32], n: false, z: false, c: false, v: false, tpidr: 0, tpidrro: 0, mem, vr: [0; 32], fpcr: 0, fpsr: 0 }
    }
}

/// A recompiled guest function.
pub type Guest = unsafe extern "C" fn(*mut Cpu);

/// One entry of a dispatch table emitted by the generator.
#[repr(C)]
#[derive(Clone, Copy)]
pub struct Entry {
    pub addr: u64,
    pub function: Guest,
}

/// The entries of a generated table.
///
/// # Safety
/// `table` must point to `len` entries that live for the whole program.
pub unsafe fn entries(table: *const Entry, len: u64) -> impl Iterator<Item = (u64, Guest)> {
    // SAFETY: guaranteed by the caller.
    unsafe { std::slice::from_raw_parts(table, len as usize) }.iter().map(|e| (e.addr, e.function))
}

static DISPATCH: OnceLock<Vec<(u64, Guest)>> = OnceLock::new();

/// Installs the dispatch table: guest entry address -> host function.
pub fn install(mut table: Vec<(u64, Guest)>) {
    table.sort_unstable_by_key(|e| e.0);
    let _ = DISPATCH.set(table);
}

pub fn lookup(addr: u64) -> Option<Guest> {
    let t = DISPATCH.get()?;
    t.binary_search_by_key(&addr, |e| e.0).ok().map(|i| t[i].1)
}

/// Untranslated instructions the interpreter executed.
pub static INTERPRETED: AtomicU64 = AtomicU64::new(0);
/// Untranslated instructions the interpreter does not cover either (each logged once).
pub static UNHANDLED: AtomicU64 = AtomicU64::new(0);

/// Calls the guest function at `target` through the dispatch table.
///
/// # Safety
/// `cpu` must point to a valid context whose memory covers what the callee touches.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn gyroship_call(cpu: *mut Cpu, target: u64) {
    match lookup(target) {
        Some(f) => unsafe { f(cpu) },
        None => panic!("no recompiled function at guest address {target:#x}"),
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

#[unsafe(no_mangle)]
pub extern "C" fn gyroship_trap(_cpu: *mut Cpu, pc: u64, imm: u32) {
    panic!("guest trap #{imm:#x} at {pc:#x}");
}

#[unsafe(no_mangle)]
pub extern "C" fn gyroship_svc(_cpu: *mut Cpu, pc: u64, imm: u32) {
    panic!("guest svc #{imm:#x} at {pc:#x} (no HLE yet)");
}

/// A computed jump inside a function landed outside its known blocks.
#[unsafe(no_mangle)]
pub extern "C" fn gyroship_miss(_cpu: *mut Cpu, pc: u64) {
    panic!("indirect jump to {pc:#x} is not a known block");
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

macro_rules! access {
    ($ld:ident, $st:ident, $t:ty) => {
        #[inline(always)]
        pub fn $ld(c: &Cpu, addr: u64) -> u64 {
            // SAFETY: the guest address space is one reservation that covers every guest address.
            unsafe { (c.mem.add(addr as usize) as *const $t).read_unaligned() as u64 }
        }
        #[inline(always)]
        pub fn $st(c: &Cpu, addr: u64, v: u64) {
            // SAFETY: as above.
            unsafe { (c.mem.add(addr as usize) as *mut $t).write_unaligned(v as $t) }
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
    unsafe { (c.mem.add(addr as usize) as *const u128).read_unaligned() }
}

#[inline(always)]
pub fn st128(c: &Cpu, addr: u64, v: u128) {
    // SAFETY: as for the other accesses.
    unsafe { (c.mem.add(addr as usize) as *mut u128).write_unaligned(v) }
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
