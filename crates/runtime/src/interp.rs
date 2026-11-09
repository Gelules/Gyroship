//! Fallback interpreter: executes, one at a time, the instructions the
//! generator does not translate (NEON lane operations, LSE atomics, system
//! registers...). Generated code calls it through `gyroship_unhandled`.
//!
//! With a null guest memory every access is a no-op, so `supported` can run
//! an instruction "dry" on a scratch context to learn whether it is covered.

use crate::Cpu;
use gyroship_analysis::decode::{Instruction, Opcode as C, Operand as O, SIMDSizeCode, ShiftStyle, SizeCode, decode};
use std::sync::atomic::{AtomicU8, AtomicU16, AtomicU32, AtomicU64, Ordering};

pub type Result<T = ()> = std::result::Result<T, &'static str>;

/// Executes one instruction; Err names what is not covered.
pub fn execute(c: &mut Cpu, pc: u64, word: u32) -> Result {
    let i = decoded(pc, word).ok_or("undecodable")?;
    exec(c, pc, &i)
}

/// Decoded instructions by PC, direct-mapped: the same few thousand
/// instructions run every frame, and decoding costs more than executing.
const CACHE: usize = 1 << 14;

fn decoded(pc: u64, word: u32) -> Option<Instruction> {
    type Slot = Option<(u64, u32, Instruction)>;
    thread_local! {
        static SLOTS: std::cell::RefCell<Vec<Slot>> = const { std::cell::RefCell::new(Vec::new()) };
    }
    SLOTS.with(|s| {
        let mut s = s.borrow_mut();
        if s.is_empty() {
            s.resize(CACHE, None);
        }
        let slot = &mut s[(pc >> 2) as usize % CACHE];
        match slot {
            Some((p, w, i)) if *p == pc && *w == word => Some(*i),
            _ => {
                let i = decode(word)?;
                *slot = Some((pc, word, i));
                Some(i)
            }
        }
    })
}

/// The opcode of an instruction word, for reports.
pub fn opcode_name(word: u32) -> String {
    decode(word).map_or_else(|| "undecodable".into(), |i| format!("{:?}", i.opcode))
}

/// An instruction word in assembly, for reports.
pub fn disassemble(word: u32) -> String {
    decode(word).map_or_else(|| "undecodable".into(), |i| i.to_string())
}

/// Whether `execute` covers the instruction (it is run on a scratch context).
pub fn supported(word: u32, pc: u64) -> bool {
    let mut c = Cpu::new(std::ptr::null_mut());
    execute(&mut c, pc, word).is_ok()
}

// ---- registers and memory ----

const LOW64: u128 = u64::MAX as u128;

fn mask(bits: u32) -> u64 {
    if bits >= 64 { u64::MAX } else { (1u64 << bits) - 1 }
}

fn sext(v: u64, bits: u32) -> i64 {
    ((v << (64 - bits)) as i64) >> (64 - bits)
}

fn gpr(c: &Cpu, o: &O) -> Result<u64> {
    Ok(match *o {
        O::Register(_, 31) => 0,
        O::RegisterOrSP(SizeCode::X, 31) => c.x[31],
        O::RegisterOrSP(SizeCode::W, 31) => c.x[31] & 0xFFFF_FFFF,
        O::Register(SizeCode::X, n) | O::RegisterOrSP(SizeCode::X, n) => c.x[n as usize],
        O::Register(SizeCode::W, n) | O::RegisterOrSP(SizeCode::W, n) => c.x[n as usize] & 0xFFFF_FFFF,
        _ => return Err("expected a general register"),
    })
}

fn set_gpr(c: &mut Cpu, o: &O, v: u64) -> Result {
    let (n, w) = match *o {
        O::Register(_, 31) => return Ok(()),
        O::Register(s, n) | O::RegisterOrSP(s, n) => (n as usize, s),
        _ => return Err("expected a general register"),
    };
    c.x[n] = if w == SizeCode::W { v & 0xFFFF_FFFF } else { v };
    Ok(())
}

fn gpr_bits(o: &O) -> Result<u32> {
    match *o {
        O::Register(s, _) | O::RegisterOrSP(s, _) => Ok(if s == SizeCode::X { 64 } else { 32 }),
        _ => Err("expected a general register"),
    }
}

fn ebits(s: SIMDSizeCode) -> u32 {
    match s {
        SIMDSizeCode::B => 8,
        SIMDSizeCode::H => 16,
        SIMDSizeCode::S => 32,
        SIMDSizeCode::D => 64,
        SIMDSizeCode::Q => 128,
    }
}

/// A whole vector operand: (register, 16 bytes, element bits).
fn vec(o: &O) -> Result<(usize, bool, u32)> {
    match *o {
        O::SIMDRegisterElements(v, n, e) => Ok((n as usize, v == SIMDSizeCode::Q, ebits(e))),
        _ => Err("expected a vector"),
    }
}

/// One element: (register, element bits, lane).
fn elem(o: &O) -> Result<(usize, u32, u32)> {
    match *o {
        O::SIMDRegisterElementsLane(_, n, e, l) => Ok((n as usize, ebits(e), u32::from(l))),
        _ => Err("expected a vector element"),
    }
}

/// A scalar FP/SIMD register: (register, bits).
fn scalar(o: &O) -> Result<(usize, u32)> {
    match *o {
        O::SIMDRegister(s, n) => Ok((n as usize, ebits(s))),
        _ => Err("expected a scalar register"),
    }
}

fn lane(v: u128, i: u32, e: u32) -> u64 {
    ((v >> (i * e)) as u64) & mask(e)
}

fn put(v: &mut u128, i: u32, e: u32, x: u64) {
    let m = u128::from(mask(e)) << (i * e);
    *v = (*v & !m) | ((u128::from(x & mask(e))) << (i * e));
}

fn lanes(q: bool, e: u32) -> u32 {
    if q { 128 / e } else { 64 / e }
}

fn set_v(c: &mut Cpu, n: usize, v: u128, q: bool) {
    c.vr[n] = if q { v } else { v & LOW64 };
}

fn imm(o: &O) -> Result<u64> {
    match *o {
        O::Immediate(v) => Ok(u64::from(v)),
        O::Imm16(v) => Ok(u64::from(v)),
        O::Imm64(v) => Ok(v),
        O::ImmShift(v, s) => Ok(u64::from(v) << s),
        _ => Err("expected an immediate"),
    }
}

fn read(c: &Cpu, addr: u64, n: usize) -> u128 {
    if c.mem.is_null() {
        return 0;
    }
    let mut b = [0u8; 16];
    // SAFETY: the guest address space covers every guest address.
    unsafe { std::ptr::copy_nonoverlapping(c.mem.add(crate::host(addr)), b.as_mut_ptr(), n) };
    u128::from_le_bytes(b)
}

fn write(c: &Cpu, addr: u64, n: usize, v: u128) {
    if c.mem.is_null() {
        return;
    }
    let b = v.to_le_bytes();
    // SAFETY: as for `read`.
    unsafe { std::ptr::copy_nonoverlapping(b.as_ptr(), c.mem.add(crate::host(addr)), n) };
}

/// The effective address of a memory operand; applies any writeback.
fn address(c: &mut Cpu, o: &O) -> Result<u64> {
    let base = |n: u16| if n == 31 { 31 } else { n as usize };
    Ok(match *o {
        O::RegPreIndex(b, off, wb) => {
            let a = c.x[base(b)].wrapping_add(off as i64 as u64);
            if wb {
                c.x[base(b)] = a;
            }
            a
        }
        O::RegPostIndex(b, off) => {
            let a = c.x[base(b)];
            c.x[base(b)] = a.wrapping_add(off as i64 as u64);
            a
        }
        O::RegPostIndexReg(b, m) => {
            let a = c.x[base(b)];
            let step = if m == 31 { 0 } else { c.x[m as usize] };
            c.x[base(b)] = a.wrapping_add(step);
            a
        }
        O::RegRegOffset(b, i, _, style, amount) => {
            let idx = if i == 31 { 0 } else { c.x[i as usize] };
            let idx = match style {
                ShiftStyle::UXTW => idx & 0xFFFF_FFFF,
                ShiftStyle::SXTW => sext(idx, 32) as u64,
                _ => idx,
            };
            c.x[base(b)].wrapping_add(idx << amount)
        }
        _ => return Err("unsupported addressing mode"),
    })
}

/// Atomically replaces the value at `addr` by `f(old)` when it returns Some; returns old.
fn atomic(c: &Cpu, addr: u64, bytes: u32, f: impl Fn(u64) -> Option<u64>) -> u64 {
    if c.mem.is_null() {
        return 0;
    }
    // SAFETY: guest atomics are naturally aligned (the architecture faults otherwise).
    let p = unsafe { c.mem.add(crate::host(addr)) };
    let (ord, fail) = (Ordering::SeqCst, Ordering::SeqCst);
    unsafe {
        match bytes {
            1 => match AtomicU8::from_ptr(p).try_update(ord, fail, |v| f(u64::from(v)).map(|x| x as u8)) {
                Ok(v) | Err(v) => u64::from(v),
            },
            2 => match AtomicU16::from_ptr(p as *mut u16).try_update(ord, fail, |v| f(u64::from(v)).map(|x| x as u16)) {
                Ok(v) | Err(v) => u64::from(v),
            },
            4 => match AtomicU32::from_ptr(p as *mut u32).try_update(ord, fail, |v| f(u64::from(v)).map(|x| x as u32)) {
                Ok(v) | Err(v) => u64::from(v),
            },
            _ => match AtomicU64::from_ptr(p as *mut u64).try_update(ord, fail, f) {
                Ok(v) | Err(v) => v,
            },
        }
    }
}

// ---- floating point lanes ----

fn f32b(x: u64) -> f32 {
    f32::from_bits(x as u32)
}
fn f64b(x: u64) -> f64 {
    f64::from_bits(x)
}

/// Applies a binary FP operation lane by lane (32- or 64-bit lanes).
fn fp2(a: u128, b: u128, e: u32, q: bool, f: impl Fn(f64, f64) -> f64, g: impl Fn(f32, f32) -> f32) -> Result<u128> {
    let mut r = 0u128;
    for i in 0..lanes(q, e) {
        let (x, y) = (lane(a, i, e), lane(b, i, e));
        let v = match e {
            32 => u64::from(g(f32b(x), f32b(y)).to_bits()),
            64 => f(f64b(x), f64b(y)).to_bits(),
            _ => return Err("half-precision lanes"),
        };
        put(&mut r, i, e, v);
    }
    Ok(r)
}

/// The quiet bit and the infinity of `e`-bit floats.
fn nan_bits(e: u32) -> (u64, u64) {
    if e == 32 { (1 << 22, 0x7F80_0000) } else { (1 << 51, 0x7FF << 52) }
}

/// Arm's result (FPCR.DN = 0) for an operation on `ops` whose IEEE result is
/// `r`: the first signalling NaN operand, quieted, else the first quiet NaN;
/// with no NaN operand, an invalid operation gives the default NaN. The host's
/// NaNs differ (x86 keeps the first operand's NaN, its default NaN is negative).
fn arm_nan(ops: &[u64], r: u64, e: u32) -> u64 {
    let (quiet, inf) = nan_bits(e);
    let nan = |x: u64| (x & (mask(e) >> 1)) > inf;
    match ops.iter().find(|&&x| nan(x) && (x & quiet) == 0).or_else(|| ops.iter().find(|&&x| nan(x))) {
        Some(x) => x | quiet,
        None if nan(r) => inf | quiet,
        None => r,
    }
}

/// FPMulAdd(acc, a, b) with Arm's NaNs: `arm_nan` on (acc, a, b), except that
/// a quiet NaN addend gives the default NaN when a * b is inf * 0.
fn fma_nan(acc: u64, a: u64, b: u64, r: u64, e: u32) -> u64 {
    let (quiet, inf) = nan_bits(e);
    let abs = |x: u64| x & (mask(e) >> 1);
    let inf_zero = |x: u64, y: u64| abs(x) == inf && abs(y) == 0;
    if abs(acc) > inf && (acc & quiet) != 0 && (inf_zero(a, b) || inf_zero(b, a)) { inf | quiet } else { arm_nan(&[acc, a, b], r, e) }
}

/// `fp2` with Arm's NaN results.
fn fp2_arm(a: u128, b: u128, e: u32, q: bool, f: impl Fn(f64, f64) -> f64, g: impl Fn(f32, f32) -> f32) -> Result<u128> {
    fp2(
        a,
        b,
        e,
        q,
        |x, y| f64b(arm_nan(&[x.to_bits(), y.to_bits()], f(x, y).to_bits(), 64)),
        |x, y| f32b(arm_nan(&[x.to_bits().into(), y.to_bits().into()], g(x, y).to_bits().into(), 32)),
    )
}

fn fp1(a: u128, e: u32, q: bool, f: impl Fn(f64) -> f64, g: impl Fn(f32) -> f32) -> Result<u128> {
    fp2(a, 0, e, q, |x, _| f(x), |x, _| g(x))
}

/// FMAXNM and FMINNM (Arm's FPMaxNum, FPMinNum): a quiet NaN against anything
/// but a quiet NaN counts as `fill` (-inf for the max, +inf for the min), so a
/// number wins; the rest goes as for FMAX: a signalling NaN first, quieted,
/// else the first quiet NaN.
fn nm64(a: f64, b: f64, f: fn(f64, f64) -> f64, fill: f64) -> f64 {
    let quiet = |x: f64| x.is_nan() && x.to_bits() & (1 << 51) != 0;
    let (a, b) = match (quiet(a), quiet(b)) {
        (true, false) => (fill, b),
        (false, true) => (a, fill),
        _ => (a, b),
    };
    f64b(arm_nan(&[a.to_bits(), b.to_bits()], f(a, b).to_bits(), 64))
}
fn nm32(a: f32, b: f32, f: fn(f32, f32) -> f32, fill: f32) -> f32 {
    let quiet = |x: f32| x.is_nan() && x.to_bits() & (1 << 22) != 0;
    let (a, b) = match (quiet(a), quiet(b)) {
        (true, false) => (fill, b),
        (false, true) => (a, fill),
        _ => (a, b),
    };
    f32b(arm_nan(&[a.to_bits().into(), b.to_bits().into()], f(a, b).to_bits().into(), 32))
}
fn fmaxnm64(a: f64, b: f64) -> f64 {
    nm64(a, b, crate::fmax64, f64::NEG_INFINITY)
}
fn fminnm64(a: f64, b: f64) -> f64 {
    nm64(a, b, crate::fmin64, f64::INFINITY)
}
fn fmaxnm32(a: f32, b: f32) -> f32 {
    nm32(a, b, crate::fmax32, f32::NEG_INFINITY)
}
fn fminnm32(a: f32, b: f32) -> f32 {
    nm32(a, b, crate::fmin32, f32::INFINITY)
}

fn all_ones(c: bool, e: u32) -> u64 {
    if c { mask(e) } else { 0 }
}

/// Integer lanes, as signed when `signed`.
fn ival(x: u64, e: u32, signed: bool) -> i128 {
    if signed { i128::from(sext(x, e)) } else { i128::from(x) }
}

fn saturate(v: i128, e: u32, signed: bool) -> u64 {
    let (lo, hi) = if signed { (-(1i128 << (e - 1)), (1i128 << (e - 1)) - 1) } else { (0, (1i128 << e) - 1) };
    (v.clamp(lo, hi) as u64) & mask(e)
}

pub(crate) fn crc(mut acc: u32, data: u64, bytes: u32, poly: u32) -> u32 {
    for k in 0..bytes {
        acc ^= ((data >> (8 * k)) & 0xFF) as u32;
        for _ in 0..8 {
            acc = if acc & 1 != 0 { (acc >> 1) ^ poly } else { acc >> 1 };
        }
    }
    acc
}

// ---- system registers (yaxpeax numbering: o0:op1:CRn:CRm:op2) ----

const NZCV: u16 = 0x5A10;
const FPCR: u16 = 0x5A20;
const FPSR: u16 = 0x5A21;
const TPIDR_EL0: u16 = 0x5E82;
const DCZID_EL0: u16 = 0x5807;
const CTR_EL0: u16 = 0x5801;

/// The performance monitors (CRn 9, and the event registers of CRn 14): reads give 0, writes are dropped.
fn pmu(r: u16) -> bool {
    (0x5C80..0x5D00).contains(&r) || (0x5F40..0x5F80).contains(&r)
}

// ---- reciprocal estimates, as the architecture defines them ----

/// 9-bit fixed-point estimate of 1/a, a in [256, 512).
fn recip_estimate(a: u64) -> u64 {
    let a = a * 2 + 1;
    let b = (1u64 << 19) / a;
    b.div_ceil(2)
}

/// 9-bit fixed-point estimate of 1/sqrt(a), a in [128, 512).
fn rsqrt_estimate(a: u64) -> u64 {
    let a = if a < 256 { a * 2 + 1 } else { ((a >> 1) << 1) * 2 + 2 };
    let mut b = 512u64;
    while a * (b + 1) * (b + 1) < 1 << 28 {
        b += 1;
    }
    b.div_ceil(2)
}

/// FRECPE on a float of `bits` (32 or 64), with the default FPCR.
fn frecpe(x: u64, bits: u32) -> u64 {
    let (ebits, fbits) = if bits == 32 { (8, 23) } else { (11, 52) };
    let bias = (1u64 << (ebits - 1)) - 1;
    let sign = x >> (bits - 1) & 1;
    let exp = (x >> fbits) & mask(ebits);
    let frac = x & mask(fbits);
    let emax = mask(ebits);
    if exp == emax && frac != 0 {
        return x | 1 << (fbits - 1); // NaN: quieted
    }
    if exp == emax {
        return sign << (bits - 1); // infinity: zero
    }
    if exp == 0 && frac == 0 {
        return sign << (bits - 1) | emax << fbits; // zero: infinity
    }
    // The fraction as 52 bits, then the 8 bits after the leading one.
    let mut f = frac << (52 - fbits);
    let mut e = exp as i64;
    if e == 0 {
        if f >> 51 & 1 == 0 {
            e = -1;
            f = (f & mask(50)) << 2;
        } else {
            f = (f & mask(51)) << 1;
        }
    }
    let result_exp = 2 * bias as i64 - 1 - e;
    if result_exp >= emax as i64 {
        return sign << (bits - 1) | emax << fbits; // too small to invert: infinity
    }
    let est = recip_estimate(1 << 8 | (f >> 44) & 0xFF);
    let mut f = (est & 0xFF) << 44;
    let mut re = result_exp;
    if re == 0 {
        f = 1 << 51 | f >> 1;
    } else if re == -1 {
        f = 1 << 50 | f >> 2;
        re = 0;
    }
    sign << (bits - 1) | (re as u64 & emax) << fbits | f >> (52 - fbits)
}

/// FRSQRTE on a float of `bits` (32 or 64), with the default FPCR.
fn frsqrte(x: u64, bits: u32) -> u64 {
    let (ebits, fbits) = if bits == 32 { (8, 23) } else { (11, 52) };
    let bias = (1u64 << (ebits - 1)) - 1;
    let sign = x >> (bits - 1) & 1;
    let exp = (x >> fbits) & mask(ebits);
    let frac = x & mask(fbits);
    let emax = mask(ebits);
    let default_nan = emax << fbits | 1 << (fbits - 1);
    if exp == emax && frac != 0 {
        return x | 1 << (fbits - 1);
    }
    if exp == 0 && frac == 0 {
        return sign << (bits - 1) | emax << fbits; // zero: infinity of the same sign
    }
    if sign == 1 {
        return default_nan;
    }
    if exp == emax {
        return 0; // +infinity: +0
    }
    let mut f = frac << (52 - fbits);
    let mut e = exp as i64;
    if e == 0 {
        while f >> 51 & 1 == 0 {
            f = (f & mask(51)) << 1;
            e -= 1;
        }
        f = (f & mask(51)) << 1;
    }
    let scaled = if e & 1 == 0 { 1 << 8 | (f >> 44) & 0xFF } else { 1 << 7 | (f >> 45) & 0x7F };
    let result_exp = (3 * bias as i64 - 1 - e) / 2;
    let est = rsqrt_estimate(scaled);
    (result_exp as u64 & emax) << fbits | (est & 0xFF) << (fbits - 8)
}

// ---- half precision ----

fn f16_to_f32(h: u64) -> f32 {
    let (sign, exp, frac) = ((h >> 15) & 1, (h >> 10) & 0x1F, (h & 0x3FF) as u32);
    let s = (sign as u32) << 31;
    f32::from_bits(match exp {
        0 if frac == 0 => s,
        0 => {
            // Subnormal: the value is frac * 2^-24.
            let v = frac as f32 * 2f32.powi(-24);
            return if sign == 1 { -v } else { v };
        }
        31 => s | 0x7F80_0000 | frac << 13,
        e => s | ((e as u32 + 112) << 23) | frac << 13,
    })
}

fn f32_to_f16(x: f32) -> u64 {
    let b = x.to_bits();
    let sign = u64::from(b >> 31) << 15;
    if x.is_nan() {
        return sign | 0x7E00 | u64::from((b >> 13) & 0x1FF);
    }
    let a = x.abs();
    if a >= 65520.0 {
        return sign | 0x7C00; // rounds to infinity
    }
    if a < 2f32.powi(-14) {
        // Subnormal or zero: round a * 2^24 to nearest even.
        return sign | (a * 2f32.powi(24)).round_ties_even() as u64;
    }
    // Normal: round the 23-bit fraction to 10 bits, ties to even (a carry into the exponent is right).
    let e = u64::from((b >> 23) & 0xFF) - 112;
    let m = u64::from(b & 0x7F_FFFF);
    let v = (e << 10) | (m >> 13);
    let rest = m & 0x1FFF;
    let up = rest > 0x1000 || (rest == 0x1000 && v & 1 == 1);
    sign | (v + u64::from(up))
}

fn exec(c: &mut Cpu, _pc: u64, i: &Instruction) -> Result {
    let o = &i.operands;
    match i.opcode {
        // ---- integer leftovers ----
        C::ADC | C::ADCS | C::SBC | C::SBCS => {
            let bits = gpr_bits(&o[0])?;
            let (a, b) = (gpr(c, &o[1])?, gpr(c, &o[2])?);
            let sub = matches!(i.opcode, C::SBC | C::SBCS);
            let b = if sub { !b & mask(bits) } else { b };
            let wide = u128::from(a) + u128::from(b) + u128::from(c.c);
            let r = (wide as u64) & mask(bits);
            if matches!(i.opcode, C::ADCS | C::SBCS) {
                let sa = sext(a, bits) as i128 + sext(b, bits) as i128 + i128::from(c.c);
                c.n = r >> (bits - 1) & 1 == 1;
                c.z = r == 0;
                c.c = wide >> bits != 0;
                c.v = sa != i128::from(sext(r, bits));
            }
            set_gpr(c, &o[0], r)
        }
        C::CRC32B | C::CRC32H | C::CRC32W | C::CRC32X | C::CRC32CB | C::CRC32CH | C::CRC32CW | C::CRC32CX => {
            let (bytes, poly) = match i.opcode {
                C::CRC32B => (1, 0xEDB8_8320),
                C::CRC32H => (2, 0xEDB8_8320),
                C::CRC32W => (4, 0xEDB8_8320),
                C::CRC32X => (8, 0xEDB8_8320),
                C::CRC32CB => (1, 0x82F6_3B78),
                C::CRC32CH => (2, 0x82F6_3B78),
                C::CRC32CW => (4, 0x82F6_3B78),
                _ => (8, 0x82F6_3B78),
            };
            let r = crc(gpr(c, &o[1])? as u32, gpr(c, &o[2])?, bytes, poly);
            set_gpr(c, &o[0], u64::from(r))
        }
        C::MRS => {
            let O::SystemReg(r) = o[1] else { return Err("mrs operand") };
            let v = match r {
                NZCV => (u64::from(c.n) << 31) | (u64::from(c.z) << 30) | (u64::from(c.c) << 29) | (u64::from(c.v) << 28),
                FPCR => c.fpcr,
                FPSR => c.fpsr,
                TPIDR_EL0 => c.tpidr,
                DCZID_EL0 => 4,           // DC ZVA zeroes 64 bytes
                CTR_EL0 => 0x8444_C004, // 64-byte cache lines
                r if pmu(r) => 0,
                _ => return Err("mrs of an unsupported system register"),
            };
            set_gpr(c, &o[0], v)
        }
        C::MSR => {
            let O::SystemReg(r) = o[0] else { return Err("msr operand") };
            let v = gpr(c, &o[1])?;
            match r {
                NZCV => crate::set_nzcv(c, (v >> 28) as u8),
                // D36: the game never sets a control bit (rounding, flush to zero,
                // default NaN, traps); one that does would change results we do not model.
                FPCR if v != 0 => return Err("msr FPCR with control bits, which are not applied (D36)"),
                FPCR => c.fpcr = v,
                FPSR => c.fpsr = v,
                TPIDR_EL0 => c.tpidr = v,
                r if pmu(r) => {}
                _ => return Err("msr of an unsupported system register"),
            }
            Ok(())
        }
        // Cache maintenance (DC CIVAC and the like) and memory tagging do nothing on the host.
        C::SYS(_) | C::STG => Ok(()),
        C::FRECPE | C::FRSQRTE => {
            let (d, q, e, src, one) = match o[0] {
                O::SIMDRegister(..) => {
                    let ((d, bits), (s, _)) = (scalar(&o[0])?, scalar(&o[1])?);
                    (d, false, bits, c.vr[s], true)
                }
                _ => {
                    let ((d, q, e), (s, _, _)) = (vec(&o[0])?, vec(&o[1])?);
                    (d, q, e, c.vr[s], false)
                }
            };
            if e != 32 && e != 64 {
                return Err("half-precision estimate");
            }
            let mut r = 0;
            for j in 0..if one { 1 } else { lanes(q, e) } {
                let x = lane(src, j, e);
                put(&mut r, j, e, if i.opcode == C::FRECPE { frecpe(x, e) } else { frsqrte(x, e) });
            }
            set_v(c, d, r, q);
            Ok(())
        }
        // Pointer authentication is never checked: signing and stripping keep the value,
        // and PACGA gives a deterministic code so that a later comparison holds.
        C::PACIA | C::PACIB | C::AUTIA | C::AUTIB | C::AUTIA1716 | C::AUTIB1716 | C::AUTIASP => Ok(()),
        C::PACGA => {
            let v = (gpr(c, &o[1])? ^ gpr(c, &o[2])?.rotate_left(17)) & 0xFFFF_FFFF_0000_0000;
            set_gpr(c, &o[0], v)
        }

        // ---- LSE atomics: [Rs, Rt, [Xn]] ----
        C::LDADD(_) | C::LDADDB(_) | C::LDADDH(_) | C::LDCLR(_) | C::LDCLRB(_) | C::LDCLRH(_) | C::LDEOR(_) | C::LDEORB(_) | C::LDEORH(_)
        | C::LDSET(_) | C::LDSETB(_) | C::LDSETH(_) | C::LDSMAX(_) | C::LDSMAXB(_) | C::LDSMAXH(_) | C::LDSMIN(_) | C::LDSMINB(_)
        | C::LDSMINH(_) | C::LDUMAX(_) | C::LDUMAXB(_) | C::LDUMAXH(_) | C::LDUMIN(_) | C::LDUMINB(_) | C::LDUMINH(_) | C::SWP(_)
        | C::SWPB(_) | C::SWPH(_) => {
            let bytes = match i.opcode {
                C::LDADDB(_) | C::LDCLRB(_) | C::LDEORB(_) | C::LDSETB(_) | C::LDSMAXB(_) | C::LDSMINB(_) | C::LDUMAXB(_) | C::LDUMINB(_) | C::SWPB(_) => 1,
                C::LDADDH(_) | C::LDCLRH(_) | C::LDEORH(_) | C::LDSETH(_) | C::LDSMAXH(_) | C::LDSMINH(_) | C::LDUMAXH(_) | C::LDUMINH(_) | C::SWPH(_) => 2,
                _ => gpr_bits(&o[1])? / 8,
            };
            let bits = 8 * bytes;
            let s = gpr(c, &o[0])? & mask(bits);
            let a = address(c, &o[2])?;
            let sx = sext(s, bits);
            let old = atomic(c, a, bytes, |v| {
                let vx = sext(v, bits);
                Some(
                    match i.opcode {
                        C::LDADD(_) | C::LDADDB(_) | C::LDADDH(_) => v.wrapping_add(s),
                        C::LDCLR(_) | C::LDCLRB(_) | C::LDCLRH(_) => v & !s,
                        C::LDEOR(_) | C::LDEORB(_) | C::LDEORH(_) => v ^ s,
                        C::LDSET(_) | C::LDSETB(_) | C::LDSETH(_) => v | s,
                        C::LDSMAX(_) | C::LDSMAXB(_) | C::LDSMAXH(_) => vx.max(sx) as u64,
                        C::LDSMIN(_) | C::LDSMINB(_) | C::LDSMINH(_) => vx.min(sx) as u64,
                        C::LDUMAX(_) | C::LDUMAXB(_) | C::LDUMAXH(_) => v.max(s),
                        C::LDUMIN(_) | C::LDUMINB(_) | C::LDUMINH(_) => v.min(s),
                        _ => s,
                    } & mask(bits),
                )
            });
            set_gpr(c, &o[1], old)
        }
        C::CAS(_) | C::CASB(_) | C::CASH(_) => {
            let bytes = match i.opcode {
                C::CASB(_) => 1,
                C::CASH(_) => 2,
                _ => gpr_bits(&o[1])? / 8,
            };
            let bits = 8 * bytes;
            let (expected, new) = (gpr(c, &o[0])? & mask(bits), gpr(c, &o[1])? & mask(bits));
            let a = address(c, &o[2])?;
            let old = atomic(c, a, bytes, |v| (v == expected).then_some(new));
            set_gpr(c, &o[0], old)
        }
        C::LDAXR | C::LDXR => {
            let bytes = gpr_bits(&o[0])? / 8;
            let a = address(c, &o[1])?;
            let v = read(c, a, bytes as usize) as u64;
            set_gpr(c, &o[0], v)
        }
        // ponytail: one guest thread runs at a time, so a store-exclusive always succeeds.
        C::STLXR | C::STXR => {
            let bytes = gpr_bits(&o[1])? / 8;
            let v = gpr(c, &o[1])?;
            let a = address(c, &o[2])?;
            write(c, a, bytes as usize, u128::from(v));
            set_gpr(c, &o[0], 0)
        }

        // ---- moves between lanes ----
        C::INS => {
            let (d, e, l) = elem(&o[0])?;
            let x = match o[1] {
                O::SIMDRegisterElementsLane(..) => {
                    let (s, _, k) = elem(&o[1])?;
                    lane(c.vr[s], k, e)
                }
                _ => gpr(c, &o[1])?,
            };
            put(&mut c.vr[d], l, e, x);
            Ok(())
        }
        C::DUP => match (&o[0], &o[1]) {
            (O::SIMDRegister(..), _) => {
                let (d, _) = scalar(&o[0])?;
                let (s, e, k) = elem(&o[1])?;
                c.vr[d] = u128::from(lane(c.vr[s], k, e));
                Ok(())
            }
            (O::SIMDRegisterElements(..), O::SIMDRegisterElementsLane(..)) => {
                let (d, q, e) = vec(&o[0])?;
                let (s, _, k) = elem(&o[1])?;
                let x = lane(c.vr[s], k, e);
                let mut r = 0;
                (0..lanes(q, e)).for_each(|j| put(&mut r, j, e, x));
                set_v(c, d, r, q);
                Ok(())
            }
            _ => {
                let (d, q, e) = vec(&o[0])?;
                let x = gpr(c, &o[1])?;
                let mut r = 0;
                (0..lanes(q, e)).for_each(|j| put(&mut r, j, e, x));
                set_v(c, d, r, q);
                Ok(())
            }
        },
        C::SMOV => {
            let (s, e, k) = elem(&o[1])?;
            let v = sext(lane(c.vr[s], k, e), e) as u64;
            set_gpr(c, &o[0], v)
        }
        C::EXT => {
            let ((d, q, _), (a, _, _), (b, _, _)) = (vec(&o[0])?, vec(&o[1])?, vec(&o[2])?);
            let k = imm(&o[3])? as u32 * 8;
            let width = if q { 128 } else { 64 };
            let (x, y) = if q { (c.vr[a], c.vr[b]) } else { (c.vr[a] & LOW64, c.vr[b] & LOW64) };
            let r = if k == 0 { x } else { (x >> k) | (y << (width - k)) };
            set_v(c, d, r, q);
            Ok(())
        }
        C::REV64 | C::REV32 | C::REV16 => {
            let ((d, q, e), (s, _, _)) = (vec(&o[0])?, vec(&o[1])?);
            let group = match i.opcode {
                C::REV64 => 64,
                C::REV32 => 32,
                _ => 16,
            };
            let per = group / e;
            let v = c.vr[s];
            let mut r = 0;
            for j in 0..lanes(q, e) {
                let base = j / per * per;
                put(&mut r, base + (per - 1 - j % per), e, lane(v, j, e));
            }
            set_v(c, d, r, q);
            Ok(())
        }
        C::ZIP1 | C::ZIP2 | C::UZP1 | C::UZP2 | C::TRN1 | C::TRN2 => {
            let ((d, q, e), (a, _, _), (b, _, _)) = (vec(&o[0])?, vec(&o[1])?, vec(&o[2])?);
            let n = lanes(q, e);
            let (x, y) = (c.vr[a], c.vr[b]);
            let cat = |k: u32| if k < n { lane(x, k, e) } else { lane(y, k - n, e) };
            let mut r = 0;
            for j in 0..n {
                let v = match i.opcode {
                    C::ZIP1 => if j % 2 == 0 { lane(x, j / 2, e) } else { lane(y, j / 2, e) },
                    C::ZIP2 => if j % 2 == 0 { lane(x, n / 2 + j / 2, e) } else { lane(y, n / 2 + j / 2, e) },
                    C::UZP1 => cat(2 * j),
                    C::UZP2 => cat(2 * j + 1),
                    C::TRN1 => if j % 2 == 0 { lane(x, j, e) } else { lane(y, j - 1, e) },
                    _ => if j % 2 == 0 { lane(x, j + 1, e) } else { lane(y, j, e) },
                };
                put(&mut r, j, e, v);
            }
            set_v(c, d, r, q);
            Ok(())
        }
        C::TBL | C::TBX => {
            let (d, q, _) = vec(&o[0])?;
            let O::SIMDRegisterGroup(_, first, _, count) = o[1] else { return Err("tbl table") };
            let (idx, _, _) = vec(&o[2])?;
            let table: Vec<u8> = (0..count as usize).flat_map(|k| c.vr[(first as usize + k) % 32].to_le_bytes()).collect();
            let mut r = if i.opcode == C::TBX { c.vr[d] } else { 0 };
            for j in 0..lanes(q, 8) {
                let t = lane(c.vr[idx], j, 8) as usize;
                if t < table.len() {
                    put(&mut r, j, 8, u64::from(table[t]));
                } else if i.opcode == C::TBL {
                    put(&mut r, j, 8, 0);
                }
            }
            set_v(c, d, r, q);
            Ok(())
        }

        // ---- structure loads and stores ----
        C::LD1 | C::ST1 | C::LD2 | C::ST2 | C::LD3 | C::ST3 | C::LD4 | C::ST4 | C::LD1R | C::LD2R | C::LD3R | C::LD4R => {
            let load = matches!(i.opcode, C::LD1 | C::LD2 | C::LD3 | C::LD4 | C::LD1R | C::LD2R | C::LD3R | C::LD4R);
            let replicate = matches!(i.opcode, C::LD1R | C::LD2R | C::LD3R | C::LD4R);
            let structure = match i.opcode {
                C::LD1 | C::ST1 | C::LD1R => 1,
                C::LD2 | C::ST2 | C::LD2R => 2,
                C::LD3 | C::ST3 | C::LD3R => 3,
                _ => 4,
            };
            match o[0] {
                O::SIMDRegisterGroup(v, first, es, count) => {
                    let (q, e) = (v == SIMDSizeCode::Q, ebits(es));
                    let bytes = (e / 8) as u64;
                    let a = address(c, &o[1])?;
                    let reg = |k: u32| (first as usize + k as usize) % 32;
                    let n = lanes(q, e);
                    if replicate {
                        for k in 0..count as u32 {
                            let x = read(c, a + u64::from(k) * bytes, bytes as usize) as u64;
                            let mut r = 0;
                            (0..n).for_each(|j| put(&mut r, j, e, x));
                            set_v(c, reg(k), r, q);
                        }
                    } else if structure == 1 {
                        // Consecutive whole registers.
                        let size = if q { 16 } else { 8 };
                        for k in 0..count as u32 {
                            let at = a + u64::from(k) * size as u64;
                            if load {
                                let x = read(c, at, size);
                                set_v(c, reg(k), x, q);
                            } else {
                                write(c, at, size, c.vr[reg(k)]);
                            }
                        }
                    } else {
                        // Interleaved elements: element j of register k is at (j * count + k).
                        for j in 0..n {
                            for k in 0..structure {
                                let at = a + (u64::from(j) * u64::from(structure) + u64::from(k)) * bytes;
                                if load {
                                    let x = read(c, at, bytes as usize) as u64;
                                    let mut r = if j == 0 { 0 } else { c.vr[reg(k)] };
                                    put(&mut r, j, e, x);
                                    set_v(c, reg(k), r, q);
                                } else {
                                    write(c, at, bytes as usize, u128::from(lane(c.vr[reg(k)], j, e)));
                                }
                            }
                        }
                    }
                    Ok(())
                }
                O::SIMDRegisterGroupLane(first, es, count, l) => {
                    let e = ebits(es);
                    let bytes = (e / 8) as u64;
                    let a = address(c, &o[1])?;
                    for k in 0..u32::from(count) {
                        let r = (first as usize + k as usize) % 32;
                        let at = a + u64::from(k) * bytes;
                        if load {
                            let x = read(c, at, bytes as usize) as u64;
                            put(&mut c.vr[r], u32::from(l), e, x);
                        } else {
                            write(c, at, bytes as usize, u128::from(lane(c.vr[r], u32::from(l), e)));
                        }
                    }
                    Ok(())
                }
                _ => Err("structure operand"),
            }
        }

        // ---- vector FP ----
        C::FADD | C::FSUB | C::FMUL | C::FDIV | C::FMAX | C::FMIN | C::FMAXNM | C::FMINNM | C::FABD | C::FMULX | C::FRECPS | C::FRSQRTS
        | C::FMLA | C::FMLS | C::FADDP | C::FMAXP | C::FMINP | C::FMAXNMP | C::FMINMNP => fp_vector(c, i),
        C::FNEG | C::FABS | C::FSQRT | C::FRINTN | C::FRINTM | C::FRINTP | C::FRINTZ | C::FRINTA | C::FRINTX | C::FRINTI => {
            let ((d, q, e), (s, _, _)) = (vec(&o[0])?, vec(&o[1])?);
            if i.opcode == C::FNEG && e == 16 {
                let mut r = 0;
                (0..lanes(q, 16)).for_each(|j| put(&mut r, j, 16, lane(c.vr[s], j, 16) ^ 0x8000));
                set_v(c, d, r, q);
                return Ok(());
            }
            let r = match i.opcode {
                C::FNEG => fp1(c.vr[s], e, q, |x| -x, |x| -x),
                C::FABS => fp1(c.vr[s], e, q, f64::abs, f32::abs),
                C::FSQRT => fp1(c.vr[s], e, q, f64::sqrt, f32::sqrt),
                C::FRINTM => fp1(c.vr[s], e, q, f64::floor, f32::floor),
                C::FRINTP => fp1(c.vr[s], e, q, f64::ceil, f32::ceil),
                C::FRINTZ => fp1(c.vr[s], e, q, f64::trunc, f32::trunc),
                C::FRINTA => fp1(c.vr[s], e, q, f64::round, f32::round),
                _ => fp1(c.vr[s], e, q, f64::round_ties_even, f32::round_ties_even),
            }?;
            set_v(c, d, r, q);
            Ok(())
        }
        C::FMAXNMV | C::FMINNMV | C::FMAXV | C::FMINV => {
            let (d, _) = scalar(&o[0])?;
            let (s, q, e) = vec(&o[1])?;
            if e != 32 {
                return Err("half-precision reduction");
            }
            let mut acc = f32b(lane(c.vr[s], 0, e));
            for j in 1..lanes(q, e) {
                let x = f32b(lane(c.vr[s], j, e));
                acc = match i.opcode {
                    C::FMAXNMV => fmaxnm32(acc, x),
                    C::FMINNMV => fminnm32(acc, x),
                    C::FMAXV => crate::fmax32(acc, x),
                    _ => crate::fmin32(acc, x),
                };
            }
            c.vr[d] = u128::from(acc.to_bits());
            Ok(())
        }
        C::FCMEQ | C::FCMGE | C::FCMGT | C::FCMLE | C::FCMLT | C::FACGE | C::FACGT => {
            let ((d, q, e), (a, _, _)) = (vec(&o[0])?, vec(&o[1])?);
            let b = match o[2] {
                O::SIMDRegisterElements(..) => Some(c.vr[vec(&o[2])?.0]),
                _ => None, // compare with +0.0
            };
            let mut r = 0;
            for j in 0..lanes(q, e) {
                let x = lane(c.vr[a], j, e);
                let y = b.map_or(0, |b| lane(b, j, e));
                let (x, y) = match e {
                    32 => (f64::from(f32b(x)), f64::from(f32b(y))),
                    64 => (f64b(x), f64b(y)),
                    _ => return Err("half-precision compare"),
                };
                let t = match i.opcode {
                    C::FCMEQ => x == y,
                    C::FCMGE => x >= y,
                    C::FCMGT => x > y,
                    C::FCMLE => x <= y,
                    C::FCMLT => x < y,
                    C::FACGE => x.abs() >= y.abs(),
                    _ => x.abs() > y.abs(),
                };
                put(&mut r, j, e, all_ones(t, e));
            }
            set_v(c, d, r, q);
            Ok(())
        }
        C::SCVTF | C::UCVTF | C::FCVTZS | C::FCVTZU | C::FCVTMS | C::FCVTMU | C::FCVTPS | C::FCVTPU | C::FCVTAS | C::FCVTAU | C::FCVTNS
        | C::FCVTNU => convert(c, i),
        C::FCVT => {
            // Scalar conversions with half precision (single <-> double is translated).
            let ((d, db), (s, sb)) = (scalar(&o[0])?, scalar(&o[1])?);
            let x = c.vr[s] as u64 & mask(sb);
            let v = match (sb, db) {
                (16, 32) => u64::from(f16_to_f32(x).to_bits()),
                (16, 64) => f64::from(f16_to_f32(x)).to_bits(),
                (32, 16) => f32_to_f16(f32b(x)),
                (64, 16) => f32_to_f16(f64b(x) as f32),
                _ => return Err("fcvt form"),
            };
            c.vr[d] = u128::from(v);
            Ok(())
        }
        C::FCVTL | C::FCVTL2 if matches!(o[1], O::SIMDRegisterElements(_, _, SIMDSizeCode::H)) => {
            let ((d, _, _), (s, _, _)) = (vec(&o[0])?, vec(&o[1])?);
            let base = if i.opcode == C::FCVTL2 { 4 } else { 0 };
            let mut r = 0;
            for j in 0..4 {
                put(&mut r, j, 32, u64::from(f16_to_f32(lane(c.vr[s], base + j, 16)).to_bits()));
            }
            c.vr[d] = r;
            Ok(())
        }
        C::FCVTN | C::FCVTN2 if matches!(o[0], O::SIMDRegisterElements(_, _, SIMDSizeCode::H)) => {
            let ((d, _, _), (s, _, _)) = (vec(&o[0])?, vec(&o[1])?);
            let upper = i.opcode == C::FCVTN2;
            let mut r = if upper { c.vr[d] & LOW64 } else { 0 };
            for j in 0..4 {
                put(&mut r, if upper { 4 + j } else { j }, 16, f32_to_f16(f32b(lane(c.vr[s], j, 32))));
            }
            c.vr[d] = r;
            Ok(())
        }
        C::CLZ => {
            let ((d, q, e), (s, _, _)) = (vec(&o[0])?, vec(&o[1])?);
            let mut r = 0;
            for j in 0..lanes(q, e) {
                put(&mut r, j, e, u64::from(lane(c.vr[s], j, e).leading_zeros()) - u64::from(64 - e));
            }
            set_v(c, d, r, q);
            Ok(())
        }
        C::ADDHN | C::ADDHN2 | C::SUBHN | C::SUBHN2 | C::RADDHN | C::RADDHN2 | C::RSUBHN | C::RSUBHN2 => {
            let ((d, _, e), (a, _, _), (b, _, _)) = (vec(&o[0])?, vec(&o[1])?, vec(&o[2])?);
            let upper = matches!(i.opcode, C::ADDHN2 | C::SUBHN2 | C::RADDHN2 | C::RSUBHN2);
            let sub = matches!(i.opcode, C::SUBHN | C::SUBHN2 | C::RSUBHN | C::RSUBHN2);
            let round = if matches!(i.opcode, C::RADDHN | C::RADDHN2 | C::RSUBHN | C::RSUBHN2) { 1u64 << (e - 1) } else { 0 };
            let wide = 2 * e;
            let count = 64 / e;
            let mut r = if upper { c.vr[d] & LOW64 } else { 0 };
            for j in 0..count {
                let (x, y) = (lane(c.vr[a], j, wide), lane(c.vr[b], j, wide));
                let v = if sub { x.wrapping_sub(y) } else { x.wrapping_add(y) }.wrapping_add(round) & mask(wide);
                put(&mut r, if upper { count + j } else { j }, e, v >> e);
            }
            c.vr[d] = r;
            Ok(())
        }
        C::PMULL | C::PMULL2 => {
            let ((d, _, e), (a, _, ae), (b, _, _)) = (vec(&o[0])?, vec(&o[1])?, vec(&o[2])?);
            let upper = i.opcode == C::PMULL2;
            let count = 64 / ae;
            let offset = if upper { count } else { 0 };
            let mut r = 0u128;
            for j in 0..count {
                let (x, y) = (lane(c.vr[a], offset + j, ae), lane(c.vr[b], offset + j, ae));
                let mut p = 0u128;
                for k in 0..ae {
                    if y >> k & 1 == 1 {
                        p ^= u128::from(x) << k;
                    }
                }
                if e == 128 {
                    r = p;
                } else {
                    put(&mut r, j, e, p as u64);
                }
            }
            c.vr[d] = r;
            Ok(())
        }
        C::FCVTL | C::FCVTL2 => {
            let ((d, _, _), (s, _, se)) = (vec(&o[0])?, vec(&o[1])?);
            if se != 32 {
                return Err("half-precision widening");
            }
            let base = if i.opcode == C::FCVTL2 { 2 } else { 0 };
            let mut r = 0;
            for j in 0..2 {
                put(&mut r, j, 64, f64::from(f32b(lane(c.vr[s], base + j, 32))).to_bits());
            }
            c.vr[d] = r;
            Ok(())
        }
        C::FCVTN | C::FCVTN2 => {
            let ((d, _, de), (s, _, _)) = (vec(&o[0])?, vec(&o[1])?);
            if de != 32 {
                return Err("half-precision narrowing");
            }
            let mut r = if i.opcode == C::FCVTN2 { c.vr[d] & LOW64 } else { 0 };
            let base = if i.opcode == C::FCVTN2 { 2 } else { 0 };
            for j in 0..2 {
                put(&mut r, base + j, 32, u64::from((f64b(lane(c.vr[s], j, 64)) as f32).to_bits()));
            }
            c.vr[d] = r;
            Ok(())
        }

        // SQSHL by register (vector); the immediate form is a shift_imm.
        C::SQSHL if matches!(o[2], O::SIMDRegisterElements(..)) => int_vector(c, i),
        // Scalar pairwise add: ADDP d0, v1.2d adds the two 64-bit lanes.
        C::ADDP if matches!(o[0], O::SIMDRegister(..)) => {
            let ((d, _), (a, _, _)) = (scalar(&o[0])?, vec(&o[1])?);
            let v = c.vr[a];
            c.vr[d] = u128::from((v as u64).wrapping_add((v >> 64) as u64));
            Ok(())
        }

        // ---- vector integer ----
        C::ADD | C::SUB | C::MUL | C::MLA | C::MLS | C::SMAX | C::SMIN | C::UMAX | C::UMIN | C::SQADD | C::UQADD | C::SQSUB | C::UQSUB
        | C::SHADD | C::UHADD | C::SHSUB | C::URHADD | C::SRHADD | C::SABD | C::UABD | C::SABA | C::UABA | C::CMEQ | C::CMGE | C::CMGT
        | C::CMHI | C::CMHS | C::CMLE | C::CMLT | C::CMTST | C::SQDMULH | C::SQRDMULH | C::USHL | C::SSHL | C::ADDP | C::UMAXP
        | C::UMINP | C::SMAXP | C::SMINP | C::AND | C::ORR | C::BIC | C::EOR | C::ORN | C::NOT | C::NEG | C::ABS | C::CNT => int_vector(c, i),
        C::SHL | C::USHR | C::SSHR | C::USRA | C::SSRA | C::SRSHR | C::URSHR | C::SRSRA | C::URSRA | C::SRI | C::SLI | C::SQSHL
        | C::SQSHLU => shift_imm(c, i),
        C::XTN | C::XTN2 | C::SQXTN | C::SQXTN2 | C::UQXTN | C::UQXTN2 | C::SQXTUN | C::SQXTUN2 | C::SHRN | C::SHRN2 | C::RSHRN
        | C::RSHRN2 | C::SQSHRN | C::SQSHRN2 | C::SQRSHRN | C::SQRSHRN2 | C::UQSHRN | C::UQRSHRN | C::SQSHRUN | C::SQRSHRUN => narrow(c, i),
        C::USHLL | C::USHLL2 | C::SSHLL | C::SSHLL2 | C::SMULL | C::SMULL2 | C::UMULL | C::UMULL2 | C::SMLAL | C::SMLAL2 | C::UMLAL
        | C::UMLAL2 | C::SMLSL | C::SMLSL2 | C::UMLSL | C::UMLSL2 | C::SADDL | C::SADDL2 | C::UADDL | C::UADDL2 | C::SSUBL | C::SSUBL2
        | C::USUBL | C::USUBL2 | C::SADDW | C::SADDW2 | C::UADDW | C::UADDW2 | C::SSUBW | C::SSUBW2 | C::USUBW | C::USUBW2
        | C::SABDL | C::UABDL | C::SQDMLAL | C::SQDMLAL2 | C::SHLL | C::SHLL2 => widen(c, i),
        C::ADDV | C::UMAXV | C::UMINV | C::SMAXV | C::SMINV | C::UADDLV | C::SADDLV => {
            let (d, _) = scalar(&o[0])?;
            let (s, q, e) = vec(&o[1])?;
            let signed = matches!(i.opcode, C::SMAXV | C::SMINV | C::SADDLV);
            let vals: Vec<i128> = (0..lanes(q, e)).map(|j| ival(lane(c.vr[s], j, e), e, signed)).collect();
            let r = match i.opcode {
                C::ADDV => (vals.iter().sum::<i128>() as u64) & mask(e),
                C::UADDLV | C::SADDLV => (vals.iter().sum::<i128>() as u64) & mask(2 * e),
                C::UMAXV | C::SMAXV => (*vals.iter().max().expect("lanes") as u64) & mask(e),
                _ => (*vals.iter().min().expect("lanes") as u64) & mask(e),
            };
            c.vr[d] = u128::from(r);
            Ok(())
        }
        C::UADDLP | C::SADDLP | C::UADALP | C::SADALP => {
            let ((d, q, e), (s, _, _)) = (vec(&o[0])?, vec(&o[1])?);
            let (signed, acc) = (matches!(i.opcode, C::SADDLP | C::SADALP), matches!(i.opcode, C::UADALP | C::SADALP));
            let half = e / 2;
            let mut r = 0;
            for j in 0..lanes(q, e) {
                let sum = ival(lane(c.vr[s], 2 * j, half), half, signed) + ival(lane(c.vr[s], 2 * j + 1, half), half, signed);
                let base = if acc { ival(lane(c.vr[d], j, e), e, signed) } else { 0 };
                put(&mut r, j, e, (base + sum) as u64);
            }
            set_v(c, d, r, q);
            Ok(())
        }
        C::UDOT | C::SDOT => {
            let ((d, q, _), (a, _, _)) = (vec(&o[0])?, vec(&o[1])?);
            let signed = i.opcode == C::SDOT;
            let b_lanes: Vec<u64> = match o[2] {
                O::SIMDRegisterElementsLane(_, b, _, l) => (0..lanes(q, 32)).map(|_| lane(c.vr[b as usize], u32::from(l), 32)).collect(),
                _ => {
                    let (b, _, _) = vec(&o[2])?;
                    (0..lanes(q, 32)).map(|j| lane(c.vr[b], j, 32)).collect()
                }
            };
            let mut r = c.vr[d];
            for j in 0..lanes(q, 32) {
                let (x, y) = (lane(c.vr[a], j, 32), b_lanes[j as usize]);
                let mut sum = ival(lane(r, j, 32), 32, false);
                for k in 0..4 {
                    sum += ival(x >> (8 * k) & 0xFF, 8, signed) * ival(y >> (8 * k) & 0xFF, 8, signed);
                }
                put(&mut r, j, 32, sum as u64);
            }
            set_v(c, d, r, q);
            Ok(())
        }
        C::MOVI | C::MVNI => {
            // Forms the generator leaves out (an ORR/BIC with immediate decodes as those opcodes).
            Err("vector immediate form")
        }
        _ => Err("opcode not covered"),
    }
}

/// Binary FP on vectors or on a scalar by element; FMLA/FMLS accumulate, pairwise forms fold.
fn fp_vector(c: &mut Cpu, i: &Instruction) -> Result {
    let o = &i.operands;
    // Destination and first source: vectors, or scalars (by-element or scalar pairwise forms).
    let (d, q, e, a) = match (&o[0], &o[1]) {
        (O::SIMDRegisterElements(..), O::SIMDRegisterElements(..)) => {
            let ((d, q, e), (a, _, _)) = (vec(&o[0])?, vec(&o[1])?);
            (d, q, e, c.vr[a])
        }
        (O::SIMDRegister(..), O::SIMDRegister(..)) => {
            let ((d, bits), (a, _)) = (scalar(&o[0])?, scalar(&o[1])?);
            (d, false, bits, c.vr[a] & u128::from(mask(bits)))
        }
        (O::SIMDRegister(..), O::SIMDRegisterElements(..)) => {
            // Scalar pairwise: FADDP s0, v1.2s.
            let (d, bits) = scalar(&o[0])?;
            let (a, _, e) = vec(&o[1])?;
            let (x, y) = (lane(c.vr[a], 0, e), lane(c.vr[a], 1, e));
            let r = match i.opcode {
                C::FADDP => fp2_arm(u128::from(x), u128::from(y), e, false, |x, y| x + y, |x, y| x + y)?,
                C::FMAXNMP => fp2(u128::from(x), u128::from(y), e, false, fmaxnm64, fmaxnm32)?,
                C::FMINMNP => fp2(u128::from(x), u128::from(y), e, false, fminnm64, fminnm32)?,
                C::FMAXP => fp2(u128::from(x), u128::from(y), e, false, crate::fmax64, crate::fmax32)?,
                C::FMINP => fp2(u128::from(x), u128::from(y), e, false, crate::fmin64, crate::fmin32)?,
                _ => return Err("scalar pairwise form"),
            };
            c.vr[d] = r & u128::from(mask(bits));
            return Ok(());
        }
        _ => return Err("FP vector operands"),
    };
    let scalar_form = matches!(o[0], O::SIMDRegister(..));
    let n = if scalar_form { 1 } else { lanes(q, e) };
    let b = match o[2] {
        O::SIMDRegisterElementsLane(..) => {
            let (b, be, l) = elem(&o[2])?;
            let x = lane(c.vr[b], l, be);
            let mut r = 0;
            (0..n).for_each(|j| put(&mut r, j, e, x));
            r
        }
        O::SIMDRegisterElements(..) => c.vr[vec(&o[2])?.0],
        O::SIMDRegister(..) => c.vr[scalar(&o[2])?.0],
        _ => return Err("FP vector operand"),
    };
    let acc = c.vr[d];
    let r = match i.opcode {
        C::FADD => fp2_arm(a, b, e, q, |x, y| x + y, |x, y| x + y),
        C::FSUB => fp2_arm(a, b, e, q, |x, y| x - y, |x, y| x - y),
        C::FMUL | C::FMULX => fp2_arm(a, b, e, q, |x, y| x * y, |x, y| x * y),
        C::FDIV => fp2_arm(a, b, e, q, |x, y| x / y, |x, y| x / y),
        C::FMAX => fp2(a, b, e, q, crate::fmax64, crate::fmax32),
        C::FMIN => fp2(a, b, e, q, crate::fmin64, crate::fmin32),
        C::FMAXNM => fp2(a, b, e, q, fmaxnm64, fmaxnm32),
        C::FMINNM => fp2(a, b, e, q, fminnm64, fminnm32),
        C::FABD => fp2(a, b, e, q, |x, y| (x - y).abs(), |x, y| (x - y).abs()),
        C::FRECPS => fp2(a, b, e, q, |x, y| (-x).mul_add(y, 2.0), |x, y| (-x).mul_add(y, 2.0)),
        C::FRSQRTS => fp2(a, b, e, q, |x, y| (-x).mul_add(y, 3.0) / 2.0, |x, y| (-x).mul_add(y, 3.0) / 2.0),
        C::FMLA | C::FMLS => {
            let neg = i.opcode == C::FMLS;
            let mut r = 0;
            for j in 0..n {
                let (x, y, z) = (lane(a, j, e), lane(b, j, e), lane(acc, j, e));
                // FMLS negates the first factor before FPMulAdd, NaN or not.
                let v = match e {
                    32 => {
                        let x = if neg { -f32b(x) } else { f32b(x) };
                        fma_nan(z, x.to_bits().into(), y, x.mul_add(f32b(y), f32b(z)).to_bits().into(), 32)
                    }
                    64 => {
                        let x = if neg { -f64b(x) } else { f64b(x) };
                        fma_nan(z, x.to_bits(), y, x.mul_add(f64b(y), f64b(z)).to_bits(), 64)
                    }
                    _ => return Err("half-precision lanes"),
                };
                put(&mut r, j, e, v);
            }
            Ok(r)
        }
        _ => {
            // Pairwise over the concatenation a:b.
            let total = lanes(q, e);
            let cat = |k: u32| if k < total { lane(a, k, e) } else { lane(b, k - total, e) };
            let mut r = 0;
            for j in 0..total {
                let (x, y) = (u128::from(cat(2 * j)), u128::from(cat(2 * j + 1)));
                let v = match i.opcode {
                    C::FADDP => fp2_arm(x, y, e, false, |x, y| x + y, |x, y| x + y)?,
                    C::FMAXP => fp2(x, y, e, false, crate::fmax64, crate::fmax32)?,
                    C::FMINP => fp2(x, y, e, false, crate::fmin64, crate::fmin32)?,
                    C::FMAXNMP => fp2(x, y, e, false, fmaxnm64, fmaxnm32)?,
                    _ => fp2(x, y, e, false, fminnm64, fminnm32)?,
                };
                put(&mut r, j, e, v as u64);
            }
            Ok(r)
        }
    }?;
    let r = if scalar_form { r & u128::from(mask(e)) } else { r };
    set_v(c, d, r, q || scalar_form);
    if scalar_form {
        c.vr[d] &= u128::from(mask(e));
    }
    Ok(())
}

/// Integer <-> FP conversions on scalars and vectors, with optional fixed-point bits.
fn convert(c: &mut Cpu, i: &Instruction) -> Result {
    let o = &i.operands;
    let fbits = match o[2] {
        O::Nothing => 0,
        ref x => imm(x)? as i32,
    };
    let to_float = matches!(i.opcode, C::SCVTF | C::UCVTF);
    let signed = matches!(i.opcode, C::SCVTF | C::FCVTZS | C::FCVTMS | C::FCVTPS | C::FCVTAS | C::FCVTNS);
    let round = |x: f64| -> f64 {
        match i.opcode {
            C::FCVTMS | C::FCVTMU => x.floor(),
            C::FCVTPS | C::FCVTPU => x.ceil(),
            C::FCVTAS | C::FCVTAU => x.round(),
            C::FCVTNS | C::FCVTNU => x.round_ties_even(),
            _ => x.trunc(),
        }
    };
    let scale = 2f64.powi(fbits);
    // One element: `x` holds `bits` bits of integer or float.
    let one = |x: u64, bits: u32| -> Result<u64> {
        Ok(if to_float {
            let v = ival(x, bits, signed) as f64 / scale;
            match bits {
                32 => u64::from((v as f32).to_bits()),
                64 => v.to_bits(),
                _ => return Err("half-precision conversion"),
            }
        } else {
            let f = match bits {
                32 => f64::from(f32b(x)),
                64 => f64b(x),
                _ => return Err("half-precision conversion"),
            };
            let v = round(f * scale);
            // Saturating, NaN to 0, as Rust's `as` casts.
            match (bits, signed) {
                (32, true) => v as i32 as u32 as u64,
                (32, false) => u64::from(v as u32),
                (_, true) => v as i64 as u64,
                (_, false) => v as u64,
            }
        })
    };
    match (&o[0], &o[1]) {
        (O::SIMDRegisterElements(..), O::SIMDRegisterElements(..)) => {
            let ((d, q, e), (s, _, _)) = (vec(&o[0])?, vec(&o[1])?);
            let mut r = 0;
            for j in 0..lanes(q, e) {
                put(&mut r, j, e, one(lane(c.vr[s], j, e), e)?);
            }
            set_v(c, d, r, q);
            Ok(())
        }
        (O::SIMDRegister(..), O::SIMDRegister(..)) => {
            let ((d, bits), (s, _)) = (scalar(&o[0])?, scalar(&o[1])?);
            c.vr[d] = u128::from(one(c.vr[s] as u64 & mask(bits), bits)?);
            Ok(())
        }
        // General register <-> FP with fixed-point bits, or the rounding variants.
        (O::SIMDRegister(..), _) if to_float => {
            let (d, bits) = scalar(&o[0])?;
            let gbits = gpr_bits(&o[1])?;
            let v = ival(gpr(c, &o[1])?, gbits, signed) as f64 / scale;
            c.vr[d] = u128::from(match bits {
                32 => u64::from((v as f32).to_bits()),
                64 => v.to_bits(),
                _ => return Err("half-precision conversion"),
            });
            Ok(())
        }
        (_, O::SIMDRegister(..)) if !to_float => {
            let (s, bits) = scalar(&o[1])?;
            let gbits = gpr_bits(&o[0])?;
            let f = match bits {
                32 => f64::from(f32b(c.vr[s] as u64)),
                64 => f64b(c.vr[s] as u64),
                _ => return Err("half-precision conversion"),
            };
            let v = round(f * scale);
            let r = match (gbits, signed) {
                (32, true) => v as i32 as u32 as u64,
                (32, false) => u64::from(v as u32),
                (_, true) => v as i64 as u64,
                (_, false) => v as u64,
            };
            set_gpr(c, &o[0], r)
        }
        _ => Err("conversion operands"),
    }
}

/// Element-wise integer operations on vectors (and vector forms of logic ops).
fn int_vector(c: &mut Cpu, i: &Instruction) -> Result {
    let o = &i.operands;
    let (d, q, e) = vec(&o[0])?;
    // Vector-immediate forms of ORR/BIC: [Vd, #imm].
    if let (C::ORR | C::BIC, O::ImmShift(..) | O::Immediate(_)) = (i.opcode, &o[1]) {
        let k = imm(&o[1])?;
        let mut r = c.vr[d];
        for j in 0..lanes(q, e) {
            let x = lane(r, j, e);
            put(&mut r, j, e, if i.opcode == C::ORR { x | k } else { x & !k });
        }
        set_v(c, d, r, q);
        return Ok(());
    }
    let a = c.vr[vec(&o[1])?.0];
    let b = match o[2] {
        O::SIMDRegisterElements(..) => Some(c.vr[vec(&o[2])?.0]),
        O::SIMDRegisterElementsLane(..) => {
            let (b, be, l) = elem(&o[2])?;
            let x = lane(c.vr[b], l, be);
            let mut r = 0;
            (0..lanes(q, e)).for_each(|j| put(&mut r, j, e, x));
            Some(r)
        }
        _ => None, // compare with zero, or a one-operand form
    };
    let bv = |j| b.map_or(0, |b| lane(b, j, e));
    let signed = matches!(
        i.opcode,
        C::SMAX | C::SMIN | C::SQADD | C::SQSUB | C::SHADD | C::SHSUB | C::SRHADD | C::SABD | C::SABA | C::CMGE | C::CMGT | C::CMLE | C::CMLT
            | C::SQDMULH | C::SQRDMULH | C::SSHL | C::SQSHL | C::NEG | C::ABS | C::SMAXP | C::SMINP
    );
    let n = lanes(q, e);
    let mut r = 0;
    for j in 0..n {
        let (x, y) = (lane(a, j, e), bv(j));
        let (sx, sy) = (ival(x, e, signed), ival(y, e, signed));
        let acc = lane(c.vr[d], j, e);
        let v: u64 = match i.opcode {
            C::ADD => x.wrapping_add(y),
            C::SUB => x.wrapping_sub(y),
            C::MUL => x.wrapping_mul(y),
            C::MLA => acc.wrapping_add(x.wrapping_mul(y)),
            C::MLS => acc.wrapping_sub(x.wrapping_mul(y)),
            C::SMAX | C::UMAX => sx.max(sy) as u64,
            C::SMIN | C::UMIN => sx.min(sy) as u64,
            C::SQADD | C::UQADD => saturate(sx + sy, e, signed),
            C::SQSUB | C::UQSUB => saturate(sx - sy, e, signed),
            C::SHADD | C::UHADD => ((sx + sy) >> 1) as u64,
            C::SHSUB => ((sx - sy) >> 1) as u64,
            C::SRHADD | C::URHADD => ((sx + sy + 1) >> 1) as u64,
            C::SABD | C::UABD => (sx - sy).unsigned_abs() as u64,
            C::SABA | C::UABA => acc.wrapping_add((sx - sy).unsigned_abs() as u64),
            C::CMEQ => all_ones(x == y, e),
            C::CMGE => all_ones(sx >= sy, e),
            C::CMGT => all_ones(sx > sy, e),
            C::CMLE => all_ones(sx <= sy, e),
            C::CMLT => all_ones(sx < sy, e),
            C::CMHI => all_ones(x > y, e),
            C::CMHS => all_ones(x >= y, e),
            C::CMTST => all_ones(x & y != 0, e),
            C::SQDMULH => saturate((2 * sx * sy) >> e, e, true),
            C::SQRDMULH => saturate((2 * sx * sy + (1 << (e - 1))) >> e, e, true),
            C::USHL | C::SSHL => {
                let s = sext(y & 0xFF, 8);
                if s >= 0 {
                    if s >= i64::from(e) { 0 } else { x << s }
                } else if signed {
                    (sx >> (-s).min(i64::from(e) - 1)) as u64
                } else if -s >= i64::from(e) {
                    0
                } else {
                    x >> -s
                }
            }
            // Saturating shift left by register (the low byte, signed): negative shifts right.
            C::SQSHL => {
                let s = sext(y & 0xFF, 8);
                if s < 0 {
                    (sx >> (-s).min(i64::from(e) - 1)) as u64
                } else if sx == 0 {
                    0
                } else if s >= i64::from(e) {
                    saturate(if sx > 0 { i128::MAX } else { i128::MIN }, e, true)
                } else {
                    saturate(sx << s, e, true)
                }
            }
            C::ADDP | C::UMAXP | C::UMINP | C::SMAXP | C::SMINP => {
                let cat = |k: u32| if k < n { lane(a, k, e) } else { b.map_or(0, |b| lane(b, k - n, e)) };
                let (p, s) = (cat(2 * j), cat(2 * j + 1));
                let (sp, ss) = (ival(p, e, signed), ival(s, e, signed));
                match i.opcode {
                    C::ADDP => p.wrapping_add(s),
                    C::UMAXP | C::SMAXP => sp.max(ss) as u64,
                    _ => sp.min(ss) as u64,
                }
            }
            C::AND => x & y,
            C::ORR => x | y,
            C::BIC => x & !y,
            C::EOR => x ^ y,
            C::ORN => x | !y,
            C::NOT => !x,
            C::NEG => (-sx) as u64,
            C::ABS => sx.unsigned_abs() as u64,
            _ => u64::from((x & mask(e)).count_ones()), // CNT
        };
        put(&mut r, j, e, v);
    }
    set_v(c, d, r, q);
    Ok(())
}

/// Shifts by an immediate, with rounding/accumulating/inserting variants.
fn shift_imm(c: &mut Cpu, i: &Instruction) -> Result {
    let o = &i.operands;
    let ((d, q, e), (s, _, _)) = (vec(&o[0])?, vec(&o[1])?);
    let k = imm(&o[2])? as u32;
    let mut r = c.vr[d];
    for j in 0..lanes(q, e) {
        let x = lane(c.vr[s], j, e);
        let acc = lane(r, j, e);
        let sx = sext(x, e) as i128;
        let round = |v: i128| if k == 0 { v } else { (v + (1 << (k - 1))) >> k };
        let v = match i.opcode {
            C::SHL => if k >= e { 0 } else { x << k },
            C::USHR => if k >= e { 0 } else { x >> k },
            C::SSHR => (sx >> k.min(e - 1)) as u64,
            C::USRA => acc.wrapping_add(if k >= e { 0 } else { x >> k }),
            C::SSRA => acc.wrapping_add((sx >> k.min(e - 1)) as u64),
            C::URSHR => round(i128::from(x)) as u64,
            C::SRSHR => round(sx) as u64,
            C::URSRA => acc.wrapping_add(round(i128::from(x)) as u64),
            C::SRSRA => acc.wrapping_add(round(sx) as u64),
            C::SRI => {
                let m = if k >= e { 0 } else { mask(e) >> k };
                (acc & !m) | ((if k >= e { 0 } else { x >> k }) & m)
            }
            C::SLI => {
                let m = if k >= e { 0 } else { mask(e) << k };
                (acc & !m) | ((x << k) & m)
            }
            C::SQSHL => saturate(sx << k, e, true),
            _ => saturate(sx << k, e, false), // SQSHLU
        };
        put(&mut r, j, e, v);
    }
    set_v(c, d, r, q);
    Ok(())
}

/// Narrowing: each wide element of the source becomes a half-width element
/// of the destination, in its lower half (or upper half for the "2" forms).
fn narrow(c: &mut Cpu, i: &Instruction) -> Result {
    let o = &i.operands;
    let ((d, _, e), (s, _, _)) = (vec(&o[0])?, vec(&o[1])?);
    let upper = matches!(i.opcode, C::XTN2 | C::SQXTN2 | C::UQXTN2 | C::SQXTUN2 | C::SHRN2 | C::RSHRN2 | C::SQSHRN2 | C::SQRSHRN2);
    let k = match o[2] {
        O::Nothing => 0,
        ref x => imm(x)? as u32,
    };
    let wide = 2 * e;
    let count = 64 / e;
    let mut r = if upper { c.vr[d] & LOW64 } else { 0 };
    for j in 0..count {
        let x = lane(c.vr[s], j, wide);
        let sx = sext(x, wide) as i128;
        let round = |v: i128| if k == 0 { v } else { (v + (1 << (k - 1))) >> k };
        let v = match i.opcode {
            C::XTN | C::XTN2 => x,
            C::SQXTN | C::SQXTN2 => saturate(sx, e, true),
            C::UQXTN | C::UQXTN2 => saturate(i128::from(x), e, false),
            C::SQXTUN | C::SQXTUN2 => saturate(sx, e, false),
            C::SHRN | C::SHRN2 => x >> k,
            C::RSHRN | C::RSHRN2 => round(i128::from(x)) as u64,
            C::SQSHRN | C::SQSHRN2 => saturate(sx >> k, e, true),
            C::SQRSHRN | C::SQRSHRN2 => saturate(round(sx), e, true),
            C::UQSHRN => saturate(i128::from(x >> k), e, false),
            C::UQRSHRN => saturate(round(i128::from(x)), e, false),
            C::SQSHRUN => saturate(sx >> k, e, false),
            _ => saturate(round(sx), e, false), // SQRSHRUN
        };
        put(&mut r, if upper { count + j } else { j }, e, v);
    }
    c.vr[d] = r;
    Ok(())
}

/// Widening: half-width elements (lower half, or upper half for "2" forms)
/// become full-width results.
fn widen(c: &mut Cpu, i: &Instruction) -> Result {
    let o = &i.operands;
    let (d, _, e) = vec(&o[0])?;
    let half = e / 2;
    let upper = matches!(
        i.opcode,
        C::USHLL2 | C::SSHLL2 | C::SMULL2 | C::UMULL2 | C::SMLAL2 | C::UMLAL2 | C::SMLSL2 | C::UMLSL2 | C::SADDL2 | C::UADDL2 | C::SSUBL2
            | C::USUBL2 | C::SADDW2 | C::UADDW2 | C::SSUBW2 | C::USUBW2 | C::SQDMLAL2 | C::SHLL2
    );
    let signed = matches!(
        i.opcode,
        C::SSHLL | C::SSHLL2 | C::SMULL | C::SMULL2 | C::SMLAL | C::SMLAL2 | C::SMLSL | C::SMLSL2 | C::SADDL | C::SADDL2 | C::SSUBL
            | C::SSUBL2 | C::SADDW | C::SADDW2 | C::SSUBW | C::SSUBW2 | C::SABDL | C::SQDMLAL | C::SQDMLAL2
    );
    let wide_first = matches!(i.opcode, C::SADDW | C::SADDW2 | C::UADDW | C::UADDW2 | C::SSUBW | C::SSUBW2 | C::USUBW | C::USUBW2);
    let a = c.vr[vec(&o[1])?.0];
    let b = match o[2] {
        O::SIMDRegisterElements(..) => Some(c.vr[vec(&o[2])?.0]),
        O::SIMDRegisterElementsLane(..) => {
            let (b, be, l) = elem(&o[2])?;
            let x = lane(c.vr[b], l, be);
            let mut r = 0;
            (0..128 / be).for_each(|j| put(&mut r, j, be, x));
            Some(r)
        }
        _ => None,
    };
    let shift = match (i.opcode, &o[2]) {
        (C::SHLL | C::SHLL2, _) => half,
        (_, O::Immediate(_) | O::Imm16(_) | O::Imm64(_)) => imm(&o[2])? as u32,
        _ => 0,
    };
    let count = 64 / half;
    let offset = if upper { count } else { 0 };
    let mut r = 0;
    for j in 0..count {
        let x = if wide_first { ival(lane(a, j, e), e, signed) } else { ival(lane(a, offset + j, half), half, signed) };
        let y = b.map_or(0, |b| ival(lane(b, offset + j, half), half, signed));
        let acc = ival(lane(c.vr[d], j, e), e, signed);
        let v = match i.opcode {
            C::USHLL | C::USHLL2 | C::SSHLL | C::SSHLL2 | C::SHLL | C::SHLL2 => x << shift,
            C::SMULL | C::SMULL2 | C::UMULL | C::UMULL2 => x * y,
            C::SMLAL | C::SMLAL2 | C::UMLAL | C::UMLAL2 => acc + x * y,
            C::SMLSL | C::SMLSL2 | C::UMLSL | C::UMLSL2 => acc - x * y,
            C::SQDMLAL | C::SQDMLAL2 => i128::from(saturate(acc + saturate(2 * x * y, e, true) as i64 as i128, e, true) as i64),
            C::SADDL | C::SADDL2 | C::UADDL | C::UADDL2 | C::SADDW | C::SADDW2 | C::UADDW | C::UADDW2 => x + y,
            C::SABDL | C::UABDL => (x - y).abs(),
            _ => x - y, // the SUBL and SUBW forms
        };
        put(&mut r, j, e, v as u64);
    }
    c.vr[d] = r;
    Ok(())
}
