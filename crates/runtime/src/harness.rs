//! The `main` of generated test harnesses: installs the dispatch table, then
//! optionally runs recompiled functions against host references.
//!
//! - no arguments: prints how many functions are installed;
//! - `adler <addr> <len> [rounds]`: runs `addr` as `adler32(1, buf, len)`;
//! - `math <exefs> <name>=<addr>...`: runs libm functions of the SDK (sinf,
//!   expf, powf...) over many inputs and compares them with the host's.

use crate::{Cpu, Guest, UNHANDLED, install, lookup};
use std::sync::atomic::Ordering;
use std::time::Instant;

pub fn main(table: Vec<(u64, Guest)>) {
    let installed = table.len();
    install(table);
    let args: Vec<String> = std::env::args().skip(1).collect();
    let hex = |s: &str| u64::from_str_radix(s.trim_start_matches("0x"), 16).expect("a hex address");
    match args.first().map(String::as_str) {
        None => println!("{installed} functions installed"),
        Some("adler") => {
            let rounds = args.get(3).map_or(10, |r| r.parse().expect("rounds"));
            adler(hex(&args[1]), args[2].parse().expect("a length"), rounds);
        }
        Some("math") => math(&args[1], &args[2..]),
        Some(other) => panic!("unknown mode {other:?}"),
    }
}

fn xorshift(s: &mut u32) -> u32 {
    *s ^= *s << 13;
    *s ^= *s >> 17;
    *s ^= *s << 5;
    *s
}

fn adler32(data: &[u8]) -> u64 {
    let (mut a, mut b) = (1u32, 0u32);
    for &x in data {
        a = (a + u32::from(x)) % 65521;
        b = (b + a) % 65521;
    }
    u64::from(b << 16 | a)
}

fn adler(target: u64, len: usize, rounds: usize) {
    const GUEST: u64 = 0x1000_0000;
    let mut mem = vec![0u8; len + (1 << 20)];
    let mut s = 0x1234_5678;
    for b in mem[..len].iter_mut() {
        *b = xorshift(&mut s) as u8;
    }
    let host = Instant::now();
    let mut expected = 0;
    for _ in 0..rounds {
        expected = adler32(&mem[..len]);
    }
    let host = host.elapsed().as_secs_f64();
    let top = GUEST + mem.len() as u64 - 64;
    let mut cpu = Cpu::new(mem.as_mut_ptr().wrapping_sub(GUEST as usize));
    let f = lookup(target).expect("no such function");
    let start = Instant::now();
    let mut result = 0;
    for _ in 0..rounds {
        (cpu.x[0], cpu.x[1], cpu.x[2], cpu.x[31], cpu.x[30]) = (1, GUEST, len as u64, top, 0);
        unsafe { f(&mut cpu) };
        result = cpu.x[0];
    }
    let guest = start.elapsed().as_secs_f64();
    let mb = (len * rounds) as f64 / 1e6;
    println!(
        "result {result:#x} expected {expected:#x} ({}); recompiled {:.0} MB/s, host Rust {:.0} MB/s, {} untranslated",
        if result == expected { "same" } else { "DIFFERENT" },
        mb / guest,
        mb / host,
        UNHANDLED.load(Ordering::Relaxed)
    );
}

/// A host function of one or two arguments, and the input range of each.
type Reference = (fn(f32, f32) -> f32, (f32, f32), (f32, f32));

/// A host reference and an input domain for each libm function.
fn reference(name: &str) -> Option<Reference> {
    let any = (-10.0, 10.0);
    Some(match name {
        "expf" => (|x, _| x.exp(), any, any),
        "exp2f" => (|x, _| x.exp2(), any, any),
        "logf" => (|x, _| x.ln(), (1e-3, 100.0), any),
        "log2f" => (|x, _| x.log2(), (1e-3, 100.0), any),
        "sinf" => (|x, _| x.sin(), any, any),
        "cosf" => (|x, _| x.cos(), any, any),
        "tanf" => (|x, _| x.tan(), (-1.5, 1.5), any),
        "sqrtf" => (|x, _| x.sqrt(), (0.0, 100.0), any),
        "powf" => (|x, y| x.powf(y), (1e-3, 10.0), (-5.0, 5.0)),
        "atan2f" => (|y, x| y.atan2(x), any, any),
        "fmodf" => (|x, y| x % y, (-100.0, 100.0), (0.1, 10.0)),
        _ => return None,
    })
}

fn ulps(a: f32, b: f32) -> u32 {
    if a.to_bits() == b.to_bits() || (a.is_nan() && b.is_nan()) {
        0
    } else if a.is_sign_negative() != b.is_sign_negative() {
        u32::MAX
    } else {
        a.to_bits().abs_diff(b.to_bits())
    }
}

fn math(exefs: &str, functions: &[String]) {
    let mut modules = Vec::new();
    for name in gyroship_loader::link::LOAD_ORDER {
        let path = std::path::Path::new(exefs).join(name);
        if path.is_file() {
            let file = std::fs::read(&path).expect("readable module");
            modules.push((name.to_string(), gyroship_loader::Module::parse_nso(&file).expect("a valid module")));
        }
    }
    let program = gyroship_loader::link::Program::link(modules).expect("linkable modules");
    let sdk = program.modules.iter().find(|m| m.name == "sdk").expect("an sdk module");
    // The SDK image at its guest address, then a stack.
    let mut mem = sdk.module.image.clone();
    mem.resize(mem.len() + (1 << 20), 0);
    let top = sdk.base + mem.len() as u64 - 64;
    let mut cpu = Cpu::new(mem.as_mut_ptr().wrapping_sub(sdk.base as usize));
    for spec in functions {
        let (name, addr) = spec.split_once('=').expect("name=addr");
        let (host, dx, dy) = reference(name).unwrap_or_else(|| panic!("no host reference for {name}"));
        let f = lookup(u64::from_str_radix(addr.trim_start_matches("0x"), 16).expect("hex")).expect("no such function");
        let (mut s, mut exact, mut worst, mut sum) = (0x9E37_79B9u32, 0u32, 0u32, 0u64);
        let n = 20_000;
        let unhandled = UNHANDLED.load(Ordering::Relaxed);
        for _ in 0..n {
            let mut pick = |(lo, hi): (f32, f32)| lo + (hi - lo) * (xorshift(&mut s) as f32 / u32::MAX as f32);
            let (x, y) = (pick(dx), pick(dy));
            cpu.vr[0] = u128::from(x.to_bits());
            cpu.vr[1] = u128::from(y.to_bits());
            (cpu.x[31], cpu.x[30]) = (top, 0);
            unsafe { f(&mut cpu) };
            let got = f32::from_bits(cpu.vr[0] as u32);
            sum = sum.wrapping_mul(31).wrapping_add(u64::from(got.to_bits()));
            let u = ulps(got, host(x, y));
            exact += u32::from(u == 0);
            worst = worst.max(u);
        }
        println!(
            "{name:<7} {n} inputs: {exact} identical to the host, worst {worst} ulp, result checksum {sum:016x}, {} untranslated",
            UNHANDLED.load(Ordering::Relaxed) - unhandled
        );
    }
}
