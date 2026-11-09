//! Hand-assembled words on a synthetic context and memory; no game data.

use gyroship_runtime::Cpu;
use gyroship_runtime::interp::{execute, supported};

const GUEST: u64 = 0x1000_0000;

fn lanes32(v: [u32; 4]) -> u128 {
    v.iter().enumerate().fold(0, |r, (i, &x)| r | u128::from(x) << (32 * i))
}

fn floats(v: [f32; 4]) -> u128 {
    lanes32(v.map(f32::to_bits))
}

fn run(c: &mut Cpu, word: u32) {
    execute(c, 0x1000, word).unwrap_or_else(|e| panic!("{word:08x}: {e}"));
}

#[test]
fn vector_lanes() {
    let mut c = Cpu::new(std::ptr::null_mut());
    c.vr[4] = lanes32([1, 2, 3, 4]);
    c.vr[5] = lanes32([5, 6, 7, 8]);
    run(&mut c, 0x4e853884); // zip1 v4.4s, v4.4s, v5.4s
    assert_eq!(c.vr[4], lanes32([1, 5, 2, 6]));

    c.vr[0] = lanes32([1, 2, 3, 4]);
    run(&mut c, 0x6e004000); // ext v0.16b, v0.16b, v0.16b, #8
    assert_eq!(c.vr[0], lanes32([3, 4, 1, 2]));

    c.vr[1] = lanes32([1, 2, 3, 4]);
    run(&mut c, 0x4ea00821); // rev64 v1.4s, v1.4s
    assert_eq!(c.vr[1], lanes32([2, 1, 4, 3]));

    c.vr[16] = floats([1.0, 2.0, 3.0, 4.0]);
    c.vr[21] = floats([2.0; 4]);
    c.vr[17] = floats([0.5, 1.0, 1.5, 2.0]);
    run(&mut c, 0x4e31ceb0); // fmla v16.4s, v21.4s, v17.4s
    assert_eq!(c.vr[16], floats([2.0, 4.0, 6.0, 8.0]));

    c.vr[6] = u128::MAX;
    c.vr[4] = lanes32([10, 11, 12, 13]);
    run(&mut c, 0x6e042486); // mov v6.s[0], v4.s[1]
    assert_eq!(c.vr[6], u128::MAX << 32 | 11, "other lanes are kept");

    c.vr[0] = u128::MAX;
    c.vr[1] = 5 << 64 | 7;
    run(&mut c, 0x5ef1b820); // addp d0, v1.2d
    assert_eq!(c.vr[0], 12, "the two lanes' sum, the rest cleared");

    c.vr[4] = lanes32([1, 0x4000_0000, 0xffff_ffff, 100]);
    c.vr[3] = lanes32([1, 1, 1, 0xff]); // shifts 1, 1, 1, -1
    run(&mut c, 0x4ea34c84); // sqshl v4.4s, v4.4s, v3.4s
    assert_eq!(c.vr[4], lanes32([2, 0x7fff_ffff, 0xffff_fffe, 50]), "saturated left, arithmetic right");
}

#[test]
fn reciprocal_estimates_match_the_hardware() {
    let mut c = Cpu::new(std::ptr::null_mut());
    c.vr[28] = floats([1.0, 2.0, 0.0, -4.0]);
    run(&mut c, 0x4ea1db9d); // frecpe v29.4s, v28.4s
    assert_eq!(c.vr[29], lanes32([0x3F7F_8000, 0x3EFF_8000, 0x7F80_0000, 0xBE7F_8000]));
    c.vr[20] = floats([1.0, 4.0, -1.0, f32::INFINITY]);
    run(&mut c, 0x6ea1da93); // frsqrte v19.4s, v20.4s
    assert_eq!(c.vr[19], lanes32([0x3F7F_8000, 0x3EFF_8000, 0x7FC0_0000, 0]));
}

#[test]
fn fmaxnm_takes_the_number_against_a_quiet_nan_only() {
    // Arm's FPMaxNum and FPMinNum: a quiet NaN loses to a number; a signalling
    // NaN wins, quieted; of two quiet NaNs, the first.
    let (one, qnan, snan, other) = (0x3F80_0000, 0x7FC0_0001, 0x7F80_0002, 0x7FC0_0003);
    let mut c = Cpu::new(std::ptr::null_mut());
    c.vr[1] = lanes32([qnan, snan, qnan, one]);
    c.vr[2] = lanes32([one, one, other, qnan]);
    let expected = lanes32([one, 0x7FC0_0002, qnan, one]);
    run(&mut c, 0x4e22c420); // fmaxnm v0.4s, v1.4s, v2.4s
    assert_eq!(c.vr[0], expected);
    run(&mut c, 0x4ea2c420); // fminnm v0.4s, v1.4s, v2.4s
    assert_eq!(c.vr[0], expected);
}

#[test]
fn atomics_and_crc_on_real_memory() {
    let mut mem = vec![0u8; 64];
    let mut c = Cpu::new(mem.as_mut_ptr().wrapping_sub(GUEST as usize));
    mem[..8].copy_from_slice(&10u64.to_le_bytes());
    (c.x[8], c.x[9]) = (GUEST, 5);
    run(&mut c, 0xf8e90108); // ldaddal x9, x8, [x8]
    assert_eq!(c.x[8], 10, "old value");
    assert_eq!(u64::from_le_bytes(mem[..8].try_into().unwrap()), 15);

    mem[8..12].copy_from_slice(&7u32.to_le_bytes());
    (c.x[20], c.x[9], c.x[10]) = (GUEST + 8, 7, 9);
    run(&mut c, 0x88e9fe8a); // casal w9, w10, [x20]
    assert_eq!((c.x[9], u32::from_le_bytes(mem[8..12].try_into().unwrap())), (7, 9), "swapped");
    c.x[9] = 7;
    run(&mut c, 0x88e9fe8a);
    assert_eq!((c.x[9], u32::from_le_bytes(mem[8..12].try_into().unwrap())), (9, 9), "no swap, old value returned");

    // CRC-32C of "123456789" is 0xE3069283.
    c.x[0] = 0xFFFF_FFFF;
    c.x[1] = u64::from_le_bytes(*b"12345678");
    run(&mut c, 0x9ac15c00); // crc32cx w0, w0, x1
    c.x[1] = u64::from(b'9');
    run(&mut c, 0x1ac15000); // crc32cb w0, w0, w1
    assert_eq!(!c.x[0] as u32, 0xE306_9283);
}

#[test]
fn a_store_exclusive_needs_its_reservation_and_the_value_it_read() {
    const LDXR: u32 = 0x885f7c01; // ldxr w1, [x0]
    const STXR: u32 = 0x88027c01; // stxr w2, w1, [x0]
    let mut mem = vec![0u8; 16];
    let mut c = Cpu::new(mem.as_mut_ptr().wrapping_sub(GUEST as usize));
    let word = |m: &Vec<u8>| u32::from_le_bytes(m[..4].try_into().unwrap());
    mem[..4].copy_from_slice(&5u32.to_le_bytes());
    c.x[0] = GUEST;
    run(&mut c, LDXR);
    assert_eq!(c.x[1], 5);
    c.x[1] = 6;
    run(&mut c, STXR);
    assert_eq!((c.x[2], word(&mem)), (0, 6), "stored");
    run(&mut c, STXR);
    assert_eq!((c.x[2], word(&mem)), (1, 6), "the reservation is used up");

    run(&mut c, LDXR);
    mem[..4].copy_from_slice(&7u32.to_le_bytes()); // another thread's write
    c.x[1] = 8;
    run(&mut c, STXR);
    assert_eq!((c.x[2], word(&mem)), (1, 7), "the word changed since the load");

    run(&mut c, LDXR);
    run(&mut c, 0xd5033f5f); // clrex
    run(&mut c, STXR);
    assert_eq!((c.x[2], word(&mem)), (1, 7), "cleared");
}

#[test]
fn dc_zva_zeroes_the_64_bytes_around_the_address_and_dc_civac_nothing() {
    let mut mem = vec![0xffu8; 192];
    let mut c = Cpu::new(mem.as_mut_ptr().wrapping_sub(GUEST as usize));
    c.x[0] = GUEST + 64 + 13;
    run(&mut c, 0xd50b7420); // dc zva, x0
    assert!(mem[..64].iter().chain(&mem[128..]).all(|&b| b == 0xff) && mem[64..128].iter().all(|&b| b == 0));
    c.x[10] = GUEST;
    run(&mut c, 0xd50b7e2a); // dc civac, x10
    assert!(mem[..64].iter().all(|&b| b == 0xff));
}

#[test]
fn carry_chain_and_coverage() {
    let mut c = Cpu::new(std::ptr::null_mut());
    (c.x[12], c.x[13], c.c) = (u64::MAX, 0, true);
    run(&mut c, 0xba0d0180); // adcs x0, x12, x13
    assert_eq!((c.x[0], c.c, c.z), (0, true, true));
    assert!(supported(0x4e040406, 0), "dup v6.4s, v0.s[0]");
    assert!(!supported(0x4e284801, 0), "aese is not covered yet");
}
