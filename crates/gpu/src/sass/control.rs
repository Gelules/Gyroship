//! Control family: branches, calls, convergence barriers, exit/kill,
//! dependency and thread barriers (SM 8.x).
//!
//! Contract for the integration:
//! - `pub enum ControlOp` with typed variants (our own code, written from the
//!   encoding facts; nothing copied from Mesa);
//! - `pub fn decode(w: Word) -> Option<ControlOp>`: Some for every opcode the
//!   family owns, None for any other;
//! - `pub fn explained(opcode: u16) -> u128`: the bits the decoder reads for
//!   that opcode (ALWAYS_EXPLAINED is added by the check);
//! - `pub const OPCODES: &[u16]`: the 12-bit opcodes the family owns.
//!
//! Shared layout (from NAK's SM70 encoder, `set_pred_src(87..90, 90, ..)` and
//! `set_rel_offset(34..)` in the Op*::encode functions):
//! - extra condition predicate: register at 87..90 (7 = PT), negation at 90.
//!   The instruction acts only when both the guard and this condition hold;
//! - branch offsets: a signed count of 32-bit words from bit 34, relative to
//!   the next instruction (the encoder subtracts ip + 4 dwords). We return it
//!   in bytes; targets are 16-byte aligned so bits 32..34 never matter;
//! - convergence barrier registers B0..B15: 4 bits at 16..20
//!   (`set_bar_dst` / `set_bar_src`).

use super::prelude::*;

pub const OPCODES: &[u16] = &[0x947, 0x941, 0x945, 0x91a, 0x94d, 0x942, 0x944, 0x948, 0x95b, 0x949, 0x918, 0xb1d, 0x950];

/// Mode of a thread-block barrier (2 bits at 77..79, from the comments in
/// NAK's OpBar encoder, given there for the register form 0x31d).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BarMode {
    Sync,
    Arrive,
    Reduce,
    Scan,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ControlOp {
    /// BSYNC (0x941, NAK OpBSync): wait until all threads registered in
    /// convergence barrier `bar` arrive.
    Bsync { bar: u8, cond: Pred },
    /// BREAK (0x942, NAK OpBreak): remove the active threads from `bar`.
    Break { bar: u8, cond: Pred },
    /// BSSY (0x945, NAK OpBSSy): set convergence barrier `bar` to the active
    /// threads, with the reconvergence point at next + `offset` bytes. NAK
    /// writes the offset to 34..64 only (forward targets).
    Bssy { bar: u8, offset: i64, cond: Pred },
    /// BRA (0x947, NAK OpBra, non-uniform branch of the encoder): jump to
    /// next + `offset` bytes. `uniform` is the .U bit 32 (NAK clears it here
    /// and sets it on the uniform-predicate opcode 0x547).
    Bra { offset: i64, cond: Pred, uniform: bool },
    /// BRX (0x949, not in NAK): indirect relative branch to
    /// next + `offset` + R`target`. The game uses offset = -next, which turns
    /// the register into an absolute program offset (jump tables).
    Brx { target: u8, offset: i64, cond: Pred },
    /// CALL (0x944, not in NAK): relative call to next + `offset`.
    /// `flag86` is set on every sample (presumably nvdisasm's .NOINC).
    Call { offset: i64, cond: Pred, flag86: bool },
    /// RET (0x950, not in NAK): return to next + `offset` + R`target`; the
    /// game uses offset = -next, so R`target` holds an absolute program
    /// offset. `flag86` as for CALL (presumably .NODEC).
    Ret { target: u8, offset: i64, cond: Pred, flag86: bool },
    /// EXIT (0x94d, NAK OpExit): bit 84 is the .KEEPREFCOUNT group (NAK
    /// writes one bit), bit 85 is .NO_ATEXIT.
    Exit { keep_ref_count: bool, no_at_exit: bool, cond: Pred },
    /// KILL (0x95b, NAK OpKill): discard the fragment.
    Kill { cond: Pred },
    /// WARPSYNC (0x948 = ALU opcode 0x148, form 4, NAK OpWarpSync): the lane
    /// mask is the generic ALU source b (an immediate at 32..64 in form 4).
    WarpSync { mask: Operand, cond: Pred },
    /// NOP (0x918, NAK OpNop).
    Nop,
    /// DEPBAR (0x91a, not in NAK): wait until scoreboard `sb` (3 bits at
    /// 44..47) has at most `count` (6 bits at 38..44) operations in flight;
    /// `le` is bit 47 (the .LE form).
    DepBar { sb: u8, count: u8, le: bool },
    /// BAR (0xb1d, NAK OpBar: opcode only, i.e. BAR.SYNC 0). Assumed, not
    /// proved: mode at 77..79 and reduction op at 74..76 (0 POPC, 1 AND, 2
    /// OR) come from NAK's commented-out code for the register form 0x31d; in
    /// the game both are always 0 (BAR.SYNC).
    Bar { mode: BarMode, red_op: u8 },
}

const COND: u128 = bits(87, 4);
const BAR_REG: u128 = bits(16, 4);
const OFFSET: u128 = bits(34, 48);
const TARGET_REG: u128 = bits(24, 8);

fn cond(w: Word) -> Pred {
    Pred::at(w, 87, 90)
}

fn bar(w: Word) -> u8 {
    w.field(16, 4) as u8
}

/// Byte offset from a dword count at 34..34+len.
fn offset(w: Word, len: u32) -> i64 {
    w.signed(34, len) * 4
}

pub fn decode(w: Word) -> Option<ControlOp> {
    use ControlOp::*;
    Some(match w.opcode() {
        0x941 => Bsync { bar: bar(w), cond: cond(w) },
        0x942 => Break { bar: bar(w), cond: cond(w) },
        0x945 => Bssy { bar: bar(w), offset: offset(w, 30), cond: cond(w) },
        0x947 => Bra { offset: offset(w, 48), cond: cond(w), uniform: w.bit(32) },
        0x949 => Brx { target: reg(w, 24), offset: offset(w, 48), cond: cond(w) },
        0x944 => Call { offset: offset(w, 48), cond: cond(w), flag86: w.bit(86) },
        0x950 => Ret { target: reg(w, 24), offset: offset(w, 48), cond: cond(w), flag86: w.bit(86) },
        0x94d => Exit { keep_ref_count: w.bit(84), no_at_exit: w.bit(85), cond: cond(w) },
        0x95b => Kill { cond: cond(w) },
        0x948 => WarpSync { mask: alu(w, false)?.b, cond: cond(w) },
        0x918 => Nop,
        0x91a => DepBar { sb: w.field(44, 3) as u8, count: w.field(38, 6) as u8, le: w.bit(47) },
        0xb1d => {
            let mode = [BarMode::Sync, BarMode::Arrive, BarMode::Reduce, BarMode::Scan][w.field(77, 2) as usize];
            Bar { mode, red_op: w.field(74, 2) as u8 }
        }
        _ => return None,
    })
}

pub fn explained(opcode: u16) -> u128 {
    match opcode {
        0x941 | 0x942 => BAR_REG | COND,
        0x945 => BAR_REG | bits(34, 30) | COND,
        0x947 => bits(32, 1) | OFFSET | COND,
        0x949 => TARGET_REG | OFFSET | COND,
        0x944 => OFFSET | bits(86, 1) | COND,
        0x950 => TARGET_REG | OFFSET | bits(86, 1) | COND,
        0x94d => bits(84, 2) | COND,
        0x95b => COND,
        0x948 => bits(32, 32) | COND,
        0x91a => bits(38, 10),
        0xb1d => bits(74, 2) | bits(77, 2),
        _ => 0,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A word with the opcode, guard PT and the given (lo, len, value) fields.
    fn word(op: u16, fields: &[(u32, u32, i64)]) -> Word {
        let mut v = op as u128 | (7 << 12);
        for &(lo, len, x) in fields {
            v |= (x as u128 & ((1u128 << len) - 1)) << lo;
        }
        Word(v)
    }

    fn p(reg: u8, not: bool) -> Pred {
        Pred { reg, not }
    }

    #[test]
    fn synthetic() {
        use ControlOp::*;
        let cases = [
            (word(0x941, &[(16, 4, 9), (87, 3, 3), (90, 1, 1)]), Bsync { bar: 9, cond: p(3, true) }),
            (word(0x942, &[(16, 4, 15), (87, 3, 6)]), Break { bar: 15, cond: p(6, false) }),
            (word(0x945, &[(16, 4, 2), (34, 30, 0x1230 / 4), (87, 3, 7)]), Bssy { bar: 2, offset: 0x1230, cond: Pred::TRUE }),
            (word(0x947, &[(32, 1, 1), (34, 48, -0x40 / 4), (87, 3, 5)]), Bra { offset: -0x40, cond: p(5, false), uniform: true }),
            (word(0x947, &[(34, 48, 0x7ff0 / 4), (87, 3, 4), (90, 1, 1)]), Bra { offset: 0x7ff0, cond: p(4, true), uniform: false }),
            (word(0x949, &[(24, 8, 54), (34, 48, -0x1fd0 / 4), (87, 3, 7)]), Brx { target: 54, offset: -0x1fd0, cond: Pred::TRUE }),
            (word(0x944, &[(34, 48, 0x780 / 4), (86, 1, 1), (87, 3, 2)]), Call { offset: 0x780, cond: p(2, false), flag86: true }),
            (word(0x950, &[(24, 8, 12), (34, 48, -0xdd0 / 4), (87, 3, 7)]), Ret { target: 12, offset: -0xdd0, cond: Pred::TRUE, flag86: false }),
            (word(0x94d, &[(84, 1, 1), (85, 1, 1), (87, 3, 1)]), Exit { keep_ref_count: true, no_at_exit: true, cond: p(1, false) }),
            (word(0x95b, &[(87, 3, 2), (90, 1, 1)]), Kill { cond: p(2, true) }),
            (word(0x948, &[(32, 32, 0xffff), (87, 3, 7)]), WarpSync { mask: Operand::plain(Src::Imm(0xffff)), cond: Pred::TRUE }),
            (word(0x918, &[]), Nop),
            (word(0x91a, &[(38, 6, 63), (44, 3, 3), (47, 1, 1)]), DepBar { sb: 3, count: 63, le: true }),
            (word(0x91a, &[(38, 6, 1), (44, 3, 5)]), DepBar { sb: 5, count: 1, le: false }),
            (word(0xb1d, &[(74, 2, 2), (77, 2, 3)]), Bar { mode: BarMode::Scan, red_op: 2 }),
            (word(0xb1d, &[(77, 2, 1)]), Bar { mode: BarMode::Arrive, red_op: 0 }),
        ];
        for (w, want) in cases {
            assert_eq!(decode(w), Some(want), "{:#034x}", w.0);
            // Every field the test sets is part of the explained mask.
            assert_eq!(w.0 & !(explained(w.opcode()) | ALWAYS_EXPLAINED), 0, "{:#034x}", w.0);
        }
        for &op in OPCODES {
            assert!(cases.iter().any(|(w, _)| w.opcode() == op), "no case for {op:#x}");
        }
    }

    #[test]
    fn others_are_none() {
        // Uniform-predicate BRA (0x547), WARPSYNC register form (0x348),
        // BAR register form (0x31d), BMOV (0x355) and a plain ALU opcode.
        for op in [0x547u16, 0x348, 0x31d, 0x355, 0x210, 0x000] {
            assert_eq!(decode(Word(op as u128)), None, "{op:#x}");
            assert!(!OPCODES.contains(&op));
        }
        for &op in OPCODES {
            assert!(decode(Word(op as u128 | 7 << 87)).is_some(), "{op:#x}");
        }
    }
}
