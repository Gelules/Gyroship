//! From a decoded instruction to one IR operation.

use crate::ir::{Addr, Alu, Cond, Op, Operand, Reg, Shift, Unary, Writeback, W};
use gyroship_analysis::decode::{decode, Instruction, Opcode, Operand as O, ShiftStyle, SizeCode};

/// MRS/MSR encodings of TPIDR_EL0 and TPIDRRO_EL0 as yaxpeax numbers them.
const TPIDR_EL0: u16 = 0x5E82;
const TPIDRRO_EL0: u16 = 0x5E83;

pub fn lift(word: u32, pc: u64) -> Op {
    match decode(word) {
        Some(i) => lift_decoded(&i, pc).unwrap_or(Op::Unhandled),
        None => Op::Unhandled,
    }
}

fn width(s: SizeCode) -> W {
    match s {
        SizeCode::X => W::W64,
        SizeCode::W => W::W32,
    }
}

/// A general register, with 31 as the zero register (`Register`) or SP (`RegisterOrSP`).
fn reg(o: &O) -> Option<(Reg, W)> {
    match *o {
        O::Register(s, 31) => Some((Reg::Zr, width(s))),
        O::Register(s, n) => Some((Reg::X(n as u8), width(s))),
        O::RegisterOrSP(s, 31) => Some((Reg::Sp, width(s))),
        O::RegisterOrSP(s, n) => Some((Reg::X(n as u8), width(s))),
        _ => None,
    }
}

fn r(o: &O) -> Option<Reg> {
    reg(o).map(|(r, _)| r)
}

fn x(n: u16) -> Reg {
    if n == 31 { Reg::Zr } else { Reg::X(n as u8) }
}

fn base(n: u16) -> Reg {
    if n == 31 { Reg::Sp } else { Reg::X(n as u8) }
}

fn imm(o: &O) -> Option<u64> {
    match *o {
        O::Immediate(v) => Some(u64::from(v)),
        O::Imm64(v) => Some(v),
        O::Imm16(v) => Some(u64::from(v)),
        O::ImmShift(v, s) => Some(u64::from(v) << s),
        _ => None,
    }
}

fn operand(o: &O) -> Option<Operand> {
    if let Some(v) = imm(o) {
        return Some(Operand::Imm(v));
    }
    match *o {
        O::Register(_, n) => Some(Operand::Shifted(x(n), Shift::Lsl, 0)),
        O::RegShift(style, amount, _, n) => {
            let ext = |signed, bits| Some(Operand::Extended { reg: x(n), signed, bits, lsl: amount });
            match style {
                ShiftStyle::LSL => Some(Operand::Shifted(x(n), Shift::Lsl, amount)),
                ShiftStyle::LSR => Some(Operand::Shifted(x(n), Shift::Lsr, amount)),
                ShiftStyle::ASR => Some(Operand::Shifted(x(n), Shift::Asr, amount)),
                ShiftStyle::ROR => Some(Operand::Shifted(x(n), Shift::Ror, amount)),
                ShiftStyle::UXTB => ext(false, 8),
                ShiftStyle::UXTH => ext(false, 16),
                ShiftStyle::UXTW => ext(false, 32),
                ShiftStyle::UXTX => ext(false, 64),
                ShiftStyle::SXTB => ext(true, 8),
                ShiftStyle::SXTH => ext(true, 16),
                ShiftStyle::SXTW => ext(true, 32),
                ShiftStyle::SXTX => ext(true, 64),
            }
        }
        _ => None,
    }
}

fn addr(o: &O) -> Option<Addr> {
    Some(match *o {
        O::RegPreIndex(b, off, wb) => {
            Addr { base: base(b), offset: i64::from(off), index: None, writeback: if wb { Writeback::Pre } else { Writeback::None } }
        }
        O::RegPostIndex(b, off) => Addr { base: base(b), offset: i64::from(off), index: None, writeback: Writeback::Post },
        O::RegRegOffset(b, i, _, style, amount) => {
            let (signed, bits) = match style {
                ShiftStyle::LSL | ShiftStyle::UXTX => (false, 64),
                ShiftStyle::SXTX => (true, 64),
                ShiftStyle::UXTW => (false, 32),
                ShiftStyle::SXTW => (true, 32),
                _ => return None,
            };
            Addr { base: base(b), offset: 0, index: Some((x(i), signed, bits, amount)), writeback: Writeback::None }
        }
        _ => return None,
    })
}

fn pc_offset(o: &O, pc: u64) -> Option<u64> {
    match *o {
        O::PCOffset(off) => Some(pc.wrapping_add(off as u64)),
        _ => None,
    }
}

fn cond(o: &O) -> Option<u8> {
    match *o {
        O::ConditionCode(c) => Some(c),
        _ => None,
    }
}

fn lift_decoded(i: &Instruction, pc: u64) -> Option<Op> {
    use Opcode as C;
    let o = &i.operands;
    let alu = |op: Alu, flags: bool| -> Option<Op> {
        let (dst, w) = reg(&o[0])?;
        Some(Op::Alu { op, w, dst, a: r(&o[1])?, b: operand(&o[2])?, flags })
    };
    // Loads and stores: (size log2, signed) from the opcode; None means "size of Rt".
    let load = |size: Option<u8>, signed: bool| -> Option<Op> {
        let (dst, w) = reg(&o[0])?;
        let size = size.unwrap_or(if w == W::W64 { 3 } else { 2 });
        Some(Op::Load { dst, w, size, signed, addr: addr(&o[1])? })
    };
    let store = |size: Option<u8>| -> Option<Op> {
        let (src, w) = reg(&o[0])?;
        let size = size.unwrap_or(if w == W::W64 { 3 } else { 2 });
        Some(Op::Store { src, size, addr: addr(&o[1])? })
    };
    Some(match i.opcode {
        C::ADD => alu(Alu::Add, false)?,
        C::ADDS => alu(Alu::Add, true)?,
        C::SUB => alu(Alu::Sub, false)?,
        C::SUBS => alu(Alu::Sub, true)?,
        C::AND => alu(Alu::And, false)?,
        C::ANDS => alu(Alu::And, true)?,
        C::ORR => alu(Alu::Orr, false)?,
        C::EOR => alu(Alu::Eor, false)?,
        C::BIC => alu(Alu::Bic, false)?,
        C::BICS => alu(Alu::Bic, true)?,
        C::ORN => alu(Alu::Orn, false)?,
        C::EON => alu(Alu::Eon, false)?,
        C::MOVZ | C::MOVN | C::MOVK => {
            let (dst, w) = reg(&o[0])?;
            let O::ImmShift(imm, shift) = o[1] else { return None };
            Op::MovWide { w, dst, imm, shift, invert: i.opcode == C::MOVN, keep: i.opcode == C::MOVK }
        }
        C::UBFM | C::SBFM | C::BFM => {
            let (dst, w) = reg(&o[0])?;
            Op::Bitfield {
                w,
                dst,
                src: r(&o[1])?,
                immr: imm(&o[2])? as u8,
                imms: imm(&o[3])? as u8,
                signed: i.opcode == C::SBFM,
                insert: i.opcode == C::BFM,
            }
        }
        C::EXTR => {
            let (dst, w) = reg(&o[0])?;
            Op::Extr { w, dst, a: r(&o[1])?, b: r(&o[2])?, lsb: imm(&o[3])? as u8 }
        }
        C::LSLV | C::LSRV | C::ASRV | C::RORV => {
            let (dst, w) = reg(&o[0])?;
            let op = match i.opcode {
                C::LSLV => Shift::Lsl,
                C::LSRV => Shift::Lsr,
                C::ASRV => Shift::Asr,
                _ => Shift::Ror,
            };
            Op::ShiftVar { op, w, dst, a: r(&o[1])?, b: r(&o[2])? }
        }
        C::MADD | C::MSUB => {
            let (dst, w) = reg(&o[0])?;
            Op::MulAdd { w, dst, a: r(&o[1])?, b: r(&o[2])?, acc: r(&o[3])?, sub: i.opcode == C::MSUB }
        }
        C::SMADDL | C::SMSUBL | C::UMADDL | C::UMSUBL => Op::MulAddLong {
            dst: r(&o[0])?,
            a: r(&o[1])?,
            b: r(&o[2])?,
            acc: r(&o[3])?,
            signed: matches!(i.opcode, C::SMADDL | C::SMSUBL),
            sub: matches!(i.opcode, C::SMSUBL | C::UMSUBL),
        },
        C::SMULH | C::UMULH => Op::MulHigh { dst: r(&o[0])?, a: r(&o[1])?, b: r(&o[2])?, signed: i.opcode == C::SMULH },
        C::UDIV | C::SDIV => {
            let (dst, w) = reg(&o[0])?;
            Op::Div { w, dst, a: r(&o[1])?, b: r(&o[2])?, signed: i.opcode == C::SDIV }
        }
        C::CSEL | C::CSINC | C::CSINV | C::CSNEG => {
            let (dst, w) = reg(&o[0])?;
            Op::CondSel {
                w,
                dst,
                a: r(&o[1])?,
                b: r(&o[2])?,
                cond: cond(&o[3])?,
                inc: matches!(i.opcode, C::CSINC | C::CSNEG),
                inv: matches!(i.opcode, C::CSINV | C::CSNEG),
            }
        }
        C::CCMP | C::CCMN => {
            let (a, w) = reg(&o[0])?;
            Op::CondCmp { w, a, b: operand(&o[1])?, nzcv: imm(&o[2])? as u8, cond: cond(&o[3])?, add: i.opcode == C::CCMN }
        }
        C::CLZ | C::RBIT | C::REV | C::REV16 | C::REV32 => {
            let (dst, w) = reg(&o[0])?;
            let op = match i.opcode {
                C::CLZ => Unary::Clz,
                C::RBIT => Unary::Rbit,
                C::REV => Unary::Rev,
                C::REV16 => Unary::Rev16,
                _ => Unary::Rev32,
            };
            Op::Unary { op, w, dst, src: r(&o[1])? }
        }
        C::ADR => Op::Const { dst: r(&o[0])?, value: pc_offset(&o[1], pc)? },
        C::ADRP => Op::Const { dst: r(&o[0])?, value: (pc & !0xFFF).wrapping_add(pc_offset(&o[1], 0)?) },
        C::LDR | C::LDUR | C::LDAR | C::LDAPR | C::LDAPUR => load(None, false)?,
        C::LDRB | C::LDURB | C::LDARB | C::LDAPRB | C::LDAPURB => load(Some(0), false)?,
        C::LDRH | C::LDURH | C::LDARH | C::LDAPRH | C::LDAPURH => load(Some(1), false)?,
        C::LDRSB | C::LDURSB | C::LDAPURSB => load(Some(0), true)?,
        C::LDRSH | C::LDURSH | C::LDAPURSH => load(Some(1), true)?,
        C::LDRSW | C::LDURSW | C::LDAPURSW => load(Some(2), true)?,
        C::STR | C::STUR | C::STLR | C::STLUR => store(None)?,
        C::STRB | C::STURB | C::STLRB | C::STLURB => store(Some(0))?,
        C::STRH | C::STURH | C::STLRH | C::STLURH => store(Some(1))?,
        C::LDP | C::LDNP | C::LDPSW => {
            let (a, w) = reg(&o[0])?;
            let signed = i.opcode == C::LDPSW;
            let size = if signed || w == W::W32 { 2 } else { 3 };
            Op::LoadPair { a, b: r(&o[1])?, w, size, signed, addr: addr(&o[2])? }
        }
        C::STP | C::STNP => {
            let (a, w) = reg(&o[0])?;
            Op::StorePair { a, b: r(&o[1])?, size: if w == W::W64 { 3 } else { 2 }, addr: addr(&o[2])? }
        }
        C::MRS => match o[1] {
            O::SystemReg(TPIDR_EL0) => Op::ReadThread { dst: r(&o[0])?, ro: false },
            O::SystemReg(TPIDRRO_EL0) => Op::ReadThread { dst: r(&o[0])?, ro: true },
            _ => return None,
        },
        // Hints, barriers and pointer authentication: the host needs none of them
        // (B-key signing is never checked, so stripping a pointer is a no-op too).
        C::HINT | C::PACIBSP | C::AUTIBSP | C::XPACI | C::XPACLRI | C::DMB(_) | C::DSB(_) | C::ISB | C::PRFM | C::PRFUM | C::CLREX => Op::Nop,
        C::B => Op::Jump(pc_offset(&o[0], pc)?),
        C::BL => Op::Call(pc_offset(&o[0], pc)?),
        C::Bcc(c) => Op::Branch(Cond::Flags(c), pc_offset(&o[0], pc)?),
        C::CBZ | C::CBNZ => {
            let (reg, w) = reg(&o[0])?;
            Op::Branch(Cond::Zero { reg, w, nonzero: i.opcode == C::CBNZ }, pc_offset(&o[1], pc)?)
        }
        C::TBZ | C::TBNZ => Op::Branch(Cond::Bit { reg: r(&o[0])?, bit: imm(&o[1])? as u8, set: i.opcode == C::TBNZ }, pc_offset(&o[2], pc)?),
        C::BR => Op::JumpIndirect(r(&o[0])?),
        C::BLR => Op::CallIndirect(r(&o[0])?),
        C::RET => match r(&o[0])? {
            Reg::X(30) => Op::Return,
            other => Op::JumpIndirect(other),
        },
        C::RETAB | C::RETAA => Op::Return,
        C::UDF | C::BRK | C::HLT => Op::Trap(imm(&o[0])? as u16),
        C::SVC => Op::Svc(imm(&o[0])? as u16),
        _ => return None,
    })
}
