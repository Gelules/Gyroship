//! From a decoded instruction to one IR operation.

use crate::ir::{Addr, Alu, Cond, FBin, FUn, Op, Operand, Reg, Select, Shift, Unary, Writeback, W};
use gyroship_analysis::decode::{decode, Instruction, Opcode, Operand as O, SIMDSizeCode, ShiftStyle, SizeCode};

/// MRS/MSR encodings of TPIDR_EL0 and TPIDRRO_EL0 as yaxpeax numbers them.
const TPIDR_EL0: u16 = 0x5E82;
const TPIDRRO_EL0: u16 = 0x5E83;
const CNTPCT_EL0: u16 = 0x5F01;
const CNTVCT_EL0: u16 = 0x5F02;

pub fn lift(word: u32, pc: u64) -> Op {
    // yield, wfe, wfi: hints for the decoder, waits for the runtime.
    if let 0xD503_203F | 0xD503_205F | 0xD503_207F = word {
        return Op::Yield(((word >> 5) & 7) as u8);
    }
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
    if let Some(op) = lift_simd(i) {
        return Some(op);
    }
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
            O::SystemReg(CNTPCT_EL0 | CNTVCT_EL0) => Op::Counter(r(&o[0])?),
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

fn simd_size(s: SIMDSizeCode) -> u8 {
    match s {
        SIMDSizeCode::B => 0,
        SIMDSizeCode::H => 1,
        SIMDSizeCode::S => 2,
        SIMDSizeCode::D => 3,
        SIMDSizeCode::Q => 4,
    }
}

/// A scalar FP/SIMD register: (number, log2 bytes).
fn vreg(o: &O) -> Option<(u8, u8)> {
    match *o {
        O::SIMDRegister(s, n) => Some((n as u8, simd_size(s))),
        _ => None,
    }
}

/// A single or double precision register: (number, is double).
fn freg(o: &O) -> Option<(u8, bool)> {
    match vreg(o)? {
        (n, 2) => Some((n, false)),
        (n, 3) => Some((n, true)),
        _ => None,
    }
}

/// A whole vector: (number, 16 bytes, log2 of the element bytes).
fn vector(o: &O) -> Option<(u8, bool, u8)> {
    match *o {
        O::SIMDRegisterElements(v, n, e) => Some((n as u8, v == SIMDSizeCode::Q, simd_size(e))),
        _ => None,
    }
}

/// `lane` (of `esize` log2 bytes) repeated over 8 or 16 bytes.
fn replicate(lane: u64, esize: u8, q: bool) -> u128 {
    let bits = 8u32 << esize;
    let lane = u128::from(if bits == 64 { lane } else { lane & ((1u64 << bits) - 1) });
    let total = if q { 128 } else { 64 };
    (0..total / bits).fold(0u128, |v, k| v | lane << (k * bits))
}

/// FP and SIMD forms the backends translate; None for everything else.
fn lift_simd(i: &Instruction) -> Option<Op> {
    use Opcode as C;
    let o = &i.operands;
    let f2 = |op: FBin| -> Option<Op> {
        let ((dst, d), (a, da), (b, db)) = (freg(&o[0])?, freg(&o[1])?, freg(&o[2])?);
        (d == da && d == db).then_some(Op::FBin { op, f64: d, dst, a, b })
    };
    let f1 = |op: FUn| -> Option<Op> {
        let ((dst, d), (src, ds)) = (freg(&o[0])?, freg(&o[1])?);
        (d == ds).then_some(Op::FUn { op, f64: d, dst, src })
    };
    let fma = |neg_acc: bool, neg_product: bool| -> Option<Op> {
        let ((dst, d), (a, _), (b, _), (acc, _)) = (freg(&o[0])?, freg(&o[1])?, freg(&o[2])?, freg(&o[3])?);
        Some(Op::FMulAdd { f64: d, dst, a, b, acc, neg_acc, neg_product })
    };
    Some(match i.opcode {
        C::LDR | C::LDUR | C::STR | C::STUR => {
            let (n, size) = vreg(&o[0])?;
            let a = addr(&o[1])?;
            if matches!(i.opcode, C::LDR | C::LDUR) { Op::LoadV { dst: n, size, addr: a } } else { Op::StoreV { src: n, size, addr: a } }
        }
        C::LDP | C::LDNP | C::STP | C::STNP => {
            let ((a, size), (b, _)) = (vreg(&o[0])?, vreg(&o[1])?);
            let at = addr(&o[2])?;
            if matches!(i.opcode, C::LDP | C::LDNP) { Op::LoadPairV { a, b, size, addr: at } } else { Op::StorePairV { a, b, size, addr: at } }
        }
        C::FMOV => match (&o[0], &o[1]) {
            (O::SIMDRegister(..), O::SIMDRegister(..)) => {
                let ((dst, size), (src, _)) = (vreg(&o[0])?, vreg(&o[1])?);
                Op::VMov { dst, src, size }
            }
            (O::SIMDRegister(..), O::Register(..)) => {
                let (dst, size) = vreg(&o[0])?;
                (size >= 2).then_some(Op::ToV { dst, src: r(&o[1])?, size })?
            }
            (O::Register(..), O::SIMDRegister(..)) => {
                let (src, size) = vreg(&o[1])?;
                (size >= 2).then_some(Op::FromV { dst: r(&o[0])?, src, size, lane: 0 })?
            }
            (O::Register(..), O::SIMDRegisterElementsLane(_, src, SIMDSizeCode::D, lane)) => {
                Op::FromV { dst: r(&o[0])?, src: *src as u8, size: 3, lane: *lane }
            }
            (O::SIMDRegister(..), O::ImmediateDouble(x)) => {
                let (dst, d) = freg(&o[0])?;
                let bits = if d { x.to_bits() } else { u64::from((*x as f32).to_bits()) };
                Op::VConst { dst, value: u128::from(bits) }
            }
            (O::SIMDRegisterElements(..), O::ImmediateDouble(x)) => {
                let (dst, q, e) = vector(&o[0])?;
                let lane = match e {
                    3 => x.to_bits(),
                    2 => u64::from((*x as f32).to_bits()),
                    _ => return None,
                };
                Op::VConst { dst, value: replicate(lane, e, q) }
            }
            _ => return None,
        },
        C::MOVI | C::MVNI => {
            let invert = i.opcode == C::MVNI;
            let (dst, value) = match (&o[0], &o[1]) {
                (O::SIMDRegister(SIMDSizeCode::D, n), O::Imm64(v)) => (*n as u8, u128::from(*v)),
                (O::SIMDRegisterElements(..), imm) => {
                    let (dst, q, e) = vector(&o[0])?;
                    let lane = match *imm {
                        O::Imm64(v) => v,
                        O::Immediate(v) => u64::from(v),
                        O::ImmShift(v, s) => u64::from(v) << s,
                        O::ImmShiftMSL(v, s) => (u64::from(v) << s) | ((1u64 << s) - 1),
                        _ => return None,
                    };
                    (dst, replicate(if invert { !lane } else { lane }, e, q))
                }
                _ => return None,
            };
            Op::VConst { dst, value }
        }
        C::UMOV => {
            let O::SIMDRegisterElementsLane(_, src, e, lane) = o[1] else { return None };
            Op::FromV { dst: r(&o[0])?, src: src as u8, size: simd_size(e), lane }
        }
        C::ORR | C::AND | C::EOR | C::BIC | C::ORN => {
            let ((dst, q, _), (a, _, _), (b, _, _)) = (vector(&o[0])?, vector(&o[1])?, vector(&o[2])?);
            let op = match i.opcode {
                C::ORR => Alu::Orr,
                C::AND => Alu::And,
                C::EOR => Alu::Eor,
                C::BIC => Alu::Bic,
                _ => Alu::Orn,
            };
            if op == Alu::Orr && a == b { Op::VMov { dst, src: a, size: if q { 4 } else { 3 } } } else { Op::VLogic { op, q, dst, a, b } }
        }
        C::FADD => f2(FBin::Add)?,
        C::FSUB => f2(FBin::Sub)?,
        C::FMUL => f2(FBin::Mul)?,
        C::FDIV => f2(FBin::Div)?,
        C::FMAX => f2(FBin::Max)?,
        C::FMIN => f2(FBin::Min)?,
        C::FMAXNM => f2(FBin::MaxNm)?,
        C::FMINNM => f2(FBin::MinNm)?,
        C::FNMUL => f2(FBin::Nmul)?,
        C::FABD => f2(FBin::Abd)?,
        C::FNEG => f1(FUn::Neg)?,
        C::FABS => f1(FUn::Abs)?,
        C::FSQRT => f1(FUn::Sqrt)?,
        C::FRINTM => f1(FUn::Floor)?,
        C::FRINTP => f1(FUn::Ceil)?,
        C::FRINTZ => f1(FUn::Trunc)?,
        C::FRINTN | C::FRINTX | C::FRINTI => f1(FUn::Nearest)?,
        C::FRINTA => f1(FUn::Round)?,
        C::BSL | C::BIT | C::BIF => {
            let ((dst, q, _), (a, _, _), (b, _, _)) = (vector(&o[0])?, vector(&o[1])?, vector(&o[2])?);
            let kind = match i.opcode {
                C::BSL => Select::Bsl,
                C::BIT => Select::Bit,
                _ => Select::Bif,
            };
            Op::VSelect { kind, q, dst, a, b }
        }
        C::FMADD => fma(false, false)?,
        C::FMSUB => fma(false, true)?,
        C::FNMADD => fma(true, true)?,
        C::FNMSUB => fma(true, false)?,
        C::FCMP | C::FCMPE => {
            let (a, d) = freg(&o[0])?;
            match o[1] {
                O::ImmediateDouble(_) => Op::FCmp { f64: d, a, b: None }, // FCMP #imm only encodes +0.0
                _ => Op::FCmp { f64: d, a, b: Some(freg(&o[1])?.0) },
            }
        }
        C::FCCMP | C::FCCMPE => {
            let ((a, d), (b, _)) = (freg(&o[0])?, freg(&o[1])?);
            Op::FCondCmp { f64: d, a, b, nzcv: imm(&o[2])? as u8, cond: cond(&o[3])? }
        }
        C::FCSEL => {
            let ((dst, d), (a, _), (b, _)) = (freg(&o[0])?, freg(&o[1])?, freg(&o[2])?);
            Op::FCondSel { f64: d, dst, a, b, cond: cond(&o[3])? }
        }
        C::FCVT => {
            let ((dst, d), (src, s)) = (freg(&o[0])?, freg(&o[1])?);
            (d != s).then_some(Op::FCvt { dst, src, to_f64: d })?
        }
        C::SCVTF | C::UCVTF => {
            if !matches!(o[2], O::Nothing) {
                return None; // fixed-point form
            }
            let (dst, d) = freg(&o[0])?;
            let (src, w) = reg(&o[1])?;
            Op::IntToF { dst, src, w, f64: d, signed: i.opcode == C::SCVTF }
        }
        C::FCVTZS | C::FCVTZU => {
            if !matches!(o[2], O::Nothing) {
                return None;
            }
            let (dst, w) = reg(&o[0])?;
            let (src, d) = freg(&o[1])?;
            Op::FToInt { dst, w, src, f64: d, signed: i.opcode == C::FCVTZS }
        }
        _ => return None,
    })
}
