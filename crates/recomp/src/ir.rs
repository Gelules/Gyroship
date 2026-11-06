//! A small IR between decoded AArch64 instructions and the backends: one
//! operation per guest instruction, with every operand resolved (registers,
//! immediates, PC-relative addresses).

/// Operation width. 32-bit results are zero-extended into the 64-bit register.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum W {
    W32,
    W64,
}

impl W {
    pub fn bits(self) -> u32 {
        match self {
            W::W32 => 32,
            W::W64 => 64,
        }
    }
}

/// A general register operand.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Reg {
    X(u8),
    /// Reads as 0, writes are dropped.
    Zr,
    Sp,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Shift {
    Lsl,
    Lsr,
    Asr,
    Ror,
}

/// The second operand of data-processing instructions.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Operand {
    Imm(u64),
    /// A register shifted by a constant.
    Shifted(Reg, Shift, u8),
    /// A register's low `bits` bits, sign- or zero-extended, then shifted left.
    Extended { reg: Reg, signed: bool, bits: u8, lsl: u8 },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Alu {
    Add,
    Sub,
    And,
    Orr,
    Eor,
    /// a & !b
    Bic,
    /// a | !b
    Orn,
    /// a ^ !b
    Eon,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Writeback {
    None,
    /// The base is updated to the address before the access.
    Pre,
    /// The access uses the base; the base then moves by `offset`.
    Post,
}

/// A memory address: base + offset, or base + extended index.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Addr {
    pub base: Reg,
    pub offset: i64,
    /// (index register, sign-extended, from bits, shift left).
    pub index: Option<(Reg, bool, u8, u8)>,
    pub writeback: Writeback,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Cond {
    /// An AArch64 condition code on NZCV.
    Flags(u8),
    /// CBZ (false) / CBNZ (true).
    Zero { reg: Reg, w: W, nonzero: bool },
    /// TBZ (false) / TBNZ (true).
    Bit { reg: Reg, bit: u8, set: bool },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Op {
    Alu { op: Alu, w: W, dst: Reg, a: Reg, b: Operand, flags: bool },
    /// MOVZ, MOVN (`invert`), MOVK (`keep`).
    MovWide { w: W, dst: Reg, imm: u16, shift: u8, invert: bool, keep: bool },
    /// UBFM, SBFM (`signed`), BFM (`insert`).
    Bitfield { w: W, dst: Reg, src: Reg, immr: u8, imms: u8, signed: bool, insert: bool },
    /// EXTR: (a:b) >> lsb.
    Extr { w: W, dst: Reg, a: Reg, b: Reg, lsb: u8 },
    /// LSLV, LSRV, ASRV, RORV.
    ShiftVar { op: Shift, w: W, dst: Reg, a: Reg, b: Reg },
    /// MADD/MSUB: acc ± a * b.
    MulAdd { w: W, dst: Reg, a: Reg, b: Reg, acc: Reg, sub: bool },
    /// SMADDL/UMADDL/SMSUBL/UMSUBL: acc ± ext32(a) * ext32(b).
    MulAddLong { dst: Reg, a: Reg, b: Reg, acc: Reg, signed: bool, sub: bool },
    /// SMULH/UMULH.
    MulHigh { dst: Reg, a: Reg, b: Reg, signed: bool },
    /// UDIV/SDIV (division by zero gives 0).
    Div { w: W, dst: Reg, a: Reg, b: Reg, signed: bool },
    /// CSEL; CSINC (`inc`), CSINV (`inv`), CSNEG (both).
    CondSel { w: W, dst: Reg, a: Reg, b: Reg, cond: u8, inc: bool, inv: bool },
    /// CCMP/CCMN (`add`): flags of a - b (a + b) if cond, else `nzcv`.
    CondCmp { w: W, a: Reg, b: Operand, nzcv: u8, cond: u8, add: bool },
    /// CLZ, RBIT, REV, REV16, REV32.
    Unary { op: Unary, w: W, dst: Reg, src: Reg },
    /// ADR/ADRP, resolved.
    Const { dst: Reg, value: u64 },
    /// Loads 1 << size bytes, extended to `w` (sign-extended if `signed`).
    Load { dst: Reg, w: W, size: u8, signed: bool, addr: Addr },
    Store { src: Reg, size: u8, addr: Addr },
    LoadPair { a: Reg, b: Reg, w: W, size: u8, signed: bool, addr: Addr },
    StorePair { a: Reg, b: Reg, size: u8, addr: Addr },
    /// MRS of TPIDR_EL0 / TPIDRRO_EL0.
    ReadThread { dst: Reg, ro: bool },
    Nop,
    // Control flow: the last operation of a block.
    Jump(u64),
    Branch(Cond, u64),
    Call(u64),
    CallIndirect(Reg),
    JumpIndirect(Reg),
    Return,
    Trap(u16),
    Svc(u16),
    /// Not translated: the runtime gets the PC and the word.
    Unhandled,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Unary {
    Clz,
    Rbit,
    Rev,
    Rev16,
    Rev32,
}
