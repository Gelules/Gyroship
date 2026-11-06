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
    // FP and SIMD registers. `size` is log2 of the bytes accessed (0 = B .. 4 = Q);
    // writing fewer than 16 bytes zeroes the rest of the register.
    LoadV { dst: u8, size: u8, addr: Addr },
    StoreV { src: u8, size: u8, addr: Addr },
    LoadPairV { a: u8, b: u8, size: u8, addr: Addr },
    StorePairV { a: u8, b: u8, size: u8, addr: Addr },
    /// MOVI/MVNI/FMOV #imm: the whole register.
    VConst { dst: u8, value: u128 },
    /// FMOV between FP registers, MOV of a whole vector: the low 1 << size bytes.
    VMov { dst: u8, src: u8, size: u8 },
    /// FMOV from a general register into the low 4 or 8 bytes (`size` 2 or 3).
    ToV { dst: u8, src: Reg, size: u8 },
    /// FMOV/UMOV of element `lane` (of 1 << size bytes) into a general register.
    FromV { dst: Reg, src: u8, size: u8, lane: u8 },
    /// AND/ORR/EOR/BIC/ORN on 8 (`q` false) or 16 bytes.
    VLogic { op: Alu, q: bool, dst: u8, a: u8, b: u8 },
    /// BSL/BIT/BIF on 8 or 16 bytes.
    VSelect { kind: Select, q: bool, dst: u8, a: u8, b: u8 },
    /// Scalar FP on single (`f64` false) or double precision.
    FBin { op: FBin, f64: bool, dst: u8, a: u8, b: u8 },
    FUn { op: FUn, f64: bool, dst: u8, src: u8 },
    /// FMADD acc + a*b; FMSUB acc - a*b; FNMADD -acc - a*b; FNMSUB -acc + a*b (fused).
    FMulAdd { f64: bool, dst: u8, a: u8, b: u8, acc: u8, neg_acc: bool, neg_product: bool },
    /// FCMP/FCMPE against a register or (None) +0.0.
    FCmp { f64: bool, a: u8, b: Option<u8> },
    FCondCmp { f64: bool, a: u8, b: u8, nzcv: u8, cond: u8 },
    FCondSel { f64: bool, dst: u8, a: u8, b: u8, cond: u8 },
    /// FCVT between single and double.
    FCvt { dst: u8, src: u8, to_f64: bool },
    /// SCVTF/UCVTF from a general register.
    IntToF { dst: u8, src: Reg, w: W, f64: bool, signed: bool },
    /// FCVTZS/FCVTZU to a general register (saturating, NaN gives 0).
    FToInt { dst: Reg, w: W, src: u8, f64: bool, signed: bool },
    // Control flow: the last operation of a block.
    Jump(u64),
    Branch(Cond, u64),
    Call(u64),
    CallIndirect(Reg),
    JumpIndirect(Reg),
    Return,
    Trap(u16),
    Svc(u16),
    /// mrs xN, CNTPCT_EL0 (or CNTVCT_EL0): the runtime's virtual counter.
    Counter(Reg),
    /// yield (1), wfe (2) or wfi (3): the guest waits, so the runtime may switch threads.
    Yield(u8),
    /// Not translated: the runtime gets the PC and the word.
    Unhandled,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FBin {
    Add,
    Sub,
    Mul,
    Div,
    /// NaN propagates; +0 > -0.
    Max,
    Min,
    /// A quiet NaN loses against a number.
    MaxNm,
    MinNm,
    /// -(a * b)
    Nmul,
    /// |a - b|
    Abd,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FUn {
    Neg,
    Abs,
    Sqrt,
    /// FRINTM, FRINTP, FRINTZ.
    Floor,
    Ceil,
    Trunc,
    /// FRINTN, and FRINTX/FRINTI in the default rounding mode: ties to even.
    Nearest,
    /// FRINTA: ties away from zero.
    Round,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Select {
    /// BSL: dst selects between a (1 bits) and b (0 bits).
    Bsl,
    /// BIT: a goes into dst where b is 1.
    Bit,
    /// BIF: a goes into dst where b is 0.
    Bif,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Unary {
    Clz,
    Rbit,
    Rev,
    Rev16,
    Rev32,
}
