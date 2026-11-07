//! NVIDIA SASS for SM 8.x (Ampere; the console runs SM 8.8), as the game
//! ships it: precompiled `nvn_gen2.raw` blobs in `data.zip`.
//!
//! A blob is a 0x18-byte header (magic 0x19866891, code size, control size,
//! scratch size, u64 hash), the code section the GPU executes, then NVN's
//! control section (magic 0x98761234), which only the driver reads.
//! Graphics code starts with a 0x80-byte shader program header (SPH);
//! compute code has none. Instructions are 128 bits, little-endian: the
//! opcode in bits 0..12 (for ALU operations, bits 9..12 are the operand
//! form), the guard predicate in 12..15 (7 = PT) and its negation in 15,
//! scheduling controls in 105..126. Field positions come from the public
//! encodings (open-gpu-doc for the SPH, Mesa's NAK for instructions).

pub mod control;
pub mod fp16;
pub mod fp32;
pub mod int;
pub mod memory;
pub mod moves;
pub mod operand;
pub mod texture;

/// What the instruction families share.
pub(crate) mod prelude {
    pub use super::operand::*;
    pub use super::{Word, bits};
    #[cfg(test)]
    pub use super::{ALWAYS_EXPLAINED, ranges};
}

/// A decoded instruction: one family's operation.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Op {
    Fp32(fp32::Fp32Op),
    Fp16(fp16::Fp16Op),
    Int(int::IntOp),
    Moves(moves::MovesOp),
    Memory(memory::MemoryOp),
    Texture(texture::TextureOp),
    Control(control::ControlOp),
}

/// Decodes one instruction; `None` for an opcode no family knows.
pub fn decode(w: Word) -> Option<Op> {
    fp32::decode(w)
        .map(Op::Fp32)
        .or_else(|| fp16::decode(w).map(Op::Fp16))
        .or_else(|| int::decode(w).map(Op::Int))
        .or_else(|| moves::decode(w).map(Op::Moves))
        .or_else(|| memory::decode(w).map(Op::Memory))
        .or_else(|| texture::decode(w).map(Op::Texture))
        .or_else(|| control::decode(w).map(Op::Control))
}

/// The bits the decoder reads for an opcode, with the opcode, guard and
/// scheduling bits every instruction has.
pub fn explained(opcode: u16) -> u128 {
    ALWAYS_EXPLAINED | fp32::explained(opcode) | fp16::explained(opcode) | int::explained(opcode) | moves::explained(opcode) | memory::explained(opcode) | texture::explained(opcode) | control::explained(opcode)
}

/// Bits every instruction has that are not operands: opcode 0..12, guard
/// 12..16, scheduling controls 105..128.
pub const ALWAYS_EXPLAINED: u128 = 0xffff | (!0u128 << 105);

/// A mask of bits lo..lo+len.
pub const fn bits(lo: u32, len: u32) -> u128 {
    ((1u128 << len) - 1) << lo
}

/// The bit ranges of a mask, for reports: "32..40 64..72".
pub fn ranges(mask: u128) -> String {
    let mut out = Vec::new();
    let mut i = 0;
    while i < 128 {
        if mask >> i & 1 != 0 {
            let start = i;
            while i < 128 && mask >> i & 1 != 0 {
                i += 1;
            }
            out.push(format!("{start}..{i}"));
        } else {
            i += 1;
        }
    }
    out.join(" ")
}

pub const BLOB_MAGIC: u32 = 0x1986_6891;
const HEADER: usize = 0x18;
pub const SPH_SIZE: usize = 0x80;

/// A shader blob, borrowed from its bytes.
#[derive(Clone, Copy, Debug)]
pub struct Blob<'a> {
    pub hash: u64,
    pub scratch: u32,
    pub code: &'a [u8],
    pub control: &'a [u8],
}

impl<'a> Blob<'a> {
    pub fn parse(b: &'a [u8]) -> Option<Blob<'a>> {
        let u32_at = |o: usize| Some(u32::from_le_bytes(b.get(o..o + 4)?.try_into().ok()?));
        if u32_at(0)? != BLOB_MAGIC {
            return None;
        }
        let (code, control) = (u32_at(4)? as usize, u32_at(8)? as usize);
        if HEADER + code + control != b.len() {
            return None;
        }
        Some(Blob {
            hash: u64::from_le_bytes(b.get(0x10..0x18)?.try_into().ok()?),
            scratch: u32_at(0xc)?,
            code: &b[HEADER..HEADER + code],
            control: &b[HEADER + code..],
        })
    }

    /// The stage, from the SPH (compute code has none).
    pub fn stage(&self) -> Stage {
        let w0 = self.code.get(0..4).map_or(0, |w| u32::from_le_bytes(w.try_into().expect("4 bytes")));
        // SPH word 0: SphType 4:0 (1 VTG, 2 PS), Version 9:5, ShaderType 13:10.
        match (w0 & 0x1f, w0 >> 5 & 0x1f, w0 >> 10 & 0xf) {
            (1, 4, 1) => Stage::Vertex,
            (1, 4, 2) => Stage::TessControl,
            (1, 4, 3) => Stage::TessEval,
            (1, 4, 4) => Stage::Geometry,
            (2, 4, 5) => Stage::Pixel,
            _ => Stage::Compute,
        }
    }

    /// The SASS: after the SPH for graphics stages.
    pub fn sass(&self) -> &'a [u8] {
        if self.stage() == Stage::Compute { self.code } else { self.code.get(SPH_SIZE..).unwrap_or_default() }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum Stage {
    Vertex,
    TessControl,
    TessEval,
    Geometry,
    Pixel,
    Compute,
}

/// One 128-bit instruction.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Word(pub u128);

impl Word {
    pub fn from_bytes(b: &[u8; 16]) -> Word {
        Word(u128::from_le_bytes(*b))
    }

    /// Bits lo..lo+len.
    pub fn field(self, lo: u32, len: u32) -> u64 {
        ((self.0 >> lo) & ((1u128 << len) - 1)) as u64
    }

    pub fn bit(self, i: u32) -> bool {
        self.0 >> i & 1 != 0
    }

    /// Bits lo..lo+len, sign-extended.
    pub fn signed(self, lo: u32, len: u32) -> i64 {
        let v = self.field(lo, len) as i64;
        (v << (64 - len)) >> (64 - len)
    }

    /// The 12-bit opcode (with the operand form for ALU operations).
    pub fn opcode(self) -> u16 {
        self.field(0, 12) as u16
    }

    /// The guard predicate: register (7 = PT) and negation.
    pub fn guard(self) -> (u8, bool) {
        (self.field(12, 3) as u8, self.bit(15))
    }

    /// Whether the guard is "always": @PT.
    pub fn always(self) -> bool {
        self.guard() == (7, false)
    }
}

/// The instructions of some SASS bytes.
pub fn words(sass: &[u8]) -> impl Iterator<Item = Word> + '_ {
    sass.as_chunks::<16>().0.iter().map(Word::from_bytes)
}

pub const BRA: u16 = 0x947;
pub const NOP: u16 = 0x918;

/// The program: every instruction up to and including the closing
/// unconditional branch to itself (the compilers' epilogue). Everything
/// after it is padding or data; `None` when there is no such branch.
pub fn program(sass: &[u8]) -> Option<&[u8]> {
    let end = words(sass).enumerate().position(|(_, w)| w.opcode() == BRA && w.always() && w.field(87, 4) == 7 && w.signed(32, 50) == -16)?;
    Some(&sass[..16 * (end + 1)])
}

/// The key of a graphics program, from its code as it lies at its GPU
/// address or in a blob's code section (SPH, then SASS): FNV-1a 64 of the
/// SPH and the program up to its closing self-branch, and that length.
/// Offline translation and the runtime's shader cache agree on it.
pub fn program_key(code: &[u8]) -> Option<(u64, usize)> {
    let len = SPH_SIZE + program(code.get(SPH_SIZE..)?)?.len();
    Some((code[..len].iter().fold(0xcbf2_9ce4_8422_2325u64, |h, &b| (h ^ u64::from(b)).wrapping_mul(0x100_0000_01b3)), len))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_program_key_covers_the_sph_and_the_program_only() {
        let mut code = vec![0u8; SPH_SIZE];
        code.extend(insn(0x94d, 7 << 87));
        code.extend(insn(BRA, 7 << 87 | (((-16i64) as u64 as u128) & ((1 << 50) - 1)) << 32));
        let (key, len) = program_key(&code).unwrap();
        assert_eq!(len, SPH_SIZE + 32);
        // What follows the program does not change the key; the program does.
        code.extend([0x55; 16]);
        assert_eq!(program_key(&code).unwrap(), (key, len));
        code[SPH_SIZE] ^= 1;
        assert_ne!(program_key(&code).unwrap().0, key);
        assert!(program_key(&code[..SPH_SIZE + 16]).is_none());
    }

    fn insn(opcode: u16, extra: u128) -> [u8; 16] {
        // @PT, operands from `extra`.
        (u128::from(opcode) | 7 << 12 | extra).to_le_bytes()
    }

    #[test]
    fn a_blob_parses_and_its_program_ends_at_the_self_branch() {
        let mut code = Vec::new();
        // SPH word 0 of a vertex shader (SphType 1, Version 4, ShaderType 1), then zeros.
        code.extend(0x020c_0481u32.to_le_bytes());
        code.resize(SPH_SIZE, 0);
        let exit = insn(0x94d, 7 << 87);
        let bra_self = insn(BRA, 7 << 87 | (((-16i64) as u64 as u128) & ((1 << 50) - 1)) << 32);
        let nop = insn(NOP, 0);
        for i in [exit, bra_self, nop, nop] {
            code.extend(i);
        }
        code.extend([0xab; 16]); // a table after the code
        let control = [0x34, 0x12, 0x76, 0x98];
        let mut blob = Vec::new();
        for v in [BLOB_MAGIC, code.len() as u32, control.len() as u32, 0] {
            blob.extend(v.to_le_bytes());
        }
        blob.extend(0x1122_3344_5566_7788u64.to_le_bytes());
        blob.extend(&code);
        blob.extend(control);
        let b = Blob::parse(&blob).unwrap();
        assert_eq!((b.hash, b.stage(), b.control.len()), (0x1122_3344_5566_7788, Stage::Vertex, 4));
        let program = program(b.sass()).unwrap();
        assert_eq!(program.len(), 32);
        assert_eq!(words(program).map(|w| w.opcode()).collect::<Vec<_>>(), [0x94d, BRA]);
        assert!(Blob::parse(&blob[..blob.len() - 1]).is_none());
    }

    #[test]
    fn fields_and_sign() {
        let w = Word(0xf << 32 | 0b101 << 12 | 1 << 15);
        assert_eq!((w.field(32, 4), w.signed(32, 4), w.guard(), w.always()), (0xf, -1, (5, true), false));
    }
}
