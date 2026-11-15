//! Texture family (SM 8.x): TEX, TLD, TLD4, TXD and TXQ.
//!
//! Contract for the integration:
//! - `pub enum TextureOp` with typed variants (our own code, written from the
//!   encoding facts);
//! - `pub fn decode(w: Word) -> Option<TextureOp>`: Some for every owned
//!   opcode, None for any other;
//! - `pub fn explained(opcode: u16) -> u128`: the bits the decoder reads for
//!   that opcode (ALWAYS_EXPLAINED is added by the check);
//! - `pub const OPCODES: &[u16]`: the 12-bit opcodes the family owns.
//!
//! Opcode layout. The bindless forms (handle in a register, `.B`) have their
//! own opcodes, whose low 9 bits differ from the other forms': TEX 0x361 for
//! 0xb60/0xf60 (NAK also has TLD 0x367, TLD4 0x364, TXD 0x36d and TXQ 0x370
//! for 0xb66, 0xb63, 0xb6c and 0xb6f). The game uses only the bindless TEX.
//! For the other forms, bits 9..12 say where the handle comes from: 5 = a
//! constant buffer, 7 = a uniform register. Forms 1 and 5 and every field position
//! below come from Mesa NAK's SM70 encoder (`OpTex/OpTld/OpTld4/OpTxd/OpTxq::
//! encode`, `set_tex_cb_ref`, `set_tex_dim`, `set_tex_lod_mode`,
//! `set_eviction_priority`, `set_tex_ndv`), SM < 100 branches. NAK never emits
//! form 7 (0xf60, 0xf63, 0xf6c); see [`TexHandle::UReg`] for how it was
//! identified.
//!
//! Fields shared by the sampling instructions (TEX, TLD, TLD4, TXD):
//! - 16..24 `dst`: first destination vector (the first two returned
//!   components); 64..72 `dst2`: the remaining components (RZ when unused).
//!   Which components are returned is `mask` (72..76, bit 0 = R .. bit 3 = A).
//! - 24..32 `a` and 32..40 `b`: the two source register vectors. Their
//!   content depends on the operation (NAK's nak_nir_lower_tex): `a` holds the
//!   array index or min-lod, then the coordinates; `b` holds the bindless
//!   handle, lod or bias, packed offsets, depth reference, sample index. For
//!   TXD, `a` holds the bindless handle and coordinates and `b` the
//!   interleaved derivatives. With `.SCR` (bit 60) the compiler rebalances up
//!   to four scalars between `a` and `b` (e.g. a 2D coordinate as a = u,
//!   b = v).
//! - 81..84 `fault`: sparse-residency predicate destination (PT = none).
//! - 84..87 eviction priority; 90 `.NODEP`.

use super::prelude::*;

pub const OPCODES: &[u16] = &[0xf60, 0xb60, 0xf6c, 0x361, 0xb66, 0xb63, 0xf63, 0xb6c, 0xb6f];

/// Where the texture (and sampler) handle comes from.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TexHandle {
    /// Form 1 (`.B`; NAK sets bit 59 with it, and so does every sample): the
    /// 32-bit handle is the first register of the `b` vector (of the `a`
    /// vector for TXD).
    Bindless,
    /// Form 5: handle read from c[bank][offset]; bits 40..54 hold the byte
    /// offset divided by 4 and 54..59 the bank (NAK `set_tex_cb_ref`).
    /// `offset` is in bytes.
    CBuf { bank: u8, offset: u16 },
    /// Form 7 (not in NAK): handle in the uniform register at bits 40..46.
    /// Evidence: bit 91 is set in every form-7 sample and in no other (bit 91
    /// marks uniform-register operands elsewhere, and NAK's SM100 encoder puts
    /// a UR handle at bit 40); bits 40..46 only hold even numbers 4..14 and
    /// 46..59 are always zero; in the same shaders ULDC loads URn and URn+1
    /// from the binding constant buffer, with URn+1 often repeating the same
    /// slot (a shared sampler). So URn is most likely the texture and URn+1
    /// the sampler. `bit91` is kept raw: its exact meaning is unproven.
    UReg { ureg: u8, bit91: bool },
}

/// Texture dimension, bits 61..64 (NAK `set_tex_dim`).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
pub enum TexDim {
    D1,
    D2,
    D3,
    Cube,
    Array1D,
    Array2D,
    ArrayCube,
    Unknown(u8),
}

/// Level-of-detail mode, bits 87..90 (NAK `set_tex_lod_mode`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LodMode {
    /// Implicit (derivatives).
    Auto,
    /// `.LZ`: level 0.
    Zero,
    /// `.LB`: bias from the `b` vector.
    Bias,
    /// `.LL`: explicit level from the `b` vector.
    Lod,
    /// `.LC`: implicit with a min-lod clamp (in the `a` vector).
    Clamp,
    /// `.LB.LC`.
    BiasClamp,
    Unknown(u8),
}

/// Cache eviction priority, bits 84..87 (NAK `set_eviction_priority`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Eviction {
    First,
    /// The default (no suffix).
    Normal,
    Last,
    LastUse,
    Unchanged,
    NoAllocate,
    Unknown(u8),
}

/// TLD4 offset mode, bits 76..78.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Tld4Offset {
    None,
    /// `.AOFFI`: one packed offset (8 bits per component) in `b`.
    AOffI,
    /// `.PTP`: one offset per gathered texel, two registers in `b`.
    PerPixel,
    Unknown(u8),
}

/// TXQ query, bits 62..64.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TexQuery {
    /// TEX_HEADER_DIMENSION (`a` = mip level).
    Dimension,
    /// TEX_HEADER_TEXTURE_TYPE.
    TextureType,
    /// TEX_HEADER_SAMPLER_POS.
    SamplerPos,
    Unknown(u8),
}

/// The destinations, sources and memory fields of a sampling instruction.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct TexIo {
    pub dst: u8,
    pub dst2: u8,
    pub fault: Pred,
    pub handle: TexHandle,
    pub a: Operand,
    pub b: Operand,
    pub mask: u8,
    pub eviction: Eviction,
    pub nodep: bool,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TextureOp {
    /// TEX (0xb60, 0xf60, bindless 0x361): sample. `scr` bit 60, `aoffi`
    /// bit 76, `ndv` bit 77 (non-divergent derivatives), `dc` bit 78 (depth
    /// compare).
    /// `f16` bit 79: the results are packed halves, (c0, c1) in the first destination
    /// and (c2, c3) in the second (NAK notes bits 79..81 as .F16; only 79 is seen).
    Tex { io: TexIo, dim: TexDim, lod: LodMode, scr: bool, aoffi: bool, ndv: bool, dc: bool, f16: bool },
    /// TLD (0xb66): texel fetch. `scr` bit 60, `aoffi` bit 76, `ms` bit 78
    /// (multisampled: sample index in `b`).
    Tld { io: TexIo, dim: TexDim, lod: LodMode, scr: bool, aoffi: bool, ms: bool, f16: bool },
    /// TLD4 (0xb63, 0xf63): gather. `comp` bits 87..89 (0 = R .. 3 = A),
    /// `scr` bit 60, `dc` bit 78.
    Tld4 { io: TexIo, dim: TexDim, comp: u8, offset: Tld4Offset, scr: bool, dc: bool },
    /// TXD (0xb6c, 0xf6c): sample with explicit derivatives. `aoffi` bit 76.
    Txd { io: TexIo, dim: TexDim, aoffi: bool },
    /// TXQ (0xb6f): texture header query. Source at 24..32, no fault
    /// predicate and no eviction field (bits 32..40 and 81..87 stay zero).
    Txq { dst: u8, dst2: u8, handle: TexHandle, src: Operand, query: TexQuery, mask: u8, nodep: bool },
}

fn handle(w: Word) -> TexHandle {
    match w.field(9, 3) {
        5 => TexHandle::CBuf { bank: w.field(54, 5) as u8, offset: (w.field(40, 14) * 4) as u16 },
        7 => TexHandle::UReg { ureg: ureg(w, 40), bit91: w.bit(91) },
        _ => TexHandle::Bindless,
    }
}

fn dim(w: Word) -> TexDim {
    use TexDim::*;
    match w.field(61, 3) as u8 {
        0 => D1,
        1 => D2,
        2 => D3,
        3 => Cube,
        4 => Array1D,
        5 => Array2D,
        7 => ArrayCube,
        v => Unknown(v),
    }
}

fn lod(w: Word) -> LodMode {
    use LodMode::*;
    match w.field(87, 3) as u8 {
        0 => Auto,
        1 => Zero,
        2 => Bias,
        3 => Lod,
        4 => Clamp,
        5 => BiasClamp,
        v => Unknown(v),
    }
}

fn eviction(w: Word) -> Eviction {
    use Eviction::*;
    match w.field(84, 3) as u8 {
        0 => First,
        1 => Normal,
        2 => Last,
        3 => LastUse,
        4 => Unchanged,
        5 => NoAllocate,
        v => Unknown(v),
    }
}

fn io(w: Word) -> TexIo {
    TexIo {
        dst: dst(w),
        dst2: reg(w, 64),
        fault: Pred { reg: w.field(81, 3) as u8, not: false },
        handle: handle(w),
        a: Operand::plain(Src::Reg(reg(w, 24))),
        b: Operand::plain(Src::Reg(reg(w, 32))),
        mask: w.field(72, 4) as u8,
        eviction: eviction(w),
        nodep: w.bit(90),
    }
}

pub fn decode(w: Word) -> Option<TextureOp> {
    let op = w.opcode();
    // The bindless TEX always carries .B (bit 59): without it, it is not one we know.
    if !OPCODES.contains(&op) || (op == 0x361 && !w.bit(59)) {
        return None;
    }
    let scr = w.bit(60);
    Some(match op & 0x1ff {
        0x160 | 0x161 => TextureOp::Tex { io: io(w), dim: dim(w), lod: lod(w), scr, aoffi: w.bit(76), ndv: w.bit(77), dc: w.bit(78), f16: w.bit(79) },
        0x166 => TextureOp::Tld { io: io(w), dim: dim(w), lod: lod(w), scr, aoffi: w.bit(76), ms: w.bit(78), f16: w.bit(79) },
        0x163 => {
            let offset = match w.field(76, 2) as u8 {
                0 => Tld4Offset::None,
                1 => Tld4Offset::AOffI,
                2 => Tld4Offset::PerPixel,
                v => Tld4Offset::Unknown(v),
            };
            TextureOp::Tld4 { io: io(w), dim: dim(w), comp: w.field(87, 2) as u8, offset, scr, dc: w.bit(78) }
        }
        0x16c => TextureOp::Txd { io: io(w), dim: dim(w), aoffi: w.bit(76) },
        _ => {
            let query = match w.field(62, 2) as u8 {
                0 => TexQuery::Dimension,
                1 => TexQuery::TextureType,
                2 => TexQuery::SamplerPos,
                v => TexQuery::Unknown(v),
            };
            TextureOp::Txq { dst: dst(w), dst2: reg(w, 64), handle: handle(w), src: Operand::plain(Src::Reg(reg(w, 24))), query, mask: w.field(72, 4) as u8, nodep: w.bit(90) }
        }
    })
}

/// dst, a, b, dst2, mask, fault, eviction, nodep.
const IO: u128 = bits(16, 24) | bits(64, 12) | bits(81, 6) | bits(90, 1);

pub fn explained(opcode: u16) -> u128 {
    if !OPCODES.contains(&opcode) {
        return 0;
    }
    let handle = match opcode >> 9 {
        // Bit 59 (.B) is required by decode() and implied by the opcode.
        1 => bits(59, 1),
        5 => bits(40, 19),
        7 => bits(40, 6) | bits(91, 1),
        _ => 0,
    };
    handle
        | match opcode & 0x1ff {
            0x160 | 0x161 => IO | bits(60, 4) | bits(76, 4) | bits(87, 3),
            0x166 => IO | bits(60, 4) | bits(76, 1) | bits(78, 2) | bits(87, 3),
            0x163 => IO | bits(60, 4) | bits(76, 3) | bits(87, 2),
            0x16c => IO | bits(61, 3) | bits(76, 1),
            _ => bits(16, 16) | bits(62, 2) | bits(64, 12) | bits(90, 1),
        }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Builds a word from (lo, len, value) fields.
    fn word(fields: &[(u32, u32, u128)]) -> Word {
        Word(fields.iter().fold(0, |w, &(lo, len, v)| {
            assert!(v < 1 << len);
            w | v << lo
        }))
    }

    fn r(n: u8) -> Operand {
        Operand::plain(Src::Reg(n))
    }

    /// Fields common to the sampling tests: dst R4, a R6, b R7, dst2 R9,
    /// mask 0b1011, fault P2, eviction LastUse, nodep.
    const IO_FIELDS: [(u32, u32, u128); 8] = [(16, 8, 4), (24, 8, 6), (32, 8, 7), (64, 8, 9), (72, 4, 0b1011), (81, 3, 2), (84, 3, 3), (90, 1, 1)];

    fn io_with(handle: TexHandle) -> TexIo {
        TexIo { dst: 4, dst2: 9, fault: Pred { reg: 2, not: false }, handle, a: r(6), b: r(7), mask: 0b1011, eviction: Eviction::LastUse, nodep: true }
    }

    /// Decodes, checks the result, and checks every set field bit is explained.
    fn check(op: u16, extra: &[(u32, u32, u128)], io: bool, want: TextureOp) {
        let mut f = vec![(0, 12, op as u128)];
        if io {
            f.extend(IO_FIELDS);
        }
        f.extend_from_slice(extra);
        let w = word(&f);
        assert_eq!(decode(w), Some(want), "{op:#05x}");
        assert_eq!(w.0 & !(explained(op) | ALWAYS_EXPLAINED), 0, "{op:#05x}: unexplained {}", ranges(w.0 & !(explained(op) | ALWAYS_EXPLAINED)));
    }

    const CB: [(u32, u32, u128); 2] = [(40, 14, 0x1c), (54, 5, 2)];
    const CB_HANDLE: TexHandle = TexHandle::CBuf { bank: 2, offset: 0x70 };
    const UR: [(u32, u32, u128); 2] = [(40, 6, 6), (91, 1, 1)];
    const UR_HANDLE: TexHandle = TexHandle::UReg { ureg: 6, bit91: true };

    #[test]
    fn tex() {
        // scr, Array2D, aoffi, ndv, dc, .F16, LodMode::Clamp.
        let mods = [(60, 1, 1), (61, 3, 5), (76, 1, 1), (77, 1, 1), (78, 1, 1), (79, 1, 1), (87, 3, 4)];
        let want = |handle| TextureOp::Tex { io: io_with(handle), dim: TexDim::Array2D, lod: LodMode::Clamp, scr: true, aoffi: true, ndv: true, dc: true, f16: true };
        check(0xb60, &[&mods[..], &CB].concat(), true, want(CB_HANDLE));
        check(0xf60, &[&mods[..], &UR].concat(), true, want(UR_HANDLE));
        check(0x361, &[&mods[..], &[(59, 1, 1)]].concat(), true, want(TexHandle::Bindless));
        // The bindless TEX without .B is not decoded.
        assert_eq!(decode(word(&[(0, 12, 0x361)])), None);
        // Defaults: 1D, auto lod, no modifiers, eviction First.
        let w = word(&[(0, 12, 0xb60)]);
        let Some(TextureOp::Tex { io, dim, lod, scr, aoffi, ndv, dc, f16: false }) = decode(w) else { panic!() };
        assert_eq!((io.eviction, dim, lod, scr, aoffi, ndv, dc), (Eviction::First, TexDim::D1, LodMode::Auto, false, false, false, false));
    }

    #[test]
    fn tld() {
        // scr, 3D, aoffi, ms, .F16, LodMode::Lod.
        let mods = [(60, 1, 1), (61, 3, 2), (76, 1, 1), (78, 1, 1), (79, 1, 1), (87, 3, 3)];
        let want = TextureOp::Tld { io: io_with(CB_HANDLE), dim: TexDim::D3, lod: LodMode::Lod, scr: true, aoffi: true, ms: true, f16: true };
        check(0xb66, &[&mods[..], &CB].concat(), true, want);
    }

    #[test]
    fn tld4() {
        // scr, Cube, per-pixel offsets, dc, component B.
        let mods = [(60, 1, 1), (61, 3, 3), (76, 2, 2), (78, 1, 1), (87, 2, 2)];
        let want = |handle| TextureOp::Tld4 { io: io_with(handle), dim: TexDim::Cube, comp: 2, offset: Tld4Offset::PerPixel, scr: true, dc: true };
        check(0xb63, &[&mods[..], &CB].concat(), true, want(CB_HANDLE));
        check(0xf63, &[&mods[..], &UR].concat(), true, want(UR_HANDLE));
    }

    #[test]
    fn txd() {
        // ArrayCube, aoffi.
        let mods = [(61, 3, 7), (76, 1, 1)];
        let want = |handle| TextureOp::Txd { io: io_with(handle), dim: TexDim::ArrayCube, aoffi: true };
        check(0xb6c, &[&mods[..], &CB].concat(), true, want(CB_HANDLE));
        check(0xf6c, &[&mods[..], &UR].concat(), true, want(UR_HANDLE));
    }

    #[test]
    fn txq() {
        let f = [(16, 8, 4), (24, 8, 6), (62, 2, 2), (64, 8, 9), (72, 4, 3), (90, 1, 1)];
        let want = TextureOp::Txq { dst: 4, dst2: 9, handle: CB_HANDLE, src: r(6), query: TexQuery::SamplerPos, mask: 3, nodep: true };
        check(0xb6f, &[&f[..], &CB].concat(), false, want);
    }

    #[test]
    fn not_owned() {
        for op in [0x000, 0x360, 0x367, 0x364, 0x36d, 0x370, 0xb69, 0x36a, 0xf66, 0xf6f, 0xd61] {
            assert_eq!(decode(Word(op)), None, "{op:#05x}");
            assert_eq!(explained(op as u16), 0);
        }
    }
}
