//! Memory family: constant, attribute, global, local and shared memory,
//! atomics, the constant-bound surface operations, and two neighbours that
//! share the opcode space (MATCH, PIXLD).
//!
//! Field positions come from Mesa NAK's SM70+ encoder (sm70_encode.rs, SM 8.x
//! branches), read for facts only; each variant names the NAK function it
//! follows. The four bound-surface opcodes (0xb97, 0xb9b, 0x593, 0x59f) are
//! never emitted by NAK: they are identified from the samples, see their docs.
//!
//! Conventions: registers are plain `u8` (255 = RZ; memory sources never carry
//! neg/abs/swizzle, so `Operand` would only add meaningless bits); destination
//! predicates are the predicate register (7 = PT, result discarded); source
//! predicates are `Pred`.

use super::prelude::*;

pub const OPCODES: &[u16] = &[
    0xb82, 0x326, 0x322, 0x321, 0x984, 0x981, 0x388, 0xb9b, 0x381, 0x986, 0xb97, 0x59f, 0x593, 0x387, 0x983, 0x38c, 0x98e, 0x3a1, 0x3a8, 0x925, 0x386,
];

/// A small enum read from a bit field, with a catch-all for encodings no
/// source documents.
macro_rules! field_enum {
    ($(#[$m:meta])* $name:ident { $($var:ident = $val:literal),* $(,)? }) => {
        $(#[$m])*
        #[derive(Clone, Copy, Debug, PartialEq, Eq)]
        pub enum $name { $($var,)* Other(u8) }
        impl $name {
            fn at(w: Word, lo: u32, len: u32) -> $name {
                match w.field(lo, len) { $($val => $name::$var,)* v => $name::Other(v as u8) }
            }
        }
    };
}

field_enum!(
    /// Access size at 73..76 (NAK `set_mem_type`); S = sign-extended.
    MemType { U8 = 0, S8 = 1, U16 = 2, S16 = 3, B32 = 4, B64 = 5, B128 = 6 }
);
field_enum!(
    /// Memory ordering at 77..81, SM 8.x encoding (NAK `set_mem_order`):
    /// nvdisasm prints .CONSTANT, .STRONG.SM/.GPU/.SYS; weak is unmarked.
    MemOrder { Weak = 0, Constant = 4, StrongCta = 5, StrongGpu = 7, StrongSys = 10 }
);
field_enum!(
    /// Cache eviction priority at 84..87 (NAK `set_eviction_priority`):
    /// .EF, default, .EL, .LU, .EU, .NA.
    Evict { First = 0, Normal = 1, Last = 2, LastUse = 3, Unchanged = 4, NoAllocate = 5 }
);
field_enum!(
    /// Atomic operation at 87..91 (87..90 for the reductions), NAK `set_atom_op`.
    AtomOp { Add = 0, Min = 1, Max = 2, Inc = 3, Dec = 4, And = 5, Or = 6, Xor = 7, Exch = 8 }
);
field_enum!(
    /// Atomic operand type at 73..76, pre-SM90 table (NAK `set_atom_type`).
    AtomType { U32 = 0, S32 = 1, U64 = 2, F32 = 3, F16x2 = 4, S64 = 5, F64 = 6 }
);
field_enum!(
    /// Shared-memory offset stride at 78..80 (NAK `OffsetStride::encode_sm75`):
    /// the address register is scaled by 1, 4, 8 or 16.
    Stride { X1 = 0, X4 = 1, X8 = 2, X16 = 3 }
);
field_enum!(
    /// LDC indexing mode at 78..80 (NAK `OpLdc`).
    LdcMode { Indexed = 0, IndexedLinear = 1, IndexedSegmented = 2, IndexedSegmentedLinear = 3 }
);
field_enum!(
    /// Surface dimensionality at 61..64 (NAK `set_image_dim`).
    ImageDim { D1 = 0, D1Buffer = 1, D1Array = 2, D2 = 3, D2Array = 4, D3 = 5 }
);
field_enum!(
    /// IPA sample location at 76..78 (NAK `OpIpa`).
    InterpLoc { Default = 0, Centroid = 1, Offset = 2 }
);
field_enum!(
    /// IPA interpolation mode at 78..80 (NAK `OpIpa`): perspective pass,
    /// flat (constant), state.
    InterpFreq { Pass = 0, Constant = 1, State = 2 }
);
field_enum!(
    /// PIXLD value at 78..81 (NAK `OpPixLd`).
    PixVal { MsCount = 0, CovMask = 1, CentroidOffset = 2, MyIndex = 3, InnerCoverage = 4 }
);

/// A memory address `[reg + ureg + offset]`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Addr {
    /// GPR at 24..32 (RZ = no register).
    pub reg: u8,
    /// The GPR is a 64-bit pair (`.64`).
    pub reg64: bool,
    /// Uniform register added to the address (only when bit 91 enables it).
    pub ureg: Option<u8>,
    /// The uniform register is a 64-bit pair.
    pub ureg64: bool,
    /// Signed byte offset, 40..64.
    pub offset: i32,
}

/// Global address. NAK `set_reg_addr`/`set_ureg_addr`: in the uniform forms
/// (0x98x, `ureg_lo` given) bit 91 turns the uniform register on; then the
/// GPR size bit is 90 and the uniform size bit 72. With bit 91 clear, and in
/// the 0x38x forms, bit 72 is the GPR's size bit (`.E`), as NAK's pre-Turing
/// encoding and its "the .E bit changes which source it applies to" note say.
fn global_addr(w: Word, ureg_lo: Option<u32>) -> Addr {
    let uniform = ureg_lo.filter(|_| w.bit(91));
    Addr {
        reg: reg(w, 24),
        reg64: if uniform.is_some() { w.bit(90) } else { w.bit(72) },
        ureg: uniform.map(|lo| ureg(w, lo)),
        ureg64: uniform.is_some() && w.bit(72),
        offset: w.signed(40, 24) as i32,
    }
}

/// Shared or local address: 32-bit GPR, optional uniform register (bit 91).
fn window_addr(w: Word, ureg_lo: Option<u32>) -> Addr {
    Addr { reg: reg(w, 24), reg64: false, ureg: ureg_lo.filter(|_| w.bit(91)).map(|lo| ureg(w, lo)), ureg64: false, offset: w.signed(40, 24) as i32 }
}

/// A texture/surface handle stored in a constant buffer: c[bank][offset].
/// Layout of NAK `set_tex_cb_ref` (40..59): word offset 40..54, bank 54..59.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct CbHandle {
    pub bank: u8,
    /// Byte offset (the field holds offset / 4).
    pub offset: u16,
}

fn cb_handle(w: Word) -> CbHandle {
    CbHandle { bank: w.field(54, 5) as u8, offset: (w.field(40, 14) * 4) as u16 }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MemoryOp {
    /// LDC 0xb82, NAK `OpLdc` (bound buffer): dst 16..24, index GPR 24..32,
    /// c[bank 54..59][byte offset 38..54], mode 78..80, size 73..76.
    Ldc { dst: u8, index: u8, bank: u8, offset: u16, mode: LdcMode, ty: MemType },
    /// ALD 0x321, NAK `OpALd`: dst 16..24, offset GPR 24..32, vertex GPR
    /// 32..40, attribute byte address 40..50, components 74..76 (+1), patch
    /// 76, physical 77, output 79.
    Ald { dst: u8, offset: u8, vtx: u8, attr: u16, comps: u8, patch: bool, phys: bool, output: bool },
    /// AST 0x322, NAK `OpASt`: data GPR 32..40, vertex GPR 64..72, offset GPR
    /// 24..32, attribute address 40..50, components 74..76 (+1), patch 76,
    /// physical 77.
    Ast { data: u8, offset: u8, vtx: u8, attr: u16, comps: u8, patch: bool, phys: bool },
    /// IPA 0x326, NAK `OpIpa`: dst 16..24, attribute byte address = 64..72
    /// times 4, location 76..78, mode 78..80, offset GPR 32..40 (used with
    /// `InterpLoc::Offset`), predicate destination 81..84.
    Ipa { dst: u8, attr: u16, loc: InterpLoc, freq: InterpFreq, offset: u8, pred_dst: u8 },
    /// LDG: 0x981 (uniform form, uniform register at 32..38, predicate source
    /// at 64..67 stored as 7 - index, its negation at 67) or 0x381 (no
    /// uniform register and no predicate source: `pred` is PT). NAK `OpLd`
    /// global: dst 16..24, size 73..76, order 77..81, eviction 84..87,
    /// predicate destination 81..84.
    Ldg { dst: u8, addr: Addr, ty: MemType, order: MemOrder, evict: Evict, pred: Pred, pred_dst: u8 },
    /// LDL 0x983, NAK `OpLd` local: dst, uniform register at 32..38, size,
    /// eviction 84..87 (NAK always writes Normal; the game uses .LU too).
    Ldl { dst: u8, addr: Addr, ty: MemType, evict: Evict },
    /// LDS 0x984, NAK `OpLd` shared: dst, uniform register at 32..38, size,
    /// stride 78..80.
    Lds { dst: u8, addr: Addr, ty: MemType, stride: Stride },
    /// STG: 0x986 (uniform register at 64..70) or 0x386, NAK `OpSt` global:
    /// data GPR 32..40, size, order, eviction.
    Stg { data: u8, addr: Addr, ty: MemType, order: MemOrder, evict: Evict },
    /// STL 0x387, NAK `OpSt` local: data 32..40, size, eviction 84..87.
    Stl { data: u8, addr: Addr, ty: MemType, evict: Evict },
    /// STS 0x388, NAK `OpSt` shared: data 32..40, size, stride 78..80.
    Sts { data: u8, addr: Addr, ty: MemType, stride: Stride },
    /// ATOMG 0x3a8, NAK `OpAtom` global with a destination (non-uniform
    /// form): dst, data 32..40, op 87..91, type 73..76, order, eviction,
    /// predicate destination 81..84.
    Atomg { dst: u8, data: u8, addr: Addr, op: AtomOp, ty: AtomType, order: MemOrder, evict: Evict, pred_dst: u8 },
    /// ATOMS 0x38c, NAK `OpAtom` shared (non-uniform form): dst, data 32..40,
    /// op 87..91, type 73..76, stride 78..80.
    Atoms { dst: u8, data: u8, addr: Addr, op: AtomOp, ty: AtomType, stride: Stride },
    /// RED 0x98e, NAK `OpAtom` global without destination: data 32..40,
    /// uniform register at 64..70, op 87..90, type, order, eviction.
    Red { data: u8, addr: Addr, op: AtomOp, ty: AtomType, order: MemOrder, evict: Evict },
    /// MATCH 0x3a1, NAK `OpMatch`: dst (lane mask) 16..24, source 24..32,
    /// 64-bit source 73, .ANY 79 (else .ALL), predicate destination 81..84.
    Match { dst: u8, src: u8, u64: bool, any: bool, pred_dst: u8 },
    /// PIXLD 0x925, NAK `OpPixLd`: dst, value 78..81, predicate destination
    /// 81..84.
    PixLd { dst: u8, val: PixVal, pred_dst: u8 },
    /// SULD.P 0xb97 (guess: constant-bound form of NAK's `OpSuLd` 0x998,
    /// one below it as TEX 0xb60 is below 0x361): dst, coordinates 24..32,
    /// handle c[54..59][40..54 * 4], dim 61..64, channel mask 72..76, order,
    /// eviction, fault predicate 81..84.
    Suld { dst: u8, coord: u8, handle: CbHandle, dim: ImageDim, mask: u8, order: MemOrder, evict: Evict, pred_dst: u8 },
    /// SUST.P 0xb9b (guess: bound form of `OpSuSt` 0x99c): coordinates
    /// 24..32, data 32..40, handle, dim, channel mask 72..76, order, eviction.
    Sust { coord: u8, data: u8, handle: CbHandle, dim: ImageDim, mask: u8, order: MemOrder, evict: Evict },
    /// SUATOM 0x593 (guess: bound form of `OpSuAtom` 0x394): dst, coordinates,
    /// data 32..40, handle, dim, op 87..91, type 73..76, .BA 72, order,
    /// eviction, fault predicate 81..84.
    Suatom { dst: u8, coord: u8, data: u8, handle: CbHandle, dim: ImageDim, op: AtomOp, ty: AtomType, ba: bool, order: MemOrder, evict: Evict, pred_dst: u8 },
    /// SURED 0x59f (guess: bound form of `OpSuAtom` without destination,
    /// 0x3a0): coordinates, data, handle, dim, op 87..90, type, .BA, order,
    /// eviction.
    Sured { coord: u8, data: u8, handle: CbHandle, dim: ImageDim, op: AtomOp, ty: AtomType, ba: bool, order: MemOrder, evict: Evict },
}

pub fn decode(w: Word) -> Option<MemoryOp> {
    use MemoryOp::*;
    let dst = dst(w);
    let ty = MemType::at(w, 73, 3);
    let order = MemOrder::at(w, 77, 4);
    let evict = Evict::at(w, 84, 3);
    let stride = Stride::at(w, 78, 2);
    let pred_dst = w.field(81, 3) as u8;
    let data = reg(w, 32);
    let atom_ty = AtomType::at(w, 73, 3);
    let (coord, handle, dim) = (reg(w, 24), cb_handle(w), ImageDim::at(w, 61, 3));
    Some(match w.opcode() {
        0xb82 => Ldc { dst, index: reg(w, 24), bank: w.field(54, 5) as u8, offset: w.field(38, 16) as u16, mode: LdcMode::at(w, 78, 2), ty },
        0x321 => Ald { dst, offset: reg(w, 24), vtx: reg(w, 32), attr: w.field(40, 10) as u16, comps: w.field(74, 2) as u8 + 1, patch: w.bit(76), phys: w.bit(77), output: w.bit(79) },
        0x322 => Ast { data, offset: reg(w, 24), vtx: reg(w, 64), attr: w.field(40, 10) as u16, comps: w.field(74, 2) as u8 + 1, patch: w.bit(76), phys: w.bit(77) },
        0x326 => Ipa { dst, attr: w.field(64, 8) as u16 * 4, loc: InterpLoc::at(w, 76, 2), freq: InterpFreq::at(w, 78, 2), offset: data, pred_dst },
        0x981 => Ldg { dst, addr: global_addr(w, Some(32)), ty, order, evict, pred: Pred { reg: 7 - w.field(64, 3) as u8, not: w.bit(67) }, pred_dst },
        0x381 => Ldg { dst, addr: global_addr(w, None), ty, order, evict, pred: Pred::TRUE, pred_dst },
        0x983 => Ldl { dst, addr: window_addr(w, Some(32)), ty, evict },
        0x984 => Lds { dst, addr: window_addr(w, Some(32)), ty, stride },
        0x986 => Stg { data, addr: global_addr(w, Some(64)), ty, order, evict },
        0x386 => Stg { data, addr: global_addr(w, None), ty, order, evict },
        0x387 => Stl { data, addr: window_addr(w, None), ty, evict },
        0x388 => Sts { data, addr: window_addr(w, None), ty, stride },
        0x3a8 => Atomg { dst, data, addr: global_addr(w, None), op: AtomOp::at(w, 87, 4), ty: atom_ty, order, evict, pred_dst },
        0x38c => Atoms { dst, data, addr: window_addr(w, None), op: AtomOp::at(w, 87, 4), ty: atom_ty, stride },
        0x98e => Red { data, addr: global_addr(w, Some(64)), op: AtomOp::at(w, 87, 3), ty: atom_ty, order, evict },
        0x3a1 => Match { dst, src: reg(w, 24), u64: w.bit(73), any: w.bit(79), pred_dst },
        0x925 => PixLd { dst, val: PixVal::at(w, 78, 3), pred_dst },
        0xb97 => Suld { dst, coord, handle, dim, mask: w.field(72, 4) as u8, order, evict, pred_dst },
        0xb9b => Sust { coord, data, handle, dim, mask: w.field(72, 4) as u8, order, evict },
        0x593 => Suatom { dst, coord, data, handle, dim, op: AtomOp::at(w, 87, 4), ty: atom_ty, ba: w.bit(72), order, evict, pred_dst },
        0x59f => Sured { coord, data, handle, dim, op: AtomOp::at(w, 87, 3), ty: atom_ty, ba: w.bit(72), order, evict },
        _ => return None,
    })
}

pub fn explained(opcode: u16) -> u128 {
    let (dst, a, b, c) = (bits(16, 8), bits(24, 8), bits(32, 8), bits(64, 8));
    let (off24, ty, order, pdst, evict) = (bits(40, 24), bits(73, 3), bits(77, 4), bits(81, 3), bits(84, 3));
    let surface = a | bits(40, 19) | bits(61, 3) | order | evict;
    match opcode {
        0xb82 => dst | a | bits(38, 21) | ty | bits(78, 2),
        0x321 => dst | a | b | bits(40, 10) | bits(74, 4) | bits(79, 1),
        0x322 => a | b | c | bits(40, 10) | bits(74, 4),
        0x326 => dst | b | c | bits(76, 4) | pdst,
        0x981 => dst | a | bits(32, 6) | off24 | bits(64, 4) | bits(72, 4) | order | pdst | evict | bits(90, 2),
        0x381 => dst | a | off24 | bits(72, 4) | order | pdst | evict,
        0x983 => dst | a | bits(32, 6) | off24 | ty | evict | bits(91, 1),
        0x984 => dst | a | bits(32, 6) | off24 | ty | bits(78, 2) | bits(91, 1),
        0x986 => a | b | off24 | bits(64, 6) | bits(72, 4) | order | evict | bits(90, 2),
        0x98e => a | b | off24 | bits(64, 6) | bits(72, 4) | order | evict | bits(87, 5),
        0x386 => a | b | off24 | bits(72, 4) | order | evict,
        0x387 => a | b | off24 | ty | evict,
        0x388 => a | b | off24 | ty | bits(78, 2),
        0x3a8 => dst | a | b | off24 | bits(72, 4) | order | pdst | evict | bits(87, 4),
        0x38c => dst | a | b | off24 | ty | bits(78, 2) | bits(87, 4),
        0x3a1 => dst | a | bits(73, 1) | bits(79, 1) | pdst,
        0x925 => dst | bits(78, 3) | pdst,
        0xb97 => dst | surface | bits(72, 4) | pdst,
        0xb9b => surface | b | bits(72, 4),
        0x593 => dst | surface | b | bits(72, 4) | pdst | bits(87, 4),
        0x59f => surface | b | bits(72, 4) | bits(87, 3),
        _ => 0,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use MemoryOp::*;

    /// A synthetic word: opcode, guard PT, then (lo, len, value) fields.
    fn word(op: u16, fields: &[(u32, u32, u64)]) -> Word {
        let mut w = op as u128 | 7 << 12;
        for &(lo, len, v) in fields {
            assert!(len == 64 || v < 1 << len, "field {lo}..{} overflows", lo + len);
            w |= (v as u128) << lo;
        }
        Word(w)
    }

    fn dec(op: u16, fields: &[(u32, u32, u64)]) -> MemoryOp {
        decode(word(op, fields)).unwrap()
    }

    #[test]
    fn foreign_opcodes() {
        assert_eq!(decode(Word(0x7b60)), None);
        assert_eq!(decode(Word(0x7998)), None);
        assert_eq!(decode(Word(0x7582)), None);
        for &op in OPCODES {
            assert!(decode(Word(op as u128)).is_some());
            assert_ne!(explained(op), 0);
        }
    }

    #[test]
    fn ldc() {
        let op = dec(0xb82, &[(16, 8, 4), (24, 8, 6), (38, 16, 0x248), (54, 5, 17), (73, 3, 5), (78, 2, 2)]);
        assert_eq!(op, Ldc { dst: 4, index: 6, bank: 17, offset: 0x248, mode: LdcMode::IndexedSegmented, ty: MemType::B64 });
    }

    #[test]
    fn attributes() {
        let op = dec(0x321, &[(16, 8, 3), (24, 8, 5), (32, 8, 9), (40, 10, 0x2fc), (74, 2, 3), (76, 1, 1), (77, 1, 1), (79, 1, 1)]);
        assert_eq!(op, Ald { dst: 3, offset: 5, vtx: 9, attr: 0x2fc, comps: 4, patch: true, phys: true, output: true });
        let op = dec(0x321, &[(16, 8, 3), (24, 8, 255), (32, 8, 255), (40, 10, 0x80)]);
        assert_eq!(op, Ald { dst: 3, offset: 255, vtx: 255, attr: 0x80, comps: 1, patch: false, phys: false, output: false });
        let op = dec(0x322, &[(24, 8, 7), (32, 8, 8), (64, 8, 10), (40, 10, 0x70), (74, 2, 1), (76, 1, 1)]);
        assert_eq!(op, Ast { data: 8, offset: 7, vtx: 10, attr: 0x70, comps: 2, patch: true, phys: false });
        let op = dec(0x326, &[(16, 8, 2), (64, 8, 0x1f), (76, 2, 2), (78, 2, 1), (32, 8, 6), (81, 3, 3)]);
        assert_eq!(op, Ipa { dst: 2, attr: 0x7c, loc: InterpLoc::Offset, freq: InterpFreq::Constant, offset: 6, pred_dst: 3 });
    }

    #[test]
    fn global_loads_stores() {
        // 0x981, uniform form: [R4 + UR8.64 - 0x10], predicate !P2 (stored as 5).
        let op = dec(0x981, &[(16, 8, 1), (24, 8, 4), (32, 6, 8), (40, 24, 0xfffff0), (64, 3, 5), (67, 1, 1), (72, 1, 1), (73, 3, 6), (77, 4, 4), (81, 3, 7), (84, 3, 3), (91, 1, 1)]);
        let addr = Addr { reg: 4, reg64: false, ureg: Some(8), ureg64: true, offset: -0x10 };
        assert_eq!(op, Ldg { dst: 1, addr, ty: MemType::B128, order: MemOrder::Constant, evict: Evict::LastUse, pred: Pred { reg: 2, not: true }, pred_dst: 7 });
        // Bit 90 is the GPR size when the uniform register is on.
        let Ldg { addr, .. } = dec(0x981, &[(90, 1, 1), (91, 1, 1)]) else { panic!() };
        assert!(addr.reg64 && !addr.ureg64 && addr.ureg == Some(0));
        // 0x381: no uniform register, bit 72 = 64-bit GPR, no predicate.
        let op = dec(0x381, &[(16, 8, 7), (24, 8, 0x2e), (40, 24, 0x50), (72, 1, 1), (73, 3, 5), (77, 4, 5), (81, 3, 7), (84, 3, 1), (64, 3, 3)]);
        let addr = Addr { reg: 0x2e, reg64: true, ureg: None, ureg64: false, offset: 0x50 };
        assert_eq!(op, Ldg { dst: 7, addr, ty: MemType::B64, order: MemOrder::StrongCta, evict: Evict::Normal, pred: Pred::TRUE, pred_dst: 7 });
        // 0x986: [R9 + UR4.64 + 0x40] <- R12.
        let op = dec(0x986, &[(24, 8, 9), (32, 8, 12), (40, 24, 0x40), (64, 6, 4), (72, 1, 1), (73, 3, 4), (77, 4, 7), (84, 3, 0), (91, 1, 1)]);
        let addr = Addr { reg: 9, reg64: false, ureg: Some(4), ureg64: true, offset: 0x40 };
        assert_eq!(op, Stg { data: 12, addr, ty: MemType::B32, order: MemOrder::StrongGpu, evict: Evict::First });
        let op = dec(0x386, &[(24, 8, 2), (32, 8, 4), (72, 1, 1), (73, 3, 6), (77, 4, 10), (84, 3, 5), (64, 6, 9)]);
        let addr = Addr { reg: 2, reg64: true, ureg: None, ureg64: false, offset: 0 };
        assert_eq!(op, Stg { data: 4, addr, ty: MemType::B128, order: MemOrder::StrongSys, evict: Evict::NoAllocate });
    }

    #[test]
    fn local_shared() {
        let op = dec(0x983, &[(16, 8, 0x1e), (24, 8, 255), (32, 6, 3), (40, 24, 0xc), (73, 3, 4), (84, 3, 3), (91, 1, 1)]);
        let addr = Addr { reg: 255, reg64: false, ureg: Some(3), ureg64: false, offset: 0xc };
        assert_eq!(op, Ldl { dst: 0x1e, addr, ty: MemType::B32, evict: Evict::LastUse });
        let op = dec(0x984, &[(16, 8, 1), (24, 8, 3), (32, 6, 5), (40, 24, 0x400), (73, 3, 3), (78, 2, 3)]);
        let addr = Addr { reg: 3, reg64: false, ureg: None, ureg64: false, offset: 0x400 };
        assert_eq!(op, Lds { dst: 1, addr, ty: MemType::S16, stride: Stride::X16 });
        let op = dec(0x387, &[(24, 8, 255), (32, 8, 0x3d), (40, 24, 8), (73, 3, 4), (84, 3, 1)]);
        let addr = Addr { reg: 255, reg64: false, ureg: None, ureg64: false, offset: 8 };
        assert_eq!(op, Stl { data: 0x3d, addr, ty: MemType::B32, evict: Evict::Normal });
        let op = dec(0x388, &[(24, 8, 4), (32, 8, 5), (40, 24, 0xffffff), (73, 3, 5), (78, 2, 1)]);
        let addr = Addr { reg: 4, reg64: false, ureg: None, ureg64: false, offset: -1 };
        assert_eq!(op, Sts { data: 5, addr, ty: MemType::B64, stride: Stride::X4 });
    }

    #[test]
    fn atomics() {
        let op = dec(0x3a8, &[(16, 8, 5), (24, 8, 2), (32, 8, 9), (40, 24, 4), (72, 1, 1), (73, 3, 1), (77, 4, 7), (81, 3, 2), (84, 3, 1), (87, 4, 8)]);
        let addr = Addr { reg: 2, reg64: true, ureg: None, ureg64: false, offset: 4 };
        assert_eq!(op, Atomg { dst: 5, data: 9, addr, op: AtomOp::Exch, ty: AtomType::S32, order: MemOrder::StrongGpu, evict: Evict::Normal, pred_dst: 2 });
        let op = dec(0x38c, &[(16, 8, 2), (24, 8, 0x13), (32, 8, 6), (40, 24, 0x40), (73, 3, 2), (78, 2, 2), (87, 4, 7)]);
        let addr = Addr { reg: 0x13, reg64: false, ureg: None, ureg64: false, offset: 0x40 };
        assert_eq!(op, Atoms { dst: 2, data: 6, addr, op: AtomOp::Xor, ty: AtomType::U64, stride: Stride::X8 });
        // RED with the uniform register on: [RZ.64 + UR4.64], MAX.
        let op = dec(0x98e, &[(24, 8, 255), (32, 8, 1), (64, 6, 4), (72, 1, 1), (73, 3, 3), (77, 4, 7), (84, 3, 1), (87, 3, 2), (90, 1, 1), (91, 1, 1)]);
        let addr = Addr { reg: 255, reg64: true, ureg: Some(4), ureg64: true, offset: 0 };
        assert_eq!(op, Red { data: 1, addr, op: AtomOp::Max, ty: AtomType::F32, order: MemOrder::StrongGpu, evict: Evict::Normal });
        // RED with bit 91 clear: bit 72 is the GPR size, no uniform register.
        let Red { addr, op, .. } = dec(0x98e, &[(24, 8, 2), (72, 1, 1), (87, 3, 6)]) else { panic!() };
        assert_eq!((addr.reg64, addr.ureg, op), (true, None, AtomOp::Or));
    }

    #[test]
    fn match_pixld() {
        let op = dec(0x3a1, &[(16, 8, 255), (24, 8, 0x13), (73, 1, 1), (81, 3, 1)]);
        assert_eq!(op, Match { dst: 255, src: 0x13, u64: true, any: false, pred_dst: 1 });
        let op = dec(0x3a1, &[(16, 8, 4), (24, 8, 2), (79, 1, 1), (81, 3, 7)]);
        assert_eq!(op, Match { dst: 4, src: 2, u64: false, any: true, pred_dst: 7 });
        let op = dec(0x925, &[(16, 8, 1), (78, 3, 4), (81, 3, 7)]);
        assert_eq!(op, PixLd { dst: 1, val: PixVal::InnerCoverage, pred_dst: 7 });
        assert_eq!(dec(0x925, &[(78, 3, 6)]), PixLd { dst: 0, val: PixVal::Other(6), pred_dst: 0 });
    }

    #[test]
    fn surfaces() {
        let handle = CbHandle { bank: 3, offset: 0x30 };
        let common = [(24, 8, 10), (40, 14, 0xc), (54, 5, 3), (61, 3, 4), (84, 3, 1)];
        let op = dec(0xb97, &[&common[..], &[(16, 8, 7), (72, 4, 1), (77, 4, 0), (81, 3, 7)]].concat());
        assert_eq!(op, Suld { dst: 7, coord: 10, handle, dim: ImageDim::D2Array, mask: 1, order: MemOrder::Weak, evict: Evict::Normal, pred_dst: 7 });
        let op = dec(0xb9b, &[&common[..], &[(32, 8, 0x30), (72, 4, 0xf), (77, 4, 5)]].concat());
        assert_eq!(op, Sust { coord: 10, data: 0x30, handle, dim: ImageDim::D2Array, mask: 0xf, order: MemOrder::StrongCta, evict: Evict::Normal });
        let op = dec(0x593, &[&common[..], &[(16, 8, 8), (32, 8, 7), (72, 1, 1), (73, 3, 1), (77, 4, 7), (81, 3, 7), (87, 4, 8)]].concat());
        assert_eq!(op, Suatom { dst: 8, coord: 10, data: 7, handle, dim: ImageDim::D2Array, op: AtomOp::Exch, ty: AtomType::S32, ba: true, order: MemOrder::StrongGpu, evict: Evict::Normal, pred_dst: 7 });
        let op = dec(0x59f, &[&common[..], &[(32, 8, 3), (73, 3, 0), (77, 4, 7), (87, 3, 6)]].concat());
        assert_eq!(op, Sured { coord: 10, data: 3, handle, dim: ImageDim::D2Array, op: AtomOp::Or, ty: AtomType::U32, ba: false, order: MemOrder::StrongGpu, evict: Evict::Normal });
    }
}
