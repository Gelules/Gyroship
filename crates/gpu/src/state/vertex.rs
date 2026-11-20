//! Vertex input and index buffer state of the Ampere 3D class (0xc797).
//!
//! Facts come from NVIDIA's clc797.h (method offsets, fields, enums) and
//! from how Mesa's NVK emits Vulkan state as these methods
//! (nvk_cmd_draw.c, nvk_format.c).
//!
//! The readers take any register reader `r` (method -> value), not a `Reg`:
//! a draw's snapshot calls them with its own closure, inlined (no call through
//! a vtable for each of its ~190 register reads).

/// Vertex streams and vertex attributes: 32 of each. clc797.h names each
/// array in two halves (`..._A(j)` for 0..15, `..._B(j)` for 16..31) at
/// contiguous offsets, so element n always sits at the `_A` base plus
/// n * pitch (NVK indexes SET_VERTEX_STREAM_A_FORMAT(b),
/// SET_VERTEX_STREAM_INSTANCE_A(b), SET_VERTEX_STREAM_SIZE_A(b) and
/// SET_VERTEX_ATTRIBUTE_A(a) with b, a up to 31).
const COUNT: u32 = 32;

fn field(v: u32, hi: u32, lo: u32) -> u32 {
    (v >> lo) & (u32::MAX >> (31 - (hi - lo)))
}

/// 40-bit GPU address from an `_UPPER` (7:0) / `_LOWER` (31:0) method pair.
fn addr40<R: Fn(u32) -> u32 + ?Sized>(r: &R, upper: u32) -> u64 {
    (u64::from(field(r(upper), 7, 0)) << 32) | u64::from(r(upper + 4))
}

/// An enabled vertex stream (vertex buffer binding).
#[derive(Clone, Debug, PartialEq)]
pub struct Stream {
    /// j of SET_VERTEX_STREAM_A_FORMAT(j), 0..31.
    pub index: u8,
    /// SET_VERTEX_STREAM_A_LOCATION_A_OFFSET_UPPER (7:0) and
    /// SET_VERTEX_STREAM_A_LOCATION_B_OFFSET_LOWER (31:0).
    pub address: u64,
    /// SET_VERTEX_STREAM_A_FORMAT_STRIDE (11:0), bytes.
    pub stride: u32,
    /// Last byte the stream may read: address + size - 1, the size being
    /// SET_VERTEX_STREAM_SIZE_A_UPPER (7:0) / SET_VERTEX_STREAM_SIZE_B_LOWER
    /// (31:0) in bytes (NVK writes the binding's byte size there on Turing
    /// and later; older classes had an inclusive LIMIT address instead).
    /// Size 0 gives address - 1 (empty, wrapping).
    pub end: u64,
    /// SET_VERTEX_STREAM_INSTANCE_A_IS_INSTANCED (0:0): per instance.
    pub per_instance: bool,
    /// SET_VERTEX_STREAM_A_FREQUENCY_V (31:0), raw. NVK writes the Vulkan
    /// instance divisor here unchanged, so 1 = advance every instance.
    pub divisor: u32,
}

/// Numerical type, SET_VERTEX_ATTRIBUTE_A_NUMERICAL_TYPE (29:27).
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum AttrKind {
    Snorm,   // NUM_SNORM 1
    Unorm,   // NUM_UNORM 2
    Sint,    // NUM_SINT 3
    Uint,    // NUM_UINT 4
    Uscaled, // NUM_USCALED 5
    Sscaled, // NUM_SSCALED 6
    Float,   // NUM_FLOAT 7
    /// 0 = UNUSED_ENUM_DO_NOT_USE_BECAUSE_IT_WILL_GO_AWAY.
    Unknown(u8),
}

/// Vertex attribute format.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct AttrFormat {
    /// Component count from SET_VERTEX_ATTRIBUTE_A_COMPONENT_BIT_WIDTHS
    /// (26:21); 0 when the enum is not one NVK uses (A8B8G8R8 0x2f,
    /// G8R8 0x32, X8B8G8R8 0x33, A8 0x34) or not in clc797.h: then see
    /// `Attribute::raw`.
    pub components: u8,
    /// Bits per component in shader order x, y, z, w (0 = absent).
    /// Packed enums are named most significant field first:
    /// A2B10G10R10 = [10, 10, 10, 2] with R in bits 9:0, B10G11R11 =
    /// [11, 11, 10, 0] with R in bits 10:0 (NVK maps
    /// VK_FORMAT_A2B10G10R10_*_PACK32 and B10G11R11_UFLOAT_PACK32 to them).
    pub bits: [u8; 4],
    /// SET_VERTEX_ATTRIBUTE_A_NUMERICAL_TYPE (29:27).
    pub kind: AttrKind,
    /// SET_VERTEX_ATTRIBUTE_A_SWAP_R_AND_B (31:31): memory holds B, G, R, A
    /// (NVK sets it for VK_FORMAT_B8G8R8A8_* and A2R10G10B10_*).
    pub bgra: bool,
}

/// A programmed vertex attribute. Attribute i of SET_VERTEX_ATTRIBUTE_A(i)
/// is the shader's generic input vector i, read by the vertex shader at
/// a[0x80 + 16 * i] (x, y, z, w at +0, +4, +8, +12): NVK programs Vulkan
/// location i into SET_VERTEX_ATTRIBUTE_A(i) and NAK addresses generic
/// attribute i at NAK_ATTR_GENERIC_START (0x80) + 0x10 * i.
#[derive(Clone, Debug, PartialEq)]
pub struct Attribute {
    /// i of SET_VERTEX_ATTRIBUTE_A(i), 0..31.
    pub location: u8,
    /// SET_VERTEX_ATTRIBUTE_A_STREAM (4:0).
    pub stream: u8,
    /// SET_VERTEX_ATTRIBUTE_A_OFFSET (20:7), bytes into the stream's element.
    pub offset: u32,
    /// SET_VERTEX_ATTRIBUTE_A_SOURCE (6:6) == SOURCE_INACTIVE: nothing is
    /// fetched, the shader sees a default value. NVK's comment says an
    /// inactive R32_G32_B32_A32 attribute reads (0, 0, 0, 0); the value for
    /// other widths is not documented (presumably missing components read
    /// as (0, 0, 0, 1), unverified).
    pub constant: bool,
    pub format: AttrFormat,
    /// The whole SET_VERTEX_ATTRIBUTE_A(i) register, for unknown enums.
    pub raw: u32,
}

/// Enabled streams (SET_VERTEX_STREAM_A_FORMAT_ENABLE) and programmed
/// attributes: those whose COMPONENT_BIT_WIDTHS is not 0 (0 is no enum; a
/// never-written register reads 0). Constant attributes are kept so the
/// shader translator can substitute their value; which of them the shader
/// actually reads is in its header, not in this state.
pub fn vertex_input<R: Fn(u32) -> u32 + ?Sized>(r: &R) -> (Vec<Stream>, Vec<Attribute>) {
    // Plain loops into lists sized once (a filtered chain grew them 4, 8, 16, 32).
    let mut streams = Vec::with_capacity(COUNT as usize);
    for j in 0..COUNT {
        let format = r(0x1c00 + 16 * j);
        if field(format, 12, 12) != 1 {
            continue; // SET_VERTEX_STREAM_A_FORMAT_ENABLE
        }
        let address = addr40(r, 0x1c04 + 16 * j); // SET_VERTEX_STREAM_A_LOCATION_A/B
        let size = addr40(r, 0x600 + 8 * j); // SET_VERTEX_STREAM_SIZE_A/B
        streams.push(Stream {
            index: j as u8,
            address,
            stride: field(format, 11, 0),
            end: address.wrapping_add(size).wrapping_sub(1),
            per_instance: field(r(0x1880 + 4 * j), 0, 0) == 1,
            divisor: r(0x1c0c + 16 * j),
        });
    }
    let mut attributes = Vec::with_capacity(COUNT as usize);
    for i in 0..COUNT {
        let v = r(0x1160 + 4 * i);
        if field(v, 26, 21) == 0 {
            continue;
        }
        attributes.push(Attribute {
            location: i as u8,
            stream: field(v, 4, 0) as u8,
            offset: field(v, 20, 7),
            constant: field(v, 6, 6) == 1,
            format: attr_format(v),
            raw: v,
        });
    }
    (streams, attributes)
}

fn attr_format(v: u32) -> AttrFormat {
    let (components, bits) = match field(v, 26, 21) {
        0x01 => (4, [32, 32, 32, 32]), // R32_G32_B32_A32
        0x02 => (3, [32, 32, 32, 0]),  // R32_G32_B32
        0x03 => (4, [16, 16, 16, 16]), // R16_G16_B16_A16
        0x04 => (2, [32, 32, 0, 0]),   // R32_G32
        0x05 => (3, [16, 16, 16, 0]),  // R16_G16_B16
        0x0a => (4, [8, 8, 8, 8]),     // R8_G8_B8_A8
        0x0f => (2, [16, 16, 0, 0]),   // R16_G16
        0x12 => (1, [32, 0, 0, 0]),    // R32
        0x13 => (3, [8, 8, 8, 0]),     // R8_G8_B8
        0x18 => (2, [8, 8, 0, 0]),     // R8_G8
        0x1b => (1, [16, 0, 0, 0]),    // R16
        0x1d => (1, [8, 0, 0, 0]),     // R8
        0x30 => (4, [10, 10, 10, 2]),  // A2B10G10R10
        0x31 => (3, [11, 11, 10, 0]),  // B10G11R11
        // ponytail: A8B8G8R8, G8R8, X8B8G8R8, A8 left unknown (NVK never
        // emits them); decode them once a trace shows one.
        _ => (0, [0; 4]),
    };
    let kind = match field(v, 29, 27) {
        1 => AttrKind::Snorm,
        2 => AttrKind::Unorm,
        3 => AttrKind::Sint,
        4 => AttrKind::Uint,
        5 => AttrKind::Uscaled,
        6 => AttrKind::Sscaled,
        7 => AttrKind::Float,
        n => AttrKind::Unknown(n as u8),
    };
    AttrFormat { components, bits, kind, bgra: field(v, 31, 31) == 1 }
}

/// The wgpu vertex format that reads `f` exactly, or None. Conversions
/// needed when None:
/// - 3-component 8/16-bit: fetch as the x4 format only if one more element
///   is always in bounds, else Uint32 words and unpack in the shader;
/// - Uscaled/Sscaled: fetch as Uint/Sint of the same widths and convert to
///   float in the shader;
/// - B10G11R11 float, non-Unorm 10/10/10/2, bgra other than Unorm8x4:
///   fetch as Uint32 and unpack (or swizzle) in the shader;
/// - Unorm/Snorm 32-bit: fetch as Uint32/Sint32 and normalize in the shader;
/// - unknown widths or numerical type: see `Attribute::raw`.
pub fn wgpu_vertex_format(f: &AttrFormat) -> Option<wgpu::VertexFormat> {
    use AttrKind::*;
    use wgpu::VertexFormat as V;
    let n = f.components as usize;
    match (f.kind, f.bits, f.bgra) {
        (Unorm, [8, 8, 8, 8], true) => return Some(V::Unorm8x4Bgra),
        (_, _, true) => return None,
        (Unorm, [10, 10, 10, 2], _) => return Some(V::Unorm10_10_10_2),
        _ => {}
    }
    let w = f.bits[0];
    if n == 0 || f.bits[..n].iter().any(|&b| b != w) {
        return None;
    }
    Some(match (f.kind, w, n) {
        (Uint, 8, 1) => V::Uint8,
        (Uint, 8, 2) => V::Uint8x2,
        (Uint, 8, 4) => V::Uint8x4,
        (Sint, 8, 1) => V::Sint8,
        (Sint, 8, 2) => V::Sint8x2,
        (Sint, 8, 4) => V::Sint8x4,
        (Unorm, 8, 1) => V::Unorm8,
        (Unorm, 8, 2) => V::Unorm8x2,
        (Unorm, 8, 4) => V::Unorm8x4,
        (Snorm, 8, 1) => V::Snorm8,
        (Snorm, 8, 2) => V::Snorm8x2,
        (Snorm, 8, 4) => V::Snorm8x4,
        (Uint, 16, 1) => V::Uint16,
        (Uint, 16, 2) => V::Uint16x2,
        (Uint, 16, 4) => V::Uint16x4,
        (Sint, 16, 1) => V::Sint16,
        (Sint, 16, 2) => V::Sint16x2,
        (Sint, 16, 4) => V::Sint16x4,
        (Unorm, 16, 1) => V::Unorm16,
        (Unorm, 16, 2) => V::Unorm16x2,
        (Unorm, 16, 4) => V::Unorm16x4,
        (Snorm, 16, 1) => V::Snorm16,
        (Snorm, 16, 2) => V::Snorm16x2,
        (Snorm, 16, 4) => V::Snorm16x4,
        (Float, 16, 1) => V::Float16,
        (Float, 16, 2) => V::Float16x2,
        (Float, 16, 4) => V::Float16x4,
        (Float, 32, 1) => V::Float32,
        (Float, 32, 2) => V::Float32x2,
        (Float, 32, 3) => V::Float32x3,
        (Float, 32, 4) => V::Float32x4,
        (Uint, 32, 1) => V::Uint32,
        (Uint, 32, 2) => V::Uint32x2,
        (Uint, 32, 3) => V::Uint32x3,
        (Uint, 32, 4) => V::Uint32x4,
        (Sint, 32, 1) => V::Sint32,
        (Sint, 32, 2) => V::Sint32x2,
        (Sint, 32, 3) => V::Sint32x3,
        (Sint, 32, 4) => V::Sint32x4,
        _ => return None,
    })
}

/// SET_GLOBAL_BASE_VERTEX_INDEX: added to every index before the fetch (NVK
/// writes the Vulkan vertexOffset there).
pub fn base_vertex<R: Fn(u32) -> u32 + ?Sized>(r: &R) -> i32 {
    r(0x1434) as i32
}

/// SET_GLOBAL_BASE_INSTANCE_INDEX (NVK: the Vulkan firstInstance).
pub fn base_instance<R: Fn(u32) -> u32 + ?Sized>(r: &R) -> u32 {
    r(0x1438)
}

/// The bound index buffer and primitive restart.
#[derive(Clone, Debug, PartialEq)]
pub struct IndexBuffer {
    /// SET_INDEX_BUFFER_A_ADDRESS_UPPER (7:0), SET_INDEX_BUFFER_B_ADDRESS_LOWER (31:0).
    pub address: u64,
    /// SET_INDEX_BUFFER_E_INDEX_SIZE (1:0): ONE_BYTE 0 -> 1, TWO_BYTES 1 -> 2,
    /// FOUR_BYTES 2 -> 4; 0 for the undefined value 3.
    pub bytes_per_index: u8,
    /// SET_INDEX_BUFFER_F_FIRST (31:0): first index, in indices (NVK writes
    /// the Vulkan firstIndex here), so reading starts at
    /// address + first * bytes_per_index.
    pub first: u32,
    /// Last byte that may be read: address + size - 1, the size being
    /// SET_INDEX_BUFFER_SIZE_A_UPPER (7:0) / SET_INDEX_BUFFER_SIZE_B_LOWER
    /// (31:0) in bytes (clc797.h has no SET_INDEX_BUFFER_C/D limit).
    pub end: u64,
    /// SET_DA_PRIMITIVE_RESTART_ENABLE (0:0).
    pub restart_enable: bool,
    /// SET_DA_PRIMITIVE_RESTART_INDEX_V (31:0).
    pub restart_index: u32,
}

pub fn index_buffer<R: Fn(u32) -> u32 + ?Sized>(r: &R) -> IndexBuffer {
    let address = addr40(r, 0x17c8);
    IndexBuffer {
        address,
        bytes_per_index: [1, 2, 4, 0][field(r(0x17d8), 1, 0) as usize],
        first: r(0x17dc),
        end: address.wrapping_add(addr40(r, 0x238)).wrapping_sub(1),
        restart_enable: field(r(0x1644), 0, 0) == 1,
        restart_index: r(0x1648),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::state::Reg;
    use std::collections::HashMap;

    fn with<T>(m: &[(u32, u32)], f: impl Fn(Reg) -> T) -> T {
        let m: HashMap<u32, u32> = m.iter().copied().collect();
        let r = |a: u32| m.get(&a).copied().unwrap_or(0);
        f(&r)
    }

    #[test]
    fn streams() {
        let (s, a) = with(
            &[
                (0x1c00, 0x0000_0010),               // stream 0: stride 16, disabled
                (0x1c00 + 16 * 5, 0x1000 | 0xfff),   // stream 5 enabled, stride 4095
                (0x1c04 + 16 * 5, 0x1ab),            // upper 7:0 only
                (0x1c08 + 16 * 5, 0x8765_4320),
                (0x600 + 8 * 5, 0x1),
                (0x604 + 8 * 5, 0x10),
                (0x1880 + 4 * 5, 1),
                (0x1c0c + 16 * 5, 3),
                (0x1d00 + 16, 0x1000 | 20),      // SET_VERTEX_STREAM_B_FORMAT(1) = stream 17
                (0x1d08 + 16, 0x1000),           // its size stays 0
            ],
            |r| vertex_input(r),
        );
        assert!(a.is_empty());
        assert_eq!(
            s,
            vec![
                Stream { index: 5, address: 0xab_8765_4320, stride: 4095, end: 0xab_8765_4320 + 0x1_0000_0010 - 1, per_instance: true, divisor: 3 },
                Stream { index: 17, address: 0x1000, stride: 20, end: 0xfff, per_instance: false, divisor: 0 },
            ]
        );
    }

    #[test]
    fn attributes() {
        let (_, a) = with(
            &[
                // 0: stream 5, offset 12, R32_G32_B32 float.
                (0x1160, 5 | 12 << 7 | 0x02 << 21 | 7 << 27),
                // 3: stream 31, offset 0x3fff, R8_G8_B8_A8 unorm, swap R and B.
                (0x1160 + 4 * 3, 31 | 0x3fff << 7 | 0x0a << 21 | 2 << 27 | 1 << 31),
                // 20 (SET_VERTEX_ATTRIBUTE_B(4)): inactive R32 float.
                (0x11a0 + 4 * 4, 1 << 6 | 0x12 << 21 | 7 << 27),
                // 21: unknown widths (A8 0x34), numerical type 0.
                (0x1160 + 4 * 21, 0x34 << 21),
                // 22: widths 0 -> not programmed, skipped.
                (0x1160 + 4 * 22, 3 | 7 << 27),
            ],
            |r| vertex_input(r),
        );
        let f = |components, bits, kind, bgra| AttrFormat { components, bits, kind, bgra };
        let want = [
            (0, 5, 12, false, f(3, [32, 32, 32, 0], AttrKind::Float, false)),
            (3, 31, 0x3fff, false, f(4, [8; 4], AttrKind::Unorm, true)),
            (20, 0, 0, true, f(1, [32, 0, 0, 0], AttrKind::Float, false)),
            (21, 0, 0, false, f(0, [0; 4], AttrKind::Unknown(0), false)),
        ];
        assert_eq!(a.len(), want.len());
        for (got, (location, stream, offset, constant, format)) in a.iter().zip(want) {
            assert_eq!((got.location, got.stream, got.offset, got.constant, got.format), (location, stream, offset, constant, format));
        }
        assert_eq!(a[3].raw, 0x34 << 21);
    }

    #[test]
    fn widths_and_kinds() {
        let fmt = |w: u32, t: u32| attr_format(w << 21 | t << 27);
        let name = |w, t| wgpu_vertex_format(&fmt(w, t));
        assert_eq!(fmt(0x30, 2).bits, [10, 10, 10, 2]);
        assert_eq!(fmt(0x31, 7).bits, [11, 11, 10, 0]);
        for (t, k) in [(1, AttrKind::Snorm), (2, AttrKind::Unorm), (3, AttrKind::Sint), (4, AttrKind::Uint), (5, AttrKind::Uscaled), (6, AttrKind::Sscaled), (7, AttrKind::Float)] {
            assert_eq!(fmt(0x01, t).kind, k);
        }
        let cases = [
            (0x01, 7, Some(wgpu::VertexFormat::Float32x4)), (0x02, 4, Some(wgpu::VertexFormat::Uint32x3)), (0x03, 1, Some(wgpu::VertexFormat::Snorm16x4)),
            (0x04, 7, Some(wgpu::VertexFormat::Float32x2)), (0x05, 7, None), (0x0a, 4, Some(wgpu::VertexFormat::Uint8x4)),
            (0x0f, 7, Some(wgpu::VertexFormat::Float16x2)), (0x12, 3, Some(wgpu::VertexFormat::Sint32)), (0x13, 2, None),
            (0x18, 1, Some(wgpu::VertexFormat::Snorm8x2)), (0x1b, 2, Some(wgpu::VertexFormat::Unorm16)), (0x1d, 3, Some(wgpu::VertexFormat::Sint8)),
            (0x30, 2, Some(wgpu::VertexFormat::Unorm10_10_10_2)), (0x30, 4, None), (0x31, 7, None),
            (0x0a, 5, None), (0x12, 2, None), (0x0a, 7, None), (0x34, 2, None), (0x01, 0, None),
        ];
        for (w, t, want) in cases {
            assert_eq!(name(w, t), want, "widths {w:#x} type {t}");
        }
        let bgra = |t: u32, w: u32| wgpu_vertex_format(&attr_format(w << 21 | t << 27 | 1 << 31));
        assert_eq!(bgra(2, 0x0a), Some(wgpu::VertexFormat::Unorm8x4Bgra));
        assert_eq!(bgra(4, 0x0a), None);
        assert_eq!(bgra(2, 0x30), None);
    }

    #[test]
    fn index_buffers() {
        for (e, bytes) in [(0, 1), (1, 2), (2, 4), (3, 0)] {
            let ib = with(
                &[(0x17c8, 0x311), (0x17cc, 0x3330_1774), (0x17d8, e), (0x17dc, 9), (0x238, 0), (0x23c, 0x100), (0x1644, 1), (0x1648, 0xffff)],
                |r| index_buffer(r),
            );
            assert_eq!(
                ib,
                IndexBuffer { address: 0x11_3330_1774, bytes_per_index: bytes, first: 9, end: 0x11_3330_1873, restart_enable: true, restart_index: 0xffff }
            );
        }
        let ib = with(&[], |r| index_buffer(r));
        assert!(!ib.restart_enable && ib.end == u64::MAX);
    }

    /// The loops give what the iterator chains they replaced gave (kept here,
    /// frozen), for random register files: streams enabled or not, attributes
    /// programmed or not, with gaps.
    #[test]
    fn vertex_input_matches_the_chains_it_replaced() {
        fn chains(r: Reg) -> (Vec<Stream>, Vec<Attribute>) {
            let streams = (0..COUNT)
                .filter(|j| field(r(0x1c00 + 16 * j), 12, 12) == 1)
                .map(|j| {
                    let address = addr40(r, 0x1c04 + 16 * j);
                    let size = addr40(r, 0x600 + 8 * j);
                    Stream { index: j as u8, address, stride: field(r(0x1c00 + 16 * j), 11, 0), end: address.wrapping_add(size).wrapping_sub(1), per_instance: field(r(0x1880 + 4 * j), 0, 0) == 1, divisor: r(0x1c0c + 16 * j) }
                })
                .collect();
            let attributes = (0..COUNT)
                .map(|i| (i, r(0x1160 + 4 * i)))
                .filter(|&(_, v)| field(v, 26, 21) != 0)
                .map(|(i, v)| Attribute { location: i as u8, stream: field(v, 4, 0) as u8, offset: field(v, 20, 7), constant: field(v, 6, 6) == 1, format: attr_format(v), raw: v })
                .collect();
            (streams, attributes)
        }
        let mut x = 0x2545_f491_4f6c_dd1du64;
        let mut next = || {
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            x
        };
        for _ in 0..200 {
            // About half the registers 0 (stream disabled, attribute not programmed).
            let regs: Vec<u32> = (0..0x1000)
                .map(|_| {
                    let v = next();
                    if v >> 40 & 1 == 0 { 0 } else { v as u32 }
                })
                .collect();
            let r = |m: u32| regs[(m / 4) as usize];
            assert_eq!(vertex_input(&r), chains(&r));
        }
    }
}
