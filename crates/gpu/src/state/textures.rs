//! Texture headers (TIC), sampler headers (TSC) and the block-linear memory
//! layout of the Ampere 3D class, decoded for wgpu 30.
//!
//! Field names and bit ranges are those of clc797tex.h (MW(hi:lo) bit indices
//! over the 32-byte header; TEXSAMPn fields are bit ranges of 32-bit word n).
//! The block-linear layout follows Mesa NIL (tiling.rs, image.rs, copy.rs).
//! Values that could not be established are kept raw or returned as
//! `Unknown` / `None`, never guessed.

use std::fmt;
use wgpu::TextureFormat as T;

/// Bits hi..=lo of a 256-bit header (clc797tex.h `MW(hi:lo)`).
fn mw(h: &[u8; 32], hi: u32, lo: u32) -> u32 {
    debug_assert_eq!(hi / 128, lo / 128, "no field crosses bit 128");
    let half = (lo / 128) as usize * 16;
    let v = u128::from_le_bytes(h[half..half + 16].try_into().unwrap());
    ((v >> (lo % 128)) & ((1u128 << (hi - lo + 1)) - 1)) as u32
}

/// Bits hi..=lo of word `w` of a sampler header (clc797tex.h `TEXSAMPw_*`).
fn samp(h: &[u8; 32], w: u32, hi: u32, lo: u32) -> u32 {
    mw(h, w * 32 + hi, w * 32 + lo)
}

// ---------------------------------------------------------------- TIC ----

/// `HEADER_VERSION` (`SELECT_*`): which of the five header layouts is used.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Layout {
    /// `SELECT_ONE_D_BUFFER` (TEXHEAD_1D_*)
    OneDBuffer,
    /// `SELECT_PITCH_COLOR_KEY` (TEXHEAD_PITCHCK_*)
    PitchColorKey,
    /// `SELECT_PITCH` (TEXHEAD_PITCH_*)
    Pitch,
    /// `SELECT_BLOCKLINEAR` (TEXHEAD_BL_*)
    BlockLinear,
    /// `SELECT_BLOCKLINEAR_COLOR_KEY` (TEXHEAD_BLCK_*)
    BlockLinearColorKey,
}

/// `R/G/B/A_DATA_TYPE` (`NUM_*`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DataType {
    Snorm,
    Unorm,
    Sint,
    Uint,
    SnormForceFp16,
    UnormForceFp16,
    Float,
    Unknown(u32),
}

impl DataType {
    fn from(v: u32) -> Self {
        [Self::Unknown(0), Self::Snorm, Self::Unorm, Self::Sint, Self::Uint, Self::SnormForceFp16, Self::UnormForceFp16, Self::Float][v as usize & 7]
    }
    /// One letter as in NIL's format table: N, S, I, U, F (FORCE_FP16 only
    /// changes the filtering precision, not the format).
    fn kind(self) -> char {
        match self {
            Self::Snorm | Self::SnormForceFp16 => 'S',
            Self::Unorm | Self::UnormForceFp16 => 'N',
            Self::Sint => 'I',
            Self::Uint => 'U',
            Self::Float => 'F',
            Self::Unknown(_) => '?',
        }
    }
}

/// `X/Y/Z/W_SOURCE` (`IN_*`): where each sampled component comes from.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Source {
    Zero,
    R,
    G,
    B,
    A,
    OneInt,
    OneFloat,
    Unknown(u32),
}

impl Source {
    fn from(v: u32) -> Self {
        [Self::Zero, Self::Unknown(1), Self::R, Self::G, Self::B, Self::A, Self::OneInt, Self::OneFloat][v as usize & 7]
    }
}

/// `TEXTURE_TYPE`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TextureType {
    OneD,
    TwoD,
    ThreeD,
    Cubemap,
    OneDArray,
    TwoDArray,
    OneDBuffer,
    TwoDNoMipmap,
    CubemapArray,
    HtexTwoD,
    HtexThreeD,
    HtexTwoDArray,
    Unknown(u32),
}

impl TextureType {
    fn from(v: u32) -> Self {
        match v {
            0 => Self::OneD,
            1 => Self::TwoD,
            2 => Self::ThreeD,
            3 => Self::Cubemap,
            4 => Self::OneDArray,
            5 => Self::TwoDArray,
            6 => Self::OneDBuffer,
            7 => Self::TwoDNoMipmap,
            8 => Self::CubemapArray,
            0xa => Self::HtexTwoD,
            0xb => Self::HtexThreeD,
            0xe => Self::HtexTwoDArray,
            v => Self::Unknown(v),
        }
    }

    /// wgpu 30 `TextureViewDimension` variant. None for 1D arrays (wgpu has
    /// none: use a D2Array of height 1), texel buffers and HTEX (sparse) types.
    pub fn wgpu_view_dimension(self) -> Option<wgpu::TextureViewDimension> {
        use wgpu::TextureViewDimension as D;
        Some(match self {
            Self::OneD => D::D1,
            Self::TwoD | Self::TwoDNoMipmap => D::D2,
            Self::ThreeD => D::D3,
            Self::Cubemap => D::Cube,
            Self::TwoDArray => D::D2Array,
            Self::CubemapArray => D::CubeArray,
            _ => return None,
        })
    }
}

/// `COMPONENTS` (`COMPONENTS_SIZES_*`), kept raw; `name()` gives the
/// clc797tex.h name. Names list components MSB first, so in A8B8G8R8 the R
/// component is the low byte (checked against NIL's nil_formats.csv).
#[derive(Clone, Copy, PartialEq, Eq)]
pub struct Components(pub u32);

/// (COMPONENTS value, name, bytes per element, element width, element height
/// in texels). 0 bytes = size not established.
const COMPONENTS: &[(u32, &str, u32, u32, u32)] = &[
    (0x00, "INVALID", 0, 1, 1),
    (0x01, "R32_G32_B32_A32", 16, 1, 1),
    (0x02, "R32_G32_B32", 12, 1, 1),
    (0x03, "R16_G16_B16_A16", 8, 1, 1),
    (0x04, "R32_G32", 8, 1, 1),
    (0x05, "R32_B24G8", 8, 1, 1),
    (0x06, "ETC2_RGB", 8, 4, 4),
    (0x07, "X8B8G8R8", 4, 1, 1),
    (0x08, "A8B8G8R8", 4, 1, 1),
    (0x09, "A2B10G10R10", 4, 1, 1),
    (0x0a, "ETC2_RGB_PTA", 8, 4, 4),
    (0x0b, "ETC2_RGBA", 16, 4, 4),
    (0x0c, "R16_G16", 4, 1, 1),
    (0x0d, "G8R24", 4, 1, 1),
    (0x0e, "G24R8", 4, 1, 1),
    (0x0f, "R32", 4, 1, 1),
    (0x10, "BC6H_SF16", 16, 4, 4),
    (0x11, "BC6H_UF16", 16, 4, 4),
    (0x12, "A4B4G4R4", 2, 1, 1),
    (0x13, "A5B5G5R1", 2, 1, 1),
    (0x14, "A1B5G5R5", 2, 1, 1),
    (0x15, "B5G6R5", 2, 1, 1),
    (0x16, "B6G5R5", 2, 1, 1),
    (0x17, "BC7U", 16, 4, 4),
    (0x18, "G8R8", 2, 1, 1),
    (0x19, "EAC", 8, 4, 4),
    (0x1a, "EACX2", 16, 4, 4),
    (0x1b, "R16", 2, 1, 1),
    (0x1c, "Y8_VIDEO", 1, 1, 1),
    (0x1d, "R8", 1, 1, 1),
    (0x1e, "G4R4", 1, 1, 1),
    (0x1f, "R1", 0, 1, 1),
    (0x20, "E5B9G9R9_SHAREDEXP", 4, 1, 1),
    (0x21, "BF10GF11RF11", 4, 1, 1),
    (0x22, "G8B8G8R8", 4, 2, 1),
    (0x23, "B8G8R8G8", 4, 2, 1),
    (0x24, "DXT1", 8, 4, 4),
    (0x25, "DXT23", 16, 4, 4),
    (0x26, "DXT45", 16, 4, 4),
    (0x27, "DXN1", 8, 4, 4),
    (0x28, "DXN2", 16, 4, 4),
    (0x29, "Z24S8", 4, 1, 1),
    (0x2a, "X8Z24", 4, 1, 1),
    (0x2b, "S8Z24", 4, 1, 1),
    (0x2c, "X4V4Z24__COV4R4V", 0, 1, 1),
    (0x2d, "X4V4Z24__COV8R8V", 0, 1, 1),
    (0x2e, "V8Z24__COV4R12V", 0, 1, 1),
    (0x2f, "ZF32", 4, 1, 1),
    (0x30, "ZF32_X24S8", 8, 1, 1),
    (0x31, "X8Z24_X20V4S8__COV4R4V", 0, 1, 1),
    (0x32, "X8Z24_X20V4S8__COV8R8V", 0, 1, 1),
    (0x33, "ZF32_X20V4X8__COV4R4V", 0, 1, 1),
    (0x34, "ZF32_X20V4X8__COV8R8V", 0, 1, 1),
    (0x35, "ZF32_X20V4S8__COV4R4V", 0, 1, 1),
    (0x36, "ZF32_X20V4S8__COV8R8V", 0, 1, 1),
    (0x37, "X8Z24_X16V8S8__COV4R12V", 0, 1, 1),
    (0x38, "ZF32_X16V8X8__COV4R12V", 0, 1, 1),
    (0x39, "ZF32_X16V8S8__COV4R12V", 0, 1, 1),
    (0x3a, "Z16", 2, 1, 1),
    (0x3b, "V8Z24__COV8R24V", 0, 1, 1),
    (0x3c, "X8Z24_X16V8S8__COV8R24V", 0, 1, 1),
    (0x3d, "ZF32_X16V8X8__COV8R24V", 0, 1, 1),
    (0x3e, "ZF32_X16V8S8__COV8R24V", 0, 1, 1),
    (0x40, "ASTC_2D_4X4", 16, 4, 4),
    (0x50, "ASTC_2D_5X4", 16, 5, 4),
    (0x41, "ASTC_2D_5X5", 16, 5, 5),
    (0x51, "ASTC_2D_6X5", 16, 6, 5),
    (0x42, "ASTC_2D_6X6", 16, 6, 6),
    (0x55, "ASTC_2D_8X5", 16, 8, 5),
    (0x52, "ASTC_2D_8X6", 16, 8, 6),
    (0x44, "ASTC_2D_8X8", 16, 8, 8),
    (0x56, "ASTC_2D_10X5", 16, 10, 5),
    (0x57, "ASTC_2D_10X6", 16, 10, 6),
    (0x53, "ASTC_2D_10X8", 16, 10, 8),
    (0x45, "ASTC_2D_10X10", 16, 10, 10),
    (0x54, "ASTC_2D_12X10", 16, 12, 10),
    (0x46, "ASTC_2D_12X12", 16, 12, 12),
];

impl Components {
    fn row(self) -> Option<&'static (u32, &'static str, u32, u32, u32)> {
        COMPONENTS.iter().find(|c| c.0 == self.0)
    }
    /// The clc797tex.h `COMPONENTS_SIZES_*` name.
    pub fn name(self) -> Option<&'static str> {
        self.row().map(|c| c.1)
    }
    /// (bytes per element, element width, element height in texels);
    /// elements are 4x4.. blocks for compressed formats.
    pub fn element(self) -> Option<(u32, u32, u32)> {
        self.row().filter(|c| c.2 != 0).map(|c| (c.2, c.3, c.4))
    }
}

impl fmt::Debug for Components {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self.name() {
            Some(n) => write!(f, "{n}"),
            None => write!(f, "Unknown({:#x})", self.0),
        }
    }
}

/// A decoded texture header (one 32-byte entry of the texture header pool).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Tic {
    /// `HEADER_VERSION`
    pub layout: Layout,
    /// GPU address of level 0, layer 0: `ADDRESS_BITS31TO9` (block linear),
    /// `ADDRESS_BITS31TO5` (pitch) or `ADDRESS_BITS31TO0` (1D buffer), plus
    /// `ADDRESS_BITS48TO32`.
    pub address: u64,
    /// `COMPONENTS`
    pub components: Components,
    /// `R_DATA_TYPE`, `G_DATA_TYPE`, `B_DATA_TYPE`, `A_DATA_TYPE`
    pub data_types: [DataType; 4],
    /// `X_SOURCE`, `Y_SOURCE`, `Z_SOURCE`, `W_SOURCE`
    pub swizzle: [Source; 4],
    /// `PACK_COMPONENTS`
    pub pack_components: bool,
    /// `TEXTURE_TYPE`
    pub texture_type: TextureType,
    /// `WIDTH_MINUS_ONE` + 1 (1D buffer: `WIDTH_MINUS_ONE_BITS31TO16` and
    /// `_BITS15TO0`, in elements).
    pub width: u32,
    /// `HEIGHT_MINUS_ONE` | `HEIGHT_MINUS_ONE_BIT16` << 16, + 1 (1 for 1D buffers).
    pub height: u32,
    /// `DEPTH_MINUS_ONE` (| `DEPTH_MINUS_ONE_BIT14` << 14 in block-linear
    /// headers) + 1: slices for 3D, layers for arrays, cubes for cube maps
    /// (NVK writes array_len / 6 there).
    pub depth: u32,
    /// `MAX_MIP_LEVEL`: number of levels in memory minus one.
    pub max_mip_level: u32,
    /// (`RES_VIEW_MIN_MIP_LEVEL`, `RES_VIEW_MAX_MIP_LEVEL`): the levels the
    /// view exposes. None in color-key and 1D-buffer headers (no such field).
    pub view_levels: Option<(u32, u32)>,
    /// `S_R_G_B_CONVERSION`
    pub srgb: bool,
    /// `GOBS_PER_BLOCK_WIDTH` (raw; only ONE_GOB = 0 is defined). 0 unless block linear.
    pub gobs_per_block_width: u32,
    /// `GOBS_PER_BLOCK_HEIGHT`: log2 of the tile height in GOBs. 0 unless block linear.
    pub gob_height_log2: u32,
    /// `GOBS_PER_BLOCK_DEPTH`: log2 of the tile depth in GOBs. 0 unless block linear.
    pub gob_depth_log2: u32,
    /// `TILE_WIDTH_IN_GOBS`: log2 of the tile width in GOBs (NIL's x_log2). 0 unless block linear.
    pub tile_width_log2: u32,
    /// `GOB3D` (raw; meaning not established). false unless block linear.
    pub gob3d: bool,
    /// `GOB_DEPTH_OFFSET` (raw; meaning not established). 0 unless block linear.
    pub gob_depth_offset: u32,
    /// Row pitch in bytes: `PITCH_BITS20TO5` << 5 | `PITCH_BIT21` << 21. 0 unless pitch.
    pub pitch: u32,
    /// `NORMALIZED_COORDS`
    pub normalized_coords: bool,
    /// `DEPTH_TEXTURE`
    pub depth_texture: bool,
    /// `MULTI_SAMPLE_COUNT` (raw `MODE_*`, 0 = 1X1). None where the header has no such field.
    pub multi_sample_count: Option<u32>,
    /// `MIN_LOD_CLAMP` (unsigned 4.8 fixed point, as NVK packs it). None where absent.
    pub min_lod_clamp: Option<f32>,
    /// `COLOR_KEY_VALUE` (color-key headers only).
    pub color_key: Option<u32>,
}

/// Decodes one texture header. None for a reserved `HEADER_VERSION` (5..7):
/// the rest of the header cannot be laid out then.
pub fn tic(h: &[u8; 32]) -> Option<Tic> {
    use Layout as L;
    let layout = match mw(h, 87, 85) {
        0 => L::OneDBuffer,
        1 => L::PitchColorKey,
        2 => L::Pitch,
        3 => L::BlockLinear,
        4 => L::BlockLinearColorKey,
        _ => return None,
    };
    let bl = matches!(layout, L::BlockLinear | L::BlockLinearColorKey);
    let pitch = matches!(layout, L::Pitch | L::PitchColorKey);
    let one_d = layout == L::OneDBuffer;
    let full = matches!(layout, L::Pitch | L::BlockLinear); // has RES_VIEW_*, MULTI_SAMPLE_COUNT, MIN_LOD_CLAMP
    let low = match layout {
        L::OneDBuffer => mw(h, 63, 32) as u64,
        L::Pitch | L::PitchColorKey => (mw(h, 63, 37) as u64) << 5,
        L::BlockLinear | L::BlockLinearColorKey => (mw(h, 63, 41) as u64) << 9,
    };
    let if_bl = |hi, lo| if bl { mw(h, hi, lo) } else { 0 };
    Some(Tic {
        layout,
        address: (mw(h, 80, 64) as u64) << 32 | low,
        components: Components(mw(h, 6, 0)),
        data_types: [mw(h, 9, 7), mw(h, 12, 10), mw(h, 15, 13), mw(h, 18, 16)].map(DataType::from),
        swizzle: [mw(h, 21, 19), mw(h, 24, 22), mw(h, 27, 25), mw(h, 30, 28)].map(Source::from),
        pack_components: mw(h, 31, 31) != 0,
        texture_type: TextureType::from(mw(h, 154, 151)),
        width: if one_d { (mw(h, 111, 96) << 16 | mw(h, 143, 128)).saturating_add(1) } else { mw(h, 144, 128) + 1 },
        height: if one_d { 1 } else { (mw(h, 146, 146) << 16 | mw(h, 175, 160)) + 1 },
        depth: if one_d { 1 } else { (if_bl(145, 145) << 14 | mw(h, 189, 176)) + 1 },
        max_mip_level: if one_d { 0 } else { mw(h, 127, 124) },
        view_levels: full.then(|| (mw(h, 227, 224), mw(h, 231, 228))),
        srgb: mw(h, 150, 150) != 0,
        gobs_per_block_width: if_bl(98, 96),
        gob_height_log2: if_bl(101, 99),
        gob_depth_log2: if_bl(104, 102),
        tile_width_log2: if_bl(108, 106),
        gob3d: if_bl(109, 109) != 0,
        gob_depth_offset: if_bl(38, 37),
        pitch: if pitch { mw(h, 111, 96) << 5 | mw(h, 145, 145) << 21 } else { 0 },
        normalized_coords: !one_d && mw(h, 191, 191) != 0,
        depth_texture: !one_d && mw(h, 123, 123) != 0,
        multi_sample_count: full.then(|| mw(h, 235, 232)),
        min_lod_clamp: full.then(|| mw(h, 247, 236) as f32 / 256.0),
        color_key: (!full && !one_d).then(|| mw(h, 255, 224)),
    })
}

// ------------------------------------------------------- wgpu formats ----

const NORM16: &str = "needs Features::TEXTURE_FORMAT_16BIT_NORM";
const BC: &str = "needs Features::TEXTURE_COMPRESSION_BC";
const ETC: &str = "needs Features::TEXTURE_COMPRESSION_ETC2 (absent on most desktop GPUs: decode on the CPU)";
const ASTC: &str = "needs Features::TEXTURE_COMPRESSION_ASTC (absent on the GTX 1080: decode on the CPU)";
const RGB32: &str = "expand 12-byte texels to 16 bytes with A = 1";
const ZF32: &str = "WebGPU forbids buffer-to-texture copies into Depth32Float: upload as R32Float and copy with a shader";
const SRGB: &str = "S_R_G_B_CONVERSION is set but the wgpu format has no sRGB variant: decode in the shader";

/// (COMPONENTS name, data type letter as in NIL's nil_formats.csv or '*' for
/// any, wgpu 30 TextureFormat, its sRGB variant, channels a sample of that
/// wgpu format returns, conversion or feature needed).
type FormatRow = (&'static str, char, T, Option<T>, usize, Option<&'static str>);

const FORMATS: &[FormatRow] = &[
    ("R32_G32_B32_A32", 'F', T::Rgba32Float, None, 4, None),
    ("R32_G32_B32_A32", 'U', T::Rgba32Uint, None, 4, None),
    ("R32_G32_B32_A32", 'I', T::Rgba32Sint, None, 4, None),
    ("R32_G32_B32", 'F', T::Rgba32Float, None, 3, Some(RGB32)),
    ("R32_G32_B32", 'U', T::Rgba32Uint, None, 3, Some(RGB32)),
    ("R32_G32_B32", 'I', T::Rgba32Sint, None, 3, Some(RGB32)),
    ("R16_G16_B16_A16", 'F', T::Rgba16Float, None, 4, None),
    ("R16_G16_B16_A16", 'N', T::Rgba16Unorm, None, 4, Some(NORM16)),
    ("R16_G16_B16_A16", 'S', T::Rgba16Snorm, None, 4, Some(NORM16)),
    ("R16_G16_B16_A16", 'U', T::Rgba16Uint, None, 4, None),
    ("R16_G16_B16_A16", 'I', T::Rgba16Sint, None, 4, None),
    ("R32_G32", 'F', T::Rg32Float, None, 2, None),
    ("R32_G32", 'U', T::Rg32Uint, None, 2, None),
    ("R32_G32", 'I', T::Rg32Sint, None, 2, None),
    ("A8B8G8R8", 'N', T::Rgba8Unorm, Some(T::Rgba8UnormSrgb), 4, None),
    ("A8B8G8R8", 'S', T::Rgba8Snorm, None, 4, None),
    ("A8B8G8R8", 'U', T::Rgba8Uint, None, 4, None),
    ("A8B8G8R8", 'I', T::Rgba8Sint, None, 4, None),
    ("X8B8G8R8", 'N', T::Rgba8Unorm, Some(T::Rgba8UnormSrgb), 4, None),
    ("X8B8G8R8", 'S', T::Rgba8Snorm, None, 4, None),
    ("X8B8G8R8", 'U', T::Rgba8Uint, None, 4, None),
    ("X8B8G8R8", 'I', T::Rgba8Sint, None, 4, None),
    ("A2B10G10R10", 'N', T::Rgb10a2Unorm, None, 4, None),
    ("A2B10G10R10", 'U', T::Rgb10a2Uint, None, 4, None),
    ("R16_G16", 'F', T::Rg16Float, None, 2, None),
    ("R16_G16", 'N', T::Rg16Unorm, None, 2, Some(NORM16)),
    ("R16_G16", 'S', T::Rg16Snorm, None, 2, Some(NORM16)),
    ("R16_G16", 'U', T::Rg16Uint, None, 2, None),
    ("R16_G16", 'I', T::Rg16Sint, None, 2, None),
    ("R32", 'F', T::R32Float, None, 1, None),
    ("R32", 'U', T::R32Uint, None, 1, None),
    ("R32", 'I', T::R32Sint, None, 1, None),
    ("A4B4G4R4", 'N', T::Rgba8Unorm, Some(T::Rgba8UnormSrgb), 4, Some("expand R 3:0, G 7:4, B 11:8, A 15:12 to Rgba8")),
    ("A5B5G5R1", 'N', T::Rgba8Unorm, Some(T::Rgba8UnormSrgb), 4, Some("expand R 0, G 5:1, B 10:6, A 15:11 to Rgba8")),
    ("A1B5G5R5", 'N', T::Rgba8Unorm, Some(T::Rgba8UnormSrgb), 4, Some("expand R 4:0, G 9:5, B 14:10, A 15 to Rgba8")),
    ("B5G6R5", 'N', T::Rgba8Unorm, Some(T::Rgba8UnormSrgb), 3, Some("expand R 4:0, G 10:5, B 15:11 to Rgba8 with A = 1")),
    ("G8R8", 'N', T::Rg8Unorm, None, 2, None),
    ("G8R8", 'S', T::Rg8Snorm, None, 2, None),
    ("G8R8", 'U', T::Rg8Uint, None, 2, None),
    ("G8R8", 'I', T::Rg8Sint, None, 2, None),
    ("R16", 'F', T::R16Float, None, 1, None),
    ("R16", 'N', T::R16Unorm, None, 1, Some(NORM16)),
    ("R16", 'S', T::R16Snorm, None, 1, Some(NORM16)),
    ("R16", 'U', T::R16Uint, None, 1, None),
    ("R16", 'I', T::R16Sint, None, 1, None),
    ("R8", 'N', T::R8Unorm, None, 1, None),
    ("R8", 'S', T::R8Snorm, None, 1, None),
    ("R8", 'U', T::R8Uint, None, 1, None),
    ("R8", 'I', T::R8Sint, None, 1, None),
    ("G4R4", 'N', T::Rg8Unorm, None, 2, Some("expand R 3:0, G 7:4 to Rg8")),
    ("E5B9G9R9_SHAREDEXP", 'F', T::Rgb9e5Ufloat, None, 3, None),
    ("BF10GF11RF11", 'F', T::Rg11b10Ufloat, None, 3, None),
    ("DXT1", 'N', T::Bc1RgbaUnorm, Some(T::Bc1RgbaUnormSrgb), 4, Some(BC)),
    ("DXT23", 'N', T::Bc2RgbaUnorm, Some(T::Bc2RgbaUnormSrgb), 4, Some(BC)),
    ("DXT45", 'N', T::Bc3RgbaUnorm, Some(T::Bc3RgbaUnormSrgb), 4, Some(BC)),
    ("DXN1", 'N', T::Bc4RUnorm, None, 1, Some(BC)),
    ("DXN1", 'S', T::Bc4RSnorm, None, 1, Some(BC)),
    ("DXN2", 'N', T::Bc5RgUnorm, None, 2, Some(BC)),
    ("DXN2", 'S', T::Bc5RgSnorm, None, 2, Some(BC)),
    ("BC6H_SF16", 'F', T::Bc6hRgbFloat, None, 3, Some(BC)),
    ("BC6H_UF16", 'F', T::Bc6hRgbUfloat, None, 3, Some(BC)),
    ("BC7U", 'N', T::Bc7RgbaUnorm, Some(T::Bc7RgbaUnormSrgb), 4, Some(BC)),
    ("ETC2_RGB", 'N', T::Etc2Rgb8Unorm, Some(T::Etc2Rgb8UnormSrgb), 3, Some(ETC)),
    ("ETC2_RGB_PTA", 'N', T::Etc2Rgb8A1Unorm, Some(T::Etc2Rgb8A1UnormSrgb), 4, Some(ETC)),
    ("ETC2_RGBA", 'N', T::Etc2Rgba8Unorm, Some(T::Etc2Rgba8UnormSrgb), 4, Some(ETC)),
    ("EAC", 'N', T::EacR11Unorm, None, 1, Some(ETC)),
    ("EAC", 'S', T::EacR11Snorm, None, 1, Some(ETC)),
    ("EACX2", 'N', T::EacRg11Unorm, None, 2, Some(ETC)),
    ("EACX2", 'S', T::EacRg11Snorm, None, 2, Some(ETC)),
    ("Z16", '*', T::Depth16Unorm, None, 1, None),
    ("ZF32", '*', T::Depth32Float, None, 1, Some(ZF32)),
    ("X8Z24", '*', T::Depth32Float, None, 1, Some("unpack unorm depth bits 23:0 (TIC component R) to f32; then as ZF32: upload as R32Float and copy with a shader")),
    ("S8Z24", '*', T::Depth32Float, None, 1, Some("unpack unorm depth bits 23:0 (TIC component R) to f32, stencil 31:24 dropped; then as ZF32: upload as R32Float and copy with a shader")),
    ("Z24S8", '*', T::Depth32Float, None, 1, Some("unpack unorm depth bits 31:8 (TIC component G) to f32, stencil 7:0 dropped; then as ZF32: upload as R32Float and copy with a shader")),
    ("ZF32_X24S8", '*', T::Depth32FloatStencil8, None, 1, Some("8-byte texels, f32 depth (TIC component R) in the first word, stencil position not established; depth aspect is not a buffer-copy destination: upload as R32Float")),
    ("ASTC_2D_4X4", 'N', T::Astc { block: wgpu::AstcBlock::B4x4, channel: wgpu::AstcChannel::Unorm }, Some(T::Astc { block: wgpu::AstcBlock::B4x4, channel: wgpu::AstcChannel::UnormSrgb }), 4, Some(ASTC)),
    ("ASTC_2D_5X4", 'N', T::Astc { block: wgpu::AstcBlock::B5x4, channel: wgpu::AstcChannel::Unorm }, Some(T::Astc { block: wgpu::AstcBlock::B5x4, channel: wgpu::AstcChannel::UnormSrgb }), 4, Some(ASTC)),
    ("ASTC_2D_5X5", 'N', T::Astc { block: wgpu::AstcBlock::B5x5, channel: wgpu::AstcChannel::Unorm }, Some(T::Astc { block: wgpu::AstcBlock::B5x5, channel: wgpu::AstcChannel::UnormSrgb }), 4, Some(ASTC)),
    ("ASTC_2D_6X5", 'N', T::Astc { block: wgpu::AstcBlock::B6x5, channel: wgpu::AstcChannel::Unorm }, Some(T::Astc { block: wgpu::AstcBlock::B6x5, channel: wgpu::AstcChannel::UnormSrgb }), 4, Some(ASTC)),
    ("ASTC_2D_6X6", 'N', T::Astc { block: wgpu::AstcBlock::B6x6, channel: wgpu::AstcChannel::Unorm }, Some(T::Astc { block: wgpu::AstcBlock::B6x6, channel: wgpu::AstcChannel::UnormSrgb }), 4, Some(ASTC)),
    ("ASTC_2D_8X5", 'N', T::Astc { block: wgpu::AstcBlock::B8x5, channel: wgpu::AstcChannel::Unorm }, Some(T::Astc { block: wgpu::AstcBlock::B8x5, channel: wgpu::AstcChannel::UnormSrgb }), 4, Some(ASTC)),
    ("ASTC_2D_8X6", 'N', T::Astc { block: wgpu::AstcBlock::B8x6, channel: wgpu::AstcChannel::Unorm }, Some(T::Astc { block: wgpu::AstcBlock::B8x6, channel: wgpu::AstcChannel::UnormSrgb }), 4, Some(ASTC)),
    ("ASTC_2D_8X8", 'N', T::Astc { block: wgpu::AstcBlock::B8x8, channel: wgpu::AstcChannel::Unorm }, Some(T::Astc { block: wgpu::AstcBlock::B8x8, channel: wgpu::AstcChannel::UnormSrgb }), 4, Some(ASTC)),
    ("ASTC_2D_10X5", 'N', T::Astc { block: wgpu::AstcBlock::B10x5, channel: wgpu::AstcChannel::Unorm }, Some(T::Astc { block: wgpu::AstcBlock::B10x5, channel: wgpu::AstcChannel::UnormSrgb }), 4, Some(ASTC)),
    ("ASTC_2D_10X6", 'N', T::Astc { block: wgpu::AstcBlock::B10x6, channel: wgpu::AstcChannel::Unorm }, Some(T::Astc { block: wgpu::AstcBlock::B10x6, channel: wgpu::AstcChannel::UnormSrgb }), 4, Some(ASTC)),
    ("ASTC_2D_10X8", 'N', T::Astc { block: wgpu::AstcBlock::B10x8, channel: wgpu::AstcChannel::Unorm }, Some(T::Astc { block: wgpu::AstcBlock::B10x8, channel: wgpu::AstcChannel::UnormSrgb }), 4, Some(ASTC)),
    ("ASTC_2D_10X10", 'N', T::Astc { block: wgpu::AstcBlock::B10x10, channel: wgpu::AstcChannel::Unorm }, Some(T::Astc { block: wgpu::AstcBlock::B10x10, channel: wgpu::AstcChannel::UnormSrgb }), 4, Some(ASTC)),
    ("ASTC_2D_12X10", 'N', T::Astc { block: wgpu::AstcBlock::B12x10, channel: wgpu::AstcChannel::Unorm }, Some(T::Astc { block: wgpu::AstcBlock::B12x10, channel: wgpu::AstcChannel::UnormSrgb }), 4, Some(ASTC)),
    ("ASTC_2D_12X12", 'N', T::Astc { block: wgpu::AstcBlock::B12x12, channel: wgpu::AstcChannel::Unorm }, Some(T::Astc { block: wgpu::AstcBlock::B12x12, channel: wgpu::AstcChannel::UnormSrgb }), 4, Some(ASTC)),
];

/// The table row for this header: COMPONENTS name, and the data type shared
/// by every component the wgpu format has (mixed types have no wgpu format).
fn format_row(t: &Tic) -> Option<&'static FormatRow> {
    let name = t.components.name()?;
    FORMATS.iter().find(|r| r.0 == name && (r.1 == '*' || t.data_types[..r.4].iter().all(|d| d.kind() == r.1)))
}

/// The wgpu 30 `TextureFormat` variant holding the texture as it is in memory
/// (wgpu R = TIC component R). The TIC swizzle is not folded in: see
/// [`needs_swizzle`]; conversions and features: see [`format_notes`].
/// None when wgpu has no usable format (YUV, coverage, R1, mixed types...).
pub fn wgpu_texture_format(t: &Tic) -> Option<T> {
    format_row(t).map(|r| if t.srgb { r.3.unwrap_or(r.2) } else { r.2 })
}

/// `X/Y/Z/W_SOURCE` as their raw 3-bit `IN_*` values, x in bits 2:0 (the
/// translated programs apply them: 0 zero, 2..5 R..A, 6 integer one, 7 float one).
pub fn swizzle_code(t: &Tic) -> u32 {
    let raw = |s: Source| match s {
        Source::Zero => 0,
        Source::R => 2,
        Source::G => 3,
        Source::B => 4,
        Source::A => 5,
        Source::OneInt => 6,
        Source::OneFloat => 7,
        Source::Unknown(v) => v,
    };
    t.swizzle.iter().enumerate().map(|(i, s)| raw(*s) << (3 * i)).sum()
}

/// True when `X/Y/Z/W_SOURCE` differ from what sampling the wgpu format gives
/// (R, G, B, A for its channels, then 0, 0, 1): wgpu 30 has no view swizzle,
/// so the shader translation must apply it.
pub fn needs_swizzle(t: &Tic) -> bool {
    use Source::*;
    let channels = format_row(t).map_or(4, |r| r.4);
    let natural = [R, G, B, A];
    t.swizzle.iter().enumerate().any(|(i, &s)| {
        let want = if i < channels { natural[i] } else if i == 3 { OneFloat } else { Zero };
        !(s == want || (want == OneFloat && s == OneInt))
    })
}

/// Conversions or wgpu features the format needs before upload, if any.
pub fn format_notes(t: &Tic) -> Vec<&'static str> {
    let Some(r) = format_row(t) else { return Vec::new() };
    let mut notes: Vec<&'static str> = r.5.into_iter().collect();
    if t.srgb && r.3.is_none() {
        notes.push(SRGB);
    }
    notes
}

// ---------------------------------------------------------------- TSC ----

/// `TEXSAMP0_ADDRESS_U/_V/_P` values.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Address {
    Wrap,
    Mirror,
    ClampToEdge,
    Border,
    ClampOgl,
    MirrorOnceClampToEdge,
    MirrorOnceBorder,
    MirrorOnceClampOgl,
}

impl Address {
    fn from(v: u32) -> Self {
        use Address::*;
        [Wrap, Mirror, ClampToEdge, Border, ClampOgl, MirrorOnceClampToEdge, MirrorOnceBorder, MirrorOnceClampOgl][v as usize & 7]
    }
    /// wgpu 30 `AddressMode`. ClampToBorder needs
    /// Features::ADDRESS_MODE_CLAMP_TO_BORDER. None: no wgpu equivalent
    /// (GL-style clamp and the mirror-once modes).
    pub fn wgpu(self) -> Option<wgpu::AddressMode> {
        use wgpu::AddressMode as A;
        Some(match self {
            Self::Wrap => A::Repeat,
            Self::Mirror => A::MirrorRepeat,
            Self::ClampToEdge => A::ClampToEdge,
            Self::Border => A::ClampToBorder,
            _ => return None,
        })
    }
}

/// `TEXSAMP0_DEPTH_COMPARE_FUNC` (`ZC_*`). NVK maps Vulkan compare ops to it
/// one to one, so the wgpu meaning is the same.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CompareFunc {
    Never,
    Less,
    Equal,
    LessEqual,
    Greater,
    NotEqual,
    GreaterEqual,
    Always,
}

impl CompareFunc {
    fn from(v: u32) -> Self {
        use CompareFunc::*;
        [Never, Less, Equal, LessEqual, Greater, NotEqual, GreaterEqual, Always][v as usize & 7]
    }
    /// wgpu 30 `CompareFunction` (the variants share these names).
    pub fn wgpu(self) -> wgpu::CompareFunction {
        use wgpu::CompareFunction as C;
        [C::Never, C::Less, C::Equal, C::LessEqual, C::Greater, C::NotEqual, C::GreaterEqual, C::Always][self as usize]
    }
}

/// `TEXSAMP1_MAG_FILTER` (`MAG_*`, `VCAA_*`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MagFilter {
    Point,
    Linear,
    Vcaa4Tap,
    Vcaa8Tap,
    Unknown(u32),
}

/// `TEXSAMP1_MIN_FILTER` (`MIN_*`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MinFilter {
    Point,
    Linear,
    Aniso,
    Unknown(u32),
}

/// `TEXSAMP1_MIP_FILTER` (`MIP_*`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MipFilter {
    /// `MIP_NONE`: only the base level is sampled.
    NoMip,
    Point,
    Linear,
    Unknown(u32),
}

impl MagFilter {
    /// wgpu 30 `FilterMode`; None for the VCAA modes.
    pub fn wgpu(self) -> Option<wgpu::FilterMode> {
        match self {
            Self::Point => Some(wgpu::FilterMode::Nearest),
            Self::Linear => Some(wgpu::FilterMode::Linear),
            _ => None,
        }
    }
}

impl MinFilter {
    /// wgpu 30 `FilterMode`. MIN_ANISO is Linear plus `anisotropy_clamp`.
    pub fn wgpu(self) -> Option<wgpu::FilterMode> {
        match self {
            Self::Point => Some(wgpu::FilterMode::Nearest),
            Self::Linear | Self::Aniso => Some(wgpu::FilterMode::Linear),
            Self::Unknown(_) => None,
        }
    }
}

impl MipFilter {
    /// wgpu 30 `MipmapFilterMode`. MIP_NONE maps to Nearest with the LOD range
    /// pinned to the base level (see [`Tsc::wgpu_lod_clamp`]).
    pub fn wgpu(self) -> Option<wgpu::MipmapFilterMode> {
        match self {
            Self::NoMip | Self::Point => Some(wgpu::MipmapFilterMode::Nearest),
            Self::Linear => Some(wgpu::MipmapFilterMode::Linear),
            Self::Unknown(_) => None,
        }
    }
}

/// A decoded sampler header (one 32-byte entry of the sampler pool).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Tsc {
    /// `TEXSAMP0_ADDRESS_U`, `TEXSAMP0_ADDRESS_V`, `TEXSAMP0_ADDRESS_P` (u, v, w)
    pub address: [Address; 3],
    /// `TEXSAMP0_DEPTH_COMPARE`
    pub depth_compare: bool,
    /// `TEXSAMP0_DEPTH_COMPARE_FUNC`
    pub compare_func: CompareFunc,
    /// `TEXSAMP0_S_R_G_B_CONVERSION`
    pub srgb_conversion: bool,
    /// `TEXSAMP0_MAX_ANISOTROPY` as a ratio: 1, 2, 4, 6, 8, 10, 12 or 16
    /// (wgpu `anisotropy_clamp`, which requires all filters Linear).
    pub max_anisotropy: u16,
    /// `TEXSAMP0_FORCED_INVALID`
    pub forced_invalid: bool,
    /// `TEXSAMP1_MAG_FILTER`
    pub mag_filter: MagFilter,
    /// `TEXSAMP1_MIN_FILTER`
    pub min_filter: MinFilter,
    /// `TEXSAMP1_MIP_FILTER`
    pub mip_filter: MipFilter,
    /// `TEXSAMP1_CUBEMAP_INTERFACE_FILTERING` (raw: 0 USE_WRAP, 1 OVERRIDE_WRAP,
    /// 2 AUTO_SPAN_SEAM, 3 AUTO_CROSS_SEAM)
    pub cubemap_interface_filtering: u32,
    /// `TEXSAMP1_REDUCTION_FILTER` (raw: 0 RED_NONE, 1 RED_MINIMUM,
    /// 2 RED_MAXIMUM; wgpu 30 has no min/max reduction samplers)
    pub reduction_filter: u32,
    /// `TEXSAMP1_MIP_LOD_BIAS`: signed fixed point with 8 fraction bits (as
    /// NVK packs it). wgpu samplers have no bias: apply it in the shader.
    pub mip_lod_bias: f32,
    /// `TEXSAMP1_FLOAT_COORD_NORMALIZATION` == FORCE_UNNORMALIZED_COORDS
    pub force_unnormalized_coords: bool,
    /// `TEXSAMP1_TRILIN_OPT` (raw)
    pub trilin_opt: u32,
    /// `TEXSAMP2_MIN_LOD_CLAMP`: unsigned 4.8 fixed point
    pub min_lod: f32,
    /// `TEXSAMP2_MAX_LOD_CLAMP`: unsigned 4.8 fixed point
    pub max_lod: f32,
    /// `TEXSAMP2_S_R_G_B_BORDER_COLOR_R`, `TEXSAMP3_S_R_G_B_BORDER_COLOR_G`, `_B`
    pub srgb_border: [u8; 3],
    /// `TEXSAMP4_BORDER_COLOR_R` .. `TEXSAMP7_BORDER_COLOR_A`: raw 32-bit
    /// values (float bits for float/norm textures, integers for int ones).
    pub border_color: [u32; 4],
}

/// Decodes one sampler header.
pub fn tsc(h: &[u8; 32]) -> Tsc {
    let s = |w, hi, lo| samp(h, w, hi, lo);
    let bias = s(1, 24, 12);
    Tsc {
        address: [s(0, 2, 0), s(0, 5, 3), s(0, 8, 6)].map(Address::from),
        depth_compare: s(0, 9, 9) != 0,
        compare_func: CompareFunc::from(s(0, 12, 10)),
        srgb_conversion: s(0, 13, 13) != 0,
        max_anisotropy: [1, 2, 4, 6, 8, 10, 12, 16][s(0, 22, 20) as usize],
        forced_invalid: s(0, 31, 31) != 0,
        mag_filter: match s(1, 2, 0) {
            1 => MagFilter::Point,
            2 => MagFilter::Linear,
            3 => MagFilter::Vcaa4Tap,
            4 => MagFilter::Vcaa8Tap,
            v => MagFilter::Unknown(v),
        },
        min_filter: match s(1, 5, 4) {
            1 => MinFilter::Point,
            2 => MinFilter::Linear,
            3 => MinFilter::Aniso,
            v => MinFilter::Unknown(v),
        },
        mip_filter: match s(1, 7, 6) {
            1 => MipFilter::NoMip,
            2 => MipFilter::Point,
            3 => MipFilter::Linear,
            v => MipFilter::Unknown(v),
        },
        cubemap_interface_filtering: s(1, 9, 8),
        reduction_filter: s(1, 11, 10),
        mip_lod_bias: ((bias << 19) as i32 >> 19) as f32 / 256.0, // sign-extend 13 bits
        force_unnormalized_coords: s(1, 25, 25) != 0,
        trilin_opt: s(1, 30, 26),
        min_lod: s(2, 11, 0) as f32 / 256.0,
        max_lod: s(2, 23, 12) as f32 / 256.0,
        srgb_border: [s(2, 31, 24) as u8, s(3, 19, 12) as u8, s(3, 27, 20) as u8],
        border_color: [s(4, 31, 0), s(5, 31, 0), s(6, 31, 0), s(7, 31, 0)],
    }
}

impl Tsc {
    /// wgpu 30 `SamplerBorderColor` when the border color is one of its three
    /// constants (0 or 1.0 bit patterns); None for any other (custom) color.
    pub fn wgpu_border_color(&self) -> Option<wgpu::SamplerBorderColor> {
        use wgpu::SamplerBorderColor as B;
        match self.border_color {
            [0, 0, 0, 0] => Some(B::TransparentBlack),
            [0, 0, 0, 0x3f80_0000] => Some(B::OpaqueBlack),
            [0x3f80_0000, 0x3f80_0000, 0x3f80_0000, 0x3f80_0000] => Some(B::OpaqueWhite),
            _ => None,
        }
    }

    /// wgpu (`lod_min_clamp`, `lod_max_clamp`). MIP_NONE samples the base
    /// level only: [0, 0.25] keeps it there while the LOD still tells
    /// minification from magnification (a [0, 0] clamp would always
    /// magnify, and MIN_LOD_CLAMP would pick another level).
    pub fn wgpu_lod_clamp(&self) -> (f32, f32) {
        match self.mip_filter {
            MipFilter::NoMip => (0.0, 0.25),
            _ => (self.min_lod, self.max_lod),
        }
    }
}

// ------------------------------------------------------- block linear ----

/// Byte order inside a 512-byte GOB (64 bytes x 8 rows), split in 16-byte
/// lines, as Mesa NIL's GOBType documents them.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum GobLayout {
    /// NIL `GOBType::TegraColor` (Tegra SoCs up to Maxwell: the Switch 1 layout).
    Tegra,
    /// NIL `GOBType::TuringColor2D` (what NIL picks for color images on
    /// Turing and later classes, including 0xc797).
    Turing2D,
}

impl GobLayout {
    /// The layout [`deswizzle`] and [`level_offsets`] assume: Turing2D, the
    /// rule NIL gives Turing and later classes, and the one a reference
    /// emulator uses for Switch 2 titles (bits 5 and 6 of the GOB offset
    /// swapped against the Switch 1's Tegra layout). Depth formats may use yet
    /// another layout (NIL `FermiZS`, undocumented).
    pub const DEFAULT: GobLayout = GobLayout::Turing2D;

    /// Byte offset inside the GOB of byte `x` (0..64) of row `y` (0..8).
    pub fn offset(self, x: u32, y: u32) -> u32 {
        let half = (x / 32) * 256; // two 32-byte-wide halves of 256 bytes
        let col = (x % 32) / 16; // 16-byte column inside the half
        match self {
            GobLayout::Tegra => half + (y / 2) * 64 + col * 32 + (y % 2) * 16 + x % 16,
            GobLayout::Turing2D => half + (y / 4) * 128 + col * 64 + (y % 4) * 16 + x % 16,
        }
    }
}

/// One mip level of a block-linear surface. Width and height are in elements
/// (blocks for compressed formats). Tile sizes are the header's log2 GOB
/// counts before the per-level clamp the hardware applies (NIL Tiling::clamp).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct BlockLinear {
    pub bytes_per_texel: u32,
    pub width: u32,
    pub height: u32,
    pub depth: u32,
    pub tile_width_log2: u32,
    pub gob_height_log2: u32,
    pub gob_depth_log2: u32,
    pub gob: GobLayout,
}

/// Tile extent (bytes, rows, slices) after the clamp, and tile counts.
struct Geom {
    tw: u64,
    th: u64,
    td: u64,
    tiles: [u64; 3],
    gob: GobLayout,
}

impl Geom {
    fn offset(&self, xb: u64, y: u64, z: u64) -> u64 {
        let tile = ((z / self.td) * self.tiles[1] + y / self.th) * self.tiles[0] + xb / self.tw;
        // GOBs inside a tile: x fastest, then y, then z.
        let gob = ((z % self.td) * (self.th / 8) + (y % self.th) / 8) * (self.tw / 64) + (xb % self.tw) / 64;
        tile * self.tw * self.th * self.td + gob * 512 + self.gob.offset((xb % 64) as u32, (y % 8) as u32) as u64
    }
}

impl BlockLinear {
    fn geom(&self) -> Geom {
        let wb = self.width * self.bytes_per_texel;
        let (h, d) = (self.height, self.depth);
        let (mut x, mut y, mut z) = (self.tile_width_log2, self.gob_height_log2, self.gob_depth_log2);
        // NIL Tiling::clamp: no wide tiles on a level smaller than the tile,
        // and tile height/depth at most the next power of two of the level.
        if wb < 64 << x || h < 8 << y || d < 1 << z {
            x = 0;
        }
        y = y.min(h.div_ceil(8).next_power_of_two().trailing_zeros());
        z = z.min(d.next_power_of_two().trailing_zeros());
        let (tw, th, td) = (64u64 << x, 8u64 << y, 1u64 << z);
        let tiles = [(wb as u64).div_ceil(tw), (h as u64).div_ceil(th), (d as u64).div_ceil(td)];
        Geom { tw, th, td, tiles, gob: self.gob }
    }

    /// Offset from the level's start of byte `x` of row `y` of slice `z`.
    pub fn offset(&self, x: u32, y: u32, z: u32) -> u64 {
        self.geom().offset(u64::from(x), u64::from(y), u64::from(z))
    }

    /// Bytes the level occupies in memory (whole tiles).
    pub fn size(&self) -> u64 {
        let g = self.geom();
        g.tiles.iter().product::<u64>() * g.tw * g.th * g.td
    }

    /// Linear rows (width * bytes_per_texel bytes each, height rows per
    /// slice, depth slices). Parts of `src` that are missing stay zero.
    pub fn deswizzle(&self, src: &[u8]) -> Vec<u8> {
        let g = self.geom();
        let row = (self.width * self.bytes_per_texel) as usize;
        let mut out = vec![0; row * self.height as usize * self.depth as usize];
        for z in 0..self.depth as usize {
            for y in 0..self.height as usize {
                let dst = (z * self.height as usize + y) * row;
                // 16 aligned bytes stay contiguous in both GOB layouts.
                for x in (0..row).step_by(16) {
                    let n = 16.min(row - x);
                    let s = g.offset(x as u64, y as u64, z as u64) as usize;
                    if let Some(chunk) = src.get(s..s + n) {
                        out[dst + x..dst + x + n].copy_from_slice(chunk);
                    }
                }
            }
        }
        out
    }
}

/// Linear rows of one block-linear mip level (tiles one GOB wide, GOB layout
/// [`GobLayout::DEFAULT`]); width/height in elements. Its size in memory is
/// [`level_size`].
pub fn deswizzle(src: &[u8], bytes_per_texel: u32, width: u32, height: u32, depth: u32, gob_height_log2: u32, gob_depth_log2: u32) -> Vec<u8> {
    level(bytes_per_texel, width, height, depth, gob_height_log2, gob_depth_log2).deswizzle(src)
}

/// Bytes one block-linear mip level occupies (same parameters as [`deswizzle`]).
pub fn level_size(bytes_per_texel: u32, width: u32, height: u32, depth: u32, gob_height_log2: u32, gob_depth_log2: u32) -> u64 {
    level(bytes_per_texel, width, height, depth, gob_height_log2, gob_depth_log2).size()
}

fn level(bytes_per_texel: u32, width: u32, height: u32, depth: u32, gob_height_log2: u32, gob_depth_log2: u32) -> BlockLinear {
    BlockLinear { bytes_per_texel, width, height, depth, tile_width_log2: 0, gob_height_log2, gob_depth_log2, gob: GobLayout::DEFAULT }
}

/// The block-linear description of level `l` of a header.
pub fn tic_level(t: &Tic, l: u32) -> Option<BlockLinear> {
    let (bpe, bw, bh) = t.components.element()?;
    let three_d = matches!(t.texture_type, TextureType::ThreeD | TextureType::HtexThreeD);
    Some(BlockLinear {
        bytes_per_texel: bpe,
        width: (t.width >> l).max(1).div_ceil(bw),
        height: (t.height >> l).max(1).div_ceil(bh),
        depth: if three_d { (t.depth >> l).max(1) } else { 1 },
        tile_width_log2: t.tile_width_log2,
        gob_height_log2: t.gob_height_log2,
        gob_depth_log2: t.gob_depth_log2,
        gob: GobLayout::DEFAULT,
    })
}

/// (offset of each level from `address`, layer stride) as NIL lays images out:
/// levels one after another, layers (array slices, cube faces) every stride,
/// the stride aligned to a level-0 tile.
fn layout(t: &Tic) -> Option<(Vec<u64>, u64)> {
    let (bpe, _, bh) = t.components.element()?;
    match t.layout {
        Layout::BlockLinear | Layout::BlockLinearColorKey => {
            let mut offsets = Vec::new();
            let mut end = 0;
            for l in 0..=t.max_mip_level {
                offsets.push(end);
                end += tic_level(t, l)?.size();
            }
            let g = tic_level(t, 0)?.geom();
            Some((offsets, end.next_multiple_of(g.tw * g.th * g.td)))
        }
        Layout::Pitch | Layout::PitchColorKey => Some((vec![0], t.pitch as u64 * t.height.div_ceil(bh) as u64)),
        Layout::OneDBuffer => Some((vec![0], t.width as u64 * bpe as u64)),
    }
}

/// Offsets from `address` of mip levels 0..=MAX_MIP_LEVEL (layer 0). Empty
/// when the element size of COMPONENTS is not established.
pub fn level_offsets(t: &Tic) -> Vec<u64> {
    layout(t).map(|l| l.0).unwrap_or_default()
}

/// Bytes between two layers (array slices or cube faces).
pub fn layer_stride(t: &Tic) -> Option<u64> {
    layout(t).map(|l| l.1)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn put(h: &mut [u8; 32], hi: u32, lo: u32, v: u32) {
        let half = (lo / 128) as usize * 16;
        let mut x = u128::from_le_bytes(h[half..half + 16].try_into().unwrap());
        let mask = ((1u128 << (hi - lo + 1)) - 1) << (lo % 128);
        x = (x & !mask) | (((v as u128) << (lo % 128)) & mask);
        h[half..half + 16].copy_from_slice(&x.to_le_bytes());
    }

    fn header(fields: &[(u32, u32, u32)]) -> [u8; 32] {
        let mut h = [0; 32];
        for &(hi, lo, v) in fields {
            put(&mut h, hi, lo, v);
        }
        h
    }

    #[test]
    fn tic_block_linear_fields() {
        let address = 0x1_8765_4321_0000u64;
        let h = header(&[
            (6, 0, 0x08),                                     // A8B8G8R8
            (9, 7, 2), (12, 10, 2), (15, 13, 2), (18, 16, 7), // N N N F
            (21, 19, 4), (24, 22, 3), (27, 25, 2), (30, 28, 6), // B G R 1i
            (31, 31, 1),
            (38, 37, 2),
            (63, 41, (address as u32) >> 9),
            (80, 64, (address >> 32) as u32),
            (87, 85, 3),
            (98, 96, 0), (101, 99, 4), (104, 102, 1), (108, 106, 1), (109, 109, 1),
            (123, 123, 1),
            (127, 124, 9),
            (144, 128, 69999),
            (145, 145, 19999 >> 14),
            (146, 146, 69999 >> 16),
            (150, 150, 1),
            (154, 151, 5),
            (175, 160, 69999 & 0xffff),
            (189, 176, 19999 & 0x3fff),
            (191, 191, 1),
            (227, 224, 2), (231, 228, 7),
            (235, 232, 3),
            (247, 236, 320),
        ]);
        let t = tic(&h).unwrap();
        assert_eq!(t.layout, Layout::BlockLinear);
        assert_eq!(t.address, address);
        assert_eq!(t.components.name(), Some("A8B8G8R8"));
        assert_eq!(t.data_types, [DataType::Unorm, DataType::Unorm, DataType::Unorm, DataType::Float]);
        assert_eq!(t.swizzle, [Source::B, Source::G, Source::R, Source::OneInt]);
        assert!(t.pack_components);
        assert_eq!(t.gob_depth_offset, 2);
        assert_eq!((t.gobs_per_block_width, t.gob_height_log2, t.gob_depth_log2, t.tile_width_log2, t.gob3d), (0, 4, 1, 1, true));
        assert!(t.depth_texture);
        assert_eq!(t.max_mip_level, 9);
        assert_eq!((t.width, t.height, t.depth), (70000, 70000, 20000));
        assert!(t.srgb);
        assert_eq!(t.texture_type, TextureType::TwoDArray);
        assert_eq!(t.texture_type.wgpu_view_dimension(), Some(wgpu::TextureViewDimension::D2Array));
        assert!(t.normalized_coords);
        assert_eq!(t.view_levels, Some((2, 7)));
        assert_eq!(t.multi_sample_count, Some(3));
        assert_eq!(t.min_lod_clamp, Some(1.25));
        assert_eq!((t.pitch, t.color_key), (0, None));
    }

    #[test]
    fn tic_pitch_one_d_and_color_key() {
        let p = tic(&header(&[
            (63, 37, 0x4440_5020 >> 5), (80, 64, 0x10), (87, 85, 2),
            (111, 96, (0x1940 >> 5) & 0xffff), (145, 145, 1), // pitch 0x201940
            (144, 128, 1599), (175, 160, 899), (189, 176, 0), (154, 151, 7),
        ]))
        .unwrap();
        assert_eq!(p.layout, Layout::Pitch);
        assert_eq!(p.address, 0x10_4440_5020);
        assert_eq!(p.pitch, 0x20_1940);
        assert_eq!((p.width, p.height, p.depth), (1600, 900, 1));
        assert_eq!((p.gob_height_log2, p.tile_width_log2), (0, 0));
        assert_eq!(p.texture_type, TextureType::TwoDNoMipmap);
        assert_eq!(level_offsets(&p), vec![] as Vec<u64>); // COMPONENTS INVALID: no element size

        let b = tic(&header(&[(6, 0, 0x0f), (63, 32, 0x6abc_1234), (80, 64, 0x10), (87, 85, 0), (111, 96, 0x1234), (143, 128, 0x5677), (154, 151, 6)])).unwrap();
        assert_eq!(b.layout, Layout::OneDBuffer);
        assert_eq!(b.address, 0x10_6abc_1234);
        assert_eq!((b.width, b.height, b.depth), (0x1234_5678, 1, 1));
        assert_eq!(b.texture_type, TextureType::OneDBuffer);
        assert_eq!((b.view_levels, b.color_key), (None, None));
        assert_eq!(layer_stride(&b), Some(0x1234_5678 * 4));

        let ck = tic(&header(&[(87, 85, 4), (255, 224, 0xdead_beef), (101, 99, 2)])).unwrap();
        assert_eq!(ck.layout, Layout::BlockLinearColorKey);
        assert_eq!(ck.color_key, Some(0xdead_beef));
        assert_eq!((ck.view_levels, ck.multi_sample_count, ck.min_lod_clamp), (None, None, None));
        assert_eq!(ck.gob_height_log2, 2);

        assert_eq!(tic(&header(&[(87, 85, 5)])), None);
    }

    fn fmt_tic(comps: u32, types: [u32; 4], src: [u32; 4], srgb: bool) -> Tic {
        let mut f = vec![(6, 0, comps), (87, 85, 3), (150, 150, srgb as u32)];
        for i in 0..4 {
            f.push((9 + 3 * i, 7 + 3 * i, types[i as usize]));
            f.push((21 + 3 * i, 19 + 3 * i, src[i as usize]));
        }
        tic(&header(&f)).unwrap()
    }

    #[test]
    fn wgpu_formats() {
        const N: u32 = 2;
        const RGBA: [u32; 4] = [2, 3, 4, 5];
        let t = fmt_tic(0x08, [N; 4], RGBA, true);
        assert_eq!(wgpu_texture_format(&t), Some(T::Rgba8UnormSrgb));
        assert!(!needs_swizzle(&t) && format_notes(&t).is_empty());
        assert!(needs_swizzle(&fmt_tic(0x08, [N; 4], [4, 3, 2, 5], false))); // BGRA view

        let r8 = fmt_tic(0x1d, [N; 4], [2, 0, 0, 7], true);
        assert_eq!(wgpu_texture_format(&r8), Some(T::R8Unorm));
        assert!(!needs_swizzle(&r8));
        assert_eq!(format_notes(&r8), vec![SRGB]);
        assert!(needs_swizzle(&fmt_tic(0x1d, [N; 4], [2, 2, 2, 7], false))); // luminance

        assert_eq!(wgpu_texture_format(&fmt_tic(0x02, [7; 4], [2, 3, 4, 7], false)), Some(T::Rgba32Float));
        assert_eq!(format_notes(&fmt_tic(0x02, [7; 4], [2, 3, 4, 7], false)), vec![RGB32]);
        assert_eq!(
            wgpu_texture_format(&fmt_tic(0x44, [N; 4], RGBA, true)),
            Some(T::Astc { block: wgpu::AstcBlock::B8x8, channel: wgpu::AstcChannel::UnormSrgb })
        );
        assert_eq!(wgpu_texture_format(&fmt_tic(0x27, [1; 4], [2, 0, 0, 7], false)), Some(T::Bc4RSnorm));
        assert_eq!(wgpu_texture_format(&fmt_tic(0x24, [N; 4], RGBA, false)), Some(T::Bc1RgbaUnorm));
        let z = fmt_tic(0x29, [4, 2, 4, 4], [3, 0, 0, 7], false); // Z24S8: depth is component G
        assert_eq!(wgpu_texture_format(&z), Some(T::Depth32Float));
        assert!(needs_swizzle(&z) && format_notes(&z)[0].contains("component G"));
        assert_eq!(wgpu_texture_format(&fmt_tic(0x1b, [N; 4], [2, 0, 0, 7], false)), Some(T::R16Unorm));
        assert_eq!(format_notes(&fmt_tic(0x1b, [N; 4], [2, 0, 0, 7], false)), vec![NORM16]);
        assert_eq!(wgpu_texture_format(&fmt_tic(0x08, [1, 1, 2, 2], RGBA, false)), None); // mixed types
        assert_eq!(wgpu_texture_format(&fmt_tic(0x7f, [N; 4], RGBA, false)), None);
        assert_eq!(fmt_tic(0x7f, [N; 4], RGBA, false).components.name(), None);
    }

    #[test]
    fn tsc_fields() {
        let mut h = [0; 32];
        let mut set = |w: u32, hi: u32, lo: u32, v: u32| put(&mut h, w * 32 + hi, w * 32 + lo, v);
        set(0, 2, 0, 0); // WRAP
        set(0, 5, 3, 1); // MIRROR
        set(0, 8, 6, 3); // BORDER
        set(0, 9, 9, 1);
        set(0, 12, 10, 6); // ZC_GEQUAL
        set(0, 13, 13, 1);
        set(0, 22, 20, 7); // 16:1
        set(1, 2, 0, 2); // MAG_LINEAR
        set(1, 5, 4, 3); // MIN_ANISO
        set(1, 7, 6, 1); // MIP_NONE
        set(1, 9, 8, 2);
        set(1, 11, 10, 2);
        set(1, 24, 12, 0x1e80); // -1.5
        set(1, 25, 25, 1);
        set(1, 30, 26, 4);
        set(2, 11, 0, 128); // 0.5
        set(2, 23, 12, 3520); // 13.75
        set(2, 31, 24, 0x11);
        set(3, 19, 12, 0x22);
        set(3, 27, 20, 0x33);
        set(7, 31, 0, 0x3f80_0000);
        let s = tsc(&h);
        assert_eq!(s.address, [Address::Wrap, Address::Mirror, Address::Border]);
        assert_eq!(s.address.map(Address::wgpu), [Some(wgpu::AddressMode::Repeat), Some(wgpu::AddressMode::MirrorRepeat), Some(wgpu::AddressMode::ClampToBorder)]);
        assert_eq!(Address::MirrorOnceClampToEdge.wgpu(), None);
        assert!(s.depth_compare && s.srgb_conversion && s.force_unnormalized_coords && !s.forced_invalid);
        assert_eq!(s.compare_func.wgpu(), wgpu::CompareFunction::GreaterEqual);
        assert_eq!(s.max_anisotropy, 16);
        assert_eq!((s.mag_filter, s.min_filter, s.mip_filter), (MagFilter::Linear, MinFilter::Aniso, MipFilter::NoMip));
        assert_eq!((s.mag_filter.wgpu(), s.min_filter.wgpu(), s.mip_filter.wgpu()), (Some(wgpu::FilterMode::Linear), Some(wgpu::FilterMode::Linear), Some(wgpu::MipmapFilterMode::Nearest)));
        assert_eq!((s.cubemap_interface_filtering, s.reduction_filter, s.trilin_opt), (2, 2, 4));
        assert_eq!(s.mip_lod_bias, -1.5);
        assert_eq!((s.min_lod, s.max_lod), (0.5, 13.75));
        assert_eq!(s.wgpu_lod_clamp(), (0.0, 0.25));
        assert_eq!(s.srgb_border, [0x11, 0x22, 0x33]);
        assert_eq!(s.wgpu_border_color(), Some(wgpu::SamplerBorderColor::OpaqueBlack));
        assert_eq!(MagFilter::Vcaa4Tap.wgpu(), None);
    }

    // GOB lines from NIL's tiling.rs tables: (x, y) of the 16-byte line at
    // offset 16 * k inside each 256-byte GOB half.
    const TEGRA_LINES: [(u32, u32); 16] = [(0, 0), (0, 1), (16, 0), (16, 1), (0, 2), (0, 3), (16, 2), (16, 3), (0, 4), (0, 5), (16, 4), (16, 5), (0, 6), (0, 7), (16, 6), (16, 7)];
    const TURING_LINES: [(u32, u32); 16] = [(0, 0), (0, 1), (0, 2), (0, 3), (16, 0), (16, 1), (16, 2), (16, 3), (0, 4), (0, 5), (0, 6), (0, 7), (16, 4), (16, 5), (16, 6), (16, 7)];

    /// Straightforward linear -> block-linear: walk tiles (x, then y, then z)
    /// and their GOBs (x, then y, then z) with a running pointer. No clamp:
    /// callers pass shapes the clamp leaves alone.
    #[allow(clippy::too_many_arguments)]
    fn reference_swizzle(lines: &[(u32, u32); 16], linear: &[u8], row: u32, h: u32, d: u32, xl: u32, yl: u32, zl: u32) -> Vec<u8> {
        let (tw, th, td) = (64 << xl, 8 << yl, 1 << zl);
        let (aw, ah, ad) = (row.next_multiple_of(tw), h.next_multiple_of(th), d.next_multiple_of(td));
        let mut out = vec![0; (aw * ah * ad) as usize];
        let mut ptr = 0;
        for tz in (0..ad).step_by(td as usize) {
            for ty in (0..ah).step_by(th as usize) {
                for tx in (0..aw).step_by(tw as usize) {
                    for gz in 0..td {
                        for gy in (0..th).step_by(8) {
                            for gx in (0..tw).step_by(64) {
                                for half in 0..2 {
                                    for (k, &(lx, ly)) in lines.iter().enumerate() {
                                        for b in 0..16 {
                                            let (x, y, z) = (tx + gx + half * 32 + lx + b, ty + gy + ly, tz + gz);
                                            if x < row && y < h && z < d {
                                                out[ptr + (half * 256) as usize + k * 16 + b as usize] = linear[((z * h + y) * row + x) as usize];
                                            }
                                        }
                                    }
                                }
                                ptr += 512;
                            }
                        }
                    }
                }
            }
        }
        out
    }

    #[test]
    fn deswizzle_matches_reference() {
        // (bytes per texel, width, height, depth, x_log2, y_log2, z_log2)
        let shapes = [(4, 37, 21, 1, 0, 1, 0), (16, 5, 9, 3, 0, 0, 1), (1, 300, 40, 1, 1, 2, 0), (4, 64, 64, 1, 0, 3, 0), (2, 33, 17, 5, 0, 1, 2)];
        for (gob, lines) in [(GobLayout::Tegra, &TEGRA_LINES), (GobLayout::Turing2D, &TURING_LINES)] {
            for &(bpt, w, h, d, xl, yl, zl) in &shapes {
                let row = w * bpt;
                let linear: Vec<u8> = (0..row * h * d).map(|i| (i % 251) as u8).collect();
                let tiled = reference_swizzle(lines, &linear, row, h, d, xl, yl, zl);
                let bl = BlockLinear { bytes_per_texel: bpt, width: w, height: h, depth: d, tile_width_log2: xl, gob_height_log2: yl, gob_depth_log2: zl, gob };
                assert_eq!(bl.size(), tiled.len() as u64, "{gob:?} {w}x{h}x{d}");
                assert_eq!(bl.deswizzle(&tiled), linear, "{gob:?} {w}x{h}x{d}");
            }
        }
        // The fixed-signature entry points use the default layout, 1-GOB-wide tiles.
        let linear: Vec<u8> = (0..4 * 37 * 21).map(|i| (i % 251) as u8).collect();
        let tiled = reference_swizzle(&TURING_LINES, &linear, 4 * 37, 21, 1, 0, 1, 0);
        assert_eq!(deswizzle(&tiled, 4, 37, 21, 1, 1, 0), linear);
        assert_eq!(level_size(4, 37, 21, 1, 1, 0), tiled.len() as u64);
    }

    #[test]
    fn deswizzle_clamps_tile_height_to_the_level() {
        // 16 rows = 2 GOBs: a requested 32-GOB tile behaves as a 2-GOB tile.
        let linear: Vec<u8> = (0..64 * 16).map(|i| (i % 251) as u8).collect();
        let tiled = reference_swizzle(&TURING_LINES, &linear, 64, 16, 1, 0, 1, 0);
        assert_eq!(deswizzle(&tiled, 4, 16, 16, 1, 5, 0), linear);
        assert_eq!(level_size(4, 16, 16, 1, 5, 0), 1024);
        assert_eq!(level_size(4, 1, 1, 1, 5, 0), 512); // one GOB at least
    }

    #[test]
    fn mip_chain_offsets() {
        // 256x256 RGBA8, 9 levels, 16-GOB-high tiles: sizes worked out by hand
        // from NIL's rules (align each level to its clamped tile).
        let t = tic(&header(&[(6, 0, 0x08), (87, 85, 3), (101, 99, 4), (127, 124, 8), (144, 128, 255), (175, 160, 255), (154, 151, 1)])).unwrap();
        assert_eq!(level_offsets(&t), vec![0, 262144, 327680, 344064, 348160, 349184, 349696, 350208, 350720]);
        assert_eq!(layer_stride(&t), Some(352256)); // 351232 aligned to a 64x128 tile
        // 64x64 BC1 (8-byte 4x4 blocks), 4 levels, 2-GOB-high tiles.
        let t = tic(&header(&[(6, 0, 0x24), (87, 85, 3), (101, 99, 1), (127, 124, 3), (144, 128, 63), (175, 160, 63), (154, 151, 1)])).unwrap();
        assert_eq!(level_offsets(&t), vec![0, 2048, 2560, 3072]);
        assert_eq!(layer_stride(&t), Some(4096));
        // 3D: 32x32x8 R8, 8-slice tiles; depth halves per level.
        let t = tic(&header(&[(6, 0, 0x1d), (87, 85, 3), (101, 99, 2), (104, 102, 3), (127, 124, 1), (144, 128, 31), (175, 160, 31), (189, 176, 7), (154, 151, 2)])).unwrap();
        assert_eq!(level_offsets(&t), vec![0, 64 * 32 * 8]); // level 1: 16x16x4 = 64 B x 16 rows x 4 slices
        assert_eq!(tic_level(&t, 1).unwrap().size(), 64 * 16 * 4);
    }
}
