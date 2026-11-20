//! Fixed-function state of a draw, decoded from the 3D-class (0xc797)
//! register file for a wgpu render pipeline and render pass: viewport 0,
//! scissor 0, surface clip, color outputs and blending, depth/stencil and
//! rasterizer state.
//!
//! Enum variants carry the exact wgpu 30 names (`wgpu::BlendFactor`,
//! `BlendOperation`, `CompareFunction`, `StencilOperation`, `Face`,
//! `FrontFace`, `PolygonMode`), so the integration maps them one to one by
//! name. Hardware values with no wgpu counterpart get their own variant;
//! values not listed in clc797.h decode to `Unknown(raw)`.
//!
//! A register the command stream never wrote reads 0. Fields whose 0 is not
//! a valid encoding then decode to `Unknown(0)` (e.g. SET_STENCIL_FUNC while
//! the stencil test is off); for stencil ops, 0 is OGL_ZERO.

fn bit(v: u32, b: u32) -> bool {
    v >> b & 1 != 0
}

// The readers take any register reader (see `vertex`): a draw's snapshot reads inlined.
fn float<R: Fn(u32) -> u32 + ?Sized>(r: &R, method: u32) -> f32 {
    f32::from_bits(r(method))
}

/// wgpu::BlendFactor names, plus the hardware factors wgpu lacks.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BlendFactor {
    Zero,
    One,
    Src,
    OneMinusSrc,
    SrcAlpha,
    OneMinusSrcAlpha,
    Dst,
    OneMinusDst,
    DstAlpha,
    OneMinusDstAlpha,
    SrcAlphaSaturated,
    Constant,
    OneMinusConstant,
    Src1,
    OneMinusSrc1,
    Src1Alpha,
    OneMinusSrc1Alpha,
    /// OGL_CONSTANT_ALPHA: no wgpu factor in a color component (in the alpha
    /// component it equals `Constant`).
    ConstantAlpha,
    /// OGL_ONE_MINUS_CONSTANT_ALPHA: same remark as `ConstantAlpha`.
    OneMinusConstantAlpha,
    /// D3D_BOTHSRCALPHA (listed for source coefficients only): no wgpu factor.
    BothSrcAlpha,
    /// D3D_BOTHINVSRCALPHA (listed for source coefficients only): no wgpu factor.
    BothInvSrcAlpha,
    Unknown(u32),
}

impl BlendFactor {
    /// SET_BLEND_{COLOR,ALPHA}_{SOURCE,DEST}_COEFF_V and the per-target
    /// variants: OGL_* (0x4000.., 0xC000..) and D3D_* (1..0x13) encodings.
    pub fn decode(v: u32) -> Self {
        use BlendFactor::*;
        match v {
            0x4000 | 0x01 => Zero,
            0x4001 | 0x02 => One,
            0x4300 | 0x03 => Src,
            0x4301 | 0x04 => OneMinusSrc,
            0x4302 | 0x05 => SrcAlpha,
            0x4303 | 0x06 => OneMinusSrcAlpha,
            0x4304 | 0x07 => DstAlpha,
            0x4305 | 0x08 => OneMinusDstAlpha,
            0x4306 | 0x09 => Dst,
            0x4307 | 0x0a => OneMinusDst,
            0x4308 | 0x0b => SrcAlphaSaturated,
            0x0c => BothSrcAlpha,
            0x0d => BothInvSrcAlpha,
            // D3D_BLENDFACTOR / D3D_INVBLENDFACTOR use the 4-component constant.
            0xc001 | 0x0e => Constant,
            0xc002 | 0x0f => OneMinusConstant,
            0xc003 => ConstantAlpha,
            0xc004 => OneMinusConstantAlpha,
            0xc900 | 0x10 => Src1,
            0xc901 | 0x11 => OneMinusSrc1,
            0xc902 | 0x12 => Src1Alpha,
            0xc903 | 0x13 => OneMinusSrc1Alpha,
            _ => Unknown(v),
        }
    }
}

/// wgpu::BlendOperation names.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BlendOperation {
    Add,
    Subtract,
    ReverseSubtract,
    Min,
    Max,
    Unknown(u32),
}

impl BlendOperation {
    /// SET_BLEND_{COLOR,ALPHA}_OP_V and SET_BLEND_PER_TARGET_{COLOR,ALPHA}_OP_V:
    /// OGL_FUNC_* (0x8006..) and D3D_* (1..5).
    pub fn decode(v: u32) -> Self {
        use BlendOperation::*;
        match v {
            0x8006 | 1 => Add,
            0x800a | 2 => Subtract,
            0x800b | 3 => ReverseSubtract,
            0x8007 | 4 => Min,
            0x8008 | 5 => Max,
            _ => Unknown(v),
        }
    }
}

/// wgpu::CompareFunction names.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CompareFunction {
    Never,
    Less,
    Equal,
    LessEqual,
    Greater,
    NotEqual,
    GreaterEqual,
    Always,
    Unknown(u32),
}

impl CompareFunction {
    /// SET_DEPTH_FUNC_V, SET_STENCIL_FUNC_V, SET_BACK_STENCIL_FUNC_V:
    /// OGL_* (0x200..0x207) and D3D_* (1..8), both in the order
    /// NEVER, LESS, EQUAL, LEQUAL, GREATER, NOTEQUAL, GEQUAL, ALWAYS.
    pub fn decode(v: u32) -> Self {
        use CompareFunction::*;
        const ORDER: [CompareFunction; 8] = [Never, Less, Equal, LessEqual, Greater, NotEqual, GreaterEqual, Always];
        match v {
            0x200..=0x207 => ORDER[(v - 0x200) as usize],
            1..=8 => ORDER[(v - 1) as usize],
            _ => Unknown(v),
        }
    }
}

/// wgpu::StencilOperation names.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum StencilOperation {
    Keep,
    Zero,
    Replace,
    Invert,
    IncrementClamp,
    DecrementClamp,
    IncrementWrap,
    DecrementWrap,
    Unknown(u32),
}

impl StencilOperation {
    /// SET_{,BACK_}STENCIL_OP_{FAIL,ZFAIL,ZPASS}_V: OGL_* and D3D_* (1..8).
    /// INCRSAT/DECRSAT clamp, INCR/DECR wrap (NVK maps VK_STENCIL_OP_INCREMENT_AND_WRAP to D3D_INCR).
    pub fn decode(v: u32) -> Self {
        use StencilOperation::*;
        match v {
            0x1e00 | 1 => Keep,
            0x0000 | 2 => Zero,
            0x1e01 | 3 => Replace,
            0x1e02 | 4 => IncrementClamp,
            0x1e03 | 5 => DecrementClamp,
            0x150a | 6 => Invert,
            0x8507 | 7 => IncrementWrap,
            0x8508 | 8 => DecrementWrap,
            _ => Unknown(v),
        }
    }
}

/// wgpu::Face names (cull face), plus FRONT_AND_BACK which wgpu cannot
/// express (it culls every polygon: skip the draw's triangles).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Face {
    Front,
    Back,
    FrontAndBack,
    Unknown(u32),
}

impl Face {
    /// OGL_SET_CULL_FACE_V: FRONT 0x404, BACK 0x405, FRONT_AND_BACK 0x408.
    pub fn decode(v: u32) -> Self {
        match v {
            0x404 => Face::Front,
            0x405 => Face::Back,
            0x408 => Face::FrontAndBack,
            _ => Face::Unknown(v),
        }
    }
}

/// wgpu::FrontFace names.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FrontFace {
    Ccw,
    Cw,
    Unknown(u32),
}

impl FrontFace {
    /// OGL_SET_FRONT_FACE_V: CW 0x900, CCW 0x901.
    pub fn decode(v: u32) -> Self {
        match v {
            0x900 => FrontFace::Cw,
            0x901 => FrontFace::Ccw,
            _ => FrontFace::Unknown(v),
        }
    }
}

/// wgpu::PolygonMode names.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PolygonMode {
    Fill,
    Line,
    Point,
    Unknown(u32),
}

impl PolygonMode {
    /// SET_{FRONT,BACK}_POLYGON_MODE_V: POINT 0x1B00, LINE 0x1B01, FILL 0x1B02.
    pub fn decode(v: u32) -> Self {
        match v {
            0x1b00 => PolygonMode::Point,
            0x1b01 => PolygonMode::Line,
            0x1b02 => PolygonMode::Fill,
            _ => PolygonMode::Unknown(v),
        }
    }
}

/// SET_VIEWPORT_CLIP_CONTROL_GEOMETRY_CLIP values.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum GeometryClip {
    WzeroClip,
    Passthru,
    FrustumXyClip,
    FrustumXyzClip,
    WzeroClipNoZCull,
    FrustumZClip,
    WzeroTriFillOrClip,
    Unknown(u32),
}

/// SET_VIEWPORT_CLIP_CONTROL_Z_CLIP_RANGE values.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ZClipRange {
    /// USE_FIELD_MIN_Z_ZERO_MAX_Z_ONE
    UseFieldMinZZeroMaxZOne,
    /// MIN_Z_MAX_Z (SET_VIEWPORT_CLIP_MIN_Z/MAX_Z)
    MinZMaxZ,
    ZeroOne,
    MinusInfPlusInf,
}

/// A rectangle in window pixels.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Rect {
    pub x: u32,
    pub y: u32,
    pub width: u32,
    pub height: u32,
}

impl Rect {
    /// Two methods packing (origin 15:0, size 31:16), as
    /// SET_SURFACE_CLIP_HORIZONTAL/VERTICAL and SET_VIEWPORT_CLIP_HORIZONTAL/VERTICAL.
    fn packed(h: u32, v: u32) -> Self {
        Rect { x: h & 0xffff, width: h >> 16, y: v & 0xffff, height: v >> 16 }
    }
}

/// Viewport 0 and the window origin.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Viewport {
    /// SET_VIEWPORT_SCALE_X/Y/Z(0)_V (f32): window = offset + scale * NDC.
    pub scale: [f32; 3],
    /// SET_VIEWPORT_OFFSET_X/Y/Z(0)_V (f32).
    pub offset: [f32; 3],
    /// SET_VIEWPORT_SCALE_OFFSET_ENABLE: the viewport transform applies.
    pub transform_enable: bool,
    /// SET_VIEWPORT_Z_CLIP_RANGE: true = ZERO_TO_POSITIVE_W (NDC z in [0, 1]),
    /// false = NEGATIVE_W_TO_POSITIVE_W (NDC z in [-1, 1]).
    pub z_zero_to_one: bool,
    /// SET_VIEWPORT_CLIP_HORIZONTAL(0) X0/WIDTH and SET_VIEWPORT_CLIP_VERTICAL(0) Y0/HEIGHT.
    pub clip: Rect,
    /// SET_VIEWPORT_CLIP_MIN_Z(0)_V (f32): depth range clamp, low end.
    pub clip_min_z: f32,
    /// SET_VIEWPORT_CLIP_MAX_Z(0)_V (f32): depth range clamp, high end.
    pub clip_max_z: f32,
    /// SET_VIEWPORT_COORDINATE_SWIZZLE(0), raw. 0x6420 is identity
    /// (X_POS_X, Y_POS_Y, Z_POS_Z, W_POS_W); 0 means never written.
    pub swizzle: u32,
    /// SET_WINDOW_ORIGIN_MODE: true = LOWER_LEFT, false = UPPER_LEFT.
    pub origin_lower_left: bool,
    /// SET_WINDOW_ORIGIN_FLIP_Y. NVK always writes FALSE; reference
    /// emulators running NVN games invert the front face when it is set and
    /// leave the viewport alone, as `render_state` and `to_wgpu` do.
    pub flip_y: bool,
}

/// What to pass to `wgpu::RenderPass::set_viewport`, plus the vertex-shader
/// fix-ups wgpu needs to reproduce the hardware transform.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct WgpuViewport {
    pub x: f32,
    pub y: f32,
    pub width: f32,
    pub height: f32,
    pub min_depth: f32,
    pub max_depth: f32,
    /// The vertex shader must output -y. The hardware maps NDC +y to the
    /// bottom here (positive scale y, upper-left origin) and wgpu needs a
    /// positive height; with the flip, framebuffer positions are the
    /// hardware's, so the front face needs no change.
    pub negate_y: bool,
    /// The vertex shader must remap z to (z + w) / 2: NDC z is in [-1, 1].
    pub remap_z: bool,
}

impl Viewport {
    /// wgpu viewport for this transform. With an upper-left origin, window y
    /// grows downward like wgpu framebuffer y, so the window rectangle is
    /// x = offset - |scale|, size = 2 |scale| on each axis. Depth range:
    /// [offset_z, offset_z + scale_z] for z in [0, 1], and
    /// [offset_z - scale_z, offset_z + scale_z] for z in [-1, 1].
    pub fn to_wgpu(&self) -> Result<WgpuViewport, &'static str> {
        if !self.transform_enable {
            return Err("SET_VIEWPORT_SCALE_OFFSET disabled: positions are in window space");
        }
        if self.origin_lower_left {
            return Err("SET_WINDOW_ORIGIN lower-left: the window-to-surface y flip is not established");
        }
        if self.swizzle != 0 && self.swizzle != 0x6420 {
            return Err("SET_VIEWPORT_COORDINATE_SWIZZLE is not identity");
        }
        let [sx, sy, sz] = self.scale;
        let [ox, oy, oz] = self.offset;
        if sx < 0.0 {
            return Err("negative SET_VIEWPORT_SCALE_X (x mirror)");
        }
        let (min_depth, max_depth) = if self.z_zero_to_one { (oz, oz + sz) } else { (oz - sz, oz + sz) };
        if !(0.0 <= min_depth && min_depth <= max_depth && max_depth <= 1.0) {
            return Err("depth range reversed or outside [0, 1]");
        }
        Ok(WgpuViewport {
            x: ox - sx,
            y: oy - sy.abs(),
            width: 2.0 * sx,
            height: 2.0 * sy.abs(),
            min_depth,
            max_depth,
            negate_y: sy > 0.0,
            remap_z: !self.z_zero_to_one,
        })
    }
}

/// Scissor 0. Bounds are exclusive maxima in window pixels.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Scissor {
    /// SET_SCISSOR_ENABLE(0)_V.
    pub enable: bool,
    /// SET_SCISSOR_HORIZONTAL(0)_XMIN.
    pub xmin: u32,
    /// SET_SCISSOR_HORIZONTAL(0)_XMAX.
    pub xmax: u32,
    /// SET_SCISSOR_VERTICAL(0)_YMIN.
    pub ymin: u32,
    /// SET_SCISSOR_VERTICAL(0)_YMAX.
    pub ymax: u32,
}

/// One blend equation, with wgpu::BlendComponent field names.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct BlendComponent {
    /// SET_BLEND{,_PER_TARGET}_{COLOR,ALPHA}_SOURCE_COEFF_V.
    pub src_factor: BlendFactor,
    /// SET_BLEND{,_PER_TARGET}_{COLOR,ALPHA}_DEST_COEFF_V.
    pub dst_factor: BlendFactor,
    /// SET_BLEND{,_PER_TARGET}_{COLOR,ALPHA}_OP_V.
    pub operation: BlendOperation,
}

/// Blend state of one color target (blending enabled).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Blend {
    /// SET_BLEND{,_PER_TARGET}_SEPARATE_FOR_ALPHA_ENABLE.
    pub separate_alpha: bool,
    /// Color equation.
    pub color: BlendComponent,
    /// Alpha equation as programmed in the ALPHA_* methods.
    pub alpha: BlendComponent,
}

impl Blend {
    /// The alpha equation to give wgpu: `alpha` when SEPARATE_FOR_ALPHA is
    /// set, otherwise `color` (the GL meaning of a non-separate blend,
    /// inferred from the method name: NVK always sets it TRUE).
    pub fn wgpu_alpha(&self) -> BlendComponent {
        if self.separate_alpha { self.alpha } else { self.color }
    }
}

/// One fragment-shader color output.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ColorOutput {
    /// Shader output location: the slot index in SET_CT_SELECT.
    pub location: u32,
    /// SET_CT_SELECT_TARGETn: the color target (SET_COLOR_TARGET_*(target))
    /// written by this location. Write mask and blend are indexed by this
    /// target, as NVK programs them per attachment.
    pub target: u32,
    /// wgpu::ColorWrites bits (RED 1, GREEN 2, BLUE 4, ALPHA 8) from
    /// SET_CT_WRITE(target)_{R,G,B,A}_ENABLE, or SET_CT_WRITE(0) when
    /// SET_SINGLE_CT_WRITE_CONTROL_ENABLE is set (inferred from the name;
    /// NVK sets it only when every target has the same mask).
    pub write_mask: u8,
    /// None when SET_BLEND(target)_ENABLE is false (wgpu `blend: None`).
    /// Equation from SET_BLEND_PER_TARGET_*(target) when
    /// SET_BLEND_STATE_PER_TARGET_ENABLE is set, else from the global SET_BLEND_*.
    pub blend: Option<Blend>,
}

/// Stencil state of one face.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct StencilFace {
    /// SET_{,BACK_}STENCIL_OP_FAIL_V.
    pub fail_op: StencilOperation,
    /// SET_{,BACK_}STENCIL_OP_ZFAIL_V.
    pub depth_fail_op: StencilOperation,
    /// SET_{,BACK_}STENCIL_OP_ZPASS_V.
    pub pass_op: StencilOperation,
    /// SET_{,BACK_}STENCIL_FUNC_V.
    pub compare: CompareFunction,
    /// SET_{,BACK_}STENCIL_FUNC_REF_V.
    pub reference: u8,
    /// SET_{,BACK_}STENCIL_FUNC_MASK_V (wgpu read_mask).
    pub read_mask: u8,
    /// SET_{,BACK_}STENCIL_MASK_V (wgpu write_mask).
    pub write_mask: u8,
}

/// Depth and stencil tests. wgpu has one reference, read mask and write
/// mask for both faces: if front and back differ there, wgpu cannot express it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct DepthStencil {
    /// SET_DEPTH_TEST_ENABLE.
    pub depth_test: bool,
    /// SET_DEPTH_WRITE_ENABLE.
    pub depth_write: bool,
    /// SET_DEPTH_FUNC_V (wgpu depth_compare when depth_test, else Always).
    pub depth_compare: CompareFunction,
    /// SET_STENCIL_TEST_ENABLE.
    pub stencil_test: bool,
    /// SET_TWO_SIDED_STENCIL_TEST_ENABLE. When false the back-face methods
    /// presumably do not apply (EXT_stencil_two_side meaning, not established).
    pub two_sided: bool,
    /// SET_STENCIL_* (front face).
    pub front: StencilFace,
    /// SET_BACK_STENCIL_* (back face), as programmed.
    pub back: StencilFace,
}

/// Depth bias. wgpu::DepthBiasState has no enable: give it zeros when the
/// enable for the polygon mode is off.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct DepthBias {
    /// SET_POLY_OFFSET_POINT_ENABLE.
    pub point: bool,
    /// SET_POLY_OFFSET_LINE_ENABLE.
    pub line: bool,
    /// SET_POLY_OFFSET_FILL_ENABLE.
    pub fill: bool,
    /// SET_DEPTH_BIAS_V (f32), in minimum resolvable depth units. NVK writes
    /// the Vulkan constant factor unchanged and notes the NVIDIA driver
    /// doubles it: whether NVN's value is 2x wgpu's i32 `constant` is unknown.
    pub constant: f32,
    /// SET_SLOPE_SCALE_DEPTH_BIAS_V (f32), wgpu slope_scale.
    pub slope_scale: f32,
    /// SET_DEPTH_BIAS_CLAMP_V (f32), wgpu clamp.
    pub clamp: f32,
    /// SET_DEPTH_BIAS_CONTROL_DEPTH_FORMAT_DEPENDENT.
    pub format_dependent: bool,
}

/// Depth clipping and clamping (SET_VIEWPORT_CLIP_CONTROL).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ClipControl {
    /// SET_VIEWPORT_CLIP_CONTROL_MIN_Z_ZERO_MAX_Z_ONE.
    pub min_z_zero_max_z_one: bool,
    /// SET_VIEWPORT_CLIP_CONTROL_Z_CLIP_RANGE.
    pub z_clip_range: ZClipRange,
    /// SET_VIEWPORT_CLIP_CONTROL_PIXEL_MIN_Z: true = CLAMP, false = CLIP.
    pub pixel_min_z_clamp: bool,
    /// SET_VIEWPORT_CLIP_CONTROL_PIXEL_MAX_Z: true = CLAMP, false = CLIP.
    pub pixel_max_z_clamp: bool,
    /// SET_VIEWPORT_CLIP_CONTROL_GEOMETRY_CLIP.
    pub geometry_clip: GeometryClip,
}

impl ClipControl {
    /// Whether primitives are clipped in z (wgpu `unclipped_depth` is the
    /// negation). Known only for NVK's two settings: FRUSTUM_XYZ_CLIP (or
    /// FRUSTUM_Z_CLIP) clips; FRUSTUM_XY_CLIP with both pixel z clamps does not.
    pub fn depth_clip(&self) -> Option<bool> {
        match self.geometry_clip {
            GeometryClip::FrustumXyzClip | GeometryClip::FrustumZClip => Some(true),
            GeometryClip::FrustumXyClip if self.pixel_min_z_clamp && self.pixel_max_z_clamp => Some(false),
            _ => None,
        }
    }
}

/// Rasterizer state.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Raster {
    /// OGL_SET_CULL_ENABLE (wgpu cull_mode None when false).
    pub cull_enable: bool,
    /// OGL_SET_CULL_FACE_V.
    pub cull_face: Face,
    /// OGL_SET_FRONT_FACE_V, inverted when SET_WINDOW_ORIGIN_FLIP_Y is set:
    /// then wgpu's meaning with an upper-left origin (NVK passes the Vulkan
    /// winding through with FLIP_Y false).
    pub front_face: FrontFace,
    /// SET_FRONT_POLYGON_MODE_V.
    pub front_polygon_mode: PolygonMode,
    /// SET_BACK_POLYGON_MODE_V (wgpu has one mode for both faces).
    pub back_polygon_mode: PolygonMode,
    pub depth_bias: DepthBias,
    pub clip: ClipControl,
    /// SET_ANTI_ALIAS_ALPHA_CONTROL_ALPHA_TO_COVERAGE (wgpu alpha_to_coverage_enabled).
    pub alpha_to_coverage: bool,
    /// SET_ANTI_ALIAS_ALPHA_CONTROL_ALPHA_TO_ONE (no wgpu equivalent).
    pub alpha_to_one: bool,
}

/// The fixed-function state of one draw.
#[derive(Clone, Debug, PartialEq)]
pub struct RenderState {
    pub viewport: Viewport,
    pub scissor: Scissor,
    /// SET_SURFACE_CLIP_HORIZONTAL X/WIDTH and SET_SURFACE_CLIP_VERTICAL Y/HEIGHT:
    /// the render area; wgpu scissors must stay inside it.
    pub surface_clip: Rect,
    /// SET_CT_SELECT_TARGET_COUNT (raw, 4 bits; at most 8 outputs are decoded).
    pub ct_count: u32,
    /// SET_BLEND_STATE_PER_TARGET_ENABLE.
    pub blend_per_target: bool,
    /// SET_SINGLE_CT_WRITE_CONTROL_ENABLE.
    pub single_ct_write: bool,
    /// One entry per selected output location, in location order.
    pub outputs: Vec<ColorOutput>,
    /// SET_BLEND_CONST_RED/GREEN/BLUE/ALPHA_V (f32): wgpu set_blend_constant.
    pub blend_constant: [f32; 4],
    pub depth_stencil: DepthStencil,
    pub raster: Raster,
    /// SET_USER_CLIP_ENABLE: bit i enables the vertex program's clip distance i.
    pub user_clip: u8,
}

impl RenderState {
    /// `wgpu::RenderPass::set_scissor_rect(x, y, width, height)`: scissor 0
    /// (if enabled) intersected with the surface clip.
    pub fn wgpu_scissor(&self) -> Result<[u32; 4], &'static str> {
        if self.viewport.origin_lower_left {
            return Err("SET_WINDOW_ORIGIN lower-left: the window-to-surface y flip is not established");
        }
        let c = self.surface_clip;
        let (mut x0, mut y0, mut x1, mut y1) = (c.x, c.y, c.x + c.width, c.y + c.height);
        let s = self.scissor;
        if s.enable {
            (x0, y0, x1, y1) = (x0.max(s.xmin), y0.max(s.ymin), x1.min(s.xmax), y1.min(s.ymax));
        }
        Ok([x0, y0, x1.saturating_sub(x0), y1.saturating_sub(y0)])
    }
}

impl BlendFactor {
    /// The wgpu factor; None for the factors wgpu lacks.
    pub fn wgpu(self) -> Option<wgpu::BlendFactor> {
        use wgpu::BlendFactor as W;
        Some(match self {
            Self::Zero => W::Zero,
            Self::One => W::One,
            Self::Src => W::Src,
            Self::OneMinusSrc => W::OneMinusSrc,
            Self::SrcAlpha => W::SrcAlpha,
            Self::OneMinusSrcAlpha => W::OneMinusSrcAlpha,
            Self::Dst => W::Dst,
            Self::OneMinusDst => W::OneMinusDst,
            Self::DstAlpha => W::DstAlpha,
            Self::OneMinusDstAlpha => W::OneMinusDstAlpha,
            Self::SrcAlphaSaturated => W::SrcAlphaSaturated,
            Self::Constant => W::Constant,
            Self::OneMinusConstant => W::OneMinusConstant,
            Self::Src1 => W::Src1,
            Self::OneMinusSrc1 => W::OneMinusSrc1,
            Self::Src1Alpha => W::Src1Alpha,
            Self::OneMinusSrc1Alpha => W::OneMinusSrc1Alpha,
            _ => return None,
        })
    }
}

impl BlendOperation {
    pub fn wgpu(self) -> Option<wgpu::BlendOperation> {
        use wgpu::BlendOperation as W;
        Some(match self {
            Self::Add => W::Add,
            Self::Subtract => W::Subtract,
            Self::ReverseSubtract => W::ReverseSubtract,
            Self::Min => W::Min,
            Self::Max => W::Max,
            Self::Unknown(_) => return None,
        })
    }
}

impl BlendComponent {
    pub fn wgpu(self) -> Option<wgpu::BlendComponent> {
        Some(wgpu::BlendComponent { src_factor: self.src_factor.wgpu()?, dst_factor: self.dst_factor.wgpu()?, operation: self.operation.wgpu()? })
    }
}

impl CompareFunction {
    pub fn wgpu(self) -> Option<wgpu::CompareFunction> {
        use wgpu::CompareFunction as W;
        Some(match self {
            Self::Never => W::Never,
            Self::Less => W::Less,
            Self::Equal => W::Equal,
            Self::LessEqual => W::LessEqual,
            Self::Greater => W::Greater,
            Self::NotEqual => W::NotEqual,
            Self::GreaterEqual => W::GreaterEqual,
            Self::Always => W::Always,
            Self::Unknown(_) => return None,
        })
    }
}

impl StencilOperation {
    pub fn wgpu(self) -> Option<wgpu::StencilOperation> {
        use wgpu::StencilOperation as W;
        Some(match self {
            Self::Keep => W::Keep,
            Self::Zero => W::Zero,
            Self::Replace => W::Replace,
            Self::Invert => W::Invert,
            Self::IncrementClamp => W::IncrementClamp,
            Self::DecrementClamp => W::DecrementClamp,
            Self::IncrementWrap => W::IncrementWrap,
            Self::DecrementWrap => W::DecrementWrap,
            Self::Unknown(_) => return None,
        })
    }
}

impl StencilFace {
    pub fn wgpu(self) -> Option<wgpu::StencilFaceState> {
        Some(wgpu::StencilFaceState { compare: self.compare.wgpu()?, fail_op: self.fail_op.wgpu()?, depth_fail_op: self.depth_fail_op.wgpu()?, pass_op: self.pass_op.wgpu()? })
    }
}

impl Raster {
    /// wgpu cull mode: Ok(None) when culling is off; Err when wgpu cannot
    /// express it (FRONT_AND_BACK culls every triangle).
    pub fn wgpu_cull(&self) -> Result<Option<wgpu::Face>, &'static str> {
        if !self.cull_enable {
            return Ok(None);
        }
        match self.cull_face {
            Face::Front => Ok(Some(wgpu::Face::Front)),
            Face::Back => Ok(Some(wgpu::Face::Back)),
            Face::FrontAndBack => Err("cull FRONT_AND_BACK"),
            Face::Unknown(_) => Err("unknown cull face"),
        }
    }

    pub fn wgpu_front_face(&self) -> Option<wgpu::FrontFace> {
        match self.front_face {
            FrontFace::Ccw => Some(wgpu::FrontFace::Ccw),
            FrontFace::Cw => Some(wgpu::FrontFace::Cw),
            FrontFace::Unknown(_) => None,
        }
    }
}

fn stencil_face<R: Fn(u32) -> u32 + ?Sized>(r: &R, ops: u32, func: u32, reference: u32, read_mask: u32, write_mask: u32) -> StencilFace {
    StencilFace {
        fail_op: StencilOperation::decode(r(ops)),
        depth_fail_op: StencilOperation::decode(r(ops + 4)),
        pass_op: StencilOperation::decode(r(ops + 8)),
        compare: CompareFunction::decode(r(func)),
        reference: r(reference) as u8,
        read_mask: r(read_mask) as u8,
        write_mask: r(write_mask) as u8,
    }
}

fn blend_component<R: Fn(u32) -> u32 + ?Sized>(r: &R, op: u32, src: u32, dst: u32) -> BlendComponent {
    BlendComponent {
        src_factor: BlendFactor::decode(r(src)),
        dst_factor: BlendFactor::decode(r(dst)),
        operation: BlendOperation::decode(r(op)),
    }
}

/// Decodes the fixed-function state of the current draw.
pub fn render_state<R: Fn(u32) -> u32 + ?Sized>(r: &R) -> RenderState {
    let origin = r(0x13ac); // SET_WINDOW_ORIGIN
    let viewport = Viewport {
        scale: [float(r, 0x0a00), float(r, 0x0a04), float(r, 0x0a08)],
        offset: [float(r, 0x0a0c), float(r, 0x0a10), float(r, 0x0a14)],
        transform_enable: bit(r(0x192c), 0),
        z_zero_to_one: bit(r(0x0d7c), 0),
        clip: Rect::packed(r(0x0c00), r(0x0c04)),
        clip_min_z: float(r, 0x0c08),
        clip_max_z: float(r, 0x0c0c),
        swizzle: r(0x0a18),
        origin_lower_left: bit(origin, 0),
        flip_y: bit(origin, 4),
    };
    let (h, v) = (r(0x0e04), r(0x0e08));
    let scissor = Scissor { enable: bit(r(0x0e00), 0), xmin: h & 0xffff, xmax: h >> 16, ymin: v & 0xffff, ymax: v >> 16 };

    let ct = r(0x121c); // SET_CT_SELECT
    let blend_per_target = bit(r(0x12e4), 0);
    let single_ct_write = bit(r(0x0f90), 0);
    let outputs = (0..(ct & 0xf).min(8))
        .map(|location| {
            let target = ct >> (4 + 3 * location) & 7;
            let w = r(0x1a00 + 4 * if single_ct_write { 0 } else { target }); // SET_CT_WRITE
            let write_mask = (w & 1 | w >> 3 & 2 | w >> 6 & 4 | w >> 9 & 8) as u8;
            // SEPARATE_FOR_ALPHA, COLOR_OP, COLOR_SOURCE, COLOR_DEST, ALPHA_OP, ALPHA_SOURCE, ALPHA_DEST
            let m: [u32; 7] = if blend_per_target {
                let b = 0x1e00 + 32 * target;
                [b, b + 4, b + 8, b + 0xc, b + 0x10, b + 0x14, b + 0x18]
            } else {
                [0x133c, 0x1340, 0x1344, 0x1348, 0x134c, 0x1350, 0x1358]
            };
            let blend = bit(r(0x1360 + 4 * target), 0).then(|| Blend {
                separate_alpha: bit(r(m[0]), 0),
                color: blend_component(r, m[1], m[2], m[3]),
                alpha: blend_component(r, m[4], m[5], m[6]),
            });
            ColorOutput { location, target, write_mask, blend }
        })
        .collect();

    let depth_stencil = DepthStencil {
        depth_test: bit(r(0x12cc), 0),
        depth_write: bit(r(0x12e8), 0),
        depth_compare: CompareFunction::decode(r(0x130c)),
        stencil_test: bit(r(0x1380), 0),
        two_sided: bit(r(0x1594), 0),
        front: stencil_face(r, 0x1384, 0x1390, 0x1394, 0x1398, 0x139c),
        back: stencil_face(r, 0x1598, 0x15a4, 0x0f54, 0x0f5c, 0x0f58),
    };

    let cc = r(0x193c); // SET_VIEWPORT_CLIP_CONTROL
    let geometry_clip = match cc >> 11 & 7 {
        0 => GeometryClip::WzeroClip,
        1 => GeometryClip::Passthru,
        2 => GeometryClip::FrustumXyClip,
        3 => GeometryClip::FrustumXyzClip,
        4 => GeometryClip::WzeroClipNoZCull,
        5 => GeometryClip::FrustumZClip,
        6 => GeometryClip::WzeroTriFillOrClip,
        g => GeometryClip::Unknown(g),
    };
    let z_clip_range = match cc >> 16 & 3 {
        0 => ZClipRange::UseFieldMinZZeroMaxZOne,
        1 => ZClipRange::MinZMaxZ,
        2 => ZClipRange::ZeroOne,
        _ => ZClipRange::MinusInfPlusInf,
    };
    let aa = r(0x153c); // SET_ANTI_ALIAS_ALPHA_CONTROL
    let raster = Raster {
        cull_enable: bit(r(0x1918), 0),
        cull_face: Face::decode(r(0x1920)),
        front_face: match FrontFace::decode(r(0x191c)) {
            FrontFace::Cw if bit(origin, 4) => FrontFace::Ccw,
            FrontFace::Ccw if bit(origin, 4) => FrontFace::Cw,
            f => f,
        },
        front_polygon_mode: PolygonMode::decode(r(0x0dac)),
        back_polygon_mode: PolygonMode::decode(r(0x0db0)),
        depth_bias: DepthBias {
            point: bit(r(0x0dc0), 0),
            line: bit(r(0x0dc4), 0),
            fill: bit(r(0x0dc8), 0),
            constant: float(r, 0x15bc),
            slope_scale: float(r, 0x156c),
            clamp: float(r, 0x187c),
            format_dependent: bit(r(0x1110), 0),
        },
        clip: ClipControl {
            min_z_zero_max_z_one: bit(cc, 0),
            z_clip_range,
            pixel_min_z_clamp: bit(cc, 3),
            pixel_max_z_clamp: bit(cc, 4),
            geometry_clip,
        },
        alpha_to_coverage: bit(aa, 0),
        alpha_to_one: bit(aa, 4),
    };

    RenderState {
        viewport,
        scissor,
        surface_clip: Rect::packed(r(0x0ff4), r(0x0ff8)),
        ct_count: ct & 0xf,
        blend_per_target,
        single_ct_write,
        outputs,
        blend_constant: [float(r, 0x131c), float(r, 0x1320), float(r, 0x1324), float(r, 0x1328)],
        depth_stencil,
        raster,
        user_clip: r(0x1510) as u8,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    fn decode(regs: &[(u32, u32)]) -> RenderState {
        let m: HashMap<u32, u32> = regs.iter().copied().collect();
        render_state(&|a| m.get(&a).copied().unwrap_or(0))
    }

    fn fb(x: f32) -> u32 {
        x.to_bits()
    }

    #[test]
    fn viewport_d3d_style() {
        // 1600x900, NDC y up mapped to window y down (negative scale y), z in [0, 1].
        let s = decode(&[
            (0x0a00, fb(800.0)), (0x0a04, fb(-450.0)), (0x0a08, fb(1.0)),
            (0x0a0c, fb(800.0)), (0x0a10, fb(450.0)), (0x0a14, fb(0.0)),
            (0x192c, 1), (0x0d7c, 1), (0x13ac, 0x10),
            (0x0c00, 0x0640_0000), (0x0c04, 0x0384_0000), (0x0c08, fb(0.0)), (0x0c0c, fb(1.0)),
        ]);
        let v = s.viewport;
        assert_eq!((v.scale, v.offset), ([800.0, -450.0, 1.0], [800.0, 450.0, 0.0]));
        assert!(v.transform_enable && v.z_zero_to_one && !v.origin_lower_left && v.flip_y);
        assert_eq!(v.clip, Rect { x: 0, y: 0, width: 1600, height: 900 });
        assert_eq!((v.clip_min_z, v.clip_max_z), (0.0, 1.0));
        let w = v.to_wgpu().unwrap();
        assert_eq!(
            w,
            WgpuViewport { x: 0.0, y: 0.0, width: 1600.0, height: 900.0, min_depth: 0.0, max_depth: 1.0, negate_y: false, remap_z: false }
        );
    }

    #[test]
    fn viewport_gl_style_needs_shader_fixups() {
        // Positive scale y with an upper-left origin, z in [-1, 1], depth range [0.25, 0.75].
        let s = decode(&[
            (0x0a00, fb(100.0)), (0x0a04, fb(50.0)), (0x0a08, fb(0.25)),
            (0x0a0c, fb(110.0)), (0x0a10, fb(60.0)), (0x0a14, fb(0.5)),
            (0x192c, 1), (0x0d7c, 0), (0x0a18, 0x6420),
        ]);
        let w = s.viewport.to_wgpu().unwrap();
        assert_eq!(
            w,
            WgpuViewport { x: 10.0, y: 10.0, width: 200.0, height: 100.0, min_depth: 0.25, max_depth: 0.75, negate_y: true, remap_z: true }
        );
    }

    #[test]
    fn viewport_unsupported_cases_are_errors() {
        let base = [(0x0a00, fb(1.0)), (0x0a04, fb(1.0)), (0x0a08, fb(1.0)), (0x192c, 1), (0x0d7c, 1)];
        assert!(decode(&base).viewport.to_wgpu().is_ok());
        let with = |m: u32, v: u32| {
            let mut regs = base.to_vec();
            regs.push((m, v));
            decode(&regs).viewport.to_wgpu()
        };
        assert!(with(0x192c, 0).is_err()); // transform disabled
        assert!(with(0x13ac, 1).is_err()); // lower-left origin
        assert!(with(0x0a18, 0x6430).is_err()); // Y_NEG_Y swizzle
        assert!(with(0x0a00, fb(-1.0)).is_err()); // x mirror
        assert!(with(0x0a08, fb(-1.0)).is_err()); // reversed depth
    }

    #[test]
    fn scissor_and_surface_clip() {
        let regs = [(0x0e00, 1), (0x0e04, 300 << 16 | 10), (0x0e08, 0xffff << 16 | 20), (0x0ff4, 0x0640_0000), (0x0ff8, 0x0384_0000)];
        let s = decode(&regs);
        assert_eq!(s.scissor, Scissor { enable: true, xmin: 10, xmax: 300, ymin: 20, ymax: 0xffff });
        assert_eq!(s.surface_clip, Rect { x: 0, y: 0, width: 1600, height: 900 });
        assert_eq!(s.wgpu_scissor(), Ok([10, 20, 290, 880]));
        let mut off = regs.to_vec();
        off[0] = (0x0e00, 0);
        assert_eq!(decode(&off).wgpu_scissor(), Ok([0, 0, 1600, 900]));
        off.push((0x13ac, 1));
        assert!(decode(&off).wgpu_scissor().is_err());
    }

    #[test]
    fn ct_select_and_write_masks() {
        // 3 outputs: location 0 -> target 2, 1 -> 0, 2 -> 1.
        let regs = [(0x121c, 3 | 2 << 4 | 1 << 10), (0x1a08, 0x0001), (0x1a00, 0x1010), (0x1a04, 0x0100)];
        let s = decode(&regs);
        assert_eq!(s.ct_count, 3);
        let got: Vec<_> = s.outputs.iter().map(|o| (o.location, o.target, o.write_mask, o.blend)).collect();
        assert_eq!(got, vec![(0, 2, 1, None), (1, 0, 2 | 8, None), (2, 1, 4, None)]);
        // SET_SINGLE_CT_WRITE_CONTROL: every target uses SET_CT_WRITE(0).
        let mut single = regs.to_vec();
        single.push((0x0f90, 1));
        let s = decode(&single);
        assert!(s.single_ct_write && s.outputs.iter().all(|o| o.write_mask == 0xa));
        // SET_CT_SELECT count above 8 is clamped.
        assert_eq!(decode(&[(0x121c, 0xf)]).outputs.len(), 8);
    }

    #[test]
    fn blend_per_target_and_global() {
        let eq = |src, dst, operation| BlendComponent { src_factor: src, dst_factor: dst, operation };
        // Per-target state for target 1 (location 0 -> target 1).
        let s = decode(&[
            (0x121c, 1 | 1 << 4), (0x12e4, 1), (0x1364, 1),
            (0x1e20, 1), (0x1e24, 1), (0x1e28, 5), (0x1e2c, 6),
            (0x1e30, 0x800b), (0x1e34, 0x4001), (0x1e38, 0xc004),
            (0x131c, fb(0.5)), (0x1320, fb(0.25)), (0x1324, fb(1.0)), (0x1328, fb(0.0)),
        ]);
        let b = s.outputs[0].blend.unwrap();
        assert!(s.blend_per_target && b.separate_alpha);
        assert_eq!(b.color, eq(BlendFactor::SrcAlpha, BlendFactor::OneMinusSrcAlpha, BlendOperation::Add));
        assert_eq!(b.alpha, eq(BlendFactor::One, BlendFactor::OneMinusConstantAlpha, BlendOperation::ReverseSubtract));
        assert_eq!(b.wgpu_alpha(), b.alpha);
        assert_eq!(s.blend_constant, [0.5, 0.25, 1.0, 0.0]);
        // Global state (per-target disabled), non-separate alpha.
        let s = decode(&[
            (0x121c, 1), (0x12e4, 0), (0x1360, 1),
            (0x133c, 0), (0x1340, 0x800a), (0x1344, 0x4306), (0x1348, 0xc001),
            (0x134c, 4), (0x1350, 2), (0x1358, 1),
            (0x1e04, 5), // per-target method, must be ignored
        ]);
        let b = s.outputs[0].blend.unwrap();
        assert_eq!(b.color, eq(BlendFactor::Dst, BlendFactor::Constant, BlendOperation::Subtract));
        assert_eq!(b.alpha, eq(BlendFactor::One, BlendFactor::Zero, BlendOperation::Min));
        assert_eq!(b.wgpu_alpha(), b.color);
        // SET_BLEND(0) false: no blending.
        assert_eq!(decode(&[(0x121c, 1), (0x1340, 1)]).outputs[0].blend, None);
    }

    #[test]
    fn enum_tables() {
        use BlendFactor as F;
        let ogl = [
            (0x4000, F::Zero), (0x4001, F::One), (0x4300, F::Src), (0x4301, F::OneMinusSrc),
            (0x4302, F::SrcAlpha), (0x4303, F::OneMinusSrcAlpha), (0x4304, F::DstAlpha), (0x4305, F::OneMinusDstAlpha),
            (0x4306, F::Dst), (0x4307, F::OneMinusDst), (0x4308, F::SrcAlphaSaturated), (0xc001, F::Constant),
            (0xc002, F::OneMinusConstant), (0xc003, F::ConstantAlpha), (0xc004, F::OneMinusConstantAlpha),
            (0xc900, F::Src1), (0xc901, F::OneMinusSrc1), (0xc902, F::Src1Alpha), (0xc903, F::OneMinusSrc1Alpha),
        ];
        let d3d = [
            F::Zero, F::One, F::Src, F::OneMinusSrc, F::SrcAlpha, F::OneMinusSrcAlpha, F::DstAlpha, F::OneMinusDstAlpha,
            F::Dst, F::OneMinusDst, F::SrcAlphaSaturated, F::BothSrcAlpha, F::BothInvSrcAlpha, F::Constant,
            F::OneMinusConstant, F::Src1, F::OneMinusSrc1, F::Src1Alpha, F::OneMinusSrc1Alpha,
        ];
        for (v, f) in ogl {
            assert_eq!(F::decode(v), f, "{v:#x}");
        }
        for (i, f) in d3d.into_iter().enumerate() {
            assert_eq!(F::decode(i as u32 + 1), f, "{}", i + 1);
        }
        assert_eq!(F::decode(0), F::Unknown(0));
        assert_eq!(F::decode(0x14), F::Unknown(0x14));

        use BlendOperation as O;
        for (ogl, d3d, o) in [(0x8006, 1, O::Add), (0x800a, 2, O::Subtract), (0x800b, 3, O::ReverseSubtract), (0x8007, 4, O::Min), (0x8008, 5, O::Max)] {
            assert_eq!((O::decode(ogl), O::decode(d3d)), (o, o));
        }
        assert_eq!(O::decode(0), O::Unknown(0));

        use CompareFunction as C;
        let order = [C::Never, C::Less, C::Equal, C::LessEqual, C::Greater, C::NotEqual, C::GreaterEqual, C::Always];
        for (i, c) in order.into_iter().enumerate() {
            assert_eq!((C::decode(0x200 + i as u32), C::decode(1 + i as u32)), (c, c));
        }
        assert_eq!((C::decode(0), C::decode(9), C::decode(0x208)), (C::Unknown(0), C::Unknown(9), C::Unknown(0x208)));

        use StencilOperation as S;
        for (ogl, d3d, s) in [
            (0x1e00, 1, S::Keep), (0x0000, 2, S::Zero), (0x1e01, 3, S::Replace), (0x1e02, 4, S::IncrementClamp),
            (0x1e03, 5, S::DecrementClamp), (0x150a, 6, S::Invert), (0x8507, 7, S::IncrementWrap), (0x8508, 8, S::DecrementWrap),
        ] {
            assert_eq!((S::decode(ogl), S::decode(d3d)), (s, s));
        }
        assert_eq!(S::decode(9), S::Unknown(9));

        assert_eq!([Face::decode(0x404), Face::decode(0x405), Face::decode(0x408), Face::decode(0)], [Face::Front, Face::Back, Face::FrontAndBack, Face::Unknown(0)]);
        assert_eq!([FrontFace::decode(0x900), FrontFace::decode(0x901), FrontFace::decode(2)], [FrontFace::Cw, FrontFace::Ccw, FrontFace::Unknown(2)]);
        use PolygonMode as P;
        assert_eq!([P::decode(0x1b00), P::decode(0x1b01), P::decode(0x1b02), P::decode(3)], [P::Point, P::Line, P::Fill, P::Unknown(3)]);
    }

    #[test]
    fn depth_stencil_front_and_back() {
        let s = decode(&[
            (0x12cc, 1), (0x12e8, 1), (0x130c, 0x203),
            (0x1380, 1), (0x1594, 1),
            (0x1384, 1), (0x1388, 3), (0x138c, 7), (0x1390, 0x207), (0x1394, 0x80), (0x1398, 0x0f), (0x139c, 0xf0),
            (0x1598, 0x1e02), (0x159c, 0x150a), (0x15a0, 0x8508), (0x15a4, 5), (0x0f54, 0x12), (0x0f5c, 0x34), (0x0f58, 0x56),
        ])
        .depth_stencil;
        assert!(s.depth_test && s.depth_write && s.stencil_test && s.two_sided);
        assert_eq!(s.depth_compare, CompareFunction::LessEqual);
        use StencilOperation as S;
        assert_eq!(
            s.front,
            StencilFace { fail_op: S::Keep, depth_fail_op: S::Replace, pass_op: S::IncrementWrap, compare: CompareFunction::Always, reference: 0x80, read_mask: 0x0f, write_mask: 0xf0 }
        );
        assert_eq!(
            s.back,
            StencilFace { fail_op: S::IncrementClamp, depth_fail_op: S::Invert, pass_op: S::DecrementWrap, compare: CompareFunction::Greater, reference: 0x12, read_mask: 0x34, write_mask: 0x56 }
        );
    }

    #[test]
    fn rasterizer() {
        let r = decode(&[
            (0x1918, 1), (0x1920, 0x405), (0x191c, 0x900), (0x0dac, 0x1b01), (0x0db0, 0x1b00),
            (0x0dc8, 1), (0x15bc, fb(2.0)), (0x156c, fb(1.5)), (0x187c, fb(0.01)), (0x1110, 1),
            (0x193c, 0x181d), (0x153c, 0x11),
        ])
        .raster;
        assert!(r.cull_enable && r.alpha_to_coverage && r.alpha_to_one);
        assert_eq!((r.cull_face, r.front_face), (Face::Back, FrontFace::Cw));
        assert_eq!((r.front_polygon_mode, r.back_polygon_mode), (PolygonMode::Line, PolygonMode::Point));
        assert_eq!(r.depth_bias, DepthBias { point: false, line: false, fill: true, constant: 2.0, slope_scale: 1.5, clamp: 0.01, format_dependent: true });
        assert_eq!(
            r.clip,
            ClipControl { min_z_zero_max_z_one: true, z_clip_range: ZClipRange::UseFieldMinZZeroMaxZOne, pixel_min_z_clamp: true, pixel_max_z_clamp: true, geometry_clip: GeometryClip::FrustumXyzClip }
        );
        assert_eq!(r.clip.depth_clip(), Some(true));
        assert_eq!(decode(&[(0x191c, 0x900), (0x13ac, 0x10)]).raster.front_face, FrontFace::Ccw);
        assert_eq!(decode(&[(0x191c, 0x901), (0x13ac, 0x10)]).raster.wgpu_front_face(), Some(wgpu::FrontFace::Cw));
        // NVK's depth-clip-off setting: FRUSTUM_XY_CLIP, both pixel z clamps, z clip range -inf..+inf.
        let off = decode(&[(0x193c, 3 << 16 | 2 << 11 | 1 << 4 | 1 << 3)]).raster.clip;
        assert_eq!((off.z_clip_range, off.geometry_clip, off.depth_clip()), (ZClipRange::MinusInfPlusInf, GeometryClip::FrustumXyClip, Some(false)));
        assert_eq!(decode(&[(0x193c, 0)]).raster.clip.depth_clip(), None); // WZERO_CLIP, pixel clip: unknown
    }
}
