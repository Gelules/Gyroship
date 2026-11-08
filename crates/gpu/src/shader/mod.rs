//! SASS to WGSL (D24): one graphics program (SPH then SASS, as at its GPU
//! address or in a blob) becomes one WGSL module, validated by naga, with
//! the interface the renderer binds to it.
//!
//! The model:
//! - registers R0..R254 are `u32` variables (RZ reads 0); float operations
//!   reinterpret them; predicates P0..P6 are `bool` (PT is true); uniform
//!   registers and predicates likewise;
//! - constant bank b is the uniform buffer `cb{b}`, 64 KiB of `vec4<u32>`;
//! - vertex inputs, varyings and color outputs follow the SPH: attribute
//!   vector v is `@location(v)`, a varying vector v is `@location(v)`, color
//!   target t is `@location(t)` and takes its components from the registers
//!   at EXIT (four per written target, from R0; then sample mask and depth);
//! - every texture instruction gets its own texture (and sampler) binding,
//!   with where its handle comes from, so the renderer can bind the right
//!   texture at draw time;
//! - control flow: straight-line code when the program has no branch, else a
//!   loop that switches on the instruction index (any graph of branches).
//!   The SASS samples textures in divergent flow as the hardware allows, so
//!   the module turns WGSL's derivative uniformity check off.
//!
//! Bindings: stage group 0 (vertex) or 1 (pixel); constant bank b at binding
//! b; texture slot i at binding 32 + 2i and its sampler at 33 + 2i.
//!
//! Anything not supported is an error naming the instruction, never a guess.

pub mod sph;
pub mod uexpr;

use crate::sass::control::{BarMode, ControlOp};
use crate::sass::fp16::{self, Fp16Op};
use crate::sass::fp32::{self, Fp32Op, MufuFn, Rnd};
use crate::sass::int::{self, IntOp};
use crate::sass::memory::{Addr, AtomOp, AtomType, CbHandle, ImageDim, InterpFreq, InterpLoc, LdcMode, MemType, MemoryOp, Stride};
use crate::sass::moves::{self, MovesOp};
use crate::sass::operand::{Operand, PT, Pred, RZ, Src, URZ};
use crate::sass::texture::{LodMode, TexDim, TexHandle, TexQuery, TextureOp, Tld4Offset};
use crate::sass::{self, Op, SPH_SIZE, Stage, Word};
use sph::{Interp, Sph};
use uexpr::UExpr;
use std::collections::{BTreeMap, BTreeSet};
use std::fmt::Write;

/// A translated program.
#[derive(Clone, Debug)]
pub struct Translation {
    pub key: u64,
    pub stage: Stage,
    pub wgsl: String,
    pub interface: Interface,
}

/// What the renderer binds to a translated program.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Interface {
    /// Vertex: attribute location -> components read.
    pub attributes: BTreeMap<u8, u8>,
    /// Vertex: varyings written; pixel: varyings read (location -> interpolation).
    pub varyings: BTreeMap<u8, Interp>,
    /// Pixel: color target -> components written.
    pub targets: BTreeMap<u8, u8>,
    pub writes_depth: bool,
    /// Constant banks read.
    pub cbufs: BTreeSet<u8>,
    /// The bytes of each bank the program may read: up to the last 16-byte
    /// word read at a fixed offset, or 0x10000 (all of it) when one read is
    /// indexed. Banks stay declared and bound as 64 KiB (smaller declarations
    /// made the driver's compile of the large programs a hundred times slower);
    /// only these bytes must hold the bank, then zeros.
    pub cbuf_sizes: BTreeMap<u8, u32>,
    /// Texture slots, in binding order.
    pub textures: Vec<Texture>,
    /// The swizzle codes are read (at `SWIZZLE_BINDING`).
    pub swizzles: bool,
    /// Compute: images written (SUST), in binding order (`IMAGE_BINDING` + i).
    pub images: Vec<Image>,
    /// Global memory the program loads (LDG), one buffer per pointer it
    /// derives addresses from: the constant-bank word (bank, byte offset)
    /// holding the buffer's 64-bit address, in binding order (`BUFFER_BINDING` + i).
    pub buffers: Vec<(u8, u16)>,
    /// The program jumps indirectly (RET, BRX) only to targets read just before from
    /// a constant bank (a compiled switch's table): (bank, byte base the word is
    /// added to), per jump. With the bank's words, `Options::jump_targets` gives
    /// the exact blocks; else every instruction starts one (a switch case each).
    pub jump_tables: Option<Vec<(u8, u32)>>,
}

/// The binding of global-memory buffer 0; buffer i is at `BUFFER_BINDING + i`.
pub const BUFFER_BINDING: u32 = 600;

/// A storage image a program writes: where its handle is, and its dimension.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Image {
    pub handle: Handle,
    pub dim: ImageDim,
    /// Also loaded (SULD) by the program that stores to it: bound read-write.
    pub read: bool,
}

/// The binding of image slot 0; slot i is at `IMAGE_BINDING + i`.
pub const IMAGE_BINDING: u32 = 512;

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Texture {
    pub handle: Handle,
    pub dim: TexDim,
    /// Depth comparison (a comparison sampler).
    pub compare: bool,
    /// A texel fetch (no sampler).
    pub fetch: bool,
    /// Only its dimensions are read (TXQ), whatever its type.
    pub query: bool,
}

/// Where a texture instruction's 32-bit handle comes from (texture index in
/// the low 20 bits, sampler index in the high 12).
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Handle {
    CBuf { bank: u8, offset: u16 },
    /// A uniform register whose value the uniform datapath computed from
    /// constant-bank words (a texture handle or'ed with a sampler handle).
    Uniform(UExpr),
    /// A uniform register (loaded from a constant bank by ULDC in practice).
    UReg(u8),
    /// A register: known only when the shader runs.
    Bindless,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Error {
    /// Byte offset of the instruction in the SASS, if any.
    pub at: Option<usize>,
    pub what: String,
}

impl std::fmt::Display for Error {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self.at {
            Some(at) => write!(f, "{at:#06x}: {}", self.what),
            None => write!(f, "{}", self.what),
        }
    }
}

type R<T> = Result<T, String>;

/// How a vertex shader is paired with its pixel shader: wgpu needs both
/// stages to agree on each varying's interpolation, and every varying the
/// pixel shader reads must be written.
#[derive(Clone, Debug, Default, PartialEq, Eq, Hash)]
pub struct Options {
    /// The pixel shader's varyings: a vertex shader declares each with this
    /// interpolation (a varying it never writes stays (0, 0, 0, 1)).
    pub varyings: BTreeMap<u8, Interp>,
    /// Compute: the block size (the QMD's CTA_THREAD_DIMENSION).
    pub workgroup: [u32; 3],
    /// Compute: the shared memory bytes (the QMD's SHARED_MEMORY_SIZE).
    pub shared: u32,
    /// Compute: the WGSL storage format of each image slot (from its texture
    /// header); a missing one is `rgba16float`.
    pub images: Vec<String>,
    /// The type of each texture slot that TXQ queries, from its header (D2 or
    /// D3), by slot; a missing one is D2.
    pub queried: BTreeMap<usize, TexDim>,
    /// Vertex: the attributes fetched as integers, by location (true: signed):
    /// declared vec4<u32> or vec4<i32>, their bits read as they are.
    pub int_attributes: BTreeMap<u8, bool>,
    /// The instructions the indirect jumps of `Interface::jump_tables` may land on,
    /// from the bound banks' words: only they (and direct targets) start blocks.
    pub jump_targets: Option<BTreeSet<usize>>,
}

/// Per indirect jump (RET, BRX), the constant bank and byte base its target comes
/// from, when each one's register is loaded by a 32-bit LDC a few instructions
/// before, with nothing between that writes it or is a branch target (a compiled
/// switch: the table's words are byte offsets). None if one is not so.
/// ponytail: "writes it" is read from the ops' Debug text (any `dst*` field within
/// the four registers up to it); a register-writes table if ops ever need one.
fn jump_tables(ops: &[Op], target: &impl Fn(usize, i64) -> Option<usize>) -> Option<Vec<(u8, u32)>> {
    let landing: BTreeSet<usize> = ops
        .iter()
        .enumerate()
        .filter_map(|(i, op)| match op {
            Op::Control(ControlOp::Bra { offset, .. } | ControlOp::Call { offset, .. }) if *offset != -16 => target(i, *offset),
            _ => None,
        })
        .collect();
    let writes = |op: &Op, r: u8| {
        let text = format!("{op:?}");
        text.match_indices("dst").any(|(at, _)| {
            let rest = &text[at..];
            let digits: String = rest.split_once(": ").map(|(_, v)| v.chars().take_while(char::is_ascii_digit).collect()).unwrap_or_default();
            digits.parse::<u8>().is_ok_and(|d| d <= r && r - d < 4)
        })
    };
    let mut tables = Vec::new();
    for (i, op) in ops.iter().enumerate() {
        let (r, offset) = match op {
            Op::Control(ControlOp::Ret { target: r, offset, .. } | ControlOp::Brx { target: r, offset, .. }) => (*r, *offset),
            _ => continue,
        };
        let base = u32::try_from(16 * (i as i64 + 1) + offset).ok()?;
        let mut found = None;
        for j in (i.saturating_sub(8)..i).rev() {
            if landing.contains(&(j + 1)) {
                break;
            }
            match &ops[j] {
                Op::Memory(MemoryOp::Ldc { dst, bank, mode: LdcMode::Indexed, ty: MemType::B32, .. }) if *dst == r => {
                    found = Some((*bank, base));
                    break;
                }
                Op::Control(_) => break,
                other if writes(other, r) => break,
                _ => {}
            }
        }
        tables.push(found?);
    }
    Some(tables)
}

/// Translates a graphics program on its own.
pub fn translate(code: &[u8]) -> Result<Translation, Error> {
    translate_with(code, &Options::default())
}

/// Translates a graphics program for a given pairing.
pub fn translate_with(code: &[u8], options: &Options) -> Result<Translation, Error> {
    let fail = |what: &str| Error { at: None, what: what.to_string() };
    let stage = sass::stage(code);
    let (key, len, sph, sass) = match stage {
        Stage::Compute => {
            let (key, len) = sass::compute_key(code).ok_or_else(|| fail("no program (closing self-branch)"))?;
            (key, len, Sph::empty(stage), &code[..len])
        }
        Stage::Vertex | Stage::Pixel => {
            let (key, len) = sass::program_key(code).ok_or_else(|| fail("no program (SPH and closing self-branch)"))?;
            (key, len, Sph::new(stage, code).ok_or_else(|| fail("short SPH"))?, &code[SPH_SIZE..len])
        }
        _ => return Err(fail(&format!("{stage:?} programs are not supported"))),
    };
    let _ = len;
    let words: Vec<Word> = sass::words(sass).collect();
    let mut ops = Vec::with_capacity(words.len());
    for (i, w) in words.iter().enumerate() {
        ops.push(sass::decode(*w).ok_or(Error { at: Some(16 * i), what: format!("undecoded opcode {:#05x}", w.opcode()) })?);
    }
    let mut cx = Cx::new(sph);
    cx.images = options.images.clone();
    cx.queried = options.queried.clone();
    cx.int_attributes = options.int_attributes.clone();
    cx.jump_targets = options.jump_targets.clone();
    // Images the program both stores to and loads from: their loads read the image itself.
    let handles = |load: bool| -> BTreeSet<(u8, u16)> {
        ops.iter()
            .filter_map(|o| match o {
                Op::Memory(MemoryOp::Sust { handle, .. }) if !load => Some((handle.bank, handle.offset)),
                Op::Memory(MemoryOp::Suld { handle, .. }) if load => Some((handle.bank, handle.offset)),
                _ => None,
            })
            .collect()
    };
    cx.read_write = handles(true).intersection(&handles(false)).copied().collect();
    let body = cx.program(&words, &ops)?;
    Ok(Translation { key, stage, wgsl: cx.module(&body, options), interface: cx.iface })
}

/// Whether integer image coordinates `at` (`coords` of them, and the layer of
/// an array image) fall inside image `slot`.
fn image_inside(slot: usize, at: &str, coords: usize, layer: Option<&str>) -> String {
    let inside = if coords == 1 { format!("(u32({at}) < textureDimensions(img{slot}))") } else { format!("all(vec{coords}<u32>({at}) < textureDimensions(img{slot}))") };
    match layer {
        Some(l) => format!("({inside} && u32({l}) < textureNumLayers(img{slot}))"),
        None => inside,
    }
}

/// A sub-word access's bits and signedness.
fn small_mem(ty: MemType) -> Option<(u32, bool)> {
    match ty {
        MemType::U8 => Some((8, false)),
        MemType::S8 => Some((8, true)),
        MemType::U16 => Some((16, false)),
        MemType::S16 => Some((16, true)),
        _ => None,
    }
}

/// The 32-bit words of a word-sized or wider access.
fn mem_words(ty: MemType) -> Result<u32, String> {
    match ty {
        MemType::B32 => Ok(1),
        MemType::B64 => Ok(2),
        MemType::B128 => Ok(4),
        other => Err(format!("a {other:?} access")),
    }
}

/// Every instruction the translator refuses, each alone and with its byte
/// offset (for study: `translate_with` stops at the first).
pub fn gaps(code: &[u8], options: &Options) -> Vec<Error> {
    let stage = sass::stage(code);
    let (sph, sass) = match stage {
        Stage::Compute => match sass::compute_key(code) {
            Some((_, len)) => (Sph::empty(stage), &code[..len]),
            None => return vec![Error { at: None, what: "no program".into() }],
        },
        _ => match (sass::program_key(code), Sph::new(stage, code)) {
            (Some((_, len)), Some(sph)) => (sph, &code[SPH_SIZE..len]),
            _ => return vec![Error { at: None, what: "no program".into() }],
        },
    };
    let mut cx = Cx::new(sph);
    cx.images = options.images.clone();
    let mut errors = Vec::new();
    for (i, w) in sass::words(sass).enumerate() {
        let result = match sass::decode(w) {
            Some(op) => cx.instruction(&mut String::new(), i, w, &op, Some(&|_, _| Some(0))).map(|_| ()),
            None => Err(format!("undecoded opcode {:#05x}", w.opcode())),
        };
        if let Err(what) = result {
            errors.push(Error { at: Some(16 * i), what });
        }
    }
    errors
}

fn interp_attribute(mode: Interp) -> &'static str {
    match mode {
        Interp::Constant => "flat",
        Interp::ScreenLinear => "linear",
        _ => "perspective",
    }
}

/// Validates a WGSL module with naga.
pub fn validate(wgsl: &str) -> Result<(), String> {
    let module = naga::front::wgsl::parse_str(wgsl).map_err(|e| e.emit_to_string(wgsl))?;
    naga::valid::Validator::new(naga::valid::ValidationFlags::all(), naga::valid::Capabilities::all()).validate(&module).map_err(|e| e.emit_to_string(wgsl))?;
    Ok(())
}

/// The translation state.
struct Cx {
    sph: Sph,
    regs: BTreeSet<u8>,
    preds: BTreeSet<u8>,
    uregs: BTreeSet<u8>,
    upreds: BTreeSet<u8>,
    iface: Interface,
    /// Texture and sampler declarations.
    decls: String,
    helpers: BTreeSet<&'static str>,
    temps: u32,
    /// Uniform registers whose value is known from constant-bank words, as
    /// of the instruction being translated.
    // ponytail: tracked in program order, which is the execution order for
    // the handle computations seen (they come first); branchy ones would
    // need the flow graph.
    uexpr: BTreeMap<u8, UExpr>,
    /// The same for ordinary registers (MOV, LDC and LOP3 of known values), for
    /// texture handles computed in a GPR. A predicated write counts as made.
    gexpr: BTreeMap<u8, UExpr>,
    /// Compute: the storage format of each image slot (`Options::images`).
    images: Vec<String>,
    /// Images both stored to and loaded from, by constant-bank handle.
    read_write: BTreeSet<(u8, u16)>,
    /// `Options::queried`.
    queried: BTreeMap<usize, TexDim>,
    int_attributes: BTreeMap<u8, bool>,
    jump_targets: Option<BTreeSet<usize>>,
    /// Compute: the program uses shared memory (`shm`, words as atomics so
    /// that ATOMS and plain accesses can share it).
    shm: bool,
    /// Registers holding a global address derived from a pointer in a
    /// constant bank (bank, offset): `Interface::buffers`.
    roots: BTreeMap<u8, (u8, u16)>,
    /// Compute: the program reads its lane in the warp (SR_LANEID).
    lane: bool,
}

/// The high word of a 32 × 32-bit product, unsigned or signed (WGSL has no
/// 64-bit integers): the four 16-bit partial products, then the sign fix-up.
const MUL_HI: &str = "fn mul_hi(a: u32, b: u32, signed: bool) -> u32 {
    let ll = (a & 0xffffu) * (b & 0xffffu);
    let lh = (a & 0xffffu) * (b >> 16u);
    let hl = (a >> 16u) * (b & 0xffffu);
    let mid = (ll >> 16u) + (lh & 0xffffu) + (hl & 0xffffu);
    var hi = (a >> 16u) * (b >> 16u) + (lh >> 16u) + (hl >> 16u) + (mid >> 16u);
    if signed {
        hi = hi - select(0u, b, bitcast<i32>(a) < 0) - select(0u, a, bitcast<i32>(b) < 0);
    }
    return hi;
}
";

/// Shared memory read-modify-writes WGSL has no atomic for, by `op`: signed
/// min (0) and max (1), wrapping increment (2) and decrement (3).
const SHM_CAS: &str = "fn shm_cas(i: u32, op: u32, v: u32) -> u32 {
    var old: u32;
    loop {
        old = atomicLoad(&shm[i]);
        var next = old;
        switch op {
            case 0u: { next = bitcast<u32>(min(bitcast<i32>(old), bitcast<i32>(v))); }
            case 1u: { next = bitcast<u32>(max(bitcast<i32>(old), bitcast<i32>(v))); }
            case 2u: { next = select(old + 1u, 0u, old >= v); }
            default: { next = select(old - 1u, v, old == 0u || old > v); }
        }
        if atomicCompareExchangeWeak(&shm[i], old, next).exchanged { break; }
    }
    return old;
}
";

impl Cx {
    fn new(sph: Sph) -> Cx {
        Cx { sph, regs: BTreeSet::new(), preds: BTreeSet::new(), uregs: BTreeSet::new(), upreds: BTreeSet::new(), iface: Interface::default(), decls: String::new(), helpers: BTreeSet::new(), temps: 0, uexpr: BTreeMap::new(), gexpr: BTreeMap::new(), images: Vec::new(), queried: BTreeMap::new(), int_attributes: BTreeMap::new(), jump_targets: None, read_write: BTreeSet::new(), shm: false, lane: false, roots: BTreeMap::new() }
    }

    /// A shared memory access's byte address, `reg × stride + ureg + offset`, as a new `let`.
    fn shared_addr(&mut self, out: &mut String, a: &Addr, stride: Stride) -> R<String> {
        let scale = match stride {
            Stride::X1 => 1,
            Stride::X4 => 4,
            Stride::X8 => 8,
            Stride::X16 => 16,
            Stride::Other(s) => return Err(format!("shared memory stride {s}")),
        };
        self.shm = true;
        let reg = self.r(a.reg);
        let ureg = a.ureg.map_or_else(|| "0u".to_string(), |u| self.ur(u));
        let t = self.temp();
        let _ = writeln!(out, "let {t} = {reg} * {scale}u + {ureg} + {:#x}u;", a.offset as u32);
        Ok(t)
    }

    fn group(&self) -> u32 {
        if self.sph.stage == Stage::Pixel { 1 } else { 0 }
    }

    fn temp(&mut self) -> String {
        self.temps += 1;
        format!("t{}", self.temps)
    }

    // ---- operands ----

    fn r(&mut self, n: u8) -> String {
        if n == RZ {
            return "0u".into();
        }
        self.regs.insert(n);
        format!("r{n}")
    }

    fn ur(&mut self, n: u8) -> String {
        if n == URZ {
            return "0u".into();
        }
        self.uregs.insert(n);
        format!("ur{n}")
    }

    fn cb(&mut self, bank: u8, offset: u32) -> R<String> {
        if !offset.is_multiple_of(4) {
            return Err(format!("unaligned constant c[{bank}][{offset:#x}]"));
        }
        self.iface.cbufs.insert(bank);
        self.reach(bank, (offset / 16 + 1) * 16);
        Ok(format!("cb{bank}[{}u][{}u]", offset / 16, offset / 4 % 4))
    }

    /// Bank `bank` is read up to byte `end` (0x10000: anywhere, an indexed read).
    fn reach(&mut self, bank: u8, end: u32) {
        let e = self.iface.cbuf_sizes.entry(bank).or_insert(0);
        *e = (*e).max(end.min(0x10000));
    }

    /// An 8- or 16-bit constant, zero- or sign-extended to 32 bits.
    fn cb_small(&mut self, bank: u8, offset: u32, bits: u32, signed: bool) -> R<String> {
        if !offset.is_multiple_of(bits / 8) {
            return Err(format!("unaligned {bits}-bit constant c[{bank}][{offset:#x}]"));
        }
        let word = self.cb(bank, offset & !3)?;
        let at = (offset & 3) * 8;
        Ok(if signed { format!("bitcast<u32>(extractBits(bitcast<i32>({word}), {at}u, {bits}u))") } else { format!("extractBits({word}, {at}u, {bits}u)") })
    }

    /// A source's raw 32 bits, without modifiers.
    fn raw(&mut self, o: &Operand) -> R<String> {
        Ok(match o.src {
            Src::Reg(n) => self.r(n),
            Src::UReg(n) => self.ur(n),
            Src::Imm(v) => format!("{v:#x}u"),
            Src::CBuf { bank, offset } => self.cb(bank, offset.into())?,
            Src::CBufU { .. } => return Err("bindless constant buffer source".into()),
        })
    }

    /// A source as f32, with abs and neg.
    fn f(&mut self, o: &Operand) -> R<String> {
        let mut x = format!("f({})", self.raw(o)?);
        if o.abs {
            x = format!("abs({x})");
        }
        if o.neg {
            x = format!("(-{x})");
        }
        Ok(x)
    }

    /// A source as u32 with its negation as two's complement (or, with `not`, as bitwise not).
    fn i(&mut self, o: &Operand, not: bool) -> R<String> {
        let x = self.raw(o)?;
        Ok(match (o.neg, not) {
            (false, _) => x,
            (true, false) => format!("(0u - {x})"),
            (true, true) => format!("(~{x})"),
        })
    }

    fn p(&mut self, p: Pred) -> String {
        let v = if p.reg == PT {
            "true".to_string()
        } else {
            self.preds.insert(p.reg);
            format!("p{}", p.reg)
        };
        if p.not { format!("!{v}") } else { v }
    }

    fn up(&mut self, p: Pred) -> String {
        let v = if p.reg == PT {
            "true".to_string()
        } else {
            self.upreds.insert(p.reg);
            format!("up{}", p.reg)
        };
        if p.not { format!("!{v}") } else { v }
    }

    fn set(&mut self, out: &mut String, d: u8, e: &str) {
        self.gexpr.remove(&d);
        self.roots.remove(&d);
        if d != RZ {
            let r = self.r(d);
            let _ = writeln!(out, "{r} = {e};");
        }
    }

    /// A uniform operand's value, if known from constant-bank words.
    fn uval(&self, o: &Operand) -> Option<UExpr> {
        if o.neg || o.abs {
            return None;
        }
        Some(match o.src {
            Src::Imm(v) => UExpr::Imm(v),
            Src::UReg(URZ) => UExpr::Imm(0),
            Src::UReg(n) => self.uexpr.get(&n)?.clone(),
            Src::Reg(RZ) => UExpr::Imm(0),
            Src::Reg(n) => self.gexpr.get(&n)?.clone(),
            Src::CBuf { bank, offset } => UExpr::CBuf(bank, offset),
            _ => return None,
        })
    }

    fn uvals(&self, o: [&Operand; 3]) -> Option<[UExpr; 3]> {
        Some([self.uval(o[0])?, self.uval(o[1])?, self.uval(o[2])?])
    }

    /// Records a uniform register's value after its write.
    fn known(&mut self, d: u8, e: Option<UExpr>) {
        if let (Some(e), true) = (e, d != URZ) {
            self.uexpr.insert(d, e);
        }
    }

    /// After a write of `d` from `srcs` (an add or a move): the pointer it derives
    /// from, a constant-bank word at an 8-byte boundary or a register holding one.
    // ponytail: in program order, as `uexpr`; an address built across branches
    // from two pointers would need the flow graph.
    fn root_from(&mut self, d: u8, srcs: &[&Operand]) {
        let root = srcs.iter().find_map(|o| match o.src {
            Src::CBuf { bank, offset } if offset % 8 == 0 => Some((bank, offset)),
            Src::Reg(r) => self.roots.get(&r).copied(),
            _ => None,
        });
        if let (Some(root), true) = (root, d != RZ) {
            self.roots.insert(d, root);
        }
    }

    /// Records an ordinary register's value after its write.
    fn gknown(&mut self, d: u8, e: Option<UExpr>) {
        if let (Some(e), true) = (e, d != RZ) {
            self.gexpr.insert(d, e);
        }
    }

    fn set_ur(&mut self, out: &mut String, d: u8, e: &str) {
        self.uexpr.remove(&d);
        if d != URZ {
            let r = self.ur(d);
            let _ = writeln!(out, "{r} = {e};");
        }
    }

    fn set_p(&mut self, out: &mut String, d: u8, e: &str) {
        if d != PT {
            self.preds.insert(d);
            let _ = writeln!(out, "p{d} = {e};");
        }
    }

    fn set_up(&mut self, out: &mut String, d: u8, e: &str) {
        if d != PT {
            self.upreds.insert(d);
            let _ = writeln!(out, "up{d} = {e};");
        }
    }

    // ---- control flow ----

    /// The body: straight-line, or a switch over instruction indices.
    fn program(&mut self, words: &[Word], ops: &[Op]) -> Result<String, Error> {
        let at = |i: usize| move |what: String| Error { at: Some(16 * i), what };
        let target = |i: usize, offset: i64| -> Option<usize> { usize::try_from((16 * (i as i64 + 1) + offset) / 16).ok() };
        // An indirect jump (RET, BRX) may land on any instruction: each starts a block then.
        let indirect = ops.iter().any(|o| matches!(o, Op::Control(ControlOp::Ret { .. } | ControlOp::Brx { .. })));
        // A table's targets known from the banks: only they start blocks.
        let tables = if indirect { jump_tables(ops, &target) } else { None };
        self.iface.jump_tables = tables.clone();
        let mut leaders: BTreeSet<usize> = match (&self.jump_targets, &tables) {
            _ if !indirect => BTreeSet::from([0usize]),
            (Some(t), Some(_)) => t.iter().copied().chain([0]).collect(),
            _ => (0..ops.len()).collect(),
        };
        for (i, op) in ops.iter().enumerate() {
            match op {
                Op::Control(ControlOp::Bra { offset, .. } | ControlOp::Call { offset, .. }) if *offset != -16 => {
                    leaders.insert(target(i, *offset).filter(|t| *t < ops.len()).ok_or_else(|| at(i)("branch outside the program".into()))?);
                    leaders.insert(i + 1);
                }
                // After a conditional EXIT the program goes on; after an unconditional
                // one, only a branch target (already a leader) can follow.
                Op::Control(ControlOp::Exit { cond, .. }) if !(words[i].always() && *cond == Pred::TRUE) => {
                    leaders.insert(i + 1);
                }
                Op::Control(ControlOp::Ret { .. } | ControlOp::Brx { .. }) => {
                    leaders.insert(i + 1);
                }
                _ => {}
            }
        }
        leaders.retain(|&l| l < ops.len());
        let mut out = String::new();
        if leaders.len() == 1 {
            for (i, (w, op)) in words.iter().zip(ops).enumerate() {
                if self.instruction(&mut out, i, *w, op, None).map_err(at(i))? {
                    return Ok(out);
                }
            }
            let exit = self.exit();
            out.push_str(&exit);
            return Ok(out);
        }
        let starts: Vec<usize> = leaders.into_iter().collect();
        // Without backward or indirect jumps, the blocks run in order, each under its
        // `if`: lanes that branched over a block wait for the others, and the warp is
        // whole again at the next block, as SASS's BSSY/BSYNC expect before its shuffles
        // and barriers. A loop over a switch would let lanes run blocks apart.
        let backward = ops.iter().enumerate().any(|(i, op)| matches!(op, Op::Control(ControlOp::Bra { offset, .. } | ControlOp::Call { offset, .. }) if *offset != -16 && target(i, *offset).is_none_or(|t| t <= i)));
        if !indirect && !backward {
            out.push_str("var pc: u32 = 0u;\n");
            for (b, &start) in starts.iter().enumerate() {
                let end = starts.get(b + 1).copied().unwrap_or(ops.len());
                let mut block = String::new();
                let mut done = false;
                for i in start..end {
                    if self.instruction(&mut block, i, words[i], &ops[i], Some(&target)).map_err(at(i))? {
                        done = true;
                        break;
                    }
                }
                if !done {
                    let _ = writeln!(block, "pc = {end}u;");
                }
                let _ = write!(out, "if (pc == {start}u) {{\n{block}}}\n");
            }
            let exit = self.exit();
            out.push_str(&exit);
            return Ok(out);
        }
        out.push_str("var pc: u32 = 0u;\nloop {\nswitch pc {\n");
        for (b, &start) in starts.iter().enumerate() {
            let end = starts.get(b + 1).copied().unwrap_or(ops.len());
            let mut block = String::new();
            let mut done = false;
            for i in start..end {
                if self.instruction(&mut block, i, words[i], &ops[i], Some(&target)).map_err(at(i))? {
                    done = true;
                    break;
                }
            }
            if !done {
                let _ = writeln!(block, "pc = {end}u;");
            }
            let _ = write!(out, "case {start}u: {{\n{block}}}\n");
        }
        let exit = self.exit();
        let _ = write!(out, "default: {{\n{exit}}}\n}}\n}}\n{exit}");
        Ok(out)
    }

    /// One instruction; true when it ends the block (or the program).
    /// `target` is given in switch mode.
    fn instruction(&mut self, out: &mut String, i: usize, w: Word, op: &Op, target: Option<&dyn Fn(usize, i64) -> Option<usize>>) -> R<bool> {
        let (g, gnot) = w.guard();
        let guard = Pred { reg: g, not: gnot };
        let both = |cx: &mut Cx, c: Pred| -> Option<String> {
            let parts: Vec<String> = [guard, c].into_iter().filter(|p| *p != Pred::TRUE).map(|p| cx.p(p)).collect();
            (!parts.is_empty()).then(|| parts.join(" && "))
        };
        match op {
            Op::Control(ControlOp::Bra { offset: -16, .. }) => {
                // The closing self-branch: never reached after EXIT.
                let exit = self.exit();
                out.push_str(&exit);
                return Ok(true);
            }
            // RET and BRX: to next + offset + the register (an absolute byte offset when
            // offset = -next, as the game does), as an instruction index.
            Op::Control(ControlOp::Ret { target: r, offset, cond, .. } | ControlOp::Brx { target: r, offset, cond }) => {
                if target.is_none() {
                    return Err("indirect jump in straight-line code".into());
                }
                let base = (16 * (i as i64 + 1) + offset) as u32;
                let reg = self.r(*r);
                let e = format!("(({reg} + {base}u) >> 4u)");
                match both(self, *cond) {
                    None => {
                        let _ = writeln!(out, "pc = {e};");
                    }
                    Some(c) => {
                        let _ = writeln!(out, "if ({c}) {{ pc = {e}; }} else {{ pc = {}u; }}", i + 1);
                    }
                }
                return Ok(true);
            }
            // CALL.NOINC: the compiler keeps the return offset in a register for RET,
            // so the call itself is a jump.
            Op::Control(ControlOp::Bra { offset, cond, .. } | ControlOp::Call { offset, cond, .. }) => {
                let t = target.and_then(|f| f(i, *offset)).ok_or("branch in straight-line code")?;
                match both(self, *cond) {
                    None => {
                        let _ = writeln!(out, "pc = {t}u;");
                    }
                    Some(c) => {
                        let _ = writeln!(out, "if ({c}) {{ pc = {t}u; }} else {{ pc = {}u; }}", i + 1);
                    }
                }
                return Ok(true);
            }
            Op::Control(ControlOp::Exit { cond, .. }) => {
                let exit = self.exit();
                return Ok(match both(self, *cond) {
                    None => {
                        out.push_str(&exit);
                        true
                    }
                    Some(c) => {
                        let _ = write!(out, "if ({c}) {{\n{exit}}}\n");
                        false
                    }
                });
            }
            _ => {}
        }
        if guard == (Pred { reg: PT, not: true }) {
            return Ok(false);
        }
        let mut body = String::new();
        self.emit(&mut body, op)?;
        if body.is_empty() {
            return Ok(false);
        }
        if guard == Pred::TRUE {
            out.push_str(&body);
        } else {
            let c = self.p(guard);
            let _ = write!(out, "if ({c}) {{\n{body}}}\n");
        }
        Ok(false)
    }

    /// The code that ends the shader: return the outputs.
    fn exit(&mut self) -> String {
        match self.sph.stage {
            Stage::Vertex => return "return o;\n".into(),
            Stage::Compute => return "return;\n".into(),
            _ => {}
        }
        let mut out = String::new();
        let mut k = 0u8;
        for (t, m) in self.sph.targets().into_iter().enumerate() {
            if m != 0 {
                let c: Vec<String> = (0..4).map(|c| format!("f({})", self.r(4 * k + c))).collect();
                let _ = writeln!(out, "o.c{t} = vec4<f32>({});", c.join(", "));
                k += 1;
            }
        }
        if self.sph.writes_depth() {
            let d = self.r(4 * k + 1);
            let _ = writeln!(out, "o.depth = f({d});");
        }
        if self.has_ps_outputs() {
            out.push_str("return o;\n");
        } else {
            out.push_str("return;\n");
        }
        out
    }

    fn has_ps_outputs(&self) -> bool {
        self.sph.targets().iter().any(|m| *m != 0) || self.sph.writes_depth()
    }

    // ---- the module ----

    fn module(&mut self, body: &str, options: &Options) -> String {
        let g = self.group();
        let mut m = String::from("diagnostic(off, derivative_uniformity);\n\nfn f(x: u32) -> f32 { return bitcast<f32>(x); }\nfn u(x: f32) -> u32 { return bitcast<u32>(x); }\n");
        for h in &self.helpers {
            m.push_str(h);
        }
        for b in &self.iface.cbufs {
            let _ = writeln!(m, "@group({g}) @binding({b}) var<uniform> cb{b}: array<vec4<u32>, 4096>;");
        }
        m.push_str(&self.decls);
        if self.iface.swizzles {
            let _ = writeln!(m, "@group({g}) @binding({SWIZZLE_BINDING}) var<uniform> swizzles: array<vec4<u32>, {}>;", self.iface.textures.len().div_ceil(4));
        }
        let mut locals = String::new();
        for r in &self.regs {
            let _ = writeln!(locals, "var r{r}: u32;");
        }
        for p in &self.preds {
            let _ = writeln!(locals, "var p{p}: bool;");
        }
        for r in &self.uregs {
            let _ = writeln!(locals, "var ur{r}: u32;");
        }
        for p in &self.upreds {
            let _ = writeln!(locals, "var up{p}: bool;");
        }
        if self.sph.stage == Stage::Compute {
            if self.shm {
                let _ = writeln!(m, "var<workgroup> shm: array<atomic<u32>, {}>;", options.shared.div_ceil(4).max(1));
            }
            let [x, y, z] = options.workgroup.map(|n| n.max(1));
            let lane = if self.lane { ", @builtin(subgroup_invocation_id) lane: u32" } else { "" };
            let _ = write!(m, "\n@compute @workgroup_size({x}, {y}, {z})\nfn main(@builtin(local_invocation_id) lid: vec3<u32>, @builtin(workgroup_id) wid: vec3<u32>{lane}) {{\n{locals}{body}}}\n");
        } else if self.sph.stage == Stage::Vertex {
            let mut vin = String::from("struct VsIn {\n@builtin(vertex_index) vid: u32,\n@builtin(instance_index) iid: u32,\n");
            for (v, mask) in self.sph.inputs().into_iter().enumerate() {
                if mask != 0 {
                    self.iface.attributes.insert(v as u8, mask);
                    let ty = match self.int_attributes.get(&(v as u8)) {
                        Some(true) => "vec4<i32>",
                        Some(false) => "vec4<u32>",
                        None => "vec4<f32>",
                    };
                    let _ = writeln!(vin, "@location({v}) a{v}: {ty},");
                }
            }
            vin.push_str("};\n");
            let mut vout = String::from("struct VsOut {\n@builtin(position) pos: vec4<f32>,\n");
            let mut init = String::from("var o: VsOut;\no.pos = vec4<f32>(0.0, 0.0, 0.0, 1.0);\n");
            // Written varyings, and those the paired pixel shader reads; the AST of an
            // undeclared one is refused above, so every written one is here.
            let written: BTreeSet<u8> = self.sph.outputs().into_iter().enumerate().filter(|(_, m)| *m != 0).map(|(v, _)| v as u8).collect();
            for v in written.iter().chain(options.varyings.keys()).copied().collect::<BTreeSet<u8>>() {
                let mode = options.varyings.get(&v).copied().unwrap_or(Interp::Perspective);
                self.iface.varyings.insert(v, mode);
                let _ = writeln!(vout, "@location({v}) @interpolate({}) v{v}: vec4<f32>,", interp_attribute(mode));
                let _ = writeln!(init, "o.v{v} = vec4<f32>(0.0, 0.0, 0.0, 1.0);");
            }
            vout.push_str("};\n");
            let _ = write!(m, "{vin}{vout}\n@vertex\nfn main(i: VsIn) -> VsOut {{\n{init}{locals}{body}}}\n");
        } else {
            let mut pin = String::from("struct PsIn {\n@builtin(position) frag: vec4<f32>,\n@builtin(front_facing) front: bool,\n");
            for (v, comps) in self.sph.interpolation().into_iter().enumerate() {
                if let Some(mode) = comps.into_iter().find(|c| *c != Interp::Unused) {
                    self.iface.varyings.insert(v as u8, mode);
                    let _ = writeln!(pin, "@location({v}) @interpolate({}) v{v}: vec4<f32>,", interp_attribute(mode));
                }
            }
            pin.push_str("};\n");
            let mut pout = String::from("struct PsOut {\n");
            for (t, mask) in self.sph.targets().into_iter().enumerate() {
                if mask != 0 {
                    self.iface.targets.insert(t as u8, mask);
                    let _ = writeln!(pout, "@location({t}) c{t}: vec4<f32>,");
                }
            }
            if self.sph.writes_depth() {
                self.iface.writes_depth = true;
                pout.push_str("@builtin(frag_depth) depth: f32,\n");
            }
            pout.push_str("};\n");
            if self.has_ps_outputs() {
                let _ = write!(m, "{pin}{pout}\n@fragment\nfn main(i: PsIn) -> PsOut {{\nvar o: PsOut;\n{locals}{body}}}\n");
            } else {
                let _ = write!(m, "{pin}\n@fragment\nfn main(i: PsIn) {{\n{locals}{body}}}\n");
            }
        }
        m
    }

    // ---- operations ----

    fn emit(&mut self, out: &mut String, op: &Op) -> R<()> {
        match op {
            Op::Fp32(o) => self.fp32(out, o),
            Op::Fp16(o) => self.fp16(out, o),
            Op::Int(o) => self.int(out, o),
            Op::Moves(o) => self.moves(out, o),
            Op::Memory(o) => self.memory(out, o),
            Op::Texture(t) => self.texture(out, t),
            Op::Control(c) => match c {
                ControlOp::Kill { cond } if self.sph.stage == Stage::Pixel => {
                    if *cond == Pred::TRUE {
                        out.push_str("discard;\n");
                    } else {
                        let c = self.p(*cond);
                        let _ = writeln!(out, "if ({c}) {{ discard; }}");
                    }
                    Ok(())
                }
                // Warp convergence, scoreboards and NOPs: nothing per thread.
                ControlOp::Bssy { .. } | ControlOp::Bsync { .. } | ControlOp::Break { .. } | ControlOp::WarpSync { .. } | ControlOp::DepBar { .. } | ControlOp::Nop => Ok(()),
                // BAR.SYNC 0: the whole block waits, and sees the shared memory written before.
                ControlOp::Bar { mode: BarMode::Sync, red_op: 0 } if self.sph.stage == Stage::Compute => {
                    out.push_str("workgroupBarrier();\n");
                    Ok(())
                }
                other => Err(format!("unsupported: {other:?}")),
            },
        }
    }

    /// A packed-half operand as `vec2<f32>` (x the low half), after its lane
    /// selector, abs and neg.
    fn h2(&mut self, o: &Operand) -> R<String> {
        let raw = self.raw(o)?;
        let mut v = match o.swizzle {
            fp16::SWZ_H1_H0 => format!("unpack2x16float({raw})"),
            fp16::SWZ_H0_H0 => format!("unpack2x16float({raw}).xx"),
            fp16::SWZ_H1_H1 => format!("unpack2x16float({raw}).yy"),
            // .F32: the operand is one f32, for both lanes.
            _ => format!("vec2<f32>(f({raw}))"),
        };
        if o.abs {
            v = format!("abs({v})");
        }
        if o.neg {
            v = format!("(-{v})");
        }
        Ok(v)
    }

    // ponytail: half denormals are not flushed (.FTZ), and results round as
    // pack2x16float does (to nearest on the GPUs seen).
    fn fp16(&mut self, out: &mut String, op: &Fp16Op) -> R<()> {
        let pack = |v: String, sat: bool| format!("pack2x16float({})", if sat { format!("clamp({v}, vec2<f32>(0.0), vec2<f32>(1.0))") } else { v });
        match op {
            Fp16Op::I2fp { dst, src, signed, dst_bits: 32, src_bits: 32, .. } => {
                let e = self.i2f(src, *signed, 32, 0)?;
                self.set(out, *dst, &e);
            }
            // .F32: one f32 result, the low lane's (the game converts a half with
            // `HADD2.F32 R, -RZ, R.H0_H0`).
            Fp16Op::Hadd2 { dst, a, b, sat, f32, .. } => {
                let e = format!("{} + {}", self.h2(a)?, self.h2(b)?);
                let e = if *f32 { format!("u(({e}).x)") } else { pack(e, *sat) };
                self.set(out, *dst, &e);
            }
            Fp16Op::Hmul2 { dst, a, b, sat, relu: false, .. } => {
                let e = format!("{} * {}", self.h2(a)?, self.h2(b)?);
                self.set(out, *dst, &pack(e, *sat));
            }
            Fp16Op::Hfma2 { dst, a, b, c, sat, f32: false, relu: false, bf16: false, .. } => {
                let e = format!("fma({}, {}, {})", self.h2(a)?, self.h2(b)?, self.h2(c)?);
                self.set(out, *dst, &pack(e, *sat));
            }
            Fp16Op::Hmnmx2 { dst, a, b, min, f32: false, nan: false, xorsign: false, bf16: false, .. } => {
                let (x, y, m) = (self.h2(a)?, self.h2(b)?, self.p(*min));
                self.set(out, *dst, &format!("pack2x16float(select(max({x}, {y}), min({x}, {y}), {m}))"));
            }
            // MOVM.16.MT88: lane t gets row t / 4, columns 2 (t % 4) and + 1 of the
            // transpose, i.e. rows 2 (t % 4) and + 1, column t / 4 of the source, which
            // lanes 8 (t % 4) + t / 8 and + 4 hold in their half (t / 4) % 2.
            Fp16Op::Movm { dst, src, shape: 0 } if self.sph.stage == Stage::Compute => {
                self.lane = true;
                let s = self.r(*src);
                let (s0, s1, h) = (self.temp(), self.temp(), self.temp());
                let _ = writeln!(out, "let {s0} = subgroupShuffle({s}, 8u * (lane % 4u) + lane / 8u);");
                let _ = writeln!(out, "let {s1} = subgroupShuffle({s}, 8u * (lane % 4u) + 4u + lane / 8u);");
                let _ = writeln!(out, "let {h} = ((lane / 4u) % 2u) * 16u;");
                self.set(out, *dst, &format!("(extractBits({s0}, {h}, 16u) | (extractBits({s1}, {h}, 16u) << 16u))"));
            }
            // HMMA.1688/.16816.F16, D = A × B + C over the warp, in PTX mma's fragments:
            // lane t (g = t / 4, q = t % 4) holds rows g and g + 8 of A (k = 2q, 2q + 1,
            // and + 8 in the next registers), column g of B (k = 2q, 2q + 1, + 8), and
            // rows g and g + 8, columns 2q and 2q + 1 of C and D. Products add in f32 and
            // round to half after each K = 8 step (the 16816 shape as two 1688 ones).
            // ponytail: the tensor cores' exact accumulation precision is not known.
            Fp16Op::Hmma { dst, a, b, c, k: k @ (8 | 16), f32: false, src_type: 0 } if self.sph.stage == Stage::Compute => {
                self.lane = true;
                let reg = |cx: &mut Cx, r: u8, k: u8| if r == RZ { "0u".to_string() } else { cx.r(r + k) };
                let acc = self.temp();
                let (c0, c1) = (reg(self, *c, 0), reg(self, *c, 1));
                let _ = writeln!(out, "var {acc} = mat2x2<f32>(unpack2x16float({c0}), unpack2x16float({c1}));");
                for step in 0..k / 8 {
                    let (ag, ag8, bk) = (reg(self, *a, 2 * step), reg(self, *a, 2 * step + 1), reg(self, *b, step));
                    for j in 0..4 {
                        let (x, x8, y0, y1) = (self.temp(), self.temp(), self.temp(), self.temp());
                        let _ = writeln!(out, "let {x} = unpack2x16float(subgroupShuffle({ag}, 4u * (lane / 4u) + {j}u));");
                        let _ = writeln!(out, "let {x8} = unpack2x16float(subgroupShuffle({ag8}, 4u * (lane / 4u) + {j}u));");
                        let _ = writeln!(out, "let {y0} = unpack2x16float(subgroupShuffle({bk}, 8u * (lane % 4u) + {j}u));");
                        let _ = writeln!(out, "let {y1} = unpack2x16float(subgroupShuffle({bk}, 8u * (lane % 4u) + 4u + {j}u));");
                        let _ = writeln!(out, "{acc}[0] += vec2<f32>(dot({x}, {y0}), dot({x}, {y1}));\n{acc}[1] += vec2<f32>(dot({x8}, {y0}), dot({x8}, {y1}));");
                    }
                    let _ = writeln!(out, "{acc} = mat2x2<f32>(unpack2x16float(pack2x16float({acc}[0])), unpack2x16float(pack2x16float({acc}[1])));");
                }
                self.set(out, *dst, &format!("pack2x16float({acc}[0])"));
                self.set(out, dst.saturating_add(1), &format!("pack2x16float({acc}[1])"));
            }
            // HSETP2: each half compared (in f32, exact for halves), then combined with the
            // accumulator; .H_AND: dst0 gets both halves' and, dst1 its negation (NAK
            // `OpHSetP2::horizontal`). Both results are computed before either is written,
            // as a destination may also be the accumulator.
            Fp16Op::Hsetp2 { dst0, dst1, a, b, cmp, bop, accum, h_and, .. } if *bop != fp16::BoolOp::Reserved3 => {
                let (x, y, lo, hi) = (self.temp(), self.temp(), self.temp(), self.temp());
                let (ex, ey) = (self.h2(a)?, self.h2(b)?);
                let cmp = fcmp16(*cmp);
                let _ = writeln!(out, "let {x} = {ex};\nlet {y} = {ey};");
                let _ = writeln!(out, "let {lo} = {};\nlet {hi} = {};", fcmp(cmp, &format!("{x}.x"), &format!("{y}.x")), fcmp(cmp, &format!("{x}.y"), &format!("{y}.y")));
                let acc = self.p(*accum);
                let (c0, c1) = if *h_and { (format!("({lo} && {hi})"), format!("!({lo} && {hi})")) } else { (lo, hi) };
                let (r0, r1) = (self.temp(), self.temp());
                let _ = writeln!(out, "let {r0} = {};\nlet {r1} = {};", bool_op(*bop as u8, &c0, &acc), bool_op(*bop as u8, &c1, &acc));
                self.set_p(out, *dst0, &r0);
                self.set_p(out, *dst1, &r1);
            }
            // F2FP.PACK_AB: a to the high half, b to the low one.
            Fp16Op::F2fp { dst, a, b, c: RZ, merge_c: false, rnd: fp16::Round::NearestEven, .. } => {
                let (x, y) = (self.f(a)?, self.f(b)?);
                self.set(out, *dst, &format!("pack2x16float(vec2<f32>({y}, {x}))"));
            }
            other => return Err(format!("unsupported: Fp16({other:?})")),
        }
        Ok(())
    }

    fn sat(e: String, sat: bool) -> String {
        if sat { format!("clamp({e}, 0.0, 1.0)") } else { e }
    }

    fn fp32(&mut self, out: &mut String, op: &Fp32Op) -> R<()> {
        match op {
            // FSWZADD: per quad lane (2y + x, from the pixel's position), add (a + b),
            // sub (a - b) or subr (b - a), as NAK lowers derivatives: a = the
            // neighbour's value (SHFL.BFLY), b = this lane's.
            Fp32Op::Fswzadd { dst, a, b, ops, .. } if self.sph.stage == Stage::Pixel => {
                let (x, y) = (format!("f({})", self.r(*a)), format!("f({})", self.r(*b)));
                let e: Vec<String> = ops
                    .iter()
                    .map(|o| match o {
                        fp32::SwzOp::Add => Ok(format!("({x} + {y})")),
                        fp32::SwzOp::Sub => Ok(format!("({x} - {y})")),
                        fp32::SwzOp::Subr => Ok(format!("({y} - {x})")),
                        fp32::SwzOp::Mov2 => Err("FSWZADD mov2".to_string()),
                    })
                    .collect::<R<_>>()?;
                let lane = self.temp();
                let _ = writeln!(out, "let {lane} = (u32(i.frag.x) & 1u) | ((u32(i.frag.y) & 1u) << 1u);");
                self.set(out, *dst, &format!("u(select(select({}, {}, {lane} == 1u), select({}, {}, {lane} == 3u), {lane} >= 2u))", e[0], e[1], e[2], e[3]));
            }
            Fp32Op::Fadd { dst, a, b, sat, .. } => {
                let e = Self::sat(format!("{} + {}", self.f(a)?, self.f(b)?), *sat);
                self.set(out, *dst, &format!("u({e})"));
            }
            Fp32Op::Fmul { dst, a, b, sat, scale, .. } => {
                // The scale field multiplies the product by 2^(scale - 4) (4 = none, NAK's
                // only value). Assumed from use, not proved: a pixel shader computes a 4x4
                // tile index as trunc(frag.xy / screen size * 2^(6 - 4)), then (y << 2) + x.
                let factor = match scale {
                    1..=7 => 2f32.powi(i32::from(*scale) - 4),
                    _ => return Err(format!("FMUL with scale field {scale}")),
                };
                let mut e = format!("{} * {}", self.f(a)?, self.f(b)?);
                if factor != 1.0 {
                    e = format!("({e}) * {factor:?}");
                }
                let e = Self::sat(e, *sat);
                self.set(out, *dst, &format!("u({e})"));
            }
            Fp32Op::Ffma { dst, a, b, c, sat, .. } => {
                let e = Self::sat(format!("fma({}, {}, {})", self.f(a)?, self.f(b)?, self.f(c)?), *sat);
                self.set(out, *dst, &format!("u({e})"));
            }
            Fp32Op::Fmnmx { dst, a, b, min, .. } => {
                let (x, y, m) = (self.f(a)?, self.f(b)?, self.p(*min));
                self.set(out, *dst, &format!("u(select(max({x}, {y}), min({x}, {y}), {m}))"));
            }
            Fp32Op::Fsel { dst, a, b, cond } => {
                let (x, y, c) = (self.f(a)?, self.f(b)?, self.p(*cond));
                self.set(out, *dst, &format!("u(select({y}, {x}, {c}))"));
            }
            Fp32Op::Fsetp { p, q, a, b, cmp, bop, acc, .. } => {
                let (x, y) = (self.f(a)?, self.f(b)?);
                let t = self.temp();
                let _ = writeln!(out, "let {t} = {};", fcmp(*cmp, &x, &y));
                // A copy: the first destination may be the accumulator itself.
                let acc = {
                    let (a, v) = (self.p(*acc), self.temp());
                    let _ = writeln!(out, "let {v} = {a};");
                    v
                };
                let comb = |c: &str| bool_op(*bop as u8, c, &acc);
                let (e1, e2) = (comb(&t), comb(&format!("!{t}")));
                self.set_p(out, *p, &e1);
                self.set_p(out, *q, &e2);
            }
            Fp32Op::Fset { dst, a, b, cmp, bop, acc, .. } => {
                let (x, y) = (self.f(a)?, self.f(b)?);
                let acc = self.p(*acc);
                let c = bool_op(*bop as u8, &fcmp(*cmp, &x, &y), &acc);
                self.set(out, *dst, &format!("select(0u, 0x3f800000u, {c})"));
            }
            Fp32Op::Mufu { dst, func, src, f16: false } => {
                let x = self.f(src)?;
                let e = match func {
                    MufuFn::Rcp => format!("1.0 / {x}"),
                    MufuFn::Rsq => format!("inverseSqrt({x})"),
                    MufuFn::Ex2 => format!("exp2({x})"),
                    MufuFn::Lg2 => format!("log2({x})"),
                    // SM 7.0+ takes the angle in turns (NAK: fsin_normalized_2_pi).
                    MufuFn::Sin => format!("sin({x} * 6.2831853)"),
                    MufuFn::Cos => format!("cos({x} * 6.2831853)"),
                    MufuFn::Sqrt => format!("sqrt({x})"),
                    MufuFn::Tanh => format!("tanh({x})"),
                    other => return Err(format!("MUFU {other:?}")),
                };
                self.set(out, *dst, &format!("u({e})"));
            }
            // F2I to 8, 16 or 32 bits: rounded, then saturated to the destination's range
            // (PTX cvt.rni.u16.f32 and the like), written zero- or sign-extended to 32 bits.
            // NaN gives 0 (.NTZ; without it the QMD's FP32_F2I_NAN_BEHAVIOR may give
            // 0x80000000 instead: NAK's own test accepts both). The extension of the small
            // signed forms is assumed (NAK leaves a 16-bit value's high bits unspecified).
            // ponytail: .FTZ is not applied; it only matters for ceil/floor of denormals.
            Fp32Op::F2i { dst, src, signed, dst_bits: bits @ (8 | 16 | 32), src_bits: 32, rnd, .. } => {
                let (x, t, nan) = (self.temp(), self.temp(), self.temp());
                // NaN tested on the bits: drivers may fold `x != x` away.
                let _ = writeln!(out, "let {x} = {};\nlet {t} = {}({x});\nlet {nan} = (u({x}) & 0x7fffffffu) > 0x7f800000u;", self.f(src)?, round_fn(*rnd));
                // Bounds as exact float literals; the conversions see only in-range values.
                let n = u32::from(*bits);
                let e = if *signed {
                    let h = 1u64 << (n - 1);
                    format!("select(bitcast<u32>(select(i32(max({t}, -{h}.0)), {}i, {t} >= {h}.0)), 0u, {nan})", h - 1)
                } else {
                    let h = 1u64 << n;
                    format!("select(select(u32(max({t}, 0.0)), {}u, {t} >= {h}.0), 0u, {nan})", h - 1)
                };
                self.set(out, *dst, &e);
            }
            Fp32Op::I2f { dst, src, signed, dst_bits: 32, src_bits, sel, .. } => {
                let e = self.i2f(src, *signed, *src_bits, *sel)?;
                self.set(out, *dst, &e);
            }
            Fp32Op::Frnd { dst, src, dst_bits: 32, src_bits: 32, rnd, .. } => {
                let e = format!("u({}({}))", round_fn(*rnd), self.f(src)?);
                self.set(out, *dst, &e);
            }
            other => return Err(format!("unsupported: {other:?}")),
        }
        Ok(())
    }

    fn i2f(&mut self, src: &Operand, signed: bool, bits: u8, sel: u8) -> R<String> {
        let x = self.raw(src)?;
        Ok(match (signed, bits) {
            (true, 32) => format!("u(f32(bitcast<i32>({x})))"),
            (false, 32) => format!("u(f32({x}))"),
            (true, 8 | 16) => format!("u(f32(extractBits(bitcast<i32>({x}), {}u, {bits}u)))", u32::from(sel) * u32::from(bits)),
            (false, 8 | 16) => format!("u(f32(extractBits({x}, {}u, {bits}u)))", u32::from(sel) * u32::from(bits)),
            _ => return Err(format!("I2F from {bits} bits")),
        })
    }

    fn int(&mut self, out: &mut String, op: &IntOp) -> R<()> {
        match op {
            IntOp::IAdd3 { dst, a, b, c, x, carry, overflow } => {
                if overflow[1] != PT {
                    return Err("IADD3 second carry-out".into());
                }
                let (pa, pb, pc) = (self.i(a, *x)?, self.i(b, *x)?, self.i(c, *x)?);
                let mut sum = format!("{pa} + {pb} + {pc}");
                if *x {
                    for cin in carry {
                        if *cin != (Pred { reg: PT, not: true }) {
                            let p = self.p(*cin);
                            sum = format!("{sum} + select(0u, 1u, {p})");
                        }
                    }
                }
                if overflow[0] != PT {
                    if *x {
                        return Err("IADD3.X with a carry-out".into());
                    }
                    // Carry out of the 32-bit sum of three values: any partial wrap.
                    let (s1, s2) = (self.temp(), self.temp());
                    let _ = writeln!(out, "let {s1} = {pa} + {pb};\nlet {s2} = {s1} + {pc};");
                    let e = format!("({s1} < {pa}) || ({s2} < {s1})");
                    self.set_p(out, overflow[0], &e);
                    self.set(out, *dst, &s2);
                } else {
                    self.set(out, *dst, &sum);
                }
                if !*x {
                    self.root_from(*dst, &[a, b, c]);
                }
            }
            IntOp::ISetP { dst, dst1, a, b, cmp, op, signed, ex: false, accum, .. } => {
                let (x, y) = (self.raw(a)?, self.raw(b)?);
                let (x, y) = if *signed { (format!("bitcast<i32>({x})"), format!("bitcast<i32>({y})")) } else { (x, y) };
                let t = self.temp();
                let _ = writeln!(out, "let {t} = {};", icmp(*cmp as u8, &x, &y));
                // A copy: the first destination may be the accumulator itself.
                let acc = {
                    let (a, v) = (self.p(*accum), self.temp());
                    let _ = writeln!(out, "let {v} = {a};");
                    v
                };
                let (e1, e2) = (bool_op(*op as u8, &t, &acc), bool_op(*op as u8, &format!("!{t}"), &acc));
                self.set_p(out, *dst, &e1);
                self.set_p(out, *dst1, &e2);
            }
            IntOp::Lop3 { dst, a, b, c, lut, pand, pdst, psrc } => {
                let known = self.uvals([a, b, c]).map(|x| UExpr::Lop3(*lut, Box::new(x)));
                let (x, y, z) = (self.raw(a)?, self.raw(b)?, self.raw(c)?);
                let e = lop3(*lut, &x, &y, &z);
                if *pdst == PT {
                    self.set(out, *dst, &e);
                } else {
                    // The predicate output is "result != 0", or'ed with the predicate
                    // input (and'ed with .PAND). Assumed from use, not proved: the game
                    // tests bits as `LOP3.LUT P0, RZ, R, mask, RZ, 0xc0, !PT` then `@P0 BRA`.
                    let t = self.temp();
                    let _ = writeln!(out, "let {t} = {e};");
                    let p = self.p(*psrc);
                    let pe = bool_op(if *pand { 0 } else { 1 }, &format!("({t} != 0u)"), &p);
                    self.set_p(out, *pdst, &pe);
                    self.set(out, *dst, &t);
                }
                self.gknown(*dst, known);
            }
            IntOp::Sel { dst, a, b, cond } => {
                let (x, y, c) = (self.raw(a)?, self.raw(b)?, self.p(*cond));
                self.set(out, *dst, &format!("select({y}, {x}, {c})"));
            }
            IntOp::IMad { dst, a, b, c, hi: false, x: false, pdst, .. } if *pdst == PT => {
                let (x, y, z) = (self.i(a, false)?, self.i(b, false)?, self.i(c, false)?);
                self.set(out, *dst, &format!("{x} * {y} + {z}"));
                self.root_from(*dst, &[c]);
            }
            // IMAD.HI: the high word of a × b + c, c being the 64-bit pair c + 1 : c
            // (a constant c is its low word).
            IntOp::IMad { dst, a, b, c, hi: true, signed, x: false, pdst, .. } if *pdst == PT && !c.neg => {
                self.helpers.insert(MUL_HI);
                let (x, y) = (self.i(a, false)?, self.i(b, false)?);
                let (lo, hi) = match c.src {
                    Src::Reg(r) if r != RZ => (self.r(r), self.r(r.saturating_add(1))),
                    _ => (self.raw(c)?, "0u".to_string()),
                };
                let p = self.temp();
                let _ = writeln!(out, "let {p} = {x} * {y};");
                self.set(out, *dst, &format!("mul_hi({x}, {y}, {signed}) + {hi} + select(0u, 1u, {p} + {lo} < {p})"));
            }
            IntOp::IMnMx { dst, a, b, signed, min } => {
                let (x, y, m) = (self.raw(a)?, self.raw(b)?, self.p(*min));
                let e = if *signed {
                    format!("bitcast<u32>(select(max(bitcast<i32>({x}), bitcast<i32>({y})), min(bitcast<i32>({x}), bitcast<i32>({y})), {m}))")
                } else {
                    format!("select(max({x}, {y}), min({x}, {y}), {m})")
                };
                self.set(out, *dst, &e);
            }
            IntOp::Prmt { dst, a, sel, c, mode: int::PrmtMode::Index } => {
                self.helpers.insert(PRMT);
                let (x, s, z) = (self.raw(a)?, self.raw(sel)?, self.raw(c)?);
                self.set(out, *dst, &format!("prmt({x}, {s}, {z})"));
            }
            IntOp::Shf { dst, lo, shift, hi, ty, wrap, right, dst_hi } => {
                self.helpers.insert(SHF);
                let (l, s, h) = (self.raw(lo)?, self.raw(shift)?, self.raw(hi)?);
                let (wide, signed) = match ty {
                    int::ShfType::S64 => (true, true),
                    int::ShfType::U64 => (true, false),
                    int::ShfType::S32 => (false, true),
                    int::ShfType::U32 => (false, false),
                };
                let limit = if wide { 63 } else { 31 };
                let amount = if *wrap { format!("({s} & {limit}u)") } else { format!("min({s}, {}u)", limit + 1) };
                let call = if *right { format!("shf_r({l}, {h}, {amount}, {signed})") } else { format!("shf_l({l}, {h}, {amount})") };
                self.set(out, *dst, &format!("{call}.{}", if *dst_hi { "y" } else { "x" }));
            }
            // LEA, with the carry out of the add when asked (the low half of a 64-bit address).
            IntOp::Lea { dst, a, b, c, shift, hi, x: false, neg_shifted, overflow, carry } if *carry == (Pred { reg: PT, not: true }) => {
                let (lo, add, high) = (self.raw(a)?, self.i(b, false)?, self.raw(c)?);
                let s = u32::from(*shift);
                let shifted = match (*hi, s) {
                    (false, _) => format!("({lo} << {s}u)"),
                    (true, 0) => high,
                    (true, _) => format!("(({high} << {s}u) | ({lo} >> {}u))", 32 - s),
                };
                let shifted = if *neg_shifted { format!("(0u - {shifted})") } else { shifted };
                let (t, sum) = (self.temp(), self.temp());
                let _ = writeln!(out, "let {t} = {shifted};\nlet {sum} = {t} + {add};");
                self.set_p(out, *overflow, &format!("{sum} < {t}"));
                self.set(out, *dst, &sum);
                if !*hi {
                    self.root_from(*dst, &[b]);
                }
            }
            IntOp::Sgxt { dst, a, bits, signed, wrap } => {
                let (x, n) = (self.raw(a)?, self.raw(bits)?);
                let n = if *wrap { format!("({n} & 31u)") } else { n };
                let ext = if *signed { format!("bitcast<u32>(extractBits(bitcast<i32>({x}), 0u, min({n}, 32u)))") } else { format!("extractBits({x}, 0u, min({n}, 32u))") };
                self.set(out, *dst, &format!("select({x}, {ext}, {n} < 32u)"));
            }
            // FLO (PTX bfind): the highest set bit, or for signed values the highest bit
            // unlike the sign; 0xffffffff when none; .SH gives 31 - position instead.
            // WGSL's firstLeadingBit has the same definition, for u32 and i32.
            IntOp::Flo { dst, src, not, signed, sh, pdst } if *pdst == PT => {
                let x = self.raw(src)?;
                let x = if *not { format!("(~{x})") } else { x };
                let (t, r) = (self.temp(), self.temp());
                let found = if *signed { format!("bitcast<u32>(firstLeadingBit(bitcast<i32>({x})))") } else { format!("firstLeadingBit({x})") };
                let _ = writeln!(out, "let {t} = {found};\nlet {r} = select({t}, 31u - {t}, {} && {t} != 0xffffffffu);", *sh);
                self.set(out, *dst, &r);
            }
            other => return Err(format!("unsupported: {other:?}")),
        }
        Ok(())
    }

    fn moves(&mut self, out: &mut String, op: &MovesOp) -> R<()> {
        match op {
            MovesOp::Mov { dst, src, quad_lanes: 0xf } => {
                let known = self.uval(src);
                let e = self.raw(src)?;
                self.set(out, *dst, &e);
                self.gknown(*dst, known);
                self.root_from(*dst, &[src]);
            }
            MovesOp::UMov { dst, src } => {
                let known = self.uval(src);
                let e = self.raw(src)?;
                self.set_ur(out, *dst, &e);
                self.known(*dst, known);
            }
            MovesOp::Uldc { dst, src: Src::CBuf { bank, offset }, mem } => {
                let offset = u32::from(*offset);
                let small = match mem {
                    moves::MemType::U8 => Some((8, false)),
                    moves::MemType::S8 => Some((8, true)),
                    moves::MemType::U16 => Some((16, false)),
                    moves::MemType::S16 => Some((16, true)),
                    _ => None,
                };
                if let Some((bits, signed)) = small {
                    let e = self.cb_small(*bank, offset, bits, signed)?;
                    self.set_ur(out, *dst, &e);
                } else {
                    let words = match mem {
                        moves::MemType::B32 => 1,
                        moves::MemType::B64 => 2,
                        other => return Err(format!("ULDC of {other:?}")),
                    };
                    for k in 0..words {
                        let e = self.cb(*bank, offset + 4 * k)?;
                        self.set_ur(out, dst + k as u8, &e);
                        if let Ok(at) = u16::try_from(offset + 4 * k) {
                            self.uexpr.insert(dst + k as u8, UExpr::CBuf(*bank, at));
                        }
                    }
                }
            }
            // S2R: SR_TID.XYZ (0x21..0x23) and SR_CTAID.XYZ (0x25..0x27) in compute.
            MovesOp::S2r { dst, sr: sr @ (0x21..=0x23 | 0x25..=0x27) } if self.sph.stage == Stage::Compute => {
                let v = if *sr < 0x25 { "lid" } else { "wid" };
                let c = ["x", "y", "z"][usize::from((sr - 1) % 4)];
                self.set(out, *dst, &format!("{v}.{c}"));
            }
            // SR_LANEID: the thread's lane in its warp (a 32-wide subgroup on NVIDIA).
            MovesOp::S2r { dst, sr: 0 } if self.sph.stage == Stage::Compute => {
                self.lane = true;
                self.set(out, *dst, "lane");
            }
            MovesOp::Cs2r { dst, sr: 0xff, wide } => {
                self.set(out, *dst, "0u");
                if *wide {
                    self.set(out, dst.saturating_add(1), "0u");
                }
            }
            MovesOp::Plop3 { dst, src, src2_uniform, lut } => {
                let a = self.p(src[0]);
                let b = self.p(src[1]);
                let c = if *src2_uniform { self.up(src[2]) } else { self.p(src[2]) };
                for (d, l) in dst.iter().zip(lut) {
                    let e = format!("((({l:#x}u >> ((u32({a}) << 2u) | (u32({b}) << 1u) | u32({c}))) & 1u) != 0u)");
                    self.set_p(out, *d, &e);
                }
            }
            // As SHF, on the uniform datapath.
            MovesOp::UShf { dst, low, shift, high, ty, wrap, right, hi } => {
                self.helpers.insert(SHF);
                let (l, s, h) = (self.raw(low)?, self.raw(shift)?, self.raw(high)?);
                let (wide, signed) = match ty {
                    moves::ShfType::S64 => (true, true),
                    moves::ShfType::U64 => (true, false),
                    moves::ShfType::S32 => (false, true),
                    moves::ShfType::U32 => (false, false),
                };
                let limit = if wide { 63 } else { 31 };
                let amount = if *wrap { format!("({s} & {limit}u)") } else { format!("min({s}, {}u)", limit + 1) };
                let call = if *right { format!("shf_r({l}, {h}, {amount}, {signed})") } else { format!("shf_l({l}, {h}, {amount})") };
                let known = self.uvals([low, shift, high]).map(|parts| UExpr::Shf { parts: Box::new(parts), limit, wrap: *wrap, right: *right, signed, hi: *hi });
                self.set_ur(out, *dst, &format!("{call}.{}", if *hi { "y" } else { "x" }));
                self.known(*dst, known);
            }
            // R2P: predicate i takes bit i of the register, for the mask's bits.
            MovesOp::R2p { src, mask } => {
                let v = self.r(*src);
                for i in 0..7u8 {
                    if mask >> i & 1 != 0 {
                        self.set_p(out, i, &format!("(({v} >> {i}u) & 1u) != 0u"));
                    }
                }
            }
            // P2R: the mask's bits of the register replaced by predicates P0..P6 (bit i =
            // Pi). Guessed from Maxwell's P2R (no NAK encoder): a byte-select form, if
            // Ampere has one, is not decoded; a mask past P6 is refused.
            MovesOp::P2r { dst, a, mask } if *mask <= 0x7f => {
                let x = self.r(*a);
                let mut e = format!("({x} & {:#x}u)", !*mask);
                for i in (0..7u8).filter(|i| mask >> i & 1 != 0) {
                    let p = self.p(Pred { reg: i, not: false });
                    e = format!("{e} | select(0u, {:#x}u, {p})", 1u32 << i);
                }
                self.set(out, *dst, &format!("({e})"));
            }
            // VOTE / VOTEU: all, any or all-equal of the predicate over the warp's active
            // lanes (WGSL subgroup operations count only active invocations, as the
            // hardware does), and its ballot (VOTEU: to a uniform register and predicate;
            // RZ / URZ / PT: not written).
            MovesOp::Vote { uniform, dst, mode, vote, pred } if *mode != moves::VoteMode::Reserved3 => {
                let p = self.p(*pred);
                let (v, b) = (self.temp(), self.temp());
                let e = match mode {
                    moves::VoteMode::All => format!("subgroupAll({p})"),
                    moves::VoteMode::Any => format!("subgroupAny({p})"),
                    _ => format!("(subgroupAll({p}) || !subgroupAny({p}))"),
                };
                let _ = writeln!(out, "let {v} = {e};\nlet {b} = subgroupBallot({p}).x;");
                if *uniform {
                    self.set_up(out, *vote, &v);
                    self.set_ur(out, *dst, &b);
                } else {
                    self.set_p(out, *vote, &v);
                    self.set(out, *dst, &b);
                }
            }
            // As LOP3, on the uniform datapath.
            MovesOp::ULop3 { dst, a, b, c, lut, pand, pdst, psrc } => {
                let known = self.uvals([a, b, c]).map(|x| UExpr::Lop3(*lut, Box::new(x)));
                let (x, y, z) = (self.raw(a)?, self.raw(b)?, self.raw(c)?);
                let e = lop3(*lut, &x, &y, &z);
                if *pdst == PT {
                    self.set_ur(out, *dst, &e);
                } else {
                    let t = self.temp();
                    let _ = writeln!(out, "let {t} = {e};");
                    let p = self.up(*psrc);
                    let pe = bool_op(if *pand { 0 } else { 1 }, &format!("({t} != 0u)"), &p);
                    self.set_up(out, *pdst, &pe);
                    self.set_ur(out, *dst, &t);
                }
                self.known(*dst, known);
            }
            // SHFL.BFLY: the value of lane (lane ^ n). In a quad segment
            // (c = 0x1c03: segment mask 0x1c, last lane 3) these are the quad
            // swaps (lanes 2y + x, as Vulkan's quads); over the whole warp
            // (c = 0x1f), a subgroup shuffle.
            MovesOp::Shfl { dst, src, lane: Src::Imm(n), c: Src::Imm(c), mode: moves::ShflMode::Bfly, in_bounds } => {
                let v = self.r(*src);
                let e = match (n, c) {
                    (1, 0x1c03) => format!("quadSwapX({v})"),
                    (2, 0x1c03) => format!("quadSwapY({v})"),
                    (3, 0x1c03) => format!("quadSwapDiagonal({v})"),
                    (n, 0x1f) => format!("subgroupShuffleXor({v}, {n}u)"),
                    (n, c) => return Err(format!("SHFL.BFLY {n} with c {c:#x}")),
                };
                self.set(out, *dst, &e);
                self.set_p(out, *in_bounds, "true");
            }
            MovesOp::UIAdd3 { dst, a, b, c, x: false, overflow, .. } if overflow == &[PT, PT] => {
                let known = self.uvals([a, b, c]).map(|x| UExpr::Add(Box::new(x)));
                let (x, y, z) = (self.i(a, false)?, self.i(b, false)?, self.i(c, false)?);
                self.set_ur(out, *dst, &format!("{x} + {y} + {z}"));
                self.known(*dst, known);
            }
            MovesOp::UISetP { dst, a, b, cmp, set_op, signed, ex: false, accum, .. } => {
                let (x, y) = (self.raw(a)?, self.raw(b)?);
                let (x, y) = if *signed { (format!("bitcast<i32>({x})"), format!("bitcast<i32>({y})")) } else { (x, y) };
                let t = self.temp();
                let _ = writeln!(out, "let {t} = {};", icmp(*cmp as u8, &x, &y));
                let acc = self.up(*accum);
                let (e1, e2) = (bool_op(*set_op as u8, &t, &acc), bool_op(*set_op as u8, &format!("!{t}"), &acc));
                self.set_up(out, dst[0], &e1);
                self.set_up(out, dst[1], &e2);
            }
            other => return Err(format!("unsupported: {other:?}")),
        }
        Ok(())
    }

    fn memory(&mut self, out: &mut String, op: &MemoryOp) -> R<()> {
        match op {
            MemoryOp::Ipa { dst, attr, loc: InterpLoc::Default, freq, offset: RZ, pred_dst: PT } => {
                let e = match attr {
                    0x70..=0x7c => format!("u(i.frag.{})", ["x", "y", "z", "w"][usize::from((attr - 0x70) / 4)]),
                    0x80..=0x27c => {
                        let (v, c) = (usize::from((attr - 0x80) / 16), usize::from(attr / 4 % 4));
                        let mode = self.sph.interpolation()[v][c];
                        let x = format!("i.v{v}[{c}]");
                        match (mode, freq) {
                            // Pass on a perspective input is the value over w, linearly interpolated.
                            (Interp::Perspective, InterpFreq::Pass) => format!("u({x} * i.frag.w)"),
                            (Interp::ScreenLinear, InterpFreq::Pass) | (Interp::Constant, _) => format!("u({x})"),
                            (m, f) => return Err(format!("IPA {f:?} of a {m:?} input")),
                        }
                    }
                    // The front face (NAK_ATTR_FRONT_FACE): non-zero when front-facing.
                    0x3fc => "select(0u, 0xffffffffu, i.front)".to_string(),
                    _ => return Err(format!("IPA of attribute {attr:#x}")),
                };
                self.set(out, *dst, &e);
            }
            MemoryOp::Ald { dst, offset: RZ, vtx: RZ, attr, comps, patch: false, phys: false, output: false } if self.sph.stage == Stage::Vertex => {
                for k in 0..u16::from(*comps) {
                    let a = attr + 4 * k;
                    let e = match a {
                        0x80..=0x27c => {
                            let v = (a - 0x80) / 16;
                            // An integer attribute's bits are its value; a float one's are reinterpreted.
                            if self.int_attributes.contains_key(&(v as u8)) { format!("bitcast<u32>(i.a{v}[{}])", a / 4 % 4) } else { format!("u(i.a{v}[{}])", a / 4 % 4) }
                        }
                        0x2f8 => "i.iid".to_string(),
                        0x2fc => "i.vid".to_string(),
                        _ => return Err(format!("ALD of attribute {a:#x}")),
                    };
                    self.set(out, dst.saturating_add(k as u8), &e);
                }
            }
            MemoryOp::Ast { data, offset: RZ, vtx: RZ, attr, comps, patch: false, phys: false } if self.sph.stage == Stage::Vertex => {
                for k in 0..u16::from(*comps) {
                    let a = attr + 4 * k;
                    let v = if *data == RZ { "0u".to_string() } else { self.r(data + k as u8) };
                    match a {
                        0x70..=0x7c => {
                            let _ = writeln!(out, "o.pos[{}] = f({v});", (a - 0x70) / 4);
                        }
                        0x80..=0x27c => {
                            let vec = (a - 0x80) / 16;
                            if self.sph.outputs()[usize::from(vec)] == 0 {
                                return Err(format!("AST to {a:#x}, an output the SPH does not declare"));
                            }
                            let _ = writeln!(out, "o.v{vec}[{}] = f({v});", a / 4 % 4);
                        }
                        // Point size: not a WGSL output.
                        0x6c => {}
                        // ponytail: clip and cull distances (NAK_ATTR_CLIP_CULL_DIST_0..7) are
                        // dropped; wgpu's CLIP_DISTANCES feature can carry them when a scene needs it.
                        0x2c0..=0x2dc => {}
                        _ => return Err(format!("AST to attribute {a:#x}")),
                    }
                }
            }
            // SULD: a typed image load, as a texel fetch of the image's texture; the enabled
            // channels go to consecutive registers, and no access faults.
            // From an image the program also stores to, the load reads the image itself.
            MemoryOp::Suld { dst, coord, handle, dim, mask, pred_dst, .. } => {
                let (tdim, ty) = match dim {
                    ImageDim::D1 => (TexDim::D1, "texture_1d<f32>"),
                    ImageDim::D2 => (TexDim::D2, "texture_2d<f32>"),
                    ImageDim::D3 => (TexDim::D3, "texture_3d<f32>"),
                    ImageDim::D2Array => (TexDim::Array2D, "texture_2d_array<f32>"),
                    d => return Err(format!("SULD from a {d:?} image")),
                };
                let (at, coords, layer) = self.image_coords(*coord, *dim)?;
                let v = if self.read_write.contains(&(handle.bank, handle.offset)) {
                    let slot = self.image_slot(*handle, *dim)?;
                    // Outside the image: zeros (WGSL may return any texel there).
                    let inside = image_inside(slot, &at, coords, layer.as_deref());
                    format!("select(vec4<u32>(0u), bitcast<vec4<u32>>(textureLoad(img{slot}, {at}{})), {inside})", layer.as_ref().map(|l| format!(", {l}")).unwrap_or_default())
                } else {
                    let slot = self.texture_slot(TexHandle::CBuf { bank: handle.bank, offset: handle.offset }, tdim, ty, false, true);
                    self.swizzled(slot, &format!("bitcast<vec4<u32>>(textureLoad(tex{slot}, {at}{}, 0))", layer.as_ref().map(|l| format!(", {l}")).unwrap_or_default()))
                };
                let t = self.temp();
                let _ = writeln!(out, "let {t} = {v};");
                for (k, c) in (0..4u8).filter(|c| mask >> c & 1 != 0).enumerate() {
                    if *dst != RZ {
                        self.set(out, dst.saturating_add(k as u8), &format!("{t}[{c}]"));
                    }
                }
                self.set_p(out, *pred_dst, "false");
            }
            MemoryOp::Sust { coord, data, handle, dim, mask: 0xf, .. } if self.sph.stage == Stage::Compute => {
                let slot = self.image_slot(*handle, *dim)?;
                let (at, coords, layer) = self.image_coords(*coord, *dim)?;
                // The image's texel type: float, or the register bits for integer formats.
                let format = self.images.get(slot).cloned().unwrap_or_default();
                let (ty, zero, cast) = if format.ends_with("uint") { ("vec4<u32>", "0u", "") } else if format.ends_with("sint") { ("vec4<i32>", "0i", "bitcast<i32>") } else { ("vec4<f32>", "0.0", "f") };
                let v: Vec<String> = (0..4).map(|k| if *data == RZ { zero.to_string() } else { format!("{cast}({})", self.r(data + k)) }).collect();
                // A store outside the image is dropped, as on the console: WGSL may instead
                // write any texel (the edge, racing with its own store).
                let inside = image_inside(slot, &at, coords, layer.as_deref());
                let layer = layer.map(|l| format!(", {l}")).unwrap_or_default();
                let _ = writeln!(out, "if ({inside}) {{ textureStore(img{slot}, {at}{layer}, {ty}({})); }}", v.join(", "));
            }
            // LDG through a pointer from a constant bank: the address minus that pointer
            // indexes the buffer bound for it (`Interface::buffers`). Buffers are under 4 GiB:
            // the low words are enough. The base is a register pair derived from the pointer,
            // or a uniform pair holding it (ULDC.64) plus a 32-bit register, taken as
            // unsigned (NAK's uniform-plus-register global address).
            MemoryOp::Ldg { dst, addr: Addr { reg, reg64, ureg, ureg64, offset }, ty, pred, pred_dst, .. } if self.sph.stage == Stage::Compute && *pred == Pred::TRUE => {
                let ((bank, at), reg) = match (*reg64, *ureg) {
                    (true, None) => (*self.roots.get(reg).ok_or("LDG from an address not derived from a constant-bank pointer")?, self.r(*reg)),
                    (false, Some(u)) if *ureg64 => {
                        let root = match (self.uexpr.get(&u), self.uexpr.get(&u.saturating_add(1))) {
                            (Some(&UExpr::CBuf(b, o)), Some(&UExpr::CBuf(b1, o1))) if o % 8 == 0 && b1 == b && o.checked_add(4) == Some(o1) => (b, o),
                            _ => return Err("LDG from a uniform base not loaded from a constant-bank pointer".into()),
                        };
                        (root, format!("({} + {})", self.ur(u), self.r(*reg)))
                    }
                    _ => return Err(format!("LDG address form {:?}", (reg64, ureg, ureg64))),
                };
                let n = match self.iface.buffers.iter().position(|b| *b == (bank, at)) {
                    Some(n) => n,
                    None => {
                        let g = self.group();
                        let n = self.iface.buffers.len();
                        let _ = writeln!(self.decls, "@group({g}) @binding({}) var<storage, read> gm{n}: array<u32>;", BUFFER_BINDING + n as u32);
                        self.iface.buffers.push((bank, at));
                        n
                    }
                };
                let base = self.cb(bank, u32::from(at))?;
                let t = self.temp();
                let _ = writeln!(out, "let {t} = {reg} + {:#x}u - {base};", *offset as u32);
                match small_mem(*ty) {
                    Some((bits, signed)) => {
                        let w = format!("gm{n}[{t} >> 2u]");
                        let e = if signed { format!("bitcast<u32>(extractBits(bitcast<i32>({w}), ({t} & 3u) * 8u, {bits}u))") } else { format!("extractBits({w}, ({t} & 3u) * 8u, {bits}u)") };
                        self.set(out, *dst, &e);
                    }
                    None => {
                        let words = mem_words(*ty)?;
                        let v = self.temp();
                        let loads: Vec<String> = (0..words).map(|k| format!("gm{n}[({t} >> 2u) + {k}u]")).collect();
                        let _ = writeln!(out, "let {v} = array<u32, {words}>({});", loads.join(", "));
                        for k in 0..words {
                            self.set(out, dst.saturating_add(k as u8), &format!("{v}[{k}]"));
                        }
                    }
                }
                self.set_p(out, *pred_dst, "true");
            }
            // LDSM: row r of matrix i is at the address lane 8i + r gives; lane t takes row
            // t / 4, elements 2 (t % 4) and + 1 (MT8N8, transposed: column t / 4 of rows
            // 2 (t % 4) and + 1), as one register per matrix.
            MemoryOp::Ldsm { dst, addr, count, shape: shape @ (0 | 1) } if self.sph.stage == Stage::Compute => {
                self.lane = true;
                let at = self.shared_addr(out, addr, Stride::X1)?;
                let half = |row: &str| format!("extractBits(atomicLoad(&shm[({row} + (lane / 4u) * 2u) >> 2u]), (({row} + (lane / 4u) * 2u) & 2u) * 8u, 16u)");
                let mut words = Vec::new();
                for i in 0..u32::from(*count) {
                    let (r0, r1) = (self.temp(), self.temp());
                    let v = if *shape == 0 {
                        let _ = writeln!(out, "let {r0} = subgroupShuffle({at}, {}u + lane / 4u);", 8 * i);
                        format!("atomicLoad(&shm[({r0} + (lane % 4u) * 4u) >> 2u])")
                    } else {
                        let _ = writeln!(out, "let {r0} = subgroupShuffle({at}, {}u + 2u * (lane % 4u));", 8 * i);
                        let _ = writeln!(out, "let {r1} = subgroupShuffle({at}, {}u + 2u * (lane % 4u) + 1u);", 8 * i);
                        format!("({} | ({} << 16u))", half(&r0), half(&r1))
                    };
                    let w = self.temp();
                    let _ = writeln!(out, "let {w} = {v};");
                    words.push(w);
                }
                for (i, w) in words.iter().enumerate() {
                    self.set(out, dst.saturating_add(i as u8), w);
                }
            }
            // Shared memory: words of `shm`; a byte or half-word is a part of its word.
            MemoryOp::Lds { dst, addr, ty, stride } if self.sph.stage == Stage::Compute => {
                let t = self.shared_addr(out, addr, *stride)?;
                let word = |k: u32| format!("atomicLoad(&shm[({t} >> 2u) + {k}u])");
                match small_mem(*ty) {
                    Some((bits, signed)) => {
                        let e = if signed { format!("bitcast<u32>(extractBits(bitcast<i32>({}), ({t} & 3u) * 8u, {bits}u))", word(0)) } else { format!("extractBits({}, ({t} & 3u) * 8u, {bits}u)", word(0)) };
                        self.set(out, *dst, &e);
                    }
                    None => {
                        let n = mem_words(*ty)?;
                        let v = self.temp();
                        let loads: Vec<String> = (0..n).map(word).collect();
                        let _ = writeln!(out, "let {v} = array<u32, {n}>({});", loads.join(", "));
                        for k in 0..n {
                            if *dst != RZ {
                                self.set(out, dst.saturating_add(k as u8), &format!("{v}[{k}]"));
                            }
                        }
                    }
                }
            }
            MemoryOp::Sts { data, addr, ty, stride } if self.sph.stage == Stage::Compute => {
                let t = self.shared_addr(out, addr, *stride)?;
                match small_mem(*ty) {
                    Some((bits, _)) => {
                        let d = self.r(*data);
                        let _ = writeln!(out, "atomicAnd(&shm[{t} >> 2u], ~({:#x}u << (({t} & 3u) * 8u)));", (1u32 << bits) - 1);
                        let _ = writeln!(out, "atomicOr(&shm[{t} >> 2u], ({d} & {:#x}u) << (({t} & 3u) * 8u));", (1u32 << bits) - 1);
                    }
                    None => {
                        for k in 0..mem_words(*ty)? {
                            let d = if *data == RZ { "0u".to_string() } else { self.r(data + k as u8) };
                            let _ = writeln!(out, "atomicStore(&shm[({t} >> 2u) + {k}u], {d});");
                        }
                    }
                }
            }
            MemoryOp::Atoms { dst, data, addr, op, ty: ty @ (AtomType::U32 | AtomType::S32), stride } if self.sph.stage == Stage::Compute => {
                let t = self.shared_addr(out, addr, *stride)?;
                let d = self.r(*data);
                let at = format!("&shm[{t} >> 2u]");
                let signed = *ty == AtomType::S32;
                let e = match op {
                    AtomOp::Add => format!("atomicAdd({at}, {d})"),
                    AtomOp::And => format!("atomicAnd({at}, {d})"),
                    AtomOp::Or => format!("atomicOr({at}, {d})"),
                    AtomOp::Xor => format!("atomicXor({at}, {d})"),
                    AtomOp::Exch => format!("atomicExchange({at}, {d})"),
                    AtomOp::Min if !signed => format!("atomicMin({at}, {d})"),
                    AtomOp::Max if !signed => format!("atomicMax({at}, {d})"),
                    AtomOp::Min | AtomOp::Max | AtomOp::Inc | AtomOp::Dec => {
                        self.helpers.insert(SHM_CAS);
                        let code = match op {
                            AtomOp::Min => 0,
                            AtomOp::Max => 1,
                            AtomOp::Inc => 2,
                            _ => 3,
                        };
                        format!("shm_cas({t} >> 2u, {code}u, {d})")
                    }
                    AtomOp::Other(o) => return Err(format!("ATOMS operation {o}")),
                };
                let v = self.temp();
                let _ = writeln!(out, "let {v} = {e};");
                self.set(out, *dst, &v);
            }
            MemoryOp::Ldc { dst, index, bank, offset, mode: LdcMode::Indexed, ty } => {
                if let Some((bits, signed)) = small_mem(*ty) {
                    let e = if *index == RZ {
                        self.cb_small(*bank, u32::from(*offset), bits, signed)?
                    } else {
                        // The byte address wraps in the bank, as the word loads below.
                        // ponytail: a half-word straddling two words reads only the first.
                        let ix = self.r(*index);
                        self.iface.cbufs.insert(*bank);
                        self.reach(*bank, 0x10000);
                        let t = self.temp();
                        let _ = writeln!(out, "let {t} = ({ix} + {}u) & 0xffffu;", u32::from(*offset));
                        let word = format!("cb{bank}[{t} >> 4u][({t} >> 2u) & 3u]");
                        if signed { format!("bitcast<u32>(extractBits(bitcast<i32>({word}), ({t} & 3u) * 8u, {bits}u))") } else { format!("extractBits({word}, ({t} & 3u) * 8u, {bits}u)") }
                    };
                    self.set(out, *dst, &e);
                    return Ok(());
                }
                let words = match ty {
                    MemType::B32 => 1u32,
                    MemType::B64 => 2,
                    other => return Err(format!("LDC of {other:?}")),
                };
                if *index == RZ {
                    for k in 0..words {
                        let at = u32::from(*offset) + 4 * k;
                        let e = self.cb(*bank, at)?;
                        self.set(out, dst.saturating_add(k as u8), &e);
                        self.gknown(dst.saturating_add(k as u8), u16::try_from(at).ok().map(|at| UExpr::CBuf(*bank, at)));
                    }
                } else {
                    let ix = self.r(*index);
                    self.iface.cbufs.insert(*bank);
                    self.reach(*bank, 0x10000);
                    for k in 0..words {
                        let t = self.temp();
                        let _ = writeln!(out, "let {t} = ({ix} + {}u) & 0xffffu;", u32::from(*offset) + 4 * k);
                        self.set(out, dst.saturating_add(k as u8), &format!("cb{bank}[{t} >> 4u][({t} >> 2u) & 3u]"));
                    }
                }
            }
            other => return Err(format!("unsupported: {other:?}")),
        }
        Ok(())
    }

    /// The image slot of a constant-bank handle, one per handle, declared at its
    /// first use: written, or read and written (`read_write`).
    fn image_slot(&mut self, handle: CbHandle, dim: ImageDim) -> R<usize> {
        let ty = match dim {
            ImageDim::D1 => "texture_storage_1d",
            ImageDim::D2 => "texture_storage_2d",
            ImageDim::D3 => "texture_storage_3d",
            ImageDim::D2Array => "texture_storage_2d_array",
            d => return Err(format!("a {d:?} image")),
        };
        let h = Handle::CBuf { bank: handle.bank, offset: handle.offset };
        if let Some(slot) = self.iface.images.iter().position(|i| i.handle == h) {
            return Ok(slot);
        }
        let slot = self.iface.images.len();
        let read = self.read_write.contains(&(handle.bank, handle.offset));
        self.iface.images.push(Image { handle: h, dim, read });
        let format = self.images.get(slot).cloned().unwrap_or_else(|| "rgba16float".into());
        let g = self.group();
        let access = if read { "read_write" } else { "write" };
        let _ = writeln!(self.decls, "@group({g}) @binding({}) var img{slot}: {ty}<{format}, {access}>;", IMAGE_BINDING + slot as u32);
        Ok(slot)
    }

    /// An image access's integer coordinates from the registers at `base`: the
    /// position (and how many components it has), then the layer of an array
    /// image (x, y, layer: surface instructions put the layer last).
    fn image_coords(&mut self, base: u8, dim: ImageDim) -> R<(String, usize, Option<String>)> {
        let (coords, layered) = match dim {
            ImageDim::D1 => (1, false),
            ImageDim::D2 => (2, false),
            ImageDim::D3 => (3, false),
            ImageDim::D2Array => (2, true),
            d => return Err(format!("a {d:?} image")),
        };
        let mut cs: Vec<String> = (0..coords as u8 + u8::from(layered)).map(|k| format!("i32({})", if base == RZ { "0u".to_string() } else { self.r(base + k) })).collect();
        let layer = layered.then(|| cs.pop()).flatten();
        let at = if coords == 1 { cs[0].clone() } else { format!("vec{coords}<i32>({})", cs.join(", ")) };
        Ok((at, coords, layer))
    }

    /// A texture slot for an instruction's handle: its declaration, and its
    /// sampler's unless it is fetched or queried.
    fn texture_slot(&mut self, handle: TexHandle, dim: TexDim, ty: &str, compare: bool, fetch: bool) -> usize {
        let slot = self.iface.textures.len();
        let handle = match handle {
            TexHandle::CBuf { bank, offset } => Handle::CBuf { bank, offset },
            TexHandle::UReg { ureg, .. } => match self.uexpr.get(&ureg) {
                Some(e) => Handle::Uniform(e.clone()),
                None => Handle::UReg(ureg),
            },
            TexHandle::Bindless => Handle::Bindless,
        };
        self.iface.textures.push(Texture { handle, dim, compare, fetch, query: false });
        let g = self.group();
        let _ = writeln!(self.decls, "@group({g}) @binding({}) var tex{slot}: {ty};", 32 + 2 * slot);
        if !fetch {
            let _ = writeln!(self.decls, "@group({g}) @binding({}) var smp{slot}: {};", 33 + 2 * slot, if compare { "sampler_comparison" } else { "sampler" });
        }
        slot
    }

    /// The enabled components of `v` (a vec4 or, with `scalar`, one value), in
    /// order, to dst, dst + 1, dst2, dst2 + 1.
    fn tex_results(&mut self, out: &mut String, v: &str, scalar: bool, mask: u8, (dst, dst2): (u8, u8), f16: bool) -> R<()> {
        let comps: Vec<usize> = (0..4).filter(|c| mask >> c & 1 != 0).collect();
        // .F16: the enabled components as halves, two per register: the first pair to
        // dst, the second to dst2 (dst + 1 when there is no dst2); nothing else is written.
        if f16 {
            let value = |c: Option<&usize>| match c {
                Some(_) if scalar => format!("f({v})"),
                Some(c) => format!("f({v}[{c}])"),
                None => "0.0".to_string(),
            };
            let pairs: Vec<String> = comps.chunks(2).map(|p| format!("pack2x16float(vec2<f32>({}, {}))", value(p.first()), value(p.get(1)))).collect();
            let second = if dst2 != RZ { dst2 } else { dst.saturating_add(1) };
            for (d, e) in [dst, second].into_iter().zip(&pairs) {
                self.set(out, d, e);
            }
            return Ok(());
        }
        let dsts = [Some(dst), Some(dst).filter(|d| *d != RZ).map(|d| d + 1), Some(dst2), Some(dst2).filter(|d| *d != RZ).map(|d| d + 1)];
        for (k, c) in comps.iter().enumerate() {
            let Some(Some(d)) = dsts.get(k) else { return Err("texture destination".into()) };
            let value = if scalar { v.to_string() } else { format!("{v}[{c}]") };
            self.set(out, *d, &value);
        }
        Ok(())
    }

    fn texture(&mut self, out: &mut String, op: &TextureOp) -> R<()> {
        // `gather`: TLD4's component (its sources are TEX's, without a LOD).
        let (io, dim, fetch, lod, scr, aoffi, dc, f16, gather) = match op {
            TextureOp::Tex { io, dim, lod, scr, aoffi, dc, f16, .. } => (io, *dim, false, *lod, *scr, *aoffi, *dc, *f16, None),
            TextureOp::Tld { io, dim, lod, scr, aoffi, ms: false, f16 } => (io, *dim, true, *lod, *scr, *aoffi, false, *f16, None),
            TextureOp::Tld4 { io, dim: dim @ (TexDim::D2 | TexDim::Array2D | TexDim::Cube | TexDim::ArrayCube), comp, offset: offset @ (Tld4Offset::None | Tld4Offset::AOffI), scr, dc } => (io, *dim, false, LodMode::Auto, *scr, matches!(offset, Tld4Offset::AOffI), *dc, false, Some(*comp)),
            TextureOp::Txd { io, dim, aoffi } => return self.txd(out, io, *dim, *aoffi),
            // The size of a level (the texture is taken as 2D): width, height, 1, levels.
            // TXQ: the texture declared as the type its header has (`Options::queried`).
            TextureOp::Txq { dst, dst2, handle, src, query: TexQuery::Dimension, mask, .. } => {
                let three_d = self.queried.get(&self.iface.textures.len()) == Some(&TexDim::D3);
                let (dim, ty, depth) = if three_d { (TexDim::D3, "texture_3d<f32>", "") } else { (TexDim::D2, "texture_2d<f32>", ", 1u") };
                let slot = self.texture_slot(*handle, dim, ty, false, true);
                self.iface.textures[slot].query = true;
                let level = self.raw(src)?;
                let t = self.temp();
                let _ = writeln!(out, "let {t} = vec4<u32>(textureDimensions(tex{slot}, {level}){depth}, textureNumLevels(tex{slot}));");
                return self.tex_results(out, &t, false, *mask, (*dst, *dst2), false);
            }
            other => return Err(format!("unsupported: {other:?}")),
        };
        if io.fault != Pred::TRUE {
            return Err("sparse texture access (fault predicate)".into());
        }
        let (coords, array, ty) = match (dim, dc) {
            (TexDim::D1, false) => (1, false, "texture_1d<f32>"),
            (TexDim::D2, false) => (2, false, "texture_2d<f32>"),
            (TexDim::D3, false) => (3, false, "texture_3d<f32>"),
            (TexDim::Cube, false) if !fetch => (3, false, "texture_cube<f32>"),
            (TexDim::Array2D, false) => (2, true, "texture_2d_array<f32>"),
            (TexDim::ArrayCube, false) if !fetch => (3, true, "texture_cube_array<f32>"),
            (TexDim::D2, true) => (2, false, "texture_depth_2d"),
            (TexDim::Array2D, true) => (2, true, "texture_depth_2d_array"),
            (TexDim::Cube, true) => (3, false, "texture_depth_cube"),
            (TexDim::ArrayCube, true) => (3, true, "texture_depth_cube_array"),
            (d, c) => return Err(format!("texture {d:?} (compare {c}, fetch {fetch})")),
        };
        // The logical sources (NAK's lowering): a = [array index] coordinates;
        // b = [bindless handle] [lod or bias] [offset] [depth reference].
        #[derive(Clone, Copy, PartialEq)]
        enum Item {
            Array,
            Coord(usize),
            Handle,
            Lod,
            Offset,
            Dref,
        }
        let mut a: Vec<Item> = array.then_some(Item::Array).into_iter().chain((0..coords).map(Item::Coord)).collect();
        let mut b = Vec::new();
        if io.handle == TexHandle::Bindless {
            b.push(Item::Handle);
        }
        match lod {
            LodMode::Auto | LodMode::Zero => {}
            LodMode::Lod | LodMode::Bias => b.push(Item::Lod),
            other => return Err(format!("texture LOD mode {other:?}")),
        }
        if aoffi {
            b.push(Item::Offset);
        }
        if dc {
            b.push(Item::Dref);
        }
        if scr {
            // Up to four scalars, a's then b's, spread over the two vectors.
            // ponytail: the split for sources beyond the coordinates (LOD,
            // handle, reference) is the coordinates' rule extended, assumed.
            a.append(&mut b);
            let keep = match a.len() {
                4 | 3 => 2,
                2 | 1 => 1,
                n => return Err(format!(".SCR texture with {n} sources")),
            };
            b = a.split_off(keep);
        }
        let base = |o: &Operand| match o.src {
            Src::Reg(n) => Ok(n),
            other => Err(format!("texture vector {other:?}")),
        };
        let (ra, rb) = (base(&io.a)?, base(&io.b)?);
        let key = |item: Item| match item {
            Item::Array => 0usize,
            Item::Coord(c) => 1 + c,
            Item::Handle => 8,
            Item::Lod => 9,
            Item::Offset => 10,
            Item::Dref => 11,
        };
        let mut regs: BTreeMap<usize, String> = BTreeMap::new();
        let mut handle_reg = None;
        for (list, start) in [(&a, ra), (&b, rb)] {
            for (k, item) in list.iter().enumerate() {
                let r = if start == RZ { "0u".to_string() } else { self.r(start + k as u8) };
                regs.insert(key(*item), r);
                if *item == Item::Handle {
                    handle_reg = Some(start.saturating_add(k as u8));
                }
            }
        }
        let get = |item: Item| -> R<String> { regs.get(&key(item)).cloned().ok_or_else(|| "texture source missing".to_string()) };
        let known = handle_reg.and_then(|r| self.uval(&Operand { src: Src::Reg(r), neg: false, abs: false, swizzle: 0 }));
        let slot = self.texture_slot(io.handle, dim, ty, dc, fetch);
        // A bindless handle whose register holds known constant-bank words.
        if let Some(e) = known {
            self.iface.textures[slot].handle = Handle::Uniform(e);
        }
        let cs: Vec<String> = (0..coords).map(|c| get(Item::Coord(c))).collect::<R<_>>()?;
        let vec = |cast: &str, sc: &str| if coords == 1 { format!("{cast}({})", cs[0]) } else { format!("vec{coords}<{sc}>({})", cs.iter().map(|c| format!("{cast}({c})")).collect::<Vec<_>>().join(", ")) };
        let arr = if array { format!(", i32({})", get(Item::Array)?) } else { String::new() };
        let call = if fetch {
            let mut xy = vec("bitcast<i32>", "i32");
            if aoffi {
                // Four signed 4-bit offsets, x in the low bits.
                let off = get(Item::Offset)?;
                let parts: Vec<String> = (0..coords).map(|c| format!("extractBits(bitcast<i32>({off}), {}u, 4u)", 4 * c)).collect();
                xy = if coords == 1 { format!("({xy} + {})", parts[0]) } else { format!("({xy} + vec{coords}<i32>({}))", parts.join(", ")) };
            }
            let level = match lod {
                LodMode::Lod => format!("bitcast<i32>({})", get(Item::Lod)?),
                _ => "0".to_string(),
            };
            if dim == TexDim::D1 || dim == TexDim::D3 || dim == TexDim::D2 || array {
                // Outside the level (or the layers): zeros, as the console's texel fetch
                // gives (WGSL may return any texel there, which made results vary).
                let size = format!("textureDimensions(tex{slot}, {level})");
                let mut inside = if coords == 1 { format!("(u32({xy}) < {size})") } else { format!("all(vec{coords}<u32>({xy}) < {size})") };
                if array {
                    inside = format!("({inside} && u32(i32({})) < textureNumLayers(tex{slot}))", get(Item::Array)?);
                }
                format!("select(vec4<f32>(0.0), textureLoad(tex{slot}, {xy}{arr}, {level}), {inside})")
            } else {
                return Err(format!("texel fetch of {dim:?}"));
            }
        } else {
            let mut uv = vec("f", "f32");
            if aoffi {
                // ponytail: WGSL offsets are constants; the register's texel offset
                // is added to the coordinates in level-0 texels instead (exact at
                // level 0, approximate on smaller levels).
                if matches!(dim, TexDim::Cube | TexDim::ArrayCube) {
                    return Err("cube texture sample with an offset".into());
                }
                let off = get(Item::Offset)?;
                // TEX: signed 4-bit offsets, x lowest. TLD4: signed 6-bit offsets, one per
                // byte (the wider gather range; as Mesa's NAK packs them, not proved here).
                let (stride, bits) = if gather.is_some() { (8, 6) } else { (4, 4) };
                let parts: Vec<String> = (0..coords).map(|c| format!("f32(extractBits(bitcast<i32>({off}), {}u, {bits}u))", stride * c)).collect();
                uv = if coords == 1 {
                    format!("({uv} + {} / f32(textureDimensions(tex{slot}, 0)))", parts[0])
                } else {
                    format!("({uv} + vec{coords}<f32>({}) / vec{coords}<f32>(textureDimensions(tex{slot}, 0)))", parts.join(", "))
                };
            }
            match (dc, lod) {
                // TLD4: the gather's arguments after the component, used below.
                (false, _) if gather.is_some() => format!("tex{slot}, smp{slot}, {uv}{arr}"),
                (true, _) if gather.is_some() => format!("textureGatherCompare(tex{slot}, smp{slot}, {uv}{arr}, f({}))", get(Item::Dref)?),
                (false, LodMode::Auto) => format!("textureSample(tex{slot}, smp{slot}, {uv}{arr})"),
                (false, LodMode::Zero) => format!("textureSampleLevel(tex{slot}, smp{slot}, {uv}{arr}, 0.0)"),
                (false, LodMode::Lod) => format!("textureSampleLevel(tex{slot}, smp{slot}, {uv}{arr}, f({}))", get(Item::Lod)?),
                (false, LodMode::Bias) => format!("textureSampleBias(tex{slot}, smp{slot}, {uv}{arr}, f({}))", get(Item::Lod)?),
                (true, LodMode::Auto) => format!("textureSampleCompare(tex{slot}, smp{slot}, {uv}{arr}, f({}))", get(Item::Dref)?),
                (true, LodMode::Zero) => format!("textureSampleCompareLevel(tex{slot}, smp{slot}, {uv}{arr}, f({}))", get(Item::Dref)?),
                (dc, other) => return Err(format!("texture sample (compare {dc}) with LOD mode {other:?}")),
            }
        };
        let t = self.temp();
        if let (Some(comp), false) = (gather, dc) {
            // The texture header's swizzle picks the gathered component (or a constant),
            // as it does for a sample; WGSL wants the component as a constant, so all
            // four are gathered and one chosen. Assumed, not proved on the console: the
            // swizzle applies before the component selection (as in OpenGL).
            self.iface.swizzles = true;
            let (k, g) = (self.temp(), self.temp());
            let _ = writeln!(out, "let {k} = (swizzles[{}][{}] >> {}u) & 7u;", slot / 4, slot % 4, 3 * u32::from(comp));
            let gathers: Vec<String> = (0..4).map(|c| format!("bitcast<vec4<u32>>(textureGather({c}, {call}))")).collect();
            let _ = writeln!(out, "let {g} = array<vec4<u32>, 4>({});", gathers.join(", "));
            let _ = writeln!(out, "let {t} = select(vec4<u32>(select(select(0u, 1u, {k} == 6u), 0x3f800000u, {k} == 7u)), {g}[({k} + 2u) & 3u], {k} >= 2u && {k} <= 5u);");
        } else if gather.is_some() {
            let _ = writeln!(out, "let {t} = bitcast<vec4<u32>>({call});");
        } else if dc {
            let _ = writeln!(out, "let {t} = bitcast<u32>({call});");
        } else {
            let _ = writeln!(out, "let {t} = {};", self.swizzled(slot, &format!("bitcast<vec4<u32>>({call})")));
        }
        self.tex_results(out, &t, dc && gather.is_none(), io.mask, (io.dst, io.dst2), f16)
    }

    /// A sampled or fetched value with its texture header's swizzle applied.
    fn swizzled(&mut self, slot: usize, v: &str) -> String {
        self.helpers.insert(SWZ);
        self.iface.swizzles = true;
        format!("swz({v}, swizzles[{}][{}])", slot / 4, slot % 4)
    }

    /// TXD: a = [bindless handle] coordinates [array index | offset << 16],
    /// b = the derivatives interleaved (d/dx, d/dy per coordinate), as NAK
    /// lowers it.
    fn txd(&mut self, out: &mut String, io: &crate::sass::texture::TexIo, dim: TexDim, aoffi: bool) -> R<()> {
        if io.fault != Pred::TRUE {
            return Err("sparse texture access (fault predicate)".into());
        }
        let (coords, array, ty) = match dim {
            TexDim::D2 => (2, false, "texture_2d<f32>"),
            TexDim::Array2D => (2, true, "texture_2d_array<f32>"),
            TexDim::D3 => (3, false, "texture_3d<f32>"),
            TexDim::Cube => (3, false, "texture_cube<f32>"),
            d => return Err(format!("TXD of {d:?}")),
        };
        if aoffi || io.handle == TexHandle::Bindless {
            return Err("TXD with an offset or a bindless handle".into());
        }
        let (Src::Reg(ra), Src::Reg(rb)) = (io.a.src, io.b.src) else { return Err("TXD sources".into()) };
        let slot = self.texture_slot(io.handle, dim, ty, false, false);
        let reg = |cx: &mut Cx, base: u8, k: u8| if base == RZ { "0u".to_string() } else { cx.r(base + k) };
        let vec = |cx: &mut Cx, base: u8, ks: &[u8]| {
            let parts: Vec<String> = ks.iter().map(|&k| format!("f({})", reg(cx, base, k))).collect();
            format!("vec{coords}<f32>({})", parts.join(", "))
        };
        let uv = vec(self, ra, &(0..coords).collect::<Vec<u8>>());
        let arr = if array { format!(", i32({} & 0xffffu)", reg(self, ra, coords)) } else { String::new() };
        let ddx = vec(self, rb, &(0..coords).map(|c| 2 * c).collect::<Vec<u8>>());
        let ddy = vec(self, rb, &(0..coords).map(|c| 2 * c + 1).collect::<Vec<u8>>());
        let t = self.temp();
        let v = self.swizzled(slot, &format!("bitcast<vec4<u32>>(textureSampleGrad(tex{slot}, smp{slot}, {uv}{arr}, {ddx}, {ddy}))"));
        let _ = writeln!(out, "let {t} = {v};");
        self.tex_results(out, &t, false, io.mask, (io.dst, io.dst2), false)
    }
}

fn round_fn(r: Rnd) -> &'static str {
    match r {
        Rnd::NearestEven => "round",
        Rnd::NegInf => "floor",
        Rnd::PosInf => "ceil",
        Rnd::Zero => "trunc",
    }
}

/// A float comparison (`*u` forms are also true when an operand is NaN).
fn fcmp(c: fp32::FCmp, a: &str, b: &str) -> String {
    use fp32::FCmp::*;
    match c {
        F => "false".into(),
        Lt => format!("({a} < {b})"),
        Eq => format!("({a} == {b})"),
        Le => format!("({a} <= {b})"),
        Gt => format!("({a} > {b})"),
        Ne => format!("(({a} < {b}) || ({a} > {b}))"),
        Ge => format!("({a} >= {b})"),
        Num => format!("(({a} == {a}) && ({b} == {b}))"),
        Nan => format!("(({a} != {a}) || ({b} != {b}))"),
        Ltu => format!("!({a} >= {b})"),
        Equ => format!("!(({a} < {b}) || ({a} > {b}))"),
        Leu => format!("!({a} > {b})"),
        Gtu => format!("!({a} <= {b})"),
        Neu => format!("({a} != {b})"),
        Geu => format!("!({a} < {b})"),
        T => "true".into(),
    }
}

/// The fp16 family's comparison as the fp32 one (the same 4-bit codes).
fn fcmp16(c: fp16::FloatCmp) -> fp32::FCmp {
    use fp32::FCmp::*;
    [F, Lt, Eq, Le, Gt, Ne, Ge, Num, Nan, Ltu, Equ, Leu, Gtu, Neu, Geu, T][c as usize]
}

/// An integer comparison by its 3-bit code (false, lt, eq, le, gt, ne, ge, true).
fn icmp(code: u8, a: &str, b: &str) -> String {
    match code {
        0 => "false".into(),
        1 => format!("({a} < {b})"),
        2 => format!("({a} == {b})"),
        3 => format!("({a} <= {b})"),
        4 => format!("({a} > {b})"),
        5 => format!("({a} != {b})"),
        6 => format!("({a} >= {b})"),
        _ => "true".into(),
    }
}

/// Combines a comparison with an accumulator: 0 and, 1 or, 2 xor.
fn bool_op(op: u8, c: &str, acc: &str) -> String {
    match (op, acc) {
        (0, "true") | (1, "false") | (2, "false") => c.to_string(),
        (0, _) => format!("({c} && {acc})"),
        (1, _) => format!("({c} || {acc})"),
        _ => format!("({c} != {acc})"),
    }
}

/// LOP3: the truth table `lut` over a (0xf0), b (0xcc), c (0xaa), bitwise.
fn lop3(lut: u8, a: &str, b: &str, c: &str) -> String {
    match lut {
        0x00 => return "0u".into(),
        0xff => return "0xffffffffu".into(),
        0xf0 => return a.into(),
        0xcc => return b.into(),
        0xaa => return c.into(),
        0x0f => return format!("(~{a})"),
        0xc0 => return format!("({a} & {b})"),
        0xa0 => return format!("({a} & {c})"),
        0x88 => return format!("({b} & {c})"),
        0xfc => return format!("({a} | {b})"),
        0xfa => return format!("({a} | {c})"),
        0xee => return format!("({b} | {c})"),
        0x3c => return format!("({a} ^ {b})"),
        0x5a => return format!("({a} ^ {c})"),
        0x66 => return format!("({b} ^ {c})"),
        0x96 => return format!("({a} ^ {b} ^ {c})"),
        0x80 => return format!("({a} & {b} & {c})"),
        0xfe => return format!("({a} | {b} | {c})"),
        0x30 => return format!("({a} & ~{b})"),
        0x50 => return format!("({a} & ~{c})"),
        _ => {}
    }
    let terms: Vec<String> = (0..8u8)
        .filter(|k| lut >> k & 1 != 0)
        .map(|k| {
            let lit = |bit: u8, v: &str| if k >> bit & 1 != 0 { v.to_string() } else { format!("~{v}") };
            format!("({} & {} & {})", lit(2, a), lit(1, b), lit(0, c))
        })
        .collect();
    format!("({})", terms.join(" | "))
}

const PRMT: &str = "fn prmt(a: u32, s: u32, c: u32) -> u32 {
    var r = 0u;
    for (var k = 0u; k < 4u; k++) {
        let n = (s >> (4u * k)) & 0xfu;
        let j = n & 7u;
        var byte = select((c >> (8u * (j & 3u))) & 0xffu, (a >> (8u * j)) & 0xffu, j < 4u);
        if ((n & 8u) != 0u) { byte = select(0u, 0xffu, (byte & 0x80u) != 0u); }
        r |= byte << (8u * k);
    }
    return r;
}
";

/// A texture header's component swizzle (`swizzle_code`: 3 bits per
/// component, 0 zero, 2..5 R..A, 6 integer one, 7 float one) on a result.
const SWZ: &str = "fn swz(v: vec4<u32>, s: u32) -> vec4<u32> {
    var r = vec4<u32>(0u);
    for (var c = 0u; c < 4u; c++) {
        let k = (s >> (3u * c)) & 7u;
        if (k >= 2u && k <= 5u) { r[c] = v[k - 2u]; } else if (k == 6u) { r[c] = 1u; } else if (k == 7u) { r[c] = 0x3f800000u; }
    }
    return r;
}
";

/// The binding of the per-slot swizzle codes (`array<vec4<u32>, n>` for 4n
/// slots, slot i at [i / 4][i % 4]), above every bank and texture binding.
pub const SWIZZLE_BINDING: u32 = 999;

const SHF: &str = "fn shf_r(lo: u32, hi: u32, s: u32, arith: bool) -> vec2<u32> {
    let fill = select(0u, select(0u, 0xffffffffu, bitcast<i32>(hi) < 0), arith);
    let h_s = select(hi >> (s & 31u), bitcast<u32>(bitcast<i32>(hi) >> (s & 31u)), arith);
    if (s == 0u) { return vec2<u32>(lo, hi); }
    if (s < 32u) { return vec2<u32>((lo >> s) | (hi << (32u - s)), h_s); }
    if (s < 64u) { return vec2<u32>(h_s, fill); }
    return vec2<u32>(fill, fill);
}
fn shf_l(lo: u32, hi: u32, s: u32) -> vec2<u32> {
    if (s == 0u) { return vec2<u32>(lo, hi); }
    if (s < 32u) { return vec2<u32>(lo << s, (hi << s) | (lo >> (32u - s))); }
    if (s < 64u) { return vec2<u32>(0u, lo << (s & 31u)); }
    return vec2<u32>(0u, 0u);
}
";

#[cfg(test)]
mod tests {
    use super::*;

    /// A word: opcode, guard PT, fields (lo, value).
    fn w(op: u16, fields: &[(u32, u128)]) -> [u8; 16] {
        fields.iter().fold(u128::from(op) | 7 << 12, |v, (lo, x)| v | x << lo).to_le_bytes()
    }

    fn exit() -> [u8; 16] {
        w(0x94d, &[(87, 7)])
    }

    fn bra(offset: i64) -> [u8; 16] {
        w(0x947, &[(87, 7), (32, (offset as u64 as u128) & ((1 << 50) - 1))])
    }

    fn program(sph: &[(usize, u32)], insns: &[[u8; 16]]) -> Vec<u8> {
        let mut code = vec![0u8; SPH_SIZE];
        for (word, v) in sph {
            code[4 * word..4 * word + 4].copy_from_slice(&v.to_le_bytes());
        }
        for i in insns {
            code.extend(i);
        }
        code.extend(bra(-16));
        code
    }

    fn ok(code: &[u8]) -> Translation {
        let t = translate(code).unwrap_or_else(|e| panic!("{e}"));
        validate(&t.wgsl).unwrap_or_else(|e| panic!("{e}\n{}", t.wgsl));
        t
    }

    #[test]
    fn a_pass_through_vertex_shader_translates_and_validates() {
        // ALD R0..R1 = attribute 1 xy; ALD R2..R3 = attribute 0 xy; AST position xy, varying 0 xy.
        let ald = |dst: u128, attr: u128| w(0x321, &[(16, dst), (24, 255), (32, 255), (40, attr), (74, 1)]);
        let ast = |data: u128, attr: u128| w(0x322, &[(24, 255), (32, data), (64, 255), (40, attr), (74, 1)]);
        let code = program(&[(0, 0x020c_0481), (6, 0x33), (13, 0x3_3000)], &[ald(0, 0x90), ald(2, 0x80), ast(2, 0x70), ast(0, 0x80), exit()]);
        let t = ok(&code);
        assert_eq!(t.stage, Stage::Vertex);
        assert_eq!(t.interface.attributes, BTreeMap::from([(0, 3), (1, 3)]));
        assert!(t.wgsl.contains("o.pos[0] = f(r2);"), "{}", t.wgsl);
    }

    #[test]
    fn a_pixel_shader_with_a_texture_and_a_branch_translates_and_validates() {
        // IPA R1 = v0.x, R2 = v0.y; FSETP.GT P0 = R1 > 0.5; @!P0 BRA over the TEX;
        // TEX R0 = tex(c[2][0x70]).r at (R1, R2) with .SCR; EXIT (R0 to target 0).
        let ipa = |dst: u128, attr: u128| w(0x326, &[(16, dst), (64, attr / 4), (32, 255), (81, 7)]);
        let fsetp = w(0x80b, &[(24, 1), (32, 0x3f00_0000), (76, 4), (81, 0), (84, 7), (87, 7)]);
        let skip = {
            let mut b = bra(16);
            b[1] = (b[1] & 0x0f) | 0x80; // guard @!P0
            b
        };
        let tex = w(0xb60, &[(16, 0), (24, 1), (32, 2), (64, 255), (40, 0x70 / 4), (54, 2), (61, 1), (72, 1), (81, 7), (84, 1), (60, 1)]);
        let code = program(&[(0, 0x000c_5482), (6, 0xa), (18, 1)], &[ipa(1, 0x80), ipa(2, 0x84), fsetp, skip, tex, exit()]);
        let t = ok(&code);
        assert_eq!(t.interface.textures, vec![Texture { handle: Handle::CBuf { bank: 2, offset: 0x70 }, dim: TexDim::D2, compare: false, fetch: false, query: false }]);
        assert_eq!(t.interface.targets, BTreeMap::from([(0, 1)]));
        // A forward branch only: blocks in order, no loop.
        assert!(t.wgsl.contains("if (pc == ") && !t.wgsl.contains("switch pc"), "{}", t.wgsl);
    }

    #[test]
    fn a_tex_with_half_results_packs_two_components_per_register() {
        // TEX.F16 R18, R19 (mask RGB) at (R1, R2): (R, G) to R18, (B, 0) to R19, nothing to R20.
        let ipa = |dst: u128, attr: u128| w(0x326, &[(16, dst), (64, attr / 4), (32, 255), (81, 7)]);
        let tex = w(0xb60, &[(16, 18), (24, 1), (32, 2), (64, 19), (40, 0x70 / 4), (54, 2), (61, 1), (72, 7), (79, 1), (81, 7), (84, 1), (60, 1)]);
        let t = ok(&program(&[(0, 0x000c_5482), (6, 0xa), (18, 1)], &[ipa(1, 0x80), ipa(2, 0x84), tex, exit()]));
        assert!(t.wgsl.contains("r18 = pack2x16float(vec2<f32>(f(") && t.wgsl.contains("[1]))"), "{}", t.wgsl);
        assert!(t.wgsl.contains("r19 = pack2x16float(vec2<f32>(f(") && t.wgsl.contains("[2]), 0.0))"), "{}", t.wgsl);
        assert!(!t.wgsl.contains("r20 ="), "{}", t.wgsl);
    }

    #[test]
    fn a_bindless_handle_made_of_constant_words_is_tracked() {
        // R7 = c[2][0x258]; R5 = c[2][0x5c0]; R4 = R7 | R5; TEX.SCR 2D at (R1, R2), handle R4.
        let ipa = |dst: u128, attr: u128| w(0x326, &[(16, dst), (64, attr / 4), (32, 255), (81, 7)]);
        let ldc =|dst: u128, offset: u128| w(0xb82, &[(16, dst), (24, 255), (54, 2), (38, offset), (73, 4)]);
        let lop3 = w(0x212, &[(16, 4), (24, 7), (32, 5), (64, 255), (72, 0xfc), (81, 7), (87, 7)]);
        let tex = w(0x361, &[(16, 0), (24, 1), (32, 4), (64, 255), (59, 1), (61, 1), (72, 1), (81, 7), (84, 1), (60, 1)]);
        let code = program(&[(0, 0x000c_5482), (6, 0xa), (18, 1)], &[ipa(1, 0x80), ipa(2, 0x84), ldc(7, 0x258), ldc(5, 0x5c0), lop3, tex, exit()]);
        let t = ok(&code);
        let want = Handle::Uniform(UExpr::Lop3(0xfc, Box::new([UExpr::CBuf(2, 0x258), UExpr::CBuf(2, 0x5c0), UExpr::Imm(0)])));
        assert_eq!(t.interface.textures[0].handle, want);
    }

    #[test]
    fn a_vertex_shader_takes_its_pixel_shaders_interpolation() {
        let ald = w(0x321, &[(16, 0), (24, 255), (32, 255), (40, 0x80), (74, 1)]);
        let ast = w(0x322, &[(24, 255), (32, 0), (64, 255), (40, 0x80), (74, 1)]);
        let code = program(&[(0, 0x020c_0481), (6, 0x3), (13, 0x3_0000)], &[ald, ast, exit()]);
        // The pixel shader reads varying 0 linearly and varying 1, which this stage never writes.
        let options = Options { varyings: BTreeMap::from([(0, Interp::ScreenLinear), (1, Interp::Constant)]), ..Default::default() };
        let t = translate_with(&code, &options).unwrap();
        validate(&t.wgsl).unwrap_or_else(|e| panic!("{e}\n{}", t.wgsl));
        assert!(t.wgsl.contains("@location(0) @interpolate(linear) v0"), "{}", t.wgsl);
        assert!(t.wgsl.contains("@location(1) @interpolate(flat) v1"), "{}", t.wgsl);
        assert_eq!(t.interface.varyings, options.varyings);
    }

    #[test]
    fn front_face_predicates_from_a_register_and_quad_swaps_translate_and_validate() {
        // IPA R0 = front face; R2P P0, P1 = bits 0, 1 of R0; SHFL.BFLY R0 = R0 of lane ^ 1
        // in its quad; EXIT (R0 to target 0).
        let ipa = w(0x326, &[(16, 0), (64, 0x3fc / 4), (32, 255), (81, 7)]);
        let r2p = w(0x804, &[(24, 0), (32, 3)]);
        let shfl = w(0xf89, &[(16, 0), (24, 0), (53, 1), (40, 0x1c03), (58, 3), (81, 7)]);
        let t = ok(&program(&[(0, 0x000c_5482), (18, 1)], &[ipa, r2p, shfl, exit()]));
        for want in ["select(0u, 0xffffffffu, i.front)", "p1 = ((r0 >> 1u) & 1u) != 0u;", "r0 = quadSwapX(r0);"] {
            assert!(t.wgsl.contains(want), "{want}\n{}", t.wgsl);
        }
    }

    #[test]
    fn a_compute_program_stores_to_an_image_by_thread_id() {
        // No SPH. S2R R0..R2 = tid.x, tid.y, ctaid.z; SUST R4..R7 at (R0, R1, R2) to the
        // 3D image whose handle is c[0][0x20]; EXIT.
        let s2r = |dst: u128, sr: u128| w(0x919, &[(16, dst), (72, sr)]);
        let sust = w(0xb9b, &[(24, 0), (32, 4), (40, 0x20 / 4), (54, 0), (61, 5), (72, 0xf), (84, 1)]);
        let mut code = Vec::new();
        for i in [s2r(0, 0x21), s2r(1, 0x22), s2r(2, 0x27), sust, exit(), bra(-16)] {
            code.extend(i);
        }
        let options = Options { workgroup: [8, 8, 1], images: vec!["rgba16float".into()], ..Default::default() };
        let t = translate_with(&code, &options).unwrap_or_else(|e| panic!("{e}"));
        validate(&t.wgsl).unwrap_or_else(|e| panic!("{e}\n{}", t.wgsl));
        assert_eq!(t.stage, Stage::Compute);
        assert_eq!(t.interface.images, vec![Image { handle: Handle::CBuf { bank: 0, offset: 0x20 }, dim: ImageDim::D3, read: false }]);
        for want in ["@workgroup_size(8, 8, 1)", "r0 = lid.x;", "r2 = wid.z;", "texture_storage_3d<rgba16float, write>", "textureStore(img0, vec3<i32>(i32(r0), i32(r1), i32(r2))"] {
            assert!(t.wgsl.contains(want), "{want}\n{}", t.wgsl);
        }
        // Integer images store the register bits.
        for (format, want) in [("r32uint", "vec4<u32>((r4), (r5), (r6), (r7))"), ("r32sint", "vec4<i32>(bitcast<i32>(r4), bitcast<i32>(r5),")] {
            let t = translate_with(&code, &Options { images: vec![format.into()], ..options.clone() }).unwrap_or_else(|e| panic!("{e}"));
            validate(&t.wgsl).unwrap_or_else(|e| panic!("{e}\n{}", t.wgsl));
            assert!(t.wgsl.contains(&format!("texture_storage_3d<{format}, write>")) && t.wgsl.contains(want), "{want}\n{}", t.wgsl);
        }
    }

    #[test]
    fn an_image_loaded_and_stored_by_one_program_is_bound_read_write() {
        // S2R R0, R1 = tid.x, tid.y; SULD R4..R7 = image c[0][0x30] at (R0, R1); SUST it back.
        let s2r = |dst: u128, sr: u128| w(0x919, &[(16, dst), (72, sr)]);
        let suld = w(0xb97, &[(16, 4), (24, 0), (40, 0x30 / 4), (54, 0), (61, 3), (72, 0xf), (81, 7)]);
        let sust = w(0xb9b, &[(24, 0), (32, 4), (40, 0x30 / 4), (54, 0), (61, 3), (72, 0xf)]);
        let mut code = Vec::new();
        for i in [s2r(0, 0x21), s2r(1, 0x22), suld, sust, exit(), bra(-16)] {
            code.extend(i);
        }
        let t = translate_with(&code, &Options { workgroup: [8, 8, 1], ..Default::default() }).unwrap_or_else(|e| panic!("{e}"));
        validate(&t.wgsl).unwrap_or_else(|e| panic!("{e}\n{}", t.wgsl));
        assert_eq!(t.interface.images, vec![Image { handle: Handle::CBuf { bank: 0, offset: 0x30 }, dim: ImageDim::D2, read: true }]);
        assert!(t.interface.textures.is_empty());
        for want in ["texture_storage_2d<rgba16float, read_write>", "textureLoad(img0, vec2<i32>(i32(r0), i32(r1)))", "textureStore(img0,"] {
            assert!(t.wgsl.contains(want), "{want}\n{}", t.wgsl);
        }
    }

    fn compute(insns: &[[u8; 16]], options: &Options) -> Translation {
        let mut code = Vec::new();
        for i in insns.iter().chain(&[exit(), bra(-16)]) {
            code.extend(i);
        }
        let t = translate_with(&code, options).unwrap_or_else(|e| panic!("{e}"));
        validate(&t.wgsl).unwrap_or_else(|e| panic!("{e}\n{}", t.wgsl));
        t
    }

    #[test]
    fn array_images_take_their_layer_after_the_position() {
        // S2R R4, R5, R6 = tid.x, tid.y, ctaid.z; SULD.D2Array R8..R11 = image c[0][0x30]
        // at (R4, R5) layer R6, SUST it back, and SUST it to c[0][0x40] (write only).
        let s2r = |dst: u128, sr: u128| w(0x919, &[(16, dst), (72, sr)]);
        let suld = w(0xb97, &[(16, 8), (24, 4), (40, 0x30 / 4), (54, 0), (61, 4), (72, 0xf), (81, 7)]);
        let sust = |h: u128| w(0xb9b, &[(24, 4), (32, 8), (40, h / 4), (54, 0), (61, 4), (72, 0xf)]);
        let t = compute(&[s2r(4, 0x21), s2r(5, 0x22), s2r(6, 0x27), suld, sust(0x30), sust(0x40)], &Options { workgroup: [8, 8, 1], ..Default::default() });
        assert_eq!(t.interface.images.iter().map(|i| (i.dim, i.read)).collect::<Vec<_>>(), [(ImageDim::D2Array, true), (ImageDim::D2Array, false)]);
        for want in [
            "texture_storage_2d_array<rgba16float, read_write>",
            "texture_storage_2d_array<rgba16float, write>",
            "textureLoad(img0, vec2<i32>(i32(r4), i32(r5)), i32(r6))",
            "u32(i32(r6)) < textureNumLayers(img0)",
            "textureStore(img1, vec2<i32>(i32(r4), i32(r5)), i32(r6), vec4<f32>(f(r8),",
        ] {
            assert!(t.wgsl.contains(want), "{want}\n{}", t.wgsl);
        }
    }

    #[test]
    fn tld4_on_an_array_texture_gathers_the_swizzled_component_with_its_layer() {
        // TLD4.G.SCR R28, R22 = c[0][0x28] Array2D at a = (R32 layer, R33 x), b = (R43 y).
        let tld4 = w(0xb63, &[(16, 28), (24, 32), (32, 43), (64, 22), (40, 0x28 / 4), (54, 0), (61, 5), (60, 1), (72, 0xf), (81, 7), (84, 1), (87, 1)]);
        let t = compute(&[tld4], &Options { workgroup: [8, 8, 1], ..Default::default() });
        assert_eq!(t.interface.textures, vec![Texture { handle: Handle::CBuf { bank: 0, offset: 0x28 }, dim: TexDim::Array2D, compare: false, fetch: false, query: false }]);
        for want in ["texture_2d_array<f32>", "textureGather(0, tex0, smp0, vec2<f32>(f(r33), f(r43)), i32(r32))", "textureGather(3, tex0", "(swizzles[0][0] >> 3u) & 7u", "r28 =", "r29 =", "r22 =", "r23 ="] {
            assert!(t.wgsl.contains(want), "{want}\n{}", t.wgsl);
        }
        // .AOFFI: the offset follows y in b (R44), 6-bit fields one byte apart.
        let tld4 = w(0xb63, &[(16, 28), (24, 32), (32, 43), (64, 22), (40, 0x28 / 4), (54, 0), (61, 5), (60, 1), (72, 0xf), (76, 1), (81, 7), (84, 1), (87, 1)]);
        let t = compute(&[tld4], &Options { workgroup: [8, 8, 1], ..Default::default() });
        for want in ["extractBits(bitcast<i32>(r44), 0u, 6u)", "extractBits(bitcast<i32>(r44), 8u, 6u)", "textureGather(0, tex0, smp0, (vec2<f32>(f(r33), f(r43)) + "] {
            assert!(t.wgsl.contains(want), "{want}\n{}", t.wgsl);
        }
    }

    #[test]
    fn banks_are_declared_as_far_as_the_program_reads_them() {
        // LDC R4 = c[3][0x24] (a fixed offset); S2R R20 = tid.x; LDC R5 = c[1][R20] (indexed).
        let fixed = w(0xb82, &[(16, 4), (24, 255), (54, 3), (38, 0x24), (73, 4)]);
        let s2r = w(0x919, &[(16, 20), (72, 0x21)]);
        let indexed = w(0xb82, &[(16, 5), (24, 20), (54, 1), (73, 4)]);
        let t = compute(&[fixed, s2r, indexed], &Options { workgroup: [1, 1, 1], ..Default::default() });
        assert_eq!(t.interface.cbuf_sizes, BTreeMap::from([(1, 0x10000), (3, 0x30)]));
        // Both still declared as 64 KiB.
        for want in ["var<uniform> cb3: array<vec4<u32>, 4096>;", "var<uniform> cb1: array<vec4<u32>, 4096>;"] {
            assert!(t.wgsl.contains(want), "{want}\n{}", t.wgsl);
        }
    }

    #[test]
    fn a_jump_table_read_from_a_bank_gives_exact_blocks_once_its_targets_are_known() {
        // S2R R20 = tid.x; LDC R54 = c[1][R20]; BRX R54 (absolute: base 0); then two
        // blocks at instructions 3 and 5, each S2R then EXIT.
        let s2r = |dst: u128| w(0x919, &[(16, dst), (72, 0x21)]);
        let ldc = w(0xb82, &[(16, 54), (24, 20), (54, 1), (73, 4)]);
        let brx = w(0x949, &[(24, 54), (34, ((-48i64 / 4) as u64 as u128) & ((1 << 48) - 1)), (87, 7)]);
        let body = [s2r(20), ldc, brx, s2r(1), exit(), s2r(2), exit()];
        let options = Options { workgroup: [8, 1, 1], ..Default::default() };
        let t = compute(&body, &options);
        assert_eq!(t.interface.jump_tables, Some(vec![(1, 0)]));
        let cases = |t: &Translation| t.wgsl.matches("case ").count();
        assert!(cases(&t) >= body.len(), "{}", t.wgsl);
        let t = compute(&body, &Options { jump_targets: Some(BTreeSet::from([3, 5])), ..options.clone() });
        assert_eq!(cases(&t), 3, "{}", t.wgsl);
        // R54 written between the load and the jump: not a table.
        let body = [s2r(20), ldc, s2r(54), brx, s2r(1), exit()];
        assert_eq!(compute(&body, &options).interface.jump_tables, None);
    }

    #[test]
    fn shared_memory_stores_loads_atomics_and_barriers() {
        // S2R R0 = tid.x; STS [R0 x4] = R0; BAR.SYNC; ATOMS.ADD R1 = [0x40], R0;
        // ATOMS.MAX.S32 R2 = [0x44], R0; LDS.U8 R3 = [R0 + 1]; EXIT.
        let s2r = w(0x919, &[(16, 0), (72, 0x21)]);
        let sts = w(0x388, &[(24, 0), (32, 0), (73, 4), (78, 1)]);
        let bar = w(0xb1d, &[]);
        let add = w(0x38c, &[(16, 1), (24, 255), (32, 0), (40, 0x40)]);
        let max = w(0x38c, &[(16, 2), (24, 255), (32, 0), (40, 0x44), (73, 1), (87, 2)]);
        let lds = w(0x984, &[(16, 3), (24, 0), (40, 1)]);
        let mut code = Vec::new();
        for i in [s2r, sts, bar, add, max, lds, exit(), bra(-16)] {
            code.extend(i);
        }
        let t = translate_with(&code, &Options { workgroup: [64, 1, 1], shared: 0x48, ..Default::default() }).unwrap_or_else(|e| panic!("{e}"));
        validate(&t.wgsl).unwrap_or_else(|e| panic!("{e}\n{}", t.wgsl));
        for want in ["var<workgroup> shm: array<atomic<u32>, 18>;", "* 4u +", "atomicStore(&shm[(t1 >> 2u) + 0u], r0);", "workgroupBarrier();", "atomicAdd(&shm[t2 >> 2u], r0)", "shm_cas(t4 >> 2u, 1u, r0)", "extractBits(atomicLoad(&shm[(t6 >> 2u) + 0u]), (t6 & 3u) * 8u, 8u)"] {
            assert!(t.wgsl.contains(want), "{want}\n{}", t.wgsl);
        }
    }

    #[test]
    fn unsupported_operations_are_errors_with_their_offset() {
        // A vertex shader with S2R (system register read) at 0x10.
        let code = program(&[(0, 0x020c_0481)], &[w(0x918, &[]), w(0x919, &[(16, 0), (72, 0x21)]), exit()]);
        let e = translate(&code).unwrap_err();
        assert_eq!(e.at, Some(0x10));
    }

    #[test]
    fn lop3_tables_match_their_bitwise_meaning() {
        assert_eq!(lop3(0xc0, "a", "b", "c"), "(a & b)");
        // 0xe8 (majority) takes the general path: minterms 3, 5, 6, 7.
        assert_eq!(lop3(0xe8, "a", "b", "c"), "((~a & b & c) | (a & ~b & c) | (a & b & ~c) | (a & b & c))");
    }
}
