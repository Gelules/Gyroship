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

use crate::sass::control::ControlOp;
use crate::sass::fp16::Fp16Op;
use crate::sass::fp32::{self, Fp32Op, MufuFn, Rnd};
use crate::sass::int::{self, IntOp};
use crate::sass::memory::{ImageDim, InterpFreq, InterpLoc, LdcMode, MemType, MemoryOp};
use crate::sass::moves::{self, MovesOp};
use crate::sass::operand::{Operand, PT, Pred, RZ, Src, URZ};
use crate::sass::texture::{LodMode, TexDim, TexHandle, TexQuery, TextureOp};
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
    /// Texture slots, in binding order.
    pub textures: Vec<Texture>,
    /// The swizzle codes are read (at `SWIZZLE_BINDING`).
    pub swizzles: bool,
    /// Compute: images written (SUST), in binding order (`IMAGE_BINDING` + i).
    pub images: Vec<Image>,
}

/// A storage image a program writes: where its handle is, and its dimension.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Image {
    pub handle: Handle,
    pub dim: ImageDim,
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
    /// Compute: the WGSL storage format of each image slot (from its texture
    /// header); a missing one is `rgba16float`.
    pub images: Vec<String>,
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
    let body = cx.program(&words, &ops)?;
    Ok(Translation { key, stage, wgsl: cx.module(&body, options), interface: cx.iface })
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
    /// Compute: the storage format of each image slot (`Options::images`).
    images: Vec<String>,
}

impl Cx {
    fn new(sph: Sph) -> Cx {
        Cx { sph, regs: BTreeSet::new(), preds: BTreeSet::new(), uregs: BTreeSet::new(), upreds: BTreeSet::new(), iface: Interface::default(), decls: String::new(), helpers: BTreeSet::new(), temps: 0, uexpr: BTreeMap::new(), images: Vec::new() }
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
        Ok(format!("cb{bank}[{}u][{}u]", offset / 16, offset / 4 % 4))
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
        let mut leaders = BTreeSet::from([0usize]);
        for (i, op) in ops.iter().enumerate() {
            match op {
                Op::Control(ControlOp::Bra { offset, .. }) if *offset != -16 => {
                    leaders.insert(target(i, *offset).filter(|t| *t < ops.len()).ok_or_else(|| at(i)("branch outside the program".into()))?);
                    leaders.insert(i + 1);
                }
                // After a conditional EXIT the program goes on; after an unconditional
                // one, only a branch target (already a leader) can follow.
                Op::Control(ControlOp::Exit { cond, .. }) if !(words[i].always() && *cond == Pred::TRUE) => {
                    leaders.insert(i + 1);
                }
                Op::Control(c @ (ControlOp::Call { .. } | ControlOp::Ret { .. } | ControlOp::Brx { .. })) => return Err(at(i)(format!("unsupported control flow: {c:?}"))),
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
            Op::Control(ControlOp::Bra { offset, cond, .. }) => {
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
            let [x, y, z] = options.workgroup.map(|n| n.max(1));
            let _ = write!(m, "\n@compute @workgroup_size({x}, {y}, {z})\nfn main(@builtin(local_invocation_id) lid: vec3<u32>, @builtin(workgroup_id) wid: vec3<u32>) {{\n{locals}{body}}}\n");
        } else if self.sph.stage == Stage::Vertex {
            let mut vin = String::from("struct VsIn {\n@builtin(vertex_index) vid: u32,\n@builtin(instance_index) iid: u32,\n");
            for (v, mask) in self.sph.inputs().into_iter().enumerate() {
                if mask != 0 {
                    self.iface.attributes.insert(v as u8, mask);
                    let _ = writeln!(vin, "@location({v}) a{v}: vec4<f32>,");
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
            Op::Fp16(Fp16Op::I2fp { dst, src, signed, dst_bits: 32, src_bits: 32, .. }) => {
                let e = self.i2f(src, *signed, 32, 0)?;
                self.set(out, *dst, &e);
                Ok(())
            }
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
                other => Err(format!("unsupported: {other:?}")),
            },
            other => Err(format!("unsupported: {other:?}")),
        }
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
                let acc = self.p(*acc);
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
            Fp32Op::F2i { dst, src, signed, dst_bits: 32, src_bits: 32, rnd, .. } => {
                let x = format!("{}({})", round_fn(*rnd), self.f(src)?);
                // WGSL's float to integer conversions saturate, as F2I does; NaN gives 0.
                let e = if *signed { format!("bitcast<u32>(i32({x}))") } else { format!("u32({x})") };
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
            }
            IntOp::ISetP { dst, dst1, a, b, cmp, op, signed, ex: false, accum, .. } => {
                let (x, y) = (self.raw(a)?, self.raw(b)?);
                let (x, y) = if *signed { (format!("bitcast<i32>({x})"), format!("bitcast<i32>({y})")) } else { (x, y) };
                let t = self.temp();
                let _ = writeln!(out, "let {t} = {};", icmp(*cmp as u8, &x, &y));
                let acc = self.p(*accum);
                let (e1, e2) = (bool_op(*op as u8, &t, &acc), bool_op(*op as u8, &format!("!{t}"), &acc));
                self.set_p(out, *dst, &e1);
                self.set_p(out, *dst1, &e2);
            }
            IntOp::Lop3 { dst, a, b, c, lut, pand, pdst, psrc } => {
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
            }
            IntOp::Sel { dst, a, b, cond } => {
                let (x, y, c) = (self.raw(a)?, self.raw(b)?, self.p(*cond));
                self.set(out, *dst, &format!("select({y}, {x}, {c})"));
            }
            IntOp::IMad { dst, a, b, c, hi: false, x: false, pdst, .. } if *pdst == PT => {
                let (x, y, z) = (self.i(a, false)?, self.i(b, false)?, self.i(c, false)?);
                self.set(out, *dst, &format!("{x} * {y} + {z}"));
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
            IntOp::Lea { dst, a, b, c, shift, hi, x: false, neg_shifted, overflow, carry } if *overflow == PT && *carry == (Pred { reg: PT, not: true }) => {
                let (lo, add, high) = (self.raw(a)?, self.i(b, false)?, self.raw(c)?);
                let s = u32::from(*shift);
                let shifted = match (*hi, s) {
                    (false, _) => format!("({lo} << {s}u)"),
                    (true, 0) => high,
                    (true, _) => format!("(({high} << {s}u) | ({lo} >> {}u))", 32 - s),
                };
                let shifted = if *neg_shifted { format!("(0u - {shifted})") } else { shifted };
                self.set(out, *dst, &format!("{shifted} + {add}"));
            }
            IntOp::Sgxt { dst, a, bits, signed, wrap } => {
                let (x, n) = (self.raw(a)?, self.raw(bits)?);
                let n = if *wrap { format!("({n} & 31u)") } else { n };
                let ext = if *signed { format!("bitcast<u32>(extractBits(bitcast<i32>({x}), 0u, min({n}, 32u)))") } else { format!("extractBits({x}, 0u, min({n}, 32u))") };
                self.set(out, *dst, &format!("select({x}, {ext}, {n} < 32u)"));
            }
            other => return Err(format!("unsupported: {other:?}")),
        }
        Ok(())
    }

    fn moves(&mut self, out: &mut String, op: &MovesOp) -> R<()> {
        match op {
            MovesOp::Mov { dst, src, quad_lanes: 0xf } => {
                let e = self.raw(src)?;
                self.set(out, *dst, &e);
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
            MovesOp::Cs2r { dst, sr: 0xff, wide } => {
                self.set(out, *dst, "0u");
                if *wide {
                    self.set(out, dst + 1, "0u");
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
                        0x80..=0x27c => format!("u(i.a{}[{}])", (a - 0x80) / 16, a / 4 % 4),
                        0x2f8 => "i.iid".to_string(),
                        0x2fc => "i.vid".to_string(),
                        _ => return Err(format!("ALD of attribute {a:#x}")),
                    };
                    self.set(out, dst + k as u8, &e);
                }
            }
            MemoryOp::Ast { data, offset: RZ, vtx: RZ, attr, comps, patch: false, phys: false } if self.sph.stage == Stage::Vertex => {
                for k in 0..u16::from(*comps) {
                    let a = attr + 4 * k;
                    let v = self.r(data + k as u8);
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
            MemoryOp::Sust { coord, data, handle, dim, mask: 0xf, .. } if self.sph.stage == Stage::Compute => {
                let (coords, ty) = match dim {
                    ImageDim::D1 => (1, "texture_storage_1d"),
                    ImageDim::D2 => (2, "texture_storage_2d"),
                    ImageDim::D3 => (3, "texture_storage_3d"),
                    d => return Err(format!("SUST to a {d:?} image")),
                };
                let slot = self.iface.images.len();
                self.iface.images.push(Image { handle: Handle::CBuf { bank: handle.bank, offset: handle.offset }, dim: *dim });
                let format = self.images.get(slot).cloned().unwrap_or_else(|| "rgba16float".into());
                let g = self.group();
                let _ = writeln!(self.decls, "@group({g}) @binding({}) var img{slot}: {ty}<{format}, write>;", IMAGE_BINDING + slot as u32);
                let reg = |cx: &mut Cx, base: u8, k: u8| if base == RZ { "0u".to_string() } else { cx.r(base + k) };
                let cs: Vec<String> = (0..coords).map(|k| format!("i32({})", reg(self, *coord, k))).collect();
                let at = if coords == 1 { cs[0].clone() } else { format!("vec{coords}<i32>({})", cs.join(", ")) };
                let v: Vec<String> = (0..4).map(|k| format!("f({})", reg(self, *data, k))).collect();
                let _ = writeln!(out, "textureStore(img{slot}, {at}, vec4<f32>({}));", v.join(", "));
            }
            MemoryOp::Ldc { dst, index, bank, offset, mode: LdcMode::Indexed, ty } => {
                let small = match ty {
                    MemType::U8 => Some((8, false)),
                    MemType::S8 => Some((8, true)),
                    MemType::U16 => Some((16, false)),
                    MemType::S16 => Some((16, true)),
                    _ => None,
                };
                if let (Some((bits, signed)), RZ) = (small, *index) {
                    let e = self.cb_small(*bank, u32::from(*offset), bits, signed)?;
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
                        let e = self.cb(*bank, u32::from(*offset) + 4 * k)?;
                        self.set(out, dst + k as u8, &e);
                    }
                } else {
                    let ix = self.r(*index);
                    self.iface.cbufs.insert(*bank);
                    for k in 0..words {
                        let t = self.temp();
                        let _ = writeln!(out, "let {t} = ({ix} + {}u) & 0xffffu;", u32::from(*offset) + 4 * k);
                        self.set(out, dst + k as u8, &format!("cb{bank}[{t} >> 4u][({t} >> 2u) & 3u]"));
                    }
                }
            }
            other => return Err(format!("unsupported: {other:?}")),
        }
        Ok(())
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
        self.iface.textures.push(Texture { handle, dim, compare, fetch });
        let g = self.group();
        let _ = writeln!(self.decls, "@group({g}) @binding({}) var tex{slot}: {ty};", 32 + 2 * slot);
        if !fetch {
            let _ = writeln!(self.decls, "@group({g}) @binding({}) var smp{slot}: {};", 33 + 2 * slot, if compare { "sampler_comparison" } else { "sampler" });
        }
        slot
    }

    /// The enabled components of `v` (a vec4 or, with `scalar`, one value), in
    /// order, to dst, dst + 1, dst2, dst2 + 1.
    fn tex_results(&mut self, out: &mut String, v: &str, scalar: bool, mask: u8, dst: u8, dst2: u8) -> R<()> {
        let dsts = [Some(dst), Some(dst).filter(|d| *d != RZ).map(|d| d + 1), Some(dst2), Some(dst2).filter(|d| *d != RZ).map(|d| d + 1)];
        let comps: Vec<usize> = (0..4).filter(|c| mask >> c & 1 != 0).collect();
        for (k, c) in comps.iter().enumerate() {
            let Some(Some(d)) = dsts.get(k) else { return Err("texture destination".into()) };
            let value = if scalar { v.to_string() } else { format!("{v}[{c}]") };
            self.set(out, *d, &value);
        }
        Ok(())
    }

    fn texture(&mut self, out: &mut String, op: &TextureOp) -> R<()> {
        let (io, dim, fetch, lod, scr, aoffi, dc) = match op {
            TextureOp::Tex { io, dim, lod, scr, aoffi, dc, .. } => (io, *dim, false, *lod, *scr, *aoffi, *dc),
            TextureOp::Tld { io, dim, lod, scr, aoffi, ms: false } => (io, *dim, true, *lod, *scr, *aoffi, false),
            TextureOp::Txd { io, dim, aoffi } => return self.txd(out, io, *dim, *aoffi),
            // The size of a level (the texture is taken as 2D): width, height, 1, levels.
            TextureOp::Txq { dst, dst2, handle, src, query: TexQuery::Dimension, mask, .. } => {
                let slot = self.texture_slot(*handle, TexDim::D2, "texture_2d<f32>", false, true);
                let level = self.raw(src)?;
                let t = self.temp();
                let _ = writeln!(out, "let {t} = vec4<u32>(textureDimensions(tex{slot}, {level}), 1u, textureNumLevels(tex{slot}));");
                return self.tex_results(out, &t, false, *mask, *dst, *dst2);
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
        for (list, start) in [(&a, ra), (&b, rb)] {
            for (k, item) in list.iter().enumerate() {
                let r = if start == RZ { "0u".to_string() } else { self.r(start + k as u8) };
                regs.insert(key(*item), r);
            }
        }
        let get = |item: Item| -> R<String> { regs.get(&key(item)).cloned().ok_or_else(|| "texture source missing".to_string()) };
        let slot = self.texture_slot(io.handle, dim, ty, dc, fetch);
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
                format!("textureLoad(tex{slot}, {xy}{arr}, {level})")
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
                let parts: Vec<String> = (0..coords).map(|c| format!("f32(extractBits(bitcast<i32>({off}), {}u, 4u))", 4 * c)).collect();
                uv = if coords == 1 {
                    format!("({uv} + {} / f32(textureDimensions(tex{slot}, 0)))", parts[0])
                } else {
                    format!("({uv} + vec{coords}<f32>({}) / vec{coords}<f32>(textureDimensions(tex{slot}, 0)))", parts.join(", "))
                };
            }
            match (dc, lod) {
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
        if dc {
            let _ = writeln!(out, "let {t} = bitcast<u32>({call});");
        } else {
            let _ = writeln!(out, "let {t} = {};", self.swizzled(slot, &format!("bitcast<vec4<u32>>({call})")));
        }
        self.tex_results(out, &t, dc, io.mask, io.dst, io.dst2)
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
        self.tex_results(out, &t, false, io.mask, io.dst, io.dst2)
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
        assert_eq!(t.interface.textures, vec![Texture { handle: Handle::CBuf { bank: 2, offset: 0x70 }, dim: TexDim::D2, compare: false, fetch: false }]);
        assert_eq!(t.interface.targets, BTreeMap::from([(0, 1)]));
        assert!(t.wgsl.contains("switch pc"), "{}", t.wgsl);
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
        assert_eq!(t.interface.images, vec![Image { handle: Handle::CBuf { bank: 0, offset: 0x20 }, dim: ImageDim::D3 }]);
        for want in ["@workgroup_size(8, 8, 1)", "r0 = lid.x;", "r2 = wid.z;", "texture_storage_3d<rgba16float, write>", "textureStore(img0, vec3<i32>(i32(r0), i32(r1), i32(r2))"] {
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
