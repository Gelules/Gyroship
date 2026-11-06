//! Backend B: Cranelift IR, emitted as one native object file. Each guest
//! function becomes the exported symbol `f_<addr>` with the platform C
//! calling convention, so it links with the runtime like the Rust backend.
//!
//! ponytail: guest registers live in the `Cpu` struct and every access is a
//! load or store; caching them in SSA values between calls is the upgrade if
//! this backend wins.

use crate::ir::{Addr, Alu, Cond, Op, Operand, Reg, Shift, Unary, Writeback, W};
use crate::{Callee, Input};
use cranelift_codegen::ir::condcodes::IntCC;
use cranelift_codegen::ir::types::{I8, I16, I32, I64, Type};
use cranelift_codegen::ir::{
    AbiParam, AliasRegionData, Block, FuncRef, Function, InstBuilder, MemFlagsData, Signature, UserFuncName, Value,
};
use cranelift_codegen::settings::{self, Configurable};
use cranelift_codegen::Context;
use cranelift_frontend::{FunctionBuilder, FunctionBuilderContext, Switch};
use cranelift_module::{DataDescription, FuncId, Linkage, Module, default_libcall_names};
use cranelift_object::{ObjectBuilder, ObjectModule};
use std::collections::HashMap;

pub type Result<T> = std::result::Result<T, String>;

/// Field offsets of the runtime's `Cpu`, which checks them at compile time.
mod layout {
    pub use gyroship_runtime::{
        OFFSET_C as C, OFFSET_MEM as MEM, OFFSET_N as N, OFFSET_TPIDR as TPIDR, OFFSET_TPIDRRO as TPIDRRO, OFFSET_V as V,
        OFFSET_X as X, OFFSET_Z as Z,
    };
}

struct Runtime {
    call: FuncId,
    unhandled: FuncId,
    trap: FuncId,
    svc: FuncId,
    miss: FuncId,
}

pub struct Object {
    module: ObjectModule,
    sig: Signature,
    /// Every guest function this object refers to: its own (exported), others (imported).
    ids: HashMap<u64, FuncId>,
    defined: Vec<u64>,
    rt: Runtime,
    ctx: Context,
    fbc: FunctionBuilderContext,
}

impl Object {
    /// An object that will define the functions at `defined`; direct calls to
    /// any other guest function become imports, resolved when linking.
    pub fn new(defined: &[u64], verify: bool) -> Result<Object> {
        let mut flags = settings::builder();
        flags.set("opt_level", "speed").map_err(|e| e.to_string())?;
        flags.set("enable_verifier", if verify { "true" } else { "false" }).map_err(|e| e.to_string())?;
        let isa = cranelift_native::builder()?.finish(settings::Flags::new(flags)).map_err(|e| e.to_string())?;
        let builder = ObjectBuilder::new(isa, "guest", default_libcall_names()).map_err(|e| e.to_string())?;
        let mut module = ObjectModule::new(builder);
        let cc = module.isa().default_call_conv();
        let ptr = module.target_config().pointer_type();
        let sig_with = |params: &[Type]| {
            let mut s = Signature::new(cc);
            s.params.extend(params.iter().map(|&t| AbiParam::new(t)));
            s
        };
        let sig = sig_with(&[ptr]);
        let mut import = |name: &str, params: &[Type]| module.declare_function(name, Linkage::Import, &sig_with(params)).map_err(|e| e.to_string());
        let rt = Runtime {
            call: import("gyroship_call", &[ptr, I64])?,
            unhandled: import("gyroship_unhandled", &[ptr, I64, I32])?,
            trap: import("gyroship_trap", &[ptr, I64, I32])?,
            svc: import("gyroship_svc", &[ptr, I64, I32])?,
            miss: import("gyroship_miss", &[ptr, I64])?,
        };
        let mut ids = HashMap::new();
        for &a in defined {
            ids.insert(a, module.declare_function(&crate::rust::name(a), Linkage::Export, &sig).map_err(|e| e.to_string())?);
        }
        Ok(Object { module, sig, ids, defined: defined.to_vec(), rt, ctx: Context::new(), fbc: FunctionBuilderContext::new() })
    }

    /// Defines `symbol`, the dispatch table of this object: for each defined
    /// function, its guest address then its host address (16 bytes), and
    /// `<symbol>_len`, the entry count.
    pub fn table(&mut self, symbol: &str) -> Result<()> {
        let err = |e: cranelift_module::ModuleError| e.to_string();
        let mut contents = vec![0u8; 16 * self.defined.len()];
        for (i, a) in self.defined.iter().enumerate() {
            contents[16 * i..16 * i + 8].copy_from_slice(&a.to_le_bytes());
        }
        let mut table = DataDescription::new();
        table.define(contents.into_boxed_slice());
        table.set_align(8);
        for (i, a) in self.defined.iter().enumerate() {
            let f = self.module.declare_func_in_data(self.ids[a], &mut table);
            table.write_function_addr((16 * i + 8) as u32, f);
        }
        let id = self.module.declare_data(symbol, Linkage::Export, false, false).map_err(err)?;
        self.module.define_data(id, &table).map_err(err)?;
        let mut len = DataDescription::new();
        len.define(Box::new((self.defined.len() as u64).to_le_bytes()));
        len.set_align(8);
        let id = self.module.declare_data(&format!("{symbol}_len"), Linkage::Export, false, false).map_err(err)?;
        self.module.define_data(id, &len).map_err(err)
    }

    pub fn function(&mut self, input: &Input) -> Result<()> {
        let f = input.function;
        let id = self.ids[&f.start];
        self.ctx.func = Function::with_name_signature(UserFuncName::user(0, id.as_u32()), self.sig.clone());
        let mut refs = HashMap::new();
        let config = self.module.target_config();
        let module = &mut self.module;
        let rt = &self.rt;
        let ids = &mut self.ids;
        let sig = &self.sig;
        let mut b = FunctionBuilder::new(&mut self.ctx.func, &mut self.fbc);
        let cpu_region = b.func.dfg.alias_regions.insert(AliasRegionData { user_id: 1, description: "cpu".into() });
        let heap_region = b.func.dfg.alias_regions.insert(AliasRegionData { user_id: 2, description: "guest".into() });
        let cpu_flags = MemFlagsData::trusted().with_alias_region(Some(cpu_region));
        let heap_flags =
            MemFlagsData::new().with_notrap().with_alias_region(Some(heap_region));
        let entry = b.create_block();
        b.append_block_params_for_function_params(entry);
        b.switch_to_block(entry);
        let cpu = b.block_params(entry)[0];
        let mem = b.ins().load(I64, cpu_flags, cpu, layout::MEM);
        let blocks: Vec<Block> = input.cfg.blocks.iter().map(|_| b.create_block()).collect();
        let index = |a: u64| input.cfg.blocks.binary_search_by_key(&a, |b| b.start).ok();
        let first = blocks[index(f.start).unwrap_or(0)];
        b.ins().jump(first, &[]);
        let mut e = Emit { b, cpu, mem, cpu_flags, heap_flags, module, rt, ids, sig, refs: &mut refs, callee: input.callee };
        for (bi, block) in input.cfg.blocks.iter().enumerate() {
            e.b.switch_to_block(blocks[bi]);
            let end = block.start + block.len;
            let mut ended = false;
            for pc in (block.start..end).step_by(4) {
                let word = (input.word)(pc);
                let op = crate::lift::lift(word, pc);
                match op {
                    Op::Jump(t) => {
                        ended = true;
                        e.goto(t, index(t).map(|i| blocks[i]));
                    }
                    Op::Branch(cnd, t) => {
                        ended = true;
                        let c = e.cond(cnd);
                        let taken = e.target_block(t, index(t).map(|i| blocks[i]));
                        let next = e.target_block(pc + 4, index(pc + 4).map(|i| blocks[i]));
                        e.b.ins().brif(c, taken, &[], next, &[]);
                    }
                    Op::Call(t) => {
                        let ret = e.b.ins().iconst(I64, (pc + 4) as i64);
                        e.set(Reg::X(30), W::W64, ret);
                        e.call(t);
                    }
                    Op::CallIndirect(r) => {
                        let t = e.get(r, W::W64);
                        let ret = e.b.ins().iconst(I64, (pc + 4) as i64);
                        e.set(Reg::X(30), W::W64, ret);
                        let f = e.func(e.rt.call);
                        e.b.ins().call(f, &[cpu, t]);
                    }
                    Op::JumpIndirect(r) => {
                        ended = true;
                        let t = e.get(r, W::W64);
                        match input.cfg.tables.iter().find(|t| t.branch == pc) {
                            Some(table) => {
                                let mut sw = Switch::new();
                                let mut targets = table.targets.clone();
                                targets.sort_unstable();
                                targets.dedup();
                                for x in targets {
                                    if let Some(i) = index(x) {
                                        sw.set_entry(u128::from(x), blocks[i]);
                                    }
                                }
                                let miss = e.b.create_block();
                                sw.emit(&mut e.b, t, miss);
                                e.b.switch_to_block(miss);
                                let f = e.func(e.rt.miss);
                                e.b.ins().call(f, &[cpu, t]);
                                e.b.ins().return_(&[]);
                            }
                            None => {
                                let f = e.func(e.rt.call);
                                e.b.ins().call(f, &[cpu, t]);
                                e.b.ins().return_(&[]);
                            }
                        }
                    }
                    Op::Return => {
                        ended = true;
                        e.b.ins().return_(&[]);
                    }
                    Op::Trap(imm) => {
                        ended = true;
                        e.runtime(e.rt.trap, pc, u32::from(imm));
                        e.b.ins().return_(&[]);
                    }
                    Op::Svc(imm) => e.runtime(e.rt.svc, pc, u32::from(imm)),
                    Op::Unhandled => e.runtime(e.rt.unhandled, pc, word),
                    ref op => e.data(op),
                }
            }
            if !ended {
                e.goto(end, index(end).map(|i| blocks[i]));
            }
        }
        e.b.seal_all_blocks();
        e.b.finalize(config);
        self.module.define_function(id, &mut self.ctx).map_err(|e| format!("{e:?}"))?;
        self.module.clear_context(&mut self.ctx);
        Ok(())
    }

    pub fn finish(self) -> Result<Vec<u8>> {
        self.module.finish().emit().map_err(|e| e.to_string())
    }
}

struct Emit<'a, 'b> {
    b: FunctionBuilder<'b>,
    cpu: Value,
    mem: Value,
    cpu_flags: MemFlagsData,
    heap_flags: MemFlagsData,
    module: &'a mut ObjectModule,
    rt: &'a Runtime,
    ids: &'a mut HashMap<u64, FuncId>,
    sig: &'a Signature,
    refs: &'a mut HashMap<FuncId, FuncRef>,
    callee: &'a dyn Fn(u64) -> Callee,
}

fn ty(w: W) -> Type {
    match w {
        W::W32 => I32,
        W::W64 => I64,
    }
}

fn mask(bits: u32) -> i64 {
    if bits >= 64 { -1 } else { ((1u64 << bits) - 1) as i64 }
}

impl Emit<'_, '_> {
    fn func(&mut self, id: FuncId) -> FuncRef {
        *self.refs.entry(id).or_insert_with(|| self.module.declare_func_in_func(id, self.b.func))
    }

    fn get(&mut self, r: Reg, w: W) -> Value {
        let slot = match r {
            Reg::X(n) => i32::from(n),
            Reg::Sp => 31,
            Reg::Zr => return self.b.ins().iconst(ty(w), 0),
        };
        let v = self.b.ins().load(I64, self.cpu_flags, self.cpu, layout::X + 8 * slot);
        if w == W::W32 { self.b.ins().ireduce(I32, v) } else { v }
    }

    fn set(&mut self, r: Reg, w: W, v: Value) {
        let slot = match r {
            Reg::X(n) => i32::from(n),
            Reg::Sp => 31,
            Reg::Zr => return,
        };
        let v = if w == W::W32 { self.b.ins().uextend(I64, v) } else { v };
        self.b.ins().store(self.cpu_flags, v, self.cpu, layout::X + 8 * slot);
    }

    fn flag(&mut self, off: i32) -> Value {
        self.b.ins().load(I8, self.cpu_flags, self.cpu, off)
    }

    fn set_flags(&mut self, n: Value, z: Value, c: Value, v: Value) {
        for (val, off) in [(n, layout::N), (z, layout::Z), (c, layout::C), (v, layout::V)] {
            self.b.ins().store(self.cpu_flags, val, self.cpu, off);
        }
    }

    /// NZCV of a + b or a - b, and the result.
    fn add_sub_flags(&mut self, a: Value, b: Value, sub: bool) -> (Value, [Value; 4]) {
        let i = &mut self.b.ins();
        let _ = i;
        let r = if sub { self.b.ins().isub(a, b) } else { self.b.ins().iadd(a, b) };
        let n = self.b.ins().icmp_imm_s(IntCC::SignedLessThan, r, 0);
        let z = self.b.ins().icmp_imm_s(IntCC::Equal, r, 0);
        let c = if sub { self.b.ins().icmp(IntCC::UnsignedGreaterThanOrEqual, a, b) } else { self.b.ins().icmp(IntCC::UnsignedLessThan, r, a) };
        let ab = self.b.ins().bxor(a, b);
        let ar = self.b.ins().bxor(a, r);
        let t = if sub {
            self.b.ins().band(ab, ar)
        } else {
            let nab = self.b.ins().bnot(ab);
            self.b.ins().band(nab, ar)
        };
        let v = self.b.ins().icmp_imm_s(IntCC::SignedLessThan, t, 0);
        (r, [n, z, c, v])
    }

    fn cond(&mut self, c: Cond) -> Value {
        match c {
            Cond::Flags(code) => self.flags_cond(code),
            Cond::Zero { reg, w, nonzero } => {
                let v = self.get(reg, w);
                self.b.ins().icmp_imm_s(if nonzero { IntCC::NotEqual } else { IntCC::Equal }, v, 0)
            }
            Cond::Bit { reg, bit, set } => {
                let v = self.get(reg, W::W64);
                let m = self.b.ins().band_imm_s(v, 1i64 << bit);
                self.b.ins().icmp_imm_s(if set { IntCC::NotEqual } else { IntCC::Equal }, m, 0)
            }
        }
    }

    fn flags_cond(&mut self, code: u8) -> Value {
        if code >= 14 {
            return self.b.ins().iconst(I8, 1);
        }
        let r = match code >> 1 {
            0 => self.flag(layout::Z),
            1 => self.flag(layout::C),
            2 => self.flag(layout::N),
            3 => self.flag(layout::V),
            4 => {
                let (c, z) = (self.flag(layout::C), self.flag(layout::Z));
                self.b.ins().icmp(IntCC::UnsignedGreaterThan, c, z) // c && !z for 0/1 values
            }
            5 => {
                let (n, v) = (self.flag(layout::N), self.flag(layout::V));
                self.b.ins().icmp(IntCC::Equal, n, v)
            }
            _ => {
                let (n, v, z) = (self.flag(layout::N), self.flag(layout::V), self.flag(layout::Z));
                let ge = self.b.ins().icmp(IntCC::Equal, n, v);
                self.b.ins().icmp(IntCC::UnsignedGreaterThan, ge, z)
            }
        };
        if code & 1 == 1 { self.b.ins().bxor_imm_s(r, 1) } else { r }
    }

    fn operand(&mut self, o: Operand, w: W) -> Value {
        match o {
            Operand::Imm(v) => self.b.ins().iconst(ty(w), if w == W::W32 { (v & 0xFFFF_FFFF) as i64 } else { v as i64 }),
            Operand::Shifted(r, sh, amount) => {
                let v = self.get(r, w);
                let a = i64::from(amount);
                match (sh, amount) {
                    (_, 0) if sh != Shift::Ror => v,
                    (Shift::Lsl, _) => self.b.ins().ishl_imm_u(v, a),
                    (Shift::Lsr, _) => self.b.ins().ushr_imm_u(v, a),
                    (Shift::Asr, _) => self.b.ins().sshr_imm_u(v, a),
                    (Shift::Ror, _) => self.b.ins().rotr_imm_u(v, a),
                }
            }
            Operand::Extended { reg, signed, bits, lsl } => {
                let v = self.get(reg, W::W64);
                let v = match bits {
                    64 => v,
                    b => {
                        let narrow = self.b.ins().ireduce(match b { 8 => I8, 16 => I16, _ => I32 }, v);
                        if signed { self.b.ins().sextend(I64, narrow) } else { self.b.ins().uextend(I64, narrow) }
                    }
                };
                let v = if lsl > 0 { self.b.ins().ishl_imm_u(v, i64::from(lsl)) } else { v };
                if w == W::W32 { self.b.ins().ireduce(I32, v) } else { v }
            }
        }
    }

    fn address(&mut self, a: &Addr) -> (Value, Value) {
        let base = self.get(a.base, W::W64);
        let at = match a.index {
            Some((r, signed, bits, sh)) => {
                let i = self.operand(Operand::Extended { reg: r, signed, bits, lsl: sh }, W::W64);
                self.b.ins().iadd(base, i)
            }
            None if a.writeback == Writeback::Post || a.offset == 0 => base,
            None => self.b.ins().iadd_imm_s(base, a.offset),
        };
        match a.writeback {
            Writeback::None => {}
            Writeback::Pre => self.set(a.base, W::W64, at),
            Writeback::Post => {
                let next = self.b.ins().iadd_imm_s(at, a.offset);
                self.set(a.base, W::W64, next);
            }
        }
        let host = self.b.ins().iadd(self.mem, at);
        (at, host)
    }

    fn load(&mut self, host: Value, off: i32, w: W, size: u8, signed: bool) -> Value {
        let (t, f) = (ty(w), self.heap_flags);
        let i = self.b.ins();
        match (size, signed) {
            (0, false) => i.uload8(t, f, host, off),
            (0, true) => i.sload8(t, f, host, off),
            (1, false) => i.uload16(t, f, host, off),
            (1, true) => i.sload16(t, f, host, off),
            (2, _) if w == W::W32 => i.load(I32, f, host, off),
            (2, false) => i.uload32(f, host, off),
            (2, true) => i.sload32(f, host, off),
            _ => i.load(I64, f, host, off),
        }
    }

    fn store(&mut self, v: Value, host: Value, off: i32, size: u8) {
        let f = self.heap_flags;
        let i = self.b.ins();
        match size {
            0 => i.istore8(f, v, host, off),
            1 => i.istore16(f, v, host, off),
            2 => i.istore32(f, v, host, off),
            _ => i.store(f, v, host, off),
        };
    }

    fn data(&mut self, op: &Op) {
        match *op {
            Op::Alu { op, w, dst, a, b, flags } => {
                let (x, y) = (self.get(a, w), self.operand(b, w));
                let r = match op {
                    Alu::Add | Alu::Sub if flags => {
                        let (r, [n, z, c, v]) = self.add_sub_flags(x, y, op == Alu::Sub);
                        self.set_flags(n, z, c, v);
                        r
                    }
                    Alu::Add => self.b.ins().iadd(x, y),
                    Alu::Sub => self.b.ins().isub(x, y),
                    Alu::And => self.b.ins().band(x, y),
                    Alu::Orr => self.b.ins().bor(x, y),
                    Alu::Eor => self.b.ins().bxor(x, y),
                    Alu::Bic => self.b.ins().band_not(x, y),
                    Alu::Orn => self.b.ins().bor_not(x, y),
                    Alu::Eon => self.b.ins().bxor_not(x, y),
                };
                if flags && !matches!(op, Alu::Add | Alu::Sub) {
                    let n = self.b.ins().icmp_imm_s(IntCC::SignedLessThan, r, 0);
                    let z = self.b.ins().icmp_imm_s(IntCC::Equal, r, 0);
                    let zero = self.b.ins().iconst(I8, 0);
                    self.set_flags(n, z, zero, zero);
                }
                self.set(dst, w, r);
            }
            Op::MovWide { w, dst, imm, shift, invert, keep } => {
                let v = u64::from(imm) << shift;
                let r = if keep {
                    let old = self.get(dst, w);
                    let cleared = self.b.ins().band_imm_s(old, !(0xFFFFi64 << shift));
                    self.b.ins().bor_imm_s(cleared, v as i64)
                } else {
                    let v = if invert { !v } else { v };
                    self.b.ins().iconst(ty(w), if w == W::W32 { (v & 0xFFFF_FFFF) as i64 } else { v as i64 })
                };
                self.set(dst, w, r);
            }
            Op::Bitfield { w, dst, src, immr, imms, signed, insert } => {
                let (n, s, r) = (i64::from(w.bits()), i64::from(imms), i64::from(immr));
                let v = self.get(src, w);
                let out = if insert {
                    let old = self.get(dst, w);
                    let (field, m) = if s >= r {
                        let f = self.b.ins().ushr_imm_u(v, r);
                        (f, mask((s - r + 1) as u32))
                    } else {
                        let f = self.b.ins().ishl_imm_u(v, n - r);
                        (f, mask((s + 1) as u32) << (n - r))
                    };
                    let f = self.b.ins().band_imm_s(field, m);
                    let keep = self.b.ins().band_imm_s(old, !m);
                    self.b.ins().bor(keep, f)
                } else if signed {
                    if s >= r {
                        let up = self.b.ins().ishl_imm_u(v, n - 1 - s);
                        self.b.ins().sshr_imm_u(up, n - (s - r + 1))
                    } else {
                        let up = self.b.ins().ishl_imm_u(v, n - (s + 1));
                        let ext = self.b.ins().sshr_imm_u(up, n - (s + 1));
                        self.b.ins().ishl_imm_u(ext, n - r)
                    }
                } else if s >= r {
                    let f = self.b.ins().ushr_imm_u(v, r);
                    self.b.ins().band_imm_s(f, mask((s - r + 1) as u32))
                } else {
                    let f = self.b.ins().band_imm_s(v, mask((s + 1) as u32));
                    self.b.ins().ishl_imm_u(f, n - r)
                };
                self.set(dst, w, out);
            }
            Op::Extr { w, dst, a, b, lsb } => {
                let lo = self.get(b, w);
                let r = if lsb == 0 {
                    lo
                } else {
                    let hi = self.get(a, w);
                    let l = self.b.ins().ushr_imm_u(lo, i64::from(lsb));
                    let h = self.b.ins().ishl_imm_u(hi, i64::from(w.bits()) - i64::from(lsb));
                    self.b.ins().bor(l, h)
                };
                self.set(dst, w, r);
            }
            Op::ShiftVar { op, w, dst, a, b } => {
                let (x, y) = (self.get(a, w), self.get(b, w));
                let r = match op {
                    Shift::Lsl => self.b.ins().ishl(x, y),
                    Shift::Lsr => self.b.ins().ushr(x, y),
                    Shift::Asr => self.b.ins().sshr(x, y),
                    Shift::Ror => self.b.ins().rotr(x, y),
                };
                self.set(dst, w, r);
            }
            Op::MulAdd { w, dst, a, b, acc, sub } => {
                let (x, y, z) = (self.get(a, w), self.get(b, w), self.get(acc, w));
                let p = self.b.ins().imul(x, y);
                let r = if sub { self.b.ins().isub(z, p) } else { self.b.ins().iadd(z, p) };
                self.set(dst, w, r);
            }
            Op::MulAddLong { dst, a, b, acc, signed, sub } => {
                let (x, y, z) = (self.get(a, W::W32), self.get(b, W::W32), self.get(acc, W::W64));
                let (x, y) = if signed {
                    (self.b.ins().sextend(I64, x), self.b.ins().sextend(I64, y))
                } else {
                    (self.b.ins().uextend(I64, x), self.b.ins().uextend(I64, y))
                };
                let p = self.b.ins().imul(x, y);
                let r = if sub { self.b.ins().isub(z, p) } else { self.b.ins().iadd(z, p) };
                self.set(dst, W::W64, r);
            }
            Op::MulHigh { dst, a, b, signed } => {
                let (x, y) = (self.get(a, W::W64), self.get(b, W::W64));
                let r = if signed { self.b.ins().smulhi(x, y) } else { self.b.ins().umulhi(x, y) };
                self.set(dst, W::W64, r);
            }
            Op::Div { w, dst, a, b, signed } => {
                let (x, y) = (self.get(a, w), self.get(b, w));
                let t = ty(w);
                let zero = self.b.ins().icmp_imm_s(IntCC::Equal, y, 0);
                let one = self.b.ins().iconst(t, 1);
                let mut safe = self.b.ins().select(zero, one, y);
                if signed {
                    // MIN / -1 overflows; dividing by 1 gives the architectural result (MIN).
                    let min = if w == W::W32 { i64::from(i32::MIN) } else { i64::MIN };
                    let is_min = self.b.ins().icmp_imm_s(IntCC::Equal, x, min);
                    let is_m1 = self.b.ins().icmp_imm_s(IntCC::Equal, y, -1);
                    let both = self.b.ins().band(is_min, is_m1);
                    safe = self.b.ins().select(both, one, safe);
                }
                let q = if signed { self.b.ins().sdiv(x, safe) } else { self.b.ins().udiv(x, safe) };
                let z = self.b.ins().iconst(t, 0);
                let r = self.b.ins().select(zero, z, q);
                self.set(dst, w, r);
            }
            Op::CondSel { w, dst, a, b, cond, inc, inv } => {
                let c = self.flags_cond(cond);
                let (x, y) = (self.get(a, w), self.get(b, w));
                let y = match (inc, inv) {
                    (false, false) => y,
                    (true, false) => self.b.ins().iadd_imm_s(y, 1),
                    (false, true) => self.b.ins().bnot(y),
                    (true, true) => self.b.ins().ineg(y),
                };
                let r = self.b.ins().select(c, x, y);
                self.set(dst, w, r);
            }
            Op::CondCmp { w, a, b, nzcv, cond, add } => {
                let c = self.flags_cond(cond);
                let (x, y) = (self.get(a, w), self.operand(b, w));
                let (_, flags) = self.add_sub_flags(x, y, !add);
                let mut out = [flags[0]; 4];
                for (k, f) in flags.into_iter().enumerate() {
                    let imm = self.b.ins().iconst(I8, i64::from((nzcv >> (3 - k)) & 1));
                    out[k] = self.b.ins().select(c, f, imm);
                }
                self.set_flags(out[0], out[1], out[2], out[3]);
            }
            Op::Unary { op, w, dst, src } => {
                let v = self.get(src, w);
                let r = match op {
                    Unary::Clz => self.b.ins().clz(v),
                    Unary::Rbit => self.b.ins().bitrev(v),
                    Unary::Rev => self.b.ins().bswap(v),
                    Unary::Rev16 => {
                        let m = if w == W::W32 { 0x00FF_00FFi64 } else { 0x00FF_00FF_00FF_00FF };
                        let lo = self.b.ins().ushr_imm_u(v, 8);
                        let lo = self.b.ins().band_imm_s(lo, m);
                        let hi = self.b.ins().band_imm_s(v, m);
                        let hi = self.b.ins().ishl_imm_u(hi, 8);
                        self.b.ins().bor(lo, hi)
                    }
                    Unary::Rev32 => {
                        let s = self.b.ins().bswap(v);
                        self.b.ins().rotr_imm_u(s, 32)
                    }
                };
                self.set(dst, w, r);
            }
            Op::Const { dst, value } => {
                let v = self.b.ins().iconst(I64, value as i64);
                self.set(dst, W::W64, v);
            }
            Op::Load { dst, w, size, signed, addr } => {
                let (_, host) = self.address(&addr);
                let v = self.load(host, 0, w, size, signed);
                self.set(dst, w, v);
            }
            Op::Store { src, size, addr } => {
                let v = self.get(src, W::W64);
                let (_, host) = self.address(&addr);
                self.store(v, host, 0, size);
            }
            Op::LoadPair { a, b, w, size, signed, addr } => {
                let (_, host) = self.address(&addr);
                let v1 = self.load(host, 0, w, size, signed);
                let v2 = self.load(host, 1 << size, w, size, signed);
                self.set(a, w, v1);
                self.set(b, w, v2);
            }
            Op::StorePair { a, b, size, addr } => {
                let (v1, v2) = (self.get(a, W::W64), self.get(b, W::W64));
                let (_, host) = self.address(&addr);
                self.store(v1, host, 0, size);
                self.store(v2, host, 1 << size, size);
            }
            Op::ReadThread { dst, ro } => {
                let v = self.b.ins().load(I64, self.cpu_flags, self.cpu, if ro { layout::TPIDRRO } else { layout::TPIDR });
                self.set(dst, W::W64, v);
            }
            Op::Nop => {}
            _ => unreachable!("control flow is emitted by the block"),
        }
    }

    fn runtime(&mut self, id: FuncId, pc: u64, imm: u32) {
        let f = self.func(id);
        let pc = self.b.ins().iconst(I64, pc as i64);
        let imm = self.b.ins().iconst(I32, i64::from(imm));
        self.b.ins().call(f, &[self.cpu, pc, imm]);
    }

    fn call(&mut self, t: u64) {
        match (self.callee)(t) {
            Callee::Direct(t) => {
                let id = match self.ids.get(&t) {
                    Some(&id) => id,
                    None => {
                        let id = self.module.declare_function(&crate::rust::name(t), Linkage::Import, self.sig).expect("a fresh import");
                        self.ids.insert(t, id);
                        id
                    }
                };
                let f = self.func(id);
                self.b.ins().call(f, &[self.cpu]);
            }
            Callee::Dispatch(t) => {
                let f = self.func(self.rt.call);
                let t = self.b.ins().iconst(I64, t as i64);
                self.b.ins().call(f, &[self.cpu, t]);
            }
        }
    }

    /// Jumps to a block of the function, or tail-calls `t`.
    fn goto(&mut self, t: u64, block: Option<Block>) {
        match block {
            Some(blk) => {
                self.b.ins().jump(blk, &[]);
            }
            None => {
                self.call(t);
                self.b.ins().return_(&[]);
            }
        }
    }

    /// A block that continues at `t`: the function's own, or a new one that tail-calls it.
    fn target_block(&mut self, t: u64, block: Option<Block>) -> Block {
        if let Some(blk) = block {
            return blk;
        }
        let here = self.b.current_block().expect("in a block");
        let tail = self.b.create_block();
        self.b.switch_to_block(tail);
        self.call(t);
        self.b.ins().return_(&[]);
        self.b.switch_to_block(here);
        tail
    }
}
