//! Backend B: Cranelift IR, emitted as one native object file. Each guest
//! function becomes the exported symbol `f_<addr>(cpu, resume)` with the
//! platform C calling convention, so it links with the runtime like the Rust
//! backend.
//!
//! Control transfers follow D20 (see `gyroship_runtime`): `RET` writes
//! `cpu.pc`; after each call, `cpu.pc` must be the continuation, or the
//! function's rescue block takes over. The rescue block switches on the
//! function's resume points (its entries, and the continuations of its calls
//! and SVCs); a miss returns (after a call), dispatches (after an indirect
//! jump) or is fatal (when entered with `resume`).
//!
//! ponytail: guest registers live in the `Cpu` struct and every access is a
//! load or store; caching them in SSA values between calls is the upgrade if
//! this backend wins.

use crate::ir::{Addr, Alu, Cond, FBin, FUn, Op, Operand, Reduce, Reg, Rmw, Select, Shift, Unary, VCmp, VFp, VInt, VIntUn, Writeback, W, ZERO};
use crate::{Callee, Input};
use cranelift_codegen::ir::condcodes::{FloatCC, IntCC};
use cranelift_codegen::ir::types::{F32, F32X4, F64, F64X2, I8, I8X16, I16, I16X8, I32, I32X4, I64, I64X2, Type};
use cranelift_codegen::ir::{
    AbiParam, AliasRegionData, AtomicRmwOp, Block, ConstantData, Endianness, FuncRef, Function, InstBuilder, MemFlagsData, Signature,
    UserFuncName, Value,
};
use cranelift_codegen::isa::OwnedTargetIsa;
use cranelift_codegen::settings::{self, Configurable};
use cranelift_codegen::Context;
use cranelift_frontend::{FunctionBuilder, FunctionBuilderContext, Switch};
use cranelift_module::{DataDescription, FuncId, Linkage, Module, default_libcall_names};
use cranelift_object::{ObjectBuilder, ObjectModule};
use std::collections::{BTreeMap, BTreeSet, HashMap};

pub type Result<T> = std::result::Result<T, String>;

/// Field offsets of the runtime's `Cpu`, which checks them at compile time.
mod layout {
    pub use gyroship_runtime::{
        OFFSET_C as C, OFFSET_MEM as MEM, OFFSET_N as N, OFFSET_PC as PC, OFFSET_STACK_LIMIT as STACK_LIMIT, OFFSET_TPIDR as TPIDR,
        OFFSET_TPIDRRO as TPIDRRO, OFFSET_V as V, OFFSET_VR as VR, OFFSET_WORK as WORK, OFFSET_X as X, OFFSET_Z as Z,
    };
}

/// What the rescue block does when the target is not a resume point.
const RETURN: i64 = 0;
const DISPATCH: i64 = 1;
const MISS: i64 = 2;

struct Runtime {
    call: FuncId,
    unhandled: FuncId,
    trap: FuncId,
    svc: FuncId,
    miss: FuncId,
    counter: FuncId,
    yield_: FuncId,
    probe: FuncId,
    crc32: FuncId,
}

/// The functions of one output: an object file (`ObjectModule`), or JIT
/// code in tests.
pub struct Object<M: Module = ObjectModule> {
    module: M,
    sig: Signature,
    /// Every guest function this object refers to: its own (exported), others (imported).
    ids: HashMap<u64, FuncId>,
    /// Start and size of each function this object defines.
    defined: Vec<(u64, u64)>,
    /// Resume points of the functions generated so far.
    resume_points: usize,
    rt: Runtime,
    ctx: Context,
    fbc: FunctionBuilderContext,
}

/// The host target, optimizing for speed, with the IR verifier if asked.
pub fn isa(verify: bool) -> Result<OwnedTargetIsa> {
    let mut flags = settings::builder();
    flags.set("opt_level", "speed").map_err(|e| e.to_string())?;
    flags.set("enable_verifier", if verify { "true" } else { "false" }).map_err(|e| e.to_string())?;
    cranelift_native::builder()?.finish(settings::Flags::new(flags)).map_err(|e| e.to_string())
}

impl Object<ObjectModule> {
    /// An object that will define the functions in `defined` (start, size);
    /// direct calls to any other guest function become imports, resolved
    /// when linking.
    pub fn new(defined: &[(u64, u64)], verify: bool) -> Result<Object> {
        let builder = ObjectBuilder::new(isa(verify)?, "guest", default_libcall_names()).map_err(|e| e.to_string())?;
        Object::with_module(ObjectModule::new(builder), defined)
    }

    pub fn finish(self) -> Result<Vec<u8>> {
        self.module.finish().emit().map_err(|e| e.to_string())
    }
}

impl<M: Module> Object<M> {
    /// Like [`Object::new`], into any Cranelift module.
    pub fn with_module(mut module: M, defined: &[(u64, u64)]) -> Result<Object<M>> {
        let cc = module.isa().default_call_conv();
        let ptr = module.target_config().pointer_type();
        let sig_with = |params: &[Type]| {
            let mut s = Signature::new(cc);
            s.params.extend(params.iter().map(|&t| AbiParam::new(t)));
            s
        };
        let sig = sig_with(&[ptr, I64]);
        let mut crc_sig = sig_with(&[I32, I64, I32]);
        crc_sig.returns.push(AbiParam::new(I32));
        let crc32 = module.declare_function("gyroship_crc32", Linkage::Import, &crc_sig).map_err(|e| e.to_string())?;
        let mut import = |name: &str, params: &[Type]| module.declare_function(name, Linkage::Import, &sig_with(params)).map_err(|e| e.to_string());
        let rt = Runtime {
            call: import("gyroship_call", &[ptr, I64])?,
            unhandled: import("gyroship_unhandled", &[ptr, I64, I32])?,
            trap: import("gyroship_trap", &[ptr, I64, I32])?,
            svc: import("gyroship_svc", &[ptr, I64, I32])?,
            miss: import("gyroship_miss", &[ptr, I64])?,
            counter: import("gyroship_counter", &[ptr, I64, I32])?,
            yield_: import("gyroship_yield", &[ptr, I64, I32])?,
            probe: import("gyroship_probe", &[ptr, I64])?,
            crc32,
        };
        let mut ids = HashMap::new();
        for &(a, _) in defined {
            ids.insert(a, module.declare_function(&crate::rust::name(a), Linkage::Export, &sig).map_err(|e| e.to_string())?);
        }
        Ok(Object { module, sig, ids, defined: defined.to_vec(), resume_points: 0, rt, ctx: Context::new(), fbc: FunctionBuilderContext::new() })
    }

    /// Defines `symbol`, the dispatch table of this object: for each defined
    /// function, its guest address, its size, then its host address (the
    /// runtime's `Entry`, 24 bytes), and `<symbol>_len`, the entry count.
    pub fn table(&mut self, symbol: &str) -> Result<()> {
        let err = |e: cranelift_module::ModuleError| e.to_string();
        let mut contents = vec![0u8; 24 * self.defined.len()];
        for (i, (a, len)) in self.defined.iter().enumerate() {
            contents[24 * i..24 * i + 8].copy_from_slice(&a.to_le_bytes());
            contents[24 * i + 8..24 * i + 16].copy_from_slice(&len.to_le_bytes());
        }
        let mut table = DataDescription::new();
        table.define(contents.into_boxed_slice());
        table.set_align(8);
        for (i, (a, _)) in self.defined.iter().enumerate() {
            let f = self.module.declare_func_in_data(self.ids[a], &mut table);
            table.write_function_addr((24 * i + 16) as u32, f);
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
        let (cpu, resume) = (b.block_params(entry)[0], b.block_params(entry)[1]);
        let mem = b.ins().load(I64, cpu_flags, cpu, layout::MEM);
        let blocks: Vec<Block> = input.cfg.blocks.iter().map(|_| b.create_block()).collect();
        let index = |a: u64| input.cfg.blocks.binary_search_by_key(&a, |b| b.start).ok();
        let first = blocks[index(f.start).unwrap_or(0)];
        // The rescue block: (target, what to do when it is not a resume point).
        let rescue = b.create_block();
        b.append_block_param(rescue, I64);
        b.append_block_param(rescue, I8);
        let resuming = b.ins().icmp_imm_s(IntCC::NotEqual, resume, 0);
        let miss = b.ins().iconst(I8, MISS);
        b.ins().brif(resuming, rescue, &[resume.into(), miss.into()], first, &[]);
        // Resume points: guest address -> block. Entries, then continuations.
        let mut points: BTreeMap<u64, Block> = f.entries.iter().filter_map(|&a| Some((a, blocks[index(a)?]))).collect();
        let mut e = Emit { b, cpu, mem, cpu_flags, heap_flags, module, rt, ids, sig, refs: &mut refs, callee: input.callee, pending: Vec::new() };
        // Loop heads: blocks a branch at the end of a block reaches backward.
        let heads: BTreeSet<u64> = input
            .cfg
            .blocks
            .iter()
            .filter(|b| b.len >= 4)
            .filter_map(|b| {
                let last = b.start + b.len - 4;
                match crate::lift::lift((input.word)(last), last) {
                    Op::Jump(t) | Op::Branch(_, t) if t <= last => Some(t),
                    _ => None,
                }
            })
            .collect();
        for (bi, block) in input.cfg.blocks.iter().enumerate() {
            e.b.switch_to_block(blocks[bi]);
            // The guest work done: the block's instructions, counted as it starts.
            let work = e.b.ins().load(I64, e.cpu_flags, cpu, layout::WORK);
            let work = e.b.ins().iadd_imm_s(work, (block.len / 4) as i64);
            e.b.ins().store(e.cpu_flags, work, cpu, layout::WORK);
            // A loop head or the entry: past PREEMPT_WORK instructions, a
            // preemption point for computations that make no SVC.
            if heads.contains(&block.start) || block.start == f.start {
                let over = e.b.ins().icmp_imm_s(IntCC::UnsignedGreaterThanOrEqual, work, gyroship_runtime::PREEMPT_WORK as i64);
                let (slow, go) = (e.b.create_block(), e.b.create_block());
                e.b.set_cold_block(slow);
                e.b.ins().brif(over, slow, &[], go, &[]);
                e.b.switch_to_block(slow);
                e.runtime(e.rt.yield_, block.start, gyroship_runtime::YIELD_PREEMPT);
                e.b.ins().jump(go, &[]);
                e.b.switch_to_block(go);
            }
            let end = block.start + block.len;
            let mut ended = false;
            for pc in (block.start..end).step_by(4) {
                let word = (input.word)(pc);
                let op = crate::lift::lift(word, pc);
                if (input.probe)(pc) {
                    let f = e.func(e.rt.probe);
                    let at = e.b.ins().iconst(I64, pc as i64);
                    e.b.ins().call(f, &[cpu, at]);
                }
                // After a call or an SVC, execution goes on in a block of its own,
                // so that the continuation is a resume point.
                let mut continuation = |e: &mut Emit<M>, ended: &mut bool| -> Block {
                    let cont = if pc + 4 < end {
                        e.b.create_block()
                    } else {
                        *ended = true;
                        e.target_block(end, index(end).map(|i| blocks[i]))
                    };
                    if pc + 4 < end || index(end).is_some() {
                        points.insert(pc + 4, cont);
                    }
                    cont
                };
                match op {
                    Op::Jump(t) if t == pc => {
                        ended = true;
                        e.runtime(e.rt.trap, pc, gyroship_runtime::TRAP_SELF_LOOP);
                        e.b.ins().return_(&[]);
                    }
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
                    Op::Call(_) | Op::CallIndirect(_) => {
                        let target = match op {
                            Op::CallIndirect(r) => Some(e.get(r, W::W64)),
                            _ => None,
                        };
                        let ret = e.b.ins().iconst(I64, (pc + 4) as i64);
                        e.set(Reg::X(30), W::W64, ret);
                        match (op, target) {
                            (Op::Call(t), _) => e.call(t),
                            (_, Some(t)) => {
                                let f = e.func(e.rt.call);
                                e.b.ins().call(f, &[cpu, t]);
                            }
                            _ => unreachable!(),
                        }
                        let cont = continuation(&mut e, &mut ended);
                        e.returned(pc + 4, cont, rescue);
                        if !ended {
                            e.b.switch_to_block(cont);
                        }
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
                                let mode = e.b.ins().iconst(I8, DISPATCH);
                                e.b.ins().jump(rescue, &[t.into(), mode.into()]);
                            }
                        }
                    }
                    Op::Return => {
                        ended = true;
                        let lr = e.get(Reg::X(30), W::W64);
                        e.b.ins().store(e.cpu_flags, lr, cpu, layout::PC);
                        e.b.ins().return_(&[]);
                    }
                    Op::Trap(imm) => {
                        ended = true;
                        e.runtime(e.rt.trap, pc, u32::from(imm));
                        e.b.ins().return_(&[]);
                    }
                    Op::Svc(_) | Op::Counter(_) | Op::Yield(_) => {
                        let (id, imm) = match op {
                            Op::Svc(imm) => (e.rt.svc, u32::from(imm)),
                            Op::Counter(r) => (e.rt.counter, reg_number(r)),
                            Op::Yield(kind) => (e.rt.yield_, u32::from(kind)),
                            _ => unreachable!(),
                        };
                        e.runtime(id, pc, imm);
                        if pc + 4 < end {
                            let cont = continuation(&mut e, &mut ended);
                            e.b.ins().jump(cont, &[]);
                            e.b.switch_to_block(cont);
                        } else if let Some(i) = index(end) {
                            points.insert(end, blocks[i]);
                        }
                    }
                    Op::Unhandled => e.runtime(e.rt.unhandled, pc, word),
                    ref op => e.data(op, pc, word),
                }
            }
            if !ended {
                e.goto(end, index(end).map(|i| blocks[i]));
            }
        }
        e.fill_pending();
        e.rescue(rescue, &points);
        let resume_points = points.len();
        e.b.seal_all_blocks();
        e.b.finalize(config);
        self.module.define_function(id, &mut self.ctx).map_err(|e| format!("{e:?}"))?;
        self.module.clear_context(&mut self.ctx);
        self.resume_points += resume_points;
        Ok(())
    }

    /// Resume points of the functions generated so far.
    pub fn resume_points(&self) -> usize {
        self.resume_points
    }

    /// The module, and the id of each guest function it declares.
    pub fn into_parts(self) -> (M, HashMap<u64, FuncId>) {
        (self.module, self.ids)
    }
}

struct Emit<'a, 'b, M: Module> {
    b: FunctionBuilder<'b>,
    cpu: Value,
    mem: Value,
    cpu_flags: MemFlagsData,
    heap_flags: MemFlagsData,
    module: &'a mut M,
    rt: &'a Runtime,
    ids: &'a mut HashMap<u64, FuncId>,
    sig: &'a Signature,
    refs: &'a mut HashMap<FuncId, FuncRef>,
    /// Tail-call blocks still to fill: (block, guest target).
    pending: Vec<(Block, u64)>,
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

impl<M: Module> Emit<'_, '_, M> {
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
        // The 39-bit mask keeps a wild address inside the reservation (D18).
        let masked = self.b.ins().band_imm_s(at, (gyroship_runtime::mem::SPACE - 1) as i64);
        let host = self.b.ins().iadd(self.mem, masked);
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

    /// An acquire load (LDAR and the like): one atomic load of 1 << size bytes,
    /// which Cranelift neither splits nor merges with another load; extended
    /// to `w` as [`Emit::load`] does.
    fn load_ordered(&mut self, host: Value, w: W, size: u8, signed: bool) -> Value {
        let narrow = [I8, I16, I32, I64][usize::from(size.min(3))];
        let v = self.b.ins().atomic_load(narrow, self.heap_flags, host);
        match (narrow == ty(w), signed) {
            (true, _) => v,
            (false, true) => self.b.ins().sextend(ty(w), v),
            (false, false) => self.b.ins().uextend(ty(w), v),
        }
    }

    /// The low 1 << size bytes of a 64-bit value, as the integer type of that width.
    fn narrow(&mut self, v: Value, size: u8) -> Value {
        if size < 3 { self.b.ins().ireduce([I8, I16, I32][usize::from(size)], v) } else { v }
    }

    /// Writes a value of 1 << size bytes to `dst`, zero-extended: X for 8 bytes,
    /// W (upper half cleared) otherwise.
    fn set_widened(&mut self, dst: Reg, v: Value, size: u8) {
        match size {
            3 => self.set(dst, W::W64, v),
            2 => self.set(dst, W::W32, v),
            _ => {
                let v = self.b.ins().uextend(I32, v);
                self.set(dst, W::W32, v);
            }
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

    /// A data operation, the instruction `word` at `pc`.
    fn data(&mut self, op: &Op, pc: u64, word: u32) {
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
            Op::Load { dst, w, size, signed, ordered, addr } => {
                let (_, host) = self.address(&addr);
                let v = if ordered { self.load_ordered(host, w, size, signed) } else { self.load(host, 0, w, size, signed) };
                self.set(dst, w, v);
            }
            Op::Store { src, size, ordered, addr } => {
                let v = self.get(src, W::W64);
                let (_, host) = self.address(&addr);
                if ordered {
                    // A store, then MFENCE: Cranelift's atomic_store on x64.
                    let v = self.narrow(v, size);
                    self.b.ins().atomic_store(self.heap_flags, v, host);
                } else {
                    self.store(v, host, 0, size);
                }
            }
            Op::Fence => {
                self.b.ins().fence();
            }
            // Host atomics are sequentially consistent: at least the order any Arm
            // variant (none, A, L, AL) asks for.
            Op::AtomicRmw { op, size, src, dst, addr } => {
                let s = self.get(src, W::W64);
                let s = self.narrow(s, size);
                let (_, host) = self.address(&addr);
                let (rmw, s) = match op {
                    Rmw::Add => (AtomicRmwOp::Add, s),
                    Rmw::Clr => (AtomicRmwOp::And, self.b.ins().bnot(s)),
                    Rmw::Eor => (AtomicRmwOp::Xor, s),
                    Rmw::Set => (AtomicRmwOp::Or, s),
                    Rmw::Smax => (AtomicRmwOp::Smax, s),
                    Rmw::Smin => (AtomicRmwOp::Smin, s),
                    Rmw::Umax => (AtomicRmwOp::Umax, s),
                    Rmw::Umin => (AtomicRmwOp::Umin, s),
                    Rmw::Swp => (AtomicRmwOp::Xchg, s),
                };
                let t = [I8, I16, I32, I64][usize::from(size.min(3))];
                let old = self.b.ins().atomic_rmw(t, self.heap_flags, rmw, host, s);
                self.set_widened(dst, old, size);
            }
            Op::AtomicCas { size, cmp, new, addr } => {
                let (c, n) = (self.get(cmp, W::W64), self.get(new, W::W64));
                let (c, n) = (self.narrow(c, size), self.narrow(n, size));
                let (_, host) = self.address(&addr);
                let old = self.b.ins().atomic_cas(self.heap_flags, host, c, n);
                self.set_widened(cmp, old, size);
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
            Op::Crc { dst, acc, data, size, castagnoli } => {
                let (a, d) = (self.get(acc, W::W32), self.get(data, W::W64));
                let kind = self.b.ins().iconst(I32, i64::from(u8::from(castagnoli) << 2 | size));
                let f = self.func(self.rt.crc32);
                let call = self.b.ins().call(f, &[a, d, kind]);
                let r = self.b.inst_results(call)[0];
                self.set(dst, W::W32, r);
            }
            Op::LoadV { dst, size, addr } => {
                let (_, host) = self.address(&addr);
                let v = self.b.ins().load(vtype(size), self.heap_flags, host, 0);
                self.vset(dst, v);
            }
            Op::StoreV { src, size, addr } => {
                let v = self.vget(src, size);
                let (_, host) = self.address(&addr);
                self.b.ins().store(self.heap_flags, v, host, 0);
            }
            Op::LoadPairV { a, b, size, addr } => {
                let (_, host) = self.address(&addr);
                let t = vtype(size);
                let v1 = self.b.ins().load(t, self.heap_flags, host, 0);
                let v2 = self.b.ins().load(t, self.heap_flags, host, 1 << size);
                self.vset(a, v1);
                self.vset(b, v2);
            }
            Op::StorePairV { a, b, size, addr } => {
                let (v1, v2) = (self.vget(a, size), self.vget(b, size));
                let (_, host) = self.address(&addr);
                self.b.ins().store(self.heap_flags, v1, host, 0);
                self.b.ins().store(self.heap_flags, v2, host, 1 << size);
            }
            Op::VConst { dst, value } => {
                let lo = self.b.ins().iconst(I64, value as u64 as i64);
                let hi = self.b.ins().iconst(I64, (value >> 64) as u64 as i64);
                let off = vr_offset(dst);
                self.b.ins().store(self.cpu_flags, lo, self.cpu, off);
                self.b.ins().store(self.cpu_flags, hi, self.cpu, off + 8);
            }
            Op::VMov { dst, src, size } => {
                let v = self.vget(src, size);
                self.vset(dst, v);
            }
            Op::ToV { dst, src, size } => {
                let v = self.get(src, if size == 3 { W::W64 } else { W::W32 });
                self.vset(dst, v);
            }
            Op::FromV { dst, src, size, lane } => {
                let off = vr_offset(src) + (i32::from(lane) << size);
                let v = self.b.ins().load(vtype(size), self.cpu_flags, self.cpu, off);
                let w = if size == 3 { W::W64 } else { W::W32 };
                let v = if size < 2 { self.b.ins().uextend(I32, v) } else { v };
                self.set(dst, w, v);
            }
            Op::VLogic { op, q, dst, a, b } => {
                let size = if q { 4 } else { 3 };
                let (x, y) = (self.vget(a, size), self.vget(b, size));
                let i = self.b.ins();
                let r = match op {
                    Alu::And => i.band(x, y),
                    Alu::Orr => i.bor(x, y),
                    Alu::Eor => i.bxor(x, y),
                    Alu::Bic => i.band_not(x, y),
                    _ => i.bor_not(x, y),
                };
                self.vset(dst, r);
            }
            Op::VSelect { kind, q, dst, a, b } => {
                let size = if q { 4 } else { 3 };
                let (d, x, y) = (self.vget(dst, size), self.vget(a, size), self.vget(b, size));
                // bitselect(c, t, f) takes t where c is 1.
                let r = match kind {
                    Select::Bsl => self.b.ins().bitselect(d, x, y),
                    Select::Bit => self.b.ins().bitselect(y, x, d),
                    Select::Bif => self.b.ins().bitselect(y, d, x),
                };
                self.vset(dst, r);
            }
            Op::FBin { op, f64, dst, a, b } => {
                let (x, y) = (self.fget(a, f64), self.fget(b, f64));
                let r = match op {
                    FBin::Add => self.b.ins().fadd(x, y),
                    FBin::Sub => self.b.ins().fsub(x, y),
                    FBin::Mul => self.b.ins().fmul(x, y),
                    FBin::Div => self.b.ins().fdiv(x, y),
                    FBin::Max => self.b.ins().fmax(x, y),
                    FBin::Min => self.b.ins().fmin(x, y),
                    FBin::MaxNm | FBin::MinNm => {
                        let m = if op == FBin::MaxNm { self.b.ins().fmax(x, y) } else { self.b.ins().fmin(x, y) };
                        let x_nan = self.b.ins().fcmp(FloatCC::Unordered, x, x);
                        let y_nan = self.b.ins().fcmp(FloatCC::Unordered, y, y);
                        let r = self.b.ins().select(y_nan, x, m);
                        self.b.ins().select(x_nan, y, r)
                    }
                    FBin::Nmul => {
                        let p = self.b.ins().fmul(x, y);
                        self.b.ins().fneg(p)
                    }
                    FBin::Abd => {
                        let d = self.b.ins().fsub(x, y);
                        self.b.ins().fabs(d)
                    }
                };
                if matches!(op, FBin::Add | FBin::Sub | FBin::Mul | FBin::Div) {
                    let nan = self.b.ins().fcmp(FloatCC::Unordered, r, r);
                    self.unless_nan(nan, pc, word, |e| e.vset(dst, r));
                } else {
                    self.vset(dst, r);
                }
            }
            Op::FUn { op, f64, dst, src } => {
                let x = self.fget(src, f64);
                let r = match op {
                    FUn::Neg => self.b.ins().fneg(x),
                    FUn::Abs => self.b.ins().fabs(x),
                    FUn::Sqrt => self.b.ins().sqrt(x),
                    FUn::Floor => self.b.ins().floor(x),
                    FUn::Ceil => self.b.ins().ceil(x),
                    FUn::Trunc => self.b.ins().trunc(x),
                    FUn::Nearest => self.b.ins().nearest(x),
                    FUn::Round => {
                        // |x| rounded half up, with the sign of x: exact, since |x| - floor(|x|) is.
                        let a = self.b.ins().fabs(x);
                        let fl = self.b.ins().floor(a);
                        let frac = self.b.ins().fsub(a, fl);
                        let (half, one) = if f64 {
                            (self.b.ins().f64const(0.5), self.b.ins().f64const(1.0))
                        } else {
                            (self.b.ins().f32const(0.5), self.b.ins().f32const(1.0))
                        };
                        let up = self.b.ins().fcmp(FloatCC::GreaterThanOrEqual, frac, half);
                        let bumped = self.b.ins().fadd(fl, one);
                        let r = self.b.ins().select(up, bumped, fl);
                        self.b.ins().fcopysign(r, x)
                    }
                };
                self.vset(dst, r);
            }
            Op::FMulAdd { f64, dst, a, b, acc, neg_acc, neg_product } => {
                let (x, y, z) = (self.fget(a, f64), self.fget(b, f64), self.fget(acc, f64));
                let x = if neg_product { self.b.ins().fneg(x) } else { x };
                let z = if neg_acc { self.b.ins().fneg(z) } else { z };
                let r = self.b.ins().fma(x, y, z);
                self.vset(dst, r);
            }
            Op::FCmp { f64, a, b } => {
                let x = self.fget(a, f64);
                let y = match b {
                    Some(b) => self.fget(b, f64),
                    None if f64 => self.b.ins().f64const(0.0),
                    None => self.b.ins().f32const(0.0),
                };
                let [n, z, c, v] = self.fcmp_flags(x, y);
                self.set_flags(n, z, c, v);
            }
            Op::FCondCmp { f64, a, b, nzcv, cond } => {
                let c = self.flags_cond(cond);
                let (x, y) = (self.fget(a, f64), self.fget(b, f64));
                let flags = self.fcmp_flags(x, y);
                let mut out = flags;
                for (k, f) in flags.into_iter().enumerate() {
                    let imm = self.b.ins().iconst(I8, i64::from((nzcv >> (3 - k)) & 1));
                    out[k] = self.b.ins().select(c, f, imm);
                }
                self.set_flags(out[0], out[1], out[2], out[3]);
            }
            Op::FCondSel { f64, dst, a, b, cond } => {
                let c = self.flags_cond(cond);
                let (x, y) = (self.fget(a, f64), self.fget(b, f64));
                let r = self.b.ins().select(c, x, y);
                self.vset(dst, r);
            }
            Op::FCvt { dst, src, to_f64 } => {
                let x = self.fget(src, !to_f64);
                let r = if to_f64 { self.b.ins().fpromote(F64, x) } else { self.b.ins().fdemote(F32, x) };
                self.vset(dst, r);
            }
            Op::IntToF { dst, src, w, f64, signed, fbits } => {
                let x = self.get(src, w);
                let t = if f64 { F64 } else { F32 };
                let mut r = if signed { self.b.ins().fcvt_from_sint(t, x) } else { self.b.ins().fcvt_from_uint(t, x) };
                // Fixed point: times 2^-fbits, exact (a normal power of two, the result at least 2^-64).
                if fbits > 0 {
                    let scale = if f64 { self.b.ins().f64const(2f64.powi(-i32::from(fbits))) } else { self.b.ins().f32const(2f32.powi(-i32::from(fbits))) };
                    r = self.b.ins().fmul(r, scale);
                }
                self.vset(dst, r);
            }
            Op::FToInt { dst, w, src, f64, signed } => {
                let x = self.fget(src, f64);
                let r = if signed { self.b.ins().fcvt_to_sint_sat(ty(w), x) } else { self.b.ins().fcvt_to_uint_sat(ty(w), x) };
                self.set(dst, w, r);
            }
            Op::LoadLane { dst, size, lane, q, addr } => {
                let (_, host) = self.address(&addr);
                let v = self.b.ins().load(vtype(size), self.heap_flags, host, 0);
                self.put_lanes(dst, size, lane, q, v);
            }
            Op::ToLane { dst, src, size, lane, q } => {
                let v = self.get(src, if size == 3 { W::W64 } else { W::W32 });
                let v = if size < 2 { self.b.ins().ireduce(vtype(size), v) } else { v };
                self.put_lanes(dst, size, lane, q, v);
            }
            Op::VShuffle { dst, a, b, mask } => {
                let x = self.vget(a, 4);
                // One source when b is a or unused; zeroed bytes repeat the lanes before them,
                // which keeps the common masks (pshufd and the like) recognizable.
                let one = a == b || mask.iter().all(|&m| m < 16 || m == ZERO);
                let y = if one { x } else { self.vget(b, 4) };
                let k = mask.iter().position(|&m| m == ZERO).unwrap_or(16).max(1);
                let mut idx = [0u8; 16];
                for i in 0..16 {
                    idx[i] = if mask[i] == ZERO { idx[i % k] } else if one { mask[i] % 16 } else { mask[i] };
                }
                let imm = self.b.func.dfg.immediates.push(ConstantData::from(idx.to_vec()));
                let mut r = self.b.ins().shuffle(x, y, imm);
                if mask.contains(&ZERO) {
                    let keep = self.vconst(I8X16, mask.map(|m| if m == ZERO { 0 } else { 0xFF }));
                    r = self.b.ins().band(r, keep);
                }
                self.b.ins().store(self.cpu_flags, r, self.cpu, vr_offset(dst));
            }
            Op::VFp { op, f64, q, dst, a, b, lane } => {
                let (t, eb) = if f64 { (F64X2, 8) } else { (F32X4, 4) };
                let (x, y) = if op == VFp::AddP {
                    let (even, odd) = self.pairs(a, b, eb, q);
                    (self.cast(t, even), self.cast(t, odd))
                } else {
                    let y = match lane {
                        Some(l) => {
                            let e = self.b.ins().load(t.lane_type(), self.cpu_flags, self.cpu, vr_offset(b) + i32::from(l) * eb as i32);
                            self.b.ins().splat(t, e)
                        }
                        None => self.vload(t, b),
                    };
                    (self.vload(t, a), y)
                };
                let r = match op {
                    VFp::Add | VFp::AddP => self.b.ins().fadd(x, y),
                    VFp::Sub => self.b.ins().fsub(x, y),
                    VFp::Mul => self.b.ins().fmul(x, y),
                    VFp::Mla | VFp::Mls => {
                        let acc = self.vload(t, dst);
                        let x = if op == VFp::Mls { self.b.ins().fneg(x) } else { x };
                        self.b.ins().fma(x, y, acc)
                    }
                };
                let nan = self.b.ins().fcmp(FloatCC::Unordered, r, r);
                let lanes = self.b.ins().vhigh_bits(I32, nan);
                let lanes = if q { lanes } else { self.b.ins().band_imm_s(lanes, 0b11) };
                self.unless_nan(lanes, pc, word, |e| e.vput(dst, r, q));
            }
            Op::VFUn { op, f64, q, dst, src } => {
                let x = self.vload(if f64 { F64X2 } else { F32X4 }, src);
                let r = if op == FUn::Neg { self.b.ins().fneg(x) } else { self.b.ins().fabs(x) };
                self.vput(dst, r, q);
            }
            Op::VFCmp { cmp, f64, q, dst, a, b } => {
                let t = if f64 { F64X2 } else { F32X4 };
                let x = self.vload(t, a);
                let y = match b {
                    Some(b) => self.vload(t, b),
                    None => self.vconst(t, [0; 16]),
                };
                let cc = match cmp {
                    VCmp::Eq => FloatCC::Equal,
                    VCmp::Ge => FloatCC::GreaterThanOrEqual,
                    VCmp::Gt => FloatCC::GreaterThan,
                    VCmp::Le => FloatCC::LessThanOrEqual,
                    VCmp::Lt => FloatCC::LessThan,
                };
                let r = self.b.ins().fcmp(cc, x, y);
                self.vput(dst, r, q);
            }
            Op::VInt { op, size, q, pairwise, dst, a, b } => {
                let t = lanes_type(size);
                let (x, y) = if pairwise {
                    let (even, odd) = self.pairs(a, b, 1 << size, q);
                    (self.cast(t, even), self.cast(t, odd))
                } else {
                    (self.vload(t, a), self.vload(t, b))
                };
                let i = self.b.ins();
                let r = match op {
                    VInt::Add => i.iadd(x, y),
                    VInt::Sub => i.isub(x, y),
                    VInt::UMin => i.umin(x, y),
                    VInt::UMax => i.umax(x, y),
                    VInt::SMin => i.smin(x, y),
                    VInt::SMax => i.smax(x, y),
                };
                self.vput(dst, r, q);
            }
            Op::VIntToF { f64, q, signed, dst, src } => {
                let (it, ft) = if f64 { (I64X2, F64X2) } else { (I32X4, F32X4) };
                let x = self.vload(it, src);
                let r = if signed { self.b.ins().fcvt_from_sint(ft, x) } else { self.b.ins().fcvt_from_uint(ft, x) };
                self.vput(dst, r, q);
            }
            Op::VIntUn { op, size, q, dst, src } => {
                let t = lanes_type(size);
                let x = self.vload(t, src);
                let r = match op {
                    VIntUn::Shl(k) => self.b.ins().ishl_imm_s(x, i64::from(k)),
                    VIntUn::CmpZero(cmp) => {
                        let zero = self.vconst(t, [0; 16]);
                        let cc = match cmp {
                            VCmp::Eq => IntCC::Equal,
                            VCmp::Ge => IntCC::SignedGreaterThanOrEqual,
                            VCmp::Gt => IntCC::SignedGreaterThan,
                            VCmp::Le => IntCC::SignedLessThanOrEqual,
                            VCmp::Lt => IntCC::SignedLessThan,
                        };
                        self.b.ins().icmp(cc, x, zero)
                    }
                    VIntUn::Cnt => self.b.ins().popcnt(x),
                };
                self.vput(dst, r, q);
            }
            Op::VReduce { op, size, q, dst, src } => {
                let x = self.vload(lanes_type(size), src);
                // The lanes left, of `eb` bytes, in the low `span` bytes of v. The long sums
                // widen first: the low 8 bytes, or the sum of both halves widened.
                let (mut v, eb, mut span) = match op {
                    Reduce::UAddL | Reduce::SAddL => {
                        let unsigned = op == Reduce::UAddL;
                        let low = if unsigned { self.b.ins().uwiden_low(x) } else { self.b.ins().swiden_low(x) };
                        let v = if q {
                            let high = if unsigned { self.b.ins().uwiden_high(x) } else { self.b.ins().swiden_high(x) };
                            self.b.ins().iadd(low, high)
                        } else {
                            low
                        };
                        (v, 2u32 << size, 16)
                    }
                    _ => (x, 1u32 << size, if q { 16 } else { 8 }),
                };
                let t = lanes_type(eb.trailing_zeros() as u8);
                // Halving: the upper half of the span folded onto the lower.
                while span > eb {
                    span /= 2;
                    let bytes = self.cast(I8X16, v);
                    let idx: Vec<u8> = (0..16).map(|i| ((i + span) % 16) as u8).collect();
                    let imm = self.b.func.dfg.immediates.push(ConstantData::from(idx));
                    let upper = self.b.ins().shuffle(bytes, bytes, imm);
                    let upper = self.cast(t, upper);
                    let i = self.b.ins();
                    v = match op {
                        Reduce::UMax => i.umax(v, upper),
                        Reduce::UMin => i.umin(v, upper),
                        Reduce::SMax => i.smax(v, upper),
                        Reduce::SMin => i.smin(v, upper),
                        _ => i.iadd(v, upper),
                    };
                }
                let r = self.b.ins().extractlane(v, 0);
                self.vset(dst, r);
            }
            _ => unreachable!("control flow is emitted by the block"),
        }
    }

    /// The low 1 << size bytes of a vector register, as an integer (or I8X16 for 16).
    fn vget(&mut self, n: u8, size: u8) -> Value {
        self.b.ins().load(vtype(size), self.cpu_flags, self.cpu, vr_offset(n))
    }

    fn fget(&mut self, n: u8, f64: bool) -> Value {
        self.b.ins().load(if f64 { F64 } else { F32 }, self.cpu_flags, self.cpu, vr_offset(n))
    }

    /// Writes a value to a vector register; one narrower than 16 bytes zeroes the rest.
    fn vset(&mut self, n: u8, v: Value) {
        let t = self.b.func.dfg.value_type(v);
        let v = if t.is_vector() {
            v
        } else {
            let lanes = match t {
                I8 => I8X16,
                I16 => I16X8,
                I32 => I32X4,
                F32 => F32X4,
                F64 => F64X2,
                _ => I64X2,
            };
            self.b.ins().scalar_to_vector(lanes, v)
        };
        self.b.ins().store(self.cpu_flags, v, self.cpu, vr_offset(n));
    }

    /// A whole vector register as a vector of type `t`.
    fn vload(&mut self, t: Type, n: u8) -> Value {
        self.b.ins().load(t, self.cpu_flags, self.cpu, vr_offset(n))
    }

    /// Writes a whole vector to a register, or (`q` false) its low 8 bytes, zeroing the rest.
    fn vput(&mut self, n: u8, v: Value, q: bool) {
        let v = if q {
            v
        } else {
            let t = self.b.func.dfg.value_type(v);
            let low = self.vconst(t, [0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0, 0, 0, 0, 0, 0, 0, 0]);
            self.b.ins().band(v, low)
        };
        self.b.ins().store(self.cpu_flags, v, self.cpu, vr_offset(n));
    }

    fn vconst(&mut self, t: Type, bytes: [u8; 16]) -> Value {
        let c = self.b.func.dfg.constants.insert(ConstantData::from(bytes.to_vec()));
        self.b.ins().vconst(t, c)
    }

    /// The same 16 bytes as a vector of type `t` (lanes in little-endian order).
    fn cast(&mut self, t: Type, v: Value) -> Value {
        if self.b.func.dfg.value_type(v) == t { v } else { self.b.ins().bitcast(t, MemFlagsData::new().with_endianness(Endianness::Little), v) }
    }

    /// The even and the odd lanes (of `eb` bytes) of the concatenation a:b,
    /// as two I8X16; with `q` false, a and b are 8 bytes each and only the low
    /// 8 bytes of the results count.
    fn pairs(&mut self, a: u8, b: u8, eb: u32, q: bool) -> (Value, Value) {
        let width = if q { 16 } else { 8 };
        let n = width / eb;
        let cat = |k: u32| if k < n { k * eb } else { 16 + (k - n) * eb };
        let x = self.vget(a, 4);
        let y = if a == b { x } else { self.vget(b, 4) };
        let mut half = |odd: u32| {
            let idx: Vec<u8> = (0..16).map(|i| (cat(2 * (i % width / eb) + odd) + i % eb) as u8).collect();
            let imm = self.b.func.dfg.immediates.push(ConstantData::from(idx));
            self.b.ins().shuffle(x, y, imm)
        };
        (half(0), half(1))
    }

    /// Writes `v` (1 << size bytes) into one lane of a vector register, or into every lane.
    fn put_lanes(&mut self, n: u8, size: u8, lane: Option<u8>, q: bool, v: Value) {
        match lane {
            Some(l) => {
                self.b.ins().store(self.cpu_flags, v, self.cpu, vr_offset(n) + (i32::from(l) << size));
            }
            None => {
                let s = self.b.ins().splat(lanes_type(size), v);
                self.vput(n, s, q);
            }
        }
    }

    /// Stores an FP result with `store` unless `nan` (nonzero: a live lane is NaN).
    /// A NaN result is the only one where x86 and Arm differ (which NaN, the default
    /// NaN's sign): those rare results take the interpreter, the reference, instead.
    fn unless_nan(&mut self, nan: Value, pc: u64, word: u32, store: impl FnOnce(&mut Self)) {
        let (slow, fast, done) = (self.b.create_block(), self.b.create_block(), self.b.create_block());
        self.b.set_cold_block(slow);
        self.b.ins().brif(nan, slow, &[], fast, &[]);
        self.b.switch_to_block(slow);
        self.runtime(self.rt.unhandled, pc, word);
        self.b.ins().jump(done, &[]);
        self.b.switch_to_block(fast);
        store(self);
        self.b.ins().jump(done, &[]);
        self.b.switch_to_block(done);
    }

    /// NZCV of an FP compare: equal 0110, less 1000, greater 0010, unordered 0011.
    fn fcmp_flags(&mut self, x: Value, y: Value) -> [Value; 4] {
        let n = self.b.ins().fcmp(FloatCC::LessThan, x, y);
        let z = self.b.ins().fcmp(FloatCC::Equal, x, y);
        let c = self.b.ins().fcmp(FloatCC::UnorderedOrGreaterThanOrEqual, x, y);
        let v = self.b.ins().fcmp(FloatCC::Unordered, x, y);
        [n, z, c, v]
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
                let zero = self.b.ins().iconst(I64, 0);
                self.b.ins().call(f, &[self.cpu, zero]);
            }
            Callee::Dispatch(t) => {
                let f = self.func(self.rt.call);
                let t = self.b.ins().iconst(I64, t as i64);
                self.b.ins().call(f, &[self.cpu, t]);
            }
        }
    }

    /// A tail call to `t`. Below the host stack limit, the call is handed to
    /// the runtime's driver instead (`cpu.pc = t`, return), so that chains of
    /// tail calls cannot overflow the host stack.
    fn tail_call(&mut self, t: u64) {
        let resolved = match (self.callee)(t) {
            Callee::Direct(t) | Callee::Dispatch(t) => t,
        };
        let target = self.b.ins().iconst(I64, resolved as i64);
        self.unless_deep(target);
        self.call(t);
        self.b.ins().return_(&[]);
    }

    /// Below the host stack limit, hands the jump to `target` to the driver
    /// (`cpu.pc = target`, return); otherwise goes on in a fresh block.
    fn unless_deep(&mut self, target: Value) {
        let sp = self.b.ins().get_stack_pointer(I64);
        let limit = self.b.ins().load(I64, self.cpu_flags, self.cpu, layout::STACK_LIMIT);
        let deep = self.b.ins().icmp(IntCC::UnsignedLessThan, sp, limit);
        let (handoff, go) = (self.b.create_block(), self.b.create_block());
        self.b.ins().brif(deep, handoff, &[], go, &[]);
        self.b.switch_to_block(handoff);
        self.b.ins().store(self.cpu_flags, target, self.cpu, layout::PC);
        self.b.ins().return_(&[]);
        self.b.switch_to_block(go);
    }

    /// After a call that should come back to `next`: goes on in `cont` when
    /// `cpu.pc` says so, else in the rescue block.
    fn returned(&mut self, next: u64, cont: Block, rescue: Block) {
        let pc = self.b.ins().load(I64, self.cpu_flags, self.cpu, layout::PC);
        let ok = self.b.ins().icmp_imm_s(IntCC::Equal, pc, next as i64);
        let mode = self.b.ins().iconst(I8, RETURN);
        self.b.ins().brif(ok, cont, &[], rescue, &[pc.into(), mode.into()]);
    }

    /// The rescue block: jumps to `target` if it is one of `points`; else
    /// returns, dispatches or fails, as its mode says.
    fn rescue(&mut self, rescue: Block, points: &BTreeMap<u64, Block>) {
        self.b.switch_to_block(rescue);
        let (target, mode) = (self.b.block_params(rescue)[0], self.b.block_params(rescue)[1]);
        let mut sw = Switch::new();
        for (&a, &blk) in points {
            sw.set_entry(u128::from(a), blk);
        }
        let (other, not_miss, dispatch, fail, ret) = (self.b.create_block(), self.b.create_block(), self.b.create_block(), self.b.create_block(), self.b.create_block());
        sw.emit(&mut self.b, target, other);
        self.b.switch_to_block(other);
        let is_miss = self.b.ins().icmp_imm_s(IntCC::Equal, mode, MISS);
        self.b.ins().brif(is_miss, fail, &[], not_miss, &[]);
        self.b.switch_to_block(not_miss);
        let is_dispatch = self.b.ins().icmp_imm_s(IntCC::Equal, mode, DISPATCH);
        self.b.ins().brif(is_dispatch, dispatch, &[], ret, &[]);
        for (blk, id) in [(fail, self.rt.miss), (dispatch, self.rt.call)] {
            self.b.switch_to_block(blk);
            if blk == dispatch {
                // An indirect tail call nests host frames like a direct one.
                self.unless_deep(target);
            }
            let f = self.func(id);
            self.b.ins().call(f, &[self.cpu, target]);
            self.b.ins().return_(&[]);
        }
        self.b.switch_to_block(ret);
        self.b.ins().return_(&[]);
    }

    /// Jumps to a block of the function, or tail-calls `t`.
    fn goto(&mut self, t: u64, block: Option<Block>) {
        match block {
            Some(blk) => {
                self.b.ins().jump(blk, &[]);
            }
            None => self.tail_call(t),
        }
    }

    /// A block that continues at `t`: the function's own, or a new one that
    /// tail-calls it. The new one is filled after the body (`fill_pending`):
    /// the builder must not leave the current block before it is terminated.
    fn target_block(&mut self, t: u64, block: Option<Block>) -> Block {
        if let Some(blk) = block {
            return blk;
        }
        let tail = self.b.create_block();
        self.pending.push((tail, t));
        tail
    }

    fn fill_pending(&mut self) {
        for (tail, t) in std::mem::take(&mut self.pending) {
            self.b.switch_to_block(tail);
            self.tail_call(t);
        }
    }
}

/// The register number the runtime writes (31 for XZR: nothing).
fn reg_number(r: Reg) -> u32 {
    match r {
        Reg::X(n) => u32::from(n),
        Reg::Zr | Reg::Sp => 31,
    }
}

fn vr_offset(n: u8) -> i32 {
    layout::VR + 16 * i32::from(n)
}

/// A vector of lanes of 1 << size bytes.
fn lanes_type(size: u8) -> Type {
    match size {
        0 => I8X16,
        1 => I16X8,
        2 => I32X4,
        _ => I64X2,
    }
}

/// The type that moves 1 << size bytes of a vector register.
fn vtype(size: u8) -> Type {
    match size {
        0 => I8,
        1 => I16,
        2 => I32,
        3 => I64,
        _ => I8X16,
    }
}
