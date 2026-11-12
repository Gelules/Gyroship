//! MME64, the macro method expander of the Turing and later 3D classes
//! (Ampere's 0xc797 included). A macro is a program in instruction RAM that
//! the pushbuffer calls with parameters; it reads them, computes, and emits
//! methods back into its class. NVN draws and clears only through macros.
//!
//! The instruction layout comes from NVIDIA's open-gpu-doc
//! (`MME-MacroMethodExpander/tu104`). The behaviour of each operation follows
//! the public descriptions of the hardware (open-gpu-doc, Mesa's nouveau
//! notes); this is our own implementation.

/// Data RAM words (DREAD/DWRITE).
pub const DATA_WORDS: usize = 0xc00;
/// Instruction RAM, in 32-bit words (three per group).
pub const CODE_WORDS: usize = 3 * 0x1000;
/// Macro start address slots.
pub const MACROS: usize = 0x80;
/// A macro running longer than this many groups is a runaway.
const STEP_LIMIT: u64 = 1 << 22;

// The class methods a macro emits to read memory (open-gpu-doc `clc797.h`).
const SET_MME_MEM_ADDRESS_A: u32 = 0x550;
const SET_MME_MEM_ADDRESS_B: u32 = 0x554;
const SET_MME_DATA_RAM_ADDRESS: u32 = 0x558;
pub const MME_DMA_READ: u32 = 0x55c;
pub const MME_DMA_READ_FIFOED: u32 = 0x560;
/// The most words one DMA read takes (Lichii's bound).
const DMA_LIMIT: u32 = 0x1000;

/// What a running macro reads from and writes to its class.
pub trait Host {
    /// The next parameter of the call (`LOAD0`, `LOAD1`), if any is left.
    fn param(&mut self) -> Option<u32>;
    /// The shadowed value of the method at byte address `method` (`STATE`).
    fn state(&mut self, method: u32) -> u32;
    /// A method the macro emits (byte address, data).
    fn method(&mut self, method: u32, data: u32);
    /// The class's current value of a method (what the macro last emitted to it).
    fn reg(&mut self, method: u32) -> u32;
    /// Words of GPU memory, for MME_DMA_READ(_FIFOED); left as they are where unmapped.
    fn read(&mut self, addr: u64, words: &mut [u32]);
}

/// Why a macro stopped early.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Fault {
    NoMacro(usize),
    OutOfCode(usize),
    BadRegister(u8),
    Runaway,
}

/// One instruction group, decoded.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct Group {
    end_next: bool,
    pred_mode: u8,
    pred: u8,
    alu: [Alu; 2],
    /// (method source, emit source) of each output.
    out: [(u8, u8); 2],
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct Alu {
    op: u8,
    dst: u8,
    src: [u8; 2],
    imm: u16,
}

// Registers: R0..R23, then the special sources.
const ZERO: u8 = 24;
const IMMED: u8 = 25;
const IMMED_PAIR: u8 = 26;
const IMMED32: u8 = 27;
const LOAD0: u8 = 28;
const LOAD1: u8 = 29;

// Output sources.
const OUT_NONE: u8 = 0;
const OUT_LOAD0: u8 = 3;
const OUT_LOAD1: u8 = 4;

/// Which of (alu 0, alu 1, out 0, out 1) a predicate mode makes conditional
/// (open-gpu-doc `pred.h`, in its order). A part runs when its letter is
/// not the predicate's state: `U` always, `T` when the predicate register is
/// zero, `F` when it is non-zero. This is the behaviour of Mesa's simulator,
/// which its tests check against the hardware; the game's macro 29 (disable
/// a pipeline stage only if the data RAM marks it enabled) needs it too.
const PRED_MODES: [&[u8; 4]; 16] = [
    b"UUUU", b"TTTT", b"FFFF", b"TTUU", b"FFUU", b"TFUU", b"TUUU", b"FUUU", b"UUTT", b"UUTF", b"UUTU", b"UUFT", b"UUFF", b"UUFU", b"UUUT", b"UUUF",
];

impl Group {
    /// Decodes the three words of a group, as sent: the high bits first.
    fn decode(w: [u32; 3]) -> Group {
        let bits = (u128::from(w[0]) << 64) | (u128::from(w[1]) << 32) | u128::from(w[2]);
        let f = |lo: u32, width: u32| ((bits >> lo) & ((1u128 << width) - 1)) as u32;
        let alu = |base: u32| Alu { op: f(base, 5) as u8, dst: f(base + 5, 5) as u8, src: [f(base + 10, 5) as u8, f(base + 15, 5) as u8], imm: f(base + 20, 16) as u16 };
        Group {
            end_next: f(0, 1) != 0,
            pred_mode: f(1, 4) as u8,
            pred: f(5, 5) as u8,
            alu: [alu(10), alu(46)],
            out: [(f(82, 3) as u8, f(85, 4) as u8), (f(89, 3) as u8, f(92, 4) as u8)],
        }
    }

    fn reads(&self, reg: u8, out: u8) -> bool {
        self.pred == reg || self.alu.iter().any(|a| a.src.contains(&reg)) || self.out.iter().any(|&(m, e)| m == out || e == out)
    }
}

// The interpreter's register file (`Run::r`): R0..R23, ZERO (never written),
// LOAD0 and LOAD1 at their own numbers, the slot that takes the writes to ZERO
// at IMMED's number (immediates are constants), and the two ALU results at the
// numbers no source may name.
const R_DISCARD: u8 = IMMED;
const R_ALU: [usize; 2] = [30, 31];

/// A value the interpreter reads: `r[idx] | konst` (`idx` ZERO for a constant).
#[derive(Debug, Clone, Copy)]
struct Src {
    idx: u8,
    konst: u32,
}

impl Src {
    const ZERO: Src = Src { idx: ZERO, konst: 0 };

    const fn reg(idx: u8) -> Src {
        Src { idx, konst: 0 }
    }

    const fn konst(konst: u32) -> Src {
        Src { idx: ZERO, konst }
    }
}

/// One ALU of a [`Fast`] group.
#[derive(Debug, Clone, Copy)]
struct FastAlu {
    op: u8,
    /// The register written: R0..R23, or R_DISCARD for ZERO and bad numbers.
    dst: u8,
    /// A bad destination number: the fault once the ALU has run.
    bad_dst: Option<u8>,
    imm: u16,
    x: Src,
    y: Src,
}

/// A group decoded for the interpreter: its sources resolved to the register
/// file or constants, its predicate mode to the parts that run.
#[derive(Debug, Clone, Copy)]
struct Fast {
    /// Parts that run (bit 0: ALU 0, 1: ALU 1, 2: output 0, 3: output 1) when the
    /// predicate reads zero, and when it does not (all without a predicate mode).
    if_zero: u8,
    if_set: u8,
    pred: Src,
    /// Parameters taken into LOAD0 (bit 0) and LOAD1 (bit 1) first.
    loads: u8,
    end_next: bool,
    /// Bad source numbers: the predicate's (read with a predicate mode only), then
    /// each ALU's first, checked in the order the unit reads them (`bad`: any).
    bad: bool,
    bad_pred: Option<u8>,
    bad_src: [Option<u8>; 2],
    alu: [FastAlu; 2],
    /// Each output's method and emit sources; whether they are not OUT_NONE in
    /// `outs`, bits 0 (method) and 1 (emit) for output 0, 2 and 3 for output 1.
    out: [(Src, Src); 2],
    outs: u8,
}

impl Fast {
    fn new(g: &Group) -> Fast {
        let imm = |k: usize| g.alu[k].imm as i16 as u32;
        // What ALU `i` (0 for the predicate) reads for a source number; ZERO and bad
        // numbers read 0 (a bad one faults before its value counts).
        let src = |i: usize, reg: u8| match reg {
            0..=23 | LOAD0 | LOAD1 => Src::reg(reg),
            IMMED => Src::konst(imm(i)),
            IMMED_PAIR => Src::konst(imm(i ^ 1)),
            IMMED32 => Src::konst((u32::from(g.alu[0].imm) << 16) | u32::from(g.alu[1].imm)),
            _ => Src::ZERO,
        };
        let out = |s: u8| {
            let (i0, i1) = (u32::from(g.alu[0].imm), u32::from(g.alu[1].imm));
            match s {
                1 => Src::reg(R_ALU[0] as u8),
                2 => Src::reg(R_ALU[1] as u8),
                OUT_LOAD0 => Src::reg(LOAD0),
                OUT_LOAD1 => Src::reg(LOAD1),
                5 => Src::konst(i0),
                6 => Src::konst(i1),
                8 => Src::konst(i0 >> 12),
                9 => Src::konst(i1 >> 12),
                10 => Src::konst((i0 << 16) | i1),
                _ => Src::ZERO,
            }
        };
        let bad = |reg: u8| (reg > LOAD1).then_some(reg);
        // A part runs when its letter is not the predicate's state; an output with
        // neither source would do nothing.
        let letters = PRED_MODES[usize::from(g.pred_mode)];
        let outs = g.out.iter().enumerate().fold(0, |m, (k, &(method, emit))| m | u8::from(method != OUT_NONE) << (2 * k) | u8::from(emit != OUT_NONE) << (2 * k + 1));
        let quiet = (0..2).fold(0, |m, k| m | u8::from(outs >> (2 * k) & 3 == 0) << (2 + k));
        let runs = |state: u8| letters.iter().enumerate().fold(0, |m, (k, &c)| m | u8::from(c != state) << k) & !quiet;
        let bad_pred = if g.pred_mode == 0 { None } else { bad(g.pred) };
        let bad_src = g.alu.map(|a| a.src.into_iter().find_map(bad));
        Fast {
            if_zero: runs(b'F'),
            if_set: runs(b'T'),
            pred: if g.pred_mode == 0 { Src::ZERO } else { src(0, g.pred) },
            loads: u8::from(g.reads(LOAD0, OUT_LOAD0)) | u8::from(g.reads(LOAD1, OUT_LOAD1)) << 1,
            end_next: g.end_next,
            bad: bad_pred.is_some() || bad_src.iter().any(Option::is_some),
            bad_pred,
            bad_src,
            alu: std::array::from_fn(|i| {
                let a = g.alu[i];
                FastAlu { op: a.op, dst: if a.dst < ZERO { a.dst } else { R_DISCARD }, bad_dst: (a.dst > ZERO).then_some(a.dst), imm: a.imm, x: src(i, a.src[0]), y: src(i, a.src[1]) }
            }),
            out: g.out.map(|(method, emit)| (out(method), out(emit))),
            outs,
        }
    }
}

/// The macro unit of one 3D class: its RAMs and the pointers that load them.
#[derive(Debug, Clone)]
pub struct Mme {
    pub code: Vec<u32>,
    pub starts: Vec<u32>,
    pub data: Vec<u32>,
    code_ptr: usize,
    start_ptr: usize,
    /// `code` decoded group by group for the interpreter: derived, kept in step
    /// by `load_code`, so that a call decodes nothing (rebuild it if `code` is
    /// ever set another way).
    groups: Vec<Fast>,
    /// What the calls did, for `--perf` (host statistics, not GPU state).
    pub stats: Stats,
    /// Each macro's last call: a hash of the parameters it took.
    last: Vec<Option<u64>>,
    /// A call running more groups than this is a runaway (STEP_LIMIT).
    limit: u64,
}

/// What macro calls did, for the speed work: the calls; those a memo could
/// replay (same macro and parameters as its last call, no register or memory
/// read); LOOPs with a zero count (run once here; Lichii skips them); taken
/// branches with immediate bit 14 (Lichii cancels a pending END with them), in
/// all and with an END pending.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct Stats {
    pub calls: u64,
    pub repeats: u64,
    pub zero_loops: u64,
    pub bit14: u64,
    pub bit14_end: u64,
    /// Groups the calls that ended ran.
    pub groups: u64,
}

impl std::ops::AddAssign for Stats {
    fn add_assign(&mut self, o: Stats) {
        (self.calls, self.repeats, self.zero_loops, self.bit14, self.bit14_end, self.groups) = (self.calls + o.calls, self.repeats + o.repeats, self.zero_loops + o.zero_loops, self.bit14 + o.bit14, self.bit14_end + o.bit14_end, self.groups + o.groups);
    }
}

impl Default for Mme {
    fn default() -> Mme {
        Mme { code: vec![0; CODE_WORDS], starts: vec![u32::MAX; MACROS], data: vec![0; DATA_WORDS], code_ptr: 0, start_ptr: 0, groups: vec![Fast::new(&Group::decode([0; 3])); CODE_WORDS / 3], stats: Stats::default(), last: vec![None; MACROS], limit: STEP_LIMIT }
    }
}

/// The state of one macro call.
struct Run<'a, H: Host> {
    host: &'a mut H,
    data: &'a mut [u32],
    stats: &'a mut Stats,
    /// The register file: R0..R23, ZERO, LOAD0/1, the ALU results (`R_ALU`).
    r: [u32; 32],
    carry: u32,
    method: u32,
    inc: u32,
    stop: bool,
    /// Words MME_DMA_READ_FIFOED queued after the call's parameters.
    fifo: std::collections::VecDeque<u32>,
    /// For [`Stats`]: a hash of the parameters taken, whether the call read
    /// registers or memory, and whether a taken branch had immediate bit 14.
    params: u64,
    reads: bool,
    bit14: bool,
}

impl Mme {
    /// LOAD_MME_INSTRUCTION_RAM_POINTER (0x114), in groups: the game loads a
    /// macro of n words at pointer p, then the next one at p + n/3.
    pub fn set_code_pointer(&mut self, groups: u32) {
        self.code_ptr = 3 * groups as usize;
    }

    /// LOAD_MME_INSTRUCTION_RAM (0x118); the pointer advances.
    pub fn load_code(&mut self, word: u32) {
        if let Some(w) = self.code.get_mut(self.code_ptr) {
            *w = word;
            let g = self.code_ptr / 3;
            if let Some(&[a, b, c]) = self.code.get(g * 3..g * 3 + 3) {
                self.groups[g] = Fast::new(&Group::decode([a, b, c]));
            }
        }
        self.code_ptr += 1;
    }

    /// LOAD_MME_START_ADDRESS_RAM_POINTER (0x11c).
    pub fn set_start_pointer(&mut self, index: u32) {
        self.start_ptr = index as usize;
    }

    /// LOAD_MME_START_ADDRESS_RAM (0x120): a start in groups (the 96-bit
    /// instruction words of the public notes; the game's starts are not
    /// multiples of 3 and its 110 macros follow each other in group units).
    pub fn load_start(&mut self, group: u32) {
        if let Some(s) = self.starts.get_mut(self.start_ptr) {
            *s = group;
        }
        self.start_ptr += 1;
    }

    /// Macro `index`'s groups from its start up to the next macro's start (or
    /// 64 groups), decoded, one line each (for study).
    pub fn listing(&self, index: usize) -> Vec<String> {
        let Some(&start) = self.starts.get(index).filter(|s| **s != u32::MAX) else { return Vec::new() };
        let next = self.starts.iter().filter(|&&s| s > start && s != u32::MAX).min().copied().unwrap_or(start + 64);
        (start..next.min(start + 256))
            .filter_map(|ip| {
                let at = ip as usize * 3;
                let w = self.code.get(at..at + 3)?;
                Some(format!("{ip:4}: {:?}", Group::decode([w[0], w[1], w[2]])))
            })
            .collect()
    }

    /// Runs macro `index` to its end.
    pub fn call<H: Host>(&mut self, index: usize, host: &mut H) -> Result<(), Fault> {
        let start = *self.starts.get(index).ok_or(Fault::NoMacro(index))? as usize;
        if start == u32::MAX as usize {
            return Err(Fault::NoMacro(index));
        }
        let Mme { groups, data, stats, last, limit, .. } = self;
        let mut run = Run { host, data, stats, r: [0; 32], carry: 0, method: 0, inc: 0, stop: false, fifo: Default::default(), params: 0, reads: false, bit14: false };
        let (mut ip, mut end_next) = (start, false);
        let mut lp = None::<(usize, usize, u32)>; // (start, end, remaining)
        for step in 1..=*limit {
            let g = groups.get(ip).ok_or(Fault::OutOfCode(ip))?;
            let mut next = ip + 1;
            if g.loads & 1 != 0 {
                run.r[usize::from(LOAD0)] = run.param();
            }
            if g.loads & 2 != 0 {
                run.r[usize::from(LOAD1)] = run.param();
            }
            if let Some(r) = g.bad_pred {
                return Err(Fault::BadRegister(r));
            }
            let mask = if run.get(g.pred) != 0 { g.if_set } else { g.if_zero };
            if g.bad
                && let Some(r) = (0..2).filter(|i| mask >> i & 1 != 0).find_map(|i| g.bad_src[i])
            {
                return Err(Fault::BadRegister(r));
            }
            // Both ALUs read the registers before either writes its destination (the
            // game's macros rely on it: macro code computes a value in one ALU while the
            // other still reads the register's old value); only the carry chains.
            let xy = g.alu.map(|a| (run.get(a.x), run.get(a.y)));
            for (i, a) in g.alu.iter().enumerate() {
                if mask >> i & 1 != 0 {
                    run.alu(a, i, xy[i], ip, &mut next, &mut lp)?;
                }
            }
            if std::mem::take(&mut run.bit14) {
                run.stats.bit14 += 1;
                run.stats.bit14_end += u64::from(end_next);
            }
            for (i, &out) in g.out.iter().enumerate() {
                if mask >> (2 + i) & 1 != 0 {
                    run.out(out, g.outs >> (2 * i));
                }
            }
            if end_next || run.stop {
                run.stats.calls += 1;
                run.stats.groups += step;
                let last = last.get_mut(index).and_then(|l| l.replace(run.params));
                run.stats.repeats += u64::from(!run.reads && last == Some(run.params));
                return Ok(());
            }
            end_next = g.end_next;
            if let Some((s, e, n)) = lp
                && ip == e
                && n > 0
            {
                lp = Some((s, e, n - 1));
                next = s + 1;
            }
            ip = next;
        }
        Err(Fault::Runaway)
    }
}

impl<H: Host> Run<'_, H> {
    fn get(&self, s: Src) -> u32 {
        self.r[usize::from(s.idx & 31)] | s.konst
    }

    fn alu(&mut self, a: &FastAlu, i: usize, (x, y): (u32, u32), ip: usize, next: &mut usize, lp: &mut Option<(usize, usize, u32)>) -> Result<(), Fault> {
        let wide = |p: u64| (p as u32, (p >> 32) as u32);
        let branch = |taken: bool| taken == (a.imm & 0x8000 != 0);
        let res = match a.op {
            0 => {
                let r = x.wrapping_add(y);
                self.carry = u32::from(r < x);
                r
            }
            1 => x.wrapping_add(y).wrapping_add(self.carry),
            2 => {
                let r = x.wrapping_sub(y);
                self.carry = u32::from(r > x);
                r
            }
            3 => x.wrapping_sub(y).wrapping_sub(self.carry),
            4 => {
                let (lo, hi) = wide((i64::from(x as i32) * i64::from(y as i32)) as u64);
                self.carry = hi;
                lo
            }
            5 => self.carry,
            6 => {
                let (lo, hi) = wide(u64::from(x) * u64::from(y));
                self.carry = hi;
                lo
            }
            // EXTENDED: a barrier for MME_DMA_READ_FIFOED, which nothing here needs.
            7 => 0,
            8 => x.leading_zeros(),
            9 => x << (y & 31),
            10 => x >> (y & 31),
            11 => ((x as i32) >> (y & 31)) as u32,
            12 => x & y,
            13 => !(x & y),
            14 => x | y,
            15 => x ^ y,
            // MERGE: `bits` bits of y from `src` replace those of x at `dst`.
            16 => {
                let (dst, bits, src) = (u32::from(a.imm >> 10) & 0x3f, u32::from(a.imm >> 5) & 0x1f, u32::from(a.imm) & 0x1f);
                let mask = ((1u64 << bits) - 1) << dst;
                ((u64::from(x) & !mask) | ((u64::from(y >> src) << dst) & mask)) as u32
            }
            17..=21 => {
                if compare(a.op - 17 + 25, x, y) {
                    u32::MAX
                } else {
                    0
                }
            }
            22 => {
                self.reads = true;
                self.host.state(u32::from(x.wrapping_add(y) as u16) * 4)
            }
            // LOOP: the next `imm - 1` groups run x times; none when x is 0. Mesa's
            // simulator runs them once, but NVK skips padding words with a LOOP of
            // parameter loads whose count is 0 for packed draws (D37).
            23 => {
                self.stats.zero_loops += u64::from(x == 0);
                match x {
                    0 => *next = ip + usize::from(a.imm),
                    _ => *lp = Some((ip, ip + usize::from(a.imm) - 1, x - 1)),
                }
                0
            }
            24 => {
                *next = ip + usize::from(a.imm & 0x7fff);
                0
            }
            25..=29 => {
                if branch(compare(a.op, x, y)) {
                    self.bit14 |= a.imm & 0x4000 != 0;
                    // A 13-bit signed offset; its most negative value means "stop".
                    let offset = i32::from(((a.imm as i16) << 3) >> 3);
                    if offset == -0x1000 {
                        self.stop = true;
                    } else {
                        *next = (ip as i64 + i64::from(offset)) as usize;
                    }
                }
                0
            }
            30 => self.data.get(x as usize).copied().unwrap_or(0),
            _ => {
                if let Some(d) = self.data.get_mut(x as usize) {
                    *d = y;
                }
                0
            }
        };
        self.r[R_ALU[i & 1]] = res;
        self.r[usize::from(a.dst & 31)] = res;
        a.bad_dst.map_or(Ok(()), |r| Err(Fault::BadRegister(r)))
    }

    /// The call's parameters, then what DMA reads queued; 0 once both run out.
    fn param(&mut self) -> u32 {
        match self.host.param() {
            Some(v) => {
                self.params = (self.params.rotate_left(5) ^ u64::from(v)).wrapping_mul(0x517c_c1b7_2722_0a95);
                v
            }
            None => self.fifo.pop_front().unwrap_or(0),
        }
    }

    /// MME_DMA_READ(_FIFOED): `count` words at SET_MME_MEM_ADDRESS_A/B go to
    /// data RAM from SET_MME_DATA_RAM_ADDRESS, or after the parameters.
    fn dma(&mut self, method: u32, count: u32) {
        self.reads = true;
        let addr = u64::from(self.host.reg(SET_MME_MEM_ADDRESS_A) & 0x1_ffff) << 32 | u64::from(self.host.reg(SET_MME_MEM_ADDRESS_B));
        let mut words = vec![0; count.min(DMA_LIMIT) as usize];
        self.host.read(addr, &mut words);
        if method == MME_DMA_READ_FIFOED {
            self.fifo.extend(words);
        } else {
            let at = self.host.reg(SET_MME_DATA_RAM_ADDRESS) as usize;
            for (d, w) in self.data.iter_mut().skip(at).zip(words) {
                *d = w;
            }
        }
    }

    /// An output: its method source (`outs` bit 0) sets the method and the
    /// increment, its emit source (bit 1) emits a value.
    fn out(&mut self, (method, emit): (Src, Src), outs: u8) {
        if outs & 1 != 0 {
            let d = self.get(method);
            self.method = (d & 0xfff) << 2;
            self.inc = (d >> 12) & 0xf;
        }
        if outs & 2 != 0 {
            let d = self.get(emit);
            self.host.method(self.method, d);
            if matches!(self.method, MME_DMA_READ | MME_DMA_READ_FIFOED) {
                self.dma(self.method, d);
            }
            self.method += self.inc * 4;
        }
    }
}

/// The comparisons, by branch opcode: BLT, BLTU, BLE, BLEU, BEQ.
fn compare(op: u8, x: u32, y: u32) -> bool {
    match op {
        25 => (x as i32) < (y as i32),
        26 => x < y,
        27 => (x as i32) <= (y as i32),
        28 => x <= y,
        _ => x == y,
    }
}

/// The interpreter as it was before [`Fast`]: each group decoded as it runs,
/// each register read checked. A test runs both on random programs.
#[cfg(test)]
mod reference {
    use super::*;

    pub(super) struct Old<'a, H: Host> {
        host: &'a mut H,
        regs: [u32; 24],
        load: [u32; 2],
        alu_res: [u32; 2],
        carry: u32,
        method: u32,
        inc: u32,
        stop: bool,
        fifo: std::collections::VecDeque<u32>,
        sources: [(u32, u32); 2],
        params: u64,
        reads: bool,
        bit14: bool,
    }

    pub(super) fn call<H: Host>(mme: &mut Mme, index: usize, host: &mut H) -> Result<(), Fault> {
        let start = *mme.starts.get(index).ok_or(Fault::NoMacro(index))? as usize;
        if start == u32::MAX as usize {
            return Err(Fault::NoMacro(index));
        }
        let mut run = Old { host, regs: [0; 24], load: [0; 2], alu_res: [0; 2], carry: 0, method: 0, inc: 0, stop: false, fifo: Default::default(), sources: [(0, 0); 2], params: 0, reads: false, bit14: false };
        let (mut ip, mut end_next) = (start, false);
        let mut lp = None::<(usize, usize, u32)>; // (start, end, remaining)
        for step in 1..=mme.limit {
            let w = (ip < CODE_WORDS / 3).then(|| &mme.code[ip * 3..ip * 3 + 3]).ok_or(Fault::OutOfCode(ip))?;
            let g = Group::decode([w[0], w[1], w[2]]);
            let mut next = ip + 1;
            if g.reads(LOAD0, OUT_LOAD0) {
                run.load[0] = run.param();
            }
            if g.reads(LOAD1, OUT_LOAD1) {
                run.load[1] = run.param();
            }
            let mask = match g.pred_mode {
                0 => [true; 4],
                m => {
                    let v = run.read(&g, 0, g.pred)? != 0;
                    PRED_MODES[usize::from(m)].map(|c| c != if v { b'T' } else { b'F' })
                }
            };
            run.sources = [(0, 0); 2];
            for (i, _) in mask[..2].iter().enumerate().filter(|m| *m.1) {
                let a = g.alu[i];
                run.sources[i] = (run.read(&g, i, a.src[0])?, run.read(&g, i, a.src[1])?);
            }
            for (i, _) in mask[..2].iter().enumerate().filter(|m| *m.1) {
                run.alu(mme, &g, i, ip, &mut next, &mut lp)?;
            }
            if std::mem::take(&mut run.bit14) {
                mme.stats.bit14 += 1;
                mme.stats.bit14_end += u64::from(end_next);
            }
            for i in 0..2 {
                if mask[2 + i] {
                    run.out(mme, &g, i);
                }
            }
            if end_next || run.stop {
                mme.stats.calls += 1;
                mme.stats.groups += step;
                let last = mme.last.get_mut(index).and_then(|l| l.replace(run.params));
                mme.stats.repeats += u64::from(!run.reads && last == Some(run.params));
                return Ok(());
            }
            end_next = g.end_next;
            if let Some((s, e, n)) = lp
                && ip == e
                && n > 0
            {
                lp = Some((s, e, n - 1));
                next = s + 1;
            }
            ip = next;
        }
        Err(Fault::Runaway)
    }

    impl<H: Host> Old<'_, H> {
        fn read(&self, g: &Group, i: usize, reg: u8) -> Result<u32, Fault> {
            Ok(match reg {
                0..=23 => self.regs[usize::from(reg)],
                ZERO => 0,
                // Immediates are signed for the ALUs.
                IMMED => g.alu[i & 1].imm as i16 as u32,
                IMMED_PAIR => g.alu[(i & 1) ^ 1].imm as i16 as u32,
                IMMED32 => (u32::from(g.alu[0].imm) << 16) | u32::from(g.alu[1].imm),
                LOAD0 => self.load[0],
                LOAD1 => self.load[1],
                r => return Err(Fault::BadRegister(r)),
            })
        }

        fn alu(&mut self, mme: &mut Mme, g: &Group, i: usize, ip: usize, next: &mut usize, lp: &mut Option<(usize, usize, u32)>) -> Result<(), Fault> {
            let (x, y) = self.sources[i];
            let a = g.alu[i];
            let wide = |p: u64| (p as u32, (p >> 32) as u32);
            let branch = |taken: bool| taken == (a.imm & 0x8000 != 0);
            let res = match a.op {
                0 => {
                    let r = x.wrapping_add(y);
                    self.carry = u32::from(r < x);
                    r
                }
                1 => x.wrapping_add(y).wrapping_add(self.carry),
                2 => {
                    let r = x.wrapping_sub(y);
                    self.carry = u32::from(r > x);
                    r
                }
                3 => x.wrapping_sub(y).wrapping_sub(self.carry),
                4 => {
                    let (lo, hi) = wide((i64::from(x as i32) * i64::from(y as i32)) as u64);
                    self.carry = hi;
                    lo
                }
                5 => self.carry,
                6 => {
                    let (lo, hi) = wide(u64::from(x) * u64::from(y));
                    self.carry = hi;
                    lo
                }
                // EXTENDED: a barrier for MME_DMA_READ_FIFOED, which nothing here needs.
                7 => 0,
                8 => x.leading_zeros(),
                9 => x << (y & 31),
                10 => x >> (y & 31),
                11 => ((x as i32) >> (y & 31)) as u32,
                12 => x & y,
                13 => !(x & y),
                14 => x | y,
                15 => x ^ y,
                // MERGE: `bits` bits of y from `src` replace those of x at `dst`.
                16 => {
                    let (dst, bits, src) = (u32::from(a.imm >> 10) & 0x3f, u32::from(a.imm >> 5) & 0x1f, u32::from(a.imm) & 0x1f);
                    let mask = ((1u64 << bits) - 1) << dst;
                    ((u64::from(x) & !mask) | ((u64::from(y >> src) << dst) & mask)) as u32
                }
                17..=21 => {
                    if compare(a.op - 17 + 25, x, y) {
                        u32::MAX
                    } else {
                        0
                    }
                }
                22 => {
                    self.reads = true;
                    self.host.state(u32::from(x.wrapping_add(y) as u16) * 4)
                }
                // LOOP: the next `imm - 1` groups run x times; none when x is 0. Mesa's
                // simulator runs them once, but NVK skips padding words with a LOOP of
                // parameter loads whose count is 0 for packed draws (D37).
                23 => {
                    mme.stats.zero_loops += u64::from(x == 0);
                    match x {
                        0 => *next = ip + usize::from(a.imm),
                        _ => *lp = Some((ip, ip + usize::from(a.imm) - 1, x - 1)),
                    }
                    0
                }
                24 => {
                    *next = ip + usize::from(a.imm & 0x7fff);
                    0
                }
                25..=29 => {
                    if branch(compare(a.op, x, y)) {
                        self.bit14 |= a.imm & 0x4000 != 0;
                        // A 13-bit signed offset; its most negative value means "stop".
                        let offset = i32::from(((a.imm as i16) << 3) >> 3);
                        if offset == -0x1000 {
                            self.stop = true;
                        } else {
                            *next = (ip as i64 + i64::from(offset)) as usize;
                        }
                    }
                    0
                }
                30 => mme.data.get(x as usize).copied().unwrap_or(0),
                _ => {
                    if let Some(d) = mme.data.get_mut(x as usize) {
                        *d = y;
                    }
                    0
                }
            };
            self.alu_res[i] = res;
            match a.dst {
                0..=23 => self.regs[usize::from(a.dst)] = res,
                ZERO => {}
                r => return Err(Fault::BadRegister(r)),
            }
            Ok(())
        }

        /// The call's parameters, then what DMA reads queued; 0 once both run out.
        fn param(&mut self) -> u32 {
            match self.host.param() {
                Some(v) => {
                    self.params = (self.params.rotate_left(5) ^ u64::from(v)).wrapping_mul(0x517c_c1b7_2722_0a95);
                    v
                }
                None => self.fifo.pop_front().unwrap_or(0),
            }
        }

        /// MME_DMA_READ(_FIFOED): `count` words at SET_MME_MEM_ADDRESS_A/B go to
        /// data RAM from SET_MME_DATA_RAM_ADDRESS, or after the parameters.
        fn dma(&mut self, mme: &mut Mme, method: u32, count: u32) {
            self.reads = true;
            let addr = u64::from(self.host.reg(SET_MME_MEM_ADDRESS_A) & 0x1_ffff) << 32 | u64::from(self.host.reg(SET_MME_MEM_ADDRESS_B));
            let mut words = vec![0; count.min(DMA_LIMIT) as usize];
            self.host.read(addr, &mut words);
            if method == MME_DMA_READ_FIFOED {
                self.fifo.extend(words);
            } else {
                let at = self.host.reg(SET_MME_DATA_RAM_ADDRESS) as usize;
                for (d, w) in mme.data.iter_mut().skip(at).zip(words) {
                    *d = w;
                }
            }
        }

        fn out(&mut self, mme: &mut Mme, g: &Group, i: usize) {
            let (method, emit) = g.out[i];
            let value = |src: u8| {
                let (i0, i1) = (u32::from(g.alu[0].imm), u32::from(g.alu[1].imm));
                match src {
                    1 => self.alu_res[0],
                    2 => self.alu_res[1],
                    OUT_LOAD0 => self.load[0],
                    OUT_LOAD1 => self.load[1],
                    5 => i0,
                    6 => i1,
                    8 => i0 >> 12,
                    9 => i1 >> 12,
                    10 => (i0 << 16) | i1,
                    _ => 0,
                }
            };
            if method != OUT_NONE {
                let d = value(method);
                self.method = (d & 0xfff) << 2;
                self.inc = (d >> 12) & 0xf;
            }
            if emit != OUT_NONE {
                let d = value(emit);
                self.host.method(self.method, d);
                if matches!(self.method, MME_DMA_READ | MME_DMA_READ_FIFOED) {
                    self.dma(mme, self.method, d);
                }
                self.method += self.inc * 4;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[derive(Default)]
    struct Fake {
        params: Vec<u32>,
        state: Vec<(u32, u32)>,
        emitted: Vec<(u32, u32)>,
    }

    impl Host for Fake {
        fn param(&mut self) -> Option<u32> {
            (!self.params.is_empty()).then(|| self.params.remove(0))
        }
        fn state(&mut self, method: u32) -> u32 {
            self.state.iter().find(|s| s.0 == method).map_or(0, |s| s.1)
        }
        fn method(&mut self, method: u32, data: u32) {
            self.emitted.push((method, data));
        }
        fn reg(&mut self, method: u32) -> u32 {
            self.emitted.iter().rev().find(|e| e.0 == method).map_or(0, |e| e.1)
        }
        /// Memory holds its own word address, plus 0x100 per 4 GiB.
        fn read(&mut self, addr: u64, words: &mut [u32]) {
            for (i, w) in words.iter_mut().enumerate() {
                *w = (addr / 4) as u32 + i as u32 + (addr >> 32) as u32 * 0x100;
            }
        }
    }

    /// Encodes a group from its fields (the inverse of `Group::decode`).
    fn enc(g: &Group) -> [u32; 3] {
        let mut b = u128::from(g.end_next) | u128::from(g.pred_mode) << 1 | u128::from(g.pred) << 5;
        for (base, a) in [(10, g.alu[0]), (46, g.alu[1])] {
            b |= u128::from(a.op) << base | u128::from(a.dst) << (base + 5) | u128::from(a.src[0]) << (base + 10) | u128::from(a.src[1]) << (base + 15) | u128::from(a.imm) << (base + 20);
        }
        b |= u128::from(g.out[0].0) << 82 | u128::from(g.out[0].1) << 85 | u128::from(g.out[1].0) << 89 | u128::from(g.out[1].1) << 92;
        [(b >> 64) as u32, (b >> 32) as u32, b as u32]
    }

    const NOP: Alu = Alu { op: 0, dst: ZERO, src: [ZERO, ZERO], imm: 0 };

    fn group(alu0: Alu, alu1: Alu, out: [(u8, u8); 2]) -> Group {
        Group { end_next: false, pred_mode: 0, pred: ZERO, alu: [alu0, alu1], out }
    }

    fn load(mme: &mut Mme, slot: u32, groups: &[Group]) {
        let start = (mme.code_ptr / 3) as u32;
        for g in groups {
            for w in enc(g) {
                mme.load_code(w);
            }
        }
        mme.set_start_pointer(slot);
        mme.load_start(start);
    }

    fn run(groups: &[Group], params: &[u32]) -> Fake {
        let mut mme = Mme::default();
        load(&mut mme, 0, groups);
        let mut host = Fake { params: params.to_vec(), ..Fake::default() };
        mme.call(0, &mut host).expect("the macro runs");
        host
    }

    fn alu(op: u8, dst: u8, x: u8, y: u8, imm: u16) -> Alu {
        Alu { op, dst, src: [x, y], imm }
    }

    fn last(mut g: Group) -> Group {
        g.end_next = true;
        g
    }

    #[test]
    fn decoding_inverts_encoding() {
        let g = Group { end_next: true, pred_mode: 9, pred: 3, alu: [alu(16, 5, 6, IMMED, 0xabcd), alu(29, ZERO, 7, 8, 0x8003)], out: [(1, 10), (6, 4)] };
        assert_eq!(Group::decode(enc(&g)), g);
    }

    #[test]
    fn a_macro_emits_its_parameters_with_an_increment() {
        // Method 0x1000 (index 0x400), increment 1, then two loads emitted.
        let set = group(NOP, NOP, [(5, 3), (OUT_NONE, 4)]);
        let mut g = set;
        g.alu[0].imm = (1 << 12) | 0x400;
        let end = last(group(NOP, NOP, [(OUT_NONE, OUT_NONE); 2]));
        let host = run(&[g, end, group(NOP, NOP, [(OUT_NONE, OUT_NONE); 2])], &[7, 9]);
        assert_eq!(host.emitted, vec![(0x1000, 7), (0x1004, 9)]);
    }

    #[test]
    fn the_group_after_end_next_runs_then_the_macro_stops() {
        // R0 = 5 (end_next), R0 = R0 + 1 emitted, then a group that must not run.
        let a = last(group(alu(0, 0, ZERO, IMMED, 5), NOP, [(OUT_NONE, OUT_NONE); 2]));
        let mut b = group(alu(0, 0, 0, IMMED, 1), NOP, [(5, 1), (OUT_NONE, OUT_NONE)]);
        b.alu[1].imm = 0;
        let mut c = group(NOP, NOP, [(5, 5), (OUT_NONE, OUT_NONE)]);
        c.alu[0].imm = 0x10;
        let host = run(&[a, b, c], &[]);
        // Method index 1 (byte 4) comes from alu 0's immediate in b: data = alu result 6.
        assert_eq!(host.emitted, vec![(4, 6)]);
    }

    #[test]
    fn a_loop_repeats_its_body() {
        // LOOP param (3) times over one group that emits R1 += 2.
        let mut set = group(NOP, NOP, [(5, OUT_NONE), (OUT_NONE, OUT_NONE)]);
        set.alu[0].imm = 0x200;
        let lp = group(alu(0, 0, LOAD0, ZERO, 0), alu(23, ZERO, LOAD0, ZERO, 2), [(OUT_NONE, OUT_NONE); 2]);
        let body = group(alu(0, 1, 1, IMMED, 2), NOP, [(OUT_NONE, 1), (OUT_NONE, OUT_NONE)]);
        let end = last(group(NOP, NOP, [(OUT_NONE, OUT_NONE); 2]));
        let host = run(&[set, lp, body, end, group(NOP, NOP, [(OUT_NONE, OUT_NONE); 2])], &[3]);
        assert_eq!(host.emitted, vec![(0x800, 2), (0x800, 4), (0x800, 6)]);
    }

    #[test]
    fn a_zero_count_loop_skips_its_body() {
        // LOOP param (0) times over a group that takes and emits a parameter, then a
        // group that takes and emits the next one: it is the second parameter, the
        // body took none (as NVK's padding loops need on hardware).
        let mut set = group(NOP, NOP, [(5, OUT_NONE), (OUT_NONE, OUT_NONE)]);
        set.alu[0].imm = 0x200;
        let lp = group(alu(23, ZERO, LOAD0, ZERO, 2), NOP, [(OUT_NONE, OUT_NONE); 2]);
        let take = group(alu(0, 1, LOAD0, ZERO, 0), NOP, [(OUT_NONE, 1), (OUT_NONE, OUT_NONE)]);
        let host = run(&[set, lp, take, last(take), group(NOP, NOP, [(OUT_NONE, OUT_NONE); 2])], &[0, 7]);
        assert_eq!(host.emitted, vec![(0x800, 7)]);
    }

    #[test]
    fn a_taken_branch_skips_and_stop_ends() {
        // BEQ param, 0 (expect true) +2 skips the emit; then BEQ zero, zero to "stop".
        let set = {
            let mut g = group(NOP, NOP, [(5, OUT_NONE), (OUT_NONE, OUT_NONE)]);
            g.alu[0].imm = 0x10;
            g
        };
        let beq = group(alu(29, ZERO, LOAD0, ZERO, 0x8000 | 2), NOP, [(OUT_NONE, OUT_NONE); 2]);
        let emit = group(NOP, NOP, [(OUT_NONE, 6), (OUT_NONE, OUT_NONE)]);
        let stop = group(alu(29, ZERO, ZERO, ZERO, 0x8000 | 0x1000), NOP, [(OUT_NONE, OUT_NONE); 2]);
        let never = group(NOP, NOP, [(OUT_NONE, 5), (OUT_NONE, OUT_NONE)]);
        assert_eq!(run(&[set, beq, emit, stop, never], &[0]).emitted, vec![]);
        let mut e = emit;
        e.alu[1].imm = 42;
        assert_eq!(run(&[set, beq, e, stop, never], &[1]).emitted, vec![(0x40, 42)]);
    }

    #[test]
    fn predication_masks_parts_and_state_reads_the_shadow() {
        // R0 = STATE(0x400 >> 2 + 0) ; then with pred R0 (UUTF): out 0 runs if R0 == 0, out 1 if not.
        let mut set = group(alu(22, 0, IMMED, ZERO, 0x100), NOP, [(5, OUT_NONE), (OUT_NONE, OUT_NONE)]);
        set.alu[0].imm = 0x100;
        let mut p = group(NOP, NOP, [(OUT_NONE, 5), (OUT_NONE, 6)]);
        p.pred_mode = 9;
        p.pred = 0;
        p.alu[0].imm = 1;
        p.alu[1].imm = 2;
        let end = last(group(NOP, NOP, [(OUT_NONE, OUT_NONE); 2]));
        let mut mme = Mme::default();
        load(&mut mme, 3, &[set, p, end, group(NOP, NOP, [(OUT_NONE, OUT_NONE); 2])]);
        let mut host = Fake { state: vec![(0x400, 77)], ..Fake::default() };
        mme.call(3, &mut host).unwrap();
        assert_eq!(host.emitted, vec![(0x400, 2)]);
        let mut host = Fake::default();
        mme.call(3, &mut host).unwrap();
        assert_eq!(host.emitted, vec![(0x400, 1)]);
    }

    #[test]
    fn dma_reads_fill_data_ram_and_queue_after_the_parameters() {
        let emit = |method: Option<u32>, data: u16| {
            let mut g = group(NOP, NOP, [(if method.is_some() { 5 } else { OUT_NONE }, 6), (OUT_NONE, OUT_NONE)]);
            g.alu[0].imm = method.unwrap_or(0) as u16;
            g.alu[1].imm = data;
            g
        };
        let none = [(OUT_NONE, OUT_NONE); 2];
        // Address 0x1_0000_0040, data RAM index 2, read 3 words; then a FIFOED read of 2.
        let setup = [emit(Some((1 << 12) | (SET_MME_MEM_ADDRESS_A / 4)), 1), emit(None, 0x40), emit(None, 2), emit(None, 3), emit(None, 2)];
        // R0 = DREAD [3], emitted to 0x1000; then three loads.
        let mut dread = group(alu(30, 0, IMMED, ZERO, 3), NOP, none);
        dread.alu[0].imm = 3;
        let mut set = group(NOP, alu(0, ZERO, 0, ZERO, 0), [(5, OUT_NONE), (OUT_NONE, 2)]);
        set.alu[0].imm = 0x400;
        let load = group(NOP, NOP, [(OUT_NONE, OUT_LOAD0), (OUT_NONE, OUT_NONE)]);
        let host = run(&[&setup[..], &[dread, set, load, last(load), load]].concat(), &[7]);
        let word = |i: u32| 0x4000_0110 + i;
        let expected = vec![(0x550, 1), (0x554, 0x40), (0x558, 2), (MME_DMA_READ, 3), (MME_DMA_READ_FIFOED, 2), (0x1000, word(1)), (0x1000, 7), (0x1000, word(0)), (0x1000, word(1))];
        assert_eq!(host.emitted, expected);
    }

    #[test]
    fn both_alus_read_registers_before_either_writes() {
        // R0 = 5; then in one group R0 = R0 + 1 and R1 = R0 + 0 (the old R0): emit R1, R0.
        let set = group(alu(0, 0, ZERO, IMMED, 5), NOP, [(OUT_NONE, OUT_NONE); 2]);
        let mut both = group(alu(0, 0, 0, IMMED, 1), alu(0, 1, 0, ZERO, 0), [(5, 2), (OUT_NONE, 1)]);
        both.alu[0].imm = 1;
        let end = last(group(NOP, NOP, [(OUT_NONE, OUT_NONE); 2]));
        let host = run(&[set, both, end, group(NOP, NOP, [(OUT_NONE, OUT_NONE); 2])], &[]);
        // Method index 1 (byte 4) from alu 0's immediate.
        assert_eq!(host.emitted, vec![(4, 5), (4, 6)]);
    }

    #[test]
    fn data_ram_merge_and_carry() {
        // DWRITE [5] = param; R0 = DREAD [5]; R1 = MERGE(0, R0, 8 bits from 4 to 16);
        // ADD/ADDC of 0xffffffff + 1 carries into the high word.
        let w = group(alu(31, ZERO, IMMED, LOAD0, 5), NOP, [(OUT_NONE, OUT_NONE); 2]);
        let r = group(alu(30, 0, IMMED, ZERO, 5), NOP, [(OUT_NONE, OUT_NONE); 2]);
        let m = group(alu(16, 1, ZERO, 0, (16 << 10) | (8 << 5) | 4), NOP, [(OUT_NONE, 1), (OUT_NONE, OUT_NONE)]);
        let c = group(alu(0, 2, LOAD1, IMMED, 1), alu(1, 3, ZERO, ZERO, 0), [(OUT_NONE, 1), (OUT_NONE, 2)]);
        let end = last(group(NOP, NOP, [(OUT_NONE, OUT_NONE); 2]));
        let host = run(&[w, r, m, c, end, group(NOP, NOP, [(OUT_NONE, OUT_NONE); 2])], &[0xabcd, 0xffff_ffff]);
        assert_eq!(host.emitted, vec![(0, 0xbc_0000), (0, 0), (0, 1)]);
    }

    /// A host answering from fixed functions, logging each call in order:
    /// (0 param, 1 state, 2 method, 3 reg, 4 read; then two values).
    #[derive(Default)]
    struct Logged {
        params: std::collections::VecDeque<u32>,
        log: Vec<(u8, u32, u32)>,
    }

    impl Host for Logged {
        fn param(&mut self) -> Option<u32> {
            let p = self.params.pop_front();
            self.log.push((0, p.unwrap_or(0), u32::from(p.is_some())));
            p
        }
        fn state(&mut self, method: u32) -> u32 {
            self.log.push((1, method, 0));
            method.wrapping_mul(0x9e37_79b9) >> 7
        }
        fn method(&mut self, method: u32, data: u32) {
            self.log.push((2, method, data));
        }
        fn reg(&mut self, method: u32) -> u32 {
            self.log.push((3, method, 0));
            self.log.iter().rev().find(|e| e.0 == 2 && e.1 == method).map_or(0, |e| e.2)
        }
        fn read(&mut self, addr: u64, words: &mut [u32]) {
            self.log.push((4, addr as u32, words.len() as u32));
            for (i, w) in words.iter_mut().enumerate() {
                *w = (addr as u32).wrapping_add(7 * i as u32);
            }
        }
    }

    /// xorshift64, for random programs.
    struct Rng(u64);

    impl Rng {
        fn next(&mut self) -> u64 {
            self.0 ^= self.0 << 13;
            self.0 ^= self.0 >> 7;
            self.0 ^= self.0 << 17;
            self.0
        }

        /// A register number below `valid`, or one in 7 times any of the 32.
        fn reg(&mut self, valid: u64) -> u8 {
            let v = self.next();
            (if v.is_multiple_of(7) { (v >> 3) % 32 } else { (v >> 3) % valid }) as u8
        }
    }

    #[test]
    fn the_interpreter_matches_the_reference_on_random_programs() {
        let mut rng = Rng(0x9e37_79b9_7f4a_7c15);
        let mut faults = 0;
        for case in 0..5000 {
            // 1 to 12 random groups at a random start: mostly valid register numbers,
            // some predicates and ENDs, now and then a method aimed at the DMA reads.
            let mut mme = Mme { limit: 3000, ..Mme::default() };
            let at = 1 + rng.next() % 64;
            mme.set_code_pointer(at as u32);
            for _ in 0..1 + rng.next() % 12 {
                let mut alu = [NOP; 2];
                for a in &mut alu {
                    *a = Alu { op: (rng.next() % 32) as u8, dst: rng.reg(25), src: [rng.reg(30), rng.reg(30)], imm: rng.next() as u16 };
                }
                let pred_mode = if rng.next().is_multiple_of(2) { 0 } else { (rng.next() % 16) as u8 };
                let out = [0; 2].map(|_| ((rng.next() % 8) as u8, (rng.next() % 16) as u8));
                let mut g = Group { end_next: rng.next().is_multiple_of(6), pred_mode, pred: rng.reg(30), alu, out };
                if rng.next().is_multiple_of(12) {
                    let dma = if rng.next().is_multiple_of(2) { MME_DMA_READ } else { MME_DMA_READ_FIFOED };
                    g.out[0].0 = 5;
                    g.alu[0].imm = (dma / 4) as u16 | ((rng.next() % 2) << 12) as u16;
                }
                for w in enc(&g) {
                    mme.load_code(w);
                }
            }
            mme.set_start_pointer(0);
            mme.load_start(at as u32);
            for _ in 0..8 {
                mme.data[(rng.next() % 16) as usize] = rng.next() as u32;
            }
            let params: std::collections::VecDeque<u32> = (0..rng.next() % 8).map(|_| (rng.next() % 64) as u32).collect();
            let (mut a, mut b) = (mme.clone(), mme);
            let (mut ha, mut hb) = (Logged { params: params.clone(), ..Logged::default() }, Logged { params, ..Logged::default() });
            let (ra, rb) = (a.call(0, &mut ha), reference::call(&mut b, 0, &mut hb));
            assert_eq!(ra, rb, "case {case}");
            assert_eq!(ha.log, hb.log, "case {case}");
            assert_eq!((a.data, a.stats), (b.data, b.stats), "case {case}");
            faults += usize::from(ra.is_err());
        }
        // Both endings, and every fault kind's path, are well covered.
        assert!((500..4500).contains(&faults), "{faults} faults");
    }

    #[test]
    fn an_endless_macro_is_a_runaway() {
        let spin = group(alu(24, ZERO, ZERO, ZERO, 0x8000), NOP, [(OUT_NONE, OUT_NONE); 2]);
        let mut mme = Mme::default();
        load(&mut mme, 0, &[spin]);
        assert_eq!(mme.call(0, &mut Fake::default()), Err(Fault::Runaway));
        assert_eq!(mme.call(1, &mut Fake::default()), Err(Fault::NoMacro(1)));
    }
}
