//! The GPU's command processor: it decodes a channel's pushbuffer segments
//! (NVIDIA's public method header format), keeps which class each
//! subchannel holds and the last value of every method, runs the 3D class's
//! macros (MME64), and reports what the CPU side waits for: semaphore
//! releases, from the host (SEMAPHOREA..D, SEM_ADDR..SEM_EXECUTE), 3D and
//! compute (SET_REPORT_SEMAPHORE_A..D) and the copy engine (LAUNCH_DMA).
//! Addresses are GPU addresses; the caller translates them. The 3D class's
//! clears and draws come out with the state they use, and the memory its
//! method stream writes (constant-buffer loads, inline uploads) comes out
//! in stream order.
//!
//! The copy engine's pitch-to-pitch copies come out as effects, before the
//! semaphore their LAUNCH_DMA may release.
//!
//! ponytail: block-linear and remapped copies, and the 2D class, are
//! reported, not executed.

use crate::draw::Mem;
use crate::mme::{self, Mme};
use std::collections::VecDeque;
use crate::state::raster::{self, RenderState};
use crate::state::resources::{self, Bindings, MemWrite, Pool, Resources, SamplerBinding, Unsupported};
use crate::state::vertex::{self, Attribute, IndexBuffer, Stream};

/// A semaphore release: `value` (4 or 8 bytes) at `addr`; with `timestamp`,
/// a u64 GPU time at `addr + 8` (and zero in between for a 4-byte value).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Release {
    /// The method that released it (its byte offset).
    pub method: u32,
    pub addr: u64,
    pub value: u64,
    pub width: u8,
    pub timestamp: bool,
}

/// A semaphore the channel waits for before going on (an acquire of the
/// host class: SEMAPHORED or SEM_EXECUTE, clc76f.h).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Acquire {
    pub addr: u64,
    pub value: u64,
    /// A 64-bit semaphore (SEM_EXECUTE_PAYLOAD_SIZE), else 32-bit.
    pub wide: bool,
    pub op: AcquireOp,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AcquireOp {
    /// Memory equals the value.
    Equal,
    /// Memory is at or past the value, with wrap-around (ACQ_GEQ, ACQ_CIRC_GEQ).
    CircGeq,
    /// Memory is at least the value, unsigned (ACQ_STRICT_GEQ).
    StrictGeq,
    /// Memory and the value share a set bit (ACQ_AND).
    And,
}

impl Acquire {
    /// Whether the semaphore's memory value `m` lets the channel go on.
    pub fn met(&self, m: u64) -> bool {
        let (m, v) = if self.wide { (m, self.value) } else { (m & 0xffff_ffff, self.value & 0xffff_ffff) };
        match self.op {
            AcquireOp::Equal => m == v,
            AcquireOp::CircGeq if self.wide => m.wrapping_sub(v) as i64 >= 0,
            AcquireOp::CircGeq => (m as u32).wrapping_sub(v as u32) as i32 >= 0,
            AcquireOp::StrictGeq => m >= v,
            AcquireOp::And => m & v != 0,
        }
    }

    /// Reads the semaphore; one that cannot be read does not hold the channel.
    fn met_in(&self, read: Mem) -> bool {
        let mut b = [0u8; 8];
        let n = if self.wide { 8 } else { 4 };
        !read(self.addr, &mut b[..n]) || self.met(u64::from_le_bytes(b))
    }
}

/// What a segment did besides storing methods.
#[derive(Clone, Debug, PartialEq)]
pub enum Effect {
    Release(Release),
    /// A method the fake does not execute but that matters (a copy, an
    /// acquire, a reduction): its class, method byte offset and value.
    Ignored { class: u32, method: u32, value: u32 },
    /// A macro stopped early.
    MacroFault { index: usize, fault: mme::Fault },
    /// CLEAR_SURFACE.
    Clear(Clear),
    /// DRAW_INDEX_BUFFER or DRAW_VERTEX_ARRAY.
    Draw(Box<Draw>),
    /// A copy engine copy, pitch to pitch.
    Copy(Copy),
    /// A compute dispatch: the GPU address of its QMD.
    Dispatch(u64),
    /// A semaphore an inline-to-memory upload releases once its data is in.
    Semaphore(crate::state::qmd::Release),
    /// Bytes the 3D method stream writes to memory, before what follows reads it.
    Write(MemWrite),
    /// A 3D resource method whose effect is not applied.
    Unsupported(Unsupported),
}

/// A draw, with the state it uses.
#[derive(Clone, Debug, PartialEq)]
pub struct Draw {
    /// BEGIN's primitive (4 = triangles).
    pub topology: u32,
    pub index: Option<IndexBuffer>,
    /// First vertex (vertex arrays).
    pub first: u32,
    pub count: u32,
    pub base_vertex: i32,
    /// The GPU address of each enabled pipeline stage's program (SPH then
    /// SASS): 0 vertex before fetch, 1 vertex, 2 tessellation init, 3
    /// tessellation, 4 geometry, 5 pixel.
    pub programs: [Option<u64>; 6],
    pub streams: Vec<Stream>,
    pub attributes: Vec<Attribute>,
    pub render: RenderState,
    /// The color target of each output location (SET_CT_SELECT).
    pub targets: Vec<Surface>,
    /// The depth-stencil target, when SET_ZT_SELECT enables one.
    pub depth: Option<Surface>,
    /// Constant buffers by stage (0 vertex .. 4 pixel) and bank.
    pub cbufs: Bindings,
    pub headers: Pool,
    pub samplers: Pool,
    pub sampler_binding: SamplerBinding,
}

/// A pitch-to-pitch copy (clc7b5.h): `line_count` lines of `line_length`
/// bytes, from `src` every `pitch_in` bytes to `dst` every `pitch_out`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Copy {
    pub src: u64,
    pub dst: u64,
    pub pitch_in: u32,
    pub pitch_out: u32,
    pub line_length: u32,
    pub line_count: u32,
    /// A block-linear destination (then `pitch_out` does not apply).
    pub dst_block: Option<BlockDst>,
}

/// A block-linear copy destination (clc7b5.h SET_DST_*), in bytes: the
/// copy engine without remapping moves bytes.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct BlockDst {
    /// SET_DST_WIDTH (bytes), SET_DST_HEIGHT, SET_DST_DEPTH.
    pub width: u32,
    pub height: u32,
    pub depth: u32,
    /// SET_DST_LAYER: the slice the lines go to.
    pub layer: u32,
    /// DST_ORIGIN_X (bytes) and DST_ORIGIN_Y.
    pub x: u32,
    pub y: u32,
    /// SET_DST_BLOCK_SIZE_HEIGHT (7:4) and _DEPTH (11:8), log2 GOBs.
    pub gob_height_log2: u32,
    pub gob_depth_log2: u32,
}

/// A render target as the 3D class describes it.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Surface {
    /// GPU address.
    pub addr: u64,
    pub width: u32,
    pub height: u32,
    /// The class's format number (SET_COLOR_TARGET_FORMAT or SET_ZT_FORMAT).
    pub format: u32,
    /// Array size (at least 1).
    pub layers: u32,
}

/// A clear: one color target and/or the depth-stencil target, with the
/// class's clear values (raw words: color floats, depth float, stencil).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Clear {
    pub layer: u32,
    /// Target, clear value, channel mask (bit 0 = R .. bit 3 = A).
    pub color: Option<(Surface, [u32; 4], u8)>,
    /// Target, depth value if cleared, stencil value if cleared.
    pub depth: Option<(Surface, Option<u32>, Option<u8>)>,
    /// The scissor that bounds the clear: x, y, width, height.
    pub scissor: Option<[u32; 4]>,
}

/// Methods below this byte offset belong to the host, whatever the subchannel.
const HOST_METHODS: u32 = 0x100;
const SET_OBJECT: u32 = 0x0;
// Host (AMPERE_CHANNEL_GPFIFO_B and its family).
const SEMAPHORE_A: u32 = 0x10;
const SEMAPHORE_B: u32 = 0x14;
const SEMAPHORE_C: u32 = 0x18;
const SEMAPHORE_D: u32 = 0x1c;
const SEM_ADDR_LO: u32 = 0x5c;
const SEM_ADDR_HI: u32 = 0x60;
const SEM_PAYLOAD_LO: u32 = 0x64;
const SEM_PAYLOAD_HI: u32 = 0x68;
const SEM_EXECUTE: u32 = 0x6c;
// 3D and compute.
const REPORT_SEMAPHORE_A: u32 = 0x1b00;
const REPORT_SEMAPHORE_B: u32 = 0x1b04;
const REPORT_SEMAPHORE_C: u32 = 0x1b08;
const REPORT_SEMAPHORE_D: u32 = 0x1b0c;
// Copy engine.
const DMA_SEMAPHORE_A: u32 = 0x240;
const DMA_SEMAPHORE_B: u32 = 0x244;
const DMA_SEMAPHORE_PAYLOAD: u32 = 0x248;
const DMA_LAUNCH: u32 = 0x300;
/// LINE_LENGTH_IN .. LOAD_INLINE_DATA: inline-to-memory uploads.
const I2M_METHODS: std::ops::RangeInclusive<u32> = 0x180..=0x1b4;
// Compute (clc7c0.h): the QMD address >> 8, and the two launch methods.
const SEND_PCAS_A: u32 = 0x2b4;
const SEND_SIGNALING_PCAS_B: u32 = 0x2bc;
const SEND_SIGNALING_PCAS2_B: u32 = 0x2c0;
const DMA_OFFSET_IN_UPPER: u32 = 0x400; // IN_LOWER + 4, OUT_UPPER + 8, OUT_LOWER + 0xc
const DMA_PITCH_IN: u32 = 0x410; // PITCH_OUT + 4, LINE_LENGTH_IN + 8, LINE_COUNT + 0xc
// The 3D class's macro unit.
const LOAD_MME_INSTRUCTION_RAM_POINTER: u32 = 0x114;
const LOAD_MME_INSTRUCTION_RAM: u32 = 0x118;
const LOAD_MME_START_ADDRESS_RAM_POINTER: u32 = 0x11c;
const LOAD_MME_START_ADDRESS_RAM: u32 = 0x120;
const SET_MME_SHADOW_RAM_CONTROL: u32 = 0x124;
/// CALL_MME_MACRO(i) at 0x3800 + 8i starts a call; CALL_MME_DATA(i), 4 bytes
/// further, passes the next parameters.
const CALL_MME: std::ops::Range<u32> = 0x3800..0x3c00;
// Render targets and clears (clc797.h).
const COLOR_TARGET: u32 = 0x800; // + 64 j: A, B, WIDTH, HEIGHT, FORMAT, MEMORY, THIRD_DIMENSION
const COLOR_CLEAR_VALUE: u32 = 0xd80;
const Z_CLEAR_VALUE: u32 = 0xd90;
const STENCIL_CLEAR_VALUE: u32 = 0xda0;
const SCISSOR_ENABLE: u32 = 0xe00; // HORIZONTAL + 4, VERTICAL + 8
const ZT_A: u32 = 0xfe0; // B + 4, FORMAT + 8
const CLEAR_SURFACE_CONTROL: u32 = 0x10f8;
const CT_SELECT: u32 = 0x121c;
const ZT_SIZE_A: u32 = 0x1228; // SIZE_B + 4, SIZE_C + 8
const ZT_SELECT: u32 = 0x1538;
const CLEAR_SURFACE: u32 = 0x19d0;
// Draws.
const VERTEX_ARRAY_START: u32 = 0xd74;
const DRAW_VERTEX_ARRAY: u32 = 0xd78;
const BEGIN: u32 = 0x1618;
const DRAW_INDEX_BUFFER: u32 = 0x17e0;
const PIPELINE_SHADER: u32 = 0x2000; // + 64 j; PROGRAM_ADDRESS_A/B at + 0x14/0x18
/// Draw methods not executed yet: reported, not silently dropped.
const OTHER_DRAWS: [u32; 9] = [0x26c, 0x274, 0x304, 0x1214, 0x1218, 0x17e4, 0x17e8, 0x17ec, 0x17f0];

/// A channel's state between submits.
#[derive(Clone, Debug)]
pub struct Channel {
    /// The class bound to each subchannel (SET_OBJECT).
    pub classes: [u32; 8],
    /// Host method values, by dword.
    host: Vec<u32>,
    /// Each subchannel's method values, by dword.
    engine: Vec<u32>,
    /// The values the 3D macros read (MME shadow RAM), by dword like `engine`.
    shadow: Vec<u32>,
    /// A method whose data words continue into the next segment.
    pending: Option<Pending>,
    /// The macro unit, out of the channel while a macro runs.
    mme: Option<Box<Mme>>,
    /// A macro call still gathering its parameters: subchannel, index, parameters.
    call: Option<(usize, usize, Vec<u32>)>,
    /// The macro running now, if any.
    running: Option<usize>,
    /// The 3D class's constant-buffer bindings and memory writes.
    resources: Resources,
    /// The inline-to-memory uploads of compute and I2M subchannels.
    uploads: std::collections::BTreeMap<usize, Resources>,
    /// Segments not run yet, with where to start in each.
    queue: VecDeque<(Vec<u32>, usize)>,
    /// The acquire the channel waits for.
    wait: Option<Acquire>,
    /// Every method executed, when set (for study tools).
    pub trace: Option<Vec<Traced>>,
}

/// One executed method, for tracing.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Traced {
    pub subchannel: usize,
    pub class: u32,
    pub method: u32,
    pub value: u32,
    /// The macro that emitted it.
    pub from_macro: Option<usize>,
}

#[derive(Clone, Copy, Debug)]
struct Pending {
    op: u32,
    subchannel: usize,
    method: u32,
    left: u32,
}

impl Default for Channel {
    fn default() -> Channel {
        Channel {
            classes: [0; 8],
            host: vec![0; (HOST_METHODS / 4) as usize],
            engine: vec![0; 8 * 0x1000],
            shadow: vec![0; 8 * 0x1000],
            pending: None,
            mme: Some(Box::default()),
            call: None,
            running: None,
            resources: Resources::default(),
            uploads: Default::default(),
            queue: VecDeque::new(),
            wait: None,
            trace: None,
        }
    }
}

/// What a running macro sees: its parameters, and the channel it emits into.
struct MacroHost<'a> {
    channel: &'a mut Channel,
    subchannel: usize,
    params: std::vec::IntoIter<u32>,
    effects: &'a mut Vec<Effect>,
}

impl mme::Host for MacroHost<'_> {
    fn param(&mut self) -> u32 {
        self.params.next().unwrap_or(0)
    }
    fn state(&mut self, method: u32) -> u32 {
        self.channel.shadow.get(self.subchannel * 0x1000 + (method / 4) as usize).copied().unwrap_or(0)
    }
    fn method(&mut self, method: u32, data: u32) {
        self.channel.write(self.subchannel, method, data, self.effects);
    }
}

impl Channel {
    /// Runs one pushbuffer segment, every acquire taken as met (for replays
    /// without memory).
    pub fn run(&mut self, words: &[u32]) -> Vec<Effect> {
        self.run_with(words, &mut |_, _| false)
    }

    /// Queues one pushbuffer segment and runs the channel as far as its
    /// acquires allow; `read` reads the semaphores.
    pub fn run_with(&mut self, words: &[u32], read: Mem) -> Vec<Effect> {
        self.queue.push_back((words.to_vec(), 0));
        self.resume(read)
    }

    /// The acquire the channel waits for, if any.
    pub fn waiting(&self) -> Option<Acquire> {
        self.wait
    }

    /// Gives up the acquire the channel waits for (a watchdog's decision).
    pub fn skip_wait(&mut self) -> Option<Acquire> {
        self.wait.take()
    }

    /// Runs the queued segments until they are done or an acquire is not met.
    pub fn resume(&mut self, read: Mem) -> Vec<Effect> {
        let mut effects = Vec::new();
        while let Some((words, at)) = self.queue.pop_front() {
            if let Some(w) = self.wait {
                if !w.met_in(read) {
                    self.queue.push_front((words, at));
                    break;
                }
                self.wait = None;
            }
            if let Some(next) = self.segment(&words, at, &mut effects) {
                self.queue.push_front((words, next));
            }
        }
        effects
    }

    /// Runs a segment from word `start`; the word to go on from when an
    /// acquire stopped it.
    fn segment(&mut self, words: &[u32], start: usize, effects: &mut Vec<Effect>) -> Option<usize> {
        let mut i = start;
        while i < words.len() {
            if self.wait.is_some() {
                return Some(i);
            }
            let Some(mut p) = self.pending.take() else {
                let h = words[i];
                i += 1;
                let (op, count, subchannel, method) = (h >> 29, h >> 16 & 0x1fff, (h >> 13 & 7) as usize, h & 0xfff);
                match op {
                    // Incrementing, non-incrementing, increment once: data words follow.
                    1 | 3 | 5 if count > 0 => self.pending = Some(Pending { op, subchannel, method, left: count }),
                    // Immediate: the value is in the header.
                    4 => self.write(subchannel, method * 4, count, effects),
                    // End of the segment.
                    7 => break,
                    // Sub-device masks and empty methods carry no data.
                    _ => {}
                }
                continue;
            };
            self.write(p.subchannel, p.method * 4, words[i], effects);
            i += 1;
            p.left -= 1;
            match p.op {
                1 => p.method += 1,
                // Increment once: the first word to the method, the rest to the next one.
                5 => {
                    p.method += 1;
                    p.op = 3;
                }
                _ => {}
            }
            if p.left > 0 {
                self.pending = Some(p);
            }
        }
        if self.wait.is_some() {
            return Some(i);
        }
        // A call whose parameters do not continue in the next segment runs now.
        if self.pending.is_none() {
            self.flush_call(effects);
        }
        None
    }

    /// Runs the macro call gathered so far, if any.
    fn flush_call(&mut self, effects: &mut Vec<Effect>) {
        let Some((subchannel, index, params)) = self.call.take() else { return };
        let Some(mut unit) = self.mme.take() else { return };
        self.running = Some(index);
        let result = unit.call(index, &mut MacroHost { channel: self, subchannel, params: params.into_iter(), effects });
        self.running = None;
        self.mme = Some(unit);
        if let Err(fault) = result {
            effects.push(Effect::MacroFault { index, fault });
        }
    }

    fn write(&mut self, subchannel: usize, method: u32, value: u32, effects: &mut Vec<Effect>) {
        let three_d = self.classes[subchannel] & 0xff == 0x97;
        // The macro unit is the channel's: the 3D and compute classes both load and call it
        // (NVK calls its macros on the compute class too); a macro emits to its caller.
        let macros = three_d || self.classes[subchannel] & 0xff == 0xc0;
        if macros && CALL_MME.contains(&method) {
            let index = ((method - CALL_MME.start) / 8) as usize;
            match &mut self.call {
                Some((s, i, params)) if method % 8 == 4 && *s == subchannel && *i == index => params.push(value),
                _ => {
                    self.flush_call(effects);
                    self.call = Some((subchannel, index, vec![value]));
                }
            }
            return;
        }
        self.flush_call(effects);
        if let Some(t) = &mut self.trace {
            t.push(Traced { subchannel, class: if method < HOST_METHODS { 0 } else { self.classes[subchannel] }, method, value, from_macro: self.running });
        }
        if method < HOST_METHODS {
            self.host[(method / 4) as usize] = value;
            if method == SET_OBJECT {
                self.classes[subchannel] = value;
            }
            if let Some(a) = self.acquire(method, value) {
                self.wait = Some(a);
                return;
            }
            if let Some(e) = self.host_effect(method, value) {
                effects.push(e);
            }
            return;
        }
        let i = subchannel * 0x1000 + (method / 4) as usize;
        if i >= self.engine.len() {
            return;
        }
        // The shadow RAM tracks method values (modes 0 and 1), is left alone
        // (passthrough), or replaces them (replay).
        let mut value = value;
        if method != SET_MME_SHADOW_RAM_CONTROL {
            match self.reg(subchannel, SET_MME_SHADOW_RAM_CONTROL) & 3 {
                0 | 1 => self.shadow[i] = value,
                3 => value = self.shadow[i],
                _ => {}
            }
        }
        self.engine[i] = value;
        if macros && let Some(unit) = self.mme.as_mut() {
            match method {
                LOAD_MME_INSTRUCTION_RAM_POINTER => unit.set_code_pointer(value),
                LOAD_MME_INSTRUCTION_RAM => unit.load_code(value),
                LOAD_MME_START_ADDRESS_RAM_POINTER => unit.set_start_pointer(value),
                LOAD_MME_START_ADDRESS_RAM => unit.load_start(value),
                _ => {}
            }
        }
        if three_d {
            let regs = &self.engine[subchannel * 0x1000..(subchannel + 1) * 0x1000];
            effects.extend(self.resources.method(method, value, &|m| regs.get((m / 4) as usize).copied().unwrap_or(0)).into_iter().map(Effect::Write));
            effects.extend(self.resources.unsupported.drain(..).map(Effect::Unsupported));
            effects.extend(self.resources.semaphores.drain(..).map(Effect::Semaphore));
        } else if matches!(self.classes[subchannel] & 0xff, 0xc0 | 0x40) && (I2M_METHODS).contains(&method) {
            // Compute and inline-to-memory classes share the 3D class's upload methods.
            let regs = &self.engine[subchannel * 0x1000..(subchannel + 1) * 0x1000];
            let r = self.uploads.entry(subchannel).or_default();
            effects.extend(r.method(method, value, &|m| regs.get((m / 4) as usize).copied().unwrap_or(0)).into_iter().map(Effect::Write));
            effects.extend(r.unsupported.drain(..).map(Effect::Unsupported));
            effects.extend(r.semaphores.drain(..).map(Effect::Semaphore));
        }
        if self.classes[subchannel] & 0xff == 0xb5 && method == DMA_LAUNCH {
            effects.extend(self.copy(subchannel, value));
        }
        if let Some(e) = self.engine_effect(subchannel, method, value) {
            effects.push(e);
        }
    }

    fn host_reg(&self, method: u32) -> u32 {
        self.host[(method / 4) as usize]
    }

    fn reg(&self, subchannel: usize, method: u32) -> u32 {
        self.engine[subchannel * 0x1000 + (method / 4) as usize]
    }

    /// SEMAPHORED operations (4:0) 1 ACQUIRE, 4 ACQ_GEQ, 8 ACQ_AND, on
    /// SEMAPHOREA/B/C; SEM_EXECUTE operations (2:0) 0 ACQUIRE, 2 ACQ_STRICT_GEQ,
    /// 3 ACQ_CIRC_GEQ, 4 ACQ_AND, on SEM_ADDR/PAYLOAD (64-bit with bit 24).
    fn acquire(&self, method: u32, value: u32) -> Option<Acquire> {
        let op = match (method, value) {
            (SEMAPHORE_D, v) => [(1, AcquireOp::Equal), (4, AcquireOp::CircGeq), (8, AcquireOp::And)].into_iter().find(|(k, _)| v & 0x1f == *k)?.1,
            (SEM_EXECUTE, v) => [(0, AcquireOp::Equal), (2, AcquireOp::StrictGeq), (3, AcquireOp::CircGeq), (4, AcquireOp::And)].into_iter().find(|(k, _)| v & 7 == *k)?.1,
            _ => return None,
        };
        Some(if method == SEMAPHORE_D {
            Acquire { addr: u64::from(self.host_reg(SEMAPHORE_A) & 0xff) << 32 | u64::from(self.host_reg(SEMAPHORE_B) & !3), value: u64::from(self.host_reg(SEMAPHORE_C)), wide: false, op }
        } else {
            let wide = value >> 24 & 1 == 1;
            let lo = u64::from(self.host_reg(SEM_PAYLOAD_LO));
            Acquire {
                addr: u64::from(self.host_reg(SEM_ADDR_HI) & 0xff) << 32 | u64::from(self.host_reg(SEM_ADDR_LO) & !3),
                value: if wide { lo | u64::from(self.host_reg(SEM_PAYLOAD_HI)) << 32 } else { lo },
                wide,
                op,
            }
        })
    }

    fn host_effect(&self, method: u32, value: u32) -> Option<Effect> {
        let ignored = Effect::Ignored { class: 0, method, value };
        match method {
            // Operation in bits 4:0 (2 = release); bit 24: 4 bytes, else 16 with a timestamp.
            SEMAPHORE_D => Some(match value & 0x1f {
                2 => Effect::Release(Release {
                    method,
                    addr: u64::from(self.host_reg(SEMAPHORE_A) & 0xff) << 32 | u64::from(self.host_reg(SEMAPHORE_B) & !3),
                    value: u64::from(self.host_reg(SEMAPHORE_C)),
                    width: 4,
                    timestamp: value >> 24 & 1 == 0,
                }),
                _ => ignored,
            }),
            // Operation in bits 2:0 (1 = release); bit 24: 64-bit payload; bit 25: timestamp.
            SEM_EXECUTE => Some(match value & 7 {
                1 => {
                    let wide = value >> 24 & 1 == 1;
                    let lo = u64::from(self.host_reg(SEM_PAYLOAD_LO));
                    Effect::Release(Release {
                        method,
                        addr: u64::from(self.host_reg(SEM_ADDR_HI) & 0xff) << 32 | u64::from(self.host_reg(SEM_ADDR_LO) & !3),
                        value: if wide { lo | u64::from(self.host_reg(SEM_PAYLOAD_HI)) << 32 } else { lo },
                        width: if wide { 8 } else { 4 },
                        timestamp: value >> 25 & 1 == 1,
                    })
                }
                _ => ignored,
            }),
            _ => None,
        }
    }

    /// CLEAR_SURFACE: Z (bit 0), stencil (1), R..A (2..5), MRT index (9:6),
    /// array layer (25:10); the MRT index goes through SET_CT_SELECT.
    /// The color target output location `location` writes (SET_CT_SELECT).
    fn color_target(&self, sub: usize, location: u32) -> Surface {
        let reg = |m: u32| self.reg(sub, m);
        let j = reg(CT_SELECT) >> (4 + 3 * location) & 7;
        let t = COLOR_TARGET + 64 * j;
        Surface { addr: u64::from(reg(t)) << 32 | u64::from(reg(t + 4)), width: reg(t + 8), height: reg(t + 0xc), format: reg(t + 0x10) & 0xff, layers: (reg(t + 0x18) & 0xfff_ffff).max(1) }
    }

    /// The depth-stencil target, if SET_ZT_SELECT enables one.
    fn depth_target(&self, sub: usize) -> Option<Surface> {
        let reg = |m: u32| self.reg(sub, m);
        (reg(ZT_SELECT) & 1 != 0).then(|| Surface { addr: u64::from(reg(ZT_A)) << 32 | u64::from(reg(ZT_A + 4)), width: reg(ZT_SIZE_A), height: reg(ZT_SIZE_A + 4), format: reg(ZT_A + 8) & 0x1f, layers: (reg(ZT_SIZE_A + 8) & 0xffff).max(1) })
    }

    fn clear(&self, sub: usize, value: u32) -> Clear {
        let reg = |m: u32| self.reg(sub, m);
        let mask = (value >> 2 & 0xf) as u8;
        let color = (mask != 0).then(|| (self.color_target(sub, value >> 6 & 0xf), [0, 1, 2, 3].map(|i| reg(COLOR_CLEAR_VALUE + 4 * i)), mask));
        let (z, s) = (value & 1 != 0, value & 2 != 0);
        let depth = self.depth_target(sub).filter(|_| z || s).map(|surface| (surface, z.then(|| reg(Z_CLEAR_VALUE)), s.then(|| reg(STENCIL_CLEAR_VALUE) as u8)));
        // Scissor 0 bounds the clear when SET_CLEAR_SURFACE_CONTROL says so and it is on.
        let scissor = (reg(CLEAR_SURFACE_CONTROL) & 0x100 != 0 && reg(SCISSOR_ENABLE) & 1 != 0).then(|| {
            let (h, v) = (reg(SCISSOR_ENABLE + 4), reg(SCISSOR_ENABLE + 8));
            let (x0, x1, y0, y1) = (h & 0xffff, h >> 16, v & 0xffff, v >> 16);
            [x0, y0, x1.saturating_sub(x0), y1.saturating_sub(y0)]
        });
        Clear { layer: value >> 10 & 0xffff, color, depth, scissor }
    }

    fn draw(&self, sub: usize, method: u32, count: u32) -> Draw {
        let reg = |m: u32| self.reg(sub, m);
        let indexed = method == DRAW_INDEX_BUFFER;
        let programs = std::array::from_fn(|j| {
            let p = PIPELINE_SHADER + 64 * j as u32;
            (reg(p) & 1 != 0).then(|| u64::from(reg(p + 0x14) & 0xff) << 32 | u64::from(reg(p + 0x18)))
        });
        let (streams, attributes) = vertex::vertex_input(&reg);
        let render = raster::render_state(&reg);
        Draw {
            topology: reg(BEGIN) & 0xffff,
            index: indexed.then(|| vertex::index_buffer(&reg)),
            first: if indexed { 0 } else { reg(VERTEX_ARRAY_START) },
            count,
            base_vertex: if indexed { vertex::base_vertex(&reg) } else { 0 },
            programs,
            streams,
            attributes,
            targets: render.outputs.iter().map(|o| self.color_target(sub, o.location)).collect(),
            render,
            depth: self.depth_target(sub),
            cbufs: self.resources.bindings(),
            headers: resources::header_pool(&reg),
            samplers: resources::sampler_pool(&reg),
            sampler_binding: resources::sampler_binding(&reg),
        }
    }

    /// LAUNCH_DMA's copy: DATA_TRANSFER_TYPE (1:0) 0 = none; SRC/DST_MEMORY_LAYOUT
    /// (7, 8) 1 = pitch; MULTI_LINE_ENABLE (9); REMAP_ENABLE (10).
    fn copy(&self, sub: usize, value: u32) -> Option<Effect> {
        let reg = |m: u32| self.reg(sub, m);
        // The source must be pitch (bit 7); the destination may be block linear.
        if value & 3 == 0 {
            return None;
        }
        if value >> 7 & 1 == 0 || value >> 10 & 1 != 0 {
            return Some(Effect::Ignored { class: self.classes[sub], method: DMA_LAUNCH, value });
        }
        let at = |m: u32| u64::from(reg(m) & 0x1_ffff) << 32 | u64::from(reg(m + 4));
        Some(Effect::Copy(Copy {
            src: at(DMA_OFFSET_IN_UPPER),
            dst: at(DMA_OFFSET_IN_UPPER + 8),
            pitch_in: reg(DMA_PITCH_IN),
            pitch_out: reg(DMA_PITCH_IN + 4),
            line_length: reg(DMA_PITCH_IN + 8),
            line_count: if value >> 9 & 1 != 0 { reg(DMA_PITCH_IN + 0xc) } else { 1 },
            dst_block: (value >> 8 & 1 == 0).then(|| {
                let block = reg(0x70c);
                BlockDst { width: reg(0x710), height: reg(0x714), depth: reg(0x718), layer: reg(0x71c), x: reg(0x74c), y: reg(0x750), gob_height_log2: block >> 4 & 0xf, gob_depth_log2: block >> 8 & 0xf }
            }),
        }))
    }

    fn engine_effect(&self, subchannel: usize, method: u32, value: u32) -> Option<Effect> {
        let class = self.classes[subchannel];
        let ignored = Effect::Ignored { class, method, value };
        match (class & 0xff, method) {
            // 3D and compute: operation in bits 1:0 (0 = release); bit 28: one word, else four.
            // REPORT_ONLY (2) of REPORT_NONE (27:23 = 0) writes the payload too.
            (0x97 | 0xc0, REPORT_SEMAPHORE_D) => Some(match value & 3 {
                op if op == 0 || (op == 2 && value >> 23 & 0x1f == 0) => Effect::Release(Release {
                    method,
                    addr: u64::from(self.reg(subchannel, REPORT_SEMAPHORE_A) & 0xff) << 32 | u64::from(self.reg(subchannel, REPORT_SEMAPHORE_B)),
                    value: u64::from(self.reg(subchannel, REPORT_SEMAPHORE_C)),
                    width: 4,
                    timestamp: value >> 28 & 1 == 0,
                }),
                // REPORT_ONLY of a counter (pipeline statistics, pixel counts): the
                // 64-bit value then, in the four-word form, the timestamp (NVK's query
                // layout). ponytail: counters are not kept; they read 0.
                2 => Effect::Release(Release {
                    method,
                    addr: u64::from(self.reg(subchannel, REPORT_SEMAPHORE_A) & 0xff) << 32 | u64::from(self.reg(subchannel, REPORT_SEMAPHORE_B)),
                    value: 0,
                    width: if value >> 28 & 1 == 0 { 8 } else { 4 },
                    timestamp: value >> 28 & 1 == 0,
                }),
                _ => ignored,
            }),
            (0x97, CLEAR_SURFACE) => Some(Effect::Clear(self.clear(subchannel, value))),
            // A launch: PCAS_B with SCHEDULE (bit 1), or PCAS2_B whose PCAS_ACTION (3:0) schedules.
            (0xc0, SEND_SIGNALING_PCAS_B) if value & 2 != 0 => Some(Effect::Dispatch(u64::from(self.reg(subchannel, SEND_PCAS_A)) << 8)),
            (0xc0, SEND_SIGNALING_PCAS2_B) if matches!(value & 0xf, 2 | 3 | 9 | 10 | 11) => Some(Effect::Dispatch(u64::from(self.reg(subchannel, SEND_PCAS_A)) << 8)),
            (0x97, DRAW_INDEX_BUFFER | DRAW_VERTEX_ARRAY) => Some(Effect::Draw(Box::new(self.draw(subchannel, method, value)))),
            (0x97, m) if OTHER_DRAWS.contains(&m) => Some(ignored),
            // Copy engine: after the copy (`copy`), a semaphore by bits 4:3 (1 one word, 2 four).
            (0xb5, DMA_LAUNCH) => Some(match value >> 3 & 3 {
                kind @ (1 | 2) => Effect::Release(Release {
                    method,
                    addr: u64::from(self.reg(subchannel, DMA_SEMAPHORE_A) & 0x1ff_ffff) << 32 | u64::from(self.reg(subchannel, DMA_SEMAPHORE_B)),
                    value: u64::from(self.reg(subchannel, DMA_SEMAPHORE_PAYLOAD)),
                    width: 4,
                    timestamp: kind == 2,
                }),
                _ => return None,
            }),
            _ => None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    /// SET_INDEX_BUFFER_A; B + 4, E (size) + 0x10, F (first) + 0x14.
    const INDEX_BUFFER_A: u32 = 0x17c8;

    /// A method header: operation, count (or immediate value), subchannel, byte offset.
    fn header(op: u32, count: u32, subchannel: u32, method: u32) -> u32 {
        op << 29 | count << 16 | subchannel << 13 | (method / 4)
    }

    #[test]
    fn binds_classes_and_releases_a_report_semaphore() {
        let mut c = Channel::default();
        let words = [
            header(1, 1, 0, SET_OBJECT),
            0xc797,
            header(1, 4, 0, REPORT_SEMAPHORE_A),
            0x10,
            0x5fc1_1000,
            2,
            0x1000_f010, // release, one word
        ];
        let effects = c.run(&words);
        assert_eq!(c.classes[0], 0xc797);
        assert_eq!(effects, vec![Effect::Release(Release { method: REPORT_SEMAPHORE_D, addr: 0x10_5fc1_1000, value: 2, width: 4, timestamp: false })]);
    }

    #[test]
    fn host_semaphore_release_and_immediates() {
        let mut c = Channel::default();
        // Immediate data and a NOP-like method first, then SEMAPHOREA..D on subchannel 6.
        let words = [header(4, 0x100, 0, 0x8), header(1, 4, 6, SEMAPHORE_A), 0x10, 0x5fc1_2000, 8, 0x0110_0002];
        assert_eq!(c.run(&words), vec![Effect::Release(Release { method: SEMAPHORE_D, addr: 0x10_5fc1_2000, value: 8, width: 4, timestamp: false })]);
        // Without bit 24 the release is 16 bytes with a timestamp.
        let words = [header(1, 1, 6, SEMAPHORE_D), 2];
        assert!(matches!(c.run(&words)[..], [Effect::Release(Release { timestamp: true, .. })]));
    }

    #[test]
    fn data_continues_into_the_next_segment_and_non_incrementing_methods_repeat() {
        let mut c = Channel::default();
        assert!(c.run(&[header(1, 1, 1, SET_OBJECT), 0xc7c0, header(1, 4, 1, REPORT_SEMAPHORE_A), 0, 0x1000]).is_empty());
        // The last two words of the method arrive in the next segment.
        let effects = c.run(&[7, 0x1000_0000]);
        assert_eq!(effects, vec![Effect::Release(Release { method: REPORT_SEMAPHORE_D, addr: 0x1000, value: 7, width: 4, timestamp: false })]);
        // Non-incrementing: every word goes to the same method; only the last stays.
        assert!(c.run(&[header(3, 3, 1, REPORT_SEMAPHORE_C), 1, 2, 3]).is_empty());
        assert_eq!(c.reg(1, REPORT_SEMAPHORE_C), 3);
        // End of segment stops decoding.
        assert!(c.run(&[header(7, 0, 0, 0), header(1, 1, 1, REPORT_SEMAPHORE_D), 0x1000_0000]).is_empty());
    }

    #[test]
    fn copy_engine_semaphore_and_ignored_operations() {
        let mut c = Channel::default();
        let words = [header(1, 1, 4, SET_OBJECT), 0xc7b5, header(1, 3, 4, DMA_SEMAPHORE_A), 0x1, 0x2000, 5, header(1, 1, 4, DMA_LAUNCH), 1 << 3];
        assert_eq!(c.run(&words), vec![Effect::Release(Release { method: DMA_LAUNCH, addr: 0x1_0000_2000, value: 5, width: 4, timestamp: false })]);
        // A pitch-to-pitch copy of two 0x40-byte lines, then its one-word semaphore.
        let words = [header(1, 8, 4, DMA_OFFSET_IN_UPPER), 0x12, 0x3400, 0x12, 0x5600, 0x100, 0x80, 0x40, 2, header(1, 1, 4, DMA_LAUNCH), 2 | 3 << 7 | 1 << 9 | 1 << 3];
        let copy = Copy { src: 0x12_0000_3400, dst: 0x12_0000_5600, pitch_in: 0x100, pitch_out: 0x80, line_length: 0x40, line_count: 2, dst_block: None };
        assert_eq!(c.run(&words), vec![Effect::Copy(copy), Effect::Release(Release { method: DMA_LAUNCH, addr: 0x1_0000_2000, value: 5, width: 4, timestamp: false })]);
        // A reduction is reported as ignored, not executed.
        assert!(matches!(c.run(&[header(1, 1, 0, SEMAPHORE_D), 0x10])[..], [Effect::Ignored { method: SEMAPHORE_D, value: 0x10, .. }]));
    }

    #[test]
    fn an_acquire_holds_the_channel_until_memory_meets_it() {
        let mut c = Channel::default();
        // Wait until the word at 0x10_0000_1000 is >= 5 (ACQ_GEQ), then release 7 at 0x2000.
        let acquire = [header(1, 4, 0, SEMAPHORE_A), 0x10, 0x1000, 5, 0x1004];
        let release = [header(1, 4, 0, SEMAPHORE_A), 0, 0x2000, 7, 0x0100_0002];
        let mut memory = 4u32;
        let mut read = |addr: u64, buf: &mut [u8]| {
            assert_eq!(addr, 0x10_0000_1000);
            buf.copy_from_slice(&memory.to_le_bytes()[..buf.len()]);
            true
        };
        assert!(c.run_with(&[&acquire[..], &release[..]].concat(), &mut read).is_empty());
        assert_eq!(c.waiting().map(|a| (a.value, a.op)), Some((5, AcquireOp::CircGeq)));
        // A later segment queues behind the wait.
        assert!(c.run_with(&release, &mut read).is_empty());
        memory = 5;
        let mut read = |_: u64, buf: &mut [u8]| {
            buf.copy_from_slice(&memory.to_le_bytes()[..buf.len()]);
            true
        };
        let r = Effect::Release(Release { method: SEMAPHORE_D, addr: 0x2000, value: 7, width: 4, timestamp: false });
        assert_eq!(c.resume(&mut read), vec![r.clone(), r]);
        assert_eq!(c.waiting(), None);
        // Wrap-around: 0xffff_fffe is not yet at 2, 3 is.
        let a = Acquire { addr: 0, value: 2, wide: false, op: AcquireOp::CircGeq };
        assert!(!a.met(0xffff_fffe) && a.met(3) && a.met(2));
        assert!(Acquire { op: AcquireOp::And, ..a }.met(6) && !Acquire { op: AcquireOp::Equal, ..a }.met(3));
    }

    /// The three words of an MME64 group, high first.
    fn group(bits: u128) -> [u32; 3] {
        [(bits >> 64) as u32, (bits >> 32) as u32, bits as u32]
    }

    #[test]
    fn a_loaded_macro_runs_when_called_and_its_methods_take_effect() {
        // Group 0: method REPORT_SEMAPHORE_A (index 0x6c0, increment 1) from immediate 0,
        // emit LOAD0 then LOAD1; group 1 (end next): emit LOAD0, LOAD1; group 2: nothing.
        let code = [group((0x16c0 << 30) | (5 << 82) | (3 << 85) | (4 << 92)), group(1 | (3 << 85) | (4 << 92)), group(0)].concat();
        let mut words = vec![header(1, 1, 0, SET_OBJECT), 0xc797, header(1, 1, 0, LOAD_MME_INSTRUCTION_RAM_POINTER), 0, header(3, 9, 0, LOAD_MME_INSTRUCTION_RAM)];
        words.extend(code);
        words.extend([header(1, 2, 0, LOAD_MME_START_ADDRESS_RAM_POINTER), 5, 0]);
        // Macro 5, called with its four parameters (increment once: then CALL_MME_DATA(5)).
        words.extend([header(5, 4, 0, CALL_MME.start + 5 * 8), 0x10, 0x5fc1_1000, 2, 0x1000_f010]);
        let mut c = Channel::default();
        assert_eq!(c.run(&words), vec![Effect::Release(Release { method: REPORT_SEMAPHORE_D, addr: 0x10_5fc1_1000, value: 2, width: 4, timestamp: false })]);
        // An unloaded macro is a fault.
        assert!(matches!(c.run(&[header(1, 1, 0, CALL_MME.start + 8), 0])[..], [Effect::MacroFault { index: 1, .. }]));
    }

    #[test]
    fn clear_surface_describes_the_selected_target() {
        let mut c = Channel::default();
        // Color target 1 at 0x81_0000_2000, 1600x900, RF16_GF16_BF16_AF16; MRT 0 -> target 1.
        let t = COLOR_TARGET + 64;
        let words = [
            header(1, 1, 0, SET_OBJECT),
            0xc797,
            header(1, 5, 0, t),
            0x81,
            0x2000,
            1600,
            900,
            0xca,
            header(1, 1, 0, t + 0x18),
            1,
            header(1, 1, 0, CT_SELECT),
            1 | 1 << 4,
            header(1, 4, 0, COLOR_CLEAR_VALUE),
            0x3f00_0000,
            0,
            0x3f00_0000,
            0x3f80_0000,
            header(1, 1, 0, CLEAR_SURFACE),
            0x3c,
        ];
        let surface = Surface { addr: 0x81_0000_2000, width: 1600, height: 900, format: 0xca, layers: 1 };
        let expected = Clear { layer: 0, color: Some((surface, [0x3f00_0000, 0, 0x3f00_0000, 0x3f80_0000], 0xf)), depth: None, scissor: None };
        assert_eq!(c.run(&words), vec![Effect::Clear(expected)]);
    }

    #[test]
    fn an_indexed_draw_names_its_programs_and_indices() {
        let mut c = Channel::default();
        let vs = PIPELINE_SHADER + 64;
        let ps = PIPELINE_SHADER + 5 * 64;
        let words = [
            header(1, 1, 0, SET_OBJECT),
            0xc797,
            header(1, 1, 0, vs),
            0x11,
            header(1, 2, 0, vs + 0x14),
            0x10,
            0x6ab9_0900,
            header(1, 1, 0, ps),
            0x51,
            header(1, 2, 0, ps + 0x14),
            0x11,
            0x5987_6400,
            header(1, 2, 0, INDEX_BUFFER_A),
            0x81,
            0x2000,
            header(1, 2, 0, INDEX_BUFFER_A + 0x10),
            1,
            6,
            header(4, 4, 0, BEGIN),
            header(1, 1, 0, DRAW_INDEX_BUFFER),
            3,
        ];
        let mut programs = [None; 6];
        programs[1] = Some(0x10_6ab9_0900);
        programs[5] = Some(0x11_5987_6400);
        let effects = c.run(&words);
        let [Effect::Draw(d)] = &effects[..] else { panic!("{effects:?}") };
        assert_eq!((d.topology, d.first, d.count, d.programs), (4, 0, 3, programs));
        let ib = d.index.as_ref().unwrap();
        assert_eq!((ib.address, ib.bytes_per_index, ib.first), (0x81_0000_2000, 2, 6));
    }

    #[test]
    fn constant_buffer_loads_come_out_in_stream_order_and_bind_to_draws() {
        let mut c = Channel::default();
        let words = [
            header(1, 1, 0, SET_OBJECT),
            0xc797,
            // SET_CONSTANT_BUFFER_SELECTOR_A..C: 0x100 bytes at 0x12_3456_7800.
            header(1, 3, 0, 0x2380),
            0x100,
            0x12,
            0x3456_7800,
            // Bind it to the pixel stage's bank 2, then load two words at 0x70.
            header(1, 1, 0, 0x2410 + 4 * 32),
            0x21,
            header(1, 1, 0, 0x238c),
            0x70,
            header(3, 2, 0, 0x2390),
            0x1111_1111,
            0x2222_2222,
            header(1, 1, 0, DRAW_VERTEX_ARRAY),
            3,
        ];
        let effects = c.run(&words);
        let writes: Vec<(u64, Vec<u8>)> = effects.iter().filter_map(|e| if let Effect::Write(w) = e { Some((w.address, w.data.clone())) } else { None }).collect();
        assert_eq!(writes, vec![(0x12_3456_7870, vec![0x11; 4]), (0x12_3456_7874, vec![0x22; 4])]);
        let Some(Effect::Draw(d)) = effects.last() else { panic!("{effects:?}") };
        assert_eq!(d.cbufs[4][2], Some((0x12_3456_7800, 0x100)));
        assert!(d.index.is_none());
    }

    #[test]
    fn replay_mode_takes_values_from_the_shadow() {
        let mut c = Channel::default();
        c.run(&[header(1, 1, 0, SET_OBJECT), 0xc797, header(1, 1, 0, REPORT_SEMAPHORE_C), 5]);
        c.run(&[header(1, 1, 0, SET_MME_SHADOW_RAM_CONTROL), 3, header(1, 1, 0, REPORT_SEMAPHORE_C), 9]);
        assert_eq!(c.reg(0, REPORT_SEMAPHORE_C), 5);
        c.run(&[header(1, 1, 0, SET_MME_SHADOW_RAM_CONTROL), 2, header(1, 1, 0, REPORT_SEMAPHORE_C), 9]);
        assert_eq!((c.reg(0, REPORT_SEMAPHORE_C), c.shadow[(REPORT_SEMAPHORE_C / 4) as usize]), (9, 5));
    }
}
