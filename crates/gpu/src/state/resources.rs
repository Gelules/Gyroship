//! What a draw's shaders read from memory (constant buffers, texture header
//! and sampler pools) and the memory the 3D method stream writes by itself
//! (constant-buffer loads and the class's inline-to-memory upload).
//!
//! Facts come from clc797.h (open-gpu-doc, MIT) and from how Mesa NVK/NAK
//! (MIT) drive the same methods:
//! - Bind group j of BIND_GROUP_CONSTANT_BUFFER(j) is a shader stage:
//!   0 vertex, 1 tessellation control, 2 tessellation evaluation,
//!   3 geometry, 4 pixel (NVK binds group = Mesa stage number, and
//!   SET_ROOT_TABLE_VISIBILITY names exactly five groups, 0..4).
//! - LOAD_CONSTANT_BUFFER(i) is an array only so that an incrementing
//!   method burst works: every element writes one dword at the current
//!   LOAD_CONSTANT_BUFFER_OFFSET, which then advances by 4 (NVK sends
//!   LOAD_CONSTANT_BUFFER_OFFSET followed by data with a 1INC burst; the
//!   game's macros write LOAD_CONSTANT_BUFFER(0) repeatedly).
//! - A 32-bit texture handle holds the texture header index in bits 19:0 and
//!   the sampler index in bits 31:20 (NAK merges a texture and a sampler
//!   handle with exactly these masks before the texture instruction).
//! - This class has no bindless-texture constant-buffer slot method:
//!   SET_BINDLESS_TEXTURE (0x2608, CONSTANT_BUFFER_SLOT_SELECT) exists up to
//!   clc597.h and is absent from clc697.h and clc797.h. On SM70+ the texture
//!   instruction itself names the bank and byte offset of its handle (NAK's
//!   "cbuf" texture reference), so that location comes from the shader.

use super::Reg;
use super::qmd::{self, Reduction};

// Method byte offsets, clc797.h.
const LINE_LENGTH_IN: u32 = 0x0180;
const LINE_COUNT: u32 = 0x0184;
const OFFSET_OUT_UPPER: u32 = 0x0188;
const OFFSET_OUT: u32 = 0x018c;
const PITCH_OUT: u32 = 0x0190;
const LAUNCH_DMA: u32 = 0x01b0;
const LOAD_INLINE_DATA: u32 = 0x01b4;
/// SET_I2M_SEMAPHORE_A (upper 7:0), _B (lower), _C (payload).
const I2M_SEMAPHORE: u32 = 0x01dc;
const SET_ROOT_TABLE_SELECTOR: u32 = 0x0504;
const LOAD_ROOT_TABLE: u32 = 0x0508;
const UNBIND_ALL: u32 = 0x10f4;
const SET_SAMPLER_BINDING: u32 = 0x1234;
const SET_TEX_SAMPLER_POOL_A: u32 = 0x155c;
const SET_TEX_SAMPLER_POOL_B: u32 = 0x1560;
const SET_TEX_SAMPLER_POOL_C: u32 = 0x1564;
const SET_TEX_HEADER_POOL_A: u32 = 0x1574;
const SET_TEX_HEADER_POOL_B: u32 = 0x1578;
const SET_TEX_HEADER_POOL_C: u32 = 0x157c;
const SET_CONSTANT_BUFFER_SELECTOR_A: u32 = 0x2380;
const SET_CONSTANT_BUFFER_SELECTOR_B: u32 = 0x2384;
const SET_CONSTANT_BUFFER_SELECTOR_C: u32 = 0x2388;
const LOAD_CONSTANT_BUFFER_OFFSET: u32 = 0x238c;
/// LOAD_CONSTANT_BUFFER(i) = 0x2390 + 4i; the next method starts at 0x2400.
const LOAD_CONSTANT_BUFFER: std::ops::Range<u32> = 0x2390..0x2400;
/// BIND_GROUP_CONSTANT_BUFFER(j) = 0x2410 + 32j; the stride-32 array ends
/// before SET_TRAP_HANDLER_A (0x25f8).
const BIND_GROUP_CONSTANT_BUFFER: u32 = 0x2410;

/// Bind groups (shader stages), see the module doc.
pub const STAGES: usize = 5;
/// BIND_GROUP_CONSTANT_BUFFER_SHADER_SLOT is 5 bits wide (8:4).
pub const BANKS: usize = 32;
/// Bytes per texture header: the TEXHEAD_* fields of clc797tex.h span
/// 256 bits (NVK_TEXTURE_HEADER_SIZE is 32 too).
pub const HEADER_SIZE: u64 = 32;
/// Bytes per sampler: clc797tex.h defines TEXSAMP0..TEXSAMP7, 8 dwords.
pub const SAMPLER_SIZE: u64 = 32;

/// Bytes the stream writes to guest memory (little endian, as the GPU does).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MemWrite {
    pub address: u64,
    pub data: Vec<u8>,
}

/// Something the stream did that this decoder does not turn into effects.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Unsupported {
    /// LAUNCH_DMA_DST_MEMORY_LAYOUT_BLOCKLINEAR: the inline data is consumed
    /// but not written (the SET_DST_* block-linear layout is not decoded).
    I2mBlockLinear { address: u64, bytes: u64 },
    /// Pitch upload of several lines whose LINE_LENGTH_IN is not a multiple
    /// of 4: whether each line starts on a new LOAD_INLINE_DATA dword is not
    /// established, so the data is consumed but not written.
    I2mLinePacking { address: u64, line_length: u32, line_count: u32 },
    /// LOAD_CONSTANT_BUFFER at or past SET_CONSTANT_BUFFER_SELECTOR_A_SIZE:
    /// what the hardware does is not established; nothing is written.
    CbufLoadPastSize { address: u64, offset: u32, size: u32 },
    /// A method whose effect is not applied: SET_ROOT_TABLE_SELECTOR,
    /// LOAD_ROOT_TABLE, UNBIND_ALL (which groups it clears is not stated),
    /// BIND_GROUP_CONSTANT_BUFFER(j) for j >= 5, LOAD_INLINE_DATA with no
    /// upload pending.
    Method { method: u32, value: u32 },
}

/// A texture header pool or sampler pool.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Pool {
    /// SET_TEX_HEADER_POOL_A_OFFSET_UPPER (7:0) / _B_OFFSET_LOWER, or the
    /// SET_TEX_SAMPLER_POOL_A/B equivalents.
    pub address: u64,
    /// SET_TEX_HEADER_POOL_C_MAXIMUM_INDEX (21:0) or
    /// SET_TEX_SAMPLER_POOL_C_MAXIMUM_INDEX (19:0): the last valid index
    /// (NVK writes entry count - 1).
    pub maximum_index: u32,
    entry_size: u64,
}

impl Pool {
    /// Address of entry `index`, or None past MAXIMUM_INDEX.
    pub fn entry(&self, index: u32) -> Option<u64> {
        (index <= self.maximum_index).then(|| self.address + index as u64 * self.entry_size)
    }
}

fn address(upper: u32, lower: u32) -> u64 {
    ((upper & 0xff) as u64) << 32 | lower as u64
}

/// The texture header pool (SET_TEX_HEADER_POOL_A/B/C). The per-draw readers
/// take any register reader, inlined into a draw's snapshot (see `vertex`).
pub fn header_pool<R: Fn(u32) -> u32 + ?Sized>(r: &R) -> Pool {
    Pool {
        address: address(r(SET_TEX_HEADER_POOL_A), r(SET_TEX_HEADER_POOL_B)),
        maximum_index: r(SET_TEX_HEADER_POOL_C) & 0x3f_ffff,
        entry_size: HEADER_SIZE,
    }
}

/// The sampler pool (SET_TEX_SAMPLER_POOL_A/B/C).
pub fn sampler_pool<R: Fn(u32) -> u32 + ?Sized>(r: &R) -> Pool {
    Pool {
        address: address(r(SET_TEX_SAMPLER_POOL_A), r(SET_TEX_SAMPLER_POOL_B)),
        maximum_index: r(SET_TEX_SAMPLER_POOL_C) & 0xf_ffff,
        entry_size: SAMPLER_SIZE,
    }
}

/// SET_SAMPLER_BINDING_V (0:0).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum SamplerBinding {
    /// V_INDEPENDENTLY: the sampler index is the handle's own (bits 31:20).
    Independently,
    /// V_VIA_HEADER_BINDING: by its name, the sampler follows the texture
    /// header binding; how that applies to handles is not verified (the
    /// game uses Independently).
    ViaHeaderBinding,
}

pub fn sampler_binding<R: Fn(u32) -> u32 + ?Sized>(r: &R) -> SamplerBinding {
    if r(SET_SAMPLER_BINDING) & 1 == 0 { SamplerBinding::Independently } else { SamplerBinding::ViaHeaderBinding }
}

/// A 32-bit texture handle as read by a texture instruction.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TexHandle {
    /// Bits 19:0: index into the texture header pool.
    pub header: u32,
    /// Bits 31:20: index into the sampler pool.
    pub sampler: u32,
}

impl TexHandle {
    pub fn new(handle: u32) -> Self {
        TexHandle { header: handle & 0xf_ffff, sampler: handle >> 20 }
    }
}

/// An inline-to-memory upload started by LAUNCH_DMA, fed by LOAD_INLINE_DATA.
#[derive(Clone, Debug)]
struct Upload {
    address: u64,
    line_length: u64,
    pitch: u64,
    total: u64,
    done: u64,
    write: bool,
    /// Released once the data is in.
    semaphore: Option<qmd::Release>,
}

/// Constant-buffer bindings and the stream's own memory writes.
#[derive(Default, Clone, Debug)]
pub struct Resources {
    banks: Bindings,
    /// LOAD_CONSTANT_BUFFER_OFFSET_V (15:0), advanced by 4 per data dword.
    cbuf_offset: u32,
    upload: Option<Upload>,
    /// Everything not turned into effects, in stream order (the command
    /// processor drains it after each method).
    pub unsupported: Vec<Unsupported>,
    /// Semaphores the finished uploads release, in order (drained likewise).
    pub semaphores: Vec<qmd::Release>,
}

/// The constant buffers bound to every stage and bank: (address, size).
pub type Bindings = [[Option<(u64, u32)>; BANKS]; STAGES];
/// A draw's banks: those of the only stages it runs, vertex (0) then pixel (4);
/// `try_draw` refuses a program on another stage before it reads a bank.
pub type DrawBanks = [[Option<(u64, u32)>; BANKS]; 2];

/// The methods [`Resources::method`] acts on, as (first, last, step): any other
/// only sets its register, and the command processor skips it (`pushbuf::ACTED`).
pub(crate) const ACTED: [(u32, u32, u32); 5] = [
    (LAUNCH_DMA, LOAD_INLINE_DATA, 4),
    (SET_ROOT_TABLE_SELECTOR, LOAD_ROOT_TABLE, 4),
    (UNBIND_ALL, UNBIND_ALL, 4),
    (LOAD_CONSTANT_BUFFER_OFFSET, LOAD_CONSTANT_BUFFER.end - 4, 4),
    (BIND_GROUP_CONSTANT_BUFFER, BIND_GROUP_END - 1, 32),
];
/// BIND_GROUP_CONSTANT_BUFFER(j) for j = 0..15 (stages 5 and up are reported).
const BIND_GROUP_END: u32 = 0x25f8;

impl Resources {
    /// Feeds one 3D method; `r` reads the register file after this write, and
    /// `write` receives each guest write it makes (address, bytes), in order.
    pub fn method(&mut self, method: u32, value: u32, r: Reg, write: &mut dyn FnMut(u64, &[u8])) {
        match method {
            LOAD_CONSTANT_BUFFER_OFFSET => self.cbuf_offset = value & 0xffff,
            m if LOAD_CONSTANT_BUFFER.contains(&m) => self.load_cbuf(value, r, write),
            m if (BIND_GROUP_CONSTANT_BUFFER..BIND_GROUP_END).contains(&m) && (m - BIND_GROUP_CONSTANT_BUFFER).is_multiple_of(32) => {
                let stage = ((m - BIND_GROUP_CONSTANT_BUFFER) / 32) as usize;
                if stage >= STAGES {
                    self.unsupported.push(Unsupported::Method { method, value });
                } else {
                    // BIND_GROUP_CONSTANT_BUFFER_VALID (0:0), _SHADER_SLOT (8:4):
                    // the selector at this moment becomes the bank.
                    let bank = ((value >> 4) & 0x1f) as usize;
                    self.banks[stage][bank] = (value & 1 == 1).then(|| selector(r));
                }
            }
            LAUNCH_DMA => self.launch(value, r),
            LOAD_INLINE_DATA => self.inline_data(method, value, write),
            SET_ROOT_TABLE_SELECTOR | LOAD_ROOT_TABLE | UNBIND_ALL => self.unsupported.push(Unsupported::Method { method, value }),
            _ => {}
        }
    }

    /// The banks a draw reads: the vertex stage's (0), then the pixel stage's (4).
    pub fn draw_banks(&self) -> DrawBanks {
        [self.banks[0], self.banks[4]]
    }

    /// (address, size) bound to bank `bank` of stage `stage`.
    pub fn cbuf(&self, stage: usize, bank: usize) -> Option<(u64, u32)> {
        *self.banks.get(stage)?.get(bank)?
    }

    fn load_cbuf(&mut self, value: u32, r: Reg, write: &mut dyn FnMut(u64, &[u8])) {
        let (address, size) = selector(r);
        let offset = self.cbuf_offset;
        self.cbuf_offset += 4;
        if offset + 4 > size {
            self.unsupported.push(Unsupported::CbufLoadPastSize { address, offset, size });
            return;
        }
        write(address + offset as u64, &value.to_le_bytes());
    }

    fn launch(&mut self, value: u32, r: Reg) {
        // LAUNCH_DMA_COMPLETION_TYPE (5:4) == RELEASE_SEMAPHORE (2): SEMAPHORE_STRUCT_SIZE
        // (12) 0 = four words (with a timestamp), REDUCTION_ENABLE (1), REDUCTION_OP
        // (15:13), REDUCTION_FORMAT (3:2) 1 = signed.
        let semaphore = ((value >> 4) & 3 == 2).then(|| {
            use Reduction::*;
            qmd::Release {
                addr: address(r(I2M_SEMAPHORE), r(I2M_SEMAPHORE + 4)),
                payload: u64::from(r(I2M_SEMAPHORE + 8)),
                reduction: (value >> 1 & 1 == 1).then(|| [Add, Min, Max, Inc, Dec, And, Or, Xor][(value >> 13 & 7) as usize]),
                signed: value >> 2 & 3 == 1,
                wide: false,
                timestamp: value >> 12 & 1 == 0,
            }
        });
        let address = address(r(OFFSET_OUT_UPPER), r(OFFSET_OUT));
        let (line_length, line_count) = (r(LINE_LENGTH_IN), r(LINE_COUNT));
        let total = line_length as u64 * line_count as u64;
        // LAUNCH_DMA_DST_MEMORY_LAYOUT (0:0): 0 BLOCKLINEAR, 1 PITCH.
        let pitch_layout = value & 1 == 1;
        let packing_known = line_count <= 1 || line_length % 4 == 0;
        if !pitch_layout {
            self.unsupported.push(Unsupported::I2mBlockLinear { address, bytes: total });
        } else if !packing_known {
            self.unsupported.push(Unsupported::I2mLinePacking { address, line_length, line_count });
        }
        if total == 0 {
            self.semaphores.extend(semaphore);
        }
        self.upload = (total > 0).then(|| Upload {
            address,
            line_length: line_length as u64,
            pitch: r(PITCH_OUT) as u64,
            total,
            done: 0,
            write: pitch_layout && packing_known,
            semaphore,
        });
    }

    fn inline_data(&mut self, method: u32, value: u32, write: &mut dyn FnMut(u64, &[u8])) {
        let Some(u) = &mut self.upload else {
            self.unsupported.push(Unsupported::Method { method, value });
            return;
        };
        let n = (u.total - u.done).min(4);
        if u.write {
            for (k, byte) in value.to_le_bytes().into_iter().take(n as usize).enumerate() {
                // Line i / LINE_LENGTH_IN lands at OFFSET_OUT + line * PITCH_OUT.
                let i = u.done + k as u64;
                write(u.address + i / u.line_length * u.pitch + i % u.line_length, &[byte]);
            }
        }
        u.done += n;
        if u.done == u.total {
            self.semaphores.extend(u.semaphore);
            self.upload = None;
        }
    }
}

/// SET_CONSTANT_BUFFER_SELECTOR_B_ADDRESS_UPPER (7:0) / _C_ADDRESS_LOWER and
/// SET_CONSTANT_BUFFER_SELECTOR_A_SIZE (16:0, bytes).
fn selector(r: Reg) -> (u64, u32) {
    (
        address(r(SET_CONSTANT_BUFFER_SELECTOR_B), r(SET_CONSTANT_BUFFER_SELECTOR_C)),
        r(SET_CONSTANT_BUFFER_SELECTOR_A) & 0x1_ffff,
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    /// Writes `m = v` into the register file, then feeds it; the guest writes
    /// it made, contiguous ones merged (as the channel does).
    fn feed(res: &mut Resources, regs: &mut HashMap<u32, u32>, m: u32, v: u32) -> Vec<MemWrite> {
        regs.insert(m, v);
        let r = |x: u32| regs.get(&x).copied().unwrap_or(0);
        let mut out: Vec<MemWrite> = Vec::new();
        res.method(m, v, &r, &mut |address, bytes| match out.last_mut() {
            Some(w) if w.address + w.data.len() as u64 == address => w.data.extend_from_slice(bytes),
            _ => out.push(MemWrite { address, data: bytes.to_vec() }),
        });
        out
    }

    fn select(res: &mut Resources, regs: &mut HashMap<u32, u32>, size: u32, addr: u64) {
        feed(res, regs, 0x2380, size);
        feed(res, regs, 0x2384, (addr >> 32) as u32);
        feed(res, regs, 0x2388, addr as u32);
    }

    #[test]
    fn bind_captures_the_selector_at_bind_time() {
        let (mut res, mut regs) = (Resources::default(), HashMap::new());
        // Size 16:0 keeps 0x10000; upper 7:0 drops 0xff00.
        select(&mut res, &mut regs, 0xfffe_0000 | 0x1_0000, 0xff12_3456_7800);
        feed(&mut res, &mut regs, 0x2410 + 4 * 32, 0x21); // pixel, bank 2, valid
        select(&mut res, &mut regs, 0x100, 0x99_0000_0000);
        assert_eq!(res.cbuf(4, 2), Some((0x12_3456_7800, 0x1_0000)));
        feed(&mut res, &mut regs, 0x2410, 0x1f1); // vertex, bank 31
        assert_eq!(res.cbuf(0, 31), Some((0x99_0000_0000, 0x100)));
        feed(&mut res, &mut regs, 0x2410 + 4 * 32, 0x20); // valid = false unbinds
        assert_eq!(res.cbuf(4, 2), None);
        assert_eq!(res.cbuf(5, 0), None);
        assert_eq!(res.cbuf(0, 32), None);
        feed(&mut res, &mut regs, 0x2410 + 5 * 32, 0x01); // no sixth group
        assert_eq!(res.unsupported, vec![Unsupported::Method { method: 0x2410 + 5 * 32, value: 1 }]);
        // BIND_GROUP_RESERVED_A..D are not bindings.
        feed(&mut res, &mut regs, 0x2400, 0x01);
        assert_eq!(res.unsupported.len(), 1);
    }

    #[test]
    fn cbuf_loads_advance_whatever_the_array_index() {
        let (mut res, mut regs) = (Resources::default(), HashMap::new());
        select(&mut res, &mut regs, 0x80, 0x10_0000_1000);
        feed(&mut res, &mut regs, 0x238c, 0xffff_0070); // OFFSET_V is 15:0
        let a = feed(&mut res, &mut regs, 0x2390, 0x1122_3344);
        let b = feed(&mut res, &mut regs, 0x239c, 0x5566_7788);
        assert_eq!(a, vec![MemWrite { address: 0x10_0000_1070, data: vec![0x44, 0x33, 0x22, 0x11] }]);
        assert_eq!(b, vec![MemWrite { address: 0x10_0000_1074, data: vec![0x88, 0x77, 0x66, 0x55] }]);
        // Writes follow the selector of the moment, at the running offset.
        select(&mut res, &mut regs, 0x80, 0x20_0000_0000);
        assert_eq!(feed(&mut res, &mut regs, 0x23fc, 1)[0].address, 0x20_0000_0078);
        // 0x7c is the last dword of a 0x80-byte buffer; 0x80 is past it.
        feed(&mut res, &mut regs, 0x238c, 0x7c);
        assert_eq!(feed(&mut res, &mut regs, 0x2390, 2).len(), 1);
        assert!(feed(&mut res, &mut regs, 0x2390, 3).is_empty());
        assert_eq!(res.unsupported, vec![Unsupported::CbufLoadPastSize { address: 0x20_0000_0000, offset: 0x80, size: 0x80 }]);
        // 0x2400 is BIND_GROUP_RESERVED_A(0), not a load.
        assert!(feed(&mut res, &mut regs, 0x2400, 4).is_empty());
    }

    fn i2m(res: &mut Resources, regs: &mut HashMap<u32, u32>, len: u32, count: u32, pitch: u32, dst: u64, launch: u32) {
        feed(res, regs, 0x0180, len);
        feed(res, regs, 0x0184, count);
        feed(res, regs, 0x0188, (dst >> 32) as u32);
        feed(res, regs, 0x018c, dst as u32);
        feed(res, regs, 0x0190, pitch);
        feed(res, regs, 0x01b0, launch);
    }

    #[test]
    fn inline_upload_pitch_single_line() {
        let (mut res, mut regs) = (Resources::default(), HashMap::new());
        i2m(&mut res, &mut regs, 6, 1, 0, 0xab_0000_0100, 0x1011);
        let a = feed(&mut res, &mut regs, 0x01b4, 0x0403_0201);
        let b = feed(&mut res, &mut regs, 0x01b4, 0x0807_0605);
        assert_eq!(a, vec![MemWrite { address: 0xab_0000_0100, data: vec![1, 2, 3, 4] }]);
        assert_eq!(b, vec![MemWrite { address: 0xab_0000_0104, data: vec![5, 6] }]);
        // The upload is over: more data has nowhere to go.
        assert!(feed(&mut res, &mut regs, 0x01b4, 9).is_empty());
        assert_eq!(res.unsupported, vec![Unsupported::Method { method: 0x01b4, value: 9 }]);
    }

    #[test]
    fn inline_upload_pitch_lines() {
        let (mut res, mut regs) = (Resources::default(), HashMap::new());
        i2m(&mut res, &mut regs, 8, 2, 0x100, 0x1000, 0x01);
        let w: Vec<u64> = (0..4).flat_map(|v| feed(&mut res, &mut regs, 0x01b4, v)).map(|w| w.address).collect();
        assert_eq!(w, vec![0x1000, 0x1004, 0x1100, 0x1104]);
        assert!(res.unsupported.is_empty());
        // 6-byte lines: packing across dwords is not established.
        i2m(&mut res, &mut regs, 6, 2, 0x100, 0x1000, 0x01);
        assert!((0..3).all(|v| feed(&mut res, &mut regs, 0x01b4, v).is_empty()));
        assert_eq!(res.unsupported, vec![Unsupported::I2mLinePacking { address: 0x1000, line_length: 6, line_count: 2 }]);
    }

    #[test]
    fn inline_upload_block_linear_is_reported() {
        let (mut res, mut regs) = (Resources::default(), HashMap::new());
        i2m(&mut res, &mut regs, 16, 1, 0, 0x2000, 0x1010);
        assert!((0..4).all(|v| feed(&mut res, &mut regs, 0x01b4, v).is_empty()));
        assert_eq!(res.unsupported, vec![Unsupported::I2mBlockLinear { address: 0x2000, bytes: 16 }]);
        // RELEASE_SEMAPHORE completion: a one-word increment at 0x12_0000_4000 once the data is in.
        for (m, v) in [(0x01dc, 0x12), (0x01e0, 0x4000), (0x01e4, 9)] {
            feed(&mut res, &mut regs, m, v);
        }
        i2m(&mut res, &mut regs, 4, 1, 0, 0x3000, 0x21 | 1 << 12 | 1 << 1 | 3 << 13);
        assert!(res.semaphores.is_empty());
        assert_eq!(feed(&mut res, &mut regs, 0x01b4, 7)[0].address, 0x3000);
        assert_eq!(res.semaphores, vec![qmd::Release { addr: 0x12_0000_4000, payload: 9, reduction: Some(Reduction::Inc), signed: false, wide: false, timestamp: false }]);
        assert_eq!(res.unsupported.len(), 1);
    }

    #[test]
    fn pools_handles_and_sampler_binding() {
        let regs = HashMap::from([
            (0x1574, 0xff10),
            (0x1578, 0x4440_5000),
            (0x157c, 0xffc0_20ff),
            (0x155c, 0x10),
            (0x1560, 0x4440_2000),
            (0x1564, 0xfff0_017f),
            (0x1234, 1),
        ]);
        let r = |x: u32| regs.get(&x).copied().unwrap_or(0);
        let h = header_pool(&r);
        assert_eq!((h.address, h.maximum_index), (0x10_4440_5000, 0x20ff));
        assert_eq!(h.entry(0x179), Some(0x10_4440_5000 + 0x179 * 32));
        assert_eq!(h.entry(0x2100), None);
        let s = sampler_pool(&r);
        assert_eq!((s.address, s.maximum_index), (0x10_4440_2000, 0x17f));
        assert_eq!(s.entry(0x107), Some(0x10_4440_2000 + 0x107 * 32));
        assert_eq!(s.entry(0x180), None);
        assert_eq!(TexHandle::new(0x1070_0179), TexHandle { header: 0x179, sampler: 0x107 });
        assert_eq!(TexHandle::new(0xffff_ffff), TexHandle { header: 0xf_ffff, sampler: 0xfff });
        assert_eq!(sampler_binding(&r), SamplerBinding::ViaHeaderBinding);
        assert_eq!(sampler_binding(&|_| 0), SamplerBinding::Independently);
    }

    #[test]
    fn root_table_and_unbind_all_are_reported() {
        let (mut res, mut regs) = (Resources::default(), HashMap::new());
        for (m, v) in [(0x0504, 0x100), (0x0508, 5), (0x10f4, 0x100)] {
            feed(&mut res, &mut regs, m, v);
        }
        assert_eq!(res.unsupported.len(), 3);
    }

}
