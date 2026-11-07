//! The GPU's command processor: it decodes a channel's pushbuffer segments
//! (NVIDIA's public method header format), keeps which class each
//! subchannel holds and the last value of every method, runs the 3D class's
//! macros (MME64), and reports what the CPU side waits for: semaphore
//! releases, from the host (SEMAPHOREA..D, SEM_ADDR..SEM_EXECUTE), 3D and
//! compute (SET_REPORT_SEMAPHORE_A..D) and the copy engine (LAUNCH_DMA).
//! Addresses are GPU addresses; the caller translates them.
//!
//! ponytail: semaphores only; drawing, copies and inline writes come with
//! the renderer (M3).

use crate::mme::{self, Mme};

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

/// What a segment did besides storing methods.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Effect {
    Release(Release),
    /// A method the fake does not execute but that matters (a copy, an
    /// acquire, a reduction): its class, method byte offset and value.
    Ignored { class: u32, method: u32, value: u32 },
    /// A macro stopped early.
    MacroFault { index: usize, fault: mme::Fault },
    /// CLEAR_SURFACE.
    Clear(Clear),
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
    /// Runs one pushbuffer segment.
    pub fn run(&mut self, words: &[u32]) -> Vec<Effect> {
        let mut effects = Vec::new();
        let mut i = 0;
        while i < words.len() {
            let Some(mut p) = self.pending.take() else {
                let h = words[i];
                i += 1;
                let (op, count, subchannel, method) = (h >> 29, h >> 16 & 0x1fff, (h >> 13 & 7) as usize, h & 0xfff);
                match op {
                    // Incrementing, non-incrementing, increment once: data words follow.
                    1 | 3 | 5 if count > 0 => self.pending = Some(Pending { op, subchannel, method, left: count }),
                    // Immediate: the value is in the header.
                    4 => self.write(subchannel, method * 4, count, &mut effects),
                    // End of the segment.
                    7 => break,
                    // Sub-device masks and empty methods carry no data.
                    _ => {}
                }
                continue;
            };
            self.write(p.subchannel, p.method * 4, words[i], &mut effects);
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
        // A call whose parameters do not continue in the next segment runs now.
        if self.pending.is_none() {
            self.flush_call(&mut effects);
        }
        effects
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
        if three_d && CALL_MME.contains(&method) {
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
        if three_d && let Some(unit) = self.mme.as_mut() {
            match method {
                LOAD_MME_INSTRUCTION_RAM_POINTER => unit.set_code_pointer(value),
                LOAD_MME_INSTRUCTION_RAM => unit.load_code(value),
                LOAD_MME_START_ADDRESS_RAM_POINTER => unit.set_start_pointer(value),
                LOAD_MME_START_ADDRESS_RAM => unit.load_start(value),
                _ => {}
            }
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
    fn clear(&self, sub: usize, value: u32) -> Clear {
        let reg = |m: u32| self.reg(sub, m);
        let mask = (value >> 2 & 0xf) as u8;
        let color = (mask != 0).then(|| {
            let mrt = value >> 6 & 0xf;
            let j = reg(CT_SELECT) >> (4 + 3 * mrt) & 7;
            let t = COLOR_TARGET + 64 * j;
            let surface = Surface { addr: u64::from(reg(t)) << 32 | u64::from(reg(t + 4)), width: reg(t + 8), height: reg(t + 0xc), format: reg(t + 0x10) & 0xff, layers: (reg(t + 0x18) & 0xfff_ffff).max(1) };
            (surface, [0, 1, 2, 3].map(|i| reg(COLOR_CLEAR_VALUE + 4 * i)), mask)
        });
        let (z, s) = (value & 1 != 0, value & 2 != 0);
        let depth = ((z || s) && reg(ZT_SELECT) & 1 != 0).then(|| {
            let surface = Surface { addr: u64::from(reg(ZT_A)) << 32 | u64::from(reg(ZT_A + 4)), width: reg(ZT_SIZE_A), height: reg(ZT_SIZE_A + 4), format: reg(ZT_A + 8) & 0x1f, layers: (reg(ZT_SIZE_A + 8) & 0xffff).max(1) };
            (surface, z.then(|| reg(Z_CLEAR_VALUE)), s.then(|| reg(STENCIL_CLEAR_VALUE) as u8))
        });
        // Scissor 0 bounds the clear when SET_CLEAR_SURFACE_CONTROL says so and it is on.
        let scissor = (reg(CLEAR_SURFACE_CONTROL) & 0x100 != 0 && reg(SCISSOR_ENABLE) & 1 != 0).then(|| {
            let (h, v) = (reg(SCISSOR_ENABLE + 4), reg(SCISSOR_ENABLE + 8));
            let (x0, x1, y0, y1) = (h & 0xffff, h >> 16, v & 0xffff, v >> 16);
            [x0, y0, x1.saturating_sub(x0), y1.saturating_sub(y0)]
        });
        Clear { layer: value >> 10 & 0xffff, color, depth, scissor }
    }

    fn engine_effect(&self, subchannel: usize, method: u32, value: u32) -> Option<Effect> {
        let class = self.classes[subchannel];
        let ignored = Effect::Ignored { class, method, value };
        match (class & 0xff, method) {
            // 3D and compute: operation in bits 1:0 (0 = release); bit 28: one word, else four.
            (0x97 | 0xc0, REPORT_SEMAPHORE_D) => Some(match value & 3 {
                0 => Effect::Release(Release {
                    method,
                    addr: u64::from(self.reg(subchannel, REPORT_SEMAPHORE_A) & 0xff) << 32 | u64::from(self.reg(subchannel, REPORT_SEMAPHORE_B)),
                    value: u64::from(self.reg(subchannel, REPORT_SEMAPHORE_C)),
                    width: 4,
                    timestamp: value >> 28 & 1 == 0,
                }),
                _ => ignored,
            }),
            (0x97, CLEAR_SURFACE) => Some(Effect::Clear(self.clear(subchannel, value))),
            // Copy engine: a copy (ignored), then a semaphore by bits 4:3 (1 one word, 2 four).
            (0xb5, DMA_LAUNCH) => Some(match value >> 3 & 3 {
                kind @ (1 | 2) => Effect::Release(Release {
                    method,
                    addr: u64::from(self.reg(subchannel, DMA_SEMAPHORE_A) & 0x1ff_ffff) << 32 | u64::from(self.reg(subchannel, DMA_SEMAPHORE_B)),
                    value: u64::from(self.reg(subchannel, DMA_SEMAPHORE_PAYLOAD)),
                    width: 4,
                    timestamp: kind == 2,
                }),
                _ => ignored,
            }),
            _ => None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

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
        // An acquire is reported as ignored, not executed.
        assert!(matches!(c.run(&[header(1, 1, 0, SEMAPHORE_D), 1])[..], [Effect::Ignored { method: SEMAPHORE_D, value: 1, .. }]));
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
    fn replay_mode_takes_values_from_the_shadow() {
        let mut c = Channel::default();
        c.run(&[header(1, 1, 0, SET_OBJECT), 0xc797, header(1, 1, 0, REPORT_SEMAPHORE_C), 5]);
        c.run(&[header(1, 1, 0, SET_MME_SHADOW_RAM_CONTROL), 3, header(1, 1, 0, REPORT_SEMAPHORE_C), 9]);
        assert_eq!(c.reg(0, REPORT_SEMAPHORE_C), 5);
        c.run(&[header(1, 1, 0, SET_MME_SHADOW_RAM_CONTROL), 2, header(1, 1, 0, REPORT_SEMAPHORE_C), 9]);
        assert_eq!((c.reg(0, REPORT_SEMAPHORE_C), c.shadow[(REPORT_SEMAPHORE_C / 4) as usize]), (9, 5));
    }
}
