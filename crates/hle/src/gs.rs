//! The Switch 2 graphics supervisor (what ISelfController 2000 hands out)
//! and its TIPC sessions: syncpoints, the GPU, GPU memory, vi2. No public
//! documentation exists; the replies follow what the SDK's clients read
//! (our analysis of `nn::gs::sdk::BootGraphicsSystem` and its callees,
//! build `a7730d0f`).
//!
//! The fake GPU is complaisant: every fence it hands out is already
//! reached, and every wait it is asked about succeeds at once.

use crate::Kernel;
use crate::ipc::{Reply, Request};
use crate::service::Service;
use crate::thread::Object;

/// State the graphics sessions share (plain data).
#[derive(Debug, Default)]
pub struct State {
    /// gpu 12: the event of each GPU event id (`Kernel::events` indices).
    pub gpu_events: Vec<Option<usize>>,
    /// gsv 204: the syncpoint memory (read-only page for the guest) and its two events.
    pub syncpt_page: Option<u64>,
    pub syncpt_events: Option<(usize, usize)>,
    /// vi2 229: frames presented.
    pub presents: u64,
    /// Syncpoint values (syncpt GetCount says 32).
    pub syncpts: [u32; 32],
    /// syncpt 22: the event of each wait index.
    pub syncpt_waits: Vec<Option<usize>>,
    /// Handles handed out: GPU address spaces (gpu 33), GPU memories (gpu 101),
    /// native memories (gsv 111, their sizes) and window images (gsv 429).
    pub spaces: u32,
    pub gpu_memories: u32,
    pub native: Vec<u64>,
    pub images: u32,
    /// GPU memory calls (101, 102, 211, 212), for a summary line.
    pub memory_calls: [u64; 4],
    /// GPU memories (gpu 101): CPU address, by handle - 1.
    pub gpu_mems: Vec<u64>,
    /// GPU mappings (gpu 211): address space, GPU range (bounds included), CPU address.
    pub gpu_maps: Vec<(u32, u64, u64, u64)>,
}

impl crate::Kernel {
    /// The CPU address of a GPU address in address space `space`; addresses
    /// no mapping covers are taken as CPU addresses (NVN maps the CPU's ASLR
    /// region one to one).
    pub fn gpu_to_cpu(&mut self, space: u32, gpu: u64) -> u64 {
        match self.gs.gpu_maps.iter().rev().find(|m| m.0 == space && (m.1..=m.2).contains(&gpu)) {
            Some(m) => m.3 + (gpu - m.1),
            None => gpu,
        }
    }
}

/// The user partition of every GPU address space (gpu 203), bounds
/// included. The SDK's GPU address allocator (`gpu::sdk::util`, which
/// serves the AS's ALLOC_SPACE in user space) starts with this one free
/// range; it must hold NVN's fixed reservation of [0x8000000, 0x8000000000),
/// the CPU's ASLR region mapped one to one.
const USER_GPU_RANGE: (u64, u64) = (0x800_0000, 0xbf_ffff_ffff);

#[derive(Debug)]
pub enum Gs {
    /// The graphics supervisor session.
    Supervisor,
    Syncpt,
    Gpu,
    GpuMemory,
    /// The display and layer session (gsv 305).
    Vi2,
    /// The chat session (gsv 1202).
    Chat,
}

/// vi2: the mask of free server-side buffer slots handed back at every
/// present; any non-zero value works, three slots allow two buffers held.
const FREE_SLOTS: u64 = 0x7;

/// The syncpoint memory header gsv 204 writes: version 1 and no in-memory
/// syncpoints (n = 0), so every syncpoint read goes through syncpt IPC.
const SYNCPT_HEADER: [u32; 8] = [1, 0, 0, 0x100, 0, 0x40, 0x8, 0x80];

impl Gs {
    pub fn label(&self) -> String {
        match self {
            Gs::Supervisor => "gsv".into(),
            Gs::Syncpt => "gsv syncpt".into(),
            Gs::Gpu => "gsv gpu".into(),
            Gs::GpuMemory => "gsv gpu memory".into(),
            Gs::Vi2 => "vi2".into(),
            Gs::Chat => "gsv chat".into(),
        }
    }

    pub fn request(&mut self, k: &mut Kernel, req: &Request) -> Option<Reply> {
        let gs = |g: Gs| Some(Reply::object(Service::Gs(g)));
        Some(match (self, req.command) {
            // Boot: the process (pseudo-handle), its aruid, flags.
            (Gs::Supervisor, 4) => {
                k.log(&format!("gsv 4: aruid {:#x}", req.u64_at(0).unwrap_or(0)));
                Reply::ok()
            }
            // Client registration and permission checks.
            (Gs::Supervisor, 11 | 101 | 1501 | 1502 | 201 | 1101 | 1102 | 401 | 402) => Reply::ok(),
            // The nvdrv-like session (CMIF).
            (Gs::Supervisor, 1402) => return Some(Reply::object(Service::Nvdrv(crate::nvdrv::Nvdrv::default()))),
            (Gs::Supervisor, 202) => return gs(Gs::Syncpt),
            // Syncpoint memory: two transfer memories (RW, then R); two events back.
            (Gs::Supervisor, 204) => return Some(k.syncpoint_memory(req)),
            // QueryGrantedFeatures: the value is not read by the SDK; Lichii's.
            (Gs::Supervisor, 115) => Reply::with(&2u64.to_le_bytes()),
            (Gs::Supervisor, 497) => return gs(Gs::Gpu),
            // syncpt GetCount (at most 0x3ff), then 21.
            (Gs::Syncpt, 1) => Reply::with(&32u32.to_le_bytes()),
            (Gs::Syncpt, 21) => Reply::ok(),
            // gpu: the range of free event ids (at least two).
            (Gs::Gpu, 11) => Reply::with(&[0u32.to_le_bytes(), 31u32.to_le_bytes()].concat()),
            // The event of an id, which the HLE signals when the GPU "finishes".
            (Gs::Gpu, 12) => {
                let n = req.u32_at(0)? as usize;
                let (r, e) = k.numbered_event()?;
                put(&mut k.gs.gpu_events, n, e);
                Reply { move_handles: vec![r], ..Reply::ok() }
            }
            (Gs::Gpu, 21) => {
                let n = req.u32_at(0)? as usize;
                k.signal_gpu(n);
                Reply::ok()
            }
            (Gs::Gpu, 22) => return gs(Gs::GpuMemory),
            // CreateGraphicsDevice: device 1.
            (Gs::Gpu, 41) => Reply::with(&[1]),
            // Two address spaces, each 33 (a handle), 203 (one user range), 202 (no gsv
            // partition), then gsv 434.
            (Gs::Gpu, 33) => {
                k.gs.spaces += 1;
                Reply::with(&k.gs.spaces.to_le_bytes())
            }
            (Gs::Gpu, 203) => {
                let mut d = [0u8; 8 + 4 * 16];
                d[..8].copy_from_slice(&1u64.to_le_bytes());
                d[8..16].copy_from_slice(&USER_GPU_RANGE.0.to_le_bytes());
                d[16..24].copy_from_slice(&USER_GPU_RANGE.1.to_le_bytes());
                Reply::with(&d)
            }
            (Gs::Gpu, 202) => Reply::with(&[0u8; 16]),
            (Gs::Supervisor, 434) => {
                k.log(&format!("gsv 434: address space {}", req.u32_at(0).unwrap_or(0)));
                Reply::ok()
            }
            // GPU memory: create (a handle), destroy, map, change kind. The client then
            // waits for the event its last word names: signaled before replying. Map and
            // change kind answer the fallback kind (no compression).
            // Unmap: the mapping goes; then its event.
            (Gs::GpuMemory, 213) => {
                let (space, first, last) = (req.u32_at(0)?, req.u64_at(4)?, req.u64_at(0xc)?);
                k.gs.gpu_maps.retain(|m| m.0 != space || m.2 < first || m.1 > last);
                k.signal_gpu(req.u32_at(0x18)? as usize);
                Reply::ok()
            }
            (Gs::GpuMemory, c @ (101 | 102 | 211 | 212)) => {
                k.memory_call(c);
                match c {
                    101 => {
                        k.gs.gpu_memories += 1;
                        k.gs.gpu_mems.push(req.u64_at(0)?);
                        Reply::with(&k.gs.gpu_memories.to_le_bytes())
                    }
                    102 => {
                        k.signal_gpu(req.u32_at(4)? as usize);
                        Reply::ok()
                    }
                    211 => {
                        if k.journal.few("gpu memory 211", 4) {
                            let d = &req.data;
                            k.log(&format!(
                                "gsv gpu memory 211: space {} gpu {:#x}..={:#x} memory {} offset {:#x} perm {} cache {} kind {:#x} second kind {:#x} raw {} device {} sync {}",
                                req.u32_at(0)?, req.u64_at(4)?, req.u64_at(0xc)?, req.u32_at(0x14)?, req.u64_at(0x18)?, d.get(0x20)?, d.get(0x24)?, d.get(0x28)?, d.get(0x2c)?, d.get(0x30)?, d.get(0x34)?, d.get(0x38)?
                            ));
                        }
                        let (space, first, last, mem, offset) = (req.u32_at(0)?, req.u64_at(4)?, req.u64_at(0xc)?, req.u32_at(0x14)?, req.u64_at(0x18)?);
                        if let Some(&base) = k.gs.gpu_mems.get(mem.wrapping_sub(1) as usize) {
                            k.gs.gpu_maps.push((space, first, last, base + offset));
                        }
                        k.signal_gpu(req.u32_at(0x3c)? as usize);
                        Reply::with(&[*req.data.get(0x2c)?])
                    }
                    _ => {
                        k.signal_gpu(req.u32_at(0x20)? as usize);
                        Reply::with(&[*req.data.get(0x18)?])
                    }
                }
            }
            // syncpt: the value of a syncpoint (on the wire, its id inverted), mode 0.
            (Gs::Syncpt, 103) => {
                let v = k.gs.syncpts.get(!req.u32_at(0)? as usize).copied().unwrap_or(0);
                Reply::with(&[&u64::from(v).to_le_bytes()[..], &[0]].concat())
            }
            // The event of a wait index.
            (Gs::Syncpt, 22) => {
                let n = req.u32_at(0)? as usize;
                let (r, e) = k.numbered_event()?;
                put(&mut k.gs.syncpt_waits, n, e);
                if k.journal.few("syncpt 22", 4) {
                    k.log(&format!("gsv syncpt 22: wait index {n} -> handle {r:#x} (event {e})"));
                }
                Reply { move_handles: vec![r], ..Reply::ok() }
            }
            // Arm a wait: the fake GPU has always got there (the syncpoint is raised to the
            // threshold, 32-bit wrapping compare), so the event is signaled at once.
            (Gs::Syncpt, 201) => {
                let (n, id, threshold) = (req.u32_at(0)? as usize, !req.u32_at(4)? as usize, req.u64_at(8)? as u32);
                if k.journal.few("syncpt 201", 4) {
                    let armed = k.gs.syncpt_waits.get(n).copied().flatten().is_some();
                    let now = k.gs.syncpts.get(id).copied();
                    k.log(&format!("gsv syncpt 201: wait index {n} syncpoint {id} threshold {threshold} mode {} value {now:?} event {armed}", req.data.get(0x10).copied().unwrap_or(0)));
                }
                if let Some(s) = k.gs.syncpts.get_mut(id)
                    && (threshold.wrapping_sub(*s) as i32) > 0
                {
                    *s = threshold;
                }
                k.signal_wait(n);
                Reply::ok()
            }
            // Cancel a wait: the client then waits for the event without a timeout.
            (Gs::Syncpt, 202) => {
                k.signal_wait(req.u32_at(0)? as usize);
                Reply::ok()
            }
            // Chat (Lichii's answers): register (1201) and unregister (1203), the session
            // (1202); in it, an event id (1), its event (11), and 103 answered empty.
            (Gs::Supervisor, 1201 | 1203) => Reply::ok(),
            (Gs::Supervisor, 1202) => return gs(Gs::Chat),
            (Gs::Chat, 1) => Reply::with(&0u32.to_le_bytes()),
            (Gs::Chat, 11) => Reply { move_handles: vec![k.lone_event()?], ..Reply::ok() },
            (Gs::Chat, 103) => Reply::ok(),
            // Window images: the size of an image (1510, 435), native memory (111, 423),
            // the image (429), then 315.
            (Gs::Supervisor, 1510) => Reply::with(&[image_size(req).to_le_bytes(), 0x1000u64.to_le_bytes()].concat()),
            (Gs::Supervisor, 435) => Reply::with(&image_size(req).to_le_bytes()),
            (Gs::Supervisor, 111) => {
                k.gs.native.push(req.u64_at(4)?);
                Reply::with(&(k.gs.native.len() as u32).to_le_bytes())
            }
            (Gs::Supervisor, 423 | 315) => Reply::ok(),
            (Gs::Supervisor, 429) => {
                k.gs.images += 1;
                let native = req.u32_at(0x44)?;
                let size = k.gs.native.get(native.wrapping_sub(1) as usize).copied().unwrap_or(0);
                k.log(&format!("gsv 429: image {} (native memory {native}, {size:#x} bytes)", k.gs.images));
                Reply::with(&k.gs.images.to_le_bytes())
            }
            // vi2: open (301), then the session (305).
            (Gs::Supervisor, 301) => Reply::ok(),
            (Gs::Supervisor, 305) => return gs(Gs::Vi2),
            // Bounds the client checks: event id space (1), native events (3, at most 7),
            // timestamp recorders (5, at most 7).
            (Gs::Vi2, 1 | 3) => Reply::with(&1u32.to_le_bytes()),
            (Gs::Vi2, 5) => Reply::with(&7u32.to_le_bytes()),
            // The notification hub (11) and the native events (31): never signaled.
            (Gs::Vi2, 11 | 31) => Reply { move_handles: vec![k.lone_event()?], ..Reply::ok() },
            // Signaled notification ids (21): none.
            (Gs::Vi2, 21) => {
                if let Some(b) = req.out_buffer(0) {
                    let _ = k.mem.write(b.addr, &vec![0; b.size.min(0x80) as usize]);
                }
                Reply::ok()
            }
            // Display, layer and event bindings, recorders, checkpoints.
            (Gs::Vi2, 13 | 51 | 101 | 111 | 201 | 251 | 263 | 265 | 267 | 269 | 351) => Reply::ok(),
            // Refresh rate: 60 Hz (1); dynamic range: SDR (1) and a u16; paper white (nits);
            // maximum buffer resolution: 1080p (2).
            (Gs::Vi2, 103) => Reply::with(&1i64.to_le_bytes()),
            (Gs::Vi2, 104) => Reply::with(&[&1u64.to_le_bytes()[..], &[0, 0]].concat()),
            (Gs::Vi2, 105) => Reply::with(&100u16.to_le_bytes()),
            (Gs::Vi2, 106) => Reply::with(&2u64.to_le_bytes()),
            // Free slots of a layer.
            (Gs::Vi2, 212) => Reply::with(&FREE_SLOTS.to_le_bytes()),
            // Timestamps: none recorded, nothing pending.
            (Gs::Vi2, 53) => Reply::with(&[0u8; 12]),
            // Present (229) or present without a buffer (228): the free slots, then a
            // release fence that is already reached ({0, 0, 0}).
            (Gs::Vi2, c @ (228 | 229)) => {
                if c == 229 {
                    k.present(req);
                }
                Reply::with(&[&FREE_SLOTS.to_le_bytes()[..], &[0u8; 16]].concat())
            }
            _ => return None,
        })
    }
}

/// Stores event `e` at index `n` of a numbered table.
fn put(table: &mut Vec<Option<usize>>, n: usize, e: usize) {
    if table.len() <= n {
        table.resize(n + 1, None);
    }
    table[n] = Some(e);
}

/// gsv 1510 and 435: the size of a window image from the image preferences
/// (`+8`: `(w - 1) | (h - 1) << 16`): 4 bytes per pixel, the pitch aligned
/// on a 64-byte GOB, the height on 128 rows, rounded to a page. Our guess;
/// it gives Lichii's 0x1fe0000 for 3840 x 2160.
fn image_size(req: &Request) -> u64 {
    let d = u64::from(req.u32_at(8).unwrap_or(0));
    let (w, h) = ((d & 0xffff) + 1, (d >> 16) + 1);
    ((w * 4).next_multiple_of(64) * h.next_multiple_of(128)).next_multiple_of(0x1000)
}

impl Kernel {
    /// A new event and its readable handle (the writable one is dropped: the HLE signals it).
    fn numbered_event(&mut self) -> Option<(u32, usize)> {
        let (w, r) = self.create_event().ok()?;
        self.handles.close(w);
        Some((r, self.event_of(r)?))
    }

    /// syncpt: signals the event of wait index `n`.
    fn signal_wait(&mut self, n: usize) {
        if let Some(Some(e)) = self.gs.syncpt_waits.get(n).copied() {
            self.signal_event(e);
        }
    }

    /// Counts the GPU memory calls; a summary line every 256.
    fn memory_call(&mut self, c: u32) {
        let i = [101, 102, 211, 212].iter().position(|&x| x == c).unwrap_or(0);
        self.gs.memory_calls[i] += 1;
        let m = self.gs.memory_calls;
        if m.iter().sum::<u64>().is_multiple_of(256) {
            self.log(&format!("gsv gpu memory: {} calls (101 {}, 102 {}, 211 {}, 212 {})", m.iter().sum::<u64>(), m[0], m[1], m[2], m[3]));
        }
    }

    /// A new event nothing will signal; returns its readable handle.
    pub fn lone_event(&mut self) -> Option<u32> {
        let (w, r) = self.create_event().ok()?;
        self.handles.close(w);
        Some(r)
    }

    /// vi2 229: one more frame. The first three and every 60th are journaled.
    fn present(&mut self, req: &Request) {
        self.gs.presents += 1;
        let n = self.gs.presents;
        if n <= 3 || n.is_multiple_of(60) {
            let (layer, image) = (req.u64_at(0).unwrap_or(0), req.u32_at(0xc).unwrap_or(0));
            let (fence_id, fence) = (req.u32_at(0x50).unwrap_or(0), req.u64_at(0x58).unwrap_or(0));
            let interval = req.u32_at(0x10 + 0x18).unwrap_or(0) >> 16;
            self.log(&format!("vi2: present #{n} layer {layer:#x} image {image} acquire fence ({fence_id}, {fence}) interval {interval}"));
        }
        if self.config.frames.is_some_and(|f| n >= f) {
            self.stop(0, &format!("{n} frames presented"));
        }
    }

    /// Signals the event of GPU event id `n`, as if the GPU had finished.
    pub fn signal_gpu(&mut self, n: usize) {
        if let Some(Some(e)) = self.gs.gpu_events.get(n).copied() {
            self.signal_event(e);
            if self.journal.first(&format!("gpu event {n}")) {
                self.log(&format!("gsv gpu: event {n} signaled"));
            }
        }
    }

    /// The address and size of the transfer memory a handle names.
    pub fn transfer_range(&self, h: u32) -> Option<(u64, u64)> {
        let Object::TransferMemory(t) = self.handles.get(h)? else { return None };
        Some((self.transfers[t].addr, self.transfers[t].size))
    }

    /// gsv 204: keeps both transfer memories, writes the syncpoint header
    /// into the read-only one, and answers a writable request event and a
    /// readable acknowledge event.
    fn syncpoint_memory(&mut self, req: &Request) -> Reply {
        let (Some(&rw), Some(&r)) = (req.copy_handles.first(), req.copy_handles.get(1)) else {
            return Reply::error(crate::result::INVALID_HANDLE);
        };
        let (Some(a), Some(b)) = (self.transfer_range(rw), self.transfer_range(r)) else {
            return Reply::error(crate::result::INVALID_HANDLE);
        };
        self.retain_transfer(rw);
        self.retain_transfer(r);
        let mut page = vec![0u8; 0x200];
        for (i, v) in SYNCPT_HEADER.iter().enumerate() {
            page[4 * i..4 * i + 4].copy_from_slice(&v.to_le_bytes());
        }
        if self.mem.write_object(b.0, &page).is_err() {
            return Reply::error(crate::result::INVALID_CURRENT_MEMORY);
        }
        let (Ok((req_w, _req_r)), Ok((_ack_w, ack_r))) = (self.create_event(), self.create_event()) else {
            return Reply::error(crate::result::OUT_OF_HANDLES);
        };
        self.gs.syncpt_page = Some(b.0);
        self.gs.syncpt_events = self.event_of(_req_r).zip(self.event_of(ack_r));
        self.log(&format!("gsv 204: syncpoint memory rw {:#x}+{:#x} r {:#x}+{:#x}, header version 1 n 0", a.0, a.1, b.0, b.1));
        Reply { move_handles: vec![req_w, ack_r], ..Reply::ok() }
    }
}
