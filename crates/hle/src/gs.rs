//! The Switch 2 graphics supervisor (what ISelfController 2000 hands out)
//! and its TIPC sessions: syncpoints, the GPU, GPU memory, vi2. No public
//! documentation exists; the replies follow what the SDK's clients read
//! (our analysis of `nn::gs::sdk::BootGraphicsSystem` and its callees,
//! build `a7730d0f`).
//!
//! The fake GPU is complaisant: every fence it hands out is already
//! reached, and every wait it is asked about succeeds at once.

use crate::ipc::{Reply, Request};
use crate::nvdrv::{Decoder, GpuSide};
use crate::service::Service;
use crate::thread::Object;
use crate::{Kernel, Stop};
use gyroship_runtime::mem::Memory;
use std::sync::Arc;
use std::sync::atomic::{AtomicU32, Ordering};

/// A frame's shader work (modules and pipelines, host time) from which it
/// is journaled: the frame's whole budget at 20 frames a second.
const SLOW_COMPILE: std::time::Duration = std::time::Duration::from_millis(50);

/// How long a pipeline may compile on a worker thread before it is reported.
const OVERDUE: std::time::Duration = std::time::Duration::from_secs(5);

/// State the graphics sessions share (plain data).
#[derive(Debug, Default)]
pub struct State {
    /// gpu 12: the event of each GPU event id (`Kernel::events` indices).
    pub gpu_events: Vec<Option<usize>>,
    /// gsv 204: the syncpoint memory (read-only page for the guest) and its two events.
    pub syncpt_page: Option<u64>,
    pub syncpt_events: Option<(usize, usize)>,
    /// vi2 229: frames presented, the vsync the last one waited for, and the
    /// guest thread that presented it (its waits go to `--trace`).
    pub presents: u64,
    pub vsync: u64,
    pub presenter: u64,
    /// Syncpoint values (syncpt GetCount says 32), shared with the GPU side.
    pub syncpts: Arc<[AtomicU32; 32]>,
    /// syncpt 22: the event of each wait index.
    pub syncpt_waits: Vec<Option<usize>>,
    /// vi2 31: the native event of each index, which the change bindings signal.
    pub vi2_events: std::collections::BTreeMap<u32, usize>,
    /// vi2 229: the area the game says it drew, (height - 1) << 16 | (width - 1).
    pub drawn_area: u32,
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
    /// GPU mappings (gpu 211). GPU work takes them as they stand when it is
    /// handed over (`Phase`); a change copies them only while it holds them.
    pub maps: Arc<Mappings>,
}

#[derive(Debug, Default, Clone)]
pub struct Mappings {
    /// Address space, GPU range (bounds included), CPU address.
    pub list: Vec<(u32, u64, u64, u64)>,
    /// `list` flattened for lookups (derived: rebuilt from it after a load).
    pub table: Table,
}

/// Disjoint GPU ranges by address space, sorted: (space, first) -> (last, CPU
/// address of first). A later mapping wins where it overlaps an earlier one.
pub type Table = std::collections::BTreeMap<(u32, u64), (u64, u64)>;

/// Paints mapping `space`, `first..=last` -> `cpu` over `table`: the ranges it
/// overlaps keep only their parts outside it.
fn paint(table: &mut Table, space: u32, first: u64, last: u64, cpu: u64) {
    let mut hit = Vec::new();
    if let Some((&(s, f), &(l, c))) = table.range(..(space, first)).next_back()
        && s == space
        && l >= first
    {
        hit.push((f, l, c));
    }
    hit.extend(table.range((space, first)..=(space, last)).map(|(&(_, f), &(l, c))| (f, l, c)));
    for (f, l, c) in hit {
        table.remove(&(space, f));
        if f < first {
            table.insert((space, f), (first - 1, c));
        }
        if l > last {
            table.insert((space, last + 1), (l, c + (last + 1 - f)));
        }
    }
    table.insert((space, first), (last, cpu));
}

/// The table of `maps`, painted in order.
fn flatten(maps: &[(u32, u64, u64, u64)]) -> Table {
    let mut table = Table::new();
    for &(space, first, last, cpu) in maps {
        paint(&mut table, space, first, last, cpu);
    }
    table
}

/// The CPU address of `gpu` in `space` through `table`; itself when no range has it.
fn lookup(table: &Table, space: u32, gpu: u64) -> u64 {
    span(table, space, gpu).0
}

/// The CPU address of `gpu` in `space` and how many bytes from it stay in one
/// range of `table`: to its end, or, outside every range, to the next one's start.
fn span(table: &Table, space: u32, gpu: u64) -> (u64, u64) {
    match table.range(..=(space, gpu)).next_back() {
        Some((&(s, first), &(last, cpu))) if s == space && gpu <= last => (cpu + (gpu - first), (last - gpu).saturating_add(1)),
        _ => {
            let next = table.range((space, gpu)..).next().filter(|((s, _), _)| *s == space).map_or(u64::MAX, |((_, f), _)| *f);
            (gpu, next - gpu)
        }
    }
}

/// The CPU address of a GPU address in address space `space`, through the
/// mapping of `maps` made last that covers it; addresses no mapping covers
/// are taken as CPU addresses (NVN maps the CPU's ASLR region one to one).
pub fn gpu_to_cpu(maps: &Mappings, space: u32, gpu: u64) -> u64 {
    lookup(&maps.table, space, gpu)
}

/// Reads guest memory at a GPU address for the GPU, through `maps`; false if
/// a byte is not readable. A naturally aligned 4- or 8-byte read (the
/// semaphore an acquire polls while guest threads release it) is one Acquire
/// load. Other reads go through the mapping that covers each part.
pub fn read_gpu(mem: &Memory, maps: &Mappings, space: u32, mut gpu: u64, mut out: &mut [u8]) -> bool {
    let (mut at, mut room) = span(&maps.table, space, gpu);
    let word = match out.len() {
        4 if at.is_multiple_of(4) => mem.load_u32(at).map(u64::from),
        8 if at.is_multiple_of(8) => mem.load_u64(at),
        _ => loop {
            let n = out.len().min(usize::try_from(room).unwrap_or(usize::MAX));
            let (part, rest) = std::mem::take(&mut out).split_at_mut(n);
            if mem.read(at, part).is_err() {
                return false;
            }
            if rest.is_empty() {
                return true;
            }
            (gpu, out) = (gpu + n as u64, rest);
            (at, room) = span(&maps.table, space, gpu);
        },
    };
    word.map(|v| out.copy_from_slice(&v.to_le_bytes()[..out.len()])).is_ok()
}

impl GpuSide {
    /// [`gpu_to_cpu`] through the mappings of the work's hand-over.
    pub fn gpu_to_cpu(&self, space: u32, gpu: u64) -> u64 {
        gpu_to_cpu(&self.at.maps, space, gpu)
    }

    /// Writes `bytes` at GPU address `gpu` of `space`, each part through the
    /// mapping that covers it (the GPU writes whatever the CPU's permissions);
    /// Err with the CPU address of a part that is not writable.
    pub fn write_gpu(&mut self, space: u32, mut gpu: u64, mut bytes: &[u8]) -> Result<(), u64> {
        while !bytes.is_empty() {
            let (at, room) = span(&self.at.maps.table, space, gpu);
            let n = bytes.len().min(usize::try_from(room).unwrap_or(usize::MAX));
            self.mem.write_object(at, &bytes[..n]).map_err(|_| at)?;
            (gpu, bytes) = (gpu + n as u64, &bytes[n..]);
        }
        Ok(())
    }

    /// [`read_gpu`] through the mappings of the work's hand-over.
    pub fn read_gpu(&mut self, space: u32, gpu: u64, out: &mut [u8]) -> bool {
        read_gpu(&self.mem, &self.at.maps, space, gpu, out)
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
    /// The camera session (gsv 602, CMIF).
    Camera,
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
            Gs::Camera => "gsv camera".into(),
        }
    }

    pub fn request(&mut self, k: &mut Kernel, req: &Request) -> Option<Reply> {
        k.census_ipc(|| format!("{} {}", self.label(), req.command));
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
                let maps = Arc::make_mut(&mut k.gs.maps);
                maps.list.retain(|m| m.0 != space || m.2 < first || m.1 > last);
                maps.table = flatten(&maps.list);
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
                            let maps = Arc::make_mut(&mut k.gs.maps);
                            maps.list.push((space, first, last, base + offset));
                            paint(&mut maps.table, space, first, last, base + offset);
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
                let v = k.gs.syncpts.get(!req.u32_at(0)? as usize).map_or(0, |s| s.load(Ordering::Acquire));
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
                    let now = k.gs.syncpts.get(id).map(|s| s.load(Ordering::Acquire));
                    k.log(&format!("gsv syncpt 201: wait index {n} syncpoint {id} threshold {threshold} mode {} value {now:?} event {armed}", req.data.get(0x10).copied().unwrap_or(0)));
                }
                if k.gs.syncpts.get(id).is_some_and(|s| s.try_update(Ordering::AcqRel, Ordering::Acquire, |v| ((threshold.wrapping_sub(v) as i32) > 0).then_some(threshold)).is_ok()) {
                    k.count("syncpoint waits raised");
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
            // Camera (Lichii's answers: success, no data; the game goes on without one):
            // initialize (601), the session (602), finalize (603).
            (Gs::Supervisor, 601 | 603) => Reply::ok(),
            (Gs::Supervisor, 602) => return gs(Gs::Camera),
            (Gs::Camera, 3 | 4 | 21 | 22 | 24 | 25 | 28 | 29) => Reply::ok(),
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
            // The notification hub (11): never signaled. A native event (31), kept by its
            // index for the change bindings below.
            (Gs::Vi2, 11) => Reply { move_handles: vec![k.lone_event()?], ..Reply::ok() },
            (Gs::Vi2, 31) => {
                let n = req.u32_at(0).unwrap_or(0);
                let (r, e) = k.numbered_event()?;
                k.gs.vi2_events.insert(n, e);
                Reply { move_handles: vec![r], ..Reply::ok() }
            }
            // Change bindings (refresh rate, dynamic range, paper white, maximum resolution,
            // mode capability): the bound native event (index at +8) is signaled once, as a
            // console does when it binds, so the game queries the current values.
            (Gs::Vi2, 263 | 265 | 267 | 269 | 271) => {
                if let Some(&e) = req.u32_at(8).and_then(|n| k.gs.vi2_events.get(&n)) {
                    k.signal_event(e);
                }
                Reply::ok()
            }
            // Signaled notification ids (21): none.
            (Gs::Vi2, 21) => {
                if let Some(b) = req.out_buffer(0) {
                    let _ = k.mem.write(b.addr, &vec![0; b.size.min(0x80) as usize]);
                }
                Reply::ok()
            }
            // Display, layer and event bindings, recorders, checkpoints.
            (Gs::Vi2, 13 | 51 | 101 | 111 | 201 | 251 | 351) => Reply::ok(),
            // Refresh rate: 60 Hz ({60, 0}: the SDK accepts -1, 60 or 120); dynamic range:
            // SDR (1) and a 400-nit peak; paper white 203 nits; maximum buffer
            // resolution: 1080p (2). Values as a reference emulator answers them.
            (Gs::Vi2, 103) => Reply::with(&[60u32.to_le_bytes(), 0u32.to_le_bytes()].concat()),
            (Gs::Vi2, 104) => Reply::with(&[&1u64.to_le_bytes()[..], &400u16.to_le_bytes()].concat()),
            (Gs::Vi2, 105) => Reply::with(&203u16.to_le_bytes()),
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

/// Virtual ticks per second (31.25 MHz).
const TICKS_PER_SECOND: u64 = 31_250_000;

/// The k-th vsync of a 60 Hz display, in ticks.
fn vsync(k: u64) -> u64 {
    k * TICKS_PER_SECOND / 60
}

/// The vsync a present is shown at, after the last one `last`, with swap
/// `interval`, `first` being the first vsync from now; and whether the
/// presenting thread waits for it. A frame later than its vsync waits for the
/// next one (an even cadence), unless `late` (`--late-frames`, play mode):
/// then it is shown at once, as the console's spare window images allow (the
/// game has three): 24 frames a second in the tutorial instead of 16 (runs
/// 53n, 53v), but an uneven cadence, which looked slower and choppier in a
/// cutscene (D48). An early frame always waits.
fn next_vsync(last: u64, interval: u64, first: u64, late: bool) -> (u64, bool) {
    let due = last + interval;
    if late && first > due { (first - 1, false) } else { (due.max(first), true) }
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
        self.gs.presenter = self.sched.current.map_or(0, |t| self.sched.threads[t].id);
        let n = self.gs.presents;
        // In --trace, the presents bracket the presenting thread's waits and probe passes.
        if self.trace.is_some() {
            let stamp = (self.now(), self.gs.presenter, n);
            if let Some(t) = self.trace.as_mut() {
                t.note(stamp, "present");
            }
        }
        if let Some((path, from)) = &self.config.profile
            && n == (*from).max(1)
        {
            gyroship_runtime::profile::start(path.clone());
        }
        if self.config.perf.is_some() {
            gyroship_runtime::profile::mark_frame();
            #[cfg(windows)]
            if self.config.play && n.is_multiple_of(60) {
                use std::os::windows::io::AsRawHandle;
                for t in &self.sched.threads {
                    if let Some(cpu) = t.host_thread.as_ref().and_then(|h| gyroship_runtime::profile::cpu_time(h.as_raw_handle() as usize)) {
                        self.census.threads_cpu += &format!("{n},{},{:.1}\n", t.id, cpu.as_secs_f64() * 1e3);
                    }
                }
            }
            if self.config.play && n.is_multiple_of(60) {
                let c = &mut self.census;
                for (&(id, kind), (parked, _)) in &c.parked {
                    c.waits_csv += &format!("{n},{id},{kind},{:.1}\n", parked.as_secs_f64() * 1e3);
                }
            }
        }
        if n <= 3 || n.is_multiple_of(60) {
            let (layer, image) = (req.u64_at(0).unwrap_or(0), req.u32_at(0xc).unwrap_or(0));
            let (fence_id, fence) = (req.u32_at(0x50).unwrap_or(0), req.u64_at(0x58).unwrap_or(0));
            let interval = req.u32_at(0x10 + 0x18).unwrap_or(0) >> 16;
            self.log(&format!("vi2: present #{n} layer {layer:#x} image {image} acquire fence ({fence_id}, {fence}) interval {interval}"));
        }
        // The drawn area (present word 7): logged when it changes.
        let area = req.u32_at(0x14).unwrap_or(0);
        if area != self.gs.drawn_area {
            self.gs.drawn_area = area;
            self.log(&format!("vi2: present #{n}: drawn area {}x{}", (area & 0xffff) + 1, (area >> 16) + 1));
        }
        // The frame's acquire fence (syncpoint id inverted, value) and swap interval.
        let fence = (req.u32_at(0x50).unwrap_or(0), req.u64_at(0x58).unwrap_or(0) as u32);
        let interval = u64::from((req.u32_at(0x10 + 0x18).unwrap_or(0) >> 16).max(1));
        self.gpu_work(crate::nvdrv::GpuJob::Present { n, fence, area, interval });
    }

    /// The rest of present `n`, once its GPU part ran (`GpuSide::present`):
    /// the frame limit, the summary line, and the display's cadence, from
    /// kernel time `at`, when the present was handed over.
    pub(crate) fn present_end(&mut self, n: u64, interval: u64, at: u64) {
        if self.config.frames.is_some_and(|f| n >= f) {
            self.stop(0, &format!("{n} frames presented"));
        }
        if n.is_multiple_of(600) {
            let summary = self.summary();
            self.log(&summary);
        }
        // The display's cadence (as a console's compositor, or Lichii's pace to
        // vblank): the presenting thread resumes at the next vsync its swap
        // interval allows, on a 60 Hz grid computed from k (no drift).
        if self.config.vsync {
            let first = (at * 60).div_ceil(TICKS_PER_SECOND);
            let (next, wait) = next_vsync(self.gs.vsync, interval, first, self.config.play && self.config.late_frames);
            self.gs.vsync = next;
            if wait {
                self.block(crate::thread::Wait::Sleep, Some(vsync(next)), crate::result::SUCCESS);
            }
        }
    }
}

impl GpuSide {
    /// Present `n`'s GPU part: the channels waiting for the CPU go on, the
    /// acquire fence (syncpoint id inverted, value) is checked, the host GPU's
    /// refusals and shader work are reported, and the image of drawn `area`
    /// is hashed, shown and dumped.
    pub(crate) fn present(&mut self, d: &mut Decoder, n: u64, (fence_id, fence): (u32, u32), area: u32) -> Result<(), Stop> {
        // The GPU work submitted from here to the end of this present is this frame's.
        if let Some(s) = self.surfaces.as_mut() {
            s.present = n;
        }
        // GPU work that waited for the CPU's semaphores runs before the frame is shown.
        crate::nvdrv::poll(d, self)?;
        // The frame's acquire fence must be reached by now.
        if let Some(now) = self.syncpts.get(!fence_id as usize).map(|s| s.load(Ordering::Acquire))
            && (fence.wrapping_sub(now) as i32) > 0
        {
            self.anomaly("present fence not reached", &format!("present #{n}: syncpoint {} at {now}, fence {fence}", !fence_id))?;
        }
        // The host GPU's refusals since the last present.
        // ponytail: checked once per frame; per operation if a stop needs the guest stack.
        if let Some(s) = self.surfaces.as_mut() {
            s.flush();
            if s.submit_ended {
                return Err((crate::STOP_EXIT, "the GPU submit thread ended: frames were lost".into()));
            }
            s.next_frame();
            let skipped = std::mem::take(&mut s.skipped);
            if self.config.perf.is_some() {
                let cpu = gyroship_runtime::profile::own_cpu_time().unwrap_or_default().as_secs_f64() * 1e3;
                #[cfg(windows)]
                let submit_cpu = s.submit_thread().and_then(|h| {
                    use std::os::windows::io::AsRawHandle;
                    gyroship_runtime::profile::cpu_time(h.as_raw_handle() as usize)
                });
                #[cfg(not(windows))]
                let submit_cpu: Option<std::time::Duration> = None;
                let (st, calls) = (&s.staging, d.mme_stats().calls);
                let row = format!(
                    "{n},{cpu:.1},{},{},{},{},{},{},{:.1},{calls},{},{:.1},{:.1},{:.1},{:.1},{:.1}\n",
                    s.drawn,
                    skipped.values().sum::<u64>(),
                    st.views[0],
                    st.bind_groups[0],
                    st.flushed,
                    st.uploads[0],
                    st.uploads[1] as f64 / 1e6,
                    self.watched,
                    st.flush_ns as f64 / 1e6,
                    st.submit_ns as f64 / 1e6,
                    d.decode_ns as f64 / 1e6,
                    self.effects_ns as f64 / 1e6,
                    submit_cpu.unwrap_or_default().as_secs_f64() * 1e3
                );
                self.frames += &row;
            }
            let errors = s.errors.lock().map(|mut e| std::mem::take(&mut *e)).unwrap_or_default();
            // Shader work that held this frame back: host time, so in the play
            // journal only (the deterministic one stays byte-identical).
            let c = std::mem::take(&mut s.compile);
            s.compile_total += c.clone();
            let overdue = if self.config.play { s.overdue(OVERDUE) } else { Vec::new() };
            // Pipelines the worker threads finished since (play mode), and the cache's write.
            let finished = s.finished.lock().map(|mut f| std::mem::take(&mut *f)).unwrap_or_default();
            s.compile_total.background += finished.len() as u32;
            s.compile_total.background_time += finished.iter().map(|f| f.1).sum::<std::time::Duration>();
            let _ = s.save_pipeline_cache(false);
            if !finished.is_empty() {
                let made: Vec<String> = finished.iter().map(|(label, t)| format!("{label} {} ms", t.as_millis())).collect();
                self.log(&format!("gpu: present #{n}: made off the frame: {}", made.join(", ")));
            }
            if self.config.play && c.module_time + c.pipeline_time >= SLOW_COMPILE {
                self.log(&format!("gpu: present #{n}: {} modules in {} ms, {} pipelines in {} ms: {}", c.modules, c.module_time.as_millis(), c.pipelines, c.pipeline_time.as_millis(), c.made.join(" ")));
            }
            // D42: a pipeline the driver works on that long; its draws wait meanwhile.
            for label in overdue {
                let line = format!("gpu: present #{n}: pipeline {label} still compiling after {} s; its draws wait", OVERDUE.as_secs());
                eprintln!("gyroship: {line}");
                self.log(&line);
            }
            for (why, count) in skipped {
                // Each distinct reason is journaled once, past the anomaly's first few lines.
                if self.journal.first(&format!("gpu not done {why}")) {
                    self.log(&format!("gpu: first refusal of its kind, frame {n}: {why}"));
                }
                self.anomaly("gpu operation not done", &format!("frame {n}: {why} ({count} times)"))?;
            }
            for e in errors {
                self.anomaly("wgpu error", &format!("frame {n}: {e}"))?;
            }
        }
        // The presented image's hash, over the drawn area but its last row: the game's
        // upscaler races on that row (two threads write it; M3 report, §6.1).
        // Hashed images are read back without waiting and land at the end of the run.
        if self.config.hashes.is_some()
            && let Some(s) = self.surfaces.as_mut()
        {
            let (w, h) = ((area & 0xffff) + 1, (area >> 16) + 1);
            if !s.hash_presented(n, w, h.saturating_sub(1)) {
                self.frame_hashes.push((n, None));
            }
        }
        // `--video`: the presented image to ffmpeg (host output, not journaled).
        if let (Some(path), Some(s)) = (self.config.video.clone(), self.surfaces.as_mut())
            && self.config.picks(n)
        {
            let (w, h) = ((area & 0xffff) + 1, (area >> 16) + 1);
            if let Some(rgba) = s.presented_rgba(w, h) {
                if self.video.is_none() {
                    let fps = 60.0 / self.config.dump_every as f64;
                    match crate::video::Video::start(&path, (w, h), fps) {
                        Ok(v) => self.video = Some(v),
                        Err(e) => return Err((2, format!("--video {}: {e}", path.display()))),
                    }
                }
                if let Some(v) = self.video.as_mut() {
                    if v.size != (w, h) {
                        v.skipped += 1;
                    } else if !v.frame(&rgba) {
                        return Err((2, format!("--video {}: ffmpeg stopped reading", path.display())));
                    }
                }
            }
        }
        // The host window: show the frame, then wait for its time on the wall clock.
        let now = self.now();
        let shown = ((area & 0xffff) + 1, (area >> 16) + 1);
        if let (Some(s), Some((pace, size))) = (self.surfaces.as_mut(), self.window.as_mut()) {
            s.show(shown, *size);
            match pace(now) {
                Some(new) => *size = new,
                None => return Err((0, "window closed".into())),
            }
        }
        // The render targets this frame touched, as PNG files (game output: never committed).
        let dumped = self.config.dumps(n);
        if let (Some(dir), Some(s)) = (self.config.frames_dump.clone(), self.surfaces.as_mut())
            && dumped
        {
            let line = match s.dump(&dir, &format!("frame-{n:04}")) {
                Ok(count) => format!("gpu: frame {n}: {count} render targets dumped; {} draws done", s.drawn),
                Err(e) => format!("gpu: frame {n}: dump failed: {e}"),
            };
            self.log(&line);
        } else if let Some(s) = self.surfaces.as_mut() {
            s.untouch();
        }
        if let Some(s) = self.surfaces.as_mut() {
            s.present = n + 1;
        }
        Ok(())
    }
}

impl Kernel {

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
        self.log(&format!("gsv 204: syncpoint memory rw {:#x}+{:#x} r {:#x}+{:#x}, header version 1 n 0, events (request, acknowledge) {:?}", a.0, a.1, b.0, b.1, self.gs.syncpt_events));
        Reply { move_handles: vec![req_w, ack_r], ..Reply::ok() }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// An early frame waits for its vsync; a late one waits for the next vsync,
    /// or with `--late-frames` is shown at once, the grid then starting from the
    /// vsync just past.
    #[test]
    fn late_frames_wait_for_no_vsync_with_late_frames() {
        // Last shown at vsync 10, interval 1, now before vsync 11: wait for 11.
        assert_eq!(super::next_vsync(10, 1, 11, false), (11, true));
        assert_eq!(super::next_vsync(10, 1, 11, true), (11, true));
        // Now past vsync 13 (the first from now is 14): late.
        assert_eq!(super::next_vsync(10, 1, 14, false), (14, true));
        assert_eq!(super::next_vsync(10, 1, 14, true), (13, false));
        // Interval 2: due at 12, now before it: wait for 12.
        assert_eq!(super::next_vsync(10, 2, 12, true), (12, true));
    }

    /// The flattened table answers what a scan of the mappings from the last one
    /// does, over random overlapping, nested, adjacent and removed mappings in
    /// two address spaces, at random addresses and at each mapping's bounds.
    #[test]
    fn the_mapping_table_matches_the_reverse_scan() {
        let mut seed = 0x9e37_79b9_7f4a_7c15u64;
        let mut next = |n: u64| {
            seed ^= seed << 13;
            seed ^= seed >> 7;
            seed ^= seed << 17;
            seed % n
        };
        let (mut maps, mut table): (Vec<(u32, u64, u64, u64)>, Table) = Default::default();
        for round in 0..400 {
            let (space, first) = (next(2) as u32, next(64) * 0x1000);
            let last = first + next(16) * 0x1000 + 0xfff;
            if round % 7 == 6 {
                maps.retain(|m| m.0 != space || m.2 < first || m.1 > last);
                table = flatten(&maps);
            } else {
                let cpu = 0x1_0000_0000 + next(1 << 20) * 0x1000;
                maps.push((space, first, last, cpu));
                paint(&mut table, space, first, last, cpu);
            }
            let probes = (0..64).map(|_| (next(2) as u32, next(80 * 0x1000))).chain([first.wrapping_sub(1), first, last, last + 1].map(|gpu| (space, gpu)));
            let scan = |maps: &[(u32, u64, u64, u64)], space: u32, gpu: u64| maps.iter().rev().find(|m| m.0 == space && (m.1..=m.2).contains(&gpu)).map_or(gpu, |m| m.3 + (gpu - m.1));
            for (space, gpu) in probes.collect::<Vec<_>>() {
                let (cpu, room) = span(&table, space, gpu);
                assert_eq!(cpu, scan(&maps, space, gpu), "round {round}: space {space}, gpu {gpu:#x}");
                // The span's last byte is still contiguous with its first.
                if room < 1 << 32 {
                    assert_eq!(scan(&maps, space, gpu + room - 1), cpu + room - 1, "round {round}: space {space}, gpu {gpu:#x}, span {room:#x}");
                }
            }
        }
    }
}
