//! The nvdrv-like session gsv 1402 hands out (CMIF): Open, Ioctl, Close,
//! QueryEvent on nvhost devices. A fake: it answers the queries the SDK's
//! boot makes and executes nothing. The values follow what the SDK's
//! clients check (our analysis of build `a7730d0f`); where the client only
//! stores a value, the value is a plausible choice of ours.
//!
//! Ioctls that read and write get their input copied into their output
//! first, then the fields the client reads are filled in.

use std::collections::BTreeMap;

use gyroship_gpu::pushbuf::{Channel, Effect};
use gyroship_runtime::mem::{Memory, PAGE};
use std::sync::atomic::{AtomicU32, Ordering};
use std::sync::{Arc, Mutex};

use crate::ipc::{Reply, Request};
use crate::{After, Kernel, Stop};

/// nvdrv error words (not HOS results). Never 0x1000: the client aborts on it.
const OK: u32 = 0;
const NOT_SUPPORTED: u32 = 1;

/// The GPU's characteristics (ctrl-gpu 0x05), 0xd0 bytes.
pub fn characteristics() -> [u8; 0xd0] {
    let mut c = [0u8; 0xd0];
    let mut put = |at: usize, v: &[u8]| c[at..at + v.len()].copy_from_slice(v);
    // arch | impl = 0x17e: the chip the SDK describes (1 GPC, 6 TPCs, 2 SMs each).
    put(0x00, &0x170u32.to_le_bytes());
    put(0x04, &0x00eu32.to_le_bytes());
    put(0x08, &0xa1u32.to_le_bytes()); // rev: at most 0xff
    put(0x10, &0x10_0000u64.to_le_bytes()); // high word must be 0
    put(0x24, &0x20u32.to_le_bytes()); // bus type: AXI
    // 0x28, 0x34: no big pages. 0x2c: the compression page size; NVN needs
    // compression (it reserves 4 GiB of GPU addresses in pages the size
    // nvrm derives from it, and fails on 0).
    put(0x2c, &0x10000u32.to_le_bytes());
    put(0x38, &1u32.to_le_bytes()); // GPC mask
    // SM and SPA versions: NVN's precompiled blobs exist for SM 8.8 only.
    put(0x3c, &0x808u32.to_le_bytes());
    put(0x40, &0x808u32.to_le_bytes());
    // Flags: TSG (8), read-only syncpoint map (26), mapping modify (34), and
    // the engine classes the chip has: 2D (40), 3D (41, which also brings
    // ZCULL), compute (42) and inline-to-memory (43). nvrm builds the class
    // list from these bits; NVN needs a compute class. Without bit 34 (and the
    // as-gpu ioctl 12, below), nvrm refuses to change a mapping's kind, so NVN
    // cannot make a texture compressible and its initialisation fails,
    // leaving the texture at GPU address 0.
    put(0x50, &(1u64 << 8 | 1 << 26 | 1 << 34 | 0xf << 40).to_le_bytes());
    // Classes: 2D, 3D, compute, GPFIFO, inline-to-memory, DMA copy (NVIDIA's
    // public ids, Ampere B). NVN only accepts a known compute class; Lichii's
    // channel gets the 3D class 0xc797.
    for (i, class) in [0x902du32, 0xc797, 0xc7c0, 0xc76f, 0xa140, 0xc7b5].iter().enumerate() {
        put(0x58 + 4 * i, &class.to_le_bytes());
    }
    // Highest ioctl number per device: ctrl-gpu, tsg, dbg, channel, as, ..., prof.
    // dbg at least 2: below, nvrm leaves a string of the device info null that
    // NVN (subsdk1) always reads. as at least 12: MAPPING_MODIFY.
    for (i, n) in [0x1fu16, 0x0c, 2, 0x20, 0x0c, 0, 0, 0, 0].iter().enumerate() {
        put(0x70 + 2 * i, &n.to_le_bytes());
    }
    put(0x88, &1u32.to_le_bytes());
    c
}

/// ZCULL info (ctrl-gpu 0x02): ten alignments and region sizes the client
/// stores and never checks; any non-zero value of ours.
const ZCULL_INFO: [u32; 10] = [0x40; 10];
/// TPCs of the one GPC.
const TPC_MASK: u32 = 0x3f;
/// GetVARegions: the 64 KiB-page region, at the top of each AS.
const BIG_PAGES: u64 = 0x10_0000_0000;
/// The channel's syncpoint; Lichii's present fences carry id 1.
const CHANNEL_SYNCPT: u32 = 1;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Device {
    CtrlGpu,
    AsGpu,
    Ctrl,
    ProfGpu,
    TsgGpu,
    Gpu,
}

impl Device {
    fn from_path(p: &str) -> Option<Device> {
        Some(match p {
            "/dev/nvhost-ctrl-gpu" => Device::CtrlGpu,
            "/dev/nvhost-as-gpu" => Device::AsGpu,
            "/dev/nvhost-ctrl" => Device::Ctrl,
            "/dev/nvhost-prof-gpu" => Device::ProfGpu,
            "/dev/nvhost-tsg-gpu" => Device::TsgGpu,
            "/dev/nvhost-gpu" => Device::Gpu,
            _ => return None,
        })
    }

    fn label(self) -> &'static str {
        match self {
            Device::CtrlGpu => "ctrl-gpu",
            Device::AsGpu => "as-gpu",
            Device::Ctrl => "ctrl",
            Device::ProfGpu => "prof-gpu",
            Device::TsgGpu => "tsg-gpu",
            Device::Gpu => "gpu",
        }
    }
}

#[derive(Debug, Default)]
pub struct Nvdrv {
    /// The device of each fd: fd = index + 1, never reused; `None` once closed.
    fds: Vec<Option<Device>>,
    /// Address spaces (AllocAsEx): start and end, by fd.
    spaces: BTreeMap<u32, (u64, u64)>,
    /// The gsv address space of each AS fd (AllocAsEx +4, which the SDK's
    /// hook fills in) and of each channel fd (the AS it is bound to).
    as_space: BTreeMap<u32, u32>,
    channel_space: BTreeMap<u32, u32>,
}

/// The GPU the HLE runs: each channel's command processor and what its
/// draws used. It lives in the kernel, so that a channel waiting on a
/// semaphore can go on at a present as well as at its next submit.
#[derive(Debug, Default)]
pub struct Gpu {
    /// Each channel's command processor and the gsv address space it runs in, by fd.
    channels: BTreeMap<u32, (Channel, u32)>,
    /// Program keys by GPU address, and the keys already seen.
    programs: BTreeMap<u64, u64>,
    keys: std::collections::BTreeSet<u64>,
    /// Presents each waiting channel has waited through.
    stalls: BTreeMap<u32, u32>,
    /// Every channel's compute QMDs and what they wait for.
    scheduler: gyroship_gpu::state::qmd::Scheduler,
}

/// What GPU work needs apart from the kernel, behind a lock of its own
/// (`Kernel::gpu_side`), so that in the play engine it can run while other
/// threads use the kernel. It shares the kernel's memory, journal, tally and
/// syncpoints, and has the kernel's config as `Kernel::new` got it; `at` is
/// where the kernel stood when it handed the work over.
pub struct GpuSide {
    /// The host render targets, when something renders.
    pub surfaces: Option<gyroship_gpu::surfaces::Surfaces>,
    /// The GPU channels and what they ran.
    pub gpu: Gpu,
    /// `--hashes`: the presents with no image to hash (the others come back from the GPU).
    pub frame_hashes: Vec<(u64, Option<(u64, u64)>)>,
    /// The host window's pace hook and its last size (`boot::HostWindow`).
    pub window: Option<(crate::Pace, (u32, u32))>,
    /// Textures found rewritten by the CPU: (present, GPU address, size), for `--perf`.
    pub rewrites: Vec<(u64, u64, u64)>,
    /// `--perf`: the frames file's rows (`FRAME_COLUMNS`), and the pages the
    /// textures' write watch went over so far.
    pub frames: String,
    pub watched: u64,
    pub mem: Arc<Memory>,
    pub journal: crate::journal::Journal,
    pub tally: Arc<Mutex<crate::Tally>>,
    pub syncpts: Arc<[AtomicU32; 32]>,
    pub config: crate::Config,
    pub at: Phase,
}

/// Where the kernel stood when it handed GPU work over: the work's lines are
/// stamped with its time and thread, and it sees the mappings of that moment.
#[derive(Clone, Default)]
pub struct Phase {
    pub now: u64,
    /// The guest thread's id (0: none).
    pub thread: u64,
    /// Frames presented and submits made so far.
    pub presents: u64,
    pub submits: u64,
    pub maps: Arc<crate::gs::Mappings>,
}

/// GPU work the kernel hands over (`Kernel::gpu_work`).
pub enum GpuJob {
    /// A submit's GPFIFO entries, on channel `fd`, in address space `space`.
    Submit { fd: u32, space: u32, entries: Vec<u8> },
    /// vi2 229, present `n`: its acquire fence (syncpoint id inverted, value),
    /// its drawn area and its swap interval.
    Present { n: u64, fence: (u32, u32), area: u32, interval: u64 },
}

impl GpuSide {
    pub fn new(mem: Arc<Memory>, journal: crate::journal::Journal, tally: Arc<Mutex<crate::Tally>>, syncpts: Arc<[AtomicU32; 32]>, config: crate::Config) -> GpuSide {
        GpuSide { surfaces: None, gpu: Gpu::default(), frame_hashes: Vec::new(), window: None, rewrites: Vec::new(), frames: String::new(), watched: 0, mem, journal, tally, syncpts, config, at: Phase::default() }
    }

    /// The kernel time the work was handed over at.
    pub fn now(&self) -> u64 {
        self.at.now
    }

    /// Logs a line, stamped as the hand-over.
    pub fn log(&self, text: &str) {
        self.journal.line(self.at.now, self.at.thread, text);
    }

    pub fn count(&self, name: &'static str) -> u64 {
        crate::lock(&self.tally).count(name)
    }

    /// A named anomaly, as `Kernel::anomaly`: in strict mode (unless
    /// tolerated) the work stops there, and the kernel stops the run with its line.
    pub fn anomaly(&self, name: &'static str, text: &str) -> Result<(), Stop> {
        let (line, fatal) = crate::lock(&self.tally).anomaly(name, text, self.config.explore);
        if fatal {
            return Err((crate::STOP_EXIT, line));
        }
        if self.journal.few(name, 4) {
            self.log(&line);
        }
        Ok(())
    }
}

impl Kernel {
    /// Hands GPU work over to the GPU side, where it runs under the GPU lock
    /// alone; then what it leaves (`After`) runs under the kernel lock. The
    /// deterministic engine runs it at once. The play engine queues it
    /// (`Deferred`, Stage 4): the caller's host thread runs it once the SVC's
    /// reply is written and the kernel lock released, before its guest code
    /// goes on, while the other threads use the kernel.
    pub(crate) fn gpu_work(&mut self, job: GpuJob) {
        let phase = Phase { now: self.now(), thread: self.sched.current.map_or(0, |t| self.sched.threads[t].id), presents: self.gs.presents, submits: self.tally().counts.get("submits").copied().unwrap_or(0), maps: self.gs.maps.clone() };
        if self.config.play && self.deferred.is_none() {
            self.deferred = Some(crate::Deferred::Gpu { side: self.gpu_side.clone(), phase, job });
            return;
        }
        let done = run(&self.gpu_side, phase, job);
        self.after(done);
    }
}

/// Runs `job` on the GPU side, handed over at `phase`; what it leaves for the kernel lock.
pub fn run(side: &Mutex<GpuSide>, phase: Phase, job: GpuJob) -> Result<After, Stop> {
    let mut g = crate::lock(side);
    // The play engine's watchdog sees who holds the GPU side, and since when.
    let play = g.config.play;
    if play {
        crate::GPU_SINCE.store(crate::play::real_ticks(), Ordering::Relaxed);
        crate::GPU_GUEST.store(phase.thread, Ordering::Relaxed);
        crate::GPU_HOLDER.store(gyroship_runtime::profile::thread_id(), Ordering::Relaxed);
    }
    let at = phase.now;
    g.at = phase;
    let done = match &job {
        GpuJob::Submit { fd, space, entries } => run_gpfifo(&mut g, *fd, *space, entries),
        GpuJob::Present { n, fence, area, .. } => g.present(*n, *fence, *area),
    };
    // The mappings' snapshot goes: a change to them need not copy them.
    g.at = Phase::default();
    if play {
        crate::GPU_HOLDER.store(0, Ordering::Relaxed);
    }
    drop(g);
    done?;
    Ok(match job {
        GpuJob::Present { n, interval, .. } => After::Present { n, interval, at },
        GpuJob::Submit { .. } => After::Nothing,
    })
}

impl Nvdrv {
    pub fn label(&self) -> String {
        "nvsrv".into()
    }

    pub fn request(&mut self, k: &mut Kernel, req: &Request) -> Option<Reply> {
        let error = |e: u32| Reply::with(&e.to_le_bytes());
        Some(match req.command {
            // Open: the path, without its NUL, in the send buffer.
            0 => {
                let b = req.send_buffer(0)?;
                let mut raw = vec![0; b.size.min(0x100) as usize];
                k.mem.read(b.addr, &mut raw).ok()?;
                let path = String::from_utf8_lossy(&raw).trim_end_matches('\0').to_string();
                let Some(dev) = Device::from_path(&path) else {
                    k.log(&format!("nvsrv: open {path:?}: no such device"));
                    return Some(Reply::with(&[u32::MAX.to_le_bytes(), NOT_SUPPORTED.to_le_bytes()].concat()));
                };
                self.fds.push(Some(dev));
                let fd = self.fds.len() as u32;
                k.log(&format!("nvsrv: open {path:?} -> fd {fd}"));
                Reply::with(&[fd.to_le_bytes(), OK.to_le_bytes()].concat())
            }
            // Ioctl, Ioctl2 (one more input) and Ioctl3 (one more output).
            1 | 11 | 12 => error(self.ioctl(k, req)),
            // Close: the fd is forgotten.
            2 => {
                let fd = req.u32_at(0)?;
                if let Some(d) = self.fds.get_mut(fd.wrapping_sub(1) as usize) {
                    k.log(&format!("nvsrv: close fd {fd} ({})", d.map_or("closed", Device::label)));
                    *d = None;
                }
                error(OK)
            }
            // QueryEvent: an event nothing signals (errors only).
            4 => {
                let (fd, id) = (req.u32_at(0)?, req.u32_at(4)?);
                k.log(&format!("nvsrv: fd {fd} ({}) event {id}", self.device(fd).map_or("closed", Device::label)));
                Reply { copy_handles: vec![k.lone_event()?], ..error(OK) }
            }
            // SetGraphicsFirmwareMemoryMarginEnabled (Lichii's name).
            13 => {
                k.log(&format!("nvsrv 13: {:#x}", req.u64_at(0).unwrap_or(0)));
                Reply::ok()
            }
            _ => return None,
        })
    }

    fn device(&self, fd: u32) -> Option<Device> {
        *self.fds.get(fd.wrapping_sub(1) as usize)?
    }

    /// One ioctl: the input copied into the output, the output filled in.
    fn ioctl(&mut self, k: &mut Kernel, req: &Request) -> u32 {
        let (Some(fd), Some(code)) = (req.u32_at(0), req.u32_at(4)) else { return NOT_SUPPORTED };
        let Some(dev) = self.device(fd) else {
            k.log(&format!("nvsrv: ioctl {code:#x} on fd {fd}, which is not open"));
            return NOT_SUPPORTED;
        };
        let read = |k: &mut Kernel, b: Option<crate::ipc::Buffer>| {
            let b = b.unwrap_or_default();
            let mut v = vec![0; b.size.min(0x10000) as usize];
            let _ = k.mem.read(b.addr, &mut v);
            v
        };
        let input = read(k, req.send_buffer(0));
        let out_buf = req.out_buffer(0).unwrap_or_default();
        let mut out = input.clone();
        out.resize(out_buf.size.min(0x10000) as usize, 0);
        let extra_buf = if req.command == 12 { req.out_buffer(1).unwrap_or_default() } else { Default::default() };
        let mut extra = vec![0; extra_buf.size.min(0x10000) as usize];

        let label = dev.label();
        if k.journal.first(&format!("nvsrv ioctl {label} {code:#x}")) {
            k.log(&format!("nvsrv: fd {fd} ({label}) ioctl {code:#x}"));
        }
        let put32 = |v: &mut [u8], at: usize, x: u32| {
            if let Some(s) = v.get_mut(at..at + 4) {
                s.copy_from_slice(&x.to_le_bytes());
            }
        };
        let put64 = |v: &mut [u8], at: usize, x: u64| {
            if let Some(s) = v.get_mut(at..at + 8) {
                s.copy_from_slice(&x.to_le_bytes());
            }
        };
        match (dev, code) {
            // ctrl-gpu: characteristics, TPC masks, SM ids, ZCULL, engines, GPU time, ZBC.
            (Device::CtrlGpu, 0xc0e04705) => {
                let c = characteristics();
                let n = extra.len().min(c.len());
                extra[..n].copy_from_slice(&c[..n]);
            }
            (Device::CtrlGpu, 0xc0184706) => {
                for m in extra.as_chunks_mut::<4>().0 {
                    *m = TPC_MASK.to_le_bytes();
                }
            }
            // One entry per SM, two SMs per TPC; the client checks the last one got written.
            (Device::CtrlGpu, 0xc0084713) => {
                for (i, e) in extra.as_chunks_mut::<2>().0.iter_mut().enumerate() {
                    *e = [0, (i / 2) as u8];
                }
            }
            (Device::CtrlGpu, 0x80044701) => put32(&mut out, 0, 0x10000),
            (Device::CtrlGpu, 0x80284702) => {
                for (i, v) in ZCULL_INFO.iter().enumerate() {
                    put32(&mut out, 4 * i, *v);
                }
            }
            // Engines {type, instance, runlist, 0}: graphics and its copy engine, runlist 0.
            (Device::CtrlGpu, 0xc010471f) => {
                let engines = [[0u32, 0, 0, 0], [1, 0, 0, 0]];
                for (e, slot) in engines.iter().zip(extra.as_chunks_mut::<0x10>().0) {
                    for (j, w) in e.iter().enumerate() {
                        put32(slot, 4 * j, *w);
                    }
                }
                put32(&mut out, 0, (engines.len() * 0x10) as u32);
            }
            // GPU time: nanoseconds of the kernel's time (ticks are 32 ns).
            (Device::CtrlGpu, 0xc010471c) => put64(&mut out, 0, k.now() * 32),
            (Device::CtrlGpu, 0x402c4703) => {}
            // as-gpu: allocate, the read-only syncpoint map, the VA regions, bind a channel.
            (Device::AsGpu, 0x40284109) => {
                let at = |o: usize| u64::from_le_bytes(input.get(o..o + 8).and_then(|s| s.try_into().ok()).unwrap_or([0; 8]));
                self.spaces.insert(fd, (at(0x10), at(0x18)));
                self.as_space.insert(fd, u32_at(&input, 4));
                k.log(&format!("nvsrv: fd {fd} address space {:#x}..{:#x} (gsv space {})", at(0x10), at(0x18), u32_at(&input, 4)));
            }
            (Device::AsGpu, 0x8010410b) => {
                let (_, end) = self.spaces.get(&fd).copied().unwrap_or_default();
                put64(&mut out, 0, end + 0xffff_f000);
                put32(&mut out, 8, 0x1000);
                put32(&mut out, 0xc, 192);
            }
            (Device::AsGpu, 0xc0404108) => {
                // Small pages but for the top BIG_PAGES bytes. NVN (subsdk1)
                // builds an allocator per region, fails on an empty one, and
                // takes 512 GiB of small pages from the second AS (640 GiB).
                // Our split.
                let (start, end) = self.spaces.get(&fd).copied().unwrap_or_default();
                let split = end.saturating_sub(BIG_PAGES).max(start);
                let regions = [(start, 0x1000u32, (split - start) / 0x1000), (split, 0x10000, (end - split) / 0x10000)];
                // The regions go inline (Ioctl) or in the extra output (Ioctl3).
                let target = if req.command == 12 { &mut extra[..] } else { out.get_mut(0x10..).unwrap_or_default() };
                for ((base, page, pages), r) in regions.iter().zip(target.as_chunks_mut::<0x18>().0) {
                    put64(r, 0, *base);
                    put32(r, 8, *page);
                    put64(r, 0x10, *pages);
                }
                put32(&mut out, 8, (regions.len() * 0x18) as u32);
            }
            // MAPPING_MODIFY: a range of a mapping takes another kind (compressed
            // kind, fallback kind, buffer offset, size, map address). Kinds change
            // nothing for us: it succeeds.
            (Device::AsGpu, 0xc020410c) => {
                if k.journal.few("as mapping modify", 4) {
                    let at = |o: usize| u64::from_le_bytes(input.get(o..o + 8).and_then(|s| s.try_into().ok()).unwrap_or([0; 8]));
                    let kind = |o: usize| i16::from_le_bytes([input.get(o).copied().unwrap_or(0), input.get(o + 1).copied().unwrap_or(0)]);
                    k.log(&format!("nvsrv: fd {fd} mapping modify: kind {} (fallback {}), offset {:#x}, size {:#x}, at {:#x}", kind(0), kind(2), at(8), at(0x10), at(0x18)));
                }
            }
            // Bind a channel (its fd) to this address space.
            (Device::AsGpu, 0x40044101) => {
                if let Some(&space) = self.as_space.get(&fd) {
                    self.channel_space.insert(u32_at(&input, 0), space);
                }
            }
            // tsg-gpu: bind a channel, time slice.
            (Device::TsgGpu, 0xc018540b | 0xc0045408) => {}
            // gpu (the channel): GPFIFO allocation hands out the channel's syncpoint.
            (Device::Gpu, 0xc020481a) => {
                let value = k.gs.syncpts[CHANNEL_SYNCPT as usize].load(Ordering::Acquire);
                put32(&mut out, 0xc, CHANNEL_SYNCPT);
                put32(&mut out, 0x10, value);
                k.log(&format!("nvsrv: channel fd {fd}, {} entries, syncpoint {CHANNEL_SYNCPT} value {value}", u32_at(&input, 0)));
            }
            (Device::Gpu, 0x40044820) => k.log(&format!("nvsrv: channel fd {fd} runlist {}", u32_at(&input, 0))),
            (Device::Gpu, 0xc0104809 | 0xc018480c | 0x40084714 | 0x40044803 | 0xc008481f | 0xc010480b | 0x4004480d) => {}
            // Submit: nothing runs; the status word stays 0 and, if asked (flag 2), the
            // fence is the channel syncpoint's next value, reached at once.
            (Device::Gpu, 0xc018481b) => {
                let flags = u32_at(&input, 0xc);
                put32(&mut out, 0xc, 0);
                let n = k.count("submits");
                k.census_ipc("nvdrv submit");
                let mut fence = String::new();
                if flags & 2 != 0 {
                    let v = k.gs.syncpts[CHANNEL_SYNCPT as usize].fetch_add(1, Ordering::AcqRel).wrapping_add(1);
                    put32(&mut out, 0x10, CHANNEL_SYNCPT);
                    put32(&mut out, 0x14, v);
                    fence = format!(" -> fence ({CHANNEL_SYNCPT}, {v})");
                }
                if n <= 3 || n.is_multiple_of(600) {
                    k.log(&format!("nvsrv: submit #{n} fd {fd}, {} entries, flags {flags:#x}{fence}", u32_at(&input, 8)));
                }
                let entries = read(k, req.send_buffer(1));
                let space = self.channel_space.get(&fd).copied().unwrap_or(0);
                k.gpu_work(GpuJob::Submit { fd, space, entries });
            }
            _ => {
                let bytes: String = input.iter().take(64).map(|b| format!("{b:02x}")).collect();
                let what = format!(
                    "nvsrv: fd {fd} ({label}) ioctl {code:#x} (command {}) is not implemented; in {:#x}, out {:#x}, extra {:#x}; input {bytes}",
                    req.command,
                    input.len(),
                    out.len(),
                    extra.len()
                );
                if !k.config.explore {
                    k.stop(crate::STOP_EXIT, &what);
                }
                if k.journal.few(&what, 1) {
                    k.log(&what);
                }
                return NOT_SUPPORTED;
            }
        }
        if !out.is_empty() {
            let _ = k.mem.write(out_buf.addr, &out);
        }
        if !extra.is_empty() {
            let _ = k.mem.write(extra_buf.addr, &extra);
        }
        OK
    }
}

/// Runs a submit's GPFIFO entries (GPU address bits 2..40, length in
/// words at bits 42..63) through the channel's command processor, as far as
/// its semaphore acquires allow, and performs what they did.
fn run_gpfifo(k: &mut GpuSide, fd: u32, space: u32, entries: &[u8]) -> Result<(), Stop> {
    forget_written_textures(k, space);
    if let Some(s) = k.surfaces.as_mut() {
        s.new_submit();
    }
    let mut gpu = std::mem::take(&mut k.gpu);
    let done = (|| {
        for (rank, e) in entries.as_chunks::<8>().0.iter().map(|e| u64::from_le_bytes(*e)).enumerate() {
            let (at, len) = (e & 0xffff_fffc | (e >> 32 & 0xff) << 32, (e >> 42 & 0x1f_ffff) as usize);
            let mut raw = vec![0u8; 4 * len.min(0x10_0000)];
            let addr = k.gpu_to_cpu(space, at);
            if k.mem.read(addr, &mut raw).is_err() {
                k.anomaly("gpfifo unreadable", &format!("channel fd {fd}: pushbuffer {at:#x} ({len} words) is not readable at {addr:#x}"))?;
                continue;
            }
            let words: Vec<u32> = raw.as_chunks::<4>().0.iter().map(|w| u32::from_le_bytes(*w)).collect();
            // Raw segments of the first submits in GPFIFO order (they load the macros) and, with
            // --dump-from N, those of presents N to N + 9, for study (never committed: game data).
            let (submits, from) = (k.at.submits, k.config.dump_from);
            if let Some(dir) = &k.config.gpfifo_dump
                && (submits <= 300 || (from > 0 && (from..from + 10).contains(&k.at.presents)))
            {
                let _ = std::fs::create_dir_all(dir);
                let _ = std::fs::write(dir.join(format!("submit-{submits:04}-{rank:04}-{at:010x}.bin")), &raw);
            }
            let (channel, channel_space) = gpu.channels.entry(fd).or_insert_with(|| (Channel::default(), space));
            *channel_space = space;
            let before = channel.waiting();
            let effects = channel.run_with(&words, &mut |a, buf| k.read_gpu(space, a, buf));
            note_wait(k, fd, space, before, channel.waiting());
            gpu.effects(k, space, effects)?;
        }
        Ok(())
    })();
    k.gpu = gpu;
    done
}

/// The host's copies of textures whose guest memory was written since they
/// were made (the CPU drawing glyphs into a font atlas, say) are dropped
/// before the GPU work that may use them: they upload afresh.
fn forget_written_textures(k: &mut GpuSide, space: u32) {
    let Some(ranges) = k.surfaces.as_ref().map(|s| s.texture_ranges()) else { return };
    // (GPU start, GPU end, CPU address), by CPU address.
    let mut cpu: Vec<(u64, u64, u64)> = ranges.iter().map(|&(start, end)| (start, end, k.gpu_to_cpu(space, start))).collect();
    cpu.sort_unstable_by_key(|c| c.2);
    let (written, pages) = written_textures(&k.mem, &cpu);
    k.watched += pages;
    if let Some(s) = k.surfaces.as_mut() {
        for (start, end) in written {
            s.invalidate(start, end - start);
            crate::lock(&k.tally).count("textures rewritten");
            // Each rewrite, to compare replays (the 12c divergence; host statistics).
            if k.config.perf.is_some() {
                k.rewrites.push((k.at.presents, start, end - start));
            }
        }
    }
}

/// The textures of `cpu` (GPU start, GPU end, CPU address; sorted by CPU
/// address) whose pages were written since the last call, as (GPU start, GPU
/// end). One write-watch take per span of textures whose pages touch: the
/// span's written pages are reported and reset in one call, so a write that
/// races it shows at the next submit, and every texture of the span is tested
/// against them, so textures sharing a page all see its write. Spans share no
/// page: taking one hides nothing from another. Also the pages the takes went over.
fn written_textures(mem: &Memory, cpu: &[(u64, u64, u64)]) -> (Vec<(u64, u64)>, u64) {
    let (mut written, mut watched) = (Vec::new(), 0);
    for span in page_spans(cpu) {
        let (first, last) = (span[0].2, span.iter().map(|&(start, end, at)| at + (end - start)).max().unwrap_or(0));
        let pages = mem.take_written(first, last - first);
        watched += (last - first).div_ceil(PAGE);
        written.extend(span.iter().filter(|&&(start, end, at)| touches(&pages, at, end - start)).map(|&(start, end, _)| (start, end)));
    }
    (written, watched)
}

/// Whether `[at, at + len)` holds a byte of one of `pages` (sorted page addresses).
fn touches(pages: &[u64], at: u64, len: u64) -> bool {
    let i = pages.partition_point(|&p| p + PAGE <= at);
    len > 0 && pages.get(i).is_some_and(|&p| p < at + len)
}

/// `ranges` (GPU start, GPU end, CPU address), sorted by CPU address, cut into
/// runs whose 4 KiB pages touch or overlap: a run's pages are its members' pages.
fn page_spans(ranges: &[(u64, u64, u64)]) -> Vec<&[(u64, u64, u64)]> {
    let mut spans = Vec::new();
    let mut from = 0;
    let mut reach = 0u64;
    for (i, &(start, end, at)) in ranges.iter().enumerate() {
        if i > from && at > reach.next_multiple_of(PAGE) {
            spans.push(&ranges[from..i]);
            from = i;
        }
        reach = if i == from { at + (end - start) } else { reach.max(at + (end - start)) };
    }
    if from < ranges.len() {
        spans.push(&ranges[from..]);
    }
    spans
}

/// Lets the channels that wait on a semaphore go on, once the CPU has
/// released it (called at each present; a submit resumes its own channel).
pub fn poll(k: &mut GpuSide) -> Result<(), Stop> {
    if let Some(s) = k.surfaces.as_mut() {
        s.new_submit();
    }
    let mut gpu = std::mem::take(&mut k.gpu);
    let waiting: Vec<u32> = gpu.channels.iter().filter(|(_, (c, _))| c.waiting().is_some()).map(|(fd, _)| *fd).collect();
    let done = (|| {
        for fd in waiting {
            let Some((channel, space)) = gpu.channels.get_mut(&fd) else { continue };
            let space = *space;
            let before = channel.waiting();
            let mut effects = channel.resume(&mut |a, buf| k.read_gpu(space, a, buf));
            // ponytail: a watchdog. A semaphore no one we model releases would hang
            // the game; after two presents the channel goes on, and the log says so.
            // Find the missing writer when one shows up here.
            let stalls = gpu.stalls.entry(fd).or_default();
            *stalls = if channel.waiting().is_some() { *stalls + 1 } else { 0 };
            if *stalls >= 2
                && let Some(a) = channel.skip_wait()
            {
                *stalls = 0;
                k.anomaly("gpu watchdog", &format!("channel fd {fd} waited two presents for {:#x} {:?} {:#x}; it goes on", a.addr, a.op, a.value))?;
                effects.extend(channel.resume(&mut |a, buf| k.read_gpu(space, a, buf)));
            }
            note_wait(k, fd, space, before, channel.waiting());
            gpu.effects(k, space, effects)?;
        }
        Ok(())
    })();
    k.gpu = gpu;
    done
}

/// Performs a semaphore release or reduction in guest memory (the GPU
/// writes whatever the CPU's permissions). The payload goes last, in one host
/// atomic: a guest thread that polls it sees the whole report, and its own
/// atomics on the word are not lost.
fn semaphore(k: &mut GpuSide, space: u32, r: &gyroship_gpu::state::qmd::Release) -> Result<(), Stop> {
    let n = if r.wide { 8 } else { 4 };
    // ponytail: where the CPU cannot read the word, the reduction starts from 0, as
    // when it was a copy (identical replays); start from the word if a game needs it.
    let readable = k.read_gpu(space, r.addr, &mut [0u8; 8][..n]);
    let mut bytes = vec![0u8; n];
    if r.timestamp {
        bytes.resize(8, 0);
        bytes.extend((k.now() * 32).to_le_bytes());
    }
    let cpu = k.gpu_to_cpu(space, r.addr);
    if k.mem.publish_object(cpu, n, &bytes[n..], |old| r.apply(if readable { old } else { 0 })).is_err() {
        k.anomaly("gpu semaphore not writable", &format!("semaphore at {:#x}", r.addr))?;
    } else if let Some(s) = k.surfaces.as_mut() {
        s.invalidate(r.addr, bytes.len() as u64);
    }
    if k.journal.few("gpu semaphore release", 8) {
        k.log(&format!("gpu: semaphore {r:x?}"));
    }
    Ok(())
}

/// Logs a channel starting or stopping a wait on a semaphore (the first ones).
fn note_wait(k: &mut GpuSide, fd: u32, space: u32, before: Option<gyroship_gpu::pushbuf::Acquire>, after: Option<gyroship_gpu::pushbuf::Acquire>) {
    if before == after || !k.journal.few("gpu wait", 48) {
        return;
    }
    match after {
        Some(a) => {
            let mut b = [0u8; 8];
            let read = k.read_gpu(space, a.addr, &mut b[..if a.wide { 8 } else { 4 }]);
            let now = if read { format!("{:#x}", u64::from_le_bytes(b)) } else { "unreadable".into() };
            let cpu = k.gpu_to_cpu(space, a.addr);
            let map = k.at.maps.list.iter().rev().find(|m| m.0 == space && (m.1..=m.2).contains(&a.addr)).copied();
            k.log(&format!("gpu: channel fd {fd} waits until {:#x} (cpu {cpu:#x}, mapping {map:x?}) is {:?} {:#x} (now {now})", a.addr, a.op, a.value));
        }
        None => k.log(&format!("gpu: channel fd {fd} goes on")),
    }
}

impl Gpu {
    /// The macro units' statistics, every channel together (`--perf`).
    pub fn mme_stats(&self) -> gyroship_gpu::mme::Stats {
        let mut all = gyroship_gpu::mme::Stats::default();
        for (channel, _) in self.channels.values() {
            if let Some(m) = &channel.mme {
                all += m.stats;
            }
        }
        all
    }

    /// The key of the graphics program at a GPU address (read from guest
    /// memory once, until its closing self-branch), None when no end is found
    /// in readable memory; new programs are journaled and go to
    /// `--dump-shaders DIR` (game code: never committed).
    fn program(&mut self, k: &mut GpuSide, space: u32, gpu: u64) -> Option<u64> {
        if let Some(key) = self.programs.get(&gpu) {
            return Some(*key);
        }
        let (key, code) = gyroship_gpu::draw::read_program(&mut |gpu, buf| k.read_gpu(space, gpu, buf), gpu)?;
        self.programs.insert(gpu, key);
        if self.keys.insert(key) {
            k.log(&format!("gpu: program {key:016x}: {} bytes at {gpu:#x}", code.len()));
            if let Some(dir) = &k.config.shaders_dump {
                let _ = std::fs::create_dir_all(dir);
                let _ = std::fs::write(dir.join(format!("{key:016x}.bin")), &code);
            }
        }
        Some(key)
    }

    /// Performs a channel's effects in guest memory and on the host GPU, in order.
    fn effects(&mut self, k: &mut GpuSide, space: u32, effects: Vec<Effect>) -> Result<(), Stop> {
        for effect in effects {
            match effect {
                Effect::Release(r) => {
                    let at = k.gpu_to_cpu(space, r.addr);
                    let width = usize::from(r.width);
                    // The payload is written last (publish_object): a guest thread polling it sees the whole report.
                    let mut bytes = r.value.to_le_bytes()[..width].to_vec();
                    if r.timestamp {
                        bytes.resize(8, 0);
                        bytes.extend((k.now() * 32).to_le_bytes());
                    }
                    let what = format!("method {:#x} releases {} at {:#x} (cpu {at:#x}){}", r.method, r.value, r.addr, if r.timestamp { " with timestamp" } else { "" });
                    // NVN releases 1 at GPU address 0 after each of its sync objects: that
                    // increments the channel's syncpoint, whose value vi2's present fences
                    // name (syncpoint !0xfffffffe = 1, value = the count of such releases).
                    if r.addr == 0 && r.value == 1 && !r.timestamp {
                        k.syncpts[CHANNEL_SYNCPT as usize].fetch_add(1, Ordering::AcqRel);
                        k.count("syncpoint increments");
                    } else if k.mem.publish_object(at, width, &bytes[width..], |_| r.value).is_err() {
                        k.anomaly(if r.addr == 0 { "gpu release at 0" } else { "gpu release not writable" }, &what)?;
                    } else {
                        // Read again by later GPU work, as any GPU write.
                        if let Some(s) = k.surfaces.as_mut() {
                            s.invalidate(r.addr, bytes.len() as u64);
                        }
                        if k.journal.few("gpu release", 8) {
                            k.log(&format!("gpu: {what}"));
                        }
                    }
                }
                Effect::Ignored { class, method, value } => k.anomaly("gpu method ignored", &format!("class {class:#x} method {method:#x} = {value:#x} not executed"))?,
                Effect::Clear(c) => {
                    // Journaled with the draws of a dumped frame: targets and raw values.
                    if k.config.dumps(k.at.presents + 1) {
                        k.log(&format!("gpu: clear {c:x?}"));
                    }
                    if let Some(s) = &mut k.surfaces {
                        s.clear(&c);
                    }
                }
                Effect::Draw(d) => {
                    let draw = k.count("draws");
                    // The programs first (a new one is journaled and dumped before the draw);
                    // the draw's line, and the strings it needs, only on the frames it is written.
                    let programs: Vec<(usize, u64, Option<u64>)> = d.programs.iter().enumerate().filter_map(|(stage, p)| Some((stage, (*p)?, self.program(k, space, (*p)?)))).collect();
                    let dumped = k.config.dumps(k.at.presents + 1);
                    let logged = k.at.presents < 3 || draw <= 64 || dumped;
                    let mut outcome = String::new();
                    if let Some(mut s) = k.surfaces.take() {
                        let result = s.draw(&d, &mut |gpu, buf| k.read_gpu(space, gpu, buf));
                        if logged {
                            outcome = match result {
                                Ok(()) => format!(" -> drawn{}", if s.textures_read.is_empty() { String::new() } else { format!(", textures {:x?}", s.textures_read) }),
                                Err(why) => format!(" -> not drawn: {why}"),
                            };
                        }
                        // One frame, draw by draw (game output: never committed).
                        if let (Some(dir), Some(n)) = (&k.config.frames_dump, k.config.dump_draws)
                            && k.at.presents + 1 == n
                        {
                            let _ = s.dump(dir, &format!("frame-{n:04}-draw-{draw:06}"));
                        }
                        k.surfaces = Some(s);
                    }
                    // The first frames' draws, one line each.
                    if logged {
                        let keys: Vec<String> = programs
                            .iter()
                            .map(|&(stage, gpu, key)| match key {
                                Some(key) => format!("stage {stage} {key:016x}"),
                                None => format!("stage {stage} program at {gpu:#x}: no end found in readable memory"),
                            })
                            .collect();
                        let index = d.index.as_ref().map(|i| (i.address, i.bytes_per_index, i.first));
                        let targets: Vec<String> = d.targets.iter().map(|t| format!("{:#x} {:#x} {}x{}", t.addr, t.format, t.width, t.height)).collect();
                        let outputs: Vec<String> = d.render.outputs.iter().map(|o| format!("{}->{} mask {:x}{}", o.location, o.target, o.write_mask, if o.blend.is_some() { " blend" } else { "" })).collect();
                        k.log(&format!("gpu: draw {draw}: topology {} count {} index {index:x?} first {}: {}; targets [{}]; outputs [{}]; viewport scale {:?} offset {:?} origin {}, clip {:?}{outcome}", d.topology, d.count, d.first, keys.join(", "), targets.join(", "), outputs.join(", "), d.render.viewport.scale, d.render.viewport.offset, d.render.viewport.origin_lower_left as u8 | (d.render.viewport.flip_y as u8) << 1, d.render.surface_clip));
                    }
                }
                // The copy engine, line by line (the GPU writes whatever the CPU's permissions).
                Effect::Copy(c) => {
                    use gyroship_gpu::state::textures::{BlockLinear, GobLayout};
                    let mut line = vec![0u8; c.line_length.min(64 << 20) as usize];
                    let mut ok = u64::from(c.line_length) * u64::from(c.line_count) <= 256 << 20;
                    // A block-linear destination: 16 aligned bytes stay contiguous in a GOB.
                    let block = c.dst_block.map(|b| {
                        let l = BlockLinear { bytes_per_texel: 1, width: b.width, height: b.height, depth: b.depth.max(1), tile_width_log2: 0, gob_height_log2: b.gob_height_log2, gob_depth_log2: b.gob_depth_log2, gob: GobLayout::DEFAULT };
                        (b, l)
                    });
                    for i in 0..c.line_count {
                        if !ok {
                            break;
                        }
                        ok = k.read_gpu(space, c.src + u64::from(i) * u64::from(c.pitch_in), &mut line);
                        match &block {
                            None => {
                                let at = k.gpu_to_cpu(space, c.dst + u64::from(i) * u64::from(c.pitch_out));
                                ok = ok && k.mem.write_object(at, &line).is_ok();
                            }
                            Some((b, l)) => {
                                let mut j = 0;
                                while ok && j < line.len() {
                                    let x = b.x + j as u32;
                                    let n = ((16 - x % 16) as usize).min(line.len() - j);
                                    let at = k.gpu_to_cpu(space, c.dst + l.offset(x, b.y + i, b.layer));
                                    ok = k.mem.write_object(at, &line[j..j + n]).is_ok();
                                    j += n;
                                }
                            }
                        }
                    }
                    if !ok {
                        k.anomaly("gpu copy not done", &format!("copy {c:x?}: memory not readable or writable"))?;
                    }
                    let span = match &block {
                        Some((_, l)) => l.size(),
                        None => u64::from(c.pitch_out) * u64::from(c.line_count.max(1)),
                    };
                    if let Some(s) = &mut k.surfaces {
                        s.invalidate(c.dst, span);
                    }
                }
                // A PCAS action: the QMDs it lets run, each then its releases and its dependent QMD.
                Effect::Dispatch(first, action, pools) => {
                    use gyroship_gpu::state::qmd;
                    let mut work = std::collections::VecDeque::from([(first, action)]);
                    // Lichii's bound on a chain of dependents.
                    let mut chain = 256;
                    while let Some((at, action)) = work.pop_front() {
                        let mut q = [0u8; qmd::SIZE];
                        let readable = k.read_gpu(space, at, &mut q);
                        if !self.scheduler.signal(at, action, || readable.then_some(q)) {
                            continue;
                        }
                        let dispatch = k.count("dispatches");
                        let releases = qmd::releases(&q);
                        let launch = qmd::launch(&q);
                        let program = gyroship_gpu::draw::read_compute_program(&mut |gpu, buf| k.read_gpu(space, gpu, buf), launch.program);
                        let key = program.as_ref().map_or_else(|| "not readable".to_string(), |(key, code)| format!("{key:016x}, {} bytes", code.len()));
                        if let (Some(dir), Some((key, code))) = (&k.config.shaders_dump, &program) {
                            let _ = std::fs::create_dir_all(dir);
                            let _ = std::fs::write(dir.join(format!("{key:016x}.bin")), code);
                        }
                        if k.journal.few("gpu dispatch", 8) {
                            k.log(&format!("gpu: compute dispatch {dispatch}: QMD {at:#x} ({action:?}) version {:?}, program {key} at {:#x}, grid {:?} block {:?} shared {:#x} cbufs {:x?}; releases {releases:x?}; dependent {:x?}", qmd::version(&q), launch.program, launch.grid, launch.block, launch.shared, launch.cbufs, qmd::dependent(&q)));
                        }
                        // The program runs before its releases signal it done; then its dependent QMD.
                        if let Some(mut s) = k.surfaces.take() {
                            let result = s.dispatch(&launch, pools, &mut |gpu, buf| k.read_gpu(space, gpu, buf));
                            let read = std::mem::take(&mut s.textures_read);
                            k.surfaces = Some(s);
                            if let Err(e) = result {
                                k.anomaly("gpu compute not run", &format!("dispatch {dispatch}: program {key}, grid {:?} block {:?} shared {:#x}: {e}", launch.grid, launch.block, launch.shared))?;
                            }
                            // Each program's texture sources, once (as draws log theirs).
                            if !read.is_empty() && k.journal.first(&format!("gpu compute reads {key}")) {
                                k.log(&format!("gpu: compute program {key} reads textures {read:x?}"));
                            }
                        }
                        for r in releases {
                            semaphore(k, space, &r)?;
                        }
                        if let Some(next) = qmd::dependent(&q)
                            && chain > 0
                        {
                            chain -= 1;
                            work.push_back(next);
                        }
                    }
                }
                Effect::Semaphore(r) => semaphore(k, space, &r)?,
                // The GPU writes whatever the CPU's permissions.
                Effect::Write(w) => {
                    if let Some(s) = &mut k.surfaces {
                        s.invalidate(w.address, w.data.len() as u64);
                    }
                    // Contiguous writes come merged: split where the mappings are.
                    if let Err(at) = k.write_gpu(space, w.address, &w.data) {
                        k.anomaly("gpu write not done", &format!("{} bytes at {:#x} (cpu {at:#x}) not writable", w.data.len(), w.address))?;
                    }
                }
                Effect::Unsupported(u) => k.anomaly("gpu unsupported", &format!("not applied: {u:x?}"))?,
                Effect::MacroFault { index, fault } => k.anomaly("gpu macro fault", &format!("macro {index} stopped: {fault:?}"))?,
                Effect::MacroRead { addr, words } => {
                    if k.journal.few("gpu macro read", 20) {
                        k.log(&format!("gpu: macro read {words} words at {addr:#x}"));
                    }
                }
            }
        }
        Ok(())
    }
}

fn u32_at(v: &[u8], at: usize) -> u32 {
    u32::from_le_bytes(v.get(at..at + 4).and_then(|s| s.try_into().ok()).unwrap_or([0; 4]))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The checks the SDK's client makes on the characteristics (proved in our analysis).
    #[test]
    fn characteristics_pass_the_client_checks() {
        let c = characteristics();
        let u32_ = |o: usize| u32_at(&c, o);
        let s16 = |o: usize| i16::from_le_bytes([c[o], c[o + 1]]);
        assert_eq!(u32_(0) | u32_(4), 0x17e);
        assert!(u32_(8) <= 0xff);
        assert_eq!(u32_(0x14), 0, "high word of 0x10");
        assert_eq!(u32_(0x28) | u32_(0x34), 0, "no big pages");
        assert_eq!(s16(0x70), 0x1f);
        assert!(s16(0x72) >= 0xb);
        assert_eq!(s16(0x76), 0x20);
        // as-gpu: nvrm needs ioctl 11 (sdk+0x62694c) and 12, MAPPING_MODIFY (sdk+0x627678).
        assert!(s16(0x78) >= 0xc);
        let flags = u64::from_le_bytes(c[0x50..0x58].try_into().unwrap());
        assert_ne!(flags & 1 << 8, 0, "TSG");
        assert_ne!(flags & 1 << 34, 0, "mapping modify");
        assert_eq!(flags & (1 << 22 | 1 << 30), 0);
    }

    /// What the GPU path leaves for a release with a timestamp and two reductions,
    /// one where the CPU cannot read the word (the reduction then starts from 0).
    #[test]
    fn releases_and_reductions_leave_the_bytes_they_left_before() {
        use gyroship_gpu::pushbuf::Release;
        use gyroship_gpu::state::qmd::{self, Reduction};
        use gyroship_runtime::mem::{Block, Memory, attr, perm, state};
        let page = crate::layout::ALIAS.0;
        let k = Kernel::new(Memory::new().unwrap(), crate::journal::Journal::memory(), crate::Config::default());
        k.mem.map(page, 0x2000, Block::new(state::NORMAL, perm::RW)).unwrap();
        k.mem.write(page, &[0xff; 0x20]).unwrap();
        k.mem.write(page + 0x20, &5u32.to_le_bytes()).unwrap();
        k.mem.write(page + 0x1000, &5u32.to_le_bytes()).unwrap();
        // The second page lent with no permission (a transfer memory): the GPU writes it, the CPU cannot read it.
        k.mem.set_attr(page + 0x1000, 0x1000, attr::LOCKED, attr::LOCKED).unwrap();
        k.mem.protect(page + 0x1000, 0x1000, 0).unwrap();
        let add = |addr| qmd::Release { addr, payload: 3, reduction: Some(Reduction::Add), signed: false, wide: false, timestamp: false };
        let effects = vec![
            Effect::Release(Release { method: 0x1b0c, addr: page, value: 7, width: 4, timestamp: true }),
            Effect::Semaphore(add(page + 0x20)),
            Effect::Semaphore(add(page + 0x1000)),
        ];
        Gpu::default().effects(&mut k.gpu_side(), 0, effects).unwrap();
        let mut b = [0u8; 0x24];
        k.mem.read(page, &mut b).unwrap();
        // Payload 7, zero padding, timestamp 0 (no thread runs: virtual time 0); nothing past it.
        assert_eq!(b[..0x10], [7, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]);
        assert_eq!(b[0x10..0x20], [0xff; 0x10]);
        assert_eq!(b[0x20..], 8u32.to_le_bytes());
        let mut c = [0u8; 4];
        k.mem.read_object(page + 0x1000, &mut c).unwrap();
        assert_eq!(c, 3u32.to_le_bytes());
    }

    /// In strict mode GPU work ends at its first anomaly: the kernel stops the
    /// run with the anomaly's line, and the effects after it are not applied.
    #[test]
    fn a_strict_anomaly_ends_the_gpu_work_there() {
        use gyroship_gpu::state::resources::MemWrite;
        use gyroship_runtime::mem::{Block, Memory, perm, state};
        let page = crate::layout::ALIAS.0;
        let k = Kernel::new(Memory::new().unwrap(), crate::journal::Journal::memory(), crate::Config::default());
        k.mem.map(page, 0x1000, Block::new(state::NORMAL, perm::RW)).unwrap();
        let effects = vec![Effect::Ignored { class: 0xc797, method: 0x123, value: 1 }, Effect::Write(MemWrite { address: page, data: vec![1; 4] })];
        let done = Gpu::default().effects(&mut k.gpu_side(), 0, effects);
        assert_eq!(done, Err((crate::STOP_EXIT, "anomaly gpu method ignored: class 0xc797 method 0x123 = 0x1 not executed".into())));
        let mut b = [9u8; 4];
        k.mem.read(page, &mut b).unwrap();
        assert_eq!(b, [0; 4], "the write after the anomaly is not applied");
        assert_eq!(k.tally().anomalies["gpu method ignored"], 1);
    }

    /// The play engine leaves GPU work for after the kernel lock, as the SVC's
    /// deferred work; the deterministic engine runs it at once. Here a
    /// present's GPU part finds its fence not reached, then its rest runs.
    #[test]
    fn the_play_engine_runs_gpu_work_after_the_kernel_lock() {
        use gyroship_runtime::mem::Memory;
        for play in [false, true] {
            let config = crate::Config { play, explore: true, vsync: false, ..crate::Config::default() };
            let mut k = Kernel::new(Memory::new().unwrap(), crate::journal::Journal::memory(), config);
            k.gpu_work(GpuJob::Present { n: 1, fence: (!CHANNEL_SYNCPT, 5), area: 0, interval: 1 });
            let ran = |k: &Kernel| k.tally().anomalies.get("present fence not reached").copied();
            if play {
                assert_eq!(ran(&k), None, "queued, not run");
                let done = k.deferred.take().expect("a deferred job").run();
                assert!(matches!(done, Ok(After::Present { n: 1, interval: 1, at: 0 })));
                k.after(done);
            }
            assert!(k.deferred.is_none());
            assert_eq!(ran(&k), Some(1));
        }
    }
}

#[cfg(test)]
mod page_span_tests {
    use super::{page_spans, written_textures};
    use gyroship_runtime::mem::{Block, Memory, perm, state};

    /// Textures that share a written page all see the write, once: the span's
    /// pages are taken in one call and every texture is tested against them.
    #[test]
    fn textures_sharing_a_written_page_all_see_it() {
        const AT: u64 = 0x10_0000_0000;
        let m = Memory::new().unwrap();
        m.map(AT, 0x3000, Block::new(state::NORMAL, perm::RW)).unwrap();
        // (GPU start, GPU end, CPU address): two textures on page 1, one on page 2.
        let cpu = [(0x100, 0x900, AT + 0x1000), (0x900, 0xa00, AT + 0x1800), (0xa00, 0xb00, AT + 0x2000)];
        written_textures(&m, &cpu);
        m.write(AT + 0x1ff0, &[1]).unwrap();
        assert_eq!(written_textures(&m, &cpu).0, [(0x100, 0x900), (0x900, 0xa00)]);
        assert!(written_textures(&m, &cpu).0.is_empty(), "taken once");
    }

    #[test]
    fn spans_join_ranges_whose_pages_touch() {
        // (GPU start, GPU end, CPU address); CPU ranges [at, at + len).
        let r = [(0, 0x100, 0x1000), (0, 0x10, 0x1f00), (0, 0x2000, 0x2000), (0, 0x10, 0x5000), (0, 0x10, 0x6ff0), (0, 0x20, 0x9000)];
        let spans: Vec<Vec<u64>> = page_spans(&r).iter().map(|s| s.iter().map(|x| x.2).collect()).collect();
        // 0x1000.. and 0x1f00.. share a page; 0x2000.. starts at the next page boundary
        // (touching); 0x5000 starts a page after the run's last page; 0x6ff0 too.
        assert_eq!(spans, vec![vec![0x1000, 0x1f00, 0x2000], vec![0x5000], vec![0x6ff0], vec![0x9000]]);
        assert!(page_spans(&[]).is_empty());
    }
}
