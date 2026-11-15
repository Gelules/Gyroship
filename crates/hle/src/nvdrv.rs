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
use std::time::{Duration, Instant};

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
pub(crate) const CHANNEL_SYNCPT: u32 = 1;

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

/// What the HLE's GPU performs: the programs its draws used and every
/// channel's compute QMDs. It lives in the GPU side; the channels' command
/// processors live in the [`Decoder`].
#[derive(Debug, Default)]
pub struct Gpu {
    /// Program keys by GPU address, and the keys already seen.
    programs: BTreeMap<u64, u64>,
    keys: std::collections::BTreeSet<u64>,
    /// Every channel's compute QMDs and what they wait for.
    scheduler: gyroship_gpu::state::qmd::Scheduler,
}

/// The GPU channels' command processors, which decode pushbuffers into
/// effects: decoding reads guest memory and writes none. Behind a lock of
/// its own (`Kernel::decoder`), always taken before the GPU side's, so that
/// decoding can run apart from the effects. It lives in the kernel, so that
/// a channel waiting on a semaphore can go on at a present as well as at its
/// next submit. It shares the GPU side's memory, journal, tally and config;
/// `at` is where the kernel stood when it handed the job over.
pub struct Decoder {
    /// Each channel's command processor and the gsv address space it runs in, by fd.
    channels: BTreeMap<u32, (Channel, u32)>,
    /// Presents each waiting channel has waited through.
    stalls: BTreeMap<u32, u32>,
    /// Host nanoseconds so far reading and decoding pushbuffers (the command
    /// processor, macros included), and, apart, waiting for the GPU thread
    /// (its barriers, and with --gpu-sync the end of each job), for `--perf`.
    pub decode_ns: u64,
    pub gpu_wait_ns: u64,
    mem: Arc<Memory>,
    journal: crate::journal::Journal,
    tally: Arc<Mutex<crate::Tally>>,
    config: crate::Config,
    at: Phase,
    /// Play mode: the GPU thread, which applies the ops while this decodes.
    pub gpu: Option<GpuThread>,
    /// With the GPU thread, the guest memory each op sent and not yet applied
    /// may write, by its number: CPU ranges, or None for anywhere. Pruned as
    /// ops are applied, never cleared: a job's ops may still be queued when
    /// the next job decodes.
    writes: Vec<(u64, Option<Ranges>)>,
}

/// The GPU thread (play mode): it applies the decoded ops in order while the
/// submitting thread decodes the next GPFIFO entries. A job returns once its
/// ops are sent; then a present waits until the previous present is applied,
/// so at most one present is in flight behind the frame being decoded. With
/// --gpu-sync every job waits until its ops are applied (S2's join).
pub struct GpuThread {
    ops: std::sync::mpsc::SyncSender<(Phase, GpuOp)>,
    progress: Arc<Progress>,
    /// Ops sent so far, and up to the last present's `End`.
    sent: u64,
    presented: u64,
    /// Host nanoseconds the submitting thread waited for room: a full op
    /// channel, and a present's wait for the previous one (`--perf`).
    queue_wait_ns: u64,
    thread: std::thread::JoinHandle<()>,
}

/// How far the GPU thread got; its waiters wake at each op. `armed` is the
/// kernel's count of guest waits on GPU work (`gs::State::armed`): the GPU
/// thread wakes the boot thread for them. The mutex is a leaf: it is taken
/// under the kernel lock or the decoder lock, and nothing is taken under it.
#[derive(Default)]
pub(crate) struct Progress {
    state: Mutex<Applied>,
    changed: std::sync::Condvar,
    pub(crate) armed: std::sync::atomic::AtomicUsize,
}

/// Ops applied so far (those after a stop are skipped, and counted), the
/// last job whose `End` is applied (`Phase::job`), the stop the GPU thread
/// met, and whether it ended.
#[derive(Default)]
struct Applied {
    ops: u64,
    jobs: u64,
    stop: Option<Stop>,
    ended: bool,
}

impl Applied {
    /// The stop the GPU thread met, or its end.
    fn stopped(&self) -> Option<Stop> {
        self.stop.clone().or_else(|| self.ended.then(gpu_thread_ended))
    }
}

impl Progress {
    /// The last job done, and whether the GPU thread still applies ops.
    pub(crate) fn jobs(&self) -> (u64, bool) {
        let state = crate::lock(&self.state);
        (state.jobs, state.stop.is_none() && !state.ended)
    }

    /// The stop the GPU thread met, or its end.
    pub(crate) fn stopped(&self) -> Option<Stop> {
        crate::lock(&self.state).stopped()
    }

    /// Waits until job `job` is done, a stop or the end, `patience` at most:
    /// the stop (or end), and the last job done.
    fn drain(&self, job: u64, patience: Duration) -> (Option<Stop>, u64) {
        let state = crate::lock(&self.state);
        let (state, _) = self.changed.wait_timeout_while(state, patience, |s| s.jobs < job && s.stop.is_none() && !s.ended).unwrap_or_else(std::sync::PoisonError::into_inner);
        (state.stopped(), state.jobs)
    }
}

/// Ops in flight at most: about two heavy frames' GPFIFO entries.
const IN_FLIGHT: usize = 256;

/// CPU ranges of guest memory: (start, end).
type Ranges = Vec<(u64, u64)>;

/// A pushbuffer read with the GPU thread: the ops applied when it was read,
/// and its CPU range (start, end).
type PushRead = (u64, u64, u64);

/// The stop of a run whose GPU thread ended (a panic).
fn gpu_thread_ended() -> Stop {
    (crate::STOP_EXIT, "the GPU thread ended: GPU work was lost".into())
}

/// The guest memory `effects` may write once applied, as CPU ranges through
/// `maps`; None when that cannot be told before (a compute dispatch's QMD
/// releases, a block-linear copy). Releases and semaphores count 16 bytes.
fn writes_of(maps: &crate::gs::Mappings, space: u32, effects: &[Effect]) -> Option<Ranges> {
    let mut out = Vec::new();
    for e in effects {
        let (gpu, len) = match e {
            // A release of 1 at address 0 increments the channel's syncpoint: no memory.
            Effect::Release(r) if r.addr == 0 && r.value == 1 && !r.timestamp => continue,
            Effect::Release(r) => (r.addr, 16),
            Effect::Semaphore(r) => (r.addr, 16),
            Effect::Write(w) => (w.address, w.data.len() as u64),
            Effect::Copy(c) if c.dst_block.is_none() => (c.dst, u64::from(c.line_count.max(1) - 1) * u64::from(c.pitch_out) + u64::from(c.line_length)),
            Effect::Copy(_) | Effect::Dispatch(..) => return None,
            _ => continue,
        };
        out.extend(crate::gs::cpu_ranges(maps, space, gpu, len));
    }
    Some(out)
}

/// GPU work decoded, for the GPU side to perform in order ([`GpuSide::apply`]).
pub enum GpuOp {
    /// A submit starts in address space `space`: the take of written textures, then a new submit.
    Begin { space: u32 },
    /// What a GPFIFO entry, or a channel resumed at a present, did in `space`;
    /// with the GPU thread, the entry's pushbuffer read: the ops applied when
    /// it was read, and its CPU range (the read-ahead check).
    Effects { space: u32, effects: Vec<Effect>, read: Option<PushRead> },
    /// Present `n` starts (`GpuSide::begin_present`): its resumed channels' ops follow.
    PresentBegin { n: u64 },
    /// The rest of present `n`, after its frame's ops (`GpuSide::finish_present`).
    PresentEnd { n: u64, fence: (u32, u32), area: u32, perf: Option<FrameIn> },
    /// The end of a job's ops (a submit's or a present's): once applied, the
    /// GPU thread counts the job done (`Applied::jobs`), which gpu memory 213
    /// and 102 and the drain at a run's end wait for.
    End,
}

/// A frame's `--perf` values only the presenting thread can read (its handle
/// on the GPU thread is the decoder's), for its row (`FRAME_COLUMNS`).
pub struct FrameIn {
    pub cpu: Duration,
    pub macro_calls: u64,
    pub decode_ns: u64,
    pub gpu_wait_ns: u64,
    pub gpu_cpu: Duration,
    pub queue_wait_ns: u64,
}

/// A named anomaly met by GPU work handed over at `at`, as `Kernel::anomaly`:
/// in strict mode (unless tolerated) the work stops there, and the kernel
/// stops the run with its line.
fn anomaly(tally: &Mutex<crate::Tally>, journal: &crate::journal::Journal, explore: bool, at: &Phase, name: &'static str, text: &str) -> Result<(), Stop> {
    let (line, fatal) = crate::lock(tally).anomaly(name, text, explore);
    if fatal {
        return Err((crate::STOP_EXIT, line));
    }
    if journal.few(name, 4) {
        journal.line(at.now, at.thread, &line);
    }
    Ok(())
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
    /// `--video`: ffmpeg, once the first image came.
    pub video: Option<crate::video::Video>,
    /// The host window's pace hook and its last size (`boot::HostWindow`).
    pub window: Option<(crate::Pace, (u32, u32))>,
    /// Textures found rewritten by the CPU: (present, GPU address, size), for `--perf`.
    pub rewrites: Vec<(u64, u64, u64)>,
    /// `--perf`: the frames file's rows (`FRAME_COLUMNS`), and the pages the
    /// textures' write watch went over so far.
    pub frames: String,
    pub watched: u64,
    /// Host nanoseconds so far performing GPU ops (the take of written
    /// textures, guest writes, and translating and recording draws), for `--perf`.
    pub effects_ns: u64,
    /// With the GPU thread: the number of the op it applies, and the guest
    /// memory the ops applied wrote, in op order: (op, start, end). Pruned at
    /// each pushbuffer read checked, of the ops applied before that read.
    pub applying: u64,
    pub wrote: Option<std::collections::VecDeque<(u64, u64, u64)>>,
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

impl Phase {
    /// The job's number: jobs (submits and presents) are numbered in
    /// hand-over order, each counted just before `Kernel::gpu_work`.
    pub(crate) fn job(&self) -> u64 {
        self.presents + self.submits
    }
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
        GpuSide { surfaces: None, gpu: Gpu::default(), frame_hashes: Vec::new(), video: None, window: None, rewrites: Vec::new(), frames: String::new(), watched: 0, effects_ns: 0, applying: 0, wrote: None, mem, journal, tally, syncpts, config, at: Phase::default() }
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

    /// A named anomaly (see [`anomaly`]).
    pub fn anomaly(&self, name: &'static str, text: &str) -> Result<(), Stop> {
        anomaly(&self.tally, &self.journal, self.config.explore, &self.at, name, text)
    }

    /// Performs a decoded op; its host time goes to `effects_ns`, but a
    /// present's end, which reports the frame.
    pub fn apply(&mut self, op: GpuOp) -> Result<(), Stop> {
        let started = Instant::now();
        let done = match op {
            GpuOp::PresentEnd { n, fence, area, perf } => return self.finish_present(n, fence, area, perf),
            GpuOp::PresentBegin { n } => {
                self.begin_present(n);
                Ok(())
            }
            GpuOp::Begin { space } => {
                forget_written_textures(self, space);
                if let Some(s) = self.surfaces.as_mut() {
                    // After the take: textures written in part may be refreshed in place.
                    s.new_submit(true);
                }
                Ok(())
            }
            GpuOp::Effects { space, effects, read } => self.check_read(read).and_then(|()| {
                let mut gpu = std::mem::take(&mut self.gpu);
                let done = gpu.effects(self, space, effects);
                self.gpu = gpu;
                done
            }),
            GpuOp::End => Ok(()),
        };
        self.effects_ns += started.elapsed().as_nanos() as u64;
        done
    }

    /// With the GPU thread: an op applied after this op's pushbuffer was read
    /// (`applied`: the ops applied then) and before this op wrote into it: the
    /// read missed the write.
    fn check_read(&mut self, read: Option<PushRead>) -> Result<(), Stop> {
        let (Some((applied, start, end)), Some(w)) = (read, self.wrote.as_mut()) else { return Ok(()) };
        // Ops are applied in order: what those applied before the read wrote, it saw, as every later read.
        while w.front().is_some_and(|x| x.0 < applied) {
            w.pop_front();
        }
        let missed = w.iter().find(|&&(_, s, e)| s < end && start < e).copied();
        match missed {
            Some((op, s, e)) => self.anomaly("gpu wrote a pushbuffer read ahead", &format!("op {op} wrote {s:#x}..{e:#x}, in a pushbuffer decoded before it ran")),
            None => Ok(()),
        }
    }

    /// With the GPU thread: the op being applied writes `len` bytes of guest
    /// memory at `at` (checked against the pushbuffers decoded ahead).
    fn note_write(&mut self, at: u64, len: usize) {
        if let Some(w) = &mut self.wrote {
            w.push_back((self.applying, at, at + len as u64));
        }
    }
}

impl Decoder {
    pub fn new(mem: Arc<Memory>, journal: crate::journal::Journal, tally: Arc<Mutex<crate::Tally>>, config: crate::Config) -> Decoder {
        Decoder { channels: BTreeMap::new(), stalls: BTreeMap::new(), decode_ns: 0, gpu_wait_ns: 0, mem, journal, tally, config, at: Phase::default(), gpu: None, writes: Vec::new() }
    }

    /// Logs a line, stamped as the hand-over.
    fn log(&self, text: &str) {
        self.journal.line(self.at.now, self.at.thread, text);
    }

    /// A named anomaly (see [`anomaly`]).
    fn anomaly(&self, name: &'static str, text: &str) -> Result<(), Stop> {
        anomaly(&self.tally, &self.journal, self.config.explore, &self.at, name, text)
    }

    /// GPFIFO entry `e` (GPU address bits 2..40, length in words at bits
    /// 42..63), number `rank` of a submit on channel `fd` in `space`: its
    /// pushbuffer read and run through the channel's command processor, as
    /// far as its semaphore acquires allow, and with the GPU thread its read
    /// (`GpuOp::Effects`). None if it cannot be read.
    #[allow(clippy::type_complexity)]
    fn decode(&mut self, fd: u32, space: u32, rank: usize, e: u64) -> Result<Option<(Vec<Effect>, Option<PushRead>)>, Stop> {
        let decoding = Instant::now();
        let (at, len) = (e & 0xffff_fffc | (e >> 32 & 0xff) << 32, (e >> 42 & 0x1f_ffff) as usize);
        let mut raw = vec![0u8; 4 * len.min(0x10_0000)];
        let addr = crate::gs::gpu_to_cpu(&self.at.maps, space, at);
        // With the GPU thread, the earlier ops may not be applied yet: what they write here would be
        // missed (the GPU thread checks). Those it applied are seen (their count is read under the
        // lock they set it under), and need no barrier any more.
        let read = match &self.gpu {
            Some(t) => {
                let applied = crate::lock(&t.progress.state).ops;
                let k = self.writes.partition_point(|w| w.0 < applied);
                self.writes.drain(..k);
                Some((applied, addr, addr + raw.len() as u64))
            }
            None => None,
        };
        if self.mem.read(addr, &mut raw).is_err() {
            self.anomaly("gpfifo unreadable", &format!("channel fd {fd}: pushbuffer {at:#x} ({len} words) is not readable at {addr:#x}"))?;
            return Ok(None);
        }
        let words: Vec<u32> = raw.as_chunks::<4>().0.iter().map(|w| u32::from_le_bytes(*w)).collect();
        // Raw segments of the first submits in GPFIFO order (they load the macros) and, with
        // --dump-from N, those of presents N to N + 9, for study (never committed: game data).
        let (submits, from) = (self.at.submits, self.config.dump_from);
        if let Some(dir) = &self.config.gpfifo_dump
            && (submits <= 300 || (from > 0 && (from..from + 10).contains(&self.at.presents)))
        {
            let _ = std::fs::create_dir_all(dir);
            let _ = std::fs::write(dir.join(format!("submit-{submits:04}-{rank:04}-{at:010x}.bin")), &raw);
        }
        let Decoder { channels, mem, at: phase, gpu, writes, .. } = self;
        let (channel, channel_space) = channels.entry(fd).or_insert_with(|| (Channel::default(), space));
        *channel_space = space;
        let before = channel.waiting();
        let mut waited = Duration::ZERO;
        let effects = channel.run_with(&words, &mut |a, buf| {
            waited += barrier(gpu.as_ref(), writes, &phase.maps, space, a, buf.len() as u64);
            crate::gs::read_gpu(mem, &phase.maps, space, a, buf)
        });
        let after = channel.waiting();
        self.note_wait(fd, space, before, after);
        self.decode_ns += decoding.elapsed().saturating_sub(waited).as_nanos() as u64;
        self.gpu_wait_ns += waited.as_nanos() as u64;
        Ok(Some((effects, read)))
    }

    /// The channels that wait on a semaphore.
    fn waiting(&self) -> Vec<u32> {
        self.channels.iter().filter(|(_, (c, _))| c.waiting().is_some()).map(|(fd, _)| *fd).collect()
    }

    /// Channel `fd` goes on, as far as the CPU has released its semaphore:
    /// what it did, and its address space.
    fn resume(&mut self, fd: u32) -> Result<Option<(u32, Vec<Effect>)>, Stop> {
        let decoding = Instant::now();
        let Decoder { channels, stalls, mem, journal, tally, config, at: phase, gpu, writes, .. } = self;
        let Some((channel, space)) = channels.get_mut(&fd) else { return Ok(None) };
        let space = *space;
        let mut waited = Duration::ZERO;
        let read = &mut |a, buf: &mut [u8]| {
            waited += barrier(gpu.as_ref(), writes, &phase.maps, space, a, buf.len() as u64);
            crate::gs::read_gpu(mem, &phase.maps, space, a, buf)
        };
        let before = channel.waiting();
        let mut effects = channel.resume(read);
        // ponytail: a watchdog. A semaphore no one we model releases would hang
        // the game; after two presents the channel goes on, and the log says so.
        // Find the missing writer when one shows up here.
        let stalls = stalls.entry(fd).or_default();
        *stalls = if channel.waiting().is_some() { *stalls + 1 } else { 0 };
        if *stalls >= 2
            && let Some(a) = channel.skip_wait()
        {
            *stalls = 0;
            anomaly(tally, journal, config.explore, phase, "gpu watchdog", &format!("channel fd {fd} waited two presents for {:#x} {:?} {:#x}; it goes on", a.addr, a.op, a.value))?;
            effects.extend(channel.resume(read));
        }
        let after = channel.waiting();
        self.note_wait(fd, space, before, after);
        self.decode_ns += decoding.elapsed().saturating_sub(waited).as_nanos() as u64;
        self.gpu_wait_ns += waited.as_nanos() as u64;
        Ok(Some((space, effects)))
    }

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

    /// The presenting thread's `--perf` values for its frame's row (None without `--perf`).
    pub(crate) fn frame_in(&self) -> Option<FrameIn> {
        self.config.perf.as_ref()?;
        Some(FrameIn {
            cpu: gyroship_runtime::profile::own_cpu_time().unwrap_or_default(),
            macro_calls: self.mme_stats().calls,
            decode_ns: self.decode_ns,
            gpu_wait_ns: self.gpu_wait_ns,
            gpu_cpu: self.gpu.as_ref().and_then(GpuThread::cpu_time).unwrap_or_default(),
            queue_wait_ns: self.gpu.as_ref().map_or(0, |t| t.queue_wait_ns),
        })
    }

    /// Sends an op to the GPU thread; the stop it met, if any, instead.
    fn send(&mut self, phase: &Phase, op: GpuOp) -> Result<(), Stop> {
        self.gpu.as_mut().map_or(Err(gpu_thread_ended()), |t| t.send(phase, op))
    }

    /// Sends what an entry or a resumed channel did in `space`; the guest
    /// memory it may write joins `writes` once sent: no barrier waits for an
    /// op never sent.
    fn send_effects(&mut self, phase: &Phase, space: u32, effects: Vec<Effect>, read: Option<PushRead>) -> Result<(), Stop> {
        let (op, w) = (self.gpu.as_ref().map_or(0, |t| t.sent), writes_of(&phase.maps, space, &effects));
        self.send(phase, GpuOp::Effects { space, effects, read })?;
        self.writes.push((op, w));
        Ok(())
    }

    /// Play mode: a submit's ops, decoded here while the GPU thread applies them.
    fn submit_ops(&mut self, phase: &Phase, fd: u32, space: u32, entries: &[u8]) -> Result<(), Stop> {
        self.send(phase, GpuOp::Begin { space })?;
        for (rank, e) in entries.as_chunks::<8>().0.iter().map(|e| u64::from_le_bytes(*e)).enumerate() {
            if let Some((effects, read)) = self.decode(fd, space, rank, e)? {
                self.send_effects(phase, space, effects, read)?;
            }
        }
        Ok(())
    }

    /// Play mode: present `n`'s ops. The channels waiting for the CPU go on
    /// (behind the barrier) before the frame is shown; the rest of the present
    /// carries the values only this thread can read.
    fn present_ops(&mut self, phase: &Phase, n: u64, fence: (u32, u32), area: u32) -> Result<(), Stop> {
        self.send(phase, GpuOp::PresentBegin { n })?;
        for fd in self.waiting() {
            if let Some((space, effects)) = self.resume(fd)? {
                self.send_effects(phase, space, effects, None)?;
            }
        }
        let perf = self.frame_in();
        self.send(phase, GpuOp::PresentEnd { n, fence, area, perf })
    }

    /// A job's end, once its ops are `sent`: `End`, even after a decode stop
    /// (the job then counts as done: the drain, 213 and 102 wait for it).
    /// Then a present waits until the previous present is applied, and with
    /// --gpu-sync every job until its own ops are. The first stop in stream
    /// order wins: the GPU thread's comes from an earlier op than the decoder's.
    fn end_job(&mut self, phase: &Phase, present: bool, sent: Result<(), Stop>) -> Result<(), Stop> {
        let ended = self.send(phase, GpuOp::End);
        let Some(t) = self.gpu.as_mut() else { return sent.and(ended) };
        let since = Instant::now();
        let waited = if self.config.gpu_sync {
            let joined = t.join();
            self.gpu_wait_ns += since.elapsed().as_nanos() as u64;
            joined
        } else if present {
            let previous = std::mem::replace(&mut t.presented, t.sent);
            let applied = t.wait_for(previous);
            t.queue_wait_ns += since.elapsed().as_nanos() as u64;
            applied
        } else {
            Ok(())
        };
        waited.and(sent).and(ended)
    }
}

/// With the GPU thread, a read of `len` bytes at GPU address `at` in `space`
/// sees what the ops sent before it write, as without the GPU thread: it
/// waits until the last op sent that may write those bytes is applied. The
/// time waited; a stop is left for the next send to report.
fn barrier(gpu: Option<&GpuThread>, writes: &[(u64, Option<Ranges>)], maps: &crate::gs::Mappings, space: u32, at: u64, len: u64) -> Duration {
    let Some(t) = gpu else { return Duration::ZERO };
    let read = crate::gs::cpu_ranges(maps, space, at, len);
    let writer = writes.iter().rev().find(|(_, w)| w.as_ref().is_none_or(|w| w.iter().any(|&(s, e)| read.iter().any(|&(a, b)| s < b && a < e))));
    let Some(&(op, _)) = writer else { return Duration::ZERO };
    let since = Instant::now();
    let _ = t.wait_for(op + 1);
    since.elapsed()
}

impl GpuThread {
    /// Starts the GPU thread, with the guest threads' stack reservation:
    /// shader translation and validation run on it. After an op that can end
    /// a guest's wait on GPU work (syncpoint 1 moved, a submit ended), while
    /// one is armed, it wakes `boot`, which signals it under the kernel lock:
    /// the GPU thread never takes that lock.
    pub fn start(side: Arc<Mutex<GpuSide>>, boot: std::thread::Thread) -> std::io::Result<GpuThread> {
        let (ops, rx) = std::sync::mpsc::sync_channel::<(Phase, GpuOp)>(IN_FLIGHT);
        let progress = Arc::new(Progress::default());
        let p = progress.clone();
        let syncpts = crate::lock(&side).syncpts.clone();
        let thread = std::thread::Builder::new().name("gpu".into()).stack_size(gyroship_runtime::fiber::STACK_RESERVE).spawn(move || {
            /// However the thread ends (a panic too), its waiters wake and see it.
            struct Ended(Arc<Progress>);
            impl Drop for Ended {
                fn drop(&mut self) {
                    crate::lock(&self.0.state).ended = true;
                    self.0.changed.notify_all();
                }
            }
            let _ended = Ended(p.clone());
            let channel = &syncpts[CHANNEL_SYNCPT as usize];
            let mut count = channel.load(Ordering::SeqCst);
            for (n, (phase, op)) in (0u64..).zip(rx) {
                let job = matches!(op, GpuOp::End).then(|| phase.job());
                let stopped = crate::lock(&p.state).stop.is_some();
                let done = if stopped { Ok(()) } else { apply_op(&side, phase, op, n) };
                let mut state = crate::lock(&p.state);
                state.ops += 1;
                // ponytail: the last job ended, not each one before it: exact while one thread hands jobs over at a time (T10 at boot, then T17).
                if let Some(j) = job {
                    state.jobs = state.jobs.max(j);
                }
                if let Err(e) = done {
                    state.stop.get_or_insert(e);
                }
                drop(state);
                p.changed.notify_all();
                // A guest may wait for this (`gs.rs`, `arm`): the boot thread signals the waits reached.
                let now = channel.load(Ordering::SeqCst);
                if (now != count || job.is_some()) && p.armed.load(Ordering::SeqCst) > 0 {
                    boot.unpark();
                }
                count = now;
            }
        })?;
        Ok(GpuThread { ops, progress, sent: 0, presented: 0, queue_wait_ns: 0, thread })
    }

    /// Sends an op; the stop the thread met, if any, instead. A full channel
    /// waits for room, timed (`queue_wait_ns`).
    fn send(&mut self, phase: &Phase, op: GpuOp) -> Result<(), Stop> {
        use std::sync::mpsc::TrySendError;
        if let Some(e) = crate::lock(&self.progress.state).stop.clone() {
            return Err(e);
        }
        match self.ops.try_send((phase.clone(), op)) {
            Ok(()) => {}
            Err(TrySendError::Full(op)) => {
                let since = Instant::now();
                self.ops.send(op).map_err(|_| gpu_thread_ended())?;
                self.queue_wait_ns += since.elapsed().as_nanos() as u64;
            }
            Err(TrySendError::Disconnected(_)) => return Err(gpu_thread_ended()),
        }
        self.sent += 1;
        Ok(())
    }

    /// Waits until every op sent so far is applied; the stop the thread met, if any.
    fn join(&self) -> Result<(), Stop> {
        self.wait_for(self.sent)
    }

    /// Waits until `n` ops are applied; the stop the thread met, if any.
    fn wait_for(&self, n: u64) -> Result<(), Stop> {
        // A target never sent would wait for good, with no lock the watchdog sees held.
        debug_assert!(n <= self.sent, "waiting for op {n} of {} sent", self.sent);
        let mut state = crate::lock(&self.progress.state);
        while state.ops < n && state.stop.is_none() && !state.ended {
            state = self.progress.changed.wait(state).unwrap_or_else(std::sync::PoisonError::into_inner);
        }
        match &state.stop {
            Some(e) => Err(e.clone()),
            None if state.ops < n => Err(gpu_thread_ended()),
            None => Ok(()),
        }
    }

    /// The thread's CPU time so far (`--perf`; Windows only).
    pub fn cpu_time(&self) -> Option<std::time::Duration> {
        #[cfg(windows)]
        {
            use std::os::windows::io::AsRawHandle;
            gyroship_runtime::profile::cpu_time(self.thread.as_raw_handle() as usize)
        }
        #[cfg(not(windows))]
        None
    }
}

/// On the GPU thread: op number `n`, handed over at `phase`, under the GPU
/// side's lock, which the play engine's watchdog sees held.
fn apply_op(side: &Mutex<GpuSide>, phase: Phase, op: GpuOp, n: u64) -> Result<(), Stop> {
    let mut g = crate::lock(side);
    crate::GPU_SINCE.store(crate::play::real_ticks(), Ordering::Relaxed);
    crate::GPU_GUEST.store(phase.thread, Ordering::Relaxed);
    crate::GPU_HOLDER.store(gyroship_runtime::profile::thread_id(), Ordering::Relaxed);
    g.at = phase;
    g.applying = n;
    // An anomaly's stop names its op: the run may stop on it a job later, or from the boot thread.
    let done = g.apply(op).map_err(|(code, line)| match code {
        crate::STOP_EXIT => (code, format!("{line} (GPU op {n}, handed over by thread {} at tick {}, present {})", g.at.thread, g.at.now, g.at.presents)),
        _ => (code, line),
    });
    // The mappings' snapshot goes: a change to them need not copy them.
    g.at = Phase::default();
    crate::GPU_HOLDER.store(0, Ordering::Relaxed);
    done
}

impl Kernel {
    /// Play mode: from now on the GPU thread applies the submits' ops. Called
    /// on the boot thread, which then runs `play::run_play`: the GPU thread
    /// wakes it for the guests' waits on GPU work.
    pub fn start_gpu_thread(&mut self) -> std::io::Result<()> {
        let t = GpuThread::start(self.gpu_side.clone(), std::thread::current())?;
        self.gpu_side().wrote = Some(std::collections::VecDeque::new());
        self.gpu_progress = Some(t.progress.clone());
        crate::lock(&self.decoder).gpu = Some(t);
        // Which mode a measurement ran in.
        self.log(if self.config.gpu_sync { "gpu: the GPU thread applies the GPU work; a job returns once its ops are applied (--gpu-sync)" } else { "gpu: the GPU thread applies the GPU work; a job returns once its ops are sent, with one present in flight" });
        Ok(())
    }

    /// Hands GPU work over to the GPU side, where it runs under the GPU lock
    /// alone; then what it leaves (`After`) runs under the kernel lock. The
    /// deterministic engine runs it at once. The play engine queues it
    /// (`Deferred`, Stage 4): the caller's host thread runs it once the SVC's
    /// reply is written and the kernel lock released, before its guest code
    /// goes on, while the other threads use the kernel.
    pub(crate) fn gpu_work(&mut self, job: GpuJob) {
        let phase = Phase { now: self.now(), thread: self.sched.current.map_or(0, |t| self.sched.threads[t].id), presents: self.gs.presents, submits: self.tally().counts.get("submits").copied().unwrap_or(0), maps: self.gs.maps.clone() };
        if self.config.play && self.deferred.is_none() {
            self.deferred = Some(crate::Deferred::Gpu { side: self.gpu_side.clone(), decoder: self.decoder.clone(), phase, job });
            return;
        }
        // ponytail: in play mode this would run a second GPU job in one SVC, under the kernel lock,
        // and none hands one over (one IPC request per SVC). Its barrier, full channel or present
        // wait would then stall every guest thread, though never deadlock: the GPU thread never
        // takes that lock. Queue a second job if a game ever does that.
        let done = run(&self.gpu_side, &self.decoder, phase, job);
        self.after(done);
    }

    /// The GPU jobs handed over so far: the last one's `Phase::job`.
    pub(crate) fn gpu_jobs(&self) -> u64 {
        let submits = self.tally().counts.get("submits").copied().unwrap_or(0);
        self.gs.presents + submits
    }

    /// Play mode: waits until the GPU thread has done every job handed over
    /// (but this scope's own, never run once the run stops), met a stop or
    /// ended, GPU_WAIT at most; the stop it met, if any. A wait cut short is
    /// journaled: what the GPU thread still had to do may be missing.
    pub(crate) fn drain_gpu(&mut self) -> Option<Stop> {
        let p = self.gpu_progress.clone()?;
        let handed = self.gpu_jobs() - u64::from(matches!(self.deferred, Some(crate::Deferred::Gpu { .. })));
        let (stop, done) = p.drain(handed, crate::GPU_WAIT);
        if stop.is_none() && done < handed {
            let line = format!("gpu: the GPU thread had done {done} of {handed} jobs after {} s: the last frames' lines, rows and hashes may be missing", crate::GPU_WAIT.as_secs());
            eprintln!("gyroship: {line}");
            self.log(&line);
        }
        stop
    }
}

/// Runs `job`, handed over at `phase`: the decoder decodes it and the GPU
/// side performs it, op by op (decoder lock first), on the GPU thread if
/// there is one; what it leaves for the kernel lock.
pub fn run(side: &Mutex<GpuSide>, decoder: &Mutex<Decoder>, phase: Phase, job: GpuJob) -> Result<After, Stop> {
    let at = phase.now;
    let mut d = crate::lock(decoder);
    d.at = phase.clone();
    let done = if d.gpu.is_some() {
        // Play mode: the GPU thread applies the job's ops while this thread decodes.
        let sent = match &job {
            GpuJob::Submit { fd, space, entries } => d.submit_ops(&phase, *fd, *space, entries),
            GpuJob::Present { n, fence, area, .. } => d.present_ops(&phase, *n, *fence, *area),
        };
        d.end_job(&phase, matches!(job, GpuJob::Present { .. }), sent)
    } else {
        let mut g = crate::lock(side);
        // The play engine's watchdog sees who holds the GPU side, and since when.
        let play = g.config.play;
        if play {
            crate::GPU_SINCE.store(crate::play::real_ticks(), Ordering::Relaxed);
            crate::GPU_GUEST.store(phase.thread, Ordering::Relaxed);
            crate::GPU_HOLDER.store(gyroship_runtime::profile::thread_id(), Ordering::Relaxed);
        }
        g.at = phase;
        let done = match &job {
            GpuJob::Submit { fd, space, entries } => run_gpfifo(&mut d, &mut g, *fd, *space, entries),
            GpuJob::Present { n, fence, area, .. } => g.present(&mut d, *n, *fence, *area),
        };
        // The mappings' snapshot goes: a change to them need not copy them.
        g.at = Phase::default();
        if play {
            crate::GPU_HOLDER.store(0, Ordering::Relaxed);
        }
        done
    };
    d.at = Phase::default();
    drop(d);
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
        k.census_ipc(|| format!("nvdrv {label} {code:#x}"));
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
                // With the GPU thread, that fence would be reached before the work it fences:
                // syncpoint 1 would run ahead of the GPU thread for good. NVN never asks (flags
                // 0x104). Before the count: a strict stop must leave no job the drain waits for.
                if flags & 2 != 0 && k.gpu_progress.is_some() {
                    k.anomaly("gpu submit fence", &format!("channel fd {fd}: submit flags {flags:#x} ask for a fence, which would be reached before the GPU thread applies the work"));
                }
                let n = k.count("submits");
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

/// Runs a submit's GPFIFO entries through the channel's command processor,
/// as far as its semaphore acquires allow, and performs what they did: a
/// `Begin` op, then the effects of each entry once it is decoded.
fn run_gpfifo(d: &mut Decoder, k: &mut GpuSide, fd: u32, space: u32, entries: &[u8]) -> Result<(), Stop> {
    k.apply(GpuOp::Begin { space })?;
    for (rank, e) in entries.as_chunks::<8>().0.iter().map(|e| u64::from_le_bytes(*e)).enumerate() {
        if let Some((effects, _)) = d.decode(fd, space, rank, e)? {
            k.apply(GpuOp::Effects { space, effects, read: None })?;
        }
    }
    Ok(())
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
    rewritten(k, written);
}

/// Drops the textures (GPU start, GPU end) the CPU wrote.
fn rewritten(k: &mut GpuSide, written: Vec<(u64, u64)>) {
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

/// After a copy engine write of `len` bytes at GPU address `start` (whose
/// textures `Surfaces::invalidate` already drops): its pages' write watch is
/// reset, or the next submit's take would see our own write as the CPU's and
/// drop the textures uploaded afresh from it (a streaming pool, uploaded twice
/// per write). Any other texture on a reset page (sharing one, or the same
/// memory at another GPU address) is dropped as that take would drop it.
fn forget_copied(k: &mut GpuSide, space: u32, start: u64, len: u64) {
    let pages = k.mem.reset_written(k.gpu_to_cpu(space, start), len);
    if pages.is_empty() {
        return;
    }
    // Leaves out the textures the copy drops (pending invalidations).
    let Some(ranges) = k.surfaces.as_ref().map(|s| s.texture_ranges()) else { return };
    let written = ranges.into_iter().filter(|&(a, b)| touches(&pages, k.gpu_to_cpu(space, a), b - a)).collect();
    rewritten(k, written);
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
/// released it (at each present without the GPU thread, `present_ops` with
/// it; a submit resumes its own channel).
pub fn poll(d: &mut Decoder, k: &mut GpuSide) -> Result<(), Stop> {
    for fd in d.waiting() {
        if let Some((space, effects)) = d.resume(fd)? {
            k.apply(GpuOp::Effects { space, effects, read: None })?;
        }
    }
    Ok(())
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
    } else {
        k.note_write(cpu, bytes.len());
        if let Some(s) = k.surfaces.as_mut() {
            s.invalidate(r.addr, bytes.len() as u64);
        }
    }
    if k.journal.few("gpu semaphore release", 8) {
        k.log(&format!("gpu: semaphore {r:x?}"));
    }
    Ok(())
}

impl Decoder {
    /// Logs a channel starting or stopping a wait on a semaphore (the first ones).
    fn note_wait(&self, fd: u32, space: u32, before: Option<gyroship_gpu::pushbuf::Acquire>, after: Option<gyroship_gpu::pushbuf::Acquire>) {
        if before == after || !self.journal.few("gpu wait", 48) {
            return;
        }
        match after {
            Some(a) => {
                let mut b = [0u8; 8];
                let read = crate::gs::read_gpu(&self.mem, &self.at.maps, space, a.addr, &mut b[..if a.wide { 8 } else { 4 }]);
                let now = if read { format!("{:#x}", u64::from_le_bytes(b)) } else { "unreadable".into() };
                let cpu = crate::gs::gpu_to_cpu(&self.at.maps, space, a.addr);
                let map = self.at.maps.list.iter().rev().find(|m| m.0 == space && (m.1..=m.2).contains(&a.addr)).copied();
                self.log(&format!("gpu: channel fd {fd} waits until {:#x} (cpu {cpu:#x}, mapping {map:x?}) is {:?} {:#x} (now {now})", a.addr, a.op, a.value));
            }
            None => self.log(&format!("gpu: channel fd {fd} goes on")),
        }
    }
}

impl Gpu {
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
                        // SeqCst: the GPU thread then reads the armed count, and the kernel
                        // reads this after publishing it (gs.rs `signal_reached`): one sees the other.
                        k.syncpts[CHANNEL_SYNCPT as usize].fetch_add(1, Ordering::SeqCst);
                        k.count("syncpoint increments");
                    } else if k.mem.publish_object(at, width, &bytes[width..], |_| r.value).is_err() {
                        k.anomaly(if r.addr == 0 { "gpu release at 0" } else { "gpu release not writable" }, &what)?;
                    } else {
                        k.note_write(at, bytes.len());
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
                    // How far past dst the lines go: past the span below when PITCH_OUT is
                    // stale or zero (a 1D copy), or the region leaves the surface.
                    let mut reach = 0;
                    for i in 0..c.line_count {
                        if !ok {
                            break;
                        }
                        ok = k.read_gpu(space, c.src + u64::from(i) * u64::from(c.pitch_in), &mut line);
                        match &block {
                            None => {
                                let off = u64::from(i) * u64::from(c.pitch_out);
                                let at = k.gpu_to_cpu(space, c.dst + off);
                                ok = ok && k.mem.write_object(at, &line).is_ok();
                                reach = reach.max(off + line.len() as u64);
                            }
                            Some((b, l)) => {
                                let mut j = 0;
                                while ok && j < line.len() {
                                    let x = b.x + j as u32;
                                    let n = ((16 - x % 16) as usize).min(line.len() - j);
                                    let off = l.offset(x, b.y + i, b.layer);
                                    let at = k.gpu_to_cpu(space, c.dst + off);
                                    ok = k.mem.write_object(at, &line[j..j + n]).is_ok();
                                    reach = reach.max(off + n as u64);
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
                    k.note_write(k.gpu_to_cpu(space, c.dst), reach as usize);
                    if let Some(s) = &mut k.surfaces {
                        s.invalidate(c.dst, span);
                        // A refresh in place takes every byte written (the rest of the
                        // cache sees the span alone, as before: same drops, same takes).
                        if reach > span {
                            s.mark_dirty(c.dst, reach);
                        }
                    }
                    forget_copied(k, space, c.dst, span);
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
                    k.note_write(k.gpu_to_cpu(space, w.address), w.data.len());
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
        k.gpu_side().apply(GpuOp::Effects { space: 0, effects, read: None }).unwrap();
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

    /// The copy engine's writes are the GPU's: the next submit's take does not
    /// report them (it would upload again the textures they fed), a CPU write
    /// after them still shows.
    #[test]
    fn a_copy_is_not_taken_for_a_cpu_write() {
        use gyroship_runtime::mem::{Block, Memory, perm, state};
        let page = crate::layout::ALIAS.0;
        let k = Kernel::new(Memory::new().unwrap(), crate::journal::Journal::memory(), crate::Config::default());
        k.mem.map(page, 0x3000, Block::new(state::NORMAL, perm::RW)).unwrap();
        k.mem.write(page, &[7; 0x100]).unwrap();
        k.mem.take_written(page, 0x3000);
        // 0x20 lines of the same 0x100 bytes into pages 1 and 2.
        let copy = gyroship_gpu::pushbuf::Copy { src: page, dst: page + 0x1000, pitch_in: 0, pitch_out: 0x100, line_length: 0x100, line_count: 0x20, dst_block: None };
        k.gpu_side().apply(GpuOp::Effects { space: 0, effects: vec![Effect::Copy(copy)], read: None }).unwrap();
        let mut b = [0u8; 0x100];
        k.mem.read(page + 0x2f00, &mut b).unwrap();
        assert_eq!(b, [7; 0x100], "copied");
        assert!(k.mem.take_written(page, 0x3000).is_empty());
        k.mem.write(page + 0x1010, &[1]).unwrap();
        assert_eq!(k.mem.take_written(page, 0x3000), [page + 0x1000]);
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
        let done = k.gpu_side().apply(GpuOp::Effects { space: 0, effects, read: None });
        assert_eq!(done, Err((crate::STOP_EXIT, "anomaly gpu method ignored: class 0xc797 method 0x123 = 0x1 not executed".into())));
        let mut b = [9u8; 4];
        k.mem.read(page, &mut b).unwrap();
        assert_eq!(b, [0; 4], "the write after the anomaly is not applied");
        assert_eq!(k.tally().anomalies["gpu method ignored"], 1);
    }

    /// The play engine leaves GPU work for after the kernel lock, as the SVC's
    /// deferred work; the deterministic engine runs it at once. Here a
    /// present's GPU part finds its fence not reached (with the GPU thread,
    /// that thread checks it), then its rest runs.
    #[test]
    fn the_play_engine_runs_gpu_work_after_the_kernel_lock() {
        use gyroship_runtime::mem::Memory;
        for (play, threaded) in [(false, false), (true, false), (true, true)] {
            let config = crate::Config { play, explore: true, vsync: false, ..crate::Config::default() };
            let mut k = Kernel::new(Memory::new().unwrap(), crate::journal::Journal::memory(), config);
            if threaded {
                k.start_gpu_thread().unwrap();
            }
            // As vi2 229 does: the present is counted, then handed over.
            k.gs.presents = 1;
            k.gpu_work(GpuJob::Present { n: 1, fence: (!CHANNEL_SYNCPT, 5), area: 0, interval: 1 });
            let ran = |k: &Kernel| k.tally().anomalies.get("present fence not reached").copied();
            if play {
                assert_eq!(ran(&k), None, "queued, not run");
                let done = k.deferred.take().expect("a deferred job").run();
                assert!(matches!(done, Ok(After::Present { n: 1, interval: 1, at: 0 })));
                k.after(done);
            }
            assert!(k.deferred.is_none());
            assert_eq!(k.drain_gpu(), None);
            assert_eq!(ran(&k), Some(1));
        }
    }

    /// A method header: operation, count, subchannel, byte offset.
    fn header(op: u32, count: u32, subchannel: u32, method: u32) -> u32 {
        op << 29 | count << 16 | subchannel << 13 | (method / 4)
    }

    /// A host semaphore (SEMAPHOREA..D): `value` released at `addr` (one
    /// word), or acquired (until the word is at least `value`).
    fn semaphore(addr: u64, value: u32, release: bool) -> Vec<u32> {
        vec![header(1, 4, 0, 0x10), (addr >> 32) as u32, addr as u32, value, if release { 0x0110_0002 } else { 4 }]
    }

    /// Where `play_submit` puts its two semaphore words and its pushbuffers.
    const WORDS: u64 = crate::layout::ALIAS.0;
    const PUSHBUFFERS: u64 = WORDS + 0x100;

    /// A play-mode kernel, strict unless `explore`, with the GPU thread if
    /// `threaded` (joined with `sync`, as --gpu-sync), and the page at `WORDS` mapped.
    fn play_kernel(threaded: bool, explore: bool, sync: bool) -> Kernel {
        use gyroship_runtime::mem::{Block, Memory, perm, state};
        let config = crate::Config { play: true, explore, vsync: false, gpu_sync: sync, ..crate::Config::default() };
        let mut k = Kernel::new(Memory::new().unwrap(), crate::journal::Journal::memory(), config);
        k.mem.map(WORDS, 0x1000, Block::new(state::NORMAL, perm::RW)).unwrap();
        if threaded {
            k.start_gpu_thread().unwrap();
        }
        k
    }

    /// A submit on channel `fd` of `segments` (a GPFIFO entry each, written
    /// from `at`), handed over as the ioctl does: the job to run.
    fn hand_over(k: &mut Kernel, fd: u32, mut at: u64, segments: &[Vec<u32>]) -> crate::Deferred {
        let mut entries = Vec::new();
        for words in segments {
            let bytes: Vec<u8> = words.iter().flat_map(|w| w.to_le_bytes()).collect();
            k.mem.write(at, &bytes).unwrap();
            entries.extend((at | (words.len() as u64) << 42).to_le_bytes());
            at += bytes.len() as u64;
        }
        // The submit's number goes into its Phase.
        k.count("submits");
        k.gpu_work(GpuJob::Submit { fd, space: 0, entries });
        k.deferred.take().expect("a deferred job")
    }

    /// The next present, with acquire `fence`, handed over as vi2 229 does: the job to run.
    fn hand_present(k: &mut Kernel, fence: (u32, u32)) -> crate::Deferred {
        k.gs.presents += 1;
        k.gpu_work(GpuJob::Present { n: k.gs.presents, fence, area: 0, interval: 1 });
        k.deferred.take().expect("a deferred job")
    }

    /// One submit of `segments` on channel 1 (`hand_over` at `PUSHBUFFERS`),
    /// not joined: the kernel, and the job to run.
    fn play_submit(threaded: bool, explore: bool, segments: &[Vec<u32>]) -> (Kernel, crate::Deferred) {
        let mut k = play_kernel(threaded, explore, false);
        let job = hand_over(&mut k, 1, PUSHBUFFERS, segments);
        (k, job)
    }

    fn word(k: &Kernel, at: u64) -> u32 {
        let mut b = [0u8; 4];
        k.mem.read(at, &mut b).unwrap();
        u32::from_le_bytes(b)
    }

    /// Whether the thread running a job finishes within 5 s (a busy machine
    /// may take a while).
    fn finishes<T>(running: &std::thread::JoinHandle<T>) -> bool {
        let since = Instant::now();
        while !running.is_finished() && since.elapsed() < std::time::Duration::from_secs(5) {
            std::thread::sleep(std::time::Duration::from_millis(1));
        }
        running.is_finished()
    }

    /// The GPU thread applies what the submitting thread decodes, in order:
    /// guest memory and the tally end as when one thread does both, joined
    /// or not.
    #[test]
    fn the_gpu_thread_leaves_what_one_thread_leaves() {
        let segments = [semaphore(WORDS, 1, true), [semaphore(WORDS, 1, false), semaphore(WORDS + 8, 2, true)].concat()];
        for (threaded, sync) in [(false, false), (true, true), (true, false)] {
            let mut k = play_kernel(threaded, false, sync);
            let job = hand_over(&mut k, 1, PUSHBUFFERS, &segments);
            assert!(matches!(job.run(), Ok(After::Nothing)));
            assert_eq!(k.drain_gpu(), None);
            assert_eq!((word(&k, WORDS), word(&k, WORDS + 8)), (1, 2), "threaded: {threaded}, sync: {sync}");
            assert!(k.tally().anomalies.is_empty());
        }
    }

    /// In play mode a job returns once its ops are sent, before the GPU
    /// thread applies them; with --gpu-sync once they are applied.
    #[test]
    fn a_job_returns_before_its_ops_are_applied() {
        for sync in [false, true] {
            let mut k = play_kernel(true, false, sync);
            let job = hand_over(&mut k, 1, PUSHBUFFERS, &[semaphore(WORDS, 1, true)]);
            // The GPU thread applies nothing while the test holds the GPU side.
            let side = k.gpu_side.clone();
            let held = crate::lock(&side);
            let running = std::thread::spawn(move || job.run());
            if sync {
                std::thread::sleep(std::time::Duration::from_millis(100));
                assert!(!running.is_finished(), "--gpu-sync: the job waits until its ops are applied");
            } else {
                assert!(finishes(&running), "the job returns with its ops queued");
                assert_eq!(word(&k, WORDS), 0);
            }
            drop(held);
            assert!(matches!(running.join().unwrap(), Ok(After::Nothing)));
            assert_eq!(k.drain_gpu(), None);
            assert_eq!(word(&k, WORDS), 1, "sync: {sync}");
        }
    }

    /// An acquire reads what a job before it released, still queued: its
    /// read waits until that release is applied.
    #[test]
    fn an_acquire_waits_for_a_release_of_the_job_before() {
        let mut k = play_kernel(true, false, false);
        let first = hand_over(&mut k, 1, PUSHBUFFERS, &[semaphore(WORDS, 1, true)]);
        let second = hand_over(&mut k, 1, PUSHBUFFERS + 0x100, &[[semaphore(WORDS, 1, false), semaphore(WORDS + 8, 2, true)].concat()]);
        let side = k.gpu_side.clone();
        let held = crate::lock(&side);
        assert!(matches!(first.run(), Ok(After::Nothing)), "returns with its ops queued");
        let running = std::thread::spawn(move || second.run());
        std::thread::sleep(std::time::Duration::from_millis(100));
        assert!(!running.is_finished(), "its acquire waits for the release queued before it");
        drop(held);
        assert!(matches!(running.join().unwrap(), Ok(After::Nothing)));
        assert_eq!(k.drain_gpu(), None);
        assert_eq!(word(&k, WORDS + 8), 2);
        assert!(crate::lock(&k.decoder).waiting().is_empty());
        assert!(k.tally().anomalies.is_empty());
    }

    /// One present in flight: a present returns once the previous one is
    /// applied (the first at once); with --gpu-sync once it is applied itself.
    #[test]
    fn a_present_waits_for_the_present_before_it() {
        for sync in [false, true] {
            let mut k = play_kernel(true, false, sync);
            let first = hand_present(&mut k, (!CHANNEL_SYNCPT, 0));
            let second = hand_present(&mut k, (!CHANNEL_SYNCPT, 0));
            let side = k.gpu_side.clone();
            let held = crate::lock(&side);
            let first = std::thread::spawn(move || first.run());
            if sync {
                std::thread::sleep(std::time::Duration::from_millis(100));
                assert!(!first.is_finished(), "--gpu-sync: present 1 waits until it is applied");
                drop(held);
                assert!(matches!(first.join().unwrap(), Ok(After::Present { n: 1, .. })));
                continue;
            }
            assert!(finishes(&first), "present 1 returns with its ops queued");
            let second = std::thread::spawn(move || second.run());
            std::thread::sleep(std::time::Duration::from_millis(100));
            assert!(!second.is_finished(), "present 2 waits until present 1 is applied");
            drop(held);
            assert!(matches!(first.join().unwrap(), Ok(After::Present { n: 1, .. })));
            assert!(matches!(second.join().unwrap(), Ok(After::Present { n: 2, .. })));
            assert_eq!(k.drain_gpu(), None);
        }
    }

    /// A present's acquire fence is checked once its frame's ops are applied:
    /// NVN's increment, queued in a submit before it, counts.
    #[test]
    fn the_present_fence_is_checked_after_the_frames_ops() {
        let mut k = play_kernel(true, true, false);
        let submit = hand_over(&mut k, 1, PUSHBUFFERS, &[semaphore(0, 1, true)]);
        let present = hand_present(&mut k, (!CHANNEL_SYNCPT, 1));
        let side = k.gpu_side.clone();
        let held = crate::lock(&side);
        assert!(matches!(submit.run(), Ok(After::Nothing)));
        assert!(matches!(present.run(), Ok(After::Present { n: 1, .. })));
        assert_eq!(syncpoint(&k), 0, "nothing applied yet");
        drop(held);
        assert_eq!(k.drain_gpu(), None);
        assert_eq!(syncpoint(&k), 1);
        assert_eq!(k.tally().anomalies.get("present fence not reached"), None);
    }

    /// A stop the GPU thread meets after its job returned reaches the drain,
    /// the boot round and the next job.
    #[test]
    fn a_gpu_thread_stop_reaches_the_next_job() {
        let mut k = play_kernel(true, false, false);
        // SEMAPHORED with an operation no one implements: an ignored method.
        let first = hand_over(&mut k, 1, PUSHBUFFERS, &[vec![header(1, 1, 0, 0x1c), 0x10], semaphore(WORDS + 8, 2, true)]);
        let side = k.gpu_side.clone();
        let held = crate::lock(&side);
        assert!(matches!(first.run(), Ok(After::Nothing)), "returns before the GPU thread meets the anomaly");
        drop(held);
        let stop = k.drain_gpu().expect("the GPU thread's stop");
        assert!(stop.1.contains("gpu method ignored"), "{stop:?}");
        assert_eq!(word(&k, WORDS + 8), 0, "the release after the anomaly is not applied");
        assert_eq!(k.gpu_progress.as_ref().and_then(|p| p.stopped()).as_ref(), Some(&stop));
        let next = hand_over(&mut k, 1, PUSHBUFFERS + 0x100, &[semaphore(WORDS, 1, true)]);
        assert_eq!(next.run().err(), Some(stop));
    }

    /// An acquire reads what the entries before it released, as with one
    /// thread: decoding it waits until the GPU thread has applied them.
    #[test]
    fn an_acquire_waits_for_the_ops_sent_before_it() {
        let segments = [semaphore(WORDS, 1, true), [semaphore(WORDS, 1, false), semaphore(WORDS + 8, 2, true)].concat()];
        let (mut k, job) = play_submit(true, false, &segments);
        // The GPU thread cannot apply anything while the test holds the GPU side.
        let side = k.gpu_side.clone();
        let held = crate::lock(&side);
        let running = std::thread::spawn(move || job.run());
        std::thread::sleep(std::time::Duration::from_millis(100));
        assert!(!running.is_finished());
        drop(held);
        assert!(matches!(running.join().unwrap(), Ok(After::Nothing)));
        assert_eq!(k.drain_gpu(), None);
        assert_eq!(word(&k, WORDS + 8), 2, "the acquire was met: the release after it ran");
        assert!(crate::lock(&k.decoder).waiting().is_empty());
    }

    /// In strict mode an anomaly on the GPU thread ends the submit there: the
    /// ops after it are not applied, and the joined submit (--gpu-sync)
    /// returns its stop, which names its op. The drain and the boot round see
    /// the same stop at once.
    #[test]
    fn a_strict_anomaly_on_the_gpu_thread_ends_the_submit() {
        // SEMAPHORED with an operation no one implements: an ignored method.
        let segments = [vec![header(1, 1, 0, 0x1c), 0x10], semaphore(WORDS + 8, 2, true)];
        let mut k = play_kernel(true, false, true);
        let job = hand_over(&mut k, 1, PUSHBUFFERS, &segments);
        let Err(stop) = job.run() else { panic!("no stop") };
        assert!(stop.0 == crate::STOP_EXIT && stop.1.contains("gpu method ignored") && stop.1.contains("(GPU op 1, handed over by thread 0 at tick 0, present 0)"), "{stop:?}");
        assert_eq!(word(&k, WORDS + 8), 0, "the release after the anomaly is not applied");
        assert_eq!(k.tally().anomalies["gpu method ignored"], 1);
        let since = Instant::now();
        assert_eq!(k.drain_gpu().as_ref(), Some(&stop));
        assert_eq!(k.gpu_progress.as_ref().and_then(|p| p.stopped()).as_ref(), Some(&stop));
        assert!(since.elapsed() < std::time::Duration::from_secs(1));
    }

    /// The drain waits for every job handed over, even one whose ops are not
    /// sent yet: here the submit runs 100 ms after the drain starts.
    #[test]
    fn a_drain_waits_for_a_job_handed_over_but_not_yet_run() {
        let (mut k, job) = play_submit(true, false, &[semaphore(WORDS, 1, true)]);
        let running = std::thread::spawn(move || {
            std::thread::sleep(std::time::Duration::from_millis(100));
            job.run()
        });
        let since = Instant::now();
        assert_eq!(k.drain_gpu(), None);
        assert!(since.elapsed() >= std::time::Duration::from_millis(90));
        assert_eq!(word(&k, WORDS), 1);
        assert!(matches!(running.join().unwrap(), Ok(After::Nothing)));
    }

    /// A submit that a decode stop ends still ends its job: the drain does
    /// not wait GPU_WAIT for it, and the ops sent before the stop are applied.
    #[test]
    fn a_decode_stop_still_ends_its_job() {
        let mut k = play_kernel(true, false, false);
        let words = semaphore(WORDS, 1, true);
        k.mem.write(PUSHBUFFERS, &words.iter().flat_map(|w| w.to_le_bytes()).collect::<Vec<u8>>()).unwrap();
        // The second entry's pushbuffer is not mapped.
        let entries = [PUSHBUFFERS | (words.len() as u64) << 42, 0x7000_0000 | 4 << 42].iter().flat_map(|e| e.to_le_bytes()).collect();
        k.count("submits");
        k.gpu_work(GpuJob::Submit { fd: 1, space: 0, entries });
        let done = k.deferred.take().expect("a deferred job").run();
        assert!(matches!(done, Err((crate::STOP_EXIT, line)) if line.contains("gpfifo unreadable")));
        let since = Instant::now();
        assert_eq!(k.drain_gpu(), None);
        assert!(since.elapsed() < std::time::Duration::from_secs(1));
        assert_eq!(word(&k, WORDS), 1);
    }

    /// A GPU write into a pushbuffer decoded before the write was applied
    /// (the GPU thread's read-ahead) is an anomaly; one thread sees the write.
    #[test]
    fn a_gpu_write_into_a_pushbuffer_read_ahead_is_an_anomaly() {
        // The first entry releases 7 over the second one's payload word (its 4th).
        let second = PUSHBUFFERS + 5 * 4;
        let segments = [semaphore(second + 3 * 4, 7, true), semaphore(WORDS + 8, 2, true)];
        for threaded in [false, true] {
            let (mut k, job) = play_submit(threaded, true, &segments);
            // The GPU thread applies nothing before the second pushbuffer is read.
            let side = k.gpu_side.clone();
            let held = threaded.then(|| crate::lock(&side));
            let running = std::thread::spawn(move || job.run());
            std::thread::sleep(std::time::Duration::from_millis(100));
            drop(held);
            assert!(matches!(running.join().unwrap(), Ok(After::Nothing)));
            assert_eq!(k.drain_gpu(), None);
            let hits = k.tally().anomalies.get("gpu wrote a pushbuffer read ahead").copied();
            if threaded {
                assert_eq!(hits, Some(1));
            } else {
                assert_eq!((hits, word(&k, WORDS + 8)), (None, 7));
            }
        }
    }

    /// A write that a barrier waited for before the pushbuffer it lands in
    /// was read is seen by that read: no anomaly.
    #[test]
    fn a_gpu_write_seen_by_the_decoder_is_no_anomaly() {
        // The second entry acquires what the first releases (into the third's
        // payload word): decoding it waits for the first's op.
        let third = PUSHBUFFERS + 5 * 4 + 10 * 4;
        let segments = [semaphore(third + 3 * 4, 7, true), [semaphore(third + 3 * 4, 7, false), semaphore(WORDS, 1, true)].concat(), semaphore(WORDS + 8, 2, true)];
        let (mut k, job) = play_submit(true, false, &segments);
        assert!(matches!(job.run(), Ok(After::Nothing)));
        assert_eq!(k.drain_gpu(), None);
        assert!(k.tally().anomalies.is_empty());
        assert_eq!(word(&k, WORDS + 8), 7, "the third entry decoded the released payload");
    }

    /// The GPU side checks each pushbuffer read when it applies the op made
    /// from it, across submits: a write applied after the read is caught, one
    /// applied before it is not.
    #[test]
    fn a_write_into_a_pushbuffer_read_before_it_is_caught_across_submits() {
        use gyroship_gpu::state::resources::MemWrite;
        use gyroship_runtime::mem::{Block, Memory, perm, state};
        let k = Kernel::new(Memory::new().unwrap(), crate::journal::Journal::memory(), crate::Config::default());
        k.mem.map(WORDS, 0x1000, Block::new(state::NORMAL, perm::RW)).unwrap();
        let mut g = k.gpu_side();
        g.wrote = Some(std::collections::VecDeque::new());
        let mut apply = |n: u64, op: GpuOp| {
            g.applying = n;
            g.apply(op)
        };
        let read = |applied| Some((applied, WORDS, WORDS + 0x20));
        let write = vec![Effect::Write(MemWrite { address: WORDS + 0x10, data: vec![1; 4] })];
        apply(1, GpuOp::Effects { space: 0, effects: write, read: None }).unwrap();
        apply(2, GpuOp::Begin { space: 0 }).unwrap();
        // Read with op 0 alone applied: op 1's write landed in it after.
        let missed = apply(3, GpuOp::Effects { space: 0, effects: vec![], read: read(1) });
        assert!(matches!(missed, Err((crate::STOP_EXIT, line)) if line.contains("gpu wrote a pushbuffer read ahead")));
        // Read once op 1 was applied: it saw the write.
        assert_eq!(apply(4, GpuOp::Effects { space: 0, effects: vec![], read: read(2) }), Ok(()));
    }

    /// A channel resumed at a present reads what a channel resumed before it
    /// released: with the GPU thread, its read waits until that is applied.
    #[test]
    fn a_resumed_channel_sees_what_the_channel_resumed_before_it_released() {
        let (w1, w2, w3) = (WORDS + 0x10, WORDS + 0x20, WORDS + 0x30);
        for threaded in [false, true] {
            let mut k = play_kernel(threaded, false, false);
            let first = hand_over(&mut k, 1, PUSHBUFFERS, &[[semaphore(w1, 1, false), semaphore(w2, 1, true)].concat()]);
            assert!(matches!(first.run(), Ok(After::Nothing)));
            let second = hand_over(&mut k, 2, PUSHBUFFERS + 0x100, &[[semaphore(w2, 1, false), semaphore(w3, 1, true)].concat()]);
            assert!(matches!(second.run(), Ok(After::Nothing)));
            assert_eq!(crate::lock(&k.decoder).waiting(), [1, 2]);
            k.mem.write(w1, &1u32.to_le_bytes()).unwrap();
            let present = hand_present(&mut k, (!CHANNEL_SYNCPT, 0));
            // The GPU thread applies nothing for 100 ms.
            let side = k.gpu_side.clone();
            let held = threaded.then(|| crate::lock(&side));
            let running = std::thread::spawn(move || present.run());
            std::thread::sleep(std::time::Duration::from_millis(100));
            drop(held);
            assert!(matches!(running.join().unwrap(), Ok(After::Present { n: 1, .. })), "threaded: {threaded}");
            assert_eq!(k.drain_gpu(), None);
            assert_eq!(word(&k, w3), 1, "threaded: {threaded}: channel 2 read channel 1's release");
            assert!(crate::lock(&k.decoder).waiting().is_empty());
            assert!(k.tally().anomalies.is_empty());
        }
    }

    /// A TIPC request to a gsv session: `command` and its arguments.
    fn tipc(command: u32, data: &[u8]) -> Request {
        Request {
            kind: crate::ipc::Kind::Tipc(command),
            command,
            pid: None,
            copy_handles: Vec::new(),
            move_handles: Vec::new(),
            x: Vec::new(),
            a: Vec::new(),
            b: Vec::new(),
            w: Vec::new(),
            c: Vec::new(),
            data: data.to_vec(),
            object: None,
            in_objects: Vec::new(),
            close_object: false,
        }
    }

    /// syncpt 201's arguments: wait index `n`, syncpoint 1 (inverted on the wire), `threshold`, mode 0.
    fn arm_wait(n: u32, threshold: u32) -> Vec<u8> {
        [&n.to_le_bytes()[..], &(!CHANNEL_SYNCPT).to_le_bytes(), &u64::from(threshold).to_le_bytes(), &[0]].concat()
    }

    fn syncpoint(k: &Kernel) -> u32 {
        k.gs.syncpts[CHANNEL_SYNCPT as usize].load(Ordering::SeqCst)
    }

    /// Waits until the GPU thread wakes the thread that started it (here the
    /// test's), 4 s at most.
    fn woken() -> bool {
        let since = Instant::now();
        std::thread::park_timeout(std::time::Duration::from_secs(5));
        since.elapsed() < std::time::Duration::from_secs(4)
    }

    /// gpu memory 213's event waits for the GPU work handed over before the
    /// unmap, even not yet decoded (it reads through the old mapping); without
    /// the GPU thread, or with nothing queued, it is signaled at once.
    #[test]
    fn an_unmap_event_waits_for_the_queued_work() {
        use crate::gs::Gs;
        use gyroship_runtime::mem::Memory;
        let unmap = |event: u32| [&0u32.to_le_bytes()[..], &0u64.to_le_bytes(), &0xffffu64.to_le_bytes(), &[0; 4], &event.to_le_bytes()].concat();
        let mut k = Kernel::new(Memory::new().unwrap(), crate::journal::Journal::memory(), crate::Config::default());
        Gs::Gpu.request(&mut k, &tipc(12, &3u32.to_le_bytes())).unwrap();
        let e = k.gs.gpu_events[3].unwrap();
        Gs::GpuMemory.request(&mut k, &tipc(213, &unmap(3))).unwrap();
        assert!(k.events[e].signaled && k.gs.armed.is_empty());

        let (mut k, job) = play_submit(true, false, &[semaphore(WORDS, 1, true)]);
        Gs::Gpu.request(&mut k, &tipc(12, &3u32.to_le_bytes())).unwrap();
        let e = k.gs.gpu_events[3].unwrap();
        // Signaled, as the earlier calls on this id leave it.
        Gs::Gpu.request(&mut k, &tipc(21, &3u32.to_le_bytes())).unwrap();
        Gs::GpuMemory.request(&mut k, &tipc(213, &unmap(3))).unwrap();
        assert!(!k.events[e].signaled, "the submit handed over before the unmap is not applied yet");
        assert!(matches!(job.run(), Ok(After::Nothing)));
        assert!(woken(), "the GPU thread wakes the boot thread once the submit is done");
        k.check_gpu_waits();
        assert!(k.events[e].signaled && k.gs.armed.is_empty());
        assert_eq!(word(&k, WORDS), 1);
        // A destroy with nothing queued: signaled at once, never left armed.
        let destroy = [0u32.to_le_bytes(), 3u32.to_le_bytes()].concat();
        Gs::GpuMemory.request(&mut k, &tipc(102, &destroy)).unwrap();
        assert!(k.events[e].signaled);
        assert_eq!(k.tally().counts.get("gpu waits armed").copied(), Some(1));
        // A present handed over before: its resumed channels may read through the old mapping too.
        let present = hand_present(&mut k, (!CHANNEL_SYNCPT, 0));
        Gs::GpuMemory.request(&mut k, &tipc(102, &destroy)).unwrap();
        assert!(!k.events[e].signaled, "the present handed over before the destroy is not applied yet");
        assert!(matches!(present.run(), Ok(After::Present { n: 1, .. })));
        assert!(woken(), "the GPU thread wakes the boot thread once the present is done");
        k.check_gpu_waits();
        assert!(k.events[e].signaled && k.gs.armed.is_empty());
        assert_eq!(k.tally().counts.get("gpu waits armed").copied(), Some(2));
    }

    /// syncpt 201 with the GPU thread: the wait ends when the GPU thread's
    /// increments reach the threshold, and the syncpoint is never raised;
    /// without it, it is raised and signaled at once.
    #[test]
    fn syncpoint_201_is_signaled_when_the_gpu_thread_gets_there() {
        use crate::gs::Gs;
        use gyroship_runtime::mem::Memory;
        let mut k = Kernel::new(Memory::new().unwrap(), crate::journal::Journal::memory(), crate::Config::default());
        Gs::Syncpt.request(&mut k, &tipc(22, &0u32.to_le_bytes())).unwrap();
        let e = k.gs.syncpt_waits[0].unwrap();
        Gs::Syncpt.request(&mut k, &tipc(201, &arm_wait(0, 1))).unwrap();
        assert!(k.events[e].signaled);
        assert_eq!(syncpoint(&k), 1);

        let config = crate::Config { play: true, vsync: false, ..crate::Config::default() };
        let mut k = Kernel::new(Memory::new().unwrap(), crate::journal::Journal::memory(), config);
        k.start_gpu_thread().unwrap();
        Gs::Syncpt.request(&mut k, &tipc(22, &0u32.to_le_bytes())).unwrap();
        let e = k.gs.syncpt_waits[0].unwrap();
        Gs::Syncpt.request(&mut k, &tipc(201, &arm_wait(0, 0))).unwrap();
        assert!(k.events[e].signaled, "already reached");
        Gs::Syncpt.request(&mut k, &tipc(201, &arm_wait(0, 1))).unwrap();
        assert!(!k.events[e].signaled);
        assert_eq!(syncpoint(&k), 0, "no raise");
        // NVN's increment, a release of 1 at address 0, applied by the GPU thread.
        {
            let mut d = crate::lock(&k.decoder);
            let t = d.gpu.as_mut().unwrap();
            let release = gyroship_gpu::pushbuf::Release { method: 0x1b0c, addr: 0, value: 1, width: 4, timestamp: false };
            t.send(&Phase::default(), GpuOp::Effects { space: 0, effects: vec![Effect::Release(release)], read: None }).unwrap();
            t.join().unwrap();
        }
        assert!(woken(), "the GPU thread wakes the boot thread once the syncpoint moves");
        k.check_gpu_waits();
        assert!(k.events[e].signaled && k.gs.armed.is_empty());
        assert_eq!(syncpoint(&k), 1);
        assert_eq!(k.tally().counts.get("syncpoint waits raised"), None);
        // Armed again, already reached: signaled by the arm's own check.
        Gs::Syncpt.request(&mut k, &tipc(201, &arm_wait(0, 1))).unwrap();
        assert!(k.events[e].signaled);
    }

    /// A syncpoint wait still unreached 2 s after it was first armed, with the
    /// GPU thread idle, has its syncpoint raised (an anomaly); not while a
    /// submit or a present is still queued. NVN's rounds (201, 202, 201) keep its age.
    #[test]
    fn an_overdue_syncpoint_wait_is_raised_once_the_gpu_thread_is_idle() {
        use crate::gs::Gs;
        const S: u64 = 31_250_000;
        let (mut k, job) = play_submit(true, true, &[semaphore(WORDS, 1, true)]);
        Gs::Syncpt.request(&mut k, &tipc(22, &0u32.to_le_bytes())).unwrap();
        let e = k.gs.syncpt_waits[0].unwrap();
        Gs::Syncpt.request(&mut k, &tipc(201, &arm_wait(0, 5))).unwrap();
        k.sched.real_now = 3 * S;
        k.check_gpu_waits();
        assert!(!k.events[e].signaled && syncpoint(&k) == 0, "not while a submit is queued");
        assert!(matches!(job.run(), Ok(After::Nothing)));
        assert_eq!(k.drain_gpu(), None);
        let present = hand_present(&mut k, (!CHANNEL_SYNCPT, 0));
        k.check_gpu_waits();
        assert!(!k.events[e].signaled && syncpoint(&k) == 0, "not while a present is queued");
        assert!(matches!(present.run(), Ok(After::Present { n: 1, .. })));
        assert_eq!(k.drain_gpu(), None);
        k.check_gpu_waits();
        assert!(k.events[e].signaled && k.gs.armed.is_empty());
        assert_eq!(syncpoint(&k), 5);
        {
            let t = k.tally();
            assert_eq!((t.counts.get("syncpoint waits raised").copied(), t.anomalies.get("gpu syncpoint wait not reached").copied()), (Some(1), Some(1)));
        }
        Gs::Syncpt.request(&mut k, &tipc(201, &arm_wait(0, 9))).unwrap();
        k.sched.real_now = 3 * S + S / 10;
        Gs::Syncpt.request(&mut k, &tipc(202, &0u32.to_le_bytes())).unwrap();
        assert!(k.events[e].signaled, "202 signals");
        k.sched.real_now = 4 * S + 9 * S / 10;
        Gs::Syncpt.request(&mut k, &tipc(201, &arm_wait(0, 9))).unwrap();
        assert!(!k.events[e].signaled);
        k.sched.real_now = 5 * S + S / 10;
        k.check_gpu_waits();
        assert!(k.events[e].signaled, "armed 2.1 s ago, at its first 201");
        assert_eq!(syncpoint(&k), 9);
    }

    /// A syncpoint wait the client gave up (202) is never raised, however
    /// old: no one waits for it any more.
    #[test]
    fn a_syncpoint_wait_given_up_is_never_raised() {
        use crate::gs::Gs;
        use gyroship_runtime::mem::Memory;
        let config = crate::Config { play: true, explore: true, vsync: false, ..crate::Config::default() };
        let mut k = Kernel::new(Memory::new().unwrap(), crate::journal::Journal::memory(), config);
        k.start_gpu_thread().unwrap();
        Gs::Syncpt.request(&mut k, &tipc(22, &0u32.to_le_bytes())).unwrap();
        Gs::Syncpt.request(&mut k, &tipc(201, &arm_wait(0, 5))).unwrap();
        Gs::Syncpt.request(&mut k, &tipc(202, &0u32.to_le_bytes())).unwrap();
        k.sched.real_now = 3 * 31_250_000;
        k.check_gpu_waits();
        assert_eq!((syncpoint(&k), k.gs.armed.len()), (0, 1));
        assert!(k.tally().anomalies.is_empty());
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
