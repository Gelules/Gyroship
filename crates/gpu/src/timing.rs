//! GPU time per program, for finding the costly translated shaders: with
//! `GYROSHIP_GPU_TIMES=<file>`, each draw's and dispatch's pass is timestamped, and
//! every 300 flushes the totals by label, costliest first, replace the file's text.
//! Reading the timestamps back waits for the GPU at each flush: a diagnostic mode.

use std::collections::HashMap;
use std::fmt::Write as _;
use std::sync::Arc;
use std::sync::atomic::{AtomicU8, Ordering};

/// Timestamp pairs a flush can hold (a query set holds 4096); passes beyond go untimed.
const PAIRS: u32 = 2048;

/// Flushes a [`FrameClock`] can have in flight before it waits for the GPU.
const RING: usize = 16;

/// GPU time of each frame's work, for `--perf`: a timestamp where a flush's
/// encoder opens and one where it closes, read back a few flushes later
/// without waiting (a ring of small buffers), added up by present.
pub(crate) struct FrameClock {
    set: wgpu::QuerySet,
    resolve: wgpu::Buffer,
    /// Read buffer, its state (0 free, 1 mapping, 2 mapped, 3 failed) and the present it measures.
    ring: Vec<(wgpu::Buffer, Arc<AtomicU8>, u64)>,
    next: usize,
    period: f32,
    /// GPU milliseconds by present.
    pub(crate) ms: Vec<f64>,
    /// Flushes that waited for a free slot: the GPU ran a whole ring behind.
    pub(crate) stalls: u64,
}

impl FrameClock {
    pub(crate) fn new(device: &wgpu::Device, queue: &wgpu::Queue) -> Option<FrameClock> {
        if !device.features().contains(wgpu::Features::TIMESTAMP_QUERY | wgpu::Features::TIMESTAMP_QUERY_INSIDE_ENCODERS) {
            return None;
        }
        let stride = wgpu::QUERY_RESOLVE_BUFFER_ALIGNMENT;
        let set = device.create_query_set(&wgpu::QuerySetDescriptor { label: Some("frames"), ty: wgpu::QueryType::Timestamp, count: 2 * RING as u32 });
        let resolve = device.create_buffer(&wgpu::BufferDescriptor { label: Some("frames"), size: RING as u64 * stride, usage: wgpu::BufferUsages::QUERY_RESOLVE | wgpu::BufferUsages::COPY_SRC, mapped_at_creation: false });
        let read = || device.create_buffer(&wgpu::BufferDescriptor { label: Some("frames"), size: 16, usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ, mapped_at_creation: false });
        let ring = (0..RING).map(|_| (read(), Arc::new(AtomicU8::new(0)), 0)).collect();
        Some(FrameClock { set, resolve, ring, next: 0, period: queue.get_timestamp_period(), ms: Vec::new(), stalls: 0 })
    }

    /// A flush's encoder was created: its first timestamp.
    pub(crate) fn open(&self, encoder: &mut wgpu::CommandEncoder) {
        encoder.write_timestamp(&self.set, 2 * self.next as u32);
    }

    /// Before the flush's encoder is finished: its last timestamp, both copied to the slot's buffer.
    pub(crate) fn close(&mut self, device: &wgpu::Device, encoder: &mut wgpu::CommandEncoder) {
        // A submit may not copy into a buffer still mapped (or about to be).
        self.harvest(device, Some(self.next));
        let i = self.next as u32;
        let at = u64::from(i) * wgpu::QUERY_RESOLVE_BUFFER_ALIGNMENT;
        encoder.write_timestamp(&self.set, 2 * i + 1);
        encoder.resolve_query_set(&self.set, 2 * i..2 * i + 2, &self.resolve, at);
        encoder.copy_buffer_to_buffer(&self.resolve, at, &self.ring[self.next].0, 0, 16);
    }

    /// After the flush's submit: its slot is mapped when the GPU gets there.
    pub(crate) fn submitted(&mut self, device: &wgpu::Device, present: u64) {
        let (buffer, state, at) = &mut self.ring[self.next];
        *at = present;
        state.store(1, Ordering::Release);
        let done = state.clone();
        buffer.slice(..).map_async(wgpu::MapMode::Read, move |r| done.store(if r.is_ok() { 2 } else { 3 }, Ordering::Release));
        self.next = (self.next + 1) % RING;
        self.harvest(device, None);
    }

    /// Adds up the slots whose timestamps arrived; waits for the GPU first if
    /// slot `wait` is still being mapped.
    fn harvest(&mut self, device: &wgpu::Device, wait: Option<usize>) {
        let blocked = wait.is_some_and(|i| self.ring[i].1.load(Ordering::Acquire) == 1);
        self.stalls += u64::from(blocked);
        let _ = device.poll(if blocked { wgpu::PollType::wait_indefinitely() } else { wgpu::PollType::Poll });
        let FrameClock { ring, ms, period, .. } = self;
        for (buffer, state, present) in ring.iter_mut() {
            match state.load(Ordering::Acquire) {
                2 => {
                    if let Ok(view) = buffer.slice(..).get_mapped_range() {
                        let [a, b] = [0, 8].map(|o| u64::from_le_bytes(view[o..o + 8].try_into().unwrap_or_default()));
                        let at = *present as usize;
                        if ms.len() <= at {
                            ms.resize(at + 1, 0.0);
                        }
                        ms[at] += b.saturating_sub(a) as f64 * f64::from(*period) / 1e6;
                    }
                    buffer.unmap();
                    state.store(0, Ordering::Release);
                }
                3 => state.store(0, Ordering::Release),
                _ => {}
            }
        }
    }

    /// Waits for every slot in flight; the milliseconds by present.
    pub(crate) fn finish(&mut self, device: &wgpu::Device) -> Vec<f64> {
        let _ = device.poll(wgpu::PollType::wait_indefinitely());
        self.harvest(device, None);
        std::mem::take(&mut self.ms)
    }
}

pub(crate) struct Timing {
    pub(crate) set: wgpu::QuerySet,
    resolve: wgpu::Buffer,
    read: wgpu::Buffer,
    labels: Vec<String>,
    /// Milliseconds and passes by label, since the start.
    totals: HashMap<String, (f64, u64)>,
    flushes: u64,
    path: std::path::PathBuf,
    period: f32,
}

impl Timing {
    /// The features GPU timing needs (asked for when the adapter has them; they change no result).
    pub(crate) fn wanted() -> wgpu::Features {
        wgpu::Features::TIMESTAMP_QUERY | wgpu::Features::TIMESTAMP_QUERY_INSIDE_ENCODERS
    }

    pub(crate) fn new(device: &wgpu::Device, queue: &wgpu::Queue) -> Option<Timing> {
        let path = std::env::var_os("GYROSHIP_GPU_TIMES")?.into();
        if !device.features().contains(wgpu::Features::TIMESTAMP_QUERY) {
            eprintln!("gyroship: GPU timestamps not supported: no GPU times");
            return None;
        }
        let set = device.create_query_set(&wgpu::QuerySetDescriptor { label: Some("timing"), ty: wgpu::QueryType::Timestamp, count: 2 * PAIRS });
        let size = u64::from(2 * PAIRS) * 8;
        let resolve = device.create_buffer(&wgpu::BufferDescriptor { label: Some("timing"), size, usage: wgpu::BufferUsages::QUERY_RESOLVE | wgpu::BufferUsages::COPY_SRC, mapped_at_creation: false });
        let read = device.create_buffer(&wgpu::BufferDescriptor { label: Some("timing"), size, usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ, mapped_at_creation: false });
        Some(Timing { set, resolve, read, labels: Vec::new(), totals: HashMap::new(), flushes: 0, path, period: queue.get_timestamp_period() })
    }

    /// The first of the two query indices for a pass with this label, if room is left.
    pub(crate) fn next(&mut self, label: String) -> Option<u32> {
        let n = self.labels.len() as u32;
        (n < PAIRS).then(|| {
            self.labels.push(label);
            2 * n
        })
    }

    /// Before the flush's encoder is finished: its timestamps to the read buffer.
    pub(crate) fn resolve(&self, encoder: &mut wgpu::CommandEncoder) {
        let n = 2 * self.labels.len() as u32;
        if n > 0 {
            encoder.resolve_query_set(&self.set, 0..n, &self.resolve, 0);
            encoder.copy_buffer_to_buffer(&self.resolve, 0, &self.read, 0, u64::from(n) * 8);
        }
    }

    /// After the flush's submit: the passes' times added up, and the file rewritten
    /// every 300 flushes.
    pub(crate) fn collect(&mut self, device: &wgpu::Device) {
        if !self.labels.is_empty() {
            let bytes = 16 * self.labels.len() as u64;
            let slice = self.read.slice(..bytes);
            slice.map_async(wgpu::MapMode::Read, |_| ());
            if device.poll(wgpu::PollType::wait_indefinitely()).is_ok()
                && let Ok(mapped) = slice.get_mapped_range()
            {
                let ticks: Vec<u64> = mapped.as_chunks::<8>().0.iter().map(|b| u64::from_le_bytes(*b)).collect();
                for (label, pair) in self.labels.iter().zip(ticks.as_chunks::<2>().0) {
                    let ms = pair[1].saturating_sub(pair[0]) as f64 * f64::from(self.period) / 1e6;
                    let t = self.totals.entry(label.clone()).or_default();
                    t.0 += ms;
                    t.1 += 1;
                }
            }
            self.read.unmap();
            self.labels.clear();
        }
        self.flushes += 1;
        if self.flushes.is_multiple_of(300) {
            let _ = std::fs::write(&self.path, self.report());
        }
    }

    fn report(&self) -> String {
        let mut rows: Vec<_> = self.totals.iter().collect();
        rows.sort_by(|a, b| b.1.0.total_cmp(&a.1.0));
        let all: f64 = rows.iter().map(|r| r.1.0).sum();
        let mut out = format!("# GPU time by program after {} flushes: {all:.1} ms in all\n# ms  percent  passes  ms/pass  label\n", self.flushes);
        for (label, (ms, n)) in rows {
            let _ = writeln!(out, "{ms:10.1} {:6.2}% {n:8} {:8.3}  {label}", 100.0 * ms / all.max(1e-9), ms / *n as f64);
        }
        out
    }
}
