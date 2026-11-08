//! GPU time per program, for finding the costly translated shaders: with
//! `GYROSHIP_GPU_TIMES=<file>`, each draw's and dispatch's pass is timestamped, and
//! every 300 flushes the totals by label, costliest first, replace the file's text.
//! Reading the timestamps back waits for the GPU at each flush: a diagnostic mode.

use std::collections::HashMap;
use std::fmt::Write as _;

/// Timestamp pairs a flush can hold (a query set holds 4096); passes beyond go untimed.
const PAIRS: u32 = 2048;

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
    /// The features timing needs, when it is asked for.
    pub(crate) fn wanted() -> wgpu::Features {
        if std::env::var_os("GYROSHIP_GPU_TIMES").is_some() { wgpu::Features::TIMESTAMP_QUERY } else { wgpu::Features::empty() }
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
