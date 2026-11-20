//! The recording side of a frame (S12): the upload arena and what is staged
//! in it, the bind groups, the frame's command encoder with its open render
//! pass, and every command recorded into it, applied in the order the GPU
//! thread sends them ([`Rec`]). The GPU thread resolves all that reads or
//! writes guest memory and hands over plain data: blocks, slot sets and
//! pipelines are announced once, then named by id, so that no reference count
//! changes threads per draw.

use crate::draw::Slot;
use crate::surfaces::{ARENA_ALIGN, Bound, FxMap, GroupKey, Groups, Slice, Staging, SubmitJob, Submitter, arena_buffer};
use smallvec::SmallVec;
use std::sync::Arc;

/// A command for the recorder, in stream order.
pub(crate) enum Rec {
    /// A block's bytes (a constant bank, indices or vertices), before its first use.
    Block(u32, Arc<Vec<u8>>),
    /// A stage's texture slots and swizzle codes, before their first use.
    Set(u32, Arc<[Slot]>, Arc<[u32]>),
    /// A render pipeline and its bind group layouts, before its first use (ids last the run).
    Pipeline(u32, wgpu::RenderPipeline, [Option<wgpu::BindGroupLayout>; 2]),
    Draw(DrawRec),
    /// Anything else recorded into the frame's encoder; it ends the open pass.
    Run(Box<dyn FnOnce(&mut Recorder) + Send>),
}

/// A draw as recorded: its pipeline, the blocks of its indices and vertex
/// streams, its bind groups, its state and call, and the pass it opens, if any.
pub(crate) struct DrawRec {
    pub(crate) pipeline: u32,
    /// The indices' block, whether it is staged as it is (else shared: widened), and their format.
    pub(crate) index: Option<(u32, bool, wgpu::IndexFormat)>,
    pub(crate) vertex: SmallVec<[u32; 8]>,
    pub(crate) groups: SmallVec<[GroupRec; 2]>,
    pub(crate) viewport: [f32; 6],
    pub(crate) scissor: [u32; 4],
    pub(crate) blend: [f32; 4],
    pub(crate) stencil: u32,
    pub(crate) call: Call,
    pub(crate) pass: Option<Box<PassOpen>>,
}

/// A draw's bind group: its constant banks (bank, block, bytes the program may
/// read), its slot set, whether it binds the set's swizzle codes, and whether
/// its buffers take dynamic offsets.
pub(crate) struct GroupRec {
    pub(crate) banks: SmallVec<[(u8, u32, u32); 16]>,
    pub(crate) set: u32,
    pub(crate) swizzles: bool,
    pub(crate) dynamic: bool,
}

/// A draw call, and the instance it draws.
pub(crate) enum Call {
    Indexed { count: u32, base_vertex: i32, instance: u32 },
    Direct { first: u32, count: u32, instance: u32 },
}

/// A render pass to open: its color attachments (none without a pixel
/// program), its depth attachment and format, and its label when timed.
pub(crate) struct PassOpen {
    pub(crate) colors: Vec<Option<wgpu::TextureView>>,
    pub(crate) depth: Option<(wgpu::TextureView, wgpu::TextureFormat)>,
    pub(crate) timed: Option<String>,
}

/// What the open pass was last given: a draw sets only what changed (wgpu
/// validates and encodes each call again, and consecutive draws mostly repeat).
/// Pipelines and bind groups wgpu already skips when repeated.
#[derive(Default)]
struct PassState {
    viewport: Option<[f32; 6]>,
    scissor: Option<[u32; 4]>,
    blend: Option<[f32; 4]>,
    stencil: Option<u32>,
    vertex: Vec<Option<Slice>>,
    index: Option<(Slice, wgpu::IndexFormat)>,
}

/// Stores `new` in `old`; whether it differed.
fn changed<T: PartialEq>(old: &mut Option<T>, new: T) -> bool {
    let differs = old.as_ref() != Some(&new);
    if differs {
        *old = Some(new);
    }
    differs
}

pub(crate) struct Recorder {
    /// The render pass the last draw recorded into, left open for the next draws
    /// to the same targets (a pass per draw cost a begin and an end each, ~585 a
    /// frame); anything else recorded ends it first ([`Recorder::encoder`]).
    /// Declared before `encoder`, so that it ends before its encoder is dropped.
    pass: Option<(wgpu::RenderPass<'static>, PassState)>,
    /// The work recorded since the last flush, in order.
    pub(crate) encoder: Option<wgpu::CommandEncoder>,
    pub(crate) device: wgpu::Device,
    queue: wgpu::Queue,
    /// This frame's vertex, index and constant data: one buffer, written at flush.
    pub(crate) arena: (wgpu::Buffer, Vec<u8>),
    /// Offsets of the arena's staged blocks by content hash: draws share identical
    /// constant banks (64 KiB each) instead of staging each copy.
    staged: FxMap<u64, Vec<(usize, usize)>>,
    /// The frame's blocks by id; a bank's staged block by (id, bytes the program
    /// may read), and indices' and vertices' by id.
    blocks: Vec<Arc<Vec<u8>>>,
    banks: FxMap<(u32, u32), Slice>,
    data: FxMap<u32, Slice>,
    /// The frame's texture slot sets, and the run's pipelines with their bind group layouts, by id.
    sets: Vec<(Arc<[Slot]>, Arc<[u32]>)>,
    pipelines: Vec<Option<(wgpu::RenderPipeline, [Option<wgpu::BindGroupLayout>; 2])>>,
    /// Bind groups, made once while the frames use them: a bind group per draw
    /// cost wgpu and the driver more than the rest of the draw.
    bind_groups: Groups<GroupKey, wgpu::BindGroup>,
    /// GPU time per frame (`--perf`) and per program, when asked for (`timing.rs`).
    pub(crate) clock: Option<crate::timing::FrameClock>,
    pub(crate) timing: Option<crate::timing::Timing>,
    /// The counters this side keeps (staging, bind groups, recordings, layout),
    /// which `Surfaces::flush` takes over.
    pub(crate) staging: Staging,
    /// Draws that found no pass open: skipped.
    pub(crate) lost: u64,
    /// Play mode: the submit thread's queue, to sync with before the arena grows,
    /// and whether it ended (then the frames it held are lost).
    pub(crate) submit: Option<std::sync::mpsc::SyncSender<SubmitJob>>,
    pub(crate) submit_ended: bool,
}

impl Recorder {
    pub(crate) fn new(device: &wgpu::Device, queue: &wgpu::Queue, timing: Option<crate::timing::Timing>) -> Recorder {
        Recorder {
            pass: None,
            encoder: None,
            device: device.clone(),
            queue: queue.clone(),
            arena: (arena_buffer(device, 16 << 20), Vec::new()),
            staged: FxMap::default(),
            blocks: Vec::new(),
            banks: FxMap::default(),
            data: FxMap::default(),
            sets: Vec::new(),
            pipelines: Vec::new(),
            bind_groups: Groups::default(),
            clock: None,
            timing,
            staging: Staging::default(),
            lost: 0,
            submit: None,
            submit_ended: false,
        }
    }

    pub(crate) fn apply(&mut self, r: Rec) {
        match r {
            Rec::Block(id, bytes) => {
                debug_assert_eq!(id as usize, self.blocks.len(), "blocks are announced in id order");
                self.blocks.push(bytes);
            }
            Rec::Set(id, slots, swizzles) => {
                debug_assert_eq!(id as usize, self.sets.len(), "sets are announced in id order");
                self.sets.push((slots, swizzles));
            }
            Rec::Pipeline(id, pipeline, layouts) => {
                // Announced once made: in play mode, not in the order their ids were given.
                let id = id as usize;
                if self.pipelines.len() <= id {
                    self.pipelines.resize_with(id + 1, || None);
                }
                self.pipelines[id] = Some((pipeline, layouts));
            }
            Rec::Draw(d) => self.draw(d),
            Rec::Run(run) => run(self),
        }
    }

    /// The frame was flushed: its pass ends, and its blocks and sets are
    /// forgotten (their ids start again).
    pub(crate) fn end_frame(&mut self) {
        self.pass = None;
        self.staged.clear();
        self.blocks.clear();
        self.banks.clear();
        self.data.clear();
        self.sets.clear();
    }

    /// A frame was presented: the bind groups neither it nor the one before used are dropped.
    pub(crate) fn next_frame(&mut self) {
        self.bind_groups.age();
    }

    /// The encoder that collects a frame's passes: one submit per frame
    /// instead of one per operation. Give it back with [`Recorder::record`].
    pub(crate) fn encoder(&mut self) -> wgpu::CommandEncoder {
        // A pass left open for the next draw ends before anything else is recorded.
        self.pass = None;
        self.staging.recordings += 1;
        self.encoder.take().unwrap_or_else(|| {
            let mut encoder = self.device.create_command_encoder(&Default::default());
            if let Some(c) = &self.clock {
                c.open(&mut encoder);
            }
            encoder
        })
    }

    pub(crate) fn record(&mut self, encoder: wgpu::CommandEncoder) {
        self.encoder = Some(encoder);
    }

    fn draw(&mut self, d: DrawRec) {
        if !matches!(self.pipelines.get(d.pipeline as usize), Some(Some(_))) {
            // Never announced: the GPU thread's bookkeeping went wrong.
            self.lost += 1;
            return;
        }
        // In the arena, in this order: indices, vertices, then each group's banks and swizzles.
        let index = d.index.map(|(id, raw, format)| (if raw { self.stage_raw(id) } else { self.stage_data(id) }, format));
        let vertex: SmallVec<[Slice; 8]> = d.vertex.iter().map(|&id| self.stage_raw(id)).collect();
        let mut groups: SmallVec<[(wgpu::BindGroup, SmallVec<[u32; 16]>); 2]> = SmallVec::new();
        for (g, r) in d.groups.iter().enumerate() {
            let mut buffers: SmallVec<[(u32, Slice); 16]> = r.banks.iter().map(|&(b, id, window)| (u32::from(b), self.stage_bank(id, window as usize))).collect();
            if r.swizzles {
                let bytes: SmallVec<[u8; 256]> = self.sets[r.set as usize].1.iter().flat_map(|c| c.to_le_bytes()).collect();
                buffers.push((crate::shader::SWIZZLE_BINDING, self.stage(&bytes)));
            }
            // In binding order, as wgpu takes them: the banks', then the swizzles' (binding 999).
            let offsets: SmallVec<[u32; 16]> = if r.dynamic { buffers.iter().map(|(_, (_, at, _))| u32::try_from(*at).unwrap_or(u32::MAX)).collect() } else { SmallVec::new() };
            let Some(layout) = self.pipelines[d.pipeline as usize].as_ref().and_then(|(_, l)| l.iter().flatten().nth(g)) else { continue };
            let group = bind_group(&self.device, &mut self.bind_groups, &mut self.staging.bind_groups, layout, &buffers, &self.sets[r.set as usize].0, r.dynamic);
            groups.push((group, offsets));
        }
        if let Some(p) = d.pass {
            let timed = p.timed.and_then(|label| self.timing.as_mut().and_then(|t| t.next(label)));
            let mut encoder = self.encoder();
            let load = wgpu::Operations { load: wgpu::LoadOp::Load, store: wgpu::StoreOp::Store };
            let attachments: Vec<Option<wgpu::RenderPassColorAttachment>> = p.colors.iter().map(|v| v.as_ref().map(|view| wgpu::RenderPassColorAttachment { view, depth_slice: None, resolve_target: None, ops: load })).collect();
            let timestamp_writes = timed.zip(self.timing.as_ref()).map(|(i, t)| wgpu::RenderPassTimestampWrites { query_set: &t.set, beginning_of_pass_write_index: Some(i), end_of_pass_write_index: Some(i + 1) });
            let pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("draw"),
                timestamp_writes,
                color_attachments: &attachments,
                depth_stencil_attachment: p.depth.as_ref().map(|(view, f)| wgpu::RenderPassDepthStencilAttachment {
                    view,
                    depth_ops: Some(wgpu::Operations { load: wgpu::LoadOp::Load, store: wgpu::StoreOp::Store }),
                    stencil_ops: f.has_stencil_aspect().then_some(wgpu::Operations { load: wgpu::LoadOp::Load, store: wgpu::StoreOp::Store }),
                }),
                ..Default::default()
            });
            let pass = pass.forget_lifetime();
            self.record(encoder);
            self.pass = Some((pass, PassState::default()));
        }
        let (Some((pass, last)), Some((pipeline, _))) = (self.pass.as_mut(), &self.pipelines[d.pipeline as usize]) else {
            self.lost += 1;
            return;
        };
        pass.set_pipeline(pipeline);
        let [x, y, w, h, near, far] = d.viewport;
        if changed(&mut last.viewport, d.viewport) {
            pass.set_viewport(x, y, w, h, near, far);
        }
        if changed(&mut last.scissor, d.scissor) {
            let [x, y, w, h] = d.scissor;
            pass.set_scissor_rect(x, y, w, h);
        }
        if changed(&mut last.blend, d.blend) {
            let [r, g, b, a] = d.blend.map(f64::from);
            pass.set_blend_constant(wgpu::Color { r, g, b, a });
        }
        if changed(&mut last.stencil, d.stencil) {
            pass.set_stencil_reference(d.stencil);
        }
        for (g, (group, offsets)) in groups.iter().enumerate() {
            pass.set_bind_group(g as u32, group, offsets);
        }
        last.vertex.resize(last.vertex.len().max(vertex.len()), None);
        for (slot, b @ (buffer, at, size)) in vertex.iter().enumerate() {
            if changed(&mut last.vertex[slot], b.clone()) {
                pass.set_vertex_buffer(slot as u32, buffer.slice(*at..at + size));
            }
        }
        match (&index, &d.call) {
            (Some((b @ (buffer, at, size), format)), &Call::Indexed { count, base_vertex, instance }) => {
                if changed(&mut last.index, (b.clone(), *format)) {
                    pass.set_index_buffer(buffer.slice(*at..at + size), *format);
                }
                pass.draw_indexed(0..count, base_vertex, instance..instance + 1);
            }
            (_, &Call::Direct { first, count, instance }) => pass.draw(first..first + count, instance..instance + 1),
            (None, Call::Indexed { .. }) => self.lost += 1,
        }
    }

    /// Copies `bytes` (padded to 4) into this frame's upload arena; returns
    /// the buffer and the range they will occupy when the frame is submitted.
    pub(crate) fn stage(&mut self, bytes: &[u8]) -> Slice {
        self.stage_window(bytes, 0, 0)
    }

    /// As [`Recorder::stage`], the range given being at least `window` bytes: a
    /// constant bank is bound as 64 KiB but only its bound size is hashed and
    /// compared; the rest of its window is zeros, as the console reads past a bank's
    /// size (text vanished when it held the arena's next blocks).
    pub(crate) fn stage_window(&mut self, bytes: &[u8], zeros: usize, window: usize) -> Slice {
        let size = bytes.len().next_multiple_of(4).max(4);
        // Filled: the bytes, then zeros up to `zeros`; bound: `window` at least.
        let fill = size.max(zeros);
        let range = fill.max(window);
        // A fast word hash (SipHash cost 5% of a tutorial frame); bytes are compared after.
        let (words, tail) = bytes.as_chunks::<8>();
        let hash = words.iter().map(|w| u64::from_le_bytes(*w)).chain(tail.iter().map(|&b| u64::from(b))).fold(bytes.len() as u64, |h, w| (h.rotate_left(5) ^ w).wrapping_mul(0x517c_c1b7_2722_0a95));
        let capacity = self.arena.0.size();
        self.staging.calls += 1;
        if let Some(&(at, _)) = self.staged.get(&hash).and_then(|v| v.iter().find(|&&(at, filled)| filled >= fill && (at + range) as u64 <= capacity && self.arena.1.get(at..at + bytes.len()) == Some(bytes))) {
            self.staging.hits += 1;
            return (self.arena.0.clone(), at as u64, range as u64);
        }
        let at = self.arena.1.len().next_multiple_of(ARENA_ALIGN);
        if (at + range) as u64 > capacity {
            self.grow(range);
            return self.stage_window(bytes, zeros, window);
        }
        match window {
            0 => self.staging.data_bytes += size as u64,
            _ => self.staging.bank_bytes += size as u64,
        }
        self.staging.zero_bytes += (fill - size) as u64;
        let data = &mut self.arena.1;
        data.resize(at, 0);
        data.extend_from_slice(bytes);
        data.resize(at + fill, 0);
        self.staged.entry(hash).or_default().push((at, fill));
        self.staging.lay(at as u64, range as u64);
        (self.arena.0.clone(), at as u64, range as u64)
    }

    /// The constant bank of block `id` as a program reading `window` bytes of
    /// it binds it ([`Recorder::stage_window`]: the bytes it may read, then
    /// zeros to there, bound as 64 KiB): staged once a frame for each window,
    /// a read made again after a GPU write taking a new id.
    pub(crate) fn stage_bank(&mut self, id: u32, window: usize) -> Slice {
        if let Some(slice) = self.banks.get(&(id, window as u32)) {
            return slice.clone();
        }
        let bytes = self.blocks[id as usize].clone();
        let slice = self.stage_window(&bytes[..bytes.len().min(window)], window, 0x10000);
        self.banks.insert((id, window as u32), slice.clone());
        slice
    }

    /// Block `id` staged as it is (padded to 4), at the arena's end, neither
    /// hashed nor compared with what the arena holds: vertices and indices,
    /// whose blocks never repeat at another address (the memo finds them at
    /// theirs). Once a frame.
    pub(crate) fn stage_raw(&mut self, id: u32) -> Slice {
        if let Some(slice) = self.data.get(&id) {
            return slice.clone();
        }
        let bytes = self.blocks[id as usize].clone();
        let range = bytes.len().next_multiple_of(4).max(4);
        let mut at = self.arena.1.len().next_multiple_of(ARENA_ALIGN);
        if (at + range) as u64 > self.arena.0.size() {
            self.grow(range);
            at = 0;
        }
        let data = &mut self.arena.1;
        data.resize(at, 0);
        data.extend_from_slice(&bytes);
        data.resize(at + range, 0);
        self.staging.calls += 1;
        self.staging.data_bytes += range as u64;
        self.staging.lay(at as u64, range as u64);
        let slice = (self.arena.0.clone(), at as u64, range as u64);
        self.data.insert(id, slice.clone());
        slice
    }

    /// Block `id` staged as [`Recorder::stage`] does (shared with an identical
    /// block): widened indices. Once a frame.
    pub(crate) fn stage_data(&mut self, id: u32) -> Slice {
        if let Some(slice) = self.data.get(&id) {
            return slice.clone();
        }
        let bytes = self.blocks[id as usize].clone();
        let slice = self.stage(&bytes);
        self.data.insert(id, slice.clone());
        slice
    }

    /// The arena is full: its bytes are written now (it is not reused this frame)
    /// and a buffer twice as large, with room for `range`, takes over; recorded
    /// work keeps the old one alive.
    pub(crate) fn grow(&mut self, range: usize) {
        // Play mode: no frame may wait in the submit thread's queue, or this write
        // (sent with the next submit) would land before that frame's own arena copy.
        self.sync();
        self.staging.flushed += self.arena.1.len() as u64;
        self.queue.write_buffer(&self.arena.0, 0, &self.arena.1);
        let capacity = (self.arena.0.size() * 2).max((range as u64).next_power_of_two());
        self.arena = (arena_buffer(&self.device, capacity), Vec::new());
        self.staged.clear();
        // ponytail: staged blocks stay valid in the old buffer; staged again for simplicity (rare).
        self.banks.clear();
        self.data.clear();
    }

    /// Play mode: returns once the submit thread has run every job sent so far.
    fn sync(&mut self) {
        if let Some(jobs) = &self.submit {
            let (tx, rx) = std::sync::mpsc::sync_channel(1);
            let job: SubmitJob = Box::new(move |_: &mut Submitter| {
                let _ = tx.send(());
            });
            if jobs.send(job).is_err() {
                self.submit = None;
                self.submit_ended = true;
            }
            let _ = rx.recv(); // at once if the thread ended: the job was dropped
        }
    }
}

/// The bind group of `buffers` and `textures` with `layout` from `groups`, made
/// there if needed (`made`: made, asked for). The entries are hashed and
/// compared as they are, borrowed; cloned into a key only for a group made here.
fn bind_group(device: &wgpu::Device, groups: &mut Groups<GroupKey, wgpu::BindGroup>, made: &mut [u64; 2], layout: &wgpu::BindGroupLayout, buffers: &[(u32, Slice)], textures: &[Slot], dynamic: bool) -> wgpu::BindGroup {
    use std::hash::{Hash, Hasher};
    // The entries: a buffer at offset 0 when the offset is dynamic, then each
    // texture slot's view and sampler.
    let at = |a: u64| if dynamic { 0 } else { a };
    let mut h = crate::surfaces::Fx::default();
    layout.hash(&mut h);
    for (binding, (buffer, a, size)) in buffers {
        (binding, buffer, at(*a), size).hash(&mut h);
    }
    for s in textures {
        s.view.hash(&mut h);
        s.sampler.hash(&mut h);
    }
    let same = |(l, bound): &GroupKey| {
        let mut it = bound.iter();
        l == layout
            && buffers.iter().all(|(binding, (buffer, a, size))| matches!(it.next(), Some(Bound::Buffer(b, f, o, z)) if b == binding && f == buffer && *o == at(*a) && z == size))
            && textures.iter().enumerate().all(|(slot, s)| {
                matches!(it.next(), Some(Bound::View(b, v)) if *b == 32 + 2 * slot as u32 && *v == s.view)
                    && s.sampler.as_ref().is_none_or(|sampler| matches!(it.next(), Some(Bound::Sampler(b, x)) if *b == 33 + 2 * slot as u32 && x == sampler))
            })
            && it.next().is_none()
    };
    made[1] += 1;
    groups.get_or(h.finish(), same, || {
        made[0] += 1;
        let mut bound = Vec::with_capacity(buffers.len() + 2 * textures.len());
        for (binding, (buffer, a, size)) in buffers {
            bound.push(Bound::Buffer(*binding, buffer.clone(), at(*a), *size));
        }
        for (slot, s) in textures.iter().enumerate() {
            bound.push(Bound::View(32 + 2 * slot as u32, s.view.clone()));
            if let Some(sampler) = &s.sampler {
                bound.push(Bound::Sampler(33 + 2 * slot as u32, sampler.clone()));
            }
        }
        let entries: Vec<wgpu::BindGroupEntry> = bound
            .iter()
            .map(|b| match b {
                Bound::Buffer(binding, buffer, at, size) => wgpu::BindGroupEntry { binding: *binding, resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding { buffer, offset: *at, size: wgpu::BufferSize::new(*size) }) },
                Bound::View(binding, view) => wgpu::BindGroupEntry { binding: *binding, resource: wgpu::BindingResource::TextureView(view) },
                Bound::Sampler(binding, sampler) => wgpu::BindGroupEntry { binding: *binding, resource: wgpu::BindingResource::Sampler(sampler) },
            })
            .collect();
        let group = device.create_bind_group(&wgpu::BindGroupDescriptor { label: None, layout, entries: &entries });
        ((layout.clone(), bound), group)
    })
}
