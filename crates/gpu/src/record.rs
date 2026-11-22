//! The recording side of a frame (S12): the upload arena and what is staged
//! in it, the bind groups, the frame's command encoder with its open render
//! pass, and every command recorded into it, applied in the order the GPU
//! thread sends them ([`Rec`]). The GPU thread resolves all that reads or
//! writes guest memory and hands over plain data: blocks, slot sets and
//! pipelines are announced once, then named by id, so that no reference count
//! changes threads per draw.

use crate::draw::Slot;
use crate::surfaces::{ARENA_ALIGN, Bound, FxMap, GroupKey, Groups, Slice, Staging, SubmitJob, Submitter, arena_buffer};
use ash::vk;
use smallvec::SmallVec;
use std::sync::Arc;
use wgpu::hal::api::Vulkan;

/// A render pipeline as the recorder binds it: wgpu's, or a raw one (D110 step 6).
#[derive(Clone)]
pub(crate) enum Pipeline {
    Wgpu(wgpu::RenderPipeline),
    Raw(Arc<crate::raw::RawPipeline>),
}

/// A command for the recorder, in stream order.
pub(crate) enum Rec {
    /// A block's bytes (a constant bank, indices or vertices), before its first use.
    Block(u32, Arc<Vec<u8>>),
    /// A stage's texture slots and swizzle codes, before their first use.
    Set(u32, Arc<[Slot]>, Arc<[u32]>),
    /// A render pipeline and its bind group layouts, before its first use (ids last the run).
    Pipeline(u32, Pipeline, [Option<wgpu::BindGroupLayout>; 2]),
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

/// The attachments of the open pass, whichever API it is in: a draw of the other API reopens it
/// there, as they are (LOAD, STORE).
#[derive(Clone)]
struct Attachments {
    colors: Vec<Option<wgpu::TextureView>>,
    depth: Option<(wgpu::TextureView, wgpu::TextureFormat)>,
}

/// The raw path's side of the recorder (`GYROSHIP_RAW`, D110 step 6).
pub(crate) struct RawRec {
    pools: crate::raw::Pools,
    /// Descriptor sets made once while the frames use them, as `Recorder::bind_groups`; aged out,
    /// they wait in `keep` for the GPU to be done with them, then are freed.
    sets: Groups<GroupKey, (vk::DescriptorPool, vk::DescriptorSet)>,
    /// The arena buffers' raw handles.
    buffers: FxMap<wgpu::Buffer, vk::Buffer>,
    segment: Option<Segment>,
    /// What this frame's raw commands use, until its GPU work is done.
    keep: Keep,
    /// Sets whose frames are done, sent by the frames' callbacks (which never lock the recorder:
    /// a flush holds it while it submits).
    freed: (std::sync::mpsc::Sender<Vec<(vk::DescriptorPool, vk::DescriptorSet)>>, std::sync::mpsc::Receiver<Vec<(vk::DescriptorPool, vk::DescriptorSet)>>),
    /// Whether a timed pass was left untimed (said once).
    untimed: bool,
}

/// What a frame's raw commands use (wgpu does not see them): kept until its GPU work is done.
#[derive(Default)]
struct Keep {
    views: Vec<wgpu::TextureView>,
    textures: Vec<wgpu::Texture>,
    buffers: Vec<wgpu::Buffer>,
    groups: Vec<GroupKey>,
    sets: Vec<(vk::DescriptorPool, vk::DescriptorSet)>,
}

/// A raw segment: the wgpu encoder before it (which gets the transitions to the states the
/// segment begins with), the raw encoder, each texture's format with its state when the segment
/// began and now, the arena buffers it reads, and the attachments of its open rendering, if any,
/// with whether it began.
struct Segment {
    before: wgpu::CommandEncoder,
    raw: wgpu::CommandEncoder,
    states: FxMap<wgpu::Texture, (wgpu::TextureFormat, wgpu::TextureUses, wgpu::TextureUses)>,
    buffers: Vec<wgpu::Buffer>,
    rendering: Option<(Attachments, bool)>,
}

impl RawRec {
    pub(crate) fn new(device: &wgpu::Device) -> Option<RawRec> {
        Some(RawRec { pools: crate::raw::Pools::new(device)?, sets: Groups::default(), buffers: FxMap::default(), segment: None, keep: Keep::default(), freed: std::sync::mpsc::channel(), untimed: false })
    }
}

/// The raw image view of `view` (its guard dropped at once: it holds wgpu's snatch lock).
fn raw_view(view: &wgpu::TextureView) -> vk::ImageView {
    unsafe { view.as_hal::<Vulkan>().map_or(vk::ImageView::null(), |v| v.raw_handle()) }
}

/// The raw image of `texture`.
fn raw_image(texture: &wgpu::Texture) -> vk::Image {
    unsafe { texture.as_hal::<Vulkan>().map_or(vk::Image::null(), |t| t.raw_handle()) }
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
    /// The work recorded since the last flush, in order: the finished recordings (wgpu's and raw
    /// segments', D110), then the wgpu one being recorded.
    encoders: Vec<wgpu::CommandEncoder>,
    pub(crate) encoder: Option<wgpu::CommandEncoder>,
    /// Whether this frame's first encoder was made (it carries the frame clock's opening).
    opened: bool,
    /// The open pass's attachments, whichever API it is in.
    open: Option<Attachments>,
    /// The raw path's side (`GYROSHIP_RAW`).
    pub(crate) raw: Option<RawRec>,
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
    pipelines: Vec<Option<(Pipeline, [Option<wgpu::BindGroupLayout>; 2])>>,
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
    /// The kind of the frame's last recording, for `Staging::segments`: a draw
    /// pass (true) or anything else (false); none yet in this frame.
    last: Option<bool>,
    /// Play mode: the submit thread's queue, to sync with before the arena grows,
    /// and whether it ended (then the frames it held are lost).
    pub(crate) submit: Option<std::sync::mpsc::SyncSender<SubmitJob>>,
    pub(crate) submit_ended: bool,
}

impl Recorder {
    pub(crate) fn new(device: &wgpu::Device, queue: &wgpu::Queue, timing: Option<crate::timing::Timing>) -> Recorder {
        Recorder {
            pass: None,
            encoders: Vec::new(),
            encoder: None,
            opened: false,
            open: None,
            raw: None,
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
            last: None,
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
            Rec::Run(run) => {
                self.segment(false);
                run(self)
            }
        }
    }

    /// A recording of this kind (a draw pass, or anything else): a new segment when the
    /// last one was of the other kind (`Staging::segments`).
    fn segment(&mut self, pass: bool) {
        if self.last != Some(pass) {
            self.staging.segments += 1;
            self.last = Some(pass);
        }
    }

    /// The frame was flushed: its pass and raw segment end, and its blocks and sets
    /// are forgotten (their ids start again). Raw sets whose frames are done are freed.
    pub(crate) fn end_frame(&mut self) {
        self.pass = None;
        self.close_segment();
        self.open = None;
        self.last = None;
        self.staged.clear();
        self.blocks.clear();
        self.banks.clear();
        self.data.clear();
        self.sets.clear();
        if let Some(r) = &mut self.raw {
            while let Ok(sets) = r.freed.1.try_recv() {
                r.pools.free(&sets);
            }
        }
    }

    /// A frame was presented: the bind groups neither it nor the one before used are dropped
    /// (raw sets: once the GPU is done with the next frame, which they go with).
    pub(crate) fn next_frame(&mut self) {
        self.bind_groups.age();
        if let Some(r) = &mut self.raw {
            for (key, set) in r.sets.age().into_values().flatten() {
                r.keep.groups.push(key);
                r.keep.sets.push(set);
            }
        }
    }

    /// The encoder that collects a frame's passes: one submit per frame
    /// instead of one per operation. Give it back with [`Recorder::record`].
    pub(crate) fn encoder(&mut self) -> wgpu::CommandEncoder {
        // A pass left open for the next draw ends before anything else is recorded, as
        // does a raw segment.
        self.pass = None;
        self.close_segment();
        self.staging.recordings += 1;
        match self.encoder.take() {
            Some(e) => e,
            None => self.new_encoder(),
        }
    }

    /// A new wgpu encoder; the frame's first carries the frame clock's opening.
    fn new_encoder(&mut self) -> wgpu::CommandEncoder {
        let mut encoder = self.device.create_command_encoder(&Default::default());
        if !std::mem::replace(&mut self.opened, true)
            && let Some(c) = &self.clock
        {
            c.open(&mut encoder);
        }
        encoder
    }

    /// The frame's recordings to submit, in order: the last is wgpu's (an empty one follows a
    /// raw segment, for the frame clock and the queries), and carries what the raw ones use,
    /// released, and their sets sent back to be freed, once the GPU is done with the frame.
    pub(crate) fn take_encoders(&mut self) -> Vec<wgpu::CommandEncoder> {
        self.close_segment();
        let mut list = std::mem::take(&mut self.encoders);
        match self.encoder.take() {
            Some(e) => list.push(e),
            None if self.raw.is_some() && !list.is_empty() => list.push(self.device.create_command_encoder(&Default::default())),
            None => {}
        }
        self.opened = false;
        if let (Some(r), Some(last)) = (&mut self.raw, list.last()) {
            let keep = std::mem::take(&mut r.keep);
            if !(keep.views.is_empty() && keep.textures.is_empty() && keep.buffers.is_empty() && keep.sets.is_empty()) {
                let freed = r.freed.0.clone();
                last.on_submitted_work_done(move || {
                    let Keep { views, textures, buffers, groups, sets } = keep;
                    drop((views, textures, buffers, groups));
                    let _ = freed.send(sets);
                });
            }
        }
        list
    }

    /// Opens a raw segment unless one is: a wgpu pass ends first, and the wgpu encoder recorded
    /// so far (or a new one) waits before it, for the transitions.
    fn open_segment(&mut self) {
        if self.raw.as_ref().is_none_or(|r| r.segment.is_some()) {
            return;
        }
        self.pass = None;
        let before = match self.encoder.take() {
            Some(e) => e,
            None => self.new_encoder(),
        };
        let raw = self.device.create_command_encoder(&Default::default());
        self.staging.recordings += 1;
        if let Some(r) = &mut self.raw {
            r.segment = Some(Segment { before, raw, states: FxMap::default(), buffers: Vec::new(), rendering: None });
        }
    }

    /// Ends the open raw segment, if any: its rendering ends, its textures go back to the states
    /// it found them in, the encoder before it transitions what it uses to those states, and
    /// both join the frame's recordings. What it used waits in `keep`.
    fn close_segment(&mut self) {
        let Some(r) = &mut self.raw else { return };
        let Some(mut s) = r.segment.take() else { return };
        let ash = &r.pools.ash;
        let back: Vec<_> = s.states.iter().filter(|(_, (_, entry, now))| entry != now).map(|(t, &(format, entry, now))| (raw_image(t), format, now, entry)).collect();
        unsafe {
            s.raw.as_hal_mut::<Vulkan, _, _>(|e| {
                let Some(e) = e else { return };
                let cb = e.raw_handle();
                if s.rendering.as_ref().is_some_and(|(_, begun)| *begun) {
                    ash.cmd_end_rendering(cb);
                }
                for &(image, format, now, entry) in &back {
                    crate::raw::barrier(ash, cb, image, format, now, entry);
                }
            })
        };
        let uses = wgpu::BufferUses::VERTEX | wgpu::BufferUses::INDEX | wgpu::BufferUses::UNIFORM;
        s.before.transition_resources(
            s.buffers.iter().map(|buffer| wgpu::BufferTransition { buffer, state: uses }),
            s.states.iter().map(|(texture, &(_, entry, _))| wgpu::TextureTransition { texture, selector: None, state: entry }),
        );
        r.keep.textures.extend(s.states.into_keys());
        r.keep.buffers.extend(s.buffers);
        self.encoders.push(s.before);
        self.encoders.push(s.raw);
    }

    pub(crate) fn record(&mut self, encoder: wgpu::CommandEncoder) {
        self.encoder = Some(encoder);
    }

    fn draw(&mut self, d: DrawRec) {
        let raw = match self.pipelines.get(d.pipeline as usize) {
            Some(Some((Pipeline::Raw(p), _))) => Some(p.clone()),
            Some(Some((Pipeline::Wgpu(_), _))) => None,
            // Never announced: the GPU thread's bookkeeping went wrong.
            _ => {
                self.lost += 1;
                return;
            }
        };
        // In the arena, in this order: indices, vertices, then each group's banks and swizzles.
        let index = d.index.map(|(id, raw, format)| (if raw { self.stage_raw(id) } else { self.stage_data(id) }, format));
        let vertex: SmallVec<[Slice; 8]> = d.vertex.iter().map(|&id| self.stage_raw(id)).collect();
        let mut staged: SmallVec<[(SmallVec<[(u32, Slice); 16]>, SmallVec<[u32; 16]>); 2]> = SmallVec::new();
        for r in &d.groups {
            let mut buffers: SmallVec<[(u32, Slice); 16]> = r.banks.iter().map(|&(b, id, window)| (u32::from(b), self.stage_bank(id, window as usize))).collect();
            if r.swizzles {
                let bytes: SmallVec<[u8; 256]> = self.sets[r.set as usize].1.iter().flat_map(|c| c.to_le_bytes()).collect();
                buffers.push((crate::shader::SWIZZLE_BINDING, self.stage(&bytes)));
            }
            // In binding order, as wgpu takes them: the banks', then the swizzles' (binding 999).
            let offsets: SmallVec<[u32; 16]> = if r.dynamic { buffers.iter().map(|(_, (_, at, _))| u32::try_from(*at).unwrap_or(u32::MAX)).collect() } else { SmallVec::new() };
            staged.push((buffers, offsets));
        }
        match raw {
            Some(p) => self.raw_draw(d, &p, index, vertex, staged),
            None => self.wgpu_draw(d, index, vertex, staged),
        }
    }

    /// Opens a wgpu pass over `colors` and `depth`, as they are, timed as `timed` says.
    fn open_pass(&mut self, colors: &[Option<wgpu::TextureView>], depth: Option<&(wgpu::TextureView, wgpu::TextureFormat)>, timed: Option<u32>) {
        let mut encoder = self.encoder();
        let load = wgpu::Operations { load: wgpu::LoadOp::Load, store: wgpu::StoreOp::Store };
        let attachments: Vec<Option<wgpu::RenderPassColorAttachment>> = colors.iter().map(|v| v.as_ref().map(|view| wgpu::RenderPassColorAttachment { view, depth_slice: None, resolve_target: None, ops: load })).collect();
        let timestamp_writes = timed.zip(self.timing.as_ref()).map(|(i, t)| wgpu::RenderPassTimestampWrites { query_set: &t.set, beginning_of_pass_write_index: Some(i), end_of_pass_write_index: Some(i + 1) });
        let pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("draw"),
            timestamp_writes,
            color_attachments: &attachments,
            depth_stencil_attachment: depth.map(|(view, f)| wgpu::RenderPassDepthStencilAttachment {
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

    fn wgpu_draw(&mut self, d: DrawRec, index: Option<(Slice, wgpu::IndexFormat)>, vertex: SmallVec<[Slice; 8]>, staged: SmallVec<[(SmallVec<[(u32, Slice); 16]>, SmallVec<[u32; 16]>); 2]>) {
        let mut groups: SmallVec<[(wgpu::BindGroup, SmallVec<[u32; 16]>); 2]> = SmallVec::new();
        for (g, (r, (buffers, offsets))) in d.groups.iter().zip(staged).enumerate() {
            let Some(layout) = self.pipelines[d.pipeline as usize].as_ref().and_then(|(_, l)| l.iter().flatten().nth(g)) else { continue };
            let device = &self.device;
            let group = group(&mut self.bind_groups, &mut self.staging.bind_groups, layout, &buffers, &self.sets[r.set as usize].0, r.dynamic, |bound| {
                let entries: Vec<wgpu::BindGroupEntry> = bound
                    .iter()
                    .map(|b| match b {
                        Bound::Buffer(binding, buffer, at, size) => wgpu::BindGroupEntry { binding: *binding, resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding { buffer, offset: *at, size: wgpu::BufferSize::new(*size) }) },
                        Bound::View(binding, view) => wgpu::BindGroupEntry { binding: *binding, resource: wgpu::BindingResource::TextureView(view) },
                        Bound::Sampler(binding, sampler) => wgpu::BindGroupEntry { binding: *binding, resource: wgpu::BindingResource::Sampler(sampler) },
                    })
                    .collect();
                device.create_bind_group(&wgpu::BindGroupDescriptor { label: None, layout, entries: &entries })
            });
            groups.push((group, offsets));
        }
        if let Some(p) = d.pass {
            self.segment(true);
            let timed = p.timed.and_then(|label| self.timing.as_mut().and_then(|t| t.next(label)));
            self.open_pass(&p.colors, p.depth.as_ref(), timed);
            self.open = Some(Attachments { colors: p.colors, depth: p.depth });
        } else if self.pass.is_none()
            && let Some(a) = self.open.clone()
        {
            // The pass went on in raw Vulkan: it goes on here, as it is.
            self.open_pass(&a.colors, a.depth.as_ref(), None);
        }
        let (Some((pass, last)), Some((Pipeline::Wgpu(pipeline), _))) = (self.pass.as_mut(), &self.pipelines[d.pipeline as usize]) else {
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

    /// A draw of a raw pipeline, in the raw segment (opened if needed): its sets (made once while
    /// the frames use them), its pass (begun anew, or the open one reopened here from wgpu), the
    /// textures it reads and the attachments it writes in their states (a barrier between two
    /// renderings when one changes within the segment), then its state, bindings and call.
    fn raw_draw(&mut self, d: DrawRec, p: &crate::raw::RawPipeline, index: Option<(Slice, wgpu::IndexFormat)>, vertex: SmallVec<[Slice; 8]>, staged: SmallVec<[(SmallVec<[(u32, Slice); 16]>, SmallVec<[u32; 16]>); 2]>) {
        // The sets, in the pipeline's group order, and their dynamic offsets.
        let mut sets: SmallVec<[vk::DescriptorSet; 2]> = SmallVec::new();
        let mut offsets: SmallVec<[u32; 32]> = SmallVec::new();
        let mut reads: SmallVec<[Arc<[Slot]>; 2]> = SmallVec::new();
        for (g, (r, (buffers, offs))) in d.groups.iter().zip(&staged).enumerate() {
            let (Some(layout), Some((set_layout, entries)), Some(rr)) = (self.pipelines[d.pipeline as usize].as_ref().and_then(|(_, l)| l.iter().flatten().nth(g)), p.sets.get(g), self.raw.as_mut()) else { continue };
            let slots = self.sets[r.set as usize].0.clone();
            let RawRec { pools, sets: cache, buffers: handles, .. } = rr;
            let set = group(cache, &mut self.staging.bind_groups, layout, buffers, &slots, r.dynamic, |bound| {
                let resources: Vec<crate::raw::Resource> = bound
                    .iter()
                    .map(|b| match b {
                        Bound::Buffer(_, buffer, at, size) => crate::raw::Resource::Buffer(raw_buffer(handles, buffer), *at, *size),
                        Bound::View(_, view) => crate::raw::Resource::Texture(raw_view(view), wgpu::hal::vulkan::conv::derive_image_layout(wgpu::TextureUses::RESOURCE, view.texture().format())),
                        Bound::Sampler(binding, _) => crate::raw::Resource::Sampler(slots[((binding - 33) / 2) as usize].raw_sampler),
                    })
                    .collect();
                pools.set(*set_layout, entries, &resources).unwrap_or_else(|e| {
                    eprintln!("gyroship: {e}");
                    (vk::DescriptorPool::null(), vk::DescriptorSet::null())
                })
            });
            if set.1 == vk::DescriptorSet::null() {
                self.lost += 1;
                return;
            }
            sets.push(set.1);
            offsets.extend_from_slice(offs);
            reads.push(slots);
        }
        // The pass: a new one, or the open one, reopened here when it was wgpu's.
        let mut end_first = false;
        let attachments = match d.pass {
            Some(pass) => {
                self.segment(true);
                if pass.timed.is_some()
                    && let Some(r) = self.raw.as_mut()
                    && !std::mem::replace(&mut r.untimed, true)
                {
                    eprintln!("gyroship: raw Vulkan passes are not timed (GYROSHIP_GPU_TIMES)");
                }
                let a = Attachments { colors: pass.colors, depth: pass.depth };
                self.open = Some(a.clone());
                Some(a)
            }
            None => None,
        };
        self.open_segment();
        let Some(r) = self.raw.as_mut() else {
            self.lost += 1;
            return;
        };
        let RawRec { pools, segment, buffers: handles, keep, .. } = r;
        let Some(s) = segment.as_mut() else {
            self.lost += 1;
            return;
        };
        if let Some(a) = attachments {
            end_first = s.rendering.as_ref().is_some_and(|(_, begun)| *begun);
            s.rendering = Some((a, false));
        } else if s.rendering.is_none() {
            // The open pass was wgpu's: it goes on here.
            match &self.open {
                Some(a) => s.rendering = Some((a.clone(), false)),
                None => {
                    self.lost += 1;
                    return;
                }
            }
        }
        let Some((a, begun)) = s.rendering.as_mut() else { return };
        // Each texture in the state this draw needs it in: attachments written, textures read.
        let texture = |v: &wgpu::TextureView| (v.texture().clone(), v.texture().format());
        let mut needs: SmallVec<[(wgpu::Texture, wgpu::TextureFormat, wgpu::TextureUses); 16]> = a.colors.iter().flatten().map(|v| { let (t, f) = texture(v); (t, f, wgpu::TextureUses::COLOR_TARGET) }).collect();
        needs.extend(a.depth.as_ref().map(|(v, _)| { let (t, f) = texture(v); (t, f, wgpu::TextureUses::DEPTH_STENCIL_WRITE) }));
        needs.extend(reads.iter().flat_map(|slots| slots.iter()).map(|slot| { let (t, f) = texture(&slot.view); (t, f, wgpu::TextureUses::RESOURCE) }));
        let mut barriers: SmallVec<[(vk::Image, wgpu::TextureFormat, wgpu::TextureUses, wgpu::TextureUses); 4]> = SmallVec::new();
        for (t, f, need) in needs {
            match s.states.get_mut(&t) {
                None => {
                    s.states.insert(t, (f, need, need));
                }
                Some((_, _, now)) if *now != need => {
                    barriers.push((raw_image(&t), f, *now, need));
                    *now = need;
                }
                Some(_) => {}
            }
        }
        if !barriers.is_empty() && *begun {
            end_first = true;
        }
        if end_first {
            *begun = false;
        }
        // What the commands name, read before the encoder is borrowed.
        let colors: SmallVec<[Option<vk::ImageView>; 8]> = a.colors.iter().map(|v| v.as_ref().map(raw_view)).collect();
        let depth = a.depth.as_ref().map(|(v, f)| (raw_view(v), *f));
        let size = a.colors.iter().flatten().chain(a.depth.as_ref().map(|(v, _)| v)).next().map_or((1, 1), |v| (v.texture().width(), v.texture().height()));
        if !*begun {
            keep.views.extend(a.colors.iter().flatten().cloned().chain(a.depth.as_ref().map(|(v, _)| v.clone())));
        }
        let slices = index.iter().map(|(b, _)| b).chain(&vertex).chain(staged.iter().flat_map(|(buffers, _)| buffers.iter().map(|(_, b)| b)));
        for (buffer, _, _) in slices {
            if !s.buffers.contains(buffer) {
                s.buffers.push(buffer.clone());
            }
        }
        let vertex: SmallVec<[(vk::Buffer, u64); 8]> = vertex.iter().map(|(b, at, _)| (raw_buffer(handles, b), *at)).collect();
        let index = index.map(|((b, at, _), format)| (raw_buffer(handles, &b), at, format));
        let begin = !*begun;
        *begun = true;
        self.staging.raw_draws += 1;
        let ash = &pools.ash;
        unsafe {
            s.raw.as_hal_mut::<Vulkan, _, _>(|e| {
                let Some(e) = e else { return };
                let cb = e.raw_handle();
                // An instance open over other attachments, or over textures changing state, ends first.
                if end_first {
                    ash.cmd_end_rendering(cb);
                }
                for &(image, format, from, to) in &barriers {
                    crate::raw::barrier(ash, cb, image, format, from, to);
                }
                if begin {
                    crate::raw::begin_rendering(ash, cb, size, &colors, depth);
                }
                crate::raw::viewport(ash, cb, d.viewport);
                crate::raw::scissor(ash, cb, d.scissor);
                ash.cmd_set_blend_constants(cb, &d.blend);
                ash.cmd_set_stencil_reference(cb, vk::StencilFaceFlags::FRONT_AND_BACK, d.stencil);
                ash.cmd_bind_pipeline(cb, vk::PipelineBindPoint::GRAPHICS, p.pipeline);
                if !sets.is_empty() {
                    ash.cmd_bind_descriptor_sets(cb, vk::PipelineBindPoint::GRAPHICS, p.layout, 0, &sets, &offsets);
                }
                for (slot, &(buffer, at)) in vertex.iter().enumerate() {
                    ash.cmd_bind_vertex_buffers(cb, slot as u32, &[buffer], &[at]);
                }
                match (index, &d.call) {
                    (Some((buffer, at, format)), &Call::Indexed { count, base_vertex, instance }) => {
                        ash.cmd_bind_index_buffer(cb, buffer, at, wgpu::hal::vulkan::conv::map_index_format(format));
                        ash.cmd_draw_indexed(cb, count, 1, 0, base_vertex, instance);
                    }
                    (_, &Call::Direct { first, count, instance }) => ash.cmd_draw(cb, count, 1, first, instance),
                    (None, Call::Indexed { .. }) => {}
                }
            })
        };
    }

    /// A full clear in the raw segment (every draw raw, D110 step 6b), as wgpu's clear pass: a
    /// rendering of its own on `view` that clears it ([`crate::raw::clear`]).
    pub(crate) fn raw_clear(&mut self, view: &wgpu::TextureView, color: [f64; 4], depth: Option<f32>, stencil: Option<u32>) {
        let format = view.texture().format();
        let (raw, size) = (raw_view(view), (view.texture().width(), view.texture().height()));
        self.raw_alone(view, None, |ash, cb| crate::raw::clear(ash, cb, size, raw, format, color, depth, stencil));
    }

    /// A fill in the raw segment, as wgpu's fill pass: pipeline `p` drawn once over `view` with its
    /// value (`bytes`, staged; its set freed with the frame), the pass's own viewport, `scissor` and
    /// `stencil`.
    pub(crate) fn raw_fill(&mut self, view: &wgpu::TextureView, p: &crate::raw::RawPipeline, bytes: &[u8], scissor: [u32; 4], stencil: u32) {
        let (buffer, at, range) = self.stage(bytes);
        let (Some(r), Some((layout, entries))) = (self.raw.as_mut(), p.sets.first()) else { return };
        let resource = crate::raw::Resource::Buffer(raw_buffer(&mut r.buffers, &buffer), at, range);
        let set = match r.pools.set(*layout, entries, &[resource]) {
            Ok(set) => set,
            Err(e) => {
                eprintln!("gyroship: {e}");
                self.lost += 1;
                return;
            }
        };
        r.keep.sets.push(set);
        let format = view.texture().format();
        let depth = format.has_depth_aspect() || format.has_stencil_aspect();
        let (raw, size) = (raw_view(view), (view.texture().width(), view.texture().height()));
        let (pipeline, layout) = (p.pipeline, p.layout);
        self.raw_alone(view, Some(buffer), |ash, cb| {
            match depth {
                true => crate::raw::begin_rendering(ash, cb, size, &[], Some((raw, format))),
                false => crate::raw::begin_rendering(ash, cb, size, &[Some(raw)], None),
            }
            crate::raw::viewport(ash, cb, [0.0, 0.0, size.0 as f32, size.1 as f32, 0.0, 1.0]);
            crate::raw::scissor(ash, cb, scissor);
            unsafe {
                ash.cmd_set_stencil_reference(cb, vk::StencilFaceFlags::FRONT_AND_BACK, stencil);
                ash.cmd_bind_pipeline(cb, vk::PipelineBindPoint::GRAPHICS, pipeline);
                ash.cmd_bind_descriptor_sets(cb, vk::PipelineBindPoint::GRAPHICS, layout, 0, &[set.1], &[]);
                ash.cmd_draw(cb, 3, 1, 0, 0);
                ash.cmd_end_rendering(cb);
            }
        });
    }

    /// A rendering of its own on `view` in the raw segment (opened if needed), as wgpu's clear
    /// and fill passes are: the open rendering ends (the next draw begins it again), the texture
    /// goes to its attachment state and the segment reads `buffer` (an arena's), then `record`.
    fn raw_alone(&mut self, view: &wgpu::TextureView, buffer: Option<wgpu::Buffer>, record: impl FnOnce(&ash::Device, vk::CommandBuffer)) {
        self.staging.recordings += 1;
        self.open_segment();
        let Some(RawRec { pools, segment: Some(s), keep, .. }) = self.raw.as_mut() else {
            self.lost += 1;
            return;
        };
        let end = s.rendering.take().is_some_and(|(_, begun)| begun);
        let t = view.texture();
        let need = if t.format().has_depth_aspect() || t.format().has_stencil_aspect() { wgpu::TextureUses::DEPTH_STENCIL_WRITE } else { wgpu::TextureUses::COLOR_TARGET };
        let barrier = match s.states.get_mut(t) {
            None => {
                s.states.insert(t.clone(), (t.format(), need, need));
                None
            }
            Some((format, _, now)) if *now != need => Some((raw_image(t), *format, std::mem::replace(now, need), need)),
            Some(_) => None,
        };
        if let Some(b) = buffer
            && !s.buffers.contains(&b)
        {
            s.buffers.push(b);
        }
        keep.views.push(view.clone());
        let ash = &pools.ash;
        unsafe {
            s.raw.as_hal_mut::<Vulkan, _, _>(|e| {
                let Some(e) = e else { return };
                let cb = e.raw_handle();
                if end {
                    ash.cmd_end_rendering(cb);
                }
                if let Some((image, format, from, to)) = barrier {
                    crate::raw::barrier(ash, cb, image, format, from, to);
                }
                record(ash, cb);
            })
        };
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

/// The raw buffer of `buffer`, kept in `handles`.
// ponytail: the arena's buffers stay in `handles` once used raw (a few, doubling in size); dropped
// with their frames if their memory ever counts.
fn raw_buffer(handles: &mut FxMap<wgpu::Buffer, vk::Buffer>, buffer: &wgpu::Buffer) -> vk::Buffer {
    *handles.entry(buffer.clone()).or_insert_with(|| unsafe { buffer.as_hal::<Vulkan>().map_or(vk::Buffer::null(), |b| b.raw_handle()) })
}

/// The group (a wgpu bind group, or a raw descriptor set) of `buffers` and `textures` with
/// `layout` from `groups`, made there by `make` from its entries if needed (`made`: made, asked
/// for). The entries are hashed and compared as they are, borrowed; cloned into a key only for a
/// group made here.
fn group<V: Clone>(groups: &mut Groups<GroupKey, V>, made: &mut [u64; 2], layout: &wgpu::BindGroupLayout, buffers: &[(u32, Slice)], textures: &[Slot], dynamic: bool, make: impl FnOnce(&[Bound]) -> V) -> V {
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
        let group = make(&bound);
        ((layout.clone(), bound), group)
    })
}
