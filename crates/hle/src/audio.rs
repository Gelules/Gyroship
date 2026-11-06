//! Audio at boot: `aud:u` (renderer manager, audio device), `hwopus`,
//! `auddmg:u` (device manager) and `audsmx:u` (system mixer, whose stream
//! is a ring in the client's memory). Answers follow what the SDK's clients
//! read (our analysis of build `a7730d0f`); nothing is played.

use crate::Kernel;
use crate::ipc::{Reply, Request};
use crate::service::Service;
use crate::thread::Object;

/// The device name the audio device reports; any NUL-terminated name.
const DEVICE_NAME: &[u8] = b"AudioTvOutput\0";
/// The one virtual output device of the device manager.
const VIRTUAL_DEVICE: &[u8] = b"VirtualOutput0\0";
/// hwopus instance and manager sizes: free for a fake, non-zero.
const OPUS_SIZE: u32 = 0x1000;

#[derive(Debug)]
pub enum Audio {
    AudU,
    RendererManager,
    Device,
    Opus,
    OpusSession,
    /// A codec manager (OpenManager): its results ring (in the client's
    /// memory), the ring's record count, and the PCM bytes of one frame.
    OpusManager { ring: u64, slots: u32, out_stride: u32 },
    DevMgr,
    DevMgrManager,
    Mixer,
    MixerSession,
    /// A mixer stream: its ring (client memory) and, once opened, its slots,
    /// rate and data event (`Kernel::events` index).
    Stream { ring: u64, size: u64, slots: u32, rate: u32, event: Option<usize> },
}

impl Audio {
    pub fn label(&self) -> String {
        match self {
            Audio::AudU => "aud:u",
            Audio::RendererManager => "aud:u IAudioRendererManager",
            Audio::Device => "aud:u IAudioDevice",
            Audio::Opus => "hwopus",
            Audio::OpusSession => "hwopus session",
            Audio::OpusManager { .. } => "hwopus manager",
            Audio::DevMgr => "auddmg:u",
            Audio::DevMgrManager => "auddmg manager",
            Audio::Mixer => "audsmx:u",
            Audio::MixerSession => "audsmx session",
            Audio::Stream { .. } => "audsmx stream",
        }
        .into()
    }

    pub fn request(&mut self, k: &mut Kernel, req: &Request) -> Option<Reply> {
        let object = |a: Audio| Some(Reply::object(Service::Audio(a)));
        let name = |k: &mut Kernel, n: &[u8]| {
            if let Some(b) = req.out_buffer(0) {
                let _ = k.mem.write(b.addr, &n[..n.len().min(b.size as usize)]);
            }
        };
        let event = |k: &mut Kernel| k.lone_event().map(|h| Reply { copy_handles: vec![h], ..Reply::ok() });
        Some(match (self, req.command) {
            (Audio::AudU, 2) => return object(Audio::RendererManager),
            // GetAudioDeviceServiceWithRevisionInfo: a new device each time, closed after use.
            (Audio::RendererManager, 4) => return object(Audio::Device),
            // Active channel count; the active device's name (10, 13); its output event; auto-tune.
            (Audio::Device, 5) => Reply::with(&2u32.to_le_bytes()),
            (Audio::Device, 10 | 13) => {
                name(k, DEVICE_NAME);
                Reply::ok()
            }
            (Audio::Device, 12) => return event(k),
            (Audio::Device, 19) => Reply::ok(),
            // hwopus: a client session; sizes; OpenManager keeps the transfer memory.
            (Audio::Opus, 100) => return object(Audio::OpusSession),
            (Audio::OpusSession, 0 | 1) => Reply::with(&OPUS_SIZE.to_le_bytes()),
            // OpenManager: the shared memory parameter (0x60 bytes: +0x2c the results
            // ring's offset, +0x40 the output stride, +0x44 the I/O buffer count), then
            // +0x68 the address of the process memory region the offsets are in. Tags
            // 1, 2… as Lichii's.
            (Audio::OpusSession, 10) => {
                for &h in &req.copy_handles {
                    if matches!(k.handles.get(h), Some(Object::TransferMemory(_))) {
                        k.retain_transfer(h);
                    }
                }
                let (region, ring_at, out_stride, io) = (req.u64_at(0x68)?, req.u32_at(0x2c)?, req.u32_at(0x40)?, req.u32_at(0x44)?);
                let manager = Audio::OpusManager { ring: region + u64::from(ring_at), slots: io + 1, out_stride };
                let tag = 1 + k.objects.iter().filter(|o| matches!(o, Service::Audio(Audio::OpusManager { .. }))).count() as u32;
                k.log(&format!("hwopus: manager {tag}, {io} I/O buffers, {out_stride} PCM bytes per frame"));
                let ev = k.lone_event()?;
                Reply { copy_handles: vec![ev], data: tag.to_le_bytes().to_vec(), objects: vec![Service::Audio(manager)], ..Reply::ok() }
            }
            (Audio::OpusManager { .. }, 0) => Reply::ok(),
            // Execute(tasks of 16 bytes, in the X or A buffer): every task completes at
            // once, in order, as a frame of silence. A 64-byte record per task in the
            // results ring (+0x80): the task's first 8 bytes (tag, instance, I/O buffer,
            // sequence), +0x10 the PCM byte count, Result 0; then the producer index
            // (+4), last. The output buffers stay as the client zeroed them.
            (Audio::OpusManager { ring, slots, out_stride }, 10) => {
                let b = req.send_buffer(0)?;
                let mut tasks = vec![0u8; b.size.min(0xff0) as usize];
                k.mem.read(b.addr, &mut tasks).ok()?;
                let mut w = [0u8; 4];
                k.mem.read(*ring + 4, &mut w).ok()?;
                let mut w = u32::from_le_bytes(w) % *slots;
                let n = tasks.len() / 16;
                for t in tasks.as_chunks::<16>().0 {
                    let mut e = [0u8; 0x40];
                    e[..8].copy_from_slice(&t[..8]);
                    e[0x10..0x14].copy_from_slice(&out_stride.to_le_bytes());
                    k.mem.write(*ring + 0x80 + u64::from(w) * 0x40, &e).ok()?;
                    w = (w + 1) % *slots;
                }
                k.mem.write(*ring + 4, &w.to_le_bytes()).ok()?;
                if k.journal.few("hwopus execute", 3) {
                    k.log(&format!("hwopus: {n} tasks decoded as silence, producer index {w}"));
                }
                Reply::ok()
            }
            // auddmg: the manager; one virtual device; its format is supported, its
            // profile null; streams attach.
            (Audio::DevMgr, 0) => return object(Audio::DevMgrManager),
            (Audio::DevMgrManager, 0) => {
                name(k, VIRTUAL_DEVICE);
                Reply::with(&1u32.to_le_bytes())
            }
            // GetVirtualDeviceProfile (0x228 bytes): zero (no latency at +0x218), but for the
            // device id at +0x220, which CreateStream gets back and refuses if zero.
            (Audio::DevMgrManager, 401) => {
                if let Some(b) = req.out_buffer(0) {
                    let mut profile = vec![0u8; b.size.min(0x1000) as usize];
                    if let Some(id) = profile.get_mut(0x220..0x224) {
                        id.copy_from_slice(&1u32.to_le_bytes());
                    }
                    let _ = k.mem.write(b.addr, &profile);
                }
                Reply::ok()
            }
            (Audio::DevMgrManager, 300 | 700) => Reply::ok(),
            // audsmx: a user session; CreateStream (id 1) on the client's ring.
            (Audio::Mixer, 0) => return object(Audio::MixerSession),
            (Audio::MixerSession, 1) => {
                let (ring, size) = (req.u64_at(0x10)?, req.u64_at(0x18)?);
                Reply { data: 1u32.to_le_bytes().to_vec(), objects: vec![Service::Audio(Audio::Stream { ring, size, slots: 0, rate: 0, event: None })], ..Reply::ok() }
            }
            // OpenStream: frames and rate; the ring has frames + 1 slots.
            (Audio::Stream { ring, size, slots, rate, .. }, 0) => {
                (*rate, *slots) = (req.u32_at(8)?, req.u32_at(0xc)? + 1);
                k.log(&format!("audsmx: stream 1 ring {ring:#x}+{size:#x}, {slots} slots, {rate} Hz"));
                Reply::ok()
            }
            // The stream's data event, which the silent sink signals once per ring's worth
            // of frames (frames × 31.25 MHz / rate ticks; the first one period from now).
            (Audio::Stream { ring, slots, rate, event, .. }, 7) => {
                let (w, r) = k.create_event().ok()?;
                k.handles.close(w);
                *event = k.event_of(r);
                if let Some(e) = *event
                    && *rate > 0
                    && *slots > 1
                {
                    let action = crate::thread::TimerAction::AudioSink { ring: *ring, slots: *slots, event: e };
                    let (num, den) = (u64::from(*slots - 1) * 31_250_000, u64::from(*rate));
                    k.timers.push(crate::thread::Timer { start: k.now(), k: 1, num, den, action });
                    k.log(&format!("audsmx: silent sink every {} frames at {rate} Hz", *slots - 1));
                }
                Reply { copy_handles: vec![r], ..Reply::ok() }
            }
            (Audio::Stream { .. }, 4) => Reply::ok(),
            _ => return None,
        })
    }
}
