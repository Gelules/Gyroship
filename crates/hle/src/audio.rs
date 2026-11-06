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
    OpusManager,
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
            Audio::OpusManager => "hwopus manager",
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
            (Audio::OpusSession, 10) => {
                for &h in &req.copy_handles {
                    if matches!(k.handles.get(h), Some(Object::TransferMemory(_))) {
                        k.retain_transfer(h);
                    }
                }
                let ev = k.lone_event()?;
                Reply { copy_handles: vec![ev], data: 0u32.to_le_bytes().to_vec(), objects: vec![Service::Audio(Audio::OpusManager)], ..Reply::ok() }
            }
            (Audio::OpusManager, 0) => Reply::ok(),
            // auddmg: the manager; one virtual device; its format is supported, its
            // profile null; streams attach.
            (Audio::DevMgr, 0) => return object(Audio::DevMgrManager),
            (Audio::DevMgrManager, 0) => {
                name(k, VIRTUAL_DEVICE);
                Reply::with(&1u32.to_le_bytes())
            }
            (Audio::DevMgrManager, 401) => {
                if let Some(b) = req.out_buffer(0) {
                    let _ = k.mem.write(b.addr, &vec![0; b.size.min(0x1000) as usize]);
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
            // The stream's data event, which the silent sink will signal.
            (Audio::Stream { event, .. }, 7) => {
                let (w, r) = k.create_event().ok()?;
                k.handles.close(w);
                *event = k.event_of(r);
                Reply { copy_handles: vec![r], ..Reply::ok() }
            }
            (Audio::Stream { .. }, 4) => Reply::ok(),
            _ => return None,
        })
    }
}
