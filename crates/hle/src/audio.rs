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
    /// `io` the I/O buffers' start; buffer i's input at io + i × (in + out), its output right after.
    OpusManager { ring: u64, slots: u32, io: u64, in_stride: u32, out_stride: u32 },
    DevMgr,
    DevMgrManager,
    Mixer,
    MixerSession,
    /// A mixer stream: its ring (client memory) and, once opened, its slots,
    /// rate and data event (`Kernel::events` index).
    Stream { ring: u64, size: u64, slots: u32, rate: u32, channels: u32, data: u64, event: Option<usize> },
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
                let (io_at, in_stride) = (req.u32_at(0x24)?, req.u32_at(0x3c)?);
                let manager = Audio::OpusManager { ring: region + u64::from(ring_at), slots: io + 1, io: region + u64::from(io_at), in_stride, out_stride };
                let tag = 1 + k.objects.iter().filter(|o| matches!(o, Service::Audio(Audio::OpusManager { .. }))).count() as u32;
                k.log(&format!("hwopus: manager {tag}, {io} I/O buffers, {out_stride} PCM bytes per frame"));
                let ev = k.lone_event()?;
                Reply { copy_handles: vec![ev], data: tag.to_le_bytes().to_vec(), objects: vec![Service::Audio(manager)], ..Reply::ok() }
            }
            (Audio::OpusManager { .. }, 0) => Reply::ok(),
            // Execute(tasks of 16 bytes, in the X or A buffer: +2 instance, +4 I/O buffer,
            // +8 input size, +0xc reset): every task completes at once, in order. Its
            // input (a big-endian size, 4 bytes of 0, the Opus packet) decodes to 960
            // s16 samples per channel in the buffer's output. A 64-byte record per task in
            // the results ring (+0x80): the task's first 8 bytes (tag, instance, I/O
            // buffer, sequence), +8 duration 0 (never host time), +0x10 the PCM byte
            // count, Result 0; then the producer index (+4), last.
            (Audio::OpusManager { ring, slots, io, in_stride, out_stride }, 10) => {
                let b = req.send_buffer(0)?;
                let mut tasks = vec![0u8; b.size.min(0xff0) as usize];
                k.mem.read(b.addr, &mut tasks).ok()?;
                let mut w = [0u8; 4];
                k.mem.read(*ring + 4, &mut w).ok()?;
                let mut w = u32::from_le_bytes(w) % *slots;
                let n = tasks.len() / 16;
                let channels = (*out_stride / (960 * 2)).clamp(1, 2) as usize;
                for t in tasks.as_chunks::<16>().0 {
                    let (instance, buffer) = (u16::from_le_bytes([t[2], t[3]]), u16::from_le_bytes([t[4], t[5]]));
                    let input = *io + u64::from(buffer) * u64::from(*in_stride + *out_stride);
                    let mut head = [0u8; 8];
                    let _ = k.mem.read_object(input, &mut head);
                    let size = (u32::from_be_bytes(head[..4].try_into().unwrap()) as usize).min(*in_stride as usize - 8);
                    let mut packet = vec![0u8; size];
                    let _ = k.mem.read_object(input + 8, &mut packet);
                    let decoder = k.opus.entry((*ring, instance)).or_insert_with(|| crate::opus::Decoder::new(channels));
                    if t[0xc] != 0 {
                        decoder.reset();
                    }
                    let mut pcm = vec![0i16; 960 * channels];
                    let decoded = decoder.decode(&packet, &mut pcm);
                    if decoded != Ok(960) {
                        // Concealment, never an error code: the game aborts on any.
                        let _ = decoder.decode(&[], &mut pcm);
                        if k.journal.few("hwopus decode", 4) {
                            k.log(&format!("hwopus: instance {instance}: packet of {size} bytes gave {decoded:?}; concealed"));
                        }
                    }
                    k.count(if pcm.iter().all(|&s| s == 0) { "opus silent frames" } else { "opus sounding frames" });
                    let bytes: Vec<u8> = pcm.iter().flat_map(|s| s.to_le_bytes()).collect();
                    let _ = k.mem.write_object(input + u64::from(*in_stride), &bytes);
                    let mut e = [0u8; 0x40];
                    e[..8].copy_from_slice(&t[..8]);
                    e[0x10..0x14].copy_from_slice(&(bytes.len() as u32).to_le_bytes());
                    k.mem.write(*ring + 0x80 + u64::from(w) * 0x40, &e).ok()?;
                    w = (w + 1) % *slots;
                }
                k.mem.write(*ring + 4, &w.to_le_bytes()).ok()?;
                if k.journal.few("hwopus execute", 3) {
                    k.log(&format!("hwopus: {n} tasks decoded, producer index {w}"));
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
                Reply { data: 1u32.to_le_bytes().to_vec(), objects: vec![Service::Audio(Audio::Stream { ring, size, slots: 0, rate: 0, channels: 0, data: 0, event: None })], ..Reply::ok() }
            }
            // OpenStream: frames and rate; the ring has frames + 1 slots.
            // Sample type, channels, rate, frames, then options (the SDK's OpenStreamImpl).
            // Its ring layout (sdk 72402ae020): read header at +0x40, write header at
            // +0x1000, samples at +0x1040 (0x80 further with option bit 0).
            (Audio::Stream { ring, size, slots, rate, channels, data, .. }, 0) => {
                (*rate, *slots, *channels) = (req.u32_at(8)?, req.u32_at(0xc)? + 1, req.u32_at(4)?);
                let (kind, options) = (req.u32_at(0)?, req.u32_at(0x14).unwrap_or(0));
                *data = *ring + 0x1040 + if options & 1 != 0 { 0x80 } else { 0 };
                k.audio_format = (*channels as u16, *rate);
                let args: String = req.data.iter().take(0x20).map(|b| format!("{b:02x}")).collect();
                k.log(&format!("audsmx: stream 1 ring {ring:#x}+{size:#x}, {slots} slots, {rate} Hz, {channels} channels, sample type {kind}, options {options:#x}, samples at {data:#x}; arguments {args}"));
                if kind != 5 {
                    k.anomaly("audio format", &format!("sample type {kind}: only f32 (5) is played"));
                }
                Reply::ok()
            }
            // The stream's data event, which the silent sink signals once per ring's worth
            // of frames (frames × 31.25 MHz / rate ticks; the first one period from now).
            (Audio::Stream { ring, slots, rate, channels, data, event, .. }, 7) => {
                let (w, r) = k.create_event().ok()?;
                k.handles.close(w);
                *event = k.event_of(r);
                if let Some(e) = *event
                    && *rate > 0
                    && *slots > 1
                {
                    let action = crate::thread::TimerAction::AudioSink { ring: *ring, slots: *slots, data: *data, channels: *channels, event: e };
                    let (num, den) = (u64::from(*slots - 1) * 31_250_000, u64::from(*rate));
                    k.timers.push(crate::thread::Timer { start: k.now(), k: 1, num, den, action });
                    k.log(&format!("audsmx: silent sink every {} frames at {rate} Hz", *slots - 1));
                }
                Reply { copy_handles: vec![r], ..Reply::ok() }
            }
            // SetStreamGain: an f32 gain, then a ramp; applied on the host side only.
            (Audio::Stream { .. }, 4) => {
                k.audio_gain = req.data.get(0..4).map_or(1.0, |b| f32::from_le_bytes(b.try_into().unwrap()));
                if k.journal.few("audsmx gain", 4) {
                    let args: String = req.data.iter().take(0x10).map(|b| format!("{b:02x}")).collect();
                    k.log(&format!("audsmx: stream gain {}; arguments {args}", k.audio_gain));
                }
                Reply::ok()
            }
            _ => return None,
        })
    }
}

/// The frames a sink plays: slots `read` up to `write` (wrapping at
/// `slots`), `channels` f32 each, from `data`, read through `read_f32s`.
pub fn ring_frames(read_f32s: &mut dyn FnMut(u64, &mut [f32]) -> bool, data: u64, channels: u32, slots: u32, read: u32, write: u32) -> Vec<f32> {
    let n = (write + slots - read) % slots;
    let mut out = vec![0f32; (n * channels) as usize];
    let stride = u64::from(channels) * 4;
    // Two runs at most: up to the ring's end, then from its start.
    let first = n.min(slots - read);
    let (a, b) = out.split_at_mut((first * channels) as usize);
    if !read_f32s(data + u64::from(read) * stride, a) || !read_f32s(data, b) {
        out.fill(0.0);
    }
    out
}

/// A WAV file of interleaved f32 samples (IEEE float, format 3).
pub fn wav(samples: &[f32], channels: u16, rate: u32) -> Vec<u8> {
    let bytes = (samples.len() * 4) as u32;
    let mut w = Vec::with_capacity(44 + bytes as usize);
    w.extend(b"RIFF");
    w.extend((36 + bytes).to_le_bytes());
    w.extend(b"WAVEfmt ");
    w.extend(16u32.to_le_bytes());
    w.extend(3u16.to_le_bytes());
    w.extend(channels.to_le_bytes());
    w.extend(rate.to_le_bytes());
    w.extend((rate * u32::from(channels) * 4).to_le_bytes());
    w.extend((channels * 4).to_le_bytes());
    w.extend(32u16.to_le_bytes());
    w.extend(b"data");
    w.extend(bytes.to_le_bytes());
    for s in samples {
        w.extend(s.to_le_bytes());
    }
    w
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Stereo frame i holds (i, -i): the frames come out in order, across the
    /// ring's end (read 2040, write 10, 2049 slots: 19 frames).
    #[test]
    fn a_sink_reads_its_frames_in_order_across_the_end() {
        const DATA: u64 = 0x1000;
        let ring: Vec<f32> = (0..2049).flat_map(|i| [i as f32, -(i as f32)]).collect();
        let mut read = |addr: u64, out: &mut [f32]| {
            let at = ((addr - DATA) / 4) as usize;
            out.copy_from_slice(&ring[at..at + out.len()]);
            true
        };
        let frames = ring_frames(&mut read, DATA, 2, 2049, 2040, 10);
        let want: Vec<f32> = (2040..2049).chain(0..10).flat_map(|i| [i as f32, -(i as f32)]).collect();
        assert_eq!(frames, want);
        assert!(ring_frames(&mut read, DATA, 2, 2049, 5, 5).is_empty());
    }

    #[test]
    fn wav_header_describes_float_stereo() {
        let w = wav(&[0.5, -0.5], 2, 48_000);
        assert_eq!(&w[0..4], b"RIFF");
        assert_eq!(u32::from_le_bytes(w[4..8].try_into().unwrap()), 36 + 8);
        assert_eq!(u16::from_le_bytes(w[20..22].try_into().unwrap()), 3);
        assert_eq!(u32::from_le_bytes(w[28..32].try_into().unwrap()), 48_000 * 8);
        assert_eq!(w.len(), 44 + 8);
        assert_eq!(f32::from_le_bytes(w[44..48].try_into().unwrap()), 0.5);
    }
}
