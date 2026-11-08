//! Opus decoding for hwopus (D31): libopus 1.3.1, as `unsafe-libopus`
//! transcribes it to Rust (no C toolchain, no CPU detection, so the same PCM
//! on every machine). A decoder's whole state is one block of bytes, which
//! is what a save state keeps.
//!
//! ponytail: the CELT part of the state points at libopus's static mode
//! table; a state restored in another process needs that pointer rewritten
//! (M6, save states). Within one process a byte copy is enough.

use unsafe_libopus::{OPUS_GET_FINAL_RANGE_REQUEST, OpusDecoder, opus_decode, opus_decoder_ctl, opus_decoder_get_size, opus_decoder_init};

/// A 48 kHz decoder of 1 or 2 channels.
pub struct Decoder {
    /// The libopus state, 8-byte aligned.
    state: Vec<u64>,
    pub channels: usize,
}

impl Decoder {
    pub fn new(channels: usize) -> Decoder {
        // SAFETY: a pure size query.
        let size = unsafe { opus_decoder_get_size(channels as i32) }.max(0) as usize;
        let mut d = Decoder { state: vec![0; size.div_ceil(8)], channels };
        d.reset();
        d
    }

    fn st(&mut self) -> *mut OpusDecoder {
        self.state.as_mut_ptr().cast()
    }

    /// Back to a fresh state (hwopus's reset flag).
    pub fn reset(&mut self) {
        let channels = self.channels as i32;
        // SAFETY: the buffer is opus_decoder_get_size bytes, aligned.
        let r = unsafe { opus_decoder_init(self.st(), 48_000, channels) };
        assert_eq!(r, 0, "opus_decoder_init({channels} channels)");
    }

    /// Decodes `packet` (empty: loss concealment) into `pcm`, interleaved,
    /// which bounds the samples per channel; returns how many, or libopus's
    /// error code.
    pub fn decode(&mut self, packet: &[u8], pcm: &mut [i16]) -> Result<usize, i32> {
        let frame = (pcm.len() / self.channels) as i32;
        let data = if packet.is_empty() { std::ptr::null() } else { packet.as_ptr() };
        // SAFETY: a valid state, the packet's bytes, and room for `frame` samples per channel.
        let n = unsafe { opus_decode(self.st(), data, packet.len() as i32, pcm.as_mut_ptr(), frame, 0) };
        if n < 0 { Err(n) } else { Ok(n as usize) }
    }

    /// The range coder's final state after the last packet: equal to the
    /// encoder's when decoding is bit exact.
    pub fn final_range(&mut self) -> u32 {
        let mut r = 0u32;
        // SAFETY: a valid state; the request writes one u32.
        unsafe { opus_decoder_ctl!(self.st(), OPUS_GET_FINAL_RANGE_REQUEST, &mut r) };
        r
    }

    /// The state's bytes (see the module note on restoring them).
    pub fn bytes(&self) -> Vec<u8> {
        self.state.iter().flat_map(|w| w.to_le_bytes()).collect()
    }

    /// Takes the state `bytes` gave, from a decoder of this process.
    pub fn restore(&mut self, bytes: &[u8]) {
        for (w, b) in self.state.iter_mut().zip(bytes.as_chunks::<8>().0) {
            *w = u64::from_le_bytes(*b);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// One CELT fullband 20 ms frame, mono (TOC 0xf8), with no data: libopus
    /// conceals it as 960 samples.
    #[test]
    fn a_toc_only_packet_gives_960_samples() {
        let mut d = Decoder::new(1);
        let mut pcm = vec![1i16; 960];
        assert_eq!(d.decode(&[0xf8], &mut pcm), Ok(960));
        let mut s = Decoder::new(2);
        let mut pcm = vec![0i16; 2 * 5760];
        assert_eq!(s.decode(&[], &mut pcm), Ok(5760), "concealment of a missing packet fills the request");
    }

    /// A synthetic second of 440 Hz, encoded here, decodes to a sounding
    /// signal; a decoder that takes a copy of the state midway decodes the
    /// rest identically (what a save state relies on).
    #[test]
    fn a_copied_state_decodes_the_rest_identically() {
        use unsafe_libopus::{OPUS_APPLICATION_AUDIO, opus_encode, opus_encoder_create, opus_encoder_destroy};
        let mut error = 0;
        // SAFETY: valid arguments; the encoder is destroyed below.
        let encoder = unsafe { opus_encoder_create(48_000, 2, OPUS_APPLICATION_AUDIO, &mut error) };
        assert_eq!(error, 0);
        let packets: Vec<Vec<u8>> = (0..50)
            .map(|f| {
                let pcm: Vec<i16> = (0..960).flat_map(|i| {
                    let v = (8000.0 * ((f * 960 + i) as f32 * 440.0 * std::f32::consts::TAU / 48_000.0).sin()) as i16;
                    [v, v]
                }).collect();
                let mut out = vec![0u8; 1500];
                // SAFETY: 960 stereo samples in, room for 1500 bytes out.
                let n = unsafe { opus_encode(encoder, pcm.as_ptr(), 960, out.as_mut_ptr(), out.len() as i32) };
                assert!(n > 0, "encode: {n}");
                out.truncate(n as usize);
                out
            })
            .collect();
        // SAFETY: the encoder created above.
        unsafe { opus_encoder_destroy(encoder) };
        let (mut d, mut copy) = (Decoder::new(2), None::<Decoder>);
        let mut energy = 0f64;
        for (i, p) in packets.iter().enumerate() {
            let mut pcm = vec![0i16; 2 * 960];
            assert_eq!(d.decode(p, &mut pcm), Ok(960));
            energy += pcm.iter().map(|&s| f64::from(s).powi(2)).sum::<f64>();
            if i == 25 {
                let mut c = Decoder::new(2);
                c.restore(&d.bytes());
                copy = Some(c);
            } else if let Some(c) = copy.as_mut() {
                let mut q = vec![0i16; 2 * 960];
                assert_eq!(c.decode(p, &mut q), Ok(960));
                assert_eq!(q, pcm, "packet {i}");
            }
        }
        assert!(energy / (50.0 * 1920.0) > 1e6, "a sounding signal (RMS above 1000)");
    }

    /// The RFC 8251 test vectors (Xiph's, not game data) when they are under
    /// $GYROSHIP_DATA/opus/opus_newvectors: every packet's final range must
    /// equal the encoder's, and the PCM must match the reference closely. A
    /// state copied midway decodes the rest identically.
    #[test]
    fn rfc8251_vectors_meet_the_conformance_criteria() {
        let Some(dir) = std::env::var_os("GYROSHIP_DATA").map(|d| std::path::PathBuf::from(d).join("opus/opus_newvectors")).filter(|d| d.is_dir()) else {
            eprintln!("no RFC 8251 vectors: test skipped");
            return;
        };
        for n in 1..=12 {
            let bit = std::fs::read(dir.join(format!("testvector{n:02}.bit"))).unwrap();
            let reference = std::fs::read(dir.join(format!("testvector{n:02}.dec"))).unwrap();
            let mut d = Decoder::new(2);
            let mut copy: Option<(Decoder, Vec<i16>)> = None;
            let (mut out, mut at, mut packets) = (Vec::<i16>::new(), 0, 0);
            while at + 8 <= bit.len() {
                let len = u32::from_be_bytes(bit[at..at + 4].try_into().unwrap()) as usize;
                let range = u32::from_be_bytes(bit[at + 4..at + 8].try_into().unwrap());
                let packet = &bit[at + 8..at + 8 + len];
                at += 8 + len;
                let mut pcm = vec![0i16; 2 * 5760];
                let samples = d.decode(packet, &mut pcm).unwrap_or_else(|e| panic!("vector {n} packet {packets}: error {e}"));
                assert_eq!(d.final_range(), range, "vector {n} packet {packets}: final range");
                out.extend_from_slice(&pcm[..2 * samples]);
                // Halfway, a second decoder takes a copy of the state and follows.
                if packets == 100 {
                    let mut c = Decoder::new(2);
                    c.restore(&d.bytes());
                    copy = Some((c, Vec::new()));
                } else if let Some((c, tail)) = copy.as_mut() {
                    let mut p = vec![0i16; 2 * 5760];
                    let k = c.decode(packet, &mut p).unwrap();
                    tail.extend_from_slice(&p[..2 * k]);
                    assert_eq!(&p[..2 * k], &pcm[..2 * samples], "vector {n} packet {packets}: copied state");
                }
                packets += 1;
            }
            let reference: Vec<i16> = reference.as_chunks::<2>().0.iter().map(|b| i16::from_le_bytes(*b)).collect();
            assert_eq!(out.len(), reference.len(), "vector {n}: length");
            let (signal, noise) = out.iter().zip(&reference).fold((0f64, 0f64), |(s, e), (&a, &b)| (s + f64::from(b).powi(2), e + (f64::from(a) - f64::from(b)).powi(2)));
            let snr = 10.0 * (signal / noise.max(1.0)).log10();
            eprintln!("vector {n}: {packets} packets, SNR {snr:.1} dB");
            assert!(snr > 40.0, "vector {n}: SNR {snr:.1} dB");
        }
    }
}
