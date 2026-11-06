//! `time:u` (IStaticService) and its clocks. Times come from the virtual
//! clock, never from the host: the run starts at a fixed date. The network
//! clock is never synchronized (rule 9).

use crate::Kernel;
use crate::ipc::{Reply, Request};
use crate::service::Service;

/// POSIX time of the run's start: 2026-01-01 00:00:00 UTC.
pub const EPOCH: i64 = 1_767_225_600;
/// Size of time's shared memory.
pub const SHARED_SIZE: u64 = 0x1000;
/// ClockSnapshot, as GetClockSnapshot writes it.
const SNAPSHOT_SIZE: usize = 0xD0;

#[derive(Debug)]
pub enum Time {
    Static,
    /// 0 user, 1 network, 4 local, 5 ephemeral network.
    SystemClock(u32),
    SteadyClock,
    ZoneService,
}

impl Time {
    pub fn label(&self) -> String {
        match self {
            Time::Static => "time:u".into(),
            Time::SystemClock(k) => format!("time ISystemClock {k}"),
            Time::SteadyClock => "time ISteadyClock".into(),
            Time::ZoneService => "time ITimeZoneService".into(),
        }
    }

    pub fn request(&mut self, k: &mut Kernel, req: &Request) -> Option<Reply> {
        let time = |t: Time| Some(Reply::object(Service::Time(t)));
        let seconds = EPOCH + (k.now() / 31_250_000) as i64;
        Some(match (self, req.command) {
            (Time::Static, c @ (0 | 1 | 4 | 5)) => return time(Time::SystemClock(c)),
            (Time::Static, 2) => return time(Time::SteadyClock),
            (Time::Static, 3) => return time(Time::ZoneService),
            // GetSharedMemoryNativeHandle.
            (Time::Static, 20) => {
                let m = match k.time_shared {
                    Some(m) => m,
                    None => {
                        let m = k.create_shared_memory(SHARED_SIZE);
                        k.time_shared = Some(m);
                        fill_shared(k, m);
                        m
                    }
                };
                Reply { copy_handles: vec![k.handles.add(crate::thread::Object::SharedMemory(m))?], ..Reply::ok() }
            }
            // IsStandardUserSystemClockAutomaticCorrectionEnabled.
            (Time::Static, 100) => Reply::with(&[0]),
            // GetClockSnapshot: zeros but the times, into the output buffer.
            (Time::Static, 400) => {
                if let Some(b) = req.b.first().filter(|b| b.size as usize >= SNAPSHOT_SIZE) {
                    let mut s = vec![0u8; SNAPSHOT_SIZE];
                    s[0x40..0x48].copy_from_slice(&seconds.to_le_bytes());
                    s[0x48..0x50].copy_from_slice(&seconds.to_le_bytes());
                    let _ = k.mem.write(b.addr, &s);
                }
                Reply::ok()
            }
            // GetCurrentTime. The network clock is never set.
            (Time::SystemClock(1 | 5), 0) => Reply::error(NETWORK_NOT_SET),
            (Time::SystemClock(_), 0) => Reply::with(&seconds.to_le_bytes()),
            // GetSystemClockContext.
            (Time::SystemClock(_), 2) => Reply::with(&[0; 0x20]),
            // GetCurrentTimePoint: seconds since boot and a clock source id.
            (Time::SteadyClock, 0) => Reply::with(&[&((k.now() / 31_250_000) as i64).to_le_bytes()[..], &[0; 16]].concat()),
            _ => return None,
        })
    }
}

/// The steady clock's source id: any non-zero value, fixed by the seed.
fn clock_source(k: &Kernel) -> [u8; 16] {
    let s = k.config.seed ^ 0x7469_6D65; // "time"
    [s.to_le_bytes(), (!s).to_le_bytes()].concat().try_into().expect("16 bytes")
}

/// time's shared memory. Each record is read under a sequence counter
/// (even: stable), then one of two copies.
/// - 0x00 steady clock context, 0x38 local system clock context, 0x80
///   network system clock context, 0xC8 automatic correction: the public
///   Switch 1 layout (switchbrew), an assumption for the Switch 2.
/// - 0xD0 continuous adjustment: two 0x38-byte entries at 0xD8 (base tick,
///   multiplier, shift, offset, bound, then the source id at +0x28), as
///   `nn::time::detail::SharedMemoryClient::GetStandardSteadyClockCurrentTimespec`
///   reads them. A zero source id is an error there (time 103), which makes
///   `clock_gettime(CLOCK_MONOTONIC)` fail; zero parameters mean steady time
///   = tick time.
fn fill_shared(k: &mut Kernel, m: usize) {
    let id = clock_source(k);
    let mut s = vec![0u8; 0x110 + 0x38];
    for copy in [0x08, 0x20] {
        s[copy + 8..copy + 24].copy_from_slice(&id);
    }
    for copy in [0x40, 0x60] {
        // Offset from the steady clock to POSIX time, then its time point (0, the source id).
        s[copy..copy + 8].copy_from_slice(&EPOCH.to_le_bytes());
        s[copy + 16..copy + 32].copy_from_slice(&id);
    }
    for entry in [0xD8, 0xD8 + 0x38] {
        s[entry + 0x28..entry + 0x38].copy_from_slice(&id);
    }
    k.write_shared(m, 0, &s);
}

/// time's ClockIsNotSynchronized-like result for the network clock (module
/// 116, description 102): there is never a network time.
pub const NETWORK_NOT_SET: u32 = 116 | 102 << 9;
