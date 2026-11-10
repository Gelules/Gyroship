//! `time:u` (IStaticService) and its clocks. Times come from the kernel's
//! clock (virtual, or real in the play engine), never from the host's
//! calendar: the run starts at a fixed date. The network clock is never
//! synchronized (rule 9).

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
            // GetClockSnapshot, into the output buffer.
            (Time::Static, 400) => {
                if let Some(b) = req.out_buffer(0).filter(|b| b.size as usize >= SNAPSHOT_SIZE) {
                    let s = snapshot(k, seconds);
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

/// Year, month (1-12), day of a day count since 1970-01-01 (proleptic Gregorian).
fn civil(days: i64) -> (i64, u8, u8) {
    let z = days + 719_468;
    let era = z.div_euclid(146_097);
    let doe = z - era * 146_097;
    let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let day = (doy - (153 * mp + 2) / 5 + 1) as u8;
    let month = if mp < 10 { mp + 3 } else { mp - 9 } as u8;
    (yoe + era * 400 + i64::from(month <= 2), month, day)
}

/// A ClockSnapshot (public Switch 1 layout, 0xD0 bytes) at POSIX time
/// `seconds`, in UTC: the user clock is set, the network clock is not.
fn snapshot(k: &Kernel, seconds: i64) -> Vec<u8> {
    let id = clock_source(k);
    let steady = (k.now() / 31_250_000) as i64;
    let mut s = vec![0u8; SNAPSHOT_SIZE];
    // User clock context: offset from steady time, then the steady time point it refers to.
    s[0x00..0x08].copy_from_slice(&EPOCH.to_le_bytes());
    s[0x10..0x20].copy_from_slice(&id);
    s[0x40..0x48].copy_from_slice(&seconds.to_le_bytes());
    // User calendar time and its additional information (day of week, day of year, zone "UTC").
    let days = seconds.div_euclid(86_400);
    let secs = seconds.rem_euclid(86_400);
    let (year, month, day) = civil(days);
    s[0x50..0x52].copy_from_slice(&(year as u16).to_le_bytes());
    s[0x52..0x57].copy_from_slice(&[month, day, (secs / 3600) as u8, (secs / 60 % 60) as u8, (secs % 60) as u8]);
    s[0x60..0x64].copy_from_slice(&((days + 4).rem_euclid(7) as u32).to_le_bytes());
    s[0x64..0x68].copy_from_slice(&((days - civil_days(year)) as u32).to_le_bytes());
    s[0x68..0x6B].copy_from_slice(b"UTC");
    // Steady clock time point, location name, then automatic correction off.
    s[0x90..0x98].copy_from_slice(&steady.to_le_bytes());
    s[0x98..0xA8].copy_from_slice(&id);
    s[0xA8..0xAB].copy_from_slice(b"UTC");
    s
}

/// Days from 1970-01-01 to January 1st of `year`.
fn civil_days(year: i64) -> i64 {
    let y = year - 1;
    365 * (year - 1970) + (y / 4 - y / 100 + y / 400) - (1969 / 4 - 1969 / 100 + 1969 / 400)
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn calendar_dates() {
        assert_eq!(civil(0), (1970, 1, 1));
        assert_eq!(EPOCH / 86_400, 20_454);
        assert_eq!(civil(20_454), (2026, 1, 1));
        assert_eq!(civil(20_454 + 58), (2026, 2, 28));
        assert_eq!(civil(20_454 + 59), (2026, 3, 1));
        assert_eq!(civil(11_016), (2000, 2, 29));
        assert_eq!(civil_days(2026), 20_454);
        // 2026-01-01 is a Thursday (0 is Sunday).
        assert_eq!((20_454 + 4) % 7, 4);
    }
}
