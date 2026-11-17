//! Switch 2 Joy-Con over Bluetooth LE (`--joycon`, D87): the pair as one more
//! controller. Windows cannot add them as devices (they refuse the standard
//! pairing), so a thread scans for them, connects without pairing, and their
//! notified reports go straight into the state the hid samples read. Nothing is
//! stored on a controller: the only frames ever written are player light 1 and a
//! read of the factory stick calibration. Buttons and sticks only: no mouse, no
//! motion, no rumble.

use crate::pads;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering::Relaxed};

/// One Joy-Con's state, and when its last report came (ms on `clock`; 0: never).
#[derive(Default)]
pub struct Side {
    state: pads::Shared,
    heard: AtomicU64,
}

/// Left, then right.
pub type Pair = [Side; 2];
const SIDES: [&str; 2] = ["left", "right"];

/// A Joy-Con silent for longer than this (ms) holds nothing: the longest gap
/// measured between two reports of a live link is 179 ms, and Windows takes 2 s
/// or more to call a link lost.
const SILENCE: u64 = 400;

/// Milliseconds since the first call, never 0.
pub fn clock() -> u64 {
    static START: std::sync::OnceLock<std::time::Instant> = std::sync::OnceLock::new();
    START.get_or_init(std::time::Instant::now).elapsed().as_millis() as u64 + 1
}

/// Nintendo's Bluetooth company identifier; a controller's manufacturer data
/// starts with `CONTROLLER`, then its product, little-endian: a left or a right
/// Joy-Con 2.
const NINTENDO: u16 = 0x0553;
const CONTROLLER: [u8; 5] = [1, 0, 3, 0x7e, 5];
const PRODUCTS: [u16; 2] = [0x2067, 0x2066];

/// The side of a Joy-Con 2 advertising, and whether it is in sync mode. After a
/// plain press it calls its console, whose address its data then carries (bytes
/// 10 to 15), and drops any other host within a third of a second: only the
/// sync button's mode, without that address, lets Gyroship in.
pub fn advertising(company: u16, data: &[u8]) -> Option<(usize, bool)> {
    if company != NINTENDO || data.len() < 16 || data[..5] != CONTROLLER {
        return None;
    }
    let side = PRODUCTS.iter().position(|&p| p == u16::from_le_bytes([data[5], data[6]]))?;
    Some((side, data[10..16] == [0; 6]))
}

/// The input report's buttons as (byte, mask, Npad bit). SL, SR, Capture and C
/// are not the game's; Home is Gyroship's own (`HOME`).
const BUTTONS: [(usize, u8, u32); 16] = [
    (4, 0x01, 3),  // Y
    (4, 0x02, 2),  // X
    (4, 0x04, 1),  // B
    (4, 0x08, 0),  // A
    (4, 0x40, 7),  // R
    (4, 0x80, 9),  // ZR
    (5, 0x01, 11), // Minus
    (5, 0x02, 10), // Plus
    (5, 0x04, 5),  // right stick
    (5, 0x08, 4),  // left stick
    (6, 0x01, 15), // Down
    (6, 0x02, 13), // Up
    (6, 0x04, 14), // Right
    (6, 0x08, 12), // Left
    (6, 0x40, 6),  // L
    (6, 0x80, 8),  // ZL
];
/// Home, kept in a side's `held` above the Npad bits to see it go down: like a
/// pad's Guide, it switches --boost perfect and never reaches the game.
const HOME: u64 = 1 << 63;
/// The Npad buttons a Joy-Con can hold.
const NPAD: u64 = 0xffff;

/// The buttons of an input report: Npad bits, and `HOME`.
pub fn buttons(report: &[u8]) -> u64 {
    let on = |byte: usize, mask: u8| report.get(byte).is_some_and(|b| b & mask != 0);
    BUTTONS.iter().filter(|b| on(b.0, b.1)).fold(if on(5, 0x10) { HOME } else { 0 }, |v, b| v | 1 << b.2)
}

/// A stick's two 12-bit values, packed in three bytes: x, which grows to the
/// right, then y, which grows upwards.
fn unpack(s: &[u8]) -> [u16; 2] {
    [u16::from(s[0]) | (u16::from(s[1]) & 0x0f) << 8, u16::from(s[1]) >> 4 | u16::from(s[2]) << 4]
}

/// A stick's factory calibration: its centre, then its travel above and below
/// the centre, on each axis.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Calibration {
    centre: [u16; 2],
    above: [u16; 2],
    below: [u16; 2],
}

/// The middle of the range and the travel of the sticks measured: for a
/// controller whose calibration cannot be read.
pub const TYPICAL: Calibration = Calibration { centre: [2048, 2048], above: [1200, 1200], below: [1200, 1200] };

/// The calibration in a factory block (0x40 bytes of flash at 0x13080): nine
/// bytes at 0x28, three packed pairs. Erased flash (all ones), a zero or a short
/// block give the typical values.
pub fn calibration(block: &[u8]) -> Calibration {
    let Some(raw) = block.get(0x28..0x31) else { return TYPICAL };
    let [centre, above, below] = [0, 3, 6].map(|i| unpack(&raw[i..i + 3]));
    if [centre, above, below].as_flattened().iter().any(|&v| v == 0 || v == 0xfff) {
        return TYPICAL;
    }
    Calibration { centre, above, below }
}

/// A raw stick as the pads': each axis over its calibrated travel, then the pads'
/// dead zone.
// ponytail: the pads' own dead zone (XInput's, 24 %); a smaller one if the Joy-Con's feel asks for it.
pub fn stick(raw: [u16; 2], c: &Calibration) -> [i32; 2] {
    let axis = |i: usize| {
        let d = f32::from(raw[i]) - f32::from(c.centre[i]);
        ((d / f32::from(if d >= 0.0 { c.above[i] } else { c.below[i] })).clamp(-1.0, 1.0) * 32767.0) as i16
    };
    // The pads' stick takes SDL's Y, which grows downwards.
    pads::stick(axis(0), -axis(1))
}

/// One input report of the Joy-Con of `side`, at `now` (63 bytes: a clock, the
/// buttons at 4, the left stick at 10 and the right one at 13), into its state:
/// what is held, what went down since the last sample (a tap shorter than a
/// sample counts once), and its own stick; the other stick's bytes are not its
/// to say. True when Home goes down.
pub fn report(side: usize, b: &[u8], c: &Calibration, to: &Side, now: u64) -> bool {
    if b.len() < 16 {
        return false;
    }
    let held = buttons(b);
    let down = held & !to.state.held.swap(held, Relaxed);
    to.state.pressed.fetch_or(down & NPAD, Relaxed);
    let at = 10 + 3 * side;
    to.state.sticks.store(pads::pack(stick(unpack(&b[at..at + 3]), c), side as u32), Relaxed);
    to.heard.store(now, Relaxed);
    down & HOME != 0
}

/// A lost Joy-Con holds nothing: no button stays down, its stick rests.
pub fn release(side: &Side) {
    side.heard.store(0, Relaxed);
}

/// What the pair adds to a sample at `now`: its buttons, held or pressed since
/// the last one, and each side's stick. A side that is lost or silent adds
/// nothing, whatever its last report said: no button stays down.
pub fn sample(pair: &Pair, now: u64) -> (u64, [i32; 2], [i32; 2]) {
    let live = |s: &Side| {
        let heard = s.heard.load(Relaxed);
        heard != 0 && now.saturating_sub(heard) <= SILENCE
    };
    let buttons = pair.iter().fold(0, |v, s| {
        let pressed = s.state.pressed.swap(0, Relaxed);
        if live(s) { v | s.state.held.load(Relaxed) | pressed } else { v }
    });
    let sticks = |s: &Side| if live(s) { s.state.sticks() } else { ([0, 0], [0, 0]) };
    (buttons & NPAD, sticks(&pair[0]).0, sticks(&pair[1]).1)
}

/// The two frames Gyroship writes, and nothing else, ever: player light 1, and
/// the read of 0x40 bytes of flash at 0x13080. A frame is a command, 0x91 (a
/// request), 0x01 (over Bluetooth), a subcommand, 0, the data's length, 0, 0,
/// then the data.
const LIGHT: [u8; 16] = [0x09, 0x91, 0x01, 0x07, 0, 8, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0];
const READ: [u8; 16] = [0x02, 0x91, 0x01, 0x04, 0, 8, 0, 0, 0x40, 0x7e, 0, 0, 0x80, 0x30, 0x01, 0];

/// The block of a notified answer to `READ`: after the answer's header (the
/// command, 1, 1, the subcommand, four more bytes), the length and the address
/// asked, which it must repeat.
pub fn block(answer: &[u8]) -> Option<&[u8]> {
    let at = answer.windows(4).position(|w| w == [READ[0], 1, 1, READ[3]])?;
    (answer.get(at + 12..at + 16)? == &READ[12..]).then(|| &answer[at + 16..])
}

/// Starts the thread that finds, connects and reads the pair.
pub fn start(pair: Arc<Pair>) {
    #[cfg(windows)]
    if let Err(e) = std::thread::Builder::new().name("joycon".into()).spawn(move || driver::run(&pair)) {
        eprintln!("gyroship: joycon: {e}");
    }
    #[cfg(not(windows))]
    {
        let _ = (pair, SIDES, LIGHT);
        eprintln!("gyroship: --joycon: Windows only");
    }
}

#[cfg(windows)]
mod driver {
    use super::*;
    use std::sync::mpsc::{self, Sender};
    use std::time::{Duration, Instant};
    use windows::Devices::Bluetooth::Advertisement::{BluetoothLEAdvertisementReceivedEventArgs, BluetoothLEAdvertisementWatcher, BluetoothLEAdvertisementWatcherStoppedEventArgs, BluetoothLEScanningMode};
    use windows::Devices::Bluetooth::GenericAttributeProfile::{GattCharacteristic, GattClientCharacteristicConfigurationDescriptorValue as Descriptor, GattValueChangedEventArgs, GattWriteOption};
    use windows::Devices::Bluetooth::{BluetoothCacheMode, BluetoothLEDevice, BluetoothLEPreferredConnectionParameters};
    use windows::Foundation::TypedEventHandler;
    use windows::Storage::Streams::{DataReader, DataWriter, IBuffer};
    use windows::core::{Error, GUID, HRESULT, IInspectable, Result};

    /// The controller's characteristics: the input report (notified), the commands
    /// (written without response) and their answers (notified).
    const INPUT: u128 = 0xab7de9be_89fe_49ad_828f_118f09df7fd2;
    const COMMAND: u128 = 0x649d4ac9_8eb7_4e6c_af44_1ea54fe5f005;
    const ANSWERS: u128 = 0xc765a961_d9d8_4d36_a20a_5315b111836a;

    /// What the system's callbacks tell the thread.
    enum Event {
        Seen { side: usize, address: u64, sync: bool },
        Lost(usize),
        /// A link's connection interval, in units of 1.25 ms.
        Interval(usize, u16),
        /// The scan ended by itself, with this Bluetooth error.
        Stopped(i32),
    }

    /// A connected Joy-Con, and what must live as long as its link: the services,
    /// the characteristics, the request for a fast link.
    struct Link(BluetoothLEDevice, #[allow(dead_code)] Box<dyn std::any::Any>);

    /// A failure of ours, as one of Windows': E_FAIL and why.
    fn failed(why: &str) -> Error {
        Error::new(HRESULT(0x8000_4005_u32 as i32), why)
    }

    fn bytes(b: &IBuffer) -> Result<Vec<u8>> {
        let mut v = vec![0; b.Length()? as usize];
        if !v.is_empty() {
            DataReader::FromBuffer(b)?.ReadBytes(&mut v)?;
        }
        Ok(v)
    }

    /// Windows reports a write that did not reach the controller as a status, not as an error.
    fn reached(status: i32) -> Result<()> {
        if status == 0 { Ok(()) } else { Err(failed("the controller did not take a write")) }
    }

    fn notify(c: &GattCharacteristic) -> Result<()> {
        reached(c.WriteClientCharacteristicConfigurationDescriptorAsync(Descriptor::Notify)?.join()?.0)
    }

    /// Writes one of the two frames, without response.
    fn write(c: &GattCharacteristic, frame: &'static [u8; 16]) -> Result<()> {
        let w = DataWriter::new()?;
        w.WriteBytes(frame)?;
        reached(c.WriteValueWithOptionAsync(&w.DetachBuffer()?, GattWriteOption::WriteWithoutResponse)?.join()?.0)
    }

    /// Connects to the Joy-Con of `side` at `address` and starts its reports.
    fn connect(side: usize, address: u64, pair: &Arc<Pair>, events: &Sender<Event>) -> Result<Link> {
        // A device Windows has not just seen comes back null: an error without a code.
        let device = BluetoothLEDevice::FromBluetoothAddressAsync(address)?.join().map_err(|e| if e.code().is_ok() { failed("Windows does not know it yet") } else { e })?;
        let s = events.clone();
        device.ConnectionStatusChanged(&TypedEventHandler::<BluetoothLEDevice, IInspectable>::new(move |d, _| {
            if d.ok()?.ConnectionStatus()?.0 == 0 {
                let _ = s.send(Event::Lost(side));
            }
            Ok(())
        }))?;
        // The uncached discovery is what connects.
        let discovery = device.GetGattServicesWithCacheModeAsync(BluetoothCacheMode::Uncached)?.join()?;
        if discovery.Status()?.0 != 0 {
            return Err(failed("out of reach"));
        }
        eprintln!("gyroship: joycon: the {} one answers: let go of its button", SIDES[side]);
        let services = discovery.Services()?;
        let mut found = [const { None }; 3];
        for s in &services {
            let Ok(list) = s.GetCharacteristicsAsync().and_then(|op| op.join()).and_then(|r| r.Characteristics()) else { continue };
            for c in &list {
                if let Some(i) = [INPUT, COMMAND, ANSWERS].iter().position(|&u| c.Uuid().is_ok_and(|id| id == GUID::from_u128(u))) {
                    found[i] = Some(c);
                }
            }
        }
        let [Some(input), command, answers] = found else { return Err(failed("it has no input report")) };
        // The factory calibration, when the controller answers within a second.
        let mut c = TYPICAL;
        if let (Some(command), Some(answers)) = (&command, &answers) {
            let (tx, rx) = mpsc::channel();
            let token = answers.ValueChanged(&TypedEventHandler::<GattCharacteristic, GattValueChangedEventArgs>::new(move |_, a| {
                let _ = tx.send(bytes(&a.ok()?.CharacteristicValue()?)?);
                Ok(())
            }))?;
            if notify(answers).and_then(|()| write(command, &READ)).is_ok() {
                let end = Instant::now() + Duration::from_secs(1);
                while let Ok(answer) = rx.recv_timeout(end.saturating_duration_since(Instant::now())) {
                    if let Some(b) = block(&answer) {
                        c = calibration(b);
                        break;
                    }
                }
            }
            let _ = answers.RemoveValueChanged(token);
        }
        // The handler first, then the descriptor: reports flow without any command.
        let to = pair.clone();
        input.ValueChanged(&TypedEventHandler::<GattCharacteristic, GattValueChangedEventArgs>::new(move |_, a| {
            if report(side, &bytes(&a.ok()?.CharacteristicValue()?)?, &c, &to[side], clock()) {
                gyroship_hle::practice::toggle_perfect();
            }
            Ok(())
        }))?;
        notify(&input)?;
        // Windows' own interval is 60 ms a few seconds after the connection; its
        // throughput preset asks for 15 ms. The request lives as long as the link, and
        // each change of the interval is told.
        let s = events.clone();
        let _ = device.ConnectionParametersChanged(&TypedEventHandler::<BluetoothLEDevice, IInspectable>::new(move |d, _| {
            let _ = s.send(Event::Interval(side, d.ok()?.GetConnectionParameters()?.ConnectionInterval()?));
            Ok(())
        }));
        let request = BluetoothLEPreferredConnectionParameters::ThroughputOptimized().and_then(|p| device.RequestPreferredConnectionParameters(&p));
        // 1: accepted.
        match request.as_ref().map(|r| r.Status().map_or(-1, |s| s.0)) {
            Ok(1) => {}
            Ok(status) => eprintln!("gyroship: joycon: Windows refused a fast link to the {} one (status {status}): expect input lag", SIDES[side]),
            Err(e) => eprintln!("gyroship: joycon: no fast link to the {} one ({e}): Windows 11 is needed, expect input lag", SIDES[side]),
        }
        if let Some(command) = &command {
            let _ = write(command, &LIGHT);
        }
        // A link that fell during all this would never be told lost.
        if device.ConnectionStatus()?.0 == 0 {
            return Err(failed("it dropped the link"));
        }
        eprintln!("gyroship: joycon: the {} one is connected{}", SIDES[side], if c == TYPICAL { " (typical calibration)" } else { "" });
        Ok(Link(device, Box::new((services, input, command, answers, request))))
    }

    /// Scans while a side is missing and connects what shows up in sync mode; a
    /// lost side lets go of everything and is scanned for again.
    pub fn run(pair: &Arc<Pair>) {
        let (tx, rx) = mpsc::channel();
        let watcher = (|| {
            let w = BluetoothLEAdvertisementWatcher::new()?;
            w.SetScanningMode(BluetoothLEScanningMode::Passive)?;
            let s = tx.clone();
            w.Received(&TypedEventHandler::<BluetoothLEAdvertisementWatcher, BluetoothLEAdvertisementReceivedEventArgs>::new(move |_, a| {
                let a = a.ok()?;
                let list = a.Advertisement()?.ManufacturerData()?;
                for i in 0..list.Size()? {
                    let m = list.GetAt(i)?;
                    if let Some((side, sync)) = advertising(m.CompanyId()?, &bytes(&m.Data()?)?) {
                        let _ = s.send(Event::Seen { side, address: a.BluetoothAddress()?, sync });
                    }
                }
                Ok(())
            }))?;
            let s = tx.clone();
            w.Stopped(&TypedEventHandler::<BluetoothLEAdvertisementWatcher, BluetoothLEAdvertisementWatcherStoppedEventArgs>::new(move |_, a| {
                let _ = s.send(Event::Stopped(a.ok()?.Error()?.0));
                Ok(())
            }))?;
            w.Start()?;
            Ok::<_, Error>(w)
        })();
        let watcher = match watcher {
            Ok(w) => w,
            Err(e) => return eprintln!("gyroship: joycon: no Bluetooth scan ({e}): is Bluetooth on?"),
        };
        eprintln!("gyroship: joycon: hold the sync button of each Joy-Con (the small round one between SL and SR); the Switch 2 must be off");
        let mut links: [Option<Link>; 2] = [None, None];
        // No tight retries: a controller is left alone for 2 s after a failed try.
        let (mut told, mut retry) = ([false; 2], [Instant::now(); 2]);
        for event in rx {
            match event {
                Event::Seen { side, address, sync } if links[side].is_none() && Instant::now() >= retry[side] => {
                    if !sync {
                        if !std::mem::replace(&mut told[side], true) {
                            eprintln!("gyroship: joycon: the {} one is calling its console: hold its sync button", SIDES[side]);
                        }
                        continue;
                    }
                    match connect(side, address, pair, &tx) {
                        Ok(link) => links[side] = Some(link),
                        Err(e) => {
                            eprintln!("gyroship: joycon: the {} one did not connect ({e}): keep holding its sync button", SIDES[side]);
                            retry[side] = Instant::now() + Duration::from_secs(2);
                        }
                    }
                    // Scanning takes radio time from the links: none once both are up.
                    if links.iter().all(Option::is_some) {
                        let _ = watcher.Stop();
                    }
                }
                // A loss told by an earlier, failed try must not end a link that is up.
                Event::Lost(side) if links[side].as_ref().is_some_and(|l| !l.0.ConnectionStatus().is_ok_and(|s| s.0 != 0)) => {
                    if let Some(Link(device, _)) = links[side].take() {
                        let _ = device.Close();
                    }
                    release(&pair[side]);
                    told[side] = false;
                    eprintln!("gyroship: joycon: the {} one is lost: hold its sync button to connect it again", SIDES[side]);
                    let _ = watcher.Start();
                }
                // Past 20 ms the controller is read too rarely for a 60 Hz game.
                Event::Interval(side, units) if links[side].is_some() || units != 0 => {
                    let ms = f32::from(units) * 1.25;
                    eprintln!("gyroship: joycon: the {} link's interval is {ms} ms{}", SIDES[side], if units > 16 { ": slow, expect input lag" } else { "" });
                }
                // 0: the scan was stopped by this thread.
                Event::Stopped(error) if error != 0 => eprintln!("gyroship: joycon: the Bluetooth scan stopped (error {error}): is Bluetooth on?"),
                Event::Seen { .. } | Event::Lost(_) | Event::Interval(..) | Event::Stopped(_) => {}
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A report with these button bytes (4, 5, 6) and stick bytes (left, right).
    fn input(buttons: [u8; 3], left: [u8; 3], right: [u8; 3]) -> Vec<u8> {
        let mut b = vec![0; 63];
        b[4..7].copy_from_slice(&buttons);
        b[7] = 0xe0;
        b[10..13].copy_from_slice(&left);
        b[13..16].copy_from_slice(&right);
        b
    }

    /// x and y as a stick's three bytes.
    fn packed(x: u16, y: u16) -> [u8; 3] {
        [x as u8, (x >> 8) as u8 | (y as u8) << 4, (y >> 4) as u8]
    }

    const REST: [u8; 3] = [0xff, 0xf7, 0x7f];

    #[test]
    fn buttons_follow_the_map_measured_on_both_controllers() {
        let one = |byte: usize, mask: u8| {
            let mut b = [0; 3];
            b[byte - 4] = mask;
            buttons(&input(b, REST, REST))
        };
        // The right Joy-Con: A, B, X, Y, R, ZR, Plus, its stick.
        assert_eq!([one(4, 8), one(4, 4), one(4, 2), one(4, 1), one(4, 0x40), one(4, 0x80), one(5, 2), one(5, 4)], [1, 1 << 1, 1 << 2, 1 << 3, 1 << 7, 1 << 9, 1 << 10, 1 << 5]);
        // The left one: Up, Down, Left, Right, L, ZL, Minus, its stick.
        assert_eq!([one(6, 2), one(6, 1), one(6, 8), one(6, 4), one(6, 0x40), one(6, 0x80), one(5, 1), one(5, 8)], [1 << 13, 1 << 15, 1 << 12, 1 << 14, 1 << 6, 1 << 8, 1 << 11, 1 << 4]);
        // SL and SR of both, Capture and C are no button of the game; Home is not one either.
        assert_eq!([one(4, 0x10), one(4, 0x20), one(6, 0x10), one(6, 0x20), one(5, 0x20), one(5, 0x40)], [0; 6]);
        assert_eq!(one(5, 0x10) & NPAD, 0);
        assert_eq!(buttons(&[]), 0, "a short report holds nothing");
    }

    #[test]
    fn a_stick_is_centred_scaled_by_its_travel_and_points_up() {
        let mut flash = [0xff; 0x40];
        flash[0x28..0x31].copy_from_slice(&[packed(2100, 2000), packed(1000, 1200), packed(1100, 1300)].concat());
        let c = calibration(&flash);
        assert_eq!(c, Calibration { centre: [2100, 2000], above: [1000, 1200], below: [1100, 1300] });
        assert_eq!(stick([2100, 2000], &c), [0, 0]);
        assert_eq!(stick([2130, 1980], &c), [0, 0], "at rest, a little off the centre");
        assert_eq!(stick([3100, 2000], &c), [32767, 0], "right, at its travel above the centre");
        assert_eq!(stick([1000, 2000], &c), [-32767, 0], "left, at its travel below");
        assert_eq!(stick([2100, 3200], &c), [0, 32767], "up is up");
        assert_eq!(stick([2100, 0], &c), [0, -32767], "past the travel is the travel");
        // Erased flash, a short block, zeros, a centre without its travel: the typical values.
        let mut half = flash;
        half[0x2b..0x31].fill(0xff);
        assert_eq!([calibration(&[0xff; 0x40]), calibration(&flash[..0x30]), calibration(&[0; 0x40]), calibration(&half)], [TYPICAL; 4]);
        assert_eq!(unpack(&REST), [2047, 2047]);
    }

    #[test]
    fn each_side_fills_its_own_stick_and_a_tap_counts_once() {
        let pair = Pair::default();
        let (up, right) = (packed(2048, 3248), packed(3248, 2048));
        // The left Joy-Con: Up held, its stick up; what it says of the right stick is not read.
        assert!(!report(0, &input([0, 0, 2], up, right), &TYPICAL, &pair[0], 10));
        assert_eq!(sample(&pair, 10), (1 << 13, [0, 32767], [0, 0]));
        // The right one: A down then up between two samples, its stick to the right.
        assert!(!report(1, &input([8, 0, 0], REST, right), &TYPICAL, &pair[1], 11));
        assert!(!report(1, &input([0, 0, 0], REST, right), &TYPICAL, &pair[1], 12));
        assert_eq!(sample(&pair, 12), (1 | 1 << 13, [0, 32767], [32767, 0]));
        assert_eq!(sample(&pair, 13).0, 1 << 13, "the tap was one press");
        // A button held over a sample and let go is not held one sample more.
        report(1, &input([8, 0, 0], REST, right), &TYPICAL, &pair[1], 14);
        assert_eq!(sample(&pair, 14).0, 1 | 1 << 13);
        report(1, &input([8, 0, 0], REST, right), &TYPICAL, &pair[1], 15);
        report(1, &input([0, 0, 0], REST, right), &TYPICAL, &pair[1], 16);
        assert_eq!(sample(&pair, 16).0, 1 << 13);
        // Home goes down once, and is never a button of the game.
        assert!(report(1, &input([0, 0x10, 0], REST, right), &TYPICAL, &pair[1], 17));
        assert!(!report(1, &input([0, 0x10, 0], REST, right), &TYPICAL, &pair[1], 18));
        assert_eq!(sample(&pair, 18).0, 1 << 13);
        // A short report changes nothing, on either side.
        for side in [0, 1] {
            assert!(!report(side, &[0xff; 15], &TYPICAL, &pair[side], 19));
        }
        assert_eq!(sample(&pair, 19), (1 << 13, [0, 32767], [32767, 0]));
    }

    /// No stuck button: a lost Joy-Con, or one silent for too long, holds nothing,
    /// even if a late report of its link comes after the loss.
    #[test]
    fn a_lost_or_silent_side_lets_go_of_everything() {
        let pair = Pair::default();
        // ZL held and the stick up on the left, ZR held and the stick to the right on the right.
        let (left, right) = (input([0, 0, 0x80], packed(2048, 3248), REST), input([0x80, 0, 0], REST, packed(3248, 2048)));
        assert_eq!(sample(&pair, clock()), (0, [0, 0], [0, 0]), "nothing before the first report");
        report(0, &left, &TYPICAL, &pair[0], 1000);
        report(1, &right, &TYPICAL, &pair[1], 1000);
        assert_eq!(sample(&pair, 1000 + SILENCE), (1 << 8 | 1 << 9, [0, 32767], [32767, 0]));
        report(1, &right, &TYPICAL, &pair[1], 1300);
        assert_eq!(sample(&pair, 1001 + SILENCE), (1 << 9, [0, 0], [32767, 0]), "the left one fell silent");
        release(&pair[1]);
        assert_eq!(sample(&pair, 1002 + SILENCE), (0, [0, 0], [0, 0]), "the right one is lost");
        // A report still on its way when the loss was told: gone with the silence.
        report(1, &right, &TYPICAL, &pair[1], 1500);
        assert_eq!(sample(&pair, 1501 + SILENCE), (0, [0, 0], [0, 0]));
    }

    #[test]
    fn only_a_joy_con_is_seen_and_only_sync_mode_is_answered() {
        let data = |product: u16, host: u8| [&CONTROLLER[..], &product.to_le_bytes(), &[0, 1, 0], &[host; 6], &[0x0f, 0, 0, 0, 0, 0, 0, 0]].concat();
        assert_eq!(advertising(NINTENDO, &data(0x2067, 0)), Some((0, true)));
        assert_eq!(advertising(NINTENDO, &data(0x2066, 0)), Some((1, true)));
        assert_eq!(advertising(NINTENDO, &data(0x2066, 0x40)), Some((1, false)), "it calls its console");
        assert_eq!(advertising(NINTENDO, &data(0x2069, 0)), None, "another controller");
        assert_eq!(advertising(0x004c, &data(0x2066, 0)), None, "another maker");
        assert_eq!(advertising(NINTENDO, &[3, 0, 1, 0, 0x80, 1, 2, 3, 4, 5, 6, 1]), None, "the console itself");
        let mut other = data(0x2066, 0);
        other[0] = 3;
        assert_eq!(advertising(NINTENDO, &other), None, "not a controller's data");
        assert_eq!(advertising(NINTENDO, &data(0x2066, 0)[..15]), None);
    }

    /// The two frames are the light and one flash read, byte for byte what the
    /// controllers answered in the trial, and an answer counts only when it
    /// repeats the address asked.
    #[test]
    fn the_frames_written_are_the_light_and_the_calibration_read() {
        assert_eq!(LIGHT, [0x09, 0x91, 0x01, 0x07, 0, 8, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0]);
        assert_eq!(READ, [0x02, 0x91, 0x01, 0x04, 0, 8, 0, 0, 0x40, 0x7e, 0, 0, 0x80, 0x30, 0x01, 0]);
        let answer = |address: [u8; 4]| [&[0x02, 1, 1, 0x04, 0x10, 0x78, 0, 0, 0x40, 0, 0, 0][..], &address, &[0xab; 0x40]].concat();
        assert_eq!(block(&answer([0x80, 0x30, 1, 0])), Some(&[0xab; 0x40][..]));
        assert_eq!(block(&answer([0xc0, 0x30, 1, 0])), None, "another read's answer");
        assert_eq!(block(&[0x09, 1, 1, 0x07, 0x10, 0x78, 0, 0]), None);
    }
}
