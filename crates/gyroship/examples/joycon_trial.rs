//! The Joy-Con 2 trial of the M6 plan (step 8, J0): finds one Switch 2 Joy-Con over
//! Bluetooth LE, connects without pairing, and logs what a driver needs to know: the
//! connection interval before and after Windows' throughput preset, the report rate,
//! the raw report at each button change, the stick ranges, the firmware version and
//! the factory stick calibration. It stores nothing on the controller: every frame it
//! writes passes a whitelist. The log holds Bluetooth addresses: keep it out of Git.
//! `cargo run --release -p gyroship --example joycon_trial -- <L|R|scan> <log file> [03|37]`

#[cfg(not(windows))]
fn main() {
    eprintln!("joycon_trial: Windows only (Bluetooth LE through WinRT)");
    std::process::exit(2);
}

#[cfg(windows)]
fn main() {
    trial::main()
}

#[cfg(windows)]
mod trial {
    use std::collections::BTreeMap;
    use std::io::Write;
    use std::sync::mpsc::{self, Receiver, Sender};
    use std::time::{Duration, Instant};
    use windows::Devices::Bluetooth::Advertisement::{BluetoothLEAdvertisementReceivedEventArgs, BluetoothLEAdvertisementWatcher, BluetoothLEAdvertisementWatcherStoppedEventArgs, BluetoothLEScanningMode};
    use windows::Devices::Bluetooth::GenericAttributeProfile::{GattCharacteristic, GattClientCharacteristicConfigurationDescriptorValue, GattValueChangedEventArgs, GattWriteOption};
    use windows::Devices::Bluetooth::{BluetoothAdapter, BluetoothCacheMode, BluetoothLEDevice, BluetoothLEPreferredConnectionParameters};
    use windows::Foundation::{IReference, TypedEventHandler};
    use windows::Storage::Streams::{DataReader, DataWriter, IBuffer};
    use windows::core::{GUID, HRESULT, IInspectable};

    /// A step's failure: `<step>: <reason>`, then `hint:` lines.
    type R<T> = Result<T, String>;

    /// The controller's characteristics the trial uses, by UUID: the common input
    /// report (63 bytes, notified), the command one (written without response), the
    /// answers' (notified: the common one, then each side's own, which puts 14 zero
    /// bytes first) and each side's command one (17 zero bytes, then the frame).
    const ROLES: [(u128, &str); 7] = [
        (0xab7de9be_89fe_49ad_828f_118f09df7fd2, "input"),
        (0x649d4ac9_8eb7_4e6c_af44_1ea54fe5f005, "command"),
        (0xc765a961_d9d8_4d36_a20a_5315b111836a, "rsp-a"),
        (0x63a3810f_aec7_474b_9010_3d52403cb996, "rsp-side"),
        (0x640ca58e_0e88_410c_a7f3_426faf2b690b, "rsp-side"),
        (0xce49a830_dced_48ae_931e_c8cf88aadbea, "cmd-side"),
        (0x65a724b3_f1e7_4a61_8078_a342376b27ff, "cmd-side"),
    ];
    /// Nintendo's Bluetooth company identifier, and the start of a controller's
    /// manufacturer data: its product identifier follows, little-endian.
    const NINTENDO: u16 = 0x0553;
    const CONTROLLER: [u8; 5] = [1, 0, 3, 0x7e, 5];
    /// The only flash addresses read: factory calibration of the primary and of the
    /// secondary stick (9 bytes at +0x28 of each block), then the user calibration.
    const READS: [u32; 3] = [0x01_3080, 0x01_30c0, 0x1f_c040];
    const SCAN: Duration = Duration::from_secs(30);
    const HOLD: Duration = Duration::from_secs(12);

    /// A command frame: command, 0x91 (a request), 0x01 (over Bluetooth), subcommand,
    /// 0, the data's length, 0, 0, then the data.
    fn frame(cmd: u8, sub: u8, data: &[u8]) -> Vec<u8> {
        [&[cmd, 0x91, 0x01, sub, 0, data.len() as u8, 0, 0], data].concat()
    }

    /// The frame that reads 0x40 bytes of flash at `addr`.
    fn read(addr: u32) -> Vec<u8> {
        frame(0x02, 0x04, &[[0x40, 0x7e, 0, 0], addr.to_le_bytes()].concat())
    }

    /// The only frames the trial may write: a feature mask and its enabling (buttons
    /// and sticks, 03, or the console's own mask, 37), the player lights, the firmware
    /// version, and a read of at most 0x40 bytes at one of `READS`. Everything else is
    /// refused: firmware updates, flash and memory writes, pairing, shutdown.
    fn allowed(f: &[u8]) -> bool {
        if f.len() < 8 || f[1] != 0x91 || f[2] != 0x01 || f[4] != 0 || f[5] as usize != f.len() - 8 || f[6] != 0 || f[7] != 0 {
            return false;
        }
        match (f[0], f[3], &f[8..]) {
            (0x0c, 0x02 | 0x04, [0x03 | 0x37, 0, 0, 0]) => true,
            (0x09, 0x07, [lights, 0, 0, 0, 0, 0, 0, 0]) => *lights <= 0x0f,
            (0x10, 0x01, []) => true,
            (0x02, 0x04, [n, 0x7e, 0, 0, addr @ ..]) => (1..=0x40).contains(n) && READS.iter().any(|r| addr == r.to_le_bytes()),
            _ => false,
        }
    }

    /// The whitelist checked against itself at each start (an example's tests do not run
    /// with `cargo test`): every frame the trial sends passes, the dangerous ones do not.
    fn self_check() {
        let mut sent = vec![frame(0x09, 0x07, &[1, 0, 0, 0, 0, 0, 0, 0]), frame(0x10, 0x01, &[])];
        sent.extend([0x03, 0x37].into_iter().flat_map(|m| [0x02, 0x04].map(|sub| frame(0x0c, sub, &[m, 0, 0, 0]))));
        sent.extend(READS.map(read));
        assert!(sent.iter().all(|f| allowed(f)), "the whitelist refuses a frame of the trial");
        let mut refused = vec![read(0x1f_a000), frame(0x02, 0x04, &[0x41, 0x7e, 0, 0, 0x80, 0x30, 1, 0]), frame(0x0c, 0x02, &[0xff, 0, 0, 0]), frame(0x09, 0x07, &[0x10, 0, 0, 0, 0, 0, 0, 0])];
        // Firmware update, flash write, erase and memory write, pairing, the 03 family
        // (pairing data, report selection), shutdown, NFC, rumble, battery, the unknown.
        for cmd in [0x0d, 0x15, 0x03, 0x06, 0x01, 0x0a, 0x0b, 0x07, 0x08, 0x11, 0x16] {
            refused.extend((0..=0x0d).flat_map(|sub| [frame(cmd, sub, &[]), frame(cmd, sub, &[0x40, 0x7e, 0, 0, 0x80, 0x30, 1, 0]), frame(cmd, sub, &[3, 0, 0, 0])]));
        }
        refused.extend([0x02, 0x03, 0x05].map(|sub| frame(0x02, sub, &[0x40, 0x7e, 0, 0, 0x80, 0x30, 1, 0])));
        refused.extend([0x01, 0x03, 0x05, 0x06].map(|sub| frame(0x0c, sub, &[3, 0, 0, 0])));
        let mut usb = read(READS[0]);
        usb[2] = 0;
        refused.push(usb);
        assert!(!refused.iter().any(|f| allowed(f)), "the whitelist lets a refused frame pass");
    }

    fn hex(b: &[u8]) -> String {
        b.iter().map(|x| format!("{x:02x}")).collect()
    }

    fn le32(b: &[u8]) -> u32 {
        u32::from_le_bytes([b[0], b[1], b[2], b[3]])
    }

    /// A stick's two 12-bit values, packed in three bytes.
    fn unpack(s: &[u8]) -> (u16, u16) {
        (s[0] as u16 | (s[1] as u16 & 0x0f) << 8, s[1] as u16 >> 4 | (s[2] as u16) << 4)
    }

    fn bytes(b: &IBuffer) -> windows::core::Result<Vec<u8>> {
        let mut v = vec![0; b.Length()? as usize];
        if !v.is_empty() {
            DataReader::FromBuffer(b)?.ReadBytes(&mut v)?;
        }
        Ok(v)
    }

    fn buffer(b: &[u8]) -> windows::core::Result<IBuffer> {
        let w = DataWriter::new()?;
        w.WriteBytes(b)?;
        w.DetachBuffer()
    }

    fn err(e: &windows::core::Error) -> String {
        format!("ERR {} {}", e.code(), e.message())
    }

    /// A WinRT failure nothing handles, as its step's failure.
    fn w<T>(step: &str, r: windows::core::Result<T>) -> R<T> {
        r.map_err(|e| format!("{step}: {}", err(&e)))
    }

    /// The ATT error byte of a GATT result, when there is one.
    fn perr(p: windows::core::Result<IReference<u8>>) -> String {
        p.and_then(|p| p.Value()).map_or("-".into(), |e| format!("{e:02X}"))
    }

    /// A notification's source.
    #[derive(Clone, Copy, PartialEq)]
    enum Role {
        Input,
        RspA,
        RspSide,
    }

    impl Role {
        fn name(self) -> &'static str {
            ["input", "rsp-a", "rsp-side"][self as usize]
        }
    }

    /// What the system's callbacks send to the main thread, stamped as it happens:
    /// callbacks only copy, the main thread prints.
    enum Ev {
        Adv { at: Instant, addr: u64, atype: i32, kind: i32, conn: bool, rssi: i16, mfr: Vec<(u16, Vec<u8>)> },
        Stopped(i32),
        Status(Instant, i32),
        Params(Instant, [u16; 3]),
        Value(Instant, Role, Vec<u8>),
        Enter,
    }

    /// Minimum, mean and maximum of a series.
    #[derive(Default)]
    struct Stat {
        n: u32,
        sum: f64,
        min: f64,
        max: f64,
    }

    impl Stat {
        fn add(&mut self, v: f64) {
            self.min = if self.n == 0 { v } else { self.min.min(v) };
            (self.max, self.sum, self.n) = (self.max.max(v), self.sum + v, self.n + 1);
        }
    }

    impl std::fmt::Display for Stat {
        fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
            if self.n == 0 { write!(f, "-") } else { write!(f, "{:.1}/{:.1}/{:.1}", self.min, self.sum / self.n as f64, self.max) }
        }
    }

    struct Log {
        start: Instant,
        file: std::fs::File,
    }

    impl Log {
        /// One line, on stdout and in the file, stamped with its event's time.
        fn at(&mut self, at: Instant, text: String) {
            let line = format!("[{:09.3}] {text}", at.saturating_duration_since(self.start).as_secs_f64());
            println!("{line}");
            let _ = writeln!(self.file, "{line}");
        }
    }

    macro_rules! say {
        ($t:expr, $($a:tt)*) => { $t.log.at(Instant::now(), format!($($a)*)) };
    }

    struct Trial {
        log: Log,
        rx: Receiver<Ev>,
        /// Kept to the end of the run, as the working tools do: whether Windows still
        /// finds the device once the watcher is gone is not documented.
        watcher: Option<BluetoothLEAdvertisementWatcher>,
        device: Option<BluetoothLEDevice>,
        step: &'static str,
        /// The last input report and its time; whether one was ever printed.
        last: Option<(Instant, Vec<u8>)>,
        seen: bool,
        /// The measuring window: its start, its reports per second, the gaps between
        /// reports (ms) and the steps of their counter.
        win: Instant,
        per_s: Vec<u32>,
        gaps: Stat,
        steps: Stat,
        /// Minimum and maximum of each stick value: left slot x and y, right slot x and y.
        sticks: [(u16, u16); 4],
        answers: Vec<(Role, Vec<u8>, Instant)>,
        changed: bool,
        enter: bool,
    }

    impl Trial {
        /// Handles the events until `until`, or until `done`; a lost link ends the run.
        fn pump(&mut self, until: Instant, done: impl Fn(&Trial) -> bool) -> R<()> {
            while !done(self) {
                let Ok(ev) = self.rx.recv_timeout(until.saturating_duration_since(Instant::now())) else { break };
                match ev {
                    Ev::Status(at, status) => {
                        self.log.at(at, format!("status {status}"));
                        // A queued 0 of an earlier failed try must not end a run whose link is up.
                        if status == 0 && self.device.as_ref().is_some_and(|d| d.ConnectionStatus().is_ok_and(|s| s.0 == 0)) {
                            return Err(format!("link: dropped during {}\nhint: battery, range, a pairing attempt by Windows (see paired=), or the console waking up and taking the controller back", self.step));
                        }
                    }
                    Ev::Params(at, [interval, latency, timeout]) => {
                        self.changed = true;
                        self.log.at(at, format!("params changed interval={interval} ({:.2} ms) latency={latency} timeout={timeout}", interval as f64 * 1.25));
                    }
                    Ev::Value(at, Role::Input, b) => self.report(at, b),
                    Ev::Value(at, role, b) => {
                        self.log.at(at, format!("rx {} len={} {}", role.name(), b.len(), hex(&b)));
                        self.answers.push((role, b, at));
                    }
                    Ev::Enter => self.enter = true,
                    Ev::Adv { .. } | Ev::Stopped(_) => {}
                }
            }
            Ok(())
        }

        /// One input report: counted in the window, compared with the previous one
        /// (a line with the whole report when the buttons change), sticks tracked.
        fn report(&mut self, at: Instant, b: Vec<u8>) {
            if !self.seen {
                self.seen = true;
                self.log.at(at, format!("rpt first len={} {}", b.len(), hex(&b)));
            }
            let counted = at.checked_duration_since(self.win).and_then(|d| self.per_s.get_mut(d.as_secs() as usize)).map(|n| *n += 1).is_some();
            if b.len() >= 16 {
                if let Some((before, old)) = &self.last && old.len() >= 16 {
                    if counted {
                        self.gaps.add(at.saturating_duration_since(*before).as_secs_f64() * 1000.0);
                        self.steps.add(le32(&b).wrapping_sub(le32(old)) as f64);
                    }
                    if b[4..8] != old[4..8] {
                        self.log.at(at, format!("btn {:08X} -> {:08X} {}", le32(&old[4..]), le32(&b[4..]), hex(&b)));
                    }
                }
                let (left, right) = (unpack(&b[10..13]), unpack(&b[13..16]));
                for (range, v) in self.sticks.iter_mut().zip([left.0, left.1, right.0, right.1]) {
                    *range = (range.0.min(v), range.1.max(v));
                }
            }
            self.last = Some((at, b));
        }

        /// A measuring window of at most `secs` seconds, then its line: the reports of
        /// each whole second. The line comes even when the link is lost inside the window.
        fn window(&mut self, label: &str, secs: usize, done: impl Fn(&Trial) -> bool) -> R<()> {
            (self.win, self.per_s, self.gaps, self.steps) = (Instant::now(), vec![0; secs], Stat::default(), Stat::default());
            let ended = self.pump(self.win + Duration::from_secs(secs as u64), done);
            self.per_s.truncate(self.win.elapsed().as_secs() as usize);
            say!(self, "rate {label} per_s={:?} total={} gap_ms={} ctr_step={}", self.per_s, self.per_s.iter().sum::<u32>(), self.gaps, self.steps);
            ended
        }

        /// The window's last report, as `rpt <label> len=<n> <hex>`.
        fn show(&mut self, label: &str) -> Vec<u8> {
            let b = self.last.as_ref().filter(|l| l.0 >= self.win).map(|l| l.1.clone()).unwrap_or_default();
            say!(self, "rpt {label} len={} {}", b.len(), hex(&b));
            b
        }

        /// Writes one whitelisted frame to the command characteristic and waits a second
        /// for its answer; without one, writes it again to the side's own characteristic,
        /// behind 17 zero bytes. Gives the answer, from its header on.
        fn send(&mut self, command: &GattCharacteristic, side: Option<&GattCharacteristic>, f: &[u8]) -> R<Option<Vec<u8>>> {
            assert!(allowed(f), "refused frame {}", hex(f));
            let head = [f[0], 1, 1, f[3]];
            // An answer: notified after this write, with this command's header; a read's
            // also repeats its address (the three reads have the same header, and an
            // answer may come on both answer characteristics).
            let find = |t: &Trial, since: Instant| {
                t.answers.iter().filter(|a| a.2 >= since).find_map(|(role, b, at)| {
                    let i = b.windows(4).position(|w| w == head)?;
                    (f[0] != 0x02 || b.get(i + 12..i + 16) == f.get(12..16)).then(|| (*role, b[i..].to_vec(), i, *at))
                })
            };
            for (name, c, zeros) in [("plain", Some(command), 0), ("side", side, 17)] {
                let Some(c) = c else { continue };
                let out = [&vec![0; zeros][..], f].concat();
                self.answers.clear();
                let sent = Instant::now();
                let wrote = buffer(&out).and_then(|b| c.WriteValueWithOptionAsync(&b, GattWriteOption::WriteWithoutResponse)).and_then(|op| op.join());
                say!(self, "tx {name} {} write={}", hex(&out), wrote.map_or_else(|e| err(&e), |s| s.0.to_string()));
                self.pump(Instant::now() + Duration::from_secs(1), |t| find(t, sent).is_some())?;
                // Unspaced writes were seen dropped: 100 ms before the next one.
                std::thread::sleep(Duration::from_millis(100));
                if let Some((role, answer, at_byte, at)) = find(self, sent) {
                    self.log.at(at, format!("rsp {:02X}/{:02X} on={} off={at_byte} ack={} ms={}", f[0], f[3], role.name(), hex(answer.get(4..6).unwrap_or_default()), at.saturating_duration_since(sent).as_millis()));
                    return Ok(Some(answer));
                }
                say!(self, "rsp {:02X}/{:02X} NONE", f[0], f[3]);
            }
            Ok(None)
        }

        /// The link's parameters: the interval in 1.25 ms, the latency in connection
        /// events, the timeout in 10 ms; all zero when not connected.
        fn params(&mut self, device: &BluetoothLEDevice, when: &str) {
            match device.GetConnectionParameters().and_then(|p| Ok((p.ConnectionInterval()?, p.ConnectionLatency()?, p.LinkTimeout()?))) {
                Ok((interval, latency, timeout)) => say!(self, "params {when} interval={interval} ({:.2} ms) latency={latency} timeout={timeout} ({} ms)", interval as f64 * 1.25, timeout as u32 * 10),
                Err(e) => say!(self, "params {when} {}", err(&e)),
            }
        }
    }

    /// What a Nintendo advertisement says: the product, the kind, the host it names.
    fn describe(d: &[u8]) -> (Option<u16>, &'static str, String) {
        let pid = (d.len() >= 7 && d[..5] == CONTROLLER).then(|| u16::from_le_bytes([d[5], d[6]]));
        let host = d.get(10..16).map_or("none".into(), |h| h.iter().rev().map(|b| format!("{b:02X}")).collect());
        let kind = match d {
            [3, ..] => "console",
            _ if d.len() < 16 => "other",
            _ if d[9] == 0x81 => "wake",
            _ if d[10..16] == [0; 6] => "standard",
            _ => "reconnection",
        };
        (pid, kind, host)
    }

    /// Scans for at most 30 s; the address and the kind of the first connectable
    /// advertisement of the wanted product in sync mode or, without one, of one that
    /// calls its console, when a product is wanted.
    fn scan(t: &mut Trial, tx: &Sender<Ev>, wanted: Option<(char, u16)>) -> R<Option<(u64, &'static str)>> {
        t.step = "scan";
        let watcher = w("scan", BluetoothLEAdvertisementWatcher::new())?;
        w("scan", watcher.SetScanningMode(BluetoothLEScanningMode::Passive))?;
        // Handlers first: the watcher does not start without one.
        let s = tx.clone();
        let received = w("scan", watcher.Received(&TypedEventHandler::<BluetoothLEAdvertisementWatcher, BluetoothLEAdvertisementReceivedEventArgs>::new(move |_, a| {
            let a = a.ok()?;
            let list = a.Advertisement()?.ManufacturerData()?;
            let mut mfr = Vec::new();
            for i in 0..list.Size()? {
                let m = list.GetAt(i)?;
                mfr.push((m.CompanyId()?, bytes(&m.Data()?)?));
            }
            let _ = s.send(Ev::Adv { at: Instant::now(), addr: a.BluetoothAddress()?, atype: a.BluetoothAddressType().map_or(-1, |x| x.0), kind: a.AdvertisementType()?.0, conn: a.IsConnectable().unwrap_or(false), rssi: a.RawSignalStrengthInDBm().unwrap_or(0), mfr });
            Ok(())
        })))?;
        let s = tx.clone();
        let stopped = w("scan", watcher.Stopped(&TypedEventHandler::<BluetoothLEAdvertisementWatcher, BluetoothLEAdvertisementWatcherStoppedEventArgs>::new(move |_, a| {
            let _ = s.send(Ev::Stopped(a.ok()?.Error()?.0));
            Ok(())
        })))?;
        const RADIO: &str = "hint: turn Bluetooth on (Settings > Bluetooth & devices) and check the adapter in Device Manager; error 5, 7 or 8 is a policy or privacy setting";
        watcher.Start().map_err(|e| format!("scan: Start {}\n{RADIO}", err(&e)))?;
        say!(t, "scan start passive 30 s");
        if wanted.is_some() {
            say!(t, "do: press and HOLD the Joy-Con's sync button (the small round one between SL and SR) until \"connected\"");
        }
        let (begin, mut held, mut error, mut target, mut calling) = (Instant::now(), wanted.is_none(), 0, None, None);
        let (mut total, mut nintendo) = (0, 0);
        let mut seen = BTreeMap::new();
        while target.is_none() && error == 0 && begin.elapsed() < SCAN {
            if !held && begin.elapsed() >= HOLD {
                held = true;
                say!(t, "do: nothing seen yet: keep holding the sync button; is the Joy-Con charged and detached?");
            }
            match t.rx.recv_timeout(if held { SCAN } else { HOLD }.saturating_sub(begin.elapsed())) {
                Ok(Ev::Adv { at, addr, atype, kind, conn, rssi, mfr }) => {
                    total += 1;
                    for (_, d) in mfr.into_iter().filter(|m| m.0 == NINTENDO) {
                        nintendo += 1;
                        let (pid, what, host) = describe(&d);
                        let pid_text = pid.map_or("-".into(), |p| format!("{p:04X}"));
                        let entry = seen.entry((addr, d.clone())).or_insert((0u32, at, at, pid_text.clone(), what));
                        (entry.0, entry.2) = (entry.0 + 1, at);
                        // One line per power of two: a controller advertises many times a second.
                        if entry.0.is_power_of_two() {
                            t.log.at(at, format!("adv addr={addr:012X} atype={atype} type={kind} conn={conn} rssi={rssi} n={} len={} data={} pid={pid_text} kind={what} host={host}", entry.0, d.len(), hex(&d)));
                        }
                        // Type 0: connectable, undirected.
                        if (kind == 0 || conn) && pid.is_some() && pid == wanted.map(|w| w.1) {
                            if what == "standard" {
                                target = Some((addr, what));
                            } else if calling.is_none() {
                                // After a plain press the Joy-Con calls its console and drops any
                                // other host within a third of a second (seen on real hardware):
                                // only its sync mode lets us in.
                                calling = Some((addr, what));
                                held = true;
                                say!(t, "do: the Joy-Con is calling its console: hold its sync button until \"connected\"");
                            }
                        }
                    }
                }
                Ok(Ev::Stopped(e)) => {
                    error = e;
                    say!(t, "scan stopped err={e}");
                }
                _ => {}
            }
        }
        let status = watcher.Status().map_or(-1, |s| s.0);
        let _ = (watcher.Stop(), watcher.RemoveReceived(received), watcher.RemoveStopped(stopped));
        t.watcher = Some(watcher);
        // Without a sync advertisement in 30 s, the calling one is tried all the same.
        let target = target.or(calling);
        say!(t, "scan end adv_total={total} nintendo={nintendo} watcher={status} target={} kind={}", target.map_or("none".into(), |x| format!("{:012X}", x.0)), target.map_or("-", |x| x.1));
        for ((addr, _), (n, first, last, pid, what)) in &seen {
            say!(t, "advsum addr={addr:012X} pid={pid} kind={what} n={n} first={:.3} last={:.3}", first.saturating_duration_since(t.log.start).as_secs_f64(), last.saturating_duration_since(t.log.start).as_secs_f64());
        }
        match (target, wanted) {
            (Some(found), _) => Ok(Some(found)),
            _ if error != 0 => Err(format!("scan: watcher stopped, BluetoothError {error}\n{RADIO}")),
            (None, None) => Ok(None),
            _ if total == 0 => Err("scan: no advertisement of any kind\nhint: the radio or its driver delivers nothing".into()),
            _ if nintendo == 0 => Err("scan: no Nintendo advertisement\nhint: the Joy-Con is not advertising: hold the sync button, charge it, or wait a few minutes if several attempts were just made".into()),
            (None, Some((side, _))) => Err(format!("scan: no {side} Joy-Con among {nintendo} Nintendo advertisements\nhint: see the advsum lines: another product, kind=console (the Switch 2 is on: turn it off) or kind=wake (Home was pressed)")),
        }
    }

    fn run(t: &mut Trial, tx: Sender<Ev>, wanted: Option<(char, u16)>, mask: u8) -> R<()> {
        match BluetoothAdapter::GetDefaultAsync().and_then(|op| op.join()) {
            Ok(a) => say!(t, "adapter addr={:012X} le={} central={}", a.BluetoothAddress().unwrap_or(0), a.IsLowEnergySupported().unwrap_or(false), a.IsCentralRoleSupported().unwrap_or(false)),
            // A null object comes back as an error of code 0.
            Err(e) if e.code() == HRESULT(0) => say!(t, "adapter NONE"),
            Err(e) => say!(t, "adapter {}", err(&e)),
        }
        let Some((addr, _)) = scan(t, &tx, wanted)? else { return Ok(()) };
        let side = wanted.map_or('R', |w| w.0);

        // The device object: no radio traffic yet. The one-argument form is the one
        // known to work with these controllers.
        t.step = "open";
        let device = match BluetoothLEDevice::FromBluetoothAddressAsync(addr).and_then(|op| op.join()) {
            Ok(d) => d,
            Err(e) if e.code() == HRESULT(0) => return Err("open: null\nhint: run again: the scan must have just seen the controller".into()),
            Err(e) => return Err(format!("open: {}\nhint: on access denied, check the Bluetooth privacy settings and close any other program using the controller", err(&e))),
        };
        let paired = device.DeviceInformation().and_then(|i| i.Pairing()).and_then(|p| p.IsPaired());
        say!(t, "device name=\"{}\" id={} atype={} status={} paired={}", device.Name().unwrap_or_default(), device.DeviceId().unwrap_or_default(), device.BluetoothAddressType().map_or(-1, |a| a.0), device.ConnectionStatus().map_or(-1, |s| s.0), paired.as_ref().map_or("?".into(), |p| p.to_string()));
        if matches!(paired, Ok(true)) {
            // The standard pairing makes the controller drop the link: stop before any attempt.
            return Err("paired: Windows lists this controller as a paired device\nhint: remove it in Settings > Bluetooth & devices > Devices, and never use \"Add device\" for it".into());
        }
        let s = tx.clone();
        let status_token = w("open", device.ConnectionStatusChanged(&TypedEventHandler::<BluetoothLEDevice, IInspectable>::new(move |d, _| {
            let _ = s.send(Ev::Status(Instant::now(), d.ok()?.ConnectionStatus()?.0));
            Ok(())
        })))?;
        t.device = Some(device.clone());

        // The uncached service discovery is what connects; Windows bounds each try (about 7 s).
        t.step = "gatt";
        let mut services = None;
        for attempt in 1..=3 {
            let begin = Instant::now();
            let found = w("gatt", device.GetGattServicesWithCacheModeAsync(BluetoothCacheMode::Uncached).and_then(|op| op.join()))?;
            let status = found.Status().map_or(-1, |s| s.0);
            let list = found.Services().ok().filter(|_| status == 0);
            let count = list.as_ref().and_then(|l| l.Size().ok()).unwrap_or(0);
            say!(t, "gatt try={attempt} status={status} perr={} ms={} services={count}", perr(found.ProtocolError()), begin.elapsed().as_millis());
            if status == 0 {
                if count == 0 {
                    return Err("gatt: no service\nhint: a stale Windows cache or a link that fell at once: remove the device in Settings, switch Bluetooth off and on, run again".into());
                }
                services = list;
                break;
            }
            if attempt < 3 {
                say!(t, "do: keep holding (or press and hold now) the sync button");
                std::thread::sleep(Duration::from_millis(500));
            }
        }
        let Some(services) = services else {
            return Err("gatt: unreachable after 3 tries\nhint: the controller stopped advertising (hold the sync button until \"connected\"), is out of range, is cooling down (wait a few minutes), or answers only its console (see kind= in \"scan end\"); status 3 = another program holds it".into());
        };
        say!(t, "do: connected. Put the Joy-Con down and do not touch it until told (about 30 s)");
        t.params(&device, "connect");
        // What the link did during the tries; a link already lost ends the run here.
        t.pump(Instant::now(), |_| false)?;

        t.step = "chr";
        let mtu = |s: &windows::Devices::Bluetooth::GenericAttributeProfile::GattDeviceService| s.Session().and_then(|x| x.MaxPduSize()).map_or("?".into(), |m| m.to_string());
        let mut found: BTreeMap<&str, GattCharacteristic> = BTreeMap::new();
        let mut service = None;
        for s in &services {
            let uuid = s.Uuid().unwrap_or_default();
            say!(t, "svc uuid={uuid:?} handle={:04X} mtu={}", s.AttributeHandle().unwrap_or(0), mtu(&s));
            // Without a cache argument, as the working tools do.
            let list = match s.GetCharacteristicsAsync().and_then(|op| op.join()) {
                Ok(r) if r.Status().is_ok_and(|x| x.0 == 0) => r.Characteristics(),
                Ok(r) => {
                    say!(t, "chrs uuid={uuid:?} status={}", r.Status().map_or(-1, |x| x.0));
                    continue;
                }
                Err(e) => {
                    say!(t, "chrs uuid={uuid:?} {}", err(&e));
                    continue;
                }
            };
            let Ok(list) = list else {
                say!(t, "chrs uuid={uuid:?} no list");
                continue;
            };
            for c in &list {
                let id = c.Uuid().unwrap_or_default();
                let role = ROLES.iter().find(|r| GUID::from_u128(r.0) == id).map_or("-", |r| r.1);
                say!(t, "chr uuid={id:?} handle={:04X} props={:03X} role={role}", c.AttributeHandle().unwrap_or(0), c.CharacteristicProperties().map_or(0, |p| p.0));
                if role == "input" {
                    service = Some(s.clone());
                }
                found.insert(role, c);
            }
        }
        t.pump(Instant::now(), |_| false)?;
        let (Some(input), Some(command)) = (found.get("input").cloned(), found.get("command").cloned()) else {
            return Err("chr: input or command characteristic not found\nhint: send back the chr lines above: they show what the device exposes".into());
        };
        let side_command = found.get("cmd-side");

        // Answers first, then the reports. Each time the handler, then the descriptor.
        t.step = "sub";
        let mut tokens = Vec::new();
        for (role, c) in [(Role::RspA, found.get("rsp-a")), (Role::RspSide, found.get("rsp-side")), (Role::Input, Some(&input))] {
            let Some(c) = c else { continue };
            let s = tx.clone();
            let token = w("sub", c.ValueChanged(&TypedEventHandler::<GattCharacteristic, GattValueChangedEventArgs>::new(move |_, a| {
                let _ = s.send(Ev::Value(Instant::now(), role, bytes(&a.ok()?.CharacteristicValue()?)?));
                Ok(())
            })))?;
            tokens.push((c, token));
            let wrote = c.WriteClientCharacteristicConfigurationDescriptorWithResultAsync(GattClientCharacteristicConfigurationDescriptorValue::Notify).and_then(|op| op.join());
            let status = wrote.as_ref().map_or(-1, |r| r.Status().map_or(-1, |s| s.0));
            say!(t, "sub {} status={status} perr={}", role.name(), wrote.map_or_else(|e| err(&e), |r| perr(r.ProtocolError())));
            if role == Role::Input && status != 0 {
                return Err("sub: input\nhint: perr 05 or 0F = the controller wants an encrypted link; 03 = write not permitted".into());
            }
        }

        // Do reports flow before any feature command?
        t.step = "default";
        t.window("default", 1, |_| false)?;

        // Buttons and sticks alone (03) is what a driver wants and what no known tool
        // sends: the console's mask (37) when it gives nothing.
        t.step = "features";
        let (mut mask, mut bad) = (mask, None);
        loop {
            t.send(&command, side_command, &frame(0x0c, 0x02, &[mask, 0, 0, 0]))?;
            t.send(&command, side_command, &frame(0x0c, 0x04, &[mask, 0, 0, 0]))?;
            t.window(&format!("mask{mask:02X}"), 2, |_| false)?;
            let last = t.show(&format!("mask{mask:02X}"));
            let what = if t.per_s.iter().sum::<u32>() == 0 { "no reports" } else { "zero sticks" };
            if !last.is_empty() && last.get(10..16).is_some_and(|sticks| sticks != [0; 6]) {
                break;
            }
            if mask == 0x37 {
                // The verdict of the run; the interval and the memory are still worth reading.
                say!(t, "reports: {what} after mask 37, going on");
                bad = Some(what);
                break;
            }
            say!(t, "fallback: mask 03 gives {what}, sending mask 37");
            mask = 0x37;
        }
        // Light 1 steady: the proof the owner can see.
        t.send(&command, side_command, &frame(0x09, 0x07, &[1, 0, 0, 0, 0, 0, 0, 0]))?;

        // The gate of the whole step: Windows' interval, and the rate it gives. Nothing
        // here ends the run: both windows always happen.
        t.step = "interval";
        t.params(&device, "before");
        t.window("before", 10, |_| false)?;
        t.params(&device, "after-w1");
        let s = tx.clone();
        let changed = device.ConnectionParametersChanged(&TypedEventHandler::<BluetoothLEDevice, IInspectable>::new(move |d, _| {
            let p = d.ok()?.GetConnectionParameters()?;
            let _ = s.send(Ev::Params(Instant::now(), [p.ConnectionInterval()?, p.ConnectionLatency()?, p.LinkTimeout()?]));
            Ok(())
        }));
        // The request lives to the end of the run: what dropping it does is not documented.
        let request = BluetoothLEPreferredConnectionParameters::ThroughputOptimized().and_then(|preset| {
            say!(t, "preset throughput min={} max={} latency={} timeout={}", preset.MinConnectionInterval()?, preset.MaxConnectionInterval()?, preset.ConnectionLatency()?, preset.LinkTimeout()?);
            device.RequestPreferredConnectionParameters(&preset)
        });
        // 1 = accepted, not yet applied; 2 = device not available; 3 = access denied.
        say!(t, "request status={}", request.as_ref().map_or_else(err, |r| r.Status().map_or(-1, |s| s.0).to_string()));
        t.changed = false;
        t.pump(Instant::now() + Duration::from_secs(5), |t| t.changed)?;
        t.params(&device, "after-request");
        t.window("after", 10, |_| false)?;
        t.show("rest");

        t.step = "memory";
        if let Some(answer) = t.send(&command, side_command, &frame(0x10, 0x01, &[]))? && let Some(d) = answer.get(8..15) {
            say!(t, "fw {} v={}.{}.{} type={} bt={}.{}.{}", hex(&answer[8..]), d[0], d[1], d[2], d[3], d[4], d[5], d[6]);
        }
        for addr in READS {
            // The answer: 8 bytes of header, the length and the address, then the block.
            let Some(answer) = t.send(&command, side_command, &read(addr))? else { continue };
            let block = answer.get(16..).unwrap_or_default();
            say!(t, "mem {addr:06X} len={} {}", block.len(), hex(block));
            if addr != READS[2] && let Some(raw) = block.get(0x28..0x31) {
                say!(t, "cal {addr:06X} raw={} centre={:?} p2={:?} p3={:?}", hex(raw), unpack(&raw[..3]), unpack(&raw[3..6]), unpack(&raw[6..]));
            }
        }

        // The owner's part: each line of a button change carries the whole report. It is
        // a measuring window too: with two controllers, both links are up by then.
        if bad != Some("no reports") {
            t.step = "tour";
            let (order, tap) = if side == 'R' { ("A, B, X, Y, R, ZR, Plus, stick click, Home, C, SL, SR", "ZR") } else { ("Up, Down, Left, Right, L, ZL, Minus, stick click, Capture, SL, SR", "ZL") };
            say!(t, "do: hold the Joy-Con upright. Press and release, one per second, in this order:");
            say!(t, "do: {order}");
            say!(t, "do: push the stick fully UP, hold it, tap {tap} once; fully RIGHT, hold, tap {tap} once");
            say!(t, "do: roll the stick twice around its full circle, let go, then press Enter here");
            // An Enter typed before this point is not the tour's end.
            t.pump(Instant::now(), |_| false)?;
            (t.enter, t.sticks) = (false, [(u16::MAX, 0); 4]);
            let ended = t.window("tour", 120, |t| t.enter);
            let [lx, ly, rx, ry] = t.sticks;
            say!(t, "sticks slotL x={}..{} y={}..{} slotR x={}..{} y={}..{}", lx.0, lx.1, ly.0, ly.1, rx.0, rx.1, ry.0, ry.1);
            ended?;
        }
        // The rate under the console's own mask, the one the known tools measured with.
        if mask == 0x03 {
            t.step = "mask37";
            t.send(&command, side_command, &frame(0x0c, 0x02, &[0x37, 0, 0, 0]))?;
            t.send(&command, side_command, &frame(0x0c, 0x04, &[0x37, 0, 0, 0]))?;
            t.window("mask37-end", 5, |_| false)?;
            t.show("mask37-end");
        }

        t.step = "end";
        t.params(&device, "end");
        say!(t, "mtu end={}", service.as_ref().map_or("?".into(), mtu));
        for (c, token) in tokens {
            let _ = c.RemoveValueChanged(token);
        }
        let _ = (changed.and_then(|token| device.RemoveConnectionParametersChanged(token)), device.RemoveConnectionStatusChanged(status_token), request.and_then(|r| r.Close()));
        for s in &services {
            let _ = s.Close();
        }
        let _ = device.Close();
        match bad {
            Some(what) => Err(format!("reports: {what} after mask 37\nhint: next to try, in a new version: each side's own report, then the descriptor write of the captured hosts")),
            None => Ok(()),
        }
    }

    fn usage() -> ! {
        eprintln!("usage: joycon_trial <L|R|scan> <log file> [03|37]");
        std::process::exit(2)
    }

    pub fn main() {
        let args: Vec<String> = std::env::args().collect();
        let wanted = match args.get(1).map(String::as_str) {
            Some("R") => Some(('R', 0x2066)),
            Some("L") => Some(('L', 0x2067)),
            Some("scan") => None,
            _ => usage(),
        };
        let Some(path) = args.get(2) else { usage() };
        let mask = match args.get(3).map(String::as_str) {
            None | Some("03") => 0x03,
            Some("37") => 0x37,
            _ => usage(),
        };
        self_check();
        // Appended to: a second run must not erase what a failed one found.
        let file = std::fs::OpenOptions::new().create(true).append(true).open(path).unwrap_or_else(|e| {
            eprintln!("J0 FAIL log: {path}: {e}");
            std::process::exit(1)
        });
        let (tx, rx) = mpsc::channel();
        // Enter ends the owner's part.
        let enter = tx.clone();
        std::thread::spawn(move || for _ in std::io::stdin().lines() {
            if enter.send(Ev::Enter).is_err() {
                break;
            }
        });
        let now = Instant::now();
        let mut t = Trial { log: Log { start: now, file }, rx, watcher: None, device: None, step: "start", last: None, seen: false, win: now, per_s: Vec::new(), gaps: Stat::default(), steps: Stat::default(), sticks: [(u16::MAX, 0); 4], answers: Vec::new(), changed: false, enter: false };
        // The wall clock, to line up the logs of two controllers run together.
        let unix_ms = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map_or(0, |d| d.as_millis());
        say!(t, "J0 start side={} pid={} mask={mask:02X} unix_ms={unix_ms}", wanted.map_or("scan".into(), |w| w.0.to_string()), wanted.map_or("-".into(), |w| format!("{:04X}", w.1)));
        let code = match run(&mut t, tx, wanted, mask) {
            Ok(()) => {
                say!(t, "J0 end ok");
                0
            }
            Err(e) => {
                // What was still queued says what the link did before the failure.
                while t.pump(Instant::now(), |_| false).is_err() {}
                for (n, line) in e.lines().enumerate() {
                    say!(t, "{}{line}", if n == 0 { "J0 FAIL " } else { "" });
                }
                1
            }
        };
        let _ = t.log.file.flush();
        std::process::exit(code)
    }
}
