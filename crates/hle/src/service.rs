//! The services, as plain data (D21): one enum, so that their state can be
//! saved with the rest of the guest state.
//!
//! Policy for a command a service does not know: the core services stop the
//! run with the command and its arguments, so that each stop says what to
//! write next; exploration mode answers success with empty outputs instead.

use crate::Kernel;
use crate::ipc::{Reply, Request};

/// sm's result NotAllowed (module 21, description 8): a service the NPDM
/// does not grant.
pub const SM_NOT_ALLOWED: u32 = 21 | 8 << 9;

/// Services granted although the NPDM's list does not name them. `bsd:s`:
/// the SDK's socket layer asks for it (Lichii sees it too) and aborts when
/// it is refused, so the console must grant it; its rule is not known.
const ALSO_GRANTED: [&str; 1] = ["bsd:s"];

#[derive(Debug, Default)]
pub enum Service {
    /// The place of a service that is answering a request, or closed.
    #[default]
    Closed,
    /// `sm:` (TIPC): RegisterClient, GetServiceHandle.
    Sm(Sm),
    /// `lm`: ILogService.
    Lm,
    /// lm's ILogger: the game's log lines go to the journal.
    Logger,
    /// Exploration mode only: a service with no implementation yet.
    Stub(String),
    /// `appletOE` and the interfaces it hands out.
    Applet(crate::am::Applet),
    /// `time:u` and its clocks.
    Time(crate::time::Time),
    /// `pctl` (IParentalControlServiceFactory) and IParentalControlService.
    Pctl,
    ParentalControl,
    /// `fsp-srv` and its objects.
    Fs(crate::fs::Fs),
    /// `acc:u0` and its profiles.
    Account(crate::account::Account),
    /// `nifm:u`, never connected.
    Nifm(crate::account::Nifm),
    /// `ldr:ro`.
    Ldr,
    /// `hid` and its objects.
    Hid(crate::hid::Hid),
    /// Save file systems and their files.
    Save(crate::save::Save),
    /// `aud:u`, `hwopus`, `auddmg:u`, `audsmx:u` and their objects.
    Audio(crate::audio::Audio),
    /// The graphics supervisor and its sessions.
    Gs(crate::gs::Gs),
    /// The nvdrv-like session of the graphics supervisor.
    Nvdrv(crate::nvdrv::Nvdrv),
    /// `bsd:s` and `bsd:u`, offline.
    Bsd(crate::bsd::Bsd),
    /// `csrng`: random bytes from the run's seed.
    Csrng { state: u64 },
    /// aoc's IPurchaseEventManager.
    PurchaseEvents,
    /// `prepo:u`: play reports, dropped.
    Prepo,
    /// `ssl`: only SetInterfaceVersion; no context, so no connection (offline).
    Ssl,
    /// `chat:u` and the session its command 0 opens (Lichii's empty answers).
    Chat { session: bool },
    /// `aoc:u` and `ectx:aw`: no command seen yet.
    Aoc,
    Ectx,
}

#[derive(Debug, Default)]
pub struct Sm {
    registered: bool,
}

impl Service {
    /// How the journal names it.
    pub fn label(&self) -> String {
        match self {
            Service::Closed => "closed".to_string(),
            Service::Sm(_) => "sm:".to_string(),
            Service::Lm => "lm".to_string(),
            Service::Logger => "lm ILogger".to_string(),
            Service::Stub(name) => format!("{name} (stub)"),
            Service::Applet(a) => a.label(),
            Service::Time(t) => t.label(),
            Service::Fs(f) => f.label(),
            Service::Bsd(b) => b.name.clone(),
            Service::Gs(g) => g.label(),
            Service::Hid(h) => h.label(),
            Service::Audio(a) => a.label(),
            Service::Save(s) => s.label(),
            Service::Nvdrv(n) => n.label(),
            Service::Account(a) => a.label(),
            Service::Nifm(n) => n.label(),
            Service::Ldr => "ldr:ro".to_string(),
            Service::Pctl => "pctl".to_string(),
            Service::ParentalControl => "pctl IParentalControlService".to_string(),
            Service::Aoc => "aoc:u".to_string(),
            Service::Csrng { .. } => "csrng".to_string(),
            Service::PurchaseEvents => "aoc IPurchaseEventManager".to_string(),
            Service::Prepo => "prepo:u".to_string(),
            Service::Ssl => "ssl".to_string(),
            Service::Chat { session: false } => "chat:u".to_string(),
            Service::Chat { session: true } => "chat:u session".to_string(),
            Service::Ectx => "ectx:aw".to_string(),
        }
    }

    pub fn request(&mut self, k: &mut Kernel, req: &Request) -> Reply {
        let answer = match self {
            Service::Applet(a) => a.request(k, req),
            Service::Time(t) => t.request(k, req),
            Service::Fs(f) => f.request(k, req),
            Service::Bsd(b) => b.request(k, req),
            Service::Gs(g) => g.request(k, req),
            Service::Hid(h) => h.request(k, req),
            Service::Audio(a) => a.request(k, req),
            Service::Save(s) => s.request(k, req),
            Service::Nvdrv(n) => n.request(k, req),
            Service::Account(a) => a.request(k, req),
            Service::Nifm(n) => n.request(k, req),
            _ => None,
        };
        if let Some(reply) = answer {
            return reply;
        }
        match (self, req.command) {
            // CreateService, CreateServiceWithoutInitialize; Initialize.
            (Service::Pctl, 0 | 1) => Reply::object(Service::ParentalControl),
            (Service::ParentalControl, 1) => Reply::ok(),
            // RegisterProcessHandle.
            (Service::Ldr, 4) => Reply::ok(),
            // csrng GenerateRandomBytes: a stream fixed by the run's seed.
            (Service::Csrng { state }, 0) => {
                if let Some(b) = req.out_buffer(0) {
                    let bytes: Vec<u8> = (0..b.size.min(0x10000).div_ceil(8)).flat_map(|_| {
                        *state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
                        let mut z = *state;
                        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
                        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
                        (z ^ (z >> 31)).to_le_bytes()
                    }).collect();
                    let _ = k.mem.write(b.addr, &bytes[..b.size as usize]);
                }
                Reply::ok()
            }
            // aoc:u: CreateEcPurchasedEventManager, CheckAddOnContentMountStatus,
            // CountAddOnContent (none); the manager: SetDefaultDeliveryTarget, the event.
            (Service::Aoc, 100) => Reply::object(Service::PurchaseEvents),
            (Service::Aoc, 50) => Reply::ok(),
            (Service::Aoc, 2) => Reply::with(&0u32.to_le_bytes()),
            (Service::PurchaseEvents, 0) => Reply::ok(),
            (Service::PurchaseEvents, 2) => match k.lone_event() {
                Some(h) => Reply { copy_handles: vec![h], ..Reply::ok() },
                None => Reply::error(crate::result::OUT_OF_HANDLES),
            },
            // prepo: reports are accepted and dropped (offline).
            (Service::Prepo, _) => Reply::ok(),
            (Service::Ssl, 5) => Reply::ok(),
            // chat:u: open a session, then 3 and 11 (11 at every frame) succeed empty.
            (Service::Chat { session: false }, 0) => Reply::object(Service::Chat { session: true }),
            (Service::Chat { session: false }, 3 | 11) => Reply::ok(),
            // The session's event: never signaled.
            (Service::Chat { session: true }, 1) => match k.lone_event() {
                Some(h) => Reply { copy_handles: vec![h], ..Reply::ok() },
                None => Reply::error(crate::result::OUT_OF_HANDLES),
            },
            (Service::Sm(sm), 0) => {
                sm.registered = true;
                Reply::ok()
            }
            (Service::Sm(_), 1) => k.get_service(req.u64_at(0).unwrap_or(0)),
            // OpenLogger.
            (Service::Lm, 0) => Reply::object(Service::Logger),
            // Log.
            (Service::Logger, 0) => {
                if let Some(b) = req.send_buffer(0) {
                    k.game_log(b.addr, b.size);
                }
                Reply::ok()
            }
            // SetDestination.
            (Service::Logger, 1) => Reply::ok(),
            (Service::Stub(name), c) => {
                let key = format!("stub {name} {c}");
                if k.journal.first(&key) {
                    k.log(&format!("ipc {name} {c} answered by a stub ({} bytes of arguments)", req.data.len()));
                }
                Reply::ok()
            }
            (svc, c) => {
                let args: String = req.data.iter().take(64).map(|b| format!("{b:02x}")).collect();
                let what = format!("ipc {} command {c} is not implemented; arguments {args}", svc.label());
                if !k.config.explore {
                    k.stop(crate::STOP_EXIT, &what);
                }
                if k.journal.first(&format!("unknown {} {c}", svc.label())) {
                    k.log(&what);
                }
                Reply::error(crate::result::NOT_IMPLEMENTED)
            }
        }
    }
}

/// A service name as sm packs it: up to 8 bytes, zero-padded.
pub fn name_of(packed: u64) -> String {
    let b = packed.to_le_bytes();
    let len = b.iter().position(|&c| c == 0).unwrap_or(8);
    String::from_utf8_lossy(&b[..len]).into_owned()
}

impl Kernel {
    /// sm GetServiceHandle: a service the NPDM grants and the HLE implements.
    fn get_service(&mut self, packed: u64) -> Reply {
        let name = name_of(packed);
        if !self.allowed_services.contains(&name) && !ALSO_GRANTED.contains(&name.as_str()) {
            self.log(&format!("sm: {name:?} is not in the NPDM's service list: NotAllowed"));
            return Reply::error(SM_NOT_ALLOWED);
        }
        let svc = match name.as_str() {
            "lm" => Service::Lm,
            "appletOE" => Service::Applet(crate::am::Applet::Service),
            "time:u" => Service::Time(crate::time::Time::Static),
            "pctl" => Service::Pctl,
            "fsp-srv" => Service::Fs(crate::fs::Fs::Proxy),
            "acc:u0" => Service::Account(crate::account::Account::Service),
            "nifm:u" => Service::Nifm(crate::account::Nifm::Service),
            "ldr:ro" => Service::Ldr,
            "hid" => Service::Hid(crate::hid::Hid::Service { styles: 0, ids: Vec::new() }),
            "aud:u" => Service::Audio(crate::audio::Audio::AudU),
            "hwopus" => Service::Audio(crate::audio::Audio::Opus),
            "auddmg:u" => Service::Audio(crate::audio::Audio::DevMgr),
            "audsmx:u" => Service::Audio(crate::audio::Audio::Mixer),
            "bsd:s" | "bsd:u" => Service::Bsd(crate::bsd::Bsd::new(name.clone())),
            "aoc:u" => Service::Aoc,
            "csrng" => Service::Csrng { state: self.config.seed ^ 0x6373_726E },
            "prepo:u" => Service::Prepo,
            "ssl" => Service::Ssl,
            "chat:u" => Service::Chat { session: false },
            "ectx:aw" => Service::Ectx,
            _ if self.config.explore => Service::Stub(name.clone()),
            _ => self.stop(crate::STOP_EXIT, &format!("sm: service {name:?} is not implemented")),
        };
        self.log(&format!("sm: GetServiceHandle {name:?}"));
        Reply::object(svc)
    }

    /// One packet for lm's ILogger: its text chunks become `game:` lines.
    fn game_log(&mut self, addr: u64, size: u64) {
        let mut packet = vec![0; size.min(0x10000) as usize];
        if self.mem.read(addr, &mut packet).is_err() {
            self.log("game: (unreadable log packet)");
            return;
        }
        for text in log_texts(&packet) {
            for line in text.lines() {
                self.log(&format!("game: {line}"));
            }
        }
    }
}

/// The text chunks (key 2) of an lm log packet: a 0x18-byte header, then
/// chunks of a key byte, a ULEB128 size and the data.
pub fn log_texts(packet: &[u8]) -> Vec<String> {
    let mut out = Vec::new();
    let mut at = 0x18;
    while at < packet.len() {
        let key = packet[at];
        at += 1;
        let (mut size, mut shift) = (0usize, 0);
        while let Some(&b) = packet.get(at) {
            at += 1;
            size |= usize::from(b & 0x7F) << shift;
            shift += 7;
            if b & 0x80 == 0 || shift > 28 {
                break;
            }
        }
        let Some(data) = packet.get(at..at + size) else { break };
        if key == 2 {
            out.push(String::from_utf8_lossy(data).trim_end_matches(['\0', '\n']).to_string());
        }
        at += size;
    }
    out
}
