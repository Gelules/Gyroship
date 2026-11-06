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
        }
    }

    pub fn request(&mut self, k: &mut Kernel, req: &Request) -> Reply {
        match (self, req.command) {
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
                k.log(&what);
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
        if !self.allowed_services.contains(&name) {
            self.log(&format!("sm: {name:?} is not in the NPDM's service list: NotAllowed"));
            return Reply::error(SM_NOT_ALLOWED);
        }
        let svc = match name.as_str() {
            "lm" => Service::Lm,
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
