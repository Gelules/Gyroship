//! IPC (D21): one entry for SendSyncRequest (0x21, message in the TLS) and
//! SendSyncRequestWithUserBuffer (0x22). The HIPC envelope (header, special
//! header with PID and handles, buffer descriptors) carries either TIPC (sm
//! and the Switch 2 graphics stack) or CMIF (`SFCI`/`SFCO`, with domains).
//! Layouts follow the public documentation (switchbrew).

use crate::Kernel;
use crate::result::*;
use crate::service::Service;
use crate::thread::Object;

const SFCI: u32 = u32::from_le_bytes(*b"SFCI");
const SFCO: u32 = u32::from_le_bytes(*b"SFCO");
/// TIPC command ids are offset by this in the message type.
const TIPC: u16 = 16;
/// TIPC 0xF: close the session.
const TIPC_CLOSE: u16 = 15;

/// A buffer descriptor: guest address and size.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Buffer {
    pub addr: u64,
    pub size: u64,
}

/// What kind of request a message is.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Kind {
    /// CMIF request (types 4 and 6, the latter with an inline context).
    Request,
    /// IHipcManager (types 5 and 7).
    Control,
    /// Session close (HIPC type 2, TIPC type 15).
    Close,
    /// TIPC command (type 16 + id).
    Tipc(u32),
}

/// A parsed request.
#[derive(Clone, Debug)]
pub struct Request {
    pub kind: Kind,
    pub command: u32,
    pub pid: Option<u64>,
    pub copy_handles: Vec<u32>,
    pub move_handles: Vec<u32>,
    /// Pointer (X), send (A), receive (B) and exchange (W) buffers.
    pub x: Vec<Buffer>,
    pub a: Vec<Buffer>,
    pub b: Vec<Buffer>,
    pub w: Vec<Buffer>,
    /// Receive lists (C): where pointer outputs go.
    pub c: Vec<Buffer>,
    /// The command's arguments (after the CMIF header, or the TIPC header).
    pub data: Vec<u8>,
    /// Domain: the target object, and the objects passed in.
    pub object: Option<u32>,
    pub in_objects: Vec<u32>,
    /// Domain command 2: close the object.
    pub close_object: bool,
}

impl Request {
    /// The send buffer of an "auto-select" argument: the pointer or the A buffer, whichever is used.
    /// The output buffer of an argument: the B buffer or the receive list, whichever is used.
    pub fn out_buffer(&self, i: usize) -> Option<Buffer> {
        [self.b.get(i), self.c.get(i)].into_iter().flatten().copied().find(|b| b.size != 0)
    }

    pub fn send_buffer(&self, i: usize) -> Option<Buffer> {
        [self.x.get(i), self.a.get(i)].into_iter().flatten().copied().find(|b| b.size != 0)
    }

    pub fn u32_at(&self, off: usize) -> Option<u32> {
        Some(u32::from_le_bytes(self.data.get(off..off + 4)?.try_into().ok()?))
    }

    pub fn u64_at(&self, off: usize) -> Option<u64> {
        Some(u64::from_le_bytes(self.data.get(off..off + 8)?.try_into().ok()?))
    }
}

/// What a command answers.
#[derive(Debug, Default)]
pub struct Reply {
    pub result: u32,
    pub data: Vec<u8>,
    pub copy_handles: Vec<u32>,
    pub move_handles: Vec<u32>,
    /// New sub-interfaces: sessions of their own, or objects of the domain.
    pub objects: Vec<Service>,
}

impl Reply {
    pub fn ok() -> Reply {
        Reply::default()
    }

    pub fn error(result: u32) -> Reply {
        Reply { result, ..Reply::default() }
    }

    pub fn with(data: &[u8]) -> Reply {
        Reply { data: data.to_vec(), ..Reply::default() }
    }

    pub fn object(service: Service) -> Reply {
        Reply { objects: vec![service], ..Reply::default() }
    }
}

/// A client session: one service object, or a domain of them. Objects
/// live in `Kernel::objects`, so that cloned sessions share them.
#[derive(Debug, Clone, Copy)]
pub struct Session {
    pub object: usize,
    /// After ConvertCurrentObjectToDomain: its index in `Kernel::domains`.
    pub domain: Option<usize>,
}

/// A domain: object id -> index in `Kernel::objects` (the converted
/// session's own object is 1).
#[derive(Debug, Default)]
pub struct Domain {
    pub objects: std::collections::BTreeMap<u32, usize>,
    next: u32,
}

fn word(buf: &[u8], i: usize) -> Option<u32> {
    Some(u32::from_le_bytes(buf.get(4 * i..4 * i + 4)?.try_into().ok()?))
}

/// Parses the HIPC envelope of `buf`; `domain` says whether CMIF data
/// starts with a domain header. `None`: malformed.
pub fn parse(buf: &[u8], domain: bool) -> Option<Request> {
    let (w0, w1) = (word(buf, 0)?, word(buf, 1)?);
    let ty = (w0 & 0xFFFF) as u16;
    let (nx, na, nb, nw) = ((w0 >> 16 & 0xF) as usize, (w0 >> 20 & 0xF) as usize, (w0 >> 24 & 0xF) as usize, (w0 >> 28) as usize);
    let data_words = (w1 & 0x3FF) as usize;
    let mut i = 2;
    let (mut pid, mut copy_handles, mut move_handles) = (None, Vec::new(), Vec::new());
    if w1 & 0x8000_0000 != 0 {
        let sh = word(buf, i)?;
        i += 1;
        if sh & 1 != 0 {
            pid = Some(u64::from(word(buf, i)?) | u64::from(word(buf, i + 1)?) << 32);
            i += 2;
        }
        for _ in 0..(sh >> 1 & 0xF) {
            copy_handles.push(word(buf, i)?);
            i += 1;
        }
        for _ in 0..(sh >> 5 & 0xF) {
            move_handles.push(word(buf, i)?);
            i += 1;
        }
    }
    let mut x = Vec::new();
    for _ in 0..nx {
        let (d0, d1) = (u64::from(word(buf, i)?), u64::from(word(buf, i + 1)?));
        i += 2;
        x.push(Buffer { addr: d1 | (d0 >> 12 & 0xF) << 32 | (d0 >> 6 & 7) << 36, size: d0 >> 16 });
    }
    let abw = |n: usize, i: &mut usize| -> Option<Vec<Buffer>> {
        let mut v = Vec::new();
        for _ in 0..n {
            let (d0, d1, d2) = (u64::from(word(buf, *i)?), u64::from(word(buf, *i + 1)?), u64::from(word(buf, *i + 2)?));
            *i += 3;
            v.push(Buffer { addr: d1 | (d2 >> 28) << 32 | (d2 >> 2 & 7) << 36, size: d0 | (d2 >> 24 & 0xF) << 32 });
        }
        Some(v)
    };
    let (a, b, w) = (abw(na, &mut i)?, abw(nb, &mut i)?, abw(nw, &mut i)?);
    let raw_start = 4 * i;
    let raw = buf.get(raw_start..raw_start + 4 * data_words)?;
    // Receive lists after the data: mode 0 none, 2 one, n + 2 n of them.
    let mut c = Vec::new();
    let mode = (w1 >> 10 & 0xF) as usize;
    for k in 0..mode.saturating_sub(2).max(usize::from(mode == 2)) {
        let at = raw_start / 4 + data_words + 2 * k;
        let (lo, hi) = (u64::from(word(buf, at)?), u64::from(word(buf, at + 1)?));
        c.push(Buffer { addr: lo | (hi & 0xFFFF) << 32, size: hi >> 16 });
    }
    let mut r = Request { kind: Kind::Close, command: 0, pid, copy_handles, move_handles, x, a, b, w, c, data: Vec::new(), object: None, in_objects: Vec::new(), close_object: false };
    match ty {
        2 => {}
        TIPC_CLOSE => r.kind = Kind::Close,
        t if t >= TIPC => {
            r.kind = Kind::Tipc(u32::from(t - TIPC));
            r.command = u32::from(t - TIPC);
            r.data = raw.to_vec();
        }
        4..=7 => {
            r.kind = if matches!(ty, 5 | 7) { Kind::Control } else { Kind::Request };
            // CMIF data starts 16-byte aligned in the buffer.
            let mut at = raw_start.next_multiple_of(16) - raw_start;
            let mut payload_end = raw.len();
            if domain && r.kind == Kind::Request {
                let h = raw.get(at..at + 16)?;
                let count = h[1] as usize;
                let len = u16::from_le_bytes([h[2], h[3]]) as usize;
                r.object = Some(u32::from_le_bytes(h[4..8].try_into().ok()?));
                r.close_object = h[0] == 2;
                at += 16;
                payload_end = at + len;
                for k in 0..count {
                    let o = payload_end + 4 * k;
                    r.in_objects.push(u32::from_le_bytes(raw.get(o..o + 4)?.try_into().ok()?));
                }
                if r.close_object {
                    return Some(r);
                }
            }
            let head = raw.get(at..at + 16)?;
            if u32::from_le_bytes(head[0..4].try_into().ok()?) != SFCI {
                return None;
            }
            r.command = u32::from_le_bytes(head[8..12].try_into().ok()?);
            r.data = raw.get(at + 16..payload_end.max(at + 16))?.to_vec();
        }
        _ => return None,
    }
    Some(r)
}

/// Writes the reply to `req` into a message buffer of `size` bytes.
fn encode(req: &Request, reply: &Reply, objects: &[u32], domain: bool, size: usize) -> Vec<u8> {
    let mut words: Vec<u32> = vec![0, 0];
    let handles = !reply.copy_handles.is_empty() || !reply.move_handles.is_empty();
    if handles {
        words.push((reply.copy_handles.len() as u32) << 1 | (reply.move_handles.len() as u32) << 5);
        words.extend(&reply.copy_handles);
        words.extend(&reply.move_handles);
    }
    let raw_start = 4 * words.len();
    let mut raw: Vec<u8> = Vec::new();
    match req.kind {
        Kind::Tipc(_) => {
            raw.extend(reply.result.to_le_bytes());
            raw.extend(&reply.data);
        }
        _ => {
            raw.resize(raw_start.next_multiple_of(16) - raw_start, 0);
            if domain && req.kind == Kind::Request {
                raw.extend((objects.len() as u32).to_le_bytes());
                raw.extend([0; 12]);
            }
            raw.extend(SFCO.to_le_bytes());
            raw.extend(0u32.to_le_bytes());
            raw.extend(reply.result.to_le_bytes());
            raw.extend(0u32.to_le_bytes());
            raw.extend(&reply.data);
            if domain {
                for o in objects {
                    raw.extend(o.to_le_bytes());
                }
            }
        }
    }
    raw.resize(raw.len().next_multiple_of(4), 0);
    let ty: u32 = match req.kind {
        Kind::Tipc(c) => c + u32::from(TIPC),
        Kind::Control | Kind::Request | Kind::Close => 0,
    };
    words[0] = ty;
    words[1] = (raw.len() / 4) as u32 | if handles { 0x8000_0000 } else { 0 };
    let mut out: Vec<u8> = words.iter().flat_map(|w| w.to_le_bytes()).collect();
    out.extend(raw);
    out.resize(out.len().max(size.min(0x100)), 0);
    out
}

impl Kernel {
    /// A new session to a new object; returns its handle.
    pub fn open_session(&mut self, service: Service) -> Result<u32, u32> {
        self.objects.push(service);
        self.add_session(Session { object: self.objects.len() - 1, domain: None })
    }

    fn add_session(&mut self, session: Session) -> Result<u32, u32> {
        let s = self.sessions.len();
        let h = self.handles.add(Object::Session(s)).ok_or(OUT_OF_HANDLES)?;
        self.sessions.push(Some(session));
        Ok(h)
    }

    /// SendSyncRequest (`buf` = the TLS) and SendSyncRequestWithUserBuffer.
    pub fn send_request(&mut self, handle: u32, buf: u64, size: u64) -> u32 {
        let Some(Object::Session(s)) = self.handles.get(handle) else { return INVALID_HANDLE };
        if self.sessions.get(s).is_none_or(Option::is_none) {
            return SESSION_CLOSED;
        }
        let size = size.min(0x10000) as usize;
        let mut msg = vec![0; size];
        if self.mem.read(buf, &mut msg).is_err() {
            return INVALID_POINTER;
        }
        let domain = self.sessions[s].is_some_and(|x| x.domain.is_some());
        let Some(req) = parse(&msg, domain) else {
            let words: Vec<String> = msg.chunks(4).take(16).map(|c| format!("{:08x}", u32::from_le_bytes(c.try_into().unwrap_or([0; 4])))).collect();
            let what = format!("ipc: malformed message on session {s}: {}", words.join(" "));
            self.stop(crate::STOP_EXIT, &what);
        };
        let (reply, objects) = self.dispatch(s, &req);
        if req.kind == Kind::Close {
            self.sessions[s] = None;
        }
        let out = encode(&req, &reply, &objects, domain, size);
        if self.mem.write(buf, &out[..out.len().min(size)]).is_err() {
            return INVALID_POINTER;
        }
        SUCCESS
    }

    /// Runs one request; returns the reply and, in a domain, the ids of its new objects.
    fn dispatch(&mut self, s: usize, req: &Request) -> (Reply, Vec<u32>) {
        let session = self.sessions[s].expect("an open session");
        match req.kind {
            Kind::Close => return (Reply::ok(), Vec::new()),
            Kind::Control => return (self.control(s, req), Vec::new()),
            _ => {}
        }
        // The target: the session's object, or an object of its domain.
        let target = match (session.domain, req.object) {
            (Some(d), Some(id)) => {
                if req.close_object {
                    let known = self.domains[d].objects.remove(&id).is_some();
                    if self.journal.all_svcs || !known {
                        self.log(&format!("ipc domain {d}: close object {id}{}", if known { "" } else { " (unknown)" }));
                    }
                    return (Reply::ok(), Vec::new());
                }
                match self.domains[d].objects.get(&id) {
                    Some(&o) => o,
                    None => {
                        let ids: Vec<u32> = self.domains[d].objects.keys().copied().collect();
                        self.anomaly("ipc unknown domain object", &format!("domain {d}: command {} for unknown object {id} (known {ids:?})", req.command));
                        return (Reply::error(crate::result::INVALID_HANDLE), Vec::new());
                    }
                }
            }
            _ => session.object,
        };
        let mut svc = std::mem::take(&mut self.objects[target]);
        let label = svc.label();
        let mut reply = svc.request(self, req);
        self.objects[target] = svc;
        let mut ids = Vec::new();
        for o in std::mem::take(&mut reply.objects) {
            self.objects.push(o);
            let o = self.objects.len() - 1;
            match (session.domain, req.object) {
                (Some(d), Some(_)) => {
                    let domain = &mut self.domains[d];
                    domain.next += 1;
                    domain.objects.insert(domain.next, o);
                    ids.push(domain.next);
                }
                _ => match self.add_session(Session { object: o, domain: None }) {
                    Ok(h) => reply.move_handles.push(h),
                    Err(e) => reply.result = e,
                },
            }
        }
        let key = format!("ipc {label} {}", req.command);
        let key = format!("{key} {:#x}", reply.result);
        if self.journal.all_svcs || self.journal.first(&key) {
            let status = if reply.result == SUCCESS { String::new() } else { format!(" -> {:#x}", reply.result) };
            self.log(&format!("ipc {label} {}{status}", req.command));
        }
        (reply, ids)
    }

    /// IHipcManager: domains, clones and the pointer buffer size.
    fn control(&mut self, s: usize, req: &Request) -> Reply {
        let session = self.sessions[s].expect("an open session");
        let reply = match req.command {
            0 => {
                // ConvertCurrentObjectToDomain: the session's object becomes object 1.
                self.domains.push(Domain { objects: std::collections::BTreeMap::from([(1, session.object)]), next: 1 });
                self.sessions[s] = Some(Session { domain: Some(self.domains.len() - 1), ..session });
                Reply::with(&1u32.to_le_bytes())
            }
            // CloneCurrentObject, CloneCurrentObjectEx: one more session to the same object (or domain).
            2 | 4 => match self.add_session(session) {
                Ok(h) => Reply { move_handles: vec![h], ..Reply::ok() },
                Err(e) => Reply::error(e),
            },
            // QueryPointerBufferSize.
            3 => Reply::with(&0x8000u16.to_le_bytes()),
            c => {
                let what = format!("ipc control command {c} is not implemented");
                if !self.config.explore {
                    self.stop(crate::STOP_EXIT, &what);
                }
                self.log(&what);
                Reply::error(NOT_IMPLEMENTED)
            }
        };
        if self.journal.first(&format!("ipc control {}", req.command)) {
            self.log(&format!("ipc control {}", req.command));
        }
        reply
    }

    /// ConnectToNamedPort: only `sm:` exists.
    pub fn connect_to_named_port(&mut self, name_addr: u64) -> Result<u32, u32> {
        let mut name = [0; 12];
        self.mem.read(name_addr, &mut name).map_err(|_| INVALID_POINTER)?;
        let len = name.iter().position(|&b| b == 0).unwrap_or(12);
        match &name[..len] {
            b"sm:" => self.open_session(Service::Sm(Default::default())),
            other => {
                let what = format!("ConnectToNamedPort {:?}: no such port", String::from_utf8_lossy(other));
                self.log(&what);
                Err(NOT_FOUND)
            }
        }
    }
}
