//! `bsd:s`/`bsd:u`, always offline (rule 9, D23). A console without a
//! network still creates sockets and has its loopback, so sockets here are
//! descriptors of ours over an in-memory loopback: a socket binds to a
//! local address, listens, connects to a local listener, and the pair
//! exchanges bytes through buffers. No host socket is ever opened, and any
//! non-local address fails as on an unplugged console. Command layouts
//! follow the public documentation (switchbrew).

use std::collections::VecDeque;

use crate::Kernel;
use crate::ipc::{Reply, Request};

/// BSD errno values.
pub const EBADF: u32 = 9;
pub const EAGAIN: u32 = 35;
pub const EINVAL: u32 = 22;
pub const EADDRINUSE: u32 = 48;
pub const EADDRNOTAVAIL: u32 = 49;
pub const ENETDOWN: u32 = 50;
pub const ENETUNREACH: u32 = 51;
pub const ECONNREFUSED: u32 = 61;
pub const ENOTCONN: u32 = 57;
/// fcntl.
const F_GETFL: u32 = 3;
const F_SETFL: u32 = 4;
/// poll events.
const POLLIN: u16 = 1;
const POLLOUT: u16 = 4;
/// The first ephemeral port.
const EPHEMERAL: u16 = 49152;

#[derive(Debug, Default)]
pub struct Socket {
    open: bool,
    port: u16,
    listening: bool,
    /// Listener: accepted-side descriptors waiting for Accept.
    pending: VecDeque<i32>,
    /// Connected: the other end, and the bytes it sent us.
    peer: Option<i32>,
    inbox: VecDeque<u8>,
    flags: u32,
}

#[derive(Debug)]
pub struct Bsd {
    pub name: String,
    /// Descriptors from 3, never reused.
    pub sockets: Vec<Socket>,
    next_port: u16,
}

impl Bsd {
    pub fn new(name: String) -> Bsd {
        Bsd { name, sockets: Vec::new(), next_port: EPHEMERAL }
    }

    fn get(&mut self, fd: i32) -> Option<&mut Socket> {
        usize::try_from(fd - 3).ok().and_then(|i| self.sockets.get_mut(i)).filter(|s| s.open)
    }

    fn create(&mut self) -> i32 {
        self.sockets.push(Socket { open: true, ..Socket::default() });
        self.sockets.len() as i32 + 2
    }

    /// The listener on local `port`, if any.
    fn listener(&self, port: u16) -> Option<i32> {
        self.sockets.iter().position(|s| s.open && s.listening && s.port == port).map(|i| i as i32 + 3)
    }

    pub fn request(&mut self, k: &mut Kernel, req: &Request) -> Option<Reply> {
        // Most bsd commands answer (i32 return value, u32 errno), some a u32 more.
        let answer = |ret: i32, errno: u32| Reply::with(&[ret.to_le_bytes(), errno.to_le_bytes()].concat());
        let answer3 = |ret: i32, errno: u32, n: u32| Reply::with(&[ret.to_le_bytes(), errno.to_le_bytes(), n.to_le_bytes()].concat());
        let fd = req.u32_at(0).unwrap_or(0) as i32;
        let read_buffer = |k: &Kernel, i: usize, max: u64| -> Vec<u8> {
            let b = req.send_buffer(i).unwrap_or_default();
            let mut v = vec![0; b.size.min(max) as usize];
            let _ = k.mem.read(b.addr, &mut v);
            v
        };
        // A local sockaddr_in: length, family, port (big endian), address.
        let local = |port: u16| -> [u8; 16] {
            let mut sa = [0u8; 16];
            sa[0] = 16;
            sa[1] = 2;
            sa[2..4].copy_from_slice(&port.to_be_bytes());
            sa[4..8].copy_from_slice(&[127, 0, 0, 1]);
            sa
        };
        let reply = match req.command {
            // RegisterClient (0; 36 on Switch 2): a client id; StartMonitoring.
            0 | 36 => Reply::with(&0u64.to_le_bytes()),
            1 => Reply::ok(),
            // Socket, SocketExempt(domain, type, protocol).
            2 | 3 => {
                let fd = self.create();
                if k.journal.few("bsd socket", 4) {
                    let (domain, kind, protocol) = (req.u32_at(0)?, req.u32_at(4)?, req.u32_at(8)?);
                    k.log(&format!("bsd: socket {fd} (domain {domain}, type {kind}, protocol {protocol}); offline, loopback only"));
                }
                answer(fd, 0)
            }
            // Bind(fd, sockaddr): local addresses only; port 0 picks an ephemeral one.
            13 => {
                let sa = read_buffer(k, 0, 16);
                let (port, ip) = (u16::from_be_bytes([sa[2], sa[3]]), [sa[4], sa[5], sa[6], sa[7]]);
                let is_local = ip == [0; 4] || ip[0] == 127;
                let port = if port == 0 {
                    self.next_port += 1;
                    self.next_port - 1
                } else {
                    port
                };
                let taken = self.sockets.iter().any(|s| s.open && s.port == port);
                k.log(&format!("bsd: bind {fd} to {}.{}.{}.{}:{port}{}", ip[0], ip[1], ip[2], ip[3], if is_local { "" } else { ": not available offline" }));
                match self.get(fd) {
                    None => answer(-1, EBADF),
                    Some(_) if !is_local => answer(-1, EADDRNOTAVAIL),
                    Some(_) if taken => answer(-1, EADDRINUSE),
                    Some(s) => {
                        s.port = port;
                        answer(0, 0)
                    }
                }
            }
            // Listen(fd, backlog).
            18 => match self.get(fd) {
                Some(s) if s.port != 0 => {
                    s.listening = true;
                    answer(0, 0)
                }
                Some(_) => answer(-1, EINVAL),
                None => answer(-1, EBADF),
            },
            // GetSockName / GetPeerName(fd) into the B buffer.
            16 | 15 => {
                let port = match self.get(fd) {
                    Some(s) if req.command == 16 => Some(s.port),
                    Some(s) => s.peer.map(|_| s.port),
                    None => None,
                };
                match port {
                    Some(port) => {
                        if let Some(b) = req.out_buffer(0) {
                            let sa = local(port);
                            let _ = k.mem.write(b.addr, &sa[..b.size.min(16) as usize]);
                        }
                        answer3(0, 0, 16)
                    }
                    None if self.get(fd).is_some() => answer3(-1, ENOTCONN, 0),
                    None => answer3(-1, EBADF, 0),
                }
            }
            // Connect(fd, sockaddr): to a local listener only.
            14 => {
                let sa = read_buffer(k, 0, 16);
                let (port, ip) = (u16::from_be_bytes([sa[2], sa[3]]), [sa[4], sa[5], sa[6], sa[7]]);
                let target = (ip[0] == 127).then(|| self.listener(port)).flatten();
                k.log(&format!("bsd: connect {fd} to {}.{}.{}.{}:{port}{}", ip[0], ip[1], ip[2], ip[3], if target.is_some() { "" } else { ": unreachable offline" }));
                if self.get(fd).is_none() {
                    answer(-1, EBADF)
                } else if let Some(l) = target {
                    let server = self.create();
                    self.next_port += 1;
                    let client_port = self.next_port - 1;
                    let s = self.get(server)?;
                    (s.peer, s.port) = (Some(fd), port);
                    let c = self.get(fd)?;
                    (c.peer, c.port) = (Some(server), client_port);
                    self.get(l)?.pending.push_back(server);
                    answer(0, 0)
                } else if ip[0] == 127 {
                    answer(-1, ECONNREFUSED)
                } else {
                    answer(-1, ENETUNREACH)
                }
            }
            // Accept(fd): a queued connection and its peer's address.
            12 => match self.get(fd) {
                Some(s) if s.listening => match s.pending.pop_front() {
                    Some(n) => {
                        if let Some(b) = req.out_buffer(0) {
                            let sa = local(0);
                            let _ = k.mem.write(b.addr, &sa[..b.size.min(16) as usize]);
                        }
                        answer3(n, 0, 16)
                    }
                    None => answer3(-1, EAGAIN, 0),
                },
                Some(_) => answer3(-1, EINVAL, 0),
                None => answer3(-1, EBADF, 0),
            },
            // Send(fd, flags) / Write(fd) from the A buffer: into the peer's inbox.
            10 | 24 => {
                let data = read_buffer(k, 0, 0x10000);
                let peer = self.get(fd).map(|s| s.peer);
                match peer {
                    Some(Some(p)) => match self.get(p) {
                        Some(ps) => {
                            ps.inbox.extend(&data);
                            answer(data.len() as i32, 0)
                        }
                        None => answer(-1, ENOTCONN),
                    },
                    Some(None) => answer(-1, ENOTCONN),
                    None => answer(-1, EBADF),
                }
            }
            // Recv(fd, flags) / Read(fd) into the B buffer.
            8 | 25 => {
                let size = req.out_buffer(0).map_or(0, |b| b.size.min(0x10000) as usize);
                match self.get(fd) {
                    Some(s) if s.inbox.is_empty() && s.peer.is_some() => answer(-1, EAGAIN),
                    Some(s) => {
                        let n = size.min(s.inbox.len());
                        let bytes: Vec<u8> = s.inbox.drain(..n).collect();
                        if let Some(b) = req.out_buffer(0) {
                            let _ = k.mem.write(b.addr, &bytes);
                        }
                        answer(n as i32, 0)
                    }
                    None => answer(-1, EBADF),
                }
            }
            // Fcntl(fd, cmd, flags).
            20 => {
                let (cmd, flags) = (req.u32_at(4)?, req.u32_at(8).unwrap_or(0));
                match self.get(fd) {
                    Some(s) if cmd == F_GETFL => answer(s.flags as i32, 0),
                    Some(s) if cmd == F_SETFL => {
                        s.flags = flags;
                        answer(0, 0)
                    }
                    Some(_) => answer(-1, EINVAL),
                    None => answer(-1, EBADF),
                }
            }
            // SetSockOpt (accepted) and GetSockOpt (zero) on an open socket.
            21 => if self.get(fd).is_some() { answer(0, 0) } else { answer(-1, EBADF) },
            17 => {
                if let Some(b) = req.out_buffer(0) {
                    let _ = k.mem.write(b.addr, &vec![0; b.size.min(0x100) as usize]);
                }
                if self.get(fd).is_some() { answer3(0, 0, 4) } else { answer3(-1, EBADF, 0) }
            }
            // Poll(nfds, timeout ms): readiness from the loopback; with nothing ready the
            // caller sleeps its timeout (no early wake-up).
            6 => {
                let (nfds, timeout) = (req.u32_at(0)? as usize, req.u32_at(4)? as i32);
                let mut fds = read_buffer(k, 0, 8 * nfds.min(64) as u64);
                let mut ready = 0;
                for p in fds.as_chunks_mut::<8>().0 {
                    let (pfd, events) = (i32::from_le_bytes(p[0..4].try_into().ok()?), u16::from_le_bytes([p[4], p[5]]));
                    let revents = match self.get(pfd) {
                        Some(s) => {
                            let readable = !s.inbox.is_empty() || !s.pending.is_empty();
                            (if readable { POLLIN } else { 0 } | if s.peer.is_some() { POLLOUT } else { 0 }) & events
                        }
                        None => 0x20, // POLLNVAL
                    };
                    p[6..8].copy_from_slice(&revents.to_le_bytes());
                    ready += i32::from(revents != 0);
                }
                if let Some(b) = req.out_buffer(0) {
                    let _ = k.mem.write(b.addr, &fds[..fds.len().min(b.size as usize)]);
                }
                if ready == 0 && timeout != 0 {
                    k.sleep(if timeout < 0 { i64::MAX } else { i64::from(timeout) * 1_000_000 });
                }
                answer(ready, 0)
            }
            // Close.
            26 => match self.get(fd) {
                Some(s) => {
                    s.open = false;
                    answer(0, 0)
                }
                None => answer(-1, EBADF),
            },
            _ => return None,
        };
        Some(reply)
    }
}
