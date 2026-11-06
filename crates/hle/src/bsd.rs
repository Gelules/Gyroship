//! `bsd:s`/`bsd:u`, always offline (rule 9): clients register and monitor,
//! but every socket is refused, and no host socket is ever opened.

use crate::Kernel;
use crate::ipc::{Reply, Request};

/// The errno a refused socket gets: ENETDOWN (BSD numbering).
pub const ENETDOWN: u32 = 50;

#[derive(Debug)]
pub struct Bsd {
    pub name: String,
}

impl Bsd {
    pub fn request(&mut self, k: &mut Kernel, req: &Request) -> Option<Reply> {
        // Most bsd commands answer (i32 return value, u32 errno).
        let answer = |ret: i32, errno: u32| Reply::with(&[ret.to_le_bytes(), errno.to_le_bytes()].concat());
        Some(match req.command {
            // RegisterClient (0; 36 on Switch 2): a client id; StartMonitoring.
            0 | 36 => Reply::with(&0u64.to_le_bytes()),
            1 => Reply::ok(),
            // Socket, SocketExempt: refused.
            2 | 3 => {
                if k.journal.first("bsd socket") {
                    k.log("bsd: sockets are refused: Gyroship is offline");
                }
                answer(-1, ENETDOWN)
            }
            _ => return None,
        })
    }
}
