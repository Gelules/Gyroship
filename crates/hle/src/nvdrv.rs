//! The nvdrv-like session gsv 1402 hands out (CMIF): Open, Ioctl, Close,
//! QueryEvent on nvhost devices. A fake: it answers the queries the SDK's
//! boot makes and executes nothing.

use crate::Kernel;
use crate::ipc::{Reply, Request};

#[derive(Debug, Default)]
pub struct Nvdrv {
    /// Open devices, by fd (from 1, never reused).
    pub fds: Vec<String>,
}

impl Nvdrv {
    pub fn label(&self) -> String {
        "nvsrv".into()
    }

    pub fn request(&mut self, k: &mut Kernel, req: &Request) -> Option<Reply> {
        Some(match req.command {
            // SetGraphicsFirmwareMemoryMarginEnabled.
            13 => {
                k.log(&format!("nvsrv 13: {:#x}", req.u64_at(0).unwrap_or(0)));
                Reply::ok()
            }
            _ => return None,
        })
    }
}
