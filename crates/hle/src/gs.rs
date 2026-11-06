//! The Switch 2 graphics supervisor (what ISelfController 2000 hands out)
//! and its TIPC sessions: syncpoints, the GPU, GPU memory, vi2. No public
//! documentation exists; the replies follow what the SDK's clients read
//! (our analysis of `nn::gs::sdk::BootGraphicsSystem` and its callees,
//! build `a7730d0f`).
//!
//! The fake GPU is complaisant: every fence it hands out is already
//! reached, and every wait it is asked about succeeds at once.

use crate::Kernel;
use crate::ipc::{Reply, Request};
use crate::service::Service;
use crate::thread::Object;

/// State the graphics sessions share (plain data).
#[derive(Debug, Default)]
pub struct State {
    /// gpu 12: the event of each GPU event id (`Kernel::events` indices).
    pub gpu_events: Vec<Option<usize>>,
    /// gsv 204: the syncpoint memory (read-only page for the guest) and its two events.
    pub syncpt_page: Option<u64>,
    pub syncpt_events: Option<(usize, usize)>,
}

#[derive(Debug)]
pub enum Gs {
    /// The graphics supervisor session.
    Supervisor,
    Syncpt,
    Gpu,
    GpuMemory,
}

/// The syncpoint memory header gsv 204 writes: version 1 and no in-memory
/// syncpoints (n = 0), so every syncpoint read goes through syncpt IPC.
const SYNCPT_HEADER: [u32; 8] = [1, 0, 0, 0x100, 0, 0x40, 0x8, 0x80];

impl Gs {
    pub fn label(&self) -> String {
        match self {
            Gs::Supervisor => "gsv".into(),
            Gs::Syncpt => "gsv syncpt".into(),
            Gs::Gpu => "gsv gpu".into(),
            Gs::GpuMemory => "gsv gpu memory".into(),
        }
    }

    pub fn request(&mut self, k: &mut Kernel, req: &Request) -> Option<Reply> {
        let gs = |g: Gs| Some(Reply::object(Service::Gs(g)));
        Some(match (self, req.command) {
            // Boot: the process (pseudo-handle), its aruid, flags.
            (Gs::Supervisor, 4) => {
                k.log(&format!("gsv 4: aruid {:#x}", req.u64_at(0).unwrap_or(0)));
                Reply::ok()
            }
            // Client registration and permission checks.
            (Gs::Supervisor, 11 | 101 | 1501 | 1502 | 201 | 1101 | 1102 | 401 | 402) => Reply::ok(),
            // The nvdrv-like session (CMIF).
            (Gs::Supervisor, 1402) => return Some(Reply::object(Service::Nvdrv(crate::nvdrv::Nvdrv::default()))),
            (Gs::Supervisor, 202) => return gs(Gs::Syncpt),
            // Syncpoint memory: two transfer memories (RW, then R); two events back.
            (Gs::Supervisor, 204) => return Some(k.syncpoint_memory(req)),
            // QueryGrantedFeatures: the value is not read by the SDK; Lichii's.
            (Gs::Supervisor, 115) => Reply::with(&2u64.to_le_bytes()),
            (Gs::Supervisor, 497) => return gs(Gs::Gpu),
            // syncpt GetCount (at most 0x3ff), then 21.
            (Gs::Syncpt, 1) => Reply::with(&32u32.to_le_bytes()),
            (Gs::Syncpt, 21) => Reply::ok(),
            // gpu: the range of free event ids (at least two).
            (Gs::Gpu, 11) => Reply::with(&[0u32.to_le_bytes(), 31u32.to_le_bytes()].concat()),
            // The event of an id, which the HLE signals when the GPU "finishes".
            (Gs::Gpu, 12) => {
                let n = req.u32_at(0)? as usize;
                let (w, r) = k.create_event().ok()?;
                let e = k.event_of(r)?;
                k.handles.close(w);
                if k.gs.gpu_events.len() <= n {
                    k.gs.gpu_events.resize(n + 1, None);
                }
                k.gs.gpu_events[n] = Some(e);
                Reply { move_handles: vec![r], ..Reply::ok() }
            }
            (Gs::Gpu, 21) => {
                let n = req.u32_at(0)? as usize;
                k.signal_gpu(n);
                Reply::ok()
            }
            (Gs::Gpu, 22) => return gs(Gs::GpuMemory),
            _ => return None,
        })
    }
}

impl Kernel {
    /// Signals the event of GPU event id `n`, as if the GPU had finished.
    pub fn signal_gpu(&mut self, n: usize) {
        if let Some(Some(e)) = self.gs.gpu_events.get(n).copied() {
            self.signal_event(e);
            if self.journal.first(&format!("gpu event {n}")) {
                self.log(&format!("gsv gpu: event {n} signaled"));
            }
        }
    }

    /// The address and size of the transfer memory a handle names.
    pub fn transfer_range(&self, h: u32) -> Option<(u64, u64)> {
        let Object::TransferMemory(t) = self.handles.get(h)? else { return None };
        Some((self.transfers[t].addr, self.transfers[t].size))
    }

    /// gsv 204: keeps both transfer memories, writes the syncpoint header
    /// into the read-only one, and answers a writable request event and a
    /// readable acknowledge event.
    fn syncpoint_memory(&mut self, req: &Request) -> Reply {
        let (Some(&rw), Some(&r)) = (req.copy_handles.first(), req.copy_handles.get(1)) else {
            return Reply::error(crate::result::INVALID_HANDLE);
        };
        let (Some(a), Some(b)) = (self.transfer_range(rw), self.transfer_range(r)) else {
            return Reply::error(crate::result::INVALID_HANDLE);
        };
        self.retain_transfer(rw);
        self.retain_transfer(r);
        let mut page = vec![0u8; 0x200];
        for (i, v) in SYNCPT_HEADER.iter().enumerate() {
            page[4 * i..4 * i + 4].copy_from_slice(&v.to_le_bytes());
        }
        if self.mem.write_object(b.0, &page).is_err() {
            return Reply::error(crate::result::INVALID_CURRENT_MEMORY);
        }
        let (Ok((req_w, _req_r)), Ok((_ack_w, ack_r))) = (self.create_event(), self.create_event()) else {
            return Reply::error(crate::result::OUT_OF_HANDLES);
        };
        self.gs.syncpt_page = Some(b.0);
        self.gs.syncpt_events = self.event_of(_req_r).zip(self.event_of(ack_r));
        self.log(&format!("gsv 204: syncpoint memory rw {:#x}+{:#x} r {:#x}+{:#x}, header version 1 n 0", a.0, a.1, b.0, b.1));
        Reply { move_handles: vec![req_w, ack_r], ..Reply::ok() }
    }
}
