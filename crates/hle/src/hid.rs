//! `hid`: one controller (FullKey, slot 0) plugged in and at rest, in the
//! shared memory layout the SDK reads (the Switch 1 one, our reading of the
//! Switch 2 SDK's readers). Every lifo holds one null entry: its sequence
//! is even (no write in progress), so the readers never spin.

use crate::Kernel;
use crate::ipc::{Reply, Request};
use crate::service::Service;
use crate::thread::Object;

pub const SHARED_SIZE: u64 = 0x40000;
/// Touch screen, mouse and keyboard lifos.
const BASIC_LIFOS: [usize; 3] = [0x400, 0x3400, 0x3800];
/// Npad slots: 0x9a00 + slot × 0x5000, slots 0..=9.
const NPAD: usize = 0x9a00;
const NPAD_SLOT: usize = 0x5000;
/// Button lifos (FullKey, Handheld, JoyDual, JoyLeft, JoyRight) and six-axis lifos in a slot.
const SLOT_LIFOS: [usize; 11] = [0x28, 0x378, 0x6c8, 0xa18, 0xd68, 0x1758, 0x1e60, 0x2568, 0x2c70, 0x3378, 0x3a80];
/// The Npad condition block: hold type at +8, "initialized" at +0xc.
const CONDITION: usize = 0x3e200;
/// FullKey style, and the attribute "connected" of a button state.
const FULL_KEY: u32 = 1;
const CONNECTED: u32 = 1;

#[derive(Debug)]
pub enum Hid {
    /// `hid` (IHidServer): the style set and ids the game supports.
    Service { styles: u32, ids: Vec<u32> },
    AppletResource,
    VibrationList,
}

impl Hid {
    pub fn label(&self) -> String {
        match self {
            Hid::Service { .. } => "hid".into(),
            Hid::AppletResource => "hid IAppletResource".into(),
            Hid::VibrationList => "hid IActiveVibrationDeviceList".into(),
        }
    }

    pub fn request(&mut self, k: &mut Kernel, req: &Request) -> Option<Reply> {
        Some(match (self, req.command) {
            (Hid::Service { .. }, 0) => return Some(Reply::object(Service::Hid(Hid::AppletResource))),
            // GetSharedMemoryHandle.
            (Hid::AppletResource, 0) => {
                let m = k.hid_memory();
                Reply { copy_handles: vec![k.handles.add(Object::SharedMemory(m))?], ..Reply::ok() }
            }
            // Activations, touch screen size, six-axis and Jcms start.
            (Hid::Service { .. }, 11 | 21 | 31 | 66 | 1004 | 3112 | 3150) => Reply::ok(),
            // ActivateNpadWithRevision: the condition becomes "initialized".
            (Hid::Service { .. }, 109) => {
                k.npad_condition(1);
                Reply::ok()
            }
            // InitializeJcms: the client lends a buffer the server fills; it starts empty.
            (Hid::Service { .. }, 3110) => {
                if let Some(&h) = req.copy_handles.first()
                    && let Some((addr, size)) = k.transfer_range(h)
                {
                    k.retain_transfer(h);
                    let _ = k.mem.write_object(addr, &vec![0; size as usize]);
                    k.log(&format!("hid 3110: jcms buffer {addr:#x}+{size:#x} zeroed"));
                }
                Reply::ok()
            }
            // SetSupportedNpadStyleSet / GetSupportedNpadStyleSet.
            (Hid::Service { styles, .. }, 100) => {
                let s = req.u32_at(0)?;
                if *styles != s {
                    k.log(&format!("hid: supported style set {s:#x}"));
                }
                *styles = s;
                Reply::ok()
            }
            (Hid::Service { styles, .. }, 101) => Reply::with(&styles.to_le_bytes()),
            // SetSupportedNpadIdType: the ids, in the pointer buffer.
            (Hid::Service { ids, .. }, 102) => {
                if let Some(b) = req.send_buffer(0) {
                    let mut raw = vec![0; b.size.min(0x100) as usize];
                    let _ = k.mem.read(b.addr, &mut raw);
                    *ids = raw.chunks(4).filter(|c| c.len() == 4).map(|c| u32::from_le_bytes(c.try_into().unwrap())).collect();
                    k.log(&format!("hid: supported npad ids {ids:x?}"));
                }
                Reply::ok()
            }
            // SetNpadJoyHoldType.
            (Hid::Service { .. }, 120) => {
                k.npad_condition(req.u64_at(8)? as u32);
                Reply::ok()
            }
            // CreateActiveVibrationDeviceList; ActivateVibrationDevice; vibration values.
            (Hid::Service { .. }, 203) => return Some(Reply::object(Service::Hid(Hid::VibrationList))),
            (Hid::VibrationList, 0) => Reply::ok(),
            (Hid::Service { .. }, 201 | 213) => Reply::ok(),
            _ => return None,
        })
    }
}

impl Kernel {
    /// hid's shared memory, created and filled at the first request.
    fn hid_memory(&mut self) -> usize {
        if let Some(m) = self.hid_shared {
            return m;
        }
        let m = self.create_shared_memory(SHARED_SIZE);
        self.hid_shared = Some(m);
        // One null entry in every lifo: tail 0, count 1, entry 0 all zero.
        let one = 1u64.to_le_bytes();
        for l in BASIC_LIFOS {
            self.write_shared(m, l + 0x18, &one);
        }
        for s in 0..10 {
            for l in SLOT_LIFOS {
                self.write_shared(m, NPAD + s * NPAD_SLOT + l + 0x18, &one);
            }
        }
        // Slot 0: a FullKey controller, connected, nothing pressed.
        self.write_shared(m, NPAD, &FULL_KEY.to_le_bytes());
        self.write_shared(m, NPAD + 0x28 + 0x20 + 0x28, &CONNECTED.to_le_bytes());
        self.npad_condition(1);
        m
    }

    /// The Npad condition: hold type and "initialized".
    fn npad_condition(&mut self, hold: u32) {
        let m = self.hid_memory_index();
        if let Some(m) = m {
            self.write_shared(m, CONDITION + 8, &hold.to_le_bytes());
            self.write_shared(m, CONDITION + 0xc, &[1]);
        }
    }

    fn hid_memory_index(&self) -> Option<usize> {
        self.hid_shared
    }
}
