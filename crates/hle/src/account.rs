//! `acc:u0` with one local user, `nifm:u` that never connects (rule 9),
//! and `ldr:ro`, which only registers the process.

use crate::Kernel;
use crate::ipc::{Reply, Request};
use crate::service::Service;

/// The one local user's id: fixed by the run's seed, never zero.
pub fn user_id(k: &Kernel) -> [u8; 16] {
    let s = k.config.seed ^ 0x7573_6572; // "user"
    [s.to_le_bytes(), s.rotate_left(17).to_le_bytes()].concat().try_into().expect("16 bytes")
}

/// The local user's nickname.
const NICKNAME: &[u8] = b"Gyroship";

#[derive(Debug)]
pub enum Account {
    /// `acc:u0`.
    Service,
    Profile,
    /// IManagerForApplication (acc:u0 101), the online account: no command seen yet.
    Baas,
}

impl Account {
    pub fn label(&self) -> String {
        match self {
            Account::Service => "acc:u0".into(),
            Account::Profile => "acc IProfile".into(),
            Account::Baas => "acc IManagerForApplication".into(),
        }
    }

    pub fn request(&mut self, k: &mut Kernel, req: &Request) -> Option<Reply> {
        let uid = user_id(k);
        Some(match (self, req.command) {
            // GetUserCount.
            (Account::Service, 0) => Reply::with(&1u32.to_le_bytes()),
            // GetUserExistence(uid).
            (Account::Service, 1) => Reply::with(&[u8::from(req.data.get(..16) == Some(&uid[..]))]),
            // ListAllUsers, ListOpenUsers: the user, into the C/B buffer.
            (Account::Service, 2 | 3) => {
                if let Some(b) = req.out_buffer(0).filter(|b| b.size >= 16) {
                    let _ = k.mem.write(b.addr, &uid);
                }
                Reply::ok()
            }
            // GetLastOpenedUser.
            (Account::Service, 4) => Reply::with(&uid),
            // GetProfile(uid).
            (Account::Service, 5) => Reply::object(Service::Account(Account::Profile)),
            // IsUserAccountSwitchLocked; InitializeApplicationInfo(V2).
            (Account::Service, 150) => Reply::with(&[0]),
            // GetBaasAccountManagerForApplication: the manager of the user's online account.
            (Account::Service, 101) => return Some(Reply::object(crate::service::Service::Account(Account::Baas))),
            (Account::Service, 100 | 140 | 160) => Reply::ok(),
            // GetBase: the user id, a last-edit time, the nickname (0x21 bytes).
            (Account::Profile, 0 | 1) => {
                let mut base = vec![0u8; 0x38];
                base[..16].copy_from_slice(&uid);
                base[0x18..0x18 + NICKNAME.len()].copy_from_slice(NICKNAME);
                if req.command == 0
                    && let Some(b) = req.out_buffer(0)
                {
                    // Get also fills the user data buffer: none.
                    let _ = k.mem.write(b.addr, &vec![0; b.size.min(0x80) as usize]);
                }
                Reply::with(&base)
            }
            _ => return None,
        })
    }
}

/// nifm's result for a request that cannot reach a network (module 110,
/// description 2400): Gyroship is offline.
pub const NIFM_NO_NETWORK: u32 = 110 | 2400 << 9;

#[derive(Debug)]
pub enum Nifm {
    /// `nifm:u`.
    Service,
    General,
    /// A connection request and its two events (state changed, something else).
    Request { events: [u32; 2] },
}

impl Nifm {
    pub fn label(&self) -> String {
        match self {
            Nifm::Service => "nifm:u".into(),
            Nifm::General => "nifm IGeneralService".into(),
            Nifm::Request { .. } => "nifm IRequest".into(),
        }
    }

    pub fn request(&mut self, k: &mut Kernel, req: &Request) -> Option<Reply> {
        Some(match (self, req.command) {
            // CreateGeneralServiceOld, CreateGeneralService.
            (Nifm::Service, 4 | 5) => Reply::object(Service::Nifm(Nifm::General)),
            // CreateRequest.
            (Nifm::General, 4) => {
                let events = [k.new_event_handle()?, k.new_event_handle()?];
                Reply::object(Service::Nifm(Nifm::Request { events }))
            }
            // GetInternetConnectionStatus: there is none.
            (Nifm::General, 18) => Reply::error(NIFM_NO_NETWORK),
            // IsWirelessCommunicationEnabled, IsEthernetCommunicationEnabled,
            // IsAnyInternetRequestAccepted: no.
            (Nifm::General, 20..=22) => Reply::with(&[0]),
            // GetRequestState: 1, free (never submitted, or failed).
            (Nifm::Request { .. }, 0) => Reply::with(&1u32.to_le_bytes()),
            // GetResult: no network.
            (Nifm::Request { .. }, 1) => Reply::error(NIFM_NO_NETWORK),
            // GetSystemEventReadableHandles.
            (Nifm::Request { events }, 2) => {
                let copies = [k.copy_event(k.event_of(events[0])?)?, k.copy_event(k.event_of(events[1])?)?];
                Reply { copy_handles: copies.to_vec(), ..Reply::ok() }
            }
            // Cancel, Submit: a submitted request fails at once and says so.
            (Nifm::Request { .. }, 3) => Reply::ok(),
            (Nifm::Request { events }, 4) => {
                let e = k.event_of(events[0])?;
                k.signal_event(e);
                Reply::ok()
            }
            // SetRequirementPreset, SetConnectionConfirmationOption, and other settings.
            (Nifm::Request { .. }, 6 | 8 | 11 | 12 | 13 | 14 | 15 | 16 | 17 | 18 | 19 | 20) => Reply::ok(),
            // GetAppletInfo: no applet to show. The SDK then launches none, and
            // HandleNetworkRequestErrorResult reports the network as unavailable.
            (Nifm::Request { .. }, 21) => Reply::error(NIFM_NO_NETWORK),
            _ => return None,
        })
    }
}
