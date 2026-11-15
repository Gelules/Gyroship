//! am, as an application sees it through `appletOE`: the application proxy
//! and its sub-interfaces. Commands follow the public Switch 1 numbering
//! (switchbrew); the Switch 2 additions (IApplicationProxy 40 and 60,
//! IApplicationFunctions 320 and 321, ISelfController 2400) answer neutrally
//! until their meaning is known.

use crate::Kernel;
use crate::ipc::{Reply, Request};
use crate::service::Service;
use std::collections::VecDeque;

/// am's result for an empty message queue (module 128, description 3).
pub const NO_MESSAGE: u32 = 128 | 3 << 9;
/// Applet message: the focus state changed. `nn::oe::InitializeApplet`
/// waits for it, then for the focus state 1.
pub const FOCUS_STATE_CHANGED: u32 = 15;
/// am's result for an empty channel (module 128, description 2).
pub const NO_DATA_IN_CHANNEL: u32 = 128 | 2 << 9;
/// The launch parameter that names the preselected user.
const PRESELECTED_USER_MAGIC: u32 = 0xC794_97CA;
/// GetCurrentFocusState: in focus.
const IN_FOCUS: u8 = 1;
/// The applet resource user id this process gets (Lichii's: 81).
pub const ARUID: u64 = 0x51;

#[derive(Debug)]
pub enum Applet {
    /// `appletOE`.
    Service,
    Proxy,
    /// ICommonStateGetter: its message queue and the event that signals it.
    CommonState { messages: VecDeque<u32>, event: usize },
    SelfController,
    Window,
    Audio,
    Display,
    LibraryAppletCreator,
    ProcessWinding,
    /// IApplicationFunctions; the preselected user is handed out once.
    Functions { popped: bool },
    Debug,
    /// An IStorage of bytes and its accessor.
    Storage(Vec<u8>),
    StorageAccessor(Vec<u8>),
    /// Switch 2: IApplicationProxy 40 and 60, not identified yet.
    Unknown(u32),
}

impl Applet {
    pub fn label(&self) -> String {
        match self {
            Applet::Service => "appletOE".into(),
            Applet::Proxy => "IApplicationProxy".into(),
            Applet::CommonState { .. } => "ICommonStateGetter".into(),
            Applet::SelfController => "ISelfController".into(),
            Applet::Window => "IWindowController".into(),
            Applet::Audio => "IAudioController".into(),
            Applet::Display => "IDisplayController".into(),
            Applet::LibraryAppletCreator => "ILibraryAppletCreator".into(),
            Applet::ProcessWinding => "IProcessWindingController".into(),
            Applet::Functions { .. } => "IApplicationFunctions".into(),
            Applet::Storage(_) => "am IStorage".into(),
            Applet::StorageAccessor(_) => "am IStorageAccessor".into(),
            Applet::Debug => "IDebugFunctions".into(),
            Applet::Unknown(c) => format!("IApplicationProxy {c} interface"),
        }
    }

    pub fn request(&mut self, k: &mut Kernel, req: &Request) -> Option<Reply> {
        let applet = |a: Applet| Some(Reply::object(Service::Applet(a)));
        Some(match (self, req.command) {
            // OpenApplicationProxy (10 on Switch 2, 0 on Switch 1).
            (Applet::Service, 0 | 10) => return applet(Applet::Proxy),
            (Applet::Proxy, 0) => {
                let (_, event) = k.create_event().ok()?;
                let e = k.event_of(event)?;
                // The first message is the focus change InitializeApplet waits for.
                k.signal_event(e);
                return applet(Applet::CommonState { messages: VecDeque::from([FOCUS_STATE_CHANGED]), event: e });
            }
            (Applet::Proxy, 1) => return applet(Applet::SelfController),
            (Applet::Proxy, 2) => return applet(Applet::Window),
            (Applet::Proxy, 3) => return applet(Applet::Audio),
            (Applet::Proxy, 4) => return applet(Applet::Display),
            (Applet::Proxy, 10) => return applet(Applet::ProcessWinding),
            (Applet::Proxy, 11) => return applet(Applet::LibraryAppletCreator),
            (Applet::Proxy, 20) => return applet(Applet::Functions { popped: false }),
            (Applet::Proxy, 1000) => return applet(Applet::Debug),
            (Applet::Proxy, c @ (40 | 60)) => return applet(Applet::Unknown(c)),
            // GetEventHandle.
            (Applet::CommonState { event, .. }, 0) => Reply { copy_handles: vec![k.copy_event(*event)?], ..Reply::ok() },
            // ReceiveMessage.
            (Applet::CommonState { messages, event }, 1) => match messages.pop_front() {
                Some(m) => {
                    if messages.is_empty() {
                        k.events[*event].signaled = false;
                    }
                    Reply::with(&m.to_le_bytes())
                }
                None => Reply::error(NO_MESSAGE),
            },
            // GetOperationMode: handheld (0); GetPerformanceMode: normal (0), as Lichii,
            // which reaches the title screen so.
            (Applet::CommonState { .. }, 5) => Reply::with(&[0]),
            (Applet::CommonState { .. }, 6) => Reply::with(&0u32.to_le_bytes()),
            // GetCurrentFocusState.
            (Applet::CommonState { .. }, 9) => Reply::with(&[IN_FOCUS]),
            // SetCpuBoostMode, SetRequestExitToLibraryAppletAtExecuteNextProgramEnabled.
            // 650 and 651 (Switch 2, around the camera): Lichii's answer, success.
            (Applet::CommonState { .. }, 66 | 650 | 651 | 900) => Reply::ok(),
            // SetScreenShotPermission, SetOperationModeChangedNotification,
            // SetPerformanceModeChangedNotification, SetFocusHandlingMode,
            // SetOutOfFocusSuspendingEnabled, Switch 2 2400.
            // LockExit / UnlockExit (around a save), and the earlier settings.
            (Applet::SelfController, 1 | 2 | 10..=13 | 16 | 19 | 100 | 2400) => Reply::ok(),
            // GetGraphicsSupervisor: a session of its own, even from a domain, as a copied handle
            // (the client finds no moved one).
            (Applet::SelfController, 2000) => match k.open_session(Service::Gs(crate::gs::Gs::Supervisor)) {
                Ok(h) => Reply { copy_handles: vec![h], ..Reply::ok() },
                Err(e) => Reply::error(e),
            },
            // GetAccumulatedSuspendedTickChangedEvent.
            (Applet::SelfController, 91) => Reply { copy_handles: vec![k.new_event_handle()?], ..Reply::ok() },
            // GetAppletResourceUserId, AcquireForegroundRights.
            (Applet::Window, 1) => Reply::with(&ARUID.to_le_bytes()),
            (Applet::Window, 10) => Reply::ok(),
            // PopLaunchParameter: the preselected user (kind 2), once; nothing else.
            (Applet::Functions { popped }, 1) => {
                if req.u32_at(0) != Some(2) || *popped {
                    return Some(Reply::error(NO_DATA_IN_CHANNEL));
                }
                *popped = true;
                let mut p = vec![0u8; 0x88];
                p[0..4].copy_from_slice(&PRESELECTED_USER_MAGIC.to_le_bytes());
                p[4] = 1;
                p[8..24].copy_from_slice(&crate::account::user_id(k));
                return applet(Applet::Storage(p));
            }
            // GetDesiredLanguage: the console's language code, en-US unless --language.
            (Applet::Functions { .. }, 21) => {
                let code = k.config.language.unwrap_or("en-US");
                if code != "en-US" {
                    k.log(&format!("am: desired language {code} (--language)"));
                }
                let mut bytes = [0u8; 8];
                bytes[..code.len()].copy_from_slice(code.as_bytes());
                Reply::with(&bytes)
            }
            // IStorage Open; IStorageAccessor GetSize, Read(offset) into the B buffer.
            (Applet::Storage(bytes), 0) => return applet(Applet::StorageAccessor(bytes.clone())),
            (Applet::StorageAccessor(bytes), 0) => Reply::with(&(bytes.len() as u64).to_le_bytes()),
            (Applet::StorageAccessor(bytes), 11) => {
                let offset = req.u64_at(0)? as usize;
                if let Some(b) = req.out_buffer(0) {
                    let chunk = bytes.get(offset..).unwrap_or(&[]);
                    let _ = k.mem.write(b.addr, &chunk[..chunk.len().min(b.size as usize)]);
                }
                Reply::ok()
            }
            // GetGpuErrorDetectedSystemEvent.
            (Applet::Functions { .. }, 130) => Reply { copy_handles: vec![k.new_event_handle()?], ..Reply::ok() },
            // IApplicationProxy 60 (Switch 2), a graphics scale controller:
            // SetGraphicsScaleChangedNotificationEnabled, GetCurrentGraphicsScale (Lichii's 10).
            (Applet::Unknown(60), 1050) => Reply::ok(),
            (Applet::Unknown(60), 10) => Reply::with(&10u32.to_le_bytes()),
            // EnsureSaveData: nothing more to allocate.
            (Applet::Functions { .. }, 20) => Reply::with(&0u64.to_le_bytes()),
            // NotifyRunning: not "becoming foreground" again.
            (Applet::Functions { .. }, 40) => Reply::with(&[0]),
            // GetFriendInvitationStorageChannelEvent: never signaled.
            (Applet::Functions { .. }, 140) => Reply { copy_handles: vec![k.lone_event()?], ..Reply::ok() },
            // 210 (Switch 2): an event, the operation mode refresh by Lichii's guess; never signaled.
            (Applet::Functions { .. }, 210) => Reply { copy_handles: vec![k.lone_event()?], ..Reply::ok() },
            // Switch 2 320 and 321: two booleans, both off.
            (Applet::Functions { .. }, 320 | 321) => Reply::with(&[0]),
            // GetPseudoDeviceId: a fixed id derived from the run's seed.
            (Applet::Functions { .. }, 50) => {
                let s = k.config.seed;
                Reply::with(&[s.to_le_bytes(), (!s).to_le_bytes()].concat())
            }
            _ => return None,
        })
    }
}

impl Kernel {
    /// The event a readable handle designates.
    pub fn event_of(&self, h: u32) -> Option<usize> {
        match self.handles.get(h)? {
            crate::thread::Object::ReadableEvent(e) => Some(e),
            _ => None,
        }
    }

    /// A new readable handle to event `e`, for a reply's copy handles.
    pub fn copy_event(&mut self, e: usize) -> Option<u32> {
        self.handles.add(crate::thread::Object::ReadableEvent(e))
    }

    /// A new event that nothing signals yet; returns its readable handle.
    pub fn new_event_handle(&mut self) -> Option<u32> {
        self.create_event().ok().map(|(_, r)| r)
    }
}
