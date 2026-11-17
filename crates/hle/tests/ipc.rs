//! IPC on hand-built messages: HIPC, TIPC (sm), CMIF (lm) and domains.

use gyroship_hle::journal::Journal;
use gyroship_hle::result::*;
use gyroship_hle::fs::{Fs, RomFs};
use gyroship_hle::service::{SM_NOT_ALLOWED, Service};
use gyroship_hle::{Config, Kernel, boot, layout};
use gyroship_runtime::Cpu;
use gyroship_runtime::mem::{Block, Memory, perm, state};

const TLS: u64 = layout::THREAD_LOCAL.0;
const DATA: u64 = layout::ALIAS.0;
const PC: u64 = 0x70_0000_0100;

fn kernel() -> (Kernel, Cpu) {
    let mem = Memory::new().unwrap();
    mem.map(TLS, 0x1000, Block::new(state::THREAD_LOCAL, perm::RW)).unwrap();
    mem.map(DATA, 0x1000, Block::new(state::NORMAL, perm::RW)).unwrap();
    let mut cpu = Cpu::new(mem.base());
    cpu.tpidrro = TLS;
    let mut k = Kernel::new(mem, Journal::memory(), Config::default());
    k.allowed_services = vec!["lm".to_string(), "fsp-srv".to_string()];
    (k, cpu)
}

/// Sends `words` from the TLS on `handle`; returns the SVC result and the reply words.
fn send(k: &mut Kernel, cpu: &mut Cpu, handle: u32, words: &[u32]) -> (u32, Vec<u32>) {
    let bytes: Vec<u8> = words.iter().flat_map(|w| w.to_le_bytes()).collect();
    k.mem.write(TLS, &[0; 0x100]).unwrap();
    k.mem.write(TLS, &bytes).unwrap();
    cpu.x[0] = u64::from(handle);
    k.svc(cpu, PC, 0x21);
    let mut out = [0; 0x100];
    k.mem.read(TLS, &mut out).unwrap();
    (cpu.x[0] as u32, out.chunks(4).map(|c| u32::from_le_bytes(c.try_into().unwrap())).collect())
}

fn connect_sm(k: &mut Kernel, cpu: &mut Cpu) -> u32 {
    k.mem.write(DATA, b"sm:\0").unwrap();
    cpu.x[1] = DATA;
    k.svc(cpu, PC, 0x1F);
    assert_eq!(cpu.x[0], 0, "ConnectToNamedPort");
    cpu.x[1] as u32
}

fn packed(name: &str) -> [u32; 2] {
    let mut b = [0u8; 8];
    b[..name.len()].copy_from_slice(name.as_bytes());
    [u32::from_le_bytes(b[0..4].try_into().unwrap()), u32::from_le_bytes(b[4..8].try_into().unwrap())]
}

/// sm GetServiceHandle: (reply result, moved handle).
fn get_service(k: &mut Kernel, cpu: &mut Cpu, sm: u32, name: &str) -> (u32, u32) {
    let n = packed(name);
    let (r, w) = send(k, cpu, sm, &[16 + 1, 2, n[0], n[1]]);
    assert_eq!(r, SUCCESS);
    if w[1] & 0x8000_0000 != 0 {
        // Special header with one moved handle; the result follows it (offset 0x10).
        assert_eq!(w[2], 0x20);
        (w[4], w[3])
    } else {
        (w[2], 0)
    }
}

/// A CMIF request (type 4) for `command` with `args`, its data aligned after `header` words.
fn cmif(header: &[u32], command: u32, args: &[u32], domain: Option<(u32, u32)>) -> Vec<u32> {
    let mut words = header.to_vec();
    // Words before the data: the header and the size word inserted below.
    let before = words.len() + 1;
    // Padding up to a 16-byte boundary, measured from the start of the buffer.
    let pad = (4 - before % 4) % 4;
    let mut raw = vec![0u32; pad];
    let payload = [&[u32::from_le_bytes(*b"SFCI"), 1, command, 0][..], args].concat();
    if let Some((object, count)) = domain {
        raw.extend([1 | count << 8 | (4 * payload.len() as u32) << 16, object, 0, 0]);
    }
    raw.extend(payload);
    words.insert(1, raw.len() as u32);
    words.extend(raw);
    words
}

/// The CMIF reply's result: the SFCO block at the first 16-byte boundary after `skip` words.
fn cmif_result(w: &[u32], skip: usize) -> u32 {
    let at = skip.next_multiple_of(4);
    assert_eq!(w[at], u32::from_le_bytes(*b"SFCO"), "{w:x?}");
    w[at + 2]
}

#[test]
fn sm_registers_and_grants_what_the_npdm_allows() {
    let (mut k, mut cpu) = kernel();
    let sm = connect_sm(&mut k, &mut cpu);
    // RegisterClient, with the PID in a special header.
    let (r, w) = send(&mut k, &mut cpu, sm, &[16, 0x8000_0002, 1, 0, 0, 0, 0]);
    assert_eq!((r, w[0], w[1], w[2]), (SUCCESS, 16, 1, SUCCESS));
    let (result, lm) = get_service(&mut k, &mut cpu, sm, "lm");
    assert_eq!(result, SUCCESS);
    assert_ne!(lm, 0);
    assert_eq!(get_service(&mut k, &mut cpu, sm, "nvdrv"), (SM_NOT_ALLOWED, 0));
    assert_eq!(SM_NOT_ALLOWED, 0x1015);
    let text = k.journal.text();
    assert!(text.contains("sm: \"nvdrv\" is not in the NPDM's service list"), "{text}");
}

#[test]
fn lm_logs_the_game_s_lines() {
    let (mut k, mut cpu) = kernel();
    let sm = connect_sm(&mut k, &mut cpu);
    let (_, lm) = get_service(&mut k, &mut cpu, sm, "lm");
    // OpenLogger with a PID: [type, size, special, pid, pid].
    let (r, w) = send(&mut k, &mut cpu, lm, &cmif(&[4, 1, 0, 0], 0, &[0, 0], None).iter().enumerate().map(|(i, &x)| if i == 1 { x | 0x8000_0000 } else { x }).collect::<Vec<_>>());
    assert_eq!(r, SUCCESS);
    assert_eq!((w[1] >> 31, w[2]), (1, 0x20), "one moved handle: {w:x?}");
    let logger = w[3];
    assert_eq!(cmif_result(&w, 4), SUCCESS);
    // A log packet: a 0x18-byte header, then a text chunk (key 2).
    let mut packet = vec![0u8; 0x18];
    packet.extend([2, 13]);
    packet.extend(b"Sim startup\n\0");
    k.mem.write(DATA + 0x200, &packet).unwrap();
    let (addr, size) = (DATA + 0x200, packet.len() as u64);
    let x = [(size << 16 | (addr >> 36 & 7) << 6 | (addr >> 32 & 0xF) << 12) as u32, addr as u32];
    let (r, w) = send(&mut k, &mut cpu, logger, &cmif(&[4 | 1 << 16, x[0], x[1]], 0, &[], None));
    assert_eq!((r, cmif_result(&w, 2)), (SUCCESS, SUCCESS));
    let text = k.journal.text();
    assert!(text.contains(" game: Sim startup\n"), "{text}");
}

#[test]
fn domains_hand_out_object_ids() {
    let (mut k, mut cpu) = kernel();
    let sm = connect_sm(&mut k, &mut cpu);
    let (_, lm) = get_service(&mut k, &mut cpu, sm, "lm");
    // ConvertCurrentObjectToDomain (control, type 5): the session's object becomes object 1.
    let mut control = cmif(&[5], 0, &[], None);
    control[0] = 5;
    let (r, w) = send(&mut k, &mut cpu, lm, &control);
    assert_eq!((r, cmif_result(&w, 2), w[8]), (SUCCESS, SUCCESS, 1), "{w:x?}");
    // OpenLogger on object 1: the new ILogger is object 2, after the outputs.
    let (r, w) = send(&mut k, &mut cpu, lm, &cmif(&[4], 0, &[0, 0], Some((1, 0))));
    assert_eq!(r, SUCCESS);
    assert_eq!(w[4], 1, "one output object: {w:x?}");
    assert_eq!(cmif_result(&w, 8), SUCCESS);
    assert_eq!(w[12], 2, "its id: {w:x?}");
}

#[test]
fn a_closed_session_answers_session_closed() {
    let (mut k, mut cpu) = kernel();
    let sm = connect_sm(&mut k, &mut cpu);
    // TIPC close (type 15).
    let (r, _) = send(&mut k, &mut cpu, sm, &[15, 0]);
    assert_eq!(r, SUCCESS);
    let (r, _) = send(&mut k, &mut cpu, sm, &[16, 0]);
    assert_eq!(r, SESSION_CLOSED);
    assert_eq!(SESSION_CLOSED, 0xF601);
}

#[test]
fn the_npdm_s_service_list() {
    let mut npdm = vec![0u8; 0x200];
    npdm[0..4].copy_from_slice(b"META");
    npdm[0x70..0x74].copy_from_slice(&0x100u32.to_le_bytes());
    npdm[0x100..0x104].copy_from_slice(b"ACI0");
    npdm[0x128..0x12C].copy_from_slice(&0x40u32.to_le_bytes());
    // "lm" (client), "srv" (server, left out), "fsp-srv" (client).
    let sac = [&[0x01u8][..], b"lm", &[0x82], b"srv", &[0x06], b"fsp-srv"].concat();
    npdm[0x12C..0x130].copy_from_slice(&(sac.len() as u32).to_le_bytes());
    npdm[0x140..0x140 + sac.len()].copy_from_slice(&sac);
    assert_eq!(boot::npdm_services(&npdm), Some(vec!["lm".to_string(), "fsp-srv".to_string()]));
}

/// A CMIF reply's single moved handle (a new sub-interface session).
fn moved(w: &[u32]) -> u32 {
    assert_eq!((w[1] >> 31, w[2]), (1, 0x20), "one moved handle: {w:x?}");
    w[3]
}

#[test]
fn the_applet_says_it_has_the_focus() {
    let (mut k, mut cpu) = kernel();
    k.allowed_services.push("appletOE".to_string());
    let sm = connect_sm(&mut k, &mut cpu);
    let (_, oe) = get_service(&mut k, &mut cpu, sm, "appletOE");
    let (_, w) = send(&mut k, &mut cpu, oe, &cmif(&[4], 10, &[0, 0], None));
    let proxy = moved(&w);
    let (_, w) = send(&mut k, &mut cpu, proxy, &cmif(&[4], 0, &[], None));
    let state = moved(&w);
    // GetEventHandle: a copied handle to a signaled event.
    let (_, w) = send(&mut k, &mut cpu, state, &cmif(&[4], 0, &[], None));
    assert_eq!((w[1] >> 31, w[2]), (1, 0x2), "one copied handle: {w:x?}");
    let e = k.event_of(w[3]).expect("a readable event");
    assert!(k.events[e].signaled);
    // ReceiveMessage: the focus change, then nothing.
    let (_, w) = send(&mut k, &mut cpu, state, &cmif(&[4], 1, &[], None));
    assert_eq!((cmif_result(&w, 2), w[8]), (SUCCESS, 15), "{w:x?}");
    let (_, w) = send(&mut k, &mut cpu, state, &cmif(&[4], 1, &[], None));
    assert_eq!(cmif_result(&w, 2), gyroship_hle::am::NO_MESSAGE);
    // GetCurrentFocusState: in focus.
    let (_, w) = send(&mut k, &mut cpu, state, &cmif(&[4], 9, &[], None));
    assert_eq!(w[8] & 0xFF, 1);
}

#[test]
fn the_desired_language_is_en_us_unless_the_language_option_says_otherwise() {
    for (language, want) in [(None, *b"en-US\0\0\0"), (Some("zh-Hans"), *b"zh-Hans\0")] {
        let (mut k, mut cpu) = kernel();
        k.config.language = language;
        k.allowed_services.push("appletOE".to_string());
        let sm = connect_sm(&mut k, &mut cpu);
        let (_, oe) = get_service(&mut k, &mut cpu, sm, "appletOE");
        let (_, w) = send(&mut k, &mut cpu, oe, &cmif(&[4], 10, &[0, 0], None));
        let proxy = moved(&w);
        // GetApplicationFunctions, then GetDesiredLanguage: the code in the data words.
        let (_, w) = send(&mut k, &mut cpu, proxy, &cmif(&[4], 20, &[], None));
        let functions = moved(&w);
        let (_, w) = send(&mut k, &mut cpu, functions, &cmif(&[4], 21, &[], None));
        assert_eq!(cmif_result(&w, 2), SUCCESS);
        let code: Vec<u8> = w[8..10].iter().flat_map(|x| x.to_le_bytes()).collect();
        assert_eq!(code, want, "{language:?}");
    }
}

#[test]
fn time_s_shared_memory_has_a_steady_clock_source() {
    let (mut k, mut cpu) = kernel();
    k.allowed_services.push("time:u".to_string());
    let sm = connect_sm(&mut k, &mut cpu);
    let (_, time) = get_service(&mut k, &mut cpu, sm, "time:u");
    let (_, w) = send(&mut k, &mut cpu, time, &cmif(&[4], 20, &[], None));
    assert_eq!((w[1] >> 31, w[2]), (1, 0x2), "one copied handle: {w:x?}");
    let at = 0x50_0000_0000;
    (cpu.x[0], cpu.x[1], cpu.x[2], cpu.x[3]) = (u64::from(w[3]), at, 0x1000, 1);
    k.svc(&mut cpu, PC, 0x13);
    assert_eq!(cpu.x[0], 0, "MapSharedMemory");
    assert_eq!(k.mem.query(at).block, Block::new(state::SHARED, perm::R));
    // The continuous adjustment's source id, which the SDK requires to be non-zero.
    let mut id = [0; 16];
    k.mem.read(at + 0xD8 + 0x28, &mut id).unwrap();
    assert_ne!(id, [0; 16]);
}

#[test]
fn a_cloned_session_reaches_the_same_object() {
    let (mut k, mut cpu) = kernel();
    let sm = connect_sm(&mut k, &mut cpu);
    let (_, lm) = get_service(&mut k, &mut cpu, sm, "lm");
    // CloneCurrentObject (control 2): a moved handle to a second session.
    let mut control = cmif(&[5], 2, &[], None);
    control[0] = 5;
    let (r, w) = send(&mut k, &mut cpu, lm, &control);
    assert_eq!(r, SUCCESS);
    let clone = moved(&w);
    assert_ne!(clone, lm);
    let (_, w) = send(&mut k, &mut cpu, clone, &cmif(&[4], 0, &[0, 0], None));
    assert_ne!(moved(&w), 0, "OpenLogger through the clone");
    assert_eq!(k.objects.iter().filter(|o| matches!(o, gyroship_hle::service::Service::Lm)).count(), 1, "one lm object");
}

/// The NFC reader is there and never finds an amiibo: ListDevices gives one
/// handle, the activate event is never signaled, detection only searches.
#[test]
fn the_nfc_reader_is_listed_and_finds_no_tag() {
    let (mut k, mut cpu) = kernel();
    k.allowed_services.push("nfp:user".to_string());
    let sm = connect_sm(&mut k, &mut cpu);
    let (_, nfp) = get_service(&mut k, &mut cpu, sm, "nfp:user");
    let (_, w) = send(&mut k, &mut cpu, nfp, &cmif(&[4], 0, &[], None));
    let user = moved(&w);
    // ListDevices into a B buffer at DATA + 0x100.
    k.mem.write(DATA + 0x100, &[0xff; 8]).unwrap();
    let at = DATA + 0x100;
    let b = [8, at as u32, ((at >> 32 & 0xF) << 28 | (at >> 36 & 7) << 2) as u32];
    let (_, w) = send(&mut k, &mut cpu, user, &cmif(&[4 | 1 << 24, b[0], b[1], b[2]], 2, &[], None));
    assert_eq!((cmif_result(&w, 2), w[8]), (SUCCESS, 1), "{w:x?}");
    let mut handle = [0xffu8; 8];
    k.mem.read(at, &mut handle).unwrap();
    assert_eq!(handle, [0; 8]);
    // AttachActivateEvent: an event never signaled.
    let (_, w) = send(&mut k, &mut cpu, user, &cmif(&[4], 17, &[0, 0], None));
    assert_eq!((w[1] >> 31, w[2]), (1, 0x2), "one copied handle: {w:x?}");
    assert!(!k.events[k.event_of(w[3]).expect("a readable event")].signaled);
    // GetDeviceState: initialized, then searching once StartDetection ran.
    let state = |k: &mut Kernel, cpu: &mut Cpu| send(k, cpu, user, &cmif(&[4], 20, &[0, 0], None)).1[8];
    assert_eq!(state(&mut k, &mut cpu), 0);
    send(&mut k, &mut cpu, user, &cmif(&[4], 3, &[0, 0], None));
    assert_eq!(state(&mut k, &mut cpu), 1);
}

/// Offline (rule 9): a connection request fails, and nifm has no applet to show
/// for it (GetAppletInfo fails too). The SDK's HandleNetworkRequestErrorResult
/// then reports the network as still unavailable instead of launching a
/// library applet, which aborted the game in run 16.
#[test]
fn nifm_fails_requests_and_offers_no_applet() {
    let (mut k, mut cpu) = kernel();
    k.allowed_services.push("nifm:u".to_string());
    let sm = connect_sm(&mut k, &mut cpu);
    let (_, nifm) = get_service(&mut k, &mut cpu, sm, "nifm:u");
    let (_, w) = send(&mut k, &mut cpu, nifm, &cmif(&[4], 5, &[0, 0], None));
    let general = moved(&w);
    let (_, w) = send(&mut k, &mut cpu, general, &cmif(&[4], 4, &[2], None));
    let request = moved(&w);
    let no_network = gyroship_hle::account::NIFM_NO_NETWORK;
    // Submit, then GetResult and GetAppletInfo.
    send(&mut k, &mut cpu, request, &cmif(&[4], 4, &[], None));
    let (_, w) = send(&mut k, &mut cpu, request, &cmif(&[4], 1, &[], None));
    assert_eq!(cmif_result(&w, 2), no_network);
    let (_, w) = send(&mut k, &mut cpu, request, &cmif(&[4], 21, &[0], None));
    assert_eq!(cmif_result(&w, 2), no_network);
}

/// IStorage Read: the destination is checked and the read counted at once, the
/// bytes come with the deferred work, and a read over the message is an anomaly.
#[test]
fn a_romfs_read_is_checked_at_once_and_lands_after_the_svc() {
    let (mut k, mut cpu) = kernel();
    k.config.explore = true;
    let root = std::path::Path::new(env!("CARGO_TARGET_TMPDIR")).join("ipc-romfs");
    let _ = std::fs::remove_dir_all(&root);
    std::fs::create_dir_all(&root).unwrap();
    std::fs::write(root.join("c"), b"xyz").unwrap();
    k.romfs = Some(std::sync::Arc::new(RomFs::build(&root).unwrap()));
    let h = k.open_session(Service::Fs(Fs::RomStorage)).unwrap();
    // Read(0x200, 3), the file's bytes, into `to` (one B buffer); returns the CMIF result.
    let read = |k: &mut Kernel, cpu: &mut Cpu, to: u64| {
        let (r, w) = send(k, cpu, h, &cmif(&[4 | 1 << 24, 3, to as u32, ((to >> 36 & 7) << 2) as u32], 0, &[0x200, 0, 3, 0], None));
        assert_eq!(r, SUCCESS);
        cmif_result(&w, 2)
    };
    assert_eq!(read(&mut k, &mut cpu, DATA + 0x800), SUCCESS);
    let mut dest = [0; 3];
    k.mem.read(DATA + 0x800, &mut dest).unwrap();
    assert_eq!(dest, [0; 3], "the bytes come with the deferred work");
    assert_eq!(k.tally().counts["romfs reads"], 1);
    k.deferred.take().expect("a deferred read").run().unwrap();
    k.mem.read(DATA + 0x800, &mut dest).unwrap();
    assert_eq!(&dest, b"xyz");
    // Unmapped: refused at once, nothing deferred.
    assert_eq!(read(&mut k, &mut cpu, DATA + 0x1000), INVALID_POINTER);
    assert!(k.deferred.is_none());
    // Over the message (the TLS): the reply would no longer win there.
    assert_eq!(read(&mut k, &mut cpu, TLS + 0x80), SUCCESS);
    assert_eq!(k.tally().anomalies["romfs read buffer"], 2);
}

/// The Joy-Con mouse: the game lends a buffer of lifos (InitializeJcms), then starts the two
/// sides of a controller (StartJcms). Each started lifo holds one null state from then on,
/// so the game, which reads the newest state whatever their count, reads "no mouse" and not
/// what its own stack held.
#[test]
fn a_started_joy_con_mouse_has_a_null_state_to_read() {
    let (mut k, mut cpu) = kernel();
    k.allowed_services.push("hid".to_string());
    let sm = connect_sm(&mut k, &mut cpu);
    let (_, hid) = get_service(&mut k, &mut cpu, sm, "hid");
    // The game's buffer: one page, lent as a transfer memory the game may still read.
    k.mem.write(DATA, &[0xff; 0x1000]).unwrap();
    (cpu.x[1], cpu.x[2], cpu.x[3]) = (DATA, 0x1000, u64::from(perm::R));
    k.svc(&mut cpu, PC, 0x15);
    assert_eq!(cpu.x[0], 0, "CreateTransferMemory");
    let lent = cpu.x[1] as u32;
    // InitializeJcms: the PID, one copied handle, the applet resource user id and the size.
    let init: Vec<u32> = cmif(&[4, 1 | 1 << 1, 0, 0, lent], 3110, &[0x51, 0, 0x1000, 0], None).iter().enumerate().map(|(i, &x)| if i == 1 { x | 0x8000_0000 } else { x }).collect();
    let (_, w) = send(&mut k, &mut cpu, hid, &init);
    assert_eq!(cmif_result(&w, 2), SUCCESS);
    let count = |k: &Kernel, lifo: u64| {
        let mut b = [0; 8];
        k.mem.read(DATA + lifo * 0x2c8 + 0x18, &mut b).unwrap();
        u64::from_le_bytes(b)
    };
    assert_eq!([count(&k, 0), count(&k, 1), count(&k, 5)], [0; 3], "the buffer starts empty");
    // StartJcms for Npad 0, side 0 then side 1 (the handle: id, then side), then for a lifo
    // that ends past the buffer (Npad 2, side 1): nothing written, no error.
    for (handle, counts) in [(0x000, [1, 0, 0]), (0x100, [1, 1, 0]), (0x102, [1, 1, 0])] {
        let (_, w) = send(&mut k, &mut cpu, hid, &cmif(&[4], 3112, &[handle, 0, 0x51, 0], None));
        assert_eq!(cmif_result(&w, 2), SUCCESS);
        assert_eq!([count(&k, 0), count(&k, 1), count(&k, 5)], counts, "after handle {handle:#x}");
    }
    // The state itself: sequence, sampling number, counts and attributes all zero.
    let mut state = [0xff; 0x28];
    k.mem.read(DATA + 0x2c8 + 0x20, &mut state).unwrap();
    assert_eq!(state, [0; 0x28]);
    // A lifo that already holds states keeps them when it is started again.
    k.mem.write_object(DATA + 0x18, &5u64.to_le_bytes()).unwrap();
    send(&mut k, &mut cpu, hid, &cmif(&[4], 3112, &[0x000, 0, 0x51, 0], None));
    assert_eq!(count(&k, 0), 5);
    // FinalizeJcms: the buffer is the game's again, a later start writes nothing.
    send(&mut k, &mut cpu, hid, &cmif(&[4], 3111, &[0x51, 0], None));
    send(&mut k, &mut cpu, hid, &cmif(&[4], 3112, &[0x001, 0, 0x51, 0], None));
    assert_eq!(count(&k, 2), 0);
}

/// There is no system applet here. The controller screen (`nn::hid::ShowControllerSupport`,
/// library applet 0x0C) is created, takes an argument and starts, and has already closed,
/// cancelled: the one answer the SDK hands back to the game, as any refusal on the way makes
/// it abort.
#[test]
fn the_controller_screen_closes_at_once_cancelled() {
    let (mut k, mut cpu) = kernel();
    k.config.explore = true;
    k.allowed_services.push("appletOE".to_string());
    let sm = connect_sm(&mut k, &mut cpu);
    let (_, oe) = get_service(&mut k, &mut cpu, sm, "appletOE");
    let (_, w) = send(&mut k, &mut cpu, oe, &cmif(&[4], 10, &[0, 0], None));
    let proxy = moved(&w);
    let signaled = |k: &Kernel, w: &[u32]| {
        assert_eq!((w[1] >> 31, w[2]), (1, 0x2), "one copied handle: {w:x?}");
        k.events[k.event_of(w[3]).expect("a readable event")].signaled
    };
    // GetSelfController, GetLibraryAppletLaunchableEvent: the SDK waits on it before anything.
    let (_, w) = send(&mut k, &mut cpu, proxy, &cmif(&[4], 1, &[], None));
    let controller = moved(&w);
    let (_, w) = send(&mut k, &mut cpu, controller, &cmif(&[4], 9, &[], None));
    assert!(signaled(&k, &w));
    // GetLibraryAppletCreator; another applet (the keyboard, 0x11) is not answered.
    let (_, w) = send(&mut k, &mut cpu, proxy, &cmif(&[4], 11, &[], None));
    let creator = moved(&w);
    let (_, w) = send(&mut k, &mut cpu, creator, &cmif(&[4], 3, &[0x11, 0, 7, 0], None));
    assert_eq!(cmif_result(&w, 2), NOT_IMPLEMENTED);
    // CreateLibraryAppletEx(the controller screen, mode 0, a thread id).
    let (_, w) = send(&mut k, &mut cpu, creator, &cmif(&[4], 3, &[0x100_000C, 0, 7, 0], None));
    let screen = moved(&w);
    // One argument: CreateStorage(0x14), Open, Write(0) from an A buffer, PushInData.
    let (_, w) = send(&mut k, &mut cpu, creator, &cmif(&[4], 10, &[0x14, 0], None));
    let storage = moved(&w);
    let (_, w) = send(&mut k, &mut cpu, storage, &cmif(&[4], 0, &[], None));
    let accessor = moved(&w);
    let at = DATA + 0x300;
    k.mem.write(at, &[0x14, 0, 0, 0, 0x30, 4, 0, 0, 1, 0, 0, 0, 0, 0x10, 0, 0, 1, 0, 0, 0]).unwrap();
    let a = [0x14, at as u32, ((at >> 32 & 0xF) << 28 | (at >> 36 & 7) << 2) as u32];
    let (_, w) = send(&mut k, &mut cpu, accessor, &cmif(&[4 | 1 << 20, a[0], a[1], a[2]], 10, &[0, 0], None));
    assert_eq!(cmif_result(&w, 2), SUCCESS);
    let (_, w) = send(&mut k, &mut cpu, screen, &cmif(&[4], 100, &[], None));
    assert_eq!(cmif_result(&w, 2), SUCCESS);
    // Start; the state event is signaled (the screen has closed) and GetResult says cancelled.
    let (_, w) = send(&mut k, &mut cpu, screen, &cmif(&[4], 10, &[], None));
    assert_eq!(cmif_result(&w, 2), SUCCESS);
    let (_, w) = send(&mut k, &mut cpu, screen, &cmif(&[4], 0, &[], None));
    assert!(signaled(&k, &w));
    let (_, w) = send(&mut k, &mut cpu, screen, &cmif(&[4], 30, &[], None));
    assert_eq!(cmif_result(&w, 2), gyroship_hle::am::CANCELLED);
    assert_eq!(gyroship_hle::am::CANCELLED, 0x2C80, "module 128, description 22");
    assert_eq!(k.tally().counts["controller screens cancelled"], 1);
    let text = k.journal.text();
    assert!(text.contains("am: the controller screen is asked (library applet 0x100000c, mode 0): none here, it closes at once, cancelled"), "{text}");
    assert!(text.contains("am: applet argument of 0x14 bytes: 1400000030040000010000000010000001000000"), "{text}");
    assert!(text.contains("ipc ILibraryAppletCreator command 3 is not implemented; arguments 11000000"), "{text}");
}
