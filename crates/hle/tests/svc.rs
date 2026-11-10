//! The SVC framework on a synthetic address space.

use gyroship_hle::journal::Journal;
use gyroship_hle::result::*;
use gyroship_hle::svc::CURRENT_PROCESS;
use gyroship_hle::{Config, Kernel, STOP_EXIT, layout};
use gyroship_runtime::mem::{Block, Info, Memory, attr, perm, state};
use gyroship_runtime::{self as rt, Cpu};

const CODE: u64 = 0x70_0000_0000;
const PC: u64 = CODE + 0x100;
/// A stack page, for out-pointers.
const STACK: u64 = layout::STACK.0;

fn kernel(config: Config) -> (Kernel, Cpu) {
    let mem = Memory::new().unwrap();
    mem.map(CODE, 0x4000, Block::new(state::CODE, perm::RX)).unwrap();
    mem.map(STACK, 0x1000, Block::new(state::STACK, perm::RW)).unwrap();
    let cpu = Cpu::new(mem.base());
    (Kernel::new(mem, Journal::memory(), config), cpu)
}

/// Calls `svc #imm` with the given x0.. arguments; returns w0.
fn svc(k: &mut Kernel, cpu: &mut Cpu, imm: u32, args: &[u64]) -> u32 {
    cpu.x[..args.len()].copy_from_slice(args);
    k.svc(cpu, PC, imm);
    cpu.x[0] as u32
}

#[test]
fn query_memory_writes_a_memory_info() {
    let (mut k, mut cpu) = kernel(Config::default());
    assert_eq!(svc(&mut k, &mut cpu, 0x06, &[STACK, 0, CODE + 0x10]), SUCCESS);
    assert_eq!(cpu.x[1], 0, "page info");
    let mut out = [0; 0x28];
    k.mem.read(STACK, &mut out).unwrap();
    assert_eq!(out, Info { base: CODE, size: 0x4000, block: Block::new(state::CODE, perm::RX) }.to_bytes());
    // Past the address space: the Inaccessible block that ends rtld's walk.
    svc(&mut k, &mut cpu, 0x06, &[STACK, 0, 1 << 39]);
    k.mem.read(STACK, &mut out).unwrap();
    assert_eq!(out[16..20], state::INACCESSIBLE.to_le_bytes());
    assert_eq!(svc(&mut k, &mut cpu, 0x06, &[CODE, 0, 0]), INVALID_POINTER, "out-pointer in read-only code");
}

#[test]
fn get_info_reports_the_platform() {
    let (mut k, mut cpu) = kernel(Config::default());
    let mut info = |k: &mut Kernel, kind: u64, handle: u64, sub: u64| (svc(k, &mut cpu, 0x29, &[0, kind, handle, sub]), cpu.x[1]);
    assert_eq!(info(&mut k, 0x21, CURRENT_PROCESS, 0), (SUCCESS, 0x1000));
    assert_eq!(info(&mut k, 0x10, CURRENT_PROCESS, 0), (SUCCESS, 0x100_0000));
    assert_eq!(info(&mut k, 0x02, CURRENT_PROCESS, 0), (SUCCESS, layout::ALIAS.0));
    assert_eq!(info(&mut k, 0x16, CURRENT_PROCESS, 0), (SUCCESS, 0x5000), "the mapped pages");
    assert_eq!(info(&mut k, 0x00, 0, 0).0, INVALID_HANDLE);
    assert_eq!(info(&mut k, 0x99, CURRENT_PROCESS, 0).0, INVALID_ENUM_VALUE);
    let (r2, e2) = info(&mut k, 0x0B, 0, 2);
    let (_, e3) = info(&mut k, 0x0B, 0, 3);
    assert_eq!(r2, SUCCESS);
    assert_ne!(e2, e3);
    // The same seed gives the same entropy.
    let (mut k2, mut cpu2) = kernel(Config::default());
    assert_eq!(svc(&mut k2, &mut cpu2, 0x29, &[0, 0x0B, 0, 2]), SUCCESS);
    assert_eq!(cpu2.x[1], e2);
}

#[test]
fn physical_memory_backs_the_alias_region() {
    let (mut k, mut cpu) = kernel(Config::default());
    let alias = layout::ALIAS.0;
    assert_eq!(svc(&mut k, &mut cpu, 0x2C, &[alias + 0x1000, 0x1000]), SUCCESS);
    // A wider range maps only what is still unmapped.
    assert_eq!(svc(&mut k, &mut cpu, 0x2C, &[alias, 0x3000]), SUCCESS);
    let q = k.mem.query(alias);
    assert_eq!((q.base, q.size, q.block), (alias, 0x3000, Block::new(state::NORMAL, perm::RW)));
    k.mem.write(alias + 0x2ff8, &[1; 8]).unwrap();
    assert_eq!(svc(&mut k, &mut cpu, 0x2C, &[STACK, 0x1000]), INVALID_MEMORY_REGION);
    assert_eq!(svc(&mut k, &mut cpu, 0x2C, &[alias + 0x10, 0x1000]), INVALID_ADDRESS);
    assert_eq!(svc(&mut k, &mut cpu, 0x2D, &[alias, 0x2000]), SUCCESS);
    assert_eq!(k.mem.query(alias).block.state, state::FREE);
    assert_eq!(k.mem.query(alias + 0x2000).block.state, state::NORMAL);
}

#[test]
fn map_memory_moves_to_the_stack_region() {
    let (mut k, mut cpu) = kernel(Config::default());
    let (alias, dst) = (layout::ALIAS.0, layout::STACK.0 + 0x10_0000);
    svc(&mut k, &mut cpu, 0x2C, &[alias, 0x2000]);
    k.mem.write(alias, b"thread stack").unwrap();
    assert_eq!(svc(&mut k, &mut cpu, 0x04, &[dst, alias, 0x2000]), SUCCESS);
    assert_eq!(k.mem.query(alias).block.attr, attr::LOCKED);
    let mut b = [0; 12];
    k.mem.read(dst, &mut b).unwrap();
    assert_eq!(&b, b"thread stack");
    assert_eq!(svc(&mut k, &mut cpu, 0x04, &[alias + 0x4000, alias, 0x2000]), INVALID_MEMORY_REGION);
    assert_eq!(svc(&mut k, &mut cpu, 0x05, &[dst, alias, 0x2000]), SUCCESS);
    assert_eq!(k.mem.query(dst).block.state, state::FREE);
}

#[test]
fn permissions_and_attributes() {
    let (mut k, mut cpu) = kernel(Config::default());
    // RELRO: read-only, then permission-locked, as nn::ro::ProtectRelro does.
    assert_eq!(svc(&mut k, &mut cpu, 0x02, &[CODE + 0x1000, 0x1000, u64::from(perm::R)]), SUCCESS);
    assert_eq!(svc(&mut k, &mut cpu, 0x03, &[CODE + 0x1000, 0x1000, 0x10, 0x10]), SUCCESS);
    assert_eq!(k.mem.query(CODE + 0x1000).block, Block { state: state::CODE, perm: perm::R, attr: attr::PERMISSION_LOCKED });
    assert_eq!(svc(&mut k, &mut cpu, 0x02, &[CODE, 0x1000, u64::from(perm::X)]), INVALID_NEW_MEMORY_PERMISSION);
    assert_eq!(svc(&mut k, &mut cpu, 0x03, &[CODE, 0x1000, 0x1, 0x1]), INVALID_COMBINATION, "Locked is the kernel's");
    // Switch 2: the GPU-sharable bit, and only it.
    assert_eq!(svc(&mut k, &mut cpu, 0x80, &[STACK, 0x1000, 0x20, 0x20]), SUCCESS);
    assert_eq!(k.mem.query(STACK).block.attr, attr::GPU_SHARABLE);
    assert_eq!(svc(&mut k, &mut cpu, 0x80, &[STACK, 0x1000, 0x10, 0]), INVALID_COMBINATION);
}

#[test]
fn notifications_and_debug_strings_are_logged() {
    let (mut k, mut cpu) = kernel(Config::default());
    assert_eq!(svc(&mut k, &mut cpu, 0x26, &[0x8000_0007, 0, 0]), SUCCESS);
    k.mem.write(STACK, b"hello from the guest\n").unwrap();
    assert_eq!(svc(&mut k, &mut cpu, 0x27, &[STACK, 21]), SUCCESS);
    let text = k.journal.text();
    assert!(text.contains("svc 0x26 Break notification 0x80000007"), "{text}");
    assert!(text.contains("svc 0x27 OutputDebugString \"hello from the guest\""), "{text}");
}

#[test]
fn exploration_mode_answers_not_implemented() {
    let (mut k, mut cpu) = kernel(Config { explore: true, ..Config::default() });
    assert_eq!(svc(&mut k, &mut cpu, 0x99, &[1, 2]), NOT_IMPLEMENTED);
    let text = k.journal.text();
    assert!(text.contains("mode exploration (not a validation run)"), "{text}");
    assert!(text.contains("svc 0x99 ? at 0x7000000100 is not implemented; x0=0x1 x1=0x2"), "{text}");
}

/// The same calls give the same journal.
#[test]
fn the_journal_is_deterministic() {
    let run = || {
        let (mut k, mut cpu) = kernel(Config::default());
        for (imm, a) in [(0x29, [0, 0x0B, 0, 1]), (0x2C, [layout::ALIAS.0, 0x1000, 0, 0]), (0x06, [STACK, 0, layout::ALIAS.0, 0]), (0x29, [0, 0x99, CURRENT_PROCESS, 0])] {
            svc(&mut k, &mut cpu, imm, &a);
        }
        k.journal.text().to_string()
    };
    let first = run();
    assert_eq!(first.lines().count(), 5, "{first}");
    assert_eq!(first, run());
}

#[test]
fn guest_code_reaches_the_kernel_through_the_runtime() {
    let (k, mut cpu) = kernel(Config::default());
    gyroship_hle::install(k);
    (cpu.x[1], cpu.x[2]) = (0x21, CURRENT_PROCESS);
    // SAFETY: a valid context.
    unsafe { rt::gyroship_svc(&mut cpu, PC, 0x29) };
    assert_eq!((cpu.x[0], cpu.x[1]), (0, 0x1000));
    gyroship_hle::uninstall();
}

/// One kernel, several host threads: `with` lets one in at a time (no count
/// is lost), and a thread that did not attach has no kernel.
#[test]
fn host_threads_share_one_kernel() {
    let (k, _) = kernel(Config::default());
    let shared = gyroship_hle::install(k);
    let workers: Vec<_> = (0..4)
        .map(|_| {
            std::thread::spawn(move || {
                gyroship_hle::attach(shared);
                for _ in 0..10_000 {
                    gyroship_hle::with(|k| k.count("lock test"));
                }
            })
        })
        .collect();
    for w in workers {
        w.join().unwrap();
    }
    assert!(std::thread::spawn(|| gyroship_hle::with(|_| ())).join().is_err(), "a thread that did not attach has no kernel");
    assert_eq!(gyroship_hle::uninstall().unwrap().counts["lock test"], 40_000);
}

/// Only does something when started by `a_fault_inside_the_kernel_exits`.
#[test]
fn kernel_fault_child() {
    if std::env::var("GYROSHIP_KERNEL_FAULT_CHILD").is_err() {
        return;
    }
    // A fault that waited for the kernel lock would hang: this ends the child instead.
    std::thread::spawn(|| {
        std::thread::sleep(std::time::Duration::from_secs(20));
        std::process::exit(99);
    });
    let (k, cpu) = kernel(Config::default());
    gyroship_runtime::mem::install_fault_handler();
    gyroship_hle::install(k);
    // HLE code faults: this thread holds the kernel lock (black_box: a release
    // build would drop a load whose value is unused).
    gyroship_hle::with(|_| {
        let _ = std::hint::black_box(rt::ld64(&cpu, 0x1234));
    });
}

#[test]
fn a_fault_inside_the_kernel_exits() {
    let out = std::process::Command::new(std::env::current_exe().unwrap())
        .args(["--exact", "kernel_fault_child", "--nocapture", "--test-threads=1"])
        .env("GYROSHIP_KERNEL_FAULT_CHILD", "1")
        .output()
        .unwrap();
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(out.status.code(), Some(gyroship_runtime::mem::FAULT_EXIT), "{stderr}");
    assert!(stderr.contains("guest fault: read at guest address 0x1234 "), "{stderr}");
}

/// Only does something when started by `a_fault_report_waits_for_another_threads_lock`.
#[test]
fn fault_wait_child() {
    let Ok(log) = std::env::var("GYROSHIP_FAULT_WAIT_CHILD") else { return };
    std::thread::spawn(|| {
        std::thread::sleep(std::time::Duration::from_secs(20));
        std::process::exit(99);
    });
    let (mut k, cpu) = kernel(Config::default());
    k.journal = Journal::file(std::path::Path::new(&log)).unwrap();
    let base = cpu.mem as usize;
    gyroship_runtime::mem::install_fault_handler();
    let shared = gyroship_hle::install(k);
    let barrier = std::sync::Arc::new(std::sync::Barrier::new(2));
    let b = barrier.clone();
    let faulter = std::thread::spawn(move || {
        gyroship_hle::attach(shared);
        let cpu = Cpu::new(base as *mut u8);
        b.wait();
        let _ = std::hint::black_box(rt::ld64(&cpu, 0x1234));
    });
    // This thread holds the kernel while the other one faults.
    gyroship_hle::with(|_| {
        barrier.wait();
        std::thread::sleep(std::time::Duration::from_millis(300));
    });
    let _ = faulter.join();
}

/// A guest fault while another host thread holds the kernel: the report
/// waits for the lock, then reaches the journal.
#[test]
fn a_fault_report_waits_for_another_threads_lock() {
    let log = std::path::Path::new(env!("CARGO_TARGET_TMPDIR")).join("fault-wait.log");
    let _ = std::fs::remove_file(&log);
    let out = std::process::Command::new(std::env::current_exe().unwrap())
        .args(["--exact", "fault_wait_child", "--nocapture", "--test-threads=1"])
        .env("GYROSHIP_FAULT_WAIT_CHILD", &log)
        .output()
        .unwrap();
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(out.status.code(), Some(gyroship_runtime::mem::FAULT_EXIT), "{stderr}");
    let journal = std::fs::read_to_string(&log).unwrap();
    assert_eq!(journal.matches("guest fault: read at guest address 0x1234 ").count(), 1, "{journal}");
    assert!(journal.contains("stack at the last SVC:"), "{journal}");
}

/// Only does something when started by `stops_are_logged`.
#[test]
fn stop_child() {
    let Ok(mode) = std::env::var("GYROSHIP_STOP_CHILD") else { return };
    let (mut k, mut cpu) = kernel(Config::default());
    match mode.as_str() {
        "forbidden" => svc(&mut k, &mut cpu, 0x40, &[]),
        "abort" => svc(&mut k, &mut cpu, 0x26, &[0, 0, 4]),
        "unknown" => svc(&mut k, &mut cpu, 0x99, &[7]),
        "anomaly" => {
            k.anomaly("gpu watchdog", "channel fd 3 waited two presents");
            0
        }
        _ => unreachable!(),
    };
}

/// Exploration counts every anomaly; the summary names them apart from the
/// activity. (Strict mode stops: `stops_are_logged`.)
#[test]
fn anomalies_are_counted_and_summarized() {
    let (mut k, _) = kernel(Config { explore: true, ..Config::default() });
    assert_eq!(k.summary(), "summary: 0 presents; activity: ; anomalies: none");
    k.anomaly("gpu release at 0", "method 0x1b0c releases 2 at 0x0");
    k.count("draws");
    k.count("draws");
    assert_eq!(k.summary(), "summary: 0 presents; activity: draws 2; anomalies: gpu release at 0 1");
    for _ in 0..6 {
        k.anomaly("gpu watchdog", "w");
    }
    assert_eq!(k.anomalies["gpu watchdog"], 6);
    assert_eq!(k.journal.text().matches("anomaly gpu watchdog: w").count(), 4, "journaled the first times only");
}

#[test]
fn stops_are_logged() {
    for (mode, line) in [
        ("forbidden", "forbidden svc 0x40 CreateSession at 0x7000000100: the KAC does not grant it"),
        ("abort", "svc 0x26 Break at 0x7000000100: the guest aborts, reason 0x0 info 0x0+0x4"),
        ("unknown", "svc 0x99 ? at 0x7000000100 is not implemented; x0=0x7"),
        ("anomaly", "anomaly gpu watchdog: channel fd 3 waited two presents"),
    ] {
        let out = std::process::Command::new(std::env::current_exe().unwrap())
            .args(["--exact", "stop_child", "--nocapture", "--test-threads=1"])
            .env("GYROSHIP_STOP_CHILD", mode)
            .output()
            .unwrap();
        let stderr = String::from_utf8_lossy(&out.stderr);
        assert_eq!(out.status.code(), Some(STOP_EXIT), "{mode}: {stderr}");
        assert!(stderr.contains(line), "{mode}: {stderr}");
    }
}

#[test]
fn transfer_memory_locks_until_its_last_handle_closes() {
    let (mut k, mut cpu) = kernel(Config::default());
    let alias = layout::ALIAS.0;
    svc(&mut k, &mut cpu, 0x2C, &[alias, 0x2000]);
    assert_eq!(svc(&mut k, &mut cpu, 0x15, &[0, alias, 0x2000, u64::from(perm::R)]), SUCCESS);
    let h = cpu.x[1];
    assert_eq!(k.mem.query(alias).block, Block { state: state::NORMAL, perm: perm::R, attr: attr::LOCKED });
    // A service that keeps it holds a reference of its own.
    let t = k.retain_transfer(h as u32).unwrap();
    assert_eq!(svc(&mut k, &mut cpu, 0x16, &[h]), SUCCESS);
    assert_eq!(k.mem.query(alias).block.attr, attr::LOCKED, "still held by the service");
    k.release_transfer(t);
    assert_eq!(k.mem.query(alias).block, Block::new(state::NORMAL, perm::RW));
    // Memory already lent cannot be lent again, nor can unmapped memory.
    svc(&mut k, &mut cpu, 0x15, &[0, alias, 0x1000, 0]);
    assert_eq!(svc(&mut k, &mut cpu, 0x15, &[0, alias, 0x1000, 0]), INVALID_CURRENT_MEMORY);
    assert_eq!(svc(&mut k, &mut cpu, 0x15, &[0, alias + 0x10_0000, 0x1000, 0]), INVALID_CURRENT_MEMORY);
}
