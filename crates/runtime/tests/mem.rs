//! Guest memory on synthetic layouts.

use gyroship_runtime::mem::{self, Block, Error, Info, Memory, attr, perm, state};
use gyroship_runtime::{Cpu, ld64, st64};
use std::sync::atomic::{AtomicBool, AtomicU32, AtomicU64, Ordering};

const TEXT: u64 = 0x70_0000_0000;
const STACK: u64 = 0x20_0000_0000;
const HEAP: u64 = 0x10_0000_0000;

/// Walks the address space the way rtld does: query, jump to base + size,
/// stop when the address wraps to zero.
fn walk(m: &Memory) -> Vec<Info> {
    let mut out = Vec::new();
    let mut addr = 0u64;
    loop {
        let info = m.query(addr);
        out.push(info);
        addr = info.base.wrapping_add(info.size);
        if addr == 0 || out.len() > 100 {
            return out;
        }
    }
}

/// Several host threads check and copy while another one splits and merges
/// the next page's block: every check sees whole blocks (the table lock), and
/// `Memory` is Sync (the scoped threads share it).
#[test]
fn host_threads_share_one_memory() {
    let m = Memory::new().unwrap();
    m.map(HEAP, 0x2000, Block::new(state::NORMAL, perm::RW)).unwrap();
    std::thread::scope(|s| {
        s.spawn(|| {
            for _ in 0..20_000 {
                m.protect(HEAP + 0x1000, 0x1000, perm::R).unwrap();
                m.protect(HEAP + 0x1000, 0x1000, perm::RW).unwrap();
            }
        });
        for t in 0..4u64 {
            let m = &m;
            s.spawn(move || {
                let mut b = [0; 8];
                for i in 0..20_000u64 {
                    m.write(HEAP + 8 * t, &i.to_le_bytes()).unwrap();
                    m.read(HEAP + 8 * t, &mut b).unwrap();
                    assert_eq!(u64::from_le_bytes(b), i);
                }
            });
        }
    });
    assert_eq!(m.blocks().count(), 1, "merged back");
}

#[test]
fn walk_sees_free_gaps_and_the_inaccessible_tail() {
    let m = Memory::new().unwrap();
    m.map(TEXT, 0x4000, Block::new(state::CODE, perm::RX)).unwrap();
    m.map(TEXT + 0x4000, 0x1000, Block::new(state::CODE, perm::R)).unwrap();
    m.map(TEXT + 0x5000, 0x1000, Block::new(state::CODE_DATA, perm::RW)).unwrap();
    m.map(TEXT + 0x6000, 0x2000, Block::new(state::CODE_DATA, perm::RW)).unwrap();
    m.map(STACK, 0x10000, Block::new(state::STACK, perm::RW)).unwrap();
    let free = Block::new(state::FREE, 0);
    let expected = [
        Info { base: 0, size: STACK, block: free },
        Info { base: STACK, size: 0x10000, block: Block::new(state::STACK, perm::RW) },
        Info { base: STACK + 0x10000, size: TEXT - STACK - 0x10000, block: free },
        Info { base: TEXT, size: 0x4000, block: Block::new(state::CODE, perm::RX) },
        Info { base: TEXT + 0x4000, size: 0x1000, block: Block::new(state::CODE, perm::R) },
        // data and bss merge: same state, permission and attribute.
        Info { base: TEXT + 0x5000, size: 0x3000, block: Block::new(state::CODE_DATA, perm::RW) },
        Info { base: TEXT + 0x8000, size: mem::SPACE - TEXT - 0x8000, block: free },
        Info { base: mem::SPACE, size: mem::SPACE.wrapping_neg(), block: Block::new(state::INACCESSIBLE, 0) },
    ];
    assert_eq!(walk(&m), expected);
    assert_eq!(m.query(TEXT + 0x6800), expected[5]);
}

#[test]
fn memory_info_layout() {
    let info = Info { base: 0x1122_3344_5566_7788, size: 0x4000, block: Block { state: state::CODE, perm: perm::RX, attr: attr::LOCKED } };
    let b = info.to_bytes();
    assert_eq!(&b[0..8], &0x1122_3344_5566_7788u64.to_le_bytes());
    assert_eq!(&b[8..16], &0x4000u64.to_le_bytes());
    assert_eq!(&b[16..28], &[3, 0, 0, 0, 1, 0, 0, 0, 5, 0, 0, 0]);
    assert_eq!(&b[28..], &[0; 12]);
}

#[test]
fn bad_ranges_are_refused() {
    let m = Memory::new().unwrap();
    let rw = Block::new(state::NORMAL, perm::RW);
    assert_eq!(m.map(HEAP + 1, 0x1000, rw), Err(Error::InvalidRange));
    assert_eq!(m.map(HEAP, 0, rw), Err(Error::InvalidRange));
    assert_eq!(m.map(mem::SPACE - 0x1000, 0x2000, rw), Err(Error::InvalidRange));
    m.map(HEAP, 0x3000, rw).unwrap();
    assert_eq!(m.map(HEAP + 0x2000, 0x2000, rw), Err(Error::NotFree));
    assert_eq!(m.unmap(HEAP + 0x2000, 0x2000), Err(Error::InvalidState));
    assert_eq!(m.protect(HEAP - 0x1000, 0x2000, perm::R), Err(Error::InvalidState));
}

/// The atomic accessors read and write the bytes the copies do, check the
/// guest's permission as `read` and `write` do, refuse unaligned words, and
/// write nothing when they only load or when `update_u32` refuses.
#[test]
fn atomic_words_check_like_read_and_write() {
    let m = Memory::new().unwrap();
    m.map(HEAP, 0x1000, Block::new(state::NORMAL, perm::RW)).unwrap();
    m.map(HEAP + 0x1000, 0x1000, Block::new(state::NORMAL, perm::R)).unwrap();
    let (rw, r) = (HEAP + 0x10, HEAP + 0x1010);
    m.store_u32(rw, 0x1122_3344).unwrap();
    let mut b = [0; 4];
    m.read(rw, &mut b).unwrap();
    assert_eq!(b, [0x44, 0x33, 0x22, 0x11]);
    m.write(rw + 8, &0x0102_0304_0506_0708u64.to_le_bytes()).unwrap();
    assert_eq!(m.load_u64(rw + 8), Ok(0x0102_0304_0506_0708));
    m.take_written(HEAP, 0x2000);
    assert_eq!(m.update_u32(rw, |v| (v < 5).then_some(v.wrapping_sub(1))), Ok(Err(0x1122_3344)));
    assert_eq!(m.load_u32(rw), Ok(0x1122_3344));
    assert!(m.take_written(HEAP, 0x2000).is_empty(), "loads and refused updates write nothing");
    assert_eq!(m.update_u32(rw, |v| Some(v | 0x4000_0000)), Ok(Ok(0x1122_3344)));
    assert_eq!(m.load_u32(rw), Ok(0x5122_3344));
    // The guest's permission: R loads only.
    assert!(m.load_u32(r).is_ok());
    assert_eq!(m.store_u32(r, 1), Err(Error::InvalidState));
    assert_eq!(m.update_u32(r, Some), Err(Error::InvalidState));
    // Unmapped, unaligned, past the space.
    assert_eq!(m.load_u32(HEAP + 0x2000), Err(Error::InvalidState));
    assert_eq!(m.load_u32(rw + 2), Err(Error::InvalidRange));
    assert_eq!(m.load_u64(rw + 4), Err(Error::InvalidRange));
    assert_eq!(m.load_u32(mem::SPACE), Err(Error::InvalidRange));
}

/// `publish_object` leaves what `write_object` of the new word and the rest leaves:
/// the same bytes, written pages and refusals (then nothing is written), at every
/// alignment, across pages and blocks, and where the guest may only read.
#[test]
fn publish_object_leaves_what_write_object_leaves() {
    const COPY: u64 = 0x10_0000;
    let m = Memory::new().unwrap();
    for at in [HEAP, HEAP + COPY] {
        m.map(at, 0x2000, Block::new(state::NORMAL, perm::RW)).unwrap();
        // Guest R, host RW (the GPU still writes it), then host read-only (refused).
        m.map(at + 0x2000, 0x1000, Block::new(state::SHARED, perm::R)).unwrap();
        m.map(at + 0x3000, 0x1000, Block::new(state::CODE, perm::R)).unwrap();
        m.write_object(at, &(0..0x3000).map(|i| i as u8).collect::<Vec<_>>()).unwrap();
    }
    let mut seed = 0x9e37_79b9_7f4a_7c15u64;
    let mut next = |n: u64| {
        seed ^= seed << 13;
        seed ^= seed >> 7;
        seed ^= seed << 17;
        seed % n
    };
    m.take_written(HEAP, 0x4000);
    m.take_written(HEAP + COPY, 0x4000);
    for trial in 0..4000 {
        let (at, width) = (HEAP + next(0x4000 - 20), [4usize, 8][next(2) as usize]);
        let rest: Vec<u8> = (0..[0, 4, 8, 12][next(4) as usize]).map(|_| next(256) as u8).collect();
        let key = next(u64::MAX);
        let f = move |old: u64| old.wrapping_mul(3) ^ key;
        let mut old = [0u8; 8];
        m.read(at, &mut old[..width]).unwrap();
        let word = f(u64::from_le_bytes(old)).to_le_bytes();
        let plain = m.write_object(at, &[&word[..width], &rest[..]].concat());
        let published = m.publish_object(at + COPY, width, &rest, f);
        assert_eq!(plain.is_ok(), published.is_ok(), "trial {trial} at {at:#x}");
        if let Ok(o) = published {
            assert_eq!(o.to_le_bytes()[..width], old[..width], "trial {trial}: old word");
        }
        let (mut a, mut b) = (vec![0u8; 0x4000], vec![0u8; 0x4000]);
        m.read(HEAP, &mut a).unwrap();
        m.read(HEAP + COPY, &mut b).unwrap();
        assert_eq!(a, b, "trial {trial} at {at:#x}, width {width}, rest {}", rest.len());
        let copied: Vec<u64> = m.take_written(HEAP + COPY, 0x4000).iter().map(|p| p - COPY).collect();
        assert_eq!(m.take_written(HEAP, 0x4000), copied, "trial {trial}: written pages");
    }
}

/// A poller on another host thread that acquire-loads a report's payload never sees
/// it with an older report's timestamp: the timestamp goes first, the payload last
/// with Release (how a guest thread waits for the GPU).
#[test]
fn a_poller_that_sees_a_payload_sees_its_timestamp() {
    const REPORTS: u64 = 200_000;
    let m = Memory::new().unwrap();
    m.map(HEAP, 0x1000, Block::new(state::NORMAL, perm::RW)).unwrap();
    let base = m.base() as usize;
    let done = AtomicBool::new(false);
    std::thread::scope(|s| {
        let done = &done;
        s.spawn(move || {
            // SAFETY: a mapped page that outlives the scope; aligned words.
            let (payload, stamp) = unsafe { (AtomicU32::from_ptr((base as *mut u8).add(HEAP as usize).cast()), AtomicU64::from_ptr((base as *mut u8).add(HEAP as usize + 8).cast())) };
            while !done.load(Ordering::Acquire) {
                let p = u64::from(payload.load(Ordering::Acquire));
                let t = stamp.load(Ordering::Relaxed);
                assert!(t >= p, "payload {p} seen with the timestamp {t} of an older report");
            }
        });
        for i in 1..=REPORTS {
            m.publish_object(HEAP, 4, &[&[0u8; 4][..], &i.to_le_bytes()[..]].concat(), |_| i).unwrap();
        }
        done.store(true, Ordering::Release);
    });
}

/// GPU reductions and guest atomics on one word lose no update: the reduction is one
/// compare-exchange loop on the host word, as a guest LDADD is one LOCK XADD on it.
#[test]
fn reductions_and_guest_atomics_on_one_word_lose_nothing() {
    const N: u32 = 100_000;
    let m = Memory::new().unwrap();
    m.map(HEAP, 0x1000, Block::new(state::NORMAL, perm::RW)).unwrap();
    let base = m.base() as usize;
    std::thread::scope(|s| {
        for _ in 0..3 {
            s.spawn(move || {
                // SAFETY: as above.
                let word = unsafe { AtomicU32::from_ptr((base as *mut u8).add(HEAP as usize).cast()) };
                for _ in 0..N {
                    word.fetch_add(1, Ordering::SeqCst);
                }
            });
        }
        for _ in 0..N {
            m.publish_object(HEAP, 4, &[], |old| old + 1).unwrap();
        }
    });
    assert_eq!(m.load_u32(HEAP), Ok(4 * N));
}

/// The write watch reports the pages written (by the host here; the guest's
/// stores are plain stores too) in the range asked, once: the take resets them.
#[test]
fn written_pages_are_taken_once() {
    let m = Memory::new().unwrap();
    m.map(HEAP, 0x3000, Block::new(state::NORMAL, perm::RW)).unwrap();
    m.take_written(HEAP, 0x3000);
    assert!(m.take_written(HEAP, 0x3000).is_empty());
    m.write(HEAP + 0x1010, &[1, 2, 3]).unwrap();
    m.write(HEAP + 0x2010, &[4]).unwrap();
    assert_eq!(m.take_written(HEAP + 0x1008, 0x10), [HEAP + 0x1000], "the page of the range asked, from an unaligned base");
    assert_eq!(m.take_written(HEAP, 0x3000), [HEAP + 0x2000], "the other page, still written");
    assert!(m.take_written(HEAP, 0x3000).is_empty(), "both taken");
    assert!(m.take_written(HEAP, 0).is_empty() && m.take_written(mem::SPACE - 0x1000, 0x2000).is_empty());
}

/// A write that races a take shows at that take or the next one. Each round
/// one thread writes the round's number once, while the other keeps taking
/// the page and copying the word after each take that reports it (as a
/// texture re-uploads); once the write is done, one more take must leave the
/// copy up to date.
#[test]
fn a_write_racing_the_take_shows_at_the_next_take() {
    const ROUNDS: u32 = 1_000;
    let m = Memory::new().unwrap();
    m.map(HEAP, 0x1000, Block::new(state::NORMAL, perm::RW)).unwrap();
    // SAFETY: an aligned word of a page mapped for the whole test.
    let word = unsafe { AtomicU32::from_ptr(m.base().add(HEAP as usize + 0x10).cast()) };
    let (round, wrote) = (AtomicU32::new(0), AtomicU32::new(0));
    let lost = std::thread::scope(|s| {
        s.spawn(|| {
            for r in 1..=ROUNDS {
                while round.load(Ordering::Acquire) != r {
                    std::thread::yield_now();
                }
                // A varying delay: the write lands anywhere in the other thread's take.
                for _ in 0..r % 97 {
                    std::hint::spin_loop();
                }
                word.store(r, Ordering::Release);
                wrote.store(r, Ordering::Release);
            }
        });
        let (mut copy, mut lost) = (0, Vec::new());
        for r in 1..=ROUNDS {
            round.store(r, Ordering::Release);
            loop {
                let done = wrote.load(Ordering::Acquire) == r;
                if !m.take_written(HEAP, 0x1000).is_empty() {
                    copy = word.load(Ordering::Acquire);
                }
                if done {
                    break;
                }
                std::thread::yield_now();
            }
            if copy != r {
                lost.push(r);
            }
        }
        lost
    });
    assert!(lost.is_empty(), "writes lost in rounds {lost:?}");
}

#[test]
fn protect_splits_then_merges() {
    let m = Memory::new().unwrap();
    m.map(HEAP, 0x3000, Block::new(state::NORMAL, perm::RW)).unwrap();
    m.protect(HEAP + 0x1000, 0x1000, perm::R).unwrap();
    let perms: Vec<_> = m.blocks().map(|i| (i.base, i.size, i.block.perm)).collect();
    assert_eq!(perms, [(HEAP, 0x1000, perm::RW), (HEAP + 0x1000, 0x1000, perm::R), (HEAP + 0x2000, 0x1000, perm::RW)]);
    assert_eq!(m.write(HEAP + 0xff8, &[0; 16]), Err(Error::InvalidState));
    m.protect(HEAP + 0x1000, 0x1000, perm::RW).unwrap();
    assert_eq!(m.blocks().count(), 1);
    m.unmap(HEAP + 0x1000, 0x1000).unwrap();
    assert_eq!(m.blocks().count(), 2);
    assert_eq!(m.query(HEAP + 0x1000).block.state, state::FREE);
}

#[test]
fn map_move_locks_the_source_and_unmap_copies_back() {
    let m = Memory::new().unwrap();
    m.map(HEAP, 0x2000, Block::new(state::NORMAL, perm::RW)).unwrap();
    m.write(HEAP + 0x10, b"stack bytes").unwrap();
    m.map_move(STACK, HEAP, 0x2000, state::STACK).unwrap();
    let src = m.query(HEAP).block;
    assert_eq!((src.perm, src.attr), (0, attr::LOCKED));
    let mut buf = [0; 11];
    assert_eq!(m.read(HEAP + 0x10, &mut buf), Err(Error::InvalidState));
    m.read(STACK + 0x10, &mut buf).unwrap();
    assert_eq!(&buf, b"stack bytes");
    m.write(STACK + 0x10, b"STACK").unwrap();
    // A second move of a locked source is refused.
    assert_eq!(m.map_move(STACK + 0x10000, HEAP, 0x2000, state::STACK), Err(Error::InvalidState));
    // The destination must be the moved range: the locked source itself is refused.
    assert_eq!(m.unmap_move(HEAP, HEAP, 0x2000, state::STACK), Err(Error::InvalidState));
    m.unmap_move(STACK, HEAP, 0x2000, state::STACK).unwrap();
    assert_eq!(m.query(STACK).block.state, state::FREE);
    m.read(HEAP + 0x10, &mut buf).unwrap();
    assert_eq!(&buf, b"STACK bytes");
    assert_eq!(m.query(HEAP).block, Block::new(state::NORMAL, perm::RW));
}

/// Runs this test binary again in fault mode `mode`; returns its exit code and stderr.
fn run_child(mode: &str) -> (Option<i32>, String) {
    let out = std::process::Command::new(std::env::current_exe().unwrap())
        .args(["--exact", "fault_child", "--nocapture", "--test-threads=1"])
        .env("GYROSHIP_FAULT_CHILD", mode)
        .output()
        .unwrap();
    (out.status.code(), String::from_utf8_lossy(&out.stderr).into_owned())
}

/// Only does something when started by `faults_are_logged_with_the_guest_address`.
#[test]
fn fault_child() {
    let Ok(mode) = std::env::var("GYROSHIP_FAULT_CHILD") else { return };
    let m = Memory::new().unwrap();
    m.map(HEAP, 0x1000, Block::new(state::CODE, perm::R)).unwrap();
    mem::install_fault_handler();
    let cpu = Cpu::new(m.base());
    match mode.as_str() {
        // black_box: a release build would drop a load whose value is unused.
        "hole" => {
            let _ = std::hint::black_box(ld64(&cpu, 0x1234));
        }
        "wild" => {
            let _ = std::hint::black_box(ld64(&cpu, mem::SPACE + 0x5678));
        }
        "readonly" => st64(&cpu, HEAP + 8, 1),
        "two" => {
            // Two host threads fault at once: one report.
            let base = m.base() as usize;
            let start = std::sync::Arc::new(std::sync::Barrier::new(2));
            let threads: Vec<_> = (0..2)
                .map(|_| {
                    let start = start.clone();
                    std::thread::spawn(move || {
                        let cpu = Cpu::new(base as *mut u8);
                        start.wait();
                        let _ = std::hint::black_box(ld64(&cpu, 0x1234));
                    })
                })
                .collect();
            for t in threads {
                let _ = t.join();
            }
        }
        _ => unreachable!(),
    }
    eprintln!("no fault");
}

#[test]
fn only_the_first_fault_reports() {
    let (code, stderr) = run_child("two");
    assert_eq!(code, Some(mem::FAULT_EXIT), "{stderr}");
    assert_eq!(stderr.matches("guest fault:").count(), 1, "{stderr}");
}

#[test]
fn faults_are_logged_with_the_guest_address() {
    for (mode, line) in [
        ("hole", "guest fault: read at guest address 0x1234 "),
        // The 39-bit mask keeps a wild address inside the reservation.
        ("wild", "guest fault: read at guest address 0x5678 "),
        ("readonly", "guest fault: write at guest address 0x1000000008 "),
    ] {
        let (code, stderr) = run_child(mode);
        assert_eq!(code, Some(mem::FAULT_EXIT), "{mode}: {stderr}");
        assert!(stderr.contains(line), "{mode}: {stderr}");
    }
}
