//! The RomFS image over synthetic files, read back by an independent reader.

mod common;

use common::*;
use gyroship_hle::Deferred;
use gyroship_hle::fs::{Fs, RomFs};
use gyroship_hle::service::Service;
use gyroship_hle::thread::Outcome;
use gyroship_runtime::Cpu;
use gyroship_runtime::mem::{Block, Memory, perm, state};
use std::sync::Arc;

fn u32_at(b: &[u8], o: usize) -> u32 {
    u32::from_le_bytes(b[o..o + 4].try_into().unwrap())
}
fn u64_at(b: &[u8], o: usize) -> u64 {
    u64::from_le_bytes(b[o..o + 8].try_into().unwrap())
}

/// The format's name hash, written again here.
fn name_hash(parent: u32, name: &str) -> u32 {
    name.bytes().fold(parent ^ 123_456_789, |h, c| h.rotate_right(5) ^ u32::from(c))
}

/// Finds `path` ("dir/file") in the image like a guest reader: hash buckets, then entries.
fn lookup(r: &RomFs, path: &str) -> Option<(u64, u64)> {
    let mut header = [0u8; 0x50];
    r.read(0, &mut header).unwrap();
    let table = |i: usize| {
        let (off, size) = (u64_at(&header, 8 + 16 * i), u64_at(&header, 16 + 16 * i));
        let mut t = vec![0u8; size as usize];
        r.read(off, &mut t).unwrap();
        t
    };
    let (dir_hash, dirs, file_hash, files) = (table(0), table(1), table(2), table(3));
    let data = u64_at(&header, 0x48);
    let mut parts: Vec<&str> = path.split('/').collect();
    let file = parts.pop()?;
    let mut dir = 0u32;
    for d in parts {
        let mut at = u32_at(&dir_hash, 4 * (name_hash(dir, d) as usize % (dir_hash.len() / 4)));
        loop {
            if at == u32::MAX {
                return None;
            }
            let e = at as usize;
            let len = u32_at(&dirs, e + 0x14) as usize;
            if u32_at(&dirs, e) == dir && &dirs[e + 0x18..e + 0x18 + len] == d.as_bytes() {
                break;
            }
            at = u32_at(&dirs, e + 0x10);
        }
        dir = at;
    }
    let mut at = u32_at(&file_hash, 4 * (name_hash(dir, file) as usize % (file_hash.len() / 4)));
    while at != u32::MAX {
        let e = at as usize;
        let len = u32_at(&files, e + 0x1C) as usize;
        if u32_at(&files, e) == dir && &files[e + 0x20..e + 0x20 + len] == file.as_bytes() {
            return Some((data + u64_at(&files, e + 8), u64_at(&files, e + 0x10)));
        }
        at = u32_at(&files, e + 0x18);
    }
    None
}

#[test]
fn files_are_found_and_read_back() {
    let root = std::path::Path::new(env!("CARGO_TARGET_TMPDIR")).join("romfs-test");
    let _ = std::fs::remove_dir_all(&root);
    std::fs::create_dir_all(root.join("sub")).unwrap();
    let big: Vec<u8> = (0..20u8).collect();
    std::fs::write(root.join("a.bin"), &big).unwrap();
    std::fs::write(root.join("b.txt"), b"hello").unwrap();
    std::fs::write(root.join("sub").join("c"), b"xyz").unwrap();
    let r = RomFs::build(&root).unwrap();

    let (a, alen) = lookup(&r, "a.bin").expect("a.bin");
    let (b, blen) = lookup(&r, "b.txt").expect("b.txt");
    let (c, clen) = lookup(&r, "sub/c").expect("sub/c");
    assert_eq!((alen, blen, clen), (20, 5, 3));
    assert!(lookup(&r, "missing").is_none());
    assert_eq!((a, b), (0x200, 0x200 + 32), "file data from 0x200, each file 16-byte aligned");
    let mut buf = vec![0u8; 3];
    r.read(c, &mut buf).unwrap();
    assert_eq!(buf, b"xyz");
    // A read across the end of a.bin, its padding and the start of b.txt.
    let mut buf = vec![0u8; 32 + 5 - 18];
    r.read(a + 18, &mut buf).unwrap();
    assert_eq!(&buf[..2], &big[18..]);
    assert!(buf[2..14].iter().all(|&x| x == 0));
    assert_eq!(&buf[14..], b"hello");
    // Same files, same image.
    let mut h1 = vec![0u8; r.size() as usize];
    r.read(0, &mut h1).unwrap();
    let mut h2 = vec![0u8; r.size() as usize];
    RomFs::build(&root).unwrap().read(0, &mut h2).unwrap();
    assert_eq!(h1, h2);
}

/// A layer file takes the dump file's place, bytes and size; the other files,
/// and the dump itself, stay as they are.
#[test]
fn a_layer_file_replaces_the_dumps() {
    let (root, layer) = (folder("romfs-layer-dump"), folder("romfs-layer"));
    std::fs::create_dir_all(root.join("sub")).unwrap();
    std::fs::create_dir_all(layer.join("sub")).unwrap();
    std::fs::write(root.join("data.zip"), b"zip").unwrap();
    std::fs::write(root.join("x_build.cvar"), b"{\"a\": true}").unwrap();
    std::fs::write(root.join("sub").join("c"), b"xyz").unwrap();
    std::fs::write(layer.join("x_build.cvar"), b"{\"a\": false, \"b\": 1.0}").unwrap();
    std::fs::write(layer.join("sub").join("c"), b"longer than xyz").unwrap();
    std::fs::write(layer.join("not_in_the_dump"), b"ignored").unwrap();
    let (r, replaced) = RomFs::layered(&root, Some(&layer)).unwrap();
    assert_eq!(replaced, ["sub/c", "x_build.cvar"]);
    let read = |path: &str| {
        let (at, len) = lookup(&r, path).expect(path);
        let mut buf = vec![0; len as usize];
        r.read(at, &mut buf).unwrap();
        buf
    };
    assert_eq!(read("x_build.cvar"), b"{\"a\": false, \"b\": 1.0}");
    assert_eq!(read("sub/c"), b"longer than xyz");
    assert_eq!(read("data.zip"), b"zip");
    assert!(lookup(&r, "not_in_the_dump").is_none());
    assert_eq!(std::fs::read(root.join("x_build.cvar")).unwrap(), b"{\"a\": true}", "the dump is untouched");
}

/// An archive as the game's engine reads it: an index block ("VZIP", a record
/// count, the names' size; a 40-byte record per entry; the names, each ended
/// by a zero), then each entry's raw bytes.
fn archive_of(entries: &[(&str, &[u8])]) -> Vec<u8> {
    let names: Vec<u8> = entries.iter().flat_map(|e| [e.0.as_bytes(), &[0]].concat()).collect();
    let mut zip = [&b"VZIP"[..], &(entries.len() as u32).to_le_bytes(), &(names.len() as u32).to_le_bytes(), &vec![0; 40 * entries.len()][..], &names[..]].concat();
    let mut name = 0u32;
    for (k, &(n, data)) in entries.iter().enumerate() {
        // A record: its name's offset; from its byte 8, where the bytes are, and their two sizes.
        let (record, at) = (12 + 40 * k, zip.len() as u64);
        zip[record..record + 4].copy_from_slice(&name.to_le_bytes());
        zip[record + 8..record + 32].copy_from_slice(&[at, data.len() as u64, data.len() as u64].map(u64::to_le_bytes).concat());
        zip.extend(data);
        name += n.len() as u32 + 1;
    }
    zip
}

/// A layer's `<archive>.entries` folder replaces entries inside the dump's
/// archive: the image serves the dump's bytes, but for the index record of
/// the entry, which points at the new content after them.
#[test]
fn a_layer_entry_replaces_one_inside_the_dumps_archive() {
    let (root, layer) = (folder("romfs-entries-dump"), folder("romfs-entries"));
    let zip = archive_of(&[("aa", b"first"), ("bb", b"second")]);
    std::fs::write(root.join("data.zip"), &zip).unwrap();
    let entries = layer.join("data.zip.entries");
    std::fs::create_dir_all(&entries).unwrap();
    std::fs::write(entries.join("bb"), b"a longer second entry").unwrap();
    let (r, replaced) = RomFs::layered(&root, Some(&layer)).unwrap();
    assert_eq!(replaced, ["data.zip entries bb"]);
    let (at, len) = lookup(&r, "data.zip").expect("data.zip");
    let mut served = vec![0; len as usize];
    r.read(at, &mut served).unwrap();
    // The second record starts at 52: its bytes 8 to 32 changed, nothing else.
    let [new, stored, size] = [60, 68, 76].map(|o| u64_at(&served, o));
    assert!(new >= zip.len() as u64 && (stored, size) == (21, 21), "the record points past the dump's bytes");
    assert_eq!(&served[new as usize..][..21], b"a longer second entry");
    assert_eq!((&served[..60], &served[84..zip.len()]), (&zip[..60], &zip[84..]), "the dump's other bytes");
    // The same bytes whatever a read's bounds, across the record and the dump's end too.
    let mut pieces = vec![0; len as usize];
    for (i, piece) in pieces.chunks_mut(7).enumerate() {
        r.read(at + 7 * i as u64, piece).unwrap();
    }
    assert_eq!(pieces, served);
    // A name the archive lacks is refused; the dump is only read.
    std::fs::write(entries.join("cc"), b"x").unwrap();
    assert!(RomFs::layered(&root, Some(&layer)).unwrap_err().to_string().contains("no entry named cc"));
    assert_eq!(std::fs::read(root.join("data.zip")).unwrap(), zip, "the dump is untouched");
}

/// An empty folder of the test's own.
fn folder(name: &str) -> std::path::PathBuf {
    let root = std::path::Path::new(env!("CARGO_TARGET_TMPDIR")).join(name);
    let _ = std::fs::remove_dir_all(&root);
    std::fs::create_dir_all(&root).unwrap();
    root
}

const READER: u64 = 0x1000;
const DEST: u64 = DATA + 0x800;

/// Guest code: IStorage Read of the image's only file ("xyz", its data at
/// 0x200) into DEST, then a look at DEST as soon as the SVC returns.
unsafe extern "C" fn reader(cpu: *mut Cpu, _: u64) {
    unsafe {
        let root = folder("romfs-hook");
        std::fs::write(root.join("c"), b"xyz").unwrap();
        let h = gyroship_hle::with(|k| {
            k.romfs = Some(Arc::new(RomFs::build(&root).unwrap()));
            k.open_session(Service::Fs(Fs::RomStorage)).unwrap()
        });
        // CMIF request 0, one B buffer (DEST, 3 bytes); data: offset 0x200, size 3.
        let sfci = u32::from_le_bytes(*b"SFCI");
        let msg = [4 | 1 << 24, 11, 3, DEST as u32, ((DEST >> 36 & 7) << 2) as u32, 0, 0, 0, sfci, 1, 0, 0, 0x200, 0, 3, 0];
        for (i, w) in msg.into_iter().enumerate() {
            store(cpu, (*cpu).tpidrro + 4 * i as u64, w);
        }
        record("svc", svc(cpu, 0x21, &[u64::from(h)]));
        record("result", u64::from(load(cpu, (*cpu).tpidrro + 0x18)));
        record("bytes", u64::from(load(cpu, DEST)));
        ret(cpu);
    }
}

/// An IStorage Read made through the real hooks: the bytes land after the
/// kernel lock is released, before the guest resumes.
#[test]
fn a_read_lands_before_the_guest_resumes() {
    install(&[(READER, reader)]);
    let (outcome, events, journal) = run(READER);
    assert_eq!(outcome, Outcome::Finished, "{journal}");
    assert_eq!(events, [("svc", 0), ("result", 0), ("bytes", u64::from(u32::from_le_bytes(*b"xyz\0")))]);
}

/// Deferred reads on several host threads at once, over one image and one
/// guest memory, with no kernel anywhere: first opens race on each file's
/// handle, reads cross file ends, and every byte lands right.
#[test]
fn deferred_reads_run_on_several_host_threads() {
    let root = folder("romfs-threads");
    for f in 0..16u32 {
        let bytes: Vec<u8> = (0..40_000u32).map(|i| (f * 31 + i * 7 + i / 251) as u8).collect();
        std::fs::write(root.join(format!("f{f:02}")), bytes).unwrap();
    }
    let image = Arc::new(RomFs::build(&root).unwrap());
    let size = image.size();
    // The expected bytes, from an image of its own: the shared one has opened nothing.
    let mut whole = vec![0; size as usize];
    RomFs::build(&root).unwrap().read(0, &mut whole).unwrap();
    let span = size.next_multiple_of(0x1000);
    let mem = Arc::new(Memory::new().unwrap());
    mem.map(DATA, 8 * span, Block::new(state::NORMAL, perm::RW)).unwrap();
    std::thread::scope(|s| {
        for t in 0..8u64 {
            let (image, mem) = (&image, &mem);
            s.spawn(move || {
                // Thread t copies the whole image to its own span, in chunks that cross files.
                let chunk = 3000 + 517 * t;
                let mut at = 0;
                while at < size {
                    let len = chunk.min(size - at);
                    Deferred::RomFsRead { romfs: image.clone(), mem: mem.clone(), offset: at, addr: DATA + t * span + at, len }.run().unwrap();
                    at += len;
                }
            });
        }
    });
    for t in 0..8 {
        let mut copy = vec![0; size as usize];
        mem.read(DATA + t * span, &mut copy).unwrap();
        assert_eq!(copy, whole, "thread {t}");
    }
}
