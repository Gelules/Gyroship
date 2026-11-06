//! The RomFS image over synthetic files, read back by an independent reader.

use gyroship_hle::fs::RomFs;

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
