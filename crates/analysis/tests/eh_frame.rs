//! Synthetic unwind tables; no game data is involved.

use gyroship_analysis::eh_frame::{function_ranges, Range};

fn put(buf: &mut [u8], off: usize, v: &[u8]) {
    buf[off..off + v.len()].copy_from_slice(v);
}

fn rel(target: i64, at: usize) -> [u8; 4] {
    ((target - at as i64) as i32).to_le_bytes()
}

/// CIE @0x100 ("zR", FDE pointers pcrel|sdata4), FDEs @0x120 and @0x140,
/// eh_frame_hdr @0x200 with a datarel|sdata4 search table.
fn image(functions: &[(i64, i32)], table: &[(i64, i64)]) -> Vec<u8> {
    let mut img = vec![0u8; 0x300];
    put(&mut img, 0x100, &0x1Cu32.to_le_bytes());
    put(&mut img, 0x104, &0u32.to_le_bytes());
    put(&mut img, 0x108, &[1, b'z', b'R', 0, 4, 0x78, 30, 1, 0x1B]);
    for (i, (start, len)) in functions.iter().enumerate() {
        let fde = 0x120 + 0x20 * i;
        put(&mut img, fde, &0x1Cu32.to_le_bytes());
        put(&mut img, fde + 4, &((fde + 4 - 0x100) as u32).to_le_bytes());
        put(&mut img, fde + 8, &rel(*start, fde + 8));
        put(&mut img, fde + 12, &len.to_le_bytes());
    }
    put(&mut img, 0x200, &[1, 0x1B, 0x03, 0x3B]);
    put(&mut img, 0x204, &rel(0x100, 0x204));
    put(&mut img, 0x208, &(table.len() as u32).to_le_bytes());
    for (i, (start, fde)) in table.iter().enumerate() {
        put(&mut img, 0x20C + 8 * i, &((start - 0x200) as i32).to_le_bytes());
        put(&mut img, 0x210 + 8 * i, &((fde - 0x200) as i32).to_le_bytes());
    }
    img
}

#[test]
fn reads_every_fde_from_the_search_table() {
    let img = image(&[(0x40, 0x20), (0x80, 0x10)], &[(0x40, 0x120), (0x80, 0x140)]);
    let ranges = function_ranges(&img, 0x200).unwrap();
    assert_eq!(ranges, [Range { start: 0x40, len: 0x20 }, Range { start: 0x80, len: 0x10 }]);
}

#[test]
fn rejects_inconsistent_tables() {
    let unsorted = image(&[(0x40, 0x20), (0x80, 0x10)], &[(0x80, 0x140), (0x40, 0x120)]);
    assert!(function_ranges(&unsorted, 0x200).unwrap_err().0.contains("not sorted"));
    let wrong_start = image(&[(0x44, 0x20)], &[(0x40, 0x120)]);
    assert!(function_ranges(&wrong_start, 0x200).unwrap_err().0.contains("starts at"));
    let truncated = image(&[(0x40, 0x20)], &[(0x40, 0x120)]);
    assert!(function_ranges(&truncated[..0x20E], 0x200).is_err());
}
