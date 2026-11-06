//! Synthetic unwind tables; no game data is involved.

use gyroship_analysis::eh_frame::{function_ranges, landing_pads, Range};

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
    assert_eq!(ranges, [Range { start: 0x40, len: 0x20, lsda: None }, Range { start: 0x80, len: 0x10, lsda: None }]);
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

#[test]
fn reads_the_lsda_pointer_and_its_landing_pads() {
    let mut img = vec![0u8; 0x300];
    // CIE "zLR": LSDA pcrel|sdata4, FDE pointers pcrel|sdata4.
    put(&mut img, 0x100, &0x1Cu32.to_le_bytes());
    put(&mut img, 0x108, &[1, b'z', b'L', b'R', 0, 4, 0x78, 30, 2, 0x1B, 0x1B]);
    // FDE for 0x40..0x80 whose augmentation data points at the LSDA at 0x180.
    put(&mut img, 0x120, &0x1Cu32.to_le_bytes());
    put(&mut img, 0x124, &((0x124 - 0x100) as u32).to_le_bytes());
    put(&mut img, 0x128, &rel(0x40, 0x128));
    put(&mut img, 0x12C, &0x40u32.to_le_bytes());
    put(&mut img, 0x130, &[4]);
    put(&mut img, 0x131, &rel(0x180, 0x131));
    put(&mut img, 0x200, &[1, 0x1B, 0x03, 0x3B]);
    put(&mut img, 0x204, &rel(0x100, 0x204));
    put(&mut img, 0x208, &1u32.to_le_bytes());
    put(&mut img, 0x20C, &(0x40i32 - 0x200).to_le_bytes());
    put(&mut img, 0x210, &(0x120i32 - 0x200).to_le_bytes());
    // LSDA: no lpstart, no type table, uleb call sites; pads at +0x30, none, +0x30 again, +0x24.
    put(&mut img, 0x180, &[0xFF, 0xFF, 0x01, 16, 0x00, 0x08, 0x30, 0x00, 0x08, 0x04, 0x00, 0x00, 0x0C, 0x04, 0x30, 0x01, 0x10, 0x04, 0x24, 0x00]);
    let ranges = function_ranges(&img, 0x200).unwrap();
    assert_eq!(ranges, [Range { start: 0x40, len: 0x40, lsda: Some(0x180) }]);
    assert_eq!(landing_pads(&img, 0x40, 0x180).unwrap(), [0x64, 0x70]);
    img[0x183] = 15; // the length cuts a record in half
    assert!(landing_pads(&img, 0x40, 0x180).is_err());
}
