//! Game functions replaced by ours (`--hooks FILE`): the file says where (the
//! game's addresses stay in data files, D21), the code here says what. Only a
//! function the game reaches by indirect calls can be replaced this way: the
//! dispatch table serves those, while a direct call was bound when the code
//! was generated.

use gyroship_runtime::{self as rt, Cpu, Entry, Guest};
use std::sync::atomic::{AtomicU64, Ordering::Relaxed};

/// The hooks a `--hooks` file can name.
const HOOKS: &[(&str, Guest)] = &[("requirement-met", requirement_met), ("quit", quit)];

/// Where `quit` writes: the address its line gives.
static QUIT_FLAG: AtomicU64 = AtomicU64::new(0);

/// A menu button's handler that quits the game, in place of one that did nothing:
/// sets the byte the game's frame loop polls to end the application (what the game
/// itself sets when the console asks it to close; its address is the line's third
/// word), so the game shuts itself down. The answer is the empty handler's: a null
/// variant through x8, 24 zero bytes.
unsafe extern "C" fn quit(cpu: *mut Cpu, _resume: u64) {
    // SAFETY: the running thread's context, as for any recompiled function.
    let c = unsafe { &mut *cpu };
    rt::st8(c, QUIT_FLAG.load(Relaxed), 1);
    let out = c.x[8];
    for word in [0, 8, 0x10] {
        rt::st64(c, out + word, 0);
    }
    c.pc = c.x[30]; // RET (D20)
}

/// A menu button requirement's test, always met: the boolean variant the
/// game's own test returns, through x8 (the AAPCS64 indirect result
/// register; x0 is the call's context): value 1, then its type, 3 (a
/// boolean), at +0x10.
unsafe extern "C" fn requirement_met(cpu: *mut Cpu, _resume: u64) {
    // SAFETY: the running thread's context, as for any recompiled function.
    let c = unsafe { &mut *cpu };
    let out = c.x[8];
    rt::st64(c, out, 1);
    rt::st64(c, out + 8, 0);
    rt::st64(c, out + 0x10, 3);
    c.pc = c.x[30]; // RET (D20)
}

/// Gives the function at each line's address (`<hex address> <hook>`, `#`
/// starts a comment) the named hook's code; returns what it replaced, for
/// the journal. A `quit` line has a third word, the address of the byte it
/// sets: the same one on every such line.
pub fn apply(table: &mut [Entry], text: &str) -> Result<Vec<String>, String> {
    let mut done = Vec::new();
    let mut quit_flag = None;
    for (n, line) in text.lines().enumerate() {
        let bad = |what: &str| format!("line {}: {what}: {line}", n + 1);
        let hex = |word: &str| u64::from_str_radix(word.trim_start_matches("0x"), 16).ok();
        let mut words = line.split('#').next().unwrap_or("").split_whitespace();
        let Some(addr) = words.next() else { continue };
        let addr = hex(addr).ok_or_else(|| bad("not a hex address"))?;
        let name = words.next().ok_or_else(|| bad("no hook named"))?;
        let &(_, hook) = HOOKS.iter().find(|(h, _)| *h == name).ok_or_else(|| bad("unknown hook"))?;
        let mut what = format!("{name} at {addr:#x}");
        if name == "quit" {
            let flag = words.next().and_then(hex).ok_or_else(|| bad("quit needs the hex address of the byte it sets"))?;
            if quit_flag.replace(flag).is_some_and(|other| other != flag) {
                return Err(bad("quit lines name different bytes"));
            }
            QUIT_FLAG.store(flag, Relaxed);
            what += &format!(", which sets the byte at {flag:#x}");
        }
        let e = table.iter_mut().find(|e| e.addr == addr).ok_or_else(|| bad("no recompiled function starts there"))?;
        e.function = hook;
        done.push(what);
    }
    Ok(done)
}

#[cfg(test)]
mod tests {
    use super::*;
    use gyroship_runtime::mem::{Block, Memory, perm, state};

    unsafe extern "C" fn original(_: *mut Cpu, _: u64) {}

    #[test]
    fn the_requirement_is_met_through_x8_and_returns() {
        let m = Memory::new().unwrap();
        m.map(0x10_0000, 0x1000, Block::new(state::NORMAL, perm::RW)).unwrap();
        let (out, context) = (0x10_0000, 0x10_0800);
        m.write(out, &[0xaa; 0x18]).unwrap();
        m.write(context, &[0x55; 0x18]).unwrap();
        let mut c = Cpu::new(m.base());
        (c.x[0], c.x[8], c.x[30]) = (context, out, 0x7100_1234);
        // SAFETY: a context whose memory covers what the hook writes.
        unsafe { requirement_met(&mut c, 0) };
        let mut got = [0u8; 0x18];
        m.read(out, &mut got).unwrap();
        let mut want = [0u8; 0x18];
        (want[0], want[0x10]) = (1, 3);
        assert_eq!(got, want, "met (1), a boolean (3 at +0x10)");
        m.read(context, &mut got).unwrap();
        assert_eq!(got, [0x55; 0x18], "x0, the call's context, is left alone");
        assert_eq!(c.pc, 0x7100_1234, "returns to x30");
    }

    /// The quit hook sets the one byte its line names and answers a null variant; a
    /// line without that address is refused.
    #[test]
    fn quit_sets_the_byte_its_line_names_and_answers_null() {
        let m = Memory::new().unwrap();
        m.map(0x10_0000, 0x1000, Block::new(state::NORMAL, perm::RW)).unwrap();
        let (out, flag) = (0x10_0000, 0x10_0900);
        m.write(out, &[0xaa; 0x20]).unwrap();
        m.write(flag - 1, &[0x55; 3]).unwrap();
        let mut table = vec![Entry { addr: 0x71_0000_0000, len: 4, function: original }, Entry { addr: 0x71_0000_0010, len: 4, function: original }];
        for (line, why) in [("7100000000 quit", "the byte it sets"), ("7100000000 quit zz", "the byte it sets"), ("7100000000 quit 100900\n7100000010 quit 100901", "different bytes")] {
            assert!(apply(&mut table, line).unwrap_err().contains(why), "{line}");
        }
        assert_eq!(table[1].function as *const (), original as *const (), "a refused line replaces nothing");
        let done = apply(&mut table, "7100000000 quit 100900 # the game's flag\n7100000010 quit 0x100900").unwrap();
        assert_eq!(done, ["quit at 0x7100000000, which sets the byte at 0x100900", "quit at 0x7100000010, which sets the byte at 0x100900"]);
        let mut c = Cpu::new(m.base());
        (c.x[8], c.x[30]) = (out, 0x7100_1234);
        let registers = c.x;
        // SAFETY: a context whose memory covers what the hook writes.
        unsafe { (table[0].function)(&mut c, 0) };
        let (mut got, mut near) = ([0u8; 0x20], [0u8; 3]);
        m.read(out, &mut got).unwrap();
        m.read(flag - 1, &mut near).unwrap();
        assert_eq!((&got[..0x18], &got[0x18..]), (&[0u8; 0x18][..], &[0xaa; 8][..]), "24 zero bytes through x8, no more");
        assert_eq!(near, [0x55, 1, 0x55], "the byte, alone");
        assert_eq!((c.pc, c.x), (0x7100_1234, registers), "returns to x30, the registers as they were");
    }

    #[test]
    fn a_hooks_file_replaces_named_functions_only() {
        let mut table = vec![Entry { addr: 0x71_0000_0000, len: 4, function: original }, Entry { addr: 0x71_0000_0010, len: 4, function: original }];
        let done = apply(&mut table, "# test\n7100000010 requirement-met # second\n\n").unwrap();
        assert_eq!(done, ["requirement-met at 0x7100000010"]);
        assert_eq!(table[0].function as *const (), original as *const ());
        assert_eq!(table[1].function as *const (), requirement_met as *const ());
        assert!(apply(&mut table, "7100000020 requirement-met").unwrap_err().contains("no recompiled function"));
        assert!(apply(&mut table, "7100000000 nothing").unwrap_err().contains("unknown hook"));
        assert!(apply(&mut table, "zz requirement-met").unwrap_err().contains("hex"));
    }
}
