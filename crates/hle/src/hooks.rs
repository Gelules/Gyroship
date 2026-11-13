//! Game functions replaced by ours (`--hooks FILE`): the file says where (the
//! game's addresses stay in data files, D21), the code here says what. Only a
//! function the game reaches by indirect calls can be replaced this way: the
//! dispatch table serves those, while a direct call was bound when the code
//! was generated.

use gyroship_runtime::{self as rt, Cpu, Entry, Guest};

/// The hooks a `--hooks` file can name.
const HOOKS: &[(&str, Guest)] = &[("requirement-met", requirement_met)];

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
/// the journal.
pub fn apply(table: &mut [Entry], text: &str) -> Result<Vec<String>, String> {
    let mut done = Vec::new();
    for (n, line) in text.lines().enumerate() {
        let bad = |what: &str| format!("line {}: {what}: {line}", n + 1);
        let mut words = line.split('#').next().unwrap_or("").split_whitespace();
        let Some(addr) = words.next() else { continue };
        let addr = u64::from_str_radix(addr.trim_start_matches("0x"), 16).map_err(|_| bad("not a hex address"))?;
        let name = words.next().ok_or_else(|| bad("no hook named"))?;
        let &(_, hook) = HOOKS.iter().find(|(h, _)| *h == name).ok_or_else(|| bad("unknown hook"))?;
        let e = table.iter_mut().find(|e| e.addr == addr).ok_or_else(|| bad("no recompiled function starts there"))?;
        e.function = hook;
        done.push(format!("{name} at {addr:#x}"));
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
