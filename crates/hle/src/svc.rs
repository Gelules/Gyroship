//! The SVC table (D21). A number is implemented, forbidden by the process's
//! kernel capabilities (KAC), or not implemented yet; the last two stop the
//! run with a journal line, unless exploration mode is on.
//!
//! Convention of the Horizon kernel, as the SDK's wrappers use it: arguments
//! in x0..x7, the Result in w0, outputs in x1 and x2. The recompiled wrapper
//! copies outputs to memory itself; only QueryMemory (and later
//! GetThreadContext3) write guest memory directly.

use crate::result::*;
use crate::thread::{CORES, PRIORITIES, State, THREAD_LIMIT, Wait};
use crate::{Kernel, STOP_EXIT, layout};
use gyroship_runtime::Cpu;
use gyroship_runtime::mem::{self, Block, attr, perm, state};

/// SVCs the SDK contains but the game's KAC does not grant: dead code.
const FORBIDDEN: [u32; 11] = [0x40, 0x41, 0x43, 0x44, 0x45, 0x4B, 0x50, 0x51, 0x52, 0x70, 0x72];

/// Pseudo-handles of the current thread and process.
pub const CURRENT_THREAD: u64 = 0xFFFF_8000;
pub const CURRENT_PROCESS: u64 = 0xFFFF_8001;

/// The public name of an SVC number (switchbrew).
pub fn name(imm: u32) -> &'static str {
    match imm {
        0x01 => "SetHeapSize",
        0x02 => "SetMemoryPermission",
        0x03 => "SetMemoryAttribute",
        0x04 => "MapMemory",
        0x05 => "UnmapMemory",
        0x06 => "QueryMemory",
        0x07 => "ExitProcess",
        0x08 => "CreateThread",
        0x09 => "StartThread",
        0x0A => "ExitThread",
        0x0B => "SleepThread",
        0x0C => "GetThreadPriority",
        0x0D => "SetThreadPriority",
        0x0E => "GetThreadCoreMask",
        0x0F => "SetThreadCoreMask",
        0x10 => "GetCurrentProcessorNumber",
        0x11 => "SignalEvent",
        0x12 => "ClearEvent",
        0x13 => "MapSharedMemory",
        0x14 => "UnmapSharedMemory",
        0x15 => "CreateTransferMemory",
        0x16 => "CloseHandle",
        0x17 => "ResetSignal",
        0x18 => "WaitSynchronization",
        0x19 => "CancelSynchronization",
        0x1A => "ArbitrateLock",
        0x1B => "ArbitrateUnlock",
        0x1C => "WaitProcessWideKeyAtomic",
        0x1D => "SignalProcessWideKey",
        0x1F => "ConnectToNamedPort",
        0x21 => "SendSyncRequest",
        0x22 => "SendSyncRequestWithUserBuffer",
        0x23 => "SendAsyncRequestWithUserBuffer",
        0x25 => "GetThreadId",
        0x26 => "Break",
        0x27 => "OutputDebugString",
        0x28 => "ReturnFromException",
        0x29 => "GetInfo",
        0x2C => "MapPhysicalMemory",
        0x2D => "UnmapPhysicalMemory",
        0x32 => "SetThreadActivity",
        0x33 => "GetThreadContext3",
        0x34 => "WaitForAddress",
        0x35 => "SignalToAddress",
        0x36 => "SynchronizePreemptionState",
        0x40 => "CreateSession",
        0x41 => "AcceptSession",
        0x43 => "ReplyAndReceive",
        0x44 => "ReplyAndReceiveWithUserBuffer",
        0x45 => "CreateEvent",
        0x4B => "CreateCodeMemory",
        0x50 => "CreateSharedMemory",
        0x51 => "MapTransferMemory",
        0x52 => "UnmapTransferMemory",
        0x70 => "CreatePort",
        0x72 => "ConnectToPort",
        0x80 => "SetGpuMemoryAttribute",
        _ => "?",
    }
}

fn args(a: &[u64; 8]) -> String {
    a.iter().enumerate().map(|(i, v)| format!(" x{i}={v:#x}")).collect()
}

/// The Result of a failed memory operation.
fn mem_result(e: mem::Error) -> u32 {
    match e {
        mem::Error::InvalidRange => INVALID_ADDRESS,
        mem::Error::NotFree | mem::Error::InvalidState => INVALID_CURRENT_MEMORY,
        mem::Error::Host(code) => panic!("the host refused a memory operation (Windows error {code})"),
    }
}

fn done(r: Result<(), mem::Error>) -> u32 {
    r.map_or_else(mem_result, |()| SUCCESS)
}

impl Kernel {
    /// Handles `svc #imm` at `pc`.
    pub fn svc(&mut self, cpu: &mut Cpu, pc: u64, imm: u32) {
        let a: [u64; 8] = cpu.x[..8].try_into().expect("8 registers");
        self.frame = (pc, cpu.x[30], cpu.x[29]);
        let call = format!("svc {imm:#04x} {}", name(imm));
        if FORBIDDEN.contains(&imm) {
            self.stop(STOP_EXIT, &format!("forbidden {call} at {pc:#x}: the KAC does not grant it;{}", args(&a)));
        }
        // (Result, what makes this call worth a new journal line, what to log)
        let (result, kind, detail) = match imm {
            0x02 => (self.set_permission(a[0], a[1], a[2] as u32), 0, format!("{:#x}+{:#x} perm {}", a[0], a[1], a[2] as u32)),
            0x03 => (self.set_attribute(a[0], a[1], a[2] as u32, a[3] as u32, attr::UNCACHED | attr::PERMISSION_LOCKED), 0, format!("{:#x}+{:#x} mask {:#x} value {:#x}", a[0], a[1], a[2] as u32, a[3] as u32)),
            0x04 => (self.map_memory(a[0], a[1], a[2]), 0, format!("{:#x} <- {:#x} size {:#x}", a[0], a[1], a[2])),
            0x05 => (self.unmap_memory(a[0], a[1], a[2]), 0, format!("{:#x} -> {:#x} size {:#x}", a[0], a[1], a[2])),
            0x06 => {
                let info = self.mem.query(a[2]);
                cpu.x[1] = 0;
                let r = if self.mem.write(a[0], &info.to_bytes()).is_ok() { SUCCESS } else { INVALID_POINTER };
                (r, 0, format!("{:#x} -> {:#x}+{:#x} state {:#x} perm {} attr {:#x}", a[2], info.base, info.size, info.block.state, info.block.perm, info.block.attr))
            }
            0x07 => self.stop(0, &format!("{call} at {pc:#x}: the guest exits")),
            0x08 => match self.create_thread(a[1], a[2], a[3], a[4] as u32, a[5] as i32) {
                Ok(h) => {
                    cpu.x[1] = u64::from(h);
                    (SUCCESS, 0, format!("entry {:#x} arg {:#x} stack {:#x} priority {} core {} -> {h:#x}", a[1], a[2], a[3], a[4] as u32, a[5] as i32))
                }
                Err(e) => (e, 0, format!("priority {} core {}", a[4] as u32, a[5] as i32)),
            },
            0x09 => (self.start_thread(a[0] as u32), 0, format!("{:#x}", a[0] as u32)),
            0x0A => {
                self.end_current();
                (SUCCESS, 0, String::new())
            }
            0x0B => {
                // A long sleep (a frame or more): its caller's stack, once per call site.
                if a[0] as i64 >= 20_000_000 && self.journal.first(&format!("long sleep {:#x}", self.frame.1)) {
                    self.log(&format!("sleep of {} ns, from:", a[0] as i64));
                    for line in self.backtrace() {
                        self.log(&line);
                    }
                }
                self.sleep(a[0] as i64);
                (SUCCESS, u64::from(a[0] as i64 <= 0), format!("{}", a[0] as i64))
            }
            0x0C => match self.thread_of(a[1] as u32) {
                Some(t) => {
                    cpu.x[1] = u64::from(self.sched.threads[t].priority);
                    (SUCCESS, 0, format!("{:#x} -> {}", a[1] as u32, cpu.x[1]))
                }
                None => (INVALID_HANDLE, 0, format!("{:#x}", a[1] as u32)),
            },
            0x0D => (self.set_priority(a[0] as u32, a[1] as u32), 0, format!("{:#x} priority {}", a[0] as u32, a[1] as u32)),
            0x0E => match self.thread_of(a[2] as u32) {
                Some(t) => {
                    let th = &self.sched.threads[t];
                    (cpu.x[1], cpu.x[2]) = (u64::from(th.ideal as u32), th.affinity);
                    (SUCCESS, 0, format!("{:#x} -> core {} mask {:#x}", a[2] as u32, cpu.x[1], cpu.x[2]))
                }
                None => (INVALID_HANDLE, 0, format!("{:#x}", a[2] as u32)),
            },
            0x0F => (self.set_core_mask(a[0] as u32, a[1] as i32, a[2]), 0, format!("{:#x} core {} mask {:#x}", a[0] as u32, a[1] as i32, a[2])),
            0x10 => {
                // w0 is the core number, not a Result.
                let core = self.sched.current.map_or(0, |t| self.sched.threads[t].core);
                cpu.x[0] = core as u64;
                if self.journal.all_svcs || self.journal.first("0x10") {
                    self.log(&format!("{call} -> {core}"));
                }
                return;
            }
            0x25 => match self.thread_of(a[1] as u32) {
                Some(t) => {
                    cpu.x[1] = self.sched.threads[t].id;
                    (SUCCESS, 0, format!("{:#x} -> {}", a[1] as u32, cpu.x[1]))
                }
                None => (INVALID_HANDLE, 0, format!("{:#x}", a[1] as u32)),
            },
            0x36 => (SUCCESS, 0, String::new()),
            0x15 => match self.create_transfer(a[1], a[2], a[3] as u32) {
                Ok(h) => {
                    cpu.x[1] = u64::from(h);
                    (SUCCESS, 0, format!("{:#x}+{:#x} owner keeps perm {} -> {h:#x}", a[1], a[2], a[3] as u32))
                }
                Err(e) => (e, 0, format!("{:#x}+{:#x} perm {}", a[1], a[2], a[3] as u32)),
            },
            0x13 => (self.map_shared(a[0] as u32, a[1], a[2], a[3] as u32), 0, format!("{:#x} at {:#x}+{:#x} perm {}", a[0] as u32, a[1], a[2], a[3] as u32)),
            0x14 => (self.unmap_shared(a[0] as u32, a[1], a[2]), 0, format!("{:#x} at {:#x}+{:#x}", a[0] as u32, a[1], a[2])),
            0x1F => match self.connect_to_named_port(a[1]) {
                Ok(h) => {
                    cpu.x[1] = u64::from(h);
                    (SUCCESS, 0, format!("-> {h:#x}"))
                }
                Err(e) => (e, 0, String::new()),
            },
            0x21 => (self.send_request(a[0] as u32, cpu.tpidrro, 0x100), 0, format!("{:#x}", a[0] as u32)),
            0x22 => (self.send_request(a[2] as u32, a[0], a[1]), 0, format!("{:#x} buffer {:#x}+{:#x}", a[2] as u32, a[0], a[1])),
            0x11 => (self.signal_event_handle(a[0] as u32), 0, format!("{:#x}", a[0] as u32)),
            0x12 => (self.clear_event(a[0] as u32), 0, format!("{:#x}", a[0] as u32)),
            0x16 => (self.close_handle(a[0] as u32), 0, format!("{:#x}", a[0] as u32)),
            0x17 => (self.reset_signal(a[0] as u32), 0, format!("{:#x}", a[0] as u32)),
            0x18 => {
                let mut first = [0; 4];
                let _ = self.mem.read(a[1], &mut first);
                let h = u32::from_le_bytes(first);
                let state = match self.handles.get(h) {
                    Some(crate::thread::Object::ReadableEvent(e)) => format!(", event {e} signaled {}", self.events[e].signaled),
                    _ => String::new(),
                };
                let detail = format!("{} handles at {:#x} (first {h:#x}{state}) timeout {}", a[2] as u32, a[1], a[3] as i64);
                (self.wait_synchronization(cpu, a[1], a[2] as u32, a[3] as i64), 0, detail)
            }
            0x19 => (self.cancel_synchronization(a[0] as u32), 0, format!("{:#x}", a[0] as u32)),
            0x1A => (self.arbitrate_lock(a[0] as u32, a[1], a[2] as u32), 0, format!("owner {:#x} at {:#x} tag {:#x}", a[0] as u32, a[1], a[2] as u32)),
            0x1B => (self.arbitrate_unlock(a[0]), 0, format!("{:#x}", a[0])),
            0x1C => (self.wait_key(a[0], a[1], a[2] as u32, a[3] as i64), 0, format!("mutex {:#x} key {:#x} tag {:#x} timeout {}", a[0], a[1], a[2] as u32, a[3] as i64)),
            0x1D => {
                self.signal_key(a[0], a[1] as i32);
                (SUCCESS, 0, format!("key {:#x} count {}", a[0], a[1] as i32))
            }
            0x34 => (self.wait_address(a[0], a[1] as u32, a[2], a[3] as i64), u64::from(a[1] as u32), format!("{:#x} type {} value {:#x} timeout {}", a[0], a[1] as u32, a[2], a[3] as i64)),
            0x35 => (self.signal_address(a[0], a[1] as u32, a[2] as i32, a[3] as i32), u64::from(a[1] as u32), format!("{:#x} type {} value {} count {}", a[0], a[1] as u32, a[2] as i32, a[3] as i32)),
            0x26 => {
                let reason = a[0] as u32;
                if reason & 0x8000_0000 == 0 {
                    self.stop(STOP_EXIT, &format!("{call} at {pc:#x}: the guest aborts, reason {reason:#x} info {:#x}+{:#x}", a[1], a[2]));
                }
                (SUCCESS, u64::from(reason), format!("notification {reason:#x}"))
            }
            0x27 => {
                let mut text = vec![0; a[1].min(4096) as usize];
                let r = if self.mem.read(a[0], &mut text).is_ok() { SUCCESS } else { INVALID_POINTER };
                (r, u64::MAX, format!("{:?}", String::from_utf8_lossy(&text).trim_end()))
            }
            0x29 => {
                let (r, value) = self.info(a[1] as u32, a[2] as u32, a[3]);
                cpu.x[1] = value;
                (r, a[1] << 32 | (a[3] & 0xFFFF_FFFF), format!("type {:#x} handle {:#x} sub {:#x} -> {value:#x}", a[1] as u32, a[2] as u32, a[3]))
            }
            0x2C => (self.map_physical(a[0], a[1]), 0, format!("{:#x}+{:#x}", a[0], a[1])),
            0x2D => (self.unmap_physical(a[0], a[1]), 0, format!("{:#x}+{:#x}", a[0], a[1])),
            0x80 => (self.set_attribute(a[0], a[1], a[2] as u32, a[3] as u32, attr::GPU_SHARABLE), 0, format!("{:#x}+{:#x} mask {:#x} value {:#x}", a[0], a[1], a[2] as u32, a[3] as u32)),
            _ => {
                let what = format!("{call} at {pc:#x} is not implemented;{}", args(&a));
                if !self.config.explore {
                    self.stop(STOP_EXIT, &what);
                }
                self.log(&what);
                cpu.x[0] = u64::from(NOT_IMPLEMENTED);
                return;
            }
        };
        cpu.x[0] = u64::from(result);
        // OutputDebugString (kind u64::MAX) is always logged, like every
        // failure but those the SDK expects of futexes and waits.
        let expected = matches!(result, SUCCESS | INVALID_STATE | TIMED_OUT | CANCELLED);
        let key = format!("{imm:#x} {kind:#x} {result:#x}");
        // Unexpected failures: their first four occurrences.
        if self.journal.all_svcs || self.journal.first(&key) || kind == u64::MAX || (!expected && self.journal.few(&key, 4)) {
            let status = if result == SUCCESS { String::new() } else { format!(" -> {result:#x}") };
            self.log(&format!("{call} {detail}{status}"));
        }
    }

    fn start_thread(&mut self, h: u32) -> u32 {
        match self.thread_of(h) {
            Some(t) if self.sched.threads[t].state == State::Created => {
                let now = self.now();
                self.ready(t, now);
                SUCCESS
            }
            Some(_) => INVALID_STATE,
            None => INVALID_HANDLE,
        }
    }

    /// SleepThread: 0 yields to threads of the same priority, -1 also pulls
    /// a waiting thread from another core, anything positive sleeps that
    /// many nanoseconds (a tick is 32 ns).
    pub(crate) fn sleep(&mut self, ns: i64) {
        let t = self.sched.current.expect("a running thread");
        let now = self.now();
        if ns == -1 {
            self.yield_with_migration(t);
        } else if ns <= 0 {
            // ponytail: -2 (yield to any thread) behaves as 0; the SDK only uses -1.
            self.ready(t, now);
        } else {
            self.block(Wait::Sleep, Some(now + ns_to_ticks(ns)), SUCCESS);
        }
    }

    fn set_priority(&mut self, h: u32, priority: u32) -> u32 {
        let Some(t) = self.thread_of(h) else { return INVALID_HANDLE };
        if !PRIORITIES.contains(&priority) {
            return INVALID_PRIORITY;
        }
        self.sched.threads[t].priority = priority;
        SUCCESS
    }

    /// SetThreadCoreMask. `core` -1: no preferred core; -2: the process's
    /// default core (0), alone in the mask, whatever `mask` says; -3: keep
    /// the ideal core, which must be in the mask. A thread off its new mask
    /// moves to its ideal core, or to the highest core of the mask.
    fn set_core_mask(&mut self, h: u32, core: i32, mask: u64) -> u32 {
        let Some(t) = self.thread_of(h) else { return INVALID_HANDLE };
        let (ideal, mask) = if core == -2 {
            (0, 1)
        } else {
            if mask & !((1u64 << CORES) - 1) != 0 {
                return INVALID_CORE_ID;
            }
            if mask == 0 {
                return INVALID_COMBINATION;
            }
            let ideal = match core {
                0..=5 => core,
                -3 => self.sched.threads[t].ideal,
                -1 => -1,
                _ => return INVALID_CORE_ID,
            };
            if ideal >= 0 && mask & 1 << ideal == 0 {
                return INVALID_COMBINATION;
            }
            (ideal, mask)
        };
        let th = &mut self.sched.threads[t];
        (th.ideal, th.affinity) = (ideal, mask);
        if mask & 1 << th.core == 0 {
            let to = if ideal >= 0 { ideal as usize } else { 63 - mask.leading_zeros() as usize };
            self.move_thread(t, to);
        }
        SUCCESS
    }

    fn set_permission(&mut self, addr: u64, size: u64, p: u32) -> u32 {
        if !matches!(p, 0 | perm::R | perm::RW) {
            return INVALID_NEW_MEMORY_PERMISSION;
        }
        done(self.mem.protect(addr, size, p))
    }

    /// SetMemoryAttribute and SetGpuMemoryAttribute: `mask` must stay within `allowed`.
    fn set_attribute(&mut self, addr: u64, size: u64, mask: u32, value: u32, allowed: u32) -> u32 {
        if mask & !allowed != 0 || value & !mask != 0 {
            return INVALID_COMBINATION;
        }
        done(self.mem.set_attr(addr, size, mask, value))
    }

    /// MapMemory moves heap or alias memory to the stack region.
    fn map_memory(&mut self, dst: u64, src: u64, size: u64) -> u32 {
        if !layout::contains(layout::STACK, dst, size) {
            return INVALID_MEMORY_REGION;
        }
        done(self.mem.map_move(dst, src, size, state::STACK))
    }

    fn unmap_memory(&mut self, dst: u64, src: u64, size: u64) -> u32 {
        if !layout::contains(layout::STACK, dst, size) {
            return INVALID_MEMORY_REGION;
        }
        done(self.mem.unmap_move(dst, src, size, state::STACK))
    }

    /// MapPhysicalMemory backs every unmapped page of the range, which must
    /// lie in the alias region (the SDK's VAMM).
    fn map_physical(&mut self, addr: u64, size: u64) -> u32 {
        self.physical(addr, size, true)
    }

    /// UnmapPhysicalMemory releases every page of the range that it backed.
    fn unmap_physical(&mut self, addr: u64, size: u64) -> u32 {
        self.physical(addr, size, false)
    }

    fn physical(&mut self, addr: u64, size: u64, map: bool) -> u32 {
        if !addr.is_multiple_of(mem::PAGE) || !size.is_multiple_of(mem::PAGE) || size == 0 {
            return INVALID_ADDRESS;
        }
        if !layout::contains(layout::ALIAS, addr, size) {
            return INVALID_MEMORY_REGION;
        }
        let end = addr + size;
        // The blocks of the range, collected first: mapping changes the table.
        let mut parts = Vec::new();
        let mut at = addr;
        while at < end {
            let info = self.mem.query(at);
            let to = (info.base + info.size).min(end);
            parts.push((at, to - at, info.block.state));
            at = to;
        }
        for (start, len, st) in parts {
            let r = match (map, st) {
                (true, state::FREE) => self.mem.map(start, len, Block::new(state::NORMAL, perm::RW)),
                (false, state::NORMAL) => self.mem.unmap(start, len),
                (_, state::FREE | state::NORMAL) => Ok(()),
                _ => return INVALID_CURRENT_MEMORY,
            };
            if let Err(e) = r {
                return mem_result(e);
            }
        }
        SUCCESS
    }

    /// GetInfo: fixed values of the platform (D18, D21), recorded in the journal.
    fn info(&mut self, kind: u32, handle: u32, sub: u64) -> (u32, u64) {
        let handle = u64::from(handle);
        let process = |v: u64| if handle == CURRENT_PROCESS { (SUCCESS, v) } else { (INVALID_HANDLE, 0) };
        let none = |v: u64| if handle == 0 { (SUCCESS, v) } else { (INVALID_HANDLE, 0) };
        match kind {
            0x00 => process(0x3F),
            0x02 => process(layout::ALIAS.0),
            0x03 => process(layout::ALIAS.1),
            0x04 => process(layout::HEAP.0),
            0x05 => process(layout::HEAP.1),
            0x08 => none(0),
            // ponytail: no idle time until the scheduler exists.
            0x0A => none(0),
            0x0B if sub < 4 => none(splitmix(self.config.seed.wrapping_add(sub))),
            0x0C => process(layout::ASLR.0),
            0x0D => process(layout::ASLR.1),
            0x0E => process(layout::STACK.0),
            0x0F => process(layout::STACK.1),
            // Non-zero, or the SDK leaves its VAMM off.
            0x10 => process(0x100_0000),
            0x11 => process(0),
            0x14 => process(layout::THREAD_LOCAL.0),
            0x15 => process(self.config.non_system_memory),
            0x16 => process(self.mem.blocks().map(|b| b.size).sum()),
            0x17 => process(1),
            0x18 => process(THREAD_LIMIT - self.sched.threads.iter().filter(|t| t.state != State::Done).count() as u64),
            0x1C => process(0),
            0x1D => process(layout::REMOTE.0),
            0x1E => process(layout::REMOTE.1),
            0x1F | 0x20 => process(0),
            // rtld requires module segments aligned to it, and they are aligned to 4 KiB.
            0x21 => process(mem::PAGE),
            _ => (INVALID_ENUM_VALUE, 0),
        }
    }
}

/// A 64-bit mix of `x` (splitmix64's finalizer): fixed entropy from the seed.
fn splitmix(x: u64) -> u64 {
    let mut z = x.wrapping_add(0x9E37_79B9_7F4A_7C15);
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

/// Nanoseconds to ticks of 31.25 MHz (32 ns each), rounded up.
pub fn ns_to_ticks(ns: i64) -> u64 {
    (ns as u64).div_ceil(32)
}
