//! Guest memory: one host reservation for the whole 39-bit guest address
//! space, and the block table that answers `svcQueryMemory`.
//!
//! The table is the truth: it gives each mapped range its Horizon state,
//! permission and attribute, and it lists the pages a save state must keep.
//! Host pages are committed when the guest maps them, and their host
//! protection follows the guest permission (see [`Block::host_protection`]).
//! A host access violation inside the reservation becomes a logged guest
//! fault (see [`install_fault_handler`]).
//!
//! One `Memory` serves every host thread: the block table sits behind a lock,
//! and the HLE reaches guest bytes through checked accesses that run once the
//! check has released it.

use std::collections::BTreeMap;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{RwLock, RwLockReadGuard, RwLockWriteGuard};

use windows_sys::Win32::Foundation::EXCEPTION_ACCESS_VIOLATION;
use windows_sys::Win32::System::Diagnostics::Debug::{AddVectoredExceptionHandler, EXCEPTION_CONTINUE_SEARCH, EXCEPTION_POINTERS};
/// MEM_WRITE_WATCH (winnt.h), which windows-sys does not list.
const MEM_WRITE_WATCH: windows_sys::Win32::System::Memory::VIRTUAL_ALLOCATION_TYPE = 0x0020_0000;
use windows_sys::Win32::System::Memory::{
    GetWriteWatch, MEM_COMMIT, MEM_DECOMMIT, MEM_RELEASE, MEM_RESERVE, PAGE_NOACCESS, PAGE_PROTECTION_FLAGS, PAGE_READONLY, PAGE_READWRITE, ResetWriteWatch, VirtualAlloc, VirtualFree, VirtualProtect,
};

/// Size of the guest address space (39 bits, from the NPDM).
pub const SPACE: u64 = 1 << 39;
pub const PAGE: u64 = 0x1000;
/// Reserved after the space and never committed (wider than any access).
const GUARD: u64 = 0x1_0000;

/// Horizon memory states (the low byte, as `svcQueryMemory` reports it).
pub mod state {
    pub const FREE: u32 = 0x00;
    pub const CODE: u32 = 0x03;
    pub const CODE_DATA: u32 = 0x04;
    pub const NORMAL: u32 = 0x05;
    pub const SHARED: u32 = 0x06;
    pub const STACK: u32 = 0x0B;
    pub const THREAD_LOCAL: u32 = 0x0C;
    pub const INACCESSIBLE: u32 = 0x10;
}

/// Horizon permission bits.
pub mod perm {
    pub const R: u32 = 1;
    pub const W: u32 = 2;
    pub const X: u32 = 4;
    pub const RW: u32 = R | W;
    pub const RX: u32 = R | X;
}

/// Horizon attribute bits.
pub mod attr {
    pub const LOCKED: u32 = 0x01;
    pub const IPC_LOCKED: u32 = 0x02;
    pub const DEVICE_SHARED: u32 = 0x04;
    pub const UNCACHED: u32 = 0x08;
    pub const PERMISSION_LOCKED: u32 = 0x10;
    /// Switch 2: set by `svcSetGpuMemoryAttribute` (0x80).
    pub const GPU_SHARABLE: u32 = 0x20;
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Block {
    pub state: u32,
    pub perm: u32,
    pub attr: u32,
}

impl Block {
    pub const fn new(state: u32, perm: u32) -> Block {
        Block { state, perm, attr: 0 }
    }

    /// The host protection of a mapped block. Shared memory and locked
    /// pages (transfer memory, the source of a `svcMapMemory`) stay writable
    /// on the host even when the guest only has R: the HLE writes into them.
    fn host_protection(&self) -> PAGE_PROTECTION_FLAGS {
        if self.perm & perm::W != 0 || self.state == state::SHARED || self.attr & attr::LOCKED != 0 {
            PAGE_READWRITE
        } else if self.perm & perm::R != 0 {
            PAGE_READONLY
        } else {
            PAGE_NOACCESS
        }
    }
}

/// What `svcQueryMemory` reports for one block.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Info {
    pub base: u64,
    pub size: u64,
    pub block: Block,
}

impl Info {
    /// The guest `MemoryInfo` structure (0x28 bytes): base, size, state,
    /// attribute, permission, IPC and device reference counts, padding.
    pub fn to_bytes(&self) -> [u8; 0x28] {
        let mut b = [0; 0x28];
        b[0..8].copy_from_slice(&self.base.to_le_bytes());
        b[8..16].copy_from_slice(&self.size.to_le_bytes());
        b[16..20].copy_from_slice(&self.block.state.to_le_bytes());
        b[20..24].copy_from_slice(&self.block.attr.to_le_bytes());
        b[24..28].copy_from_slice(&self.block.perm.to_le_bytes());
        b
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Error {
    /// Address or size not page-aligned, empty, or outside the address space.
    InvalidRange,
    /// Part of the range is already mapped.
    NotFree,
    /// Part of the range is not mapped, or not in the expected state.
    InvalidState,
    /// The host refused to commit or protect (Windows error code).
    Host(u32),
}

/// The guest address space, shared by every host thread that runs guest code
/// or the HLE. The block table sits behind a lock: map, unmap, protect and
/// attribute changes take it to write, checks take it to read, and the access
/// a check allows runs once it is released. Changes made of several calls
/// (query, then map) rely on the kernel lock, under which every table change
/// runs.
pub struct Memory {
    base: *mut u8,
    blocks: RwLock<Table>,
}

// SAFETY: `base` never changes and the reservation is ours until drop; the
// table is behind its lock. Guest bytes are shared with guest code on other
// host threads by design: Memory hands out no Rust reference to them that
// outlives one access, only raw copies, as the generated code's own accesses.
unsafe impl Send for Memory {}
unsafe impl Sync for Memory {}

/// Mapped blocks only: start -> (end, block). Gaps are Free.
#[derive(Default)]
struct Table(BTreeMap<u64, (u64, Block)>);

/// Base of the reservation the fault handler reports on (0 when none).
static RESERVATION: AtomicUsize = AtomicUsize::new(0);

impl Memory {
    /// Reserves the address space, plus a guard that is never committed: an
    /// access that starts below 2^39 and runs past it faults inside the
    /// reservation. Nothing is committed yet. The fault handler reports on
    /// the newest reservation.
    pub fn new() -> Result<Memory, Error> {
        // SAFETY: a fresh reservation, no existing memory is touched.
        // Write-watched: the kernel notes the pages written, so the host knows when
        // memory it copied (textures) changed, without faults or hashing.
        let base = unsafe { VirtualAlloc(std::ptr::null(), (SPACE + GUARD) as usize, MEM_RESERVE | MEM_WRITE_WATCH, PAGE_NOACCESS) } as *mut u8;
        if base.is_null() {
            return Err(Error::Host(last_error()));
        }
        RESERVATION.store(base as usize, Ordering::Release);
        Ok(Memory { base, blocks: RwLock::default() })
    }

    // The table's lock, recovered from poison like the crate's other locks.
    fn table(&self) -> RwLockReadGuard<'_, Table> {
        self.blocks.read().unwrap_or_else(|e| e.into_inner())
    }

    fn table_mut(&self) -> RwLockWriteGuard<'_, Table> {
        self.blocks.write().unwrap_or_else(|e| e.into_inner())
    }

    /// Whether a page of `[addr, addr + len)` was written since its watch was
    /// last reset (by the guest or the host alike).
    pub fn written(&self, addr: u64, len: u64) -> bool {
        let Some(end) = addr.checked_add(len).filter(|&e| e <= SPACE && len > 0) else { return false };
        let (mut page, mut count, mut granularity) = (std::ptr::null_mut::<core::ffi::c_void>(), 1usize, 0u32);
        // SAFETY: a range of our reservation; room for one address.
        let r = unsafe { GetWriteWatch(0, self.base.add(addr as usize).cast(), (end - addr) as usize, &mut page, &mut count, &mut granularity) };
        r == 0 && count > 0
    }

    /// Starts watching `[addr, addr + len)` afresh.
    pub fn reset_written(&self, addr: u64, len: u64) {
        if addr.checked_add(len).is_some_and(|e| e <= SPACE) && len > 0 {
            // SAFETY: a range of our reservation.
            unsafe { ResetWriteWatch(self.base.add(addr as usize).cast(), len as usize) };
        }
    }

    /// Host address of guest address 0 (`Cpu.mem`).
    pub fn base(&self) -> *mut u8 {
        self.base
    }

    /// The block that contains `addr`, as `svcQueryMemory` reports it: Free
    /// between mappings, and Inaccessible from the end of the space to 2^64.
    pub fn query(&self, addr: u64) -> Info {
        self.table().query(addr)
    }

    /// Maps fresh zeroed pages over a Free range.
    pub fn map(&self, addr: u64, size: u64, block: Block) -> Result<(), Error> {
        self.map_in(&mut self.table_mut(), addr, size, block)
    }

    /// Unmaps a fully mapped range; its pages are decommitted.
    pub fn unmap(&self, addr: u64, size: u64) -> Result<(), Error> {
        self.unmap_in(&mut self.table_mut(), addr, size)
    }

    /// Changes the permission of a fully mapped range.
    pub fn protect(&self, addr: u64, size: u64, perm: u32) -> Result<(), Error> {
        self.update_in(&mut self.table_mut(), addr, size, |b| b.perm = perm)
    }

    /// Sets the attribute bits in `mask` to `value` over a fully mapped range.
    pub fn set_attr(&self, addr: u64, size: u64, mask: u32, value: u32) -> Result<(), Error> {
        self.update_in(&mut self.table_mut(), addr, size, |b| b.attr = (b.attr & !mask) | (value & mask))
    }

    /// `svcMapMemory`: moves `size` bytes from `src` (mapped RW, no
    /// attribute) to the Free range at `dst`, which takes `dst_state`. The
    /// source stays in place, without permission and locked.
    pub fn map_move(&self, dst: u64, src: u64, size: u64, dst_state: u32) -> Result<(), Error> {
        let mut t = self.table_mut();
        t.expect(src, size, |b| b.perm == perm::RW && b.attr == 0)?;
        self.map_in(&mut t, dst, size, Block::new(dst_state, perm::RW))?;
        // SAFETY: both ranges are committed, readable and writable, and distinct (dst was Free).
        // Under the write lock: no check sees the move half done.
        unsafe { std::ptr::copy_nonoverlapping(self.base.add(src as usize), self.base.add(dst as usize), size as usize) };
        self.update_in(&mut t, src, size, |b| (b.perm, b.attr) = (0, attr::LOCKED))
    }

    /// `svcUnmapMemory`: the inverse of [`Memory::map_move`]. Nothing
    /// changes unless `dst` is in `dst_state` without attribute and `src`
    /// is locked: then the two ranges cannot overlap.
    pub fn unmap_move(&self, dst: u64, src: u64, size: u64, dst_state: u32) -> Result<(), Error> {
        let mut t = self.table_mut();
        t.expect(src, size, |b| b.perm == 0 && b.attr == attr::LOCKED)?;
        t.expect(dst, size, |b| b.state == dst_state && b.perm == perm::RW && b.attr == 0)?;
        // SAFETY: both ranges are committed and host-writable (locked pages stay
        // RW on the host), and distinct: one is locked, the other is not.
        unsafe { std::ptr::copy_nonoverlapping(self.base.add(dst as usize), self.base.add(src as usize), size as usize) };
        self.unmap_in(&mut t, dst, size)?;
        self.update_in(&mut t, src, size, |b| (b.perm, b.attr) = (perm::RW, 0))
    }

    /// Reads a guest-supplied buffer for the HLE. Every byte must be mapped
    /// with guest permission R: guest addresses are checked here, never trusted.
    pub fn read(&self, addr: u64, out: &mut [u8]) -> Result<(), Error> {
        self.accessible(addr, out.len() as u64, perm::R)?;
        // SAFETY: checked just above.
        unsafe { std::ptr::copy_nonoverlapping(self.base.add(addr as usize), out.as_mut_ptr(), out.len()) };
        Ok(())
    }

    /// Writes a guest-supplied buffer for the HLE; every byte needs guest
    /// permission W.
    pub fn write(&self, addr: u64, data: &[u8]) -> Result<(), Error> {
        self.accessible(addr, data.len() as u64, perm::W)?;
        // SAFETY: checked just above.
        unsafe { std::ptr::copy_nonoverlapping(data.as_ptr(), self.base.add(addr as usize), data.len()) };
        Ok(())
    }

    /// Writes memory the HLE owns (shared memory, locked pages): the guest
    /// may only have R there, but the host pages are writable.
    pub fn write_object(&self, addr: u64, data: &[u8]) -> Result<(), Error> {
        self.table().host_accessible(addr, data.len() as u64, PAGE_READWRITE)?;
        // SAFETY: checked just above.
        unsafe { std::ptr::copy_nonoverlapping(data.as_ptr(), self.base.add(addr as usize), data.len()) };
        Ok(())
    }

    /// Reads memory the HLE owns, whatever the guest's permission.
    pub fn read_object(&self, addr: u64, out: &mut [u8]) -> Result<(), Error> {
        self.table().host_accessible(addr, out.len() as u64, PAGE_READWRITE)?;
        // SAFETY: checked just above.
        unsafe { std::ptr::copy_nonoverlapping(self.base.add(addr as usize), out.as_mut_ptr(), out.len()) };
        Ok(())
    }

    /// Whether every byte of `[addr, addr + len)` is mapped with the guest
    /// permission `need`: the check `read` (R) and `write` (W) make.
    pub fn accessible(&self, addr: u64, len: u64, need: u32) -> Result<(), Error> {
        self.table().accessible(addr, len, need)
    }

    /// The mapped blocks, in address order: a snapshot taken under the read lock.
    pub fn blocks(&self) -> impl Iterator<Item = Info> {
        let blocks: Vec<Info> = self.table().0.iter().map(|(&base, &(end, block))| Info { base, size: end - base, block }).collect();
        blocks.into_iter()
    }

    // The bodies of the table changes, on the write guard their caller holds
    // (the lock is not re-entrant).

    fn map_in(&self, t: &mut Table, addr: u64, size: u64, block: Block) -> Result<(), Error> {
        let end = check(addr, size)?;
        if block.state == state::FREE {
            return Err(Error::InvalidState);
        }
        if t.0.range(..end).next_back().is_some_and(|(_, &(e, _))| e > addr) {
            return Err(Error::NotFree);
        }
        // SAFETY: the range is inside our reservation and unmapped.
        if unsafe { VirtualAlloc(self.base.add(addr as usize).cast(), size as usize, MEM_COMMIT, block.host_protection()) }.is_null() {
            return Err(Error::Host(last_error()));
        }
        t.0.insert(addr, (end, block));
        t.merge(addr, end);
        Ok(())
    }

    fn unmap_in(&self, t: &mut Table, addr: u64, size: u64) -> Result<(), Error> {
        let end = t.mapped(addr, size)?;
        t.split(addr);
        t.split(end);
        let starts: Vec<u64> = t.0.range(addr..end).map(|(&s, _)| s).collect();
        for s in starts {
            t.0.remove(&s);
        }
        // SAFETY: committed pages of our reservation.
        if unsafe { VirtualFree(self.base.add(addr as usize).cast(), size as usize, MEM_DECOMMIT) } == 0 {
            return Err(Error::Host(last_error()));
        }
        Ok(())
    }

    /// Applies `f` to every block of a fully mapped range, then reprotects it.
    fn update_in(&self, t: &mut Table, addr: u64, size: u64, f: impl Fn(&mut Block)) -> Result<(), Error> {
        let end = t.mapped(addr, size)?;
        t.split(addr);
        t.split(end);
        for (&start, (e, block)) in t.0.range_mut(addr..end) {
            f(block);
            let mut old = 0;
            // SAFETY: committed pages of our reservation.
            if unsafe { VirtualProtect(self.base.add(start as usize).cast(), (*e - start) as usize, block.host_protection(), &mut old) } == 0 {
                return Err(Error::Host(last_error()));
            }
        }
        t.merge(addr, end);
        Ok(())
    }
}

impl Table {
    fn query(&self, addr: u64) -> Info {
        if addr >= SPACE {
            return Info { base: SPACE, size: SPACE.wrapping_neg(), block: Block::new(state::INACCESSIBLE, 0) };
        }
        let before = self.0.range(..=addr).next_back();
        if let Some((&start, &(end, block))) = before
            && addr < end
        {
            return Info { base: start, size: end - start, block };
        }
        let start = before.map_or(0, |(_, &(end, _))| end);
        let end = self.0.range(addr..).next().map_or(SPACE, |(&s, _)| s);
        Info { base: start, size: end - start, block: Block::new(state::FREE, 0) }
    }

    fn accessible(&self, addr: u64, len: u64, need: u32) -> Result<(), Error> {
        let end = addr.checked_add(len).filter(|&e| e <= SPACE).ok_or(Error::InvalidRange)?;
        let mut at = addr;
        while at < end {
            let info = self.query(at);
            if info.block.state == state::FREE || info.block.perm & need != need {
                return Err(Error::InvalidState);
            }
            at = info.base + info.size;
        }
        Ok(())
    }

    fn host_accessible(&self, addr: u64, len: u64, protection: PAGE_PROTECTION_FLAGS) -> Result<(), Error> {
        let end = addr.checked_add(len).filter(|&e| e <= SPACE).ok_or(Error::InvalidRange)?;
        let mut at = addr;
        while at < end {
            let info = self.query(at);
            if info.block.state == state::FREE || info.block.host_protection() != protection {
                return Err(Error::InvalidState);
            }
            at = info.base + info.size;
        }
        Ok(())
    }

    /// Checks that `[addr, addr + size)` is entirely mapped; returns its end.
    fn mapped(&self, addr: u64, size: u64) -> Result<u64, Error> {
        self.expect(addr, size, |_| true)
    }

    /// Checks that the range is entirely mapped by blocks that satisfy `ok`.
    fn expect(&self, addr: u64, size: u64, ok: impl Fn(&Block) -> bool) -> Result<u64, Error> {
        let end = check(addr, size)?;
        let mut at = addr;
        while at < end {
            let info = self.query(at);
            if info.block.state == state::FREE || !ok(&info.block) {
                return Err(Error::InvalidState);
            }
            at = info.base + info.size;
        }
        Ok(end)
    }

    /// Cuts the block that strictly contains `at` in two.
    fn split(&mut self, at: u64) {
        if let Some((&start, &(end, block))) = self.0.range(..at).next_back()
            && at < end
        {
            self.0.insert(start, (at, block));
            self.0.insert(at, (end, block));
        }
    }

    /// Joins equal blocks that touch, around `[addr, end)`, as the kernel does.
    fn merge(&mut self, addr: u64, end: u64) {
        let first = self.0.range(..addr).next_back().map_or(addr, |(&s, _)| s);
        let starts: Vec<u64> = self.0.range(first..=end).map(|(&s, _)| s).collect();
        let mut keep = starts[0];
        for &s in &starts[1..] {
            let (kend, kblock) = self.0[&keep];
            let (send, sblock) = self.0[&s];
            if kend == s && kblock == sblock {
                self.0.remove(&s);
                self.0.insert(keep, (send, kblock));
            } else {
                keep = s;
            }
        }
    }
}

impl Drop for Memory {
    fn drop(&mut self) {
        let _ = RESERVATION.compare_exchange(self.base as usize, 0, Ordering::AcqRel, Ordering::Acquire);
        // SAFETY: releases our own reservation; nothing points into it any more.
        unsafe { VirtualFree(self.base.cast(), 0, MEM_RELEASE) };
    }
}

fn check(addr: u64, size: u64) -> Result<u64, Error> {
    let end = addr.checked_add(size).ok_or(Error::InvalidRange)?;
    if size == 0 || !addr.is_multiple_of(PAGE) || !size.is_multiple_of(PAGE) || end > SPACE {
        return Err(Error::InvalidRange);
    }
    Ok(end)
}

fn last_error() -> u32 {
    std::io::Error::last_os_error().raw_os_error().unwrap_or(0) as u32
}

/// Installs the handler that turns a host access violation inside the guest
/// reservation into a logged guest fault, then stops the process. Other
/// exceptions go to the next handler.
pub fn install_fault_handler() {
    static ONCE: std::sync::Once = std::sync::Once::new();
    // SAFETY: registers a handler that only reads the exception record.
    ONCE.call_once(|| unsafe {
        AddVectoredExceptionHandler(1, Some(on_exception));
    });
}

/// Exit code of a process stopped by a guest fault.
pub const FAULT_EXIT: i32 = 0x47;

unsafe extern "system" fn on_exception(info: *mut EXCEPTION_POINTERS) -> i32 {
    // SAFETY: Windows passes a valid record.
    let rec = unsafe { &*(*info).ExceptionRecord };
    let base = RESERVATION.load(Ordering::Acquire);
    if rec.ExceptionCode != EXCEPTION_ACCESS_VIOLATION || base == 0 || rec.NumberParameters < 2 {
        return EXCEPTION_CONTINUE_SEARCH;
    }
    let host = rec.ExceptionInformation[1];
    if host < base || host - base >= (SPACE + GUARD) as usize {
        return EXCEPTION_CONTINUE_SEARCH;
    }
    let kind = match rec.ExceptionInformation[0] {
        0 => "read",
        1 => "write",
        _ => "execute",
    };
    let rip = rec.ExceptionAddress as u64;
    let near = crate::function_at_host(rip).map_or_else(|| "none".to_string(), |g| format!("{g:#x}"));
    let what = format!("guest fault: {kind} at guest address {:#x} (nearest recompiled function: {near})", host - base);
    eprintln!("gyroship: {what}");
    if let Some(h) = crate::HOOKS.get() {
        (h.fault)(&what);
    }
    std::process::exit(FAULT_EXIT);
}
