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
use std::sync::atomic::{AtomicBool, AtomicPtr, AtomicU32, AtomicU64, AtomicUsize, Ordering};
use std::sync::{OnceLock, RwLock, RwLockReadGuard, RwLockWriteGuard};

use windows_sys::Win32::Foundation::EXCEPTION_ACCESS_VIOLATION;
use windows_sys::Win32::System::Diagnostics::Debug::{AddVectoredExceptionHandler, EXCEPTION_CONTINUE_EXECUTION, EXCEPTION_CONTINUE_SEARCH, EXCEPTION_POINTERS};
/// MEM_WRITE_WATCH (winnt.h) and WRITE_WATCH_FLAG_RESET (memoryapi.h):
/// windows-sys lists them under SystemServices, a feature this crate does not enable.
const MEM_WRITE_WATCH: windows_sys::Win32::System::Memory::VIRTUAL_ALLOCATION_TYPE = 0x0020_0000;
const WRITE_WATCH_FLAG_RESET: u32 = 0x01;
use windows_sys::Win32::System::Memory::{
    GetWriteWatch, MEM_COMMIT, MEM_DECOMMIT, MEM_RELEASE, MEM_RESERVE, MEMORY_BASIC_INFORMATION, PAGE_NOACCESS, PAGE_PROTECTION_FLAGS, PAGE_READONLY, PAGE_READWRITE, VirtualAlloc, VirtualFree, VirtualProtect,
    VirtualQuery,
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
    /// Address or size not aligned (a page for mappings, the word for atomic
    /// accesses), empty, or outside the address space.
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
/// a check allows runs under the same read lock (copies chunk by chunk, see
/// CHUNK), so that no unmap decommits pages under it. Changes made of several
/// calls (query, then map) rely on the kernel lock, under which every table
/// change runs.
pub struct Memory {
    base: *mut u8,
    blocks: RwLock<Table>,
    /// The pages takes found clean, made at the first take (see [`Armed`]).
    armed: OnceLock<Option<&'static Armed>>,
}

// SAFETY: `base` never changes and the reservation is ours until drop; the
// table is behind its lock. Guest bytes are shared with guest code on other
// host threads by design: Memory hands out no Rust reference to them that
// outlives one access, only raw copies and host atomics, as the generated
// code's own accesses.
unsafe impl Send for Memory {}
unsafe impl Sync for Memory {}

/// Mapped blocks only: start -> (end, block). Gaps are Free.
#[derive(Default)]
struct Table(BTreeMap<u64, (u64, Block)>);

/// HLE copies run in chunks of this size, each checked and copied under the
/// table's read lock: an unmap on another host thread (play mode) waits for one
/// chunk at most, and a copy of 128 MiB never holds the lock throughout.
const CHUNK: usize = 1 << 20;

/// Base of the reservation the fault handler reports on (0 when none).
static RESERVATION: AtomicUsize = AtomicUsize::new(0);

/// The pages a take found clean (see [`Memory::take_written`]), one bit per
/// guest page, set while the page is read-only on the host: its next write
/// faults, the fault handler makes it writable again and clears its bit, and
/// the write, retried, reaches the OS write watch. Takes ask the watch only
/// about pages not armed: its cost is per page asked (2 ms a frame for the
/// tutorial's 265k texture pages), and most texture pages are never written
/// again. Leaked: the fault handler may hold one while its `Memory` drops.
///
/// Never pass guest memory to an OS call that writes it (ReadFile, recv): on
/// an armed page it would fail with ERROR_NOACCESS instead of faulting.
struct Armed {
    /// The reservation's host base.
    base: usize,
    bits: Box<[AtomicU64]>,
    /// Arming, disarming and reprotecting exclude each other: a page must never
    /// stay read-only unarmed, as its next write would be taken for a guest
    /// fault. A spin lock: the fault handler takes no other lock and allocates
    /// nothing (a faulting HLE copy holds the table's read lock).
    busy: AtomicBool,
    /// Faults since the last take: past FAULTS, a fault disarms the page's
    /// whole armed run (a texture being rewritten), not one page.
    faults: AtomicU32,
}

const FAULTS: u32 = 64;
/// How far such a run reaches on each side of the fault, in pages (64 MiB).
const RUN: u64 = 0x4000;

/// The armed pages of each live `Memory` that took written pages: the fault
/// handler finds a fault's reservation here (tests hold several at once).
static ARMED: [AtomicPtr<Armed>; 8] = [const { AtomicPtr::new(std::ptr::null_mut()) }; 8];

impl Armed {
    fn lock(&self) {
        while self.busy.compare_exchange_weak(false, true, Ordering::Acquire, Ordering::Relaxed).is_err() {
            std::hint::spin_loop();
        }
    }

    fn unlock(&self) {
        self.busy.store(false, Ordering::Release);
    }

    fn get(&self, page: u64) -> bool {
        self.bits[(page / 64) as usize].load(Ordering::Relaxed) >> (page % 64) & 1 != 0
    }

    /// The first page of `[from, to)` whose bit is `on`, else `to`.
    fn find(&self, mut from: u64, to: u64, on: bool) -> u64 {
        while from < to {
            let w = self.bits[(from / 64) as usize].load(Ordering::Relaxed);
            let w = (if on { w } else { !w }) >> (from % 64);
            if w != 0 {
                return (from + u64::from(w.trailing_zeros())).min(to);
            }
            from = (from / 64 + 1) * 64;
        }
        to
    }

    /// Sets (`on`) or clears the bits of pages `[from, to)`.
    fn set(&self, mut from: u64, to: u64, on: bool) {
        while from < to {
            let (i, lo) = ((from / 64) as usize, from % 64);
            let n = (64 - lo).min(to - from);
            let mask = (u64::MAX >> (64 - n)) << lo;
            if on {
                self.bits[i].fetch_or(mask, Ordering::Relaxed);
            } else {
                self.bits[i].fetch_and(!mask, Ordering::Relaxed);
            }
            from += n;
        }
    }
}

/// The armed pages of the reservation that holds host address `host`.
fn armed_at(host: usize) -> Option<&'static Armed> {
    ARMED.iter().find_map(|slot| {
        // SAFETY: registered `Armed` are leaked, never freed.
        let a = unsafe { slot.load(Ordering::Acquire).as_ref() }?;
        (host >= a.base && host - a.base < SPACE as usize).then_some(a)
    })
}

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
        Ok(Memory { base, blocks: RwLock::default(), armed: OnceLock::new() })
    }

    /// The armed pages, made at the first take; None when every slot of `ARMED`
    /// is taken (many `Memory` at once, in tests): the write watch alone then.
    fn armed(&self) -> Option<&'static Armed> {
        *self.armed.get_or_init(|| {
            // SAFETY: zeroed atomics are valid zeros; the pages come committed on demand.
            let bits = unsafe { Box::<[AtomicU64]>::new_zeroed_slice((SPACE / PAGE / 64) as usize).assume_init() };
            let a: &'static Armed = Box::leak(Box::new(Armed { base: self.base as usize, bits, busy: AtomicBool::new(false), faults: AtomicU32::new(0) }));
            let ptr = std::ptr::from_ref(a).cast_mut();
            let free = ARMED.iter().any(|slot| slot.compare_exchange(std::ptr::null_mut(), ptr, Ordering::AcqRel, Ordering::Acquire).is_ok());
            free.then(|| {
                install_fault_handler();
                a
            })
        })
    }

    /// Arms the pages of `[s, e)` (page-aligned) mapped read-write on the host:
    /// their bits, then read-only. The table's read lock keeps their blocks as
    /// they are meanwhile.
    fn arm(&self, armed: &Armed, s: u64, e: u64) {
        let t = self.table();
        armed.lock();
        let first = t.0.range(..=s).next_back().map_or(s, |(&k, _)| k);
        for (&start, &(end, block)) in t.0.range(first..e) {
            let (a, b) = (start.max(s), end.min(e));
            if a >= b || block.host_protection() != PAGE_READWRITE {
                continue;
            }
            armed.set(a / PAGE, b / PAGE, true);
            let mut old = 0;
            // SAFETY: committed pages of our reservation (a mapped block).
            if unsafe { VirtualProtect(self.base.add(a as usize).cast(), (b - a) as usize, PAGE_READONLY, &mut old) } == 0 {
                armed.set(a / PAGE, b / PAGE, false);
            }
        }
        armed.unlock();
    }

    // The table's lock, recovered from poison like the crate's other locks.
    fn table(&self) -> RwLockReadGuard<'_, Table> {
        self.blocks.read().unwrap_or_else(|e| e.into_inner())
    }

    fn table_mut(&self) -> RwLockWriteGuard<'_, Table> {
        self.blocks.write().unwrap_or_else(|e| e.into_inner())
    }

    /// The pages of `[addr, addr + len)` written since they were last taken
    /// (by the guest or the host alike), as sorted guest page addresses.
    /// Taking them resets their watch in the same call, so a write that races
    /// the call is reported by it or by the next take. Empty for an empty range
    /// or one that leaves the space.
    ///
    /// Pages found clean are armed (see [`Armed`]): later takes skip them until
    /// a write disarms them; a write racing the arming shows in the second look
    /// at the watch that follows it.
    pub fn take_written(&self, addr: u64, len: u64) -> Vec<u64> {
        let Some(end) = addr.checked_add(len).filter(|&e| e <= SPACE && len > 0) else { return Vec::new() };
        let Some(armed) = self.armed() else { return self.watched(addr, end) };
        armed.faults.store(0, Ordering::Relaxed);
        let (mut out, mut p, last) = (Vec::new(), addr / PAGE, end.div_ceil(PAGE));
        loop {
            p = armed.find(p, last, false);
            if p == last {
                break;
            }
            let q = armed.find(p, last, true);
            let (s, e) = ((p * PAGE).max(addr), (q * PAGE).min(end));
            out.extend(self.watched(s, e));
            self.arm(armed, p * PAGE, q * PAGE);
            out.extend(self.watched(s, e));
            p = q;
        }
        out.sort_unstable();
        out.dedup();
        out
    }

    /// The pages of `[addr, addr + len)` written since they were last taken,
    /// reset as `take_written` resets them but left unarmed: for memory the
    /// host has just written itself (the GPU's copies), which may hold no
    /// texture and be rewritten by the guest at once.
    pub fn reset_written(&self, addr: u64, len: u64) -> Vec<u64> {
        let Some(end) = addr.checked_add(len).filter(|&e| e <= SPACE && len > 0) else { return Vec::new() };
        self.watched(addr, end)
    }

    /// The OS write watch's pages of `[addr, end)`, reset as they are reported.
    fn watched(&self, addr: u64, end: u64) -> Vec<u64> {
        let len = end - addr;
        // Room for every page of the range, so one call reports and resets all of it
        // (how far a reset reaches when the buffer fills is not documented).
        // ponytail: 8 bytes a page, on every span of every submit; cut a huge span
        // into fixed chunks, one call each, if this ever shows in --profile.
        let mut pages = vec![std::ptr::null_mut(); ((end.next_multiple_of(PAGE) - addr / PAGE * PAGE) / PAGE) as usize];
        let (mut count, mut granularity) = (pages.len(), 0u32);
        // SAFETY: a range of our reservation; `pages` has room for `count` addresses.
        let r = unsafe { GetWriteWatch(WRITE_WATCH_FLAG_RESET, self.base.add(addr as usize).cast(), len as usize, pages.as_mut_ptr(), &mut count, &mut granularity) };
        if r != 0 {
            return Vec::new();
        }
        let mut out: Vec<u64> = pages[..count].iter().map(|&p| p as u64 - self.base as u64).collect();
        out.sort_unstable();
        out
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
        let to = out.as_mut_ptr();
        // SAFETY: each chunk is copied under the read lock it was checked under.
        self.chunked(addr, out.len(), |t, a, n| t.accessible(a, n, perm::R), |at, n| unsafe { std::ptr::copy_nonoverlapping(self.base.add(addr as usize + at), to.add(at), n) })
    }

    /// As [`Memory::read`], appending the `len` bytes to `out` (none on an
    /// error) without zeroing their room first: copied into what `reserve`
    /// made, the length set once all are in.
    pub fn read_append(&self, addr: u64, len: usize, out: &mut Vec<u8>) -> Result<(), Error> {
        out.reserve(len);
        let start = out.len();
        let to = out.as_mut_ptr();
        // SAFETY: as in `read`, into the `len` bytes of room past `start` that `reserve` made.
        self.chunked(addr, len, |t, a, n| t.accessible(a, n, perm::R), |at, n| unsafe { std::ptr::copy_nonoverlapping(self.base.add(addr as usize + at), to.add(start + at), n) })?;
        // SAFETY: every byte up to `start + len` is written: the old ones, then the copy above.
        unsafe { out.set_len(start + len) };
        Ok(())
    }

    /// Writes a guest-supplied buffer for the HLE; every byte needs guest
    /// permission W.
    pub fn write(&self, addr: u64, data: &[u8]) -> Result<(), Error> {
        // SAFETY: as in `read`.
        self.chunked(addr, data.len(), |t, a, n| t.accessible(a, n, perm::W), |at, n| unsafe { std::ptr::copy_nonoverlapping(data.as_ptr().add(at), self.base.add(addr as usize + at), n) })
    }

    /// Writes memory the HLE owns (shared memory, locked pages): the guest
    /// may only have R there, but the host pages are writable.
    pub fn write_object(&self, addr: u64, data: &[u8]) -> Result<(), Error> {
        // SAFETY: as in `read`.
        self.chunked(addr, data.len(), |t, a, n| t.host_accessible(a, n, PAGE_READWRITE), |at, n| unsafe { std::ptr::copy_nonoverlapping(data.as_ptr().add(at), self.base.add(addr as usize + at), n) })
    }

    /// Reads memory the HLE owns, whatever the guest's permission.
    pub fn read_object(&self, addr: u64, out: &mut [u8]) -> Result<(), Error> {
        let to = out.as_mut_ptr();
        // SAFETY: as in `read`.
        self.chunked(addr, out.len(), |t, a, n| t.host_accessible(a, n, PAGE_READWRITE), |at, n| unsafe { std::ptr::copy_nonoverlapping(self.base.add(addr as usize + at), to.add(at), n) })
    }

    /// `copy(offset, len)` over `len` bytes from `addr`, CHUNK bytes at a time,
    /// each under the table's read lock once `ok` holds for it. A range longer
    /// than a chunk is checked whole first: a refused range copies nothing,
    /// unless it is unmapped meanwhile, which stops the copy at that chunk.
    fn chunked(&self, addr: u64, len: usize, ok: impl Fn(&Table, u64, u64) -> Result<(), Error>, mut copy: impl FnMut(usize, usize)) -> Result<(), Error> {
        if len > CHUNK {
            ok(&self.table(), addr, len as u64)?;
        }
        let mut at = 0;
        loop {
            let n = CHUNK.min(len - at);
            let t = self.table();
            ok(&t, addr + at as u64, n as u64)?;
            copy(at, n);
            drop(t);
            at += n;
            if at >= len {
                return Ok(());
            }
        }
    }

    /// Whether every byte of `[addr, addr + len)` is mapped with the guest
    /// permission `need`: the check `read` (R) and `write` (W) make.
    pub fn accessible(&self, addr: u64, len: u64, need: u32) -> Result<(), Error> {
        self.table().accessible(addr, len, need)
    }

    /// `f` on the naturally aligned guest word at `addr` as host atomic `A`,
    /// once the guest permission `need` is checked as `read` (R) and `write` (W)
    /// check it, under the same read lock.
    fn word<A, R>(&self, addr: u64, need: u32, f: impl FnOnce(&A) -> R) -> Result<R, Error> {
        if !addr.is_multiple_of(size_of::<A>() as u64) {
            return Err(Error::InvalidRange);
        }
        let t = self.table();
        t.accessible(addr, size_of::<A>() as u64, need)?;
        // SAFETY: mapped with `need` (checked just above, the lock still held) and
        // aligned; the reference does not leave `f`.
        Ok(f(unsafe { &*self.base.add(addr as usize).cast::<A>() }))
    }

    /// Loads a naturally aligned guest word with Acquire ordering, checked as
    /// `read` (Err(InvalidRange) when unaligned). A plain load: it never writes.
    pub fn load_u32(&self, addr: u64) -> Result<u32, Error> {
        self.word(addr, perm::R, |w: &AtomicU32| w.load(Ordering::Acquire))
    }

    pub fn load_u64(&self, addr: u64) -> Result<u64, Error> {
        self.word(addr, perm::R, |w: &AtomicU64| w.load(Ordering::Acquire))
    }

    /// Stores a naturally aligned guest word with Release ordering, checked as `write`.
    pub fn store_u32(&self, addr: u64, v: u32) -> Result<(), Error> {
        self.word(addr, perm::W, |w: &AtomicU32| w.store(v, Ordering::Release))
    }

    /// Replaces the guest word v by f(v) in one compare-exchange loop (`f` may
    /// run more than once) unless `f` returns None, which stores nothing;
    /// checked for R and W first. The inner result is the value replaced (Ok)
    /// or refused (Err).
    pub fn update_u32(&self, addr: u64, f: impl FnMut(u32) -> Option<u32>) -> Result<Result<u32, u32>, Error> {
        self.word(addr, perm::RW, |w: &AtomicU32| w.try_update(Ordering::AcqRel, Ordering::Acquire, f))
    }

    /// Writes memory the HLE owns, as `write_object` does, with one word last:
    /// `rest` goes to `addr + width`, then the `width`-byte word at `addr` (at
    /// most 8) becomes `f(old)`; returns `old`. A naturally aligned 4- or 8-byte
    /// word changes in one host atomic with Release ordering: a thread whose
    /// Acquire load sees the new value also sees `rest` and every earlier write
    /// of this thread (a GPU report: timestamp, then payload). Any other word is
    /// copied, as no Arm atomic can reach it either. Nothing is written unless
    /// the whole range is host-writable.
    pub fn publish_object(&self, addr: u64, width: usize, rest: &[u8], mut f: impl FnMut(u64) -> u64) -> Result<u64, Error> {
        assert!(width <= 8, "a word of at most 8 bytes");
        let t = self.table();
        t.host_accessible(addr, (width + rest.len()) as u64, PAGE_READWRITE)?;
        // SAFETY: the whole range is mapped and host-writable (checked just above, under `t`);
        // an aligned guest address is an aligned host one (the base is 64 KiB-aligned).
        unsafe {
            let p = self.base.add(addr as usize);
            std::ptr::copy_nonoverlapping(rest.as_ptr(), p.add(width), rest.len());
            let (set, fetch) = (Ordering::Release, Ordering::Relaxed);
            Ok(match width {
                4 if addr.is_multiple_of(4) => match AtomicU32::from_ptr(p.cast()).try_update(set, fetch, |v| Some(f(u64::from(v)) as u32)) {
                    Ok(v) | Err(v) => u64::from(v),
                },
                8 if addr.is_multiple_of(8) => match AtomicU64::from_ptr(p.cast()).try_update(set, fetch, |v| Some(f(v))) {
                    Ok(v) | Err(v) => v,
                },
                _ => {
                    let mut old = [0u8; 8];
                    std::ptr::copy_nonoverlapping(p, old.as_mut_ptr(), width);
                    let old = u64::from_le_bytes(old);
                    std::ptr::copy_nonoverlapping(f(old).to_le_bytes().as_ptr(), p, width);
                    old
                }
            })
        }
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
        let armed = self.disarm(addr, end);
        // SAFETY: committed pages of our reservation.
        let freed = unsafe { VirtualFree(self.base.add(addr as usize).cast(), size as usize, MEM_DECOMMIT) } != 0;
        armed.inspect(|a| a.unlock());
        if !freed {
            return Err(Error::Host(last_error()));
        }
        Ok(())
    }

    /// Before an unmap or a reprotect of `[addr, end)` (on the table's write
    /// lock): its pages unarmed, the armed pages locked until the caller is done.
    fn disarm(&self, addr: u64, end: u64) -> Option<&'static Armed> {
        let a = self.armed.get().copied().flatten()?;
        a.lock();
        a.set(addr / PAGE, end / PAGE, false);
        Some(a)
    }

    /// Applies `f` to every block of a fully mapped range, then reprotects it.
    fn update_in(&self, t: &mut Table, addr: u64, size: u64, f: impl Fn(&mut Block)) -> Result<(), Error> {
        let end = t.mapped(addr, size)?;
        t.split(addr);
        t.split(end);
        let armed = self.disarm(addr, end);
        let mut result = Ok(());
        for (&start, (e, block)) in t.0.range_mut(addr..end) {
            f(block);
            let mut old = 0;
            // SAFETY: committed pages of our reservation.
            if unsafe { VirtualProtect(self.base.add(start as usize).cast(), (*e - start) as usize, block.host_protection(), &mut old) } == 0 {
                result = Err(Error::Host(last_error()));
                break;
            }
        }
        armed.inspect(|a| a.unlock());
        result?;
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
        if let Some(Some(a)) = self.armed.get() {
            let ptr = std::ptr::from_ref(*a).cast_mut();
            ARMED.iter().for_each(|slot| _ = slot.compare_exchange(ptr, std::ptr::null_mut(), Ordering::AcqRel, Ordering::Acquire));
        }
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
    if rec.ExceptionCode != EXCEPTION_ACCESS_VIOLATION || rec.NumberParameters < 2 {
        return EXCEPTION_CONTINUE_SEARCH;
    }
    let host = rec.ExceptionInformation[1];
    // A write to an armed page: writable again, unarmed, and retried.
    if rec.ExceptionInformation[0] == 1
        && let Some(a) = armed_at(host)
    {
        let page = ((host - a.base) as u64) / PAGE;
        a.lock();
        let hit = a.get(page);
        if hit {
            let (from, to) = if a.faults.fetch_add(1, Ordering::Relaxed) < FAULTS {
                (page, page + 1)
            } else {
                let mut from = page;
                while from > page.saturating_sub(RUN) && a.get(from - 1) {
                    from -= 1;
                }
                (from, a.find(page, (page + RUN).min(SPACE / PAGE), false))
            };
            a.set(from, to, false);
            let mut old = 0;
            // SAFETY: armed pages are committed pages of a read-write block.
            unsafe { VirtualProtect((a.base + (from * PAGE) as usize) as *const _, ((to - from) * PAGE) as usize, PAGE_READWRITE, &mut old) };
        }
        a.unlock();
        // Not armed: another thread may have just disarmed it.
        if hit || writable(host) {
            return EXCEPTION_CONTINUE_EXECUTION;
        }
    }
    if base == 0 {
        return EXCEPTION_CONTINUE_SEARCH;
    }
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
    crate::fatal(&what)
}

/// Whether the host page at `host` is writable now.
fn writable(host: usize) -> bool {
    // SAFETY: a query; it reads nothing at `host`.
    let mut info: MEMORY_BASIC_INFORMATION = unsafe { std::mem::zeroed() };
    unsafe { VirtualQuery(host as *const _, &mut info, size_of::<MEMORY_BASIC_INFORMATION>()) != 0 && info.Protect == PAGE_READWRITE }
}
