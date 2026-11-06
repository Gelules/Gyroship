//! Synchronization SVCs, with the public semantics of Horizon (switchbrew):
//! WaitSynchronization on threads and events, the mutex arbiter
//! (ArbitrateLock/Unlock), process-wide condition variables, and waits and
//! signals on addresses. Waiters wake by priority, then in arrival order.
//!
//! Guest words are read and written between two switch points, so they are
//! atomic with respect to every other guest thread.

use crate::Kernel;
use crate::result::*;
use crate::svc::{CURRENT_THREAD, ns_to_ticks};
use crate::thread::{Object, State, Wait};
use gyroship_runtime::Cpu;

/// Set in a mutex word while threads wait for the mutex.
pub const HAS_LISTENERS: u32 = 0x4000_0000;

/// An event: a writable end signals it, a readable end waits for it.
#[derive(Clone, Copy, Debug, Default)]
pub struct Event {
    pub signaled: bool,
}

/// Most handles WaitSynchronization takes.
const MAX_WAIT: u32 = 64;

impl Kernel {
    /// A new event; returns its (writable, readable) handles.
    pub fn create_event(&mut self) -> Result<(u32, u32), u32> {
        let e = self.events.len();
        let w = self.handles.add(Object::WritableEvent(e)).ok_or(OUT_OF_HANDLES)?;
        let Some(r) = self.handles.add(Object::ReadableEvent(e)) else {
            self.handles.close(w);
            return Err(OUT_OF_HANDLES);
        };
        self.events.push(Event::default());
        Ok((w, r))
    }

    /// Signals event `e`: every thread waiting for it wakes.
    pub fn signal_event(&mut self, e: usize) {
        self.events[e].signaled = true;
        self.signal(Object::ReadableEvent(e));
    }

    fn object(&self, h: u32) -> Option<Object> {
        if u64::from(h) == CURRENT_THREAD {
            return self.sched.current.map(Object::Thread);
        }
        self.handles.get(h)
    }

    fn signaled(&self, o: Object) -> bool {
        match o {
            Object::Thread(t) => self.sched.threads[t].state == State::Done,
            Object::ReadableEvent(e) => self.events[e].signaled,
            Object::WritableEvent(_) | Object::Session(_) | Object::SharedMemory(_) => false,
        }
    }

    /// Wakes the threads in WaitSynchronization on `o`, with its index in their lists.
    pub fn signal(&mut self, o: Object) {
        for t in self.waiters(|w| matches!(w, Wait::Sync(list) if list.contains(&o))) {
            let Wait::Sync(list) = &self.sched.threads[t].wait else { unreachable!() };
            let i = list.iter().position(|x| *x == o).expect("in the list") as u64;
            self.wake(t, SUCCESS, Some(i));
        }
    }

    fn read32(&self, addr: u64) -> Option<u32> {
        let mut b = [0; 4];
        self.mem.read(addr, &mut b).ok()?;
        Some(u32::from_le_bytes(b))
    }

    fn write32(&mut self, addr: u64, v: u32) -> Option<()> {
        self.mem.write(addr, &v.to_le_bytes()).ok()
    }

    fn deadline(&self, timeout: i64) -> Option<u64> {
        (timeout > 0).then(|| self.now() + ns_to_ticks(timeout))
    }

    /// WaitSynchronization: x1 gets the index of the first signaled object.
    pub fn wait_synchronization(&mut self, cpu: &mut Cpu, ptr: u64, count: u32, timeout: i64) -> u32 {
        if count > MAX_WAIT {
            return OUT_OF_RANGE;
        }
        let mut raw = vec![0; 4 * count as usize];
        if count > 0 && self.mem.read(ptr, &mut raw).is_err() {
            return INVALID_POINTER;
        }
        let mut objects = Vec::new();
        for h in raw.chunks(4).map(|c| u32::from_le_bytes(c.try_into().expect("4 bytes"))) {
            match self.object(h) {
                Some(o @ (Object::Thread(_) | Object::ReadableEvent(_))) => objects.push(o),
                _ => return INVALID_HANDLE,
            }
        }
        let t = self.sched.current.expect("a running thread");
        if std::mem::take(&mut self.sched.threads[t].cancel) {
            return CANCELLED;
        }
        if let Some(i) = objects.iter().position(|&o| self.signaled(o)) {
            cpu.x[1] = i as u64;
            return SUCCESS;
        }
        if timeout == 0 {
            return TIMED_OUT;
        }
        let deadline = self.deadline(timeout);
        self.block(Wait::Sync(objects), deadline, TIMED_OUT);
        SUCCESS
    }

    /// CancelSynchronization: wakes the thread out of WaitSynchronization,
    /// or makes its next one return Cancelled.
    pub fn cancel_synchronization(&mut self, h: u32) -> u32 {
        let Some(t) = self.thread_of(h) else { return INVALID_HANDLE };
        if self.sched.threads[t].state == State::Waiting && matches!(self.sched.threads[t].wait, Wait::Sync(_)) {
            self.wake(t, CANCELLED, None);
        } else {
            self.sched.threads[t].cancel = true;
        }
        SUCCESS
    }

    /// ResetSignal: clears a signaled readable event, InvalidState if it was not.
    pub fn reset_signal(&mut self, h: u32) -> u32 {
        match self.object(h) {
            Some(Object::ReadableEvent(e)) if self.events[e].signaled => {
                self.events[e].signaled = false;
                SUCCESS
            }
            Some(Object::ReadableEvent(_)) => INVALID_STATE,
            _ => INVALID_HANDLE,
        }
    }

    pub fn signal_event_handle(&mut self, h: u32) -> u32 {
        match self.object(h) {
            Some(Object::WritableEvent(e)) => {
                self.signal_event(e);
                SUCCESS
            }
            _ => INVALID_HANDLE,
        }
    }

    pub fn clear_event(&mut self, h: u32) -> u32 {
        match self.object(h) {
            Some(Object::WritableEvent(e) | Object::ReadableEvent(e)) => {
                self.events[e].signaled = false;
                SUCCESS
            }
            _ => INVALID_HANDLE,
        }
    }

    /// CloseHandle. ponytail: the object itself lives on (threads, events);
    /// add reference counts when an object must die with its last handle.
    pub fn close_handle(&mut self, h: u32) -> u32 {
        if self.handles.close(h).is_some() { SUCCESS } else { INVALID_HANDLE }
    }

    /// ArbitrateLock: waits for the mutex at `addr`, unless its word is no
    /// longer `owner` with waiters (then the caller retries).
    pub fn arbitrate_lock(&mut self, owner: u32, addr: u64, tag: u32) -> u32 {
        if !addr.is_multiple_of(4) {
            return INVALID_ADDRESS;
        }
        let Some(v) = self.read32(addr) else { return INVALID_CURRENT_MEMORY };
        if v != owner | HAS_LISTENERS {
            return SUCCESS;
        }
        if self.thread_of(owner).is_none() {
            return INVALID_HANDLE;
        }
        self.block(Wait::Lock { addr, tag }, None, SUCCESS);
        SUCCESS
    }

    /// ArbitrateUnlock: hands the mutex at `addr` to its best waiter, or frees it.
    pub fn arbitrate_unlock(&mut self, addr: u64) -> u32 {
        if !addr.is_multiple_of(4) {
            return INVALID_ADDRESS;
        }
        let w = self.waiters(|w| matches!(w, Wait::Lock { addr: a, .. } if *a == addr));
        let value = match w.first() {
            None => 0,
            Some(&t) => {
                let Wait::Lock { tag, .. } = self.sched.threads[t].wait else { unreachable!() };
                tag | if w.len() > 1 { HAS_LISTENERS } else { 0 }
            }
        };
        if self.write32(addr, value).is_none() {
            return INVALID_CURRENT_MEMORY;
        }
        if let Some(&t) = w.first() {
            self.wake(t, SUCCESS, None);
        }
        SUCCESS
    }

    /// WaitProcessWideKeyAtomic: releases the mutex at `addr`, marks the
    /// condition `key` as waited on, and waits for a signal; once signaled,
    /// the thread takes the mutex back before it returns.
    pub fn wait_key(&mut self, addr: u64, key: u64, tag: u32, timeout: i64) -> u32 {
        let r = self.arbitrate_unlock(addr);
        if r != SUCCESS {
            return r;
        }
        if !key.is_multiple_of(4) {
            return INVALID_ADDRESS;
        }
        if self.write32(key, 1).is_none() {
            return INVALID_CURRENT_MEMORY;
        }
        if timeout == 0 {
            return TIMED_OUT;
        }
        let deadline = self.deadline(timeout);
        self.block(Wait::CondVar { key, addr, tag }, deadline, TIMED_OUT);
        SUCCESS
    }

    /// SignalProcessWideKey: up to `count` waiters of `key` (all when
    /// `count` <= 0) go on to take their mutex.
    pub fn signal_key(&mut self, key: u64, count: i32) {
        let w = self.waiters(|w| matches!(w, Wait::CondVar { key: k, .. } if *k == key));
        let n = if count <= 0 { w.len() } else { w.len().min(count as usize) };
        for &t in &w[..n] {
            let Wait::CondVar { addr, tag, .. } = self.sched.threads[t].wait else { unreachable!() };
            match self.read32(addr) {
                None => self.wake(t, INVALID_CURRENT_MEMORY, None),
                Some(0) => {
                    let _ = self.write32(addr, tag);
                    self.wake(t, SUCCESS, None);
                }
                Some(v) => {
                    // Owned: wait for it like ArbitrateLock, with no deadline any more.
                    let _ = self.write32(addr, v | HAS_LISTENERS);
                    let th = &mut self.sched.threads[t];
                    (th.wait, th.deadline) = (Wait::Lock { addr, tag }, None);
                }
            }
        }
        if n == w.len() {
            let _ = self.write32(key, 0);
        }
    }

    /// WaitForAddress. Types: 0 wait if less than, 1 decrement and wait if
    /// less than, 2 wait if equal (32-bit, signed); 3, new on Switch 2, is
    /// taken to be wait if equal on 64 bits (an assumption, logged once).
    pub fn wait_address(&mut self, addr: u64, kind: u32, value: u64, timeout: i64) -> u32 {
        if !addr.is_multiple_of(if kind == 3 { 8 } else { 4 }) {
            return INVALID_ADDRESS;
        }
        match kind {
            0..=2 => {
                let Some(v) = self.read32(addr) else { return INVALID_CURRENT_MEMORY };
                let (v, value) = (v as i32, value as i32);
                if !(if kind == 2 { v == value } else { v < value }) {
                    return INVALID_STATE;
                }
                if kind == 1 && self.write32(addr, v.wrapping_sub(1) as u32).is_none() {
                    return INVALID_CURRENT_MEMORY;
                }
            }
            3 => {
                if self.journal.first("WaitForAddress type 3") {
                    self.log("svc 0x34 WaitForAddress type 3: assumed to wait if the 64-bit value is equal");
                }
                let mut b = [0; 8];
                if self.mem.read(addr, &mut b).is_err() {
                    return INVALID_CURRENT_MEMORY;
                }
                if u64::from_le_bytes(b) != value {
                    return INVALID_STATE;
                }
            }
            _ => return INVALID_ENUM_VALUE,
        }
        if timeout == 0 {
            return TIMED_OUT;
        }
        let deadline = self.deadline(timeout);
        self.block(Wait::Address(addr), deadline, TIMED_OUT);
        SUCCESS
    }

    /// SignalToAddress. Types: 0 signal, 1 increment if equal, then signal,
    /// 2 modify by the number of waiters if equal, then signal. Wakes up to
    /// `count` waiters (all when `count` <= 0).
    pub fn signal_address(&mut self, addr: u64, kind: u32, value: i32, count: i32) -> u32 {
        if !addr.is_multiple_of(4) {
            return INVALID_ADDRESS;
        }
        let w = self.waiters(|w| *w == Wait::Address(addr));
        let new = match kind {
            0 => None,
            1 => Some(value.wrapping_add(1)),
            2 => Some(match (count <= 0, w.len()) {
                (true, 0) => value.wrapping_add(1),
                (true, _) => value.wrapping_sub(2),
                (false, 0) => value.wrapping_add(1),
                (false, n) if n <= count as usize => value.wrapping_sub(1),
                (false, _) => value,
            }),
            _ => return INVALID_ENUM_VALUE,
        };
        if let Some(new) = new {
            let Some(v) = self.read32(addr) else { return INVALID_CURRENT_MEMORY };
            if v as i32 != value {
                return INVALID_STATE;
            }
            if new != value && self.write32(addr, new as u32).is_none() {
                return INVALID_CURRENT_MEMORY;
            }
        }
        let n = if count <= 0 { w.len() } else { w.len().min(count as usize) };
        for &t in &w[..n] {
            self.wake(t, SUCCESS, None);
        }
        SUCCESS
    }
}
