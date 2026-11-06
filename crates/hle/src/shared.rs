//! Shared memory a service hands out (time, hid): the kernel object keeps
//! the bytes, and the guest maps them with MapSharedMemory. The HLE writes
//! through the object, into the guest pages too when they are mapped.

use crate::result::*;
use crate::thread::Object;
use crate::{Kernel, layout};
use gyroship_runtime::mem::{Block, PAGE, perm, state};

#[derive(Debug)]
pub struct SharedMemory {
    pub data: Vec<u8>,
    /// Where the guest mapped it.
    pub mapped: Option<u64>,
}

impl Kernel {
    /// A new zeroed shared memory of `size` bytes; returns its index.
    pub fn create_shared_memory(&mut self, size: u64) -> usize {
        self.shared.push(SharedMemory { data: vec![0; size as usize], mapped: None });
        self.shared.len() - 1
    }

    fn shared_of(&self, h: u32) -> Option<usize> {
        match self.handles.get(h)? {
            Object::SharedMemory(m) => Some(m),
            _ => None,
        }
    }

    /// MapSharedMemory: the whole object, at a free range of the ASLR region.
    pub fn map_shared(&mut self, h: u32, addr: u64, size: u64, p: u32) -> u32 {
        let Some(m) = self.shared_of(h) else { return INVALID_HANDLE };
        if !addr.is_multiple_of(PAGE) || size != self.shared[m].data.len().next_multiple_of(PAGE as usize) as u64 {
            return INVALID_SIZE;
        }
        if !matches!(p, perm::R | perm::RW) {
            return INVALID_NEW_MEMORY_PERMISSION;
        }
        if !layout::contains(layout::ASLR, addr, size) {
            return INVALID_MEMORY_REGION;
        }
        if self.mem.map(addr, size, Block::new(state::SHARED, p)).is_err() {
            return INVALID_CURRENT_MEMORY;
        }
        let data = std::mem::take(&mut self.shared[m].data);
        self.mem.write_object(addr, &data).expect("just mapped");
        self.shared[m] = SharedMemory { data, mapped: Some(addr) };
        SUCCESS
    }

    /// UnmapSharedMemory: the guest's bytes go back to the object.
    pub fn unmap_shared(&mut self, h: u32, addr: u64, size: u64) -> u32 {
        let Some(m) = self.shared_of(h) else { return INVALID_HANDLE };
        if self.shared[m].mapped != Some(addr) {
            return INVALID_STATE;
        }
        let mut data = std::mem::take(&mut self.shared[m].data);
        // Shared pages stay readable on the host, whatever the guest's permission.
        if self.mem.read_object(addr, &mut data).is_err() || self.mem.unmap(addr, size).is_err() {
            self.shared[m].data = data;
            return INVALID_CURRENT_MEMORY;
        }
        self.shared[m] = SharedMemory { data, mapped: None };
        SUCCESS
    }

    /// Writes `bytes` at `offset` of shared memory `m`.
    pub fn write_shared(&mut self, m: usize, offset: usize, bytes: &[u8]) {
        self.shared[m].data[offset..offset + bytes.len()].copy_from_slice(bytes);
        if let Some(addr) = self.shared[m].mapped {
            self.mem.write_object(addr + offset as u64, bytes).expect("mapped shared memory");
        }
    }
}
