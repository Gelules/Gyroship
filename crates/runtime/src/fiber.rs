//! Windows fibers: each guest thread runs on one, all of them on a single
//! host thread (D19). Switching is cooperative and deterministic.

use std::ffi::c_void;
use windows_sys::Win32::System::Threading::{ConvertFiberToThread, ConvertThreadToFiber, CreateFiberEx, DeleteFiber, SwitchToFiber};

/// The x87 and SSE control state switches with the fiber.
const FIBER_FLAG_FLOAT_SWITCH: u32 = 1;

/// A fiber, by its Windows handle.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Fiber(pub *mut c_void);

// SAFETY: a handle to a process-wide fiber object; moving it to another
// thread runs nothing. `switch` and `delete` keep their own contracts.
unsafe impl Send for Fiber {}

/// Host stack of a guest thread: reserved, committed on use. Generated code
/// hands tail calls back to the driver past `STACK_BUDGET`, and guest
/// recursion is bounded by the guest stack.
pub const STACK_RESERVE: usize = 16 << 20;

/// Turns the calling host thread into a fiber, so that it can switch to
/// others and be switched back to.
pub fn convert_thread() -> Fiber {
    // SAFETY: the calling thread is not a fiber yet.
    let f = unsafe { ConvertThreadToFiber(std::ptr::null()) };
    assert!(!f.is_null(), "ConvertThreadToFiber failed: {}", std::io::Error::last_os_error());
    Fiber(f)
}

/// Undoes [`convert_thread`] on the calling thread.
pub fn convert_back() {
    // SAFETY: the calling thread was converted by `convert_thread`.
    unsafe { ConvertFiberToThread() };
}

/// A new fiber that will run `entry(arg)` when first switched to. `entry`
/// must never return: a fiber ends by switching away for good, and another
/// fiber deletes it.
pub fn create(entry: unsafe extern "system" fn(*mut c_void), arg: usize) -> Fiber {
    // SAFETY: creates a fiber; nothing runs until a switch.
    let f = unsafe { CreateFiberEx(64 << 10, STACK_RESERVE, FIBER_FLAG_FLOAT_SWITCH, Some(entry), arg as *const c_void) };
    assert!(!f.is_null(), "CreateFiberEx failed: {}", std::io::Error::last_os_error());
    Fiber(f)
}

/// Switches to `to`; returns when some fiber switches back to this one.
///
/// # Safety
/// `to` must be a live fiber of this host thread, and the caller must hold
/// no borrow that the other fibers need.
pub unsafe fn switch(to: Fiber) {
    // SAFETY: guaranteed by the caller.
    unsafe { SwitchToFiber(to.0) };
}

/// Deletes a fiber that is not running. Rust frames left on its stack are
/// not dropped: a fiber that can be deleted must hold nothing that needs it.
///
/// # Safety
/// `f` must not be the running fiber, and nothing may switch to it again.
pub unsafe fn delete(f: Fiber) {
    // SAFETY: guaranteed by the caller.
    unsafe { DeleteFiber(f.0) };
}
