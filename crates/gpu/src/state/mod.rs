//! The 3D class's (0xc797) state at a draw, decoded from its register file
//! for wgpu: vertex input, fixed-function state, resource bindings and the
//! memory the method stream writes, texture and sampler headers.
//!
//! Facts come from NVIDIA's open-gpu-doc headers (clc797.h, clc797tex.h)
//! and from how Mesa's NVK/NAK/NIL drive the same hardware; where neither
//! settles a value, the decoders keep it raw or say so.

pub mod qmd;
pub mod raster;
pub mod resources;
pub mod textures;
pub mod vertex;

/// Reads a 3D-class register by its byte method offset (clc797.h:
/// SET_COLOR_TARGET_A(0) is 0x800); never-written registers read 0.
pub type Reg<'a> = &'a dyn Fn(u32) -> u32;
