//! AArch64 code generator: lifts each instruction to a small IR, then emits
//! it through one of two backends (Rust source or Cranelift objects), chosen
//! in M1 by measurement.

pub mod ir;
pub mod lift;
pub mod clif;
pub mod rust;

use gyroship_analysis::cfg::Cfg;
use gyroship_analysis::discover::Function;

/// How a direct call reaches its target.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Callee {
    /// A generated function, called by name.
    Direct(u64),
    /// Through the runtime's dispatch table.
    Dispatch(u64),
}

/// One function to generate.
pub struct Input<'a> {
    pub function: &'a Function,
    pub cfg: &'a Cfg,
    /// The guest word at an address.
    pub word: &'a dyn Fn(u64) -> u32,
    /// Where a direct call to an address goes (PLT stubs already resolved).
    pub callee: &'a dyn Fn(u64) -> Callee,
}
