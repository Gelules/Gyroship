//! Function discovery, control-flow graphs, jump tables and call graph for AArch64 modules.

pub mod cfg;
pub mod decode;
pub mod discover;
pub mod eh_frame;
pub mod insn;
