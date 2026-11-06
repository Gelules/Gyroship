//! AArch64 decoding: yaxpeax-arm 0.5.0 (base forms, no aliases), behind fixes
//! for the defects found by checking it against llvm-objdump on every
//! distinct code word of the game's modules (journal: DECISIONS D13).

pub use yaxpeax_arm::armv8::a64::{Instruction, Opcode, Operand, SIMDSizeCode, ShiftStyle, SizeCode};
use yaxpeax_arch::{Decoder, U8Reader};
use yaxpeax_arm::armv8::a64::InstDecoder;

/// Decodes one instruction word. `None` means "do not translate this":
/// unallocated, or an instruction this decoder gets wrong.
pub fn decode(word: u32) -> Option<Instruction> {
    // decode_into can return Ok without writing the opcode.
    let mut inst = Instruction { opcode: Opcode::Invalid, ..Instruction::default() };
    InstDecoder::default().decode_into(&mut inst, &mut U8Reader::new(&word.to_le_bytes())).ok()?;
    match inst.opcode {
        Opcode::Invalid => return None,
        // Wrong arrangement or lost "2" form in 0.5.0; the game never uses them.
        Opcode::FMLAL | Opcode::FMLAL2 | Opcode::FMLSL | Opcode::FMLSL2 => return None,
        _ => {}
    }
    // MSR (register): 0.5.0 keeps only 4 bits of Rt, so x16..x30 become x0..x14.
    if word & 0xFFF0_0000 == 0xD510_0000 {
        inst.operands[1] = Operand::Register(SizeCode::X, (word & 31) as u16);
    }
    // 64-bit ADD/SUB (extended register) with UXTW: 0.5.0 reports LSL.
    if word & 0x9FE0_E000 == 0x8B20_4000
        && let Operand::RegShift(_, amount, size, rm) = inst.operands[2]
    {
        inst.operands[2] = Operand::RegShift(ShiftStyle::UXTW, amount, size, rm);
    }
    Some(inst)
}
