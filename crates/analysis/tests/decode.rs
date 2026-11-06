//! The decoder fixes, on standard A64 encodings (no game data).

use gyroship_analysis::decode::{decode, Opcode, Operand, ShiftStyle, SizeCode};

#[test]
fn msr_register_reads_all_five_bits_of_rt() {
    for (word, rt) in [(0xd51b4210, 16), (0xd51b4211, 17), (0xd51b421f, 31), (0xd51bd056, 22)] {
        let inst = decode(word).unwrap();
        assert_eq!(inst.opcode, Opcode::MSR, "{word:#x}");
        assert_eq!(inst.operands[1], Operand::Register(SizeCode::X, rt), "{word:#x}");
    }
    // msr TPIDR_EL0, x22 prints with the right register.
    assert!(decode(0xd51bd056).unwrap().to_string().contains("x22"));
}

#[test]
fn extended_add_sub_keeps_uxtw() {
    // add x8, x24, w26, uxtw #4; cmp x27, w28, uxtw; adds x21, x10, w12, uxtw; subs x8, x8, w11, uxtw
    for (word, amount, rm) in [(0x8b3a5308, 4, 26), (0xeb3c437f, 0, 28), (0xab2c4155, 0, 12), (0xeb2b4108, 0, 11)] {
        let inst = decode(word).unwrap();
        assert_eq!(inst.operands[2], Operand::RegShift(ShiftStyle::UXTW, amount, SizeCode::W, rm), "{word:#x}");
    }
    // add x8, x9, w10, sxtw is left alone.
    assert_eq!(decode(0x8b2ac128).unwrap().operands[2], Operand::RegShift(ShiftStyle::SXTW, 0, SizeCode::W, 10));
}

#[test]
fn refuses_unallocated_and_untrusted_encodings() {
    assert!(decode(0x0e68a8d4).is_none(), "unallocated AdvSIMD: yaxpeax returns Ok without an opcode");
    assert!(decode(0x4f8700b8).is_none(), "fmlal by element: decoded wrongly by 0.5.0");
    assert_eq!(decode(0xd2a00000).unwrap().opcode, Opcode::MOVZ);
    assert_eq!(decode(0xd65f03c0).unwrap().opcode, Opcode::RET);
}
