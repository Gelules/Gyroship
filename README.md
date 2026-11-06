# Gyroship

Gyroship is a native PC port of Star Fox (2026, Nintendo Switch 2), written in Rust.
It statically recompiles the game's AArch64 executable and runs it on a high-level
emulation runtime, in the spirit of N64Recomp and XenonRecomp. It also aims to be a
practice tool for speedrunners (save states, mission teleport with a chosen starting
state) and a verified reference for the game's boost mechanics.

This repository contains no Nintendo code or assets. Each player supplies their own
unencrypted dump, as plain folders. Gyroship never decrypts anything and
never goes online.

Status: milestone M0 (tooling and verification loop). The game does not run yet.

## Building

Requires a recent stable Rust toolchain (MSVC on Windows).

    cargo build --release

## Tools

`gyroship-tools` works on your own dump:

    gyroship-tools nso info    <exefs/main>               # segments, symbols, relocations
    gyroship-tools nso2elf -o <dir> <exefs/main> ...      # ELF files for disassemblers
    gyroship-tools link    <exefs>                        # place, relocate and link all modules
    gyroship-tools capture --title Gyroship -o shot.png   # capture a window (Windows)

`tools/ghidra/analyze_modules.sh` imports the exported ELF files into a headless Ghidra
project (main at `0x7100000000`) and writes function and string lists.

## Running the test scene

Until the game renders, `gyroship` shows a placeholder scene used to prove the visual
verification loop:

    gyroship --headless --frames 60 --dump-frames artifacts/frames/run1 --dump-every 10
    gyroship --headless --backend dx12 --replay inputs.txt --log run.log
    gyroship --frames 600                                 # windowed

The time step is fixed: frame N always renders the same image, on Vulkan and DX12 alike.

## Layout

    crates/loader     NSO executables, relocations, ELF export
    crates/analysis   function discovery, CFG, jump tables (M1)
    crates/recomp     AArch64 code generator (M1)
    crates/runtime    guest CPU context, memory, dispatch (M2)
    crates/hle        OS services; network always offline (M2)
    crates/gpu        wgpu rendering; NVN translation later
    crates/practice   save states, teleport, practice menu (M6)
    crates/research   boost measurements and verified formulas (M7)
    crates/gyroship   the executable
    crates/tools      gyroship-tools
    tools/ghidra      headless Ghidra scripts

Generated code (`generated/`) and run outputs (`artifacts/`) are derived from the
player's own game files and are never committed.
