#!/bin/sh
# Imports the ELF exports of one exefs into a Ghidra project, one program per
# module at a fixed base address, runs auto-analysis, then ExportInventory.
#
# usage: analyze_modules.sh <elf dir> <project dir> <project name> <out dir> [module...]
# Needs GHIDRA_DIR. GHIDRA_HEADLESS_MAXMEM (for example 10G) bounds the JVM heap.
#
# Module bases: main sits at 0x7100000000 (the usual convention of Switch
# modding tools, so offsets read as main+X); the other modules get their own
# 256 MiB-aligned slots so that addresses never overlap between programs.
set -eu

elf_dir=$1 project_dir=$2 project=$3 out_dir=$4
shift 4
[ $# -gt 0 ] || set -- rtld subsdk3 subsdk0 subsdk2 subsdk1 sdk main

headless="$GHIDRA_DIR/ghidra_12.2_PUBLIC/support/analyzeHeadless.bat"
script_dir=$(cd "$(dirname "$0")" && pwd)
mkdir -p "$project_dir" "$out_dir"

for module in "$@"; do
    case $module in
        rtld) base=7000000000 ;;
        main) base=7100000000 ;;
        subsdk0) base=7200000000 ;;
        subsdk1) base=7210000000 ;;
        subsdk2) base=7220000000 ;;
        subsdk3) base=7230000000 ;;
        sdk) base=7240000000 ;;
        *) echo "unknown module $module" >&2; exit 1 ;;
    esac
    # main holds 94% of the code: skip the heaviest analyzers there (16 GB machine).
    light=""
    [ "$module" = main ] && light="-preScript DisableHeavyAnalyzers.java"
    echo "=== $module at 0x$base"
    # shellcheck disable=SC2086
    "$headless" "$project_dir" "$project" \
        -import "$elf_dir/$module.elf" -overwrite \
        -loader ElfLoader -loader-imagebase "$base" $light \
        -scriptPath "$script_dir" -postScript ExportInventory.java "$out_dir" \
        -log "$out_dir/$module.ghidra.log"
done
