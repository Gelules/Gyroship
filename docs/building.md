# Building

## Requirements

- Windows on a 64-bit x86 PC (there is no Linux build), with Rust's stable MSVC toolchain, 1.97 or
  later (Cranelift 0.137 asks for it), and Microsoft's C++ build tools: mimalloc compiles C code,
  and the game's object files go through the MSVC linker. Cargo downloads the dependencies.
- A GPU with a Vulkan driver and subgroups of exactly 32 lanes, and 16 GB of RAM (see [status.md](status.md)).
  Nothing checks the GPU before the game's first start: a refused one shows then, by exit code 3
  and a message that says why. The processor sets the frame rate in the first mission's city: 60
  on the test PC's 8 cores, down to 42 on the cloud PC's 4 (see [status.md](status.md)).
- Your own dump of the game, as plain folders: `exefs` and, next to it, `romfs`, decrypted, with the
  game's update applied. Gyroship decrypts nothing. It was developed against one build of the game;
  the game does not check the build at start (the launcher's Play tab says whether yours is the
  update the templates are made for), and whether another build runs is not known
  (`gyroship-tools nso info <exefs>/main` prints a build's ID).
- Optional: SDL3, as the 64-bit Windows `SDL3.dll` (the bindings follow the SDL 3.4 headers), for
  controllers and sound, and ffmpeg on the `PATH` for `--video`. The repository ships neither.

On a PC without Rust or the C++ build tools, winget installs them (the second one asks for
administrator rights):

    winget install --id Rustlang.Rustup --exact
    winget install --id Microsoft.VisualStudio.BuildTools --exact --override "--passive --wait --norestart --add Microsoft.VisualStudio.Workload.VCTools --add Microsoft.VisualStudio.Component.VC.Tools.x86.x64 --add Microsoft.VisualStudio.Component.Windows11SDK.26100"

Three executables are built with Cargo: `gyroship-tools.exe`, the command line that ports the game;
`gyroship.exe`, the game, ported from your dump by the tools; and `gyroship-launcher.exe`, an
optional window over both. After updating the repository, build them again, the tools before the
game.

The commands are written for Git Bash or PowerShell, from the repository root; in cmd.exe, program
paths take backslashes (`target\release\gyroship-tools`). Quote paths that contain spaces. The
launcher looks for the tools and the game in Cargo's default `target` folders of the repository
named in its Setup tab: leave `CARGO_TARGET_DIR` unset.

## gyroship-tools.exe

    cargo build --release

This builds every crate of the repository but the launcher, and gives one executable,
`target/release/gyroship-tools.exe`: not the game yet. Build the whole workspace this way, not
`-p gyroship-tools`: it also downloads every crate the game is built from. The tools find the
repository from their own place, in its `target/release` (or at its top, in a release): they work
wherever the repository is, as long as they stay there. A launcher already built, on this PC or
another, can install Rust and the C++ build tools through winget, and build the tools, on your
click ([playing.md](playing.md)).

## gyroship.exe, the game

`<exefs>` is the `exefs` folder of your dump: keep the dump outside the repository; Gyroship only
reads it, never changes it.

1. Recompile the game: Cranelift compiles every function of `<exefs>` into native object files,
   next to a Cargo project, `harness`, that links them with Gyroship's runtime into the game's
   executable. The untranslated instructions that `recompile` counts are not an error: a fallback
   interpreter runs them. It takes about two minutes on six threads (`--threads N`) and, when last
   measured, up to 3.4 GB of memory; it writes six object files, about 1.7 GB. `--probes FILE`
   compiles in a probe list, which a practice file needs (`templates/probes.txt`, see
   [practice.md](practice.md)).

       target/release/gyroship-tools recompile <exefs> --out generated/game

2. Build the generated project, in its folder: it gives `generated/game/target/release/gyroship.exe`,
   about 1.7 GB. Its first build also compiles the runtime and its dependencies, at the versions
   of the repository's `Cargo.lock` (the recompile copies it there), which the tools' build already
   downloaded; after a new recompile, a build only links the game again (about 16 s when last
   measured).
   `generated/game` then takes about 5.5 GB of disk.

       cd generated/game
       cargo build --release
       cd ../..

The project holds absolute paths: after moving `generated/game` or the repository, recompile and
build again. For controllers and sound, put `SDL3.dll` next to
`gyroship.exe`, or name it with `--sdl FILE`; `cargo clean` in `generated/game` deletes it.

## gyroship-launcher.exe

The launcher is a workspace of its own, built with Tauri 2, so that its dependencies stay out of
everything else:

    cd crates/launcher
    cargo build --release
    cd ../..

This gives `crates/launcher/target/release/gyroship-launcher.exe`, about 9 MB. A plain Cargo build is
enough, with no Node.js and no Tauri command line: the launcher's page is embedded in its
executable. Its first build compiled 224 crates in about two minutes. The executable needs no other
file: copied to a folder of its own, it keeps its settings there, out of `target`, which `cargo
clean` deletes. Its Setup tab can also build the tools (step 2), then recompiles and builds the
game.

## Making a release

A release is what players download, and what the launcher's updates install
([playing.md](playing.md)): one archive, `gyroship-<version>-windows-x64.zip`, holding a folder
`Gyroship-<version>` with the repository's files at the release's tag, and the two executables at
its top. A launcher looks for its updates in the repository its `Cargo.toml` names, through
GitHub's API: the repository must be public for that. For version 0.2.0, from the repository root:

1. Set `version = "0.2.0"` in `crates/launcher/Cargo.toml`. The tag must be `v` and exactly that
   version: a launcher compares its own version with the latest release's tag, and one that takes
   itself for an older version finds its own release out at each start.
2. Build the tools and the launcher without this machine's paths, which panic messages would
   otherwise carry: set `RUSTFLAGS` to
   `--remap-path-prefix=<repository>=. --remap-path-prefix=<Cargo's home>=cargo`, then run
   `cargo build --release` at the root and in `crates/launcher`.
3. Commit `crates/launcher/Cargo.toml` and `crates/launcher/Cargo.lock`, then tag that commit and
   push both:

       git tag v0.2.0
       git push origin HEAD v0.2.0

4. Make the archive and publish it; `--verify-tag` stops `gh` from making a tag of its own on
   another commit:

       git archive --format=zip --prefix=Gyroship-0.2.0/ --add-file=crates/launcher/target/release/gyroship-launcher.exe --add-file=target/release/gyroship-tools.exe -o gyroship-0.2.0-windows-x64.zip v0.2.0
       gh release create v0.2.0 gyroship-0.2.0-windows-x64.zip --verify-tag --title "Gyroship 0.2.0" --notes "What changed, for the players."
