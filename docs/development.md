# Verification and tools

## Repeatable runs and verification

Without `--play`, the default engine runs every game thread as a fiber on one host thread, over six
virtual cores with clocks of their own: a discrete-event simulation, designed so that every run is
the same ([status.md](status.md) gives the exceptions seen); it slows the game down whenever the PC is late. Its
journal is stamped with virtual time, never with wall-clock time or a host address, so two runs
compare byte for byte. Such a run can serve as a reference: a later run is compared with its journal
and its hashes.

    gyroship boot <exefs> --headless --explore --replay <inputs.txt> --hashes <hashes.csv> --log <run.log>

| Option | Effect |
| --- | --- |
| `--headless` | No window, no sound output, no live input; the host GPU is used only to dump, film, measure or hash frames, or with `--gpu`. |
| `--frames N`, `--seconds N` | End after N presented frames, or stop past N seconds (virtual time; real time in play mode). |
| `--replay FILE`, `--record FILE` | Play back, or write, a file of inputs. |
| `--dump-frames DIR` | One PNG per render target of each frame; `--dump-every K` and `--dump-from N` choose the frames. |
| `--hashes FILE` | A CSV with a hash of each presented image. |
| `--video FILE`, `--dump-audio FILE` | The presented images as H.264 video, through ffmpeg (`--dump-from` and `--dump-every` choose them too); everything the game plays, as a float WAV. |
| `--perf FILE`, `--trace FILE` | Each frame's host time, and its GPU time where the GPU can time it, as CSV, with a summary on stderr and files beside it with each frame's and each thread's costs; one line per pass of each `trace:` probe, which the journal never sees. |

A headless run with neither `--frames` nor `--seconds`, and no replay, stops after 60 s. Make
`<inputs.txt>` with `--record <inputs.txt>` in a window run: it is written when the run ends, in a
folder that must exist (a file that cannot be written is reported only then, and the recording is
lost). Recorded with `--play`, it does not replay that session exactly; its replay with the default
engine repeats. The file has one line per change of state, `<sample> <buttons> [L=x,y] [R=x,y]
[K=usage,...] [ML=dx,dy,c] [MR=dx,dy,c] [AL=x,y,z] [AR=x,y,z]`, then `<sample> end`, which stops
the run. Samples are the game's input samples, 60 per second; button names (`A`, `B`, `X`, `Y`,
`L`, `R`, `ZL`, `ZR`, `LS`, `RS`, `Plus`, `Minus`, `Left`, `Up`, `Right`, `Down` and a few more
that `--record` writes) are joined by `+`, and `-` stands for none; sticks go from -32767 to
32767; `K=` holds keyboard keys as hex USB HID usages; `ML=` and `MR=` give the left and right
Joy-Con a mouse: its counts in X and Y at each sample while the line holds, then 1 when it rests
on a surface (0 in the air); `AL=` and `AR=` give their acceleration, in thousandths of g (the
game takes a left Joy-Con whose x is above 0.9 g, or a right one whose x is below -0.9 g, for one
lying on its side). A mouse or an acceleration that a line drops is gone: its Joy-Con gets one
null state. While a line has a mouse, Npad 0 shows a Joy-Con pair: used as mice when the game
supports that style but not FullKey's (it asked for mouse controls), else a plain pair when the
game supports one, as the console shows a pair held apart (the game's options offer mouse aiming
only then); otherwise (no mouse, or neither pair style supported) the FullKey controller. The console's controller screen, when the game opens it, closes at
once: confirmed with Npad 0 if the style Npad 0 shows is one the game supports, cancelled
otherwise. The tutorial's mouse lesson then waits for exactly one Joy-Con lying on its side with
its mouse moving: a replay passed with `AR=-1000,0,0` and `MR=` moving 120 counts per sample, not
with 6. `--joycon-mouse` gives them from live Joy-Con. `#` starts a
comment.

Play mode is never a reference: its threads are scheduled by Windows on the real clock, so runs do
not repeat; its journal is stamped with real time; a replay there follows the wall clock, so the
game sees the inputs at other moments than in the recorded run. Play in play mode, prove with the
default engine. `--dump-frames` in play mode without a window crashed when last tried, before frames
were recorded on a thread of their own; it was not tried since.

Exit codes of the game's executable: 0 a normal end (the game exited by itself, `--frames` reached,
a replay's `end`, the window closed); 2 a bad command line, a start that cannot go on, or ffmpeg
failing under `--video`; 3 a GPU that cannot run the game; 71 a fault in the recompiled game (a
memory fault, or an instruction that cannot be executed); 72 a run the emulation stopped (an
unknown system call, service or command, or an anomaly, which `--explore` lets pass; the game
aborting; a save that cannot be read or written; the time limit; a deadlock; in play mode, one of
the GPU's threads that ended).

## gyroship-tools

    nso info <module>                    one module: segments, build ID, symbols, relocations
    nso2elf <module>... --out <dir>      AArch64 ELF files for disassemblers
    link <exefs>                         loads, relocates and links the modules; unresolved imports
    discover <exefs>                     functions, control-flow graphs and jump tables, counted
    functions <module>                   function ranges from the unwind tables; --compare FILE
    recompile <exefs> --out <dir>        every function through Cranelift, and the game's project
    experiment <exefs> --out <dir>       a sample of functions as a project to build and measure
    measure <program> [args...]          wall time and peak memory of a process tree (Windows)
    capture --title <text> --out <png>   one frame of a window found by its title (Windows)
    gpfifo <dir>                         replays a --dump-gpfifo folder and reports each submit
    shaders <archive>                    inventory of the shaders in the game's asset archive
    sass <file>                          lists a graphics program dumped with --dump-shaders
    translate <file>...                  translates dumped programs to WGSL and validates them
    datazip-find --zip <archive> <pattern>...   entries of the asset archive that hold a byte pattern
    text-sheet --zip <archive> --practice <file> --out <png>   Gyroship's in-game texts, as an image
    boost-check <trace>...               the boost model against --trace files, bit for bit

`gyroship-tools <subcommand> --help` gives each one's options. Exit codes: 0, 1 (a check failed),
2 (an error).

## Repository layout

    crates/loader     executables: modules, relocations, linking, ELF export
    crates/analysis   AArch64 decoder, function discovery, control-flow graphs, jump tables
    crates/recomp     AArch64 code generator (Cranelift object files)
    crates/runtime    guest CPU context and memory, dispatch, fallback interpreter, profiler
    crates/hle        system services, boot and options, both engines, saves, practice, hooks;
                      in play mode, the GPU and decoder threads
    crates/gpu        the console's GPU commands and shaders on wgpu, the frame recorder (in play
                      mode, its own thread and the submit thread); caches; asset archive reader
    crates/practice   the texts Gyroship adds in game, and the reader of the game's fonts
    crates/research   the three vehicles' boost, reimplemented and checked against the game
    crates/gyroship   the host side, a library: window, pacing, keyboard, SDL3 controllers and sound, Joy-Con
    crates/tools      the gyroship-tools command line
    crates/launcher   the launcher: a Tauri window over the command line (a workspace of its own)
    tools/ghidra      headless Ghidra scripts (not needed to play)
    templates/        the practice file, its probe list and the hooks file, for the targeted build
    docs/             building, playing, practice tools and mods, status, verification
    research/         write-ups of the game's mechanics, with the code behind their numbers
    vendor/naga       naga 30.0.1 with one performance patch
    generated/, artifacts/   ignored by Git: code and run outputs derived from your game
