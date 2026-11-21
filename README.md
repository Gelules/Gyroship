# Gyroship

Gyroship is a native PC port of Star Fox (2026, Nintendo Switch 2), written in Rust. The game's
AArch64 executable is statically recompiled to native code and runs on a high-level emulation of the
console's system services; its graphics commands and shaders are translated to wgpu. It also holds
practice aids for speedrunners, which still need data files the repository does not provide, and a
write-up of the game's boost mechanics whose rules were checked against the running game.

- It contains no game code or assets and decrypts nothing. You supply the game's files yourself,
  dumped from your own console and already decrypted: its `exefs` folder (the executable's modules)
  and its `romfs` folder (the data). These files, the code generated from them and the outputs of
  its runs are never committed.
- The game never goes online. The emulated network services answer that there is no connection and
  no host socket is ever opened, so rankings, downloads, online play and GameShare do not work, by
  design: pressing GAMESHARE does nothing (by replay; not yet tried by hand), where a console
  without a connection would show an error, and the run's log file says why. (The optional
  launcher opens no connection either, but Microsoft's WebView2, which draws its window, makes
  requests of its own: see The launcher.)
- It is not for runs submitted to leaderboards: official runs are done on the console.

## Status (November 2026)

The game boots; its tutorial, its menus and its first mission play from start to end. It was
played on one PC, the test PC: a 2018 desktop CPU with 8 cores, 16 GB of RAM, a GeForce GTX 1080
with 8 GB of video memory, Windows 11. One other machine ran it, from recorded inputs only: a cloud
PC with 4 cores of a 2.8 GHz server CPU, 28 GB of RAM and an RTX A4500. No other player has tried
it yet.

- Played by hand by the author, in a window, with `--play --explore` (the real-time engine; never
  in the default strict mode, without `--explore`): the title screen, the tutorial, the menus and
  the first mission from start to end, several times. Eight missions, the first one among them,
  were loaded and played through the game's debug menu in an earlier build; later sessions played
  one Blue-Marine and two Landmaster missions, part of Bolse, and, with another vehicle (see
  Another vehicle in a mission), the first mission with the Blue-Marine and with the Landmaster,
  then part of Meteo with the Landmaster. Only the first mission is recorded as played from its
  start to its end.
- Replayed without a window with the default, deterministic engine: recorded inputs give the same
  journal (the run's log file) and image hashes from run to run, with rare exceptions from the
  host: the checks accept a few known image variants and extra texture reloads, one run once
  differed on 120 images, unexplained, and one once stopped on an apparent loss of the GPU device,
  then passed when run again. Changes are checked against two such replays, run with `--explore`:
  the tutorial (1,600 frames) and the first mission (11,500 frames). On the cloud PC, the
  tutorial's replay gave the same journal as on the test PC, but for the GPU's name. Entered through
  the game's own debug menu, all 16 missions of its development mission select start and reach
  gameplay in this engine; nothing more was checked for them.
- Frame rates, in play mode. On the test PC, from replays of recorded inputs: 60 frames per second
  on the title screen; about 45 in the main menu, where the GTX 1080 takes 21 to 22 ms per frame;
  30 in the first mission's intro cutscene, the rate the game asks for there; 60 in its city, with
  and without a window (in a window, about 2% of the frames of its heaviest 1,000 miss a refresh).
  Played by hand in an earlier build, the Blue-Marine and Landmaster missions gave 30 to 60; not
  measured since. On the cloud PC, from the same replay: 57 to 60 on the title screen and in the
  menu, but the city falls from 58 to 42
  as it fills, held back by the 4 cores: each frame needs 1.7 to 1.9 times the CPU time it needs
  on the test PC. That run ended before the city's heaviest part. The game's own clock stays in
  real time when frames are late.

Known problems:

- Graphics cards: two NVIDIA cards have run the game, the GTX 1080 and the cloud PC's RTX A4500.
  The A4500 froze while loading the tutorial in an earlier build; later builds ran there with the
  same driver, and the cause of the freeze was never proven. The game's shaders need GPU subgroups
  of exactly 32 lanes; a GPU without them is refused at start. No AMD or Intel GPU was tried, and
  this check is expected to refuse them. Vulkan is the backend in use; DX12 failed at the game's
  frame 71 when last tried.
- Memory: the game ran with 16 GB and with 28 GB of RAM. 16 GB is tight: in play mode its
  processes committed up to 15 GB in the first mission, and in an earlier build only 1.3 GB of RAM
  stayed free in its heaviest scene.
- First runs stutter: each shader is translated and each pipeline compiled when first needed (63 to
  125 ms per new material on the test PC), and in play mode objects can be missing when they first
  appear: a few frames as a rule, up to about two seconds on the test PC for a material the GPU
  driver takes long to compile. `--pipeline-cache` keeps both between runs. With empty caches, a
  replay of the first mission on the cloud PC compiled 444 pipelines while their draws waited, some
  for more than 5 s, with stops of up to 3 s; run again with the caches filled, it had almost no
  waits. Long frames remain anyway: in a window
  with filled caches, a replay from the title screen into the first mission's city (11,500 frames,
  menus and loadings included) had 13 and 15 frames of 100 ms or more in two runs on the test PC,
  the longest about 1 s.
- Play mode is experimental and says so at start; one memory fault was seen in it, once, cause
  unknown. 18 of the game's 2,828 shader programs do not translate yet; none is used when entering a
  mission. Without `--explore`, the run stops (exit code 72) at the first system call, draw or
  computation the emulation cannot do; with it, that one is logged or counted, then skipped, and the
  picture may miss something. One known case: a cutscene reloaded after the first mission's results has
  computations that cannot be done.
- Rendering is compared with the console by eye only; a few defects reported in earlier builds were
  never re-checked one by one. Sound needs SDL3 and a window; in play mode a gap or a dropped tick
  every few minutes is expected, and the sound trails the picture by about 0.1 s.
- No rumble, no button remapping, no amiibo (the reader never finds a figure), no system applet.
  Mouse aiming from live Joy-Con (`--joycon-mouse`, below) was tried on one pair: both mice
  report, and the tutorial's mouse mode works, its posture screen passed with the right Joy-Con
  on its side (the left one's posture is untried). A replay can give a Joy-Con pair mice and
  acceleration (`ML=`, `MR=`, `AL=`, `AR=`, below): one took the game into its mouse mode from the
  tutorial's question, through the console's controller screen (answered at once) and the
  lesson's posture screen, and the mouse then steered. That replay reaches the tutorial's mouse
  question in a minute or two, through a layer of edited level files, which the repository does
  not contain.
  The two mouse lines of the game's options (Preferences) behave as on the console: offered from
  the main menu with a save while a mouse is there (by replay, then with that pair), greyed in a
  new game's tutorial and without a mouse (by replay).
  Switch 2 Joy-Con give their buttons and sticks through `--joycon`, played on
  one pair only: the tutorial, then sessions of 6 and 12 minutes. A gamepad missing from SDL3's database is not recognised. The picture is
  stretched to the window, without 16:9 bars.

## Requirements

- Windows on a 64-bit x86 PC (there is no Linux build), with Rust's stable MSVC toolchain, 1.97 or
  later (Cranelift 0.137 asks for it), and Microsoft's C++ build tools: mimalloc compiles C code,
  and the game's object files go through the MSVC linker. Cargo downloads the dependencies.
- A GPU with a Vulkan driver and subgroups of exactly 32 lanes, and 16 GB of RAM (see Status).
  Nothing checks the GPU before the game's first start: a refused one shows then, by exit code 3
  and a message that says why. The processor sets the frame rate in the first mission's city: 60
  on the test PC's 8 cores, down to 42 on the cloud PC's 4 (see Status).
- Your own dump of the game, as plain folders: `exefs` and, next to it, `romfs`, decrypted, with the
  game's update applied. Gyroship decrypts nothing. It was developed against one build of the game;
  nothing checks the build at start, and whether another build runs is not known
  (`gyroship-tools nso info <exefs>/main` prints a build's ID).
- Optional: SDL3, as the 64-bit Windows `SDL3.dll` (the bindings follow the SDL 3.4 headers), for
  controllers and sound, and ffmpeg on the `PATH` for `--video`. The repository ships neither.

## Building

Three executables are built with Cargo: `gyroship-tools.exe`, the command line that ports the game;
`gyroship.exe`, the game, ported from your dump by the tools; and `gyroship-launcher.exe`, an
optional window over both. After updating the repository, build them again, the tools before the
game.

The commands are written for Git Bash or PowerShell, from the repository root; in cmd.exe, program
paths take backslashes (`target\release\gyroship-tools`). Quote paths that contain spaces. The
launcher looks for the tools and the game in Cargo's default `target` folders of the repository
named in its Setup tab: leave `CARGO_TARGET_DIR` unset.

### gyroship-tools.exe

    cargo build --release

This builds every crate of the repository but the launcher, and gives one executable,
`target/release/gyroship-tools.exe`: not the game yet. Build the whole workspace this way, not
`-p gyroship-tools`: it also downloads every crate the game is built from, which the launcher's
offline build of the game needs. The tools carry the repository's path: after moving the
repository, build them again.

### gyroship.exe, the game

`<exefs>` is the `exefs` folder of your dump: keep the dump outside the repository; Gyroship only
reads it, never changes it.

1. Recompile the game: Cranelift compiles every function of `<exefs>` into native object files,
   next to a Cargo project, `harness`, that links them with Gyroship's runtime into the game's
   executable. The untranslated instructions that `recompile` counts are not an error: a fallback
   interpreter runs them. It takes about two minutes on six threads (`--threads N`) and, when last
   measured, up to 3.4 GB of memory; it writes six object files, about 1.7 GB. `--probes FILE`
   compiles in a probe list, which a practice file needs (see Practice tools).

       target/release/gyroship-tools recompile <exefs> --out generated/game

2. Build the generated project, in its folder: it gives `generated/game/target/release/gyroship.exe`,
   about 1.7 GB. Its first build also compiles the runtime and its dependencies (the launcher's
   build of it adds `--offline`, and then uses only the crates the tools' build downloaded); after
   a new recompile, a build only links the game again (about 16 s when last measured).
   `generated/game` then takes about 5.5 GB of disk.

       cd generated/game
       cargo build --release
       cd ../..

The project holds absolute paths: after moving `generated/game`, recompile and build again; after
moving the repository, build the tools first. For controllers and sound, put `SDL3.dll` next to
`gyroship.exe`, or name it with `--sdl FILE`; `cargo clean` in `generated/game` deletes it.

### gyroship-launcher.exe

The launcher is a workspace of its own, built with Tauri 2, so that its dependencies stay out of
everything else:

    cd crates/launcher
    cargo build --release
    cd ../..

This gives `crates/launcher/target/release/gyroship-launcher.exe`, about 9 MB. A plain Cargo build is
enough, with no Node.js and no Tauri command line: the launcher's page is embedded in its
executable. Its first build compiled 224 crates in about two minutes. The executable needs no other
file: copied to a folder of its own, it keeps its settings there, out of `target`, which `cargo
clean` deletes. Its Setup tab recompiles and builds the game, never the tools: build
`gyroship-tools.exe` first.

## Playing

Start the game from the repository root, with the `<exefs>` it was built from; its data is read from
the `romfs` folder next to it. `<save dir>` and `<cache dir>` are folders of your choice, best outside
the repository too, created when missing. Each run rewrites its journal: `boot.log` in the current
folder, or the file `--log FILE` names, in a folder that must exist. Git ignores `artifacts/`, not
`boot.log`, and the journal is derived from your game: create `artifacts/logs` first.

    generated/game/target/release/gyroship boot <exefs> --play --explore --save <save dir> --pipeline-cache <cache dir> --log artifacts/logs/play.log

Below, `gyroship` stands for `generated/game/target/release/gyroship`. It opens a window titled
"Gyroship", 1920 x 1080 at start;
closing it ends the run cleanly. `gyroship boot <exefs> --help` lists the 41 options by name; the
developer options this file does not describe are documented, for most, on the fields of `Config`
in `crates/hle/src/lib.rs`. A misspelt option, a valued option given twice, or one left without its
value at the end of the line stops the start with exit code 2 (any next word is taken as the value).

| Option | Effect |
| --- | --- |
| `--play` | The real-time engine, the one to play with: one host thread per game thread, on the wall clock, and the GPU's work on threads of its own. |
| `--explore` | What the emulation cannot do is logged or counted, then skipped, instead of ending the run; every session played so far used it. |
| `--late-frames` | In play mode, a frame later than its vsync is shown at once, not at the next vsync: more frames, an uneven cadence. |
| `--fullscreen` | Borderless fullscreen from the start; Alt+Enter switches at any time. |
| `--save DIR` | Saves are loaded from `DIR` and written back there each time the game commits them. |
| `--language CODE` | The console language the game reads at boot to pick one of its 14 text regions; default `en-US`. |
| `--pipeline-cache DIR` | Keeps between runs the GPU driver's compiled pipelines (one file per GPU, Vulkan only) and, in `DIR/shaders`, the translated shaders. |
| `--backend VALUE` | `auto` (the default: Vulkan and DX12 are both offered to wgpu, which took Vulkan on the test PC), `vulkan` or `dx12`. |
| `--sdl FILE` | The SDL3 library to load, instead of `SDL3.dll` next to the game's executable. |
| `--joycon` | Switch 2 Joy-Con over Bluetooth LE, without pairing (Windows 11): see below. |
| `--joycon-mouse` | `--joycon`, plus each Joy-Con's mouse and accelerometer, for the game's mouse mode: see below. |

Without `--save`, every run starts from an empty save kept in memory, and progress is lost; `DIR`
holds one sub-folder per save, with the game's save files as plain files. Under `--pipeline-cache
DIR`, `DIR/shaders` holds one folder per build of the shader translator; those of other builds are
never removed, so delete `shaders` to reclaim space. The cache is derived from the game: keep it out of any repository.
`--language` takes, in any letter case, `ja`, `en-US`, `fr`, `de`, `it`, `es`, `zh-CN`, `ko`, `nl`,
`pt`, `ru`, `zh-TW`, `en-GB`, `fr-CA`, `es-419`, `zh-Hans`, `zh-Hant`, `pt-BR`, `pl` or `th`; `ru`,
`pl` and `th` have no region of their own and give US English. Only English, French and Japanese
were seen running. The command line and the logs stay in English.

The keyboard always works, by physical key (US labels: the same places on any layout). No key moves
the right stick or clicks a stick; Alt+Enter and G (the practice switch) never reach the game.

    key -> console button
    Enter -> A        Q -> L     P -> Plus     arrow keys -> D-pad
    Backspace -> B    E -> R     M -> Minus    W A S D -> left stick, at full deflection
    X -> X, Y -> Y    1 -> ZL    3 -> ZR

Controllers and sound go through SDL3, loaded at run time. If the start prints `gyroship: no SDL3
(...): keyboard only, no sound`, the file was not found or could not be loaded; the window is then
meant to run with the keyboard alone and no sound, which was never tried. A gamepad SDL3 recognises
should work; plugging and unplugging while the game runs are handled in the code, unplugging never
tried. Buttons go by position, as on the console: the east face button is A, south is B, north is X,
west is Y, so A is the button labelled B on an Xbox pad and Circle on a PlayStation pad. The
shoulders are L and R, a trigger past half its travel is ZL or ZR, Start is Plus and Back is Minus.
The game sees one controller: several gamepads and the keyboard act as one. An Xbox 360 pad was
played by hand; a PS5 pad was only tried on the title screen.

Switch 2 Joy-Con do not go through SDL3 nor through Windows' Bluetooth settings (never add them
there: they refuse the standard pairing). With `--joycon`, Gyroship scans for them itself, connects
without pairing and reads their reports; it stores nothing on them, and the only things it sends
are player light 1 and a read of the factory stick calibration. Turn the console off, start the
game, then hold the sync button of each Joy-Con (the small round one between SL and SR) until the
terminal says it is connected: after a plain press a Joy-Con only calls its console and drops
anyone else. Do it again after a loss. The pair acts as one more gamepad: its buttons where the
console has them, Home as a gamepad's Guide; SL, SR, Capture and C do nothing; no rumble, no
gyroscope (no script of the game reads it). With `--joycon-mouse` instead, Gyroship also sends each Joy-Con
the console's own feature mask and its enabling (two more fixed frames, each sent once the
controller has answered the one before), and gives the game each Joy-Con's mouse (its moves, and
whether it rests on a surface) and acceleration, as the `ML=`, `MR=`, `AL=` and `AR=` words of a
replay do; `--record` writes them. The terminal says when a Joy-Con's mouse first reports, or why
it did not start. While a mouse is there, the game sees a Joy-Con pair instead of the gamepad
(below). Tried on one pair (see "Known problems"); the mouse's counts go to the game unscaled.
Windows
11 is needed for a fast link: Gyroship asks for Windows' 15 ms
connection interval (60 ms otherwise) and prints each link's interval whenever it changes. A
Joy-Con that is lost, or silent for 0.4 s, holds no button until its reports come back. With
another gamepad plugged in, of its stick and the Joy-Con's the one pushed farther counts. The
protocol is not public: this follows community research and measurements on one pair (firmware
2.1.9), which then played through `--joycon`, both links at 15 ms: the tutorial, then sessions of 6
and 12 minutes (the first mission twice, Macbeth, Bolse). The stick clicks and the right stick's full travel were not
used, and no Joy-Con was switched off in flight.
`cargo run --release -p gyroship --example joycon_trial -- <L|R|scan> <log file>` is the trial those
measurements came from; it stores nothing on the controller.

## Practice tools

Read this first. These options need two files made for your own build of the game, which hold its
addresses and structure offsets: a probe list, compiled into the game, and a practice file. The
repository holds neither, and no tool builds them from a dump: `templates/` holds both to fill in,
and Making a practice file, below, says how the values were found. QUIT GAME, the game's debug menu
and its mission timer need a hooks file or a layer folder of replacement files instead, which the
repository does not hold either. There are no save states, no speed change and no menu of
Gyroship's own; options are chosen at launch.

### Using a practice file

1. Compile the probe list into the game: recompile with `--probes <probe list>`, then build the
   game again (see Building); a changed list means both again. In the launcher: the Setup tab's
   Probe list field, then its two steps.
2. Start the game with `--practice <practice file>`, and `--boost infinite`, `--boost perfect` or
   `--hud`, which need it. In the launcher: the Practice file field, then Boost and "Speed, frame
   rate and release markers on screen".

       gyroship boot <exefs> --play --explore --practice <practice file> --boost perfect --hud

A line that breaks the rules below, or whose `<pc>` is not a `trace:` probe compiled into the game,
stops the start with exit code 2 and a message that names the line. Nothing else is checked: not
that a probe sits at the right place, nor where a field points, and `--boost infinite` writes
wherever its fields point.

A probe list has one `<hex address> <name>` per line (`#` starts a comment): Gyroship is called each
time the game's code passes that address, and a name `trace:<label> <item>...` makes a probe that a
practice line can name. A practice file has one action per line; a comment is a whole line that
starts with `#`, never the end of a line:

    boost <pc> <vehicle> name=TYPE@EXPR ...     a vehicle's boost update and the fields it uses
    hud-speed <pc> s=ITEM m=ITEM b=ITEM         where to read the speed shown, m * (s + b)
    hud-at <x> <y>                              where the texts go, in pixels of the image
    marker <vehicle> x0 y0 x1 y1 x2 y2 x3 y3    a release marker in the boost bar, as a quad
    font <region tag | latin> <entry>           the asset archive entry that holds a region's font
    ms-keep <pc> ms=wN                          the milliseconds of a level time about to be shown
    ms <pc> out=xN                              its two-digit seconds, which gain them

`<pc>` is the address, in hex without `0x` (unlike in the probe list), of a `trace:` probe compiled
into the game; `<vehicle>` is `arwing`, `blue-marine` or `landmaster`. EXPR is a register `xN`, a
hex number written with `0x`, or `[EXPR]`, followed by `+0xHEX` or `-0xHEX` offsets; an ITEM of
`hud-speed` is `sN` (a float register) or `f32@EXPR`. Boost fields are `gauge`, `speed`, `lock`,
`boosting` and the settings `warp`, `accel`, `decel`, `max`, `brake`, `brake_recover`,
`brake_floor`, `gauge_max`, `fill`, `drain`: `u8@EXPR` for `lock` and `boosting`, `f32@EXPR` for
the others; `--boost infinite` needs `gauge` and `lock`, `--boost perfect` all 14 except on a
`landmaster` line. At most four markers per vehicle are drawn. A region tag names one of the game's
14 text regions as `ll-CC` (`en-US`, `fr-FR`, `ja-JP`, `es-419`...), not a `--language` code;
`<entry>` is the entry's name, a hex number without `0x`; font lines are read only with `--hud`. An
`ms` line takes the milliseconds that the last `ms-keep` line passed on the same game thread kept,
whatever their order in the file.

- `--boost infinite`: as each boost update starts, the meter goes back to a full bar and the
  boost-and-brake lock is lifted, so Boost held never runs out. Verified by deterministic replays on
  the three vehicles; played by hand in the Arwing and the Landmaster, not in the Blue-Marine. At
  rest its bar keeps its active look.
- `--boost perfect`: hold Boost, and Gyroship ends each boost at the moment the research found
  best. Arwing: 30 frames of boost, then 60 of cooldown. Blue-Marine: boosts of 68, 45 and 34 frames
  from rest, then 27 frames every time (1/2, 1/3, 1/4, then 1/5 of the meter): the rule a player can
  follow by hand, not the frame-perfect schedule. Landmaster: nothing to do, holding is best. The
  controller's Guide (PS, Home) button, or the G key, switches it off for full boosts and on again.
  Verified by deterministic replays, bit for bit against the model; the switch is only unit-tested.
  Not covered: the Arwing after a boost gate, the Landmaster's leans.
- `--hud`: at the `hud-at` position, the vehicle's speed and the frames per second, drawn with one
  of the game's fonts read from your dump (a `font` line is required), in the game's language. In
  the boost bar, the magenta `marker` quads of the vehicle in flight show where to let go; the file
  used so far has one for the Arwing and four for the Blue-Marine. Markers need the vehicle's
  `boost` line; on the Arwing and the Blue-Marine it must carry the ten settings, and no marker
  shows while they differ from the measured ones. At the start of a flight they show about a second
  before the game's own HUD. Shown in the window and in dumped frames only, never in the hashes, the
  video or the journal. Seen on the three vehicles, in English, French and Japanese.
- `ms-keep` and `ms` lines: the level time to the millisecond, with no option beyond `--practice`.
  Where the game formats a level time, its two-digit seconds gain a point and three digits, so the
  game itself shows "MM:SS.mmm" on its results screen and in its own mission timer, which shows only
  when a setting of the game is switched on through a layer file. The time advances by one sixtieth
  of a second per game update: the three digits show that step, not a finer clock. Verified by a
  deterministic replay up to the results screen; not compared with the console's time.

### Making a practice file

Copy `templates/practice.txt` and `templates/probes.txt` outside the repository and fill them in:
their comments say what goes where. The values hold for one build of the game: write its build ID
(`gyroship-tools nso info <exefs>/main`) at the top of both. The author found them this way:

1. Export the game's modules with `gyroship-tools nso2elf` and import them into Ghidra with
   `tools/ghidra/analyze_modules.sh`, from Git Bash (its first lines give its arguments;
   `GHIDRA_DIR` must name the folder that holds `ghidra_12.2_PUBLIC`, the Ghidra it calls). It
   places each module where Gyroship does: an address in Ghidra is an address for the probe list.
2. Find the functions. The game's logic is made of visual scripts compiled to native code, which
   keep their function and property names as strings: `tools/ghidra/FindByStrings.java`, a
   post-script for Ghidra's `analyzeHeadless` run on that project with `-process main.elf
   -noanalysis` (its first lines give its arguments), lists and decompiles the functions that refer
   to matching strings (`boost`, for example).
3. In the decompiled code, follow each value from the function's arguments through its pointers,
   and write that path as an EXPR.
4. Put a `trace:` probe at each place, with the items of its practice line without their names
   (`f32@EXPR`, not `gauge=f32@EXPR`), compile the list in, and replay recorded inputs with
   `--trace FILE`: one line per pass, with each item's value. Check that each field moves as the
   game's rules say (`research/boost/README.md`): the meter rising while boosting, the lock set at
   each release with anything in the meter and when the meter is full, then cleared once it is
   back to 0.
5. Markers: dump frames of the boost bar with `--dump-frames`, fit its fill edge against the traced
   meter, and draw a thin quad where the edge stands at the release level. Fonts:
   `gyroship-tools datazip-find --zip <archive> fonts/` lists the archive entries that hold a font
   (no leading slash: Git Bash would turn `/fonts/` into a Windows path), and
   `gyroship-tools text-sheet --zip <archive> --practice <file> --out <png>` draws Gyroship's texts
   in the 14 regions with them.
6. Last, replay with `--practice`, `--boost` and `--hud`, and look at the dumped frames.

### The game's own debug menu

The game contains its developers' debug tools, locked in the retail build. Mission select and the
shield, wings, bombs, lives and laser cheats are those of its debug menu, not Gyroship's. Two
generic mechanisms unlock that menu, fed by your own files:

- `--romfs-layer DIR`: a file of `DIR` replaces the dump's file at the same path under `romfs`; one
  the dump lacks is ignored. Each file of `DIR/<archive>.entries`, where `<archive>` is the asset
  archive's path under `romfs`, replaces the entry of that name in the game's asset archive, as its
  decoded content; a name the archive lacks stops the start. The dump is only read.
- `--hooks FILE`: one `<hex address> <hook>` per line (`#` starts a comment). The game function at
  that address, which must be one the game reaches by indirect calls, is replaced by the hook.
  `requirement-met` makes a menu button's requirement always met. `quit`, written `<hex address>
  quit <hex address of a byte>`, takes the place of a menu button's handler that does nothing: it
  sets the byte the game's frame loop polls to end the application, and the game shuts itself down.

A run with a layer, hooks or practice actions that write is never a reference run: its journal says
what was changed. The debug menu writes to the save: point `--save` at a copy while using it.

### QUIT GAME

QUIT GAME is added in two places, in the game's 14 text regions, each through replaced entries of
the game's asset archive built from the player's own dump: a third choice in the QUIT confirmation
of the pause menu (no hook: the game's own handler quits), and a sixth line of the title menu, under
TRAINING, which needs the `quit` hook (without it the line shows and does nothing). The game then
shuts itself down and the process exits with code 0; the save is unchanged. Verified by
deterministic replays in English, French and Japanese and by a real-time run without a window, then
used in a window, in French: the title menu line twice and the pause menu choice three times, in
two missions (not in the tutorial). The repository holds only the generic mechanism and no script
that builds those entries: this is not packaged for other players yet.

### Another vehicle in a mission

An experiment made the same way, with replaced archive entries built from the player's own dump: a
mission can spawn another of the game's three vehicles than its own. Two entries per mission are
rebuilt: its root level, whose player spawner then names the other vehicle, and its package list,
which then loads the files that vehicle needs, inserted where the list's dependency order wants
them. Tried without a window on seven pairs of mission and vehicle, among them each vehicle in a
mission of each other kind: the vehicle is there, its code runs, the mission goes on and nothing is
journaled as an anomaly. Then, in a window: the first mission with the Blue-Marine for about 9
minutes, and with the Landmaster to its results screen, then Meteo with the Landmaster for 2
minutes. As for QUIT GAME, the repository holds no script that builds those entries.

## The launcher

`crates/launcher` is an optional window over the command line (see Building for its build). It
needs Microsoft's WebView2 runtime, which Windows 11 has. Start
`crates/launcher/target/release/gyroship-launcher`, or a copy of it, and first fill in its Setup
tab's Repository and Data folder fields, as full paths (the data folder must exist to play): the
tools and the generated game are found in the first, and `<data folder>` below is the second.

- **Play**: the build to start (the generated one, or builds kept as
  `<data folder>/bin/<name>/gyroship.exe`, or `harness.exe` for those from before its renaming;
  a chosen build that is gone gives way to the first one found), your dump's `exefs` folder, the
  save and cache folders, the language, full screen, the Joy-Con and their mouse mode, the SDL3
  library, the practice tools and the recording of inputs, then Play. Paths are typed or pasted, as
  full paths. The game plays on the save folder itself, or on a fresh copy of it that the launcher
  makes at each start (`<data folder>/save-copy`: the folder itself is never written), or on a new,
  empty save at each start (`<data folder>/save-new`). Mod picks one layer folder of
  `<data folder>/practice` (a folder whose name starts with `layer`), shown by the first line of
  its `about.txt`; the repository holds no layer. The window closes while the game
  runs, so that its memory is the game's, and comes back with how the session ended.
  `gyroship-launcher --play` starts the game at once with the saved settings, for a shortcut: after
  a game that ended normally, the launcher ends too, without a window.
- **Setup**: those two fields, then the recompile and the build of the game (see Building), one at
  a time, their output on the page; the tools' build stays yours. The build step uses four jobs (`-j 4`, for 16 GB of RAM)
  and never fetches a crate (`--offline`): its crates are those the tools' build fetched. It takes
  `<data folder>/cargo` as cargo's home when that folder exists and `CARGO_HOME` is not set.
- **Last session**: the exit code and what it means, the journal's summary line and the game's last
  messages.

Its settings (`launcher.json`), its lock file and its window's data stay next to its executable;
journals and recordings go to `<data folder>/logs`. One launcher runs at a time: a second one says
so and ends. It opens no
connection and downloads nothing itself; the WebView2 runtime is Microsoft's and makes requests of
its own. Checked with stand-ins for the game, by a script clicking its buttons, then used for one
real session from its Play button: 12 minutes with both Joy-Con, ended by the game itself with exit
code 0. `--play`, the Setup tab, the save copies, the Mod list and the mouse mode box have not run
on the real game. With the Joy-Con, tick "Show the
game's messages in a console": the messages that say when each one is connected go there.

## Repeatable runs and verification

Without `--play`, the default engine runs every game thread as a fiber on one host thread, over six
virtual cores with clocks of their own: a discrete-event simulation, designed so that every run is
the same (Status gives the exceptions seen); it slows the game down whenever the PC is late. Its
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

## Research

`research/` holds write-ups of the game's mechanics, each with the code that reproduces its numbers;
no game files, logs or captures. One so far, `research/boost/README.md`: how each vehicle's boost
meter and boost speed work, and when to let go (the rules that `--boost perfect` applies, above).
The same rules, in Rust in `crates/research`, match the running game to the last bit on 21,191
frames of six deterministic runs. Not covered yet: boost gates, loops and U-turns, the Landmaster's
leans, the Blue-Marine by hand, and frame rates other than 60 per second, which a slower PC can
fall to.

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
    templates/        a practice file and its probe list, to fill in for your build
    research/         write-ups of the game's mechanics, with the code behind their numbers
    vendor/naga       naga 30.0.1 with one performance patch
    generated/, artifacts/   ignored by Git: code and run outputs derived from your game
