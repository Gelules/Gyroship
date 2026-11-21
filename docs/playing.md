# Playing

Start the game from the repository root, with the `<exefs>` it was built from; its data is read from
the `romfs` folder next to it. `<save dir>` and `<cache dir>` are folders of your choice, best outside
the repository too, created when missing. Each run rewrites its journal: `boot.log` in the current
folder, or the file `--log FILE` names, in a folder that must exist. Git ignores `artifacts/`, not
`boot.log`, and the journal is derived from your game: create `artifacts/logs` first.

    generated/game/target/release/gyroship boot <exefs> --play --explore --save <save dir> --pipeline-cache <cache dir> --log artifacts/logs/play.log

Below, `gyroship` stands for `generated/game/target/release/gyroship`. It opens a window titled
"Gyroship", 1920 x 1080 at start;
closing it ends the run cleanly. `gyroship boot <exefs> --help` lists the 44 options by name; the
practice options are in [practice.md](practice.md), those of replays and checks in
[development.md](development.md), and the developer options these pages do not describe are
documented, for most, on the fields of `Config` in `crates/hle/src/lib.rs`. A misspelt option, a valued option given twice, or one left without its
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
(see the `ML=` words in [development.md](development.md)). Tried on one pair (see
[status.md](status.md#known-problems)); the mouse's counts go to the game unscaled.
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

## The launcher

`crates/launcher` is an optional window over the command line ([building.md](building.md) builds
it; a release has it at its top). It needs Microsoft's WebView2 runtime, which Windows 11 has. Start
`crates/launcher/target/release/gyroship-launcher`, or a copy of it, and first fill in its Setup
tab's Repository and Data folder fields, as full paths (the data folder must exist to play): the
tools and the generated game are found in the first, and `<data folder>` below is the second. A
launcher inside a copy of Gyroship fills in the repository itself.
While Setup lacks something (a folder not set, not a full path or not there, or no build of the
game yet), its tab shows in red with a warning sign, Setup lists what is missing, and the strip at
the bottom of Play says so.

- **Play**: the build to start (the generated one, or builds kept as
  `<data folder>/bin/<name>/gyroship.exe`, or `harness.exe` for those from before its renaming;
  a chosen build that is gone gives way to the first one found), your dump's `exefs` folder (the
  launcher checks that it holds `main`, with `romfs` next to it, and says whether its build is the
  update the templates are made for), the
  save and cache folders, the language, full screen, the Joy-Con and their mouse mode, the SDL3
  library, the practice tools and the recording of inputs, then Play. Paths are typed or pasted, as
  full paths: Play refuses a save, cache, hooks, mod or practice path that is not one, as the game
  would look for it from wherever the launcher runs. The game plays on the save folder itself, or on a fresh copy of it that the launcher
  makes at each start (`<data folder>/save-copy`: the folder itself is never written), or on a new,
  empty save at each start (`<data folder>/save-new`). Practice: the practice file, the boost mode
  and the three overlays, one box each (speed, frame rate, boost release markers); without a
  practice file they are greyed. Mods: Mod picks one layer folder of `<data folder>/practice` (a
  folder whose name starts with `layer`), shown by the first line of its `about.txt` and listed by
  it, so that alike mods sit together; the repository holds no layer. "Unlock the game's debug
  menu" starts the game with the repository's `templates/hooks.txt`, with any mod, unless the Hooks
  file field names a file of your own, which then always wins. The window closes while the game
  runs, so that its memory is the game's, and comes back with how the session ended.
  `gyroship-launcher --play` starts the game at once with the saved settings, for a shortcut: after
  a game that ended normally, the launcher ends too, without a window.
- **Setup**, in four steps, each with a lamp (green: done; red: to fix). Set up everything, at the
  top, does those not done yet in order, reading the lamps again after each one, and stops at the
  first that fails or stays undone; it waits for both folders and for the exefs folder set in Play.
  One by one:
  1. the two folders, with a button that sets the empty Save and Shader cache folders to
     `<data folder>\save` and `<data folder>\pipeline-cache`;
  2. what building and playing need: Rust 1.97 or later, Microsoft's C++ build tools and the
     Visual C++ runtime, each found without the network (Cargo's version, `vswhere`, the runtime's
     DLL) and, when missing, installed on your click through winget, Windows' package manager
     (`Rustlang.Rustup`; `Microsoft.VisualStudio.BuildTools` with the x64 C++ compiler and the
     Windows SDK, behind an administrator prompt, never restarting the PC by itself;
     `Microsoft.VCRedist.2015+.x64`). Clicking accepts the package's licence. Then Build the
     tools, `cargo build --release` in the repository: a release brings them, a Git clone builds
     them, and again after each update;
  3. Recompile, which translates the game's code into object files and a project in
     `generated/game` (about two minutes), with the repository's `Cargo.lock`;
  4. Build, which compiles Gyroship's runtime and links it with them into `gyroship.exe` (minutes
     the first time, then about 16 s).

  One step at a time, its output on the page. The builds use four jobs (`-j 4`, for 16 GB of RAM)
  and download the crates they lack, at the versions of that lock; they take
  `<data folder>/cargo` as cargo's home when that folder exists and `CARGO_HOME` is not set.

  Below the steps, **Updates**. When the launcher first opens its window, it asks GitHub for the
  latest release of the repository (through GitHub's API and Windows' own `curl`), unless "Look
  for updates at start" is off; a `--play` start whose game ends normally opens no window, so it
  asks nothing. Check now does the same at any time. When a newer release is out, a copy of
  Gyroship from a release installs it at once, without a question: the launcher downloads its
  archive (at most 30 minutes), unpacks it next to the current copy into a `.part` folder,
  carries the settings over (the repository, the game built in it, and the templates and
  `SDL3.dll` the new copy has, point to it; every other path stays) with the `SDL3.dll` put next
  to the game, renames the folder `Gyroship-<version>`, starts the new copy's launcher and ends. A
  failure before that rename leaves the current copy as it was. The new launcher then recompiles
  and builds the game for its version, before your eyes. Once the game is built, and in that
  window only, it offers to delete the previous copy; it refuses while that copy holds a file or
  folder of your settings, however the path is written, or while its launcher still runs. Later,
  delete the folder in the file explorer. A shortcut to the previous copy's launcher still starts
  that copy: point it to the new one. Nothing is downloaded for a folder that is already there: a
  release whose launcher kept the previous version number is said, not installed again. A Git
  clone is only told that a new version is out (update it with `git pull`, then build the tools
  and the game again).
- **Last session**: the exit code and what it means, the journal's summary line and the game's last
  messages.

The window fits its size: below about 760 pixels wide, the tabs move to the top, and below about
560 the fields go to one column. Its settings (`launcher.json`), its lock file and its window's data stay next to its
executable; journals and recordings go to `<data folder>/logs`. One launcher runs at a time: a
second one says so and ends. It opens no connection itself and goes online through other programs
only: at start, unless turned off, Windows' `curl` asks GitHub for a new release and downloads it;
on your click, its Install buttons start winget, which downloads from the Rust project or
Microsoft, and its builds let Cargo download the crates they lack. The game never goes online.
The WebView2 runtime that draws its window is Microsoft's and makes requests of its own. Checked with stand-ins for the game, by a
script clicking its buttons, then used for one real session from its Play button: 12 minutes with
both Joy-Con, ended by the game itself with exit code 0. `--play`, the Setup tab, the save copies,
the Mod list, the debug menu box, the overlay boxes and the mouse mode box have not run on the real
game. The updates were checked on synthetic archives only: no release is published yet. With the Joy-Con, tick "Show the game's messages": a window then shows the messages that say
when each one is connected.
