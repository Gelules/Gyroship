# Status (November 2026)

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
  [Another vehicle in a mission](practice.md#another-vehicle-in-a-mission)), the first mission with the Blue-Marine and with the Landmaster,
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
- The launcher: one real session from its Play button (12 minutes with both Joy-Con, ended by the
  game with exit code 0); its other paths ran with stand-ins for the game, by a script clicking its
  buttons. `--play`, the Setup tab, the save copies, the Mod list, the debug menu box, the overlay
  boxes and the mouse mode box have not run on the real game.

## Known problems

- Updates: the launcher's updates were checked on synthetic archives only. No release is published
  yet, and the launcher sees one only once the repository is public: until then, each start notes
  in Setup that no release is published. A copy from a release, or any copy without a `.git`
  folder, installs a new version by itself; a Git clone is only told, and is updated with
  `git pull`, then its tools and game are built again.
- Offline by design: the emulated network services answer that there is no connection and no host
  socket is ever opened, so rankings, downloads, online play and GameShare do not work. Pressing
  GAMESHARE does nothing (by replay; not yet tried by hand), where a console without a connection
  would show an error, and the run's log file says why.
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
- `--renderer vulkan` and the precompiled shaders (the launcher's "Use the precompiled shaders")
  are new: they ran in replays without a window, with the same images as the usual way, and were
  never played by hand. On the test PC, the GPU driver's own cache already held every shader those
  replays met, so how much stutter they spare a first run is not measured yet.
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
  Mouse aiming from live Joy-Con (`--joycon-mouse`, [playing.md](playing.md)) was tried on one pair: both mice
  report, and the tutorial's mouse mode works, its posture screen passed with the right Joy-Con
  on its side (the left one's posture is untried). A replay can give a Joy-Con pair mice and
  acceleration (`ML=`, `MR=`, `AL=`, `AR=`, [development.md](development.md)): one took the game into its mouse mode from the
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
