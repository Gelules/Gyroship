# Practice tools and mods

The practice tools read the game's own memory, at places that differ from one build of the game to
another. `templates/` holds the three files they need for the update Gyroship targets, whose build
ID heads each of them (`gyroship-tools nso info <exefs>/main` prints yours; the launcher's Play tab
says whether your dump is that update): `probes.txt`, the probe list compiled into the game;
`practice.txt`, the practice file; `hooks.txt`, which unlocks the game's debug menu. With that
build, use them as they are. With another one, Making a practice file, below, says how the values
were found. The
mods (QUIT GAME, the game's mission timer, another vehicle in a mission, the eShop demo's lines,
the tutorial's mouse lesson) are layer folders of replacement files that `gyroship-tools mods`
builds from your own dump (Building the mods, below): the repository holds no file of the game,
only the values read in it that the builds need, in `templates/mods.txt`. There are no save
states, no speed change and no menu of Gyroship's own; options are chosen at launch.

## Using a practice file

1. Compile the probe list into the game: recompile with `--probes templates/probes.txt`, then build
   the game again (see [building.md](building.md)); a changed list means both again. In the
   launcher: the Setup tab's Practice probes field, then Recompile and Build.
2. Start the game with `--practice templates/practice.txt`, and the options that need it: `--boost
   infinite` or `--boost perfect`, and the overlays, `--hud-speed`, `--hud-fps` and `--hud-markers`,
   one each, or `--hud` for all three. In the launcher: the Practice file field (the full path of
   `templates/practice.txt`), then Boost and the three On screen boxes.

       gyroship boot <exefs> --play --explore --practice templates/practice.txt --boost perfect --hud-speed --hud-markers

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
`<entry>` is the entry's name, a hex number without `0x`; font lines are read only with an overlay. An
`ms` line takes the milliseconds that the last `ms-keep` line passed on the same game thread kept,
whatever their order in the file.

- `--boost infinite`: as each boost update starts, the meter goes back to a full bar and the
  boost-and-brake lock is lifted, so Boost held never runs out. Verified by deterministic replays on
  the three vehicles; played by hand in the Arwing and the Landmaster, not in the Blue-Marine. At
  rest its bar keeps its active look.
- `--boost perfect`: hold Boost, and Gyroship ends each boost where the research says to let go.
  Arwing: 30 frames of boost, then 60 of cooldown. Blue-Marine: boosts of 68, 45 and 34 frames
  from rest, then 27 frames every time (1/2, 1/3, 1/4, then 1/5 of the meter): the rule a player can
  follow by hand, not the frame-perfect schedule. Landmaster: nothing to do, holding is best. The
  controller's Guide (PS, Home) button, or the G key, switches it off for full boosts and on again.
  Verified by deterministic replays, bit for bit against the model; the switch is only unit-tested.
  Not covered: the Arwing after a boost gate, the Landmaster's leans.
- The overlays, each on its own: `--hud-speed`, at the `hud-at` position, the vehicle's speed;
  `--hud-fps`, the frames per second, under the speed line, or in its place without it;
  `--hud-markers`, in the boost bar, the magenta `marker` quads of the vehicle in flight, where to
  let go (`templates/practice.txt` has one for the Arwing and four for the Blue-Marine). `--hud`
  draws all three. The texts use one of the game's fonts read from your dump, in the game's
  language: any overlay needs a `font` line. Markers need the vehicle's `boost` line; on the Arwing
  and the Blue-Marine it must carry the ten settings, and no marker shows while they differ from the
  measured ones. At the start of a flight they show about a second before the game's own HUD.
  Shown in the window and in dumped frames only, never in the hashes, the video or the journal. Seen
  on the three vehicles, in English, French and Japanese.
- `ms-keep` and `ms` lines: the level time to the millisecond, with no option beyond `--practice`.
  Where the game formats a level time, its two-digit seconds gain a point and three digits, so the
  game itself shows "MM:SS.mmm" on its results screen and in its own mission timer, which shows only
  when a setting of the game is switched on through a layer file. The time advances by one sixtieth
  of a second per game update: the three digits show that step, not a finer clock. Verified by a
  deterministic replay up to the results screen; not compared with the console's time.

## Making a practice file

For another build of the game, copy `templates/practice.txt`, `templates/probes.txt` and
`templates/hooks.txt`, and change each address and offset for that build; their comments say what
each line points at. The author found the values this way:

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
6. Last, replay with `--practice`, `--boost` and the overlays, and look at the dumped frames.

## The game's own debug menu

The game contains its developers' debug tools, locked in the retail build. Mission select and the
shield, wings, bombs, lives and laser cheats are those of its debug menu, not Gyroship's. No mod
unlocks it: `--hooks templates/hooks.txt` does, for the build Gyroship targets, with any mod or
none (the launcher's "Unlock the game's debug menu" box). Two generic mechanisms are behind it and
behind the mods:

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

## QUIT GAME

QUIT GAME is added in two places, in the game's 14 text regions, each through replaced entries of
the game's asset archive built from the player's own dump: a third choice in the QUIT confirmation
of the pause menu (no hook: the game's own handler quits), and a sixth line of the title menu, under
TRAINING, which needs the `quit` hook, in `templates/hooks.txt` (the launcher's "Unlock the game's
debug menu" box; without it the line shows and does nothing). The game then
shuts itself down and the process exits with code 0; the save is unchanged. Verified by
deterministic replays in English, French and Japanese and by a real-time run without a window, then
used in a window, in French: the title menu line twice and the pause menu choice three times, in
two missions (not in the tutorial). `gyroship-tools mods` builds those entries from your own dump
(Building the mods, below).

## Another vehicle in a mission

An experiment made the same way, with replaced archive entries built from the player's own dump: a
mission can spawn another of the game's three vehicles than its own. Two entries per mission are
rebuilt: its root level, whose player spawner then names the other vehicle, and its package list,
which then loads the files that vehicle needs, inserted where the list's dependency order wants
them. The three levels of Andross and the tutorial keep the Arwing. Tried without a window on seven pairs of mission and vehicle, among them each vehicle in a
mission of each other kind: the vehicle is there, its code runs, the mission goes on and nothing is
journaled as an anomaly. Then, in a window: the first mission with the Blue-Marine for about 9
minutes, and with the Landmaster to its results screen, then Meteo with the Landmaster for 2
minutes. As for QUIT GAME, `gyroship-tools mods` builds those entries from your own dump.

## Building the mods

`gyroship-tools mods <exefs> --out <folder>` builds every mod from your own dump, one layer folder
each, in a few seconds; the launcher's Setup does it into `<data folder>\practice`, where Play's
Mod list finds them (step 6, Build the mods). Each layer is the practice layer plus its own changes:

| Folder | Mod |
|---|---|
| `layer` | QUIT GAME in the pause and title menus, and the mission timer |
| `layer-demo` | the eShop demo's lines in the main menu (play on a fresh copy of the save: METEO starts a new campaign) |
| `layer-mouse` | the tutorial's mouse question within a minute or two of flight |
| `layer-arwing`, `layer-landmaster`, `layer-bluemarine` | that vehicle in every mission whose own is another |

A layer holds the entries of the game's asset archive it replaces, rebuilt from your own: each edit
inserts or replaces a few lines of the engine's JSON text, keeps every other byte, and checks what
it relies on, so a dump of another build stops the build with a message instead of giving a wrong
layer. The values read in the game that the edits need are in `templates/mods.txt`, for the update
Gyroship targets. Nothing of the game is in the repository or in a release: each player makes their
own layers. On the targeted update, every file of the six layers is byte for byte the one the
scripts that first made them wrote.
