# Gyroship

**Star Fox (2026) on your PC with the practice tools and mods a speedrunner dreams of.**

Gyroship turns your own copy of Star Fox for Nintendo Switch 2 into a real Windows program, at 60
fps. Then it goes further than a port: a boost that lets go at the perfect moment,
your speed and release markers on screen, the level time to the millisecond, the game's own debug
menu with mission select and cheats, mods that swap your vehicle, and Switch 2 Joy-Con with their
mouse controls.

You bring the game: your own dump, from your own console. Gyroship holds no game code or assets
and decrypts nothing, the game never goes online, and what you build from your dump stays yours
alone. It is for practice: official runs are done on the console.

## What makes Gyroship different

| | |
|---|---|
| 🎯 **Speedrun practice** | Perfect boost release, infinite boost, speed and release markers on screen, the level time to the millisecond |
| 🧩 **Mods** | Another vehicle in a mission, QUIT GAME, the game's mission timer, the eShop demo's lines, a shortcut to the tutorial's mouse lesson |
| 🛠️ **The developers' debug menu** | Mission select, and the shield, wings, bombs, lives and laser cheats, with any mod |
| 🎮 **Switch 2 Joy-Con** | Straight over Bluetooth, no pairing in Windows: hold sync and play |
| 🖱️ **Mouse controls** | Each Joy-Con becomes the mouse the game expects, as on the console |
| 🔬 **Research you can check** | The boost of each vehicle, taken from the game's code and matched to the running game, bit for bit |

## Practice like no console lets you

Every practice tool is a switch in the launcher, and the files they need for the game's update come
with Gyroship: no reverse engineering.

- **A perfect boost, every time.** Hold Boost and each boost ends where the research says to let
  go, in the Arwing and the Blue-Marine. One button turns it off when you want a full boost.
- **An infinite boost**, to learn a route without watching the meter.
- **Your speed, the frame rate, the release markers**: pick any of them, and they show over the
  game, in its own font and language. The markers sit in the boost bar, right where to let go.
- **The level time in milliseconds**, frame by frame (the game counts sixtieths of a second), on
  the results screen and in the game's mission timer.
- **Mission select and cheats** from the game's own debug menu: jump straight to the part you
  practise, with the shield, wings, bombs and lives you want.
- **Record your inputs** and replay them.

Why let go at a third of the meter in the Arwing, at a fifth in the Blue-Marine, and never in the
Landmaster?
[research/boost/README.md](research/boost/README.md) explains each vehicle's boost, from the game's
own code, with the numbers to prove it.

## Mods

- **Swap your vehicle**: the Landmaster or the Blue-Marine in the Arwing's missions, the Arwing in
  theirs.
- **QUIT GAME**, in the title menu and in the pause menu.
- **The game's mission timer** on screen, to the millisecond.
- **The eShop demo's menu lines** in the full game.
- **The tutorial's mouse lesson in a minute or two**, to practise the mouse controls.

Pick a mod in the launcher and play; with the debug menu box ticked, the mods' QUIT GAME also works
from the title menu. Mods are built from your own copy of the game; their build scripts are on
their way to this repository.

## Switch 2 Joy-Con and mouse controls

Gyroship talks to Switch 2 Joy-Con itself, over Bluetooth: no pairing in Windows, no extra driver.
Turn your Switch 2 off, hold each one's sync button when the game starts, and play. Turn on mouse mode and each Joy-Con also
becomes a mouse, exactly as the game expects on the console: its mouse lesson, its mouse aiming,
its options. Xbox, PlayStation and other gamepads work too, and so does the keyboard.

## The whole game

- **A steady 60 fps** on both PCs it was tested on: a desktop with an Intel Core i9-9900K, a
  GeForce GTX 1080 and 16 GB of RAM, and a Shadow cloud PC with 4 cores of an AMD EPYC 7543P, an
  NVIDIA RTX A4500 and 28 GB of RAM.
- **The tutorial, the menus and every mission**, recompiled ahead of time into native code for
  your PC.
- **A launcher that guides you**: it prepares the game step by step, tells you what is missing and
  explains every option.
- **Always up to date**: the launcher tells you when a new version of Gyroship is out, and a copy
  from a release downloads it, installs it and builds your game again, by itself.
- **14 languages** for the game's texts, as on the console.
- **Offline**: the game talks to no server.

## What works today

| Part | |
|---|---|
| Native port | The game's code recompiled for your PC, its graphics on Vulkan |
| Frame rate | 60 fps, on both test PCs |
| Practice tools | Boost modes, on-screen speed, frame rate and markers, millisecond timer |
| Debug menu | Mission select and cheats, one box in the launcher |
| Mods | Chosen in the launcher; their build scripts are on their way |
| Controls | Keyboard, Xbox, PlayStation and other gamepads, Switch 2 Joy-Con |
| Mouse controls | Through the Joy-Con, as on the console |
| Sound and languages | The game's audio and its 14 text regions |
| Launcher | Setup step by step, then Play; a release's copy updates itself |

Details and known issues: [docs/status.md](docs/status.md).

## Get started

1. **What you need**: Windows 11 (or Windows 10 with Microsoft's WebView2 runtime), 64-bit, an
   NVIDIA graphics card, 16 GB of RAM, your own
   dump of the game with its update, and `SDL3.dll` for gamepads and sound. Building Gyroship and
   its launcher needs Rust (1.97 or later) and Microsoft's C++ build tools: install them first
   ([docs/building.md](docs/building.md)); the launcher then checks them, with the Visual C++
   runtime, and installs any that goes missing.
2. **Build** Gyroship and its launcher: [docs/building.md](docs/building.md).
3. **Open the launcher**: its Setup tab walks you through preparing the game, then press Play
   ([docs/playing.md](docs/playing.md)).
4. **Practise**: turn on the practice tools in the launcher ([docs/practice.md](docs/practice.md)).

## Documentation

- [Building](docs/building.md): what to install, and the three programs to build.
- [Playing](docs/playing.md): the launcher, controls, gamepads and Joy-Con, the options to play
  with.
- [Practice tools and mods](docs/practice.md): the boost modes, what shows on screen, the debug
  menu, QUIT GAME, another vehicle in a mission.
- [Boost research](research/boost/README.md): each vehicle's boost, and when to let go.
- [Status](docs/status.md) and [verification](docs/development.md): what was tested, known issues,
  replays and tools for developers.

## License

Gyroship is free software: you can share and change it under the terms of the GNU General Public
License, version 3 or any later version ([LICENSE](LICENSE)).
