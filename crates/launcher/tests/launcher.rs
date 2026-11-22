//! The launcher's decisions, on synthetic settings and files.

use gyroship_launcher::{LAUNCHER, Last, Settings, builds, checks, flags, layers, load, logs, outcome, prepare_save, report, save, session, step};
use std::fs;
use std::path::{Path, PathBuf};

/// An empty folder of this test's own.
fn scratch(name: &str) -> PathBuf {
    let dir = Path::new(env!("CARGO_TARGET_TMPDIR")).join(name);
    let _ = fs::remove_dir_all(&dir);
    fs::create_dir_all(&dir).unwrap();
    dir
}

fn words(line: &[String]) -> Vec<&str> {
    line.iter().map(String::as_str).collect()
}

/// `rel`'s slashes as the separator, under `root`: the paths the launcher itself joins.
fn at(root: &Path, rel: &str) -> PathBuf {
    rel.split('/').fold(root.to_path_buf(), |p, part| p.join(part))
}

fn text(p: &Path) -> String {
    p.to_string_lossy().into_owned()
}

/// Gyroship's sources, as far as the launcher can tell: the runtime's crate.
fn sources_of_gyroship(root: &Path) {
    fs::create_dir_all(at(root, "crates/runtime")).unwrap();
    fs::write(at(root, "crates/runtime/Cargo.toml"), "").unwrap();
}

/// Windows' own tar makes the archives, as tools/release.sh does: `what`, taken in `from`.
#[cfg(windows)]
fn zip(to: &Path, from: &Path, what: &[&str]) {
    let tar = at(Path::new(&std::env::var_os("SystemRoot").unwrap()), "System32/tar.exe");
    assert!(std::process::Command::new(tar).args(["-a", "-c", "-f"]).arg(to).arg("-C").arg(from).args(what).status().unwrap().success());
}

/// Settings come back as saved; a file that does not parse is kept aside, not lost.
#[test]
fn settings_come_back_and_a_broken_file_is_kept_aside() {
    let dir = scratch("settings");
    let file = dir.join("launcher.json");
    assert_eq!(load(&file), (Settings::default(), None), "no file yet: the defaults, and nothing to say");
    let chosen = Settings { build: "b/gyroship.exe".into(), language: "fr".into(), joycon: true, joycon_mouse: true, save_mode: "copy".into(), boost: "perfect".into(), ..Settings::default() };
    save(&file, &chosen).unwrap();
    assert_eq!(load(&file), (chosen, None));
    // A field this version does not know, and one it knows missing: still read.
    fs::write(&file, r#"{"language": "ja", "later": 1}"#).unwrap();
    assert_eq!(load(&file).0, Settings { language: "ja".into(), ..Settings::default() });
    // Hand-made and broken: the defaults, a word for the player, and the text kept.
    fs::write(&file, "{ \"language\": \"fr\", ").unwrap();
    let (settings, said) = load(&file);
    assert_eq!(settings, Settings::default());
    assert!(said.unwrap().contains("launcher.json.bad"));
    assert_eq!(fs::read_to_string(dir.join("launcher.json.bad")).unwrap(), "{ \"language\": \"fr\", ");
    // Saved in another encoding by some editor: kept aside all the same.
    fs::write(&file, b"\xff\xfe{\x00").unwrap();
    assert!(load(&file).1.unwrap().contains("launcher.json.bad"));
    assert_eq!(fs::read(dir.join("launcher.json.bad")).unwrap(), b"\xff\xfe{\x00");
    // Saved before the renaming: the generated harness.exe is now the generated gyroship.exe,
    // whose old file a new port leaves behind; any other harness.exe is left as chosen.
    let repo = dir.join("repo");
    let generated = |exe: &str| ["generated", "game", "target", "release", exe].iter().fold(repo.clone(), |p, part| p.join(part)).to_string_lossy().into_owned();
    let before = |build: String| Settings { repo: repo.to_string_lossy().into(), build, ..Settings::default() };
    save(&file, &before(generated("harness.exe"))).unwrap();
    assert_eq!(load(&file).0.build, generated("gyroship.exe"));
    let kept = dir.join("data/bin/a/harness.exe").to_string_lossy().into_owned();
    save(&file, &before(kept.clone())).unwrap();
    assert_eq!(load(&file).0.build, kept);
    // Saved with the one overlay switch: the three overlays, and the old switch never written.
    fs::write(&file, r#"{"hud": true}"#).unwrap();
    let three = Settings { hud_speed: true, hud_fps: true, hud_markers: true, ..Settings::default() };
    assert_eq!(load(&file).0, three);
    save(&file, &three).unwrap();
    assert!(!fs::read_to_string(&file).unwrap().contains("\"hud\""));
}

/// The game's command line: only what is set, and the practice options only with their file.
#[test]
fn the_command_line_holds_what_is_set() {
    let logs = Path::new("logs");
    let (log, record) = (logs.join("play-3.log").to_string_lossy().into_owned(), logs.join("play-3.txt").to_string_lossy().into_owned());
    let (log, record) = (log.as_str(), record.as_str());
    let bare = Settings { exefs: "game/exefs".into(), boost: "infinite".into(), hud_speed: true, hud_fps: true, hud_markers: true, ..Settings::default() };
    assert_eq!(words(&flags(&bare, logs, "play-3")), ["boot", "game/exefs", "--play", "--explore", "--log", log], "no practice file: no --boost, no overlay");
    let full = Settings {
        exefs: "game/exefs".into(),
        save: "save".into(),
        cache: "cache".into(),
        language: "fr".into(),
        sdl: "SDL3.dll".into(),
        hooks: "a.hooks".into(),
        layer: "layer".into(),
        practice: "practice.txt".into(),
        boost: "perfect".into(),
        hud_speed: true,
        hud_markers: true,
        fullscreen: true,
        joycon: true,
        joycon_mouse: true,
        record: true,
        vulkan: true,
        ..Settings::default()
    };
    let expected = ["boot", "game/exefs", "--play", "--explore", "--save", "save", "--pipeline-cache", "cache", "--language", "fr", "--sdl", "SDL3.dll", "--hooks", "a.hooks", "--romfs-layer", "layer", "--practice", "practice.txt", "--boost", "perfect", "--renderer", "vulkan", "--log", log, "--record", record, "--fullscreen", "--joycon", "--joycon-mouse", "--hud-speed", "--hud-markers"];
    assert_eq!(words(&flags(&full, logs, "play-3")), expected);
    // The debug menu box: the repository's hooks file, unless the player gave one of their own.
    let debug = Settings { exefs: "game/exefs".into(), repo: "repo".into(), debug_menu: true, ..Settings::default() };
    let template = Path::new("repo").join("templates").join("hooks.txt").to_string_lossy().into_owned();
    assert_eq!(words(&flags(&debug, logs, "play-3")), ["boot", "game/exefs", "--play", "--explore", "--hooks", template.as_str(), "--log", log]);
    let own = Settings { hooks: "mine.hooks".into(), ..debug.clone() };
    assert_eq!(words(&flags(&own, logs, "play-3"))[4..6], ["--hooks", "mine.hooks"]);
    assert!(!flags(&Settings { debug_menu: false, ..debug }, logs, "play-3").contains(&"--hooks".to_string()));
}

/// Sessions are numbered after the highest in the folder; the last one's end is reported
/// with the journal's summary and the game's last messages.
#[test]
fn a_session_gets_the_next_number_and_its_end_is_reported() {
    let dir = scratch("sessions");
    assert_eq!(session(&dir), "play-1");
    assert_eq!(report(&dir), None, "no session yet");
    for name in ["play-2.log", "play-9.err.txt", "play-10.log", "other-40.log", "play-x.log"] {
        fs::write(dir.join(name), "").unwrap();
    }
    assert_eq!(session(&dir), "play-11");
    fs::write(dir.join("play-10.log"), "  1 t0 boot\n  2 t1 summary: 1200 presents; anomalies: none\n  3 t1 probe a: 1 passes\n").unwrap();
    fs::write(dir.join("play-10.err.txt"), "gyroship: play mode\ngyroship: window closed\n").unwrap();
    fs::write(dir.join("last.json"), serde_json::to_string(&Last { session: "play-10".into(), code: 72, seconds: 95 }).unwrap()).unwrap();
    let r = report(&dir).unwrap();
    assert_eq!((r.session.as_str(), r.ended, r.code, r.seconds), ("play-10", true, 72, 95));
    assert!(r.outcome.contains("emulation stopped"), "{}", r.outcome);
    assert!(outcome(2).contains("command line"));
    assert_eq!(r.summary, "1200 presents; anomalies: none");
    assert_eq!(r.messages, "gyroship: play mode\ngyroship: window closed\n");
    assert!(outcome(0).contains("normally") && outcome(3).contains("GPU") && outcome(71).contains("fault") && outcome(1234).contains("unexpected"));
    // A launcher that stopped before its game left the start's record: no end to show.
    fs::write(dir.join("last.json"), serde_json::to_string(&Last { session: "play-10".into(), code: gyroship_launcher::RUNNING, seconds: 0 }).unwrap()).unwrap();
    let r = report(&dir).unwrap();
    assert!(!r.ended && r.outcome.contains("not recorded"), "{}", r.outcome);
    assert_eq!(logs(&Settings { data: "data".into(), ..Settings::default() }), Path::new("data").join("logs"));
}

/// Builds: the generated one first, then those kept under the data folder (named before
/// or after the renaming), then the chosen one; existing files only.
#[test]
fn builds_are_the_generated_one_those_kept_and_the_chosen_one() {
    let dir = scratch("builds");
    let (repo, data) = (dir.join("repo"), dir.join("data"));
    for folder in [repo.join("generated/game/target/release"), data.join("bin/b"), data.join("bin/a"), data.join("bin/empty"), dir.join("elsewhere")] {
        fs::create_dir_all(&folder).unwrap();
    }
    let s = Settings { repo: repo.to_string_lossy().into(), data: data.to_string_lossy().into(), ..Settings::default() };
    assert!(repo.is_absolute(), "the test's folder is a full path");
    assert_eq!(builds(&s), Vec::<String>::new());
    // Part by part, as the launcher joins it: the same text when chosen from the list.
    let generated = ["generated", "game", "target", "release", "gyroship.exe"].iter().fold(repo.clone(), |p, part| p.join(part));
    for exe in [&generated, &repo.join("generated/game/target/release/harness.exe"), &data.join("bin/b/harness.exe"), &data.join("bin/a/gyroship.exe"), &dir.join("elsewhere/gyroship.exe")] {
        fs::write(exe, "").unwrap();
    }
    let found = |s: &Settings| builds(s).iter().map(|b| b.strip_prefix(dir.to_str().unwrap()).unwrap().replace('\\', "/")).collect::<Vec<_>>();
    // The generated build is the renamed one: a harness.exe left beside it is a stale build.
    assert_eq!(found(&s), ["/repo/generated/game/target/release/gyroship.exe", "/data/bin/a/gyroship.exe", "/data/bin/b/harness.exe"]);
    // A chosen build elsewhere comes last; one already found, or one that is gone, adds nothing.
    let chosen = |b: &Path| Settings { build: b.to_string_lossy().into(), ..s.clone() };
    assert_eq!(found(&chosen(&dir.join("elsewhere/gyroship.exe"))).last().unwrap(), "/elsewhere/gyroship.exe");
    assert_eq!(found(&chosen(&generated)).len(), 3);
    assert_eq!(found(&chosen(&dir.join("elsewhere/gone.exe"))).len(), 3);
}

/// Folders not set, or not full paths: nothing is looked for under the current folder, even
/// where it holds builds and layers. The one test that moves the process's current folder.
#[test]
fn relative_folders_are_never_looked_in() {
    let dir = scratch("relative");
    for folder in ["repo/generated/game/target/release", "data/bin/a", "data/practice/layer"] {
        fs::create_dir_all(dir.join(folder)).unwrap();
    }
    for exe in ["repo/generated/game/target/release/gyroship.exe", "data/bin/a/gyroship.exe"] {
        fs::write(dir.join(exe), "").unwrap();
    }
    std::env::set_current_dir(&dir).unwrap();
    let s = Settings { repo: "repo".into(), data: "data".into(), build: "data/bin/a/gyroship.exe".into(), ..Settings::default() };
    assert_eq!(builds(&s), Vec::<String>::new());
    assert_eq!(layers(&s), Vec::new());
}

/// Layers: the data folder's practice folders named `layer…`, each shown by the first line of
/// its about.txt, or by its name, in the order of what they show.
#[test]
fn layers_are_the_practice_folders_and_their_about_lines() {
    let dir = scratch("layers");
    let practice = dir.join("data").join("practice");
    for folder in ["layer", "layer-arwing", "layer-demo", "layer-landmaster", "vehicle"] {
        fs::create_dir_all(practice.join(folder)).unwrap();
    }
    fs::write(practice.join("layer-arwing/about.txt"), "  Force the Arwing \nsecond line\n").unwrap();
    fs::write(practice.join("layer-demo/about.txt"), "Add the demo's lines\n").unwrap();
    fs::write(practice.join("layer-landmaster/about.txt"), "Force the Landmaster\n").unwrap();
    fs::write(practice.join("layer-file"), "").unwrap();
    let s = Settings { data: dir.join("data").to_string_lossy().into(), ..Settings::default() };
    let shown: Vec<(String, String)> = layers(&s);
    let path = |name: &str| practice.join(name).to_string_lossy().into_owned();
    // By label, whatever the folders' names: the two "Force" together.
    let expected = [("layer-demo", "Add the demo's lines"), ("layer-arwing", "Force the Arwing"), ("layer-landmaster", "Force the Landmaster"), ("layer", "layer")];
    assert_eq!(shown, expected.map(|(f, l)| (path(f), l.to_string())));
}

/// The checks the page shows: what the Setup tab lacks, the dump's problem, the port's steps.
#[test]
fn checks_say_what_the_setup_and_the_dump_lack() {
    let dir = scratch("checks");
    let none = checks(&Settings::default());
    assert_eq!(none.setup, ["Repository: not set", "Data folder: not set", "No game build yet: recompile the game, then build it"]);
    assert_eq!(none.exefs.as_deref(), Some("Your game's exefs folder: not set"));
    assert!(!none.folders && !none.tools && !none.recompiled && !none.built);
    let (repo, data, dump) = (dir.join("repo"), dir.join("data"), dir.join("dump"));
    let text = |p: &Path| p.to_string_lossy().into_owned();
    let s = Settings { repo: text(&repo), data: text(&data), exefs: text(&dump.join("exefs")), ..Settings::default() };
    let c = checks(&s);
    // A data folder not there yet is no problem: the launcher makes it (`make_folders`).
    assert_eq!(c.setup, ["Repository: folder not found", "No game build yet: recompile the game, then build it"]);
    assert_eq!(checks(&Settings { data: "data".into(), ..s.clone() }).setup[1], "Data folder: not a full path");
    assert_eq!(c.exefs.as_deref(), Some("Your game's exefs folder: folder not found"));
    for folder in ["target/release", "generated/game/target/release"] {
        fs::create_dir_all(repo.join(folder)).unwrap();
    }
    fs::create_dir_all(&data).unwrap();
    fs::create_dir_all(dump.join("exefs")).unwrap();
    let c = checks(&s);
    assert_eq!(c.setup, ["No game build yet: recompile the game, then build it"]);
    assert_eq!(c.exefs.as_deref(), Some("Your game's exefs folder: no romfs folder next to it"));
    assert!(c.folders && !c.tools && !c.recompiled && !c.built);
    fs::create_dir_all(dump.join("romfs")).unwrap();
    for file in ["target/release/gyroship-tools.exe", "generated/game/Cargo.toml", "generated/game/target/release/gyroship.exe"] {
        fs::write(repo.join(file), "").unwrap();
    }
    // A folder with no game module in it, a romfs folder for one: no exefs folder.
    let c = checks(&s);
    assert_eq!((c.exefs.as_deref(), c.dump.as_deref()), (Some("Your game's exefs folder: no main module in it (is it the exefs folder?)"), None));
    // The dump's build, from its main module's header: the one the repository's templates name
    // (a synthetic one here), or another one.
    let id = "0123456789abcdef0123456789abcdef01234567";
    fs::create_dir_all(repo.join("templates")).unwrap();
    fs::write(repo.join("templates/practice.txt"), format!("# A practice file for the build\n# {id} (a test's)\n")).unwrap();
    assert_eq!(gyroship_launcher::templates_build(&text(&repo)).as_deref(), Some(id));
    let header = |id: &[u8]| [b"NSO0".as_slice(), &[0; 0x3c], id, &[0; 0x2c]].concat();
    let ours: Vec<u8> = (0..20).map(|i| u8::from_str_radix(&id[2 * i..2 * i + 2], 16).unwrap()).collect();
    fs::write(dump.join("exefs/main"), header(&ours)).unwrap();
    let everything = gyroship_launcher::Checks { setup: Vec::new(), folders: true, exefs: None, dump: Some(id.into()), dump_known: Some(true), tools: true, recompiled: true, built: true, shaders: false };
    assert_eq!(checks(&s), everything);
    // The shaders: a precompiler's signature (any shader build's folder) newer than the game.
    let cache = dir.join("cache");
    let cached = Settings { cache: text(&cache), ..s.clone() };
    assert!(!checks(&cached).shaders, "no signature yet");
    fs::create_dir_all(cache.join("vulkan/0123456789abcdef")).unwrap();
    fs::write(cache.join("vulkan/0123456789abcdef/precompiled.txt"), "a signature\n").unwrap();
    assert!(checks(&cached).shaders);
    let rebuilt = std::time::SystemTime::now() + std::time::Duration::from_secs(30);
    fs::File::options().write(true).open(repo.join("generated/game/target/release/gyroship.exe")).unwrap().set_modified(rebuilt).unwrap();
    assert!(!checks(&cached).shaders, "a game built since");
    // A recompile after the build: Build is to do again, Recompile stays done.
    let later = std::time::SystemTime::now() + std::time::Duration::from_secs(60);
    fs::File::options().write(true).open(repo.join("generated/game/Cargo.toml")).unwrap().set_modified(later).unwrap();
    let c = checks(&s);
    assert!(c.recompiled && !c.built, "{c:?}");
    // Tools newer than the recompile: Recompile is to do again.
    fs::File::options().write(true).open(repo.join("target/release/gyroship-tools.exe")).unwrap().set_modified(later + std::time::Duration::from_secs(60)).unwrap();
    assert!(!checks(&s).recompiled);
    fs::write(dump.join("exefs/main"), header(&[0xab; 20])).unwrap();
    let c = checks(&s);
    assert_eq!((c.exefs, c.dump.as_deref(), c.dump_known), (None, Some("abababababababababababababababababababab"), Some(false)));
    // The repository not set yet: the templates' build is unknown, so the dump is neither.
    assert_eq!(checks(&Settings { repo: String::new(), ..s.clone() }).dump_known, None);
    fs::write(dump.join("exefs/main"), b"not a module").unwrap();
    assert_eq!(checks(&s).dump, None);
    assert!(checks(&s).exefs.unwrap().contains("no main module"));
    assert_eq!(checks(&Settings { repo: "repo".into(), ..s }).setup[0], "Repository: not a full path");
}

/// The folders the player set and that are not there yet: made, the data folder first; a
/// relative one refused, before anything is made.
#[test]
fn the_folders_not_there_yet_are_made() {
    let dir = scratch("folders");
    let (data, save, cache) = (at(&dir, "new/data"), at(&dir, "new/data/save"), at(&dir, "elsewhere/cache"));
    let s = Settings { data: text(&data), save: text(&save), cache: text(&cache), ..Settings::default() };
    gyroship_launcher::make_folders(&s).unwrap();
    assert!(data.is_dir() && save.is_dir() && cache.is_dir());
    gyroship_launcher::make_folders(&s).unwrap();
    gyroship_launcher::make_folders(&Settings::default()).unwrap();
    let relative = Settings { data: text(&dir.join("other")), save: "save".into(), ..Settings::default() };
    assert!(gyroship_launcher::make_folders(&relative).unwrap_err().contains("the save folder is not a full path"));
    assert!(!dir.join("other").exists(), "nothing made");
}

/// The tools: the repository's own build, else a release's at its top, else those downloaded next
/// to the launcher, else where the tools step puts them.
#[test]
fn the_tools_are_found_in_the_repository_then_next_to_the_launcher() {
    let dir = scratch("tools");
    let (repo, launcher) = (dir.join("repo"), dir.join("launcher"));
    fs::create_dir_all(repo.join("target/release")).unwrap();
    fs::create_dir_all(&launcher).unwrap();
    let repo_text = repo.to_string_lossy().into_owned();
    let tools = || gyroship_launcher::tools_in(&repo_text, Some(&launcher));
    let built = PathBuf::from_iter([repo_text.as_str(), "target", "release", "gyroship-tools.exe"]);
    assert_eq!(tools(), built, "none yet: where the tools step puts them");
    fs::write(launcher.join("gyroship-tools.exe"), "").unwrap();
    assert_eq!(tools(), launcher.join("gyroship-tools.exe"));
    fs::write(repo.join("gyroship-tools.exe"), "").unwrap();
    assert_eq!(tools(), repo.join("gyroship-tools.exe"));
    fs::write(&built, "").unwrap();
    assert_eq!(tools(), built);
    assert_eq!(gyroship_launcher::tools_in(&repo_text, None), built);
}

/// The save the game gets: the folder itself, a fresh copy of it, or a fresh empty one; the
/// chosen folder is never written by the last two, nor emptied when it overlaps theirs.
#[test]
fn the_save_is_the_folder_a_fresh_copy_or_a_fresh_empty_one() {
    let dir = scratch("saves");
    let (data, save) = (dir.join("data"), dir.join("save"));
    fs::create_dir_all(save.join("slot")).unwrap();
    fs::write(save.join("slot/game.save"), "progress").unwrap();
    let s = Settings { data: data.to_string_lossy().into(), save: save.to_string_lossy().into(), ..Settings::default() };
    assert_eq!(prepare_save(&s).unwrap(), s.save, "the folder itself");
    let copy = Settings { save_mode: "copy".into(), ..s.clone() };
    let copied = PathBuf::from(prepare_save(&copy).unwrap());
    assert_eq!(copied, data.join("save-copy"));
    assert_eq!(fs::read_to_string(copied.join("slot/game.save")).unwrap(), "progress");
    // What the game wrote in the copy goes at the next start; the folder itself never moved.
    fs::write(copied.join("slot/game.save"), "new campaign").unwrap();
    fs::write(copied.join("extra"), "").unwrap();
    prepare_save(&copy).unwrap();
    assert_eq!(fs::read_to_string(copied.join("slot/game.save")).unwrap(), "progress");
    assert!(!copied.join("extra").exists());
    let fresh = Settings { save_mode: "new".into(), ..s.clone() };
    let new = PathBuf::from(prepare_save(&fresh).unwrap());
    assert_eq!((new.clone(), fs::read_dir(&new).unwrap().count()), (data.join("save-new"), 0));
    // What the game saved there goes at the next start too.
    fs::create_dir_all(new.join("slot")).unwrap();
    fs::write(new.join("slot/game.save"), "tutorial").unwrap();
    prepare_save(&fresh).unwrap();
    assert_eq!(fs::read_dir(&new).unwrap().count(), 0);
    assert_eq!(fs::read_to_string(save.join("slot/game.save")).unwrap(), "progress");
    // A save folder that is, holds or lies in the folder to empty: refused, and left alone.
    for chosen in [data.join("save-copy"), data.clone(), data.join("SAVE-NEW/slot")] {
        let mode = if chosen.ends_with("slot") { "new" } else { "copy" };
        let said = prepare_save(&Settings { save: chosen.to_string_lossy().into(), save_mode: mode.into(), ..s.clone() }).unwrap_err();
        assert!(said.contains("choose another save folder"), "{said}");
    }
    // The same folder written another way (the verbatim form here; a mapped drive or a junction
    // likewise): refused all the same.
    #[cfg(windows)]
    {
        let verbatim = format!(r"\\?\{}", data.join("save-copy").display());
        let said = prepare_save(&Settings { save: verbatim, ..copy.clone() }).unwrap_err();
        assert!(said.contains("choose another save folder"), "{said}");
    }
    assert!(copied.join("slot/game.save").exists());
    // Nothing to copy, or no such mode.
    assert!(prepare_save(&Settings { save: String::new(), ..copy.clone() }).unwrap_err().contains("not set"));
    assert!(prepare_save(&Settings { save: dir.join("none").to_string_lossy().into(), ..copy.clone() }).unwrap_err().contains("does not exist"));
    assert!(prepare_save(&Settings { save_mode: "move".into(), ..s.clone() }).unwrap_err().contains("no save mode"));
}

/// The setup steps' command lines; the build fetches nothing; a path missing, or not a full
/// one, is said.
#[test]
fn setup_steps_are_the_tools_command_lines() {
    // Full paths, as the steps want them; nothing here touches the file system.
    let root = Path::new(env!("CARGO_TARGET_TMPDIR"));
    let (repo, exefs) = (root.join("repo").to_string_lossy().into_owned(), root.join("exefs").to_string_lossy().into_owned());
    let s = Settings { repo: repo.clone(), exefs: exefs.clone(), probes: "probes.txt".into(), ..Settings::default() };
    let tools = Path::new(&repo).join("target/release/gyroship-tools.exe");
    // One kind of separator in what the player reads: the folder is also an argument, as text.
    let out = [repo.as_str(), "generated", "game"].join(std::path::MAIN_SEPARATOR_STR);
    let generated = PathBuf::from(&out);
    let (program, line, folder) = step(&s, "recompile").unwrap();
    assert_eq!((program.as_path(), words(&line), folder.as_path()), (tools.as_path(), vec!["recompile", exefs.as_str(), "--out", out.as_str(), "--probes", "probes.txt"], Path::new(&repo)));
    assert_eq!(words(&step(&Settings { probes: String::new(), ..s.clone() }, "recompile").unwrap().1), ["recompile", exefs.as_str(), "--out", out.as_str()]);
    // The builds: Cargo, found where rustup puts it, without --offline (the first ones download
    // the crates they need, on the player's click).
    let (program, line, folder) = step(&s, "build").unwrap();
    assert_eq!((program, words(&line), folder), (gyroship_launcher::cargo_exe(), vec!["build", "--release", "-j", "4"], generated));
    let (program, line, folder) = step(&s, "tools").unwrap();
    assert_eq!((program, words(&line), folder), (gyroship_launcher::cargo_exe(), vec!["build", "--release", "-j", "4"], PathBuf::from(&repo)));
    // The shaders: the game built (else the one chosen) precompiles them into the cache folder.
    let cache = root.join("cache").to_string_lossy().into_owned();
    let game = PathBuf::from_iter([repo.as_str(), "generated", "game", "target", "release", "gyroship.exe"]);
    let (program, line, folder) = step(&Settings { cache: cache.clone(), ..s.clone() }, "shaders").unwrap();
    assert_eq!((program, words(&line), folder), (game, vec!["precompile", exefs.as_str(), "--pipeline-cache", cache.as_str()], PathBuf::from(&repo)));
    let chosen = step(&Settings { cache: cache.clone(), build: "C:\\kept\\gyroship.exe".into(), ..s.clone() }, "shaders").unwrap().0;
    assert_eq!(chosen, PathBuf::from("C:\\kept\\gyroship.exe"));
    assert!(step(&s, "shaders").unwrap_err().contains("the shader cache folder is not set"));
    // The installs, through winget, need no folder of the player's.
    let (program, line, _) = step(&Settings::default(), "install-rust").unwrap();
    let common = ["--exact", "--source", "winget", "--accept-package-agreements", "--accept-source-agreements", "--disable-interactivity"];
    assert_eq!((program, words(&line)), (PathBuf::from("winget"), [["install", "--id", "Rustlang.Rustup"].as_slice(), &common].concat()));
    let line = step(&Settings::default(), "install-cpp").unwrap().1;
    assert_eq!(words(&line)[2..4], ["Microsoft.VisualStudio.BuildTools", "--exact"]);
    // The installer's own arguments go as one word, --wait among them (--override replaces the
    // package's), never restarting the PC by itself, with the compiler and the SDK named.
    let over = line.last().unwrap();
    assert!(["--wait", "--norestart", "Microsoft.VisualStudio.Workload.VCTools", "Microsoft.VisualStudio.Component.VC.Tools.x86.x64", "Windows11SDK"].iter().all(|w| over.contains(w)), "{over}");
    assert_eq!(step(&Settings::default(), "install-runtime").unwrap().1[2], "Microsoft.VCRedist.2015+.x64");
    assert!(step(&Settings { exefs: String::new(), ..s.clone() }, "recompile").unwrap_err().contains("the exefs folder is not set"));
    assert!(step(&Settings::default(), "build").unwrap_err().contains("the repository is not set"));
    // The tools run in the repository, where a relative path would mean another place: refused.
    assert!(step(&Settings { repo: "repo".into(), ..s.clone() }, "build").unwrap_err().contains("the repository is not a full path: repo"));
    assert!(step(&Settings { exefs: "game/exefs".into(), ..s.clone() }, "recompile").unwrap_err().contains("the exefs folder is not a full path"));
    assert!(step(&s, "format-the-disk").unwrap_err().contains("no setup step"));
}

/// Updates (D109, D112): GitHub's answer for the latest release, read; only a newer one, with the
/// repository's own page and both its archives, is taken, with the SHA-256 GitHub gives for each.
#[test]
fn a_newer_release_is_read_from_githubs_answer() {
    use gyroship_launcher::{Asset, REPOSITORY, is_newer, release_from};
    assert!(is_newer("v0.2.0", "0.1.0") && is_newer("0.10.0", "0.9.9") && is_newer("v1.0.0-rc.1", "0.9.0"));
    assert!(!is_newer("v0.1.0", "0.1.0") && !is_newer("v0.1.0-beta", "0.1.0") && !is_newer("v1.0", "0.1.0") && !is_newer("latest", "0.1.0"));
    let answer = |tag: &str, page: &str, assets: &str| format!(r#"{{"tag_name": "{tag}", "html_url": "{page}", "draft": false, "assets": [{assets}]}}"#);
    let asset = |name: &str, url: &str, digest: &str| format!(r#"{{"name": "{name}", "browser_download_url": "{url}", "size": 1234, "digest": "{digest}"}}"#);
    let page = format!("{REPOSITORY}/releases/tag/v0.2.0");
    let url = |name: &str| format!("{REPOSITORY}/releases/download/v0.2.0/{name}");
    let (programs, source, sum) = ("gyroship-0.2.0-windows-x64.zip", "gyroship-0.2.0-source.zip", "AB".repeat(32));
    let assets = [asset("SHA256SUMS.txt", &url("SHA256SUMS.txt"), ""), asset(programs, &url(programs), &format!("sha256:{sum}")), asset(source, &url(source), "sha256:not hex")].join(",");
    let r = release_from(&answer("v0.2.0", &page, &assets), "0.1.0").unwrap().unwrap();
    assert_eq!((r.tag.as_str(), r.page.as_str()), ("v0.2.0", page.as_str()));
    assert_eq!(r.programs, Asset { url: url(programs), size: 1234, sha256: Some(sum.to_lowercase()) });
    assert_eq!(r.source, Asset { url: url(source), size: 1234, sha256: None }, "a digest that is no SHA-256: none");
    assert_eq!(release_from(&answer("v0.1.0", &page, &assets), "0.1.0").unwrap(), None, "not newer: nothing");
    // Without its sources' archive: refused, the page named.
    let alone = asset(programs, &url(programs), "");
    assert!(release_from(&answer("v0.2.0", &page, &alone), "0.1.0").unwrap_err().contains("no gyroship-0.2.0-source.zip: get it from"));
    // An archive served from elsewhere, another repository's page, a tag that names no folder:
    // refused.
    let with_source = |programs: String| [programs, asset(source, &url(source), "")].join(",");
    let foreign = with_source(asset(programs, "https://example.com/gyroship-0.2.0-windows-x64.zip", ""));
    assert!(release_from(&answer("v0.2.0", &page, &foreign), "0.1.0").unwrap_err().contains("no gyroship-0.2.0-windows-x64.zip"));
    // `..` segments, which curl and browsers resolve: another repository's archive or page.
    let away = with_source(asset(programs, &format!("{REPOSITORY}/releases/download/../../../../someone/else/releases/download/v0.2.0/{programs}"), ""));
    assert!(release_from(&answer("v0.2.0", &page, &away), "0.1.0").is_err());
    assert!(!gyroship_launcher::is_release_page(&format!("{REPOSITORY}/releases/../../someone/else")));
    let other = with_source(asset("gyroship-0.1.9-windows-x64.zip", &url("gyroship-0.1.9-windows-x64.zip"), ""));
    assert!(release_from(&answer("v0.2.0", &page, &other), "0.1.0").is_err(), "another version's archive");
    assert!(release_from(&answer("v0.2.0", "https://github.com/someone/else/releases/tag/v0.2.0", &assets), "0.1.0").is_err());
    assert!(release_from(&answer("v0.2.0+../../x", &page, &assets), "0.1.0").is_err());
    assert!(release_from("{}", "0.1.0").unwrap_err().contains("GitHub's answer"));
}

/// SHA-256 as sha256sum and GitHub write it (FIPS 180-2's first example).
#[test]
fn a_sha256_is_written_as_sha256sum_writes_it() {
    let file = scratch("sha256").join("abc");
    fs::write(&file, "abc").unwrap();
    assert_eq!(gyroship_launcher::sha256_of(&file).unwrap(), "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad");
}

/// SDL3 for the game: the one chosen, else the one a release puts next to the launcher, else none
/// named (the game looks next to itself).
#[test]
fn sdl3_is_the_chosen_one_else_the_one_next_to_the_launcher() {
    use gyroship_launcher::sdl_for;
    let dir = scratch("sdl");
    let chosen = Settings { sdl: r"C:\mine\SDL3.dll".into(), ..Settings::default() };
    assert_eq!(sdl_for(&Settings::default(), &dir), "");
    fs::write(dir.join("SDL3.dll"), "").unwrap();
    assert_eq!(sdl_for(&Settings::default(), &dir), text(&dir.join("SDL3.dll")));
    assert_eq!(sdl_for(&chosen, &dir), r"C:\mine\SDL3.dll");
}

/// Who installs a release itself (D112): a launcher in a folder of its own, playing from a
/// release's sources. One inside the sources, or with no repository set, or a Git clone's, is
/// only told.
#[test]
fn only_a_launcher_apart_from_a_releases_sources_updates_itself() {
    use gyroship_launcher::{install_root, update_blocker};
    let dir = scratch("blockers");
    let (programs, repo, clone) = (dir.join("programs"), dir.join("gyroship-0.1.0"), dir.join("clone"));
    sources_of_gyroship(&repo);
    sources_of_gyroship(&clone);
    fs::create_dir_all(clone.join(".git")).unwrap();
    // A launcher apart from any sources: a path only, outside the repository that holds the tests'
    // own folders.
    let launcher = std::env::temp_dir().join("gyroship-nowhere").join(LAUNCHER);
    let s = Settings { repo: text(&repo), ..Settings::default() };
    assert_eq!((install_root(&launcher), update_blocker(&launcher, &s)), (None, None));
    assert!(update_blocker(&launcher, &Settings::default()).unwrap().contains("not set"));
    assert!(update_blocker(&launcher, &Settings { repo: text(&programs), ..s.clone() }).unwrap().contains("holds no Gyroship sources"));
    assert!(update_blocker(&launcher, &Settings { repo: text(&clone), ..s.clone() }).unwrap().contains("git pull"));
    // A launcher built in a clone, or put inside a release's sources.
    let deep = at(&clone, "crates/launcher/target/release").join(LAUNCHER);
    assert_eq!(install_root(&deep), Some(clone.clone()));
    assert!(update_blocker(&deep, &s).unwrap().contains("built in a Git clone"));
    assert!(update_blocker(&repo.join(LAUNCHER), &s).unwrap().contains("inside Gyroship's sources"));
}

/// An update's sources (D112): the release's source archive unpacked next to the repository and
/// renamed once whole; the settings that pointed into the repository follow it, the player's own
/// stay. A stopped update's folder is removed, a folder of the same name that is not one is not.
#[cfg(windows)]
#[test]
fn a_releases_sources_install_next_to_the_repository() {
    use gyroship_launcher::{install_sources, rebase, target};
    let dir = scratch("update-sources");
    let (repo, new, part) = (dir.join("gyroship"), dir.join("gyroship-0.2.0"), dir.join("gyroship-0.2.0.part"));
    sources_of_gyroship(&repo);
    let staged = at(&dir, "staged/gyroship-0.2.0");
    sources_of_gyroship(&staged);
    fs::create_dir_all(staged.join("templates")).unwrap();
    fs::write(at(&staged, "templates/practice.txt"), "new").unwrap();
    // As git archive makes it, with its top folder; and without one.
    let (archive, flat) = (dir.join("gyroship-0.2.0-source.zip"), dir.join("flat.zip"));
    zip(&archive, &dir.join("staged"), &["gyroship-0.2.0"]);
    zip(&flat, &staged, &["crates", "templates"]);
    assert_eq!(install_sources(&archive, "v0.2.0", &repo).unwrap(), new);
    assert!(at(&new, "templates/practice.txt").is_file() && !part.exists() && repo.exists());
    // Installed: that folder is taken, and sources of that version want its programs.
    assert!(target(&repo, "v0.2.0").unwrap_err().contains("already exists"));
    assert!(target(&new, "v0.2.0").unwrap_err().contains("get the programs of v0.2.0"));
    // The settings follow: the repository, the game built there, a template it has; a template of
    // the player's own, which the new sources lack, stays (and so keeps the previous sources from
    // being deleted), as do the player's folders.
    let s = Settings {
        repo: text(&repo),
        build: text(&at(&repo, "generated/game/target/release/gyroship.exe")),
        practice: text(&at(&repo, "templates/practice.txt")),
        probes: text(&at(&repo, "templates/my-probes.txt")),
        data: text(&dir.join("data")),
        save: text(&at(&dir, "data/save")),
        ..Settings::default()
    };
    let moved = Settings { repo: text(&new), build: text(&at(&new, "generated/game/target/release/gyroship.exe")), practice: text(&at(&new, "templates/practice.txt")), ..s.clone() };
    assert_eq!(rebase(&s, &repo, &new, &new), moved);
    assert!(install_sources(&archive, "v0.2.0", &repo).unwrap_err().contains("already exists"));
    // A folder named as an update's but without its mark: not the launcher's to remove.
    fs::remove_dir_all(&new).unwrap();
    fs::create_dir_all(&part).unwrap();
    assert!(install_sources(&flat, "v0.2.0", &repo).is_err());
    assert!(part.exists());
    // With the mark, a stopped update's: removed, and the archive without a top folder installs.
    fs::write(part.join(".gyroship-update"), "").unwrap();
    assert_eq!(install_sources(&flat, "v0.2.0", &repo).unwrap(), new);
    assert!(at(&new, "crates/runtime/Cargo.toml").is_file() && !new.join(".gyroship-update").exists() && !part.exists());
    // An archive with no sources in it: refused, and nothing left behind.
    fs::remove_dir_all(&new).unwrap();
    let bad = dir.join("bad.zip");
    zip(&bad, &staged, &["templates"]);
    assert!(install_sources(&bad, "v0.2.0", &repo).unwrap_err().contains("no Gyroship sources"));
    assert!(!new.exists() && !part.exists());
    assert!(install_sources(&archive, "v0.2.0/../x", &repo).unwrap_err().contains("no version"));
}

/// An update's programs (D112): the release's archive puts the launcher, the tools, SDL3.dll and
/// the licences in place of those next to the launcher, each kept as `.old` until the next
/// launcher removes it, even the launcher that runs; the launcher's other files stay. An archive
/// without the tools changes nothing.
#[cfg(windows)]
#[test]
fn a_releases_programs_replace_the_launchers_own() {
    use gyroship_launcher::{RELEASED, install_programs, remove_replaced};
    let dir = scratch("update-programs");
    let (folder, staged) = (dir.join("programs"), dir.join("staged"));
    fs::create_dir_all(&folder).unwrap();
    fs::create_dir_all(&staged).unwrap();
    for n in RELEASED {
        fs::write(staged.join(n), format!("new {n}")).unwrap();
    }
    let (archive, toolless) = (dir.join("gyroship-0.2.0-windows-x64.zip"), dir.join("toolless.zip"));
    zip(&archive, &staged, &RELEASED);
    zip(&toolless, &staged, &[LAUNCHER, "SDL3.dll"]);
    fs::write(folder.join("launcher.json"), "{}").unwrap();
    fs::write(folder.join("gyroship-tools.exe"), "old tools").unwrap();
    // The launcher that runs: a copy of Windows' ping, waiting under the launcher's name.
    let ping = at(Path::new(&std::env::var_os("SystemRoot").unwrap()), "System32/PING.EXE");
    fs::copy(ping, folder.join(LAUNCHER)).unwrap();
    let mut running = std::process::Command::new(folder.join(LAUNCHER)).args(["-n", "60", "127.0.0.1"]).stdout(std::process::Stdio::null()).spawn().unwrap();
    // Without the tools: refused, nothing changed.
    assert!(install_programs(&toolless, &folder).unwrap_err().contains("no gyroship-tools.exe"));
    assert_eq!(fs::read_to_string(folder.join("gyroship-tools.exe")).unwrap(), "old tools");
    assert!(!folder.join(".gyroship-programs").exists() && !folder.join("SDL3.dll").exists());
    install_programs(&archive, &folder).unwrap();
    for n in RELEASED {
        assert_eq!(fs::read_to_string(folder.join(n)).unwrap(), format!("new {n}"), "{n}");
    }
    assert_eq!(fs::read_to_string(folder.join("gyroship-tools.exe.old")).unwrap(), "old tools");
    assert!(folder.join("gyroship-launcher.exe.old").is_file() && !folder.join("SDL3.dll.old").exists() && folder.join("launcher.json").is_file());
    assert!(running.try_wait().unwrap().is_none(), "the launcher replaced still runs");
    // Its file stays while it runs; the others go.
    remove_replaced(&folder);
    assert!(folder.join("gyroship-launcher.exe.old").is_file() && !folder.join("gyroship-tools.exe.old").exists());
    let _ = running.kill();
    running.wait().unwrap();
    for _ in 0..50 {
        remove_replaced(&folder);
        if !folder.join("gyroship-launcher.exe.old").exists() {
            break;
        }
        std::thread::sleep(std::time::Duration::from_millis(100));
    }
    assert!(!folder.join("gyroship-launcher.exe.old").exists());
}

/// The previous sources go on the player's click (D112): only a release's, next to the
/// repository, once the game is built there, and when they hold nothing the player chose.
#[cfg(windows)]
#[test]
fn the_previous_sources_go_only_when_nothing_of_the_players_is_in_them() {
    use gyroship_launcher::delete_previous;
    let dir = scratch("previous");
    let (old, new, away) = (dir.join("gyroship-0.1.0"), dir.join("gyroship-0.2.0"), at(&dir, "away/gyroship-0.0.9"));
    for root in [&old, &new, &away] {
        sources_of_gyroship(root);
    }
    let s = Settings { repo: text(&new), save: text(&old.join("save")), ..Settings::default() };
    assert!(delete_previous(&s, &old, &new).unwrap_err().contains("holds your save folder"));
    // The same save folder written otherwise: relative (any folder, for all the launcher knows),
    // verbatim, through `..`.
    fs::create_dir_all(old.join("save")).unwrap();
    for save in ["save".to_string(), format!(r"\\?\{}", old.join("save").display()), text(&at(&dir, "x/../gyroship-0.1.0/save"))] {
        assert!(delete_previous(&Settings { save: save.clone(), ..s.clone() }, &old, &new).unwrap_err().contains("holds your save folder"), "{save}");
    }
    let s = Settings { save: String::new(), ..s };
    for (not, why) in [(&new, "the repository"), (&dir, "no sources"), (&away, "not next to the repository")] {
        assert!(delete_previous(&s, not, &new).unwrap_err().contains("no previous release's sources"), "{why}");
    }
    let mut this = text(&new);
    this.push(std::path::MAIN_SEPARATOR);
    assert!(delete_previous(&s, Path::new(&this), &new).unwrap_err().contains("no previous release's sources"), "the repository, written with a final separator");
    fs::create_dir_all(old.join(".git")).unwrap();
    assert!(delete_previous(&s, &old, &new).is_err(), "a Git clone");
    fs::remove_dir_all(old.join(".git")).unwrap();
    // Until the repository's game is built, the previous sources keep the only one.
    assert!(delete_previous(&s, &old, &new).unwrap_err().contains("not built"));
    fs::create_dir_all(at(&new, "generated/game/target/release")).unwrap();
    fs::write(at(&new, "generated/game/target/release/gyroship.exe"), "").unwrap();
    delete_previous(&s, &old, &new).unwrap();
    assert!(!old.exists() && new.exists() && away.exists());
}

/// Rust's version, as `cargo --version` prints it.
#[test]
fn versions_are_read_from_cargo() {
    use gyroship_launcher::version;
    assert_eq!(version("cargo 1.99.1 (58cf8f9c3 2026-10-13)"), Some((1, 99, 1)));
    assert_eq!(version("cargo 1.100.0-nightly (abc 2026-12-01)"), Some((1, 100, 0)));
    assert_eq!(version("cargo"), None);
    assert_eq!(version("error: no default toolchain"), None);
    assert!((1, 96) < gyroship_launcher::RUST_OLDEST && (1, 99) >= gyroship_launcher::RUST_OLDEST);
}
