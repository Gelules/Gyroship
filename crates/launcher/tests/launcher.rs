//! The launcher's decisions, on synthetic settings and files.

use gyroship_launcher::{Last, Settings, builds, flags, layers, load, logs, outcome, prepare_save, report, save, session, step};
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
}

/// The game's command line: only what is set, and the practice options only with their file.
#[test]
fn the_command_line_holds_what_is_set() {
    let logs = Path::new("logs");
    let (log, record) = (logs.join("play-3.log").to_string_lossy().into_owned(), logs.join("play-3.txt").to_string_lossy().into_owned());
    let (log, record) = (log.as_str(), record.as_str());
    let bare = Settings { exefs: "game/exefs".into(), boost: "infinite".into(), hud: true, ..Settings::default() };
    assert_eq!(words(&flags(&bare, logs, "play-3")), ["boot", "game/exefs", "--play", "--explore", "--log", log], "no practice file: no --boost, no --hud");
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
        hud: true,
        fullscreen: true,
        joycon: true,
        joycon_mouse: true,
        record: true,
        ..Settings::default()
    };
    let expected = ["boot", "game/exefs", "--play", "--explore", "--save", "save", "--pipeline-cache", "cache", "--language", "fr", "--sdl", "SDL3.dll", "--hooks", "a.hooks", "--romfs-layer", "layer", "--practice", "practice.txt", "--boost", "perfect", "--log", log, "--record", record, "--fullscreen", "--joycon", "--joycon-mouse", "--hud"];
    assert_eq!(words(&flags(&full, logs, "play-3")), expected);
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
/// its about.txt, or by its name.
#[test]
fn layers_are_the_practice_folders_and_their_about_lines() {
    let dir = scratch("layers");
    let practice = dir.join("data").join("practice");
    for folder in ["layer", "layer-arwing", "vehicle"] {
        fs::create_dir_all(practice.join(folder)).unwrap();
    }
    fs::write(practice.join("layer-arwing/about.txt"), "  The Arwing everywhere \nsecond line\n").unwrap();
    fs::write(practice.join("layer-file"), "").unwrap();
    let s = Settings { data: dir.join("data").to_string_lossy().into(), ..Settings::default() };
    let path = |name: &str| practice.join(name).to_string_lossy().into_owned();
    assert_eq!(layers(&s), [(path("layer"), "layer".to_string()), (path("layer-arwing"), "The Arwing everywhere".to_string())]);
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
    let (program, line, folder) = step(&s, "build").unwrap();
    assert_eq!((program.as_path(), words(&line), folder), (Path::new("cargo"), vec!["build", "--release", "--offline", "-j", "4"], generated));
    assert!(step(&Settings { exefs: String::new(), ..s.clone() }, "recompile").unwrap_err().contains("the exefs folder is not set"));
    assert!(step(&Settings::default(), "build").unwrap_err().contains("the repository is not set"));
    // The tools run in the repository, where a relative path would mean another place: refused.
    assert!(step(&Settings { repo: "repo".into(), ..s.clone() }, "build").unwrap_err().contains("the repository is not a full path: repo"));
    assert!(step(&Settings { exefs: "game/exefs".into(), ..s.clone() }, "recompile").unwrap_err().contains("the exefs folder is not a full path"));
    assert!(step(&s, "format-the-disk").unwrap_err().contains("no setup step"));
}
