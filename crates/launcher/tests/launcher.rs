//! The launcher's decisions, on synthetic settings and files.

use gyroship_launcher::{Last, Settings, builds, flags, load, logs, outcome, report, save, session, step};
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
    let chosen = Settings { build: "b/harness.exe".into(), language: "fr".into(), joycon: true, boost: "perfect".into(), ..Settings::default() };
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
        record: true,
        ..Settings::default()
    };
    let expected = ["boot", "game/exefs", "--play", "--explore", "--save", "save", "--pipeline-cache", "cache", "--language", "fr", "--sdl", "SDL3.dll", "--hooks", "a.hooks", "--romfs-layer", "layer", "--practice", "practice.txt", "--boost", "perfect", "--log", log, "--record", record, "--fullscreen", "--joycon", "--hud"];
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

/// Builds: the generated one first, then those kept under the data folder, existing files only.
#[test]
fn builds_are_the_generated_one_and_those_kept() {
    let dir = scratch("builds");
    let (repo, data) = (dir.join("repo"), dir.join("data"));
    for folder in [repo.join("generated/game/target/release"), data.join("bin/b"), data.join("bin/a"), data.join("bin/empty")] {
        fs::create_dir_all(&folder).unwrap();
    }
    let s = Settings { repo: repo.to_string_lossy().into(), data: data.to_string_lossy().into(), ..Settings::default() };
    assert!(repo.is_absolute(), "the test's folder is a full path");
    assert_eq!(builds(&s), Vec::<String>::new());
    for exe in [repo.join("generated/game/target/release/harness.exe"), data.join("bin/b/harness.exe"), data.join("bin/a/harness.exe")] {
        fs::write(exe, "").unwrap();
    }
    let found = builds(&s);
    let ends: Vec<_> = found.iter().map(|b| Path::new(b).parent().unwrap().file_name().unwrap().to_str().unwrap()).collect();
    assert_eq!(ends, ["release", "a", "b"]);
    // Folders not set, or not full paths: nothing is looked for under the current folder.
    std::env::set_current_dir(&dir).unwrap();
    assert_eq!(builds(&Settings { repo: "repo".into(), data: "data".into(), ..Settings::default() }), Vec::<String>::new());
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
