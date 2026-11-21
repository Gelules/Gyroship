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

/// A copy of Gyroship, as far as its launcher can tell: the launcher at its top, beside the
/// runtime's sources.
fn copy_of_gyroship(root: &Path) {
    fs::create_dir_all(at(root, "crates/runtime")).unwrap();
    fs::write(at(root, "crates/runtime/Cargo.toml"), "").unwrap();
    fs::write(root.join(LAUNCHER), "").unwrap();
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
        ..Settings::default()
    };
    let expected = ["boot", "game/exefs", "--play", "--explore", "--save", "save", "--pipeline-cache", "cache", "--language", "fr", "--sdl", "SDL3.dll", "--hooks", "a.hooks", "--romfs-layer", "layer", "--practice", "practice.txt", "--boost", "perfect", "--log", log, "--record", record, "--fullscreen", "--joycon", "--joycon-mouse", "--hud-speed", "--hud-markers"];
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
    assert_eq!(c.setup, ["Repository: folder not found", "Data folder: folder not found", "No game build yet: recompile the game, then build it"]);
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
    let everything = gyroship_launcher::Checks { setup: Vec::new(), folders: true, exefs: None, dump: Some(id.into()), dump_known: Some(true), tools: true, recompiled: true, built: true };
    assert_eq!(checks(&s), everything);
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

/// Updates (D109): GitHub's answer for the latest release, read; only a newer one, with the
/// repository's own page and archive, is taken.
#[test]
fn a_newer_release_is_read_from_githubs_answer() {
    use gyroship_launcher::{REPOSITORY, is_newer, release_from};
    assert!(is_newer("v0.2.0", "0.1.0") && is_newer("0.10.0", "0.9.9") && is_newer("v1.0.0-rc.1", "0.9.0"));
    assert!(!is_newer("v0.1.0", "0.1.0") && !is_newer("v0.1.0-beta", "0.1.0") && !is_newer("v1.0", "0.1.0") && !is_newer("latest", "0.1.0"));
    let answer = |tag: &str, page: &str, assets: &str| format!(r#"{{"tag_name": "{tag}", "html_url": "{page}", "draft": false, "assets": [{assets}]}}"#);
    let asset = |name: &str, url: &str| format!(r#"{{"name": "{name}", "browser_download_url": "{url}", "size": 1234, "digest": "sha256:00"}}"#);
    let page = format!("{REPOSITORY}/releases/tag/v0.2.0");
    let zip = format!("{REPOSITORY}/releases/download/v0.2.0/gyroship-0.2.0-windows-x64.zip");
    let assets = [asset("gyroship-tools.exe", &format!("{REPOSITORY}/releases/download/v0.2.0/gyroship-tools.exe")), asset("gyroship-0.2.0-windows-x64.zip", &zip)].join(",");
    let r = release_from(&answer("v0.2.0", &page, &assets), "0.1.0").unwrap().unwrap();
    assert_eq!((r.tag.as_str(), r.page.as_str(), r.zip.as_str(), r.size), ("v0.2.0", page.as_str(), zip.as_str(), 1234));
    assert_eq!(release_from(&answer("v0.1.0", &page, &assets), "0.1.0").unwrap(), None, "not newer: nothing");
    // An archive served from elsewhere, another repository's page, a tag that names no folder:
    // refused.
    let foreign = asset("gyroship-0.2.0-windows-x64.zip", "https://example.com/gyroship-0.2.0-windows-x64.zip");
    assert!(release_from(&answer("v0.2.0", &page, &foreign), "0.1.0").unwrap_err().contains("no gyroship-"));
    // `..` segments, which curl and browsers resolve: another repository's archive or page.
    let away = asset("gyroship-0.2.0-windows-x64.zip", &format!("{REPOSITORY}/releases/download/../../../../someone/else/releases/download/v0.2.0/gyroship-0.2.0-windows-x64.zip"));
    assert!(release_from(&answer("v0.2.0", &page, &away), "0.1.0").is_err());
    assert!(!gyroship_launcher::is_release_page(&format!("{REPOSITORY}/releases/../../someone/else")));
    let other = asset("gyroship-0.1.9-windows-x64.zip", &format!("{REPOSITORY}/releases/download/v0.2.0/gyroship-0.1.9-windows-x64.zip"));
    assert!(release_from(&answer("v0.2.0", &page, &other), "0.1.0").is_err(), "another version's archive");
    assert!(release_from(&answer("v0.2.0", "https://github.com/someone/else/releases/tag/v0.2.0", &assets), "0.1.0").is_err());
    assert!(release_from(&answer("v0.2.0+../../x", &page, &assets), "0.1.0").is_err());
    assert!(release_from("{}", "0.1.0").unwrap_err().contains("GitHub's answer"));
}

/// An update: the release's archive unpacked next to the current copy and renamed once whole;
/// the copy's own paths in the settings follow it, the player's stay, SDL3.dll goes with the
/// game. A stopped update's folder is removed, a folder of the same name that is not one is not.
#[cfg(windows)]
#[test]
fn a_release_installs_next_to_the_current_copy() {
    use gyroship_launcher::{install, install_root, update_blocker};
    let dir = scratch("update");
    let (old, new, part) = (dir.join("Gyroship-0.1.0"), dir.join("Gyroship-0.2.0"), dir.join("Gyroship-0.2.0.part"));
    copy_of_gyroship(&old);
    let staged = at(&dir, "staged/Gyroship-0.2.0");
    copy_of_gyroship(&staged);
    fs::create_dir_all(staged.join("templates")).unwrap();
    fs::write(at(&staged, "templates/practice.txt"), "new").unwrap();
    fs::write(staged.join("gyroship-tools.exe"), "").unwrap();
    // Windows' own tar makes the archives, as git archive would: with a top folder, or without.
    let tar = |zip: &Path, from: &Path, what: &[&str]| {
        let tar = at(Path::new(&std::env::var_os("SystemRoot").unwrap()), "System32/tar.exe");
        assert!(std::process::Command::new(tar).args(["-a", "-c", "-f"]).arg(zip).arg("-C").arg(from).args(what).status().unwrap().success());
    };
    let (zip, flat) = (dir.join("gyroship-0.2.0-windows-x64.zip"), dir.join("flat.zip"));
    tar(&zip, &dir.join("staged"), &["Gyroship-0.2.0"]);
    tar(&flat, &staged, &[LAUNCHER, "gyroship-tools.exe", "crates", "templates"]);
    fs::create_dir_all(at(&old, "generated/game/target/release")).unwrap();
    fs::write(at(&old, "generated/game/target/release/SDL3.dll"), "sdl").unwrap();
    let s = Settings {
        repo: text(&old),
        build: text(&at(&old, "generated/game/target/release/gyroship.exe")),
        practice: text(&at(&old, "templates/practice.txt")),
        probes: text(&at(&old, "templates/my-probes.txt")),
        sdl: text(&at(&old, "generated/game/target/release/SDL3.dll")),
        layer: text(&at(&old, "my-layer")),
        data: text(&old.join("Data")),
        save: text(&dir.join("save")),
        ..Settings::default()
    };
    assert_eq!(gyroship_launcher::install(&zip, "v0.2.0", &old, &s).unwrap(), new);
    assert!(new.join(LAUNCHER).is_file() && at(&new, "templates/practice.txt").is_file() && !part.exists());
    assert_eq!(fs::read_to_string(at(&new, "generated/game/target/release/SDL3.dll")).unwrap(), "sdl");
    // The copy's own files move; a file of the player's in its templates, which the new copy does
    // not have, stays (and so keeps the old copy from being deleted), as do the player's folders.
    let moved = Settings {
        repo: text(&new),
        build: text(&at(&new, "generated/game/target/release/gyroship.exe")),
        practice: text(&at(&new, "templates/practice.txt")),
        sdl: text(&at(&new, "generated/game/target/release/SDL3.dll")),
        ..s.clone()
    };
    assert_eq!(load(&new.join("launcher.json")), (moved, None));
    // Installed: that folder is taken, and that copy, at its version, needs no update.
    assert!(gyroship_launcher::target(&old, "v0.2.0").unwrap_err().contains("already exists"));
    assert!(gyroship_launcher::target(&new, "v0.2.0").unwrap_err().contains("without setting its version"));
    // A release's tools are at its top, until the tools step builds its own.
    assert_eq!(gyroship_launcher::tools_exe(&text(&new)), new.join("gyroship-tools.exe"));
    fs::create_dir_all(at(&new, "target/release")).unwrap();
    fs::write(at(&new, "target/release/gyroship-tools.exe"), "").unwrap();
    assert_eq!(gyroship_launcher::tools_exe(&text(&new)), at(&new, "target/release/gyroship-tools.exe"));
    assert!(install(&zip, "v0.2.0", &old, &s).unwrap_err().contains("already exists"));
    // A folder named as an update's but without its mark: not the launcher's to remove.
    fs::remove_dir_all(&new).unwrap();
    fs::create_dir_all(&part).unwrap();
    assert!(install(&flat, "v0.2.0", &old, &s).is_err());
    assert!(part.exists());
    // With the mark, a stopped update's: removed, and the archive without a top folder installs.
    fs::write(part.join(".gyroship-update"), "").unwrap();
    assert_eq!(install(&flat, "v0.2.0", &old, &s).unwrap(), new);
    assert!(new.join(LAUNCHER).is_file() && !new.join(".gyroship-update").exists() && !part.exists());
    // An archive with no launcher in it: refused, and nothing left behind.
    fs::remove_dir_all(&new).unwrap();
    fs::remove_file(staged.join(LAUNCHER)).unwrap();
    let bad = dir.join("bad.zip");
    tar(&bad, &dir.join("staged"), &["Gyroship-0.2.0"]);
    assert!(install(&bad, "v0.2.0", &old, &s).unwrap_err().contains("no gyroship-launcher.exe"));
    assert!(!new.exists() && !part.exists());
    assert!(install(&zip, "v0.2.0/../x", &old, &s).unwrap_err().contains("no version"));
    // The copy a launcher belongs to: a release's, at its top; a Git clone's, deeper, which
    // updates through Git.
    assert_eq!((install_root(&old.join(LAUNCHER)), update_blocker(&old.join(LAUNCHER))), (Some(old.clone()), None));
    fs::create_dir_all(old.join(".git")).unwrap();
    let deep = at(&old, "crates/launcher/target/release").join(LAUNCHER);
    assert_eq!(install_root(&deep), Some(old.clone()));
    assert!(update_blocker(&deep).unwrap().contains("git pull"));
}

/// The previous copy goes on the player's click: only a release's folder next to this one, whose
/// launcher has ended, and which holds nothing the player chose.
#[cfg(windows)]
#[test]
fn the_previous_copy_goes_only_when_nothing_of_the_players_is_in_it() {
    use gyroship_launcher::delete_previous;
    use std::os::windows::fs::OpenOptionsExt;
    let dir = scratch("previous");
    let (old, new, away) = (dir.join("Gyroship-0.1.0"), dir.join("Gyroship-0.2.0"), at(&dir, "away/Gyroship-0.0.9"));
    for root in [&old, &new, &away] {
        copy_of_gyroship(root);
    }
    let s = Settings { repo: text(&new), save: text(&old.join("save")), ..Settings::default() };
    assert!(delete_previous(&s, &old, &new).unwrap_err().contains("holds your save folder"));
    // The same save folder written otherwise: relative (any folder, for all the launcher knows),
    // verbatim, through `..`.
    fs::create_dir_all(old.join("save")).unwrap();
    for save in ["save".to_string(), format!(r"\\?\{}", old.join("save").display()), text(&at(&dir, "x/../Gyroship-0.1.0/save"))] {
        assert!(delete_previous(&Settings { save: save.clone(), ..s.clone() }, &old, &new).unwrap_err().contains("holds your save folder"), "{save}");
    }
    let s = Settings { save: String::new(), ..s };
    for (not, why) in [(&new, "this copy"), (&dir, "no copy"), (&away, "not next to this one")] {
        assert!(delete_previous(&s, not, &new).unwrap_err().contains("no previous copy"), "{why}");
    }
    let mut this = text(&new);
    this.push(std::path::MAIN_SEPARATOR);
    assert!(delete_previous(&s, Path::new(&this), &new).unwrap_err().contains("no previous copy"), "this copy, written with a final separator");
    fs::create_dir_all(old.join(".git")).unwrap();
    assert!(delete_previous(&s, &old, &new).is_err(), "a Git clone");
    fs::remove_dir_all(old.join(".git")).unwrap();
    // Until this copy's game is built, the previous copy keeps the only one.
    assert!(delete_previous(&s, &old, &new).unwrap_err().contains("not built"));
    fs::create_dir_all(at(&new, "generated/game/target/release")).unwrap();
    fs::write(at(&new, "generated/game/target/release/gyroship.exe"), "").unwrap();
    let running = fs::OpenOptions::new().write(true).create(true).truncate(false).share_mode(0).open(old.join("launcher.lock")).unwrap();
    assert!(delete_previous(&s, &old, &new).unwrap_err().contains("still running"));
    drop(running);
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
