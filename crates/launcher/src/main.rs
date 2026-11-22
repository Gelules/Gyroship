//! Gyroship's launcher (M6 step 9c): a window over the command line. It picks a build
//! of the game and its options, starts it, says how the last session ended, runs the
//! setup steps (the installs, the tools' build, the recompile, the game's build) and
//! installs new releases (D109, D112), one thing at a time. Its settings and its webview's
//! data live next to its executable.
#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

use gyroship_launcher::{Checks, Last, Needs, RUNNING, Report, Settings, Update, VERSION, builds, checks, flags, install_root, latest_newer, layers, load, logs, needs, prepare_save, report, save, sdl_for, session, step, update_blocker};
use std::io::{BufRead, BufReader};
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::sync::atomic::{AtomicBool, Ordering::Relaxed};
use std::time::{Duration, Instant};
use tauri::{AppHandle, Emitter, Manager, RunEvent, State};

/// A game, a setup step or an update is running: one thing at a time (a port takes gigabytes
/// of memory, and so does the game).
#[derive(Default)]
struct Busy(AtomicBool);

/// The copy this launcher was updated from (`--updated-from`): the only one it may delete.
struct UpdatedFrom(Option<PathBuf>);

/// The launcher's folder: its settings and its webview's data stay with it.
fn folder() -> PathBuf {
    std::env::current_exe().ok().and_then(|e| Some(e.parent()?.to_path_buf())).unwrap_or_default()
}

fn settings_file() -> PathBuf {
    folder().join("launcher.json")
}

/// A word for the player from a launcher that has no window to say it in.
#[cfg(windows)]
fn say(text: &str) {
    #[link(name = "user32")]
    unsafe extern "system" {
        fn MessageBoxW(owner: *mut std::ffi::c_void, text: *const u16, caption: *const u16, kind: u32) -> i32;
    }
    let wide = |s: &str| s.encode_utf16().chain([0]).collect::<Vec<u16>>();
    // 0x40: the information icon, and one OK button.
    unsafe { MessageBoxW(std::ptr::null_mut(), wide(text).as_ptr(), wide("Gyroship launcher: already running").as_ptr(), 0x40) };
}

/// No console for the programs started, unless the player wants the game's: then a new
/// one. The game writes to it only when the launcher hands it no standard handle of its
/// own, and a launcher started from a terminal has some: it forgets them first.
fn window_of(command: &mut Command, console: bool) {
    #[cfg(windows)]
    {
        unsafe extern "system" {
            fn SetStdHandle(id: u32, handle: *mut std::ffi::c_void) -> i32;
        }
        // The standard input, output and error handles' ids: -10, -11 and -12.
        for id in [-10i32, -11, -12].into_iter().filter(|_| console) {
            unsafe { SetStdHandle(id as u32, std::ptr::null_mut()) };
        }
        std::os::windows::process::CommandExt::creation_flags(command, if console { 0x10 } else { 0x0800_0000 });
    }
    #[cfg(not(windows))]
    let _ = (command, console);
}

/// The launcher's window. Its page starts with the launcher's version (`__VERSION__`), whether
/// it is this run's first window (`__FIRST_WINDOW__`: the page then looks for updates), and
/// `said`: after a game, how the session ended (`__AFTER_GAME__`); a word for the player
/// (`__NOTICE__`); after an update, the copy it came from (`__UPDATED_FROM__`).
fn open_window(app: &AppHandle, said: &str) -> tauri::Result<()> {
    static OPENED: AtomicBool = AtomicBool::new(false);
    let first = !OPENED.swap(true, Relaxed);
    let page = tauri::WebviewUrl::App("index.html".into());
    let script = format!("window.__VERSION__ = {}; window.__FIRST_WINDOW__ = {first}; {said}", serde_json::Value::String(VERSION.into()));
    tauri::WebviewWindowBuilder::new(app, "main", page).title("Gyroship launcher").inner_size(960.0, 820.0).data_directory(folder().join("webview-data")).initialization_script(script).build().map(drop)
}

/// The settings, and a word for the player when their file did not parse. With no repository
/// set yet, the Gyroship folder the launcher is in, if it is in one (a release's).
#[tauri::command]
fn get_settings() -> (Settings, Option<String>) {
    let (mut s, said) = load(&settings_file());
    if s.repo.is_empty()
        && let Some(root) = std::env::current_exe().ok().and_then(|e| install_root(&e))
    {
        s.repo = root.to_string_lossy().into_owned();
    }
    (s, said)
}

#[tauri::command]
fn set_settings(settings: Settings) -> Result<(), String> {
    save(&settings_file(), &settings)
}

#[tauri::command]
fn list_builds(settings: Settings) -> Vec<String> {
    builds(&settings)
}

#[tauri::command]
fn list_layers(settings: Settings) -> Vec<(String, String)> {
    layers(&settings)
}

/// What the page flags: the Setup tab's problems, the dump's, the port's progress.
#[tauri::command]
fn check(settings: Settings) -> Checks {
    checks(&settings)
}

/// What building and running the game needs on this PC: Rust, the C++ tools, the runtime.
/// Off the window's thread: the three programs it runs take a moment.
#[tauri::command(async)]
fn list_needs() -> Needs {
    needs()
}

/// This launcher's version, the latest release when it is newer (D109), and why this launcher
/// cannot install it itself, if it cannot (D112: a Git clone, a launcher inside the sources, a
/// folder already taken; said at start, rather than downloaded again at each one). An error when
/// GitHub cannot be asked.
#[tauri::command(async)]
fn check_update(settings: Settings) -> Result<Update, String> {
    let exe = std::env::current_exe().map_err(|e| e.to_string())?;
    let release = latest_newer()?;
    let taken = || gyroship_launcher::target(Path::new(&settings.repo), &release.as_ref()?.tag).err();
    let blocker = update_blocker(&exe, &settings).or_else(taken);
    Ok(Update { version: VERSION.into(), release, blocker })
}

/// Installs the latest release in the background (D112), telling the page how far it went
/// ("update"): its sources next to the repository's, its programs in place of this launcher's.
/// Then starts the new launcher, which rebuilds the game, and ends; a failure comes as
/// "update-failed". Never while a game or a step runs.
#[tauri::command]
fn apply_update(app: AppHandle, busy: State<Busy>, settings: Settings) -> Result<(), String> {
    let exe = std::env::current_exe().map_err(|e| e.to_string())?;
    if let Some(why) = update_blocker(&exe, &settings) {
        return Err(why);
    }
    if busy.0.swap(true, Relaxed) {
        return Err("a game or a setup step is running".into());
    }
    std::thread::spawn(move || {
        let page = app.clone();
        let started = gyroship_launcher::update(&settings, &exe, |text| drop(page.emit("update", text))).and_then(|old| {
            // The new launcher, at this one's place: it waits for this one's lock, freed as it ends.
            let mut child = Command::new(&exe).arg("--updated-from").arg(&old).current_dir(folder()).spawn().map_err(|e| format!("{}: {e}", exe.display()))?;
            // A new launcher that ends at once shows no message of its own: said here, and this one stays.
            std::thread::sleep(Duration::from_secs(3));
            match child.try_wait() {
                Ok(Some(status)) => Err(format!("the new launcher stopped at its start ({status}): start {} yourself", exe.display())),
                _ => Ok(()),
            }
        });
        match started {
            Ok(()) => app.exit(0),
            Err(e) => {
                app.state::<Busy>().0.store(false, Relaxed);
                let _ = app.emit("update-failed", e);
            }
        }
    });
    Ok(())
}

/// A release's page, in the player's browser: the repository's release pages only.
#[tauri::command]
fn open_release(url: String) -> Result<(), String> {
    if !gyroship_launcher::is_release_page(&url) {
        return Err(format!("not a release page of {}: {url}", gyroship_launcher::REPOSITORY));
    }
    Command::new("explorer.exe").arg(url).spawn().map(drop).map_err(|e| e.to_string())
}

/// The sources the update came from, deleted on the player's click: those this launcher was
/// started for, checked against the settings as saved, never what the page says. Off the
/// window's thread (sources hold a built game, thousands of files), and busy meanwhile, so that
/// the window waits for its end.
#[tauri::command(async)]
fn delete_previous(app: AppHandle) -> Result<(), String> {
    let old = app.state::<UpdatedFrom>().0.clone().ok_or("this launcher was not started by an update")?;
    let busy = app.state::<Busy>();
    if busy.0.swap(true, Relaxed) {
        return Err("a game, a setup step or an update is running".into());
    }
    let s = load(&settings_file()).0;
    let done = gyroship_launcher::delete_previous(&s, &old, Path::new(&s.repo));
    busy.0.store(false, Relaxed);
    done
}

/// The journals' folder, once the data folder is set: without it, `logs` would be
/// looked for under wherever the launcher happens to run.
fn journals(settings: &Settings) -> Result<PathBuf, String> {
    if Path::new(&settings.data).is_absolute() { Ok(logs(settings)) } else { Err("the data folder (Setup tab) is not set, or is not a full path".into()) }
}

#[tauri::command]
fn last_session(settings: Settings) -> Option<Report> {
    report(&journals(&settings).ok()?)
}

/// The journals' folder, in the file explorer (which wants backslashes, and opens
/// another folder without a word when its own is missing).
#[tauri::command]
fn open_logs(settings: Settings) -> Result<(), String> {
    let folder = std::path::absolute(journals(&settings)?).map_err(|e| e.to_string())?;
    if !folder.is_dir() {
        return Err(format!("{} does not exist yet: play a game first", folder.display()));
    }
    Command::new("explorer.exe").arg(folder).spawn().map(drop).map_err(|e| e.to_string())
}

#[tauri::command]
fn play(app: AppHandle, settings: Settings) -> Result<(), String> {
    start(&app, settings, false)
}

/// Starts the game. The window goes while it runs, so that the webview's memory is the
/// game's to use, and comes back with the session's end. A `shortcut` start (`--play`)
/// shows no window after a game that ended normally: the launcher ends with it.
fn start(app: &AppHandle, settings: Settings, shortcut: bool) -> Result<(), String> {
    let busy = app.state::<Busy>();
    for (what, path) in [("game build", &settings.build), ("game's exefs folder", &settings.exefs), ("data folder (Setup tab)", &settings.data)] {
        if path.is_empty() {
            return Err(format!("the {what} is not set"));
        }
        if !Path::new(path).is_absolute() || !Path::new(path).exists() {
            return Err(format!("the {what} does not exist, or is not a full path: {path}"));
        }
    }
    // The game would look for a relative path from the launcher's current folder, which an update
    // changes: full paths only.
    for (what, path) in [("save folder", &settings.save), ("shader cache folder", &settings.cache), ("hooks file", &settings.hooks), ("mod", &settings.layer), ("practice file", &settings.practice)] {
        if !path.is_empty() && !Path::new(path).is_absolute() {
            return Err(format!("the {what} is not a full path: {path}"));
        }
    }
    if busy.0.swap(true, Relaxed) {
        return Err("a game or a setup step is already running".into());
    }
    let started = (|| {
        let logs = logs(&settings);
        std::fs::create_dir_all(&logs).map_err(|e| format!("{}: {e}", logs.display()))?;
        let name = session(&logs);
        // The save the game gets: the chosen folder, a copy of it, or an empty one; and SDL3, the
        // one chosen or the one a release puts next to the launcher.
        let played = Settings { save: prepare_save(&settings)?, sdl: sdl_for(&settings, &folder()), ..settings.clone() };
        let mut game = Command::new(&settings.build);
        game.args(flags(&played, &logs, &name));
        if !settings.console {
            let messages = std::fs::File::create(logs.join(format!("{name}.err.txt"))).map_err(|e| e.to_string())?;
            game.stderr(messages).stdout(Stdio::null());
        }
        window_of(&mut game, settings.console);
        let child = game.spawn().map_err(|e| format!("{}: {e}", settings.build))?;
        Ok::<_, String>((child, logs, name))
    })();
    let (mut child, logs, name) = match started {
        Ok(s) => s,
        Err(e) => {
            busy.0.store(false, Relaxed);
            return Err(e);
        }
    };
    let record = move |code: i32, seconds: u64| {
        if let Ok(text) = serde_json::to_string(&Last { session: name.clone(), code, seconds }) {
            let _ = std::fs::write(logs.join("last.json"), text);
        }
    };
    // Should the launcher stop before the game does, the session is still the last one.
    record(RUNNING, 0);
    if let Some(w) = app.get_webview_window("main") {
        let _ = w.destroy();
    }
    let app = app.clone();
    std::thread::spawn(move || {
        let began = Instant::now();
        let code = child.wait().ok().and_then(|s| s.code()).unwrap_or(-1);
        record(code, began.elapsed().as_secs());
        if shortcut && code == 0 {
            return app.exit(0);
        }
        // A game that ends at once must not find the old window still closing: its
        // name must be free, and the launcher stays busy (so alive) until the new one is up.
        while app.get_webview_window("main").is_some() {
            std::thread::sleep(Duration::from_millis(10));
        }
        let again = app.clone();
        let _ = app.run_on_main_thread(move || {
            let opened = open_window(&again, "window.__AFTER_GAME__ = true;");
            again.state::<Busy>().0.store(false, Relaxed);
            if opened.is_err() {
                again.exit(1);
            }
        });
    });
    Ok(())
}

/// Runs a setup step and sends its output to the page, line by line ("line"), then
/// its exit code ("done").
#[tauri::command]
fn run_step(app: AppHandle, busy: State<Busy>, settings: Settings, kind: String) -> Result<(), String> {
    let (program, arguments, folder) = step(&settings, &kind)?;
    if !folder.is_dir() {
        return Err(format!("{} does not exist{}", folder.display(), if kind == "build" { ": recompile the game first" } else { "" }));
    }
    // Recompile runs the tools: a release brings them, the tools step builds them, or they were
    // downloaded next to the launcher.
    if kind == "recompile" && !program.is_file() {
        return Err(format!("{} does not exist: build the tools first (Build the tools, above), or put gyroship-tools.exe next to the launcher", program.display()));
    }
    if busy.0.swap(true, Relaxed) {
        return Err("a game or a setup step is already running".into());
    }
    let mut command = Command::new(&program);
    command.args(&arguments).current_dir(&folder).stdout(Stdio::piped()).stderr(Stdio::piped()).stdin(Stdio::null());
    // Cargo's crates go to the data folder's `cargo`, when there is one and cargo's home is
    // not set otherwise.
    let home = Path::new(&settings.data).join("cargo");
    if ["build", "tools"].contains(&kind.as_str()) && std::env::var_os("CARGO_HOME").is_none() && home.is_dir() {
        command.env("CARGO_HOME", home);
    }
    window_of(&mut command, false);
    let mut child = match command.spawn() {
        Ok(c) => c,
        Err(e) => {
            busy.0.store(false, Relaxed);
            return Err(format!("{}: {e}", program.display()));
        }
    };
    let _ = app.emit("line", format!("> {} {}", program.display(), arguments.join(" ")));
    let readers: Vec<_> = [child.stdout.take().map(|o| Box::new(o) as Box<dyn std::io::Read + Send>), child.stderr.take().map(|e| Box::new(e) as Box<dyn std::io::Read + Send>)]
        .into_iter()
        .flatten()
        .map(|stream| {
            let app = app.clone();
            std::thread::spawn(move || {
                // A progress line redrawn in place (carriage returns) is one line here.
                for line in BufReader::new(stream).split(b'\n').map_while(Result::ok) {
                    let text = String::from_utf8_lossy(&line);
                    let _ = app.emit("line", text.rsplit('\r').find(|t| !t.is_empty()).unwrap_or("").to_string());
                }
            })
        })
        .collect();
    std::thread::spawn(move || {
        let code = child.wait().ok().and_then(|s| s.code()).unwrap_or(-1);
        // The last lines, but not for ever: a helper process the step left behind may
        // hold its output open long after it.
        let until = Instant::now() + Duration::from_secs(2);
        while readers.iter().any(|r| !r.is_finished()) && Instant::now() < until {
            std::thread::sleep(Duration::from_millis(50));
        }
        app.state::<Busy>().0.store(false, Relaxed);
        let _ = app.emit("done", code);
    });
    Ok(())
}

fn main() {
    // One launcher at a time: a second one would not know that the first is waiting for
    // its game, and would start another beside it. The lock goes with the process.
    #[cfg(windows)]
    let _only = {
        use std::os::windows::fs::OpenOptionsExt;
        // After an update, the launcher that started this one holds the lock until it ends, a few
        // seconds later: waited for, 30 s at most.
        let patience = if std::env::args().any(|a| a == "--updated-from") { 150 } else { 0 };
        let mut tries = 0;
        loop {
            match std::fs::OpenOptions::new().write(true).create(true).truncate(false).share_mode(0).open(folder().join("launcher.lock")) {
                // 32: a sharing violation, the other launcher holds the file.
                Err(e) if e.raw_os_error() == Some(32) && tries < patience => tries += 1,
                Err(e) if e.raw_os_error() == Some(32) => return say("The launcher is already running: use its window, or close it first.\n\nDuring a game it has no window: it comes back when the game ends."),
                lock => break lock.ok(),
            }
            std::thread::sleep(Duration::from_millis(200));
        }
    };
    // The programs an update replaced, now that the launcher they belonged to has ended.
    gyroship_launcher::remove_replaced(&folder());
    let app = tauri::Builder::default()
        .manage(Busy::default())
        .manage(UpdatedFrom(std::env::args().skip_while(|a| a != "--updated-from").nth(1).map(PathBuf::from)))
        .setup(|app| {
            // `--play` (a shortcut's argument): the game at once, with the saved settings.
            let direct = std::env::args().any(|a| a == "--play").then(|| start(app.handle(), load(&settings_file()).0, true));
            // `--updated-from FOLDER`: started by the launcher of that older copy, after an update.
            let updated = std::env::args().skip_while(|a| a != "--updated-from").nth(1);
            match direct {
                Some(Ok(())) => {}
                Some(Err(e)) => open_window(app.handle(), &format!("window.__NOTICE__ = {};", serde_json::Value::String(format!("The game did not start: {e}"))))?,
                None => open_window(app.handle(), &updated.map_or(String::new(), |old| format!("window.__UPDATED_FROM__ = {};", serde_json::Value::String(old))))?,
            }
            Ok(())
        })
        // A setup step or an update half done would leave a half-written folder: the window
        // stays until it ends. (During a game there is no window to close.)
        .on_window_event(|window, event| {
            if let tauri::WindowEvent::CloseRequested { api, .. } = event
                && window.state::<Busy>().0.load(Relaxed)
            {
                api.prevent_close();
                let _ = window.emit("notice", "A setup step or an update is running: close the window once it ends.");
            }
        })
        .invoke_handler(tauri::generate_handler![get_settings, set_settings, list_builds, list_layers, check, list_needs, check_update, apply_update, open_release, delete_previous, last_session, open_logs, play, run_step])
        .build(tauri::generate_context!())
        .expect("the launcher could not start");
    app.run(|app, event| {
        // Without a window the launcher goes on while a game runs: it waits for its end.
        if let RunEvent::ExitRequested { api, code: None, .. } = event
            && app.state::<Busy>().0.load(Relaxed)
        {
            api.prevent_exit();
        }
    });
}
