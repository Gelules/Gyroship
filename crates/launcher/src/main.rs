//! Gyroship's launcher (M6 step 9c): a window over the command line. It picks a build
//! of the game and its options, starts it, says how the last session ended, and runs
//! the setup steps (recompile the game, build it), one thing at a time. Its settings
//! and its webview's data live next to its executable.
#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

use gyroship_launcher::{Last, RUNNING, Report, Settings, builds, flags, load, logs, report, save, session, step};
use std::io::{BufRead, BufReader};
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::sync::atomic::{AtomicBool, Ordering::Relaxed};
use std::time::{Duration, Instant};
use tauri::{AppHandle, Emitter, Manager, RunEvent, State};

/// A game or a setup step is running: one thing at a time (a port takes gigabytes of
/// memory, and so does the game).
#[derive(Default)]
struct Busy(AtomicBool);

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

/// The launcher's window. Its page starts with `said`: after a game, how the session
/// ended (`__AFTER_GAME__`); a word for the player (`__NOTICE__`).
fn open_window(app: &AppHandle, said: &str) -> tauri::Result<()> {
    let page = tauri::WebviewUrl::App("index.html".into());
    tauri::WebviewWindowBuilder::new(app, "main", page).title("Gyroship launcher").inner_size(960.0, 820.0).data_directory(folder().join("webview-data")).initialization_script(said).build().map(drop)
}

/// The settings, and a word for the player when their file did not parse.
#[tauri::command]
fn get_settings() -> (Settings, Option<String>) {
    load(&settings_file())
}

#[tauri::command]
fn set_settings(settings: Settings) -> Result<(), String> {
    save(&settings_file(), &settings)
}

#[tauri::command]
fn list_builds(settings: Settings) -> Vec<String> {
    builds(&settings)
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
        return Err(format!("{} does not exist yet: it comes with the first game", folder.display()));
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
    for (what, path) in [("build", &settings.build), ("dump's exefs folder", &settings.exefs), ("data folder (Setup tab)", &settings.data)] {
        if path.is_empty() {
            return Err(format!("the {what} is not set"));
        }
        if !Path::new(path).is_absolute() || !Path::new(path).exists() {
            return Err(format!("the {what} does not exist, or is not a full path: {path}"));
        }
    }
    if busy.0.swap(true, Relaxed) {
        return Err("a game or a setup step is already running".into());
    }
    let started = (|| {
        let logs = logs(&settings);
        std::fs::create_dir_all(&logs).map_err(|e| format!("{}: {e}", logs.display()))?;
        let name = session(&logs);
        let mut game = Command::new(&settings.build);
        game.args(flags(&settings, &logs, &name));
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
    // The tools are the repository's own build, the one step the launcher does not do.
    if kind == "recompile" && !program.is_file() {
        return Err(format!("{} does not exist: build the tools first (cargo build --release, in the repository)", program.display()));
    }
    if busy.0.swap(true, Relaxed) {
        return Err("a game or a setup step is already running".into());
    }
    let mut command = Command::new(&program);
    command.args(&arguments).current_dir(&folder).stdout(Stdio::piped()).stderr(Stdio::piped()).stdin(Stdio::null());
    // The build fetches nothing: its crates are those of the data folder's `cargo`,
    // when there is one and cargo's home is not set otherwise.
    let home = Path::new(&settings.data).join("cargo");
    if kind == "build" && std::env::var_os("CARGO_HOME").is_none() && home.is_dir() {
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
        match std::fs::OpenOptions::new().write(true).create(true).truncate(false).share_mode(0).open(folder().join("launcher.lock")) {
            // 32: a sharing violation, the other launcher holds the file.
            Err(e) if e.raw_os_error() == Some(32) => return say("The launcher is already running: use its window, or close it first.\n\nWhile a game it started runs, it has no window: it comes back when the game ends."),
            lock => lock.ok(),
        }
    };
    let app = tauri::Builder::default()
        .manage(Busy::default())
        .setup(|app| {
            // `--play` (a shortcut's argument): the game at once, with the saved settings.
            let direct = std::env::args().any(|a| a == "--play").then(|| start(app.handle(), load(&settings_file()).0, true));
            match direct {
                Some(Ok(())) => {}
                Some(Err(e)) => open_window(app.handle(), &format!("window.__NOTICE__ = {};", serde_json::Value::String(format!("The game did not start: {e}"))))?,
                None => open_window(app.handle(), "")?,
            }
            Ok(())
        })
        // A setup step half done would leave a half-written folder: the window stays
        // until it ends. (During a game there is no window to close.)
        .on_window_event(|window, event| {
            if let tauri::WindowEvent::CloseRequested { api, .. } = event
                && window.state::<Busy>().0.load(Relaxed)
            {
                api.prevent_close();
                let _ = window.emit("notice", "A setup step is running: the window can close when it ends.");
            }
        })
        .invoke_handler(tauri::generate_handler![get_settings, set_settings, list_builds, last_session, open_logs, play, run_step])
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
