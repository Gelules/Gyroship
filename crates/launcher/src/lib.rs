//! What Gyroship's launcher decides, apart from its window: the player's settings and
//! their file, the game's command line, the setup steps' command lines, and what a
//! session's end means. The launcher opens no connection and downloads nothing.

use serde::{Deserialize, Serialize};
use std::fs;
use std::io::{Read, Seek, SeekFrom};
use std::path::{Path, PathBuf};

/// What the player chose; `launcher.json`, next to the launcher. Paths are the
/// player's own: an empty one leaves its option out.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct Settings {
    /// The repository (its tools, its generated build) and the data folder (kept
    /// builds under `bin`, journals under `logs`).
    pub repo: String,
    pub data: String,
    /// The game's executable, and the dump's `exefs` folder it boots (`romfs` next to it).
    pub build: String,
    pub exefs: String,
    pub save: String,
    pub cache: String,
    /// A console language code, or nothing for the game's default.
    pub language: String,
    pub fullscreen: bool,
    pub joycon: bool,
    pub sdl: String,
    /// The game's debug menu and Gyroship's practice tools.
    pub hooks: String,
    pub layer: String,
    pub practice: String,
    /// "", "perfect" or "infinite".
    pub boost: String,
    pub hud: bool,
    /// Records the session's inputs next to its journal.
    pub record: bool,
    /// The game in a console that shows its messages, instead of a file of them.
    pub console: bool,
    /// Setup: the probe list compiled into a port.
    pub probes: String,
}

/// The settings in `file`, and what to tell the player. A file that does not parse is
/// kept aside as `<file>.bad` before the defaults take its place: it may be hand-made.
pub fn load(file: &Path) -> (Settings, Option<String>) {
    let text = match fs::read(file) {
        Ok(t) => t,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => return (Settings::default(), None),
        Err(e) => return (Settings::default(), Some(format!("{}: {e}", file.display()))),
    };
    // Bytes, not text: a file saved in another encoding fails here and is kept aside too.
    match serde_json::from_slice(&text) {
        Ok(s) => (s, None),
        Err(e) => {
            let bad = PathBuf::from(format!("{}.bad", file.display()));
            let kept = if fs::copy(file, &bad).is_ok() { format!("kept as {}", bad.display()) } else { "which could not be kept aside".into() };
            (Settings::default(), Some(format!("{} does not parse ({e}): {kept}, the defaults are used", file.display())))
        }
    }
}

pub fn save(file: &Path, s: &Settings) -> Result<(), String> {
    let text = serde_json::to_string_pretty(s).map_err(|e| e.to_string())?;
    fs::write(file, text).map_err(|e| format!("{}: {e}", file.display()))
}

/// Where the sessions' journals go.
pub fn logs(s: &Settings) -> PathBuf {
    Path::new(&s.data).join("logs")
}

/// The next session's name in `logs`: `play-<n>`, after the highest there.
pub fn session(logs: &Path) -> String {
    let taken = fs::read_dir(logs).into_iter().flatten().flatten().filter_map(|e| e.file_name().to_str()?.strip_prefix("play-")?.split('.').next()?.parse::<u32>().ok()).max();
    format!("play-{}", taken.map_or(1, |n| n + 1))
}

/// The game's command line: the real-time engine, what the emulation cannot do
/// skipped, then the player's options. `--boost` and `--hud` need the practice file.
pub fn flags(s: &Settings, logs: &Path, session: &str) -> Vec<String> {
    let file = |extension: &str| logs.join(format!("{session}.{extension}")).to_string_lossy().into_owned();
    let practice = !s.practice.is_empty();
    let valued = [
        ("--save", s.save.clone()),
        ("--pipeline-cache", s.cache.clone()),
        ("--language", s.language.clone()),
        ("--sdl", s.sdl.clone()),
        ("--hooks", s.hooks.clone()),
        ("--romfs-layer", s.layer.clone()),
        ("--practice", s.practice.clone()),
        ("--boost", if practice { s.boost.clone() } else { String::new() }),
        ("--log", file("log")),
        ("--record", if s.record { file("txt") } else { String::new() }),
    ];
    let switches = [("--fullscreen", s.fullscreen), ("--joycon", s.joycon), ("--hud", s.hud && practice)];
    let mut line = vec!["boot".to_string(), s.exefs.clone(), "--play".into(), "--explore".into()];
    line.extend(valued.into_iter().filter(|v| !v.1.is_empty()).flat_map(|(name, value)| [name.to_string(), value]));
    line.extend(switches.into_iter().filter(|v| v.1).map(|v| v.0.to_string()));
    line
}

/// A setup step of the port: its program, its arguments and the folder it runs in.
/// The game's `exefs` folder is recompiled, then built; the build never fetches a
/// crate (`--offline`).
// ponytail: 4 build jobs, for 16 GB of RAM; a setting when another machine needs another.
pub fn step(s: &Settings, kind: &str) -> Result<(PathBuf, Vec<String>, PathBuf), String> {
    let need = |what: &str, value: &str| if value.is_empty() { Err(format!("{what} is not set")) } else { Ok(value.to_string()) };
    let repo = need("the repository", &s.repo)?;
    // Joined part by part: the player reads these paths, with one kind of separator.
    let (tools, generated) = (PathBuf::from_iter([repo.as_str(), "target", "release", "gyroship-tools.exe"]), PathBuf::from_iter([repo.as_str(), "generated", "game"]));
    Ok(match kind {
        "recompile" => {
            let mut line = vec!["recompile".to_string(), need("the exefs folder", &s.exefs)?, "--out".into(), generated.to_string_lossy().into_owned()];
            if !s.probes.is_empty() {
                line.extend(["--probes".to_string(), s.probes.clone()]);
            }
            (tools, line, repo.into())
        }
        "build" => ("cargo".into(), ["build", "--release", "--offline", "-j", "4"].map(String::from).to_vec(), generated),
        _ => return Err(format!("{kind} is no setup step")),
    })
}

/// The game's executables to choose from: the generated build, then those kept
/// under the data folder's `bin`. Full paths only: with a folder not set yet, nothing
/// is looked for under wherever the launcher happens to run.
pub fn builds(s: &Settings) -> Vec<String> {
    let mut kept: Vec<PathBuf> = fs::read_dir(Path::new(&s.data).join("bin")).into_iter().flatten().flatten().map(|e| e.path().join("harness.exe")).collect();
    kept.sort();
    let generated = PathBuf::from_iter([s.repo.as_str(), "generated", "game", "target", "release", "harness.exe"]);
    std::iter::once(generated).chain(kept).filter(|p| p.is_absolute() && p.is_file()).map(|p| p.to_string_lossy().into_owned()).collect()
}

/// The code recorded when a game starts, until its end replaces it.
pub const RUNNING: i32 = i32::MIN;

/// What the game's exit code means.
pub fn outcome(code: i32) -> &'static str {
    match code {
        RUNNING => "its end was not recorded: the launcher stopped before the game did",
        0 => "the game ended normally",
        2 => "a bad command line, or a start that could not go on: see the messages",
        3 => "this GPU cannot run the game: see the messages",
        71 => "a fault in the recompiled game",
        72 => "the emulation stopped the run (an unknown system call, an anomaly, the time limit or a deadlock)",
        _ => "an unexpected end",
    }
}

/// How the last session ended; `last.json` in the journals' folder.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct Last {
    pub session: String,
    pub code: i32,
    pub seconds: u64,
}

/// The last session, for the player: its end, the summary line of its journal and the
/// last of the game's messages.
#[derive(Debug, Default, PartialEq, Serialize)]
pub struct Report {
    pub session: String,
    pub code: i32,
    pub outcome: &'static str,
    pub seconds: u64,
    pub summary: String,
    pub messages: String,
    pub folder: String,
}

/// The end of a text file: its last `bytes` bytes, from a line's start.
fn tail(file: &Path, bytes: u64) -> String {
    let Ok(mut f) = fs::File::open(file) else { return String::new() };
    let length = f.metadata().map_or(0, |m| m.len());
    let mut text = Vec::new();
    if f.seek(SeekFrom::Start(length.saturating_sub(bytes))).is_err() || f.read_to_end(&mut text).is_err() {
        return String::new();
    }
    let text = String::from_utf8_lossy(&text);
    if length > bytes { text.split_once('\n').map_or("", |t| t.1).to_string() } else { text.into_owned() }
}

pub fn report(logs: &Path) -> Option<Report> {
    let last: Last = serde_json::from_str(&fs::read_to_string(logs.join("last.json")).ok()?).ok()?;
    let journal = tail(&logs.join(format!("{}.log", last.session)), 1 << 16);
    let summary = journal.lines().rev().find_map(|l| l.split_once(" summary: ")).map_or(String::new(), |l| l.1.to_string());
    let messages = tail(&logs.join(format!("{}.err.txt", last.session)), 4096);
    Some(Report { outcome: outcome(last.code), summary, messages, folder: logs.to_string_lossy().into_owned(), session: last.session, code: last.code, seconds: last.seconds })
}
