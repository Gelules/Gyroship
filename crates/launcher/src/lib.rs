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
    /// "" (the save folder itself), "copy" (a copy of it at each start) or "new" (an
    /// empty save at each start): see `prepare_save`.
    pub save_mode: String,
    pub cache: String,
    /// A console language code, or nothing for the game's default.
    pub language: String,
    pub fullscreen: bool,
    pub joycon: bool,
    /// The Joy-Con's mice too, for the game's mouse mode (which brings the pair itself).
    pub joycon_mouse: bool,
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
    match serde_json::from_slice::<Settings>(&text) {
        Ok(mut s) => {
            // The generated build was harness.exe until 2026-11-18: the same build is now
            // gyroship.exe, and a new port leaves the old file behind, never updated again.
            let old = PathBuf::from_iter([s.repo.as_str(), "generated", "game", "target", "release", "harness.exe"]);
            if !s.repo.is_empty() && Path::new(&s.build) == old {
                s.build = old.with_file_name("gyroship.exe").to_string_lossy().into_owned();
            }
            (s, None)
        }
        Err(e) => {
            let bad = PathBuf::from(format!("{}.bad", file.display()));
            let kept = if fs::copy(file, &bad).is_ok() { format!("kept as {}", bad.display()) } else { "not kept aside".into() };
            (Settings::default(), Some(format!("{} is invalid ({e}): {kept}, the defaults are used", file.display())))
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
    let switches = [("--fullscreen", s.fullscreen), ("--joycon", s.joycon), ("--joycon-mouse", s.joycon_mouse), ("--hud", s.hud && practice)];
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
    // Full paths only: the tools run in the repository, where a relative one would mean another place.
    let need = |what: &str, value: &str| match value {
        "" => Err(format!("{what} is not set")),
        v if !Path::new(v).is_absolute() => Err(format!("{what} is not a full path: {v}")),
        v => Ok(v.to_string()),
    };
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

/// The game's executables to choose from: the generated build, those kept under the
/// data folder's `bin` (`gyroship.exe`, or `harness.exe` from before its renaming),
/// then the chosen one, wherever it is. Existing files only, so a chosen build that
/// is gone is not among them. Full paths only: with a folder not set yet, nothing is
/// looked for under wherever the launcher happens to run.
pub fn builds(s: &Settings) -> Vec<String> {
    let mut kept: Vec<PathBuf> = fs::read_dir(Path::new(&s.data).join("bin")).into_iter().flatten().flatten().flat_map(|e| ["gyroship.exe", "harness.exe"].map(|n| e.path().join(n))).collect();
    kept.sort();
    let generated = PathBuf::from_iter([s.repo.as_str(), "generated", "game", "target", "release", "gyroship.exe"]);
    let mut found: Vec<String> = Vec::new();
    for p in std::iter::once(generated).chain(kept).chain([PathBuf::from(&s.build)]).filter(|p| p.is_absolute() && p.is_file()) {
        let p = p.to_string_lossy().into_owned();
        if !found.contains(&p) {
            found.push(p);
        }
    }
    found
}

/// The replacement-file layers to choose from: each folder `layer…` of the data folder's
/// `practice`, by path, with the first line of its `about.txt` to show (else its name).
/// Full paths only, as for the builds.
pub fn layers(s: &Settings) -> Vec<(String, String)> {
    let mut found: Vec<(String, String)> = fs::read_dir(Path::new(&s.data).join("practice"))
        .into_iter()
        .flatten()
        .flatten()
        .filter(|e| e.file_name().to_string_lossy().starts_with("layer") && e.path().is_absolute() && e.path().is_dir())
        .map(|e| {
            let about = fs::read_to_string(e.path().join("about.txt")).ok().and_then(|t| Some(t.lines().next()?.trim().to_string())).filter(|l| !l.is_empty());
            (e.path().to_string_lossy().into_owned(), about.unwrap_or_else(|| e.file_name().to_string_lossy().into_owned()))
        })
        .collect();
    found.sort();
    found
}

/// The save folder the game gets. With `save_mode` "copy" or "new", a folder of the data
/// folder's own, made again at each start: `save-copy`, a copy of the save folder, or
/// `save-new`, empty. The save folder itself is then never written.
pub fn prepare_save(s: &Settings) -> Result<String, String> {
    let (name, copy) = match s.save_mode.as_str() {
        "" => return Ok(s.save.clone()),
        "copy" => ("save-copy", true),
        "new" => ("save-new", false),
        other => return Err(format!("{other} is no save mode")),
    };
    if copy && !Path::new(&s.save).is_dir() {
        return Err(if s.save.is_empty() { "the save folder to copy is not set".into() } else { format!("the save folder to copy does not exist: {}", s.save) });
    }
    let to = Path::new(&s.data).join(name);
    // The folder about to be emptied must neither hold the save folder nor lie in it, however
    // either is written: as typed, and where it really is (a mapped drive, a junction, \\?\).
    if !s.save.is_empty() {
        let lower = |p: PathBuf| PathBuf::from(p.to_string_lossy().to_lowercase());
        let typed = |p: &Path| std::path::absolute(p).map(lower).map_err(|e| format!("{}: {e}", p.display()));
        let mut pairs = vec![(typed(Path::new(&s.save))?, typed(&to)?)];
        if let (Ok(save), Ok(data)) = (fs::canonicalize(&s.save), fs::canonicalize(&s.data)) {
            pairs.push((lower(save), lower(data.join(name))));
        }
        if pairs.iter().any(|(save, fresh)| save.starts_with(fresh) || fresh.starts_with(save)) {
            return Err(format!("the save folder {} overlaps the launcher's {name} folder: choose another save folder", s.save));
        }
    }
    match fs::remove_dir_all(&to) {
        Err(e) if e.kind() != std::io::ErrorKind::NotFound => return Err(format!("{}: {e}", to.display())),
        _ => {}
    }
    fs::create_dir_all(&to).map_err(|e| format!("{}: {e}", to.display()))?;
    if copy {
        copy_folder(Path::new(&s.save), &to).map_err(|e| format!("copying {} to {}: {e}", s.save, to.display()))?;
    }
    Ok(to.to_string_lossy().into_owned())
}

fn copy_folder(from: &Path, to: &Path) -> std::io::Result<()> {
    fs::create_dir_all(to)?;
    for e in fs::read_dir(from)? {
        let e = e?;
        let target = to.join(e.file_name());
        if e.file_type()?.is_dir() {
            copy_folder(&e.path(), &target)?;
        } else {
            fs::copy(e.path(), target)?;
        }
    }
    Ok(())
}

/// The code recorded when a game starts, until its end replaces it.
pub const RUNNING: i32 = i32::MIN;

/// What the game's exit code means, shown after "Ended".
pub fn outcome(code: i32) -> &'static str {
    match code {
        RUNNING => "not recorded (the launcher stopped first)",
        0 => "normally",
        2 => "bad command line or failed start: see the messages",
        3 => "this GPU cannot run the game: see the messages",
        71 => "a fault in the recompiled game",
        72 => "the emulation stopped the game: see the messages",
        _ => "unexpectedly",
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
    /// False when the launcher stopped before the game: no exit code, no duration.
    pub ended: bool,
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
    Some(Report { ended: last.code != RUNNING, outcome: outcome(last.code), summary, messages, folder: logs.to_string_lossy().into_owned(), session: last.session, code: last.code, seconds: last.seconds })
}
