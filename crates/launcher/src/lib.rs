//! What Gyroship's launcher decides, apart from its window: the player's settings and
//! their file, the game's command line, the setup steps' command lines, its updates, and
//! what a session's end means. The game stays offline. The launcher reaches the network
//! through other programs only: at start, unless the player turned it off, Windows' curl
//! asks GitHub for a newer release and fetches it (D109); on the player's click, the setup
//! steps start winget and Cargo, which download (D108).

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
    /// The game's debug menu and Gyroship's practice tools. `debug_menu`, with no hooks file:
    /// the repository's `templates/hooks.txt`, which unlocks the game's debug menu.
    pub hooks: String,
    pub debug_menu: bool,
    pub layer: String,
    pub practice: String,
    /// "", "perfect" or "infinite".
    pub boost: String,
    /// The overlays: the speed, the frame rate, the boost release markers.
    pub hud_speed: bool,
    pub hud_fps: bool,
    pub hud_markers: bool,
    /// Before the three, one switch for them all: read from an older file, never written.
    #[serde(skip_serializing)]
    pub hud: bool,
    /// Records the session's inputs next to its journal.
    pub record: bool,
    /// The game in a console that shows its messages, instead of a file of them.
    pub console: bool,
    /// No look for a new release at start (D109): the check is on unless turned off.
    pub no_update_check: bool,
    /// The game draws through Vulkan itself, with the shaders the Shaders step precompiled into
    /// the cache folder (`--renderer vulkan`, D110).
    pub vulkan: bool,
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
            // One overlay switch until 2026-11-21: the three overlays it drew.
            if std::mem::take(&mut s.hud) {
                (s.hud_speed, s.hud_fps, s.hud_markers) = (true, true, true);
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
/// skipped, then the player's options. `--boost` and the overlays need the practice file.
pub fn flags(s: &Settings, logs: &Path, session: &str) -> Vec<String> {
    let file = |extension: &str| logs.join(format!("{session}.{extension}")).to_string_lossy().into_owned();
    let practice = !s.practice.is_empty();
    // A hooks file of the player's own wins; else the debug menu's, from the repository.
    let hooks = match (s.hooks.as_str(), s.debug_menu && !s.repo.is_empty()) {
        ("", true) => PathBuf::from_iter([s.repo.as_str(), "templates", "hooks.txt"]).to_string_lossy().into_owned(),
        (own, _) => own.to_string(),
    };
    let valued = [
        ("--save", s.save.clone()),
        ("--pipeline-cache", s.cache.clone()),
        ("--language", s.language.clone()),
        ("--sdl", s.sdl.clone()),
        ("--hooks", hooks),
        ("--romfs-layer", s.layer.clone()),
        ("--practice", s.practice.clone()),
        ("--boost", if practice { s.boost.clone() } else { String::new() }),
        ("--renderer", if s.vulkan { "vulkan".into() } else { String::new() }),
        ("--log", file("log")),
        ("--record", if s.record { file("txt") } else { String::new() }),
    ];
    let switches = [
        ("--fullscreen", s.fullscreen),
        ("--joycon", s.joycon),
        ("--joycon-mouse", s.joycon_mouse),
        ("--hud-speed", s.hud_speed && practice),
        ("--hud-fps", s.hud_fps && practice),
        ("--hud-markers", s.hud_markers && practice),
    ];
    let mut line = vec!["boot".to_string(), s.exefs.clone(), "--play".into(), "--explore".into()];
    line.extend(valued.into_iter().filter(|v| !v.1.is_empty()).flat_map(|(name, value)| [name.to_string(), value]));
    line.extend(switches.into_iter().filter(|v| v.1).map(|v| v.0.to_string()));
    line
}

/// The winget packages the launcher offers to install, on the player's demand only (D108):
/// Rust through rustup (its stable MSVC toolchain, for the user alone, no administrator rights),
/// Microsoft's C++ build tools with the x64 compiler and the Windows SDK (an administrator
/// prompt; `--wait`, as `--override` replaces the package's own), the Visual C++ runtime. winget
/// downloads them; the launcher itself opens no connection.
pub const INSTALLS: [(&str, &str, &[&str]); 3] = [
    ("install-rust", "Rustlang.Rustup", &[]),
    (
        "install-cpp",
        "Microsoft.VisualStudio.BuildTools",
        &["--override", "--passive --wait --norestart --add Microsoft.VisualStudio.Workload.VCTools --add Microsoft.VisualStudio.Component.VC.Tools.x86.x64 --add Microsoft.VisualStudio.Component.Windows11SDK.26100"],
    ),
    ("install-runtime", "Microsoft.VCRedist.2015+.x64", &[]),
];

/// A setup step: its program, its arguments and the folder it runs in. An install through
/// winget; the tools' build; the game's recompile, then its build (the first builds download
/// the crates they need); its shaders precompiled into the cache folder (D110).
// ponytail: 4 build jobs, for 16 GB of RAM; a setting when another machine needs another.
pub fn step(s: &Settings, kind: &str) -> Result<(PathBuf, Vec<String>, PathBuf), String> {
    if let Some((_, id, extra)) = INSTALLS.iter().find(|i| i.0 == kind) {
        // The winget source only (not the Store's twins); no prompt: the page has no keyboard
        // for winget, and the player's click is the agreement.
        let mut line: Vec<String> = ["install", "--id", id, "--exact", "--source", "winget", "--accept-package-agreements", "--accept-source-agreements", "--disable-interactivity"].map(String::from).to_vec();
        line.extend(extra.iter().map(|w| w.to_string()));
        return Ok(("winget".into(), line, std::env::temp_dir()));
    }
    // Full paths only: the tools run in the repository, where a relative one would mean another place.
    let need = |what: &str, value: &str| match value {
        "" => Err(format!("{what} is not set")),
        v if !Path::new(v).is_absolute() => Err(format!("{what} is not a full path: {v}")),
        v => Ok(v.to_string()),
    };
    let repo = need("the repository", &s.repo)?;
    // Joined part by part: the player reads these paths, with one kind of separator.
    let (tools, generated) = (tools_exe(&repo), PathBuf::from_iter([repo.as_str(), "generated", "game"]));
    let build = ["build", "--release", "-j", "4"].map(String::from).to_vec();
    Ok(match kind {
        "tools" => (cargo_exe(), build, repo.into()),
        "recompile" => {
            let mut line = vec!["recompile".to_string(), need("the exefs folder", &s.exefs)?, "--out".into(), generated.to_string_lossy().into_owned()];
            if !s.probes.is_empty() {
                line.extend(["--probes".to_string(), s.probes.clone()]);
            }
            (tools, line, repo.into())
        }
        "build" => (cargo_exe(), build, generated),
        "shaders" => {
            let line = vec!["precompile".to_string(), need("the exefs folder", &s.exefs)?, "--pipeline-cache".into(), need("the shader cache folder", &s.cache)?];
            (game_exe(s, &repo), line, repo.into())
        }
        _ => return Err(format!("{kind} is no setup step")),
    })
}

/// The game the Shaders step precompiles for and Play plays: the build chosen, else the one the
/// Build step makes.
pub fn game_exe(s: &Settings, repo: &str) -> PathBuf {
    match s.build.as_str() {
        "" => PathBuf::from_iter([repo, "generated", "game", "target", "release", "gyroship.exe"]),
        build => build.into(),
    }
}

/// The tools of the repository `repo`: those its tools step built (`target/release`), else
/// those a release brings at its top, else those downloaded next to this launcher (run in the
/// repository, they take it as theirs), else where the tools step will put them.
pub fn tools_exe(repo: &str) -> PathBuf {
    tools_in(repo, std::env::current_exe().ok().as_deref().and_then(Path::parent))
}

/// [`tools_exe`] for a launcher in the folder `launcher`.
pub fn tools_in(repo: &str, launcher: Option<&Path>) -> PathBuf {
    let built = PathBuf::from_iter([repo, "target", "release", "gyroship-tools.exe"]);
    let (brought, beside) = (Path::new(repo).join("gyroship-tools.exe"), launcher.map(|l| l.join("gyroship-tools.exe")));
    [Some(&built), Some(&brought), beside.as_ref()].into_iter().flatten().find(|p| p.is_file()).unwrap_or(&built).clone()
}

/// Cargo's program: CARGO_HOME's, then the user's `.cargo` (where rustup puts it, before a new
/// session's PATH holds it), else `cargo` from the PATH.
pub fn cargo_exe() -> PathBuf {
    let homes = [std::env::var_os("CARGO_HOME").map(PathBuf::from), std::env::var_os("USERPROFILE").map(|h| Path::new(&h).join(".cargo"))];
    homes.into_iter().flatten().map(|h| h.join("bin").join("cargo.exe")).find(|c| c.is_file()).unwrap_or_else(|| "cargo".into())
}

/// The oldest Rust that builds Gyroship (Cranelift 0.137 asks for it).
pub const RUST_OLDEST: (u32, u32) = (1, 97);

/// What building and running the game needs on this PC, found without the network.
#[derive(Debug, Default, PartialEq, Serialize)]
pub struct Needs {
    /// winget, which installs the rest on demand.
    pub winget: bool,
    /// Rust's version, from Cargo ("1.99.1"), and whether it is recent enough.
    pub rust: Option<String>,
    pub rust_ok: bool,
    /// Microsoft's C++ build tools, which link the game.
    pub cpp: bool,
    /// The Visual C++ runtime, which the game needs to run.
    pub runtime: bool,
}

/// The version of a `cargo --version` line: "cargo 1.99.1 (58cf8f9c3 2026-10-13)".
pub fn version(line: &str) -> Option<(u32, u32, u32)> {
    let mut parts = line.split_whitespace().nth(1)?.split(['.', '-']).map(|p| p.parse::<u32>().ok());
    Some((parts.next()??, parts.next()??, parts.next()??))
}

/// A program's standard output, run without a window, or what went wrong: its error output.
/// A check never downloads: rustup would otherwise install a missing toolchain (rule 9, D108).
fn run(program: &Path, args: &[&str]) -> Result<String, String> {
    let mut command = std::process::Command::new(program);
    command.args(args).stdin(std::process::Stdio::null()).env("RUSTUP_AUTO_INSTALL", "0");
    #[cfg(windows)]
    std::os::windows::process::CommandExt::creation_flags(&mut command, 0x0800_0000);
    let out = command.output().map_err(|e| format!("{}: {e}", program.display()))?;
    match String::from_utf8_lossy(&out.stderr).trim() {
        _ if out.status.success() => Ok(String::from_utf8_lossy(&out.stdout).into_owned()),
        "" => Err(format!("{} ended with {}", program.display(), out.status)),
        said => Err(said.to_string()),
    }
}

fn output(program: &Path, args: &[&str]) -> Option<String> {
    run(program, args).ok()
}

pub fn needs() -> Needs {
    let rust = output(&cargo_exe(), &["--version"]).and_then(|l| version(&l));
    let vswhere = std::env::var_os("ProgramFiles(x86)").map(|p| Path::new(&p).join("Microsoft Visual Studio").join("Installer").join("vswhere.exe"));
    let cpp = vswhere.and_then(|w| output(&w, &["-products", "*", "-requires", "Microsoft.VisualStudio.Component.VC.Tools.x86.x64", "-property", "installationPath"]));
    Needs {
        winget: output(Path::new("winget"), &["--version"]).is_some(),
        rust_ok: rust.is_some_and(|(major, minor, _)| (major, minor) >= RUST_OLDEST),
        rust: rust.map(|(a, b, c)| format!("{a}.{b}.{c}")),
        cpp: cpp.is_some_and(|o| !o.trim().is_empty()),
        runtime: std::env::var_os("SystemRoot").is_some_and(|r| Path::new(&r).join("System32").join("vcruntime140.dll").is_file()),
    }
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
/// `practice`, by path, with the first line of its `about.txt` to show (else its name),
/// in the order of what they show, so that alike labels sit together. Full paths only, as
/// for the builds.
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
    found.sort_by_key(|(path, about)| (about.to_lowercase(), path.clone()));
    found
}

/// What the page tells the player before a game: the Setup tab's problems (its folders, or
/// no build to play yet), the dump's, and how far the port went. Full paths only.
#[derive(Debug, Default, PartialEq, Serialize)]
pub struct Checks {
    pub setup: Vec<String>,
    /// Both Setup folders are full paths that exist.
    pub folders: bool,
    pub exefs: Option<String>,
    /// The dump's build ID, read from its main module, and whether it is the one the
    /// repository's templates are made for: None when the templates' build cannot be read
    /// (the repository not set, or no templates/practice.txt).
    pub dump: Option<String>,
    pub dump_known: Option<bool>,
    /// The tools built, the game recompiled (its project there), the game built, its shaders
    /// precompiled (since that game was built).
    pub tools: bool,
    pub recompiled: bool,
    pub built: bool,
    pub shaders: bool,
}

/// Where new versions are published, the repository on GitHub (its Releases), and this
/// launcher's version, which a release's tag is compared with (D109).
pub const REPOSITORY: &str = env!("CARGO_PKG_REPOSITORY");
pub const VERSION: &str = env!("CARGO_PKG_VERSION");
/// The launcher's file, at the top of a release's folder.
pub const LAUNCHER: &str = "gyroship-launcher.exe";

/// Whether a release's tag ("v0.2.0", "0.2.0-beta") names a version after `current` ("0.1.0"):
/// major, minor, patch, as numbers.
pub fn is_newer(tag: &str, current: &str) -> bool {
    let parse = |v: &str| -> Option<Vec<u64>> { v.trim().trim_start_matches(['v', 'V']).split(['.', '-', '+']).take(3).map(|p| p.parse().ok()).collect() };
    matches!((parse(tag), parse(current)), (Some(a), Some(b)) if a.len() == 3 && b.len() == 3 && a > b)
}

/// Whether `url` is a page of the repository's releases, the only pages the launcher opens: no
/// `.` or `..` segment, which a browser would resolve to another page.
pub fn is_release_page(url: &str) -> bool {
    url.strip_prefix(REPOSITORY).and_then(|rest| rest.strip_prefix("/releases/")).is_some_and(|rest| {
        !rest.is_empty() && rest.chars().all(|c| c.is_ascii_alphanumeric() || "/.-_".contains(c)) && !rest.split('/').any(|s| s == "." || s == "..")
    })
}

/// Whether a tag can name a folder: letters, digits, dots and dashes.
fn folder_tag(tag: &str) -> bool {
    tag.chars().all(|c| c.is_ascii_alphanumeric() || ".-".contains(c))
}

/// A release newer than this launcher: its tag ("v0.2.0"), its page, and its Windows archive
/// (`gyroship-<version>-windows-x64.zip`, docs/building.md) with its size in bytes.
#[derive(Clone, Debug, Default, PartialEq, Serialize)]
pub struct Release {
    pub tag: String,
    pub page: String,
    pub zip: String,
    pub size: u64,
}

/// What the page learns at start: this launcher's version, a newer release if there is one,
/// and why this copy of Gyroship cannot install it itself, if it cannot.
#[derive(Debug, Default, PartialEq, Serialize)]
pub struct Update {
    pub version: String,
    pub release: Option<Release>,
    pub blocker: Option<String>,
}

/// The release in GitHub's answer for the latest one, when it is newer than `current`. Only the
/// repository's own page and archive are taken, and only a tag that can name a folder.
pub fn release_from(json: &str, current: &str) -> Result<Option<Release>, String> {
    #[derive(Deserialize)]
    struct Asset {
        name: String,
        browser_download_url: String,
        size: u64,
    }
    #[derive(Deserialize)]
    struct Answer {
        tag_name: String,
        html_url: String,
        #[serde(default)]
        assets: Vec<Asset>,
    }
    let answer: Answer = serde_json::from_str(json).map_err(|e| format!("GitHub's answer: {e}"))?;
    let tag = answer.tag_name;
    if !is_newer(&tag, current) {
        return Ok(None);
    }
    if !folder_tag(&tag) || !is_release_page(&answer.html_url) {
        return Err(format!("release {tag}: not one of {REPOSITORY}'s"));
    }
    // This tag's archive, at the one address GitHub gives it: nothing else is downloaded.
    let name = format!("gyroship-{}-windows-x64.zip", tag.trim_start_matches(['v', 'V']));
    let url = format!("{REPOSITORY}/releases/download/{tag}/{name}");
    let zip = answer.assets.into_iter().find(|a| a.name == name && a.browser_download_url == url);
    let zip = zip.ok_or_else(|| format!("release {tag} has no {name}: get it from {}", answer.html_url))?;
    Ok(Some(Release { tag, page: answer.html_url, zip: url, size: zip.size }))
}

/// One of Windows' own programs (curl and tar, there since Windows 10 1803): the launcher
/// carries no HTTP client and no unzipper of its own.
fn windows_program(name: &str) -> PathBuf {
    let root = std::env::var_os("SystemRoot").unwrap_or_else(|| "C:\\Windows".into());
    Path::new(&root).join("System32").join(name)
}

/// curl, for GitHub: HTTPS only, redirects followed (archives are served from another of
/// GitHub's hosts), an HTTP error as a failure.
fn curl(args: &[&str]) -> Result<String, String> {
    // -q first, as curl wants it: no .curlrc of the player's ("insecure", "proxy"...) applies.
    let mut line = vec!["-q", "--fail", "--silent", "--show-error", "--location", "--proto", "=https", "--proto-redir", "=https", "--user-agent", "Gyroship-launcher"];
    line.extend(args);
    run(&windows_program("curl.exe"), &line)
}

/// The repository's latest release when it is newer than this launcher: one request to
/// GitHub's API. A private repository, or one with no release yet, answers 404: an error.
pub fn latest_newer() -> Result<Option<Release>, String> {
    let api = format!("https://api.github.com/repos/{}/releases/latest", REPOSITORY.trim_start_matches("https://github.com/"));
    let answer = curl(&["--max-time", "20", "--header", "Accept: application/vnd.github+json", &api])
        .map_err(|e| if e.contains("error: 404") { format!("no release is published at {REPOSITORY} yet, or the repository is private") } else { e })?;
    release_from(&answer, VERSION)
}

/// The Gyroship folder a launcher belongs to: the first one above it with Gyroship's sources
/// (crates/runtime), as for the tools. A release has its launcher at the top; a Git clone, in
/// crates/launcher/target/release.
pub fn install_root(exe: &Path) -> Option<PathBuf> {
    exe.ancestors().skip(1).find(|d| d.join("crates").join("runtime").join("Cargo.toml").is_file()).map(Path::to_path_buf)
}

/// Why the copy of Gyroship that launcher `exe` belongs to cannot install a release itself, if
/// it cannot: a Git clone updates through Git, and a launcher outside a Gyroship folder has
/// nothing to update.
pub fn update_blocker(exe: &Path) -> Option<String> {
    match install_root(exe) {
        None => Some("this launcher is not in a Gyroship folder: get the new version from its page".into()),
        Some(root) if root.join(".git").exists() => Some("this copy is a Git clone: update it with git pull, then build the tools and the game again".into()),
        Some(_) => None,
    }
}

/// `path`'s place inside `folder`, when it lies there as written (letter case aside, as Windows
/// has it).
fn within(path: &str, folder: &Path) -> Option<PathBuf> {
    let lower = |p: &Path| PathBuf::from(p.to_string_lossy().to_lowercase());
    (!path.is_empty() && lower(Path::new(path)).starts_with(lower(folder))).then(|| Path::new(path).components().skip(folder.components().count()).collect())
}

/// Whether `path` may lie in `folder`, however either is written: made whole (`..` resolved), and
/// where it really is (a subst or mapped drive, a junction, a short name, `\\?\`), as in
/// `prepare_save`. A path that is not a full one may mean any folder: it counts as inside.
fn inside(path: &str, folder: &Path) -> bool {
    let lower = |p: PathBuf| PathBuf::from(p.to_string_lossy().to_lowercase());
    let forms = |p: &Path| [std::path::absolute(p).ok(), fs::canonicalize(p).ok()].map(|f| f.map(lower));
    let (paths, folders) = (forms(Path::new(path)), forms(folder));
    !Path::new(path).is_absolute() || paths.iter().zip(&folders).any(|pair| matches!(pair, (Some(p), Some(f)) if p.starts_with(f)))
}

/// The settings for the copy of Gyroship in `new`, from those of the copy in `old`, whose files
/// are now in `unpacked` (`new` itself, or the folder about to become it). The repository and
/// the game built there move to their place in `new`; so do the templates chosen from it and an
/// SDL3.dll next to the game, when the new copy has them (a file of the player's own there stays
/// where it is, and so keeps the old copy from being deleted). Whatever else the player chose
/// stays too.
pub fn rebase(s: &Settings, old: &Path, new: &Path, unpacked: &Path) -> Settings {
    let moved = |value: &String, from: &Path, checked: bool| match (within(value, from), within(value, old)) {
        (Some(_), Some(rest)) if rest.as_os_str().is_empty() => new.to_string_lossy().into_owned(),
        (Some(_), Some(rest)) if !checked || unpacked.join(&rest).exists() => new.join(rest).to_string_lossy().into_owned(),
        _ => value.clone(),
    };
    let templates = old.join("templates");
    Settings {
        repo: moved(&s.repo, old, false),
        build: moved(&s.build, old, false),
        sdl: moved(&s.sdl, &game_folder(old), true),
        practice: moved(&s.practice, &templates, true),
        probes: moved(&s.probes, &templates, true),
        hooks: moved(&s.hooks, &templates, true),
        ..s.clone()
    }
}

/// The game's build folder in a copy of Gyroship.
fn game_folder(root: &Path) -> PathBuf {
    root.join("generated").join("game").join("target").join("release")
}

/// Where release `tag` goes next to the copy of Gyroship in `root`, `Gyroship-<version>`, when it
/// can go there: asked before anything is downloaded.
pub fn target(root: &Path, tag: &str) -> Result<PathBuf, String> {
    if !folder_tag(tag) {
        return Err(format!("{tag}: no version a folder can be named after"));
    }
    let parent = root.parent().ok_or_else(|| format!("{}: no folder above it", root.display()))?;
    let to = parent.join(format!("Gyroship-{}", tag.trim_start_matches(['v', 'V'])));
    let lower = |p: &Path| PathBuf::from(p.to_string_lossy().to_lowercase());
    if lower(&to) == lower(root) {
        return Err(format!("this copy is the release {tag}, but its launcher says {VERSION}: the release was made without setting its version"));
    }
    if to.exists() {
        return Err(format!("{} already exists: Gyroship {tag} is installed there; start its launcher (and point your shortcuts to it), or move that folder away", to.display()));
    }
    Ok(to)
}

/// The mark of the folder an update unpacks into: one left by an update that stopped is the
/// launcher's to remove, and a folder of that name without it is not.
const MARK: &str = ".gyroship-update";

/// Installs a release's archive `zip` (tag `tag`) next to the copy of Gyroship in `root`, as
/// `Gyroship-<version>`: unpacked into `Gyroship-<version>.part`, given its settings (`rebase`)
/// and the SDL3.dll put next to the game, then renamed. The rename is the one step that makes the
/// new copy: a failure before it leaves the current copy as it was, and at most a marked `.part`
/// that the next update removes. Returns the new folder.
pub fn install(zip: &Path, tag: &str, root: &Path, s: &Settings) -> Result<PathBuf, String> {
    let fail = |what: &Path, e: std::io::Error| format!("{}: {e}", what.display());
    let to = target(root, tag)?;
    let part = PathBuf::from(format!("{}.part", to.display()));
    if part.join(MARK).is_file() {
        fs::remove_dir_all(&part).map_err(|e| fail(&part, e))?;
    }
    fs::create_dir(&part).map_err(|e| fail(&part, e))?;
    fs::write(part.join(MARK), "").map_err(|e| fail(&part, e))?;
    let unpacked = run(&windows_program("tar.exe"), &["-x", "-f", &zip.to_string_lossy(), "-C", &part.to_string_lossy()]).and_then(|_| {
        // The archive's top: the folder itself, or its one folder.
        let top = if part.join(LAUNCHER).is_file() { Some(part.clone()) } else { fs::read_dir(&part).into_iter().flatten().flatten().map(|e| e.path()).find(|p| p.join(LAUNCHER).is_file()) };
        top.filter(|t| t.join("crates").join("runtime").join("Cargo.toml").is_file()).ok_or_else(|| format!("{}: no {LAUNCHER} beside Gyroship's sources in it", zip.display()))
    });
    let top = match unpacked {
        Ok(top) => top,
        Err(e) => {
            let _ = fs::remove_dir_all(&part);
            return Err(e);
        }
    };
    // The game resolves these against the launcher's current folder, which the new launcher does not
    // share (it runs in its copy): made whole here, they keep their meaning. (The tools look for the
    // probe list from the repository, and SDL3 goes through Windows' search for DLLs.)
    let whole = |p: &String| if p.is_empty() || Path::new(p).is_absolute() { p.clone() } else { std::path::absolute(p).map_or_else(|_| p.clone(), |a| a.to_string_lossy().into_owned()) };
    let s = Settings { save: whole(&s.save), cache: whole(&s.cache), hooks: whole(&s.hooks), layer: whole(&s.layer), practice: whole(&s.practice), ..s.clone() };
    let sdl = game_folder(root).join("SDL3.dll");
    let ready = (|| -> Result<(), String> {
        if sdl.is_file() {
            fs::create_dir_all(game_folder(&top)).and_then(|()| fs::copy(&sdl, game_folder(&top).join("SDL3.dll"))).map_err(|e| fail(&sdl, e))?;
        }
        save(&top.join("launcher.json"), &rebase(&s, root, &to, &top))
    })();
    // An antivirus scanning the new executables can hold the folder for a moment (5: access denied).
    let mut tries = 0;
    let renamed = ready.and_then(|()| loop {
        match fs::rename(&top, &to) {
            Ok(()) => break Ok(()),
            Err(e) if e.raw_os_error() == Some(5) && tries < 4 => tries += 1,
            Err(e) => break Err(fail(&to, e)),
        }
        std::thread::sleep(std::time::Duration::from_secs(1));
    });
    if let Err(e) = renamed {
        let _ = fs::remove_dir_all(&part);
        return Err(e);
    }
    // Leftovers only, now: the mark (in the new copy when the archive had no top folder), and the
    // `.part` folder, still marked, which a later update removes.
    let _ = fs::remove_file(to.join(MARK));
    if top != part {
        let _ = fs::remove_dir_all(&part);
    }
    Ok(to)
}

/// Fetches a release's archive into `to`, and checks it has the size GitHub gave. Bounded: 30 s to
/// connect, 30 minutes in all, and a transfer slower than 1 KB/s for a minute stops.
// ponytail: integrity rests on HTTPS, that size and the archive's own checksums (tar checks
// them); a signature check once releases are signed. 30 minutes are about 30 KB/s for a 50 MB
// archive: a limit from `r.size` if players need slower links.
fn download(r: &Release, to: &Path) -> Result<(), String> {
    let output = to.to_string_lossy();
    curl(&["--retry", "2", "--connect-timeout", "30", "--max-time", "1800", "--retry-max-time", "1800", "--speed-limit", "1024", "--speed-time", "60", "--output", &output, &r.zip])?;
    match fs::metadata(to).map(|m| m.len()) {
        Ok(got) if got == r.size => Ok(()),
        got => Err(format!("{}: {} bytes received, {} announced", r.zip, got.unwrap_or(0), r.size)),
    }
}

/// Fetches the latest release, when it is newer, and installs it next to the copy of Gyroship
/// that launcher `exe` belongs to; `say` tells the player how far it went. Asked again here,
/// not taken from the page: only GitHub's answer says what is downloaded. Returns the new
/// copy's folder and the current one's.
pub fn update(s: &Settings, exe: &Path, say: impl Fn(String)) -> Result<(PathBuf, PathBuf), String> {
    if let Some(why) = update_blocker(exe) {
        return Err(why);
    }
    let root = install_root(exe).ok_or("no Gyroship folder")?;
    let r = latest_newer()?.ok_or("this is the latest version")?;
    // Before the download: nothing is fetched for a folder that is taken.
    target(&root, &r.tag)?;
    say(format!("Downloading Gyroship {} ({} MB)…", r.tag, r.size.div_ceil(1 << 20)));
    let zip = std::env::temp_dir().join(format!("gyroship-{}-windows-x64.zip", r.tag));
    let installed = download(&r, &zip).and_then(|()| {
        say(format!("Installing Gyroship {} next to this copy…", r.tag));
        install(&zip, &r.tag, &root, s)
    });
    let _ = fs::remove_file(&zip);
    Ok((installed?, root))
}

/// Deletes the copy of Gyroship an update came from, `old`: only a release's folder (no Git
/// clone) next to this copy (`root`), once this copy's game is built, when the old copy's
/// launcher has ended, and when it holds nothing the player chose.
pub fn delete_previous(s: &Settings, old: &Path, root: &Path) -> Result<(), String> {
    // Component by component: a final separator or a `.` changes nothing.
    let lower = |p: &Path| PathBuf::from(p.to_string_lossy().to_lowercase());
    let copy = old.join(LAUNCHER).is_file() && old.join("crates").join("runtime").join("Cargo.toml").is_file() && !old.join(".git").exists();
    if !copy || old.parent().map(lower) != root.parent().map(lower) || lower(old) == lower(root) {
        return Err(format!("{} is no previous copy of Gyroship next to this one", old.display()));
    }
    let chosen = [("repository", &s.repo), ("data folder", &s.data), ("game build", &s.build), ("exefs folder", &s.exefs), ("save folder", &s.save), ("shader cache folder", &s.cache), ("SDL3 library", &s.sdl), ("hooks file", &s.hooks), ("mod", &s.layer), ("practice file", &s.practice), ("practice probes", &s.probes)];
    if let Some((what, path)) = chosen.iter().find(|(_, p)| !p.is_empty() && inside(p, old)) {
        return Err(format!("{} holds your {what} ({path}): move it out first", old.display()));
    }
    // The previous copy keeps a game to play until this one has built its own.
    if !game_folder(root).join("gyroship.exe").is_file() {
        return Err(format!("the game is not built in {} yet: build it (Setup), then delete the previous copy", root.display()));
    }
    // Its launcher holds launcher.lock while it runs (32: a sharing violation).
    #[cfg(windows)]
    {
        use std::os::windows::fs::OpenOptionsExt;
        if let Err(e) = fs::OpenOptions::new().read(true).share_mode(0).open(old.join("launcher.lock"))
            && e.raw_os_error() == Some(32)
        {
            return Err(format!("the launcher in {} is still running: close it first", old.display()));
        }
    }
    fs::remove_dir_all(old).map_err(|e| format!("{}: {e}", old.display()))
}

/// The build the repository's templates (practice file, probe list, hooks file) are made for: the
/// first 40-digit hex word of `templates/practice.txt`, its main module's build ID. Read there, so
/// that the full build ID lives in `templates/` only (D106).
pub fn templates_build(repo: &str) -> Option<String> {
    let text = fs::read_to_string(Path::new(repo).join("templates").join("practice.txt")).ok()?;
    text.split(|c: char| !c.is_ascii_hexdigit()).find(|w| w.len() == 40).map(str::to_lowercase)
}

/// The build ID of `exefs/main`: the 20 bytes at 0x40 of its NSO0 header, in hex.
pub fn dump_build(exefs: &Path) -> Option<String> {
    let mut header = [0u8; 0x54];
    fs::File::open(exefs.join("main")).ok()?.read_exact(&mut header).ok()?;
    (&header[..4] == b"NSO0").then(|| header[0x40..0x54].iter().map(|b| format!("{b:02x}")).collect())
}

/// A folder field's problem: not set, not a full path, or not there.
fn folder_problem(what: &str, value: &str) -> Option<String> {
    let path = Path::new(value);
    if value.is_empty() {
        Some(format!("{what}: not set"))
    } else if !path.is_absolute() {
        Some(format!("{what}: not a full path"))
    } else if !path.is_dir() {
        Some(format!("{what}: folder not found"))
    } else {
        None
    }
}

pub fn checks(s: &Settings) -> Checks {
    let repo = folder_problem("Repository", &s.repo);
    // When a step's output was last written: a step is done when its output is newer than what
    // it is made from (a recompile older than the tools or the probe list, or a build older than
    // the recompile, is to do again). None sorts first: a missing earlier step dims nothing.
    let time = |path: PathBuf| fs::metadata(path).and_then(|m| m.modified()).ok();
    let at = |parts: &[&str]| if repo.is_none() { time(PathBuf::from_iter(std::iter::once(s.repo.as_str()).chain(parts.iter().copied()))) } else { None };
    let tools = if repo.is_none() { time(tools_exe(&s.repo)) } else { None };
    let (project, game) = (at(&["generated", "game", "Cargo.toml"]), at(&["generated", "game", "target", "release", "gyroship.exe"]));
    let probes = if s.probes.is_empty() { None } else { time(PathBuf::from(&s.probes)) };
    // The probes compiled in: the recompile names each one in the harness, so a list chosen after the
    // last recompile, yet older than it, still shows.
    let harness = PathBuf::from_iter([s.repo.as_str(), "generated", "game", "harness", "src", "main.rs"]);
    let probed = s.probes.is_empty() || (repo.is_none() && fs::read_to_string(harness).is_ok_and(|h| h.contains("trace:")));
    let mut setup: Vec<String> = [repo.clone(), folder_problem("Data folder", &s.data)].into_iter().flatten().collect();
    let folders = setup.is_empty();
    if builds(s).is_empty() {
        setup.push("No game build yet: recompile the game, then build it".into());
    }
    let romfs = Path::new(&s.exefs).parent().map(|p| p.join("romfs"));
    let exefs = folder_problem("Your game's exefs folder", &s.exefs).or_else(|| (!romfs.is_some_and(|r| r.is_dir())).then(|| "Your game's exefs folder: no romfs folder next to it".to_string()));
    let dump = if exefs.is_none() { dump_build(Path::new(&s.exefs)) } else { None };
    // A folder with no game module (a romfs folder, the dump's root) is no exefs folder.
    let exefs = exefs.or_else(|| dump.is_none().then(|| "Your game's exefs folder: no main module in it (is it the exefs folder?)".to_string()));
    let templates = if repo.is_none() { templates_build(&s.repo) } else { None };
    // The precompiler's signature (one per shader build, which the game names): written at each of
    // its runs, so newer than the game it ran for once done.
    let precompiled = if s.cache.is_empty() { None } else { fs::read_dir(Path::new(&s.cache).join("vulkan")).into_iter().flatten().flatten().filter_map(|d| time(d.path().join("precompiled.txt"))).max() };
    let played = if repo.is_none() { time(game_exe(s, &s.repo)) } else { None };
    Checks {
        setup,
        folders,
        exefs,
        dump_known: dump.as_ref().zip(templates).map(|(d, t)| *d == t),
        dump,
        tools: tools.is_some(),
        recompiled: project.is_some() && project >= tools.max(probes) && probed,
        built: game.is_some() && game >= project,
        shaders: played.is_some() && precompiled >= played,
    }
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
