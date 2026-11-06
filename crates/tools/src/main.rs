//! `gyroship-tools`: developer command line for Gyroship.

use clap::{Parser, Subcommand};
use gyroship_loader::Module;
use std::path::{Path, PathBuf};
use std::process::ExitCode;

#[derive(Parser)]
#[command(name = "gyroship-tools", version, about = "Gyroship developer tools")]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    /// Inspect NSO executables (segments, MOD0, symbols, relocations)
    #[command(subcommand)]
    Nso(NsoCommand),
    /// Capture one frame of a window by its exact title (Windows Graphics
    /// Capture; works without focus) and save it as PNG
    Capture {
        /// Window title, for example "Gyroship"
        #[arg(long)]
        title: String,
        /// Output PNG file
        #[arg(long, short)]
        out: PathBuf,
    },
    /// Convert NSO executables to AArch64 ELF files for disassemblers
    Nso2elf {
        /// NSO files to convert
        #[arg(required = true)]
        inputs: Vec<PathBuf>,
        /// Output folder; each module becomes <name>.elf
        #[arg(long, short)]
        out: PathBuf,
    },
}

#[derive(Subcommand)]
enum NsoCommand {
    /// Summary of one module: segments, build ID, dynamic table, symbol and relocation counts
    Info {
        nso: PathBuf,
        /// Also print every symbol (address, size, kind, name)
        #[arg(long)]
        symbols: bool,
    },
}

fn main() -> ExitCode {
    let cli = Cli::parse();
    let result = match cli.command {
        Command::Nso(NsoCommand::Info { nso, symbols }) => nso_info(&nso, symbols).map(|_| true),
        Command::Nso2elf { inputs, out } => nso2elf(&inputs, &out).map(|_| true),
        Command::Capture { title, out } => capture::window(&title, &out).map(|_| true),
    };
    match result {
        Ok(true) => ExitCode::SUCCESS,
        Ok(false) => ExitCode::from(1),
        Err(e) => {
            eprintln!("error: {e}");
            ExitCode::from(2)
        }
    }
}

type Result<T> = std::result::Result<T, Box<dyn std::error::Error>>;

#[cfg(windows)]
mod capture {
    use std::path::{Path, PathBuf};
    use windows_capture::capture::{Context, GraphicsCaptureApiHandler};
    use windows_capture::encoder::ImageFormat;
    use windows_capture::frame::Frame;
    use windows_capture::graphics_capture_api::InternalCaptureControl;
    use windows_capture::settings::{
        ColorFormat, CursorCaptureSettings, DirtyRegionSettings, DrawBorderSettings, MinimumUpdateIntervalSettings,
        SecondaryWindowSettings, Settings,
    };
    use windows_capture::window::Window;

    type Error = Box<dyn std::error::Error + Send + Sync>;

    /// Saves the first frame that arrives, then stops.
    struct OneShot {
        out: PathBuf,
    }

    impl GraphicsCaptureApiHandler for OneShot {
        type Flags = PathBuf;
        type Error = Error;

        fn new(ctx: Context<Self::Flags>) -> Result<Self, Self::Error> {
            Ok(OneShot { out: ctx.flags })
        }

        fn on_frame_arrived(&mut self, frame: &mut Frame, control: InternalCaptureControl) -> Result<(), Self::Error> {
            frame.save_as_image(&self.out, ImageFormat::Png)?;
            control.stop();
            Ok(())
        }
    }

    pub fn window(title: &str, out: &Path) -> super::Result<()> {
        let window = Window::from_name(title).map_err(|e| format!("no window titled {title:?}: {e}"))?;
        let settings = Settings::new(
            window,
            CursorCaptureSettings::WithoutCursor,
            DrawBorderSettings::WithoutBorder,
            SecondaryWindowSettings::Default,
            MinimumUpdateIntervalSettings::Default,
            DirtyRegionSettings::Default,
            ColorFormat::Rgba8,
            out.to_path_buf(),
        );
        OneShot::start(settings).map_err(|e| format!("capture failed: {e}"))?;
        println!("captured {title:?} -> {}", out.display());
        Ok(())
    }
}

#[cfg(not(windows))]
mod capture {
    pub fn window(_title: &str, _out: &std::path::Path) -> super::Result<()> {
        Err("window capture uses Windows Graphics Capture and is only available on Windows".into())
    }
}

fn nso2elf(inputs: &[PathBuf], out: &Path) -> Result<()> {
    std::fs::create_dir_all(out)?;
    for input in inputs {
        let m = Module::parse_nso(&std::fs::read(input)?).map_err(|e| format!("{}: {e}", input.display()))?;
        let name = input.file_name().ok_or("input has no file name")?.to_string_lossy();
        let target = out.join(format!("{name}.elf"));
        std::fs::write(&target, m.to_elf())?;
        println!("{} -> {} ({} bytes in memory)", input.display(), target.display(), m.mem_end());
    }
    Ok(())
}

fn nso_info(path: &Path, symbols: bool) -> Result<()> {
    let m = Module::parse_nso(&std::fs::read(path)?)?;
    println!("module      {}", path.display());
    println!("build id    {}", m.build_id.iter().map(|b| format!("{b:02x}")).collect::<String>());
    println!("flags       {:#x} (compressed {:03b}, hashed {:03b})", m.flags, m.flags & 7, (m.flags >> 3) & 7);
    for (name, s) in [("text", m.text), ("ro", m.ro), ("data", m.data)] {
        println!("{name:<11} {:#010x}..{:#010x}  {} bytes", s.addr, s.end(), s.size);
    }
    println!("bss         {:#010x}..{:#010x}", m.mod0.bss_start, m.mod0.bss_end);
    println!("mod0        {:#x}, dynamic {:#x}, eh_frame_hdr {:#x}..{:#x}, module object {:#x}", m.mod0.addr, m.mod0.dynamic, m.mod0.eh_frame_hdr_start, m.mod0.eh_frame_hdr_end, m.mod0.module_object);
    println!("soname      {}", m.soname.as_deref().unwrap_or("-"));
    println!("needed      {}", if m.needed.is_empty() { "-".to_string() } else { m.needed.join(", ") });
    let tags: Vec<String> = m.dynamic.iter().map(|(t, _)| format!("{t:#x}")).collect();
    println!("dynamic     {} entries: {}", m.dynamic.len(), tags.join(" "));
    let defined = m.symbols.iter().filter(|s| s.is_defined()).count();
    let functions = m.symbols.iter().filter(|s| s.is_defined() && s.is_function()).count();
    println!("symbols     {} ({defined} defined, {functions} defined functions, {} imported)", m.symbols.len(), m.symbols.len() - defined);
    let mut kinds = std::collections::BTreeMap::new();
    for r in &m.relocations {
        *kinds.entry(r.kind).or_insert(0u64) += 1;
    }
    let kinds: Vec<String> = kinds.iter().map(|(k, n)| format!("type {k}: {n}")).collect();
    println!("relocations {} rela ({}), {} relr words", m.relocations.len(), kinds.join(", "), m.relr_words);
    if symbols {
        for s in &m.symbols {
            let kind = if !s.is_defined() { "import" } else if s.is_function() { "func" } else { "data" };
            println!("{:#012x} {:>8} {kind:<6} {}", s.value, s.size, s.name);
        }
    }
    Ok(())
}
