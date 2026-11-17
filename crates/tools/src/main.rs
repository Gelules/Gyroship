//! `gyroship-tools`: developer command line for Gyroship.

use clap::{Parser, Subcommand};
use gyroship_loader::Module;
use std::path::{Path, PathBuf};
use std::process::ExitCode;
use std::time::Instant;

mod boost;
mod experiment;
mod gpfifo;
mod measure;
mod recompile;
mod shaders;
mod text_sheet;

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
    /// Capture one frame of the first window whose title contains the given
    /// text (Windows Graphics Capture; works without focus) and save it as PNG
    Capture {
        /// Part of the window title, for example "Gyroship"
        #[arg(long)]
        title: String,
        /// Output PNG file
        #[arg(long, short)]
        out: PathBuf,
    },
    /// Load every module of an exefs folder at its guest base, apply the
    /// relocations, resolve imports, and report (exit code 1 if any import
    /// stays unresolved)
    Link {
        /// exefs folder (main, rtld, sdk, subsdk0...)
        exefs: PathBuf,
        /// Also print every unresolved import
        #[arg(long)]
        verbose: bool,
    },
    /// Discover every function of a linked exefs and build their control-flow
    /// graphs, with jump tables; prints per-module counts
    Discover {
        /// exefs folder (main, rtld, sdk, subsdk0...)
        exefs: PathBuf,
        /// Print up to this many examples of each unusual case
        #[arg(long, default_value_t = 0)]
        examples: usize,
    },
    /// Recompile every function of an exefs with Cranelift: one object per
    /// thread plus a harness crate that links them with the runtime
    Recompile {
        /// exefs folder
        exefs: PathBuf,
        /// Output folder (under generated/, which Git ignores)
        #[arg(long, short)]
        out: PathBuf,
        /// Compiler threads, one object file each
        #[arg(long, default_value_t = 6)]
        threads: usize,
        /// Run the Cranelift verifier on every function
        #[arg(long)]
        verify: bool,
        /// Probe list: one `<hex address> <name>` per line ('#' starts a comment)
        #[arg(long)]
        probes: Option<PathBuf>,
    },
    /// Backend experiment: generate every N-th function of main (or chosen
    /// functions) as a project ready to build and measure
    Experiment(ExperimentArgs),
    /// Run a command; report its wall time and the peak memory of its whole
    /// process tree
    Measure {
        program: String,
        #[arg(trailing_var_arg = true, allow_hyphen_values = true)]
        args: Vec<String>,
    },
    /// Function ranges from a module's unwind tables (.eh_frame_hdr), optionally
    /// compared with a function list exported from Ghidra
    Functions {
        /// NSO module; its file name (main, sdk...) gives its guest base
        nso: PathBuf,
        /// Ghidra list to compare with: tab-separated address, size, name
        #[arg(long)]
        compare: Option<PathBuf>,
    },
    /// Replay a --dump-gpfifo folder through the GPU command processor
    /// (macros included) and report each submit
    Gpfifo {
        /// Folder of submit-SSSS-RRRR-GPU.bin files
        dir: PathBuf,
        /// First submit reported (earlier ones still run, silently)
        #[arg(long, default_value_t = 0)]
        first: u32,
        /// Last submit run
        #[arg(long, default_value_t = u32::MAX)]
        last: u32,
        /// Print every executed method
        #[arg(long)]
        methods: bool,
        /// After the replay, list this macro's code, decoded
        #[arg(long = "macro")]
        listing: Option<usize>,
        /// Time the reported submits without tracing them, and give the macros' figures
        #[arg(long)]
        bench: bool,
        /// With --bench: sample the reported submits, report to this file
        #[arg(long)]
        profile: Option<PathBuf>,
        /// With --bench: run the reported submits this many times
        #[arg(long, default_value_t = 1)]
        repeat: u32,
    },
    /// Inventory of the game's shaders: every nvn_gen2.raw blob of data.zip,
    /// its program and its instructions by opcode and stage (exit code 1 if a
    /// program has no closing self-branch)
    Shaders {
        /// The romfs data.zip
        zip: PathBuf,
        /// Write real instruction words, one file per opcode, into this folder
        #[arg(long)]
        samples: Option<PathBuf>,
    },
    /// Search every entry of data.zip archives for byte patterns: text, or hex
    /// with a 0x prefix (bytes in file order); print each match's entry, offset
    /// and surrounding bytes, and copy the matching entries to a folder (game
    /// data: keep it out of the repository)
    DatazipFind {
        /// The romfs data.zip files, searched in order
        #[arg(long = "zip", required = true)]
        zips: Vec<PathBuf>,
        /// Patterns: text, or 0x followed by hex bytes
        #[arg(required = true)]
        patterns: Vec<String>,
        /// Copy the entries that match into this folder
        #[arg(long)]
        out: Option<PathBuf>,
        /// Bytes shown around each match
        #[arg(long, default_value_t = 48)]
        context: usize,
    },
    /// Draw every text Gyroship adds in game, in the 14 regions, with the
    /// fonts practice.txt's font lines name in data.zip (exit code 1 if a font
    /// cannot draw a character; the sheet is game output: keep it out of the
    /// repository)
    TextSheet {
        /// The romfs data.zip
        #[arg(long)]
        zip: PathBuf,
        /// The practice file with the font lines
        #[arg(long)]
        practice: PathBuf,
        /// The PNG to write
        #[arg(long, short)]
        out: PathBuf,
    },
    /// List a dumped graphics program (--dump-shaders): its SPH and each
    /// decoded instruction
    Sass {
        /// A program file: SPH then SASS
        file: PathBuf,
    },
    /// Translate dumped graphics programs to WGSL and validate them (exit
    /// code 1 if one fails)
    Translate {
        /// Program files: SPH then SASS
        #[arg(required = true)]
        files: Vec<PathBuf>,
        /// Write each module as {key}.wgsl into this folder
        #[arg(long)]
        out: Option<PathBuf>,
    },
    /// Check the boost reimplementation (gyroship-research) against --trace
    /// files of the boost probes: every frame of every vehicle, bit for bit
    /// (exit code 1 on any difference)
    BoostCheck {
        /// Trace files written with --trace
        #[arg(required = true)]
        traces: Vec<PathBuf>,
        /// The --boost practice action the runs had on: the model applies its writes too
        #[arg(long, value_enum)]
        apply: Option<boost::Apply>,
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
        Command::BoostCheck { traces, apply } => boost::check(&traces, apply),
        Command::Gpfifo { dir, first, last, methods, listing, bench, profile, repeat } => gpfifo::run(&dir, first, last, methods, listing, bench.then_some((profile, repeat))).map(|_| true),
        Command::Shaders { zip, samples } => shaders::run(&zip, samples.as_deref()),
        Command::DatazipFind { zips, patterns, out, context } => shaders::find(&zips, &patterns, out.as_deref(), context).map(|_| true),
        Command::TextSheet { zip, practice, out } => text_sheet::run(&zip, &practice, &out),
        Command::Sass { file } => shaders::list(&file).map(|_| true),
        Command::Translate { files, out } => shaders::translate(&files, out.as_deref()),
        Command::Link { exefs, verbose } => link(&exefs, verbose),
        Command::Discover { exefs, examples } => discover(&exefs, examples).map(|_| true),
        Command::Functions { nso, compare } => functions(&nso, compare.as_deref()).map(|_| true),
        Command::Capture { title, out } => capture::window(&title, &out).map(|_| true),
        Command::Experiment(args) => run_experiment(&args).map(|_| true),
        Command::Recompile { exefs, out, threads, verify, probes } => recompile::run(&exefs, &out, threads, verify, probes.as_deref()).map(|_| true),
        Command::Measure { program, args } => measure::run(&program, &args).map(|u| {
            let gib = |b: u64| b as f64 / (1u64 << 30) as f64;
            println!("measure     {:.1} s, peak {:.2} GiB for the process tree, {:.2} GiB for one process", u.seconds, gib(u.peak_total), gib(u.peak_process));
            u.success
        }),
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
        let window = Window::from_contains_name(title)
            .map_err(|e| format!("no window whose title contains {title:?}: {e}"))?;
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

/// Loads and links every module of an exefs folder.
fn load(exefs: &Path) -> Result<gyroship_loader::link::Program> {
    use gyroship_loader::link::{LOAD_ORDER, Program};
    let mut modules = Vec::new();
    for name in LOAD_ORDER {
        let path = exefs.join(name);
        if path.is_file() {
            let m = Module::parse_nso(&std::fs::read(&path)?).map_err(|e| format!("{}: {e}", path.display()))?;
            modules.push((name.to_string(), m));
        }
    }
    if modules.is_empty() {
        return Err(format!("{}: no module here (main, rtld, sdk...): give the exefs folder of your dump", exefs.display()).into());
    }
    Ok(Program::link(modules)?)
}

fn link(exefs: &Path, verbose: bool) -> Result<bool> {
    let p = load(exefs)?;
    println!("{:<8} {:>14} {:>12} {:>9} {:>9} {:>9}", "module", "base", "size", "relr", "relative", "symbolic");
    for (m, s) in p.modules.iter().zip(&p.stats) {
        println!("{:<8} {:#14x} {:#12x} {:>9} {:>9} {:>9}", m.name, m.base, m.module.image.len(), s.relr, s.relative, s.symbolic);
    }
    let unresolved: Vec<_> = p.unresolved().collect();
    let tls = unresolved.iter().filter(|i| i.kind == gyroship_loader::link::R_AARCH64_TLS_TPREL).count();
    println!("imports  {} by name, {} unresolved ({tls} TLS, left to the runtime)", p.imports.len(), unresolved.len());
    for i in unresolved.iter().filter(|_| verbose) {
        println!("  {} {:#x} type {} {}", p.modules[i.module].name, i.slot, i.kind, i.name);
    }
    Ok(unresolved.len() == tls)
}

fn discover(exefs: &Path, examples: usize) -> Result<()> {
    use gyroship_analysis::{cfg, discover, insn};
    let p = load(exefs)?;
    let start = Instant::now();
    let d = discover::discover(&p)?;
    println!("discovery   {:.2} s: {} functions, {} PLT stubs", start.elapsed().as_secs_f64(), d.functions.len(), d.stubs.len());
    println!(
        "entries     {} landing pads, {} code pointers into function bodies, {} references into gaps that are not code",
        d.counts.landing_pads, d.counts.inner_pointers, d.counts.dropped
    );
    let show = |what: &str, list: &[u64]| {
        if examples > 0 && !list.is_empty() {
            let shown: Vec<String> = list.iter().take(examples).map(|a| format!("{a:#x}")).collect();
            println!("    {what}: {}", shown.join(" "));
        }
    };
    let row = |c: [&str; 12]| {
        println!("{:<8} {:>8} {:>6} {:>6} {:>6} {:>7} {:>6} {:>8} {:>10} {:>9} {:>8} {:>5}", c[0], c[1], c[2], c[3], c[4], c[5], c[6], c[7], c[8], c[9], c[10], c[11]);
    };
    row(["module", "fde", "asm", "stubs", "tables", "cases", "capped", "indirect", "reached", "unreached", "nop/trap", "undec"]);
    let start = Instant::now();
    for m in &p.modules {
        let fs: Vec<&discover::Function> = d.functions.iter().filter(|f| (m.base..m.end()).contains(&f.start)).collect();
        let stubs = d.stubs.iter().filter(|s| (m.base..m.end()).contains(&s.addr)).count();
        let (mut tables, mut cases, mut reached, mut padding) = (0, 0, 0, 0);
        let (mut capped, mut indirect, mut unreached, mut undecodable): (Vec<u64>, Vec<u64>, Vec<u64>, Vec<u64>) = Default::default();
        for f in &fs {
            let g = cfg::build(&m.module.image, m.base, f);
            tables += g.tables.len();
            cases += g.tables.iter().map(|t| t.targets.len()).sum::<usize>();
            capped.extend(g.tables.iter().filter(|t| t.capped).map(|t| t.branch));
            indirect.extend(&g.indirect);
            reached += g.instructions();
            undecodable.extend(&g.undecodable);
            if !f.unwind {
                continue;
            }
            // Words of an FDE that no path reaches: padding and traps, or dead code.
            let mut at = f.start;
            for b in g.blocks.iter().chain([&cfg::Block { start: f.end(), len: 0 }]) {
                for a in (at..b.start).step_by(4) {
                    let w = p.read(a, 4).map_or(0, |b| u32::from_le_bytes(b.try_into().expect("4 bytes")));
                    if w == 0xD503_201F || insn::flow(w, a) == insn::Flow::Trap {
                        padding += 1;
                    } else {
                        unreached.push(a);
                    }
                }
                at = b.start + b.len;
            }
        }
        let asm = fs.iter().filter(|f| !f.unwind).count();
        let n = |v: usize| v.to_string();
        let cells = [n(fs.len() - asm), n(asm), n(stubs), n(tables), n(cases), n(capped.len()), n(indirect.len()), reached.to_string(), n(unreached.len()), n(padding), n(undecodable.len())];
        let c: Vec<&str> = std::iter::once(m.name.as_str()).chain(cells.iter().map(String::as_str)).collect();
        row(c.try_into().expect("12 columns"));
        show("capped tables", &capped);
        show("unreached words in FDE functions", &unreached);
        show("undecodable", &undecodable);
    }
    println!("cfg         {:.2} s", start.elapsed().as_secs_f64());
    Ok(())
}

#[derive(clap::Args)]
struct ExperimentArgs {
    /// exefs folder
    exefs: PathBuf,
    /// Output folder (under generated/, which Git ignores)
    #[arg(long, short)]
    out: PathBuf,
    /// Take every N-th FDE function of main
    #[arg(long, default_value_t = 124)]
    every: usize,
    /// Generate the functions at these guest addresses instead (hex, comma-separated)
    #[arg(long, value_delimiter = ',')]
    functions: Vec<String>,
    /// With --functions, also every function they call or jump to, transitively
    #[arg(long)]
    callees: bool,
    /// Functions per generated file
    #[arg(long, default_value_t = 256)]
    per_file: usize,
    /// Files per generated crate
    #[arg(long, default_value_t = 4)]
    files_per_crate: usize,
    /// Release opt-level of the generated project
    #[arg(long, default_value_t = 1)]
    opt_level: u8,
    /// rust (source crates) or clif (one Cranelift object)
    #[arg(long, default_value = "rust")]
    backend: String,
}

fn run_experiment(a: &ExperimentArgs) -> Result<()> {
    let (out, backend) = (a.out.as_path(), a.backend.as_str());
    let p = load(&a.exefs)?;
    let d = gyroship_analysis::discover::discover(&p)?;
    let start = Instant::now();
    let s = if a.functions.is_empty() {
        experiment::sample(&p, &d, a.every)?
    } else {
        let addrs = a.functions.iter().map(|x| u64::from_str_radix(x.trim_start_matches("0x"), 16)).collect::<std::result::Result<Vec<_>, _>>()?;
        experiment::chosen(&p, &d, &addrs, a.callees)?
    };
    let (total, unhandled, top) = experiment::coverage(&s);
    println!("sample      {} functions, {total} instructions, {unhandled} untranslated ({:.2}%)", s.functions.len(), 100.0 * unhandled as f64 / total as f64);
    let top: Vec<String> = top.iter().map(|(n, c)| format!("{n} {c}")).collect();
    println!("untranslated: {}", top.join(", "));
    let mib = |b: u64| b as f64 / (1u64 << 20) as f64;
    match backend {
        "rust" => {
            let bytes = experiment::write_rust(&s, out, a.per_file, a.files_per_crate, a.opt_level)?;
            println!("generated   {:.1} MiB of Rust in {:.2} s -> {}", mib(bytes), start.elapsed().as_secs_f64(), out.display());
        }
        "clif" => {
            let (bytes, seconds) = experiment::write_clif(&s, out, false)?;
            println!("generated   {:.1} MiB object, Cranelift {seconds:.2} s (one thread) -> {}", mib(bytes), out.display());
        }
        other => return Err(format!("unknown backend {other:?}").into()),
    }
    Ok(())
}

fn functions(path: &Path, compare: Option<&Path>) -> Result<()> {
    use std::collections::BTreeMap;
    let m = Module::parse_nso(&std::fs::read(path)?)?;
    let name = path.file_name().ok_or("input has no file name")?.to_string_lossy();
    let base = gyroship_loader::link::base_of(&name).ok_or_else(|| format!("no guest base for module {name:?}"))?;
    let ranges = gyroship_analysis::eh_frame::function_ranges(&m.image, m.mod0.eh_frame_hdr_start)?;
    let covered: u64 = ranges.iter().map(|r| r.len).sum();
    let outside = ranges.iter().filter(|r| r.start + r.len > m.text.end()).count();
    println!("eh_frame     {} functions, {covered:#x} bytes = {:.1}% of text ({outside} outside text)", ranges.len(), 100.0 * covered as f64 / m.text.size as f64);
    let start = Instant::now();
    let (mut words, mut refused) = (0u64, Vec::new());
    for r in &ranges {
        for at in (r.start..r.start + r.len).step_by(4) {
            let word = u32::from_le_bytes(m.image[at as usize..at as usize + 4].try_into()?);
            words += 1;
            if gyroship_analysis::decode::decode(word).is_none() {
                refused.push((at, word));
            }
        }
    }
    let sample: Vec<String> = refused.iter().take(8).map(|(a, w)| format!("{name}+{a:#x}={w:08x}")).collect();
    println!("decode       {words} words in FDE ranges, {} refused, {:.2} s  {}", refused.len(), start.elapsed().as_secs_f64(), sample.join(" "));
    let Some(compare) = compare else { return Ok(()) };

    let mut ghidra = BTreeMap::new();
    for line in std::fs::read_to_string(compare)?.lines() {
        let mut f = line.split('\t');
        let (Some(addr), Some(size)) = (f.next(), f.next()) else { continue };
        let addr = u64::from_str_radix(addr, 16)?.wrapping_sub(base);
        if addr < m.text.end() {
            ghidra.insert(addr, size.parse::<u64>()?);
        }
    }
    let fde: BTreeMap<u64, u64> = ranges.iter().map(|r| (r.start, r.len)).collect();
    let both = fde.keys().filter(|a| ghidra.contains_key(a)).count();
    let same_len = fde.iter().filter(|(a, l)| ghidra.get(a) == Some(l)).count();
    let ghidra_only: Vec<_> = ghidra.iter().filter(|(a, _)| !fde.contains_key(a)).collect();
    let fde_only: Vec<_> = fde.iter().filter(|(a, _)| !ghidra.contains_key(a)).collect();
    let inside = ghidra_only.iter().filter(|(a, _)| fde.range(..=**a).next_back().is_some_and(|(s, l)| **a < s + l)).count();
    println!("ghidra       {} functions in text", ghidra.len());
    println!("same start   {both} ({same_len} also with the same length)");
    println!("ghidra only  {} ({inside} of them inside an FDE range)", ghidra_only.len());
    println!("eh_frame only {}", fde_only.len());
    let show = |label: &str, list: &[(&u64, &u64)]| {
        let sample: Vec<String> = list.iter().take(8).map(|(a, l)| format!("{name}+{a:#x}/{l}")).collect();
        println!("  {label}: {}", sample.join(" "));
    };
    show("ghidra only, first", &ghidra_only);
    show("eh_frame only, first", &fde_only);
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

