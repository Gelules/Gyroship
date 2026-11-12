//! `gpfifo`: replays a `--dump-gpfifo` folder (`submit-SSSS-RRRR-GPU.bin`,
//! one file per GPFIFO entry, in submit then entry order) through the
//! command processor, macros included, and reports what each submit did.

use gyroship_gpu::pushbuf::{Channel, Effect};
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

/// `bench`: time without tracing (Some), the reported submits `repeat` times,
/// sampling into the file if one is given.
pub fn run(dir: &Path, first: u32, last: u32, methods: bool, listing: Option<usize>, bench: Option<(Option<PathBuf>, u32)>) -> super::Result<()> {
    let (mut profile, repeat, bench) = (bench.clone().and_then(|b| b.0), bench.as_ref().map_or(1, |b| b.1.max(1)), bench.is_some());
    // The reported submits' words, for the bench's later rounds.
    let mut kept = Vec::new();
    let mut files: Vec<(u32, u32, PathBuf)> = Vec::new();
    for entry in std::fs::read_dir(dir)? {
        let path = entry?.path();
        let name = path.file_name().map(|n| n.to_string_lossy().into_owned()).unwrap_or_default();
        let mut parts = name.trim_end_matches(".bin").split('-');
        if let (Some("submit"), Some(s), Some(r)) = (parts.next(), parts.next(), parts.next())
            && let (Ok(s), Ok(r)) = (s.parse(), r.parse())
        {
            files.push((s, r, path));
        }
    }
    files.sort();
    if files.is_empty() {
        return Err(format!("{}: no submit-SSSS-RRRR-*.bin files", dir.display()).into());
    }
    let mut channel = Channel::default();
    // A bench times the command processor alone: no trace.
    channel.trace = (!bench).then(Vec::new);
    let mut current = None;
    let mut stats = Stats::default();
    // Over the reported submits: a digest of their methods (to compare two builds),
    // the time they took, and the macro figures before them.
    let (mut digest, mut took, mut before) = (0u64, std::time::Duration::ZERO, None);
    // Earlier submits still run (they load the macros), silently.
    for (submit, _, path) in files.into_iter().take_while(|f| f.0 <= last) {
        let raw = std::fs::read(&path)?;
        let words: Vec<u32> = raw.as_chunks::<4>().0.iter().map(|w| u32::from_le_bytes(*w)).collect();
        if submit >= first && before.is_none() {
            before = channel.mme.as_ref().map(|u| u.stats);
            if let Some(path) = profile.take() {
                gyroship_runtime::profile::start(path);
            }
        }
        let since = std::time::Instant::now();
        let effects = channel.run(&words);
        if submit >= first {
            took += since.elapsed();
        }
        if bench {
            if submit >= first && repeat > 1 {
                kept.push(words);
            }
            continue;
        }
        let trace = channel.trace.replace(Vec::new()).unwrap_or_default();
        if submit < first {
            continue;
        }
        if current != Some(submit) {
            stats.print(current);
            stats = Stats::default();
            current = Some(submit);
        }
        stats.words += words.len();
        for t in &trace {
            let words = [t.subchannel as u64, u64::from(t.class), u64::from(t.method), u64::from(t.value), t.from_macro.map_or(u64::MAX, |m| m as u64)];
            digest = words.iter().fold(digest, |h, &w| (h.rotate_left(5) ^ w).wrapping_mul(0x517c_c1b7_2722_0a95));
            stats.methods += 1;
            if let Some(m) = t.from_macro {
                *stats.by_macro.entry(m).or_default() += 1;
            }
            if methods {
                let source = t.from_macro.map_or(String::new(), |m| format!(" (macro {m})"));
                println!("  s{} {:#06x} {:#06x} = {:#010x}{source}", t.subchannel, t.class, t.method, t.value);
            }
        }
        for e in effects {
            match e {
                Effect::Release(r) => println!("  release {} at {:#x} by method {:#x}", r.value, r.addr, r.method),
                Effect::MacroFault { index, fault } => println!("  macro {index} fault {fault:?}"),
                Effect::MacroRead { addr, words } => println!("  macro read {words} words at {addr:#x}"),
                Effect::Clear(c) => println!("  clear {c:?}"),
                Effect::Draw(d) => println!("  draw {d:x?}"),
                Effect::Write(w) => stats.written += w.data.len(),
                Effect::Unsupported(u) => println!("  not applied: {u:x?}"),
                Effect::Copy(c) => println!("  copy {c:x?}"),
                Effect::Dispatch(q, action, _) => println!("  compute {action:?}, QMD at {q:#x}"),
                Effect::Semaphore(s) => println!("  semaphore {s:x?}"),
                Effect::Ignored { class, method, value } => *stats.ignored.entry((class, method, value)).or_default() += 1,
            }
        }
    }
    if bench {
        for _ in 1..repeat {
            for words in &kept {
                let since = std::time::Instant::now();
                channel.run(words);
                took += since.elapsed();
            }
        }
        gyroship_runtime::profile::finish(&|_| String::new());
        let (a, b) = (before.unwrap_or_default(), channel.mme.as_ref().map(|u| u.stats).unwrap_or_default());
        let (calls, groups) = (b.calls - a.calls, b.groups - a.groups);
        let ms = took.as_secs_f64() * 1e3;
        println!(
            "submits {first}.., {repeat} rounds: {:.1} ms a round, {} macro calls a round of {:.1} groups on average, {:.0} ns of command processing per call",
            ms / f64::from(repeat),
            calls / u64::from(repeat),
            groups as f64 / calls.max(1) as f64,
            ms * 1e6 / calls.max(1) as f64
        );
        return Ok(());
    }
    stats.print(current);
    println!("methods from submit {first} on: digest {digest:016x}");
    if let (Some(index), Some(unit)) = (listing, &channel.mme) {
        println!("macro {index}:");
        for line in unit.listing(index) {
            println!("  {line}");
        }
    }
    Ok(())
}

#[derive(Default)]
struct Stats {
    words: usize,
    methods: usize,
    /// Bytes the 3D stream wrote to memory.
    written: usize,
    by_macro: BTreeMap<usize, usize>,
    ignored: BTreeMap<(u32, u32, u32), usize>,
}

impl Stats {
    fn print(&self, submit: Option<u32>) {
        let Some(submit) = submit else { return };
        let from_macros: usize = self.by_macro.values().sum();
        let macros: Vec<String> = self.by_macro.iter().map(|(m, n)| format!("{m}:{n}")).collect();
        println!("submit {submit}: {} words, {} methods, {from_macros} of them from macros [{}], {} bytes written", self.words, self.methods, macros.join(" "), self.written);
        for ((class, method, value), n) in &self.ignored {
            println!("  not executed: class {class:#x} method {method:#x} = {value:#x} x{n}");
        }
    }
}
