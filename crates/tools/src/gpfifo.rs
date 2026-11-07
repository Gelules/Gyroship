//! `gpfifo`: replays a `--dump-gpfifo` folder (`submit-SSSS-RRRR-GPU.bin`,
//! one file per GPFIFO entry, in submit then entry order) through the
//! command processor, macros included, and reports what each submit did.

use gyroship_gpu::pushbuf::{Channel, Effect};
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

pub fn run(dir: &Path, first: u32, last: u32, methods: bool) -> super::Result<()> {
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
    channel.trace = Some(Vec::new());
    let mut current = None;
    let mut stats = Stats::default();
    // Earlier submits still run (they load the macros), silently.
    for (submit, _, path) in files.into_iter().take_while(|f| f.0 <= last) {
        let raw = std::fs::read(&path)?;
        let words: Vec<u32> = raw.as_chunks::<4>().0.iter().map(|w| u32::from_le_bytes(*w)).collect();
        let effects = channel.run(&words);
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
                Effect::Clear(c) => println!("  clear {c:?}"),
                Effect::Draw(d) => println!("  draw {d:x?}"),
                Effect::Write(w) => stats.written += w.data.len(),
                Effect::Unsupported(u) => println!("  not applied: {u:x?}"),
                Effect::Copy(c) => println!("  copy {c:x?}"),
                Effect::Ignored { class, method, value } => *stats.ignored.entry((class, method, value)).or_default() += 1,
            }
        }
    }
    stats.print(current);
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
