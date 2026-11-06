//! The journal of a run (D21): one line per event, stamped with the virtual
//! time and the guest thread, never with wall-clock time or a host address,
//! so that two runs compare byte for byte.

use std::collections::BTreeSet;
use std::io::Write;
use std::sync::atomic::Ordering;

enum Sink {
    Memory(Vec<u8>),
    File(std::io::BufWriter<std::fs::File>),
}

pub struct Journal {
    sink: Sink,
    seen: BTreeSet<String>,
    /// Every SVC, instead of the first of each kind.
    pub all_svcs: bool,
}

impl Journal {
    /// A journal kept in memory (tests).
    pub fn memory() -> Journal {
        Journal { sink: Sink::Memory(Vec::new()), seen: BTreeSet::new(), all_svcs: false }
    }

    pub fn file(path: &std::path::Path) -> std::io::Result<Journal> {
        Ok(Journal { sink: Sink::File(std::io::BufWriter::new(std::fs::File::create(path)?)), seen: BTreeSet::new(), all_svcs: false })
    }

    /// Logs one line for guest thread `thread`.
    pub fn line(&mut self, thread: u64, text: &str) {
        let time = gyroship_runtime::COUNTER.load(Ordering::Relaxed);
        let line = format!("{time:>12} t{thread} {text}\n");
        match &mut self.sink {
            Sink::Memory(v) => v.extend_from_slice(line.as_bytes()),
            Sink::File(f) => {
                let _ = f.write_all(line.as_bytes());
            }
        }
    }

    /// Whether `key` is new; it is then remembered.
    pub fn first(&mut self, key: &str) -> bool {
        !self.seen.contains(key) && self.seen.insert(key.to_string())
    }

    /// What a memory journal holds so far.
    pub fn text(&self) -> &str {
        match &self.sink {
            Sink::Memory(v) => std::str::from_utf8(v).unwrap_or(""),
            Sink::File(_) => "",
        }
    }

    pub fn flush(&mut self) {
        if let Sink::File(f) = &mut self.sink {
            let _ = f.flush();
        }
    }
}
