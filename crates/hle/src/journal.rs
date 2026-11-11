//! The journal of a run (D21): one line per event, stamped with the virtual
//! time and the guest thread, never with wall-clock time or a host address,
//! so that two runs compare byte for byte. The play engine stamps real time
//! instead: its journal is a log, never a reference (D38).

use std::collections::BTreeSet;
use std::io::Write;
use std::sync::{Arc, Mutex, MutexGuard, PoisonError};

enum Sink {
    Memory(Vec<u8>),
    File(std::io::BufWriter<std::fs::File>),
}

struct Inner {
    sink: Sink,
    seen: BTreeSet<String>,
    counts: std::collections::BTreeMap<String, u32>,
}

/// A handle on a journal; [`Journal::share`] makes another one on the same
/// lines (the GPU side's), each call taking the journal's own lock, which no
/// other lock is ever taken under.
pub struct Journal {
    inner: Arc<Mutex<Inner>>,
    /// Every SVC, instead of the first of each kind.
    pub all_svcs: bool,
}

impl Journal {
    fn with(sink: Sink) -> Journal {
        Journal { inner: Arc::new(Mutex::new(Inner { sink, seen: BTreeSet::new(), counts: Default::default() })), all_svcs: false }
    }

    /// A journal kept in memory (tests).
    pub fn memory() -> Journal {
        Journal::with(Sink::Memory(Vec::new()))
    }

    pub fn file(path: &std::path::Path) -> std::io::Result<Journal> {
        Ok(Journal::with(Sink::File(std::io::BufWriter::new(std::fs::File::create(path)?))))
    }

    /// Another handle on the same journal.
    pub fn share(&self) -> Journal {
        Journal { inner: self.inner.clone(), all_svcs: self.all_svcs }
    }

    fn inner(&self) -> MutexGuard<'_, Inner> {
        self.inner.lock().unwrap_or_else(PoisonError::into_inner)
    }

    /// Logs one line at virtual time `time` for guest thread `thread`.
    pub fn line(&self, time: u64, thread: u64, text: &str) {
        let line = format!("{time:>12} t{thread} {text}\n");
        match &mut self.inner().sink {
            Sink::Memory(v) => v.extend_from_slice(line.as_bytes()),
            Sink::File(f) => {
                let _ = f.write_all(line.as_bytes());
            }
        }
    }

    /// Whether `key` is new; it is then remembered.
    pub fn first(&self, key: &str) -> bool {
        let mut i = self.inner();
        !i.seen.contains(key) && i.seen.insert(key.to_string())
    }

    /// Whether `key` has been seen fewer than `n` times; counts this one.
    pub fn few(&self, key: &str, n: u32) -> bool {
        let mut i = self.inner();
        let c = i.counts.entry(key.to_string()).or_insert(0);
        *c += 1;
        *c <= n
    }

    /// What a memory journal holds so far.
    pub fn text(&self) -> String {
        match &self.inner().sink {
            Sink::Memory(v) => String::from_utf8_lossy(v).into_owned(),
            Sink::File(_) => String::new(),
        }
    }

    pub fn flush(&self) {
        if let Sink::File(f) = &mut self.inner().sink {
            let _ = f.flush();
        }
    }
}
