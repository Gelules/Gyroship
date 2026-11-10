//! Guest threads and their scheduler (D19).
//!
//! Every guest thread runs on a Windows fiber, all of them on one host
//! thread. Six virtual cores each have a clock in ticks of 31.25 MHz. At each
//! switch point (an SVC, a counter read, `yield`/`wfe`), the running core's
//! clock advances by [`STEP`], and the scheduler runs the core whose next
//! thread can start earliest: a discrete-event simulation, so every run is
//! the same. Within a core the highest priority runs (lowest number), in
//! arrival order at equal priority; a core with nothing to run takes a ready
//! thread that waits behind another, if its affinity allows.

use crate::Kernel;
use crate::layout;
use gyroship_runtime::fiber::{self, Fiber};
use gyroship_runtime::mem::{Block, perm, state};
use gyroship_runtime::{self as rt, Cpu};
use std::ffi::c_void;

pub const CORES: usize = 6;
/// Ticks one switch point costs: 1.024 µs.
pub const STEP: u64 = 32;
/// How far the running core may get ahead of the one furthest behind before
/// the scheduler switches to it (32.8 µs): fewer fiber switches, same order
/// in every run.
pub const QUANTUM: u64 = 1024;
pub const MAIN_PRIORITY: u32 = 44;
/// Priorities the KAC grants.
pub const PRIORITIES: std::ops::RangeInclusive<u32> = 28..=59;
const TLR_SIZE: u64 = 0x200;
/// GetInfo 0x18 counts down from this.
pub const THREAD_LIMIT: u64 = 96;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum State {
    /// Created, not started.
    Created,
    /// Runnable, from `ready_at` on.
    Ready,
    Waiting,
    Done,
}

/// What a handle designates.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Object {
    Thread(usize),
    /// The two ends of an event (an index into `Kernel::events`).
    ReadableEvent(usize),
    WritableEvent(usize),
    /// A client session (an index into `Kernel::sessions`).
    Session(usize),
    /// An index into `Kernel::shared`.
    SharedMemory(usize),
    /// An index into `Kernel::transfers`.
    TransferMemory(usize),
}

/// What a waiting thread waits for.
#[derive(Clone, Debug, PartialEq, Eq, Default)]
pub enum Wait {
    #[default]
    None,
    Sleep,
    /// WaitSynchronization: the first of these objects to be signaled.
    Sync(Vec<Object>),
    /// ArbitrateLock: the mutex word at `addr`, to be handed over with `tag`.
    Lock { addr: u64, tag: u32 },
    /// WaitProcessWideKeyAtomic: a signal on `key`, then the mutex at `addr`.
    CondVar { key: u64, addr: u64, tag: u32 },
    /// WaitForAddress.
    Address(u64),
}

/// The process's handle table: 512 entries, numbered the same in every run.
pub struct Handles {
    slots: Vec<Option<(u32, Object)>>,
    serial: u32,
}

impl Handles {
    pub const MAX: usize = 512;

    pub fn new() -> Handles {
        Handles { slots: Vec::new(), serial: 0 }
    }

    /// A new handle: the lowest free index in bits 0..15, a serial above.
    pub fn add(&mut self, o: Object) -> Option<u32> {
        let i = self.slots.iter().position(Option::is_none).unwrap_or(self.slots.len());
        if i >= Self::MAX {
            return None;
        }
        self.serial = self.serial % 0x7FFF + 1;
        let h = self.serial << 15 | i as u32;
        if i == self.slots.len() {
            self.slots.push(None);
        }
        self.slots[i] = Some((h, o));
        Some(h)
    }

    pub fn get(&self, h: u32) -> Option<Object> {
        match self.slots.get((h & 0x7FFF) as usize) {
            Some(Some((x, o))) if *x == h => Some(*o),
            _ => None,
        }
    }

    pub fn close(&mut self, h: u32) -> Option<Object> {
        let o = self.get(h)?;
        self.slots[(h & 0x7FFF) as usize] = None;
        Some(o)
    }
}

impl Default for Handles {
    fn default() -> Handles {
        Handles::new()
    }
}

pub struct Thread {
    pub id: u64,
    pub handle: u32,
    /// Owned (from `Box::into_raw`): generated code and wakers write it
    /// through this pointer.
    pub cpu: *mut Cpu,
    pub tlr: u64,
    pub priority: u32,
    /// The ideal core, or -1 when the thread has no preference.
    pub ideal: i32,
    pub core: usize,
    pub affinity: u64,
    pub state: State,
    pub ready_at: u64,
    /// Arrival order among threads of equal priority.
    pub seq: u64,
    pub deadline: Option<u64>,
    /// Written to w0 when the deadline wakes the thread.
    pub on_timeout: u32,
    /// Ticks spent running, for `GetThreadTickCount` (TLR + 0x108).
    pub cpu_time: u64,
    /// Host time spent while it was the current thread, HLE work done on its
    /// behalf included (`--perf` reports it; host state, never the guest's).
    pub host: std::time::Duration,
    /// Guest instructions it ran (from the generated code's count; `--perf`).
    pub instructions: u64,
    /// With `--work-time`, the instructions not yet turned into a whole tick.
    pub work_rem: u64,
    pub fiber: Option<Fiber>,
    pub wait: Wait,
    /// Arrival order among waiters of equal priority.
    pub wait_seq: u64,
    /// CancelSynchronization came while the thread was not waiting.
    pub cancel: bool,
}

// SAFETY: `cpu` is a heap context owned by the thread table (freed by
// `Sched::drop`) and tied to no host thread. The host thread running this
// guest thread uses it (generated code, its own hooks); kernel code under the
// kernel lock uses it only while the guest thread is not running (created, or
// parked at a switch point: `wake`, `pick`'s deadlines). `fiber` is Send.
unsafe impl Send for Thread {}

/// How a run of the threads ended.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Outcome {
    /// Every thread ended.
    Finished,
    /// Threads wait and nothing can wake them.
    Deadlock(String),
}

/// A periodic virtual timer: its k-th deadline is `start + k × num / den`
/// ticks, computed from `k` each time so that rounding never accumulates.
#[derive(Clone, Copy, Debug)]
pub struct Timer {
    pub start: u64,
    pub k: u64,
    pub num: u64,
    pub den: u64,
    pub action: TimerAction,
}

impl Timer {
    pub fn at(&self) -> u64 {
        self.start + (u128::from(self.k) * u128::from(self.num) / u128::from(self.den)) as u64
    }
}

/// What a timer does at its deadline (plain data).
#[derive(Clone, Copy, Debug)]
pub enum TimerAction {
    /// The silent audio sink of an audsmx stream: the ring's read index
    /// (+0x44) takes the write index (+0x1004) if valid, then the stream's
    /// data event (`Kernel::events` index) is signaled.
    AudioSink { ring: u64, slots: u32, data: u64, channels: u32, event: usize },
    /// hid's controller sample, into shared memory `shared` (D29).
    HidSample { shared: usize },
}

/// The scheduler's state, inside the kernel.
#[derive(Default)]
pub struct Sched {
    pub threads: Vec<Thread>,
    pub current: Option<usize>,
    /// When the current thread's host time was last charged.
    pub host_mark: Option<std::time::Instant>,
    pub clocks: [u64; CORES],
    /// The thread that runs on each core, and since when.
    on_core: [Option<(usize, u64)>; CORES],
    seq: u64,
    next_id: u64,
    next_tlr: u64,
    pub boot: Option<Fiber>,
    /// Fibers of ended threads, deleted by the next fiber that runs.
    dead: Vec<Fiber>,
    pub outcome: Option<Outcome>,
}

impl Drop for Sched {
    fn drop(&mut self) {
        for t in &self.threads {
            // SAFETY: allocated by `Box::into_raw` in `create_thread`; no fiber runs any more.
            drop(unsafe { Box::from_raw(t.cpu) });
        }
    }
}

impl Kernel {
    /// The running core's clock: the guest's present.
    pub fn now(&self) -> u64 {
        self.sched.current.map_or(0, |t| self.sched.clocks[self.sched.threads[t].core])
    }

    /// A switch point costs the running core one step.
    pub fn tick(&mut self) {
        if let Some(t) = self.sched.current {
            self.sched.clocks[self.sched.threads[t].core] += STEP;
        }
        if self.config.time_limit.is_some_and(|l| self.now() > l) {
            self.stop(crate::STOP_EXIT, "the run reached its virtual time limit");
        }
    }

    /// The index of the thread a handle (or the current-thread pseudo-handle) designates.
    pub fn thread_of(&self, h: u32) -> Option<usize> {
        if u64::from(h) == crate::svc::CURRENT_THREAD {
            return self.sched.current;
        }
        match self.handles.get(h)? {
            Object::Thread(i) => Some(i),
            _ => None,
        }
    }

    /// CreateThread: a suspended thread with its own TLR; `core` -2 means
    /// the process's default core, 0. Returns its handle.
    pub fn create_thread(&mut self, entry: u64, arg: u64, stack_top: u64, priority: u32, core: i32) -> Result<u32, u32> {
        use crate::result::*;
        if !PRIORITIES.contains(&priority) {
            return Err(INVALID_PRIORITY);
        }
        let core = match core {
            -2 => 0,
            0..=5 => core as usize,
            _ => return Err(INVALID_CORE_ID),
        };
        let index = self.sched.threads.len();
        let handle = self.handles.add(Object::Thread(index)).ok_or(OUT_OF_HANDLES)?;
        let tlr = self.alloc_tlr();
        let mut cpu = Box::new(Cpu::new(self.mem.base()));
        (cpu.x[0], cpu.x[30], cpu.x[31], cpu.pc, cpu.tpidrro) = (arg, rt::EXIT, stack_top, entry, tlr);
        self.sched.next_id += 1;
        self.sched.threads.push(Thread {
            id: self.sched.next_id,
            handle,
            cpu: Box::into_raw(cpu),
            tlr,
            priority,
            ideal: core as i32,
            core,
            affinity: 1 << core,
            state: State::Created,
            ready_at: 0,
            seq: 0,
            deadline: None,
            on_timeout: 0,
            cpu_time: 0,
            host: std::time::Duration::ZERO,
            instructions: 0,
            work_rem: 0,
            fiber: None,
            wait: Wait::None,
            wait_seq: 0,
            cancel: false,
        });
        Ok(handle)
    }

    /// 0x200 zeroed bytes in the thread-local region; pages are mapped as needed.
    fn alloc_tlr(&mut self) -> u64 {
        let tlr = layout::THREAD_LOCAL.0 + self.sched.next_tlr;
        self.sched.next_tlr += TLR_SIZE;
        if tlr.is_multiple_of(gyroship_runtime::mem::PAGE) {
            self.mem.map(tlr, gyroship_runtime::mem::PAGE, Block::new(state::THREAD_LOCAL, perm::RW)).expect("the thread-local region has room");
        }
        tlr
    }

    /// The main thread: priority 44 on core 0, x0 = 0 and w1 = its own handle.
    pub fn create_main(&mut self, entry: u64, stack_top: u64) -> u32 {
        let h = self.create_thread(entry, 0, stack_top, MAIN_PRIORITY, 0).expect("the first thread");
        let i = self.sched.threads.len() - 1;
        // SAFETY: the thread has not run.
        unsafe { (*self.sched.threads[i].cpu).x[1] = u64::from(h) };
        self.ready(i, 0);
        h
    }

    /// Makes thread `t` runnable from time `at`, behind the others of its priority.
    pub fn ready(&mut self, t: usize, at: u64) {
        self.sched.seq += 1;
        let th = &mut self.sched.threads[t];
        (th.state, th.ready_at, th.seq, th.deadline, th.wait) = (State::Ready, at, self.sched.seq, None, Wait::None);
    }

    /// The running thread waits for `wait` until woken, or until
    /// `deadline`, when w0 gets `on_timeout`.
    pub fn block(&mut self, wait: Wait, deadline: Option<u64>, on_timeout: u32) {
        let t = self.sched.current.expect("a running thread");
        self.leave_core(t);
        self.sched.seq += 1;
        let th = &mut self.sched.threads[t];
        (th.state, th.wait, th.wait_seq, th.deadline, th.on_timeout) = (State::Waiting, wait, self.sched.seq, deadline, on_timeout);
    }

    /// Wakes waiting thread `t` now, with Result `result` (and x1 = `x1`).
    pub fn wake(&mut self, t: usize, result: u32, x1: Option<u64>) {
        let cpu = self.sched.threads[t].cpu;
        // SAFETY: a waiting thread is parked at a switch point; nothing else uses its context.
        unsafe {
            (*cpu).x[0] = u64::from(result);
            if let Some(v) = x1 {
                (*cpu).x[1] = v;
            }
        }
        let now = self.now();
        self.ready(t, now);
    }

    /// Waiting threads whose wait matches, highest priority first, then in arrival order.
    pub fn waiters(&self, matches: impl Fn(&Wait) -> bool) -> Vec<usize> {
        let mut w: Vec<usize> = (0..self.sched.threads.len()).filter(|&t| self.sched.threads[t].state == State::Waiting && matches(&self.sched.threads[t].wait)).collect();
        w.sort_by_key(|&t| (self.sched.threads[t].priority, self.sched.threads[t].wait_seq));
        w
    }

    /// The running thread ends; returns the fiber to switch to.
    pub fn exit_current(&mut self) -> Fiber {
        self.end_current();
        self.reschedule().expect("another fiber: a thread or the boot fiber")
    }

    /// The running thread ends; the next switch point leaves it for good
    /// and another fiber deletes its fiber.
    pub fn end_current(&mut self) {
        let t = self.sched.current.expect("a running thread");
        self.leave_core(t);
        self.sched.threads[t].state = State::Done;
        // Threads waiting for this one's end wake up.
        self.signal(Object::Thread(t));
        if let Some(f) = self.sched.threads[t].fiber.take() {
            self.sched.dead.push(f);
        }
        self.sched.current = None;
    }

    /// Moves thread `t` to core `c`; the scheduler places it there at its next decision.
    pub fn move_thread(&mut self, t: usize, c: usize) {
        self.leave_core(t);
        self.sched.threads[t].core = c;
    }

    /// Thread `t` stops running on its core: its running time is counted,
    /// and a thread that stays ready cannot start anywhere before the time
    /// it has reached.
    fn leave_core(&mut self, t: usize) {
        let c = self.sched.threads[t].core;
        if let Some((r, since)) = self.sched.on_core[c]
            && r == t
        {
            self.sched.threads[t].cpu_time += self.sched.clocks[c] - since;
            self.sched.on_core[c] = None;
        }
        let now = self.sched.clocks[c];
        let th = &mut self.sched.threads[t];
        if th.state == State::Ready {
            th.ready_at = th.ready_at.max(now);
        }
    }

    /// Chooses what runs next; returns the fiber to switch to, or `None` to
    /// keep running the current thread.
    pub fn reschedule(&mut self) -> Option<Fiber> {
        // Host time since the last switch point goes to the thread that ran.
        let now = std::time::Instant::now();
        if let (Some(cur), Some(mark)) = (self.sched.current, self.sched.host_mark) {
            self.sched.threads[cur].host += now - mark;
        }
        self.sched.host_mark = Some(now);
        let Some(next) = self.pick() else {
            self.finish();
            return self.sched.boot;
        };
        let same = self.sched.current == Some(next);
        // Even the same thread may come back to its core: it may have just blocked and been woken by its deadline.
        self.enter(next);
        if same { None } else { Some(self.fiber_of(next)) }
    }

    fn fiber_of(&mut self, t: usize) -> Fiber {
        *self.sched.threads[t].fiber.get_or_insert_with(|| fiber::create(fiber_main, t))
    }

    /// Thread `t` runs on its core from now on.
    fn enter(&mut self, t: usize) {
        let c = self.sched.threads[t].core;
        let now = self.sched.clocks[c];
        if self.sched.on_core[c].map(|(r, _)| r) != Some(t) {
            if let Some((p, _)) = self.sched.on_core[c] {
                self.leave_core(p);
            }
            self.sched.on_core[c] = Some((t, now));
            // GetThreadTickCount reads CNTPCT + [TLR + 0x108]: rewritten only
            // when the thread comes back to its core, so that the SDK's
            // read-again loop settles.
            let th = &self.sched.threads[t];
            let offset = th.cpu_time.wrapping_sub(now);
            let tlr = th.tlr;
            self.mem.write(tlr + 0x108, &offset.to_le_bytes()).expect("the TLR is mapped");
        }
        self.sched.current = Some(t);
        // For --profile's samples (host state only).
        gyroship_runtime::profile::set_guest(self.sched.threads[t].id, self.sched.threads[t].cpu);
    }

    /// The next thread to run, after waking the threads whose deadline
    /// comes first. `None` when nothing can run any more.
    fn pick(&mut self) -> Option<usize> {
        loop {
            self.migrate();
            // (key, core, start, thread): the core whose next thread starts earliest.
            let mut best: Option<(u64, usize, u64, usize)> = None;
            for c in 0..CORES {
                let Some((start, t)) = self.next_on(c) else { continue };
                // The running thread keeps its core a quantum longer.
                let key = if self.sched.current == Some(t) { start.saturating_sub(QUANTUM) } else { start };
                if best.is_none_or(|b| (key, c) < (b.0, b.1)) {
                    best = Some((key, c, start, t));
                }
            }
            let deadline = self.sched.threads.iter().filter(|t| t.state == State::Waiting).filter_map(|t| t.deadline).min();
            // A timer fires when it comes before every thread start and deadline; when
            // nothing else could ever run, only if a thread waits for its event (else the
            // run is deadlocked, and firing it would loop forever).
            if let Some((at, i)) = self.timers.iter().enumerate().map(|(i, t)| (t.at(), i)).min()
                && best.is_none_or(|b| at < b.2)
                && deadline.is_none_or(|d| at < d)
                && (best.is_some() || deadline.is_some() || self.timer_awaited(i))
            {
                if self.config.time_limit.is_some_and(|l| at > l) {
                    self.stop(crate::STOP_EXIT, "the run reached its virtual time limit");
                }
                self.fire(i, at);
                continue;
            }
            match (best, deadline) {
                (b, Some(d)) if b.is_none_or(|b| d < b.2) => {
                    let due: Vec<usize> = (0..self.sched.threads.len()).filter(|&t| self.sched.threads[t].state == State::Waiting && self.sched.threads[t].deadline == Some(d)).collect();
                    for t in due {
                        let th = &self.sched.threads[t];
                        // SAFETY: the thread is parked at a switch point.
                        unsafe { (*th.cpu).x[0] = u64::from(th.on_timeout) };
                        self.ready(t, d);
                    }
                }
                (Some((_, c, start, t)), _) => {
                    self.sched.clocks[c] = start;
                    return Some(t);
                }
                (None, _) => return None,
            }
        }
    }

    /// Whether a thread waits for the event timer `i` signals.
    fn timer_awaited(&self, i: usize) -> bool {
        let TimerAction::AudioSink { event, .. } = self.timers[i].action else { return false };
        self.sched.threads.iter().any(|t| t.state == State::Waiting && matches!(&t.wait, Wait::Sync(list) if list.contains(&Object::ReadableEvent(event))))
    }

    /// Timer `i` reaches its deadline `at`: its action runs, the threads it
    /// wakes start at `at`, and it rearms.
    fn fire(&mut self, i: usize, at: u64) {
        let was_ready: Vec<bool> = self.sched.threads.iter().map(|t| t.state == State::Ready).collect();
        let k = self.timers[i].k;
        match self.timers[i].action {
            TimerAction::AudioSink { ring, slots, data, channels, event } => {
                // The sink plays everything written (audio::sink_take): the frames go to
                // the host, nothing goes back to the guest.
                let want = self.audio_out.is_some() || self.audio_samples.is_some();
                let (write, frames) = crate::audio::sink_take(&self.mem, ring, slots, data, channels, want);
                if let Some(frames) = frames {
                    if let Some(all) = self.audio_samples.as_mut() {
                        all.extend_from_slice(&frames);
                    }
                    if let Some(out) = self.audio_out.as_mut() {
                        out(&frames, self.audio_gain);
                    }
                }
                self.signal_event(event);
                if k == 1 || k.is_multiple_of(100) {
                    self.journal.line(at, 0, &format!("audio sink: tick {k} read index -> {write}"));
                }
            }
            TimerAction::HidSample { shared } => self.hid_sample(shared, k),
        }
        for (t, was) in was_ready.into_iter().enumerate() {
            if !was && self.sched.threads[t].state == State::Ready {
                self.sched.threads[t].ready_at = at;
            }
        }
        self.timers[i].k += 1;
    }

    /// When core `c` can start its next thread, and which: the highest
    /// priority (then earliest arrival) among those ready by then.
    fn next_on(&self, c: usize) -> Option<(u64, usize)> {
        let ready = || self.sched.threads.iter().enumerate().filter(move |(_, t)| t.state == State::Ready && t.core == c);
        let clock = self.sched.clocks[c];
        let start = ready().map(|(_, t)| t.ready_at).min()?.max(clock);
        let (t, _) = ready().filter(|(_, t)| t.ready_at <= start).min_by_key(|(_, t)| (t.priority, t.seq))?;
        Some((start, t))
    }

    /// The thread each core runs next.
    fn heads(&self) -> Vec<Option<usize>> {
        (0..CORES).map(|k| self.next_on(k).map(|(_, t)| t)).collect()
    }

    fn ready_on(&self, c: usize) -> usize {
        self.sched.threads.iter().filter(|t| t.state == State::Ready && t.core == c).count()
    }

    /// A core with nothing ready takes the best ready thread that waits
    /// behind another and may run here; failing that, the thread another
    /// core runs next, if that core has another thread to run (as the
    /// Horizon scheduler does).
    fn migrate(&mut self) {
        for c in 0..CORES {
            if self.ready_on(c) > 0 {
                continue;
            }
            let heads = self.heads();
            let allowed = |t: &Thread| t.state == State::Ready && t.affinity & 1 << c != 0;
            let rank = |i: &usize| (self.sched.threads[*i].priority, self.sched.threads[*i].seq);
            let all = 0..self.sched.threads.len();
            let waiting = all.clone().filter(|&i| allowed(&self.sched.threads[i]) && heads[self.sched.threads[i].core] != Some(i)).min_by_key(rank);
            let head = || all.clone().filter(|&i| allowed(&self.sched.threads[i]) && heads[self.sched.threads[i].core] == Some(i) && self.ready_on(self.sched.threads[i].core) > 1).min_by_key(rank);
            if let Some(i) = waiting.or_else(head) {
                self.move_thread(i, c);
            }
        }
    }

    /// SleepThread(-1), the yield the SDK uses: the thread goes behind those
    /// of its priority, and its core may take a ready thread waiting on
    /// another core, of higher priority, or of the same priority if none is
    /// queued here.
    pub fn yield_with_migration(&mut self, t: usize) {
        let now = self.now();
        self.ready(t, now);
        let (c, priority) = (self.sched.threads[t].core, self.sched.threads[t].priority);
        let heads = self.heads();
        let peers = self.sched.threads.iter().enumerate().filter(|&(i, x)| i != t && x.state == State::Ready && x.core == c && x.priority == priority).count();
        let candidate = self
            .sched
            .threads
            .iter()
            .enumerate()
            .filter(|&(i, x)| i != t && x.state == State::Ready && x.core != c && heads[x.core] != Some(i) && x.affinity & 1 << c != 0)
            .filter(|(_, x)| x.priority < priority || (x.priority == priority && peers == 0))
            .min_by_key(|(_, x)| (x.priority, x.seq))
            .map(|(i, _)| i);
        if let Some(i) = candidate {
            self.move_thread(i, c);
        }
    }

    /// No thread can run: either all ended, or a deadlock.
    fn finish(&mut self) {
        let waiting: Vec<String> = self.sched.threads.iter().filter(|t| t.state != State::Done).map(|t| format!("thread {} ({:?})", t.id, t.state)).collect();
        self.sched.outcome = Some(if waiting.is_empty() { Outcome::Finished } else { Outcome::Deadlock(waiting.join(", ")) });
        self.sched.current = None;
    }

    /// Deletes the fibers of ended threads.
    pub fn reap(&mut self) {
        for f in self.sched.dead.drain(..) {
            // SAFETY: an ended thread's fiber, which is not the running one.
            unsafe { fiber::delete(f) };
        }
    }
}

/// The start of a guest thread's fiber: the runtime's driver, then the end of the thread.
unsafe extern "system" fn fiber_main(param: *mut c_void) {
    let t = param as usize;
    crate::with(|k| k.reap());
    let cpu = crate::with(|k| k.sched.threads[t].cpu);
    // SAFETY: the thread's own context, which lives as long as the kernel.
    unsafe { rt::run(cpu) };
    // Its code returned to EXIT: it ends as with ExitThread.
    let next = crate::with(|k| k.exit_current());
    // SAFETY: a live fiber; no borrow is held.
    unsafe { fiber::switch(next) };
    unreachable!("an ended thread's fiber was switched to");
}

/// Runs the guest threads on this host thread until none can run; returns how it ended.
pub fn run() -> Outcome {
    let boot = fiber::convert_thread();
    let next = crate::with(|k| {
        k.sched.boot = Some(boot);
        k.reschedule()
    });
    if let Some(f) = next
        && f != boot
    {
        // SAFETY: a thread's fiber; no borrow is held.
        unsafe { fiber::switch(f) };
    }
    let outcome = crate::with(|k| {
        k.reap();
        for t in &mut k.sched.threads {
            if let Some(f) = t.fiber.take() {
                // SAFETY: parked fibers that nothing will switch to again.
                unsafe { fiber::delete(f) };
            }
        }
        k.sched.boot = None;
        k.sched.outcome.take().unwrap_or(Outcome::Finished)
    });
    fiber::convert_back();
    outcome
}

/// After a switch point: runs what the scheduler chose, if it is not the
/// current thread. Returns when this thread is scheduled again.
pub fn switch_to(next: Option<Fiber>) {
    if let Some(f) = next {
        // SAFETY: a live fiber of this host thread; the kernel borrow is released.
        unsafe { fiber::switch(f) };
        crate::with(|k| k.reap());
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Config;
    use crate::journal::Journal;
    use gyroship_runtime::mem::Memory;
    use std::sync::atomic::{AtomicU32, Ordering::{Acquire, Relaxed, Release}};
    use std::time::{Duration, Instant};

    const SLOTS: u32 = 9;
    const FRAMES: u32 = 100_000;

    /// The SDK's WriteStream on another core, one channel: frames 1, 2… as
    /// f32, as many as the ring has room for, then the write index (release).
    fn produce(ring: usize, deadline: Instant) {
        // SAFETY: aligned indices of a ring that stays mapped while this runs.
        let index = |o: usize| unsafe { AtomicU32::from_ptr((ring + o) as *mut u32) };
        let mut next = 1;
        while next <= FRAMES && Instant::now() < deadline {
            let (write, read) = (index(0x1004).load(Relaxed), index(0x44).load(Acquire));
            let n = (SLOTS - 1 - (write + SLOTS - read) % SLOTS).min(FRAMES + 1 - next);
            for i in 0..n {
                let at = ring + 0x1040 + 4 * ((write + i) % SLOTS) as usize;
                // SAFETY: a free slot: the sink reads it only once the write index covers it.
                unsafe { (at as *mut f32).write((next + i) as f32) };
            }
            next += n;
            index(0x1004).store((write + n) % SLOTS, Release);
        }
    }

    /// While a producer writes the ring on another thread, the sink plays
    /// every frame once, in order: it loads the write index before the frames
    /// and stores the read index after them.
    #[test]
    fn the_sink_plays_every_frame_once_while_the_game_writes() {
        const RING: u64 = layout::ALIAS.0;
        let mut k = Kernel::new(Memory::new().unwrap(), Journal::memory(), Config::default());
        k.mem.map(RING, 0x2000, Block::new(state::NORMAL, perm::RW)).unwrap();
        let (w, r) = k.create_event().unwrap();
        k.handles.close(w);
        let event = k.event_of(r).unwrap();
        k.timers.push(Timer { start: 0, k: 1, num: 1, den: 1, action: TimerAction::AudioSink { ring: RING, slots: SLOTS, data: RING + 0x1040, channels: 1, event } });
        k.audio_samples = Some(Vec::new());
        let ring = k.mem.base() as usize + RING as usize;
        let deadline = Instant::now() + Duration::from_secs(60);
        std::thread::scope(|s| {
            s.spawn(|| produce(ring, deadline));
            while k.audio_samples.as_ref().unwrap().len() < FRAMES as usize {
                assert!(Instant::now() < deadline, "the sink stalled");
                k.fire(0, 0);
            }
        });
        let want: Vec<f32> = (1..=FRAMES).map(|f| f as f32).collect();
        assert_eq!(k.audio_samples.take().unwrap(), want);
    }
}
