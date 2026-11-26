//! Submissions the host GPU may still be running (`VICTORY_ASYNC_SUBMIT`),
//! oldest first, and what is done once each has run: first its work
//! (copying what shaders wrote back into title memory, then the
//! bottom-of-pipe writes that tell the title the GPU is done), then the
//! title's fence signals queued behind it. Finished in submission order,
//! one thread at a time.
//!
//! The rule this keeps: the title never sees a signal (a fence value, a
//! memory label) before the GPU work submitted ahead of it and the copies
//! of what that work wrote have reached title memory. A signal queued while
//! a submission is in flight runs after that submission's work; one queued
//! while its work is being done also waits for it (the entry stays until
//! its work is done, and its signals run under the lock).
//!
//! After the timeline of wbkepler's #595 (rebased as #1923), without its
//! per-range conflicts: the next submission is prepared only once the one
//! in flight has finished (`Host::execute`), so the CPU and the GPU overlap
//! while the title records its next lists, never on the same memory.

use std::collections::VecDeque;
use std::sync::{Condvar, Mutex, MutexGuard};

type Signal = Box<dyn FnOnce() + Send>;

struct Entry<W> {
    /// The host fence value the submission signals.
    value: u64,
    /// Title fence signals the title queued after it.
    signals: Vec<Signal>,
    /// Taken while it is being finished.
    work: Option<W>,
}

pub(crate) struct Timeline<W> {
    entries: Mutex<VecDeque<Entry<W>>>,
    added: Condvar,
    /// Held while submissions are finished: in order, by one thread.
    finishing: Mutex<()>,
}

impl<W> Timeline<W> {
    pub(crate) const fn new() -> Self {
        Timeline {
            entries: Mutex::new(VecDeque::new()),
            added: Condvar::new(),
            finishing: Mutex::new(()),
        }
    }

    fn entries(&self) -> MutexGuard<'_, VecDeque<Entry<W>>> {
        self.entries.lock().unwrap_or_else(|e| e.into_inner())
    }

    /// A submission that signals `value` (higher than any before it), and
    /// its work once it ran.
    pub(crate) fn push(&self, value: u64, work: W) {
        let mut entries = self.entries();
        debug_assert!(entries.back().is_none_or(|e| e.value < value));
        entries.push_back(Entry {
            value,
            signals: Vec::new(),
            work: Some(work),
        });
        drop(entries);
        self.added.notify_all();
    }

    /// Submissions not finished yet.
    pub(crate) fn len(&self) -> usize {
        self.entries().len()
    }

    /// The value of the newest submission not finished yet.
    pub(crate) fn newest(&self) -> Option<u64> {
        self.entries().back().map(|e| e.value)
    }

    /// Runs `signal` once every submission pushed so far has finished: now
    /// if none is left.
    pub(crate) fn after(&self, signal: impl FnOnce() + Send + 'static) {
        let mut entries = self.entries();
        match entries.back_mut() {
            Some(last) => last.signals.push(Box::new(signal)),
            // Under the lock, so it stays behind signals still running.
            None => signal(),
        }
    }

    /// Blocks until a submission is left; the oldest one's value.
    pub(crate) fn next(&self) -> u64 {
        let mut entries = self.entries();
        loop {
            if let Some(first) = entries.front() {
                return first.value;
            }
            entries = self.added.wait(entries).unwrap_or_else(|e| e.into_inner());
        }
    }

    /// Finishes, oldest first, the submissions whose values are at most
    /// `reached`: `finish` takes each one's work, then its signals run.
    /// Returns once those are all finished, by this thread or another.
    pub(crate) fn finish(&self, reached: u64, mut finish: impl FnMut(W)) {
        let _finishing = self.finishing.lock().unwrap_or_else(|e| e.into_inner());
        loop {
            let work = {
                let mut entries = self.entries();
                match entries.front_mut() {
                    Some(first) if first.value <= reached => first.work.take(),
                    _ => return,
                }
            };
            // Outside the lock: a signal queued meanwhile goes on this
            // entry, so behind this work.
            if let Some(work) = work {
                finish(work);
            }
            let mut entries = self.entries();
            if let Some(done) = entries.pop_front() {
                // Under the lock: a signal queued now goes behind these.
                done.signals.into_iter().for_each(|s| s());
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;
    use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};

    type Log = Arc<Mutex<Vec<String>>>;

    fn note(log: &Log, what: &str) {
        log.lock().unwrap().push(what.to_owned());
    }

    /// A signal queued behind a submission runs only after its work, and
    /// one queued with nothing in flight at once.
    #[test]
    fn submissions_finish_in_order_and_signals_follow_their_work() {
        let log: Log = Arc::default();
        let t: Timeline<&str> = Timeline::new();
        let l = log.clone();
        t.after(move || note(&l, "signal 0"));
        t.push(1, "copy back 1");
        t.push(2, "copy back 2");
        let l = log.clone();
        t.after(move || note(&l, "signal 2"));
        let finish = |w: &str| note(&log, w);
        t.finish(0, finish);
        assert_eq!(*log.lock().unwrap(), ["signal 0"]);
        t.finish(1, finish);
        assert_eq!(*log.lock().unwrap(), ["signal 0", "copy back 1"], "signal 2 waits for submission 2");
        assert_eq!(t.newest(), Some(2));
        t.finish(5, finish);
        assert_eq!(*log.lock().unwrap(), ["signal 0", "copy back 1", "copy back 2", "signal 2"]);
        assert_eq!(t.len(), 0);
        t.finish(5, finish);
        assert_eq!(log.lock().unwrap().len(), 4, "finished once");
    }

    /// Copy-backs keep submission order, each submission's signals between
    /// its copy-back and the next one's.
    #[test]
    fn copy_backs_keep_submission_order() {
        let log: Log = Arc::default();
        let t: Timeline<u64> = Timeline::new();
        for v in 1..=4 {
            t.push(v, v);
            let l = log.clone();
            t.after(move || note(&l, &format!("signal {v}")));
        }
        t.finish(4, |v| note(&log, &format!("copy back {v}")));
        let want: Vec<String> = (1..=4).flat_map(|v| [format!("copy back {v}"), format!("signal {v}")]).collect();
        assert_eq!(*log.lock().unwrap(), want);
    }

    /// A signal queued while a submission's work is being done (the title
    /// signalling from another thread) still waits for that work.
    #[test]
    fn a_signal_queued_during_the_work_waits_for_it() {
        let log: Log = Arc::default();
        let t: Arc<Timeline<()>> = Arc::new(Timeline::new());
        t.push(1, ());
        let (t2, l2) = (t.clone(), log.clone());
        t.finish(1, move |()| {
            let l3 = l2.clone();
            t2.after(move || note(&l3, "signal"));
            note(&l2, "copy back");
        });
        assert_eq!(*log.lock().unwrap(), ["copy back", "signal"]);
    }

    /// Finishers racing (the thread that waits on the host fence and a
    /// submission draining the one before it) finish each submission once,
    /// in order; a title thread polling the fence the signals move never
    /// sees a value whose copy-back has not landed.
    #[test]
    fn racing_finishers_never_show_a_signal_before_its_copy_back() {
        const N: u64 = 2000;
        let t: Arc<Timeline<u64>> = Arc::new(Timeline::new());
        // Title memory the copy-backs write, and the fence the signals move.
        let memory = Arc::new(AtomicU64::new(0));
        let fence = Arc::new(AtomicU64::new(0));
        let order = Arc::new(Mutex::new(Vec::new()));
        let stop = Arc::new(AtomicBool::new(false));
        let watcher = {
            let (memory, fence, stop) = (memory.clone(), fence.clone(), stop.clone());
            std::thread::spawn(move || {
                let mut seen = 0;
                while !stop.load(Ordering::Acquire) {
                    let f = fence.load(Ordering::Acquire);
                    let m = memory.load(Ordering::Acquire);
                    assert!(m >= f, "fence {f} seen before copy-back {f} (memory {m})");
                    assert!(f >= seen, "the fence went back");
                    seen = f;
                }
            })
        };
        let finisher = |t: Arc<Timeline<u64>>, memory: Arc<AtomicU64>, order: Arc<Mutex<Vec<u64>>>| {
            std::thread::spawn(move || {
                for reached in (0..=N).step_by(7).chain([N]) {
                    t.finish(reached, |v| {
                        order.lock().unwrap().push(v);
                        memory.store(v, Ordering::Release);
                    });
                }
            })
        };
        for v in 1..=N {
            t.push(v, v);
            let fence = fence.clone();
            t.after(move || fence.store(v, Ordering::Release));
        }
        let a = finisher(t.clone(), memory.clone(), order.clone());
        let b = finisher(t.clone(), memory.clone(), order.clone());
        a.join().unwrap();
        b.join().unwrap();
        stop.store(true, Ordering::Release);
        watcher.join().unwrap();
        assert_eq!(*order.lock().unwrap(), (1..=N).collect::<Vec<_>>(), "each once, in order");
        assert_eq!(fence.load(Ordering::Acquire), N);
        assert_eq!(t.len(), 0);
    }

    /// `next` waits for a submission and names the oldest.
    #[test]
    fn next_waits_for_a_submission() {
        let t: Arc<Timeline<()>> = Arc::new(Timeline::new());
        let t2 = t.clone();
        let waiter = std::thread::spawn(move || t2.next());
        std::thread::sleep(std::time::Duration::from_millis(20));
        t.push(3, ());
        t.push(4, ());
        assert_eq!(waiter.join().unwrap(), 3);
        assert_eq!(t.next(), 3);
    }
}
