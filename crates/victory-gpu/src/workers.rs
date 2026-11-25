//! The host's persistent worker threads, shared by everything that splits
//! a submission's work across threads: texture refreshes and view reads
//! (`crate::texture::in_parallel`), and refresh copies of title memory
//! (`crate::copy_pool`).
//!
//! The threads stay alive across submissions: each keeps its
//! `thread_local!` scratch buffers warm (`crate::texture::Scratch`) instead
//! of faulting and zero-filling fresh ones, and no OS thread is created
//! per call. One set of workers for both users, so they never compete for
//! the cores with a second set.

use std::sync::{Arc, Condvar, Mutex, OnceLock};

/// Workers besides the calling thread: four threads saturated host memory
/// for copies (#2051), and the title's own threads need the rest.
const WORKERS: usize = 3;

struct Job {
    call: fn(*const (), usize),
    data: *const (),
    n: usize,
    next: std::sync::atomic::AtomicUsize,
    panicked: Mutex<Option<Box<dyn std::any::Any + Send>>>,
}

// SAFETY: `Job` is shared across worker threads only while the caller's
// `WaitGuard` keeps the caller's stack frame alive until `remaining == 0`,
// and `data` points to an `F: Fn(usize) + Sync`.
unsafe impl Sync for Job {}

struct PoolState {
    job: *const Job,
    generation: u64,
    wanted: usize,
    claimed: usize,
    remaining: usize,
    shutdown: bool,
}

// SAFETY: `job` points to a `Sync` `Job` pinned on the caller's stack for the
// duration of the batch (`remaining > 0`).
unsafe impl Send for PoolState {}

#[cfg(test)]
thread_local! {
    /// Tests: this thread leaves every item of its runs to the workers (and
    /// waits for the pool rather than running inline when it is busy).
    pub(crate) static WORKERS_ALONE: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}

fn workers_alone() -> bool {
    #[cfg(test)]
    return WORKERS_ALONE.with(std::cell::Cell::get);
    #[cfg(not(test))]
    false
}

pub(crate) struct Pool {
    workers: usize,
    run_lock: Mutex<()>,
    state: Arc<(Mutex<PoolState>, Condvar, Condvar)>,
    #[cfg(test)]
    handles: Vec<std::thread::JoinHandle<()>>,
}

impl Pool {
    pub(crate) fn new(workers: usize) -> Self {
        let state = Arc::new((
            Mutex::new(PoolState {
                job: std::ptr::null(),
                generation: 0,
                wanted: 0,
                claimed: 0,
                remaining: 0,
                shutdown: false,
            }),
            Condvar::new(),
            Condvar::new(),
        ));
        let mut live_workers = 0usize;
        #[cfg(test)]
        let mut handles = Vec::with_capacity(workers);
        for idx in 0..workers {
            let shared = Arc::clone(&state);
            let builder = std::thread::Builder::new().name(format!("victory worker {idx}"));
            if let Ok(h) = builder.spawn(move || {
                let (lock, work_cv, done_cv) = &*shared;
                let mut my_gen = 0u64;
                loop {
                    let job_ptr = {
                        let mut st = lock.lock().unwrap_or_else(|e| e.into_inner());
                        while !st.shutdown && st.generation == my_gen {
                            st = work_cv.wait(st).unwrap_or_else(|e| e.into_inner());
                        }
                        if st.shutdown {
                            break;
                        }
                        my_gen = st.generation;
                        if st.claimed >= st.wanted {
                            continue;
                        }
                        st.claimed += 1;
                        st.job
                    };
                    // SAFETY: `job_ptr` is non-null and valid until this worker
                    // decrements `st.remaining` below.
                    run_job(unsafe { &*job_ptr });
                    let mut st = lock.lock().unwrap_or_else(|e| e.into_inner());
                    st.remaining -= 1;
                    if st.remaining == 0 {
                        st.job = std::ptr::null();
                        done_cv.notify_one();
                    }
                }
            }) {
                live_workers += 1;
                #[cfg(test)]
                handles.push(h);
                #[cfg(not(test))]
                drop(h);
            }
        }
        Self {
            workers: live_workers,
            run_lock: Mutex::new(()),
            state,
            #[cfg(test)]
            handles,
        }
    }

    /// Runs `f(i)` for each `i` below `n` on up to `threads` threads, the
    /// calling one included; returns once all are done.
    pub(crate) fn run<F: Fn(usize) + Sync>(&self, threads: usize, n: usize, f: F) {
        use std::sync::atomic::AtomicUsize;

        let workers = threads.saturating_sub(1).min(self.workers);
        if workers == 0 || n < 2 {
            (0..n).for_each(f);
            return;
        }
        // Contended or re-entrant call: run inline instead of blocking or
        // deadlocking against an outer run holding `run_lock`.
        let _run = match self.run_lock.try_lock() {
            Ok(run) => run,
            Err(_) if workers_alone() => self.run_lock.lock().unwrap_or_else(|e| e.into_inner()),
            Err(_) => {
                (0..n).for_each(f);
                return;
            }
        };

        fn invoke<F: Fn(usize) + Sync>(data: *const (), i: usize) {
            // SAFETY: `data` points to `f: F` on `run`'s stack for the batch.
            unsafe { (&*data.cast::<F>())(i) };
        }

        let job = Job {
            call: invoke::<F>,
            data: (&raw const f).cast(),
            n,
            next: AtomicUsize::new(0),
            panicked: Mutex::new(None),
        };

        struct WaitGuard<'a> {
            state: &'a (Mutex<PoolState>, Condvar, Condvar),
        }
        impl Drop for WaitGuard<'_> {
            fn drop(&mut self) {
                let (lock, _, done_cv) = self.state;
                let mut st = lock.lock().unwrap_or_else(|e| e.into_inner());
                while st.remaining > 0 {
                    st = done_cv.wait(st).unwrap_or_else(|e| e.into_inner());
                }
                st.job = std::ptr::null();
            }
        }

        {
            let (lock, work_cv, _) = &*self.state;
            let mut st = lock.lock().unwrap_or_else(|e| e.into_inner());
            st.job = &raw const job;
            st.wanted = workers;
            st.claimed = 0;
            st.remaining = workers;
            st.generation = st.generation.wrapping_add(1);
            drop(st);
            work_cv.notify_all();
        }
        let guard = WaitGuard { state: &self.state };
        if !workers_alone() {
            run_job(&job);
        }
        drop(guard);
        drop(_run);

        if let Some(payload) = job.panicked.into_inner().unwrap_or_else(|e| e.into_inner()) {
            std::panic::resume_unwind(payload);
        }
    }
}

#[cfg(test)]
impl Drop for Pool {
    fn drop(&mut self) {
        let (lock, work_cv, _) = &*self.state;
        {
            let mut st = lock.lock().unwrap_or_else(|e| e.into_inner());
            st.shutdown = true;
        }
        work_cv.notify_all();
        for h in self.handles.drain(..) {
            let _ = h.join();
        }
    }
}

fn run_job(job: &Job) {
    use std::panic::{AssertUnwindSafe, catch_unwind};
    use std::sync::atomic::Ordering::Relaxed;

    loop {
        let i = job.next.fetch_add(1, Relaxed);
        if i >= job.n {
            break;
        }
        if let Err(payload) = catch_unwind(AssertUnwindSafe(|| (job.call)(job.data, i))) {
            let mut slot = job.panicked.lock().unwrap_or_else(|e| e.into_inner());
            if slot.is_none() {
                *slot = Some(payload);
            }
            job.next.store(job.n, Relaxed);
            break;
        }
    }
}

/// The shared workers, started on first use: [`WORKERS`], fewer on a host
/// with fewer than five logical CPUs.
pub(crate) fn pool() -> &'static Pool {
    static POOL: OnceLock<Pool> = OnceLock::new();
    POOL.get_or_init(|| {
        let cpus = std::thread::available_parallelism().map_or(4, |n| n.get());
        Pool::new(WORKERS.min(cpus.saturating_sub(1)))
    })
}

/// Runs `f(i)` for each `i` below `n` on up to `threads` threads (the
/// calling one and the shared workers); returns once all are done. Runs
/// inline where the workers are busy with another caller's items.
pub(crate) fn run(threads: usize, n: usize, f: impl Fn(usize) + Sync) {
    if threads < 2 || n < 2 {
        (0..n).for_each(f);
        return;
    }
    pool().run(threads, n, f);
}

/// The workers that started (none where threads cannot start).
pub(crate) fn started() -> usize {
    pool().workers
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_pool_handles_reentrancy_and_panics() {
        use std::sync::atomic::{AtomicUsize, Ordering};

        let pool = Pool::new(3);
        let sum = AtomicUsize::new(0);
        pool.run(4, 64, |i| {
            sum.fetch_add(i + 1, Ordering::Relaxed);
        });
        assert_eq!(sum.load(Ordering::Relaxed), 64 * 65 / 2);

        // Nested call runs inline without deadlocking.
        let nested = AtomicUsize::new(0);
        pool.run(4, 8, |_| {
            pool.run(4, 4, |_| {
                nested.fetch_add(1, Ordering::Relaxed);
            });
        });
        assert_eq!(nested.load(Ordering::Relaxed), 32);

        // Panic propagates cleanly and leaves pool reusable afterward.
        let caught = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            pool.run(4, 16, |i| {
                assert_ne!(i, 5, "boom");
            });
        }));
        assert!(caught.is_err());

        let after = AtomicUsize::new(0);
        pool.run(4, 16, |_| {
            after.fetch_add(1, Ordering::Relaxed);
        });
        assert_eq!(after.load(Ordering::Relaxed), 16);
    }
}
