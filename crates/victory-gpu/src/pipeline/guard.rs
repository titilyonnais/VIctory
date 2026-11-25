//! A net under the host driver's compiler. AMD's (RX 9070 XT, Adrenalin
//! 26.11) faults making a few of the title's pipelines (`amdxc64.dll`,
//! reading near address 0) and never ends a few others while it takes
//! memory; either cost the run, and the next run made the same pipeline
//! again.
//!
//! On the pipeline workers, a fault inside the driver's compile is taken
//! here: the pipeline is noted in `compiler-faults.txt` next to the
//! pipeline library and refused from then on (as a refused pipeline, its
//! draws and dispatches are dropped), the worker is parked for good and
//! another takes its place, and nothing waits for the pipeline any more. A
//! compile that has not ended after `VICTORY_PIPELINE_HANG` seconds (60 by
//! default, 0 for never) is noted the same way and its worker replaced;
//! the compile goes on, and is used if it ends. Deleting the file (after a
//! driver update, say) tries them all again; `VICTORY_PIPELINE_GUARD=0`
//! turns the net off, noted pipelines included.
//!
//! shortcut: only faults that are no deliberate probe are taken (an access
//! near address 0, an illegal or privileged instruction, an integer
//! division by zero, outside our own code); a fault elsewhere in the driver
//! still ends the run. A parked worker keeps what the driver held, memory
//! and any lock; a compile that never ends keeps its memory and its core.

use std::collections::HashSet;
use std::sync::{Condvar, Mutex, OnceLock};
use std::time::{Duration, Instant};

/// Pipelines (library names) the driver's compiler faulted on or did not
/// end, one per line, with what happened.
const FILE: &str = "compiler-faults.txt";

/// Whether the net is on, and how long a compile may take before it is
/// taken as one that never ends.
fn settings() -> (bool, Option<Duration>) {
    static SETTINGS: OnceLock<(bool, Option<Duration>)> = OnceLock::new();
    *SETTINGS.get_or_init(|| {
        let on = std::env::var("VICTORY_PIPELINE_GUARD").map_or(true, |v| v.trim() != "0");
        let secs = std::env::var("VICTORY_PIPELINE_HANG")
            .ok()
            .and_then(|v| v.trim().parse::<u64>().ok())
            .unwrap_or(60);
        (on, (secs > 0).then(|| Duration::from_secs(secs)))
    })
}

/// The noted pipelines, read from [`FILE`] on first use.
static NOTED: Mutex<Option<HashSet<String>>> = Mutex::new(None);

fn noted() -> std::sync::MutexGuard<'static, Option<HashSet<String>>> {
    let mut noted = NOTED.lock().unwrap_or_else(|e| e.into_inner());
    if noted.is_none() {
        let text = super::library::dir()
            .and_then(|d| std::fs::read_to_string(d.join(FILE)).ok())
            .unwrap_or_default();
        *noted = Some(names(&text));
    }
    noted
}

/// The names of a [`FILE`]'s lines (the first word of each).
fn names(text: &str) -> HashSet<String> {
    text.lines()
        .filter_map(|l| l.split_whitespace().next())
        .map(str::to_owned)
        .collect()
}

/// Whether the pipeline named `name` is not to be made: the driver's
/// compiler faulted on it or did not end it, in this run or an earlier one.
pub(super) fn refused(name: &str) -> bool {
    if !settings().0 {
        return false;
    }
    let refused = noted().as_ref().is_some_and(|n| n.contains(name));
    if refused {
        tracing::warn!(name, "host pipeline left out: the driver's compiler faulted on it or did not end it ({FILE})");
    }
    refused
}

/// Notes `name` as refused from now on, and in [`FILE`] for later runs.
fn note(name: &str, what: &str) {
    if !noted().get_or_insert_with(HashSet::new).insert(name.to_owned()) {
        return;
    }
    let Some(dir) = super::library::dir() else { return };
    use std::io::Write;
    let file = std::fs::OpenOptions::new().create(true).append(true).open(dir.join(FILE));
    if let Ok(mut f) = file {
        let _ = writeln!(f, "{name} {what}");
    }
}

/// The compiles under way on the workers: thread, library name, start.
static AT_WORK: Mutex<Vec<(u32, String, Instant)>> = Mutex::new(Vec::new());

/// Workers parked after a fault, for the watcher to replace.
static PARKED: (Mutex<Vec<u32>>, Condvar) = (Mutex::new(Vec::new()), Condvar::new());

thread_local! {
    /// Whether this thread is inside a guarded compile.
    static MAKING: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}

fn thread_id() -> u32 {
    // SAFETY: a plain call.
    unsafe { windows_sys::Win32::System::Threading::GetCurrentThreadId() }
}

/// Runs `create`, the driver's compile of the pipeline named `name`, under
/// the net when on a pipeline worker (elsewhere a parked thread would be
/// the title's).
pub(super) fn making<T>(name: &str, create: impl FnOnce() -> T) -> T {
    if !settings().0 || !super::make::on_worker() {
        return create();
    }
    install();
    let id = thread_id();
    AT_WORK
        .lock()
        .unwrap_or_else(|e| e.into_inner())
        .push((id, name.to_owned(), Instant::now()));
    MAKING.set(true);
    let made = create();
    MAKING.set(false);
    AT_WORK.lock().unwrap_or_else(|e| e.into_inner()).retain(|w| w.0 != id);
    made
}

/// Registers the handler and starts the watcher, once.
fn install() {
    static ONCE: std::sync::Once = std::sync::Once::new();
    ONCE.call_once(|| {
        use windows_sys::Win32::System::Diagnostics::Debug::AddVectoredExceptionHandler;
        // SAFETY: a handler that lives as long as the process; ahead of the
        // crash reporter's, which would log the fault as a crash.
        unsafe { AddVectoredExceptionHandler(1, Some(on_fault)) };
        let spawned = std::thread::Builder::new()
            .name("victory-pipeline-guard".into())
            .spawn(watch);
        if let Err(e) = spawned {
            tracing::warn!(error = %e, "no pipeline guard watcher");
        }
    });
}

/// Replaces parked workers, and notes compiles that do not end.
fn watch() {
    let mut late: Vec<(u32, String)> = Vec::new();
    loop {
        let parked = {
            let (lock, wake) = &PARKED;
            let held = lock.lock().unwrap_or_else(|e| e.into_inner());
            let (mut held, _) = wake
                .wait_timeout_while(held, Duration::from_secs(1), |p| p.is_empty())
                .unwrap_or_else(|e| e.into_inner());
            std::mem::take(&mut *held)
        };
        for id in parked {
            super::make::worker_lost(id);
        }
        let Some(limit) = settings().1 else { continue };
        let long: Vec<(u32, String)> = AT_WORK
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .iter()
            .filter(|w| w.2.elapsed() > limit && !late.iter().any(|l| l.0 == w.0 && l.1 == w.1))
            .map(|w| (w.0, w.1.clone()))
            .collect();
        for (id, name) in long {
            tracing::error!(
                name,
                secs = limit.as_secs(),
                "the driver's compiler does not end a pipeline: it is left out from now on, its worker replaced"
            );
            note(&name, &format!("not ended after {} s", limit.as_secs()));
            super::make::worker_lost(id);
            late.push((id, name));
        }
        // Forget compiles that ended since.
        let at_work = AT_WORK.lock().unwrap_or_else(|e| e.into_inner());
        late.retain(|l| at_work.iter().any(|w| w.0 == l.0 && w.1 == l.1));
    }
}

/// Whether a fault is one to take: no deliberate probe (see the module's
/// shortcut), in a guarded compile, outside our own code.
fn taken(code: i32, info: &[usize], at: usize) -> bool {
    use windows_sys::Win32::Foundation::{
        EXCEPTION_ACCESS_VIOLATION, EXCEPTION_ILLEGAL_INSTRUCTION, EXCEPTION_INT_DIVIDE_BY_ZERO,
        EXCEPTION_PRIV_INSTRUCTION,
    };
    let kind = match code {
        EXCEPTION_ACCESS_VIOLATION => info.len() >= 2 && info[1] < 0x1_0000,
        EXCEPTION_ILLEGAL_INSTRUCTION | EXCEPTION_PRIV_INSTRUCTION | EXCEPTION_INT_DIVIDE_BY_ZERO => true,
        _ => false,
    };
    kind && MAKING.try_with(std::cell::Cell::get).unwrap_or(false) && !ours(at)
}

/// Whether `at` is in our own executable's code.
fn ours(at: usize) -> bool {
    use windows_sys::Win32::System::LibraryLoader::{
        GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS, GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
        GetModuleHandleExW, GetModuleHandleW,
    };
    let mut module = std::ptr::null_mut();
    // SAFETY: plain calls; the address is only looked up.
    unsafe {
        GetModuleHandleExW(
            GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS | GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
            at as *const u16,
            &mut module,
        ) != 0
            && module == GetModuleHandleW(std::ptr::null())
    }
}

unsafe extern "system" fn on_fault(
    info: *mut windows_sys::Win32::System::Diagnostics::Debug::EXCEPTION_POINTERS,
) -> i32 {
    use windows_sys::Win32::System::Diagnostics::Debug::EXCEPTION_CONTINUE_SEARCH;
    // SAFETY: the system passes the exception being dispatched.
    let r = unsafe { &*(*info).ExceptionRecord };
    let n = (r.NumberParameters as usize).min(r.ExceptionInformation.len());
    if !taken(r.ExceptionCode, &r.ExceptionInformation[..n], r.ExceptionAddress as usize) {
        return EXCEPTION_CONTINUE_SEARCH;
    }
    let id = thread_id();
    let name = AT_WORK
        .lock()
        .unwrap_or_else(|e| e.into_inner())
        .iter()
        .find(|w| w.0 == id)
        .map(|w| w.1.clone())
        .unwrap_or_default();
    let code = r.ExceptionCode as u32;
    tracing::error!(
        name,
        code = format_args!("{code:#x}"),
        at = format_args!("{:#x}", r.ExceptionAddress as usize),
        "the driver's compiler faulted making a pipeline: it is left out from now on, its worker parked"
    );
    note(&name, &format!("fault {code:#x}"));
    AT_WORK.lock().unwrap_or_else(|e| e.into_inner()).retain(|w| w.0 != id);
    let (lock, wake) = &PARKED;
    lock.lock().unwrap_or_else(|e| e.into_inner()).push(id);
    wake.notify_one();
    loop {
        std::thread::sleep(Duration::from_secs(3600));
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A line's first word is the name; blank lines name nothing.
    #[test]
    fn noted_names_are_first_words() {
        let n = names("abc fault 0xc0000005\n\n  def not ended after 60 s\nghi\n");
        assert_eq!(n.len(), 3);
        assert!(n.contains("abc") && n.contains("def") && n.contains("ghi"));
    }

    /// Only faults in a guarded compile are taken, and of those only the
    /// kinds no driver probes with.
    #[test]
    fn only_guarded_non_probe_faults_are_taken() {
        use windows_sys::Win32::Foundation::{EXCEPTION_ACCESS_VIOLATION, EXCEPTION_BREAKPOINT};
        let elsewhere = 0x1234_5678_usize;
        assert!(!taken(EXCEPTION_ACCESS_VIOLATION, &[0, 8], elsewhere));
        MAKING.set(true);
        assert!(taken(EXCEPTION_ACCESS_VIOLATION, &[0, 8], elsewhere));
        assert!(!taken(EXCEPTION_ACCESS_VIOLATION, &[0, 0x7ff0_0000], elsewhere));
        assert!(!taken(EXCEPTION_BREAKPOINT, &[], elsewhere));
        // Our own code's faults are bugs to see, not the driver's.
        assert!(!taken(EXCEPTION_ACCESS_VIOLATION, &[0, 8], ours as fn(usize) -> bool as usize));
        MAKING.set(false);
    }
}
