//! The PC's memory, as the rules that free or hold back what we keep see it
//! (from avatarmanz/VIctory-7900XTX#206, whose `residency` module this branch
//! does not have). The system is asked at most once a second: below 3 GiB of
//! free RAM or 4 GiB of commit the PC is short of memory ([`pressured`]), and
//! stays so until there are 5 / 6 GiB again. On by default;
//! `VICTORY_RAM_PRESSURE=0`: never short, and no PC counted as limited.
use std::sync::OnceLock;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering::Relaxed};
use windows_sys::Win32::System::SystemInformation::{GlobalMemoryStatusEx, MEMORYSTATUSEX};

const GIB: u64 = 1 << 30;

/// `VICTORY_RAM_PRESSURE=0` turns this module off.
pub(crate) fn on() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("VICTORY_RAM_PRESSURE").is_none_or(|v| v != "0"))
}

fn memory_status() -> Option<MEMORYSTATUSEX> {
    // SAFETY: plain data the call fills in.
    let mut status: MEMORYSTATUSEX = unsafe { std::mem::zeroed() };
    status.dwLength = size_of::<MEMORYSTATUSEX>() as u32;
    // SAFETY: a sized structure.
    (unsafe { GlobalMemoryStatusEx(&mut status) } != 0).then_some(status)
}

/// A PC of 32 GB or less (or `VICTORY_RAM_LIMITED=1`).
pub(crate) fn limited_host_ram() -> bool {
    static LIMITED: OnceLock<bool> = OnceLock::new();
    *LIMITED.get_or_init(|| {
        on() && (std::env::var_os("VICTORY_RAM_LIMITED").is_some_and(|v| v == "1")
            || memory_status().is_some_and(|s| s.ullTotalPhys <= 32 * GIB))
    })
}

/// Whether the PC is short of memory after one that `was` (or was not), with
/// `physical` bytes of RAM free and `commit` bytes of commit left.
fn next_pressure(was: bool, physical: u64, commit: u64) -> bool {
    if was { physical < 5 * GIB || commit < 6 * GIB } else { physical <= 3 * GIB || commit <= 4 * GIB }
}

static PRESSURED: AtomicBool = AtomicBool::new(false);
/// When the system was last asked ([`crate::host::coarse_ms`]), plus one.
static SAMPLED: AtomicU64 = AtomicU64::new(0);

/// Whether the PC is short of memory (asked again when a second has gone by
/// since the last look; a failed look keeps the last answer).
pub(crate) fn pressured() -> bool {
    if !on() {
        return false;
    }
    let now = crate::host::coarse_ms() + 1;
    let last = SAMPLED.load(Relaxed);
    if (last == 0 || now >= last + 1000) && SAMPLED.compare_exchange(last, now, Relaxed, Relaxed).is_ok() {
        if let Some(status) = memory_status() {
            let (physical, commit) = (status.ullAvailPhys, status.ullAvailPageFile);
            let was = PRESSURED.load(Relaxed);
            let pressure = next_pressure(was, physical, commit);
            if pressure != was {
                PRESSURED.store(pressure, Relaxed);
                if pressure {
                    tracing::warn!(
                        physical_free_mib = physical >> 20,
                        available_commit_mib = commit >> 20,
                        "PC memory short: idle texture copies go after 5 s"
                    );
                } else {
                    tracing::info!(
                        physical_free_mib = physical >> 20,
                        available_commit_mib = commit >> 20,
                        "PC memory short no more"
                    );
                }
            }
        }
    }
    PRESSURED.load(Relaxed)
}

/// Whether this process's private bytes have reached 70 % of the smaller
/// of the PC's RAM and its commit limit (RAM plus page files), from
/// avatarmanz/VIctory-7900XTX#271. Free RAM alone does not show it: on a
/// 32 GB PC the process grows to 35–43 GB of commit while 12 GB of RAM can
/// still look free, and freed copies kept for five minutes then push it
/// into the page file. The commit limit alone does not either: with a
/// 48 GB page file it is 80 GB, far past where paging starts. Asked at most
/// once a second; a failed query keeps the last answer.
/// `VICTORY_NEAR_COMMIT=0` (or `VICTORY_RAM_PRESSURE=0`): never near.
pub(crate) fn near_commit_limit() -> bool {
    use windows_sys::Win32::System::ProcessStatus::{K32GetProcessMemoryInfo, PROCESS_MEMORY_COUNTERS};
    use windows_sys::Win32::System::Threading::GetCurrentProcess;
    static ON: OnceLock<bool> = OnceLock::new();
    static NEAR: AtomicBool = AtomicBool::new(false);
    static LOOKED: AtomicU64 = AtomicU64::new(0);
    if !*ON.get_or_init(|| on() && std::env::var_os("VICTORY_NEAR_COMMIT").is_none_or(|v| v != "0")) {
        return false;
    }
    let now = crate::host::coarse_ms() + 1;
    let last = LOOKED.load(Relaxed);
    if (last == 0 || now >= last + 1000) && LOOKED.compare_exchange(last, now, Relaxed, Relaxed).is_ok() {
        // SAFETY: plain data the call fills in.
        let mut counters: PROCESS_MEMORY_COUNTERS = unsafe { std::mem::zeroed() };
        counters.cb = size_of::<PROCESS_MEMORY_COUNTERS>() as u32;
        // SAFETY: our own process and a sized structure.
        let ok = unsafe { K32GetProcessMemoryInfo(GetCurrentProcess(), &mut counters, counters.cb) } != 0;
        if let (true, Some(status)) = (ok, memory_status()) {
            // The pagefile usage of a process is its private commit.
            let private = counters.PagefileUsage as u64;
            NEAR.store(private_near_limit(private, status.ullTotalPhys, status.ullTotalPageFile), Relaxed);
        }
    }
    NEAR.load(Relaxed)
}

fn private_near_limit(private: u64, total_physical: u64, commit_limit: u64) -> bool {
    let room = match (total_physical, commit_limit) {
        (0, 0) => return false,
        (0, b) | (b, 0) => b,
        (a, b) => a.min(b),
    };
    private.saturating_mul(10) >= room.saturating_mul(7)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn freed_copies_not_kept_near_the_commit_limit() {
        // 70 % exactly of the smaller one is near, a byte under it is not.
        assert!(private_near_limit(7 * GIB, 10 * GIB, 20 * GIB));
        assert!(!private_near_limit(7 * GIB - 1, 10 * GIB, 20 * GIB));
        assert!(private_near_limit(7 * GIB, 20 * GIB, 10 * GIB));
        // A 32 GB PC with a 48 GB page file (an 80 GB commit limit): RAM
        // sets the mark, about 22.4 GB, not 56 GB.
        assert!(!private_near_limit(20 * GIB, 32 * GIB, 80 * GIB));
        assert!(private_near_limit(23 * GIB, 32 * GIB, 80 * GIB));
        // A 64 GB PC at 30 GB keeps its copies as before.
        assert!(!private_near_limit(30 * GIB, 64 * GIB, 80 * GIB));
        // One unknown number: the other one; both unknown: never near.
        assert!(private_near_limit(23 * GIB, 0, 32 * GIB));
        assert!(private_near_limit(23 * GIB, 32 * GIB, 0));
        assert!(!private_near_limit(u64::MAX, 0, 0));
        assert!(private_near_limit(u64::MAX, u64::MAX, u64::MAX));
    }

    #[test]
    fn either_shortage_enters_pressure_and_both_must_recover() {
        assert!(!next_pressure(false, 8 * GIB, 8 * GIB));
        assert!(next_pressure(false, 3 * GIB, 8 * GIB));
        assert!(next_pressure(false, 8 * GIB, 4 * GIB));
        // Between the two thresholds: as it was.
        assert!(next_pressure(true, 4 * GIB, 8 * GIB));
        assert!(!next_pressure(false, 4 * GIB, 5 * GIB));
        assert!(next_pressure(true, 8 * GIB, 5 * GIB));
        assert!(!next_pressure(true, 5 * GIB, 6 * GIB));
    }
}
