//! Title memory the title asked to be write-combining, and what of it the
//! copy-back leaves on the card (`VICTORY_SKIP_WC_BACK=1`, off by default).
//!
//! The console's CPU writes write-combining memory but does not read it back
//! (a read there is uncached and very slow), so what shaders wrote there is
//! for the GPU. The copy-back still fetched it: in the city, per frame, about
//! 800 pieces of 64 KiB (52 MB) of copies in the card's memory were fetched
//! and merged into title memory, most of the "copying back" time. With the
//! switch, pieces of those copies that lie wholly in write-combining memory
//! stay on the card. An experiment, not a fix (port of fxceless/victory-fork
//! #868): it assumes the title's CPU never reads such memory after the card
//! wrote it, and never writes part of a page the card wrote (the next refresh
//! would send the title's older bytes of that page up).
//!
//! What counts is what the title asked for (`XMemVirtualAlloc`'s protection,
//! bit 0x400): the runtime drops that bit for the host.

use std::collections::BTreeMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Mutex, OnceLock};

/// The console protection's write-combining bit.
const WRITE_COMBINE: u32 = 0x400;

/// `VICTORY_SKIP_WC_BACK`: 0 (unset) copies everything back; 1 leaves the
/// pieces of copies in the card's memory there; 2 also takes no snapshot of
/// copies in the PC's memory, so nothing of them is compared or copied back.
fn level() -> u8 {
    static LEVEL: OnceLock<u8> = OnceLock::new();
    *LEVEL.get_or_init(|| {
        let level = match std::env::var("VICTORY_SKIP_WC_BACK").as_deref() {
            Ok("1") => 1,
            Ok("2") => 2,
            _ => 0,
        };
        if level > 0 {
            tracing::warn!(
                "VICTORY_SKIP_WC_BACK={level}: what shaders wrote to copies of write-combining title memory is not \
                 copied back ({}; experiment)",
                if level == 1 { "copies in the card's memory" } else { "copies in the card's memory and in the PC's" }
            );
        }
        level
    })
}

/// Whether pieces of write-combining title memory stay on the card.
pub(crate) fn on() -> bool {
    level() >= 1
}

/// Whether a range shaders write of a copy in the PC's memory, `[from, to)`
/// of title memory, is left to the copy: no snapshot, no copy-back.
pub(crate) fn leave_snapshot(from: usize, to: usize) -> bool {
    if level() < 2 || !runs().all_wc(from, to) {
        return false;
    }
    LEFT[2].fetch_add(1, Ordering::Relaxed);
    LEFT[3].fetch_add((to - from) as u64, Ordering::Relaxed);
    true
}

/// Title memory known to be write-combining, by start: the end of each run.
/// Runs do not overlap; neighbours may touch.
#[derive(Default)]
struct Runs {
    wc: BTreeMap<usize, usize>,
    /// Reservations made at an address of the host's choice, by start: their
    /// length, for a release that names only the start.
    reserved: BTreeMap<usize, usize>,
}

impl Runs {
    /// Nothing of `[from, to)` is write-combining any more.
    fn cut(&mut self, from: usize, to: usize) {
        if from >= to {
            return;
        }
        let over: Vec<(usize, usize)> =
            self.wc.range(..to).rev().take_while(|&(_, &end)| end > from).map(|(&s, &e)| (s, e)).collect();
        for (start, end) in over {
            self.wc.remove(&start);
            if start < from {
                self.wc.insert(start, from);
            }
            if end > to {
                self.wc.insert(to, end);
            }
        }
    }

    /// The title asked for `[from, to)` with protection `protect`.
    fn asked(&mut self, from: usize, to: usize, protect: u32) {
        self.cut(from, to);
        if from < to && protect & WRITE_COMBINE != 0 {
            self.wc.insert(from, to);
        }
    }

    /// Whether all of `[from, to)` is write-combining (runs that touch count
    /// as one).
    fn all_wc(&self, from: usize, to: usize) -> bool {
        if from >= to {
            return false;
        }
        let Some((_, &end)) = self.wc.range(..=from).next_back() else {
            return false;
        };
        let mut at = end;
        if end <= from {
            return false;
        }
        while at < to {
            match self.wc.get(&at) {
                Some(&end) => at = end,
                None => return false,
            }
        }
        true
    }
}

fn runs() -> std::sync::MutexGuard<'static, Runs> {
    static RUNS: Mutex<Runs> = Mutex::new(Runs { wc: BTreeMap::new(), reserved: BTreeMap::new() });
    RUNS.lock().unwrap_or_else(|e| e.into_inner())
}

/// The title's `XMemVirtualAlloc` gave it `[at, at + len)` with the console
/// protection `protect` (as asked, before the runtime cleaned it);
/// `reserved`: a reservation at an address of the host's choice.
pub fn title_memory_protected(at: usize, len: usize, protect: u32, reserved: bool) {
    if !on() || at == 0 || len == 0 {
        return;
    }
    let mut r = runs();
    r.asked(at, at.saturating_add(len), protect);
    if reserved {
        r.reserved.insert(at, len);
    }
}

/// The title let `[at, at + len)` go (`len` 0: the reservation that starts at
/// `at`, whole).
pub fn title_memory_released(at: usize, len: usize) {
    if !on() {
        return;
    }
    let mut r = runs();
    let len = if len == 0 { r.reserved.remove(&at).unwrap_or(0) } else { len };
    r.cut(at, at.saturating_add(len));
}

/// Per frame: pieces left on the card and their bytes; ranges of copies in
/// the PC's memory left to them and their bytes.
pub(crate) static LEFT: [AtomicU64; 4] = [const { AtomicU64::new(0) }; 4];

/// Whether the piece `[from, to)` of title memory stays on the card: the
/// switch is on and it is all write-combining.
pub(crate) fn leave(from: usize, to: usize) -> bool {
    if !on() || !runs().all_wc(from, to) {
        return false;
    }
    LEFT[0].fetch_add(1, Ordering::Relaxed);
    LEFT[1].fetch_add((to - from) as u64, Ordering::Relaxed);
    true
}

#[cfg(test)]
mod tests {
    use super::Runs;

    const MIB: usize = 1 << 20;

    #[test]
    fn only_memory_asked_write_combining_counts() {
        let mut r = Runs::default();
        r.asked(0x1000_0000, 0x1000_0000 + 4 * MIB, 0x2404);
        r.asked(0x2000_0000, 0x2000_0000 + 4 * MIB, 0x2004);
        assert!(r.all_wc(0x1000_0000, 0x1001_0000));
        assert!(r.all_wc(0x1000_0000, 0x1000_0000 + 4 * MIB));
        assert!(!r.all_wc(0x1000_0000, 0x1000_0000 + 4 * MIB + 1));
        assert!(!r.all_wc(0x0fff_ffff, 0x1000_0010));
        assert!(!r.all_wc(0x2000_0000, 0x2001_0000));
        assert!(!r.all_wc(0x3000_0000, 0x3001_0000));
        assert!(!r.all_wc(0x1000_0000, 0x1000_0000));
    }

    #[test]
    fn ordinary_memory_inside_a_write_combining_run_cuts_it() {
        let mut r = Runs::default();
        let base = 0x1_0000_0000;
        r.asked(base, base + 8 * MIB, 0x404);
        // A commit of the middle as ordinary memory.
        r.asked(base + 2 * MIB, base + 3 * MIB, 0x04);
        assert!(r.all_wc(base, base + 2 * MIB));
        assert!(r.all_wc(base + 3 * MIB, base + 8 * MIB));
        assert!(!r.all_wc(base + 2 * MIB - 0x10000, base + 2 * MIB + 0x10000));
        assert!(!r.all_wc(base + 2 * MIB, base + 2 * MIB + 0x10000));
        assert!(!r.all_wc(base, base + 8 * MIB));
        // Made write-combining again: the runs touch and count as one.
        r.asked(base + 2 * MIB, base + 3 * MIB, 0x404);
        assert!(r.all_wc(base, base + 8 * MIB));
    }

    #[test]
    fn released_memory_is_no_longer_write_combining() {
        let mut r = Runs::default();
        let base = 0x2_0000_0000;
        r.asked(base, base + 4 * MIB, 0x2404);
        r.reserved.insert(base, 4 * MIB);
        // A release that names only the start lets the whole reservation go.
        let len = r.reserved.remove(&base).unwrap_or(0);
        r.cut(base, base + len);
        assert!(!r.all_wc(base, base + 0x10000));
        assert!(!r.all_wc(base + 4 * MIB - 0x10000, base + 4 * MIB));
        assert!(r.wc.is_empty());
        // A part let go leaves the rest.
        r.asked(base, base + 4 * MIB, 0x2404);
        r.cut(base + MIB, base + 2 * MIB);
        assert!(r.all_wc(base, base + MIB));
        assert!(!r.all_wc(base + MIB, base + MIB + 0x10000));
        assert!(r.all_wc(base + 2 * MIB, base + 4 * MIB));
    }

    #[test]
    fn a_run_asked_again_over_several_replaces_them() {
        let mut r = Runs::default();
        let base = 0x3_0000_0000;
        r.asked(base, base + MIB, 0x404);
        r.asked(base + 2 * MIB, base + 3 * MIB, 0x404);
        r.asked(base + MIB / 2, base + 5 * MIB / 2, 0x4);
        assert_eq!(r.wc.len(), 2);
        assert!(r.all_wc(base, base + MIB / 2));
        assert!(!r.all_wc(base, base + MIB));
        assert!(r.all_wc(base + 5 * MIB / 2, base + 3 * MIB));
        assert!(!r.all_wc(base + 2 * MIB, base + 3 * MIB));
    }
}
