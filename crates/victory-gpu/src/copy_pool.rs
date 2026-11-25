//! A submission's refreshes, copied on several threads.
//!
//! Bringing host copies of title memory up to date is a memory copy, and
//! one thread copies far below what the memory can move: on an RTX 5070 Ti
//! host (DDR5), the world's ~8,800 scattered 64 KiB pages per frame
//! (550 MB) took 60 ms on one thread, 43 on two and 37 on four, where the
//! memory is saturated. A submission's refreshes ([`queued`]) queue their
//! copies, and [`run`] copies them on the host's shared workers
//! ([`crate::workers`], also those of `crate::texture::in_parallel`) with
//! the submitting thread; it returns once all are done, so what follows
//! sees them all.

use std::cell::RefCell;
#[cfg(test)]
use std::sync::atomic::{AtomicUsize, Ordering};

/// A copy of title memory into host memory: destination, source, bytes.
pub(crate) type Piece = (usize, usize, usize);

thread_local! {
    static QUEUE: RefCell<Option<Vec<Piece>>> = const { RefCell::new(None) };
}

/// Runs `f`; the copies [`copy`] is asked for on this thread meanwhile are
/// not made but returned, for [`run`].
pub(crate) fn queued<R>(f: impl FnOnce() -> R) -> (R, Vec<Piece>) {
    let outer = QUEUE.with(|q| q.replace(Some(Vec::new())));
    let result = f();
    let pieces = QUEUE.with(|q| q.replace(outer)).unwrap_or_default();
    (result, pieces)
}

/// Copies `len` bytes of title memory at `src` into host memory at `dst`
/// ([`crate::host::copy_mapped`]), or queues the copy inside [`queued`].
/// The caller must not read the destination before [`run`] then.
pub(crate) fn copy(dst: *mut u8, src: *const u8, len: usize) {
    let piece = (dst as usize, src as usize, len);
    let queued = QUEUE.with(|q| q.borrow_mut().as_mut().map(|q| q.push(piece)).is_some());
    if !queued {
        crate::host::copy_mapped(dst, src, len, false);
    }
}

/// Copies are cut to this size, so that one big copy is shared too.
const PIECE: usize = 1 << 20;
/// Below this many bytes in all, one thread copies (waking the workers
/// costs tens of microseconds; 4 MiB take about 0.4 ms).
const PARALLEL_FROM: usize = 4 << 20;
/// Threads copying, the submitting one included: four saturated the
/// memory above, eight gained 2%.
const THREADS: usize = 4;

/// Tests: every copy on the calling thread (to compare with the workers).
#[cfg(test)]
pub(crate) static SERIAL: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

fn serial() -> bool {
    #[cfg(test)]
    return SERIAL.load(std::sync::atomic::Ordering::Relaxed);
    #[cfg(not(test))]
    false
}

/// Tests: the pieces the shared workers copied.
#[cfg(test)]
static WORKER_PIECES: AtomicUsize = AtomicUsize::new(0);

/// Makes the copies [`queued`] returned, on several threads where they are
/// many bytes; returns once all are made.
pub(crate) fn run(pieces: Vec<Piece>) {
    let total: usize = pieces.iter().map(|p| p.2).sum();
    if total < PARALLEL_FROM || serial() || crate::workers::started() == 0 {
        for (dst, src, len) in pieces {
            crate::host::copy_mapped(dst as *mut u8, src as *const u8, len, false);
        }
        return;
    }
    let mut cut = Vec::with_capacity(pieces.len() + total / PIECE);
    for (dst, src, len) in pieces {
        let mut at = 0;
        while at < len {
            let n = (len - at).min(PIECE);
            cut.push((dst + at, src + at, n));
            at += n;
        }
    }
    crate::workers::run(THREADS, cut.len(), |i| {
        let (dst, src, len) = cut[i];
        crate::host::copy_mapped(dst as *mut u8, src as *const u8, len, false);
        #[cfg(test)]
        if std::thread::current().name().is_some_and(|n| n.starts_with("victory worker")) {
            WORKER_PIECES.fetch_add(1, Ordering::Relaxed);
        }
    });
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Past the threshold the workers copy: with the submitting thread
    /// leaving them every piece, they copy all of a run's pieces, and every
    /// byte lands where it belongs.
    #[test]
    fn the_workers_take_the_pieces_past_the_threshold() {
        let len = 8 << 20;
        let src: Vec<u8> = (0..len).map(|i| (i * 13 + i / 65536) as u8).collect();
        let mut dst = vec![0u8; len];
        let (s, d) = (src.as_ptr(), dst.as_mut_ptr());
        // 128 pieces of 64 KiB, in another order than in memory.
        let ((), pieces) = queued(|| {
            for k in 0..128usize {
                let at = (k * 37 % 128) * (64 << 10);
                // SAFETY: (both) inside the two buffers.
                copy(unsafe { d.add(at) }, unsafe { s.add(at) }, 64 << 10);
            }
        });
        assert!(crate::workers::started() > 0, "workers started");
        crate::workers::WORKERS_ALONE.with(|a| a.set(true));
        let before = WORKER_PIECES.load(Ordering::Relaxed);
        run(pieces);
        let copied = WORKER_PIECES.load(Ordering::Relaxed) - before;
        crate::workers::WORKERS_ALONE.with(|a| a.set(false));
        // At least: runs of other tests meanwhile count too.
        assert!(copied >= 128, "pieces the workers copied: {copied}");
        assert!(dst == src, "every byte where it belongs");
    }

    /// Queued copies are made by `run`, on the workers past the threshold,
    /// one big copy cut into pieces too, each byte where it belongs.
    #[test]
    fn queued_copies_are_made_on_the_workers() {
        let len = 24 << 20;
        let src: Vec<u8> = (0..len).map(|i| (i * 7 + i / 4096) as u8).collect();
        let mut dst = vec![0u8; len];
        let (s, d) = (src.as_ptr(), dst.as_mut_ptr());
        // Scattered 64 KiB pieces, then one 8 MiB copy.
        let ((), pieces) = queued(|| {
            for k in 0..128usize {
                let at = (k * 37 % 256) * (64 << 10);
                // SAFETY: (both) inside the two buffers.
                copy(unsafe { d.add(at) }, unsafe { s.add(at) }, 64 << 10);
            }
            // SAFETY: inside the two buffers.
            copy(
                unsafe { d.add(16 << 20) },
                unsafe { s.add(16 << 20) },
                8 << 20,
            );
        });
        assert_eq!(pieces.len(), 129, "queued, not made");
        assert!(dst.iter().all(|&b| b == 0));
        run(pieces);
        let mut want = vec![0u8; len];
        for k in 0..128usize {
            let at = (k * 37 % 256) * (64 << 10);
            want[at..at + (64 << 10)].copy_from_slice(&src[at..at + (64 << 10)]);
        }
        want[16 << 20..].copy_from_slice(&src[16 << 20..]);
        assert!(dst == want, "every byte where it belongs");
        // Outside `queued`, a copy is made at once.
        copy(d, s, 16);
        assert_eq!(dst[..16], src[..16]);
        // Again, to reuse the workers.
        // SAFETY: (both) inside the two buffers.
        let (at, from) = unsafe { (d.add(1 << 20), s.add(1 << 20)) };
        let ((), pieces) = queued(|| copy(at, from, 6 << 20));
        dst[1 << 20..7 << 20].fill(0);
        run(pieces);
        assert!(dst[1 << 20..7 << 20] == src[1 << 20..7 << 20]);
    }
}
