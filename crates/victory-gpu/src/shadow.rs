//! Host copies of title buffers (decision 0007, step 3 and "Copies of title
//! memory"). Title buffers are title memory, which the host GPU cannot
//! read, so a buffer a draw uses gets a persistent host buffer of its size
//! on first use (system memory shaders may also write), and the ranges each
//! submission reads (vertex, index and constant data, buffer views) are
//! brought up to date from title memory before it runs: the 64 KiB pages
//! the title wrote since they were last copied ([`crate::watch`]: read-only
//! guards and the runtime's reports), all of it where the memory is not the
//! runtime's. A title GPU address inside the buffer is the same offset
//! in its copy. Ranges shaders write go back to title memory after the
//! submission ([`Shadow::gpu_writes`]).
//!
//! The ranges brought up to date are those draws are known to read (vertex
//! and index buffer views, constant buffers in tables, root descriptors,
//! buffer views in tables) and, for shaders that index the heap
//! themselves, every buffer range the heap's views reach
//! (`crate::bindless`).

use std::collections::BTreeMap;
use std::sync::{Arc, Mutex, RwLock};

use windows::Win32::Graphics::Direct3D12::ID3D12Resource;

use crate::host::{HostList, host};
use crate::watch::PAGE;

/// A title buffer's host copy.
pub(crate) struct Shadow {
    pub(crate) buffer: ID3D12Resource,
    data: *mut u8,
    gpu: u64,
    /// Where the title buffer starts and how long it is.
    start: u64,
    len: u64,
    /// The pages changed since the copy took them (none: the memory is not
    /// the runtime's, copied whole each time).
    tracked: Option<crate::watch::Tracked>,
    /// Ranges shaders of a submission not yet copied back may be writing:
    /// title memory there is older than the copy.
    pending: Mutex<std::collections::HashSet<(usize, usize)>>,
    /// The copy's bytes before the submission ran where its shaders write:
    /// what the GPU wrote differs. Disjoint segments (offset, bytes) in
    /// order, and the 64 KiB pieces (by title address) already compared.
    before: Mutex<Snapshot>,
    /// Its first bit in the buffer views' written-pages bitmap, one per
    /// 64 KiB title page it touches (tracked memory only): writes through
    /// views of it are marked there ([`Shadow::mark_entry`]).
    marks: Option<u32>,
    /// Where it starts in `buffer` (0 but for copies sharing a buffer), and
    /// the run of the shared buffer it holds ([`Arena`]), given back when it
    /// goes.
    base: u64,
    slice: Option<Slice>,
}

/// Snapshot segments (offset, bytes) and the pieces compared.
type Snapshot = (Vec<(usize, Vec<u8>)>, std::collections::HashSet<usize>);

/// Snapshot memory kept for reuse (already touched, so no page faults).
static POOL: Mutex<Vec<Vec<u8>>> = Mutex::new(Vec::new());

/// An empty buffer with room for `n` bytes, from the pool if it has one.
fn pooled(n: usize) -> Vec<u8> {
    let mut pool = POOL.lock().unwrap_or_else(|e| e.into_inner());
    let fit = pool
        .iter()
        .enumerate()
        .filter(|(_, v)| v.capacity() >= n)
        .min_by_key(|(_, v)| v.capacity())
        .map(|(i, _)| i);
    let mut v = match fit {
        Some(i) => pool.swap_remove(i),
        None => Vec::with_capacity(n),
    };
    v.clear();
    v
}

/// Returns a snapshot buffer to the pool, which keeps the largest up to
/// [`POOL_BYTES`] in all: a world submission snapshots several copies, and
/// fresh buffers cost a page fault per 4 KiB touched (millions per frame).
fn give_back(v: Vec<u8>) {
    let mut pool = POOL.lock().unwrap_or_else(|e| e.into_inner());
    pool.push(v);
    pool.sort_by_key(|v| std::cmp::Reverse(v.capacity()));
    let mut total = 0;
    pool.retain(|v| {
        total += v.capacity();
        total <= POOL_BYTES
    });
}

/// Snapshot memory kept for reuse.
const POOL_BYTES: usize = 512 << 20;

// SAFETY: D3D12 resources are free-threaded; `data` is a persistent mapping
// written only on the submitting thread.
unsafe impl Send for Shadow {}
// SAFETY: as above.
unsafe impl Sync for Shadow {}

impl Shadow {
    /// The host GPU address of a title address inside the buffer.
    pub(crate) fn gpu(&self, address: u64) -> u64 {
        self.gpu + (address - self.start)
    }

    /// Bytes from a title address to the buffer's end.
    pub(crate) fn left(&self, address: u64) -> u64 {
        (self.start + self.len).saturating_sub(address)
    }

    /// `n` dwords of the copy at title address `address` (tests).
    #[cfg(test)]
    pub(crate) fn read_host(&self, address: u64, n: usize) -> Vec<u32> {
        let at = (address - self.start) as usize;
        assert!(at + 4 * n <= self.len as usize);
        // SAFETY: inside the mapping (checked); the GPU is done with it.
        unsafe { std::slice::from_raw_parts(self.data.add(at).cast::<u32>(), n).to_vec() }
    }

    /// Writes `bytes` into the copy at title address `address`, as shaders
    /// do (tests).
    #[cfg(test)]
    pub(crate) fn write_host(&self, address: u64, bytes: &[u8]) {
        let at = (address - self.start) as usize;
        assert!(at + bytes.len() <= self.len as usize);
        // SAFETY: inside the mapping (checked).
        unsafe { std::ptr::copy_nonoverlapping(bytes.as_ptr(), self.data.add(at), bytes.len()) };
    }

    /// Up to 64 bytes of the copy at title address `address`, as floats, and
    /// which copy it is (diagnostics: what shaders see there).
    #[cfg(feature = "diagnostics")]
    pub(crate) fn peek(&self, address: u64, len: u64) -> String {
        let at = address.saturating_sub(self.start) as usize;
        let n = (len.min(64) as usize).min((self.len as usize).saturating_sub(at)) / 4;
        // SAFETY: inside the mapping (checked); a racing GPU write only makes
        // the logged value stale.
        let v = unsafe { std::slice::from_raw_parts(self.data.add(at).cast::<f32>(), n) };
        format!("copy {:#x}+{:#x} {v:?}", self.start, self.len)
    }

    /// Offset of a title address from the buffer's start.
    pub(crate) fn offset(&self, address: u64) -> u64 {
        address - self.start + self.base
    }

    /// Where the title memory it copies starts, and its length.
    pub(crate) fn range(&self) -> (u64, u64) {
        (self.start, self.len)
    }

    /// After a submission whose shaders wrote title memory at addresses they
    /// computed (`crate::memory`), somewhere in the 64 KiB page at `page`:
    /// the bytes of the page (cut to the copy) that differ from title memory
    /// go back into it, when the title has not written the page since the
    /// copy took it (then title memory is the copy as it was before the
    /// submission, and what differs is what the GPU wrote). Returns the
    /// bytes written; a page the title wrote meanwhile keeps the title's
    /// bytes (logged once).
    // shortcut: a page the copy holds only in part (at its ends) has no
    // "copied" mark and is compared without the title-write check.
    pub(crate) fn copy_back_page(&self, page: usize) -> usize {
        let (start, end) = (self.start as usize, (self.start + self.len) as usize);
        let (from, to) = (page.max(start), (page + PAGE).min(end));
        if from >= to {
            return 0;
        }
        if let Some(t) = &self.tracked
            && t.changed(page / PAGE)
        {
            static ONCE: std::sync::Once = std::sync::Once::new();
            ONCE.call_once(|| {
                tracing::warn!(
                    page = format_args!("{page:#x}"),
                    "the title and translated shaders both wrote a page of title memory in one submission: the title's bytes kept (logged once)"
                )
            });
            return 0;
        }
        let n = to - from;
        // SAFETY: inside the mapping; only the submitting thread writes it.
        let host = unsafe { std::slice::from_raw_parts(self.data.add(from - start), n) };
        let mut title = vec![0u8; n];
        if !crate::host::read_into(from as *const u8, &mut title) || title == host {
            return 0;
        }
        #[cfg(test)]
        tests::before_copy_back_write();
        if !crate::watch::write_back(self.tracked.as_ref(), from, n, || {
            crate::host::write_mapped(from as *mut u8, host.as_ptr(), n);
        }) {
            return 0;
        }
        // Our write is not the title's: the copy holds the page as it is.
        if let Some(t) = &self.tracked {
            t.taken(&[page / PAGE]);
        }
        n
    }

    /// Brings `len` bytes at `address` up to date when `list` is submitted,
    /// and keeps the copy alive until the list is reset.
    pub(crate) fn refresh(self: &Arc<Self>, list: &HostList, address: u64, len: u64) {
        let len = len.min(self.start + self.len - address);
        list.defer_refresh(self, address as usize, len as usize);
    }

    fn new(buffer: ID3D12Resource, data: *mut u8, gpu: u64, start: u64, len: u64) -> Arc<Shadow> {
        Shadow::new_at(buffer, data, gpu, start, len, None)
    }

    /// [`Shadow::new`] of a copy in a run of a shared buffer (`data` and
    /// `gpu` are the run's).
    fn new_at(buffer: ID3D12Resource, data: *mut u8, gpu: u64, start: u64, len: u64, slice: Option<Slice>) -> Arc<Shadow> {
        let tracked = crate::watch::track_copy(start as usize, len as usize, true);
        let marks = tracked.as_ref().and_then(|_| marks().take(pages(start, len)));
        let shadow = Arc::new(Shadow {
            buffer,
            data,
            gpu,
            start,
            len,
            tracked,
            pending: Mutex::default(),
            before: Mutex::default(),
            marks,
            base: slice.map_or(0, |s| s.offset),
            slice,
        });
        if let Some(base) = marks {
            self::marks().owners.insert(base, Arc::downgrade(&shadow));
        }
        shadow
    }

    /// The view-mark entry (`victory_shader::emit`, "Write marks") of a
    /// view of this copy whose first byte is at title address `start`: the
    /// bit of that byte's page and its offset in the page. None where
    /// writes through it are not marked (memory not tracked, or no bits
    /// left): such views declare what they write ([`Shadow::gpu_writes`]).
    pub(crate) fn mark_entry(&self, start: u64) -> Option<(u32, u32)> {
        let base = self.marks?;
        let page = (start >> 16).checked_sub(self.start >> 16)?;
        Some((base + u32::try_from(page).ok()?, (start & 0xFFFF) as u32))
    }

    /// Whether writes through views of this copy are marked.
    pub(crate) fn marked(&self) -> bool {
        self.marks.is_some()
    }

    /// `[address, address + len)` cut to the buffer, as offsets in the copy.
    fn inside(&self, address: usize, len: usize) -> (usize, usize) {
        let (start, end) = (self.start as usize, (self.start + self.len) as usize);
        let from = address.clamp(start, end);
        (from - start, (address + len).clamp(from, end) - start)
    }

    /// Shaders of `list` may write `[address, address + len)` of the copy
    /// (unordered access): after the list's submission has run, the range
    /// is copied back into title memory, where the title reads it; until
    /// then no refresh copies title memory over it. Call after the
    /// [`Shadow::refresh`] that gives the shaders the title's data.
    ///
    /// shortcut: what shaders wrote is found by keeping the range's bytes
    /// before the submission (a copy per declared range and submission);
    /// a GPU-side dirty mask would save that if big ranges make it slow.
    ///
    /// `marked`: the shaders mark the pages they write through this copy's
    /// views (`victory_shader::emit`, "Write marks"): no snapshot then, only
    /// the pages they marked are compared ([`copy_back_marks`]).
    pub(crate) fn gpu_writes(self: &Arc<Self>, list: &HostList, address: u64, len: u64, marked: bool) {
        let len = len.min(self.left(address));
        list.defer_gpu_write(self, address as usize, len as usize, marked && self.marked());
    }

    /// At submission, after the range's refresh: a submitted list's shaders
    /// write this range. Its bytes are kept to tell later what the GPU wrote:
    /// per copy and submission, segments covering the ranges the submission
    /// declares (the world declares overlapping views of its 174 MB heap
    /// dozens of times; one segment over all of them copied most of the
    /// heap), in reused memory (fresh snapshots cost ~2.3 million page
    /// faults per world frame).
    ///
    /// With `marked`, nothing is kept: the pages are armed instead, so that
    /// a title write there before the copy-back keeps the page's bytes from
    /// before it ([`crate::watch::Tracked::arm`]).
    pub(crate) fn writing(&self, address: usize, len: usize, marked: bool) {
        self.pending
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .insert((address, address + len));
        if marked && let Some(t) = &self.tracked {
            t.arm(address, address + len);
            // A page left changed by an earlier partial copy-back gets no new
            // title-side snapshot: its first write was already recorded. Keep
            // this submission's host bytes instead, so its later GPU writes
            // remain distinguishable from the title's.
            let (start, end) = (self.start as usize, (self.start + self.len) as usize);
            for page in address / PAGE..(address + len).div_ceil(PAGE) {
                if !t.changed(page) {
                    continue;
                }
                let (from, to) = ((page * PAGE).max(start), ((page + 1) * PAGE).min(end));
                let mut bytes = vec![0; PAGE].into_boxed_slice();
                // SAFETY: `from..to` is inside this copy's persistent mapping.
                let host = unsafe {
                    std::slice::from_raw_parts(self.data.add(from - start), to - from)
                };
                bytes[from - page * PAGE..to - page * PAGE].copy_from_slice(host);
                t.save_host(page, bytes);
            }
            return;
        }
        let (from, to) = self.inside(address, len);
        if from >= to {
            return;
        }
        // SAFETY: inside the mapping.
        let host = |a: usize, b: usize| unsafe { std::slice::from_raw_parts(self.data.add(a), b - a) };
        let mut before = self.before.lock().unwrap_or_else(|e| e.into_inner());
        let segments = &mut before.0;
        // The segments the range overlaps or touches.
        let first = segments.partition_point(|(at, bytes)| at + bytes.len() < from);
        let last = first + segments[first..].iter().take_while(|(at, _)| *at <= to).count();
        if let [(at, bytes)] = &segments[first..last]
            && *at <= from
            && to <= at + bytes.len()
        {
            return;
        }
        let start = segments.get(first).filter(|_| first < last).map_or(from, |s| s.0.min(from));
        let end = segments[first..last].last().map_or(to, |(at, bytes)| (at + bytes.len()).max(to));
        #[cfg(feature = "diagnostics")]
        let _timed = crate::watch::Timed(std::time::Instant::now(), crate::copystat::C::SnapshotUs);
        #[cfg(feature = "diagnostics")]
        crate::copystat::add(crate::copystat::C::SnapshotBytes, (end - start) as u64);
        #[cfg(feature = "diagnostics")]
        crate::copystat::snapshot(self.start, self.len, (end - start) as u64);
        // Parts newly covered from the copy, the rest as it was taken.
        let mut grown = pooled(end - start);
        let mut at = start;
        for (taken, bytes) in segments.drain(first..last) {
            grown.extend_from_slice(host(at, taken));
            grown.extend_from_slice(&bytes);
            at = taken + bytes.len();
            give_back(bytes);
        }
        grown.extend_from_slice(host(at, end));
        segments.insert(first, (start, grown));
    }

    /// `[address, address + len)` expanded outward to 64 KiB page
    /// boundaries and cut to the copy (`[start, start + len)`): whole pages
    /// so [`crate::watch::Tracked::take`] guards them for
    /// [`copy_back_marks`].
    pub(crate) fn page_range(&self, address: u64, len: u64) -> (u64, u64) {
        let end = self.start.saturating_add(self.len);
        let from = address.clamp(self.start, end);
        let to = address.saturating_add(len).clamp(from, end);
        if from >= to {
            return (from, 0);
        }
        let page = PAGE as u64;
        let from = (from & !(page - 1)).max(self.start);
        let to = to.checked_next_multiple_of(page).unwrap_or(end).min(end);
        (from, to - from)
    }

    /// After the submission ran: copies the bytes its shaders wrote back
    /// into title memory (unmapped pages left alone), and only those: the
    /// title's threads write the same buffers while the GPU runs (its
    /// command memory shares the big GPU heap with what shaders write), and
    /// putting back the copy's older bytes there broke the title's own
    /// records (a fence list read as -1, title+0x4e9b54a). Returns the bytes
    /// written.
    /// Withheld ranges stay in the host copy.
    pub(crate) fn copy_back(
        &self,
        address: usize,
        len: usize,
        withheld: &[(usize, usize)],
    ) -> usize {
        self.copy_back_or_not(address, len, Some(withheld))
    }

    /// After the submission ran: what its shaders wrote in the range stays
    /// in the copy and does not go into title memory (they computed it
    /// from memory with no copy, `crate::memory`).
    pub(crate) fn keep_writes(&self, address: usize, len: usize) {
        self.copy_back_or_not(address, len, None);
    }

    fn copy_back_or_not(
        &self,
        address: usize,
        len: usize,
        withheld: Option<&[(usize, usize)]>,
    ) -> usize {
        let mut snapshot = self.before.lock().unwrap_or_else(|e| e.into_inner());
        let (segments, compared) = &mut *snapshot;
        let mut written = 0;
        let (from, to) = self.inside(address, len);
        let start = self.start as usize;
        if let Some(withheld) = withheld
            && let Some((taken, before)) = segments
                .iter()
                .find(|(at, bytes)| *at <= from && to <= at + bytes.len())
            && from < to
        {
            let before = &before[from - taken..to - taken];
            #[cfg(feature = "diagnostics")]
            let _timed = crate::watch::Timed(std::time::Instant::now(), crate::copystat::C::CompareUs);
            // By 64 KiB piece: the copy compared with its bytes before the
            // submission, title memory read only where the GPU wrote.
            const PIECE: usize = 64 << 10;
            let mut title = vec![0u8; PIECE];
            let mut synced = Vec::new();
            let whole = [(start + from, start + to)];
            let clipped;
            let ranges = if withheld.is_empty() {
                &whole[..]
            } else {
                clipped = outside_withheld(start + from, start + to, withheld);
                &clipped
            };
            for &(allowed_from, allowed_to) in ranges {
                let mut at = allowed_from - start;
                let to = allowed_to - start;
                while at < to {
                    let n = (PIECE - (start + at) % PIECE).min(to - at);
                    // SAFETY: inside the mapping; only the submitting thread
                    // writes it.
                    let host = unsafe { std::slice::from_raw_parts_mut(self.data.add(at), n) };
                    let old = &before[at - from..at - from + n];
                    let title = &mut title[..n];
                    // A whole piece is compared once per submission, whichever
                    // declared ranges overlap it: then what the GPU wrote there
                    // went back.
                    if n == PIECE && !compared.insert(start + at) {
                        at += n;
                        continue;
                    }
                    if self
                        .tracked
                        .as_ref()
                        .is_some_and(|t| t.was_remapped((start + at) / PAGE))
                    {
                        at += n;
                        continue;
                    }
                    #[cfg(feature = "diagnostics")]
                    crate::copystat::add(crate::copystat::C::CompareBytes, n as u64);
                    if host != old && crate::host::read_into((start + at) as *const u8, title) {
                        // Title memory with the GPU's bytes put in; the copy
                        // takes the title's bytes in turn.
                        let mut changed = false;
                        for ((t, h), o) in title.iter_mut().zip(host.iter_mut()).zip(old) {
                            if *h != *o && *t != *h {
                                *t = *h;
                                changed = true;
                            }
                            *h = *t;
                        }
                        #[cfg(test)]
                        tests::before_copy_back_write();
                        // Under the remap lock: a page remapped meanwhile
                        // keeps its new bytes.
                        let wrote = if changed {
                            crate::watch::write_back(self.tracked.as_ref(), start + at, n, || {
                                crate::host::write_mapped((start + at) as *mut u8, title.as_ptr(), n);
                            })
                        } else {
                            self.tracked
                                .as_ref()
                                .is_none_or(|t| !t.was_remapped((start + at) / PAGE))
                        };
                        if changed && wrote {
                            written += n;
                        }
                        if wrote && n == PIECE {
                            synced.push((start + at) / PAGE);
                        }
                    }
                    at += n;
                }
            }
            // Only whole pages made equal to title memory become current.
            if let Some(t) = &self.tracked {
                t.taken(&synced);
            }
        }
        // Every submitted step that declared the range is done with it (one
        // submission runs at a time), and the snapshot with the last range.
        let mut pending = self.pending.lock().unwrap_or_else(|e| e.into_inner());
        pending.remove(&(address, address + len));
        if pending.is_empty() {
            compared.clear();
            for (_, bytes) in segments.drain(..) {
                give_back(bytes);
            }
            if let Some(t) = &self.tracked {
                t.disarm();
            }
        }
        written
    }

    /// Prepares `page` of this copy to write back what shaders marked as
    /// written through its views: skips unarmed pages the title already
    /// wrote, and arms the others, so that a title write before the commit
    /// keeps the page's bytes from before it ([`MarkedSlice::commit`] reads
    /// title memory then, as the baseline where nothing was kept). Before
    /// any `watch::writing` opens shared pages: a page of another copy (or
    /// another mapping) the commits write saves its baseline that way.
    /// `withheld` ranges stay in the copy and out of title memory, without
    /// suppressing the page's other writes.
    // shortcut: a page the title wrote that was not armed (a write through
    // the heap's views no list declared) keeps the title's bytes, losing
    // the shaders' there (counted).
    fn prepare_marked(&self, page: usize, withheld: Vec<(usize, usize)>) -> Option<MarkedSlice<'_>> {
        let t = self.tracked.as_ref()?;
        let (start, end) = (self.start as usize, (self.start + self.len) as usize);
        let (from, to) = (page.max(start), (page + PAGE).min(end));
        if from >= to {
            return None;
        }
        if !t.has_saved(page / PAGE) && t.changed(page / PAGE) {
            t.unarm(page / PAGE);
            #[cfg(feature = "diagnostics")]
            crate::copystat::add(crate::copystat::C::MarkedKept, 1);
            return None;
        }
        t.arm(page, page + PAGE);
        Some(MarkedSlice {
            copy: self,
            page,
            from,
            to,
            withheld,
        })
    }

    /// Whether the title wrote none of the copy's memory since the copy took
    /// it (tracked memory only): [`Shadow::update`] would copy nothing.
    pub(crate) fn clean(&self) -> bool {
        self.tracked.as_ref().is_some_and(crate::watch::Tracked::clean)
    }

    /// Its tracker's bit: `crate::watch::clean_bit` of it is
    /// [`Shadow::clean`] (none for memory that is not tracked: never clean).
    pub(crate) fn dirty_bit(&self) -> Option<u32> {
        self.tracked.as_ref().and_then(crate::watch::Tracked::bit)
    }

    /// Advances any snapshot segments covering `[offset, offset + data.len())`
    /// to `data`, so a later declared copy-back neither takes those bytes for
    /// the GPU's nor puts them back over a later title write.
    fn sync_snapshot(&self, offset: usize, data: &[u8]) {
        let mut before = self.before.lock().unwrap_or_else(|e| e.into_inner());
        let end = offset + data.len();
        for (at, bytes) in before.0.iter_mut() {
            let (s, e) = ((*at).max(offset), (*at + bytes.len()).min(end));
            if s < e {
                bytes[s - *at..e - *at].copy_from_slice(&data[s - offset..e - offset]);
            }
        }
    }

    /// At submission: copies what changed of `[address, address + len)`
    /// since it was last copied (all of it if the memory is not
    /// write-watched). Returns the bytes copied.
    pub(crate) fn update(&self, address: usize, len: usize) -> usize {
        if self.clean() {
            return 0;
        }
        let (start, end) = (self.start as usize, (self.start + self.len) as usize);
        let (from, to) = (address.max(start), (address + len).min(end));
        if from >= to {
            return 0;
        }
        let mut busy: Vec<(usize, usize)> = self
            .pending
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .iter()
            .copied()
            .filter(|&(a, e)| a < to && e > from)
            .collect();
        if self.tracked.is_some() {
            return self.update_tracked(from, to, !busy.is_empty());
        }
        if !busy.is_empty() {
            // Around ranges shaders are writing: the gaps, each as below
            // (in the world most submissions have such ranges; copying the
            // gaps whole cost ~1 s per frame). Under them nothing: what the
            // title wrote there is not known.
            busy.sort_unstable();
            let (mut at, mut bytes) = (from, 0);
            for (a, e) in busy {
                let (a, e) = (a.max(from), e.min(to));
                if a > at {
                    bytes += self.update_untracked(at, a);
                }
                at = at.max(e);
            }
            if at < to {
                bytes += self.update_untracked(at, to);
            }
            return bytes;
        }
        self.update_untracked(from, to)
    }

    /// [`Shadow::update`] of `[from, to)` (inside the buffer) where the
    /// memory is tracked: the pages the title wrote since they were taken,
    /// in one [`crate::watch::Tracked::take`], so that a page wholly in the
    /// range is guarded even where shaders write part of it (split at those
    /// ranges it stayed changed, and [`copy_back_marks`] dropped
    /// what they wrote).
    ///
    /// `busy`: a list submitted earlier declared that shaders write in the
    /// range. A submission waits for the one before it, so none of this
    /// submission's shaders has run yet: the pages the title wrote are
    /// copied as anywhere else. Skipped, the title's constants there stayed
    /// stale for the whole submission (a dispatch read its sizes as zeros
    /// and looped until the device was removed). The same bytes go into the
    /// snapshot `writing` took, so copy-back neither takes them for the
    /// GPU's nor puts them back over a later title write.
    fn update_tracked(&self, from: usize, to: usize, busy: bool) -> usize {
        let Some(t) = &self.tracked else {
            return 0;
        };
        #[cfg(feature = "diagnostics")]
        let started = std::time::Instant::now();
        let (runs, unmapped) = t.take_split(from, to);
        #[cfg(feature = "diagnostics")]
        crate::copystat::add(crate::copystat::C::WatchUs, started.elapsed().as_micros() as u64);
        let start = self.start as usize;
        let mut before = busy.then(|| self.before.lock().unwrap_or_else(|e| e.into_inner()));
        // Copies `[a, b)` (`copy`) or takes zeros written there, into the
        // snapshot over it. A copy with no snapshot over it (none is kept
        // where nothing is busy, nor for marked ranges) may be queued
        // (`crate::copy_pool`): nothing reads those bytes before the flush
        // makes it.
        let mut sync_before = |a: usize, b: usize, copy: bool| -> usize {
            let (ra, rb) = (a - start, b - start);
            let covered = before.as_ref().is_some_and(|before| {
                before
                    .0
                    .iter()
                    .any(|(at, bytes)| *at < rb && ra < *at + bytes.len())
            });
            if copy && !covered {
                return self.copy_queued(a, b);
            }
            let copied = if copy { self.copy(a, b) } else { 0 };
            let Some(before) = &mut before else { return copied };
            let (a, b) = (ra, rb);
            for (at, bytes) in before.0.iter_mut() {
                let (s, e) = ((*at).max(a), (*at + bytes.len()).min(b));
                if s < e {
                    // SAFETY: inside the mapping.
                    let host = unsafe { std::slice::from_raw_parts(self.data.add(s), e - s) };
                    bytes[s - *at..e - *at].copy_from_slice(host);
                }
            }
            copied
        };
        let mut copied = 0;
        for (a, b) in runs {
            copied += sync_before(a, b, true);
        }
        for (a, b) in unmapped {
            // SAFETY: inside the mapping (`take_split` cuts to the buffer).
            unsafe { self.data.add(a - start).write_bytes(0, b - a) };
            sync_before(a, b, false);
            #[cfg(feature = "diagnostics")]
            crate::copystat::add(crate::copystat::C::Unmapped, (b - a) as u64);
        }
        // Parts on memory that is not the runtime's: whole, as below.
        for (a, b) in t.untracked(from, to) {
            #[cfg(feature = "diagnostics")]
            let started = std::time::Instant::now();
            copied += sync_before(a, b, true);
            #[cfg(feature = "diagnostics")]
            {
                use crate::copystat::{C, add};
                add(C::WholeBytes, (b - a) as u64);
                add(C::WholeUs, started.elapsed().as_micros() as u64);
            }
        }
        #[cfg(feature = "diagnostics")]
        {
            use crate::copystat::{C, add};
            add(C::Watches, 1);
            add(C::Checked, (to - from) as u64);
            add(C::Unchanged, (to - from).saturating_sub(copied) as u64);
        }
        copied
    }

    /// [`Shadow::update`] of untracked `[from, to)` (inside the buffer).
    fn update_untracked(&self, from: usize, to: usize) -> usize {
        #[cfg(feature = "diagnostics")]
        let started = std::time::Instant::now();
        self.copy_queued(from, to);
        #[cfg(feature = "diagnostics")]
        {
            use crate::copystat::{C, add};
            add(C::WholeBytes, (to - from) as u64);
            add(C::WholeUs, started.elapsed().as_micros() as u64);
        }
        to - from
    }

    /// As [`Shadow::copy`], queued where the caller collects a
    /// submission's refreshes ([`crate::copy_pool::queued`]): for copies
    /// nothing reads back before they are made.
    fn copy_queued(&self, from: usize, to: usize) -> usize {
        let (start, end) = (self.start as usize, (self.start + self.len) as usize);
        let (from, to) = (from.max(start), to.min(end));
        // SAFETY: inside the mapping (`from` and `to` are within the buffer).
        let at = unsafe { self.data.add(from - start) };
        crate::copy_pool::copy(at, from as *const u8, to - from);
        to - from
    }

    /// Copies title memory `[from, to)`, cut to the buffer, into the copy.
    fn copy(&self, from: usize, to: usize) -> usize {
        let (start, end) = (self.start as usize, (self.start + self.len) as usize);
        let (from, to) = (from.max(start), to.min(end));
        // SAFETY: inside the mapping (`from` and `to` are within the buffer).
        let at = unsafe { self.data.add(from - start) };
        crate::host::copy_mapped(at, from as *const u8, to - from, false);
        to - from
    }
}

impl Drop for Shadow {
    fn drop(&mut self) {
        if let Some(slice) = self.slice {
            arena().give_back(slice);
        }
        if let Some(base) = self.marks {
            let mut m = marks();
            m.owners.remove(&base);
            m.give_back(base, pages(self.start, self.len));
        }
    }
}

/// The 64 KiB title pages `[start, start + len)` touches.
fn pages(start: u64, len: u64) -> u32 {
    ((start + len.max(1) - 1) >> 16).saturating_sub(start >> 16) as u32 + 1
}

/// The bits of the buffer views' written-pages bitmap (`victory_shader::
/// emit`, "Write marks"): a run per tracked copy, a bit per title page it
/// touches, for as long as it lives.
#[derive(Default)]
struct Marks {
    /// Free runs below `end`, by first bit: their length.
    free: BTreeMap<u32, u32>,
    /// Bits handed out at some time: those past it are clear.
    end: u32,
    /// The copy holding each run, by its first bit.
    owners: BTreeMap<u32, std::sync::Weak<Shadow>>,
}

static MARKS: Mutex<Marks> = Mutex::new(Marks {
    free: BTreeMap::new(),
    end: 0,
    owners: BTreeMap::new(),
});

fn marks() -> std::sync::MutexGuard<'static, Marks> {
    MARKS.lock().unwrap_or_else(|e| e.into_inner())
}

impl Marks {
    /// A run of `n` bits: its first. None when the bitmap is full (the last
    /// bit takes the marks of offsets past it, so it is never handed out).
    fn take(&mut self, n: u32) -> Option<u32> {
        if let Some((&at, &len)) = self.free.iter().find(|&(_, &len)| len >= n) {
            self.free.remove(&at);
            if len > n {
                self.free.insert(at + n, len - n);
            }
            return Some(at);
        }
        if self.end + n < victory_shader::emit::VIEW_WRITE_BITS {
            self.end += n;
            return Some(self.end - n);
        }
        static ONCE: std::sync::Once = std::sync::Once::new();
        ONCE.call_once(|| tracing::warn!("no bits left in the views' written-pages bitmap: copies made now declare their shaders' writes"));
        None
    }

    /// The run of `n` bits at `at` is free again (joined to its neighbours).
    fn give_back(&mut self, mut at: u32, mut n: u32) {
        if let Some((&before, &len)) = self.free.range(..at).next_back()
            && before + len == at
        {
            self.free.remove(&before);
            (at, n) = (before, n + len);
        }
        if let Some(len) = self.free.remove(&(at + n)) {
            n += len;
        }
        self.free.insert(at, n);
    }
}

/// The buffer views' written-pages bitmap: the buffer, its mapping (bits
/// as dwords) and host GPU address.
struct Bitmap {
    _buffer: ID3D12Resource,
    data: usize,
    gpu: u64,
}

// SAFETY: D3D12 resources are free-threaded; the mapping is read and cleared
// only between submissions, which run one at a time.
unsafe impl Send for Bitmap {}
// SAFETY: as above.
unsafe impl Sync for Bitmap {}

/// The bitmap of other shaders, or (`memory`) of shaders that use title
/// memory, whose marks go back only while it is not withheld
/// (`crate::memory::withholding`).
fn bitmap(memory: bool) -> Option<&'static Bitmap> {
    static BITMAPS: [std::sync::OnceLock<Option<Bitmap>>; 2] = [const { std::sync::OnceLock::new() }; 2];
    BITMAPS[usize::from(memory)]
        .get_or_init(|| {
            // Written pages, then one read mark per heap index.
            let bits = victory_shader::emit::VIEW_READS + victory_shader::emit::HEAP_VIEWS;
            let bytes = u64::from(bits / 8);
            let (buffer, data) = host()?.shared_buffer(bytes)?;
            crate::host::name(&buffer, || "views' written pages and read slots".into());
            // SAFETY: our mapped buffer of `bytes` bytes.
            unsafe { std::ptr::write_bytes(data, 0, bytes as usize) };
            // SAFETY: a live buffer.
            let gpu = unsafe { buffer.GetGPUVirtualAddress() };
            Some(Bitmap {
                _buffer: buffer,
                data: data as usize,
                gpu,
            })
        })
        .as_ref()
}

/// The host GPU address of the buffer views' written-pages bitmap, for the
/// root UAV of shaders that mark their writes (`memory`: shaders that use
/// title memory, see [`bitmap`]).
pub(crate) fn view_writes(memory: bool) -> Option<u64> {
    bitmap(memory).map(|b| b.gpu)
}

/// Heap slots whose texture views shaders read: the marks seen so far (a
/// slot's stays set in the bitmap until [`forget_read`]), and the slots to
/// clear at the next [`take_reads`].
static READS: Mutex<(Vec<u32>, Vec<u32>)> = Mutex::new((Vec::new(), Vec::new()));

/// After a submission: the heap slots its shaders first read texture views
/// at (`victory_shader::emit`, "Read marks"), each once until
/// [`forget_read`]. Clears the slots forgotten since first.
pub(crate) fn take_reads() -> Vec<u32> {
    let words = (victory_shader::emit::HEAP_VIEWS / 32) as usize;
    let first = (victory_shader::emit::VIEW_READS / 32) as usize;
    // Both bitmaps: shaders that use title memory mark their reads in theirs.
    let mut maps: Vec<&mut [u32]> = [false, true]
        .into_iter()
        .filter_map(bitmap)
        // SAFETY: the mapped bitmap's read marks; the GPU is done with them
        // (one submission runs at a time).
        .map(|b| unsafe { std::slice::from_raw_parts_mut((b.data as *mut u32).add(first), words) })
        .collect();
    if maps.is_empty() {
        return Vec::new();
    }
    let mut reads = READS.lock().unwrap_or_else(|e| e.into_inner());
    let (seen, forgotten) = &mut *reads;
    seen.resize(words, 0);
    for slot in forgotten.drain(..) {
        let (w, bit) = ((slot / 32) as usize, 1u32 << (slot % 32));
        if w < words {
            maps.iter_mut().for_each(|bits| bits[w] &= !bit);
            seen[w] &= !bit;
        }
    }
    let mut new = Vec::new();
    for (w, s) in seen.iter_mut().enumerate() {
        let mut fresh = maps.iter().fold(0, |x, bits| x | bits[w]) & !*s;
        *s |= fresh;
        while fresh != 0 {
            new.push(w as u32 * 32 + fresh.trailing_zeros());
            fresh &= fresh - 1;
        }
    }
    new
}

/// The heap slots shaders read texture views at since the last call (both
/// bitmaps), whose marks are cleared: what the title used (eviction stamps,
/// `crate::resource::evict`). Between submissions, after [`take_reads`]
/// has seen every mark: its own record of slots seen keeps a slot read
/// again from being reported as new.
pub(crate) fn reads_since() -> Vec<u32> {
    let words = (victory_shader::emit::HEAP_VIEWS / 32) as usize;
    let first = (victory_shader::emit::VIEW_READS / 32) as usize;
    let mut slots = Vec::new();
    for b in [false, true].into_iter().filter_map(bitmap) {
        // SAFETY: the mapped bitmap's read marks; no submission runs.
        let bits = unsafe { std::slice::from_raw_parts_mut((b.data as *mut u32).add(first), words) };
        for (w, word) in bits.iter_mut().enumerate() {
            let mut x = std::mem::take(word);
            while x != 0 {
                slots.push(w as u32 * 32 + x.trailing_zeros());
                x &= x - 1;
            }
        }
    }
    slots.sort_unstable();
    slots.dedup();
    slots
}

/// Heap slot `slot` now holds a texture view left null: its next read is
/// marked again.
pub(crate) fn forget_read(slot: u32) {
    READS.lock().unwrap_or_else(|e| e.into_inner()).1.push(slot);
}

struct MarkedSlice<'a> {
    copy: &'a Shadow,
    page: usize,
    from: usize,
    to: usize,
    /// Ranges of the slice that stay in the copy and out of title memory.
    withheld: Vec<(usize, usize)>,
}

impl MarkedSlice<'_> {
    /// Merges shaders' changes into title memory and keeps the host slice
    /// equal to title memory, but for the withheld ranges; returns
    /// `(bytes_written, wrote, saved)`, `None` if title memory could not be
    /// read or a page was remapped. `title` is scratch memory.
    ///
    /// The baseline (title memory as the copy took it) is title memory now,
    /// read while the page is still armed, unless a write since kept the
    /// page's bytes from before it (`saved`): then title memory is read
    /// again, with that write in, and merged three ways. A page changed
    /// otherwise meanwhile keeps the title's bytes. Only the spans the
    /// merge changed are written.
    fn commit(&self, title: &mut Vec<u8>) -> Option<(usize, bool, bool)> {
        let t = self.copy.tracked.as_ref()?;
        let page = self.page / PAGE;
        let start = self.copy.start as usize;
        let n = self.to - self.from;
        title.clear();
        title.resize(n, 0);
        #[cfg(feature = "diagnostics")]
        let reading = crate::watch::Timed(std::time::Instant::now(), crate::copystat::C::MarkedReadUs);
        let read = crate::host::read_into(self.from as *const u8, title);
        // Our own writes below must not save the page for this copy again.
        t.unarm(page);
        let saved = t.saved(page);
        if !read {
            return None;
        }
        let saved = match saved {
            Some(bytes) => {
                if !crate::host::read_into(self.from as *const u8, title) {
                    return None;
                }
                Some(bytes)
            }
            None if t.changed(page) => {
                #[cfg(feature = "diagnostics")]
                crate::copystat::add(crate::copystat::C::MarkedKept, 1);
                return None;
            }
            None => None,
        };
        #[cfg(feature = "diagnostics")]
        drop(reading);
        let (mut written, mut synced) = (0, true);
        for (from, to) in outside_withheld(self.from, self.to, &self.withheld) {
            let (a, b) = (from - self.from, to - self.from);
            // SAFETY: inside the mapping; only the submitting thread writes it.
            let host = unsafe { std::slice::from_raw_parts_mut(self.copy.data.add(from - start), to - from) };
            let piece = &mut title[a..b];
            #[cfg(feature = "diagnostics")]
            let merging = crate::watch::Timed(std::time::Instant::now(), crate::copystat::C::MarkedMergeUs);
            let spans = match &saved {
                Some(bytes) => merge_into_title(piece, host, &bytes[from - self.page..to - self.page]),
                None => merge_from_host(piece, host),
            };
            #[cfg(feature = "diagnostics")]
            drop(merging);
            #[cfg(test)]
            tests::before_copy_back_write();
            // Under the remap lock: a page remapped meanwhile keeps its new
            // bytes, and the slice is not taken. The bytes between the
            // spans are title memory as read above, and a title write
            // landing there since keeps.
            let wrote = match (spans.first(), spans.last()) {
                (Some(&(first, _)), Some(&(_, last))) => {
                    crate::watch::write_back(Some(t), from + first, last - first, || {
                        for &(a, b) in &spans {
                            crate::host::write_mapped((from + a) as *mut u8, piece[a..].as_ptr(), b - a);
                        }
                    })
                }
                _ => !t.was_remapped(page),
            };
            if !wrote {
                synced = false;
                continue;
            }
            written += spans.iter().map(|(a, b)| b - a).sum::<usize>();
            // The host bytes are title memory now; so is the snapshot of
            // declared ranges over them: their copy-backs must not take the
            // title's bytes the merge took for the GPU's.
            self.copy.sync_snapshot(from - start, host);
        }
        synced.then_some((written, written > 0, saved.is_some()))
    }
}

/// Bytes compared at once by the merges before looking closer (most of a
/// marked page is what it was: one `memcmp` per chunk).
const MERGE_CHUNK: usize = 64;

/// Adds byte `at` to `spans` (in order), joining it to the last span across
/// a gap under [`SPAN_GAP`].
fn note_span(spans: &mut Vec<(usize, usize)>, at: usize) {
    match spans.last_mut() {
        Some((_, end)) if at < *end + SPAN_GAP => *end = at + 1,
        _ => spans.push((at, at + 1)),
    }
}

/// Past [`MAX_SPANS`] spans: one span over them all.
fn bound_spans(mut spans: Vec<(usize, usize)>) -> Vec<(usize, usize)> {
    if spans.len() > MAX_SPANS {
        let (first, last) = (spans[0].0, spans[spans.len() - 1].1);
        spans = vec![(first, last)];
    }
    spans
}

/// [`merge_into_title`] where the baseline is `title` itself (the title
/// wrote nothing since the copy took it): `title` takes every byte of
/// `host` that differs; `host` is left as it is (equal to `title` then).
fn merge_from_host(title: &mut [u8], host: &[u8]) -> Vec<(usize, usize)> {
    debug_assert_eq!(title.len(), host.len());
    let mut spans: Vec<(usize, usize)> = Vec::new();
    let mut at = 0;
    while at < title.len() {
        let end = (at + MERGE_CHUNK).min(title.len());
        if title[at..end] != host[at..end] {
            for i in at..end {
                if title[i] != host[i] {
                    title[i] = host[i];
                    note_span(&mut spans, i);
                }
            }
        }
        at = end;
    }
    bound_spans(spans)
}

/// Bytes closer than this between two spans the merge changed are written
/// with them (one kernel call instead of two).
const SPAN_GAP: usize = 256;
/// Past this many spans in a slice, its changes go back as one span.
const MAX_SPANS: usize = 16;

/// The three-way merge of a copy-back: where the GPU changed a byte of
/// `host` (it differs from `before`) and `title` lacks it, `title` takes it;
/// `host` takes `title`'s bytes everywhere. Returns the spans of `title`
/// changed (offsets, in order, joined across gaps under [`SPAN_GAP`]),
/// which are all that must be written into title memory. Chunks where the
/// GPU changed nothing are compared whole.
fn merge_into_title(title: &mut [u8], host: &mut [u8], before: &[u8]) -> Vec<(usize, usize)> {
    debug_assert!(title.len() == host.len() && host.len() == before.len());
    let mut spans: Vec<(usize, usize)> = Vec::new();
    let mut at = 0;
    while at < title.len() {
        let end = (at + MERGE_CHUNK).min(title.len());
        if host[at..end] == before[at..end] {
            if title[at..end] != host[at..end] {
                host[at..end].copy_from_slice(&title[at..end]);
            }
        } else {
            for i in at..end {
                if host[i] != before[i] && title[i] != host[i] {
                    title[i] = host[i];
                    note_span(&mut spans, i);
                }
                host[i] = title[i];
            }
        }
        at = end;
    }
    bound_spans(spans)
}

/// The parts of `[from, to)` not covered by `withheld`, in order.
fn outside_withheld(from: usize, to: usize, withheld: &[(usize, usize)]) -> Vec<(usize, usize)> {
    let mut held: Vec<(usize, usize)> = withheld
        .iter()
        .filter_map(|&(a, b)| {
            let (a, b) = (a.max(from), b.min(to));
            (a < b).then_some((a, b))
        })
        .collect();
    held.sort_unstable();
    let mut ranges = Vec::new();
    let mut at = from;
    for (a, b) in held {
        if b <= at {
            continue;
        }
        if at < a {
            ranges.push((at, a));
        }
        at = b;
        if at >= to {
            return ranges;
        }
    }
    if at < to {
        ranges.push((at, to));
    }
    ranges
}

/// The marked bits of `bits` below `end` (and so the bits themselves
/// cleared); marks of offsets past the bitmap land in its last bit, which no
/// copy owns. Clearing that bit's whole dword dropped the writes to the 31
/// pages before it.
fn drain_marks(bits: &mut [u32], end: u32) -> Vec<u32> {
    if let Some(last) = bits.last_mut() {
        *last &= !(1 << 31);
    }
    let used = (end as usize).div_ceil(32).min(bits.len());
    let mut marked = Vec::new();
    for (w, word) in bits[..used].iter_mut().enumerate() {
        let mut x = std::mem::take(word);
        while x != 0 {
            marked.push((w * 32) as u32 + x.trailing_zeros());
            x &= x - 1;
        }
    }
    marked
}

/// After a submission: every page its shaders marked as written through
/// views goes back ([`MarkedSlice::commit`]), except bytes in
/// `withheld` (ranges of copies that lists whose writes stay out of title
/// memory declared) and, with `memory_withheld`, pages that shaders using
/// title memory marked; the bitmaps are cleared. Before the copy-backs of
/// declared ranges, which disarm the copies.
// Marks do not tell lists apart: bytes in a withheld range stay out even if
// another list also marked the page.
pub(crate) fn copy_back_marks(withheld: &[(Arc<Shadow>, usize, usize)], memory_withheld: bool) {
    let (mut marked, mut held) = (Vec::new(), Vec::new());
    for memory in [false, true] {
        let Some(b) = bitmap(memory) else {
            continue;
        };
        let words = (victory_shader::emit::VIEW_WRITE_BITS / 32) as usize;
        // SAFETY: the mapped bitmap; the GPU is done with it (one submission
        // runs at a time).
        let bits = unsafe { std::slice::from_raw_parts_mut(b.data as *mut u32, words) };
        let into = if memory && memory_withheld { &mut held } else { &mut marked };
        into.extend(drain_marks(bits, marks().end));
    }
    #[cfg(feature = "diagnostics")]
    let _timed = crate::watch::Timed(std::time::Instant::now(), crate::copystat::C::MarkedUs);
    marked.sort_unstable();
    marked.dedup();
    let page_of = |m: &Marks, n: u32| {
        let (&base, copy) = m.owners.range(..=n).next_back()?;
        let copy = copy.upgrade()?;
        let k = n - base;
        (k < pages(copy.start, copy.len)).then(|| {
            let page = (((copy.start >> 16) + u64::from(k)) << 16) as usize;
            (copy, page)
        })
    };
    // Held back: dropped from the copies (`crate::memory::drop_withheld`).
    if !held.is_empty() {
        let m = marks();
        for (_, page) in held.into_iter().filter_map(|n| page_of(&m, n)) {
            crate::memory::drop_withheld(page, PAGE);
        }
    }
    if marked.is_empty() {
        return;
    }
    let pages: Vec<(Arc<Shadow>, usize)> = {
        let m = marks();
        marked.iter().filter_map(|&n| page_of(&m, n)).collect()
    };
    // Prepare all slices before any `watch::writing` opens shared pages.
    #[cfg(feature = "diagnostics")]
    let preparing = crate::watch::Timed(std::time::Instant::now(), crate::copystat::C::MarkedPrepUs);
    let prepared: Vec<MarkedSlice<'_>> = pages
        .iter()
        .filter_map(|(copy, page)| {
            #[cfg(feature = "diagnostics")]
            crate::copystat::add(crate::copystat::C::MarkedPages, 1);
            let held: Vec<(usize, usize)> = withheld
                .iter()
                .filter(|(c, a, n)| Arc::ptr_eq(c, copy) && *a < page + PAGE && *page < a + n)
                .map(|(_, a, n)| (*a, a + n))
                .collect();
            copy.prepare_marked(*page, held)
        })
        .collect();
    #[cfg(feature = "diagnostics")]
    drop(preparing);
    let mut results = Vec::with_capacity(prepared.len());
    let mut scratch = Vec::with_capacity(PAGE);
    for entry in &prepared {
        let res = entry.commit(&mut scratch);
        if let Some((written, _, _)) = res {
            #[cfg(feature = "diagnostics")]
            crate::copystat::add(crate::copystat::C::MarkedBytes, written as u64);
            #[cfg(not(feature = "diagnostics"))]
            let _ = written;
        }
        results.push(res);
    }
    #[cfg(feature = "diagnostics")]
    let _taking = crate::watch::Timed(std::time::Instant::now(), crate::copystat::C::MarkedTakenUs);
    // After all `Writing` guards have dropped, re-take every synced slice's
    // page (even sub-page copies cover their entire tracked slice in `page`),
    // unless a later write in this batch overlapped its bytes or a withheld
    // range was deliberately left different from title memory. The slices
    // of a page are found by page (the batch has thousands), and each
    // copy's pages are taken in one call (runs of pages, one protection
    // change each).
    let mut by_page: std::collections::HashMap<usize, Vec<usize>> = std::collections::HashMap::new();
    for (i, entry) in prepared.iter().enumerate() {
        by_page.entry(entry.page).or_default().push(i);
    }
    let wrote = |j: usize| results[j].is_some_and(|(_, w, _)| w);
    let mut take: std::collections::HashMap<*const Shadow, (&Shadow, Vec<usize>)> = std::collections::HashMap::new();
    for (i, entry) in prepared.iter().enumerate() {
        let Some((_, wrote_self, was_saved)) = results[i] else {
            continue;
        };
        let same_page = &by_page[&entry.page];
        let wrote_on_page = wrote_self || same_page.iter().any(|&j| wrote(j));
        let overlapped_later = same_page
            .iter()
            .any(|&j| j > i && wrote(j) && prepared[j].from < entry.to && entry.from < prepared[j].to);
        if (was_saved || wrote_on_page) && !overlapped_later && entry.withheld.is_empty() && entry.copy.tracked.is_some() {
            take.entry(std::ptr::from_ref(entry.copy))
                .or_insert_with(|| (entry.copy, Vec::new()))
                .1
                .push(entry.page / PAGE);
        }
    }
    for (copy, mut pages) in take.into_values() {
        pages.sort_unstable();
        pages.dedup();
        if let Some(t) = &copy.tracked {
            t.taken(&pages);
        }
    }
}

struct Entry {
    end: u64,
    shadow: Option<Arc<Shadow>>,
}

/// Title buffers by (start address, resource), with their copy once made.
/// Read by every draw's views (`find`) on the title's threads, at once.
static BUFFERS: RwLock<BTreeMap<(u64, usize), Entry>> = RwLock::new(BTreeMap::new());

fn buffers() -> std::sync::RwLockWriteGuard<'static, BTreeMap<(u64, usize), Entry>> {
    BUFFERS.write().unwrap_or_else(|e| e.into_inner())
}

fn buffers_read() -> std::sync::RwLockReadGuard<'static, BTreeMap<(u64, usize), Entry>> {
    BUFFERS.read().unwrap_or_else(|e| e.into_inner())
}

/// The key of the innermost title buffer holding `[address, end)`.
fn innermost(
    map: &BTreeMap<(u64, usize), Entry>,
    address: u64,
    end: u64,
) -> Option<(u64, usize)> {
    map.range(..=(address, usize::MAX))
        .rev()
        .take(64)
        .find(|(_, e)| e.end >= end)
        .map(|(k, _)| *k)
}

/// Counts copies made and title buffers released: [`all`] may differ when
/// it changed.
static GENERATION: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

/// The generation of [`all`]: read it before calling that.
pub(crate) fn generation() -> u64 {
    GENERATION.load(std::sync::atomic::Ordering::Acquire)
}

fn changed() {
    GENERATION.fetch_add(1, std::sync::atomic::Ordering::Release);
}

/// A title buffer exists at `start` (`len` bytes).
pub(crate) fn register(resource: usize, start: u64, len: u64) {
    buffers().insert(
        (start, resource),
        Entry {
            end: start + len,
            shadow: None,
        },
    );
}

/// The title released it; lists that still hold its copy keep it alive.
pub(crate) fn unregister(resource: usize, start: u64) {
    if buffers().remove(&(start, resource)).is_some_and(|e| e.shadow.is_some()) {
        changed();
    }
}

/// The copy of the innermost title buffer holding `[address, address +
/// len)`, made on first use.
pub(crate) fn find(address: u64, len: u64) -> Option<Arc<Shadow>> {
    let end = address.checked_add(len.max(1))?;
    // A copy made already, as it nearly always is, is found under the
    // shared lock; the copy is made under the exclusive one.
    {
        let map = buffers_read();
        let Some(key) = innermost(&map, address, end) else {
            drop(map);
            return loose(address, end);
        };
        if let Some(shadow) = map.get(&key).and_then(|e| e.shadow.clone()) {
            return Some(shadow);
        }
    }
    let mut map = buffers();
    let Some(key) = innermost(&map, address, end) else {
        drop(map);
        return loose(address, end);
    };
    let entry = map.get_mut(&key)?;
    if entry.shadow.is_none() {
        let len = entry.end - key.0;
        // Rounded up to whole slots of the small buffers' pool, so that views
        // ending in the buffer's last 256 bytes fit and a root descriptor's
        // read past the title buffer's end stays in this copy up to the
        // rounded width, none when the length is whole slots already
        // (`Host::shared_buffer`). Shaders write some buffers (unordered
        // access), so not upload memory.
        let width_bytes = len.next_multiple_of(crate::placed::SLOT_BYTES);
        let (buffer, data) = host()?.shared_buffer(width_bytes)?;
        crate::host::name(&buffer, || format!("title buffer {:#x}+{len:#x}", key.0));
        // SAFETY: a live buffer.
        let gpu = unsafe { buffer.GetGPUVirtualAddress() };
        if len >= 16 << 20 {
            tracing::info!(
                start = format_args!("{:#x}", key.0),
                len,
                "large buffer copied to the host"
            );
        } else {
            tracing::debug!(
                start = format_args!("{:#x}", key.0),
                len,
                "buffer copied to the host"
            );
        }
        entry.shadow = Some(Shadow::new(buffer, data, gpu, key.0, len));
        changed();
    }
    entry.shadow.clone()
}

/// Titles also use memory outside any buffer object as GPU memory (Grand
/// Theft Auto VI's interface vertices, from its own allocator). Such memory
/// gets copies of the aligned windows it touches.
/// shortcut: copies of windows are never freed; key them to the title's
/// allocations if memory use grows.
const WINDOW: u64 = 4 << 20;

/// Copies of title memory outside buffers, by (start, length).
static LOOSE: Mutex<BTreeMap<(u64, u64), Arc<Shadow>>> = Mutex::new(BTreeMap::new());

fn loose(address: u64, end: u64) -> Option<Arc<Shadow>> {
    let start = address & !(WINDOW - 1);
    let len = end.next_multiple_of(WINDOW) - start;
    let mut map = LOOSE.lock().unwrap_or_else(|e| e.into_inner());
    if let Some(shadow) = map.get(&(start, len)) {
        return Some(shadow.clone());
    }
    let (buffer, data) = host()?.shared_buffer(len)?;
    crate::host::name(&buffer, || format!("title memory {start:#x}+{len:#x}"));
    // SAFETY: a live buffer.
    let gpu = unsafe { buffer.GetGPUVirtualAddress() };
    tracing::debug!(
        start = format_args!("{start:#x}"),
        len,
        "loose memory copied to the host"
    );
    let shadow = Shadow::new(buffer, data, gpu, start, len);
    map.insert((start, len), shadow.clone());
    changed();
    Some(shadow)
}

/// The copy a buffer view of `len` bytes at `address`, whose elements are
/// `element` bytes, reads: the innermost title buffer's ([`find`]), unless
/// the view does not start on an element of that copy. A host view counts
/// its first element from its buffer's start; a console view starts at any
/// address, and the title places structured views at ring positions inside
/// its big GPU heap (the meshlet culling's 456-byte constants, the colour
/// bakes' parameter blocks: refused, their slots read zeros, every meshlet
/// culled). Such a view gets a copy of its own bytes, starting at its
/// address (`true` with it), which lives while a heap slot made from the
/// view or a list holds it. `element` is what the view's first byte must be
/// a multiple of in its copy: its element size, 16 for a raw view (a host
/// raw view 4 bytes off removes the device).
pub(crate) fn for_view(address: u64, len: u64, element: u64) -> Option<(Arc<Shadow>, bool)> {
    let shadow = find(address, len)?;
    if element == 0 || shadow.offset(address).is_multiple_of(element) {
        return Some((shadow, false));
    }
    #[cfg(feature = "diagnostics")]
    VIEWS[0].fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    exact(address, len, element).map(|s| (s, true))
}

/// Asks for, and copies made of, views off their copy's elements, the
/// copies' bytes and the largest (diagnostics).
#[cfg(feature = "diagnostics")]
static VIEWS: [std::sync::atomic::AtomicU64; 4] = [const { std::sync::atomic::AtomicU64::new(0) }; 4];

/// Logs [`VIEWS`] per frame over `frames`.
#[cfg(feature = "diagnostics")]
pub(crate) fn log_views(frames: f64) {
    use std::sync::atomic::Ordering::Relaxed;
    let [asks, made, bytes, largest] = [0, 1, 2, 3].map(|i| VIEWS[i].swap(0, Relaxed));
    let live = EXACT
        .lock()
        .unwrap_or_else(|e| e.into_inner())
        .1
        .values()
        .filter(|w| w.strong_count() > 0)
        .count();
    let (chunks, held) = {
        let a = arena();
        // SAFETY: live buffers.
        let size = |c: &(ID3D12Resource, usize, u64, BTreeMap<u64, u64>)| unsafe { c.0.GetDesc() }.Width;
        let held: u64 = a.chunks.iter().map(|c| size(c) - c.3.values().sum::<u64>()).sum();
        (a.chunks.len(), held)
    };
    if asks > 0 {
        tracing::info!(
            "buffer views off their copy's elements: {:.0} asked for/frame, {:.0} copies made/frame ({:.1} KB), largest {largest} bytes, {live} live in {chunks} shared buffers ({:.1} MB held)",
            asks as f64 / frames,
            made as f64 / frames,
            bytes as f64 / frames / 1e3,
            held as f64 / 1e6
        );
    }
}

/// Copies of single views' bytes ([`for_view`]), by (start, length), held
/// by the heap slots made from them and the lists that use them: a copy is
/// found again for as long as a slot shows it, so draws refresh the copy
/// the GPU reads (a capped cache dropped copies slots still showed, and a
/// new copy of the same bytes took the refreshes). With the size the map
/// is next pruned of dead entries at.
#[allow(clippy::type_complexity)]
static EXACT: Mutex<(usize, BTreeMap<(u64, u64, u64), std::sync::Weak<Shadow>>)> = Mutex::new((0, BTreeMap::new()));

fn exact(address: u64, len: u64, element: u64) -> Option<Arc<Shadow>> {
    let mut guard = EXACT.lock().unwrap_or_else(|e| e.into_inner());
    let (prune_at, map) = &mut *guard;
    if let Some(shadow) = map.get(&(address, len, element)).and_then(std::sync::Weak::upgrade) {
        return Some(shadow);
    }
    // Ring positions change every frame: entries die by the hundred.
    if map.len() >= *prune_at {
        map.retain(|_, w| w.strong_count() > 0);
        *prune_at = (2 * map.len()).max(1024);
    }
    // A run of a shared buffer, its start a whole number of elements in.
    let (buffer, data, gpu, slice) = arena().take(len, element)?;
    let shadow = Shadow::new_at(buffer, data, gpu, address, len, Some(slice));
    map.insert((address, len, element), Arc::downgrade(&shadow));
    #[cfg(feature = "diagnostics")]
    {
        use std::sync::atomic::Ordering::Relaxed;
        VIEWS[1].fetch_add(1, Relaxed);
        VIEWS[2].fetch_add(len, Relaxed);
        VIEWS[3].fetch_max(len, Relaxed);
    }
    Some(shadow)
}

/// A run of an [`Arena`] buffer: the buffer, where the run starts, its
/// bytes.
#[derive(Clone, Copy)]
struct Slice {
    chunk: usize,
    offset: u64,
    len: u64,
}

/// Shared buffers for the copies of single views ([`exact`]): they are
/// small (~2 KB on average) and come and go by the hundred a frame, and a
/// buffer of their own each cost a kernel call to make and one to free on
/// the title's threads and the submitting one (300-820 a frame), and 64 KiB
/// of memory each (~5,000 live).
#[derive(Default)]
struct Arena {
    /// Each buffer, its mapping and host address, and its free runs
    /// (offset → bytes).
    chunks: Vec<(ID3D12Resource, usize, u64, BTreeMap<u64, u64>)>,
}

/// Bytes of each arena buffer (a larger copy gets one of its size).
const CHUNK: u64 = 16 << 20;

static ARENA: Mutex<Arena> = Mutex::new(Arena { chunks: Vec::new() });

fn arena() -> std::sync::MutexGuard<'static, Arena> {
    ARENA.lock().unwrap_or_else(|e| e.into_inner())
}

impl Arena {
    /// A run of `len` bytes starting at a multiple of `element` (and of 256)
    /// in a shared buffer: the buffer, the run's mapping and host address.
    fn take(&mut self, len: u64, element: u64) -> Option<(ID3D12Resource, *mut u8, u64, Slice)> {
        let gcd = |mut a: u64, mut b: u64| {
            while b != 0 {
                (a, b) = (b, a % b);
            }
            a
        };
        let align = 256 / gcd(256, element.max(1)) * element.max(1);
        let len = len.max(1).next_multiple_of(4);
        let fits = |free: &BTreeMap<u64, u64>| {
            free.iter().find_map(|(&at, &n)| {
                let start = at.next_multiple_of(align);
                (start + len <= at + n).then_some((at, n, start))
            })
        };
        let found = self.chunks.iter().enumerate().find_map(|(c, chunk)| fits(&chunk.3).map(|f| (c, f)));
        let (chunk, (at, n, start)) = match found {
            Some(found) => found,
            None => {
                let size = CHUNK.max((len + align).next_multiple_of(1 << 16));
                let (buffer, data) = host()?.shared_buffer(size)?;
                crate::host::name(&buffer, || format!("title views {size:#x}"));
                // SAFETY: a live buffer.
                let gpu = unsafe { buffer.GetGPUVirtualAddress() };
                let free = BTreeMap::from([(0, size)]);
                self.chunks.push((buffer, data as usize, gpu, free));
                let c = self.chunks.len() - 1;
                (c, fits(&self.chunks[c].3)?)
            }
        };
        let (buffer, data, gpu, free) = &mut self.chunks[chunk];
        free.remove(&at);
        if start > at {
            free.insert(at, start - at);
        }
        if at + n > start + len {
            free.insert(start + len, at + n - start - len);
        }
        let slice = Slice { chunk, offset: start, len };
        Some((buffer.clone(), (*data + start as usize) as *mut u8, *gpu + start, slice))
    }

    /// A run goes back, joined to its free neighbours.
    fn give_back(&mut self, slice: Slice) {
        let Some((_, _, _, free)) = self.chunks.get_mut(slice.chunk) else { return };
        let (mut at, mut n) = (slice.offset, slice.len);
        if let Some((&before, &m)) = free.range(..at).next_back()
            && before + m == at
        {
            free.remove(&before);
            (at, n) = (before, n + m);
        }
        if let Some(m) = free.remove(&(at + n)) {
            n += m;
        }
        free.insert(at, n);
    }
}

/// Every host copy of title memory made so far (buffers and windows).
pub(crate) fn all() -> Vec<Arc<Shadow>> {
    let mut all: Vec<Arc<Shadow>> =
        buffers_read().values().filter_map(|e| e.shadow.clone()).collect();
    all.extend(LOOSE.lock().unwrap_or_else(|e| e.into_inner()).values().cloned());
    all
}

/// For device-removed reports: the copy of title memory whose host range
/// starts closest below host GPU address `va`, and where `va` is in it.
pub(crate) fn near(va: u64) -> String {
    let mut all: Vec<(u64, u64, u64)> = buffers_read()
        .values()
        .filter_map(|e| e.shadow.as_deref().map(|s| (s.gpu, s.start, s.len)))
        .collect();
    all.extend(
        LOOSE
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .values()
            .map(|s| (s.gpu, s.start, s.len)),
    );
    let below = all.iter().filter(|c| c.0 <= va).max();
    let above = all.iter().filter(|c| c.0 > va).min();
    let mut text = match below {
        Some(&(gpu, start, len)) => format!(
            "; the nearest title memory copy below it: {start:#x}+{len:#x} at host {gpu:#x}, offset {:#x}",
            va - gpu
        ),
        None => "; no title memory copy below it".into(),
    };
    if let Some(&(gpu, start, len)) = above {
        text.push_str(&format!(
            "; the nearest above: {start:#x}+{len:#x} at host {gpu:#x} ({} copies in all)",
            all.len()
        ));
    }
    text
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;

    /// A shader's read mark of heap slot `slot` in the bitmap of shaders
    /// that use title memory (`memory`) or of the others.
    pub(crate) fn mark_read(memory: bool, slot: u32) {
        let b = bitmap(memory).expect("a host GPU");
        let bit = victory_shader::emit::VIEW_READS + slot;
        // SAFETY: a bit of the mapped bitmap; no submission runs (the caller
        // holds the submission lock).
        unsafe { *(b.data as *mut u32).add((bit / 32) as usize) |= 1 << (bit % 32) };
    }

    /// A snapshot-backed range can overlap a write whose list missed title
    /// memory. Retire only the allowed bytes, even though both reached the GPU.
    #[test]
    fn snapshot_copy_back_keeps_overlapping_withheld_bytes_on_the_gpu() {
        let Some(host) = host() else {
            return;
        };
        const A: u32 = 0x5a5a_5a5a;
        const B: u32 = 0xa5a5_a5a5;
        let mut results = Vec::new();
        for withheld in [false, true] {
            let address = crate::watch::tests::pool(1);
            register(address, address as u64, PAGE as u64);
            let copy = find(address as u64, PAGE as u64).unwrap();
            copy.update(address, PAGE);
            let (src, data) = host.upload_buffer(16).unwrap();
            // SAFETY: a mapped 16-byte upload buffer.
            unsafe { data.cast::<[u32; 4]>().write([A, A, B, B]) };
            let a = host.new_list().unwrap();
            let b = host.new_list().unwrap();
            copy.gpu_writes(&a, (address + 64) as u64, 8, false);
            {
                // Withheld when missed: a write of a program using title memory.
                let _writer = b.writer_uses_memory(true);
                copy.gpu_writes(&b, (address + 68) as u64, 8, false);
            }
            // Only the withheld run misses (a list with no miss registered
            // is immune to the frame-global withholding state).
            if withheld {
                b.defer_missed(|| true);
            }
            // SAFETY: live buffers and open lists. The destination is a
            // copy of the title memory; both aligned ranges are inside it.
            unsafe {
                a.list.CopyBufferRegion(&copy.buffer, 64, &src, 0, 8);
                b.list.CopyBufferRegion(&copy.buffer, 68, &src, 8, 8);
            }
            a.close();
            b.close();
            host.execute(&[&a, &b]);
            // SAFETY: three aligned dwords of the test's live title page.
            let title = unsafe { ((address + 64) as *const [u32; 3]).read() };
            let shadow = copy.read_host((address + 64) as u64, 3);
            results.push((title, shadow));
            drop((a, b));
            unregister(address, address as u64);
            drop(copy);
            crate::watch::tests::free(address);
        }
        assert_eq!(
            results,
            [([A, B, B], vec![A, B, B]), ([A, 0, 0], vec![A, B, B])]
        );
    }

    /// The title writes a constant under a range an earlier list of the
    /// submission declared as written by shaders: the refresh before the
    /// submission runs still brings it to the copy, a shader's write still
    /// goes back, and a title write made after the refresh is not undone.
    #[test]
    fn title_writes_under_a_busy_range_are_copied() {
        let p = crate::watch::tests::pool(2);
        let len = 2 * PAGE;
        register(p, p as u64, len as u64);
        let Some(copy) = find(p as u64, len as u64) else {
            unregister(p, p as u64);
            return; // no host GPU here
        };
        let title = |at: usize| (p + at) as *mut u8;
        let host = |at: usize| copy.read_host((p + at) as u64 & !3, 1)[0].to_le_bytes()[at % 4];
        copy.update(p, len);
        // An earlier list declares all of it as written by shaders.
        copy.writing(p, len, false);
        // SAFETY: (all title reads and writes below) bytes of our pool.
        unsafe { title(PAGE + 20).write_volatile(0x55) };
        assert!(copy.update(p + PAGE + 16, 16) > 0, "the title's write is copied");
        assert_eq!(host(PAGE + 20), 0x55, "the title's constant reaches the copy");
        // The shaders run: one byte written. The title writes meanwhile.
        copy.write_host((p + 10) as u64, &[0xaa]);
        // SAFETY: a byte of our pool.
        unsafe { title(PAGE + 20).write_volatile(0x66) };
        copy.copy_back(p, len, &[]);
        // SAFETY: bytes of our pool.
        unsafe {
            assert_eq!(title(10).read_volatile(), 0xaa, "the shader's write goes back");
            assert_eq!(title(PAGE + 20).read_volatile(), 0x66, "the title's later write stays");
        }
        unregister(p, p as u64);
    }

    /// A title buffer over a pool and, after it, a plain allocation (memory
    /// that is not the runtime's): the pool is tracked and guarded; the
    /// plain allocation's bytes are copied at each use and never guarded,
    /// so a value the title keeps there reaches the copy, and again after
    /// the title changes it.
    #[test]
    fn bytes_off_the_runtimes_memory_are_copied_at_each_use() {
        use crate::watch::tests::{MEM_WRITE_WATCH, free};
        use windows_sys::Win32::System::Memory::{
            MEM_COMMIT, MEM_RELEASE, MEM_RESERVE, MEMORY_BASIC_INFORMATION, PAGE_READONLY, PAGE_READWRITE,
            VirtualAlloc, VirtualFree, VirtualQuery,
        };
        // A pool page and a plain one right after it, in two pages found free
        // (other tests allocate meanwhile: tried again if taken).
        let (pool, plain) = (0..8)
            .find_map(|_| {
                // SAFETY: a fresh reservation, released at once; then
                // allocations where it was, freed below.
                unsafe {
                    let p = VirtualAlloc(std::ptr::null(), 2 * PAGE, MEM_RESERVE, PAGE_READWRITE) as usize;
                    VirtualFree(p as *mut _, 0, MEM_RELEASE);
                    let at = |q: usize, kind: u32| {
                        VirtualAlloc(q as *const _, PAGE, MEM_RESERVE | MEM_COMMIT | kind, PAGE_READWRITE) as usize
                    };
                    let (pool, plain) = (at(p, MEM_WRITE_WATCH), at(p + PAGE, 0));
                    if p != 0 && pool == p && plain == p + PAGE {
                        return Some((pool, plain));
                    }
                    [pool, plain].into_iter().filter(|&q| q != 0).for_each(free);
                    None
                }
            })
            .expect("two free pages");
        let len = 2 * PAGE;
        register(pool, pool as u64, len as u64);
        let Some(copy) = find(pool as u64, len as u64) else {
            unregister(pool, pool as u64);
            [pool, plain].into_iter().for_each(free);
            return; // no host GPU here
        };
        let word = (plain + 64) as *mut u32;
        let protection = |p: usize| {
            // SAFETY: plain data filled in by the query.
            let mut info: MEMORY_BASIC_INFORMATION = unsafe { std::mem::zeroed() };
            // SAFETY: a valid out pointer of the right size.
            unsafe { VirtualQuery(p as *const _, &mut info, size_of_val(&info)) };
            info.Protect
        };
        // SAFETY: a word of our plain allocation.
        unsafe { word.write_volatile(0x1234) };
        copy.update(pool, len);
        assert_eq!(copy.read_host(word as u64, 1), [0x1234]);
        assert!(!copy.clean(), "never clean: part of it is not tracked");
        // SAFETY: as above.
        unsafe { word.write_volatile(0x5678) };
        copy.update(pool, len);
        assert_eq!(copy.read_host(word as u64, 1), [0x5678], "the title's change reaches the copy");
        assert_eq!(protection(pool), PAGE_READONLY, "the pool is guarded");
        assert_eq!(protection(plain), PAGE_READWRITE, "the plain allocation is not");
        unregister(pool, pool as u64);
        // The pages stay allocated (as the other tests' pools): a copy some
        // page table still holds must not name memory reused meanwhile.
    }

    /// Runs of the shared buffers start a whole number of elements (and of
    /// 256 bytes) in, do not overlap, and join again when given back.
    #[test]
    fn arena_runs_are_aligned_to_their_elements_and_join_again() {
        if host().is_none() {
            return; // no host GPU here
        }
        let mut arena = Arena::default();
        let (_, _, _, a) = arena.take(100, 456).unwrap();
        let (_, _, _, b) = arena.take(5000, 16).unwrap();
        let (_, _, _, c) = arena.take(4, 4).unwrap();
        assert!(a.offset % 456 == 0 && a.offset % 256 == 0 && b.offset % 256 == 0);
        let mut runs = [a, b, c].map(|s| (s.offset, s.offset + s.len));
        runs.sort_unstable();
        assert!(runs.windows(2).all(|w| w[0].1 <= w[1].0), "{runs:?}");
        for s in [b, a, c] {
            arena.give_back(s);
        }
        assert_eq!(arena.chunks.len(), 1);
        assert_eq!(arena.chunks[0].3.len(), 1, "one free run again");
        // Larger than a buffer: one of its own.
        let (_, _, _, d) = arena.take(CHUNK + 1, 4).unwrap();
        assert_eq!((arena.chunks.len(), d.chunk), (2, 1));
    }

    /// Ranges shaders write far apart are kept apart (not the copy between
    /// them); what the GPU wrote goes back once, whichever overlapping
    /// ranges declared it, and nothing else does.
    #[test]
    fn snapshots_cover_the_declared_ranges_and_writes_go_back_once() {
        let p = crate::watch::tests::pool(4);
        let len = 4 * PAGE;
        register(p, p as u64, len as u64);
        let Some(copy) = find(p as u64, len as u64) else {
            unregister(p, p as u64);
            return; // no host GPU here
        };
        copy.update(p, len);
        let bytes = |s: &Shadow| s.before.lock().unwrap().0.iter().map(|(_, b)| b.len()).sum::<usize>();
        copy.writing(p + 16, 64, false);
        copy.writing(p + 3 * PAGE, 100, false);
        assert_eq!(bytes(&copy), 164, "two segments");
        copy.writing(p + PAGE, 2 * PAGE, false);
        copy.writing(p + PAGE, PAGE, false);
        copy.writing(p + 80, 8, false);
        assert_eq!(bytes(&copy), 72 + 2 * PAGE + 100, "merged where they overlap or touch");
        // The GPU writes inside two ranges.
        copy.write_host((p + 20) as u64, &[1, 2, 3]);
        copy.write_host((p + PAGE + 5) as u64, &[9]);
        assert_eq!(copy.copy_back(p + 16, 64, &[]), 64, "the range's part of the piece");
        assert_eq!(copy.copy_back(p + PAGE, 2 * PAGE, &[]), PAGE, "the piece written");
        assert_eq!(copy.copy_back(p + PAGE, PAGE, &[]), 0, "compared once");
        assert_eq!(copy.copy_back(p + 3 * PAGE, 100, &[]), 0);
        assert_eq!(copy.copy_back(p + 80, 8, &[]), 0);
        // SAFETY: our pool.
        let title = |at: usize| unsafe { (at as *const u8).read_volatile() };
        assert_eq!([title(p + 20), title(p + 21), title(p + 22), title(p + PAGE + 5)], [1, 2, 3, 9]);
        assert_eq!(bytes(&copy), 0, "the snapshot goes with the last range");
        unregister(p, p as u64);
    }

    /// Pages shaders marked (`copy_back_marks`, `victory_shader::emit`
    /// "Write marks") go back by the bytes the GPU changed. The title writes
    /// a marked page after the copy took it (the guard saves its bytes from
    /// before) and a page nobody marked: the shader's byte and the title's
    /// both stay in the first; the second keeps the title's, and the copy's
    /// byte there (no mark) is not put back. The bits are cleared.
    /// The merge gives the bytewise rule's title and host, and its spans
    /// hold every byte it changed in title memory.
    #[test]
    fn merge_spans_hold_exactly_the_bytes_put_into_title_memory() {
        let len = 3 * 4096 + 5; // a tail past the 8-byte words
        let before: Vec<u8> = (0..len).map(|i| (i * 7 % 251) as u8).collect();
        let (mut host, mut title) = (before.clone(), before.clone());
        // The GPU's bytes: single bytes, a word, a run, and the tail.
        for i in [0, 3, 9, 100, 101, 102, 103, 104, 105, 106, 107, 2000, 9000, len - 1] {
            host[i] ^= 0x5a;
        }
        for i in 5000..5300 {
            host[i] = host[i].wrapping_add(1);
        }
        // The title's own writes, one over a GPU byte with the GPU's value.
        for i in [5, 200, 4000, 4001] {
            title[i] ^= 0xa5;
        }
        title[2000] = host[2000];
        let (mut ref_title, mut ref_host) = (title.clone(), host.clone());
        let mut changed_at = Vec::new();
        for (i, ((t, h), b)) in ref_title.iter_mut().zip(ref_host.iter_mut()).zip(&before).enumerate() {
            if *h != *b && *t != *h {
                *t = *h;
                changed_at.push(i);
            }
            *h = *t;
        }
        let read = title.clone();
        let spans = merge_into_title(&mut title, &mut host, &before);
        assert_eq!((&title, &host), (&ref_title, &ref_host), "the bytewise rule");
        assert!(!changed_at.contains(&2000), "the title had the GPU's byte already");
        for i in 0..len {
            let inside = spans.iter().any(|&(a, b)| (a..b).contains(&i));
            assert_eq!(title[i] != read[i], changed_at.contains(&i));
            if title[i] != read[i] {
                assert!(inside, "changed byte {i} is in a span");
            }
        }
        for w in spans.windows(2) {
            assert!(w[0].1 + SPAN_GAP <= w[1].0, "spans apart by the gap at least: {spans:?}");
        }
        assert!(
            spans.iter().all(|&(a, b)| title[a] != read[a] && title[b - 1] != read[b - 1]),
            "spans start and end on changes"
        );
        // Nothing changed: no span.
        let again = host.clone();
        assert!(merge_into_title(&mut title, &mut host, &again).is_empty());
    }

    /// With title memory as its own baseline, the two-way merge is the
    /// three-way one.
    #[test]
    fn merge_from_host_is_the_merge_over_the_title_itself() {
        let len = 2 * 4096 + 3;
        let title: Vec<u8> = (0..len).map(|i| (i * 5 % 241) as u8).collect();
        let mut host = title.clone();
        for i in [1, 64, 65, 700, 1000, 4095, 4096, len - 1] {
            host[i] ^= 0x33;
        }
        let (mut three, mut two, mut host3) = (title.clone(), title.clone(), host.clone());
        let spans3 = merge_into_title(&mut three, &mut host3, &title);
        let spans2 = merge_from_host(&mut two, &host);
        assert_eq!((&two, &spans2), (&three, &spans3));
        assert_eq!((&two, &host3), (&host, &host));
    }

    /// The copy-back of a marked page writes the GPU's bytes and nothing
    /// else into title memory: the rest of the page, the pages around it
    /// (written by the GPU too but not marked) and a byte the title writes
    /// between the copy-back's read and its write keep the title's bytes.
    /// Writing the whole page from the read put the page's older bytes back
    /// over that title write.
    #[test]
    fn a_marked_copy_back_changes_no_title_byte_but_the_gpus() {
        let p = crate::watch::tests::pool(3);
        let len = 3 * PAGE;
        let pattern = |i: usize| (i * 13 % 253) as u8 | 1;
        // SAFETY: our pool, before it is tracked.
        unsafe {
            for i in 0..len {
                ((p + i) as *mut u8).write_volatile(pattern(i));
            }
        }
        register(p, p as u64, len as u64);
        let (Some(copy), Some(b)) = (find(p as u64, len as u64), bitmap(false)) else {
            unregister(p, p as u64);
            return; // no host GPU here
        };
        let _submissions = host().expect("a bitmap means a host").lock_submissions();
        let base = copy.marks.expect("tracked memory gets mark bits") as usize;
        copy.update(p, len);
        copy.writing(p, len, true);
        // The shaders write bytes in each page; only page 1 is marked.
        let gpu = [(10, 0xe1u8), (PAGE + 100, 0xe2), (PAGE + 101, 0xe3), (PAGE + 30_000, 0xe4), (2 * PAGE + 7, 0xe5)];
        for &(at, v) in &gpu {
            copy.write_host((p + at) as u64, &[v]);
        }
        let bits = b.data as *mut u32;
        // SAFETY: a dword of the mapped bitmap.
        unsafe { *bits.add((base + 1) / 32) |= 1 << ((base + 1) % 32) };
        // A title thread writes page 1 once the copy-back has read it.
        const LATE: usize = PAGE + 40_000;
        BEFORE_COPY_BACK_WRITE.with(|cell| {
            *cell.borrow_mut() = Some(Box::new(move || {
                // SAFETY: a byte of our pool (its guard opens it).
                unsafe { ((p + LATE) as *mut u8).write_volatile(0x77) };
            }));
        });
        copy_back_marks(&[], false);
        BEFORE_COPY_BACK_WRITE.with(|cell| cell.borrow_mut().take());
        for i in 0..len {
            // SAFETY: a byte of our pool.
            let now = unsafe { ((p + i) as *const u8).read_volatile() };
            let want = match i {
                LATE => 0x77,
                _ if (PAGE..2 * PAGE).contains(&i) => gpu.iter().find(|g| g.0 == i).map_or(pattern(i), |g| g.1),
                _ => pattern(i),
            };
            assert_eq!(now, want, "title byte {i:#x}");
        }
        unregister(p, p as u64);
    }

    /// A page no list armed is armed when its copy-back is prepared: a
    /// title write before the commit (which reads the baseline) keeps the
    /// page's bytes from before it, and both writers' bytes stay.
    #[test]
    fn a_title_write_between_prepare_and_commit_keeps_both_writers_bytes() {
        let p = crate::watch::tests::pool(1);
        register(p, p as u64, PAGE as u64);
        let Some(copy) = find(p as u64, PAGE as u64) else {
            unregister(p, p as u64);
            return; // no host GPU here
        };
        copy.update(p, PAGE);
        copy.write_host((p + 10) as u64, &[0xaa]);
        let slice = copy.prepare_marked(p, Vec::new()).expect("an unchanged page is prepared");
        // SAFETY: a byte of our pool (its guard opens it).
        unsafe { ((p + 40) as *mut u8).write_volatile(0x11) };
        let done = slice.commit(&mut Vec::new());
        assert!(done.is_some_and(|(n, _, saved)| n > 0 && saved), "{done:?}");
        drop(slice);
        // SAFETY: bytes of our pool.
        let read = |at: usize| unsafe { ((p + at) as *const u8).read_volatile() };
        assert_eq!([read(10), read(40)], [0xaa, 0x11]);
        unregister(p, p as u64);
    }

    #[test]
    fn marked_pages_go_back_by_the_bytes_the_gpu_changed() {
        let p = crate::watch::tests::pool(3);
        let len = 3 * PAGE;
        register(p, p as u64, len as u64);
        let (Some(copy), Some(b)) = (find(p as u64, len as u64), bitmap(false)) else {
            unregister(p, p as u64);
            return; // no host GPU here
        };
        // The bitmap and `copy_back_marks` are process-wide: hold the
        // submission lock so no parallel test's submission consumes the
        // bits this test sets before its own copy-back.
        let _submissions = host().expect("a bitmap means a host").lock_submissions();
        let base = copy.marks.expect("tracked memory gets mark bits") as usize;
        assert_eq!(copy.mark_entry(p as u64 + PAGE as u64 + 8), Some((base as u32 + 1, 8)));
        copy.update(p, len);
        copy.writing(p, len, true);
        assert!(copy.before.lock().unwrap().0.is_empty(), "no snapshot");
        // The shaders write a byte in each page; the title writes pages 1
        // and 2 meanwhile.
        copy.write_host((p + 10) as u64, &[0xaa]);
        copy.write_host((p + PAGE + 20) as u64, &[0xbb]);
        copy.write_host((p + 2 * PAGE + 30) as u64, &[0xcc]);
        let title = |at: usize| (p + at) as *mut u8;
        // SAFETY: (title writes and reads below) bytes of our pool.
        unsafe {
            title(PAGE + 40).write_volatile(0x11);
            title(2 * PAGE + 50).write_volatile(0x22);
        }
        // The shaders marked pages 0 and 1.
        let bits = b.data as *mut u32;
        for n in [base, base + 1] {
            // SAFETY: a dword of the mapped bitmap.
            unsafe { *bits.add(n / 32) |= 1 << (n % 32) };
        }
        copy_back_marks(&[], false);
        // SAFETY: bytes of our pool.
        let read = |at: usize| unsafe { title(at).read_volatile() };
        assert_eq!(read(10), 0xaa, "the shader's byte, page 0");
        assert_eq!((read(PAGE + 20), read(PAGE + 40)), (0xbb, 0x11), "both writers' bytes, page 1");
        assert_eq!((read(2 * PAGE + 30), read(2 * PAGE + 50)), (0, 0x22), "page 2 not marked");
        // Only this copy's two bits: parallel tests' copies may hold bits in
        // the same dwords.
        let left = [base, base + 1]
            .iter()
            .filter(|&&n| {
                // SAFETY: a dword of the mapped bitmap.
                let word = unsafe { *bits.add(n / 32) };
                word & 1 << (n % 32) != 0
            })
            .count();
        assert_eq!(left, 0, "bits cleared");
        assert_eq!(copy.copy_back(p, len, &[]), 0, "nothing kept to compare");
        unregister(p, p as u64);
    }

    /// Marks of shaders using title memory stay out of title memory while
    /// their results are held back, and are cleared; the other shaders'
    /// marks of the same submission go back.
    #[test]
    fn marks_of_shaders_using_title_memory_are_held_back() {
        let p = crate::watch::tests::pool(2);
        let len = 2 * PAGE;
        register(p, p as u64, len as u64);
        let (Some(copy), Some(others), Some(memory)) = (find(p as u64, len as u64), bitmap(false), bitmap(true)) else {
            unregister(p, p as u64);
            return; // no host GPU here
        };
        let base = copy.marks.expect("tracked memory gets mark bits") as usize;
        copy.update(p, len);
        copy.writing(p, len, true);
        copy.write_host((p + 10) as u64, &[0xaa]);
        copy.write_host((p + PAGE + 10) as u64, &[0xbb]);
        // Page 0 marked by another shader, page 1 by one using title memory.
        for (b, n) in [(others, base), (memory, base + 1)] {
            // SAFETY: a dword of the mapped bitmap.
            unsafe { *(b.data as *mut u32).add(n / 32) |= 1 << (n % 32) };
        }
        copy_back_marks(&[], true);
        // SAFETY: bytes of our pool.
        let read = |at: usize| unsafe { ((p + at) as *const u8).read_volatile() };
        assert_eq!((read(10), read(PAGE + 10)), (0xaa, 0), "page 1 held back");
        // SAFETY: a dword of the mapped bitmap.
        assert_eq!(unsafe { *(memory.data as *const u32).add((base + 1) / 32) } & 1 << ((base + 1) % 32), 0, "cleared");
        unregister(p, p as u64);
    }

    /// What a frame withheld leaves every host copy of its memory at the
    /// frame's end (run textures9's shape): in frame N a program using title
    /// memory writes a record into the copies and the frame misses, so it
    /// stays out of title memory but in the copies for the GPU; in frame
    /// N+1, which misses nothing, a program reading the record must see
    /// title memory's bytes, not frame N's, in a title buffer's copy and in
    /// an inner buffer's copy of the same pages alike.
    #[test]
    fn what_a_frame_withholds_leaves_the_copies_at_its_end() {
        let p = crate::watch::tests::pool(2);
        let len = 2 * PAGE;
        // SAFETY: a byte of our pool.
        unsafe { ((p + 16) as *mut u8).write_volatile(0x11) };
        register(p, p as u64, len as u64);
        register(p + 1, p as u64, PAGE as u64);
        let (Some(outer), Some(inner)) = (find(p as u64, len as u64), find(p as u64 + 16, 16)) else {
            unregister(p + 1, p as u64);
            unregister(p, p as u64);
            return; // no host GPU here
        };
        assert!(!Arc::ptr_eq(&outer, &inner));
        let byte = |c: &Shadow| c.read_host(p as u64 + 16, 1)[0] & 0xff;
        for c in [&outer, &inner] {
            let (start, n) = c.range();
            c.update(start as usize, n as usize);
            c.writing(start as usize, n as usize, false);
            c.write_host(p as u64 + 16, &[0x22]);
            c.keep_writes(start as usize, n as usize);
        }
        crate::memory::drop_withheld(p + 16, 4);
        assert_eq!((byte(&outer), byte(&inner)), (0x22, 0x22), "frame N: the GPU's copies keep it");
        crate::memory::log_frame();
        for c in [&outer, &inner] {
            let (start, n) = c.range();
            c.update(start as usize, n as usize);
        }
        assert_eq!((byte(&outer), byte(&inner)), (0x11, 0x11), "frame N+1: title memory's bytes");
        // SAFETY: as above.
        assert_eq!(unsafe { ((p + 16) as *const u8).read_volatile() }, 0x11, "title memory untouched");
        unregister(p + 1, p as u64);
        unregister(p, p as u64);
    }

    #[test]
    fn shared_page_marked_copies_do_not_drop_or_dirty_each_other() {
        let p = crate::watch::tests::pool(1);
        register(p, p as u64, 4096);
        register(p + 4096, (p + 4096) as u64, 4096);
        let (Some(copy_a), Some(copy_b), Some(b)) = (
            find(p as u64, 4096),
            find((p + 4096) as u64, 4096),
            bitmap(false),
        ) else {
            unregister(p, p as u64);
            unregister(p + 4096, (p + 4096) as u64);
            return;
        };
        // As above: hold the submission lock over the process-wide bitmap.
        let _submissions = host().expect("a bitmap means a host").lock_submissions();
        let base_a = copy_a.marks.expect("tracked A gets mark bits");
        let base_b = copy_b.marks.expect("tracked B gets mark bits");
        copy_a.update(p, 4096);
        copy_b.update(p + 4096, 4096);

        // Both sub-page copies are written via unmarked-range heap UAVs (not armed).
        copy_a.write_host((p + 10) as u64, &[0xaa]);
        copy_b.write_host((p + 4096 + 20) as u64, &[0xbb]);
        let bits = b.data as *mut u32;
        for n in [base_a as usize, base_b as usize] {
            // SAFETY: a dword of the mapped bitmap.
            unsafe { *bits.add(n / 32) |= 1 << (n % 32) };
        }
        copy_back_marks(&[], false);

        // SAFETY: bytes of our pool.
        let read = |at: usize| unsafe { ((p + at) as *const u8).read_volatile() };
        assert_eq!(read(10), 0xaa, "copy A GPU write kept");
        assert_eq!(read(4096 + 20), 0xbb, "copy B GPU write not dropped");
        assert_eq!(copy_a.update(p, 4096), 0, "copy A stays taken and guarded");
        assert_eq!(
            copy_b.update(p + 4096, 4096),
            0,
            "copy B stays taken and guarded"
        );

        // Page remains guarded: title CPU write + armed marked GPU write merge cleanly.
        copy_a.writing(p, 4096, true);
        copy_a.write_host((p + 30) as u64, &[0xcc]);
        // SAFETY: byte inside our guarded pool.
        unsafe { ((p + 40) as *mut u8).write_volatile(0x11) };
        // SAFETY: a dword of the mapped bitmap.
        unsafe { *bits.add(base_a as usize / 32) |= 1 << (base_a as usize % 32) };
        copy_back_marks(&[], false);
        assert_eq!((read(30), read(40)), (0xcc, 0x11));

        unregister(p, p as u64);
        unregister(p + 4096, (p + 4096) as u64);
    }

    #[test]
    fn subpage_marked_copy_retakes_its_page_across_rounds() {
        let p = crate::watch::tests::pool(1);
        let addr = p + 4096;
        register(addr, addr as u64, 8192);
        let (Some(copy), Some(b)) = (find(addr as u64, 8192), bitmap(false)) else {
            unregister(addr, addr as u64);
            return;
        };
        // As above: hold the submission lock over the process-wide bitmap.
        let _submissions = host().expect("a bitmap means a host").lock_submissions();
        let base = copy.marks.expect("tracked memory gets mark bits") as usize;
        copy.update(addr, 8192);

        // Round 1: marked GPU write on sub-64-KiB copy must re-take its page.
        copy.write_host((addr + 10) as u64, &[0xaa]);
        let bits = b.data as *mut u32;
        // SAFETY: a dword of the mapped bitmap.
        unsafe { *bits.add(base / 32) |= 1 << (base % 32) };
        copy_back_marks(&[], false);

        // Round 2: refresh only a sub-slice `[addr + 4096, addr + 8192)` then
        // write `[addr + 20]` via marked UAV (unarmed).
        assert_eq!(copy.update(addr + 4096, 4096), 0);
        copy.write_host((addr + 20) as u64, &[0xbb]);
        // SAFETY: a dword of the mapped bitmap.
        unsafe { *bits.add(base / 32) |= 1 << (base % 32) };
        copy_back_marks(&[], false);

        // SAFETY: bytes of our pool.
        let read = |at: usize| unsafe { ((addr + at) as *const u8).read_volatile() };
        assert_eq!((read(10), read(20)), (0xaa, 0xbb));
        unregister(addr, addr as u64);
    }

    /// A marked copy-back advances the snapshot of the declared range over
    /// the same page with the copy, so that range's copy-back does not put
    /// the title's bytes it took back over a title write made since.
    #[test]
    fn a_declared_copy_back_after_a_marked_one_keeps_newer_title_writes() {
        let p = crate::watch::tests::pool(1);
        register(p, p as u64, PAGE as u64);
        let Some(copy) = find(p as u64, PAGE as u64) else {
            unregister(p, p as u64);
            return; // no host GPU here
        };
        // SAFETY: bytes of our pool.
        let write = |at: usize, v: u8| unsafe { ((p + at) as *mut u8).write_volatile(v) };
        // SAFETY: bytes of our pool.
        let read = |at: usize| unsafe { ((p + at) as *const u8).read_volatile() };
        copy.update(p, PAGE);
        copy.writing(p, PAGE, true);
        copy.writing(p, PAGE, false);
        // A shader writes byte 10; the title writes byte 40 meanwhile.
        copy.write_host((p + 10) as u64, &[0xaa]);
        write(40, 0x11);
        let slice = copy.prepare_marked(p, Vec::new()).expect("an armed page is prepared");
        assert!(slice.commit(&mut Vec::new()).is_some_and(|(n, _, _)| n > 0));
        drop(slice);
        // The title writes both bytes again before the declared copy-back.
        write(10, 0x22);
        write(40, 0x23);
        assert_eq!(copy.copy_back(p, PAGE, &[]), 0);
        assert_eq!([read(10), read(40)], [0x22, 0x23]);
        unregister(p, p as u64);
    }

    /// A withheld range (a list whose shaders read memory with no copy,
    /// #1976) keeps its bytes out of title memory, but only its own: another
    /// list's valid write on the same marked page still goes back, and the
    /// withheld bytes stay in the copy. Subsequent marks, a withheld range on
    /// another page, and a final submission without withholding are controls.
    #[test]
    fn a_withheld_range_does_not_suppress_the_pages_other_writes() {
        let p = crate::watch::tests::pool(2);
        let len = 2 * PAGE;
        register(p, p as u64, len as u64);
        let (Some(copy), Some(b)) = (find(p as u64, len as u64), bitmap(false)) else {
            unregister(p, p as u64);
            return; // no host GPU here
        };
        // The bitmap and `copy_back_marks` are process-wide: hold the
        // submission lock so no parallel test's submission consumes the
        // bits this test sets before its own copy-back.
        let _submissions = host().expect("a bitmap means a host").lock_submissions();
        let base = copy.marks.expect("tracked memory gets mark bits") as usize;
        let bits = b.data as *mut u32;
        let mark = |page: usize| {
            let n = base + page;
            // SAFETY: a dword of the mapped bitmap.
            unsafe { *bits.add(n / 32) |= 1 << (n % 32) };
        };
        // SAFETY: four-byte ranges of our pool.
        let title = |at: usize| unsafe { ((p + at) as *const [u8; 4]).read_volatile() };
        let host = |at: usize| {
            let mut bytes = [0; 4];
            // SAFETY: four bytes inside the copy's persistent mapping.
            unsafe { std::ptr::copy_nonoverlapping(copy.data.add(at), bytes.as_mut_ptr(), 4) };
            bytes
        };
        copy.update(p, len);
        copy.writing(p, len, true);
        // List A writes at 64 (valid), list B at 256 (withheld); one page.
        copy.write_host((p + 64) as u64, &[0x5a; 4]);
        copy.write_host((p + 256) as u64, &[0xa5; 4]);
        mark(0);
        copy_back_marks(&[(copy.clone(), p + 256, 4)], false);
        assert_eq!(title(64), [0x5a; 4], "the valid write on the shared page goes back");
        assert_eq!(title(256), [0; 4], "the withheld write does not");
        assert_eq!(host(256), [0xa5; 4], "and the copy keeps it");
        // B on the next page instead: A's page goes back whole.
        copy.writing(p, len, true);
        copy.write_host((p + 128) as u64, &[0x3c; 4]);
        copy.write_host((p + PAGE + 256) as u64, &[0xc3; 4]);
        mark(0);
        mark(1);
        copy_back_marks(&[(copy.clone(), p + PAGE + 256, 4)], false);
        assert_eq!(title(128), [0x3c; 4]);
        assert_eq!(title(PAGE + 256), [0; 4]);
        // Nothing withheld: both pages' subsequent writes go back.
        copy.writing(p, len, true);
        copy.write_host((p + 512) as u64, &[0x77; 4]);
        copy.write_host((p + PAGE + 512) as u64, &[0x88; 4]);
        mark(0);
        mark(1);
        copy_back_marks(&[], false);
        assert_eq!((title(512), title(PAGE + 512)), ([0x77; 4], [0x88; 4]));
        copy.copy_back(p, len, &[]);
        unregister(p, p as u64);
    }

    /// A title write consumes its saved snapshot in the first copy-back, but
    /// a withheld range leaves the page changed. A second marked submission
    /// without a refresh must still publish its new write without reviving
    /// the old withheld bytes over another title write.
    #[test]
    fn a_title_write_does_not_suppress_the_next_marked_submission() {
        let p = crate::watch::tests::pool(1);
        register(p, p as u64, PAGE as u64);
        let (Some(copy), Some(b)) = (find(p as u64, PAGE as u64), bitmap(false)) else {
            unregister(p, p as u64);
            return; // no host GPU here
        };
        // The bitmap and `copy_back_marks` are process-wide: hold the
        // submission lock so no parallel test's submission consumes the
        // bits this test sets before its own copy-back.
        let _submissions = host().expect("a bitmap means a host").lock_submissions();
        let base = copy.marks.expect("tracked memory gets mark bits") as usize;
        let bits = b.data as *mut u32;
        let mark = || {
            // SAFETY: a dword of the mapped bitmap.
            unsafe { *bits.add(base / 32) |= 1 << (base % 32) };
        };
        // SAFETY: four-byte ranges of our pool.
        let title = |at: usize| unsafe { ((p + at) as *const [u8; 4]).read_volatile() };
        let write_title = |at: usize, bytes: [u8; 4]| {
            // SAFETY: a four-byte range of our pool.
            unsafe { ((p + at) as *mut [u8; 4]).write_volatile(bytes) };
        };
        copy.update(p, PAGE);
        copy.writing(p, PAGE, true);
        write_title(256, [0x3c; 4]);
        copy.write_host((p + 64) as u64, &[0x5a; 4]);
        copy.write_host((p + 256) as u64, &[0xa5; 4]);
        mark();
        copy_back_marks(&[(copy.clone(), p + 256, 4)], false);
        assert_eq!(title(64), [0x5a; 4]);
        assert_eq!(title(256), [0x3c; 4]);

        // No intervening refresh: the title writes the still-changed page
        // again, then a later marked submission writes a disjoint range.
        write_title(256, [0x6e; 4]);
        copy.writing(p, PAGE, true);
        copy.write_host((p + 512) as u64, &[0x77; 4]);
        mark();
        copy_back_marks(&[], false);
        assert_eq!(title(512), [0x77; 4], "the next submission is not skipped");
        assert_eq!(title(256), [0x6e; 4], "the later title write survives");
        assert_eq!(copy.read_host((p + 256) as u64, 1), [0x6e6e6e6e]);
        copy.copy_back(p, PAGE, &[]);
        unregister(p, p as u64);
    }

    #[test]
    fn withheld_ranges_are_cut_out_without_losing_the_gaps() {
        let page = 0x2_0000;
        assert_eq!(outside_withheld(page, page + PAGE, &[]), [(page, page + PAGE)]);
        assert_eq!(
            outside_withheld(page, page + PAGE, &[(page + 256, page + 260)]),
            [(page, page + 256), (page + 260, page + PAGE)]
        );
        assert_eq!(
            outside_withheld(
                page,
                page + 100,
                &[(page + 80, page + 120), (page + 10, page + 40), (page + 20, page + 50)]
            ),
            [(page, page + 10), (page + 50, page + 80)]
        );
        assert!(outside_withheld(page, page + PAGE, &[(page, page + PAGE)]).is_empty());
    }

    /// Only the last bit of the bitmap is the sink for offsets past it; the
    /// bits before it in its last dword belong to copies.
    #[test]
    fn marks_in_the_last_dword_are_read_but_the_sink_bit_is_not() {
        let last = victory_shader::emit::VIEW_WRITE_BITS - 1;
        let mut bits = vec![0u32; (victory_shader::emit::VIEW_WRITE_BITS / 32) as usize];
        let w = bits.len() - 1;
        bits[w] = (1 << 31) | (1 << 30) | 1;
        assert_eq!(drain_marks(&mut bits, last), [last - 31, last - 1]);
        assert_eq!(bits[w], 0, "the sink bit is cleared too");
    }

    /// Mark bits are handed out in runs and come back joined.
    #[test]
    fn mark_runs_are_reused_and_joined() {
        let mut m = Marks::default();
        assert_eq!((m.take(3), m.take(2)), (Some(0), Some(3)));
        m.give_back(0, 3);
        assert_eq!(m.take(1), Some(0), "first fit");
        m.give_back(3, 2);
        assert_eq!(m.free.iter().collect::<Vec<_>>(), [(&1, &4)], "3-4 joins the free 1-2");
        m.give_back(0, 1);
        assert_eq!(m.free.iter().collect::<Vec<_>>(), [(&0, &5)], "and 0 joins them");
        assert_eq!(m.take(5), Some(0));
        assert_eq!(pages(0x1_fff0, 0x20), 2, "a range across a page boundary");
        assert_eq!(pages(0x2_0000, 0x1_0000), 1);
    }

    /// Remapping or freeing a title memory range while a submission's
    /// declared copy-back is pending must not overwrite the new mapping with
    /// stale GPU bytes, and must leave the remapped page dirty for the next
    /// refresh.
    #[test]
    fn remapping_while_copy_back_is_pending_keeps_new_title_bytes() {
        let p = crate::watch::tests::pool(2);
        let len = 2 * PAGE;
        // SAFETY: our pool.
        unsafe {
            (p as *mut u8).write_volatile(0x11);
            ((p + PAGE) as *mut u8).write_volatile(0x33);
        }
        register(p, p as u64, len as u64);
        let Some(copy) = find(p as u64, len as u64) else {
            unregister(p, p as u64);
            crate::watch::tests::free(p);
            return; // no host GPU here
        };
        copy.update(p, len);
        copy.writing(p, len, false);
        copy.write_host(p as u64, &[0xaa]);
        copy.write_host((p + PAGE) as u64, &[0xbb]);
        // Title remaps page 0 while the submission is pending and writes 0x22.
        crate::watch::title_memory_remapped(p, PAGE);
        // SAFETY: our pool.
        unsafe { (p as *mut u8).write_volatile(0x22) };
        assert_eq!(copy.copy_back(p, len, &[]), PAGE, "only unremapped page 1 writes back");
        // SAFETY: our pool.
        let title = |at: usize| unsafe { (at as *const u8).read_volatile() };
        assert_eq!(title(p), 0x22, "remapped page keeps the new mapping's bytes");
        assert_eq!(title(p + PAGE), 0xbb, "other page keeps the GPU write");
        assert!(!copy.clean(), "remapped page remains changed for next update");
        // Subsequent submission sees the new bytes after update and copies back.
        copy.update(p, PAGE);
        assert_eq!(copy.read_host(p as u64, 1)[0] & 0xff, 0x22);
        copy.writing(p, PAGE, false);
        copy.write_host(p as u64, &[0xcc]);
        assert_eq!(copy.copy_back(p, PAGE, &[]), PAGE);
        assert_eq!(title(p), 0xcc);
        unregister(p, p as u64);
        crate::watch::tests::free(p);
    }

    /// A title write before remapping saves pre-write bytes on an armed page;
    /// remapping discards them so `copy_back_marks` does not overwrite the new
    /// mapping with stale GPU writes.
    #[test]
    fn remapping_after_title_write_discards_saved_bytes_for_marked_copy_back() {
        let p = crate::watch::tests::pool(1);
        register(p, p as u64, PAGE as u64);
        let (Some(copy), Some(b)) = (find(p as u64, PAGE as u64), bitmap(false)) else {
            unregister(p, p as u64);
            crate::watch::tests::free(p);
            return; // no host GPU here
        };
        let base = copy.marks.expect("tracked memory gets mark bits") as usize;
        copy.update(p, PAGE);
        copy.writing(p, PAGE, true);
        // Title writes first (populating Saved::bytes), GPU writes, then title remaps.
        // SAFETY: our pool.
        unsafe { ((p + 40) as *mut u8).write_volatile(0x11) };
        copy.write_host((p + 10) as u64, &[0xaa]);
        // SAFETY: mapped bitmap dword.
        unsafe { *(b.data as *mut u32).add(base / 32) |= 1 << (base % 32) };
        crate::watch::title_memory_remapped(p, PAGE);
        // SAFETY: our pool.
        unsafe { ((p + 10) as *mut u8).write_volatile(0x22) };
        copy_back_marks(&[], false);
        assert_eq!(copy.copy_back(p, PAGE, &[]), 0);
        // SAFETY: our pool.
        let byte10 = unsafe { ((p + 10) as *const u8).read_volatile() };
        assert_eq!(byte10, 0x22, "remapped page is not overwritten by marked write");
        assert!(!copy.clean(), "remapped page stays changed");
        unregister(p, p as u64);
        crate::watch::tests::free(p);
    }

    thread_local! {
        static BEFORE_COPY_BACK_WRITE: std::cell::RefCell<Option<Box<dyn FnOnce()>>> =
            const { std::cell::RefCell::new(None) };
    }

    pub(crate) fn before_copy_back_write() {
        let hook = BEFORE_COPY_BACK_WRITE.with(|cell| cell.borrow_mut().take());
        if let Some(hook) = hook {
            hook();
        }
    }

    /// Pausing `copy_back_or_not` after its initial `was_remapped` check while
    /// another thread remaps the page (and finishes under `title_memory_remapping`)
    /// prevents `write_mapped` from landing in the replacement mapping.
    #[test]
    fn remapping_after_copy_back_check_before_write_skips_write_back() {
        let p = crate::watch::tests::pool(1);
        // SAFETY: our pool.
        unsafe { (p as *mut u8).write_volatile(0x11) };
        register(p, p as u64, PAGE as u64);
        let Some(copy) = find(p as u64, PAGE as u64) else {
            unregister(p, p as u64);
            crate::watch::tests::free(p);
            return; // no host GPU here
        };
        copy.update(p, PAGE);
        copy.writing(p, PAGE, false);
        copy.write_host(p as u64, &[0xaa]);
        BEFORE_COPY_BACK_WRITE.with(|cell| {
            *cell.borrow_mut() = Some(Box::new(move || {
                std::thread::spawn(move || {
                    let _remap = crate::watch::title_memory_remapping();
                    // SAFETY: our pool.
                    unsafe { (p as *mut u8).write_volatile(0x22) };
                    crate::watch::title_memory_remapped(p, PAGE);
                })
                .join()
                .unwrap();
            }));
        });
        assert_eq!(copy.copy_back(p, PAGE, &[]), 0);
        // SAFETY: our pool.
        let title = unsafe { (p as *const u8).read_volatile() };
        assert_eq!(title, 0x22, "paused copy-back must not overwrite remapped page");
        assert!(!copy.clean(), "remapped page remains dirty");
        unregister(p, p as u64);
        crate::watch::tests::free(p);
    }

    /// Pausing a marked copy-back (`MarkedSlice::commit`) after `t.saved()` has moved the pre-write
    /// snapshot out while another thread remaps the page prevents `write_mapped`
    /// from overwriting the replacement mapping.
    #[test]
    fn remapping_after_marked_snapshot_taken_before_write_skips_write_back() {
        let p = crate::watch::tests::pool(1);
        register(p, p as u64, PAGE as u64);
        let Some(copy) = find(p as u64, PAGE as u64) else {
            unregister(p, p as u64);
            crate::watch::tests::free(p);
            return; // no host GPU here
        };
        copy.update(p, PAGE);
        copy.writing(p, PAGE, true);
        // Title writes first (populating Saved::bytes), then GPU writes.
        // SAFETY: our pool.
        unsafe { ((p + 40) as *mut u8).write_volatile(0x11) };
        copy.write_host((p + 10) as u64, &[0xaa]);
        BEFORE_COPY_BACK_WRITE.with(|cell| {
            *cell.borrow_mut() = Some(Box::new(move || {
                std::thread::spawn(move || {
                    let _remap = crate::watch::title_memory_remapping();
                    // SAFETY: our pool.
                    unsafe { ((p + 10) as *mut u8).write_volatile(0x22) };
                    crate::watch::title_memory_remapped(p, PAGE);
                })
                .join()
                .unwrap();
            }));
        });
        let slice = copy.prepare_marked(p, Vec::new()).expect("an armed page is prepared");
        assert_eq!(slice.commit(&mut Vec::new()), None, "the remapped page is not written back");
        drop(slice);
        assert_eq!(copy.copy_back(p, PAGE, &[]), 0);
        // SAFETY: our pool.
        let byte10 = unsafe { ((p + 10) as *const u8).read_volatile() };
        assert_eq!(byte10, 0x22, "paused marked copy-back must not overwrite remapped page");
        assert!(!copy.clean(), "remapped page remains dirty");
        unregister(p, p as u64);
        crate::watch::tests::free(p);
    }

    /// A copy-back write waits while `title_memory_remapping()` is held (as
    /// across OS map/unmap + `title_memory_remapped`), then sees the remap.
    #[test]
    fn copy_back_write_waits_for_in_flight_remapping_guard() {
        let p = crate::watch::tests::pool(1);
        // SAFETY: our pool.
        unsafe { (p as *mut u8).write_volatile(0x11) };
        register(p, p as u64, PAGE as u64);
        let Some(copy) = find(p as u64, PAGE as u64) else {
            unregister(p, p as u64);
            crate::watch::tests::free(p);
            return; // no host GPU here
        };
        copy.update(p, PAGE);
        copy.writing(p, PAGE, false);
        copy.write_host(p as u64, &[0xaa]);
        let (entered_tx, entered_rx) = std::sync::mpsc::channel();
        let (remap_done_tx, remap_done_rx) = std::sync::mpsc::channel();
        let handle = std::thread::spawn(move || {
            let _remap = crate::watch::title_memory_remapping();
            entered_tx.send(()).unwrap();
            // Wait until copy-back has passed the pre-write hook and is waiting on MAPPING.
            std::thread::sleep(std::time::Duration::from_millis(20));
            // SAFETY: our pool.
            unsafe { (p as *mut u8).write_volatile(0x22) };
            crate::watch::title_memory_remapped(p, PAGE);
            drop(_remap);
            remap_done_tx.send(()).unwrap();
        });
        BEFORE_COPY_BACK_WRITE.with(|cell| {
            *cell.borrow_mut() = Some(Box::new(move || {
                entered_rx.recv().unwrap();
            }));
        });
        assert_eq!(copy.copy_back(p, PAGE, &[]), 0);
        remap_done_rx.recv().unwrap();
        handle.join().unwrap();
        // SAFETY: our pool.
        let title = unsafe { (p as *const u8).read_volatile() };
        assert_eq!(title, 0x22);
        unregister(p, p as u64);
        crate::watch::tests::free(p);
    }
}
