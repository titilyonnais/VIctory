//! Title memory the title asked to be write-combining, and what of it the
//! copy-back leaves on the card (`VICTORY_SKIP_WC_BACK=1` or
//! `VICTORY_WC_LAZY=1`, both off by default).
//!
//! The console's CPU writes write-combining memory but does not read it back
//! (a read there is uncached and very slow), so what shaders wrote there is
//! for the GPU. The copy-back still fetched it: in the city, per frame, about
//! 800 pieces of 64 KiB (52 MB) of copies in the card's memory were fetched
//! and merged into title memory, most of the "copying back" time.
//!
//! `VICTORY_SKIP_WC_BACK=1` (port of fxceless/victory-fork #868): pieces of
//! those copies that lie wholly in write-combining memory stay on the card,
//! and title memory keeps its older bytes. It assumes nothing reads that
//! memory after the card wrote it; a green patch by the boat says something
//! does.
//!
//! `VICTORY_WC_LAZY=1`, the safe form (#868's "first touch", #1178's lazy
//! copy-back): such a piece stays on the card, and its title memory is made
//! inaccessible until something wants it. Then it is fetched from the card,
//! alone, and merged as the copy-back would have, before the access goes on:
//! the title's reads and writes (a fault, [`on_fault`]), our own reads of it
//! (a fault too), the kernel's writes into it (a file read, [`take_back`]
//! from `crate::watch::kernel_writing`), memory let go, and other host copies
//! of the same memory, which would not see it written ([`take_back_for`],
//! when they ask what was written). A piece taken back is fetched by the
//! copy-back from then on ("hot"). Only whole kernel pages of the title's
//! own read-write memory are left: not views of sections, which the title
//! also sees elsewhere.
//!
//! What counts is what the title asked for (`XMemVirtualAlloc`'s protection,
//! bit 0x400): the runtime drops that bit for the host.

use std::collections::{BTreeMap, HashMap, HashSet};
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, LazyLock, Mutex, OnceLock};

use crate::shadow::Shadow;

/// The console protection's write-combining bit.
const WRITE_COMBINE: u32 = 0x400;
/// The kernel's page: the unit of a protection.
const KERNEL_PAGE: usize = 4096;
/// The unit of what is remembered as ever left.
const BLOCK: usize = 64 << 10;

/// Whether pieces of write-combining title memory stay on the card, one way
/// or the other. (Leaving copies in the PC's memory alone as well, with no
/// snapshot and no compare, was tried: the picture went black.)
pub(crate) fn on() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| {
        let skip = std::env::var_os("VICTORY_SKIP_WC_BACK").is_some_and(|v| v == "1");
        if lazy() {
            tracing::warn!(
                "VICTORY_WC_LAZY=1: what shaders wrote to copies in the card's memory of write-combining title \
                 memory stays there, and is fetched when something touches that memory"
            );
        } else if skip {
            tracing::warn!(
                "VICTORY_SKIP_WC_BACK=1: what shaders wrote to copies in the card's memory of write-combining title \
                 memory is not copied back (experiment)"
            );
        }
        skip || lazy()
    })
}

/// Whether what stays on the card is fetched when something touches it
/// (`VICTORY_WC_LAZY=1`; it wins over `VICTORY_SKIP_WC_BACK`).
pub(crate) fn lazy() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("VICTORY_WC_LAZY").is_some_and(|v| v == "1"))
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
    // (What is left there was taken back by `crate::watch::title_memory_freed`.)
    let mut r = runs();
    let len = if len == 0 { r.reserved.remove(&at).unwrap_or(0) } else { len };
    r.cut(at, at.saturating_add(len));
}

/// Per frame: pieces left on the card and their bytes.
pub(crate) static LEFT: [AtomicU64; 2] = [const { AtomicU64::new(0) }; 2];

/// Per frame, with `VICTORY_WC_LAZY=1`: pieces taken back, their bytes, the
/// microseconds that took, and the times something wanted them.
static BACK: [AtomicU64; 4] = [const { AtomicU64::new(0) }; 4];
/// Per frame: the times something wanted pieces, by [`Why`].
static WHY: [AtomicU64; 5] = [const { AtomicU64::new(0) }; 5];
/// Per frame: write-combining pieces fetched as before, by why: taken back
/// before (hot), not whole kernel pages, not the title's own read-write
/// memory, the protection refused.
static NOT: [AtomicU64; 4] = [const { AtomicU64::new(0) }; 4];
const HOT: usize = 0;
const UNALIGNED: usize = 1;
const NOT_PLAIN: usize = 2;
const REFUSED: usize = 3;
/// Since the start: pieces taken back that the card could not give (the
/// title memory keeps its older bytes).
static LOST: AtomicU64 = AtomicU64::new(0);

/// What wanted pieces left on the card.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Why {
    /// A read of their title memory faulted (the title's or ours).
    Read,
    /// A write to it faulted.
    Write,
    /// The kernel is about to write it (a file read).
    Kernel,
    /// Another host copy of it asks what was written there, or the memory
    /// goes.
    Copy,
    /// Nothing but this table holds their copy any more.
    Orphan,
}

/// A piece of a copy in the card's memory left there: `len` bytes at
/// offset `lo` of the copy.
struct Piece {
    shadow: Arc<Shadow>,
    lo: usize,
    len: usize,
}

#[derive(Default)]
struct Table {
    /// The pieces left, by title address and copy.
    left: BTreeMap<(usize, u64), Piece>,
    /// The longest piece ever left.
    longest: usize,
    /// Pieces taken back (copy, offset): fetched as before from then on.
    hot: HashSet<(u64, usize)>,
    /// The 64 KiB blocks that ever held a piece left.
    ever: HashSet<usize>,
    /// Title memory written by taking back, for the guards to be told.
    landed: Vec<(usize, usize)>,
    /// Copies of pieces taken back, let go outside a fault.
    grave: Vec<Arc<Shadow>>,
    /// What took pieces back, the first times, to be said.
    notes: Vec<String>,
    noted: usize,
    /// The frame last tended.
    frame: u64,
    #[cfg(not(test))]
    gpu: Option<Gpu>,
    #[cfg(not(test))]
    no_gpu: bool,
}

static TABLE: LazyLock<Mutex<Table>> = LazyLock::new(Mutex::default);
/// Pieces in the table.
static COUNT: AtomicUsize = AtomicUsize::new(0);
/// Whether a piece was ever left (faults are looked at from then on).
static EVER: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

fn table() -> std::sync::MutexGuard<'static, Table> {
    TABLE.lock().unwrap_or_else(|e| e.into_inner())
}

thread_local! {
    /// This thread is taking pieces back (a fault in that is not ours).
    static INSIDE: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}

/// This thread takes pieces back until it is dropped.
struct Inside;

impl Inside {
    fn enter() -> Option<Inside> {
        let free = INSIDE.try_with(|c| !c.replace(true)).unwrap_or(false);
        free.then_some(Inside)
    }
}

impl Drop for Inside {
    fn drop(&mut self) {
        let _ = INSIDE.try_with(|c| c.set(false));
    }
}

/// The keys of the pieces of `left` that hold some of `seeds`, and of
/// those that share a kernel page with them, and so on: what must be taken
/// back together (letting one in lets in the pages it shares with
/// another). `len` gives a piece's bytes; `longest` is the most.
fn together<V>(
    left: &BTreeMap<(usize, u64), V>,
    len: impl Fn(&V) -> usize,
    longest: usize,
    seeds: &[(usize, usize)],
) -> Vec<(usize, u64)> {
    let mut keys: Vec<(usize, u64)> = Vec::new();
    let mut todo: Vec<(usize, usize)> = seeds
        .iter()
        .filter(|&&(a, b)| a < b)
        .map(|&(a, b)| (a / KERNEL_PAGE * KERNEL_PAGE, b.div_ceil(KERNEL_PAGE) * KERNEL_PAGE))
        .collect();
    while let Some((from, to)) = todo.pop() {
        for (k, v) in left.range((from.saturating_sub(longest), 0)..(to, 0)) {
            let end = k.0 + len(v);
            if end > from && !keys.contains(k) {
                keys.push(*k);
                todo.push((k.0, end.div_ceil(KERNEL_PAGE) * KERNEL_PAGE));
            }
        }
    }
    keys.sort_unstable();
    keys
}

/// Whether `[from, to)` is whole kernel pages of the title's own committed
/// read-write memory (no view of a section: the title sees those elsewhere
/// too).
fn plain(from: usize, to: usize) -> bool {
    use windows_sys::Win32::System::Memory::{
        MEM_COMMIT, MEM_PRIVATE, MEMORY_BASIC_INFORMATION, PAGE_READWRITE, VirtualQuery,
    };
    if from >= to || !from.is_multiple_of(KERNEL_PAGE) || !to.is_multiple_of(KERNEL_PAGE) {
        return false;
    }
    let mut at = from;
    while at < to {
        // SAFETY: plain data filled in by the query.
        let mut info: MEMORY_BASIC_INFORMATION = unsafe { std::mem::zeroed() };
        // SAFETY: a valid out pointer of the right size.
        let n = unsafe { VirtualQuery(at as *const _, &mut info, size_of_val(&info)) };
        if n == 0 || info.State != MEM_COMMIT || info.Type != MEM_PRIVATE || info.Protect != PAGE_READWRITE {
            return false;
        }
        at = (info.BaseAddress as usize).saturating_add(info.RegionSize);
    }
    true
}

/// Gives `[from, from + len)` the protection `to`; the one it had before.
fn protect(from: usize, len: usize, to: u32) -> Option<u32> {
    let mut old = 0;
    // SAFETY: a change of protection of title memory the caller names.
    let done = unsafe { windows_sys::Win32::System::Memory::VirtualProtect(from as *const _, len, to, &mut old) } != 0;
    done.then_some(old)
}

/// Whether an access to `at` (`write`: a write) can be made now.
fn accessible(at: usize, write: bool) -> bool {
    use windows_sys::Win32::System::Memory::{
        MEM_COMMIT, MEMORY_BASIC_INFORMATION, PAGE_READONLY, PAGE_READWRITE, VirtualQuery,
    };
    // SAFETY: plain data filled in by the query.
    let mut info: MEMORY_BASIC_INFORMATION = unsafe { std::mem::zeroed() };
    // SAFETY: a valid out pointer of the right size.
    let n = unsafe { VirtualQuery(at as *const _, &mut info, size_of_val(&info)) };
    n != 0 && info.State == MEM_COMMIT && (info.Protect == PAGE_READWRITE || (!write && info.Protect == PAGE_READONLY))
}

/// Whether the piece `[lo, hi)` of the copy `shadow` stays on the card: the
/// switch is on and its title memory is all write-combining. With
/// `VICTORY_WC_LAZY=1` also: it was not taken back before, it is whole
/// kernel pages of the title's own read-write memory, and that memory could
/// be made inaccessible.
pub(crate) fn leave(shadow: &Arc<Shadow>, lo: usize, hi: usize) -> bool {
    if !on() || lo >= hi {
        return false;
    }
    let from = shadow.span().0 as usize + lo;
    if !runs().all_wc(from, from + (hi - lo)) {
        return false;
    }
    if !lazy() {
        count_left(hi - lo);
        return true;
    }
    tend();
    leave_lazy(shadow, lo, hi)
}

fn count_left(bytes: usize) {
    LEFT[0].fetch_add(1, Ordering::Relaxed);
    LEFT[1].fetch_add(bytes as u64, Ordering::Relaxed);
}

/// [`leave`] with `VICTORY_WC_LAZY=1`, for a piece of write-combining memory.
fn leave_lazy(shadow: &Arc<Shadow>, lo: usize, hi: usize) -> bool {
    use windows_sys::Win32::System::Memory::{PAGE_NOACCESS, PAGE_READWRITE};
    let (from, len) = (shadow.span().0 as usize + lo, hi - lo);
    let id = shadow.id();
    let mut t = table();
    if t.left.contains_key(&(from, id)) {
        count_left(len);
        return true;
    }
    if t.hot.contains(&(id, lo)) {
        NOT[HOT].fetch_add(1, Ordering::Relaxed);
        return false;
    }
    if !from.is_multiple_of(KERNEL_PAGE) || !len.is_multiple_of(KERNEL_PAGE) {
        NOT[UNALIGNED].fetch_add(1, Ordering::Relaxed);
        return false;
    }
    if !plain(from, from + len) {
        NOT[NOT_PLAIN].fetch_add(1, Ordering::Relaxed);
        return false;
    }
    #[cfg(not(test))]
    if t.gpu.is_none() {
        if t.no_gpu {
            return false;
        }
        match crate::host::host().map(|h| Gpu::new(&h.device)) {
            Some(Ok(gpu)) => t.gpu = Some(gpu),
            failed => {
                let error = failed.and_then(|r| r.err()).map(|e| e.to_string()).unwrap_or_default();
                tracing::warn!(%error, "VICTORY_WC_LAZY=1: no copy queue to fetch pieces with; they are fetched as before");
                t.no_gpu = true;
                return false;
            }
        }
    }
    match protect(from, len, PAGE_NOACCESS) {
        Some(PAGE_READWRITE) => {}
        Some(old) => {
            // Changed between the look and the change: as it was.
            let _ = protect(from, len, old);
            NOT[REFUSED].fetch_add(1, Ordering::Relaxed);
            return false;
        }
        None => {
            NOT[REFUSED].fetch_add(1, Ordering::Relaxed);
            return false;
        }
    }
    t.left.insert((from, id), Piece { shadow: shadow.clone(), lo, len });
    t.longest = t.longest.max(len);
    for block in from / BLOCK..(from + len).div_ceil(BLOCK) {
        t.ever.insert(block);
    }
    COUNT.fetch_add(1, Ordering::Relaxed);
    EVER.store(true, Ordering::Release);
    count_left(len);
    true
}

/// Takes the pieces `keys` of the table back: fetched from the card, their
/// title memory let in again, what shaders wrote merged into it. Under the
/// table's lock, inside [`Inside`]. `at` and `code`: the address wanted and
/// the instruction that wanted it, for the notes.
fn take_back_keys(t: &mut Table, keys: &[(usize, u64)], why: Why, at: usize, code: usize) {
    use windows_sys::Win32::System::Memory::PAGE_READWRITE;
    let began = std::time::Instant::now();
    let pieces: Vec<(usize, Piece)> = keys.iter().filter_map(|k| t.left.remove(k).map(|p| (k.0, p))).collect();
    if pieces.is_empty() {
        return;
    }
    COUNT.fetch_sub(pieces.len(), Ordering::Relaxed);
    let mut card = Vec::new();
    let fetched = fetch(t, &pieces, &mut card);
    let (mut put, mut bytes, mut title) = (0usize, 0u64, Vec::new());
    let mut offset = 0;
    for (from, p) in &pieces {
        let _ = protect(*from, p.len, PAGE_READWRITE);
        if fetched {
            put += p.shadow.lazy_merge(p.lo, &card[offset..offset + p.len], &mut title).unwrap_or(0);
            t.landed.push((*from, p.len));
        } else if LOST.fetch_add(1, Ordering::Relaxed) == 0 {
            tracing::warn!(at = format_args!("{from:#x}"), "VICTORY_WC_LAZY=1: a piece could not be fetched from the card");
        }
        offset += p.len;
        bytes += p.len as u64;
        t.hot.insert((p.shadow.id(), p.lo));
    }
    let us = began.elapsed().as_micros() as u64;
    BACK[0].fetch_add(pieces.len() as u64, Ordering::Relaxed);
    BACK[1].fetch_add(bytes, Ordering::Relaxed);
    BACK[2].fetch_add(us, Ordering::Relaxed);
    BACK[3].fetch_add(1, Ordering::Relaxed);
    WHY[why as usize].fetch_add(1, Ordering::Relaxed);
    if t.noted < 64 {
        t.noted += 1;
        let (from, p) = &pieces[0];
        let (start, size) = p.shadow.span();
        // SAFETY: no arguments.
        let thread = unsafe { windows_sys::Win32::System::Threading::GetCurrentThreadId() };
        t.notes.push(format!(
            "{why:?} of {at:#x} by code at {code:#x} (thread {thread}), frame {}: {} pieces taken back ({bytes} bytes, \
             {put} written, {us} us), the first {from:#x}+{:#x}, offset {:#x} of the copy of {start:#x}+{size:#x} (copy {})",
            crate::frame::current(),
            pieces.len(),
            p.len,
            p.lo,
            p.shadow.id()
        ));
    }
    t.grave.extend(pieces.into_iter().map(|(_, p)| p.shadow));
}

/// Fetches the pieces' bytes from the card into `out`, one after the other.
#[cfg(not(test))]
fn fetch(t: &mut Table, pieces: &[(usize, Piece)], out: &mut Vec<u8>) -> bool {
    let Some(gpu) = t.gpu.as_mut() else {
        return false;
    };
    let list: Vec<(&windows::Win32::Graphics::Direct3D12::ID3D12Resource, usize, usize)> =
        pieces.iter().map(|(_, p)| (&p.shadow.buffer, p.lo, p.len)).collect();
    gpu.fetch(&list, out).is_ok()
}

/// For tests: the card's side of each copy, by copy.
#[cfg(test)]
static TEST_CARD: Mutex<BTreeMap<u64, Vec<u8>>> = Mutex::new(BTreeMap::new());

#[cfg(test)]
fn fetch(_: &mut Table, pieces: &[(usize, Piece)], out: &mut Vec<u8>) -> bool {
    let card = TEST_CARD.lock().unwrap_or_else(|e| e.into_inner());
    out.clear();
    for (_, p) in pieces {
        let Some(side) = card.get(&p.shadow.id()) else {
            return false;
        };
        out.extend_from_slice(&side[p.lo..p.lo + p.len]);
    }
    true
}

/// A fault at `at` (`write`: a write), of the instruction at `code`: if it is
/// title memory of pieces left on the card, they are taken back and the
/// access can be made again (true). True as well if another thread took
/// them back since this one faulted.
pub(crate) fn on_fault(at: usize, write: bool, code: usize) -> bool {
    if !EVER.load(Ordering::Acquire) {
        return false;
    }
    let Some(_inside) = Inside::enter() else {
        return false;
    };
    let mut t = table();
    let keys = together(&t.left, |p| p.len, t.longest, &[(at, at + 1)]);
    if !keys.is_empty() {
        take_back_keys(&mut t, &keys, if write { Why::Write } else { Why::Read }, at, code);
        return true;
    }
    t.ever.contains(&(at / BLOCK)) && accessible(at, write)
}

/// Something other than the title's CPU is about to touch `[at, at + len)`
/// (`why`): the pieces left there are taken back first.
pub(crate) fn take_back(at: usize, len: usize, why: Why) {
    if COUNT.load(Ordering::Acquire) == 0 {
        return;
    }
    let Some(_inside) = Inside::enter() else {
        return;
    };
    let mut t = table();
    let keys = together(&t.left, |p| p.len, t.longest, &[(at, at.saturating_add(len.max(1)))]);
    if !keys.is_empty() {
        take_back_keys(&mut t, &keys, why, at, 0);
    }
}

/// A host copy of `[from, to)` asks what was written there (`asker`: the
/// copy, if it is a buffer's; None for a texture's): pieces of other copies
/// left there are taken back, so that it finds them written. Its own are not
/// (it holds their bytes on the card).
pub(crate) fn take_back_for(asker: Option<u64>, from: usize, to: usize) {
    if COUNT.load(Ordering::Acquire) == 0 || from >= to {
        return;
    }
    let Some(_inside) = Inside::enter() else {
        return;
    };
    let mut t = table();
    let others: Vec<(usize, usize)> = together(&t.left, |p| p.len, t.longest, &[(from, to)])
        .into_iter()
        .filter(|k| Some(k.1) != asker && k.0 < to)
        .filter_map(|k| t.left.get(&k).map(|p| (k.0, k.0 + p.len)))
        .filter(|&(a, b)| a < to && b > from)
        .collect();
    if others.is_empty() {
        return;
    }
    let keys = together(&t.left, |p| p.len, t.longest, &others);
    take_back_keys(&mut t, &keys, Why::Copy, from, 0);
}

/// Whether pieces are left on the card now.
pub(crate) fn any_left() -> bool {
    COUNT.load(Ordering::Acquire) != 0
}

/// Once a frame, outside faults: the guards are told of title memory that
/// taking back wrote, copies only the table holds have their pieces taken
/// back and are let go, and the first notes are said.
fn tend() {
    let frame = crate::frame::current();
    let (landed, notes, grave, orphans) = {
        let mut t = table();
        if t.frame == frame {
            return;
        }
        t.frame = frame;
        let mut held: HashMap<u64, (usize, usize)> = HashMap::new();
        for p in t.left.values() {
            held.entry(p.shadow.id()).or_insert((0, Arc::strong_count(&p.shadow))).0 += 1;
        }
        let orphans: HashSet<u64> = held.into_iter().filter(|(_, (n, all))| n == all).map(|(id, _)| id).collect();
        (std::mem::take(&mut t.landed), std::mem::take(&mut t.notes), std::mem::take(&mut t.grave), orphans)
    };
    if !orphans.is_empty()
        && let Some(_inside) = Inside::enter()
    {
        let mut t = table();
        let keys: Vec<(usize, u64)> = t.left.keys().filter(|k| orphans.contains(&k.1)).copied().collect();
        let keys = together(&t.left, |p| p.len, t.longest, &keys.iter().map(|k| (k.0, k.0 + 1)).collect::<Vec<_>>());
        take_back_keys(&mut t, &keys, Why::Orphan, 0, 0);
    }
    for (at, len) in landed {
        crate::watch::landing(at, len);
    }
    for note in notes {
        tracing::info!("VICTORY_WC_LAZY=1: {note}");
    }
    drop(grave);
}

/// The meter's lines, per frame over `frames`.
pub(crate) fn meter(frames: f64) {
    if !on() {
        return;
    }
    let take = |a: &AtomicU64| a.swap(0, Ordering::Relaxed) as f64 / frames;
    let (n, bytes) = (take(&LEFT[0]), take(&LEFT[1]));
    tracing::info!(
        "write-combining title memory not copied back, per frame ({}): {n:.0} pieces of copies in the card's \
         memory left there ({:.1} MB)",
        if lazy() { "VICTORY_WC_LAZY=1" } else { "VICTORY_SKIP_WC_BACK=1" },
        bytes / 1e6
    );
    if lazy() {
        let back = BACK.each_ref().map(take);
        let why = WHY.each_ref().map(take);
        let not = NOT.each_ref().map(take);
        tracing::info!(
            "write-combining pieces taken back on demand, per frame (VICTORY_WC_LAZY=1): {:.2} pieces ({:.2} MB, \
             {:.3} ms) at {:.2} times: {:.2} reads, {:.2} writes, {:.2} for the kernel, {:.2} for other copies, \
             {:.2} orphans; fetched as before: {:.0} hot, {:.0} not whole pages, {:.0} not plain memory, {:.0} \
             refused; {} left now, {} lost since the start",
            back[0],
            back[1] / 1e6,
            back[2] / 1e3,
            back[3],
            why[0],
            why[1],
            why[2],
            why[3],
            why[4],
            not[0],
            not[1],
            not[2],
            not[3],
            COUNT.load(Ordering::Relaxed),
            LOST.load(Ordering::Relaxed)
        );
    }
}

/// What fetches pieces from the card outside the copy-back: a copy queue,
/// list, fence and readback buffer of its own, used under the table's lock
/// alone (a fault may come from a thread that holds any other lock).
#[cfg(not(test))]
struct Gpu {
    queue: windows::Win32::Graphics::Direct3D12::ID3D12CommandQueue,
    allocator: windows::Win32::Graphics::Direct3D12::ID3D12CommandAllocator,
    list: windows::Win32::Graphics::Direct3D12::ID3D12GraphicsCommandList,
    fence: windows::Win32::Graphics::Direct3D12::ID3D12Fence,
    event: windows::Win32::Foundation::HANDLE,
    value: u64,
    back: windows::Win32::Graphics::Direct3D12::ID3D12Resource,
    data: *const u8,
}

// SAFETY: Direct3D objects are free-threaded; the mapping is used under the
// table's lock alone.
#[cfg(not(test))]
unsafe impl Send for Gpu {}

/// The readback buffer's bytes: what one fetch brings.
#[cfg(not(test))]
const FETCH_BYTES: usize = 4 << 20;

#[cfg(not(test))]
impl Gpu {
    fn new(device: &windows::Win32::Graphics::Direct3D12::ID3D12Device) -> windows::core::Result<Gpu> {
        use windows::Win32::Graphics::Direct3D12::*;
        use windows::Win32::Graphics::Dxgi::Common::DXGI_SAMPLE_DESC;
        // SAFETY: plain object creation and mapping.
        unsafe {
            let queue: ID3D12CommandQueue = device.CreateCommandQueue(&D3D12_COMMAND_QUEUE_DESC {
                Type: D3D12_COMMAND_LIST_TYPE_COPY,
                Priority: D3D12_COMMAND_QUEUE_PRIORITY_HIGH.0,
                ..Default::default()
            })?;
            let allocator: ID3D12CommandAllocator = device.CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_COPY)?;
            let list: ID3D12GraphicsCommandList = device.CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_COPY, &allocator, None)?;
            list.Close()?;
            let fence: ID3D12Fence = device.CreateFence(0, D3D12_FENCE_FLAG_NONE)?;
            let event = windows::Win32::System::Threading::CreateEventW(None, false, false, None)?;
            let heap = D3D12_HEAP_PROPERTIES { Type: D3D12_HEAP_TYPE_READBACK, ..Default::default() };
            let desc = D3D12_RESOURCE_DESC {
                Dimension: D3D12_RESOURCE_DIMENSION_BUFFER,
                Width: FETCH_BYTES as u64,
                Height: 1,
                DepthOrArraySize: 1,
                MipLevels: 1,
                SampleDesc: DXGI_SAMPLE_DESC { Count: 1, Quality: 0 },
                Layout: D3D12_TEXTURE_LAYOUT_ROW_MAJOR,
                ..Default::default()
            };
            let mut back: Option<ID3D12Resource> = None;
            device.CreateCommittedResource(&heap, D3D12_HEAP_FLAG_NONE, &desc, D3D12_RESOURCE_STATE_COPY_DEST, None, &mut back)?;
            let back = back.ok_or_else(windows::core::Error::empty)?;
            let mut data = std::ptr::null_mut();
            back.Map(0, None, Some(&mut data))?;
            Ok(Gpu { queue, allocator, list, fence, event, value: 0, back, data: data as *const u8 })
        }
    }

    /// Fetches `pieces` (copy, offset, bytes) into `out`, one after the
    /// other, as many at a time as the readback buffer holds, and waits.
    fn fetch(
        &mut self,
        pieces: &[(&windows::Win32::Graphics::Direct3D12::ID3D12Resource, usize, usize)],
        out: &mut Vec<u8>,
    ) -> windows::core::Result<()> {
        use windows::Win32::Graphics::Direct3D12::ID3D12CommandList;
        use windows::core::Interface;
        out.clear();
        let mut i = 0;
        while i < pieces.len() {
            let first = i;
            let mut bytes = 0;
            while i < pieces.len() && bytes + pieces[i].2 <= FETCH_BYTES {
                bytes += pieces[i].2;
                i += 1;
            }
            if i == first {
                return Err(windows::core::Error::empty());
            }
            // SAFETY: our own objects, under the table's lock; the copies
            // are live (the table holds them); the list's last run was waited
            // for.
            unsafe {
                self.allocator.Reset()?;
                self.list.Reset(&self.allocator, None)?;
                let mut at = 0u64;
                for &(copy, lo, n) in &pieces[first..i] {
                    self.list.CopyBufferRegion(&self.back, at, copy, lo as u64, n as u64);
                    at += n as u64;
                }
                self.list.Close()?;
                self.queue.ExecuteCommandLists(&[Some(self.list.cast::<ID3D12CommandList>()?)]);
                self.value += 1;
                self.queue.Signal(&self.fence, self.value)?;
                if self.fence.GetCompletedValue() < self.value {
                    self.fence.SetEventOnCompletion(self.value, self.event)?;
                    windows::Win32::System::Threading::WaitForSingleObject(self.event, windows::Win32::System::Threading::INFINITE);
                }
                out.extend_from_slice(std::slice::from_raw_parts(self.data, bytes));
            }
        }
        Ok(())
    }
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

#[cfg(test)]
mod lazy_tests {
    use super::*;
    use windows_sys::Win32::System::Memory::{
        MEM_COMMIT, MEM_RELEASE, MEM_RESERVE, MEMORY_BASIC_INFORMATION, PAGE_NOACCESS, PAGE_READWRITE, VirtualAlloc,
        VirtualFree, VirtualQuery,
    };

    #[test]
    fn pieces_that_share_pages_are_taken_back_together() {
        let k = KERNEL_PAGE;
        let base = 0x10_0000;
        let mut left: BTreeMap<(usize, u64), usize> = BTreeMap::new();
        left.insert((base, 1), 16 * k);
        // Another copy's piece over the second half of the first.
        left.insert((base + 8 * k, 2), 16 * k);
        // And a third over the end of that one.
        left.insert((base + 20 * k, 3), 16 * k);
        // Apart.
        left.insert((0x20_0000, 1), 16 * k);
        let len = |n: &usize| *n;
        let all = vec![(base, 1), (base + 8 * k, 2), (base + 20 * k, 3)];
        assert_eq!(together(&left, len, 16 * k, &[(base + 5, base + 6)]), all);
        assert_eq!(together(&left, len, 16 * k, &[(base + 35 * k, base + 35 * k + 1)]), all);
        assert_eq!(together(&left, len, 16 * k, &[(0x20_0000 + 16 * k - 1, 0x20_0000 + 16 * k)]), vec![(0x20_0000, 1)]);
        assert!(together(&left, len, 16 * k, &[(0x20_0000 + 16 * k, 0x20_0000 + 17 * k)]).is_empty());
        assert!(together(&left, len, 16 * k, &[(base, base)]).is_empty());
    }

    /// Title memory of the title's own, read-write.
    struct Memory(usize);

    impl Memory {
        fn new(len: usize, fill: u8) -> Memory {
            // SAFETY: a fresh allocation.
            let at = unsafe { VirtualAlloc(std::ptr::null(), len, MEM_COMMIT | MEM_RESERVE, PAGE_READWRITE) } as usize;
            assert_ne!(at, 0);
            // SAFETY: inside it.
            unsafe { std::ptr::write_bytes(at as *mut u8, fill, len) };
            Memory(at)
        }
    }

    impl Drop for Memory {
        fn drop(&mut self) {
            // SAFETY: ours.
            unsafe { VirtualFree(self.0 as *mut _, 0, MEM_RELEASE) };
        }
    }

    fn protection(at: usize) -> u32 {
        // SAFETY: plain data filled in by the query.
        let mut info: MEMORY_BASIC_INFORMATION = unsafe { std::mem::zeroed() };
        // SAFETY: a valid out pointer of the right size.
        unsafe { VirtualQuery(at as *const _, &mut info, size_of_val(&info)) };
        info.Protect
    }

    /// A copy in the card's memory of `[title, title + len)`: its CPU side
    /// holds what title memory holds now, its card side `card`.
    fn copy(title: usize, len: usize, card: Vec<u8>) -> Arc<Shadow> {
        let side: &'static mut [u8] = Box::leak(vec![0u8; len].into_boxed_slice());
        // SAFETY: title memory the caller made.
        side.copy_from_slice(unsafe { std::slice::from_raw_parts(title as *const u8, len) });
        let copy = Shadow::unbacked(title as u64, side.as_mut_ptr(), len as u64, true);
        TEST_CARD.lock().unwrap().insert(copy.id(), card);
        copy
    }

    fn bytes(at: usize, len: usize) -> Vec<u8> {
        // SAFETY: title memory the test made.
        unsafe { std::slice::from_raw_parts(at as *const u8, len) }.to_vec()
    }

    #[test]
    fn a_piece_left_on_the_card_comes_back_at_the_first_read_or_write() {
        crate::watch::install();
        let memory = Memory::new(2 * BLOCK, 0x11);
        let title = memory.0;
        let mut card = vec![0x11u8; 2 * BLOCK];
        card[100..200].fill(0x77);
        card[BLOCK + 4..BLOCK + 12].fill(0x66);
        let copy = copy(title, 2 * BLOCK, card);
        // The title wrote a byte the copy has not taken yet.
        // SAFETY: inside the memory.
        unsafe { std::ptr::write_volatile((title + 300) as *mut u8, 0x33) };
        let before = BACK[0].load(Ordering::Relaxed);
        assert!(leave_lazy(&copy, 0, BLOCK));
        assert!(leave_lazy(&copy, BLOCK, 2 * BLOCK));
        assert!(leave_lazy(&copy, 0, BLOCK), "a piece still left stays left");
        assert_eq!(protection(title), PAGE_NOACCESS);
        assert_eq!(protection(title + BLOCK), PAGE_NOACCESS);
        // A read: what shaders wrote is there, and the title's byte too.
        // SAFETY: inside the memory.
        let read = unsafe { std::ptr::read_volatile((title + 150) as *const u8) };
        assert_eq!(read, 0x77);
        assert_eq!(protection(title), PAGE_READWRITE);
        assert_eq!(bytes(title + 98, 4), [0x11, 0x11, 0x77, 0x77]);
        assert_eq!(bytes(title + 300, 1), [0x33]);
        assert_eq!(protection(title + BLOCK), PAGE_NOACCESS, "the other piece is still left");
        // A write: the shaders' bytes come back under it.
        // SAFETY: inside the memory.
        unsafe { std::ptr::write_volatile((title + BLOCK + 5) as *mut u8, 0x55) };
        assert_eq!(bytes(title + BLOCK + 2, 12), [0x11, 0x11, 0x66, 0x55, 0x66, 0x66, 0x66, 0x66, 0x66, 0x66, 0x11, 0x11]);
        assert!(BACK[0].load(Ordering::Relaxed) >= before + 2);
        // Taken back once: fetched as before from then on.
        assert!(!leave_lazy(&copy, 0, BLOCK));
        assert!(!leave_lazy(&copy, BLOCK, 2 * BLOCK));
    }

    #[test]
    fn other_copies_and_the_kernel_take_pieces_back_and_the_copy_itself_does_not() {
        crate::watch::install();
        let memory = Memory::new(2 * BLOCK, 0x22);
        let title = memory.0;
        let mut card = vec![0x22u8; 2 * BLOCK];
        card[10..20].fill(0x99);
        card[BLOCK..BLOCK + 4].fill(0x88);
        let copy = copy(title, 2 * BLOCK, card);
        assert!(leave_lazy(&copy, 0, BLOCK));
        assert!(leave_lazy(&copy, BLOCK, 2 * BLOCK));
        take_back_for(Some(copy.id()), title, title + 2 * BLOCK);
        assert_eq!(protection(title), PAGE_NOACCESS, "the copy asking holds its own pieces");
        assert_eq!(protection(title + BLOCK), PAGE_NOACCESS);
        take_back_for(None, title + 8, title + 9);
        assert_eq!(protection(title), PAGE_READWRITE);
        assert_eq!(bytes(title + 9, 2), [0x22, 0x99]);
        assert_eq!(protection(title + BLOCK), PAGE_NOACCESS);
        take_back(title + BLOCK + 100, 10, Why::Kernel);
        assert_eq!(protection(title + BLOCK), PAGE_READWRITE);
        assert_eq!(bytes(title + BLOCK + 3, 2), [0x88, 0x22]);
    }

    #[test]
    fn only_whole_pages_of_plain_memory_are_left() {
        let memory = Memory::new(2 * BLOCK, 0);
        assert!(plain(memory.0, memory.0 + 2 * BLOCK));
        assert!(!plain(memory.0 + 1, memory.0 + BLOCK));
        assert!(!plain(memory.0, memory.0 + BLOCK - 1));
        assert!(!plain(memory.0, memory.0 + 2 * BLOCK + KERNEL_PAGE), "past the allocation");
        let _ = protect(memory.0 + BLOCK, KERNEL_PAGE, windows_sys::Win32::System::Memory::PAGE_READONLY);
        assert!(!plain(memory.0, memory.0 + 2 * BLOCK));
        assert!(plain(memory.0, memory.0 + BLOCK));
    }
}
