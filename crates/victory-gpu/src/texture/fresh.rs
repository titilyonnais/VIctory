//! Textures the host GPU wrote by a copy or resolve that the title then
//! writes again with its CPU (`VICTORY_OCCLUSION_FRESH`).
//!
//! A texture a host copy or resolve wrote ([`super::host_written`]) was
//! never uploaded from title memory again: a copy does not reach title
//! memory, so what lies there is older than what the card holds. But a
//! title that writes the texture's memory itself after that copy wants
//! that write seen, and it was dropped, for the rest of the run: the
//! occlusion texture the CPU rasterises its occluders into each frame and
//! `CS_InitDepthPyramidWithCpuOcclusion3x3` (dispatch 42683eebec081520)
//! folds into the depth pyramid kept the occluders of an old frame, and
//! hid fences, trees and poles that were in sight.
//!
//! So a print of the texture's title memory is taken when the host writes
//! it; a list that samples it afterwards looks at the memory again (once a
//! frame), and one found changed is the title's again: it is uploaded
//! from title memory, as any texture the title fills. Only textures of up
//! to `VICTORY_OCCLUSION_FRESH_MAX` bytes (4 MiB by default; 0: any) are
//! watched, the occluder textures being small; others are kept as the host
//! wrote them, as before. `VICTORY_OCCLUSION_FRESH=0`: all of them are.

use std::cell::RefCell;
use std::collections::HashMap;
use std::sync::Mutex;
use std::sync::atomic::{AtomicU64, Ordering::Relaxed};

use crate::object::This;

/// Whether textures the host wrote are uploaded again once the title
/// writes their memory (`VICTORY_OCCLUSION_FRESH=0`: never, as before).
pub(crate) fn on() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("VICTORY_OCCLUSION_FRESH").is_none_or(|v| v != "0"))
}

/// The most bytes of title memory a watched texture has
/// (`VICTORY_OCCLUSION_FRESH_MAX`, in bytes; 0: no limit).
fn most() -> u64 {
    static MOST: std::sync::OnceLock<u64> = std::sync::OnceLock::new();
    *MOST.get_or_init(|| {
        std::env::var("VICTORY_OCCLUSION_FRESH_MAX")
            .ok()
            .and_then(|v| v.trim().parse().ok())
            .map_or(DEFAULT_MOST, |v: u64| if v == 0 { u64::MAX } else { v })
    })
}

const DEFAULT_MOST: u64 = 4 << 20;

/// Up to this many bytes a texture's memory is hashed whole; a larger one
/// is sampled ([`super::fingerprint`], 48 KiB).
const WHOLE: u64 = 256 << 10;

/// What is known of a texture the host wrote.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct Mark {
    /// Its title memory: address and bytes.
    at: u64,
    len: u64,
    /// The print of that memory when the host last wrote the texture.
    print: u64,
    /// The frame it was last looked at in (`u64::MAX`: never).
    looked: u64,
}

/// What a look at a marked texture finds.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Look {
    /// Looked at already this frame: as it was.
    Again,
    /// Its memory is as the host found it.
    Same,
    /// The title wrote it since: it is the title's again.
    Written,
}

/// Judges a mark whose memory prints as `now` (asked only if needed) in
/// `frame`, and notes the look.
fn look(mark: &mut Mark, frame: u64, now: impl FnOnce(u64, u64) -> u64) -> Look {
    if mark.looked == frame {
        return Look::Again;
    }
    mark.looked = frame;
    if now(mark.at, mark.len) == mark.print { Look::Same } else { Look::Written }
}

/// Whether a texture of `len` bytes of title memory is watched.
fn watched_len(len: u64, most: u64) -> bool {
    len > 0 && len <= most
}

static MARKS: Mutex<Option<HashMap<usize, Mark>>> = Mutex::new(None);

/// Since the last report line: prints taken when the host wrote a
/// texture, looks at marked textures, textures found written by the title
/// (uploaded again), and textures the host wrote that are not watched (too
/// big, or not the title's to fill).
pub(crate) static STATS: [AtomicU64; 4] = [const { AtomicU64::new(0) }; 4];

thread_local! {
    static READ: RefCell<Vec<u8>> = const { RefCell::new(Vec::new()) };
}

/// A print of title memory `[at, at + len)`, never 0: of all of it up to
/// [`WHOLE`], sampled above. Pages not there count as zeros.
fn print(at: u64, len: u64) -> u64 {
    if len > WHOLE {
        return super::fingerprint(at, len);
    }
    READ.with(|read| {
        let mut read = read.borrow_mut();
        read.clear();
        read.resize(len as usize, 0);
        crate::host::copy_mapped(read.as_mut_ptr(), at as *const u8, len as usize, false);
        let d = super::digest(&read).0;
        (d as u64 ^ (d >> 64) as u64) | 1
    })
}

/// The host writes `resource` by a copy or resolve (while a list is
/// recorded): its title memory as it is now is the host's from here on.
pub(crate) fn mark(resource: This) {
    if !on() {
        return;
    }
    let span = crate::resource::title_fills(resource)
        .then(|| super::title_span(resource, 0))
        .flatten()
        .filter(|&(at, len)| at != 0 && watched_len(len, most()));
    let Some((at, len)) = span else {
        STATS[3].fetch_add(1, Relaxed);
        if let Some(map) = MARKS.lock().unwrap_or_else(|e| e.into_inner()).as_mut() {
            map.remove(&(resource as usize));
        }
        return;
    };
    let print = print(at, len);
    STATS[0].fetch_add(1, Relaxed);
    // The host copy no longer holds what it was last filled from: an
    // upload of those very bytes is not one "that brings nothing new".
    if let Some(copy) = crate::resource::made_copy(resource) {
        *copy.holds.lock().unwrap_or_else(|e| e.into_inner()) = None;
    }
    let mark = Mark { at, len, print, looked: u64::MAX };
    MARKS.lock().unwrap_or_else(|e| e.into_inner()).get_or_insert_with(HashMap::new).insert(resource as usize, mark);
}

/// A list samples `resource`, which the host wrote: whether the title has
/// written its memory since (once a frame). Then it is no longer marked.
pub(crate) fn written_since(resource: This) -> bool {
    if !on() {
        return false;
    }
    let Some(mut mark) = MARKS
        .lock()
        .unwrap_or_else(|e| e.into_inner())
        .as_ref()
        .and_then(|m| m.get(&(resource as usize)).copied())
    else {
        return false;
    };
    // (Printed with the table unlocked: other threads mark and look.)
    let found = look(&mut mark, crate::frame::current(), print);
    if found != Look::Again {
        STATS[1].fetch_add(1, Relaxed);
    }
    let mut marks = MARKS.lock().unwrap_or_else(|e| e.into_inner());
    let Some(map) = marks.as_mut() else {
        return false;
    };
    let key = resource as usize;
    // Marked again by a host write in between (or forgotten): that stands.
    if !map.get(&key).is_some_and(|now| now.print == mark.print && now.at == mark.at) {
        return false;
    }
    if found != Look::Written {
        if let Some(now) = map.get_mut(&key) {
            now.looked = mark.looked;
        }
        return false;
    }
    map.remove(&key);
    STATS[2].fetch_add(1, Relaxed);
    #[cfg(feature = "diagnostics")]
    sent_again(resource, mark.len);
    true
}

/// The title released `resource`.
pub(crate) fn forget(resource: usize) {
    if let Some(map) = MARKS.lock().unwrap_or_else(|e| e.into_inner()).as_mut() {
        map.remove(&resource);
    }
}

/// The first textures sent again, logged.
#[cfg(feature = "diagnostics")]
fn sent_again(resource: This, len: u64) {
    static SAID: AtomicU64 = AtomicU64::new(0);
    if SAID.fetch_add(1, Relaxed) >= 32 {
        return;
    }
    let desc = crate::resource::host_resource(resource).map(|t| {
        // SAFETY: a live resource.
        let d = unsafe { t.GetDesc() };
        (d.Width, d.Height, d.Format.0, d.MipLevels)
    });
    tracing::info!(
        resource = format_args!("{:#x}", resource as usize),
        bytes = len,
        host = ?desc,
        "a texture the host copied into was written by the title since: uploaded from title memory again (VICTORY_OCCLUSION_FRESH=0: kept)"
    );
}

#[cfg(any(test, feature = "diagnostics"))]
/// How the texels of a format are read for their range.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Texel {
    F32,
    F16,
    Unorm16,
    Unorm8,
    /// Anything else: 32-bit words as numbers.
    Word,
}

#[cfg(any(test, feature = "diagnostics"))]
impl Texel {
    /// Of a DXGI format number.
    pub(crate) fn of(format: i32) -> Texel {
        match format {
            2 | 6 | 16 | 39 | 40 | 41 => Texel::F32,
            10 | 34 | 54 => Texel::F16,
            11 | 35 | 53 | 55 | 56 => Texel::Unorm16,
            28 | 49 | 61 | 87 => Texel::Unorm8,
            _ => Texel::Word,
        }
    }
}

#[cfg(any(test, feature = "diagnostics"))]
/// The smallest and the largest value among `bytes` read as `texel`s
/// (NaNs left out); `None` if there is none.
pub(crate) fn range(bytes: &[u8], texel: Texel) -> Option<(f64, f64)> {
    let values: Box<dyn Iterator<Item = f64>> = match texel {
        Texel::F32 => Box::new(bytes.chunks_exact(4).map(|b| f64::from(f32::from_le_bytes([b[0], b[1], b[2], b[3]])))),
        Texel::F16 => Box::new(bytes.chunks_exact(2).map(|b| half(u16::from_le_bytes([b[0], b[1]])))),
        Texel::Unorm16 => Box::new(bytes.chunks_exact(2).map(|b| f64::from(u16::from_le_bytes([b[0], b[1]])) / 65535.0)),
        Texel::Unorm8 => Box::new(bytes.iter().map(|&b| f64::from(b) / 255.0)),
        Texel::Word => Box::new(bytes.chunks_exact(4).map(|b| f64::from(u32::from_le_bytes([b[0], b[1], b[2], b[3]])))),
    };
    values.filter(|v| !v.is_nan()).fold(None, |r, v| match r {
        None => Some((v, v)),
        Some((lo, hi)) => Some((lo.min(v), hi.max(v))),
    })
}

#[cfg(any(test, feature = "diagnostics"))]
/// A half-precision float.
fn half(h: u16) -> f64 {
    let sign = if h & 0x8000 != 0 { -1.0 } else { 1.0 };
    let exp = i32::from((h >> 10) & 0x1f);
    let frac = f64::from(h & 0x3ff);
    match exp {
        0 => sign * frac * 2f64.powi(-24),
        31 if frac == 0.0 => sign * f64::INFINITY,
        31 => f64::NAN,
        _ => sign * (1.0 + frac / 1024.0) * 2f64.powi(exp - 15),
    }
}

/// The watched dispatch: textures its tables read are looked at
/// (diagnostics). `VICTORY_OCCLUSION_WATCH`: a shader hash prefix (by
/// default `42683eeb`, `CS_InitDepthPyramidWithCpuOcclusion3x3`).
#[cfg(feature = "diagnostics")]
pub(crate) mod watch {
    use std::sync::Mutex;
    use std::sync::atomic::{AtomicU64, Ordering::Relaxed};

    use crate::object::This;

    thread_local! {
        /// The thread binds the tables of the watched dispatch.
        pub(crate) static READER: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
    }

    fn prefix() -> &'static str {
        static PREFIX: std::sync::OnceLock<String> = std::sync::OnceLock::new();
        PREFIX.get_or_init(|| {
            std::env::var("VICTORY_OCCLUSION_WATCH")
                .map(|v| v.trim().to_ascii_lowercase())
                .unwrap_or_else(|_| "42683eeb".into())
        })
    }

    /// Whether a shader of this hash is the watched dispatch's.
    pub(crate) fn watched(hash: &str) -> bool {
        let prefix = prefix();
        !prefix.is_empty() && hash.starts_with(prefix)
    }

    /// Since the last report line.
    #[derive(Default)]
    struct Seen {
        binds: u64,
        reads: u64,
        /// Of them: the host copied into it (kept, not uploaded), the card
        /// owns it (target, depth or unordered access: never uploaded),
        /// its title memory changed since the last look.
        host_written: u64,
        card_owned: u64,
        changed: u64,
        /// The range of all texels looked at, and of the last texture.
        all: Option<(f64, f64)>,
        last: Option<(f64, f64)>,
        /// By texture: frame and print of the last look.
        prints: std::collections::HashMap<usize, (u64, u64)>,
    }

    static SEEN: Mutex<Option<Seen>> = Mutex::new(None);
    static SAID: AtomicU64 = AtomicU64::new(0);

    /// The watched dispatch is bound.
    pub(crate) fn bound() {
        SEEN.lock().unwrap_or_else(|e| e.into_inner()).get_or_insert_with(Seen::default).binds += 1;
    }

    /// The watched dispatch reads `resource` (a shader resource view),
    /// looked at once a frame: the range of the texels its title memory
    /// holds (in any order: the range does not depend on the tiling).
    pub(crate) fn read(resource: This) {
        let frame = crate::frame::current();
        {
            let seen = SEEN.lock().unwrap_or_else(|e| e.into_inner());
            if seen.as_ref().and_then(|s| s.prints.get(&(resource as usize))).is_some_and(|p| p.0 == frame) {
                return;
            }
        }
        let host_written = crate::texture::host_wrote(resource as usize);
        let flags = crate::resource::host_copy(resource).map(|t| t.flags());
        let card_owned = flags.is_some_and(|f| f & 0x7 != 0);
        let format = crate::resource::host_resource(resource).map(|t| {
            // SAFETY: a live resource.
            let d = unsafe { t.GetDesc() };
            (d.Format.0, d.Width, d.Height)
        });
        let span = crate::texture::title_span(resource, 0).filter(|&(at, len)| at != 0 && len > 0);
        let (print, range) = match (span, format) {
            (Some((at, len)), Some((f, ..))) => {
                let len = len.min(4 << 20) as usize;
                let mut bytes = vec![0u8; len];
                crate::host::copy_mapped(bytes.as_mut_ptr(), at as *const u8, len, false);
                let d = crate::texture::digest(&bytes).0;
                (d as u64 | 1, super::range(&bytes, super::Texel::of(f)))
            }
            _ => (0, None),
        };
        let mut seen = SEEN.lock().unwrap_or_else(|e| e.into_inner());
        let seen = seen.get_or_insert_with(Seen::default);
        let before = seen.prints.insert(resource as usize, (frame, print));
        let changed = before.is_some_and(|(_, p)| p != print);
        seen.reads += 1;
        seen.host_written += u64::from(host_written);
        seen.card_owned += u64::from(card_owned);
        seen.changed += u64::from(changed);
        if let Some((lo, hi)) = range {
            seen.all = Some(seen.all.map_or((lo, hi), |(a, b)| (a.min(lo), b.max(hi))));
            seen.last = Some((lo, hi));
        }
        if SAID.fetch_add(1, Relaxed) < 24 {
            tracing::info!(
                resource = format_args!("{:#x}", resource as usize),
                frame,
                host = ?format,
                ?flags,
                host_written,
                card_owned,
                changed,
                ?range,
                "occlusion watch: the watched dispatch reads a texture (VICTORY_OCCLUSION_WATCH)"
            );
        }
    }

    /// The report line (with the frame-rate line).
    pub(crate) fn line(frames: f64) {
        let [marks, looks, sent, unwatched] = super::STATS.each_ref().map(|c| c.swap(0, Relaxed) as f64 / frames);
        let seen = SEEN.lock().unwrap_or_else(|e| e.into_inner()).as_mut().map(std::mem::take);
        let seen = seen.unwrap_or_default();
        if marks + looks + sent + unwatched == 0.0 && seen.binds == 0 {
            return;
        }
        let per = |n: u64| n as f64 / frames;
        let range = |r: Option<(f64, f64)>| r.map_or_else(|| "none".to_string(), |(lo, hi)| format!("{lo:.6}..{hi:.6}"));
        tracing::info!(
            "occlusion: textures the host copied into, per frame: {marks:.1} marked, {looks:.1} looked at again, {sent:.2} found \
             written by the title since and uploaded again (VICTORY_OCCLUSION_FRESH={}), {unwatched:.1} not watched; \
             dispatch {} bound {:.1} times per frame, read {:.1} textures through its tables ({:.1} host-copied, {:.1} card-owned, \
             {:.1} changed since the frame before), texels {} (last {})",
            u8::from(super::on()),
            prefix(),
            per(seen.binds),
            per(seen.reads),
            per(seen.host_written),
            per(seen.card_owned),
            per(seen.changed),
            range(seen.all),
            range(seen.last),
        );
    }
}

#[cfg(test)]
mod tests {
    use super::{Look, Mark, Texel, half, look, range, watched_len};

    fn mark(print: u64) -> Mark {
        Mark { at: 0x1000, len: 64, print, looked: u64::MAX }
    }

    #[test]
    fn a_texture_the_title_wrote_after_the_host_is_found_once_a_frame() {
        let mut m = mark(7);
        // As the host left it: nothing to send.
        assert_eq!(look(&mut m, 10, |_, _| 7), Look::Same);
        // Looked at already this frame: the memory is not read again.
        assert_eq!(look(&mut m, 10, |_, _| panic!("read twice in a frame")), Look::Again);
        // The title wrote it in the next frame: it is the title's again.
        assert_eq!(look(&mut m, 11, |_, _| 8), Look::Written);
    }

    #[test]
    fn a_mark_looks_at_its_own_memory() {
        let mut m = mark(5);
        let mut asked = None;
        let _ = look(&mut m, 1, |at, len| {
            asked = Some((at, len));
            5
        });
        assert_eq!(asked, Some((0x1000, 64)));
        assert_eq!(m.looked, 1);
    }

    #[test]
    fn only_small_textures_are_watched() {
        assert!(watched_len(64 << 10, 4 << 20));
        assert!(watched_len(4 << 20, 4 << 20));
        assert!(!watched_len((4 << 20) + 1, 4 << 20));
        assert!(!watched_len(0, u64::MAX));
        assert!(watched_len(1 << 40, u64::MAX));
    }

    #[test]
    fn the_range_of_float_texels_leaves_nans_out() {
        let mut bytes = Vec::new();
        for v in [0.5f32, -1.0, f32::NAN, 0.25, 1.0] {
            bytes.extend_from_slice(&v.to_le_bytes());
        }
        assert_eq!(range(&bytes, Texel::F32), Some((-1.0, 1.0)));
        assert_eq!(range(&[], Texel::F32), None);
        assert_eq!(range(&f32::NAN.to_le_bytes(), Texel::F32), None);
    }

    #[test]
    fn the_range_of_other_texels() {
        assert_eq!(range(&[0, 255, 51], Texel::Unorm8), Some((0.0, 1.0)));
        let words: Vec<u8> = [0u16, 65535].iter().flat_map(|w| w.to_le_bytes()).collect();
        assert_eq!(range(&words, Texel::Unorm16), Some((0.0, 1.0)));
        let halves: Vec<u8> = [0x3c00u16, 0xc000, 0x3800].iter().flat_map(|w| w.to_le_bytes()).collect();
        assert_eq!(range(&halves, Texel::F16), Some((-2.0, 1.0)));
        assert_eq!(range(&7u32.to_le_bytes(), Texel::Word), Some((7.0, 7.0)));
    }

    #[test]
    fn halves_and_formats() {
        assert_eq!(half(0x3c00), 1.0);
        assert_eq!(half(0x0001), 2f64.powi(-24));
        assert!(half(0x7c01).is_nan());
        assert_eq!(half(0xfc00), f64::NEG_INFINITY);
        assert_eq!(Texel::of(41), Texel::F32);
        assert_eq!(Texel::of(40), Texel::F32);
        assert_eq!(Texel::of(54), Texel::F16);
        assert_eq!(Texel::of(56), Texel::Unorm16);
        assert_eq!(Texel::of(61), Texel::Unorm8);
        assert_eq!(Texel::of(71), Texel::Word);
    }
}
