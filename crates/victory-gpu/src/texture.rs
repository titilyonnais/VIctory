//! Texture contents the title writes itself (decision 0007, "Texture
//! memory"): streamed pages, layout-library copies, movie frames, glyph
//! atlases. Such a texture lives in console memory in its tiled layout;
//! before a submission samples its host copy, the memory is detiled into a
//! staging buffer when its content changed, and the staging buffer is
//! copied into the host texture.
//!
//! "Changed" means a page of the memory changed since the last detiling
//! ([`crate::watch`], which follows writes through every mapping of the
//! same physical pages) in the runtime's memory; elsewhere, a content hash
//! of the whole memory at every submission that samples it.
//!
//! shortcut: a change detiles the whole texture, not the pages written.
//! Textures read through descriptor tables are brought up to date, and for
//! shaders that index the heap themselves every texture with a resource
//! object the heap views (`crate::bindless`); textures the title placed in
//! memory itself are not made for the heap (see `descriptor::host`). Plane
//! 0 is uploaded (every mip and array slice), depth and stencil planes are
//! not.

use std::sync::{Arc, Mutex};

use windows::Win32::Graphics::Direct3D12::*;

use crate::host::{HostList, host};
use crate::object::This;
use windows::core::Interface;

macro_rules! once {
    ($($arg:tt)*) => {{
        static ONCE: std::sync::Once = std::sync::Once::new();
        ONCE.call_once(|| tracing::warn!($($arg)*));
    }};
}

/// A texture's upload state: how a staging buffer lays out the host
/// texture's subresources, and what was last detiled. The buffer itself is
/// taken for each upload from a small pool shared by all textures
/// ([`staging_buffer`]; kept for each texture's life, the buffers took its
/// size again in system memory: 2.6-2.8 GB with placed textures made on
/// read, run integ2).
pub(crate) struct Staging {
    /// A staging buffer's bytes.
    size: u64,
    footprints: Vec<D3D12_PLACED_SUBRESOURCE_FOOTPRINT>,
    rows: Vec<u32>,
    layout: victory_formats::texture::Layout,
    /// Mips per array slice.
    mips: u32,
    /// The first level uploaded: the levels before it are never read
    /// ([`note_reads`]), and `prefix` bytes of the memory hold the rest.
    first: u32,
    prefix: usize,
    memory: *const u8,
    /// The memory's pages changed since detiled (none: not the runtime's
    /// memory, hashed instead), of its first `prefix` bytes.
    tracked: Option<crate::watch::Tracked>,
    /// Per subresource, the bytes of the texture's memory it lies in (from
    /// its start, whole layout blocks); empty where the layout gives none.
    spans: Vec<(usize, usize)>,
    last: Mutex<Last>,
    /// Uploads [`record_upload`] recorded, and the subresources they
    /// copied.
    uploads: std::sync::atomic::AtomicU64,
    copied: std::sync::atomic::AtomicU64,
}

/// What was last detiled: the hash of the memory (not tracked), or of each
/// subresource's bytes (tracked).
#[derive(Default)]
struct Last {
    hash: Option<blake3::Hash>,
    hashes: Vec<Option<blake3::Hash>>,
}

/// Per subresource of `layout` (`mips` per array slice; a 3D texture's
/// are its mips), the bytes of the texture's memory it lies in, rounded out
/// to the layout's blocks; empty if the layout cannot say.
fn spans(layout: &victory_formats::texture::Layout, count: u32, mips: u32) -> Vec<(usize, usize)> {
    use victory_formats::texture::Dimension;
    let d = &layout.desc;
    let block = layout.alignment.max(256);
    (0..count)
        .map(|i| {
            let (mip, slice) = (i % mips, i / mips);
            let m = layout.mips.get(mip as usize)?;
            let (z0, z1) = if d.dimension == Dimension::Tex3d { (0, m.depth.max(1) - 1) } else { (slice, slice) };
            let first = layout.offset(mip, 0, 0, z0).ok()?;
            let last = layout.offset(mip, m.width.max(1) - 1, m.height.max(1) - 1, z1).ok()? + u64::from(d.element_bytes);
            let (from, to) = (first / block * block, last.div_ceil(block) * block);
            Some((from as usize, (to.min(layout.size)) as usize))
        })
        .collect::<Option<Vec<_>>>()
        .unwrap_or_default()
}

/// Staging buffers for detiles to reuse, by size: the menu movie rewrites
/// the same textures every frame, and a fresh 12.7 MB of upload buffers a
/// frame cost ~5 ms (run textures8). Uploads run only inside
/// `Host::execute`, which waits for each submission, so the buffers of one
/// submission are free from the next on.
struct Buffers {
    /// (bytes, buffer, mapping), oldest first.
    free: std::collections::VecDeque<(u64, ID3D12Resource, usize)>,
    free_bytes: u64,
    /// Taken by the submission being prepared.
    used: Vec<(u64, ID3D12Resource, usize)>,
}

// SAFETY: the mappings are persistent mappings of the upload buffers.
unsafe impl Send for Buffers {}

static BUFFERS: Mutex<Buffers> = Mutex::new(Buffers {
    free: std::collections::VecDeque::new(),
    free_bytes: 0,
    used: Vec::new(),
});

/// Free staging buffers kept at most.
const KEPT_BYTES: u64 = 64 << 20;

/// Called before a submission's uploads: the buffers of the last one are
/// free again (the oldest dropped past `KEPT_BYTES`).
pub(crate) fn recycle() {
    {
        let mut c = CHUNKS.lock().unwrap_or_else(|e| e.into_inner());
        (c.current, c.used) = (0, 0);
        c.all.truncate(CHUNKS_KEPT);
    }
    let mut b = BUFFERS.lock().unwrap_or_else(|e| e.into_inner());
    let b = &mut *b;
    for e in b.used.drain(..) {
        b.free_bytes += e.0;
        b.free.push_back(e);
    }
    while b.free_bytes > KEPT_BYTES
        && let Some((n, ..)) = b.free.pop_front()
    {
        b.free_bytes -= n;
    }
}

/// A free staging buffer of `size` bytes, or a new one.
/// Sizes are taken in powers of two (at least 64 KiB), so a buffer serves
/// textures of other sizes too: a new shot's textures came in new sizes and
/// made buffers 10 % of the submitting thread (run evict5).
fn staging_buffer(size: u64) -> Option<(ID3D12Resource, *mut u8)> {
    let class = size.max(1 << 16).next_power_of_two();
    let taken = {
        let mut b = BUFFERS.lock().unwrap_or_else(|e| e.into_inner());
        let i = b.free.iter().position(|e| e.0 == class);
        i.and_then(|i| b.free.remove(i)).inspect(|_| b.free_bytes -= class)
    };
    // Made without the pool's lock (refreshes run on several threads).
    let e = match taken {
        Some(e) => e,
        None => {
            let (buffer, data) = host()?.upload_buffer(class)?;
            crate::host::name(&buffer, || "texture staging".into());
            (class, buffer, data as usize)
        }
    };
    let made = (e.1.clone(), e.2 as *mut u8);
    BUFFERS.lock().unwrap_or_else(|e| e.into_inner()).used.push(e);
    Some(made)
}

/// Staging space carved from large upload buffers, front to back: most
/// textures take a part of one instead of a buffer of their own (making
/// upload buffers was 10 % of the submitting thread in a new shot, run
/// evict5). Rewound at each submission ([`recycle`]): the last one has run.
struct Chunks {
    /// (buffer, mapping), [`CHUNK`] bytes each.
    all: Vec<(ID3D12Resource, usize)>,
    /// The chunk being carved and the bytes taken from it.
    current: usize,
    used: u64,
}

// SAFETY: the mappings are persistent mappings of the upload buffers.
unsafe impl Send for Chunks {}

static CHUNKS: Mutex<Chunks> = Mutex::new(Chunks { all: Vec::new(), current: 0, used: 0 });

const CHUNK: u64 = 32 << 20;
/// Chunks kept between submissions (more are made while one needs them).
const CHUNKS_KEPT: usize = 16;

/// `size` bytes of staging space: a buffer, the offset of the space in it
/// (a multiple of 512, as texture copies need) and its mapping there.
/// Space larger than a chunk is a buffer of its own ([`staging_buffer`]).
fn staging_space(size: u64) -> Option<(ID3D12Resource, u64, *mut u8)> {
    if size > CHUNK {
        return staging_buffer(size).map(|(b, data)| (b, 0, data));
    }
    let mut c = CHUNKS.lock().unwrap_or_else(|e| e.into_inner());
    loop {
        if let Some((buffer, data)) = c.all.get(c.current) {
            let at = c.used.next_multiple_of(512);
            if at + size <= CHUNK {
                let made = (buffer.clone(), at, (*data + at as usize) as *mut u8);
                c.used = at + size;
                return Some(made);
            }
            (c.current, c.used) = (c.current + 1, 0);
            continue;
        }
        let (buffer, data) = host()?.upload_buffer(CHUNK)?;
        crate::host::name(&buffer, || "texture staging chunk".into());
        c.all.push((buffer, data as usize));
    }
}

// SAFETY: the title memory is only read, under `last`'s lock.
unsafe impl Send for Staging {}
// SAFETY: as above.
unsafe impl Sync for Staging {}

/// RAII pin keeping a destination `HostTexture` alive against eviction while
/// a recorded list holds a pending `host_writes` mark for it. Dropping without
/// [`PendingHostWrite::record`] releases the pin without marking `host_written`.
pub(crate) struct PendingHostWrite(Arc<crate::resource::HostTexture>);

impl PendingHostWrite {
    /// The pinned host `ID3D12Resource`, guaranteed to match the pinned copy.
    pub(crate) fn resource(&self) -> &ID3D12Resource {
        &self.0.texture
    }

    /// Transfers this pin to `list`'s deferred step so `host_written` is set on
    /// submission and the pin is released when `list` resets or drops.
    pub(crate) fn record(self, list: &HostList) {
        use std::sync::atomic::Ordering::Release;
        let guard = self;
        list.defer_host(move || {
            guard
                .0
                .pending_host_writes
                .fetch_or(crate::resource::HOST_WRITTEN, Release);
            guard.0.host_written.store(true, Release);
        });
    }
}

impl Drop for PendingHostWrite {
    fn drop(&mut self) {
        self.0
            .pending_host_writes
            .fetch_sub(1, std::sync::atomic::Ordering::Release);
    }
}

/// Atomically acquires `resource`'s current host texture and pins it against
/// concurrent eviction before any GPU write is recorded into it. If eviction
/// already claimed the copy with `EVICTED`, detaches it and acquires a fresh
/// replacement.
pub(crate) fn prepare_host_write(resource: This) -> Option<PendingHostWrite> {
    use std::sync::atomic::Ordering::{AcqRel, Acquire};
    for _ in 0..16 {
        let copy = crate::resource::host_copy(resource)?;
        if copy
            .pending_host_writes
            .fetch_update(AcqRel, Acquire, |cur| {
                (cur & crate::resource::EVICTED == 0).then_some(cur + 1)
            })
            .is_ok()
        {
            return Some(PendingHostWrite(copy));
        }
        crate::resource::detach_if_same(resource, &copy);
    }
    None
}

/// A host copy or resolve that `list` records writes `resource`: from the
/// submission that runs `list` on, its host texture's contents are the
/// host's, not title memory's, which stays as it was. Marked when that
/// submission is made, not when `list` is recorded: a list recorded before
/// it but run after it must not upload title memory over the host's texels
/// (submissions run in the order they are made, which need not be the
/// order their lists were recorded in), and one run before it must still
/// get its upload.
#[cfg(test)]
pub(crate) fn host_writes(list: &HostList, resource: This) {
    if let Some(write) = prepare_host_write(resource) {
        write.record(list);
    }
}

fn is_host_written(resource: This) -> bool {
    crate::resource::host_copy(resource)
        .is_some_and(|c| c.host_written.load(std::sync::atomic::Ordering::Acquire))
}

/// Whether `resource`'s host copy takes its contents from title memory:
/// a texture the host GPU does not render to or write.
pub(crate) fn title_written(texture: &ID3D12Resource) -> bool {
    const TARGET_DEPTH_UNORDERED: i32 = 0x7;
    // SAFETY: a live resource.
    unsafe { texture.GetDesc() }.Flags.0 & TARGET_DEPTH_UNORDERED == 0
}

/// Copies [`alias_copy`] recorded since the last frame-rate line.
pub(crate) static ALIAS_COPIES: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

/// Records, once per list, the copy of `from`'s host texture into `to`'s:
/// two title textures at one memory that cannot share a host texture
/// (`resource::alias_source`: block-compressed against plain formats of
/// the same bytes, or a cast the host refused), `to` (a placed view's)
/// reading what the GPU wrote through `from`. Through a buffer, each
/// subresource with its own format, as many as have the same bytes per
/// row and rows in both; `to` no longer takes its contents from title
/// memory.
pub(crate) fn alias_copy(list: &HostList, from: This, to: This) {
    const COPY_DEST: u32 = 0x400;
    const COPY_SOURCE: u32 = 0x800;
    let (Some(a), Some(to_write), Some(host)) = (
        crate::resource::host_resource(from),
        prepare_host_write(to),
        host(),
    ) else {
        return;
    };
    let b = to_write.resource().clone();
    // An odd key: textures' own first uses are keyed by the host texture.
    if !list.first_use(to as usize | 1) {
        return;
    }
    // SAFETY: live resources.
    let (da, db) = unsafe { (a.GetDesc(), b.GetDesc()) };
    let count = |d: &D3D12_RESOURCE_DESC| {
        if d.Dimension == D3D12_RESOURCE_DIMENSION_TEXTURE3D {
            u32::from(d.MipLevels)
        } else {
            u32::from(d.MipLevels) * u32::from(d.DepthOrArraySize).max(1)
        }
    };
    let footprints = |d: &D3D12_RESOURCE_DESC, n: u32| {
        let mut f = vec![D3D12_PLACED_SUBRESOURCE_FOOTPRINT::default(); n as usize];
        let mut rows = vec![0u32; n as usize];
        let mut total = 0;
        // SAFETY: outputs sized for `n` subresources.
        unsafe {
            host.device
                .GetCopyableFootprints(d, 0, n, 0, Some(f.as_mut_ptr()), Some(rows.as_mut_ptr()), None, Some(&mut total));
        }
        (f, rows, total)
    };
    let (fa, ra, total) = footprints(&da, count(&da));
    let (fb, rb, _) = footprints(&db, count(&db));
    // Subresources in order while both have the same bytes there (mips of
    // a block-compressed texture past its last whole block do not).
    let n = (0..fa.len().min(fb.len()))
        .take_while(|&i| {
            let (x, y) = (fa[i], fb[i]);
            x.Offset == y.Offset && x.Footprint.RowPitch == y.Footprint.RowPitch && ra[i] == rb[i] && x.Footprint.Depth == y.Footprint.Depth
        })
        .count() as u32;
    if n == 0 {
        once!(?da, ?db, "an aliased texture not copied: its layouts differ on the host");
        return;
    }
    let Some(buffer) = host.default_buffer(total) else { return };
    list.keep(&buffer);
    to_write.record(list);
    crate::list::ensure_state(list, from, COPY_SOURCE);
    crate::list::ensure_state(list, to, COPY_DEST);
    let place = |r: &ID3D12Resource, f: D3D12_PLACED_SUBRESOURCE_FOOTPRINT| D3D12_TEXTURE_COPY_LOCATION {
        // SAFETY: a borrowed pointer the copy does not release.
        pResource: unsafe { std::mem::transmute_copy(r) },
        Type: D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT,
        Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 { PlacedFootprint: f },
    };
    let sub = |r: &ID3D12Resource, i: u32| D3D12_TEXTURE_COPY_LOCATION {
        // SAFETY: as above.
        pResource: unsafe { std::mem::transmute_copy(r) },
        Type: D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX,
        Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 { SubresourceIndex: i },
    };
    let l = &list.list;
    // SAFETY: an open list; live resources in the states named.
    unsafe {
        for i in 0..n {
            l.CopyTextureRegion(&place(&buffer, fa[i as usize]), 0, 0, 0, &sub(&a, i), None);
        }
        l.ResourceBarrier(&[crate::host::transition(
            &buffer,
            D3D12_RESOURCE_STATE_COPY_DEST,
            D3D12_RESOURCE_STATE_COPY_SOURCE,
        )]);
        for i in 0..n {
            l.CopyTextureRegion(&sub(&b, i), 0, 0, 0, &place(&buffer, fb[i as usize]), None);
        }
    }
    ALIAS_COPIES.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    static LOGGED: Mutex<Vec<(usize, usize)>> = Mutex::new(Vec::new());
    let mut logged = LOGGED.lock().unwrap_or_else(|e| e.into_inner());
    if logged.len() < 64 && !logged.contains(&(from as usize, to as usize)) {
        logged.push((from as usize, to as usize));
        tracing::info!(from = ?from, to = ?to, a = da.Format.0, b = db.Format.0, w = da.Width, h = da.Height, subresources = n, "aliased texture copied");
    }
}

/// Has `resource`'s host copy brought up to date at the submission that
/// runs `list`: its staging buffer detiled again if the memory changed, and
/// then copied into the host texture by the submission's own upload list,
/// once for all its lists (`crate::bindless::uploads`).
///
/// shortcut: a title copy into texture memory recorded in the same
/// submission shows from the next one on.
pub(crate) fn upload(list: &HostList, resource: This, texture: &ID3D12Resource) {
    // Copies and bound tables read the texture whole.
    if let Some(copy) = crate::resource::host_copy(resource) {
        note_reads(&copy, 0);
    }
    // shortcut: a texture the host copied into keeps the host's contents
    // even if the title later writes its memory too.
    // Once per list for each host texture, however many title textures of
    // its memory the list samples.
    if is_host_written(resource) {
        #[cfg(feature = "diagnostics")]
        {
            static SEEN: Mutex<Option<std::collections::HashSet<usize>>> = Mutex::new(None);
            if SEEN.lock().unwrap_or_else(|e| e.into_inner()).get_or_insert_default().insert(resource as usize) {
                let memory = crate::resource::texture_memory(resource).map(|(m, _)| m);
                // SAFETY: a live resource.
                let desc = unsafe { texture.GetDesc() };
                tracing::debug!(?resource, ?memory, ?desc, "a sampled texture keeps what the host copied into it: not uploaded from title memory");
            }
        }
        return;
    }
    if !list.first_use(texture.as_raw() as usize) {
        return;
    }
    let (Some(staging), Some(copy)) = (staging(resource, texture), crate::resource::host_copy(resource)) else {
        return;
    };
    // Each list recorded the copy before (run perf7: 149 textures uploaded
    // 3,195 times in a frame, 17 ms of GPU time), changed or not.
    list.defer_texture(staging, copy);
}

/// Records into `list` the copy of `buffer` (laid out as `staging` says)
/// into host texture `copy`. Returns the barrier that leaves it readable by
/// shaders, for the caller to record with the others after every copy: one
/// barrier per texture between the copies kept the GPU from overlapping
/// them (new shots copy hundreds of textures, run evict6).
#[must_use]
pub(crate) fn record_upload(
    list: &HostList,
    copy: &Arc<crate::resource::HostTexture>,
    staging: &Staging,
    (buffer, at, subresources): &Upload,
) -> Option<D3D12_RESOURCE_BARRIER> {
    const COPY_DEST: u32 = 0x400;
    const SHADER_RESOURCE: u32 = 0x40 | 0x80;
    if let Some(b) = list.move_texture(copy, COPY_DEST) {
        // SAFETY: an open list; the barrier names a live texture.
        unsafe { list.list.ResourceBarrier(&[b]) };
    }
    copies(list, &copy.texture, staging, (buffer, *at), subresources, String::new);
    list.keep(buffer);
    staging.uploads.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    staging.copied.fetch_add(subresources.len() as u64, std::sync::atomic::Ordering::Relaxed);
    list.move_texture(copy, SHADER_RESOURCE)
}

/// The uploads [`record_upload`] recorded into `resource`'s host texture
/// (tests: counts of their own texture, whatever other tests upload
/// meanwhile).
#[cfg(test)]
pub(crate) fn uploads(resource: This) -> u64 {
    let staging = crate::resource::host_copy(resource).and_then(|c| c.staging.lock().unwrap_or_else(|e| e.into_inner()).clone());
    staging.map_or(0, |s| s.uploads.load(std::sync::atomic::Ordering::Relaxed))
}

/// The subresources those uploads copied (tests, as [`uploads`]).
#[cfg(test)]
pub(crate) fn copied(resource: This) -> u64 {
    let staging = crate::resource::host_copy(resource).and_then(|c| c.staging.lock().unwrap_or_else(|e| e.into_inner()).clone());
    staging.map_or(0, |s| s.copied.load(std::sync::atomic::Ordering::Relaxed))
}


/// Records the copy of `subresources` of `buffer` (laid out as `staging`
/// says) into `texture` (in the copy destination state); `name` names it
/// in diagnostics notes.
#[cfg_attr(not(feature = "diagnostics"), allow(unused_variables))]
fn copies(
    list: &HostList,
    texture: &ID3D12Resource,
    staging: &Staging,
    (buffer, at): (&ID3D12Resource, u64),
    subresources: &[u32],
    name: impl Fn() -> String,
) {
    for &i in subresources {
        let i = i as usize;
        let mut footprint = staging.footprints[i];
        footprint.Offset += at;
        let footprint = &footprint;
        #[cfg(feature = "diagnostics")]
        {
            let f = footprint.Footprint;
            // SAFETY: live resources.
            let (gpu, d) = unsafe { (buffer.GetGPUVirtualAddress(), texture.GetDesc()) };
            let len = u64::from(f.RowPitch) * u64::from(staging.rows[i]) * u64::from(f.Depth);
            list.note_read(
                || {
                    format!(
                        "texture upload {} ({}x{}x{} f{} mips {}) sub {i} of {} from staging {gpu:#x}+{:#x}: offset {:#x} {}x{}x{} pitch {}",
                        name(),
                        d.Width,
                        d.Height,
                        d.DepthOrArraySize,
                        d.Format.0,
                        d.MipLevels,
                        staging.footprints.len(),
                        staging.size,
                        footprint.Offset,
                        f.Width,
                        f.Height,
                        f.Depth,
                        f.RowPitch
                    )
                },
                (gpu + footprint.Offset, len),
            );
        }
        let to = D3D12_TEXTURE_COPY_LOCATION {
            // SAFETY: a borrowed pointer the location never releases.
            pResource: unsafe { std::mem::transmute_copy(texture) },
            Type: D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX,
            Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 {
                SubresourceIndex: i as u32,
            },
        };
        let from = D3D12_TEXTURE_COPY_LOCATION {
            // SAFETY: as above.
            pResource: unsafe { std::mem::transmute_copy(buffer) },
            Type: D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT,
            Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 {
                PlacedFootprint: *footprint,
            },
        };
        // SAFETY: an open list; both locations name live resources.
        unsafe { list.list.CopyTextureRegion(&to, 0, 0, 0, &from, None) };
    }
}

/// What a refresh gives: a staging buffer, where its space starts in it,
/// and the subresources detiled into it.
pub(crate) type Upload = (ID3D12Resource, u64, Vec<u32>);

/// Runs `f(i)` for each `i` below `n` on up to a quarter of the CPUs (four
/// at most: the title's own threads need the rest), here when that is
/// fewer than two.
pub(crate) fn in_parallel(n: usize, f: impl Fn(usize) + Sync) {
    // On the host's persistent workers (`crate::workers`, shared with the
    // refresh copies): a thread made per call lost its warm [`Scratch`]
    // when it ended (#2786: 1.6 ms a batch, 0.06 on kept threads).
    let threads = std::thread::available_parallelism().map_or(4, |n| n.get() / 4).clamp(1, 4).min(n);
    crate::workers::run(threads, n, f);
}

/// Sorts `spans` and merges those that overlap or touch, dropping empty
/// ones.
fn merge_spans(spans: &mut Vec<(usize, usize)>) {
    spans.retain(|&(a, b)| a < b);
    if spans.len() <= 1 {
        return;
    }
    spans.sort_unstable();
    let mut write = 0;
    for read in 1..spans.len() {
        let (a, b) = spans[read];
        if a <= spans[write].1 {
            spans[write].1 = spans[write].1.max(b);
        } else {
            write += 1;
            spans[write] = (a, b);
        }
    }
    spans.truncate(write + 1);
}

/// [`Staging::refresh`] of each of `work`, those whose memory may have
/// changed on several threads at once (reading title memory and detiling
/// cost ~6 ms of frame a MB on one thread in a new shot, run evict5).
pub(crate) fn refresh_all(work: &[(Arc<Staging>, Arc<crate::resource::HostTexture>)]) -> Vec<Option<Upload>> {
    // Most find nothing changed (their trackers' clean bits say so): those
    // are not handed to threads.
    let changed: Vec<usize> = (0..work.len()).filter(|&i| work[i].0.maybe_changed()).collect();
    let results: Vec<Mutex<Option<Upload>>> = changed.iter().map(|_| Mutex::new(None)).collect();
    in_parallel(changed.len(), |k| {
        *results[k].lock().unwrap_or_else(|e| e.into_inner()) = work[changed[k]].0.refresh();
    });
    let mut out: Vec<Option<Upload>> = vec![None; work.len()];
    for (k, r) in results.into_iter().enumerate() {
        out[changed[k]] = r.into_inner().unwrap_or_else(|e| e.into_inner());
    }
    out
}

impl Staging {
    /// Whether its memory may have changed since it was last detiled (an
    /// untracked texture always may).
    pub(crate) fn maybe_changed(&self) -> bool {
        self.tracked.as_ref().is_none_or(|t| !t.clean())
    }

    /// A staging buffer with the texture memory detiled into it if it
    /// changed since the last time (and is mapped at all), and the
    /// subresources that changed (only those are detiled and to be copied).
    pub(crate) fn refresh(&self) -> Option<Upload> {
        use std::sync::atomic::Ordering::Relaxed;
        let started = std::time::Instant::now();
        let (read, buffer) = self.refresh_now();
        let preparing = &crate::host::PREPARING;
        preparing[2].fetch_add(started.elapsed().as_micros() as u64, Relaxed);
        preparing[3].fetch_add(read as u64, Relaxed);
        buffer
    }

    /// Returns the bytes of title memory read, and the buffer it detiled
    /// into with the subresources detiled.
    fn refresh_now(&self) -> (usize, Option<Upload>) {
        // The levels read only (`first` on): past their bytes may lie
        // another texture. The rest of the layout is neither read nor
        // detiled.
        let read_from = |i: u32| i % self.mips >= self.first;
        let all = || {
            (0..self.footprints.len() as u32)
                .filter(|&i| read_from(i))
                .collect::<Vec<_>>()
        };
        let size = self.layout.size as usize;
        let prefix = self.prefix;
        let start = self.memory as usize;
        let mut last = self.last.lock().unwrap_or_else(|e| e.into_inner());
        let last = &mut *last;
        if let Some(t) = &self.tracked {
            // Most refreshes find nothing: the tracker's dense bit says so
            // without its lock (123,000 refreshes a frame, run evict3).
            if t.clean() {
                return (0, None);
            }
            let ranges = t.take(start, start + prefix);
            if ranges.is_empty() {
                return (0, None);
            }
            // Only the subresources in the pages that changed (streamed
            // textures change page by page; a whole re-read and detile of
            // each cost up to 430 MB a frame, run integ2).
            let n = self.footprints.len();
            let span = |i: usize| {
                self.spans
                    .get(i)
                    .map_or((0, prefix), |&(a, b)| (a.min(prefix), b.min(prefix)))
            };
            let touched: Vec<usize> = (0..n)
                .filter(|&i| read_from(i as u32))
                .filter(|&i| {
                    let (a, b) = span(i);
                    ranges.iter().any(|&(f, e)| f - start < b && a < e - start)
                })
                .collect();
            // Read through the copy, which stands unmapped pages: memory of
            // big pools is too slow to ask page by page. Not kept: it would
            // double each texture's system memory. Only the touched spans
            // are written, and only they are hashed and detiled.
            let mut scratch = Scratch::take(size);
            let read = &mut scratch.0[..size];
            let mut bytes = 0;
            // Overlapping and adjacent spans merged: each byte is read once.
            let mut spans: Vec<(usize, usize)> = touched.iter().map(|&i| span(i)).collect();
            merge_spans(&mut spans);
            for (a, b) in spans {
                // SAFETY: inside the buffer of the texture's size.
                crate::host::copy_mapped(unsafe { read.as_mut_ptr().add(a) }, self.memory.wrapping_add(a), b - a, false);
                bytes += b - a;
            }
            // A changed page that holds only the subresource's bytes says it
            // changed. Pages it shares with other data the title writes
            // (small textures): its own bytes decide. (Hashing a movie frame
            // cost 8 ms a frame.)
            let page = crate::watch::PAGE;
            last.hashes.resize(n, None);
            let fresh: Vec<u32> = touched
                .into_iter()
                .filter(|&i| {
                    let (a, b) = span(i);
                    let (a0, b0) = (start + a, start + b);
                    if ranges.iter().any(|&(f, e)| f.max(a0).next_multiple_of(page) + page <= e.min(b0)) {
                        last.hashes[i] = None;
                        return true;
                    }
                    let hash = blake3::hash(&read[a..b]);
                    last.hashes[i].replace(hash) != Some(hash)
                })
                .map(|i| i as u32)
                .collect();
            if fresh.is_empty() {
                return (bytes, None);
            }
            let buffer = self.detile(read, &fresh);
            if buffer.is_none() {
                for &i in &fresh {
                    last.hashes[i as usize] = None;
                }
            }
            #[cfg(feature = "diagnostics")]
            crate::copystat::add(crate::copystat::C::TexWatched, bytes as u64);
            return (bytes, buffer.map(|(b, at)| (b, at, fresh)));
        }
        if !committed(self.memory, prefix) {
            #[cfg(feature = "diagnostics")]
            tracing::debug!(
                memory = ?self.memory,
                size,
                desc = ?self.layout.desc,
                "a sampled texture's memory is not committed"
            );
            // Never uploaded: zeros, as a committed texture starts (one in
            // a heap of textures starts with whatever the range held).
            if last.hash.is_some() {
                return (0, None);
            }
            let zeros = vec![0u8; size];
            let every = all();
            let buffer = self.detile(&zeros, &every);
            last.hash = buffer.is_some().then(|| blake3::hash(&zeros));
            return (0, buffer.map(|(b, at)| (b, at, every)));
        }
        // SAFETY: committed memory of the levels' bytes.
        let tiled = unsafe { std::slice::from_raw_parts(self.memory, prefix) };
        let hash = blake3::hash(tiled);
        #[cfg(feature = "diagnostics")]
        crate::copystat::add(crate::copystat::C::TexHashed, prefix as u64);
        if last.hash == Some(hash) {
            return (prefix, None);
        }
        let every = all();
        let buffer = if prefix < size {
            // Detiled from a layout-sized copy; the levels not read are not.
            let mut scratch = Scratch::take(size);
            scratch.0[..prefix].copy_from_slice(tiled);
            self.detile(&scratch.0[..size], &every)
        } else {
            self.detile(tiled, &every)
        };
        last.hash = buffer.is_some().then_some(hash);
        #[cfg(feature = "diagnostics")]
        crate::copystat::add(crate::copystat::C::TexChanged, prefix as u64);
        (prefix, buffer.map(|(b, at)| (b, at, every)))
    }

    /// Detiles `subresources` of `tiled` (the texture's memory) into a
    /// staging buffer (`None` if none could be had or a subresource could
    /// not be read).
    fn detile(&self, tiled: &[u8], subresources: &[u32]) -> Option<(ID3D12Resource, u64)> {
        let (buffer, at, data) = staging_space(self.size)?;
        #[cfg(feature = "diagnostics")]
        // Scanned only when its message is wanted: a pass over every
        // texture read, in timing runs too.
        if tracing::enabled!(tracing::Level::DEBUG) {
            if tiled[..self.prefix.min(tiled.len())].iter().all(|&b| b == 0) {
                tracing::debug!(
                    memory = ?self.memory,
                    desc = ?self.layout.desc,
                    "a sampled texture's memory is all zero (not streamed yet, or a failed BCPack read)"
                );
            } else {
                tracing::debug!(memory = ?self.memory, desc = ?self.layout.desc, "a sampled texture detiled");
            }
        }
        for &i in subresources {
            let (footprint, rows) = (&self.footprints[i as usize], self.rows[i as usize]);
            let (mip, slice) = (i % self.mips, i / self.mips);
            let f = footprint.Footprint;
            let pitch = f.RowPitch as usize;
            let len = subresource_len(footprint.Offset, pitch, rows, f.Depth, self.size);
            // SAFETY: `subresource_len` clamps `len` to `self.size - footprint.Offset`,
            // keeping the slice strictly inside the mapped staging buffer even when
            // GetCopyableFootprints does not pad the final row of the last subresource.
            let linear = unsafe { std::slice::from_raw_parts_mut(data.add(footprint.Offset as usize), len) };
            if let Err(e) =
                self.layout
                    .detile(tiled, mip, slice, linear, pitch, pitch * rows as usize)
            {
                tracing::warn!(?e, mip, slice, "texture not detiled");
                return None;
            }
        }
        Some((buffer, at))
    }
}

/// A buffer for the title memory a refresh reads, kept by the thread for
/// its next refresh: a fresh one for each texture cost a zero-fill and a
/// page fault per 4 KiB (129,000 faults in a frame, run placed1).
struct Scratch(Vec<u8>);

thread_local! {
    static SCRATCH: std::cell::Cell<Vec<u8>> = const { std::cell::Cell::new(Vec::new()) };
}

impl Scratch {
    /// The thread's buffer, at least `size` bytes.
    fn take(size: usize) -> Self {
        let mut v = SCRATCH.take();
        if v.len() < size {
            v.resize(size, 0);
        }
        Scratch(v)
    }
}

impl Drop for Scratch {
    fn drop(&mut self) {
        // The largest textures' buffers are not kept.
        const KEEP: usize = 32 << 20;
        if self.0.len() <= KEEP {
            SCRATCH.set(std::mem::take(&mut self.0));
        }
    }
}

/// Bytes available in the staging buffer for a subresource at `offset`.
/// D3D12's `GetCopyableFootprints` omits trailing row-pitch padding on the
/// final row of the final subresource, so `pitch * rows * depth` can exceed
/// `total - offset`.
fn subresource_len(offset: u64, pitch: usize, rows: u32, depth: u32, total: u64) -> usize {
    let padded = pitch
        .saturating_mul(rows as usize)
        .saturating_mul(depth as usize);
    let remaining = usize::try_from(total.saturating_sub(offset)).unwrap_or(usize::MAX);
    padded.min(remaining)
}

/// Whether `[at, at + len)` is committed, readable memory (streamed textures
/// may be unmapped until their pages arrive).
pub(crate) fn committed(at: *const u8, len: usize) -> bool {
    use windows_sys::Win32::System::Memory::{
        MEM_COMMIT, MEMORY_BASIC_INFORMATION, PAGE_GUARD, PAGE_NOACCESS, VirtualQuery,
    };
    let mut p = at as usize;
    let Some(end) = p.checked_add(len) else {
        return false;
    };
    while p < end {
        // SAFETY: plain data filled in by the query.
        let mut info: MEMORY_BASIC_INFORMATION = unsafe { std::mem::zeroed() };
        // SAFETY: a valid out pointer of the right size.
        let n = unsafe { VirtualQuery(p as *const _, &mut info, size_of_val(&info)) };
        let unreadable = info.Protect == 0
            || info.Protect & (PAGE_NOACCESS | PAGE_GUARD) != 0;
        if n == 0 || info.State != MEM_COMMIT || unreadable {
            return false;
        }
        let next = (info.BaseAddress as usize).saturating_add(info.RegionSize);
        if next <= p {
            return false;
        }
        p = next;
    }
    true
}

/// The first mip level a shader view (its words as the title writes them)
/// can read: its most detailed level, or a later one where its minimum
/// level-of-detail clamp is past it. 0 for a view of the resource as a
/// whole and for buffers.
pub(crate) fn first_level(view: Option<&[u32; 10]>) -> u32 {
    let Some(w) = view else {
        return 0;
    };
    // `D3D12_SHADER_RESOURCE_VIEW_DESC`: format, dimension, mapping, a pad,
    // then the dimension's fields, the clamp last.
    let clamp = match w[1] {
        2 | 8 | 9 => w[6], // 1D, 3D, cube
        4 => w[7],         // 2D
        3 | 10 => w[8],    // 1D array, cube array
        5 => w[9],         // 2D array
        _ => return 0,
    };
    let clamp = f32::from_bits(clamp);
    let clamp = if clamp.is_finite() && clamp > 0.0 {
        clamp as u32
    } else {
        0
    };
    w[4].max(clamp)
}

/// Something reads `copy` from level `first` on (a shader view, or 0 for a
/// copy or a bound table): a staging made from a later level is made again.
pub(crate) fn note_reads(copy: &crate::resource::HostTexture, first: u32) {
    use std::sync::atomic::Ordering::Relaxed;
    if copy.read_from.load(Relaxed) <= first || copy.read_from.fetch_min(first, Relaxed) <= first {
        return;
    }
    let mut held = copy.staging.lock().unwrap_or_else(|e| e.into_inner());
    if held.as_ref().is_some_and(|s| s.first > first) {
        *held = None;
    }
}

/// The first level to upload of a texture of `mips` levels read from
/// `read_from` on: only single 2D textures start past level 0 (an array's
/// slices each hold the whole chain, so its unread levels are not one
/// range).
fn first_uploaded(layout: &victory_formats::texture::Layout, read_from: u32, mips: u32) -> u32 {
    use victory_formats::texture::Dimension;
    if read_from == u32::MAX || layout.slices != 1 || layout.desc.dimension != Dimension::Tex2d {
        return 0;
    }
    read_from.min(mips.saturating_sub(1))
}

/// The bytes at the start of a single texture's memory that hold its levels
/// from `first` on: the levels are stored smallest first (the mip tail,
/// then each larger level), so they end where level `first - 1` begins,
/// or where the tail ends when `first` lies in the tail.
fn readable_bytes(layout: &victory_formats::texture::Layout, first: u32) -> u64 {
    if first == 0 || layout.slices != 1 {
        return layout.size;
    }
    let k = first.min(layout.first_tail_mip) as usize;
    match k.checked_sub(1).and_then(|k| layout.mips.get(k)) {
        Some(next) if next.offset > 0 => next.offset,
        _ => layout.size,
    }
}

/// The staging state of a title texture's host texture, made on first use
/// (one for all title textures of its memory).
pub(crate) fn staging(resource: This, texture: &ID3D12Resource) -> Option<Arc<Staging>> {
    let copy = crate::resource::host_copy(resource)?;
    let mut held = copy.staging.lock().unwrap_or_else(|e| e.into_inner());
    if let Some(s) = held.as_ref() {
        return Some(s.clone());
    }
    let (memory, desc) = crate::resource::texture_memory(resource)?;
    let Some(layout) = crate::resource::texture_layout(&desc).map(|t| t.planes[0].layout.clone())
    else {
        once!(
            ?desc,
            "texture contents without a console layout not uploaded"
        );
        return None;
    };
    if layout.desc.swizzle.xor() {
        once!(?desc, "texture contents in an XOR mode not uploaded");
        return None;
    }
    let host = host()?;
    // SAFETY: a live resource.
    let d = unsafe { texture.GetDesc() };
    let count = if d.Dimension == D3D12_RESOURCE_DIMENSION_TEXTURE3D {
        u32::from(d.MipLevels)
    } else {
        u32::from(d.MipLevels) * u32::from(d.DepthOrArraySize)
    };
    let mut footprints = vec![D3D12_PLACED_SUBRESOURCE_FOOTPRINT::default(); count as usize];
    let mut rows = vec![0u32; count as usize];
    let mut total = 0;
    // SAFETY: outputs for `count` subresources.
    unsafe {
        host.device.GetCopyableFootprints(
            &d,
            0,
            count,
            0,
            Some(footprints.as_mut_ptr()),
            Some(rows.as_mut_ptr()),
            None,
            Some(&mut total),
        );
    }
    tracing::debug!(?desc, total, "texture contents uploaded from title memory");
    let mips = u32::from(d.MipLevels);
    let read_from = copy.read_from.load(std::sync::atomic::Ordering::Relaxed);
    let first = first_uploaded(&layout, read_from, mips);
    // Watched by 64 KiB pages: a write in the last page the levels read
    // share with what follows them still uploads them again (only them).
    let prefix = readable_bytes(&layout, first) as usize;
    if first > 0 {
        once!(
            ?desc,
            first,
            read = prefix,
            whole = layout.size,
            "textures only viewed from a smaller level are uploaded from it on"
        );
    }
    #[cfg(feature = "diagnostics")]
    tracing::debug!(
        memory = ?memory,
        mips,
        first,
        read = prefix,
        whole = layout.size,
        neighbours = crate::resource::placed_textures_in(
            memory as u64 + prefix as u64,
            memory as u64 + layout.size
        ),
        "texture staged"
    );
    let spans = spans(&layout, count, mips);
    let s = Arc::new(Staging {
        size: total,
        footprints,
        rows,
        layout,
        mips,
        first,
        prefix,
        memory,
        tracked: crate::watch::track(memory as usize, prefix, true),
        spans,
        last: Mutex::default(),
        uploads: Default::default(),
        copied: Default::default(),
    });
    *held = Some(s.clone());
    Some(s)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn merge_spans_coalesces_overlapping_and_adjacent_intervals() {
        let mut s = vec![(100, 200), (50, 50), (0, 64), (64, 100), (300, 400), (350, 380), (380, 450), (0, 64)];
        merge_spans(&mut s);
        assert_eq!(s, vec![(0, 200), (300, 450)]);
        let mut empty = vec![(10, 10), (20, 10)];
        merge_spans(&mut empty);
        assert!(empty.is_empty());
    }

    fn layout(w: u32, h: u32, mips: u32) -> victory_formats::texture::Layout {
        use victory_formats::texture::{Desc, Dimension, Layout, Swizzle};
        Layout::new(&Desc {
            dimension: Dimension::Tex2d,
            width: w,
            height: h,
            depth: 1,
            mips,
            element_bytes: 16,
            block: 4,
            samples: 1,
            swizzle: Swizzle::SW_64KB_S_X,
        })
        .unwrap()
    }

    /// A texture's levels are stored smallest first: the levels from any
    /// level on take the start of its memory, smaller as the level rises,
    /// and level 0 on is the whole of it.
    #[test]
    fn the_readable_levels_are_a_prefix_of_the_memory() {
        let l = layout(2048, 2048, 12);
        assert!(
            l.first_tail_mip > 0 && l.first_tail_mip < 12,
            "{}",
            l.first_tail_mip
        );
        assert_eq!(readable_bytes(&l, 0), l.size);
        let prefixes: Vec<u64> = (0..12).map(|first| readable_bytes(&l, first)).collect();
        for pair in prefixes.windows(2) {
            assert!(
                pair[1] <= pair[0],
                "smaller from a later level: {prefixes:?}"
            );
        }
        // Each level on lies inside its prefix, and none before it does.
        for first in 1..l.first_tail_mip {
            let end = readable_bytes(&l, first);
            let m = &l.mips[first as usize];
            assert!(m.offset < end, "level {first} in its prefix");
            assert!(
                l.mips[first as usize - 1].offset >= end,
                "level {} outside",
                first - 1
            );
        }
        // From a level in the tail on: the tail block alone, the same for all.
        let tail = readable_bytes(&l, l.first_tail_mip);
        assert_eq!(readable_bytes(&l, 11), tail);
        assert!(tail < l.size / 64, "a tail of {tail} bytes");
        // Level 1 on is about a quarter of the memory.
        let one = readable_bytes(&l, 1);
        assert!(one * 3 < l.size && one * 5 > l.size, "{one} of {}", l.size);
    }

    /// Views say their first level by their most detailed level or their
    /// level-of-detail clamp, per dimension; arrays and unknown views keep
    /// every level.
    #[test]
    fn views_name_their_first_level() {
        let view = |dimension: u32, most_detailed: u32, clamp_at: usize, clamp: f32| {
            let mut w = [0u32; 10];
            (w[1], w[4]) = (dimension, most_detailed);
            w[clamp_at] = clamp.to_bits();
            w
        };
        assert_eq!(first_level(None), 0);
        assert_eq!(
            first_level(Some(&view(4, 3, 7, 0.0))),
            3,
            "2D, most detailed level"
        );
        assert_eq!(
            first_level(Some(&view(4, 0, 7, 2.5))),
            2,
            "2D, clamp 2.5 reads level 2"
        );
        assert_eq!(
            first_level(Some(&view(4, 4, 7, 2.0))),
            4,
            "the later of the two"
        );
        assert_eq!(first_level(Some(&view(9, 0, 6, 5.0))), 5, "cube");
        assert_eq!(first_level(Some(&view(5, 0, 9, 1.0))), 1, "2D array");
        assert_eq!(first_level(Some(&view(4, 0, 7, f32::NAN))), 0);
        assert_eq!(first_level(Some(&view(1, 9, 7, 0.0))), 0, "a buffer");
        // Arrays are uploaded whole whatever their views read.
        let mut array = layout(256, 256, 9);
        assert_eq!(first_uploaded(&array, 3, 9), 3);
        array.slices = 2;
        assert_eq!(first_uploaded(&array, 3, 9), 0);
        assert_eq!(
            first_uploaded(&layout(256, 256, 9), u32::MAX, 9),
            0,
            "no view yet"
        );
        assert_eq!(
            first_uploaded(&layout(256, 256, 9), 20, 9),
            8,
            "at most the last level"
        );
    }

    /// A texture placed in title memory and viewed only from level 3 on
    /// (the title keeps a streamed texture's small levels and places the
    /// next texture where its larger ones would be): it is uploaded and
    /// watched from level 3 on only, so a write past those levels (the next
    /// texture's, past the 64 KiB watch page they share) does not upload it
    /// again, a write inside them does, and a view of every level later
    /// makes it whole again.
    #[test]
    fn a_texture_viewed_from_a_later_level_is_staged_from_it_only() {
        let p = crate::watch::tests::pool(8);
        let words = crate::resource::texture_words(256, 256, 28, 9);
        let Some(r) = crate::resource::texture_at(p as u64, &words) else {
            crate::watch::tests::free(p);
            return; // no host GPU here
        };
        let copy = crate::resource::host_copy(r).expect("a host copy");
        note_reads(&copy, 3);
        let s = staging(r, &copy.texture).expect("its staging");
        assert_eq!(s.first, 3);
        assert_eq!(s.prefix as u64, readable_bytes(&s.layout, 3));
        assert!(
            s.prefix * 16 < s.layout.size as usize,
            "{} of {}",
            s.prefix,
            s.layout.size
        );
        let (_, _, first) = s.refresh().expect("uploaded the first time");
        assert!(first.iter().all(|&i| i % s.mips >= 3), "levels 3 on only: {first:?}");
        assert!(s.refresh().is_none(), "nothing written since");
        let title = |at: usize| (p + at) as *mut u8;
        let past = s.prefix.next_multiple_of(crate::watch::PAGE) + 16;
        assert!(past < s.layout.size as usize);
        // SAFETY: a byte of our pool, inside the layout.
        unsafe { title(past).write_volatile(0x5a) };
        assert!(
            s.refresh().is_none(),
            "a write past the levels read is not this texture's"
        );
        // SAFETY: as above.
        unsafe { title(8).write_volatile(0xa5) };
        assert!(s.refresh().is_some(), "a write inside them is");
        // A view of every level: made again, whole.
        note_reads(&copy, 0);
        assert!(
            copy.staging.lock().unwrap().is_none(),
            "the staging from level 3 is let go"
        );
        let whole = staging(r, &copy.texture).expect("its staging again");
        assert_eq!((whole.first, whole.prefix), (0, whole.layout.size as usize));
        crate::watch::tests::free(p);
    }

    #[test]
    fn subresource_len_stays_inside_staging_total() {
        // A 4x4 RGBA8 texture has RowPitch=256, rows=4, last_row=16 -> total=784.
        // Padded size (1024) must be clamped to 784 so from_raw_parts_mut stays in bounds.
        assert_eq!(subresource_len(0, 256, 4, 1, 784), 784);
        assert_eq!(subresource_len(512, 256, 4, 1, 784), 272);
        assert_eq!(subresource_len(0, 256, 4, 1, 2048), 1024);
    }

    #[test]
    fn committed_rejects_overflow_and_noaccess_pages() {
        use windows_sys::Win32::System::Memory::{
            MEM_COMMIT, MEM_RELEASE, MEM_RESERVE, PAGE_NOACCESS, PAGE_READWRITE, VirtualAlloc,
            VirtualFree,
        };
        assert!(!committed(usize::MAX as *const u8, 16));
        // SAFETY: reserving and committing one private page for the test, freed below.
        let noaccess =
            unsafe { VirtualAlloc(std::ptr::null(), 4096, MEM_RESERVE | MEM_COMMIT, PAGE_NOACCESS) };
        assert!(!noaccess.is_null());
        assert!(!committed(noaccess.cast::<u8>(), 4096));
        // SAFETY: freeing the allocation above.
        unsafe { VirtualFree(noaccess, 0, MEM_RELEASE) };

        // Negative control: a readable committed page still passes.
        // SAFETY: reserving and committing one private page for the test, freed below.
        let rw =
            unsafe { VirtualAlloc(std::ptr::null(), 4096, MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE) };
        assert!(!rw.is_null());
        assert!(committed(rw.cast::<u8>(), 4096));
        // SAFETY: freeing the allocation above.
        unsafe { VirtualFree(rw, 0, MEM_RELEASE) };
    }

    #[cfg(test)]
    struct HeldCom(This);
    #[cfg(test)]
    impl Drop for HeldCom {
        fn drop(&mut self) {
            crate::object::slot::<extern "system" fn(This) -> u32>(self.0, 0x10)(self.0);
        }
    }

    #[cfg(test)]
    fn test_title_texture(addr: u64, fill: u8) -> HeldCom {
        const SIDE: usize = 64;
        const BYTES: usize = SIDE * SIDE * 4;
        // SAFETY: `pool(1)` provides a 64 KiB RW page for each 64x64 RGBA8 linear texture.
        unsafe { std::ptr::write_bytes(addr as *mut u8, fill, BYTES) };
        let words: [u32; 14] = [3, 0, 0, 0, SIDE as u32, 0, SIDE as u32, 1 | (1 << 16), 28, 1, 0, 0x100, 0, 0];
        let iid = crate::object::guid(0x696442be, 0xa72e, 0x4059, [0xbc, 0x79, 0x5b, 0x5c, 0x98, 0x04, 0x0f, 0xad]);
        let mut out = std::ptr::null();
        let hr = crate::resource::create_at_address(std::ptr::null(), addr, words.as_ptr().cast(), 0, std::ptr::null(), &iid, &mut out);
        assert_eq!(hr, crate::object::S_OK);
        HeldCom(out)
    }

    #[cfg(test)]
    fn seed_title_texture(host: &crate::host::Host, tex: This) {
        let seed_list = host.new_list().unwrap();
        let host_tex = crate::resource::host_resource(tex).unwrap();
        upload(&seed_list, tex, &host_tex);
        seed_list.close();
        host.execute(&[&seed_list]);
    }

    #[cfg(test)]
    fn readback_first_texel(host: &crate::host::Host, tex: This) -> [u8; 4] {
        let read_list = host.new_list().unwrap();
        let tex_after = crate::resource::host_resource(tex).unwrap();
        upload(&read_list, tex, &tex_after);
        crate::list::ensure_state(&read_list, tex, 0x800);
        let mut footprint = D3D12_PLACED_SUBRESOURCE_FOOTPRINT::default();
        let mut total = 0;
        // SAFETY: live host texture; outputs for subresource 0.
        unsafe {
            host.device.GetCopyableFootprints(&tex_after.GetDesc(), 0, 1, 0, Some(&mut footprint), None, None, Some(&mut total));
        }
        let mut readback: Option<ID3D12Resource> = None;
        // SAFETY: plain readback buffer creation and copy recording.
        unsafe {
            host.device
                .CreateCommittedResource(
                    &D3D12_HEAP_PROPERTIES { Type: D3D12_HEAP_TYPE_READBACK, ..Default::default() },
                    D3D12_HEAP_FLAG_NONE,
                    &D3D12_RESOURCE_DESC {
                        Dimension: D3D12_RESOURCE_DIMENSION_BUFFER,
                        Width: total,
                        Height: 1,
                        DepthOrArraySize: 1,
                        MipLevels: 1,
                        SampleDesc: windows::Win32::Graphics::Dxgi::Common::DXGI_SAMPLE_DESC { Count: 1, Quality: 0 },
                        Layout: D3D12_TEXTURE_LAYOUT_ROW_MAJOR,
                        ..Default::default()
                    },
                    D3D12_RESOURCE_STATE_COPY_DEST,
                    None,
                    &mut readback,
                )
                .unwrap();
        }
        let readback = readback.unwrap();
        let dst_loc = D3D12_TEXTURE_COPY_LOCATION {
            // SAFETY: borrowed pointer for recording only.
            pResource: unsafe { std::mem::transmute_copy(&readback) },
            Type: D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT,
            Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 { PlacedFootprint: footprint },
        };
        let src_loc = D3D12_TEXTURE_COPY_LOCATION {
            // SAFETY: borrowed pointer for recording only.
            pResource: unsafe { std::mem::transmute_copy(&tex_after) },
            Type: D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX,
            Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 { SubresourceIndex: 0 },
        };
        // SAFETY: open host command list and live resources in copy states.
        unsafe {
            read_list.list.CopyTextureRegion(&dst_loc, 0, 0, 0, &src_loc, None);
        }
        read_list.close();
        host.execute(&[&read_list]);

        let mut mapped = std::ptr::null_mut();
        // SAFETY: synchronous execution finished; `readback` covers all rows.
        unsafe {
            readback.Map(0, None, Some(&mut mapped)).unwrap();
            let first = mapped.cast::<u8>().add(footprint.Offset as usize).cast::<[u8; 4]>().read();
            readback.Unmap(0, None);
            first
        }
    }

    /// A texture fed from title memory cannot be evicted while a recorded list
    /// holds a pending `host_writes` mark for it; resetting/dropping an
    /// unexecuted list releases that pin, while executing the list keeps the
    /// copied GPU texels over stale title memory.
    #[test]
    fn pending_host_write_pins_texture_until_executed_and_unpins_on_reset() {
        let Some(host) = host() else {
            return; // no host GPU here
        };
        let dst_mem = crate::watch::tests::pool(1);
        let src_mem = crate::watch::tests::pool(1);
        let dst = test_title_texture(dst_mem as u64, 0x40);
        let src = test_title_texture(src_mem as u64, 0xAA);

        let null = std::ptr::null();
        let (mut alloc, mut list_drop, mut list_exec) = (null, null, null);
        let alloc_iid = crate::object::guid(0x6102dee4, 0xaf59, 0x4b09, [0xb9, 0x99, 0xb4, 0x4d, 0x73, 0xf0, 0x9b, 0x24]);
        let list_iid = crate::object::guid(0x5b160d0f, 0xac1b, 0x4185, [0x8b, 0xa8, 0xb3, 0xae, 0x42, 0xa5, 0xa4, 0x55]);
        assert_eq!(crate::list::create_allocator(null, 0, &alloc_iid, &mut alloc), crate::object::S_OK);
        let _alloc = HeldCom(alloc);
        assert_eq!(crate::list::create_list(null, 0, 0, alloc, null, &list_iid, &mut list_drop), crate::object::S_OK);
        let _l_drop = HeldCom(list_drop);
        assert_eq!(crate::list::create_list(null, 0, 0, alloc, null, &list_iid, &mut list_exec), crate::object::S_OK);
        let _l_exec = HeldCom(list_exec);

        type CloseFn = extern "system" fn(This) -> crate::object::HRESULT;
        type ResetFn = extern "system" fn(This, This, This) -> crate::object::HRESULT;
        type CopyResourceFn = extern "system" fn(This, This, This);
        let close_drop = crate::object::slot::<CloseFn>(list_drop, 0x48);
        let reset_drop = crate::object::slot::<ResetFn>(list_drop, 0x50);
        let copy_drop = crate::object::slot::<CopyResourceFn>(list_drop, 0x88);
        let close_exec = crate::object::slot::<CloseFn>(list_exec, 0x48);
        let reset_exec = crate::object::slot::<ResetFn>(list_exec, 0x50);
        let copy_exec = crate::object::slot::<CopyResourceFn>(list_exec, 0x88);

        seed_title_texture(host, dst.0);

        copy_drop(list_drop, dst.0, src.0);
        assert!(!crate::resource::give_up_now(dst.0), "pending CopyResource must pin dst against eviction");
        assert_eq!(close_drop(list_drop), crate::object::S_OK);
        assert_eq!(reset_drop(list_drop, alloc, null), crate::object::S_OK);
        assert!(crate::resource::give_up_now(dst.0), "resetting unexecuted list must allow eviction again");

        seed_title_texture(host, dst.0);

        copy_exec(list_exec, dst.0, src.0);
        assert_eq!(close_exec(list_exec), crate::object::S_OK);
        assert!(!crate::resource::give_up_now(dst.0), "pending L1 CopyResource must block eviction before L1 runs");

        let recorded = crate::list::host_list(list_exec).unwrap();
        host.execute(&[recorded]);
        assert_eq!(reset_exec(list_exec, alloc, null), crate::object::S_OK);
        assert!(!crate::resource::give_up_now(dst.0), "executed host write must keep dst un-evictable after list reset");

        assert_eq!(readback_first_texel(host, dst.0), [0xAA, 0xAA, 0xAA, 0xAA]);
    }

    /// Window 2: an eviction candidate snapshot taken before a list pins `dst`
    /// must not evict `dst` in `give_up`, and the recorded copy must keep its
    /// host texels on subsequent reads.
    #[test]
    fn stale_eviction_snapshot_skips_texture_pinned_after_candidates() {
        use std::sync::atomic::Ordering::Acquire;
        let Some(host) = host() else {
            return; // no host GPU here
        };
        let dst_mem = crate::watch::tests::pool(1);
        let src_mem = crate::watch::tests::pool(1);
        let dst = test_title_texture(dst_mem as u64, 0x40);
        let src = test_title_texture(src_mem as u64, 0xAA);

        let null = std::ptr::null();
        let (mut alloc, mut list) = (null, null);
        let alloc_iid = crate::object::guid(0x6102dee4, 0xaf59, 0x4b09, [0xb9, 0x99, 0xb4, 0x4d, 0x73, 0xf0, 0x9b, 0x24]);
        let list_iid = crate::object::guid(0x5b160d0f, 0xac1b, 0x4185, [0x8b, 0xa8, 0xb3, 0xae, 0x42, 0xa5, 0xa4, 0x55]);
        assert_eq!(crate::list::create_allocator(null, 0, &alloc_iid, &mut alloc), crate::object::S_OK);
        let _alloc = HeldCom(alloc);
        assert_eq!(crate::list::create_list(null, 0, 0, alloc, null, &list_iid, &mut list), crate::object::S_OK);
        let _list = HeldCom(list);

        type CloseFn = extern "system" fn(This) -> crate::object::HRESULT;
        type ResetFn = extern "system" fn(This, This, This) -> crate::object::HRESULT;
        type CopyResourceFn = extern "system" fn(This, This, This);
        let close = crate::object::slot::<CloseFn>(list, 0x48);
        let reset = crate::object::slot::<ResetFn>(list, 0x50);
        let copy = crate::object::slot::<CopyResourceFn>(list, 0x88);

        seed_title_texture(host, dst.0);
        let snap = crate::resource::owned(dst.0).expect("unpinned texture is an eviction candidate");

        // Record after `candidates()` snapshot was taken.
        copy(list, dst.0, src.0);
        let pinned = crate::resource::made_copy(dst.0).unwrap();
        assert_eq!(crate::resource::give_up_snapshot(&[snap]), (0, 0));

        let still = crate::resource::made_copy(dst.0).expect("stale snapshot must not evict pinned copy");
        assert!(Arc::ptr_eq(&pinned, &still));
        assert_eq!(still.pending_host_writes.load(Acquire) & crate::resource::EVICTED, 0);

        assert_eq!(close(list), crate::object::S_OK);
        let recorded = crate::list::host_list(list).unwrap();
        host.execute(&[recorded]);
        assert_eq!(reset(list, alloc, null), crate::object::S_OK);
        assert_eq!(readback_first_texel(host, dst.0), [0xAA, 0xAA, 0xAA, 0xAA]);
    }

    /// Window 1: when `give_up` has already claimed `old` with `EVICTED`
    /// before clearing `resource(dst).host`, `prepare_host_write` must never
    /// return `old`; it detaches `old` and atomically acquires + pins a fresh
    /// replacement `HostTexture`.
    #[test]
    fn prepare_host_write_detaches_claimed_evicted_copy_and_pins_replacement() {
        use std::sync::atomic::Ordering::{AcqRel, Acquire};
        use windows::core::Interface;
        if host().is_none() {
            return; // no host GPU here
        }
        let dst_mem = crate::watch::tests::pool(1);
        let dst = test_title_texture(dst_mem as u64, 0x40);
        let old = crate::resource::host_copy(dst.0).unwrap();

        // Simulate `give_up` claiming `old` via CAS before clearing `resource.host`.
        assert_eq!(
            old.pending_host_writes.compare_exchange(0, crate::resource::EVICTED, AcqRel, Acquire),
            Ok(0)
        );

        let guard = prepare_host_write(dst.0).expect("must detach EVICTED copy and acquire replacement");
        assert_ne!(guard.resource().as_raw(), old.texture.as_raw());
        let current = crate::resource::made_copy(dst.0).unwrap();
        assert!(!Arc::ptr_eq(&current, &old));
        assert!(!crate::resource::give_up_now(dst.0), "replacement must be pinned while guard lives");

        drop(guard);
        assert!(crate::resource::give_up_now(dst.0), "dropping unrecorded guard must unpin replacement");
    }
}
