//! Compiled host pipelines kept on disk, and the title pipelines worth
//! starting early (decision 0007, "Pipeline cache").
//!
//! 1. Every host pipeline goes into the host's pipeline library, named by
//!    a hash of all that its host description holds (the translated
//!    shaders' code, which changes when a translation does, our root
//!    signature, the graphics state). The library is saved under the cache
//!    folder; the host refuses a library of another adapter or driver, and
//!    an entry whose description differs, so neither is ever used.
//! 2. The title pipelines a run drew or dispatched with (made in time or
//!    not) are listed by a key known when the title creates them (their
//!    shader containers and graphics state); the next run hands those to
//!    the workers at creation, during loading. A full list keeps the
//!    latest run's first.
//!
//! 3. The pre-build (`victory translate --cache --pipelines`) lists the
//!    title pipelines it has put in the library, by the same key, so a
//!    second pre-build skips them without translating or loading anything.
//!    The list holds only for the library bytes it was saved with and for
//!    what names the pipelines (the translator, this module and `host`).
//!
//! The library is free-threaded but for loads of one pipeline on several
//! threads (`ID3D12Device1::CreatePipelineLibrary`, "Thread Safety"), so
//! pipelines of different names load at the same time. A load is not always
//! quick: the library of an AMD driver holds about 140 bytes per pipeline,
//! and what its own cache lacks is compiled inside the load, for seconds.
//! Loaded one at a time, 650 pipelines kept the eight workers to one core
//! for minutes, while each title frame could wait out its 500 ms for
//! them (`make`). Stores and serialising keep the lock: the size asked for
//! must still be right when the bytes are written.
//!
//! The library's file is mapped, not read into memory ([`Blob`];
//! Marvisak/VIctory#2129): the host reads the bytes it was made from for as
//! long as it lives, and read into a buffer they were the process's own
//! memory for the whole run (1,352 MB in a world run on
//! fxceless/victory-fork). Mapped, the pages are the file's: only those the
//! host reads are in memory, and the system can drop them without writing
//! them anywhere. A save serialises the new library straight into a
//! mapping of its staging file, not into a second buffer as big.
//!
//! The mapped file's handle stays open for as long as the mapping lives. A
//! save renames the new library over the file this run has mapped, and with
//! the handle closed Windows gives up the old file's contents: pages of the
//! mapping not in memory at that moment read as zeros from then on, which
//! the host reads as stored pipelines (device removals right after a save,
//! on the fork). With the handle open they keep reading the old file
//! (`a_replaced_library_file_still_reads_from_the_disk`). A save goes to
//! [`NEXT_FILE`] first and from there over the library: where a system
//! refuses to replace a mapped file, the new library waits there and takes
//! its place at the next start.
//! `VICTORY_LIBRARY_MAP=0`: read and written through buffers, as before.

use std::collections::{HashMap, HashSet};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Condvar, Mutex, OnceLock};

use windows::Win32::Graphics::Direct3D12::*;
use windows::core::{HSTRING, Interface, PCWSTR};

static DIR: OnceLock<PathBuf> = OnceLock::new();

/// Keeps the pipeline library and the early list in `dir` from now on.
///
/// The library's pipelines are named after the shaders they were made from,
/// so a change to the translator (`victory_shader::SOURCE_HASH`) makes every
/// name in it useless. The first run after one used to write the few
/// pipelines it made over the old library: measured on the title, a 926 MB
/// library (27 160 pipelines, a 7.5 minute warm-up) became 612 KB, and
/// bisecting between two builds paid that warm-up for each of them, in
/// turn, every time. So each translation keeps its library beside the
/// others, in `dir/<SOURCE_HASH>`, and a run only ever writes its own;
/// [`KEPT`] of the newest are kept.
pub fn use_pipeline_cache(dir: PathBuf) {
    let _ = DIR.set(keyed(&dir));
}

/// Whether `dir` is a library directory this wrote: a name that is 16
/// lowercase hex digits (a [`victory_shader::SOURCE_HASH`]) holding
/// [`LIBRARY_FILE`].
fn is_translation(dir: &Path) -> bool {
    let ours = dir
        .file_name()
        .and_then(|name| name.to_str())
        .is_some_and(|name| {
            name.len() == 16
                && name
                    .bytes()
                    .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
        });
    ours && dir.is_dir() && dir.join(LIBRARY_FILE).is_file()
}

/// How many translations keep a library: the one in use and the previous
/// build's, which is what a bisection between two builds needs.
const KEPT: usize = 2;

/// `dir` under this build's translation, with older libraries pruned to
/// [`KEPT`] (by modification time).
fn keyed(dir: &Path) -> PathBuf {
    let path = dir.join(victory_shader::SOURCE_HASH);
    let Ok(entries) = std::fs::read_dir(dir) else {
        return path;
    };
    let mut others: Vec<(std::time::SystemTime, PathBuf)> = entries
        .flatten()
        .map(|entry| entry.path())
        // Only a directory this wrote: named like a translation and holding
        // a library. Anything else in the cache directory (`backup/`, a
        // copy of one) is not ours to delete.
        .filter(|other| *other != path && is_translation(other))
        .filter_map(|other| {
            let at = other.metadata().ok()?.modified().ok()?;
            Some((at, other))
        })
        .collect();
    others.sort_by_key(|(at, _)| std::cmp::Reverse(*at));
    for (_, older) in others.into_iter().skip(KEPT - 1) {
        // A library and a list or two: gone with the translation that made
        // them. Nothing this run reads is ever one of them.
        let _ = std::fs::remove_dir_all(&older);
    }
    path
}

/// Host pipelines created, loaded from the library, and title pipelines
/// started at creation, since the start.
pub(crate) static MADE: AtomicU64 = AtomicU64::new(0);
pub(crate) static LOADED: AtomicU64 = AtomicU64::new(0);
pub(crate) static EARLY: AtomicU64 = AtomicU64::new(0);

/// Whether the library's file is mapped ([`Blob`]) and saved through a
/// mapping. On by default; `VICTORY_LIBRARY_MAP=0` uses buffers, as before.
fn mapped() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("VICTORY_LIBRARY_MAP").is_none_or(|v| v != "0"))
}

/// A whole file mapped into memory: read-only (the library's file) or
/// writable (the file a save fills). Unmapped when dropped.
struct Mapped {
    at: windows_sys::Win32::System::Memory::MEMORY_MAPPED_VIEW_ADDRESS,
    len: usize,
    /// The file, for a mapping made by [`Mapped::open`]: open until after
    /// the view is unmapped (fields drop after `drop`), so that a file a
    /// rename took the name from keeps its contents for the view (see the
    /// module's notes).
    _file: Option<std::fs::File>,
}

// SAFETY: a view of a file, read (or, for a save's, written) by one holder.
unsafe impl Send for Mapped {}

impl Mapped {
    /// Maps `file` (`len` bytes, not zero) for reading, or for reading and
    /// writing.
    fn map(file: &std::fs::File, len: usize, write: bool) -> Option<Mapped> {
        use std::os::windows::io::AsRawHandle;
        use windows_sys::Win32::Foundation::CloseHandle;
        use windows_sys::Win32::System::Memory::{
            CreateFileMappingW, FILE_MAP_READ, FILE_MAP_WRITE, MapViewOfFile, PAGE_READONLY, PAGE_READWRITE,
        };
        if len == 0 {
            return None;
        }
        // SAFETY: a file we hold open; the view keeps the mapping alive
        // after its handle is closed.
        let at = unsafe {
            let mapping = CreateFileMappingW(
                file.as_raw_handle(),
                std::ptr::null(),
                if write { PAGE_READWRITE } else { PAGE_READONLY },
                0,
                0,
                std::ptr::null(),
            );
            if mapping.is_null() {
                return None;
            }
            let access = if write { FILE_MAP_WRITE } else { FILE_MAP_READ };
            let at = MapViewOfFile(mapping, access, 0, 0, 0);
            CloseHandle(mapping);
            at
        };
        (!at.Value.is_null()).then_some(Mapped { at, len, _file: None })
    }

    /// The file at `path`, mapped for reading and held open; `None` if it is
    /// not there, is empty or cannot be mapped.
    fn open(path: &std::path::Path) -> Option<Mapped> {
        use std::os::windows::fs::OpenOptionsExt;
        // Others may read it, and rename or delete it (a save renames the
        // new library over it), but not write it: a write in place would
        // change the bytes under the host while it reads them. Opening it
        // to write or truncate fails with a sharing violation meanwhile.
        const FILE_SHARE_READ: u32 = 1;
        const FILE_SHARE_DELETE: u32 = 4;
        let file = std::fs::OpenOptions::new()
            .read(true)
            .share_mode(FILE_SHARE_READ | FILE_SHARE_DELETE)
            .open(path)
            .ok()?;
        let len = usize::try_from(file.metadata().ok()?.len()).ok()?;
        let mut mapped = Mapped::map(&file, len, false)?;
        mapped._file = Some(file);
        Some(mapped)
    }

    /// A new file of `len` bytes at `path` (replacing what is there),
    /// mapped for writing.
    fn create(path: &std::path::Path, len: usize) -> Option<Mapped> {
        let file = std::fs::OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .truncate(true)
            .open(path)
            .ok()?;
        file.set_len(len as u64).ok()?;
        Mapped::map(&file, len, true)
    }

    fn bytes(&self) -> &[u8] {
        // SAFETY: the view holds `len` readable bytes until dropped.
        unsafe { std::slice::from_raw_parts(self.at.Value.cast(), self.len) }
    }

    /// Only for a view made by [`Mapped::create`].
    fn bytes_mut(&mut self) -> &mut [u8] {
        // SAFETY: a writable view of `len` bytes, ours alone.
        unsafe { std::slice::from_raw_parts_mut(self.at.Value.cast(), self.len) }
    }
}

impl Drop for Mapped {
    fn drop(&mut self) {
        // SAFETY: our view.
        unsafe { windows_sys::Win32::System::Memory::UnmapViewOfFile(self.at) };
    }
}

/// The bytes a library is made from: the host reads them for as long as
/// the library lives, so they are kept beside it.
enum Blob {
    /// No file (an empty library).
    None,
    /// The file read into memory (`VICTORY_LIBRARY_MAP=0`, or a file that
    /// could not be mapped).
    Held(Vec<u8>),
    /// The file itself, mapped.
    Mapped(Mapped),
}

impl Blob {
    /// The library file in `dir`: a library a save left beside it
    /// ([`NEXT_FILE`]) takes its place first.
    fn of(dir: &std::path::Path) -> Blob {
        let path = dir.join(LIBRARY_FILE);
        let next = dir.join(NEXT_FILE);
        if next.exists() && std::fs::rename(&next, &path).is_err() {
            tracing::warn!("the host pipeline library saved last could not take the old one's place");
        }
        if mapped()
            && let Some(file) = Mapped::open(&path)
        {
            return Blob::Mapped(file);
        }
        match std::fs::read(&path) {
            Ok(bytes) if !bytes.is_empty() => Blob::Held(bytes),
            _ => Blob::None,
        }
    }

    fn bytes(&self) -> &[u8] {
        match self {
            Blob::None => &[],
            Blob::Held(bytes) => bytes,
            Blob::Mapped(file) => file.bytes(),
        }
    }
}

struct Library {
    library: ID3D12PipelineLibrary1,
    /// The blob the library was made from: it must outlive the library.
    _blob: Blob,
    /// Pipelines stored since the last save.
    added: u64,
    /// Title pipelines the pre-build put in the library or found there, by
    /// key, and whether that list changed since the last save.
    prebuilt: HashSet<String>,
    prebuilt_changed: bool,
}

// SAFETY: D3D12 objects are free-threaded (a pipeline library with the
// exception `NAMES` takes care of).
unsafe impl Send for Library {}

/// Names one thread at a time holds.
struct Names {
    held: Mutex<Option<HashSet<String>>>,
    released: Condvar,
}

/// A name held until this is dropped.
struct Held<'a> {
    names: &'a Names,
    name: &'a str,
}

impl Names {
    const fn new() -> Names {
        Names {
            held: Mutex::new(None),
            released: Condvar::new(),
        }
    }

    /// Holds `name`, once no other thread does.
    fn hold<'a>(&'a self, name: &'a str) -> Held<'a> {
        let mut held = self.held.lock().unwrap_or_else(|e| e.into_inner());
        while held.as_ref().is_some_and(|h| h.contains(name)) {
            held = self.released.wait(held).unwrap_or_else(|e| e.into_inner());
        }
        held.get_or_insert_default().insert(name.to_owned());
        Held { names: self, name }
    }
}

impl Drop for Held<'_> {
    fn drop(&mut self) {
        let mut held = self.names.held.lock().unwrap_or_else(|e| e.into_inner());
        if let Some(h) = held.as_mut() {
            h.remove(self.name);
        }
        drop(held);
        self.names.released.notify_all();
    }
}

/// Pipelines being loaded, or made and stored.
static NAMES: Names = Names::new();

const LIBRARY_FILE: &str = "host-pipelines.bin";
/// The library saved last, while it has not yet replaced [`LIBRARY_FILE`]:
/// every save is renamed here first, then over the library
/// ([`replace_library`]). Where the second rename is refused (a system that
/// refuses to replace a mapped file), the next start renames it over the
/// library first ([`Blob::of`]).
const NEXT_FILE: &str = "host-pipelines.next";
const EARLY_FILE: &str = "early-pipelines.txt";
const PREBUILT_FILE: &str = "prebuilt-pipelines.txt";
/// Title pipelines listed for an early start, at most.
const EARLY_MAX: usize = 8192;

fn library() -> Option<&'static Mutex<Library>> {
    static LIBRARY: OnceLock<Option<Mutex<Library>>> = OnceLock::new();
    LIBRARY
        .get_or_init(|| {
            let device = crate::host::host()?.device.cast::<ID3D12Device1>().ok()?;
            let read = |file| DIR.get().and_then(|d| std::fs::read(d.join(file)).ok());
            let blob = DIR.get().map_or(Blob::None, |d| Blob::of(d));
            let mut library = open(&device, blob);
            // The pre-build's list, if it was saved with these bytes.
            if let (Some(l), Some(list)) = (library.as_mut(), read(PREBUILT_FILE))
                && !l._blob.bytes().is_empty()
            {
                l.prebuilt = listed(&list, &tag(l._blob.bytes()));
            }
            if library.is_none() {
                tracing::warn!("no host pipeline library: pipelines are not kept");
            }
            library.map(Mutex::new)
        })
        .as_ref()
}

/// The library in `blob`, or an empty one if the host rejects the blob (an
/// adapter or driver change, damage).
fn open(device: &ID3D12Device1, blob: Blob) -> Option<Library> {
    let make = |bytes: &[u8]| -> windows::core::Result<ID3D12PipelineLibrary1> {
        // SAFETY: the bytes outlive the library (kept beside it).
        unsafe {
            device.CreatePipelineLibrary::<ID3D12PipelineLibrary>(bytes)?.cast()
        }
    };
    if !blob.bytes().is_empty() {
        let started = std::time::Instant::now();
        match make(blob.bytes()) {
            Ok(library) => {
                let ms = started.elapsed().as_millis() as u64;
                let (bytes, mapped) = (blob.bytes().len(), matches!(blob, Blob::Mapped(_)));
                tracing::info!(bytes, mapped, ms, "host pipeline library loaded");
                return Some(Library {
                    library,
                    _blob: blob,
                    added: 0,
                    prebuilt: HashSet::new(),
                    prebuilt_changed: false,
                });
            }
            Err(e) => tracing::info!(error = %e, "host pipeline library rejected; starting an empty one"),
        }
    } else {
        tracing::info!("no host pipeline library yet; starting an empty one");
    }
    // The rejected bytes are let go (a mapped file is unmapped) before the
    // empty library is made: nothing reads them any more.
    drop(blob);
    let library = make(&[]).ok()?;
    Some(Library {
        library,
        _blob: Blob::None,
        added: 0,
        prebuilt: HashSet::new(),
        prebuilt_changed: false,
    })
}

/// What the pre-build's list holds for: the library's bytes, and what
/// names its pipelines (the translation, as the shader cache keys it; the
/// host descriptions).
fn tag(blob: &[u8]) -> String {
    let limit = victory_shader::emit::loop_limit().to_le_bytes();
    let translator = victory_shader::SOURCE_HASH.as_bytes();
    let mut h = blake3::Hasher::new();
    for part in [
        blob,
        translator,
        include_bytes!("../shader_cache.rs"),
        &limit,
        include_bytes!("host.rs"),
        include_bytes!("library.rs"),
    ] {
        h.update(&(part.len() as u64).to_le_bytes());
        h.update(part);
    }
    h.finalize().to_hex().to_string()
}

/// The pre-build's list as saved: the tag it holds for, then its keys.
fn list(tag: &str, keys: &HashSet<String>) -> String {
    let keys = keys.iter().map(String::as_str);
    std::iter::once(tag).chain(keys).collect::<Vec<_>>().join("\n")
}

/// The keys of a saved list, if it holds for `tag` (else none).
fn listed(list: &[u8], tag: &str) -> HashSet<String> {
    let list = String::from_utf8_lossy(list);
    let mut lines = list.lines();
    match lines.next() == Some(tag) {
        true => lines.map(str::to_owned).collect(),
        false => HashSet::new(),
    }
}

/// Whether the pre-build has the title pipeline `key` in the library.
pub(crate) fn prebuilt(key: &str) -> bool {
    library().is_some_and(|l| l.lock().unwrap_or_else(|e| e.into_inner()).prebuilt.contains(key))
}

/// The pre-build has the title pipeline `key` in the library now.
pub(crate) fn note_prebuilt(key: String) {
    if let Some(l) = library() {
        let mut l = l.lock().unwrap_or_else(|e| e.into_inner());
        l.prebuilt_changed |= l.prebuilt.insert(key);
    }
}

/// A pipeline's name in the library: the hash of what its host
/// description holds.
pub(crate) fn name(kind: &str, root: &[u8], code: &[&[u8]], state: &str) -> String {
    let mut h = blake3::Hasher::new();
    for part in [kind.as_bytes(), root].into_iter().chain(code.iter().copied()).chain([state.as_bytes()]) {
        h.update(&(part.len() as u64).to_le_bytes());
        h.update(part);
    }
    h.finalize().to_hex()[..32].to_owned()
}

/// The pipeline named `name`: loaded from the library by `load` if it is
/// there with the same description, else made by `create` and stored. A
/// name asked for by several threads is loaded or made by one at a time
/// (the others then find it in the library); different names overlap.
pub(crate) fn pipeline(
    name: &str,
    load: impl FnOnce(&ID3D12PipelineLibrary1, PCWSTR) -> windows::core::Result<ID3D12PipelineState>,
    create: impl FnOnce() -> windows::core::Result<ID3D12PipelineState>,
) -> windows::core::Result<ID3D12PipelineState> {
    let wide = HSTRING::from(name);
    #[cfg(feature = "diagnostics")]
    LAST.set(Some((name.to_owned(), true)));
    let Some(library) = library() else {
        MADE.fetch_add(1, Ordering::Relaxed);
        return create();
    };
    let _held = NAMES.hold(name);
    // The interface alone under the lock: loads take up to seconds. Its
    // blob lives as long as the library, which is never dropped.
    let loads = library
        .lock()
        .unwrap_or_else(|e| e.into_inner())
        .library
        .clone();
    if let Ok(pso) = load(&loads, PCWSTR(wide.as_ptr())) {
        LOADED.fetch_add(1, Ordering::Relaxed);
        return Ok(pso);
    }
    // Compiled outside the lock: compiles take up to seconds.
    #[cfg(feature = "diagnostics")]
    LAST.set(Some((name.to_owned(), false)));
    let pso = create()?;
    MADE.fetch_add(1, Ordering::Relaxed);
    let mut l = library.lock().unwrap_or_else(|e| e.into_inner());
    // SAFETY: a live pipeline; a name already stored (a description that
    // differs) is refused, which leaves the library as it was.
    if unsafe { l.library.StorePipeline(PCWSTR(wide.as_ptr()), &pso) }.is_ok() {
        l.added += 1;
    }
    Ok(pso)
}

#[cfg(feature = "diagnostics")]
thread_local! {
    /// This thread's last pipeline: its name, and whether it was loaded
    /// (else made), for the pipeline log (diagnostics).
    pub(crate) static LAST: std::cell::Cell<Option<(String, bool)>> =
        const { std::cell::Cell::new(None) };
}

/// Writes the library and the early list if pipelines were added since the
/// last save and `SAVE_EVERY` has passed (or `now`).
pub(crate) fn save(now: bool) {
    static LAST: Mutex<Option<std::time::Instant>> = Mutex::new(None);
    const SAVE_EVERY: std::time::Duration = std::time::Duration::from_secs(20);
    let (Some(dir), Some(library)) = (DIR.get(), library()) else { return };
    {
        let mut last = LAST.lock().unwrap_or_else(|e| e.into_inner());
        if !now && last.is_some_and(|t| t.elapsed() < SAVE_EVERY) {
            return;
        }
        *last = Some(std::time::Instant::now());
    }
    if let Err(e) = std::fs::create_dir_all(dir) {
        tracing::warn!(error = %e, "host pipeline library not saved");
        return;
    }
    let (staged, prebuilt) = {
        let mut l = library.lock().unwrap_or_else(|e| e.into_inner());
        if l.added == 0 && !l.prebuilt_changed && !USED_CHANGED.swap(false, Ordering::Relaxed) {
            return;
        }
        // SAFETY: a plain query.
        let size = unsafe { l.library.GetSerializedSize() };
        // The pre-build's list is kept beside the bytes it holds for (also
        // when a run adds pipelines): hashed while they are at hand.
        let want_tag = !l.prebuilt.is_empty();
        let mut hashed = None;
        let serialize = |out: &mut [u8]| {
            // SAFETY: bytes of the size the library asks for, under its lock.
            unsafe { l.library.Serialize(out) }?;
            if want_tag {
                hashed = Some(tag(out));
            }
            Ok(())
        };
        match stage_library(dir, size, serialize) {
            Ok(staged) => {
                l.added = 0;
                l.prebuilt_changed = false;
                let prebuilt = hashed.map(|tag| list(&tag, &l.prebuilt));
                (staged, prebuilt)
            }
            Err(e) => {
                tracing::warn!(error = %e, "host pipeline library not serialised");
                return;
            }
        }
    };
    let used = used().as_ref().map(Used::list).unwrap_or_default();
    let bytes = staged.len();
    let written = staged
        .publish(dir)
        .and_then(|()| write_atomically(&dir.join(EARLY_FILE), used.join("\n").as_bytes()))
        .and_then(|()| match &prebuilt {
            Some(list) => write_atomically(&dir.join(PREBUILT_FILE), list.as_bytes()),
            None => Ok(()),
        });
    let (created, distinct) = created();
    let (shaders_cached, shaders_translated) = crate::shader_cache::counts();
    match written {
        Ok(()) => tracing::info!(
            bytes,
            early = used.len(),
            made = MADE.load(Ordering::Relaxed),
            loaded = LOADED.load(Ordering::Relaxed),
            started_early = EARLY.load(Ordering::Relaxed),
            created,
            distinct,
            shaders_cached,
            shaders_translated,
            "host pipeline library saved"
        ),
        Err(e) => tracing::warn!(error = %e, "host pipeline library not saved"),
    }
}

/// Writes `bytes` to a temporary file beside `path`, then renames it over
/// `path`: a reader never sees a part. Serialised so concurrent worker and
/// shutdown saves do not race on the same staging file or target rename.
fn write_atomically(path: &std::path::Path, bytes: &[u8]) -> std::io::Result<()> {
    let _guard = WRITING.lock().unwrap_or_else(|e| e.into_inner());
    let tmp = staging(path);
    let res = std::fs::write(&tmp, bytes).and_then(|()| std::fs::rename(&tmp, path));
    if res.is_err() {
        let _ = std::fs::remove_file(&tmp);
    }
    res
}

/// Held while files are renamed into place ([`write_atomically`],
/// [`Staged::publish`]).
static WRITING: Mutex<()> = Mutex::new(());

/// A temporary file beside `path`, of a name no other write uses.
fn staging(path: &std::path::Path) -> PathBuf {
    static NEXT: AtomicU64 = AtomicU64::new(0);
    let n = NEXT.fetch_add(1, Ordering::Relaxed);
    let mut name = path.file_name().unwrap_or_default().to_os_string();
    name.push(format!(".tmp-{}-{n}", std::process::id()));
    path.with_file_name(name)
}
/// A serialised library on its way to the library's place.
enum Staged {
    /// Written through a mapping into this staging file, of these bytes.
    File(PathBuf, usize),
    /// In a buffer, to write out (`VICTORY_LIBRARY_MAP=0`, or a staging
    /// file that could not be mapped).
    Bytes(Vec<u8>),
}

impl Staged {
    fn len(&self) -> usize {
        match self {
            Staged::File(_, n) => *n,
            Staged::Bytes(bytes) => bytes.len(),
        }
    }

    /// Puts the library in its place in `dir` ([`replace_library`]); no
    /// staging file is left if that fails.
    fn publish(self, dir: &std::path::Path) -> std::io::Result<()> {
        let tmp = match self {
            Staged::File(tmp, _) => tmp,
            Staged::Bytes(bytes) => {
                let tmp = staging(&dir.join(LIBRARY_FILE));
                if let Err(e) = std::fs::write(&tmp, &bytes) {
                    let _ = std::fs::remove_file(&tmp);
                    return Err(e);
                }
                tmp
            }
        };
        let res = {
            let _guard = WRITING.lock().unwrap_or_else(|e| e.into_inner());
            replace_library(dir, &tmp)
        };
        if res.is_err() {
            let _ = std::fs::remove_file(&tmp);
        }
        res
    }
}

/// The library's `size` serialised bytes (`serialize` fills them): written
/// through a mapping of a staging file in `dir` where it can (the bytes are
/// then the file's pages, not more of the process's own memory), else in a
/// buffer. Called with the library's lock held: the size asked for must
/// still be right when the bytes are written.
fn stage_library(
    dir: &std::path::Path,
    size: usize,
    serialize: impl FnOnce(&mut [u8]) -> windows::core::Result<()>,
) -> std::io::Result<Staged> {
    let failed = |e: windows::core::Error| std::io::Error::other(e.to_string());
    if mapped() {
        let tmp = staging(&dir.join(LIBRARY_FILE));
        if let Some(mut file) = Mapped::create(&tmp, size) {
            let written = serialize(file.bytes_mut()).map_err(failed);
            drop(file);
            if let Err(e) = written {
                let _ = std::fs::remove_file(&tmp);
                return Err(e);
            }
            return Ok(Staged::File(tmp, size));
        }
        let _ = std::fs::remove_file(&tmp);
    }
    let mut bytes = zeroed(size)?;
    serialize(&mut bytes).map_err(failed)?;
    Ok(Staged::Bytes(bytes))
}

/// `n` zero bytes, asked for fallibly: a library is gigabytes, and a host
/// without the memory leaves the save for later instead of aborting.
#[allow(clippy::slow_vector_initialization)] // `vec!` would abort instead.
fn zeroed(n: usize) -> std::io::Result<Vec<u8>> {
    let mut bytes = Vec::new();
    bytes
        .try_reserve_exact(n)
        .map_err(|e| std::io::Error::new(std::io::ErrorKind::OutOfMemory, e))?;
    bytes.resize(n, 0);
    Ok(bytes)
}

/// Puts the written library `tmp` in the library's place in `dir`: renamed
/// over [`NEXT_FILE`], then from there over the library's file, as a start
/// does ([`Blob::of`]). So a pending file is always the newest library
/// saved. This run may have the library's file mapped: Windows replaces a
/// mapped file it opened for sharing, and the mapping goes on reading the
/// old one (its handle is held). Where a system refuses, the new library
/// waits as [`NEXT_FILE`] for the next start.
fn replace_library(dir: &std::path::Path, tmp: &std::path::Path) -> std::io::Result<()> {
    let next = dir.join(NEXT_FILE);
    std::fs::rename(tmp, &next)?;
    // Saved either way: if refused, the next start takes it.
    let _ = std::fs::rename(&next, dir.join(LIBRARY_FILE));
    Ok(())
}


/// Title pipelines created, and their distinct keys.
static CREATED: Mutex<(u64, Option<HashSet<String>>)> = Mutex::new((0, None));

/// The title created a pipeline with `key`.
pub(crate) fn note_created(key: &str) {
    let mut created = CREATED.lock().unwrap_or_else(|e| e.into_inner());
    created.0 += 1;
    created.1.get_or_insert_with(HashSet::new).insert(key.to_owned());
}

/// Title pipelines created so far, and how many distinct ones.
fn created() -> (u64, usize) {
    let created = CREATED.lock().unwrap_or_else(|e| e.into_inner());
    (created.0, created.1.as_ref().map_or(0, HashSet::len))
}

/// Title pipeline keys drawn or dispatched with: in earlier runs (the
/// list as loaded), and in this one.
#[derive(Default)]
struct Used {
    /// Each earlier key with its place in the list as loaded: the list is
    /// kept most recent run first, so a lower place is a more recent run.
    before: HashMap<String, usize>,
    now: HashSet<String>,
}

impl Used {
    /// The list as saved: one key a line, most recent run first (a list
    /// in no order, as earlier builds saved it, reads as well).
    fn read(listed: &str) -> Used {
        let mut before = HashMap::new();
        for (place, key) in listed.lines().filter(|l| !l.is_empty()).enumerate() {
            before.entry(key.to_owned()).or_insert(place);
        }
        Used { before, now: HashSet::new() }
    }

    /// The list to keep: this run's keys, then earlier runs' others, most
    /// recent first, at most `EARLY_MAX` (the oldest go once it is full).
    fn list(&self) -> Vec<String> {
        let mut earlier: Vec<_> = self.before.iter().filter(|(k, _)| !self.now.contains(*k)).collect();
        earlier.sort_unstable_by_key(|&(_, &place)| place);
        let earlier = earlier.into_iter().map(|(k, _)| k);
        self.now.iter().chain(earlier).take(EARLY_MAX).cloned().collect()
    }
}

static USED: Mutex<Option<Used>> = Mutex::new(None);
static USED_CHANGED: AtomicBool = AtomicBool::new(false);

fn used() -> std::sync::MutexGuard<'static, Option<Used>> {
    let mut used = USED.lock().unwrap_or_else(|e| e.into_inner());
    if used.is_none() {
        let listed = DIR
            .get()
            .and_then(|d| std::fs::read_to_string(d.join(EARLY_FILE)).ok())
            .unwrap_or_default();
        let read = Used::read(&listed);
        tracing::info!(pipelines = read.before.len(), "title pipelines to start early");
        *used = Some(read);
    }
    used
}

/// Reads the early list now if not yet (a pre-build asks for no title
/// pipeline).
pub(crate) fn keep_early_list() {
    drop(used());
}

/// Whether the title pipeline with `key` was drawn with in an earlier run.
pub(crate) fn wanted_early(key: &str) -> bool {
    used().as_ref().is_some_and(|u| u.before.contains_key(key))
}

/// The title pipeline with `key` was drawn or dispatched with.
pub(crate) fn note_used(key: &str) {
    if used().get_or_insert_with(Used::default).now.insert(key.to_owned()) {
        USED_CHANGED.store(true, Ordering::Relaxed);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn names_change_with_what_goes_into_the_pipeline() {
        let a = name("graphics", b"root", &[b"vs", b"ps"], "state");
        assert_eq!(a, name("graphics", b"root", &[b"vs", b"ps"], "state"));
        assert_ne!(a, name("graphics", b"root", &[b"vs", b"ps2"], "state"));
        assert_ne!(a, name("graphics", b"root", &[b"vs", b"ps"], "state2"));
        assert_ne!(a, name("graphics", b"root2", &[b"vs", b"ps"], "state"));
        assert_ne!(a, name("mesh", b"root", &[b"vs", b"ps"], "state"));
        // Parts do not run into each other.
        assert_ne!(a, name("graphics", b"root", &[b"vsp", b"s"], "state"));
    }

    /// This run's keys stay in the list; earlier runs' fill it up to the cap.
    #[test]
    fn the_early_list_keeps_this_run_first() {
        let keys = |r: std::ops::Range<usize>| r.map(|i| format!("{i:032x}")).collect::<HashSet<_>>();
        let before = keys(0..EARLY_MAX).into_iter().enumerate().map(|(i, k)| (k, i)).collect();
        let used = Used { before, now: keys(EARLY_MAX - 10..EARLY_MAX + 10) };
        let list = used.list();
        assert_eq!(list.len(), EARLY_MAX);
        assert!(used.now.iter().all(|k| list.contains(k)));
        assert_eq!(list.iter().collect::<HashSet<_>>().len(), EARLY_MAX, "no key twice");
    }

    /// A full list makes room by the oldest runs' keys, and keeps that
    /// order across saves.
    #[test]
    fn a_full_early_list_drops_the_oldest_runs_keys() {
        let key = |i: usize| format!("{i:032x}");
        // As saved: the last run's keys, then the run before's.
        let saved: Vec<String> = (0..EARLY_MAX).map(key).collect();
        let mut used = Used::read(&saved.join("\n"));
        used.now = (EARLY_MAX..EARLY_MAX + 10).map(key).collect();
        used.now.insert(key(EARLY_MAX - 1)); // the oldest key, used again
        let list = used.list();
        assert_eq!(list.len(), EARLY_MAX);
        assert!(used.now.iter().all(|k| list.contains(k)));
        // The ten that go are the oldest ones not used again.
        let dropped: HashSet<String> = (EARLY_MAX - 11..EARLY_MAX - 1).map(key).collect();
        assert!(list.iter().all(|k| !dropped.contains(k)));
        // Read back, this run's keys come first, then the earlier ones in order.
        let again = Used::read(&list.join("\n"));
        assert!(used.now.iter().all(|k| again.before[k] < used.now.len()));
        assert!(again.before[&key(0)] < again.before[&key(1)]);
    }

    /// The pre-build's list holds only for the library bytes it was
    /// saved with.
    #[test]
    fn the_prebuilt_list_holds_for_its_library_only() {
        let keys: HashSet<String> = ["a".repeat(32), "b".repeat(32)].into();
        let saved = list(&tag(b"library"), &keys);
        assert_eq!(listed(saved.as_bytes(), &tag(b"library")), keys);
        assert!(listed(saved.as_bytes(), &tag(b"library2")).is_empty());
        assert!(listed(b"", &tag(b"library")).is_empty());
    }

    #[test]
    fn different_names_are_held_at_the_same_time() {
        static NAMES: Names = Names::new();
        // Each thread holds its name until every thread holds its own.
        let all = std::sync::Barrier::new(8);
        std::thread::scope(|s| {
            for i in 0..8 {
                let all = &all;
                s.spawn(move || {
                    let name = format!("pipeline {i}");
                    let _held = NAMES.hold(&name);
                    all.wait();
                });
            }
        });
        assert!(NAMES.held.lock().unwrap().as_ref().unwrap().is_empty());
    }

    #[test]
    fn one_name_is_held_by_one_thread_at_a_time() {
        static NAMES: Names = Names::new();
        static INSIDE: AtomicU64 = AtomicU64::new(0);
        std::thread::scope(|s| {
            for _ in 0..8 {
                s.spawn(|| {
                    for _ in 0..500 {
                        let _held = NAMES.hold("pipeline");
                        assert_eq!(INSIDE.fetch_add(1, Ordering::SeqCst), 0);
                        std::hint::spin_loop();
                        INSIDE.fetch_sub(1, Ordering::SeqCst);
                    }
                });
            }
        });
        // A holder that panics lets go of it.
        let panicked = std::thread::spawn(|| {
            let _held = NAMES.hold("pipeline");
            panic!("on purpose");
        });
        assert!(panicked.join().is_err());
        drop(NAMES.hold("pipeline"));
    }

    /// A library's pipelines load on many threads at once, each with its
    /// own description (a load with another's is refused), while others are
    /// stored and the library is serialised.
    #[test]
    fn pipelines_load_on_many_threads_at_once() {
        use windows::Win32::Graphics::Direct3D::Fxc::D3DCompile;
        use windows::core::s;
        const N: usize = 48;
        let Some(device) = crate::host::host().and_then(|h| h.device.cast::<ID3D12Device1>().ok())
        else {
            return; // no host GPU here
        };
        let shader = |i: usize| {
            let source = format!(
                "#define RS \"UAV(u0)\"\nRWByteAddressBuffer b : register(u0);\n\
                 [RootSignature(RS)] [numthreads(1, 1, 1)] void main() {{ b.Store(0, b.Load(4) * {i} + {i}); }}"
            );
            let mut blob = None;
            // SAFETY: the source outlives the call; the blob is read within it.
            unsafe {
                D3DCompile(
                    source.as_ptr().cast(),
                    source.len(),
                    None,
                    None,
                    None,
                    s!("main"),
                    s!("cs_5_1"),
                    0,
                    0,
                    &mut blob,
                    None,
                )
                .ok()?;
                let blob = blob?;
                Some(
                    std::slice::from_raw_parts(
                        blob.GetBufferPointer().cast::<u8>(),
                        blob.GetBufferSize(),
                    )
                    .to_vec(),
                )
            }
        };
        let Some(codes) = (0..2 * N).map(shader).collect::<Option<Vec<_>>>() else {
            return; // no shader compiler here
        };
        // SAFETY: the code holds its root signature.
        let root: ID3D12RootSignature =
            unsafe { device.CreateRootSignature(0, &codes[0]) }.unwrap();
        struct Shared(ID3D12Device1, ID3D12RootSignature, ID3D12PipelineLibrary1);
        // SAFETY: D3D12 objects are free-threaded.
        unsafe impl Sync for Shared {}
        impl Shared {
            fn desc(&self, code: &[u8]) -> D3D12_COMPUTE_PIPELINE_STATE_DESC {
                D3D12_COMPUTE_PIPELINE_STATE_DESC {
                    // SAFETY: a borrowed pointer the description never releases.
                    pRootSignature: unsafe { std::mem::transmute_copy(&self.1) },
                    CS: D3D12_SHADER_BYTECODE {
                        pShaderBytecode: code.as_ptr().cast(),
                        BytecodeLength: code.len(),
                    },
                    ..Default::default()
                }
            }
            fn store(&self, i: usize, code: &[u8]) {
                let name = HSTRING::from(format!("p{i}"));
                // SAFETY: the description points at live data.
                unsafe {
                    let pso: ID3D12PipelineState =
                        self.0.CreateComputePipelineState(&self.desc(code)).unwrap();
                    self.2.StorePipeline(&name, &pso).unwrap();
                }
            }
            fn load(&self, i: usize, code: &[u8]) -> windows::core::Result<ID3D12PipelineState> {
                let name = HSTRING::from(format!("p{i}"));
                // SAFETY: as above.
                unsafe { self.2.LoadComputePipeline(&name, &self.desc(code)) }
            }
            fn serialized(&self) -> Vec<u8> {
                // SAFETY: a buffer of the size asked for; nothing is stored
                // meanwhile.
                unsafe {
                    let mut blob = vec![0; self.2.GetSerializedSize()];
                    self.2.Serialize(&mut blob).unwrap();
                    blob
                }
            }
        }
        let first = open(&device, Blob::None).expect("an empty library");
        let made = Shared(device.clone(), root.clone(), first.library.clone());
        (0..N).for_each(|i| made.store(i, &codes[i]));
        let again = open(&device, Blob::Held(made.serialized())).expect("the library again");
        let shared = Shared(device, root, again.library.clone());
        let stores = Mutex::new(());
        std::thread::scope(|s| {
            for t in 0..8 {
                let (shared, codes, stores) = (&shared, &codes, &stores);
                s.spawn(move || {
                    for i in (t..N).step_by(8) {
                        shared.load(i, &codes[i]).expect("a stored pipeline");
                        assert!(
                            shared.load(i, &codes[(i + 1) % N]).is_err(),
                            "another description"
                        );
                        assert!(shared.load(N + i, &codes[N + i]).is_err(), "not stored yet");
                        let _one = stores.lock().unwrap();
                        shared.store(N + i, &codes[N + i]);
                        shared.serialized();
                    }
                });
            }
        });
        for (i, code) in codes.iter().enumerate() {
            shared.load(i, code).expect("every pipeline");
        }
    }

    #[test]
    fn a_rejected_blob_gives_an_empty_library() {
        let Some(device) = crate::host::host().and_then(|h| h.device.cast::<ID3D12Device1>().ok()) else {
            return; // no host GPU here
        };
        let library = open(&device, Blob::Held(b"not a pipeline library".repeat(16))).expect("an empty library");
        // SAFETY: a plain query on a live library.
        assert!(unsafe { library.library.GetSerializedSize() } > 0);
        assert!(matches!(library._blob, Blob::None));
    }

    /// A library made from its mapped file keeps the mapping (and the
    /// file) for as long as it lives, and a save replacing the file while
    /// it lives leaves it reading the old one.
    #[test]
    fn a_library_keeps_its_mapped_file_open() {
        let Some(device) = crate::host::host().and_then(|h| h.device.cast::<ID3D12Device1>().ok()) else {
            return; // no host GPU here
        };
        if !mapped() {
            return; // `VICTORY_LIBRARY_MAP=0`
        }
        let dir = tempfile::tempdir().unwrap();
        let empty = open(&device, Blob::None).expect("an empty library");
        // SAFETY: a buffer of the size the library asks for.
        let bytes = unsafe {
            let mut bytes = vec![0u8; empty.library.GetSerializedSize()];
            empty.library.Serialize(&mut bytes).map(|()| bytes)
        }
        .unwrap();
        std::fs::write(dir.path().join(LIBRARY_FILE), &bytes).unwrap();
        let blob = Blob::of(dir.path());
        assert!(matches!(blob, Blob::Mapped(_)), "mapped by default");
        let library = open(&device, blob).expect("the library");
        let Blob::Mapped(mapped) = &library._blob else {
            panic!("the library holds its mapped file");
        };
        assert!(mapped._file.is_some(), "the file stays open while the library lives");
        assert_eq!(mapped.bytes(), &bytes[..]);
        let staged = stage_library(dir.path(), 4096, |out| {
            out.fill(9);
            Ok(())
        })
        .unwrap();
        staged.publish(dir.path()).unwrap();
        assert_eq!(std::fs::read(dir.path().join(LIBRARY_FILE)).unwrap(), [9u8; 4096]);
        assert_eq!(mapped.bytes(), &bytes[..], "the mapping still reads the old library");
    }

    /// A mapped file reads as the file's bytes, a save writes the library
    /// through a mapping of a staging file and renames it over the mapped
    /// one, the mapping still reads the old library, and the next start
    /// reads the new one.
    #[test]
    fn the_library_file_is_mapped_and_replaced() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join(LIBRARY_FILE);
        assert!(matches!(Blob::of(dir.path()), Blob::None), "no file yet");
        std::fs::write(&path, b"").unwrap();
        assert!(matches!(Blob::of(dir.path()), Blob::None), "an empty file");
        let old: Vec<u8> = (0..200_000u32).map(|i| i as u8).collect();
        std::fs::write(&path, &old).unwrap();
        let blob = Blob::of(dir.path());
        assert!(matches!(blob, Blob::Mapped(_)) || !mapped());
        assert_eq!(blob.bytes(), &old[..]);
        // A save while the old file is mapped.
        let new: Vec<u8> = (0..300_000u32).map(|i| (i * 7) as u8).collect();
        let staged = stage_library(dir.path(), new.len(), |out| {
            out.copy_from_slice(&new);
            Ok(())
        })
        .unwrap();
        assert!(matches!(staged, Staged::File(..)) || !mapped(), "written through a mapping");
        staged.publish(dir.path()).unwrap();
        // The mapping still reads the old library; no staging file is left.
        assert_eq!(blob.bytes(), &old[..]);
        let left: Vec<_> = std::fs::read_dir(dir.path()).unwrap().flatten().map(|e| e.file_name()).collect();
        assert_eq!(left, [std::ffi::OsString::from(LIBRARY_FILE)], "{left:?}");
        drop(blob);
        // The next start reads the new one.
        assert_eq!(Blob::of(dir.path()).bytes(), &new[..]);
        assert!(!dir.path().join(NEXT_FILE).exists());
        // A serialisation that fails leaves the library as it was.
        let refused = stage_library(dir.path(), 64, |_| Err(windows::core::Error::from_hresult(windows::core::HRESULT(-1))));
        assert!(refused.is_err());
        assert_eq!(std::fs::read(&path).unwrap(), new);
        assert_eq!(std::fs::read_dir(dir.path()).unwrap().count(), 1, "no staging file left");
    }

    /// A save that cannot replace the library's file waits as the pending
    /// file, which the next start puts in the library's place.
    #[test]
    fn a_library_that_cannot_replace_the_old_one_waits_for_the_next_start() {
        use std::os::windows::fs::OpenOptionsExt;
        let dir = tempfile::tempdir().unwrap();
        let (path, next) = (dir.path().join(LIBRARY_FILE), dir.path().join(NEXT_FILE));
        let save = |byte: u8| {
            stage_library(dir.path(), 4096, |out| {
                out.fill(byte);
                Ok(())
            })
            .and_then(|staged| staged.publish(dir.path()))
        };
        save(1).unwrap();
        // Opened without delete sharing, a file cannot be renamed over.
        let held = std::fs::OpenOptions::new().read(true).share_mode(1).open(&path).unwrap();
        save(2).unwrap();
        assert_eq!(std::fs::read(&next).unwrap(), [2u8; 4096], "waits as the pending file");
        drop(held);
        assert_eq!(Blob::of(dir.path()).bytes(), [2u8; 4096]);
        assert!(!next.exists());
    }

    /// A save buffer the host cannot give is an error, not an abort.
    #[test]
    fn a_save_buffer_the_host_cannot_give_is_an_error() {
        let refused = zeroed(usize::MAX / 2).expect_err("no such memory");
        assert_eq!(refused.kind(), std::io::ErrorKind::OutOfMemory);
        assert_eq!(zeroed(100).unwrap(), [0u8; 100]);
    }

    /// The bytes at `at` of the pattern the test below writes.
    fn pattern(at: usize, out: &mut [u8]) {
        for (i, word) in out.as_chunks_mut::<4>().0.iter_mut().enumerate() {
            let v = ((at / 4 + i) as u32).wrapping_mul(2_654_435_761).wrapping_add(1);
            *word = v.to_le_bytes();
        }
    }

    /// A file of `len` bytes (whole MiB) of [`pattern`], written past the
    /// system's cache: none of its pages is in memory afterwards. False
    /// where the folder's file system does not take such writes.
    fn write_uncached(path: &std::path::Path, len: usize) -> bool {
        use std::io::Write;
        use std::os::windows::fs::OpenOptionsExt;
        // FILE_FLAG_NO_BUFFERING | FILE_FLAG_WRITE_THROUGH
        const UNCACHED: u32 = 0x2000_0000 | 0x8000_0000;
        const CHUNK: usize = 1 << 20;
        let Ok(mut file) = std::fs::OpenOptions::new()
            .write(true)
            .create(true)
            .truncate(true)
            .custom_flags(UNCACHED)
            .open(path)
        else {
            return false;
        };
        // Such writes want a buffer on a sector boundary.
        let mut store = vec![0u8; CHUNK + 4096];
        let start = store.as_ptr().align_offset(4096);
        for at in (0..len).step_by(CHUNK) {
            let chunk = &mut store[start..start + CHUNK];
            pattern(at, chunk);
            if file.write_all(chunk).is_err() {
                return false;
            }
        }
        true
    }

    /// The library's file is replaced by a save while it is mapped, and
    /// pages of the mapping that were not in memory then are read after:
    /// with the file's handle held they still read as the old file (closed,
    /// they read as zeros, which the host would take for pipelines and
    /// remove the device).
    #[test]
    fn a_replaced_library_file_still_reads_from_the_disk() {
        const LEN: usize = 8 << 20;
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join(LIBRARY_FILE);
        assert!(write_uncached(&path, LEN), "uncached writes in the temporary folder");
        let mapped = Mapped::open(&path).expect("the file, mapped");
        // A save: the new library, renamed over the mapped one.
        stage_library(dir.path(), LEN, |out| {
            out.fill(0xEE);
            Ok(())
        })
        .and_then(|staged| staged.publish(dir.path()))
        .unwrap();
        assert!(!dir.path().join(NEXT_FILE).exists(), "replaced in place");
        // Every page of the old mapping, read for the first time now.
        let mut expected = vec![0u8; 4096];
        for (page, bytes) in mapped.bytes().chunks(4096).enumerate() {
            pattern(page * 4096, &mut expected);
            assert!(bytes == &expected[..], "page {page} of the replaced library no longer reads as the old file");
        }
    }

    /// While the library's file is mapped, no one may write it in place or
    /// truncate it (the bytes would change under the host); reading it,
    /// and a save renaming a new library over it, still work.
    #[test]
    fn a_mapped_library_file_cannot_be_written_meanwhile() {
        const SHARING_VIOLATION: i32 = 32;
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join(LIBRARY_FILE);
        let old = vec![0x5Au8; 3 * 4096];
        std::fs::write(&path, &old).unwrap();
        let mapped = Mapped::open(&path).expect("the file, mapped");
        let write = std::fs::OpenOptions::new().write(true).open(&path);
        assert_eq!(write.err().and_then(|e| e.raw_os_error()), Some(SHARING_VIOLATION), "written in place");
        let truncate = std::fs::OpenOptions::new().write(true).truncate(true).open(&path);
        assert_eq!(truncate.err().and_then(|e| e.raw_os_error()), Some(SHARING_VIOLATION), "truncated");
        assert_eq!(std::fs::read(&path).unwrap(), old, "others still read it");
        stage_library(dir.path(), 4096, |out| {
            out.fill(1);
            Ok(())
        })
        .and_then(|staged| staged.publish(dir.path()))
        .unwrap();
        assert!(!dir.path().join(NEXT_FILE).exists(), "replaced in place");
        assert_eq!(std::fs::read(&path).unwrap(), [1u8; 4096]);
        assert_eq!(mapped.bytes(), &old[..]);
    }

    /// The process's own memory a library file costs: read into memory and
    /// mapped (every page read once).
    /// `cargo test --release -p victory-gpu --lib library_file_memory -- --ignored --nocapture`
    #[test]
    #[ignore = "measurement: run with --ignored --nocapture"]
    fn library_file_memory() {
        use windows_sys::Win32::System::ProcessStatus::{GetProcessMemoryInfo, PROCESS_MEMORY_COUNTERS_EX};
        use windows_sys::Win32::System::Threading::GetCurrentProcess;
        fn private() -> f64 {
            let mut c = PROCESS_MEMORY_COUNTERS_EX::default();
            // SAFETY: a plain query into our own counters.
            unsafe { GetProcessMemoryInfo(GetCurrentProcess(), (&raw mut c).cast(), size_of_val(&c) as u32) };
            c.PrivateUsage as f64 / f64::from(1u32 << 20)
        }
        const LEN: usize = 256 << 20;
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join(LIBRARY_FILE);
        std::fs::write(&path, vec![7u8; LEN]).unwrap();
        let sum = |b: &[u8]| b.iter().step_by(4096).map(|&x| u64::from(x)).sum::<u64>();
        let before = private();
        let read = std::fs::read(&path).unwrap();
        assert_eq!(sum(&read), 7 * (LEN / 4096) as u64);
        let held = private() - before;
        drop(read);
        let before = private();
        let mapped = Mapped::open(&path).unwrap();
        assert_eq!(sum(mapped.bytes()), 7 * (LEN / 4096) as u64);
        let mapped_cost = private() - before;
        drop(mapped);
        eprintln!("256 MiB library file: {held:.1} MiB private read into memory, {mapped_cost:.1} MiB mapped");
    }


    #[test]
    fn concurrent_write_atomically_never_fails_or_truncates() {
        let dir = tempfile::tempdir().unwrap();
        let target = dir.path().join("host-pipelines.bin");
        let threads: Vec<_> = (0..4)
            .map(|t| {
                let target = target.clone();
                std::thread::spawn(move || {
                    let payload = vec![b'a' + t as u8; 4096];
                    for _ in 0..30 {
                        write_atomically(&target, &payload).unwrap();
                    }
                })
            })
            .collect();
        for t in threads {
            t.join().unwrap();
        }
        let final_bytes = std::fs::read(&target).unwrap();
        assert_eq!(final_bytes.len(), 4096);
        assert!(final_bytes.iter().all(|&b| b == final_bytes[0]));
    }
}

#[cfg(test)]
mod keyed_tests {
    use super::*;

    /// Pruning keeps only directories this wrote: a `backup/` (or any
    /// directory without a hash name) in the cache directory survives,
    /// with what is in it, even when it is the oldest thing there.
    #[test]
    fn pruning_leaves_foreign_directories_alone() {
        let dir = std::env::temp_dir().join(format!("victory-keyed-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        // An old directory of ours, an old one of someone else's, and the
        // one this build uses.
        for (name, own) in [("00112233aabbccdd", true), ("backup", false)] {
            let at = dir.join(name);
            std::fs::create_dir_all(&at).unwrap();
            std::fs::write(at.join("sentinel"), b"keep me").unwrap();
            if own {
                std::fs::write(at.join(LIBRARY_FILE), b"library").unwrap();
            }
        }
        let current = dir.join(victory_shader::SOURCE_HASH);
        std::fs::create_dir_all(&current).unwrap();
        std::fs::write(current.join(LIBRARY_FILE), b"library").unwrap();

        assert_eq!(keyed(&dir), current);
        assert!(dir.join("backup").join("sentinel").is_file(), "a foreign directory was deleted");
        let _ = std::fs::remove_dir_all(&dir);
    }
}
