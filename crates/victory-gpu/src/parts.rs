//! Device methods 0x1c8 and 0x218: pipelines the title hands over already
//! compiled for the console, as bare GFX10 code blocks with their metadata
//! instead of shader containers (layouts from the descriptions logged by
//! `device::unknown_pipeline`; see also fxceless/victory-fork#113 and #1151).
//!
//! ```text
//! 0x218, one compute program (0x60 bytes; the title keeps them back to back)
//! +0x00 root signature (one of ours)
//! +0x08 header: pointer, size (0x1c): u32 0x4005cf00, u32 0x6c090102, u32 0,
//!       u16 x3 thread-group size, u16 flags (bit 4 wave32),
//!       u32 COMPUTE_PGM_RSRC1, u32 COMPUTE_PGM_RSRC2
//! +0x18 hash block: pointer, size (0x14)
//! +0x28 code: pointer (title GPU memory, 256-byte aligned), size
//! +0x38 table: pointer, size
//! +0x48 two links of the title's own keyed table, +0x58 a 64-bit hash
//!
//! 0x1c8, a graphics pipeline (0x160 bytes)
//! +0x00 root signature
//! +0x08 state block: pointer, size (0x17c to 0x200)
//! +0x18 input layout: pointer, size (semantic names inside)
//! +0x28 programs, 0x20 bytes each: code pointer, code size, table pointer,
//!       table size; the first without code ends them (two in most calls)
//! +0x118 u64 0x0401000000000160 (0x160, the description's size)
//!
//! table +0x00 u32 0x403, u32 size - 0x14, u32 code dwords; from +0x18
//!       32-byte records: u32 tag, ...; 0x41 used resource, 0x42 a place in
//!       the code and a register, 0x43 the end (a 16-byte hash follows)
//! ```
//!
//! What the title does when either method fails (its code, read in
//! `game_scarlett_master.exe`):
//! - 0x218 (only caller title+0x247fe25): it goes straight on to make the
//!   same pipeline with `CreateComputePipelineStateX` (0x1f0) from its own
//!   container description.
//! - 0x1c8 (title+0x2477578): a loop over a list of entries (the
//!   description at entry+0x40) that fills a table under a lock on success
//!   and skips the entry on failure; the table is a prebuilt cache, the
//!   draw-time creation methods make what it lacks.
//!
//! Neither code block is the program of a container (only 46 of 2,689
//! compute blocks match one), and building the translator's native program
//! from the tables translates few of them (228 of 2,689 in #113's attempt),
//! while a success makes the title skip its fallback. So both methods still
//! fail, and this module counts, per root signature, the failed calls
//! against the pipelines then made the usual way: equal counts say the
//! title went without nothing. Measured (RX 9070 XT, Series X profile, five
//! minutes into the world): all 2,689 failed 0x218 calls were followed, on
//! the same thread, by a compute creation with the same root signature;
//! the 25,450 failed 0x1c8 calls used 45 root signatures, under which
//! 79,158 graphics pipelines were made the usual way (covering 25,380 of
//! them signature by signature; nearly every graphics pipeline uses those
//! signatures, so this side is weaker evidence than the compute one).

use std::collections::HashMap;
use std::sync::Mutex;

use crate::object::This;

/// A block the title points at: its address and size in bytes.
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub(crate) struct Block {
    pub(crate) at: *const u8,
    pub(crate) size: usize,
}

impl Block {
    fn present(&self) -> bool {
        !self.at.is_null() && self.size != 0
    }
}

/// Method 0x218's description, its first 0x60 bytes.
#[repr(C)]
pub(crate) struct ComputeParts {
    pub(crate) root: This,
    pub(crate) header: Block,
    pub(crate) hash_block: Block,
    pub(crate) code: Block,
    pub(crate) table: Block,
    pub(crate) links: [usize; 2],
    pub(crate) hash: u64,
}

/// One program of a 0x1c8 description.
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub(crate) struct Program {
    pub(crate) code: Block,
    pub(crate) table: Block,
}

/// The program slots between +0x28 and +0x118.
pub(crate) const GRAPHICS_PROGRAMS: usize = 7;

/// Method 0x1c8's description, its first 0x120 bytes.
#[repr(C)]
pub(crate) struct GraphicsParts {
    /// Read through [`root`], the same word in both layouts.
    #[allow(dead_code)]
    pub(crate) root: This,
    pub(crate) state: Block,
    pub(crate) input_layout: Block,
    pub(crate) programs: [Program; GRAPHICS_PROGRAMS],
    pub(crate) tail: [u64; 2],
    pub(crate) size_word: u64,
}

impl GraphicsParts {
    /// The programs given, up to the first slot without code.
    pub(crate) fn programs(&self) -> impl Iterator<Item = &Program> {
        self.programs.iter().take_while(|p| p.code.present())
    }
}

/// A description's root signature (its first word in both layouts).
pub(crate) fn root(desc: *const u8) -> This {
    if desc.is_null() {
        return std::ptr::null();
    }
    // SAFETY: the title passes a description; only its first word is read.
    unsafe { std::ptr::addr_of!((*desc.cast::<ComputeParts>()).root).read_unaligned() }
}

/// What a description gives, for the log: block and program sizes.
pub(crate) fn shape(kind: Kind, desc: *const u8) -> String {
    if desc.is_null() {
        return "none".into();
    }
    match kind {
        Kind::Compute => {
            // SAFETY: the title passes a 0x218 description.
            let d = unsafe { desc.cast::<ComputeParts>().read_unaligned() };
            format!(
                "header {:#x}, hash block {:#x}, code {:#x}, table {:#x}, hash {:016x}, links {}",
                d.header.size,
                d.hash_block.size,
                d.code.size,
                d.table.size,
                d.hash,
                d.links.iter().filter(|&&l| l != 0).count()
            )
        }
        Kind::Graphics => {
            // SAFETY: the title passes a 0x1c8 description.
            let d = unsafe { desc.cast::<GraphicsParts>().read_unaligned() };
            let programs: Vec<_> = d
                .programs()
                .map(|p| format!("{:#x}+{:#x}", p.code.size, p.table.size))
                .collect();
            format!(
                "state {:#x}, input layout {:#x}, programs [{}], tail {:x?}, size word {:#x}",
                d.state.size,
                d.input_layout.size,
                programs.join(", "),
                d.tail,
                d.size_word
            )
        }
    }
}

/// The program table's header: tag 0x403 and its size less 0x14.
#[cfg_attr(not(test), allow(dead_code))]
pub(crate) fn table_fits(table: &[u8]) -> bool {
    let word = |i: usize| {
        table
            .get(i * 4..i * 4 + 4)
            .map(|b| u32::from_le_bytes([b[0], b[1], b[2], b[3]]))
    };
    word(0) == Some(0x403) && word(1).map(|s| s as usize + 0x14) == Some(table.len())
}

/// Which kind of pipeline.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) enum Kind {
    Graphics = 0,
    Compute = 1,
}

#[derive(Default, Clone, Copy, Debug, PartialEq, Eq)]
struct Counts {
    /// Failed 0x1c8 (graphics) and 0x218 (compute) calls.
    parts: [u32; 2],
    /// Pipelines made the usual way.
    usual: [u32; 2],
}

/// Per root signature, the failed calls and the usual creations; and for
/// 0x218, whether the very next compute creation on the same thread had the
/// same root signature (the fallback the title's code makes).
#[derive(Default, Debug)]
pub(crate) struct Book {
    roots: HashMap<usize, Counts>,
    linked: u32,
    unlinked: u32,
}

/// A report: per kind, the failed calls, the usual creations with the root
/// signatures those calls named, how many of the failed ones that covers,
/// and all usual creations.
#[derive(Debug, PartialEq, Eq)]
pub(crate) struct Tally {
    pub(crate) parts: [u32; 2],
    pub(crate) usual_same_roots: [u32; 2],
    pub(crate) covered: [u32; 2],
    pub(crate) usual: [u32; 2],
    pub(crate) roots: [u32; 2],
    pub(crate) linked: u32,
    pub(crate) unlinked: u32,
}

impl Book {
    pub(crate) fn failed(&mut self, kind: Kind, root: usize) {
        self.roots.entry(root).or_default().parts[kind as usize] += 1;
    }

    /// A usual creation; `pending` is the root signature of a failed 0x218
    /// call on this thread not yet followed by a compute creation.
    pub(crate) fn usual(&mut self, kind: Kind, root: usize, pending: Option<usize>) {
        self.roots.entry(root).or_default().usual[kind as usize] += 1;
        if let (Kind::Compute, Some(p)) = (kind, pending) {
            if p == root {
                self.linked += 1;
            } else {
                self.unlinked += 1;
            }
        }
    }

    /// A failed 0x218 call followed by another one, not by a creation.
    pub(crate) fn not_followed(&mut self) {
        self.unlinked += 1;
    }

    pub(crate) fn tally(&self) -> Tally {
        let mut t = Tally {
            parts: [0; 2],
            usual_same_roots: [0; 2],
            covered: [0; 2],
            usual: [0; 2],
            roots: [0; 2],
            linked: self.linked,
            unlinked: self.unlinked,
        };
        for c in self.roots.values() {
            for k in 0..2 {
                t.usual[k] += c.usual[k];
                if c.parts[k] != 0 {
                    t.roots[k] += 1;
                    t.parts[k] += c.parts[k];
                    t.usual_same_roots[k] += c.usual[k];
                    t.covered[k] += c.parts[k].min(c.usual[k]);
                }
            }
        }
        t
    }
}

static BOOK: Mutex<Option<Book>> = Mutex::new(None);

thread_local! {
    /// The root signature of this thread's last failed 0x218 call.
    static PENDING: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
}

fn with_book<R>(f: impl FnOnce(&mut Book) -> R) -> R {
    let mut book = BOOK.lock().unwrap_or_else(|e| e.into_inner());
    f(book.get_or_insert_with(Book::default))
}

/// A failed 0x1c8 or 0x218 call.
pub(crate) fn note_failed(kind: Kind, root: This) {
    let root = root as usize;
    let previous = if kind == Kind::Compute {
        PENDING.with(|p| p.replace(root))
    } else {
        0
    };
    with_book(|b| {
        b.failed(kind, root);
        if previous != 0 {
            b.not_followed();
        }
    });
}

/// A pipeline made the usual way, with its root signature (null if none).
pub(crate) fn note_usual(kind: Kind, root: This) {
    let pending = if kind == Kind::Compute {
        Some(PENDING.with(|p| p.replace(0))).filter(|&p| p != 0)
    } else {
        None
    };
    let (total, tally) = with_book(|b| {
        b.usual(kind, root as usize, pending);
        let t = b.tally();
        (t.usual[0] + t.usual[1], t)
    });
    if (total.is_power_of_two() && total >= 256) || total % 8192 == 0 {
        report(&tally);
    }
}

fn report(t: &Tally) {
    tracing::info!(
        graphics_failed = t.parts[0],
        graphics_roots = t.roots[0],
        graphics_usual_same_roots = t.usual_same_roots[0],
        graphics_covered = t.covered[0],
        compute_failed = t.parts[1],
        compute_roots = t.roots[1],
        compute_usual_same_roots = t.usual_same_roots[1],
        compute_covered = t.covered[1],
        compute_fallback_next = t.linked,
        compute_not_next = t.unlinked,
        usual_graphics = t.usual[0],
        usual_compute = t.usual[1],
        "pipelines 0x1c8/0x218: failed calls against the usual creations with the same root signatures"
    );
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn layouts_match_the_logged_offsets() {
        assert_eq!(std::mem::size_of::<ComputeParts>(), 0x60);
        assert_eq!(std::mem::offset_of!(ComputeParts, code), 0x28);
        assert_eq!(std::mem::offset_of!(ComputeParts, table), 0x38);
        assert_eq!(std::mem::offset_of!(ComputeParts, hash), 0x58);
        assert_eq!(std::mem::offset_of!(GraphicsParts, programs), 0x28);
        assert_eq!(std::mem::offset_of!(GraphicsParts, size_word), 0x118);
    }

    #[test]
    fn graphics_programs_end_at_the_first_without_code() {
        // As logged: state 0x184, input layout 0xbc, two programs.
        let state = [0u8; 0x184];
        let layout = [0u8; 0xbc];
        let code = [0u8; 0x400];
        let tables = [0u8; 0x180];
        let none = Block { at: std::ptr::null(), size: 0 };
        let block = |s: &[u8]| Block { at: s.as_ptr(), size: s.len() };
        let mut programs = [Program { code: none, table: none }; GRAPHICS_PROGRAMS];
        programs[0] = Program { code: block(&code[..0x200]), table: block(&tables[..0xd0]) };
        programs[1] = Program { code: block(&code[0x200..]), table: block(&tables[0xd0..]) };
        programs[3] = programs[0]; // after the end: not read
        let desc = GraphicsParts {
            root: std::ptr::null(),
            state: block(&state),
            input_layout: block(&layout),
            programs,
            tail: [0; 2],
            size_word: 0x0401_0000_0000_0160,
        };
        let sizes: Vec<_> = desc.programs().map(|p| (p.code.size, p.table.size)).collect();
        assert_eq!(sizes, [(0x200, 0xd0), (0x200, 0xb0)]);
        assert_eq!(desc.size_word & 0xffff, 0x160);
    }

    #[test]
    fn a_table_starts_with_its_tag_and_size() {
        let mut table = vec![0u8; 0x118];
        table[..4].copy_from_slice(&0x403u32.to_le_bytes());
        table[4..8].copy_from_slice(&(0x118u32 - 0x14).to_le_bytes());
        assert!(table_fits(&table));
        table[4] ^= 1;
        assert!(!table_fits(&table));
        assert!(!table_fits(&table[..6]));
    }

    #[test]
    fn the_book_matches_failed_calls_to_usual_creations() {
        let mut b = Book::default();
        // Two failed 0x218 calls under root 1, each followed by its fallback.
        b.failed(Kind::Compute, 1);
        b.usual(Kind::Compute, 1, Some(1));
        b.failed(Kind::Compute, 1);
        b.usual(Kind::Compute, 1, Some(1));
        // A failed 0x218 call under root 2, then a compute creation under 3.
        b.failed(Kind::Compute, 2);
        b.usual(Kind::Compute, 3, Some(2));
        // Three failed 0x1c8 calls under root 4, one usual creation later.
        for _ in 0..3 {
            b.failed(Kind::Graphics, 4);
        }
        b.usual(Kind::Graphics, 4, None);
        b.usual(Kind::Graphics, 5, None);
        let t = b.tally();
        assert_eq!(t.parts, [3, 3]);
        assert_eq!(t.roots, [1, 2]);
        assert_eq!(t.usual_same_roots, [1, 2]);
        assert_eq!(t.covered, [1, 2]);
        assert_eq!(t.usual, [2, 3]);
        assert_eq!((t.linked, t.unlinked), (2, 1));
    }

    #[test]
    fn a_failed_call_followed_by_another_counts_as_not_followed() {
        let mut b = Book::default();
        b.failed(Kind::Compute, 1);
        b.not_followed();
        b.failed(Kind::Compute, 1);
        b.usual(Kind::Compute, 1, Some(1));
        assert_eq!((b.tally().linked, b.tally().unlinked), (1, 1));
    }
}
