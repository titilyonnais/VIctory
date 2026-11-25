//! Small CPU-visible host buffers, placed in shared heaps.
//!
//! A committed buffer is a kernel allocation of its own; a placed one takes
//! a slot in a heap allocated once for [`HEAP_SLOTS`] of them. A slot is
//! free again when its buffer is destroyed, which the host reports through
//! `ID3DDestructionNotifier`.
//!
//! Holds buffers of up to [`SLOT_BYTES`] only; larger ones stay committed
//! (`Host::shared_buffer`).
//!
//! A placed buffer is zeroed over its width only; the rest of its slot keeps
//! an earlier buffer's bytes. Views are bounded by a buffer's width; root
//! descriptors are not, so callers that bind one past its width ask for
//! whole slots (`Host::shared_buffer`). An access past a slot's end, which
//! D3D12 leaves undefined, reaches the next slot's buffer here, where past a
//! committed buffer it may fault.
//!
//! Heaps no buffer is left in are released one a tick of the frame meter
//! (once a second while frames are presented, [`Pool::release_spare`]), each
//! only while a heap's worth of slots stays free without it. A few buffers
//! that live long and lie in many heaps hold all of those heaps, so after a
//! peak the pool can keep more memory than committed buffers would. Without
//! a peak it keeps more too: the free slots of the heaps it holds.
//!
//! A freed slot is handed out again, at the same GPU address. A use of a
//! buffer after its last reference went is undefined for any buffer: a
//! committed one may fault, here it reaches the slot's next buffer. What
//! prevents it is the same for both: whoever records a buffer's address
//! keeps the buffer until the GPU is done with it (the lists' kept
//! resources, `host::retire`).

use std::ffi::c_void;
use std::sync::{Mutex, MutexGuard};

use windows::Win32::Graphics::Direct3D::ID3DDestructionNotifier;
use windows::Win32::Graphics::Direct3D12::{
    D3D12_DEFAULT_RESOURCE_PLACEMENT_ALIGNMENT, D3D12_HEAP_DESC,
    D3D12_HEAP_FLAG_ALLOW_ONLY_BUFFERS, D3D12_RESOURCE_DESC, D3D12_RESOURCE_STATE_COMMON,
    ID3D12Device, ID3D12Heap, ID3D12Resource,
};
use windows::core::{Interface, Result};

use crate::host::{SHARED_HEAP, shared_buffer_desc};

/// The bytes of one slot, and so the widest buffer placed: the host places
/// buffers at multiples of this.
pub(crate) const SLOT_BYTES: u64 = D3D12_DEFAULT_RESOURCE_PLACEMENT_ALIGNMENT as u64;

/// Slots per heap.
const HEAP_SLOTS: usize = 256;

/// The bytes of one heap.
const HEAP_BYTES: u64 = HEAP_SLOTS as u64 * SLOT_BYTES;

/// Slots per word of a heap's [free mask](Heap::free).
const WORD_SLOTS: usize = u64::BITS as usize;

const _: () = assert!(HEAP_SLOTS.is_multiple_of(WORD_SLOTS));

/// A slot's number: its heap's place in [`Slots::heaps`], then its place in
/// the heap.
#[derive(Clone, Copy)]
struct Slot(usize);

impl Slot {
    fn heap(self) -> usize {
        self.0 / HEAP_SLOTS
    }

    fn in_heap(self) -> usize {
        self.0 % HEAP_SLOTS
    }

    fn offset_bytes(self) -> u64 {
        self.in_heap() as u64 * SLOT_BYTES
    }
}

/// One device's heaps of [`SHARED_HEAP`] memory and their slots.
pub(crate) struct Pool {
    device: ID3D12Device,
    /// A leaf lock: taken under the callers' locks (`shadow`'s maps around
    /// `Host::shared_buffer`) and, in [`destroyed`], under whatever the
    /// thread that drops a buffer holds. Nothing is locked, created or
    /// released while it is held; a heap is only cloned.
    slots: Mutex<Slots>,
}

struct Slots {
    /// A heap keeps its place while it lives, so that its slots keep their
    /// numbers; a released heap's place is `None` until a new heap takes it.
    heaps: Vec<Option<Heap>>,
    /// The host refused a heap, a buffer or the report of a destruction:
    /// nothing more is placed. The heaps stay for the buffers still in them.
    closed: bool,
}

struct Heap {
    heap: ID3D12Heap,
    /// The slots no buffer is in, a bit each.
    free: [u64; HEAP_SLOTS / WORD_SLOTS],
}

impl Heap {
    fn free_slots(&self) -> usize {
        self.free
            .iter()
            .map(|bits| bits.count_ones() as usize)
            .sum()
    }
}

/// What a placed buffer's destruction hands back: where its slot goes.
struct Home {
    pool: &'static Pool,
    slot: Slot,
}

impl Slots {
    /// The lowest free slot and its heap: low slots first, so the heaps
    /// after them empty when fewer buffers are alive. `None` when no slot is
    /// free or the pool is closed.
    fn take(&mut self) -> Option<(Slot, ID3D12Heap)> {
        if self.closed {
            return None;
        }
        self.heaps.iter_mut().enumerate().find_map(|(place, heap)| {
            let heap = heap.as_mut()?;
            let (word, bits) = heap
                .free
                .iter_mut()
                .enumerate()
                .find(|(_, bits)| **bits != 0)?;
            let bit = bits.trailing_zeros() as usize;
            *bits &= !(1 << bit);
            Some((
                Slot(place * HEAP_SLOTS + word * WORD_SLOTS + bit),
                heap.heap.clone(),
            ))
        })
    }

    /// A slot for a caller that made `heap` because none was free, and the
    /// heap back when it is not needed, for the caller to release. Threads
    /// that found no slot at the same moment each made a heap: the first
    /// adds its own, the others take its slots. A pool closed in between
    /// takes no heap.
    fn take_or_add(
        &mut self,
        heap: ID3D12Heap,
    ) -> (Option<(Slot, ID3D12Heap)>, Option<ID3D12Heap>) {
        match self.take() {
            Some(taken) => (Some(taken), Some(heap)),
            None if self.closed => (None, Some(heap)),
            None => {
                self.add(heap);
                (self.take(), None)
            }
        }
    }

    /// Adds `heap`, whose slots are all free, in the first place without a
    /// heap.
    fn add(&mut self, heap: ID3D12Heap) {
        let heap = Some(Heap {
            heap,
            free: [u64::MAX; HEAP_SLOTS / WORD_SLOTS],
        });
        match self.heaps.iter_mut().find(|place| place.is_none()) {
            Some(place) => *place = heap,
            None => self.heaps.push(heap),
        }
    }

    /// The buffer in `slot` is gone.
    fn give(&mut self, slot: Slot) {
        if let Some(Some(heap)) = self.heaps.get_mut(slot.heap())
            && let Some(bits) = heap.free.get_mut(slot.in_heap() / WORD_SLOTS)
        {
            *bits |= 1 << (slot.in_heap() % WORD_SLOTS);
        }
    }

    /// The free slots of every heap held.
    fn free_slots(&self) -> usize {
        self.heaps.iter().flatten().map(Heap::free_slots).sum()
    }

    /// Takes out a heap no buffer is in, the last such, for the caller to
    /// release. An open pool does so only while a heap's worth of slots stays
    /// free without it, so a pool that hovers around a heap's edge does not
    /// make and release one over and over; a closed pool makes no heap again,
    /// so it gives back each empty one. A placed buffer holds a reference of
    /// its own on its heap (the remarks of `ID3D12Device::CreateHeap`), so ours
    /// may go while a buffer whose destruction was reported is still torn down.
    fn spare(&mut self) -> Option<ID3D12Heap> {
        if !self.closed && self.free_slots() < 2 * HEAP_SLOTS {
            return None;
        }
        let empty = |heap: &Heap| heap.free_slots() == HEAP_SLOTS;
        let place = self
            .heaps
            .iter_mut()
            .rev()
            .find(|place| place.as_ref().is_some_and(empty))?;
        place.take().map(|heap| heap.heap)
    }
}

impl Pool {
    pub(crate) fn new(device: ID3D12Device) -> Pool {
        Pool {
            device,
            slots: Mutex::new(Slots {
                heaps: Vec::new(),
                closed: false,
            }),
        }
    }

    fn lock(&self) -> MutexGuard<'_, Slots> {
        self.slots.lock().unwrap_or_else(|e| e.into_inner())
    }

    /// A mapped [shared buffer](shared_buffer_desc) of `width_bytes` in a
    /// slot, zeroed over its width (at least one byte). The slot's bytes past
    /// the width keep what an earlier buffer left there, where a committed
    /// buffer's allocation reads zero in practice; a read past a buffer's
    /// width, which D3D12 leaves undefined, reaches them. `None` when that is
    /// wider than a slot or the pool is closed: the caller commits a buffer
    /// instead. Any thread; takes the pool's lock.
    pub(crate) fn buffer(&'static self, width_bytes: u64) -> Option<(ID3D12Resource, *mut u8)> {
        if width_bytes > SLOT_BYTES {
            return None;
        }
        let desc = shared_buffer_desc(width_bytes);
        let width = usize::try_from(desc.Width).ok()?;
        let (slot, heap) = self.take_or_grow()?;
        match self.place(&heap, slot, &desc) {
            Ok((buffer, data)) => {
                // SAFETY: the buffer's mapping, `width` bytes as the host
                // made it; nothing else has the buffer yet.
                unsafe { std::ptr::write_bytes(data, 0, width) };
                Some((buffer, data))
            }
            Err(e) => {
                self.close(&e);
                None
            }
        }
    }

    /// Releases one heap no buffer is in, if a heap's worth of free slots
    /// stays without it ([`Slots::spare`]); a closed pool releases any empty
    /// heap. Called by the frame meter once a second on the present thread,
    /// so a tick after a peak pays for one heap's release; the heap goes
    /// after the pool's lock.
    pub(crate) fn release_spare(&self) {
        let spare = self.lock().spare();
        drop(spare); // released outside the lock
    }

    /// A free slot and its heap, from a new heap when no slot is free;
    /// `None` when the pool is closed.
    fn take_or_grow(&self) -> Option<(Slot, ID3D12Heap)> {
        let (closed, taken) = {
            let mut slots = self.lock();
            (slots.closed, slots.take())
        };
        if closed || taken.is_some() {
            return taken;
        }
        let heap = match self.new_heap() {
            Ok(heap) => heap,
            Err(e) => {
                self.close(&e);
                return None;
            }
        };
        let (taken, unneeded) = self.lock().take_or_add(heap);
        drop(unneeded); // released outside the lock
        taken
    }

    /// A new heap of [`HEAP_SLOTS`] slots.
    fn new_heap(&self) -> Result<ID3D12Heap> {
        let desc = D3D12_HEAP_DESC {
            SizeInBytes: HEAP_BYTES,
            Properties: SHARED_HEAP,
            Alignment: 0,
            Flags: D3D12_HEAP_FLAG_ALLOW_ONLY_BUFFERS,
        };
        let mut heap: Option<ID3D12Heap> = None;
        // SAFETY: plain creation.
        unsafe { self.device.CreateHeap(&desc, &mut heap)? };
        // Counted over a scene, these show a pool that makes and releases a
        // heap over and over.
        tracing::debug!("small buffers: a heap made");
        heap.ok_or_else(windows::core::Error::empty)
    }

    /// The buffer `desc` describes at `slot` of `heap`, mapped, with the
    /// slot's return registered for its destruction. On an error the slot is
    /// free again: no buffer is in it.
    fn place(
        &'static self,
        heap: &ID3D12Heap,
        slot: Slot,
        desc: &D3D12_RESOURCE_DESC,
    ) -> Result<(ID3D12Resource, *mut u8)> {
        self.create_placed(heap, slot, desc)
            .inspect_err(|_| self.lock().give(slot))
    }

    /// [`Pool::place`] without the slot's return on an error: a buffer made
    /// before the error is destroyed with nothing registered for it.
    fn create_placed(
        &'static self,
        heap: &ID3D12Heap,
        slot: Slot,
        desc: &D3D12_RESOURCE_DESC,
    ) -> Result<(ID3D12Resource, *mut u8)> {
        let mut buffer: Option<ID3D12Resource> = None;
        // SAFETY: plain creation in a free slot of a live heap.
        unsafe {
            self.device.CreatePlacedResource(
                heap,
                slot.offset_bytes(),
                desc,
                D3D12_RESOURCE_STATE_COMMON,
                None,
                &mut buffer,
            )?;
        }
        let buffer = buffer.ok_or_else(windows::core::Error::empty)?;
        let mut data = std::ptr::null_mut();
        // SAFETY: mapping of a CPU-visible buffer.
        unsafe { buffer.Map(0, None, Some(&mut data))? };
        let notifier = buffer.cast::<ID3DDestructionNotifier>()?;
        let home = Box::into_raw(Box::new(Home { pool: self, slot }));
        // SAFETY: `destroyed` takes the box back, once, when the host
        // destroys the buffer.
        let registered =
            unsafe { notifier.RegisterDestructionCallback(Some(destroyed), home.cast()) };
        if let Err(e) = registered {
            // SAFETY: not registered, so the box is still ours alone.
            drop(unsafe { Box::from_raw(home) });
            return Err(e);
        }
        Ok((buffer, data.cast()))
    }

    /// Logs the heaps held and the buffers alive in them (diagnostics).
    #[cfg(feature = "diagnostics")]
    pub(crate) fn log(&self) {
        let (heaps, places, free) = {
            let slots = self.lock();
            let heaps = slots.heaps.iter().flatten().count();
            (heaps, slots.heaps.len(), slots.free_slots())
        };
        tracing::info!(
            "small buffers: {} alive in {heaps} heaps of {} MiB, {places} heaps at most",
            heaps * HEAP_SLOTS - free,
            HEAP_BYTES >> 20
        );
    }

    /// Closes the pool: one warning, and every later buffer is committed.
    fn close(&self, error: &windows::core::Error) {
        if !std::mem::replace(&mut self.lock().closed, true) {
            tracing::warn!(%error, "the host does not place small buffers: they are committed from now on");
        }
    }
}

/// The host destroyed a placed buffer (its last reference went, on any
/// thread): its slot is free. Takes the pool's lock. Never panics.
unsafe extern "system" fn destroyed(home: *mut c_void) {
    // SAFETY: the box `Pool::place` registered for this buffer, handed back
    // once.
    let home = unsafe { Box::from_raw(home.cast::<Home>()) };
    home.pool.lock().give(home.slot);
}

/// These rely on what the D3D12 runtime does, which the documentation of
/// `ID3DDestructionNotifier` does not state: the callback runs inside the
/// buffer's final release, on the thread that drops it.
#[cfg(test)]
mod tests {
    use super::*;

    /// A pool of its own for one test, on the host's device.
    fn pool() -> Option<&'static Pool> {
        let host = crate::host::host()?;
        Some(Box::leak(Box::new(Pool::new(host.device.clone()))))
    }

    fn heaps(pool: &Pool) -> usize {
        pool.lock().heaps.iter().flatten().count()
    }

    #[test]
    fn a_slot_is_a_heap_and_an_offset_in_it() {
        assert_eq!((Slot(0).heap(), Slot(0).offset_bytes()), (0, 0));
        assert_eq!((Slot(255).heap(), Slot(255).offset_bytes()), (0, 0xff_0000));
        assert_eq!((Slot(257).heap(), Slot(257).offset_bytes()), (1, 0x1_0000));
    }

    #[test]
    fn a_destroyed_buffer_frees_its_slot_and_the_next_buffer_there_reads_zero() {
        let Some(pool) = pool() else {
            return; // no host GPU here
        };
        let (first, data) = pool.buffer(SLOT_BYTES).unwrap();
        // SAFETY: a live buffer.
        let address = unsafe { first.GetGPUVirtualAddress() };
        // SAFETY: the buffer's mapping.
        unsafe { std::ptr::write_bytes(data, 0xAA, SLOT_BYTES as usize) };

        drop(first);
        let (second, data) = pool.buffer(SLOT_BYTES).unwrap();

        // SAFETY: a live buffer.
        assert_eq!(unsafe { second.GetGPUVirtualAddress() }, address);
        // SAFETY: the buffer's mapping.
        let bytes = unsafe { std::slice::from_raw_parts(data, SLOT_BYTES as usize) };
        assert!(bytes.iter().all(|&b| b == 0));
    }

    #[test]
    fn a_buffer_of_no_bytes_reads_zero_in_its_one_byte() {
        let Some(pool) = pool() else {
            return; // no host GPU here
        };
        let (first, data) = pool.buffer(1).unwrap();
        // SAFETY: the buffer's mapping.
        unsafe { data.write(0xAA) };

        drop(first);
        let (_second, data) = pool.buffer(0).unwrap();

        // SAFETY: the buffer's mapping: a byte, as the host makes it.
        assert_eq!(unsafe { data.read() }, 0);
    }

    #[test]
    fn a_buffer_wider_than_a_slot_is_not_placed() {
        let Some(pool) = pool() else {
            return; // no host GPU here
        };

        assert!(pool.buffer(SLOT_BYTES + 1).is_none());
        assert_eq!(heaps(pool), 0);
    }

    #[test]
    fn one_empty_heap_is_released_a_tick() {
        let Some(pool) = pool() else {
            return; // no host GPU here
        };
        let three_heaps: Vec<_> = (0..2 * HEAP_SLOTS + 1)
            .map(|_| pool.buffer(256).unwrap().0)
            .collect();
        drop(three_heaps);

        pool.release_spare();
        assert_eq!(heaps(pool), 2);
        pool.release_spare();
        assert_eq!(heaps(pool), 1);
    }

    #[test]
    fn an_empty_heap_stays_unless_a_heaps_worth_of_slots_stays_free_without_it() {
        let Some(pool) = pool() else {
            return; // no host GPU here
        };
        let mut two_heaps: Vec<_> = (0..HEAP_SLOTS + 1)
            .map(|_| pool.buffer(256).unwrap().0)
            .collect();
        let _in_the_first_heap = two_heaps.swap_remove(0);
        drop(two_heaps);

        pool.release_spare();

        assert_eq!(heaps(pool), 2, "255 slots would stay free without it");
    }

    /// Three heaps with the middle one released: what is left is a buffer
    /// in the first heap and a buffer in the last.
    fn a_released_middle_heap(pool: &'static Pool) -> [ID3D12Resource; 2] {
        let buffer = || pool.buffer(256).unwrap().0;
        let mut three_heaps: Vec<_> = (0..2 * HEAP_SLOTS + 1).map(|_| buffer()).collect();
        let in_the_last_heap = three_heaps.pop().unwrap();
        drop(three_heaps);
        pool.release_spare();
        [buffer(), in_the_last_heap]
    }

    #[test]
    fn an_empty_heap_below_one_in_use_is_released() {
        let Some(pool) = pool() else {
            return; // no host GPU here
        };

        let _two = a_released_middle_heap(pool);

        let slots = pool.lock();
        let places: Vec<_> = slots.heaps.iter().map(Option::is_some).collect();
        assert_eq!(places, [true, false, true]);
    }

    #[test]
    fn a_heap_in_a_released_heaps_place_gets_that_places_slots() {
        let Some(pool) = pool() else {
            return; // no host GPU here
        };
        let mut buffers = Vec::from(a_released_middle_heap(pool));

        // The two heaps left hold 512 buffers; the next one needs a third.
        buffers.extend((0..2 * HEAP_SLOTS - 1).map(|_| pool.buffer(256).unwrap().0));

        assert_eq!((heaps(pool), pool.lock().heaps.len()), (3, 3));
        let mut addresses: Vec<_> = buffers
            .iter()
            // SAFETY: a live buffer.
            .map(|buffer| unsafe { buffer.GetGPUVirtualAddress() })
            .collect();
        addresses.sort_unstable();
        addresses.dedup();
        assert_eq!(addresses.len(), 2 * HEAP_SLOTS + 1, "a slot each");
        drop(buffers);
        let free: Vec<_> = pool
            .lock()
            .heaps
            .iter()
            .flatten()
            .map(Heap::free_slots)
            .collect();
        assert_eq!(free, [HEAP_SLOTS; 3], "each back in its own heap");
    }

    #[test]
    fn a_heap_made_while_another_thread_added_one_is_handed_back() {
        let Some(pool) = pool() else {
            return; // no host GPU here
        };
        let (first, second) = (pool.new_heap().unwrap(), pool.new_heap().unwrap());
        let mut slots = pool.lock();
        slots.add(first);

        let (taken, unneeded) = slots.take_or_add(second);

        assert!(taken.is_some() && unneeded.is_some());
        assert_eq!(slots.heaps.len(), 1);
    }

    #[test]
    fn a_heap_made_while_the_pool_closed_is_handed_back() {
        let Some(pool) = pool() else {
            return; // no host GPU here
        };
        let heap = pool.new_heap().unwrap();
        pool.close(&windows::core::Error::empty());

        let (taken, unneeded) = pool.lock().take_or_add(heap);

        assert!(taken.is_none() && unneeded.is_some());
        assert_eq!(heaps(pool), 0);
    }

    #[test]
    fn a_closed_pool_hands_out_no_slot() {
        let Some(pool) = pool() else {
            return; // no host GPU here
        };
        let heap = pool.new_heap().unwrap();
        pool.lock().add(heap);
        pool.close(&windows::core::Error::empty());

        let taken = pool.lock().take();

        assert!(taken.is_none());
    }

    #[test]
    fn a_slot_the_host_refuses_a_buffer_in_is_free_again() {
        let Some(pool) = pool() else {
            return; // no host GPU here
        };
        let (slot, heap) = pool.take_or_grow().unwrap();
        // Wider than the whole heap: the host refuses it.
        let refused = D3D12_RESOURCE_DESC {
            Width: (HEAP_SLOTS as u64 + 1) * SLOT_BYTES,
            ..shared_buffer_desc(256)
        };

        let placed = pool.place(&heap, slot, &refused);

        assert!(placed.is_err());
        let slots = pool.lock();
        let free: Vec<_> = slots.heaps.iter().flatten().map(Heap::free_slots).collect();
        assert_eq!(free, [HEAP_SLOTS]);
    }

    #[test]
    fn a_closed_pool_places_nothing_and_releases_every_empty_heap() {
        let Some(pool) = pool() else {
            return; // no host GPU here
        };
        let mut three_heaps: Vec<_> = (0..2 * HEAP_SLOTS + 1)
            .map(|_| pool.buffer(256).unwrap().0)
            .collect();
        let _in_the_first_heap = three_heaps.swap_remove(0);
        pool.close(&windows::core::Error::empty());
        drop(three_heaps);
        pool.release_spare();
        pool.release_spare();

        let placed = pool.buffer(256);

        assert!(placed.is_none());
        let slots = pool.lock();
        let places: Vec<_> = slots.heaps.iter().map(Option::is_some).collect();
        assert_eq!(places, [true, false, false]);
    }

    /// Through `take_or_grow` and its locks. Whether two threads find no
    /// slot before a heap is added depends on timing; that case is pinned
    /// by `a_heap_made_while_another_thread_added_one_is_handed_back`.
    #[test]
    fn threads_that_place_buffers_at_once_leave_one_heap() {
        let Some(pool) = pool() else {
            return; // no host GPU here
        };
        let together = std::sync::Barrier::new(8);

        let buffers: Vec<_> = std::thread::scope(|scope| {
            let threads: Vec<_> = (0..8)
                .map(|_| {
                    scope.spawn(|| {
                        together.wait();
                        pool.buffer(256).unwrap().0
                    })
                })
                .collect();
            threads
                .into_iter()
                .map(|thread| thread.join().unwrap())
                .collect()
        });

        assert_eq!((buffers.len(), heaps(pool)), (8, 1));
    }
}
