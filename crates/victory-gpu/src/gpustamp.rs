//! Diagnostics: how long the GPU is busy with a submission, as opposed to
//! the wait around it (the "on the GPU" part of the frame-rate line counts
//! both). Each submission's lists run between two small lists of ours, one
//! writing a timestamp before them, one after them and resolving both into
//! a read-back buffer; submissions are waited for one at a time, so the two
//! lists are recorded once and run again for each.

use std::sync::OnceLock;

use windows::Win32::Graphics::Direct3D12::*;
use windows::Win32::Graphics::Dxgi::Common::DXGI_SAMPLE_DESC;
use windows::core::Interface;

pub(crate) struct Stamper {
    /// Writes timestamp 0.
    begin: ID3D12CommandList,
    /// Writes timestamp 1 and resolves both into `back`.
    end: ID3D12CommandList,
    back: ID3D12Resource,
    // Kept alive while the lists may run.
    _heap: ID3D12QueryHeap,
    _allocators: [ID3D12CommandAllocator; 2],
}

// SAFETY: D3D12 objects are free-threaded; the lists run under the
// submission lock, one submission at a time.
unsafe impl Send for Stamper {}
// SAFETY: as above.
unsafe impl Sync for Stamper {}

/// Whether submissions are timestamped (`VICTORY_GPU_STAMPS=0` turns it
/// off).
fn on() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| std::env::var("VICTORY_GPU_STAMPS").map_or(true, |v| v != "0"))
}

/// The stamper of `device`'s submissions, made on first use; `None` when
/// off or when its objects could not be made.
pub(crate) fn stamper(device: &ID3D12Device) -> Option<&'static Stamper> {
    static STAMPER: OnceLock<Option<Stamper>> = OnceLock::new();
    if !on() {
        return None;
    }
    STAMPER
        .get_or_init(|| match Stamper::new(device) {
            Ok(s) => Some(s),
            Err(e) => {
                tracing::warn!(error = %e, "no GPU timestamps for submissions");
                None
            }
        })
        .as_ref()
}

impl Stamper {
    fn new(device: &ID3D12Device) -> windows::core::Result<Stamper> {
        // SAFETY: plain object creation with valid descriptions, and
        // recording into lists we just made.
        unsafe {
            let mut heap: Option<ID3D12QueryHeap> = None;
            device.CreateQueryHeap(
                &D3D12_QUERY_HEAP_DESC {
                    Type: D3D12_QUERY_HEAP_TYPE_TIMESTAMP,
                    Count: 2,
                    NodeMask: 0,
                },
                &mut heap,
            )?;
            let heap = heap.ok_or_else(windows::core::Error::empty)?;
            let mut back: Option<ID3D12Resource> = None;
            device.CreateCommittedResource(
                &D3D12_HEAP_PROPERTIES {
                    Type: D3D12_HEAP_TYPE_READBACK,
                    ..Default::default()
                },
                D3D12_HEAP_FLAG_NONE,
                &D3D12_RESOURCE_DESC {
                    Dimension: D3D12_RESOURCE_DIMENSION_BUFFER,
                    Width: 16,
                    Height: 1,
                    DepthOrArraySize: 1,
                    MipLevels: 1,
                    SampleDesc: DXGI_SAMPLE_DESC {
                        Count: 1,
                        Quality: 0,
                    },
                    Layout: D3D12_TEXTURE_LAYOUT_ROW_MAJOR,
                    ..Default::default()
                },
                D3D12_RESOURCE_STATE_COPY_DEST,
                None,
                &mut back,
            )?;
            let back = back.ok_or_else(windows::core::Error::empty)?;
            let list = |record: &dyn Fn(&ID3D12GraphicsCommandList)| -> windows::core::Result<_> {
                let allocator: ID3D12CommandAllocator =
                    device.CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT)?;
                let list: ID3D12GraphicsCommandList = device.CreateCommandList(
                    0,
                    D3D12_COMMAND_LIST_TYPE_DIRECT,
                    &allocator,
                    None,
                )?;
                record(&list);
                list.Close()?;
                Ok((allocator, list.cast::<ID3D12CommandList>()?))
            };
            let (a0, begin) = list(&|l| l.EndQuery(&heap, D3D12_QUERY_TYPE_TIMESTAMP, 0))?;
            let (a1, end) = list(&|l| {
                l.EndQuery(&heap, D3D12_QUERY_TYPE_TIMESTAMP, 1);
                l.ResolveQueryData(&heap, D3D12_QUERY_TYPE_TIMESTAMP, 0, 2, &back, 0);
            })?;
            Ok(Stamper {
                begin,
                end,
                back,
                _heap: heap,
                _allocators: [a0, a1],
            })
        }
    }

    /// `group` between the timestamp lists.
    pub(crate) fn around(
        &self,
        group: &[Option<ID3D12CommandList>],
    ) -> Vec<Option<ID3D12CommandList>> {
        let mut lists = Vec::with_capacity(group.len() + 2);
        lists.push(Some(self.begin.clone()));
        lists.extend_from_slice(group);
        lists.push(Some(self.end.clone()));
        lists
    }

    /// After the stamped submission finished: microseconds between its two
    /// timestamps (`frequency` ticks per second), if they read sensibly.
    pub(crate) fn busy_us(&self, frequency: u64) -> Option<u64> {
        let mut data = std::ptr::null_mut();
        let range = D3D12_RANGE { Begin: 0, End: 16 };
        // SAFETY: a read-back buffer the finished submission resolved into.
        let [begin, end] = unsafe {
            self.back.Map(0, Some(&range), Some(&mut data)).ok()?;
            let stamps = std::ptr::read_unaligned(data.cast::<[u64; 2]>());
            self.back.Unmap(0, Some(&D3D12_RANGE { Begin: 0, End: 0 }));
            stamps
        };
        ticks_to_us(begin, end, frequency)
    }
}

/// Microseconds from timestamp `begin` to `end` at `frequency` ticks per
/// second; `None` for no frequency or stamps out of order (a reset
/// counter), and for spans over a minute (stamps never written).
pub(crate) fn ticks_to_us(begin: u64, end: u64, frequency: u64) -> Option<u64> {
    if frequency == 0 || end < begin {
        return None;
    }
    let us = u128::from(end - begin) * 1_000_000 / u128::from(frequency);
    u64::try_from(us).ok().filter(|&us| us <= 60_000_000)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ticks_become_microseconds_and_nonsense_is_dropped() {
        assert_eq!(ticks_to_us(1_000, 3_000, 1_000_000), Some(2_000));
        assert_eq!(ticks_to_us(0, 25_000_000, 100_000_000), Some(250_000));
        // Large counters do not overflow.
        assert_eq!(ticks_to_us(u64::MAX - 10, u64::MAX, 10), Some(1_000_000));
        assert_eq!(ticks_to_us(5, 5, 1_000), Some(0));
        assert_eq!(ticks_to_us(10, 5, 1_000), None, "out of order");
        assert_eq!(ticks_to_us(0, 10, 0), None, "no frequency");
        assert_eq!(
            ticks_to_us(0, u64::MAX, 1_000_000_000),
            None,
            "never written"
        );
    }

    #[test]
    fn a_stamped_submission_reads_its_busy_time() {
        let Some(host) = crate::host::host() else {
            return;
        };
        let Some(stamper) = stamper(&host.device) else {
            return;
        };
        let list = host.new_list().unwrap();
        list.close();
        let before = crate::copystat::count(crate::copystat::C::GpuTimed);
        for _ in 0..3 {
            host.execute(&[&list]);
            list.reset();
            list.close();
        }
        assert!(
            crate::copystat::count(crate::copystat::C::GpuTimed) >= before + 3,
            "each submission is timed, the stamp lists run again each time"
        );
        // An empty submission is short: well under a second.
        let busy = stamper
            .busy_us(host.timestamp_frequency())
            .expect("stamps read back");
        assert!(busy < 1_000_000, "{busy} us");
        assert_eq!(stamper.around(&[None]).len(), 3);
    }
}
