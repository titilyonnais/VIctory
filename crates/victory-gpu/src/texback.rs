//! Textures shaders write that the title's own code then reads on the CPU.
//! On the console the CPU and the GPU share memory, so what a compute pass
//! stores into a texture is in title memory for the CPU to read. Here it stays
//! in the texture's host copy. The title's ocean/water simulation is one: its
//! compute passes write compressed min/max planes and composite height and
//! displacement textures that its CPU buoyancy code samples, which is what
//! boats float on; without copy-back it reads unwritten memory and boats sink.
//!
//! For textures of the kinds in `VICTORY_TEXTURE_BACK` (comma-separated
//! `<width>x<height>f<format>`; `0` or empty for none; by default the kinds
//! the title's CPU code reads), a list in which shaders write one through
//! unordered access copies it into a read-back buffer when the list closes,
//! and once the submission has run the texels are tiled into the texture's
//! title memory in the console's layout ([`victory_formats::texture::Layout::tile`]).
//!
//! Based on `Geckomacabre/vx-layer` (`eee5667d`).
//!
//! shortcut: by kind, not by texture: textures of those kinds the title never
//! reads are copied back too (about a megabyte per such list). The copy shows
//! what the list's last write left; a title read in between sees the frame
//! before, as it would before the console's GPU has run.

use std::sync::OnceLock;

use windows::Win32::Graphics::Direct3D12::*;

use crate::object::This;

const DEFAULT_KINDS: &str = "256x256f41,256x256f16,64x64f16,128x128f36";

/// Parses `VICTORY_TEXTURE_BACK` (`<width>x<height>f<format>`, comma-separated;
/// `"0"` or empty disables copy-back).
fn parse_kinds(spec: &str) -> Vec<(u64, u32, u32)> {
    let trimmed = spec.trim();
    if trimmed.is_empty() || trimmed == "0" {
        return Vec::new();
    }
    trimmed
        .split(',')
        .filter_map(|k| {
            let (size, format) = k.trim().split_once('f')?;
            let (w, h) = size.split_once('x')?;
            Some((w.trim().parse().ok()?, h.trim().parse().ok()?, format.trim().parse().ok()?))
        })
        .collect()
}

/// The kinds copied back: (width, height, format).
fn kinds() -> &'static [(u64, u32, u32)] {
    static KINDS: OnceLock<Vec<(u64, u32, u32)>> = OnceLock::new();
    KINDS.get_or_init(|| {
        let list = std::env::var("VICTORY_TEXTURE_BACK").unwrap_or_else(|_| DEFAULT_KINDS.to_owned());
        let kinds = parse_kinds(&list);
        if !kinds.is_empty() {
            tracing::info!(
                ?kinds,
                "textures of these kinds that shaders write are copied back into title memory (VICTORY_TEXTURE_BACK)"
            );
        }
        kinds
    })
}

/// Whether title texture `resource` is of a kind copied back.
pub(crate) fn wanted(resource: This) -> bool {
    let kinds = kinds();
    !kinds.is_empty()
        && crate::resource::texture_kind(resource).is_some_and(|k| kinds.contains(&k))
}

/// A texture's copy into a read-back buffer, recorded into a list, and where
/// its texels go once the list has run.
pub(crate) struct Back {
    buffer: ID3D12Resource,
    footprints: Vec<D3D12_PLACED_SUBRESOURCE_FOOTPRINT>,
    rows: Vec<u32>,
    mips: u32,
    layout: victory_formats::texture::Layout,
    /// The texture's title memory and its layout's bytes.
    at: usize,
    size: usize,
}

/// Records into `list` the copy of `texture` (title resource `resource`'s
/// host copy, in the copy-source state) into a read-back buffer.
pub(crate) fn record(
    list: &ID3D12GraphicsCommandList,
    resource: This,
    texture: &ID3D12Resource,
) -> Option<Back> {
    let host = crate::host::host()?;
    let (layout, at, size) = crate::resource::title_texture(resource)?;
    // SAFETY: a live texture.
    let desc = unsafe { texture.GetDesc() };
    let mips = u32::from(desc.MipLevels.max(1));
    let subresources = mips * u32::from(desc.DepthOrArraySize.max(1));
    let n = subresources as usize;
    let mut footprints = vec![D3D12_PLACED_SUBRESOURCE_FOOTPRINT::default(); n];
    let mut rows = vec![0u32; n];
    let mut total = 0u64;
    // SAFETY: out arrays of `n` entries.
    unsafe {
        host.device.GetCopyableFootprints(
            &desc,
            0,
            subresources,
            0,
            Some(footprints.as_mut_ptr()),
            Some(rows.as_mut_ptr()),
            None,
            Some(&mut total),
        )
    };
    let buffer = host.read_back_buffer(total)?;
    for (i, footprint) in footprints.iter().enumerate() {
        let to = D3D12_TEXTURE_COPY_LOCATION {
            // SAFETY: a borrowed pointer the location never releases.
            pResource: unsafe { std::mem::transmute_copy(&buffer) },
            Type: D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT,
            Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 {
                PlacedFootprint: *footprint,
            },
        };
        let from = D3D12_TEXTURE_COPY_LOCATION {
            // SAFETY: as above.
            pResource: unsafe { std::mem::transmute_copy(texture) },
            Type: D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX,
            Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 {
                SubresourceIndex: i as u32,
            },
        };
        // SAFETY: an open list; the texture is in the copy-source state.
        unsafe { list.CopyTextureRegion(&to, 0, 0, 0, &from, None) };
    }
    COPIED.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    Some(Back {
        buffer,
        footprints,
        rows,
        mips,
        layout,
        at,
        size,
    })
}

/// Textures copied back since the start.
pub(crate) static COPIED: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

/// Once the list that recorded `back` has run: its texels, tiled into the
/// title's layout, into the texture's title memory.
pub(crate) fn land(back: &Back) {
    let mut tiled = vec![0u8; back.size];
    // Bytes of title memory the subresources do not cover stay as they are.
    crate::host::copy_mapped(tiled.as_mut_ptr(), back.at as *const u8, back.size, false);
    let mut data = std::ptr::null_mut();
    // SAFETY: our read-back buffer, written by the list that has run.
    unsafe {
        if back.buffer.Map(0, None, Some(&mut data)).is_ok() && !data.is_null() {
            for (i, (footprint, &rows)) in back.footprints.iter().zip(&back.rows).enumerate() {
                let (mip, slice) = (i as u32 % back.mips, i as u32 / back.mips);
                let f = footprint.Footprint;
                let pitch = f.RowPitch as usize;
                let len = pitch * rows as usize * f.Depth as usize;
                // SAFETY: the subresource's part of the mapped buffer.
                let linear = std::slice::from_raw_parts(
                    data.cast::<u8>().add(footprint.Offset as usize),
                    len,
                );
                if let Err(e) =
                    back.layout
                        .tile(&mut tiled, mip, slice, linear, pitch, pitch * rows as usize)
                {
                    static ONCE: std::sync::Once = std::sync::Once::new();
                    ONCE.call_once(|| {
                        tracing::warn!(?e, mip, slice, "a texture copied back could not be tiled")
                    });
                    back.buffer.Unmap(0, None);
                    return;
                }
            }
            back.buffer.Unmap(0, None);
        } else {
            return;
        }
    }
    crate::host::copy_mapped(back.at as *mut u8, tiled.as_ptr(), back.size, true);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_kinds_handles_default_disable_and_custom_lists() {
        assert_eq!(
            parse_kinds(DEFAULT_KINDS),
            vec![(256, 256, 41), (256, 256, 16), (64, 64, 16), (128, 128, 36)]
        );
        assert!(parse_kinds("0").is_empty());
        assert!(parse_kinds("  ").is_empty());
        assert_eq!(
            parse_kinds(" 512x512f41 , bad, 128x64f2 "),
            vec![(512, 512, 41), (128, 64, 2)]
        );
    }
}
