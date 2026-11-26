//! Host work that a host library records into a title's command list in
//! place of a console one (the host's FidelityFX upscaler, say): the host
//! device, a title texture's host copy in the state the library is told,
//! and the host list itself.

use std::sync::atomic::{AtomicBool, Ordering};

use windows::Win32::Graphics::Direct3D12::{
    ID3D12Device, ID3D12GraphicsCommandList, ID3D12Resource,
};

use crate::object::Object;

static HOST_FIDELITYFX: AtomicBool = AtomicBool::new(false);

/// Enables compatibility behavior for a successfully loaded host upscaler.
pub fn enable_host_fidelityfx() {
    HOST_FIDELITYFX.store(true, Ordering::Release);
}

pub(crate) fn host_fidelityfx_enabled() -> bool {
    HOST_FIDELITYFX.load(Ordering::Acquire)
}

/// The host D3D12 device, created on first use; `None` without a host GPU.
pub fn host_device() -> Option<ID3D12Device> {
    crate::host::host().map(|h| h.device.clone())
}

/// A title texture's host copy, moved to the host resource `state` in the
/// title list `list`, and the state it is in: `state` without the bits the
/// host copy cannot take (see `host::legal_state`). What the list records
/// next finds it there and must leave it there. The list holds the texture
/// until it is reset. `None` for a buffer, or without a host list.
pub fn host_texture(
    list: *const Object,
    resource: *const Object,
    state: u32,
) -> Option<(ID3D12Resource, u32)> {
    let host = crate::list::host_list(list)?;
    let texture = crate::resource::host_resource(resource)?.clone();
    crate::list::ensure_state(host, resource, state);
    let state = crate::host::legal_state(&texture, state);
    Some((texture, state))
}

/// Runs `f` with the host list behind the title list `list` (open for
/// recording); `None` without one. What `f` binds (descriptor heaps, root
/// signatures, pipeline) is replaced before the list's next draw or
/// dispatch.
pub fn record<R>(
    list: *const Object,
    f: impl FnOnce(&ID3D12GraphicsCommandList) -> R,
) -> Option<R> {
    let mut out = None;
    crate::list::with_state(list, |host, state| {
        out = Some(f(&host.list));
        state.host_state_lost();
    });
    out
}

/// What frame generation needs of an upscaled frame besides its depth and
/// motion vectors: the upscale's constants (as its description has them)
/// and the motion and depth bits of its creation flags.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct FrameInputs {
    pub render: [u32; 2],
    pub jitter: [f32; 2],
    pub motion_scale: [f32; 2],
    pub frame_time_ms: f32,
    pub reset: bool,
    /// Near, far, vertical field of view, metres per view-space unit.
    pub camera: [f32; 4],
    pub flags: u32,
    /// The FidelityFX descriptions of the depth and motion textures.
    pub depth_desc: [u32; 8],
    pub motion_desc: [u32; 8],
}

/// Turns on AMD frame generation for upscaled frames (diagnostics) with
/// `amd_fidelityfx_framegeneration_dx12.dll` from `dir`: the presenter's
/// swap chain becomes its proxy. The DLL's path, or why not.
#[cfg(feature = "diagnostics")]
pub fn use_frame_generation(dir: &std::path::Path, ml: bool) -> Result<String, String> {
    crate::framegen::load(dir, ml)
}

/// At a host upscale recorded into the title list `list`: its depth and
/// motion textures (title resources) and constants feed frame generation
/// for the next present, once the list has run. Nothing when it is off.
#[cfg(feature = "diagnostics")]
pub fn frame_generation_inputs(list: *const Object, depth: *const Object, motion: *const Object, frame: FrameInputs) {
    if !crate::framegen::enabled() {
        return;
    }
    let (Some(host), Some(depth), Some(motion)) = (
        crate::list::host_list(list),
        crate::resource::host_copy(depth),
        crate::resource::host_copy(motion),
    ) else {
        return;
    };
    crate::framegen::record(host, crate::framegen::Inputs { depth, motion, frame, pose: None });
}
