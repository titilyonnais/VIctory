//! Host work that a host library records into a title's command list in
//! place of a console one (the host's FidelityFX upscaler, say): the host
//! device, a title texture's host copy in the state the library is told,
//! and the host list itself.

use windows::Win32::Graphics::Direct3D12::{
    ID3D12Device, ID3D12GraphicsCommandList, ID3D12Resource,
};

use crate::object::This;

/// The host D3D12 device, created on first use; `None` without a host GPU.
pub fn host_device() -> Option<ID3D12Device> {
    crate::host::host().map(|h| h.device.clone())
}

/// The active host list for one synchronous external recording call.
pub struct Recording<'a>(&'a crate::host::HostList);

impl Recording<'_> {
    /// Move a title texture to the state the provider will see. The list
    /// holds the backing resource through submission.
    pub fn texture(&self, resource: This, state: u32) -> Option<(ID3D12Resource, u32)> {
        let texture = crate::resource::host_resource(resource)?.clone();
        crate::list::ensure_state(self.0, resource, state);
        let state = crate::host::legal_state(&texture, state);
        Some((texture, state))
    }

    /// Prepare aliases, mip uploads and depth twins like a compute SRV.
    pub fn prepare_input(&self, resource: This) -> Option<()> {
        let texture = crate::resource::host_resource(resource)?;
        let aliased = crate::resource::alias_of(resource)
            .is_some_and(|source| crate::texture::alias(self.0, resource, source));
        if !aliased && crate::texture::title_written(texture) {
            crate::texture::upload_from(self.0, resource, 0);
        }
        self.0.twin_use(resource, false, false);
        Some(())
    }

    /// Access flushes any pending host barriers before the provider runs.
    pub fn commands(&self) -> &ID3D12GraphicsCommandList {
        &self.0.list
    }
}

/// The producer recorded its output on the host's command list.
pub fn output_written(resource: This) {
    crate::texture::host_written(resource);
}

/// Settle submitted title work before releasing a provider's private GPU resources.
pub fn finish_submitted() {
    crate::queue::before_sync();
    crate::host::settle_for_fence();
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

/// Turns on AMD frame generation for upscaled frames with
/// `amd_fidelityfx_framegeneration_dx12.dll` from `dir`: the presenter's
/// swap chain becomes its proxy. The DLL's path, or why not. Call before
/// the swap chain is made.
pub fn use_frame_generation(dir: &std::path::Path, ml: bool) -> Result<String, String> {
    crate::framegen::load(dir, ml)
}

impl Recording<'_> {
    /// At a host upscale recorded here: its depth and motion textures (title
    /// resources) and constants feed frame generation for the next present,
    /// once the list has run. Nothing when it is off.
    pub fn frame_generation_inputs(&self, depth: This, motion: This, frame: FrameInputs) {
        if !crate::framegen::enabled() {
            return;
        }
        let (Some(depth), Some(motion)) = (crate::resource::host_copy(depth), crate::resource::host_copy(motion))
        else {
            return;
        };
        crate::framegen::record(self.0, crate::framegen::Inputs { depth, motion, frame, pose: None });
    }
}

/// Runs `f` with the host list behind the title list `list` (open for
/// recording); `None` without one. What `f` binds (descriptor heaps, root
/// signatures, pipeline) is replaced before the list's next draw or
/// dispatch.
pub fn record<R>(list: This, f: impl FnOnce(&Recording<'_>) -> R) -> Option<R> {
    let list = crate::list::thin::for_external(list)?;
    let mut out = None;
    crate::list::with_state(list, |host, state| {
        out = Some(f(&Recording(host)));
        state.external_compute();
    });
    out
}
