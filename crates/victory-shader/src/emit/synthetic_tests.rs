//! Synthetic GFX10 programs of what the 2020 compiler does not produce,
//! lowered to DXIL, validated by `dxil.dll` when present, and run on the
//! host GPU and WARP. Gathers at a level (`image_gather4_l`, `_l_o`), which
//! DXIL lacks: at level 0 the host's own gather (`image_gather4_lz`) of the
//! same coordinates is the reference on the hardware adapter (RDNA2, as the
//! console): it rounds the exact product of coordinate and size to 1/256 of
//! a texel, half up, which WARP does not. Lane permutes (`ds_bpermute_b32`).

use super::host::{Dispatch, Host, Root, Stride, View};
use super::{HostShader, lower, validator};
use crate::native::{
    ComputeIo, HwStage, NativeProgram, ResourceType, RootParam, RootType, StageIo, UsedResource,
};
use crate::{io, ir, resource, structure};

/// Per lane (32, wave32): the record [offset word, u, v, level] at entry 1
/// of the table at s0 (a 2D texture at entry 0, a raw UAV at entry 2); the
/// level converted from an integer (so it is whole). Gathers red at the
/// level, with the offset, and at level 0 (`lz`) into 12 dwords at lane × 48.
fn program(whole: bool) -> Vec<u32> {
    let mut w = vec![
        0xBE84_0300, // s_mov_b32 s4, s0
        0xBE85_0380, // s_mov_b32 s5, 0
        0xF40C_0302,
        0xFA00_0000, // s_load_dwordx8 s[12:19], s[4:5], 0x0
        0xF408_0502,
        0xFA00_0020, // s_load_dwordx4 s[20:23], s[4:5], 0x20
        0xF408_0A02,
        0xFA00_0040, // s_load_dwordx4 s[40:43], s[4:5], 0x40
        // The culling shaders' sampler: border, transparent black, levels
        // 0-12, linear filters.
        0xB008_01B6, // s_movk_i32 s8, 0x1b6
        0xBE89_03FF,
        0x00C0_0000, // s_mov_b32 s9, 0xc00000
        0xBE8A_03FF,
        0x1850_0000, // s_mov_b32 s10, 0x18500000
        0xBE8B_0380, // s_mov_b32 s11, 0
        0x3420_0084, // v_lshlrev_b32 v16, 4, v0
        0x1628_00B0, // v_mul_u32_u24 v20, 48, v0
        0xBF8C_C07F, // s_waitcnt lgkmcnt(0)
        0xE038_1000,
        0x8005_0C10, // buffer_load_dwordx4 v[12:15], v16, s[20:23], 0 offen
        0xBF8C_3F70, // s_waitcnt vmcnt(0)
    ];
    if whole {
        w.push(0x7E1E_0D0F); // v_cvt_f32_u32 v15, v15
    }
    w.extend([
        0xF110_0108,
        0x0043_040D, // image_gather4_l v[4:7], v[13:15], s[12:19], s[8:11]
        0xF150_0108,
        0x0043_080C, // image_gather4_l_o v[8:11], v[12:15], s[12:19], s[8:11]
        0xF11C_0108,
        0x0043_100D, // image_gather4_lz v[16:19], v[13:14], s[12:19], s[8:11]
        0xBF8C_3F70, // s_waitcnt vmcnt(0)
        0xE078_1000,
        0x800A_0414, // buffer_store_dwordx4 v[4:7], v20, s[40:43], 0 offen
        0xE078_1010,
        0x800A_0814, // buffer_store_dwordx4 v[8:11], v20, s[40:43], 0 offen offset:16
        0xE078_1020,
        0x800A_1014, // buffer_store_dwordx4 v[16:19], v20, s[40:43], 0 offen offset:32
        0xBF81_0000, // s_endpgm
    ]);
    w
}

fn build(words: &[u32]) -> Result<HostShader, crate::ShaderError> {
    let range = |kind, table_offset_dwords, dwords, stride| UsedResource {
        kind,
        root: RootType::Table,
        table_offset_dwords,
        lower_bound: 0,
        stride,
        dwords,
        user_data: 0,
        slot: 0,
    };
    let native = NativeProgram {
        stage: HwStage::Cs,
        wave64: false,
        vgprs: 32,
        layout: vec![RootParam {
            root: RootType::Table,
            table_offset_dwords: 0,
            sgprs: 1,
            base_register: 0,
            user_data: 0,
            slot: 0,
        }],
        used: vec![
            range(ResourceType::Srv, 0, 8, 0),
            range(ResourceType::Srv, 8, 4, 16),
            range(ResourceType::Uav, 16, 4, 0),
        ],
        io: StageIo::Cs(ComputeIo {
            rsrc2: 1 << 1,
            threads: [32, 1, 1],
        }),
        code: words.iter().flat_map(|w| w.to_le_bytes()).collect(),
    };
    let lowered = lowered(&native)?;
    Ok(HostShader {
        stage: HwStage::Cs,
        dxil: lowered.dxil,
        user_data: lowered.user_data,
        static_samplers: lowered
            .samplers
            .iter()
            .map(|&w| super::StaticSampler::from_words(w))
            .collect(),
        samplers: lowered.samplers,
        root_cbvs: lowered.root_cbvs,
        root_srvs: lowered.root_srvs,
        root_uavs: lowered.root_uavs,
        memory: lowered.memory,
        memory_writes: lowered.memory_writes,
        heap_usage: lowered.heap_usage,
        view_addresses: lowered.view_addresses,
        heap: lowered.heap,
        heap_writes: lowered.heap_writes,
    })
}

fn lowered(native: &NativeProgram) -> Result<lower::Lowered, crate::ShaderError> {
    let p = ir::build(&native.code, native.wave64, native.vgprs).expect("ir::build");
    let st = structure::structure(&p).expect("structure");
    let res = resource::track(&p, &native.layout, native.stage, Some(&st)).expect("track");
    let io = io::raise(&p, native).expect("io::raise");
    lower::lower(&lower::Parts {
        native,
        p: &p,
        st: &st,
        io: &io,
        res: &res,
        inputs: Vec::new(),
        outputs: Vec::new(),
        prims: Vec::new(),
        mesh_stride: 0,
    })
}

/// The first texel of a footprint (see `Lower::gather_first`).
fn first(u: f32, size: u32) -> i64 {
    let u = if u.is_nan() { 0.0 } else { u.clamp(-2.0, 2.0) };
    ((f64::from(u) * f64::from(size) * 512.0).floor() as i64 - 255) >> 9
}

#[test]
fn gathers_at_a_level_match_the_hardware() {
    // 24×16 then 12×8, 6×4, 3×2; red = level × 1000 + y × 32 + x + 1.
    let mut texels = Vec::new();
    for l in 0..4 {
        let (w, h) = (24 >> l, 16 >> l);
        for y in 0..h {
            for x in 0..w {
                texels.push([(l * 1000 + y * 32 + x + 1) as f32, 0.0, 0.0, 1.0]);
            }
        }
    }
    let texel = |l: u32, x: i64, y: i64| -> f32 {
        let (w, h) = (24i64 >> l, 16i64 >> l);
        let skip: i64 = (0..l).map(|k| (24i64 >> k) * (16 >> k)).sum();
        if (0..w).contains(&x) && (0..h).contains(&y) {
            texels[(skip + y * w + x) as usize][0]
        } else {
            0.0
        }
    };
    // Records: lanes 0-15 at level 0 where rounding decides (coordinates
    // whose f32 product with 24 lands on the other side of a texel's
    // threshold than the exact one, ties of 1/512 on the 16 texels of y,
    // NaN, infinities, zeros); lanes 16-31 at other levels and offsets.
    let mut records: Vec<[u32; 4]> = Vec::new();
    let rounded = |u: f32| ((f64::from(u * 24.0) * 512.0).floor() as i64 - 255) >> 9;
    let mut straddling = (1..24).filter_map(|i| {
        let u0 = ((f64::from(i) + 127.5 / 256.0) / 24.0) as f32;
        (-1024i32..1024)
            .map(|d| f32::from_bits(u0.to_bits().wrapping_add_signed(d)))
            .find(|&u| rounded(u) != first(u, 24))
    });
    for k in 0..16 {
        let tie = (f64::from(k - 3) + 0.5 - 1.0 / 512.0) / 16.0;
        let u = match k {
            0 => f32::NAN,
            1 => f32::INFINITY,
            2 => f32::NEG_INFINITY,
            3 => -0.0,
            _ => straddling.next().unwrap(),
        };
        records.push([0, u.to_bits(), (tie as f32).to_bits(), 0]);
    }
    for k in 0..16u32 {
        let level = [1, 2, 3, 7][k as usize % 4];
        let offset = [0x0000, 0x3F01, 0x0302, 0x3E3D][k as usize / 4];
        let (u, v) = (
            (k as f32 * 0.77 + 0.3) / 13.0 - 0.1,
            (k as f32 * 0.41 + 0.2) / 9.0,
        );
        records.push([offset, u.to_bits(), v.to_bits(), level]);
    }
    let input: Vec<u32> = records.iter().flatten().copied().collect();
    let shader = build(&program(true)).expect("lower");
    if let Ok(mut v) = validator::Validator::load(validator::Validator::default_path().as_deref()) {
        v.validate(&mut shader.dxil.clone())
            .expect("DXIL validation failed");
    }
    let views = [
        View::Mipped(24, 16, 4, &texels),
        View::Buffer {
            data: &input,
            stride: Stride::Structured(16),
        },
        View::RawUav(32 * 12),
    ];
    let _one = super::valu_tests::ONE_AT_A_TIME
        .lock()
        .unwrap_or_else(|e| e.into_inner());
    let mut ran = 0;
    for (label, warp) in [("hardware", false), ("warp", true)] {
        let Ok(host) = Host::new(warp) else {
            eprintln!("no SM 6.6 {label} device; skipped");
            continue;
        };
        let (root, pso) = host.compute_pipeline(&shader).expect("pipeline");
        let ud = vec![0u32; shader.user_data_dwords().max(1) as usize];
        let out = host
            .dispatch(
                &root,
                &pso,
                &Dispatch {
                    roots: vec![(0, Root::Constants(ud))],
                    views: &views,
                    samplers: &[],
                    groups: [1, 1, 1],
                },
            )
            .unwrap_or_else(|e| panic!("{label}: dispatch failed: {e}"));
        ran += 1;
        for (lane, r) in records.iter().enumerate() {
            let got = |k: usize| -> Vec<f32> {
                out[0][lane * 12 + 4 * k..][..4]
                    .iter()
                    .map(|&b| f32::from_bits(b))
                    .collect()
            };
            let [offset, u, v, level] = *r;
            let level = level.min(3);
            let (w, h) = (24 >> level, 16 >> level);
            let at = |ox: i64, oy: i64| {
                let (x, y) = (
                    first(f32::from_bits(u), w) + ox,
                    first(f32::from_bits(v), h) + oy,
                );
                vec![
                    texel(level, x, y + 1),
                    texel(level, x + 1, y + 1),
                    texel(level, x + 1, y),
                    texel(level, x, y),
                ]
            };
            let sext = |b: u32| i64::from((b as i32) << 26 >> 26);
            assert_eq!(got(0), at(0, 0), "{label}: lane {lane}");
            assert_eq!(
                got(1),
                at(sext(offset), sext(offset >> 8)),
                "{label}: lane {lane}, offset"
            );
            // WARP's own gather truncates where the hardware rounds.
            if lane < 16 && !warp {
                assert_eq!(got(2), got(0), "{label}: lane {lane} against its gather");
            }
        }
    }
    assert!(ran > 0, "no device");
}

#[test]
fn gathers_at_a_level_that_may_be_a_fraction_stop() {
    let Err(crate::ShaderError::Unsupported { what, .. }) = build(&program(false)) else {
        panic!("translated a gather at a level read from memory");
    };
    assert_eq!(what, "a gather at a level that may not be whole");
}

#[test]
fn permutes_read_the_lane_named_and_zero_from_lanes_left_out() {
    // Lanes below 24 run the permute; each reads lane (13 × lane + 5) mod 32
    // (its address in dwords, offset 20 bytes) of 1000 + lane.
    let shader = build(&[
        0xBE84_0300, // s_mov_b32 s4, s0
        0xBE85_0380, // s_mov_b32 s5, 0
        0xF408_0A02,
        0xFA00_0040, // s_load_dwordx4 s[40:43], s[4:5], 0x40
        0x1602_00B4, // v_mul_u32_u24 v1, 52, v0
        0x4A04_00FF,
        0x0000_03E8, // v_add_nc_u32 v2, 1000, v0
        0x7E06_02FF,
        0x0000_DEAD, // v_mov_b32 v3, 0xdead
        0x3420_0082, // v_lshlrev_b32 v16, 2, v0
        0x7D88_0098, // v_cmp_gt_u32 vcc_lo, 24, v0
        0xBE94_3C6A, // s_and_saveexec_b32 s20, vcc_lo
        0xDACC_0014,
        0x0300_0201, // ds_bpermute_b32 v3, v1, v2 offset:20
        0xBF8C_C07F, // s_waitcnt lgkmcnt(0)
        0xBEFE_0314, // s_mov_b32 exec_lo, s20
        0xE070_1000,
        0x800A_0310, // buffer_store_dword v3, v16, s[40:43], 0 offen
        0xBF81_0000, // s_endpgm
    ])
    .expect("lower");
    if let Ok(mut v) = validator::Validator::load(validator::Validator::default_path().as_deref()) {
        v.validate(&mut shader.dxil.clone())
            .expect("DXIL validation failed");
    }
    let want: Vec<u32> = (0..32)
        .map(|l| match (13 * l + 5) % 32 {
            _ if l >= 24 => 0xDEAD,
            src if src < 24 => 1000 + src,
            _ => 0,
        })
        .collect();
    let views = [
        View::Texture(1, 1, &[[0.0; 4]]),
        View::Buffer {
            data: &[0; 4],
            stride: Stride::Structured(16),
        },
        View::RawUav(32),
    ];
    let _one = super::valu_tests::ONE_AT_A_TIME
        .lock()
        .unwrap_or_else(|e| e.into_inner());
    for (label, warp) in [("hardware", false), ("warp", true)] {
        let Ok(host) = Host::new(warp) else {
            eprintln!("no SM 6.6 {label} device; skipped");
            continue;
        };
        // The program declares the console's wave of 32, which WARP refuses.
        let Ok((root, pso)) = host.compute_pipeline(&shader) else {
            assert!(warp, "{label}: no pipeline for a wave of 32");
            eprintln!("{label}: no pipeline for a wave of 32; skipped");
            continue;
        };
        let ud = vec![0u32; shader.user_data_dwords().max(1) as usize];
        let out = host
            .dispatch(
                &root,
                &pso,
                &Dispatch {
                    roots: vec![(0, Root::Constants(ud))],
                    views: &views,
                    samplers: &[],
                    groups: [1, 1, 1],
                },
            )
            .unwrap_or_else(|e| panic!("{label}: dispatch failed: {e}"));
        assert_eq!(out[0], want, "{label}");
    }
}

#[test]
fn ordered_pixel_sections_use_rasterizer_ordered_views() {
    // A pixel program's ordered section (POPS) as the title's compiler
    // writes it: the collision id after the primitive mask (s10, user data
    // ending at s9), a wait loop on the exiting wave id, a read-modify-write
    // of a UAV, the end of the section.
    let code: Vec<u32> = vec![
        0xBE84_0308, // s_mov_b32 s4, s8
        0xBE85_0380, // s_mov_b32 s5, 0
        0xF40C_0302,
        0xFA00_0000, // s_load_dwordx8 s[12:19], s[4:5], 0x0
        0x9000_9F0A, // s_lshr_b32 s0, s10, 31
        0xBF06_8000, // s_cmp_eq_u32 s0, 0
        0x8580_807E, // s_cselect_b64 s[0:1], exec, 0
        0x8102_81EF, // s_add_i32 s2, src_pops_exiting_wave_id, 1
        0xBF08_8502, // s_cmp_gt_u32 s2, 5
        0x8582_807E, // s_cselect_b64 s[2:3], exec, 0
        0x8880_0200, // s_or_b64 s[0:1], s[0:1], s[2:3]
        0x87EA_007E, // s_and_b64 vcc, exec, s[0:1]
        0xBF87_0004, // s_cbranch_vccnz +4
        0xBF8E_0001, // s_sleep 1
        0x8100_81EF, // s_add_i32 s0, src_pops_exiting_wave_id, 1
        0xBF0A_8500, // s_cmp_lt_u32 s0, 5
        0xBF84_FFFC, // s_cbranch_scc0 -4
        0x7E00_0280, // v_mov_b32 v0, 0
        0x7E02_0280, // v_mov_b32 v1, 0
        0xBF8C_C07F, // s_waitcnt lgkmcnt(0)
        0xF000_3F88,
        0x0003_0200, // image_load v[2:5], v[0:1], s[12:19] dmask:0xf dim:2D unorm dlc glc
        0xBF8C_3F70, // s_waitcnt vmcnt(0)
        0x0604_04F2, // v_add_f32 v2, 1.0, v2
        0xF020_1F08,
        0x0003_0200, // image_store v[2:5], v[0:1], s[12:19] dmask:0xf dim:2D unorm
        0xBF90_0007, // s_sendmsg sendmsg(MSG_ORDERED_PS_DONE)
        0xF800_1890,
        0x0000_0000, // exp null off, off, off, off done vm
        0xBF81_0000, // s_endpgm
    ];
    let native = |db_shader_control| NativeProgram {
        stage: HwStage::Ps,
        wave64: true,
        vgprs: 8,
        layout: vec![RootParam {
            root: RootType::Table,
            table_offset_dwords: 0,
            sgprs: 1,
            base_register: 0,
            user_data: 8,
            slot: 0,
        }],
        used: vec![UsedResource {
            kind: ResourceType::Uav,
            root: RootType::Table,
            table_offset_dwords: 0,
            lower_bound: 0,
            stride: 0,
            dwords: 8,
            user_data: 8,
            slot: 0,
        }],
        io: StageIo::Ps {
            inputs: Vec::new(),
            input_addr: 0,
            db_shader_control,
            patches: Vec::new(),
        },
        code: code.iter().flat_map(|w| w.to_le_bytes()).collect(),
    };
    let l = lowered(&native(1 << 16)).expect("lower");
    assert!(l.rovs);
    if let Ok(mut v) = validator::Validator::load(validator::Validator::default_path().as_deref()) {
        v.validate(&mut l.dxil.clone())
            .expect("DXIL validation failed");
    }
    // Without the ordered mode the collision id is no value the host has.
    let Err(crate::ShaderError::Unsupported { what, .. }) = lowered(&native(0)) else {
        panic!("translated a collision id outside an ordered section");
    };
    assert_eq!(what, "a hardware SGPR input");
}

/// Exact CPU reference for `Lower::gather_first` and `Lower::gather_address`.
fn gather_first_ref(u: f32, size: u32, mode: u32, unnormalized: bool) -> i64 {
    let (u, mul) = if unnormalized {
        let u = if u.is_nan() { 0.0 } else { u.clamp(-65536.0, 65536.0) };
        (u, 1i64)
    } else if mode == 0 {
        let red = u - u.trunc();
        (if red.is_nan() { 0.0 } else { red }, i64::from(size))
    } else if mode == 1 {
        let red = u - 2.0 * (u * 0.5).trunc();
        (if red.is_nan() { 0.0 } else { red }, i64::from(size))
    } else {
        let u = if u.is_nan() { 0.0 } else { u.clamp(-64.0, 64.0) };
        (u, i64::from(size))
    };
    ((f64::from(u) * (mul as f64) * 512.0).floor() as i64 - 255) >> 9
}

fn gather_addr_ref(coord: i64, size: i64, mode: u32) -> Option<i64> {
    match mode {
        0 => Some(coord.rem_euclid(size)),
        1 => {
            let period = 2 * size;
            let t = coord.rem_euclid(period);
            Some(if t >= size { period - 1 - t } else { t })
        }
        2 => Some(coord.clamp(0, size - 1)),
        3 => {
            let m = if coord < 0 { -1 - coord } else { coord };
            Some(m.min(size - 1))
        }
        6 => (0..size).contains(&coord).then_some(coord),
        7 => {
            let m = if coord < 0 { -1 - coord } else { coord };
            (0..size).contains(&m).then_some(m)
        }
        _ => unreachable!(),
    }
}

/// Encodes an MIMG `image_gather4*` instruction (2 dwords).
#[allow(clippy::too_many_arguments)]
fn mimg_gather(
    opcode: u32,
    dmask: u32,
    dim: u32,
    unorm: bool,
    a16: bool,
    d16: bool,
    vdst: u32,
    vaddr: u32,
    srsrc: u32,
    ssamp: u32,
) -> [u32; 2] {
    let w0 = (0x3C << 26)
        | (opcode << 18)
        | (u32::from(unorm) << 12)
        | ((dmask & 0xF) << 8)
        | ((dim & 7) << 3);
    let w1 = (u32::from(d16) << 31)
        | (u32::from(a16) << 30)
        | ((ssamp >> 2) << 21)
        | ((srsrc >> 2) << 16)
        | (vdst << 8)
        | vaddr;
    [w0, w1]
}

fn build_raw_gather(words: &[u32]) -> Result<HostShader, crate::ShaderError> {
    let range = |kind, table_offset_dwords, dwords| UsedResource {
        kind,
        root: RootType::Table,
        table_offset_dwords,
        lower_bound: 0,
        stride: 0,
        dwords,
        user_data: 0,
        slot: 0,
    };
    let native = NativeProgram {
        stage: HwStage::Cs,
        wave64: false,
        vgprs: 32,
        layout: vec![RootParam {
            root: RootType::Table,
            table_offset_dwords: 0,
            sgprs: 1,
            base_register: 0,
            user_data: 0,
            slot: 0,
        }],
        used: vec![
            range(ResourceType::Srv, 0, 8),
            range(ResourceType::Srv, 8, 4),
            range(ResourceType::Uav, 16, 4),
        ],
        io: StageIo::Cs(ComputeIo {
            rsrc2: 1 << 1,
            threads: [32, 1, 1],
        }),
        code: words.iter().flat_map(|w| w.to_le_bytes()).collect(),
    };
    let lowered = lowered(&native)?;
    let shader = HostShader {
        stage: HwStage::Cs,
        dxil: lowered.dxil,
        user_data: lowered.user_data,
        static_samplers: lowered
            .samplers
            .iter()
            .map(|&w| super::StaticSampler::from_words(w))
            .collect(),
        samplers: lowered.samplers,
        root_cbvs: lowered.root_cbvs,
        root_srvs: lowered.root_srvs,
        root_uavs: lowered.root_uavs,
        memory: lowered.memory,
        memory_writes: lowered.memory_writes,
        view_addresses: lowered.view_addresses,
        heap: lowered.heap,
        heap_writes: lowered.heap_writes,
    };
    if let Ok(mut v) = validator::Validator::load(validator::Validator::default_path().as_deref()) {
        v.validate(&mut shader.dxil.clone())
            .expect("DXIL validation failed");
    }
    Ok(shader)
}

fn f32_to_f16_bits(x: f32) -> u32 {
    let bits = x.to_bits();
    let sign = (bits >> 16) & 0x8000;
    let exp = ((bits >> 23) & 0xFF) as i32;
    let frac = bits & 0x7F_FFFF;
    if exp == 0xFF {
        return sign | 0x7C00 | if frac != 0 { 0x0200 | (frac >> 13) } else { 0 };
    }
    let e = exp - 127 + 15;
    if e >= 31 {
        return sign | 0x7C00;
    }
    if e <= 0 {
        if e < -10 {
            return sign;
        }
        let m = (frac | 0x80_0000) >> (14 - e);
        return sign | ((m + 1) >> 1);
    }
    let half = sign | ((e as u32) << 10) | (frac >> 13);
    let rem = frac & 0x1FFF;
    if rem > 0x1000 || (rem == 0x1000 && (half & 1) != 0) {
        half + 1
    } else {
        half
    }
}

fn f16_bits_to_f32(bits: u32) -> f32 {
    let b = bits & 0xFFFF;
    let sign = if b & 0x8000 != 0 { -1.0f32 } else { 1.0 };
    let (e, m) = ((b >> 10) & 0x1F, b & 0x3FF);
    sign * match e {
        0 => m as f32 * 2f32.powi(-24),
        0x1F if m == 0 => f32::INFINITY,
        0x1F => f32::NAN,
        _ => (1024 + m) as f32 * 2f32.powi(e as i32 - 25),
    }
}

/// Tests `image_gather4_l` and `image_gather4_l_o` across all supported RDNA2
/// address modes (WRAP, MIRROR, CLAMP, MIRROR_ONCE, CLAMP_BORDER,
/// MIRROR_ONCE_BORDER, mixed X/Y), all three built-in border colours (on R
/// and A channels), whole LOD bias, min/max LOD clamps, and `mip_filter = 0`,
/// checking against both the CPU reference and native hardware
/// `image_gather4_lz` at level 0 on RTX hardware and WARP.
#[test]
fn gathers_at_a_level_address_modes_and_borders_match_reference_and_hardware() {
    // 16x8 then 8x4, 4x2, 2x1; each channel distinct so dmask selection is
    // checked (R = 1000*l + 32*y + x + 1, A = R + 0.5).
    let mut texels = Vec::new();
    for l in 0..4u32 {
        let (w, h) = (16 >> l, 8 >> l);
        for y in 0..h {
            for x in 0..w {
                let r = (l * 1000 + y * 32 + x + 1) as f32;
                texels.push([r, r + 0.25, r + 0.5, r + 0.75]);
            }
        }
    }
    let sample_texel = |l: u32, cx: i64, cy: i64, mx: u32, my: u32, border: u32, ch: usize| -> f32 {
        let (w, h) = (16i64 >> l, 8i64 >> l);
        let skip: i64 = (0..l).map(|k| (16i64 >> k) * (8 >> k)).sum();
        let border_val = match border {
            2 => 1.0,
            1 if ch == 3 => 1.0,
            _ => 0.0,
        };
        match (gather_addr_ref(cx, w, mx), gather_addr_ref(cy, h, my)) {
            (Some(x), Some(y)) => texels[(skip + y * w + x) as usize][ch],
            _ => border_val,
        }
    };

    struct GatherCfg {
        label: &'static str,
        mx: u32,
        my: u32,
        border: u32,
        dmask: u32,
        min_lod: u32,
        max_lod: u32,
        bias: i32,
        mip_filter: bool,
        whole_lod: bool,
        check_hw_lz: bool,
    }
    let cfg = |label, mx, my, border, dmask, min_lod, max_lod, bias, mip_filter, whole_lod, check_hw_lz| GatherCfg {
        label,
        mx,
        my,
        border,
        dmask,
        min_lod,
        max_lod,
        bias,
        mip_filter,
        whole_lod,
        check_hw_lz,
    };
    let configs = [
        cfg("wrap", 0, 0, 0, 1, 0, 12, 0, true, true, true),
        cfg("mirror", 1, 1, 0, 1, 0, 12, 0, true, true, true),
        // Custom border_type=3 is ignored when neither axis uses border.
        cfg("clamp_edge", 2, 2, 3, 1, 0, 12, 0, true, true, true),
        cfg("mirror_once", 3, 3, 0, 1, 0, 12, 0, true, true, true),
        cfg("border_white_r", 6, 6, 2, 1, 0, 12, 0, true, true, true),
        cfg("border_opaque_black_a", 6, 6, 1, 8, 0, 12, 0, true, true, true),
        cfg("border_opaque_black_r", 6, 6, 1, 1, 0, 12, 0, true, true, true),
        // Mirror-once border (mode 7) has no exact D3D12 static-sampler mode.
        cfg("mirror_once_border_white", 7, 7, 2, 1, 0, 12, 0, true, true, false),
        cfg("mixed_wrap_clamp", 0, 2, 0, 2, 0, 12, 0, true, true, true),
        // Whole LOD bias (+1) with min_lod=1, max_lod=2.
        cfg("bias_and_clamp_lod", 2, 2, 0, 1, 1, 2, 1, true, true, false),
        // No mip filter: always samples mip 0, even with non-whole LOD.
        cfg("no_mip_filter", 0, 0, 0, 1, 0, 12, 0, false, false, true),
    ];

    // 32 lanes of [offset, u, v, level_u32_or_f32]:
    // lanes 0..16 test mip 0 (including out-of-range, negative, >1, >2, NaN, +-inf);
    // lanes 16..32 test levels 0..=5 with non-zero offsets.
    let coords_uv: [(f32, f32); 16] = [
        (0.0, 0.0),
        (1.0, 1.0),
        (-0.25, -0.375),
        (-1.25, -1.75),
        (-3.3125, 2.6875),
        (1.3125, 1.8125),
        (2.4375, -2.1875),
        (5.125, -4.625),
        (0.5 / 16.0, 0.5 / 8.0),
        (15.5 / 16.0, 7.5 / 8.0),
        (-0.03125, 1.0625),
        (0.375, 0.625),
        (40.0, -40.0),
        (f32::NAN, 0.25),
        (f32::INFINITY, 0.25),
        (f32::NEG_INFINITY, 0.25),
    ];
    let mut records = Vec::with_capacity(32);
    for (u, v) in coords_uv {
        records.push([0u32, u.to_bits(), v.to_bits(), 0]);
    }
    for k in 0..16u32 {
        let level = [0, 1, 2, 3, 5][k as usize % 5];
        let offset = [0x0000, 0x3F01, 0x023E, 0x3D03][k as usize / 4];
        let u = (k as f32 * 0.37) - 1.75;
        let v = (k as f32 * 0.29) - 1.25;
        records.push([offset, u.to_bits(), v.to_bits(), level]);
    }

    let _one = super::valu_tests::ONE_AT_A_TIME
        .lock()
        .unwrap_or_else(|e| e.into_inner());
    let mut ran = 0;
    for (backend, warp) in [("hardware", false), ("warp", true)] {
        let Ok(host) = Host::new(warp) else {
            eprintln!("no SM 6.6 {backend} device; skipped");
            continue;
        };
        for &GatherCfg {
            label,
            mx,
            my,
            border,
            dmask,
            min_lod,
            max_lod,
            bias,
            mip_filter,
            whole_lod,
            check_hw_lz,
        } in &configs
        {
            let w0 = mx | (my << 3) | (2 << 6);
            let w1 = (min_lod << 8) | (max_lod << 20);
            let bias_field = ((bias * 256) as u32) & 0x3FFF;
            let w2 = (u32::from(mip_filter) << 26) | (1 << 22) | (1 << 20) | bias_field;
            let w3 = border << 30;

            let mut code = vec![
                0xBE84_0300, // s_mov_b32 s4, s0
                0xBE85_0380, // s_mov_b32 s5, 0
                0xF40C_0302,
                0xFA00_0000, // s_load_dwordx8 s[12:19], s[4:5], 0x0
                0xF408_0502,
                0xFA00_0020, // s_load_dwordx4 s[20:23], s[4:5], 0x20
                0xF408_0A02,
                0xFA00_0040, // s_load_dwordx4 s[40:43], s[4:5], 0x40
                0xBE88_03FF,
                w0, // s_mov_b32 s8, w0
                0xBE89_03FF,
                w1, // s_mov_b32 s9, w1
                0xBE8A_03FF,
                w2, // s_mov_b32 s10, w2
                0xBE8B_03FF,
                w3,          // s_mov_b32 s11, w3
                0x3420_0084, // v_lshlrev_b32 v16, 4, v0
                0x1628_00B0, // v_mul_u32_u24 v20, 48, v0
                0xBF8C_C07F, // s_waitcnt lgkmcnt(0)
                0xE038_1000,
                0x8005_0C10, // buffer_load_dwordx4 v[12:15], v16, s[20:23], 0 offen
                0xBF8C_3F70, // s_waitcnt vmcnt(0)
            ];
            if whole_lod {
                code.push(0x7E1E_0D0F); // v_cvt_f32_u32 v15, v15
            }
            code.extend(mimg_gather(0x44, dmask, 1, false, false, false, 4, 13, 12, 8));
            code.extend(mimg_gather(0x54, dmask, 1, false, false, false, 8, 12, 12, 8));
            code.extend(mimg_gather(0x47, dmask, 1, false, false, false, 16, 13, 12, 8));
            code.extend([
                0xBF8C_3F70, // s_waitcnt vmcnt(0)
                0xE078_1000,
                0x800A_0414, // buffer_store_dwordx4 v[4:7], v20, s[40:43], 0 offen
                0xE078_1010,
                0x800A_0814, // buffer_store_dwordx4 v[8:11], v20, s[40:43], 0 offen offset:16
                0xE078_1020,
                0x800A_1014, // buffer_store_dwordx4 v[16:19], v20, s[40:43], 0 offen offset:32
                0xBF81_0000, // s_endpgm
            ]);
            let shader = build_raw_gather(&code).unwrap_or_else(|e| panic!("{label}: {e}"));
            let mut input = Vec::with_capacity(32 * 4);
            for &[off, u, v, lvl] in &records {
                let lword = if whole_lod {
                    lvl
                } else {
                    (lvl as f32 + 0.37).to_bits()
                };
                input.extend([off, u, v, lword]);
            }
            let views = [
                View::Mipped(16, 8, 4, &texels),
                View::Buffer {
                    data: &input,
                    stride: Stride::Raw,
                },
                View::RawUav(32 * 12),
            ];
            let (root, pso) = host.compute_pipeline(&shader).expect("pipeline");
            let ud = vec![0u32; shader.user_data_dwords().max(1) as usize];
            let out = host
                .dispatch(
                    &root,
                    &pso,
                    &Dispatch {
                        roots: vec![(0, Root::Constants(ud))],
                        views: &views,
                        samplers: &[],
                        groups: [1, 1, 1],
                    },
                )
                .unwrap_or_else(|e| panic!("{backend}/{label}: dispatch failed: {e}"));
            let ch = dmask.trailing_zeros() as usize;
            let sext = |b: u32| i64::from((b as i32) << 26 >> 26);
            for (lane, &[offset, u_bits, v_bits, raw_lvl]) in records.iter().enumerate() {
                let got = |k: usize| -> Vec<f32> {
                    out[0][lane * 12 + 4 * k..][..4]
                        .iter()
                        .map(|&b| f32::from_bits(b))
                        .collect()
                };
                let level = if mip_filter {
                    ((raw_lvl as i32 + bias).clamp(min_lod as i32, max_lod as i32) as u32).min(3)
                } else {
                    0
                };
                let (w, h) = (16u32 >> level, 8u32 >> level);
                let (u, v) = (f32::from_bits(u_bits), f32::from_bits(v_bits));
                let at = |ox: i64, oy: i64| {
                    let x = gather_first_ref(u, w, mx, false) + ox;
                    let y = gather_first_ref(v, h, my, false) + oy;
                    vec![
                        sample_texel(level, x, y + 1, mx, my, border, ch),
                        sample_texel(level, x + 1, y + 1, mx, my, border, ch),
                        sample_texel(level, x + 1, y, mx, my, border, ch),
                        sample_texel(level, x, y, mx, my, border, ch),
                    ]
                };
                assert_eq!(got(0), at(0, 0), "{backend}/{label}: lane {lane} _l");
                assert_eq!(
                    got(1),
                    at(sext(offset), sext(offset >> 8)),
                    "{backend}/{label}: lane {lane} _l_o"
                );
                // On finite level-0 coordinates (lanes 0..13), verify against
                // native hardware `image_gather4_lz` (exact on hardware).
                if check_hw_lz && !warp && lane < 13 && level == 0 {
                    assert_eq!(
                        got(2),
                        got(0),
                        "{backend}/{label}: lane {lane} ({u}, {v}) against hardware gather4_lz"
                    );
                }
            }
        }
        ran += 1;
    }
    assert!(ran > 0, "no device");
}

/// Tests `image_gather4_l(_o)` with `a16`, `d16`, `unorm`, and `Texture2DArray`
/// (including `a16` 2D arrays with `lod` packed in the high half of the second
/// address VGPR) on the host GPU and WARP.
#[test]
fn gathers_at_a_level_support_a16_d16_unorm_and_2d_arrays() {
    // 16x16 2D array with 3 mip levels and 4 slices; unique value per
    // (slice, level, y, x).
    let (w0, h0, levels, slices) = (16u32, 16u32, 3u32, 4u32);
    let mut array_texels = Vec::new();
    for s in 0..slices {
        for l in 0..levels {
            let (w, h) = (w0 >> l, h0 >> l);
            for y in 0..h {
                for x in 0..w {
                    let val = (s * 400 + l * 100 + y * 16 + x + 1) as f32;
                    array_texels.push([val, val + 0.5, 0.0, 1.0]);
                }
            }
        }
    }
    let array_texel = |s: i64, l: u32, cx: i64, cy: i64, mode: u32, border: u32| -> f32 {
        let s = s.clamp(0, i64::from(slices) - 1) as u32;
        let (w, h) = (i64::from(w0 >> l), i64::from(h0 >> l));
        let per_slice: usize = (0..levels).map(|k| ((w0 >> k) * (h0 >> k)) as usize).sum();
        let skip: usize = (0..l).map(|k| ((w0 >> k) * (h0 >> k)) as usize).sum();
        let bval = if border == 2 { 1.0 } else { 0.0 };
        match (gather_addr_ref(cx, w, mode), gather_addr_ref(cy, h, mode)) {
            (Some(x), Some(y)) => {
                array_texels[s as usize * per_slice + skip + (y * w + x) as usize][0]
            }
            _ => bval,
        }
    };
    let slice_ref = |s: f32| -> i64 {
        if s.is_nan() {
            0
        } else {
            s.round_ties_even().clamp(0.0, (slices - 1) as f32) as i64
        }
    };

    // 1) 2D texture (slice 0): a16 + d16 + unorm in one compute shader.
    // Input record per lane (4 dwords): [packed_uv_f16, lod_u32, unorm_u, unorm_v].
    let mut tex2d = Vec::new();
    for l in 0..levels {
        let (w, h) = (w0 >> l, h0 >> l);
        for y in 0..h {
            for x in 0..w {
                let val = (l * 100 + y * 16 + x + 1) as f32;
                tex2d.push([val, 0.0, 0.0, 1.0]);
            }
        }
    }
    let tex2d_at = |l: u32, cx: i64, cy: i64, mode: u32, border: u32| -> f32 {
        let (w, h) = (i64::from(w0 >> l), i64::from(h0 >> l));
        let skip: usize = (0..l).map(|k| ((w0 >> k) * (h0 >> k)) as usize).sum();
        let bval = if border == 2 { 1.0 } else { 0.0 };
        match (gather_addr_ref(cx, w, mode), gather_addr_ref(cy, h, mode)) {
            (Some(x), Some(y)) => tex2d[skip + (y * w + x) as usize][0],
            _ => bval,
        }
    };

    // Sampler 0 (s[8:11]): CLAMP_LAST_TEXEL, mip linear, levels 0..12.
    // Sampler 1 (s[24:27]): CLAMP_BORDER + OPAQUE_WHITE, mip linear, levels 0..12.
    let mut code_2d = vec![
        0xBE84_0300, // s_mov_b32 s4, s0
        0xBE85_0380, // s_mov_b32 s5, 0
        0xF40C_0302,
        0xFA00_0000, // s_load_dwordx8 s[12:19], s[4:5], 0x0
        0xF408_0502,
        0xFA00_0020, // s_load_dwordx4 s[20:23], s[4:5], 0x20
        0xF408_0A02,
        0xFA00_0040, // s_load_dwordx4 s[40:43], s[4:5], 0x40
        0xBE88_03FF,
        2 | (2 << 3) | (2 << 6), // s8 = CLAMP
        0xBE89_03FF,
        0x00C0_0000,
        0xBE8A_03FF,
        0x0450_0000,
        0xBE8B_0380, // s11 = 0
        0xBE98_03FF,
        6 | (6 << 3) | (2 << 6), // s24 = BORDER
        0xBE99_03FF,
        0x00C0_0000,
        0xBE9A_03FF,
        0x0450_0000,
        0xBE9B_03FF,
        2 << 30,     // s27 = OPAQUE_WHITE
        0x3420_0084, // v_lshlrev_b32 v16, 4, v0
        0x1628_00A8, // v_mul_u32_u24 v20, 40, v0 (10 dwords per lane)
        0xBF8C_C07F, // s_waitcnt lgkmcnt(0)
        0xE038_1000,
        0x8005_0C10, // buffer_load_dwordx4 v[12:15], v16, s[20:23], 0 offen
        0xBF8C_3F70, // s_waitcnt vmcnt(0)
        // v17 = f32(v13) (whole LOD); v18 = f16(v17) (whole f16 LOD).
        0x7E22_0D0D, // v_cvt_f32_u32 v17, v13
        0x7E24_1511, // v_cvt_f16_f32 v18, v17
        // Put [v12 (u|v f16), v18 (lod f16)] in v[22:23] for a16 gather.
        0x7E2C_030C, // v_mov_b32 v22, v12
        0x7E2E_0312, // v_mov_b32 v23, v18
        // Put [v14 (unorm u), v15 (unorm v), v17 (lod)] in v[24:26].
        0x7E30_030E, // v_mov_b32 v24, v14
        0x7E32_030F, // v_mov_b32 v25, v15
        0x7E34_0311, // v_mov_b32 v26, v17
    ];
    // (a) image_gather4_l v[4:7], v[22:23], s[12:19], s[8:11] a16 (CLAMP)
    code_2d.extend(mimg_gather(0x44, 1, 1, false, true, false, 4, 22, 12, 8));
    // (b) image_gather4_l v[8:9], v[22:23], s[12:19], s[8:11] a16 d16 (CLAMP)
    code_2d.extend(mimg_gather(0x44, 1, 1, false, true, true, 8, 22, 12, 8));
    // (c) image_gather4_l v[28:31], v[24:26], s[12:19], s[24:27] unorm (BORDER WHITE)
    code_2d.extend(mimg_gather(0x44, 1, 1, true, false, false, 28, 24, 12, 24));
    code_2d.extend([
        0xBF8C_3F70, // s_waitcnt vmcnt(0)
        0xE078_1000,
        0x800A_0414, // store v[4:7] at +0
        0xE074_1010,
        0x800A_0814, // buffer_store_dwordx2 v[8:9] at +16
        0xE078_1018,
        0x800A_1C14, // store v[28:31] at +24
        0xBF81_0000, // s_endpgm
    ]);
    let shader_2d = build_raw_gather(&code_2d).expect("lower 2d a16/d16/unorm");

    // 2) Texture2DArray (dim = 5): f32 gather4_l + a16 gather4_l (lod in high
    // half of second address VGPR!).
    // Input record per lane (4 dwords): [u_f32, v_f32, slice_f32, lod_u32].
    let mut code_arr = vec![
        0xBE84_0300, // s_mov_b32 s4, s0
        0xBE85_0380, // s_mov_b32 s5, 0
        0xF40C_0302,
        0xFA00_0000, // s_load_dwordx8 s[12:19], s[4:5], 0x0
        0xF408_0502,
        0xFA00_0020, // s_load_dwordx4 s[20:23], s[4:5], 0x20
        0xF408_0A02,
        0xFA00_0040, // s_load_dwordx4 s[40:43], s[4:5], 0x40
        0xBE88_03FF,
        6 | (6 << 3) | (2 << 6), // BORDER
        0xBE89_03FF,
        0x00C0_0000,
        0xBE8A_03FF,
        0x0450_0000,
        0xBE8B_03FF,
        2 << 30,     // OPAQUE_WHITE
        0x3420_0084, // v_lshlrev_b32 v16, 4, v0
        0x3428_0085, // v_lshlrev_b32 v20, 5, v0 (32 bytes = 8 dwords per lane)
        0xBF8C_C07F, // s_waitcnt lgkmcnt(0)
        0xE038_1000,
        0x8005_0C10, // buffer_load_dwordx4 v[12:15], v16, s[20:23], 0 offen
        0xBF8C_3F70, // s_waitcnt vmcnt(0)
        0x7E1E_0D0F, // v_cvt_f32_u32 v15, v15
        // Pack a16 coordinates into v[22:23]: v22 = pkrtz(v12, v13), v23 = pkrtz(v14, v15).
        // VOP3 v_cvt_pkrtz_f16_f32 (0x12f): v22 = (u, v), v23 = (slice, lod).
        0xD52F_0016,
        0x0002_1B0C, // v_cvt_pkrtz_f16_f32 v22, v12, v13
        0xD52F_0017,
        0x0002_1F0E, // v_cvt_pkrtz_f16_f32 v23, v14, v15
    ];
    // (a) f32 2DArray: image_gather4_l v[4:7], v[12:15], s[12:19], s[8:11] dim:2D_ARRAY
    code_arr.extend(mimg_gather(0x44, 1, 5, false, false, false, 4, 12, 12, 8));
    // (b) a16 2DArray: image_gather4_l v[8:11], v[22:23], s[12:19], s[8:11] dim:2D_ARRAY a16
    code_arr.extend(mimg_gather(0x44, 1, 5, false, true, false, 8, 22, 12, 8));
    code_arr.extend([
        0xBF8C_3F70, // s_waitcnt vmcnt(0)
        0xE078_1000,
        0x800A_0414, // store v[4:7] at +0
        0xE078_1010,
        0x800A_0814, // store v[8:11] at +16
        0xBF81_0000, // s_endpgm
    ]);
    let shader_arr = build_raw_gather(&code_arr).expect("lower 2d array f32/a16");

    let mut input_2d = Vec::with_capacity(32 * 4);
    let mut rec_2d = Vec::with_capacity(32);
    for k in 0..32u32 {
        let u = (k as f32 * 0.125) - 0.5;
        let v = ((31 - k) as f32 * 0.0625) - 0.25;
        let lvl = k % 4;
        let un_u = (k as f32 * 0.75) - 2.0;
        let un_v = (k as f32 * 0.5) - 1.5;
        let uv_pk = f32_to_f16_bits(u) | (f32_to_f16_bits(v) << 16);
        rec_2d.push((
            f16_bits_to_f32(uv_pk),
            f16_bits_to_f32(uv_pk >> 16),
            lvl,
            un_u,
            un_v,
        ));
        input_2d.extend([uv_pk, lvl, un_u.to_bits(), un_v.to_bits()]);
    }

    let test_slices: [f32; 12] = [
        -2.0,
        -0.4,
        0.0,
        0.5,
        1.4,
        1.5,
        2.5,
        2.6,
        3.0,
        99.0,
        f32::NAN,
        f32::INFINITY,
    ];
    let mut input_arr = Vec::with_capacity(32 * 4);
    let mut rec_arr = Vec::with_capacity(32);
    for k in 0..32u32 {
        let u = (k as f32 * 0.0625) - 0.125;
        let v = ((k * 3) % 17) as f32 * 0.0625 - 0.0625;
        let s = test_slices[k as usize % test_slices.len()];
        let lvl = k % 4;
        rec_arr.push((u, v, s, lvl));
        input_arr.extend([u.to_bits(), v.to_bits(), s.to_bits(), lvl]);
    }

    let _one = super::valu_tests::ONE_AT_A_TIME
        .lock()
        .unwrap_or_else(|e| e.into_inner());
    let mut ran = 0;
    for (backend, warp) in [("hardware", false), ("warp", true)] {
        let Ok(host) = Host::new(warp) else {
            eprintln!("no SM 6.6 {backend} device; skipped");
            continue;
        };
        // Run 2D a16/d16/unorm shader.
        let (root, pso) = host.compute_pipeline(&shader_2d).expect("pipeline 2d");
        let ud = vec![0u32; shader_2d.user_data_dwords().max(1) as usize];
        let out = host
            .dispatch(
                &root,
                &pso,
                &Dispatch {
                    roots: vec![(0, Root::Constants(ud))],
                    views: &[
                        View::Mipped(w0, h0, levels, &tex2d),
                        View::Buffer {
                            data: &input_2d,
                            stride: Stride::Raw,
                        },
                        View::RawUav(32 * 10),
                    ],
                    samplers: &[],
                    groups: [1, 1, 1],
                },
            )
            .unwrap_or_else(|e| panic!("{backend}/2d: dispatch failed: {e}"));
        for (lane, &(u16f, v16f, raw_lvl, un_u, un_v)) in rec_2d.iter().enumerate() {
            let lvl = raw_lvl.min(levels - 1);
            let (w, h) = (w0 >> lvl, h0 >> lvl);
            let x = gather_first_ref(u16f, w, 2, false);
            let y = gather_first_ref(v16f, h, 2, false);
            let want_a16 = [
                tex2d_at(lvl, x, y + 1, 2, 0),
                tex2d_at(lvl, x + 1, y + 1, 2, 0),
                tex2d_at(lvl, x + 1, y, 2, 0),
                tex2d_at(lvl, x, y, 2, 0),
            ];
            let base = lane * 10;
            let got_a16: Vec<f32> = out[0][base..base + 4]
                .iter()
                .map(|&b| f32::from_bits(b))
                .collect();
            assert_eq!(got_a16, want_a16, "{backend}/a16 lane {lane}");

            let want_d16 = [
                f32_to_f16_bits(want_a16[0]) | (f32_to_f16_bits(want_a16[1]) << 16),
                f32_to_f16_bits(want_a16[2]) | (f32_to_f16_bits(want_a16[3]) << 16),
            ];
            assert_eq!(
                &out[0][base + 4..base + 6],
                &want_d16,
                "{backend}/d16 lane {lane}"
            );

            let ux = gather_first_ref(un_u, w, 6, true);
            let uy = gather_first_ref(un_v, h, 6, true);
            let want_unorm = [
                tex2d_at(lvl, ux, uy + 1, 6, 2),
                tex2d_at(lvl, ux + 1, uy + 1, 6, 2),
                tex2d_at(lvl, ux + 1, uy, 6, 2),
                tex2d_at(lvl, ux, uy, 6, 2),
            ];
            let got_unorm: Vec<f32> = out[0][base + 6..base + 10]
                .iter()
                .map(|&b| f32::from_bits(b))
                .collect();
            assert_eq!(got_unorm, want_unorm, "{backend}/unorm lane {lane}");
        }

        // Run Texture2DArray f32 + a16 shader.
        let (root, pso) = host.compute_pipeline(&shader_arr).expect("pipeline arr");
        let ud = vec![0u32; shader_arr.user_data_dwords().max(1) as usize];
        let out = host
            .dispatch(
                &root,
                &pso,
                &Dispatch {
                    roots: vec![(0, Root::Constants(ud))],
                    views: &[
                        View::MippedArray(w0, h0, levels, slices, &array_texels),
                        View::Buffer {
                            data: &input_arr,
                            stride: Stride::Raw,
                        },
                        View::RawUav(32 * 8),
                    ],
                    samplers: &[],
                    groups: [1, 1, 1],
                },
            )
            .unwrap_or_else(|e| panic!("{backend}/2d_array: dispatch failed: {e}"));
        for (lane, &(u, v, s, raw_lvl)) in rec_arr.iter().enumerate() {
            let lvl = raw_lvl.min(levels - 1);
            let (w, h) = (w0 >> lvl, h0 >> lvl);
            let x = gather_first_ref(u, w, 6, false);
            let y = gather_first_ref(v, h, 6, false);
            let si = slice_ref(s);
            let want_f32 = [
                array_texel(si, lvl, x, y + 1, 6, 2),
                array_texel(si, lvl, x + 1, y + 1, 6, 2),
                array_texel(si, lvl, x + 1, y, 6, 2),
                array_texel(si, lvl, x, y, 6, 2),
            ];
            let base = lane * 8;
            let got_f32: Vec<f32> = out[0][base..base + 4]
                .iter()
                .map(|&b| f32::from_bits(b))
                .collect();
            assert_eq!(got_f32, want_f32, "{backend}/2d_array f32 lane {lane}");

            // a16 variant uses f16-converted (u, v, s) (all values exact in f16).
            let got_a16: Vec<f32> = out[0][base + 4..base + 8]
                .iter()
                .map(|&b| f32::from_bits(b))
                .collect();
            assert_eq!(got_a16, want_f32, "{backend}/2d_array a16 lane {lane}");
        }
        ran += 1;
    }
    assert!(ran > 0, "no device");
}

/// Unsupported sampler features (half-border, custom border on a border mode,
/// fractional bias or LOD limits, reduction filter, unorm+wrap, non-whole a16
/// LOD) remain refused cleanly.
#[test]
fn gathers_at_a_level_still_refuse_unsupported_samplers() {
    let make = |w0: u32, w1: u32, w2: u32, w3: u32, unorm: bool, a16: bool, whole: bool| {
        let mut code = vec![
            0xBE84_0300,
            0xBE85_0380,
            0xF40C_0302,
            0xFA00_0000,
            0xF408_0502,
            0xFA00_0020,
            0xF408_0A02,
            0xFA00_0040,
            0xBE88_03FF,
            w0,
            0xBE89_03FF,
            w1,
            0xBE8A_03FF,
            w2,
            0xBE8B_03FF,
            w3,
            0x3420_0084,
            0xBF8C_C07F,
            0xE038_1000,
            0x8005_0C10,
            0xBF8C_3F70,
        ];
        if whole {
            code.push(0x7E1E_0D0F); // v_cvt_f32_u32 v15, v15
            if a16 {
                code.push(0x7E1E_150F); // v_cvt_f16_f32 v15, v15
            }
        }
        let vaddr = if a16 { 14 } else { 13 };
        code.extend(mimg_gather(0x44, 1, 1, unorm, a16, false, 4, vaddr, 12, 8));
        code.extend([0xBF8C_3F70, 0xE078_1000, 0x800A_0410, 0xBF81_0000]);
        build_raw_gather(&code)
    };
    let base_w1 = 0x00C0_0000;
    let base_w2 = 0x0450_0000;
    for (name, res, expected) in [
        (
            "half_border",
            make(4 | (4 << 3), base_w1, base_w2, 0, false, false, true),
            "a gather at a level through an unsupported sampler",
        ),
        (
            "custom_border_on_border",
            make(6 | (6 << 3), base_w1, base_w2, 3 << 30, false, false, true),
            "a gather at a level through an unsupported sampler",
        ),
        (
            "fractional_lod_bias",
            make(2 | (2 << 3), base_w1, base_w2 | 0x80, 0, false, false, true),
            "a gather at a level through an unsupported sampler",
        ),
        (
            "fractional_min_lod",
            make(2 | (2 << 3), base_w1 | 0x80, base_w2, 0, false, false, true),
            "a gather at a level through an unsupported sampler",
        ),
        (
            "min_reduction",
            make(2 | (2 << 3) | (1 << 29), base_w1, base_w2, 0, false, false, true),
            "a gather at a level through an unsupported sampler",
        ),
        (
            "unorm_wrap",
            make(0, base_w1, base_w2, 0, true, false, true),
            "a gather at a level through an unsupported sampler",
        ),
        (
            "non_whole_a16_lod",
            make(2 | (2 << 3), base_w1, base_w2, 0, false, true, false),
            "a gather at a level that may not be whole",
        ),
    ] {
        let Err(crate::ShaderError::Unsupported { what, .. }) = res else {
            panic!("{name}: expected Unsupported, got {:?}", res.err());
        };
        assert_eq!(what, expected, "{name}");
    }
}

/// Null Texture2D SRVs return all zeros for `image_gather4_l(_o)` even with
/// opaque border colors (`OpaqueWhite`, `OpaqueBlack` alpha), matching native
/// `image_gather4_lz`.
#[test]
fn gathers_at_a_level_return_zero_on_null_texture_even_with_opaque_border() {
    let base_w1 = 0x00C0_0000;
    let base_w2 = 0x0450_0000;
    let configs = [
        ("wrap", 0, base_w1, base_w2, 0, 1u32),
        ("mirror", 1 | (1 << 3), base_w1, base_w2, 0, 1),
        ("clamp", 2 | (2 << 3), base_w1, base_w2, 0, 1),
        ("mirror_once", 3 | (3 << 3), base_w1, base_w2, 0, 1),
        ("border_trans_black", 6 | (6 << 3), base_w1, base_w2, 0, 8),
        (
            "border_opaque_black_rgb",
            6 | (6 << 3),
            base_w1,
            base_w2,
            1 << 30,
            1,
        ),
        (
            "border_opaque_black_alpha",
            6 | (6 << 3),
            base_w1,
            base_w2,
            1 << 30,
            8,
        ),
        (
            "border_opaque_white",
            6 | (6 << 3),
            base_w1,
            base_w2,
            2 << 30,
            1,
        ),
        (
            "mirror_once_border_white",
            7 | (7 << 3),
            base_w1,
            base_w2,
            2 << 30,
            1,
        ),
        (
            "mixed_wrap_x_border_y",
            6 << 3,
            base_w1,
            base_w2,
            2 << 30,
            1,
        ),
        (
            "bias_plus1_border_white",
            6 | (6 << 3),
            base_w1,
            base_w2 | 0x0100,
            2 << 30,
            1,
        ),
    ];
    let input: Vec<u32> = (0..32u32)
        .flat_map(|k| {
            let u = (k as f32 * 0.125) - 1.0;
            let v = ((31 - k) as f32 * 0.125) - 1.0;
            let off = ((k as i32 % 5) - 2) as u32 & 0x3F;
            [off | (off << 8), u.to_bits(), v.to_bits(), k % 3]
        })
        .collect();
    let _one = super::valu_tests::ONE_AT_A_TIME
        .lock()
        .unwrap_or_else(|e| e.into_inner());
    let mut ran = 0;
    for (backend, warp) in [("hardware", false), ("warp", true)] {
        let Ok(host) = Host::new(warp) else {
            eprintln!("no SM 6.6 {backend} device; skipped");
            continue;
        };
        for &(label, w0, w1, w2, w3, dmask) in &configs {
            let mut code = vec![
                0xBE84_0300,
                0xBE85_0380,
                0xF40C_0302,
                0xFA00_0000,
                0xF408_0502,
                0xFA00_0020,
                0xF408_0A02,
                0xFA00_0040,
                0xBE88_03FF,
                w0 | (2 << 6),
                0xBE89_03FF,
                w1,
                0xBE8A_03FF,
                w2,
                0xBE8B_03FF,
                w3,
                0x3420_0084,
                0x1628_00B0, // v_mul_u32_u24 v20, 48, v0 (12 dwords per lane)
                0xBF8C_C07F,
                0xE038_1000,
                0x8005_0C10,
                0xBF8C_3F70,
                0x7E1E_0D0F, // v_cvt_f32_u32 v15, v15
            ];
            code.extend(mimg_gather(0x44, dmask, 1, false, false, false, 4, 13, 12, 8));
            code.extend(mimg_gather(0x54, dmask, 1, false, false, false, 8, 12, 12, 8));
            code.extend(mimg_gather(0x47, dmask, 1, false, false, false, 24, 13, 12, 8));
            code.extend([
                0xBF8C_3F70,
                0xE078_1000,
                0x800A_0414,
                0xE078_1010,
                0x800A_0814,
                0xE078_1020,
                0x800A_1814,
                0xBF81_0000,
            ]);
            let shader = build_raw_gather(&code).unwrap_or_else(|e| panic!("{label}: {e}"));
            let (root, pso) = host.compute_pipeline(&shader).expect("pipeline");
            let ud = vec![0u32; shader.user_data_dwords().max(1) as usize];
            let out = host
                .dispatch(
                    &root,
                    &pso,
                    &Dispatch {
                        roots: vec![(0, Root::Constants(ud))],
                        views: &[
                            View::NullTexture,
                            View::Buffer {
                                data: &input,
                                stride: Stride::Raw,
                            },
                            View::RawUav(32 * 12),
                        ],
                        samplers: &[],
                        groups: [1, 1, 1],
                    },
                )
                .unwrap_or_else(|e| panic!("{backend}/{label}: dispatch failed: {e}"));
            assert!(
                out[0].iter().all(|&b| b == 0),
                "{backend}/{label}: expected all zeros on null texture, got non-zero word"
            );
        }
        ran += 1;
    }
    assert!(ran > 0, "no device");
}

/// `whole_f16` on `v_lshl_or_b32 dst, src0, 16, src2` checks `src2[31:16]`:
/// a non-zero upper half in `src2` (literal or unmasked VGPR) that could alter
/// the packed high-half f16 LOD is rejected, while a zero upper half (literal
/// or masked VGPR) and constant `-1.0` remain accepted.
#[test]
fn gathers_at_a_level_a16_packed_lshl_or_validates_src2_high_half() {
    // VOP3 v_lshl_or_b32 (0x36f): dst = (src0 << 16) | src2 (inline 16 = 144).
    const fn v_lshl_or_16(vdst: u8, src0: u16, src2: u16) -> [u32; 2] {
        [
            0xD400_0000 | (0x36F << 16) | (vdst as u32),
            (src0 as u32) | (144 << 9) | ((src2 as u32) << 18),
        ]
    }
    let build_packed = |pack_insts: &[u32]| {
        let mut code = vec![
            0xBE84_0300,
            0xBE85_0380,
            0xF40C_0302,
            0xFA00_0000,
            0xF408_0502,
            0xFA00_0020,
            0xF408_0A02,
            0xFA00_0040,
            0xBE88_03FF,
            2 | (2 << 3) | (2 << 6),
            0xBE89_03FF,
            0x00C0_0000,
            0xBE8A_03FF,
            0x0450_0000,
            0xBE8B_0380,
            0x3420_0084,
            0xBF8C_C07F,
            0xE038_1000,
            0x8005_0C10, // buffer_load_dwordx4 v[12:15]
            0xBF8C_3F70,
            // v14 = whole f16 LOD from uint v15:
            0x7E1E_0D0F, // v_cvt_f32_u32 v15, v15
            0x7E1C_150F, // v_cvt_f16_f32 v14, v15
        ];
        code.extend_from_slice(pack_insts);
        // image_gather4_l v[4:7], v[12:13], s[12:19], s[8:11] dim:2D_ARRAY a16
        // v12 = (u, v), v13 = (slice, lod).
        code.extend(mimg_gather(0x44, 1, 5, false, true, false, 4, 12, 12, 8));
        code.extend([0xBF8C_3F70, 0xE078_1000, 0x800A_0410, 0xBF81_0000]);
        build_raw_gather(&code)
    };

    // 1. Literal fractional upper half in src2 (`0x0200_3c00` OR `0x3c00 << 16` = 1.5) -> refused.
    let mut lit_frac = vec![0x7E20_02FF, 0x0000_3C00]; // v16 = 0x3c00 (1.0f16)
    lit_frac.extend(v_lshl_or_16(13, 256 + 16, 255));
    lit_frac.push(0x0200_3C00);
    let err = build_packed(&lit_frac).expect_err("literal fractional src2 high half must fail");
    assert!(
        err.to_string().contains("may not be whole"),
        "unexpected error: {err}"
    );

    // 2. Per-lane unmasked VGPR src2 (v13 from buffer_load; unknown high half!) -> refused.
    let per_lane_unmasked = v_lshl_or_16(13, 256 + 14, 256 + 13);
    let err = build_packed(&per_lane_unmasked)
        .expect_err("per-lane unmasked src2 high half must fail");
    assert!(
        err.to_string().contains("may not be whole"),
        "unexpected error: {err}"
    );

    // 3. Per-lane whole LOD + masked src2 (`v_and_b32 v16, 0xFFFF, v13`) -> accepted.
    let mut per_lane_masked = vec![
        0x3620_1AFF,
        0x0000_FFFF, // v_and_b32 v16, 0xFFFF, v13
    ];
    per_lane_masked.extend(v_lshl_or_16(13, 256 + 14, 256 + 16));
    build_packed(&per_lane_masked).expect("per-lane whole LOD with zeroed src2 high half");

    // 4. Constant -1.0f16 (`0xBC00`) with zero-high literal src2 (`0x0000_3C00`) -> accepted.
    let mut const_neg_one = vec![0x7E20_02FF, 0x0000_BC00]; // v16 = 0xbc00 (-1.0f16)
    const_neg_one.extend(v_lshl_or_16(13, 256 + 16, 255));
    const_neg_one.push(0x0000_3C00);
    build_packed(&const_neg_one).expect("constant -1.0f16 LOD with zero high half in src2");
}
