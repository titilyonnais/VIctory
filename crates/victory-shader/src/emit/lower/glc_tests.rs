use super::tests::{S_ENDPGM, lower_with, root};
use crate::native::{ResourceType, RootType, UsedResource};

fn validate_dxil(dxil: &[u8]) {
    #[cfg(windows)]
    {
        use super::super::validator::Validator;
        let mut v = Validator::load(Validator::default_path().as_deref())
            .expect("dxil.dll validator must load");
        let mut copy = dxil.to_vec();
        let ours = copy[4..20].to_vec();
        v.validate(&mut copy).expect("DXIL validation failed");
        assert_eq!(copy[4..20], ours[..], "container hash mismatch");
    }
    #[cfg(not(windows))]
    let _ = dxil;
}

#[test]
fn heap_uav_inference_and_glc_coherence_set_handle_properties() {
    // #724: one heap descriptor loaded and stored gets UAV handles for both,
    // including across a separate reload of the same heap address and
    // scalar `s_buffer_load_dword` (with and without `glc`), while distinct
    // read-only heap addresses remain SRVs.
    let words = [
        0xBE84_0300, // s_mov_b32 s4, s0 (resource heap base)
        0xBE85_0380, // s_mov_b32 s5, 0
        0xF408_0202,
        0xFA00_0000, // s_load_dwordx4 s[8:11], s[4:5], 0x0   (UAV @ 0)
        0xF408_0302,
        0xFA00_0020, // s_load_dwordx4 s[12:15], s[4:5], 0x20 (SRV @ 1)
        0xF408_0402,
        0xFA00_0040, // s_load_dwordx4 s[16:19], s[4:5], 0x40 (read-only constants @ 2)
        0x7E00_0280, // v_mov_b32 v0, 0
        0xE030_1000,
        0x8002_0100, // buffer_load_dword v1, v0, s[8:11], 0 offen
        0xE030_5004,
        0x8002_0200, // buffer_load_dword v2, v0, s[8:11], 0 offen offset:4 glc
        0xE030_1000,
        0x8003_0300, // buffer_load_dword v3, v0, s[12:15], 0 offen
        0xF420_0188,
        0xFA00_0000, // s_buffer_load_dword s6, s[16:19], 0x0
        // Reload UAV @ 0 into s[20:23] and read it with scalar `glc` and plain load.
        0xF408_0502,
        0xFA00_0000, // s_load_dwordx4 s[20:23], s[4:5], 0x0
        0xF421_01CA,
        0xFA00_0008, // s_buffer_load_dword s7, s[20:23], 0x8 glc
        0xF420_060A,
        0xFA00_000C, // s_buffer_load_dword s24, s[20:23], 0xC
        0x4A02_0501, // v_add_nc_u32 v1, v1, v2
        0x4A02_0701, // v_add_nc_u32 v1, v1, v3
        0x4A02_0206, // v_add_nc_u32 v1, s6, v1
        0x4A02_0207, // v_add_nc_u32 v1, s7, v1
        0x4A02_0218, // v_add_nc_u32 v1, s24, v1
        0xE070_1010,
        0x8002_0100, // buffer_store_dword v1, v0, s[8:11], 0 offen offset:16
        S_ENDPGM,
    ];
    let l = lower_with(&words, vec![root(RootType::ResourceHeap, 0, 0)], vec![]).unwrap();
    validate_dxil(&l.dxil);
    // Both reloaded `s_buffer_load_dword` uses on `s[20:23]` resolve to the
    // same UAV @ 0 (`0x500B` for `glc`, `0x100B` plain) and reuse its entry
    // handles instead of creating a new read-only SRV (`0x000B`) handle.
    assert_eq!(
        l.heap_props,
        vec![
            (0x100B, 0), // UAV @ 0 handle (load + store + reloaded scalar load)
            (0x000B, 0), // read-only SRV @ 1 handle
            (0x500B, 0), // buffer_load_dword glc + reloaded scalar glc on UAV @ 0
            (0x000B, 0), // s_buffer_load_dword on read-only heap constants @ 2
        ]
    );
}

#[test]
fn glc_only_heap_reads_stay_srv_and_table_uavs_are_globally_coherent() {
    // #725 / review invariant: a heap descriptor only read with `glc`
    // (never stored to or used atomically) must stay an SRV handle
    // (`0x000B`), never promoted to a UAV (`0x100B`/`0x500B`).
    let spin = [
        0xBE84_0300, // s_mov_b32 s4, s0 (resource heap base)
        0xBE85_0380, // s_mov_b32 s5, 0
        0xF408_0202,
        0xFA00_0000, // s_load_dwordx4 s[8:11], s[4:5], 0x0
        0x7E00_0280, // v_mov_b32 v0, 0
        // loop:
        0xE030_5000,
        0x8002_0100, // buffer_load_dword v1, v0, s[8:11], 0 offen glc
        0x7D84_0280, // v_cmp_eq_u32 vcc_lo, 0, v1
        0xBF87_FFFC, // s_cbranch_vccnz -> loop
        0xBF8A_0000, // s_barrier
        0xD834_0000,
        0x0000_0100, // ds_write_b32 v0, v1
        S_ENDPGM,
    ];
    let l = lower_with(&spin, vec![root(RootType::ResourceHeap, 0, 0)], vec![]).unwrap();
    validate_dxil(&l.dxil);
    assert!(!l.heap_props.is_empty());
    assert!(
        l.heap_props.iter().all(|&p| p == (0x000B, 0)),
        "{:?}",
        l.heap_props
    );

    // Scalar `s_buffer_load_dword glc` and `image_load glc` on read-only
    // heap descriptors likewise remain SRVs (`0x000B` and `0x0002`).
    let heap_ro_glc = [
        0xBE84_0300, // s_mov_b32 s4, s0 (resource heap base)
        0xBE85_0380, // s_mov_b32 s5, 0
        0xF408_0202,
        0xFA00_0000, // s_load_dwordx4 s[8:11], s[4:5], 0x0   (read-only raw buffer)
        0xF40C_0302,
        0xFA00_0020, // s_load_dwordx8 s[12:19], s[4:5], 0x20 (read-only 2D image)
        0x7E00_0280, // v_mov_b32 v0, 0
        0x7E02_0280, // v_mov_b32 v1, 0
        0xF421_0184,
        0xFA00_0000, // s_buffer_load_dword s6, s[8:11], 0x0 glc
        0xF000_2F08,
        0x0003_0400, // image_load v[4:7], v[0:1], s[12:19] dmask:0xf 2d glc
        0x4A08_0506, // v_add_nc_u32 v4, s6, v4
        0xD834_0000,
        0x0000_0400, // ds_write_b32 v0, v4
        S_ENDPGM,
    ];
    let lro = lower_with(
        &heap_ro_glc,
        vec![root(RootType::ResourceHeap, 0, 0)],
        vec![],
    )
    .unwrap();
    validate_dxil(&lro.dxil);
    assert_eq!(lro.heap_props, vec![(0x0002, 0x409), (0x000B, 0)]);

    // Table resources: structured buffer UAV (`idxen glc`), typed buffer
    // UAV (`glc`), 2D image UAV (`image_load glc`, `image_store glc`,
    // `image_atomic_add glc`), and a read-only table SRV whose `glc` load
    // must stay an SRV (0xB).
    let table = [
        0xBE84_0300, // s_mov_b32 s4, s0
        0xBE85_0380, // s_mov_b32 s5, 0
        0xF408_0202,
        0xFA00_0000, // s_load_dwordx4 s[8:11], s[4:5], 0x0    (structured UAV)
        0xF408_0302,
        0xFA00_0020, // s_load_dwordx4 s[12:15], s[4:5], 0x20  (typed UAV)
        0xF40C_0402,
        0xFA00_0040, // s_load_dwordx8 s[16:23], s[4:5], 0x40  (image UAV)
        0xF40C_0602,
        0xFA00_0060, // s_load_dwordx8 s[24:31], s[4:5], 0x60  (uint image UAV)
        0xF408_0802,
        0xFA00_0080, // s_load_dwordx4 s[32:35], s[4:5], 0x80  (raw SRV)
        0x7E00_0280, // v_mov_b32 v0, 0
        0x7E02_0280, // v_mov_b32 v1, 0
        0xE030_6000,
        0x8002_0200, // buffer_load_dword v2, v0, s[8:11], 0 idxen glc
        0xE00C_4000,
        0x8003_0400, // buffer_load_format_xyzw v[4:7], v0, s[12:15], 0 glc
        0xE01C_4000,
        0x8003_0400, // buffer_store_format_xyzw v[4:7], v0, s[12:15], 0 glc
        0xF000_2F08,
        0x0004_0800, // image_load v[8:11], v[0:1], s[16:23] dmask:0xf 2d glc
        0xF020_2F08,
        0x0004_0800, // image_store v[8:11], v[0:1], s[16:23] dmask:0xf 2d glc
        0xF044_2108,
        0x0006_0200, // image_atomic_add v2, v[0:1], s[24:31] dmask:0x1 2d glc
        0xE030_5000,
        0x8008_0300, // buffer_load_dword v3, v0, s[32:35], 0 offen glc
        0x4A04_0702, // v_add_nc_u32 v2, v2, v3
        0xD834_0000,
        0x0000_0200, // ds_write_b32 v0, v2
        S_ENDPGM,
    ];
    let used = vec![
        UsedResource {
            kind: ResourceType::Uav,
            root: RootType::Table,
            table_offset_dwords: 0,
            lower_bound: 0,
            stride: 4,
            dwords: 4,
            user_data: 0,
            slot: 0,
        },
        UsedResource {
            kind: ResourceType::Uav,
            root: RootType::Table,
            table_offset_dwords: 8,
            lower_bound: 0,
            stride: 0,
            dwords: 4,
            user_data: 0,
            slot: 0,
        },
        UsedResource {
            kind: ResourceType::Uav,
            root: RootType::Table,
            table_offset_dwords: 16,
            lower_bound: 0,
            stride: 0,
            dwords: 8,
            user_data: 0,
            slot: 0,
        },
        UsedResource {
            kind: ResourceType::Uav,
            root: RootType::Table,
            table_offset_dwords: 24,
            lower_bound: 0,
            stride: 0,
            dwords: 8,
            user_data: 0,
            slot: 0,
        },
        UsedResource {
            kind: ResourceType::Srv,
            root: RootType::Table,
            table_offset_dwords: 32,
            lower_bound: 0,
            stride: 0,
            dwords: 4,
            user_data: 0,
            slot: 0,
        },
    ];
    let lt = lower_with(&table, vec![root(RootType::Table, 0, 0)], used).unwrap();
    validate_dxil(&lt.dxil);
    assert!(lt.heap_props.contains(&(0x500C, 4)), "{:?}", lt.heap_props);
    assert!(
        lt.heap_props.contains(&(0x500A, 0x409)),
        "{:?}",
        lt.heap_props
    );
    assert!(
        lt.heap_props.contains(&(0x5002, 0x409)),
        "{:?}",
        lt.heap_props
    );
    assert!(
        lt.heap_props.contains(&(0x5002, 0x105)),
        "{:?}",
        lt.heap_props
    );
    assert!(lt.heap_props.contains(&(0x000B, 0)), "{:?}", lt.heap_props);
    assert!(!lt.heap_props.contains(&(0x500B, 0)), "{:?}", lt.heap_props);
}
