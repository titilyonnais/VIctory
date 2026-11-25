//! The console's rectangle lists on the host. A rectangle comes as three
//! corners in any order (the blits put the right angle at `v1`, the tiled
//! lighting at `v0`), and the hardware adds the fourth opposite the right
//! angle `vc`, at `va + vb - vc`, for the position and every attribute. The
//! host has no such primitive; a geometry shader made for the vertex shader
//! that feeds it takes each triangle `(v0, v1, v2)`, finds the right angle
//! opposite its longest side, and emits the triangle with its completion
//! `(va, vb, v3)` (`va` before `vc`, `vb` after it), both turning the same
//! way. Integer outputs (which can only be flat) keep `va`'s value at the
//! fourth corner.

use super::bitcode::{Attr, Bin, Term, V, pred};
use super::shader::Shader;
use crate::ShaderError;
use crate::dxbc::{Container, FourCC};

/// Which of a rectangle's three corners (x and y of each) holds the right
/// angle: the one opposite the longest side, else (ties, NaN) `1`. The
/// geometry shader of [`rectangle_gs`] makes the same choice on the GPU.
pub fn right_angle(xy: [[f32; 2]; 3]) -> usize {
    let edge = |i: usize, j: usize| [xy[j][0] - xy[i][0], xy[j][1] - xy[i][1]];
    let dot = |a: [f32; 2], b: [f32; 2]| a[0] * b[0] + a[1] * b[1];
    let (e01, e02, e12) = (edge(0, 1), edge(0, 2), edge(1, 2));
    let (d01, d20, d12) = (dot(e01, e01), dot(e02, e02), dot(e12, e12));
    // Opposite the strictly longest side, plus a dot-product rescue for thin
    // rectangles where `d_long + d_short` rounds to `d_long` in `f32`:
    // `4 * dot < min(leg0^2, leg1^2)` guarantees `opp^2 > max + 0.5 * min`,
    // keeping exact isosceles ties (`4 * dot == 2 * base^2`) on `v1`.
    let four_dot0 = 4.0 * dot(e01, e02);
    let four_dot2 = 4.0 * dot(e02, e12);
    if (d12 > d20 && d12 > d01) || (d01 > four_dot0 && d20 > four_dot0) {
        0
    } else if (d01 > d12 && d01 > d20) || (d20 > four_dot2 && d12 > four_dot2) {
        2
    } else {
        1
    }
}

/// The geometry shader that completes the rectangles of the vertex shader
/// container `vs` (its output signature is the shader's input and output).
pub fn rectangle_gs(vs: &[u8]) -> Result<Vec<u8>, ShaderError> {
    let c = Container::parse(vs)?;
    let part = c
        .part(FourCC::OSG1)
        .ok_or(ShaderError::Malformed("a vertex shader without outputs"))?;
    let mut elements = super::signature(part.data)?;
    elements.retain(|e| e.row.is_some());
    for e in &mut elements {
        e.usage = (1 << e.cols) - 1;
    }
    let mut s = Shader::new(false);
    s.gs = Some(6);
    let b = s.entry;
    // Corner k's values, element by element, component by component.
    let mut corners: Vec<Vec<Vec<super::bitcode::V>>> = Vec::new();
    for k in 0..3u32 {
        let mut values = Vec::new();
        for (id, e) in elements.iter().enumerate() {
            let (ty, suffix) = if e.comp == 9 {
                (s.t.f32, ".f32")
            } else {
                (s.t.i32, ".i32")
            };
            let mut comps = Vec::new();
            for col in 0..e.cols {
                let args = [s.m.i32(id as u32), s.m.i32(0), s.m.i8(col), s.m.i32(k)];
                comps.push(s.op(b, &format!("loadInput{suffix}"), Attr::ReadNone, ty, 4, &args));
            }
            values.push(comps);
        }
        corners.push(values);
    }
    // The right angle, as [`right_angle`] picks it from the position.
    let position = elements
        .iter()
        .position(|e| e.kind == 3 && e.cols >= 2 && e.comp == 9)
        .ok_or(ShaderError::Malformed("a rectangle list without a position"))?;
    let edge = |s: &mut Shader, i: usize, j: usize| -> [V; 2] {
        [
            s.m.bin(
                b,
                Bin::Sub,
                corners[j][position][0],
                corners[i][position][0],
            ),
            s.m.bin(
                b,
                Bin::Sub,
                corners[j][position][1],
                corners[i][position][1],
            ),
        ]
    };
    let dot = |s: &mut Shader, a: [V; 2], c: [V; 2]| {
        let px = s.m.bin(b, Bin::Mul, a[0], c[0]);
        let py = s.m.bin(b, Bin::Mul, a[1], c[1]);
        s.m.bin(b, Bin::Add, px, py)
    };
    let (e01, e02, e12) = (edge(&mut s, 0, 1), edge(&mut s, 0, 2), edge(&mut s, 1, 2));
    let (d01, d20, d12) = (
        dot(&mut s, e01, e01),
        dot(&mut s, e02, e02),
        dot(&mut s, e12, e12),
    );
    let four = s.m.f32(4.0f32.to_bits());
    let four_dot0 = dot(&mut s, e01, e02);
    let four_dot0 = s.m.bin(b, Bin::Mul, four, four_dot0);
    let four_dot2 = dot(&mut s, e02, e12);
    let four_dot2 = s.m.bin(b, Bin::Mul, four, four_dot2);
    let longest = |s: &mut Shader, d: V, others: [V; 2]| {
        let [x, y] = others.map(|o| s.m.cmp(b, pred::OGT, d, o));
        s.m.bin(b, Bin::And, x, y)
    };
    let wider = |s: &mut Shader, a: V, c: V, limit: V| {
        let x = s.m.cmp(b, pred::OGT, a, limit);
        let y = s.m.cmp(b, pred::OGT, c, limit);
        s.m.bin(b, Bin::And, x, y)
    };
    let at0 = {
        let strict = longest(&mut s, d12, [d20, d01]);
        let thin = wider(&mut s, d01, d20, four_dot0);
        s.m.bin(b, Bin::Or, strict, thin)
    };
    let at2 = {
        let strict = longest(&mut s, d01, [d12, d20]);
        let thin = wider(&mut s, d20, d12, four_dot2);
        s.m.bin(b, Bin::Or, strict, thin)
    };
    // `x0` if the right angle is at v0, `x2` if at v2, else `x1`.
    let pick = |s: &mut Shader, [x0, x1, x2]: [V; 3]| {
        let x = s.m.select(b, at2, x2, x1);
        s.m.select(b, at0, x0, x)
    };
    // The completing triangle (va, vb, v3), element by element.
    let mut second: [Vec<Vec<V>>; 3] = Default::default();
    for (id, e) in elements.iter().enumerate() {
        let mut cols: [Vec<V>; 3] = Default::default();
        let [c0, c1, c2] = [0, 1, 2].map(|k| &corners[k][id]);
        for ((&v0, &v1), &v2) in c0.iter().zip(c1).zip(c2) {
            let va = pick(&mut s, [v2, v0, v1]);
            let vb = pick(&mut s, [v1, v2, v0]);
            let v3 = if e.comp == 9 {
                let vc = pick(&mut s, [v0, v1, v2]);
                let sum = s.m.bin(b, Bin::Add, va, vb);
                s.m.bin(b, Bin::Sub, sum, vc)
            } else {
                va
            };
            for (k, v) in [va, vb, v3].into_iter().enumerate() {
                cols[k].push(v);
            }
        }
        for (k, c) in cols.into_iter().enumerate() {
            second[k].push(c);
        }
    }
    corners.extend(second);
    for triangle in [[0, 1, 2], [3, 4, 5]] {
        for k in triangle {
            for (id, e) in elements.iter().enumerate() {
                let suffix = if e.comp == 9 { ".f32" } else { ".i32" };
                for col in 0..e.cols {
                    let v = corners[k][id][usize::from(col)];
                    let args = [s.m.i32(id as u32), s.m.i32(0), s.m.i8(col), v];
                    s.op(b, &format!("storeOutput{suffix}"), Attr::NoUnwind, s.t.void, 5, &args);
                }
            }
            let stream = [s.m.i8(0)];
            s.op(b, "emitStream", Attr::NoUnwind, s.t.void, 97, &stream);
        }
        let stream = [s.m.i8(0)];
        s.op(b, "cutStream", Attr::NoUnwind, s.t.void, 98, &stream);
    }
    s.inputs = elements.clone();
    s.outputs = elements;
    s.m.terminate(b, Term::Ret);
    Ok(s.finish())
}

#[cfg(test)]
mod rule_tests {
    use super::right_angle;

    /// The right angle is found wherever it is; triangles that are no
    /// rectangle's take the corner opposite their longest side (their
    /// widest angle), and ties and NaN keep the blits' corner, `v1`.
    #[test]
    fn right_angles_are_found_in_any_order() {
        let (tl, bl, br, tr) = ([0.0, 0.0], [0.0, 1.0], [1.0, 1.0], [1.0, 0.0]);
        assert_eq!(right_angle([tl, bl, br]), 1);
        assert_eq!(right_angle([tl, bl, tr]), 0);
        assert_eq!(right_angle([bl, tr, tl]), 2);
        assert_eq!(right_angle([[0.0, 0.0], [3.0, 1.0], [1.0, 0.0]]), 2, "obtuse at v2");
        assert_eq!(right_angle([tl, tl, tl]), 1, "a point");
        assert_eq!(right_angle([tl, [f32::NAN, 0.0], br]), 1);
    }

    /// Thin rectangles whose squared hypotenuse rounds to the long leg's
    /// squared length in `f32` still find their right angle at every corner,
    /// under rotation, while thin isosceles ties stay at `v1`.
    #[test]
    fn thin_rectangles_find_their_right_angle() {
        let eps = 2.0f32.powi(-12);
        assert_eq!(right_angle([[0.0, 0.0], [0.0, eps], [1.0, 0.0]]), 0);
        assert_eq!(right_angle([[0.0, eps], [0.0, 0.0], [1.0, 0.0]]), 1);
        assert_eq!(right_angle([[0.0, eps], [1.0, 0.0], [0.0, 0.0]]), 2);
        assert_eq!(
            right_angle([[-0.5, 0.5], [-0.5 + eps, 0.5], [-0.5, -0.5]]),
            0
        );
        assert_eq!(
            right_angle([[0.0, 0.0], [0.0, 2.0f32.powi(-20)], [1.0, 0.0]]),
            0
        );

        // Rotated thin rectangle (right angle at each corner in turn).
        for (s, c) in [(0.8f32, 0.6f32), 0.7f32.sin_cos()] {
            let v0 = [-0.2f32, 0.3];
            let v1 = [v0[0] - s * eps, v0[1] + c * eps];
            let v2 = [v0[0] + c, v0[1] + s];
            assert_eq!(right_angle([v0, v1, v2]), 0);
            assert_eq!(right_angle([v1, v0, v2]), 1);
            assert_eq!(right_angle([v1, v2, v0]), 2);
        }

        // Thin isosceles tie (d12 == d20) and exact acute isosceles ties keep `v1`.
        assert_eq!(right_angle([[0.0, 0.0], [0.0, eps], [1.0, 0.5 * eps]]), 1);
        assert_eq!(
            right_angle([
                [0.0, 0.0],
                [8193.0 / 16384.0, 4100.0 / 16384.0],
                [4100.0 / 16384.0, 8193.0 / 16384.0],
            ]),
            1
        );
        assert_eq!(
            right_angle([
                [4100.0 / 16384.0, 8193.0 / 16384.0],
                [0.0, 0.0],
                [8193.0 / 16384.0, 4100.0 / 16384.0],
            ]),
            1
        );
    }
}

#[cfg(all(test, windows))]
mod tests {
    use super::super::bitcode::{Attr, Term};
    use super::super::container::SigElement;
    use super::super::host::{Draw, Host};
    use super::super::shader::Shader;

    fn element(name: &str, kind: u8, interp: u8, row: u32, cols: u8) -> SigElement {
        SigElement {
            name: name.into(),
            index: 0,
            kind,
            comp: 9,
            interp,
            row: Some(row),
            col: 0,
            cols,
            usage: (1 << cols) - 1,
            half: false,
        }
    }

    /// A shader copying its inputs to its outputs (vertex), or writing
    /// target 0 from TEXCOORD (pixel).
    fn shader(pixel: bool) -> Vec<u8> {
        let mut s = Shader::new(pixel);
        let b = s.entry;
        let (inputs, outputs) = if pixel {
            (
                vec![element("SV_Position", 3, 4, 0, 4), element("TEXCOORD", 0, 2, 1, 2)],
                vec![element("SV_Target", 16, 0, 0, 4)],
            )
        } else {
            (
                vec![element("POSITION", 0, 0, 0, 4), element("TEXCOORD", 0, 0, 1, 2)],
                vec![element("SV_Position", 3, 0, 0, 4), element("TEXCOORD", 0, 0, 1, 2)],
            )
        };
        let load = |s: &mut Shader, id: u32, col: u8| {
            let args = [s.m.i32(id), s.m.i32(0), s.m.i8(col), s.m.undef(s.t.i32)];
            s.op(b, "loadInput.f32", Attr::ReadNone, s.t.f32, 4, &args)
        };
        let values: Vec<_> = if pixel {
            let one = s.m.f32(1f32.to_bits());
            vec![load(&mut s, 1, 0), load(&mut s, 1, 1), one, one]
        } else {
            [(0, 0), (0, 1), (0, 2), (0, 3), (1, 0), (1, 1)]
                .into_iter()
                .map(|(id, c)| load(&mut s, id, c))
                .collect()
        };
        let mut k = 0;
        for (id, e) in outputs.iter().enumerate() {
            for col in 0..e.cols {
                let args = [s.m.i32(id as u32), s.m.i32(0), s.m.i8(col), values[k]];
                s.op(b, "storeOutput.f32", Attr::NoUnwind, s.t.void, 5, &args);
                k += 1;
            }
        }
        s.inputs = inputs;
        s.outputs = outputs;
        s.m.terminate(b, Term::Ret);
        s.finish()
    }

    /// One rectangle with a coordinate attribute, its right angle at each of
    /// the three corners in turn (the blits' top left, bottom left, bottom
    /// right; the tiled lighting's top left, bottom left, top right; and
    /// one with it last): every pixel of the square is drawn, the attribute
    /// interpolated over it, the missing half included. Three corners on one
    /// line draw nothing.
    #[test]
    fn rectangles_get_their_fourth_corner() {
        let (vs, ps) = (shader(false), shader(true));
        let gs = super::rectangle_gs(&vs).unwrap();
        if let Ok(mut v) =
            super::super::validator::Validator::load(super::super::validator::Validator::default_path().as_deref())
        {
            for c in [&vs, &gs, &ps] {
                v.validate(&mut c.clone()).unwrap();
            }
        }
        let corner = |x: f32, y: f32| [x, y, 0.5, 1.0, x, y];
        let (tl, bl, br, tr) = (corner(-0.5, 0.5), corner(-0.5, -0.5), corner(0.5, -0.5), corner(0.5, 0.5));
        let orders = [[tl, bl, br], [tl, bl, tr], [bl, tr, tl], [tl, corner(0.0, 0.0), br]];
        for warp in [false, true] {
            let Ok(host) = Host::new(warp) else { continue };
            let root = host.root_signature_for(&[]).unwrap();
            let pso = host.pipeline_gs(&root, &vs, &gs, &ps).unwrap();
            for (n, order) in orders.iter().enumerate() {
                let vertices = order.concat();
                let draw = Draw {
                    roots: Vec::new(),
                    textures: &[],
                    samplers: &[],
                    vertices: &vertices,
                    stride: 24,
                    size: (32, 32),
                    mesh_groups: 0,
                };
                let texels = host.draw(&root, &pso, &draw).unwrap();
                for (i, t) in texels.iter().enumerate() {
                    let (x, y) = ((i % 32) as f32, (i / 32) as f32);
                    let (u, v) = ((x + 0.5) / 16.0 - 1.0, 1.0 - (y + 0.5) / 16.0);
                    let inside = n < 3 && u.abs() < 0.5 && v.abs() < 0.5;
                    let want = if inside { [u, v, 1.0, 1.0] } else { [0.0; 4] };
                    let near = t.iter().zip(want).all(|(a, b)| (a - b).abs() < 1e-4);
                    assert!(near, "warp {warp} order {n} pixel ({x}, {y}): {t:?}, want {want:?}");
                }
            }
        }
    }

    /// A thin rectangle with its right angle at `v0` completes onto the
    /// `+y` side rather than flipping onto `-y` when `d12` rounds to `d20`.
    #[test]
    fn thin_rectangles_get_their_fourth_corner() {
        let (vs, ps) = (shader(false), shader(true));
        let gs = super::rectangle_gs(&vs).unwrap();
        if let Ok(mut v) = super::super::validator::Validator::load(
            super::super::validator::Validator::default_path().as_deref(),
        ) {
            for c in [&vs, &gs, &ps] {
                v.validate(&mut c.clone()).unwrap();
            }
        }
        let eps = 2.0f32.powi(-12);
        let corner = |x: f32, y: f32| [x, y, 0.5, 1.0, x, y];
        let vertices = [corner(-0.75, 0.0), corner(-0.75, eps), corner(0.75, 0.0)].concat();
        let (w, h) = (8u32, 8192u32);
        let mut ran = 0;
        for warp in [false, true] {
            let Ok(host) = Host::new(warp) else { continue };
            let root = host.root_signature_for(&[]).unwrap();
            let pso = host.pipeline_gs(&root, &vs, &gs, &ps).unwrap();
            let draw = Draw {
                roots: Vec::new(),
                textures: &[],
                samplers: &[],
                vertices: &vertices,
                stride: 24,
                size: (w, h),
                mesh_groups: 0,
            };
            let texels = host.draw(&root, &pso, &draw).unwrap();
            // Row 4095 has center y = +2^-13 (inside [0, 2^-12]); row 4096 has y = -2^-13.
            for y in [4095u32, 4096] {
                let v = 1.0 - 2.0 * (y as f32 + 0.5) / h as f32;
                for x in 0..w {
                    let u = 2.0 * (x as f32 + 0.5) / w as f32 - 1.0;
                    let inside = y == 4095 && u.abs() < 0.75;
                    let want = if inside { [u, v, 1.0, 1.0] } else { [0.0; 4] };
                    let t = texels[(y * w + x) as usize];
                    let near = t.iter().zip(want).all(|(a, b)| (a - b).abs() < 1e-4);
                    assert!(near, "warp {warp} pixel ({x}, {y}): {t:?}, want {want:?}");
                }
            }
            ran += 1;
        }
        assert!(ran > 0, "neither hardware nor WARP SM 6.6 device available");
    }
}
