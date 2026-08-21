//! 3D/4D linear algebra (`LinAlg3Vector` / `LinAlg4Vector`), tested at the
//! **`Vector` layer** for `Vector<f32x4>` / `Vector<f64x4>` on Scalar + V2 + V3.
//! `register/linalg.rs` had essentially no coverage.
//!
//! Oracles are computed in `f64` and compared with a relative tolerance (the
//! SIMD paths use FMA / different summation orders). Inputs are moderate finite
//! floats, since linalg over the full edge-case corpus (Inf/NaN/denormals/huge) would
//! make the relative error meaningless. Pure lane-routing ops (transpose,
//! zero4/one4) are bit-exact. 3D ops only check the first three lanes (the 4th
//! is documented as unused / unspecified).
#![cfg(any(
    target_arch = "x86",
    target_arch = "x86_64",
    target_arch = "wasm32",
    target_arch = "aarch64"
))]

mod harness;

use rand::RngExt;

use harness::Tol;
use thermite::Vector;
use thermite::simd::Simd;
use thermite::vector::{GenericVector, LinAlg3Vector, LinAlg4Vector};

const TRIALS: usize = 256;

// --- f64 oracles ------------------------------------------------------------

fn o_dot3(a: [f64; 4], b: [f64; 4]) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}
fn o_dot4(a: [f64; 4], b: [f64; 4]) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2] + a[3] * b[3]
}
fn o_cross3(a: [f64; 4], b: [f64; 4]) -> [f64; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}
/// Hamilton product, quaternion stored as (x, y, z, w) = lanes (0, 1, 2, 3).
fn o_quat(a: [f64; 4], b: [f64; 4]) -> [f64; 4] {
    let ([x1, y1, z1, w1], [x2, y2, z2, w2]) = (a, b);
    [
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
    ]
}
/// transpose: out[i][j] = m[j][i].
fn o_transpose(m: [[f64; 4]; 4]) -> [[f64; 4]; 4] {
    let mut t = [[0.0; 4]; 4];
    for i in 0..4 {
        for j in 0..4 {
            t[i][j] = m[j][i];
        }
    }
    t
}
/// column-major M*v: result[j] = sum_i cols[i][j] * v[i].
fn o_mat4_vec4_col(cols: [[f64; 4]; 4], v: [f64; 4]) -> [f64; 4] {
    let mut r = [0.0; 4];
    for j in 0..4 {
        for i in 0..4 {
            r[j] += cols[i][j] * v[i];
        }
    }
    r
}
/// row-major M*v: result[j] = dot(row_j, v).
fn o_mat4_vec4_row(rows: [[f64; 4]; 4], v: [f64; 4]) -> [f64; 4] {
    let mut r = [0.0; 4];
    for j in 0..4 {
        for i in 0..4 {
            r[j] += rows[j][i] * v[i];
        }
    }
    r
}

macro_rules! linalg_suite {
    ($modname:ident, $backend:ty, $reg:ident, $e:ty, $tol:expr, $bl:expr) => {
        mod $modname {
            use super::*;
            type V = Vector<<$backend as Simd>::$reg>;

            // moderate finite scalar
            fn rs(rng: &mut rand::rngs::SmallRng) -> $e {
                rng.random_range(-8.0..8.0)
            }
            fn rv(rng: &mut rand::rngs::SmallRng) -> ([$e; 4], V) {
                let a = [rs(rng), rs(rng), rs(rng), rs(rng)];
                (a, V::from_slice(&a))
            }
            fn f64x4(a: [$e; 4]) -> [f64; 4] {
                [a[0] as f64, a[1] as f64, a[2] as f64, a[3] as f64]
            }
            fn rd(v: V) -> [f64; 4] {
                let g = v.into_array();
                [g[0] as f64, g[1] as f64, g[2] as f64, g[3] as f64]
            }

            #[test]
            fn vector_ops() {
                let mut rng = harness::rng();
                for _ in 0..TRIALS {
                    let (a, va) = rv(&mut rng);
                    let (b, vb) = rv(&mut rng);
                    let (af, bf) = (f64x4(a), f64x4(b));

                    // dot3 / dot4 (scalars)
                    harness::assert_lanes_eq(concat!($bl, " [dot3]"), &[], &[va.dot3(vb) as f64], &[o_dot3(af, bf)], Tol::Rel($tol));
                    harness::assert_lanes_eq(concat!($bl, " [dot4]"), &[], &[va.dot4(vb) as f64], &[o_dot4(af, bf)], Tol::Rel($tol));

                    // cross3 (both FAST modes), first 3 lanes
                    let want3 = o_cross3(af, bf);
                    let g = rd(va.cross3::<true>(vb));
                    harness::assert_lanes_eq(concat!($bl, " [cross3<false>]"), &[], &g[..3], &want3, Tol::Rel($tol));
                    let g = rd(va.cross3::<false>(vb));
                    harness::assert_lanes_eq(concat!($bl, " [cross3<true>]"), &[], &g[..3], &want3, Tol::Rel($tol));

                    // quat4_product (full 4 lanes)
                    harness::assert_lanes_eq(concat!($bl, " [quat4_product]"), &[], &rd(va.quat4_product(vb)), &o_quat(af, bf), Tol::Rel($tol));

                    // sum_elements3 (first 3 lanes summed)
                    let want_s = af[0] + af[1] + af[2];
                    harness::assert_lanes_eq(concat!($bl, " [sum_elements3]"), &[], &[va.sum_elements3() as f64], &[want_s], Tol::Rel($tol));

                    // zero4 / one4 (bit-exact lane set)
                    harness::assert_lanes_eq(concat!($bl, " [zero4]"), &[], &rd(va.zero4()), &[af[0], af[1], af[2], 0.0], Tol::Exact);
                    harness::assert_lanes_eq(concat!($bl, " [one4]"), &[], &rd(va.one4()), &[af[0], af[1], af[2], 1.0], Tol::Exact);
                }
            }

            #[test]
            fn matrix_ops() {
                let mut rng = harness::rng();
                for _ in 0..TRIALS {
                    let cols: [V; 4] = [rv(&mut rng).1, rv(&mut rng).1, rv(&mut rng).1, rv(&mut rng).1];
                    let mf: [[f64; 4]; 4] = core::array::from_fn(|i| rd(cols[i]));
                    let (v, vv) = rv(&mut rng);
                    let vf = f64x4(v);

                    // mat4_transpose (bit-exact lane routing)
                    let t = V::mat4_transpose(&cols);
                    let got_t: Vec<f64> = t.iter().flat_map(|&r| rd(r)).collect();
                    let want_t: Vec<f64> = o_transpose(mf).iter().flatten().copied().collect();
                    harness::assert_lanes_eq(concat!($bl, " [mat4_transpose]"), &[], &got_t, &want_t, Tol::Exact);

                    // mat4_vec4_product, column-major and row-major
                    harness::assert_lanes_eq(concat!($bl, " [mat4_vec4<col>]"), &[], &rd(vv.mat4_vec4_product::<true>(&cols)), &o_mat4_vec4_col(mf, vf), Tol::Rel($tol));
                    harness::assert_lanes_eq(concat!($bl, " [mat4_vec4<row>]"), &[], &rd(vv.mat4_vec4_product::<false>(&cols)), &o_mat4_vec4_row(mf, vf), Tol::Rel($tol));
                }
            }
        }
    };
}

linalg_suite!(scalar_f32, Scalar, f32x4, f32, 2.0e-3, "scalar f32x4");
linalg_suite!(scalar_f64, Scalar, f64x4, f64, 1.0e-9, "scalar f64x4");

use thermite::backend::scalar::Scalar;

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
mod x86 {
    use super::*;
    use thermite::backend::x86_v1::X86V1;
    use thermite::backend::x86_v2::X86V2;
    use thermite::backend::x86_v3::X86V3;
    linalg_suite!(v3_f32, X86V3, f32x4, f32, 2.0e-3, "x86_v3 f32x4");
    linalg_suite!(v3_f64, X86V3, f64x4, f64, 1.0e-9, "x86_v3 f64x4");
    linalg_suite!(v2_f32, X86V2, f32x4, f32, 2.0e-3, "x86_v2 f32x4");
    linalg_suite!(v2_f64, X86V2, f64x4, f64, 1.0e-9, "x86_v2 f64x4");
    linalg_suite!(v1_f32, X86V1, f32x4, f32, 2.0e-3, "x86_v1 f32x4");
    linalg_suite!(v1_f64, X86V1, f64x4, f64, 1.0e-9, "x86_v1 f64x4");
}

#[cfg(target_arch = "wasm32")]
mod wasm {
    use super::*;
    use thermite::backend::wasm::Wasm;
    linalg_suite!(wasm_f32, Wasm, f32x4, f32, 2.0e-3, "wasm f32x4");
    linalg_suite!(wasm_f64, Wasm, f64x4, f64, 1.0e-9, "wasm f64x4");
}

#[cfg(target_arch = "aarch64")]
mod neon {
    use super::*;
    use thermite::backend::neon::Neon;
    linalg_suite!(neon_f32, Neon, f32x4, f32, 2.0e-3, "neon f32x4");
    linalg_suite!(neon_f64, Neon, f64x4, f64, 1.0e-9, "neon f64x4");
}
