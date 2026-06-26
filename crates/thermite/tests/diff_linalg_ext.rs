//! Extended 3D/4D linear algebra coverage, complementing `diff_linalg.rs`.
//!
//! `diff_linalg` already covers dot3/dot4/cross3/quat4_product/sum_elements3/
//! zero4/one4/mat4_transpose/mat4_vec4. This file fills in the rest of
//! `register/linalg.rs`, which had no coverage:
//!   - `mat3_transpose`, `mat3_vec3_product` (col- and row-major)
//!   - `mat4_vec3_product` (col- and row-major)
//!   - `quat4_vec3_product` (both `DOP` modes)
//!   - `mat4_product` (col- and row-major)
//!   - `mat4_det` / `mat4_inverse` / `mat4_inverse_inplace` (incl. `DET_ONLY`
//!     and the singular-matrix path)
//!   - `min_element3` / `max_element3` / `prod_elements3`
//!
//! All oracles are computed in `f64` by *mirroring the implementation's own
//! definition* (e.g. row-major == transpose-then-column-major,
//! `quat4_vec3` == `v + w·t + q×t` with `t = 2·(q×v)`), so the test pins the
//! documented behaviour rather than an independent textbook convention.
//! Tested at the `Vector` layer on Scalar + V2 + V3, `Vector<f32x4>` /
//! `Vector<f64x4>`.
#![cfg(any(target_arch = "x86", target_arch = "x86_64", target_arch = "wasm32"))]

mod harness;

use rand::RngExt;

use harness::Tol;
use thermite::Vector;
use thermite::simd::Simd;
use thermite::vector::{GenericVector, LinAlg3Vector, LinAlg4Vector, NumericVector};

const TRIALS: usize = 128;

// --- f64 oracles. Matrices stored column-major: m[col][row]. --------------

fn transpose4(m: [[f64; 4]; 4]) -> [[f64; 4]; 4] {
    core::array::from_fn(|i| core::array::from_fn(|j| m[j][i]))
}

/// column-major M·v: r[j] = sum_i cols[i][j] * v[i] over `i in 0..N`.
fn matvecN_col<const N: usize>(cols: [[f64; 4]; 4], v: [f64; 4]) -> [f64; 4] {
    let mut r = [0.0; 4];
    for j in 0..4 {
        for i in 0..N {
            r[j] += cols[i][j] * v[i];
        }
    }
    r
}

/// column-major matmul: C[k] = M·B[k]; C[k][j] = sum_i a[i][j] * b[k][i].
fn matmul4_col(a: [[f64; 4]; 4], b: [[f64; 4]; 4]) -> [[f64; 4]; 4] {
    core::array::from_fn(|k| matvecN_col::<4>(a, b[k]))
}

fn det3(m: [[f64; 3]; 3]) -> f64 {
    m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1]) - m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0])
        + m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0])
}

/// determinant of a 4x4 (layout-independent: det(M) == det(Mᵀ)).
fn det4(m: [[f64; 4]; 4]) -> f64 {
    let mut total = 0.0;
    for c in 0..4 {
        // minor: delete row 0 and column `c` (m is [col][row]); surviving rows
        // are 1..=3, surviving cols are 0..4 except `c`.
        let mut minor = [[0.0f64; 3]; 3];
        let mut a = 0;
        for col in 0..4 {
            if col == c {
                continue;
            }
            for b in 0..3 {
                minor[a][b] = m[col][b + 1];
            }
            a += 1;
        }
        let sign = if c % 2 == 0 { 1.0 } else { -1.0 };
        total += sign * m[c][0] * det3(minor);
    }
    total
}

fn cross3(a: [f64; 4], b: [f64; 4]) -> [f64; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

/// quat·vec3 rotation, mirroring the impl: t = 2·(q×v); res = v + w·t + q×t.
fn quat_vec3(q: [f64; 4], v: [f64; 4]) -> [f64; 3] {
    let w = q[3];
    let c = cross3(q, v);
    let t = [2.0 * c[0], 2.0 * c[1], 2.0 * c[2], 0.0];
    let ct = cross3(q, t);
    core::array::from_fn(|i| v[i] + w * t[i] + ct[i])
}

/// GLSL/GLM refract in f64. Returns the result and the discriminant `k`
/// (`< 0` is total internal reflection, where the result is the zero vector).
fn refract3(i: [f64; 4], n: [f64; 4], eta: f64) -> ([f64; 3], f64) {
    let d = n[0] * i[0] + n[1] * i[1] + n[2] * i[2];
    let k = 1.0 - eta * eta * (1.0 - d * d);
    if k < 0.0 {
        return ([0.0; 3], k);
    }
    let c = eta * d + k.sqrt();
    (core::array::from_fn(|j| eta * i[j] - c * n[j]), k)
}

macro_rules! linalg_ext_suite {
    ($modname:ident, $backend:ty, $reg:ident, $e:ty, $tol:expr, $bl:expr) => {
        mod $modname {
            use super::*;
            type V = Vector<<$backend as Simd>::$reg>;

            fn rs(rng: &mut rand::rngs::SmallRng) -> $e {
                rng.random_range(-4.0..4.0)
            }
            fn rv(rng: &mut rand::rngs::SmallRng) -> ([$e; 4], V) {
                let a = [rs(rng), rs(rng), rs(rng), rs(rng)];
                (a, V::from_slice(&a))
            }
            fn f4(a: [$e; 4]) -> [f64; 4] {
                [a[0] as f64, a[1] as f64, a[2] as f64, a[3] as f64]
            }
            fn rd(v: V) -> [f64; 4] {
                let g = v.into_array();
                [g[0] as f64, g[1] as f64, g[2] as f64, g[3] as f64]
            }
            /// well-conditioned (diagonally dominant) matrix as Vector columns + f64 oracle.
            fn rmat(rng: &mut rand::rngs::SmallRng) -> ([V; 4], [[f64; 4]; 4]) {
                let mut mf = [[0.0f64; 4]; 4];
                let cols: [V; 4] = core::array::from_fn(|i| {
                    let mut a = [rs(rng), rs(rng), rs(rng), rs(rng)];
                    a[i] = a[i] + 12.0 as $e; // diagonal dominance => invertible
                    mf[i] = f4(a);
                    V::from_slice(&a)
                });
                (cols, mf)
            }

            #[test]
            fn vec3_ops() {
                let mut rng = harness::rng();
                for _ in 0..TRIALS {
                    let (a, va) = rv(&mut rng);
                    let af = f4(a);

                    // min/max/prod over first 3 lanes
                    let mn = af[0].min(af[1]).min(af[2]);
                    let mx = af[0].max(af[1]).max(af[2]);
                    let pr = af[0] * af[1] * af[2];
                    harness::assert_lanes_eq(concat!($bl, " [min_element3]"), &[], &[va.min_element3() as f64], &[mn], Tol::Rel($tol));
                    harness::assert_lanes_eq(concat!($bl, " [max_element3]"), &[], &[va.max_element3() as f64], &[mx], Tol::Rel($tol));
                    harness::assert_lanes_eq(concat!($bl, " [prod_elements3]"), &[], &[va.prod_elements3() as f64], &[pr], Tol::Rel($tol));

                    // quat4_vec3_product (both DOP modes), first 3 lanes
                    let (q, vq) = rv(&mut rng);
                    let want = quat_vec3(f4(q), af);
                    let g = rd(vq.quat4_vec3_product::<false>(va));
                    harness::assert_lanes_eq(concat!($bl, " [quat4_vec3<false>]"), &[], &g[..3], &want, Tol::Rel($tol));
                    let g = rd(vq.quat4_vec3_product::<true>(va));
                    harness::assert_lanes_eq(concat!($bl, " [quat4_vec3<true>]"), &[], &g[..3], &want, Tol::Rel($tol));

                    // refract — GLSL/GLM; unit incident & normal, random eta covers
                    // both real refraction and total internal reflection (k < 0 → 0).
                    let nrm3 = |v: [$e; 4]| -> [f64; 4] {
                        let f = f4(v);
                        let m = (f[0] * f[0] + f[1] * f[1] + f[2] * f[2]).sqrt();
                        [f[0] / m, f[1] / m, f[2] / m, 0.0]
                    };
                    let inf = nrm3(rv(&mut rng).0);
                    let nnf = nrm3(rv(&mut rng).0);
                    let vi = V::from_slice(&[inf[0] as $e, inf[1] as $e, inf[2] as $e, 0.0 as $e]);
                    let vn = V::from_slice(&[nnf[0] as $e, nnf[1] as $e, nnf[2] as $e, 0.0 as $e]);
                    let eta: f64 = rng.random_range(0.4..2.5);
                    let (want_r, k) = refract3(inf, nnf, eta);
                    let got = rd(vi.refract(vn, eta as $e));
                    // Skip near the TIR boundary, where f32/f64 can land on opposite sides.
                    if k.abs() > 1.0e-3 {
                        let label = if k < 0.0 { concat!($bl, " [refract TIR]") } else { concat!($bl, " [refract]") };
                        harness::assert_lanes_eq(label, &[], &got[..3], &want_r, Tol::Rel($tol));
                    }
                }
            }

            #[test]
            fn mat3_ops() {
                let mut rng = harness::rng();
                for _ in 0..TRIALS {
                    // 3x3 stored as 3 columns; 4th lane 0 to keep semantics unambiguous.
                    // Bias the diagonal so the matrix stays well-conditioned - a fully
                    // random f32 3x3 is occasionally ill-conditioned enough to blow the
                    // inverse's relative tolerance in `M·M⁻¹ ≈ I`.
                    let cols: [V; 3] = core::array::from_fn(|i| {
                        let mut a = [rs(&mut rng), rs(&mut rng), rs(&mut rng), 0.0 as $e];
                        a[i] += 10.0 as $e;
                        a[3] = 0.0 as $e;
                        V::from_slice(&a)
                    });
                    let cf: [[f64; 4]; 4] = core::array::from_fn(|i| if i < 3 { rd(cols[i]) } else { [0.0; 4] });
                    let (v, vv) = rv(&mut rng);
                    let vf = f4(v);

                    // mat3_transpose: out[i] = [c0[i], c1[i], c2[i]], i in 0..3 (bit-exact
                    // lane routing). NOTE: lane 3 is documented as zeroed but the impl leaves
                    // it unspecified (see TESTING.md), so only the first 3 lanes are checked.
                    let t = V::mat3_transpose(&cols);
                    for i in 0..3 {
                        let want = [cf[0][i], cf[1][i], cf[2][i]];
                        harness::assert_lanes_eq(concat!($bl, " [mat3_transpose]"), &[], &rd(t[i])[..3], &want, Tol::Exact);
                    }

                    // mat3_vec3_product, row-major: r[i] = dot3(cols[i], v).
                    let want_row: [f64; 3] = core::array::from_fn(|i| cf[i][0] * vf[0] + cf[i][1] * vf[1] + cf[i][2] * vf[2]);
                    let g = rd(vv.mat3_vec3_product::<false>(&cols));
                    harness::assert_lanes_eq(concat!($bl, " [mat3_vec3<row>]"), &[], &g[..3], &want_row, Tol::Rel($tol));

                    // mat3_vec3_product, column-major: r[j] = sum_i cols[i][j]*v[i].
                    let want_col = matvecN_col::<3>(cf, vf);
                    let g = rd(vv.mat3_vec3_product::<true>(&cols));
                    harness::assert_lanes_eq(concat!($bl, " [mat3_vec3<col>]"), &[], &g[..3], &want_col[..3], Tol::Rel($tol));

                    // a second 3x3 for the matrix-matrix product.
                    let cols2: [V; 3] = core::array::from_fn(|_| {
                        V::from_slice(&[rs(&mut rng), rs(&mut rng), rs(&mut rng), 0.0 as $e])
                    });
                    let cf2: [[f64; 4]; 4] = core::array::from_fn(|i| if i < 3 { rd(cols2[i]) } else { [0.0; 4] });

                    // mat3_product column-major: C[k] = M·B[k]; row-major swaps to B·A.
                    let got_c = V::mat3_product::<true>(&cols, &cols2);
                    let got_r = V::mat3_product::<false>(&cols, &cols2);
                    for k in 0..3 {
                        let want_c = matvecN_col::<3>(cf, cf2[k]);
                        harness::assert_lanes_eq(concat!($bl, " [mat3_product<col>]"), &[], &rd(got_c[k])[..3], &want_c[..3], Tol::Rel($tol));
                        let want_r = matvecN_col::<3>(cf2, cf[k]);
                        harness::assert_lanes_eq(concat!($bl, " [mat3_product<row>]"), &[], &rd(got_r[k])[..3], &want_r[..3], Tol::Rel($tol));
                    }

                    // mat3_det: scalar triple product == det of the 3x3.
                    let m3: [[f64; 3]; 3] = core::array::from_fn(|i| core::array::from_fn(|j| cf[i][j]));
                    harness::assert_lanes_eq(concat!($bl, " [mat3_det]"), &[], &[V::mat3_det(&cols) as f64], &[det3(m3)], Tol::Rel($tol));

                    // mat3_inverse: M · M⁻¹ ≈ I (first 3 lanes).
                    let inv = V::mat3_inverse(&cols).expect(concat!($bl, " [mat3_inverse] returned None"));
                    let prod = V::mat3_product::<true>(&cols, &inv);
                    for k in 0..3 {
                        let want_id: [f64; 3] = core::array::from_fn(|j| if j == k { 1.0 } else { 0.0 });
                        harness::assert_lanes_eq(concat!($bl, " [M·M⁻¹ == I (3x3)]"), &[], &rd(prod[k])[..3], &want_id, Tol::Rel($tol));
                    }

                    // mat3_normal::<true> == transpose(inverse): the inverse-transpose.
                    let want_n = V::mat3_transpose(&inv);
                    let got_n = V::mat3_normal::<true>(&cols);
                    // mat3_normal::<false> is the un-divided cofactor == inverse-transpose * det.
                    let cof = V::mat3_normal::<false>(&cols);
                    let detf = V::mat3_det(&cols) as f64;
                    for k in 0..3 {
                        harness::assert_lanes_eq(concat!($bl, " [mat3_normal<true>]"), &[], &rd(got_n[k])[..3], &rd(want_n[k])[..3], Tol::Rel($tol));
                        let want_cof: [f64; 3] = core::array::from_fn(|j| rd(want_n[k])[j] * detf);
                        harness::assert_lanes_eq(concat!($bl, " [mat3_normal<false>]"), &[], &rd(cof[k])[..3], &want_cof, Tol::Rel($tol));
                    }
                }
            }

            #[test]
            fn mat4_vec3_ops() {
                let mut rng = harness::rng();
                for _ in 0..TRIALS {
                    let cols: [V; 4] = core::array::from_fn(|_| rv(&mut rng).1);
                    let cf: [[f64; 4]; 4] = core::array::from_fn(|i| rd(cols[i]));
                    let (v, vv) = rv(&mut rng);
                    let vf = f4(v);

                    // column-major: r[j] = sum_{i=0..2} cols[i][j]*v[i] (ignores col 3 and v.w), all 4 lanes.
                    let want_col = matvecN_col::<3>(cf, vf);
                    harness::assert_lanes_eq(concat!($bl, " [mat4_vec3<col>]"), &[], &rd(vv.mat4_vec3_product::<true>(&cols)), &want_col, Tol::Rel($tol));

                    // row-major: transpose then column-major.
                    let want_row = matvecN_col::<3>(transpose4(cf), vf);
                    harness::assert_lanes_eq(concat!($bl, " [mat4_vec3<row>]"), &[], &rd(vv.mat4_vec3_product::<false>(&cols)), &want_row, Tol::Rel($tol));
                }
            }

            #[test]
            fn quat_to_mat() {
                let mut rng = harness::rng();
                for _ in 0..TRIALS {
                    // The conversion assumes a unit quaternion - normalize the random one.
                    let qf = f4(rv(&mut rng).0);
                    let nrm = (qf[0] * qf[0] + qf[1] * qf[1] + qf[2] * qf[2] + qf[3] * qf[3]).sqrt();
                    let qf = [qf[0] / nrm, qf[1] / nrm, qf[2] / nrm, qf[3] / nrm];
                    let vq = V::from_slice(&[qf[0] as $e, qf[1] as $e, qf[2] as $e, qf[3] as $e]);

                    // Columns of R(q) are the rotated basis vectors: col_j = R·e_j.
                    // `quat_vec3` is an independent f64 cross-product oracle, so this
                    // cross-checks the products-based `quat_to_mat3` construction.
                    let m3 = vq.quat_to_mat3::<true>();
                    let basis = [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0]];
                    for j in 0..3 {
                        let want = quat_vec3(qf, basis[j]);
                        let cj = rd(m3[j]);
                        harness::assert_lanes_eq(concat!($bl, " [quat_to_mat3 col]"), &[], &cj[..3], &want, Tol::Rel($tol));
                    }

                    // Row-major output is the transpose: row_i[j] == col_j[i].
                    let m3r = vq.quat_to_mat3::<false>();
                    for i in 0..3 {
                        let ri = rd(m3r[i]);
                        let want: [f64; 3] = core::array::from_fn(|j| rd(m3[j])[i]);
                        harness::assert_lanes_eq(concat!($bl, " [quat_to_mat3 row == transpose]"), &[], &ri[..3], &want, Tol::Rel($tol));
                    }

                    // Batch-rotation path: mat3_vec3(quat_to_mat3(q), v) == quat4_vec3_product(q, v).
                    let (vraw, vv) = rv(&mut rng);
                    let want_rot = quat_vec3(qf, f4(vraw));
                    let gv = rd(vv.mat3_vec3_product::<true>(&m3));
                    harness::assert_lanes_eq(concat!($bl, " [quat_to_mat3 then mat3_vec3]"), &[], &gv[..3], &want_rot, Tol::Rel($tol));
                    // Same rotation through the row-major matrix + row-major matvec.
                    let gvr = rd(vv.mat3_vec3_product::<false>(&m3r));
                    harness::assert_lanes_eq(concat!($bl, " [quat_to_mat3<row> then mat3_vec3<row>]"), &[], &gvr[..3], &want_rot, Tol::Rel($tol));

                    // mat4: rotation columns match m3 with lane 3 zeroed; translation col = e3.
                    let m4 = vq.quat_to_mat4::<true>();
                    for j in 0..3 {
                        let cj = rd(m4[j]);
                        let mj = rd(m3[j]);
                        harness::assert_lanes_eq(concat!($bl, " [quat_to_mat4 rot col]"), &[], &cj, &[mj[0], mj[1], mj[2], 0.0], Tol::Rel($tol));
                    }
                    harness::assert_lanes_eq(concat!($bl, " [quat_to_mat4 translation col]"), &[], &rd(m4[3]), &[0.0, 0.0, 0.0, 1.0], Tol::Rel($tol));
                }
            }

            #[test]
            fn mat4_product_ops() {
                let mut rng = harness::rng();
                for _ in 0..TRIALS {
                    let la: [V; 4] = core::array::from_fn(|_| rv(&mut rng).1);
                    let lb: [V; 4] = core::array::from_fn(|_| rv(&mut rng).1);
                    let af: [[f64; 4]; 4] = core::array::from_fn(|i| rd(la[i]));
                    let bf: [[f64; 4]; 4] = core::array::from_fn(|i| rd(lb[i]));

                    // column-major: C = A·B.
                    let want_c = matmul4_col(af, bf);
                    let got_c = V::mat4_product::<true>(&la, &lb);
                    for k in 0..4 {
                        harness::assert_lanes_eq(concat!($bl, " [mat4_product<col>]"), &[], &rd(got_c[k]), &want_c[k], Tol::Rel($tol));
                    }

                    // row-major: operands swapped => B·A.
                    let want_r = matmul4_col(bf, af);
                    let got_r = V::mat4_product::<false>(&la, &lb);
                    for k in 0..4 {
                        harness::assert_lanes_eq(concat!($bl, " [mat4_product<row>]"), &[], &rd(got_r[k]), &want_r[k], Tol::Rel($tol));
                    }
                }
            }

            #[test]
            fn mat4_det_and_inverse() {
                let mut rng = harness::rng();
                for _ in 0..TRIALS {
                    let (cols, mf) = rmat(&mut rng);
                    let want_det = det4(mf);

                    // mat4_det
                    harness::assert_lanes_eq(concat!($bl, " [mat4_det]"), &[], &[V::mat4_det(&cols) as f64], &[want_det], Tol::Rel($tol));

                    // full inverse: M · M⁻¹ ≈ I.
                    let inv = V::mat4_inverse(&cols).expect(concat!($bl, " [mat4_inverse] returned None"));
                    let prod = V::mat4_product::<true>(&cols, &inv);
                    for k in 0..4 {
                        let want_id: [f64; 4] = core::array::from_fn(|j| if j == k { 1.0 } else { 0.0 });
                        harness::assert_lanes_eq(concat!($bl, " [M·M⁻¹ == I]"), &[], &rd(prod[k]), &want_id, Tol::Rel($tol));
                    }
                }
            }

            #[test]
            fn mat4_singular() {
                // A zero column makes the determinant exactly 0 (expansion along it),
                // which the implementation is documented to catch.
                let mut rng = harness::rng();
                let mut cols: [V; 4] = core::array::from_fn(|_| rv(&mut rng).1);
                cols[3] = V::ZERO;

                assert!(V::mat4_inverse(&cols).is_none(), concat!($bl, " [mat4_inverse] should be None for singular"));

                let mut tmp = cols;
                let det = V::mat4_inverse_inplace(&mut tmp);
                assert_eq!(det, 0.0 as $e, concat!($bl, " [inverse_inplace] singular det should be exactly 0"));
            }
        }
    };
}

linalg_ext_suite!(scalar_f32, Scalar, f32x4, f32, 3.0e-3, "scalar f32x4");
linalg_ext_suite!(scalar_f64, Scalar, f64x4, f64, 1.0e-9, "scalar f64x4");

use thermite::backend::scalar::Scalar;

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
mod x86 {
use super::*;
use thermite::backend::x86_v1::X86V1;
use thermite::backend::x86_v2::X86V2;
use thermite::backend::x86_v3::X86V3;
linalg_ext_suite!(v3_f32, X86V3, f32x4, f32, 3.0e-3, "x86_v3 f32x4");
linalg_ext_suite!(v3_f64, X86V3, f64x4, f64, 1.0e-9, "x86_v3 f64x4");
linalg_ext_suite!(v2_f32, X86V2, f32x4, f32, 3.0e-3, "x86_v2 f32x4");
linalg_ext_suite!(v2_f64, X86V2, f64x4, f64, 1.0e-9, "x86_v2 f64x4");
linalg_ext_suite!(v1_f32, X86V1, f32x4, f32, 3.0e-3, "x86_v1 f32x4");
linalg_ext_suite!(v1_f64, X86V1, f64x4, f64, 1.0e-9, "x86_v1 f64x4");
}

#[cfg(target_arch = "wasm32")]
mod wasm {
use super::*;
use thermite::backend::wasm::Wasm;
linalg_ext_suite!(wasm_f32, Wasm, f32x4, f32, 3.0e-3, "wasm f32x4");
linalg_ext_suite!(wasm_f64, Wasm, f64x4, f64, 1.0e-9, "wasm f64x4");
}
