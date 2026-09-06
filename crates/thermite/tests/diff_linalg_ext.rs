//! Extended 3D/4D linear algebra coverage, complementing `diff_linalg.rs`.
//!
//! `diff_linalg` already covers dot3/dot4/cross3/quat4_product/sum_elements3/
//! zero4/one4/mat4_transpose/mat4_vec4. This file fills in the rest of
//! `register/linalg.rs`:
//!   - `mat3_transpose`, `mat3_vec3_product` (col- and row-major)
//!   - `mat4_vec3_product` (col- and row-major)
//!   - `quat4_vec3_product` (both `FAST` modes)
//!   - `mat4_product` (col- and row-major)
//!   - `mat4_det` / `mat4_inverse` / `mat4_inverse_inplace` (incl. `DET_ONLY`
//!     and the singular-matrix path)
//!   - `min_element3` / `max_element3` / `prod_elements3`
//!
//! All oracles are computed in `f64` by *mirroring the implementation's own
//! definition* (e.g. row-major == transpose-then-column-major,
//! `quat4_vec3` == `v + w*t + cross(q,t)` with `t = 2*cross(q,v)`), so the test pins the
//! documented behaviour rather than an independent textbook convention.
//! Tested at the `Vector` layer on every backend, `Vector<f32x4>` / `Vector<f64x4>`.
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
use thermite::vector::{GenericVector, LinAlg3Vector, LinAlg4Vector, NumericVector};

const TRIALS: usize = 128;

// --- f64 oracles. Matrices stored column-major: m[col][row]. --------------

fn transpose4(m: [[f64; 4]; 4]) -> [[f64; 4]; 4] {
    core::array::from_fn(|i| core::array::from_fn(|j| m[j][i]))
}

#[allow(non_snake_case)]
fn matvecN_col<const N: usize>(cols: [[f64; 4]; 4], v: [f64; 4]) -> [f64; 4] {
    let mut r = [0.0; 4];
    for j in 0..4 {
        for i in 0..N {
            r[j] += cols[i][j] * v[i];
        }
    }
    r
}

fn matmul4_col(a: [[f64; 4]; 4], b: [[f64; 4]; 4]) -> [[f64; 4]; 4] {
    core::array::from_fn(|k| matvecN_col::<4>(a, b[k]))
}

fn det3(m: [[f64; 3]; 3]) -> f64 {
    m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1]) - m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0])
        + m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0])
}

fn det4(m: [[f64; 4]; 4]) -> f64 {
    let mut total = 0.0;
    for c in 0..4 {
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

fn quat_vec3(q: [f64; 4], v: [f64; 4]) -> [f64; 3] {
    let w = q[3];
    let c = cross3(q, v);
    let t = [2.0 * c[0], 2.0 * c[1], 2.0 * c[2], 0.0];
    let ct = cross3(q, t);
    core::array::from_fn(|i| v[i] + w * t[i] + ct[i])
}

fn refract3(i: [f64; 4], n: [f64; 4], eta: f64) -> ([f64; 3], f64) {
    let d = n[0] * i[0] + n[1] * i[1] + n[2] * i[2];
    let k = 1.0 - eta * eta * (1.0 - d * d);
    if k < 0.0 {
        return ([0.0; 3], k);
    }
    let c = eta * d + k.sqrt();
    (core::array::from_fn(|j| eta * i[j] - c * n[j]), k)
}

/// Per-slot context, emitted as items at the top of each test body.
macro_rules! ctx {
    ($reg:ident, $e:ty) => {
        type V = Vector<<S as Simd>::$reg>;

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
        #[allow(dead_code)]
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
    };
}

macro_rules! vec3_ops {
    ($reg:ident, $e:ty, $tol:expr) => {{
        ctx!($reg, $e);
        let mut rng = harness::rng();
        for _ in 0..TRIALS {
            let (a, va) = rv(&mut rng);
            let af = f4(a);

            let mn = af[0].min(af[1]).min(af[2]);
            let mx = af[0].max(af[1]).max(af[2]);
            let pr = af[0] * af[1] * af[2];
            harness::assert_lanes_eq(
                "[min_element3]",
                &[],
                &[va.min_element3() as f64],
                &[mn],
                Tol::Rel($tol),
            );
            harness::assert_lanes_eq(
                "[max_element3]",
                &[],
                &[va.max_element3() as f64],
                &[mx],
                Tol::Rel($tol),
            );
            harness::assert_lanes_eq(
                "[prod_elements3]",
                &[],
                &[va.prod_elements3() as f64],
                &[pr],
                Tol::Rel($tol),
            );

            let (q, vq) = rv(&mut rng);
            let want = quat_vec3(f4(q), af);
            let g = rd(vq.quat4_vec3_product::<true>(va));
            harness::assert_lanes_eq("[quat4_vec3<false>]", &[], &g[..3], &want, Tol::Rel($tol));
            let g = rd(vq.quat4_vec3_product::<false>(va));
            harness::assert_lanes_eq("[quat4_vec3<true>]", &[], &g[..3], &want, Tol::Rel($tol));

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
            if k.abs() > 1.0e-3 {
                let label = if k < 0.0 { "[refract TIR]" } else { "[refract]" };
                harness::assert_lanes_eq(label, &[], &got[..3], &want_r, Tol::Rel($tol));
            }
        }
    }};
}

macro_rules! mat3_ops {
    ($reg:ident, $e:ty, $tol:expr) => {{
        ctx!($reg, $e);
        let mut rng = harness::rng();
        for _ in 0..TRIALS {
            let cols: [V; 3] = core::array::from_fn(|i| {
                let mut a = [rs(&mut rng), rs(&mut rng), rs(&mut rng), 0.0 as $e];
                a[i] += 10.0 as $e;
                a[3] = 0.0 as $e;
                V::from_slice(&a)
            });
            let cf: [[f64; 4]; 4] = core::array::from_fn(|i| if i < 3 { rd(cols[i]) } else { [0.0; 4] });
            let (v, vv) = rv(&mut rng);
            let vf = f4(v);

            let t = V::mat3_transpose(&cols);
            for i in 0..3 {
                let want = [cf[0][i], cf[1][i], cf[2][i]];
                harness::assert_lanes_eq("[mat3_transpose]", &[], &rd(t[i])[..3], &want, Tol::Exact);
            }

            let want_row: [f64; 3] = core::array::from_fn(|i| cf[i][0] * vf[0] + cf[i][1] * vf[1] + cf[i][2] * vf[2]);
            let g = rd(vv.mat3_vec3_product::<false>(&cols));
            harness::assert_lanes_eq("[mat3_vec3<row>]", &[], &g[..3], &want_row, Tol::Rel($tol));

            let want_col = matvecN_col::<3>(cf, vf);
            let g = rd(vv.mat3_vec3_product::<true>(&cols));
            harness::assert_lanes_eq("[mat3_vec3<col>]", &[], &g[..3], &want_col[..3], Tol::Rel($tol));

            let cols2: [V; 3] =
                core::array::from_fn(|_| V::from_slice(&[rs(&mut rng), rs(&mut rng), rs(&mut rng), 0.0 as $e]));
            let cf2: [[f64; 4]; 4] = core::array::from_fn(|i| if i < 3 { rd(cols2[i]) } else { [0.0; 4] });

            let got_c = V::mat3_product::<true>(&cols, &cols2);
            let got_r = V::mat3_product::<false>(&cols, &cols2);
            for k in 0..3 {
                let want_c = matvecN_col::<3>(cf, cf2[k]);
                harness::assert_lanes_eq(
                    "[mat3_product<col>]",
                    &[],
                    &rd(got_c[k])[..3],
                    &want_c[..3],
                    Tol::Rel($tol),
                );
                let want_r = matvecN_col::<3>(cf2, cf[k]);
                harness::assert_lanes_eq(
                    "[mat3_product<row>]",
                    &[],
                    &rd(got_r[k])[..3],
                    &want_r[..3],
                    Tol::Rel($tol),
                );
            }

            let m3: [[f64; 3]; 3] = core::array::from_fn(|i| core::array::from_fn(|j| cf[i][j]));
            harness::assert_lanes_eq(
                "[mat3_det]",
                &[],
                &[V::mat3_det::<false>(&cols) as f64],
                &[det3(m3)],
                Tol::Rel($tol),
            );
            harness::assert_lanes_eq(
                "[mat3_det fast]",
                &[],
                &[V::mat3_det::<true>(&cols) as f64],
                &[det3(m3)],
                Tol::Rel($tol),
            );

            for fast in [false, true] {
                let inv = if fast {
                    V::mat3_inverse::<true>(&cols)
                } else {
                    V::mat3_inverse::<false>(&cols)
                }
                .expect("[mat3_inverse] returned None");

                let prod = V::mat3_product::<true>(&cols, &inv);
                for k in 0..3 {
                    let want_id: [f64; 3] = core::array::from_fn(|j| if j == k { 1.0 } else { 0.0 });
                    harness::assert_lanes_eq("[M*M^-1 == I (3x3)]", &[], &rd(prod[k])[..3], &want_id, Tol::Rel($tol));
                }
            }

            let inv = V::mat3_inverse::<false>(&cols).expect("[mat3_inverse] returned None");

            let want_n = V::mat3_transpose(&inv);
            let detf = V::mat3_det::<false>(&cols) as f64;
            for (lbl_n, lbl_c, got_n, cof) in [
                (
                    "[mat3_normal<true>]",
                    "[mat3_normal<false>]",
                    V::mat3_normal::<true, false>(&cols),
                    V::mat3_normal::<false, false>(&cols),
                ),
                (
                    "[mat3_normal<true> fast]",
                    "[mat3_normal<false> fast]",
                    V::mat3_normal::<true, true>(&cols),
                    V::mat3_normal::<false, true>(&cols),
                ),
            ] {
                for k in 0..3 {
                    harness::assert_lanes_eq(lbl_n, &[], &rd(got_n[k])[..3], &rd(want_n[k])[..3], Tol::Rel($tol));
                    let want_cof: [f64; 3] = core::array::from_fn(|j| rd(want_n[k])[j] * detf);
                    harness::assert_lanes_eq(lbl_c, &[], &rd(cof[k])[..3], &want_cof, Tol::Rel($tol));
                }
            }
        }
    }};
}

macro_rules! mat4_vec3_ops {
    ($reg:ident, $e:ty, $tol:expr) => {{
        ctx!($reg, $e);
        let mut rng = harness::rng();
        for _ in 0..TRIALS {
            let cols: [V; 4] = core::array::from_fn(|_| rv(&mut rng).1);
            let cf: [[f64; 4]; 4] = core::array::from_fn(|i| rd(cols[i]));
            let (v, vv) = rv(&mut rng);
            let vf = f4(v);

            let want_col = matvecN_col::<3>(cf, vf);
            harness::assert_lanes_eq(
                "[mat4_vec3<col>]",
                &[],
                &rd(vv.mat4_vec3_product::<true>(&cols)),
                &want_col,
                Tol::Rel($tol),
            );

            let want_row = matvecN_col::<3>(transpose4(cf), vf);
            harness::assert_lanes_eq(
                "[mat4_vec3<row>]",
                &[],
                &rd(vv.mat4_vec3_product::<false>(&cols)),
                &want_row,
                Tol::Rel($tol),
            );
        }
    }};
}

macro_rules! quat_to_mat {
    ($reg:ident, $e:ty, $tol:expr) => {{
        ctx!($reg, $e);
        let mut rng = harness::rng();
        for _ in 0..TRIALS {
            let qf = f4(rv(&mut rng).0);
            let nrm = (qf[0] * qf[0] + qf[1] * qf[1] + qf[2] * qf[2] + qf[3] * qf[3]).sqrt();
            let qf = [qf[0] / nrm, qf[1] / nrm, qf[2] / nrm, qf[3] / nrm];
            let vq = V::from_slice(&[qf[0] as $e, qf[1] as $e, qf[2] as $e, qf[3] as $e]);

            let m3 = vq.quat_to_mat3::<true>();
            let basis = [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0]];
            for j in 0..3 {
                let want = quat_vec3(qf, basis[j]);
                let cj = rd(m3[j]);
                harness::assert_lanes_eq("[quat_to_mat3 col]", &[], &cj[..3], &want, Tol::Rel($tol));
            }

            let m3r = vq.quat_to_mat3::<false>();
            for i in 0..3 {
                let ri = rd(m3r[i]);
                let want: [f64; 3] = core::array::from_fn(|j| rd(m3[j])[i]);
                harness::assert_lanes_eq("[quat_to_mat3 row == transpose]", &[], &ri[..3], &want, Tol::Rel($tol));
            }

            let (vraw, vv) = rv(&mut rng);
            let want_rot = quat_vec3(qf, f4(vraw));
            let gv = rd(vv.mat3_vec3_product::<true>(&m3));
            harness::assert_lanes_eq(
                "[quat_to_mat3 then mat3_vec3]",
                &[],
                &gv[..3],
                &want_rot,
                Tol::Rel($tol),
            );
            let gvr = rd(vv.mat3_vec3_product::<false>(&m3r));
            harness::assert_lanes_eq(
                "[quat_to_mat3<row> then mat3_vec3<row>]",
                &[],
                &gvr[..3],
                &want_rot,
                Tol::Rel($tol),
            );

            let m4 = vq.quat_to_mat4::<true>();
            for j in 0..3 {
                let cj = rd(m4[j]);
                let mj = rd(m3[j]);
                harness::assert_lanes_eq(
                    "[quat_to_mat4 rot col]",
                    &[],
                    &cj,
                    &[mj[0], mj[1], mj[2], 0.0],
                    Tol::Rel($tol),
                );
            }
            harness::assert_lanes_eq(
                "[quat_to_mat4 translation col]",
                &[],
                &rd(m4[3]),
                &[0.0, 0.0, 0.0, 1.0],
                Tol::Rel($tol),
            );
        }
    }};
}

macro_rules! mat4_product_ops {
    ($reg:ident, $e:ty, $tol:expr) => {{
        ctx!($reg, $e);
        let mut rng = harness::rng();
        for _ in 0..TRIALS {
            let la: [V; 4] = core::array::from_fn(|_| rv(&mut rng).1);
            let lb: [V; 4] = core::array::from_fn(|_| rv(&mut rng).1);
            let af: [[f64; 4]; 4] = core::array::from_fn(|i| rd(la[i]));
            let bf: [[f64; 4]; 4] = core::array::from_fn(|i| rd(lb[i]));

            let want_c = matmul4_col(af, bf);
            let got_c = V::mat4_product::<true>(&la, &lb);
            for k in 0..4 {
                harness::assert_lanes_eq("[mat4_product<col>]", &[], &rd(got_c[k]), &want_c[k], Tol::Rel($tol));
            }

            let want_r = matmul4_col(bf, af);
            let got_r = V::mat4_product::<false>(&la, &lb);
            for k in 0..4 {
                harness::assert_lanes_eq("[mat4_product<row>]", &[], &rd(got_r[k]), &want_r[k], Tol::Rel($tol));
            }
        }
    }};
}

macro_rules! mat4_det_and_inverse {
    ($reg:ident, $e:ty, $tol:expr) => {{
        ctx!($reg, $e);
        let mut rng = harness::rng();
        for _ in 0..TRIALS {
            let (cols, mf) = rmat(&mut rng);
            let want_det = det4(mf);

            harness::assert_lanes_eq(
                "[mat4_det]",
                &[],
                &[V::mat4_det::<false>(&cols) as f64],
                &[want_det],
                Tol::Rel($tol),
            );
            harness::assert_lanes_eq(
                "[mat4_det fast]",
                &[],
                &[V::mat4_det::<true>(&cols) as f64],
                &[want_det],
                Tol::Rel($tol),
            );

            for fast in [false, true] {
                let inv = if fast {
                    V::mat4_inverse::<true>(&cols)
                } else {
                    V::mat4_inverse::<false>(&cols)
                }
                .expect("[mat4_inverse] returned None");

                let prod = V::mat4_product::<true>(&cols, &inv);
                for k in 0..4 {
                    let want_id: [f64; 4] = core::array::from_fn(|j| if j == k { 1.0 } else { 0.0 });
                    harness::assert_lanes_eq("[M*M^-1 == I]", &[], &rd(prod[k]), &want_id, Tol::Rel($tol));
                }
            }
        }
    }};
}

macro_rules! mat4_singular {
    ($reg:ident, $e:ty) => {{
        ctx!($reg, $e);
        let mut rng = harness::rng();
        let mut cols: [V; 4] = core::array::from_fn(|_| rv(&mut rng).1);
        cols[3] = V::ZERO;

        assert!(
            V::mat4_inverse::<false>(&cols).is_none(),
            "[mat4_inverse] should be None for singular"
        );
        assert!(
            V::mat4_inverse::<true>(&cols).is_none(),
            "[mat4_inverse fast] should be None for singular"
        );

        let mut tmp = cols;
        let det = V::mat4_inverse_inplace::<false>(&mut tmp);
        assert_eq!(det, 0.0 as $e, "[inverse_inplace] singular det should be exactly 0");

        let mut tmp = cols;
        let det = V::mat4_inverse_inplace::<true>(&mut tmp);
        assert_eq!(
            det, 0.0 as $e,
            "[inverse_inplace fast] singular det should be exactly 0"
        );
    }};
}

macro_rules! mat4_singular_equal_columns {
    ($reg:ident, $e:ty, $tol:expr) => {{
        ctx!($reg, $e);
        let mut rng = harness::rng();
        for _ in 0..TRIALS {
            let mut cols: [V; 4] = core::array::from_fn(|_| rv(&mut rng).1);
            cols[2] = cols[1];

            for (name, got) in [
                ("[mat4_det] equal columns", V::mat4_det::<false>(&cols) as f64),
                ("[mat4_det fast] equal columns", V::mat4_det::<true>(&cols) as f64),
            ] {
                assert!(got.abs() <= $tol, "{}: {:e} exceeds {:e}", name, got, $tol);
            }
        }
    }};
}

for_each_backend_concrete! {
    fn vec3_ops_f32() { vec3_ops!(f32x4, f32, 3.0e-3) }
    fn vec3_ops_f64() { vec3_ops!(f64x4, f64, 1.0e-9) }
    fn mat3_ops_f32() { mat3_ops!(f32x4, f32, 3.0e-3) }
    fn mat3_ops_f64() { mat3_ops!(f64x4, f64, 1.0e-9) }
    fn mat4_vec3_ops_f32() { mat4_vec3_ops!(f32x4, f32, 3.0e-3) }
    fn mat4_vec3_ops_f64() { mat4_vec3_ops!(f64x4, f64, 1.0e-9) }
    fn quat_to_mat_f32() { quat_to_mat!(f32x4, f32, 3.0e-3) }
    fn quat_to_mat_f64() { quat_to_mat!(f64x4, f64, 1.0e-9) }
    fn mat4_product_ops_f32() { mat4_product_ops!(f32x4, f32, 3.0e-3) }
    fn mat4_product_ops_f64() { mat4_product_ops!(f64x4, f64, 1.0e-9) }
    fn mat4_det_and_inverse_f32() { mat4_det_and_inverse!(f32x4, f32, 3.0e-3) }
    fn mat4_det_and_inverse_f64() { mat4_det_and_inverse!(f64x4, f64, 1.0e-9) }
    fn mat4_singular_f32() { mat4_singular!(f32x4, f32) }
    fn mat4_singular_f64() { mat4_singular!(f64x4, f64) }
    fn mat4_singular_equal_columns_f32() { mat4_singular_equal_columns!(f32x4, f32, 3.0e-3) }
    fn mat4_singular_equal_columns_f64() { mat4_singular_equal_columns!(f64x4, f64, 1.0e-9) }
}
