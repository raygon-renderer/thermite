//! Correctness gate for the spherical-harmonic kernels (`sh_impl` / `sh_d_impl`).
//!
//! Three independent lines of evidence, so no error can hide in shared structure:
//!
//! 1. **Literal closed forms** through `l = 4`: hand-transcribed polynomial
//!    expressions from the standard real-SH table. These anchor the _conventions_
//!    (orthonormal, no Condon-Shortley, `l(l+1)+m` layout), since a systematic error
//!    the recurrence oracle might share with the kernel cannot pass literals.
//! 2. **An independent scalar oracle** built the classical way (`theta`/`phi` via
//!    `atan2`, `P_l^m` with the `sin^m theta` factor kept inline, factorial
//!    normalization, `libm` trig), checked over random unit directions to `L = 8`.
//! 3. **Central finite differences** for the gradients, exploiting that the kernel
//!    evaluates its polynomial form at _any_ input, unit or not, so ambient FD is
//!    exactly the derivative the kernel claims to produce.
//!
//! The kernels are pure polynomial arithmetic over `FloatVector`, so a 1-lane
//! `Vector<f64>` exercises every code path. No backend stamping needed here.

use std::f64::consts::PI;

use thermite::math::policy::DefaultPolicy;
use thermite::prelude::*;
use thermite_special::specialized::{sh_d_impl, sh_eval_d_impl, sh_eval_impl, sh_impl, sh_table_impl};
use thermite_special::{MAX_SH_DEGREE, RealPrimalMath, RealSpecialMath, ShTable};

type V64 = Vector<f64>;
type V32 = Vector<f32>;

const L: usize = 8;
const N: usize = (L + 1) * (L + 1);

fn sh64(x: f64, y: f64, z: f64) -> [f64; N] {
    let mut out = [V64::splat(0.0); N];
    sh_impl::<DefaultPolicy, f64, V64, L, N, false>(V64::splat(x), V64::splat(y), V64::splat(z), &mut out);

    let mut res = [0.0; N];
    for i in 0..N {
        res[i] = out[i].extract::<0>();
    }
    res
}

/// Flat index for `(l, m)`.
fn idx(l: usize, m: i64) -> usize {
    ((l * (l + 1)) as i64 + m) as usize
}

/// Independent scalar reference: spherical coordinates, classical recurrence with the
/// `sin^m` factor inline, factorial normalization, `std` trig.
fn oracle(l: usize, m: i64, x: f64, y: f64, z: f64) -> f64 {
    let ma = m.unsigned_abs() as usize;
    let sin_t = (x * x + y * y).sqrt();
    let phi = y.atan2(x);

    // P_m^m = (2m - 1)!! sin^m, then up the column.
    let mut pmm = 1.0;
    for k in 1..=ma {
        pmm *= (2 * k - 1) as f64 * sin_t;
    }

    let p = if l == ma {
        pmm
    } else {
        let mut p_prev = pmm;
        let mut p_cur = (2 * ma + 1) as f64 * z * pmm;
        for ll in (ma + 2)..=l {
            let p_next = ((2 * ll - 1) as f64 * z * p_cur - (ll + ma - 1) as f64 * p_prev) / (ll - ma) as f64;
            p_prev = p_cur;
            p_cur = p_next;
        }
        p_cur
    };

    // K_l^m^2 = (2l + 1)/(4 pi) * (l - m)!/(l + m)!
    let mut nf = (2 * l + 1) as f64 / (4.0 * PI);
    for k in (l - ma + 1)..=(l + ma) {
        nf /= k as f64;
    }
    let n = nf.sqrt();

    match m {
        0 => n * p,
        _ if m > 0 => core::f64::consts::SQRT_2 * n * p * (ma as f64 * phi).cos(),
        _ => core::f64::consts::SQRT_2 * n * p * (ma as f64 * phi).sin(),
    }
}

/// Simple xorshift for reproducible test directions, so no dev-dependency is needed.
struct Rng(u64);

impl Rng {
    fn next_f64(&mut self) -> f64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        (self.0 >> 11) as f64 / (1u64 << 53) as f64
    }

    /// Uniform unit vector (rejection-free: z uniform in [-1, 1], phi uniform).
    fn unit(&mut self) -> (f64, f64, f64) {
        let z = 2.0 * self.next_f64() - 1.0;
        let phi = 2.0 * PI * self.next_f64();
        let r = (1.0 - z * z).max(0.0).sqrt();
        (r * phi.cos(), r * phi.sin(), z)
    }
}

#[test]
fn sh_closed_forms() {
    // (l, m, analytic) at a fixed generic direction plus the poles and axes.
    let dirs: &[(f64, f64, f64)] = &[
        (0.267261241912424, 0.534522483824849, 0.801783725737273), // (1,2,3)/sqrt(14)
        (0.0, 0.0, 1.0),
        (0.0, 0.0, -1.0),
        (1.0, 0.0, 0.0),
        (0.0, 1.0, 0.0),
        (-0.6, 0.0, 0.8),
    ];

    for &(x, y, z) in dirs {
        let got = sh64(x, y, z);

        let pi = PI;
        // Standard real SH table (no Condon-Shortley), l <= 4 anchors.
        let want: &[(usize, i64, f64)] = &[
            (0, 0, (1.0 / (4.0 * pi)).sqrt()),
            (1, -1, (3.0 / (4.0 * pi)).sqrt() * y),
            (1, 0, (3.0 / (4.0 * pi)).sqrt() * z),
            (1, 1, (3.0 / (4.0 * pi)).sqrt() * x),
            (2, -2, (15.0 / (4.0 * pi)).sqrt() * x * y),
            (2, -1, (15.0 / (4.0 * pi)).sqrt() * y * z),
            (2, 0, (5.0 / (16.0 * pi)).sqrt() * (3.0 * z * z - 1.0)),
            (2, 1, (15.0 / (4.0 * pi)).sqrt() * x * z),
            (2, 2, (15.0 / (16.0 * pi)).sqrt() * (x * x - y * y)),
            (3, -3, (35.0 / (32.0 * pi)).sqrt() * y * (3.0 * x * x - y * y)),
            (3, -2, (105.0 / (4.0 * pi)).sqrt() * x * y * z),
            (3, -1, (21.0 / (32.0 * pi)).sqrt() * y * (5.0 * z * z - 1.0)),
            (3, 0, (7.0 / (16.0 * pi)).sqrt() * (5.0 * z * z * z - 3.0 * z)),
            (3, 1, (21.0 / (32.0 * pi)).sqrt() * x * (5.0 * z * z - 1.0)),
            (3, 2, (105.0 / (16.0 * pi)).sqrt() * z * (x * x - y * y)),
            (3, 3, (35.0 / (32.0 * pi)).sqrt() * x * (x * x - 3.0 * y * y)),
            (
                4,
                0,
                (3.0 / 16.0) * (1.0 / pi).sqrt() * (35.0 * z * z * z * z - 30.0 * z * z + 3.0),
            ),
        ];

        for &(l, m, expect) in want {
            let v = got[idx(l, m)];
            assert!(
                (v - expect).abs() < 1e-13,
                "Y({l},{m}) at ({x},{y},{z}): got {v}, want {expect}"
            );
        }
    }
}

#[test]
fn sh_vs_oracle() {
    let mut rng = Rng(0x5EED_CAFE_F00D_D00D);

    for _ in 0..200 {
        let (x, y, z) = rng.unit();
        let got = sh64(x, y, z);

        for l in 0..=L {
            for m in -(l as i64)..=(l as i64) {
                let want = oracle(l, m, x, y, z);
                let v = got[idx(l, m)];
                assert!(
                    (v - want).abs() < 1e-11,
                    "Y({l},{m}) at ({x},{y},{z}): got {v}, want {want}"
                );
            }
        }
    }
}

#[test]
fn sh_poles() {
    // At the poles sin(theta) = 0: every m != 0 harmonic vanishes exactly (the c/s
    // recurrence multiplies by (x + iy) = 0), and Y_{l,0} = +-sqrt((2l+1)/4pi).
    for &z in &[1.0f64, -1.0] {
        let got = sh64(0.0, 0.0, z);

        for l in 0..=L {
            for m in -(l as i64)..=(l as i64) {
                let v = got[idx(l, m)];
                if m == 0 {
                    let want = ((2 * l + 1) as f64 / (4.0 * PI)).sqrt() * z.powi(l as i32);
                    assert!((v - want).abs() < 1e-13, "pole Y({l},0): got {v}, want {want}");
                } else {
                    assert_eq!(v, 0.0, "pole Y({l},{m}) must be exactly zero");
                }
            }
        }
    }
}

#[test]
fn sh_gradients_finite_difference() {
    let mut rng = Rng(0x0BAD_D5EE_D0DD_BA11);
    let h = 1e-5;

    for _ in 0..50 {
        let (x, y, z) = rng.unit();

        let mut out = [V64::splat(0.0); N];
        let mut ddx = [V64::splat(0.0); N];
        let mut ddy = [V64::splat(0.0); N];
        let mut ddz = [V64::splat(0.0); N];
        sh_d_impl::<DefaultPolicy, f64, V64, L, N, false>(
            V64::splat(x),
            V64::splat(y),
            V64::splat(z),
            &mut out,
            &mut ddx,
            &mut ddy,
            &mut ddz,
        );

        // Values must match the value-only kernel bit-for-bit (same operations).
        let vals = sh64(x, y, z);
        for i in 0..N {
            assert_eq!(out[i].extract::<0>(), vals[i], "sh_d value [{i}] diverges from sh");
        }

        // Ambient central differences of the polynomial form.
        let fd = |i: usize, dx: f64, dy: f64, dz: f64| -> f64 {
            let a = sh64(x + h * dx, y + h * dy, z + h * dz)[i];
            let b = sh64(x - h * dx, y - h * dy, z - h * dz)[i];
            (a - b) / (2.0 * h)
        };

        for i in 0..N {
            for (g, (dx, dy, dz)) in [
                (ddx[i].extract::<0>(), (1.0, 0.0, 0.0)),
                (ddy[i].extract::<0>(), (0.0, 1.0, 0.0)),
                (ddz[i].extract::<0>(), (0.0, 0.0, 1.0)),
            ] {
                let want = fd(i, dx, dy, dz);
                assert!(
                    (g - want).abs() < 1e-7 * (1.0 + want.abs()),
                    "grad[{i}] at ({x},{y},{z}): got {g}, FD {want}"
                );
            }
        }
    }
}

/// The Condon-Shortley (`CS = true`) path against Sloan's published `SHEval3` (_Efficient
/// Spherical Harmonic Evaluation_, JCGT 2(2), 2013, Listing 2), transcribed from the
/// paper. An external reference for the phased convention, and a second independent
/// confirmation of the unphased basis too, because Listing 2 agrees with our
/// unphased values on even `|m|` and negates only odd `|m|`.
#[test]
fn sh_condon_shortley_vs_sloan_sheval3() {
    const L3: usize = 2;
    const N3: usize = 9;

    let dirs: &[(f64, f64, f64)] = &[
        (0.267261241912424, 0.534522483824849, 0.801783725737273),
        (-0.6, 0.0, 0.8),
        (0.0, 0.0, 1.0),
        (0.5773502691896258, -0.5773502691896258, 0.5773502691896258),
    ];

    for &(x, y, z) in dirs {
        // --- Listing 2, verbatim ---
        let mut p = [0.0f64; N3];
        let z2 = z * z;
        p[0] = 0.2820947917738781;
        p[2] = 0.4886025119029199 * z;
        p[6] = 0.9461746957575601 * z2 + -0.3153915652525201;
        let (c0, s0) = (x, y);
        let tmp_a = -0.48860251190292;
        p[3] = tmp_a * c0;
        p[1] = tmp_a * s0;
        let tmp_b = -1.092548430592079 * z;
        p[7] = tmp_b * c0;
        p[5] = tmp_b * s0;
        let (c1, s1) = (x * c0 - y * s0, x * s0 + y * c0);
        let tmp_c = 0.5462742152960395;
        p[8] = tmp_c * c1;
        p[4] = tmp_c * s1;

        let mut cs = [V64::splat(0.0); N3];
        V64::spherical_harmonics::<L3, N3, true>(V64::splat(x), V64::splat(y), V64::splat(z), &mut cs);

        let mut np = [V64::splat(0.0); N3];
        V64::spherical_harmonics::<L3, N3, false>(V64::splat(x), V64::splat(y), V64::splat(z), &mut np);

        for i in 0..N3 {
            let got = cs[i].extract::<0>();
            assert!(
                (got - p[i]).abs() < 1e-13,
                "CS sh[{i}] at ({x},{y},{z}): got {got}, SHEval3 {}",
                p[i]
            );

            // ... and the two conventions differ by exactly (-1)^|m|.
            let l = (i as f64).sqrt() as usize;
            let m_abs = (i as i64 - (l * (l + 1)) as i64).unsigned_abs();
            let sign = if m_abs % 2 == 1 { -1.0 } else { 1.0 };
            assert_eq!(got, sign * np[i].extract::<0>(), "phase relation broken at [{i}]");
        }
    }
}

/// The Condon-Shortley table must flip gradients as well as values. The `f` ratios
/// cross between adjacent columns, whose signs always disagree, so a build that only
/// negated the diagonal seeds would pass every value test and produce a wrong
/// `d/dz`. Finite differences against the phased values catch exactly that.
#[test]
fn sh_condon_shortley_gradients() {
    let mut rng = Rng(0xC047_D047_5EED_1234);
    let h = 1e-5;

    let eval_cs = |x: f64, y: f64, z: f64| -> [f64; N] {
        let mut o = [V64::splat(0.0); N];
        V64::spherical_harmonics::<L, N, true>(V64::splat(x), V64::splat(y), V64::splat(z), &mut o);
        let mut r = [0.0; N];
        for i in 0..N {
            r[i] = o[i].extract::<0>();
        }
        r
    };

    for _ in 0..25 {
        let (x, y, z) = rng.unit();

        let mut o = [V64::splat(0.0); N];
        let (mut gx, mut gy, mut gz) = ([V64::splat(0.0); N], [V64::splat(0.0); N], [V64::splat(0.0); N]);
        V64::spherical_harmonics_d::<L, N, true>(
            V64::splat(x),
            V64::splat(y),
            V64::splat(z),
            &mut o,
            &mut gx,
            &mut gy,
            &mut gz,
        );

        for i in 0..N {
            for (g, (dx, dy, dz)) in [
                (gx[i].extract::<0>(), (1.0, 0.0, 0.0)),
                (gy[i].extract::<0>(), (0.0, 1.0, 0.0)),
                (gz[i].extract::<0>(), (0.0, 0.0, 1.0)),
            ] {
                let a = eval_cs(x + h * dx, y + h * dy, z + h * dz)[i];
                let b = eval_cs(x - h * dx, y - h * dy, z - h * dz)[i];
                let want = (a - b) / (2.0 * h);
                assert!(
                    (g - want).abs() < 1e-7 * (1.0 + want.abs()),
                    "CS grad[{i}] at ({x},{y},{z}): got {g}, FD {want}"
                );
            }
        }
    }
}

#[test]
fn sh_public_trait_surface() {
    use thermite_special::{RealSpecialMathWithPolicy, ScalarSpecialMath};

    let (x, y, z) = (0.267261241912424, 0.534522483824849, 0.801783725737273);
    let want = sh64(x, y, z);

    // Default-policy trait method.
    let mut out = [V64::splat(0.0); N];
    V64::spherical_harmonics::<L, N, false>(V64::splat(x), V64::splat(y), V64::splat(z), &mut out);
    for i in 0..N {
        assert_eq!(out[i].extract::<0>(), want[i], "trait value [{i}] diverges from kernel");
    }

    // Policy variant.
    let mut out_p = [V64::splat(0.0); N];
    V64::spherical_harmonics_p::<DefaultPolicy, L, N, false>(
        V64::splat(x),
        V64::splat(y),
        V64::splat(z),
        &mut out_p,
    );
    assert_eq!(out_p[5].extract::<0>(), want[5]);

    // Scalar aggregate (exercises the &mut [E; N] Unwrap reinterpret).
    let mut out_s = [0.0f64; N];
    f64::scalar_spherical_harmonics::<L, N, false>(x, y, z, &mut out_s);
    for i in 0..N {
        assert_eq!(out_s[i], want[i], "scalar value [{i}] diverges from kernel");
    }

    // Scalar table + prebuilt-table eval (exercises the &ShTable Unwrap reinterpret).
    // A scalar is its own primal, so the table is plain `ShTable<f64, N>`.
    // `zeroed()` needs a vector type, so a bare-element table is built literally.
    let mut table_s = ShTable::<f64, N> {
        qmm: [0.0; N],
        em: [0.0; N],
        a: [0.0; N],
        nb: [0.0; N],
        f: [0.0; N],
        mf: [0.0; N],
    };
    f64::scalar_spherical_harmonics_table::<L, N, false>(&mut table_s);
    let mut out_ts = [0.0f64; N];
    f64::scalar_spherical_harmonics_with::<L, N>(&table_s, x, y, z, &mut out_ts);
    for i in 0..N {
        assert_eq!(out_ts[i], want[i], "scalar table value [{i}] diverges from kernel");
    }

    // Gradient surface, value slots only (FD already validates the derivatives).
    let mut o = [V64::splat(0.0); N];
    let (mut gx, mut gy, mut gz) = ([V64::splat(0.0); N], [V64::splat(0.0); N], [V64::splat(0.0); N]);
    V64::spherical_harmonics_d::<L, N, false>(
        V64::splat(x),
        V64::splat(y),
        V64::splat(z),
        &mut o,
        &mut gx,
        &mut gy,
        &mut gz,
    );
    assert_eq!(o[7].extract::<0>(), want[7]);

    let mut os = [0.0f64; N];
    let (mut gxs, mut gys, mut gzs) = ([0.0f64; N], [0.0f64; N], [0.0f64; N]);
    f64::scalar_spherical_harmonics_d::<L, N, false>(x, y, z, &mut os, &mut gxs, &mut gys, &mut gzs);
    assert_eq!(os[7], want[7]);
    assert_eq!(gzs[2], gz[2].extract::<0>());
}

#[test]
fn sh_f32_matches_f64() {
    let mut rng = Rng(0xF00D_F00D_F00D_F00D);

    for _ in 0..100 {
        let (x, y, z) = rng.unit();

        let mut out32 = [V32::splat(0.0); N];
        sh_impl::<DefaultPolicy, f32, V32, L, N, false>(
            V32::splat(x as f32),
            V32::splat(y as f32),
            V32::splat(z as f32),
            &mut out32,
        );

        let got64 = sh64(x, y, z);

        for i in 0..N {
            let v = out32[i].extract::<0>() as f64;
            // f32 inputs alone cost ~1e-7 relative, and the recurrence adds little.
            assert!(
                (v - got64[i]).abs() < 3e-6,
                "f32 Y[{i}] at ({x},{y},{z}): got {v}, want {}",
                got64[i]
            );
        }
    }
}

/// The two lowerings against each other. `sh_impl` reads coefficients folded into the
/// instruction stream at compile time, while `sh_table_impl` derives the same numbers
/// from their closed forms in `l` and `m` at runtime. Same recurrence, entirely
/// different provenance, so a transcription slip in either the const builder or the
/// runtime one shows up here even though both would pass the analytic tests on their
/// own.
///
/// The agreement is tight (a few ulp) rather than exact: `csqrt`'s const-eval Newton
/// iteration and the hardware `sqrt` need not round identically.
#[test]
fn sh_fast_vs_general_lowering() {
    let mut rng = Rng(0xD1FF_D1FF_1010_1010);

    for cs in [false, true] {
        for _ in 0..50 {
            let (x, y, z) = rng.unit();
            let (vx, vy, vz) = (V64::splat(x), V64::splat(y), V64::splat(z));

            let mut fast = [V64::splat(0.0); N];
            let mut table = ShTable::<V64, N>::zeroed();
            let mut slow = [V64::splat(0.0); N];

            // The unrolled kernel takes CS as a const, so both arms are spelled out.
            if cs {
                sh_impl::<DefaultPolicy, f64, V64, L, N, true>(vx, vy, vz, &mut fast);
                sh_table_impl::<V64, L, N, true>(&mut table);
            } else {
                sh_impl::<DefaultPolicy, f64, V64, L, N, false>(vx, vy, vz, &mut fast);
                sh_table_impl::<V64, L, N, false>(&mut table);
            }
            sh_eval_impl::<V64, L, N>(&table, vx, vy, vz, &mut slow);

            for i in 0..N {
                let (a, b) = (fast[i].extract::<0>(), slow[i].extract::<0>());
                assert!(
                    (a - b).abs() < 1e-14 * (1.0 + a.abs()),
                    "cs={cs} lowering mismatch at [{i}] ({x},{y},{z}): unrolled {a}, general {b}"
                );
            }
        }
    }
}

/// Gradients through the general path, diffed against the unrolled gradient kernel.
#[test]
fn sh_fast_vs_general_gradients() {
    let mut rng = Rng(0x9999_1234_ABCD_0F0F);

    for _ in 0..25 {
        let (x, y, z) = rng.unit();
        let (vx, vy, vz) = (V64::splat(x), V64::splat(y), V64::splat(z));

        let mut fo = [V64::splat(0.0); N];
        let (mut fx, mut fy, mut fz) = ([V64::splat(0.0); N], [V64::splat(0.0); N], [V64::splat(0.0); N]);
        sh_d_impl::<DefaultPolicy, f64, V64, L, N, false>(vx, vy, vz, &mut fo, &mut fx, &mut fy, &mut fz);

        let mut table = ShTable::<V64, N>::zeroed();
        sh_table_impl::<V64, L, N, false>(&mut table);
        let mut so = [V64::splat(0.0); N];
        let (mut sx, mut sy, mut sz) = ([V64::splat(0.0); N], [V64::splat(0.0); N], [V64::splat(0.0); N]);
        sh_eval_d_impl::<V64, L, N>(&table, vx, vy, vz, &mut so, &mut sx, &mut sy, &mut sz);

        for i in 0..N {
            for (label, a, b) in [
                ("value", fo[i].extract::<0>(), so[i].extract::<0>()),
                ("d/dx", fx[i].extract::<0>(), sx[i].extract::<0>()),
                ("d/dy", fy[i].extract::<0>(), sy[i].extract::<0>()),
                ("d/dz", fz[i].extract::<0>(), sz[i].extract::<0>()),
            ] {
                assert!(
                    (a - b).abs() < 1e-13 * (1.0 + a.abs()),
                    "{label} mismatch at [{i}] ({x},{y},{z}): unrolled {a}, general {b}"
                );
            }
        }
    }
}

/// Above `MAX_SH_DEGREE` the unrolled ladder does not exist, so the trait method must
/// route to the general path. Checked against the independent spherical-coordinate
/// oracle, which shares no code with either lowering.
#[test]
fn sh_beyond_max_degree() {
    const LB: usize = MAX_SH_DEGREE + 3;
    const NB: usize = (LB + 1) * (LB + 1);

    let mut rng = Rng(0x2222_7777_BEEF_5A5A);

    for _ in 0..5 {
        let (x, y, z) = rng.unit();

        let mut out = [V64::splat(0.0); NB];
        V64::spherical_harmonics::<LB, NB, false>(V64::splat(x), V64::splat(y), V64::splat(z), &mut out);

        for l in 0..=LB {
            for m in -(l as i64)..=(l as i64) {
                let got = out[((l * (l + 1)) as i64 + m) as usize].extract::<0>();
                let want = oracle(l, m, x, y, z);
                assert!(
                    (got - want).abs() < 1e-10 * (1.0 + want.abs()),
                    "L={LB} Y({l},{m}) at ({x},{y},{z}): got {got}, oracle {want}"
                );
            }
        }
    }
}

/// The `f32`/`f64` override builds its table by splatting the compile-time constants
/// rather than recomputing them. That shortcut has to agree with the arithmetic it
/// replaces, or the hoisted-table path would silently disagree with the one-shot one.
#[test]
fn sh_table_override_matches_computed() {
    let mut computed = ShTable::<V64, N>::zeroed();
    sh_table_impl::<V64, L, N, false>(&mut computed);

    let mut splatted = ShTable::<V64, N>::zeroed();
    V64::spherical_harmonics_table::<L, N, false>(&mut splatted);

    for (name, a, b) in [
        ("qmm", &computed.qmm, &splatted.qmm),
        ("em", &computed.em, &splatted.em),
        ("a", &computed.a, &splatted.a),
        ("nb", &computed.nb, &splatted.nb),
        ("f", &computed.f, &splatted.f),
        ("mf", &computed.mf, &splatted.mf),
    ] {
        for i in 0..N {
            let (x, y) = (a[i].extract::<0>(), b[i].extract::<0>());
            assert!(
                (x - y).abs() < 1e-14 * (1.0 + x.abs()),
                "table.{name}[{i}]: computed {x}, splatted {y}"
            );
        }
    }
}

/// The hoisted table has to produce bit-identical results to the one-shot call, or
/// callers would see the answer change depending on how they chose to structure
/// their loop.
#[test]
fn sh_hoisted_table_matches_one_shot() {
    let mut rng = Rng(0x5A5A_0F0F_3C3C_1E1E);

    let mut table = ShTable::<V64, N>::zeroed();
    V64::spherical_harmonics_table::<L, N, true>(&mut table);

    for _ in 0..25 {
        let (x, y, z) = rng.unit();
        let (vx, vy, vz) = (V64::splat(x), V64::splat(y), V64::splat(z));

        let mut one_shot = [V64::splat(0.0); N];
        V64::spherical_harmonics::<L, N, true>(vx, vy, vz, &mut one_shot);

        let mut hoisted = [V64::splat(0.0); N];
        V64::spherical_harmonics_with::<L, N>(&table, vx, vy, vz, &mut hoisted);

        for i in 0..N {
            let (a, b) = (one_shot[i].extract::<0>(), hoisted[i].extract::<0>());
            assert!(
                (a - b).abs() < 1e-14 * (1.0 + a.abs()),
                "hoisted vs one-shot at [{i}] ({x},{y},{z}): {a} vs {b}"
            );
        }
    }
}

/// `sh_eval_mixed_impl` against `sh_eval_impl` on the same real table.
///
/// The mixed evaluator exists so a composite can multiply by _real_ coefficients, but
/// its type bound is satisfied by a plain vector too (`W = R = V`), which makes the
/// two directly diffable. They are separate bodies (the single-type one folds its
/// recurrence into `mul_adde` where the mixed one cannot), so this is what keeps them
/// from drifting apart.
#[test]
fn sh_mixed_evaluator_matches_single_type() {
    use thermite_special::specialized::sh_eval_mixed_impl;

    let mut rng = Rng(0x0E11_2A7C_5F03_9B6D);

    let mut table = ShTable::<V64, N>::zeroed();
    sh_table_impl::<V64, L, N, false>(&mut table);

    for _ in 0..50 {
        let (x, y, z) = rng.unit();
        let (vx, vy, vz) = (V64::splat(x), V64::splat(y), V64::splat(z));

        let mut single = [V64::splat(0.0); N];
        sh_eval_impl::<V64, L, N>(&table, vx, vy, vz, &mut single);

        let mut mixed = [V64::splat(0.0); N];
        sh_eval_mixed_impl::<V64, V64, L, N>(&table, vx, vy, vz, &mut mixed);

        for i in 0..N {
            let (a, b) = (single[i].extract::<0>(), mixed[i].extract::<0>());
            assert!(
                (a - b).abs() < 1e-14 * (1.0 + a.abs()),
                "evaluator mismatch at [{i}] ({x},{y},{z}): single-type {a}, mixed {b}"
            );
        }
    }
}
