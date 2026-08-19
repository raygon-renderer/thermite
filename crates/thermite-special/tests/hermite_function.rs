//! Orthonormal Hermite functions: `hermite_function::<N>` and `hermite_function_series::<N>`.
//!
//! References come from `scripts/orthonormal_ref.py` (mpmath, 50 digits) and include
//! degrees 50 to 1000 - far past where the raw polynomial is finite in either format,
//! which is the whole reason the normalized kernel exists. The functions are `O(1)`
//! everywhere, so tolerances here are absolute, scaled only by the growth the forward
//! recurrence and the Gaussian seed are entitled to.
//!
//! Coverage: the mpmath table at low degree, the high-degree rows (inside, at and past
//! the turning point), the definitional cross-check against `hermite` where both are
//! finite, which ties the two kernels together, parity and the origin, the series against
//! the singles and against a unit coefficient, the Precision-tier residual at large `x`,
//! lane independence, and f32 including high degree inside its documented range.
#![cfg(any(
    target_arch = "x86",
    target_arch = "x86_64",
    all(target_arch = "wasm32", feature = "wasm"),
    target_arch = "aarch64"
))]

use thermite::math::policy::policies::Precision;
use thermite::prelude::*;
use thermite_special::{SpecialMath, SpecialMathWithPolicy};

type D = Vector<f64>;
type F = Vector<f32>;

include!("hermite_ref/table.rs");

#[track_caller]
fn close_abs(name: &str, got: f64, want: f64, tol: f64) {
    let err = (got - want).abs();
    assert!(err <= tol, "{name}: got {got:?}, want {want:?} (err {err:e} > {tol:e})");
}

fn col(x: f64) -> usize {
    HF_XS.iter().position(|&t| t == x).expect("x is on the table grid")
}

/// The table rows, one const instantiation each.
macro_rules! for_each_degree {
    ($mac:ident) => {
        $mac!(0);
        $mac!(1);
        $mac!(2);
        $mac!(3);
        $mac!(4);
        $mac!(5);
        $mac!(6);
        $mac!(7);
        $mac!(8);
        $mac!(9);
        $mac!(10);
        $mac!(11);
        $mac!(12);
    };
}

/// The high-degree spot checks, one const instantiation per degree in `HF_HIGH`.
fn psi_at(n: usize, x: f64) -> f64 {
    let v = D::splat(x);
    match n {
        50 => v.hermite_function::<50>(),
        100 => v.hermite_function::<100>(),
        150 => v.hermite_function::<150>(),
        300 => v.hermite_function::<300>(),
        1000 => v.hermite_function::<1000>(),
        _ => unreachable!("no const instantiation for degree {n}"),
    }
    .extract::<0>()
}

fn psi_at_f32(n: usize, x: f32) -> f32 {
    let v = F::splat(x);
    match n {
        50 => v.hermite_function::<50>(),
        100 => v.hermite_function::<100>(),
        150 => v.hermite_function::<150>(),
        300 => v.hermite_function::<300>(),
        1000 => v.hermite_function::<1000>(),
        _ => unreachable!("no const instantiation for degree {n}"),
    }
    .extract::<0>()
}

#[test]
fn matches_mpmath_at_low_degree() {
    macro_rules! check {
        ($n:literal) => {
            for (j, &x) in HF_XS.iter().enumerate() {
                let got = D::splat(x).hermite_function::<$n>().extract::<0>();
                close_abs(&format!("psi_{}({x})", $n), got, HF[$n][j], 4e-16);
            }
        };
    }
    for_each_degree!(check);
}

/// Absolute error budget for `psi_n(x)` under the default policy: the forward recurrence
/// at about `n eps / 5` (measured 4e-17 per step at worst against mpmath, degrees 50 to
/// 1000), plus the seed's `x^2/4 eps` from rounding `x*x`, which the Precision tier removes
/// on true-FMA hardware but the scalar backend in a default build does not have.
fn budget(n: usize, x: f64, eps: f64) -> f64 {
    eps / 5.0 * n as f64 + eps * x * x / 4.0
}

#[test]
fn reaches_degrees_the_raw_polynomial_cannot() {
    // Degrees 50..1000, in the oscillatory region, at the turning point, and in the tail.
    // Raw H_n overflows binary64 around degree 300 at the origin and much earlier away
    // from it, and these rows are the reason the normalized kernel exists.
    for &(n, x, want) in HF_HIGH.iter() {
        let got = psi_at(n, x);
        close_abs(&format!("psi_{n}({x})"), got, want, budget(n, x, f64::EPSILON));
        assert!(
            got != 0.0 || want == 0.0,
            "psi_{n}({x}) collapsed to zero, want {want:e}"
        );
    }
}

/// The Best-tier Dekker residual on `x^2` is gated on `HAS_TRUE_FMA`, which the scalar
/// `Vector<f64>` does not report in a default build, though the AVX2 backend does. Measured on it,
/// the residual takes `psi_300(20.7)` from 49 to 2 ulp and `psi_1000(38.9)` from 132 to
/// under 1, and does nothing at integer `x`, whose square is exact. So this test runs on
/// the wide backend and only over the rows whose `x` has a full mantissa.
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[test]
fn precision_policy_recovers_the_gaussian_residual_on_fma_hardware() {
    use thermite::math::policy::policies::Performance;
    use thermite::simd::Simd;
    type W = Vector<<thermite::backend::x86_v3::X86V3 as Simd>::f64x4>;

    macro_rules! at {
        ($p:ty, $n:expr, $x:expr) => {
            match $n {
                150 => W::splat($x).hermite_function_p::<$p, 150>(),
                300 => W::splat($x).hermite_function_p::<$p, 300>(),
                1000 => W::splat($x).hermite_function_p::<$p, 1000>(),
                _ => unreachable!(),
            }
            .extract::<0>()
        };
    }

    let (mut sum_perf, mut sum_prec, mut rows) = (0.0f64, 0.0f64, 0);
    for &(n, x, want) in HF_HIGH.iter() {
        if x.fract() == 0.0 {
            continue; // exact square: nothing to recover
        }
        let u = f64::EPSILON * want.abs();
        sum_perf += (at!(Performance, n, x) - want).abs() / u;
        sum_prec += (at!(Precision, n, x) - want).abs() / u;
        rows += 1;
    }
    assert!(rows >= 4, "expected the full-mantissa seed-stress rows in HF_HIGH");

    // Mean rather than max: the recurrence's own error is a random walk of ~100 ulp at
    // degree 1000 and can land either way on any one row, while the seed correction is a
    // consistent shift underneath it.
    let (mean_perf, mean_prec) = (sum_perf / rows as f64, sum_prec / rows as f64);
    assert!(
        mean_prec * 1.5 < mean_perf,
        "Precision mean {mean_prec:.1} ulp should be clearly under Performance's {mean_perf:.1}"
    );
}

#[test]
fn agrees_with_the_raw_polynomial_where_both_are_finite() {
    // psi_n = H_n e^{-x^2/2} / sqrt(2^n n! sqrt(pi)): the definitional cross-check that
    // ties the two kernels together at degrees where the raw one is still finite.
    let pi_qtr = core::f64::consts::PI.sqrt().sqrt();
    macro_rules! check {
        ($n:literal) => {{
            let mut norm = 1.0f64; // 2^n n!
            for k in 1..=($n as u64) {
                norm *= 2.0 * k as f64;
            }
            let norm = norm.sqrt() * pi_qtr;
            for &x in &HF_XS {
                let raw = D::splat(x).hermite::<$n>().extract::<0>();
                let want = raw * (-0.5 * x * x).exp() / norm;
                let got = D::splat(x).hermite_function::<$n>().extract::<0>();
                // The raw route cancels through the polynomial and its own exp, so the bound
                // is set by its conditioning, not by the function kernel's.
                close_abs(&format!("psi_{} vs raw at {x}", $n), got, want, 1e-12);
            }
        }};
    }
    for_each_degree!(check);
}

#[test]
fn parity_and_the_origin() {
    // psi_n(-x) = (-1)^n psi_n(x) bit for bit (the kernel squares x), and psi_n(0) = 0
    // for odd n exactly (the seed is even and the recurrence multiplies by x).
    macro_rules! check {
        ($n:literal) => {
            for &x in &HF_XS {
                let pos = D::splat(x).hermite_function::<$n>().extract::<0>();
                let neg = D::splat(-x).hermite_function::<$n>().extract::<0>();
                let want = if $n % 2 == 0 { pos } else { -pos };
                assert_eq!(neg, want, "parity psi_{}({x})", $n);
            }
            if $n % 2 == 1 {
                assert_eq!(
                    D::ZERO.hermite_function::<$n>().extract::<0>(),
                    0.0,
                    "psi_{}(0)",
                    $n
                );
            }
        };
    }
    for_each_degree!(check);
}

#[test]
fn series_sums_the_single_functions() {
    // The unit-coefficient identity (series == single) and a full sum against the table.
    // Ties Clenshaw to the forward kernel, which is tied to mpmath above.
    const N: usize = 13;
    let mut c = [0.0f64; N];
    for (k, slot) in c.iter_mut().enumerate() {
        *slot = if k % 2 == 0 { 1.0 } else { -1.0 } * 0.8f64.powi(k as i32);
    }

    for &x in &HF_XS {
        let v = D::splat(x);
        let j = col(x);

        let mut want = 0.0;
        for k in 0..N {
            let mut unit = [0.0f64; N];
            unit[k] = 1.0;
            let solo = v.hermite_function_series::<N>(&unit).extract::<0>();
            close_abs(&format!("unit series k={k} at {x}"), solo, HF[k][j], 4e-16);
            want += c[k] * HF[k][j];
        }

        let got = v.hermite_function_series::<N>(&c).extract::<0>();
        close_abs(&format!("series at {x}"), got, want, 2e-15);
    }
}

#[test]
fn series_short_forms() {
    // N = 1 and N = 2 return before the loop.
    for &x in &HF_XS {
        let v = D::splat(x);
        let j = col(x);
        let one = [2.5];
        close_abs(
            "N=1",
            v.hermite_function_series::<1>(&one).extract::<0>(),
            2.5 * HF[0][j],
            1e-15,
        );
        let two = [2.5, -1.25];
        close_abs(
            "N=2",
            v.hermite_function_series::<2>(&two).extract::<0>(),
            2.5 * HF[0][j] - 1.25 * HF[1][j],
            1e-15,
        );
    }
}

#[test]
fn series_at_high_degree_stays_in_range() {
    // A 101-term series with a single non-zero top coefficient is psi_100 itself, so the
    // Clenshaw path has to survive the same range the forward kernel does.
    let mut c = [0.0f64; 101];
    c[100] = 1.0;
    for &(n, x, want) in HF_HIGH.iter() {
        if n != 100 {
            continue;
        }
        let got = D::splat(x).hermite_function_series::<101>(&c).extract::<0>();
        // Clenshaw's own rounding is comparable to the forward kernel's, so give it 2x.
        close_abs(
            &format!("series psi_100({x})"),
            got,
            want,
            2.0 * budget(n, x, f64::EPSILON),
        );
    }
}

#[test]
fn lanes_stay_independent() {
    use thermite::backend::scalar::Scalar;
    type D4 = thermite::simd::f64x4<Scalar>;
    let xs = [HF_XS[0], HF_XS[3], HF_XS[6], HF_XS[9]];
    let got = D4::new(xs).hermite_function::<7>();
    for (lane, &x) in xs.iter().enumerate() {
        assert_eq!(
            got.as_slice()[lane],
            D::splat(x).hermite_function::<7>().extract::<0>(),
            "lane {lane}"
        );
    }
}

#[test]
fn f32_tracks_the_reference_including_high_degree() {
    macro_rules! check {
        ($n:literal) => {
            for (j, &x) in HF_XS.iter().enumerate() {
                let got = F::splat(x as f32).hermite_function::<$n>().extract::<0>() as f64;
                close_abs(&format!("f32 psi_{}({x})", $n), got, HF[$n][j], 3e-7);
            }
        };
    }
    for_each_degree!(check);

    // Raw H_n leaves binary32 around degree 48 at the origin, while psi_150 at x = 17 is well
    // inside the documented f32 range (|x| < 18.7).
    for &(n, x, want) in HF_HIGH.iter() {
        if x > 18.0 {
            continue; // documented edge of the binary32 range
        }
        let got = psi_at_f32(n, x as f32) as f64;
        // Same budget at binary32 resolution. The argument itself is `x as f32`, and for
        // the full-mantissa rows that is not the x the reference was taken at, so skip those
        // here (their job is the f64 residual test) rather than fold a slope term in.
        if x.fract() != 0.0 {
            continue;
        }
        close_abs(
            &format!("f32 psi_{n}({x})"),
            got,
            want,
            budget(n, x, f32::EPSILON as f64),
        );
        assert!(got != 0.0 || want == 0.0, "f32 psi_{n}({x}) collapsed to zero");
    }
}
