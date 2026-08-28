//! Orthonormal generalized Laguerre functions: `laguerre_function::<N>` and
//! `laguerre_function_series_n::<N>`.
//!
//! References come from `scripts/orthonormal_ref.py` (mpmath, three-term recurrence at 400
//! digits) over five weights including a half-integer and a negative one, and include
//! degrees 40 to 600 at arguments up to 2300 - where the raw polynomial has long since
//! overflowed. The functions are `O(1)` on the half-line, so tolerances are absolute.
//!
//! Coverage: the mpmath table at low degree over all weights, the high-degree rows inside,
//! near and past the turning point `4n`, the definitional cross-check against `laguerre`
//! where both are finite, the origin (`l_n^{(0)}(0) = 1`, `l_n^{(alpha > 0)}(0) = 0`), the
//! series against the singles and a unit coefficient, per-lane weights, and f32.
#![cfg(any(
    target_arch = "x86",
    target_arch = "x86_64",
    all(target_arch = "wasm32", feature = "wasm"),
    target_arch = "aarch64"
))]

use thermite::prelude::*;
use thermite_special::SpecialMath;

type D = Vector<f64>;
type F = Vector<f32>;

include!("laguerre_ref/table.rs");

#[track_caller]
fn close_abs(name: &str, got: f64, want: f64, tol: f64) {
    let err = (got - want).abs();
    assert!(err <= tol, "{name}: got {got:?}, want {want:?} (err {err:e} > {tol:e})");
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
    };
}

fn lf_at(n: usize, alpha: f64, x: f64) -> f64 {
    let (v, a) = (D::splat(x), D::splat(alpha));
    match n {
        0 => v.laguerre_function::<0>(a),
        3 => v.laguerre_function::<3>(a),
        8 => v.laguerre_function::<8>(a),
        40 => v.laguerre_function::<40>(a),
        80 => v.laguerre_function::<80>(a),
        300 => v.laguerre_function::<300>(a),
        600 => v.laguerre_function::<600>(a),
        _ => unreachable!("no const instantiation for degree {n}"),
    }
    .extract::<0>()
}

fn lf_at_f32(n: usize, alpha: f32, x: f32) -> f32 {
    let (v, a) = (F::splat(x), F::splat(alpha));
    match n {
        0 => v.laguerre_function::<0>(a),
        3 => v.laguerre_function::<3>(a),
        8 => v.laguerre_function::<8>(a),
        40 => v.laguerre_function::<40>(a),
        80 => v.laguerre_function::<80>(a),
        300 => v.laguerre_function::<300>(a),
        600 => v.laguerre_function::<600>(a),
        _ => unreachable!("no const instantiation for degree {n}"),
    }
    .extract::<0>()
}

/// Absolute error budget: the recurrence at about `n eps / 4` per step (its coefficients
/// are runtime `sqrt`s and reciprocals rather than literals, one more rounding than the
/// Hermite kernel), plus the seed's exponent. That exponent is
/// `alpha/2 ln x - lgamma(alpha+1)/2 - x/4`, and an absolute error there is a relative
/// error in the result, so each term contributes its own magnitude times eps. The
/// `alpha ln x` term is why the large-alpha rows need more room than `x` alone would give.
fn budget(n: usize, alpha: f64, x: f64, eps: f64) -> f64 {
    let expo = alpha / 2.0 * x.max(1.0).ln() + alpha.max(1.0) * alpha.max(1.0).ln() / 2.0 + x / 4.0;
    eps / 4.0 * n as f64 + eps * (expo + 4.0)
}

#[test]
fn matches_mpmath_at_low_degree_over_all_weights() {
    macro_rules! check {
        ($n:literal) => {
            for (i, &alpha) in LF_ALPHAS.iter().enumerate() {
                for (j, &x) in LF_XS.iter().enumerate() {
                    let want = LF[i][$n][j];
                    if !want.is_finite() {
                        continue; // alpha < 0 at x = 0 is a genuine pole
                    }
                    let got = D::splat(x)
                        .laguerre_function::<$n>(D::splat(alpha))
                        .extract::<0>();
                    close_abs(
                        &format!("l_{}^({alpha})({x})", $n),
                        got,
                        want,
                        budget($n, alpha, x, f64::EPSILON),
                    );
                }
            }
        };
    }
    for_each_degree!(check);
}

#[test]
fn reaches_degrees_and_arguments_the_raw_polynomial_cannot() {
    // Raw L_n(x) grows like x^n / n!: at (80, 330) that is 10^{200-118}, past binary32 by
    // forty orders of magnitude and past binary64's comfort too. These are the rows the
    // normalized kernel exists for.
    for &(n, alpha, x, want) in LF_HIGH.iter() {
        let got = lf_at(n, alpha, x);
        close_abs(
            &format!("l_{n}^({alpha})({x})"),
            got,
            want,
            budget(n, alpha, x, f64::EPSILON),
        );
        assert!(
            got != 0.0 || want == 0.0,
            "l_{n}^({alpha})({x}) collapsed to zero, want {want:e}"
        );
    }
}

#[test]
fn agrees_with_the_raw_polynomial_where_both_are_finite() {
    // l_n^a = sqrt(n! / Gamma(n+a+1)) x^{a/2} e^{-x/2} L_n^a. Integer alpha keeps the
    // Gamma ratio an exact small rational, which ties the two kernels together.
    macro_rules! check {
        ($n:literal) => {
            for &alpha in &[0.0f64, 1.0, 2.0] {
                // n! / (n+alpha)! = 1 / ((n+1)(n+2)...(n+alpha))
                let mut ratio = 1.0f64;
                for k in 1..=(alpha as u64) {
                    ratio /= ($n as f64) + k as f64;
                }
                let norm = ratio.sqrt();
                for &x in &LF_XS {
                    let raw = D::splat(x).laguerre::<$n>(D::splat(alpha)).extract::<0>();
                    let want = norm * x.powf(alpha / 2.0) * (-0.5 * x).exp() * raw;
                    let got = D::splat(x)
                        .laguerre_function::<$n>(D::splat(alpha))
                        .extract::<0>();
                    // The raw route cancels through the polynomial, so bound by its scale.
                    close_abs(
                        &format!("l_{}^({alpha}) vs raw at {x}", $n),
                        got,
                        want,
                        1e-12 * raw.abs().max(1.0),
                    );
                }
            }
        };
    }
    for_each_degree!(check);
}

#[test]
fn the_origin_is_right_for_the_ordinary_functions() {
    // l_n^{(0)}(0) = sqrt(n!/n!) * 1 * 1 * L_n(0) = 1 exactly, and l_n^{(a)}(0) = 0 for a > 0
    // (the x^{a/2} factor). The seed's exponent contains alpha/2 * ln x, which is NaN at
    // (x = 0, alpha = 0) and -inf above it, so both are patched to their limits. The
    // alpha = 0 log-normalization is separately pinned to exactly 0 rather than
    // -lgamma(1)/2, which is a few ulp off (lgamma is not exactly 0 at its zeros). With
    // s_k = k+1 exact and reciprocals of small integers, the whole recurrence at
    // (x = 0, alpha = 0) is then exact.
    macro_rules! check {
        ($n:literal) => {{
            let one = D::ZERO.laguerre_function::<$n>(D::ZERO).extract::<0>();
            assert_eq!(one, 1.0, "l_{}^(0)(0)", $n);
            for &alpha in &[0.5f64, 1.0, 2.0] {
                assert_eq!(
                    D::ZERO.laguerre_function::<$n>(D::splat(alpha)).extract::<0>(),
                    0.0,
                    "l_{}^({alpha})(0)",
                    $n
                );
            }
        }};
    }
    for_each_degree!(check);
}

#[test]
fn series_sums_the_single_functions() {
    const N: usize = 9;
    let mut c = [0.0f64; N];
    for (k, slot) in c.iter_mut().enumerate() {
        *slot = if k % 2 == 0 { 1.0 } else { -1.0 } * 0.8f64.powi(k as i32);
    }

    for (i, &alpha) in LF_ALPHAS.iter().enumerate() {
        for (j, &x) in LF_XS.iter().enumerate() {
            if !LF[i][0][j].is_finite() {
                continue;
            }
            let (v, a) = (D::splat(x), D::splat(alpha));

            let mut want = 0.0;
            for k in 0..N {
                let mut unit = [0.0f64; N];
                unit[k] = 1.0;
                let solo = v.laguerre_function_series_n::<N>(a, &unit).extract::<0>();
                close_abs(
                    &format!("unit series k={k} a={alpha} x={x}"),
                    solo,
                    LF[i][k][j],
                    budget(k, alpha, x, f64::EPSILON) * 2.0,
                );
                want += c[k] * LF[i][k][j];
            }

            let got = v.laguerre_function_series_n::<N>(a, &c).extract::<0>();
            close_abs(
                &format!("series a={alpha} x={x}"),
                got,
                want,
                budget(N, alpha, x, f64::EPSILON) * 2.0,
            );
        }
    }
}

#[test]
fn series_short_forms() {
    for (i, &alpha) in LF_ALPHAS.iter().enumerate() {
        for (j, &x) in LF_XS.iter().enumerate() {
            if !LF[i][0][j].is_finite() {
                continue;
            }
            let (v, a) = (D::splat(x), D::splat(alpha));
            let one = [2.5];
            close_abs(
                "N=1",
                v.laguerre_function_series_n::<1>(a, &one).extract::<0>(),
                2.5 * LF[i][0][j],
                budget(1, alpha, x, f64::EPSILON) * 3.0,
            );
            let two = [2.5, -1.25];
            close_abs(
                "N=2",
                v.laguerre_function_series_n::<2>(a, &two).extract::<0>(),
                2.5 * LF[i][0][j] - 1.25 * LF[i][1][j],
                budget(2, alpha, x, f64::EPSILON) * 3.0,
            );
        }
    }
}

#[test]
fn series_at_high_degree_stays_in_range() {
    let mut c = [0.0f64; 81];
    c[80] = 1.0;
    for &(n, alpha, x, want) in LF_HIGH.iter() {
        if n != 80 {
            continue;
        }
        let got = D::splat(x)
            .laguerre_function_series_n::<81>(D::splat(alpha), &c)
            .extract::<0>();
        close_abs(
            &format!("series l_80^({alpha})({x})"),
            got,
            want,
            2.0 * budget(n, alpha, x, f64::EPSILON),
        );
    }
}

// --- integer weight (`_i`) ---

#[test]
fn integer_weight_matches_mpmath() {
    // The same table rows the float form is checked against, at the integer weights.
    macro_rules! check {
        ($n:literal) => {
            for (i, &alpha) in LF_ALPHAS.iter().enumerate() {
                if alpha.fract() != 0.0 || alpha < 0.0 {
                    continue;
                }
                for (j, &x) in LF_XS.iter().enumerate() {
                    let want = LF[i][$n][j];
                    if !want.is_finite() {
                        continue;
                    }
                    let got = D::splat(x).laguerre_function_i::<$n>(alpha as i32).extract::<0>();
                    close_abs(
                        &format!("l_{}^({alpha})({x}) [i]", $n),
                        got,
                        want,
                        budget($n, alpha, x, f64::EPSILON),
                    );
                }
            }
        };
    }
    for_each_degree!(check);
}

#[test]
fn integer_weight_agrees_with_the_float_weight() {
    // The two forms differ only in how the recurrence constants are built (a scalar
    // `sqrt((k+1)(k+a+1))` against a vector `sqrt(k1 * (k1 + alpha))`), so they should
    // agree to a couple of ulp of the O(1) envelope, not merely to the reference.
    macro_rules! check {
        ($n:literal) => {
            for a in 0..=6i32 {
                for &x in &LF_XS {
                    let v = D::splat(x);
                    let want = v.laguerre_function::<$n>(D::splat(a as f64)).extract::<0>();
                    let got = v.laguerre_function_i::<$n>(a).extract::<0>();
                    close_abs(&format!("l_{}^({a})({x}) i vs f", $n), got, want, 1e-14);
                }
            }
        };
    }
    for_each_degree!(check);
}

#[test]
fn integer_weight_zero_is_bit_identical_to_the_float_form() {
    // At alpha = 0 both forms take their uniform shortcut and the seed is exactly (f, f),
    // so there is nothing left to round differently.
    macro_rules! check {
        ($n:literal) => {
            for &x in &LF_XS {
                let v = D::splat(x);
                assert_eq!(
                    v.laguerre_function_i::<$n>(0).extract::<0>(),
                    v.laguerre_function::<$n>(D::ZERO).extract::<0>(),
                    "l_{}^(0)({x})",
                    $n
                );
            }
        };
    }
    for_each_degree!(check);
}

#[test]
fn integer_weight_reaches_the_high_degree_rows() {
    for &(n, alpha, x, want) in LF_HIGH.iter() {
        if alpha.fract() != 0.0 || alpha < 0.0 {
            continue;
        }
        let a = alpha as i32;
        let got = match n {
            0 => D::splat(x).laguerre_function_i::<0>(a),
            3 => D::splat(x).laguerre_function_i::<3>(a),
            8 => D::splat(x).laguerre_function_i::<8>(a),
            40 => D::splat(x).laguerre_function_i::<40>(a),
            80 => D::splat(x).laguerre_function_i::<80>(a),
            300 => D::splat(x).laguerre_function_i::<300>(a),
            600 => D::splat(x).laguerre_function_i::<600>(a),
            _ => unreachable!(),
        }
        .extract::<0>();
        close_abs(
            &format!("l_{n}^({alpha})({x}) [i]"),
            got,
            want,
            budget(n, alpha, x, f64::EPSILON),
        );
        assert!(got != 0.0 || want == 0.0, "l_{n}^({alpha})({x}) [i] collapsed to zero");
    }
}

#[test]
fn integer_series_agrees_with_the_float_series() {
    const N: usize = 9;
    let mut c = [0.0f64; N];
    for (k, slot) in c.iter_mut().enumerate() {
        *slot = if k % 2 == 0 { 1.0 } else { -1.0 } * 0.8f64.powi(k as i32);
    }

    for a in 0..=4i32 {
        for &x in &LF_XS {
            let v = D::splat(x);
            let want = v.laguerre_function_series_n::<N>(D::splat(a as f64), &c).extract::<0>();
            let got = v.laguerre_function_series_i_n::<N>(a, &c).extract::<0>();
            close_abs(&format!("series a={a} x={x} i vs f"), got, want, 1e-14);
        }
    }
}

#[test]
fn integer_series_with_a_unit_coefficient_is_the_single_function() {
    const N: usize = 9;
    for a in 0..=4i32 {
        for k in 0..N {
            let mut unit = [0.0f64; N];
            unit[k] = 1.0;
            for &x in &LF_XS {
                let v = D::splat(x);
                let got = v.laguerre_function_series_i_n::<N>(a, &unit).extract::<0>();
                let want = match k {
                    0 => v.laguerre_function_i::<0>(a),
                    1 => v.laguerre_function_i::<1>(a),
                    2 => v.laguerre_function_i::<2>(a),
                    3 => v.laguerre_function_i::<3>(a),
                    4 => v.laguerre_function_i::<4>(a),
                    5 => v.laguerre_function_i::<5>(a),
                    6 => v.laguerre_function_i::<6>(a),
                    7 => v.laguerre_function_i::<7>(a),
                    _ => v.laguerre_function_i::<8>(a),
                }
                .extract::<0>();
                close_abs(&format!("unit series k={k} a={a} x={x}"), got, want, 1e-14);
            }
        }
    }
}

#[test]
fn integer_weight_f32_tracks_the_reference() {
    macro_rules! check {
        ($n:literal) => {
            for (i, &alpha) in LF_ALPHAS.iter().enumerate() {
                if alpha.fract() != 0.0 || alpha < 0.0 {
                    continue;
                }
                for (j, &x) in LF_XS.iter().enumerate() {
                    let want = LF[i][$n][j];
                    if !want.is_finite() {
                        continue;
                    }
                    let got = F::splat(x as f32)
                        .laguerre_function_i::<$n>(alpha as i32)
                        .extract::<0>() as f64;
                    close_abs(
                        &format!("f32 l_{}^({alpha})({x}) [i]", $n),
                        got,
                        want,
                        budget($n, alpha, x, f32::EPSILON as f64),
                    );
                }
            }
        };
    }
    for_each_degree!(check);
}

#[test]
fn integer_weight_lanes_stay_independent() {
    // The weight is scalar here, so unlike the float form it is uniform by construction -
    // but x is still a vector and must not be reduced across.
    use thermite::backend::scalar::Scalar;
    type D4 = thermite::simd::f64x4<Scalar>;

    let xs = [LF_XS[1], LF_XS[3], LF_XS[5], LF_XS[7]];
    for a in 0..=4i32 {
        let got = D4::new(xs).laguerre_function_i::<6>(a);
        for (lane, &x) in xs.iter().enumerate() {
            assert_eq!(
                got.as_slice()[lane],
                D::splat(x).laguerre_function_i::<6>(a).extract::<0>(),
                "a={a} lane={lane}"
            );
        }
    }
}

#[test]
fn integer_weight_product_seed_matches_the_log_seed_across_the_cap() {
    // Integer weights up to LAGUERRE_PRODUCT_SEED_CAP (170 f64 / 29 f32) seed by
    // x^{a/2}/sqrt(a!) directly. Above it they take the float form's log path. Both sides
    // of the cap must agree with the float form, including at x far past the range where
    // e^{-x/4} underflows, where the answer is 0 and never NaN (inf * 0).
    let xs = [0.0, 0.5, 10.0, 100.0, 250.0, 400.0, 1000.0, 3000.0, 5000.0];
    for &a in &[1i32, 2, 3, 28, 29, 30, 31, 100, 169, 170, 171] {
        for &x in &xs {
            let want = D::splat(x).laguerre_function::<3>(D::splat(a as f64)).extract::<0>();
            let got = D::splat(x).laguerre_function_i::<3>(a).extract::<0>();
            assert!(
                got.is_finite() && want.is_finite(),
                "a={a} x={x}: got {got:?} want {want:?}"
            );
            close_abs(
                &format!("f64 l_3^({a})({x}) product vs log seed"),
                got,
                want,
                budget(3, a as f64, x, f64::EPSILON),
            );
        }
    }
    for &a in &[1i32, 2, 3, 15, 28, 29, 30, 31, 34] {
        for &x in &xs {
            let want = D::splat(x).laguerre_function::<3>(D::splat(a as f64)).extract::<0>();
            let got = F::splat(x as f32).laguerre_function_i::<3>(a).extract::<0>() as f64;
            assert!(got.is_finite(), "f32 a={a} x={x}: got {got:?} want {want:?}");
            close_abs(
                &format!("f32 l_3^({a})({x}) product vs log seed"),
                got,
                want,
                budget(3, a as f64, x, f32::EPSILON as f64),
            );
        }
    }
}

#[test]
fn weights_can_vary_per_lane() {
    use thermite::backend::scalar::Scalar;
    type D4 = thermite::simd::f64x4<Scalar>;

    let alphas = [0.0, 0.5, 1.0, 2.0];
    for &x in &LF_XS[1..] {
        let got = D4::splat(x).laguerre_function::<6>(D4::new(alphas));
        for (lane, &alpha) in alphas.iter().enumerate() {
            let want = D::splat(x).laguerre_function::<6>(D::splat(alpha)).extract::<0>();
            assert_eq!(got.as_slice()[lane], want, "x={x} lane={lane}");
        }
    }
}

#[test]
fn f32_tracks_the_reference_including_high_degree() {
    macro_rules! check {
        ($n:literal) => {
            for (i, &alpha) in LF_ALPHAS.iter().enumerate() {
                for (j, &x) in LF_XS.iter().enumerate() {
                    let want = LF[i][$n][j];
                    if !want.is_finite() {
                        continue;
                    }
                    let got = F::splat(x as f32)
                        .laguerre_function::<$n>(F::splat(alpha as f32))
                        .extract::<0>() as f64;
                    close_abs(
                        &format!("f32 l_{}^({alpha})({x})", $n),
                        got,
                        want,
                        budget($n, alpha, x, f32::EPSILON as f64),
                    );
                }
            }
        };
    }
    for_each_degree!(check);

    // Raw L_n leaves binary32 range at (40, 120) already, while x up to 350 is the documented
    // f32 range of the function.
    for &(n, alpha, x, want) in LF_HIGH.iter() {
        if x > 340.0 {
            continue;
        }
        let got = lf_at_f32(n, alpha as f32, x as f32) as f64;
        close_abs(
            &format!("f32 l_{n}^({alpha})({x})"),
            got,
            want,
            budget(n, alpha, x, f32::EPSILON as f64),
        );
        assert!(got != 0.0 || want == 0.0, "f32 l_{n}^({alpha})({x}) collapsed to zero");
    }
}

/// Relative error of the seed alone, in units of the target's epsilon.
fn seed_ulps(got: f64, want: f64, eps: f64) -> f64 {
    ((got - want) / want).abs() / eps
}

/// Seed budget in ulp: 16 for the pieces (the worst measured near-peak row is 12, at
/// `alpha = 1400`), plus the exponent's own conditioning. Near the
/// peak (`|k - x| < 0.2 (k + x)`) bd0 is a series and the exponent is small. Away from it
/// bd0 is `k ln(k/x) + x - k` directly and its two terms cancel to half an ulp of their
/// size, which the exponential turns into that many ulp of relative error.
fn seed_budget(alpha: f64, x: f64) -> f64 {
    let v = (alpha - x) / (alpha + x);
    let cancel = if v.abs() < 0.2 {
        0.0
    } else {
        (alpha * (alpha / x).ln()).abs() + (x - alpha).abs()
    };
    16.0 + cancel / 2.0
}

#[test]
fn seed_matches_mpmath_across_its_branches() {
    // l_0 is the seed itself: small-Gamma weights below STIRLERR_MIN, Loader's saddle-point
    // form above it, x from half the peak to 2.5x it. A combined-log seed sits at ~25 ulp
    // at (400, 400) and ~75 at (1400, 1400), while these rows hold the saddle-point form
    // under 3.
    for &(alpha, x, want) in LF_SEED.iter() {
        let budget = seed_budget(alpha, x);
        let got = D::splat(x).laguerre_function::<0>(D::splat(alpha)).extract::<0>();
        let u = seed_ulps(got, want, f64::EPSILON);
        assert!(u <= budget, "f64 l_0^({alpha})({x}): {u:.1} ulp (budget {budget:.1})");
        if alpha.fract() == 0.0 {
            let got = D::splat(x).laguerre_function_i::<0>(alpha as i32).extract::<0>();
            let u = seed_ulps(got, want, f64::EPSILON);
            assert!(
                u <= budget,
                "f64 l_0^({alpha})({x}) [i]: {u:.1} ulp (budget {budget:.1})"
            );
        }
        if want.abs() > 1e-30 && x < 350.0 {
            let got = F::splat(x as f32)
                .laguerre_function::<0>(F::splat(alpha as f32))
                .extract::<0>() as f64;
            let u = seed_ulps(got, want, f32::EPSILON as f64);
            assert!(u <= budget, "f32 l_0^({alpha})({x}): {u:.1} ulp (budget {budget:.1})");
        }
    }
}

/// `cargo nextest run -p thermite-special --release --test laguerre_function -E 'test(seed_ulp_report)' --run-ignored all --no-capture`
#[test]
#[ignore]
fn seed_ulp_report() {
    for &(alpha, x, want) in LF_SEED.iter() {
        let got = D::splat(x).laguerre_function::<0>(D::splat(alpha)).extract::<0>();
        let bd0 = alpha * (alpha / x).ln() + x - alpha;
        println!(
            "a={alpha:7} x={x:8} bd0={bd0:9.3} f64 {:7.2} ulp",
            seed_ulps(got, want, f64::EPSILON)
        );
    }
}
