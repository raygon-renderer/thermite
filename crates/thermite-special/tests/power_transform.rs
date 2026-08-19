//! The power-transform family beyond `boxcox`: `boxcox_1p`, `inv_boxcox`, `inv_boxcox_1p`,
//! `yeo_johnson` and `inv_yeo_johnson`.
//!
//! References from `scripts/power_transform_ref.py` (mpmath at 60 digits). Two things the
//! tables are shaped around:
//!
//! The Yeo-Johnson reference writes out all four published cases rather than the sign fold
//! the kernel uses, so agreement is evidence the fold is right and not a shared assumption.
//! Its rows come in `(y, -y)` pairs at the same lambda for that reason, since the fold is
//! exactly what relates them.
//!
//! The `_1p` forms exist for small arguments, so the small-argument rows are the point, not
//! filler. Each one is paired with a measurement of the spelling it replaces: the tests
//! evaluate the naive form here rather than asserting anything about another library.
#![cfg(any(
    target_arch = "x86",
    target_arch = "x86_64",
    all(target_arch = "wasm32", feature = "wasm"),
    target_arch = "aarch64"
))]

use thermite::prelude::*;
use thermite_special::RealSpecialMath;

type D = Vector<f64>;
type F = Vector<f32>;

/// Bit-for-bit lane agreement, with NaN counting as equal to NaN, since the lane tests deliberately
/// include out-of-domain lanes, and `assert_eq!` would reject the very case being checked.
#[track_caller]
fn same(name: &str, got: f64, want: f64) {
    assert!(
        got == want || (got.is_nan() && want.is_nan()),
        "{name}: got {got:?}, want {want:?}"
    );
}

#[track_caller]
fn close(name: &str, got: f64, want: f64, tol: f64) {
    let rel = if want == 0.0 {
        got.abs()
    } else {
        ((got - want) / want).abs()
    };
    assert!(rel <= tol, "{name}: got {got:?}, want {want:?} (rel {rel:e})");
}

fn boxcox_1p(x: f64, lambda: f64) -> f64 {
    D::splat(x).boxcox_1p(D::splat(lambda)).extract::<0>()
}
fn inv_boxcox(y: f64, lambda: f64) -> f64 {
    D::splat(y).inv_boxcox(D::splat(lambda)).extract::<0>()
}
fn inv_boxcox_1p(y: f64, lambda: f64) -> f64 {
    D::splat(y).inv_boxcox_1p(D::splat(lambda)).extract::<0>()
}
fn yeo_johnson(y: f64, lambda: f64) -> f64 {
    D::splat(y).yeo_johnson(D::splat(lambda)).extract::<0>()
}
fn inv_yeo_johnson(z: f64, lambda: f64) -> f64 {
    D::splat(z).inv_yeo_johnson(D::splat(lambda)).extract::<0>()
}

#[rustfmt::skip]
const BOXCOX_1P: [(f64, f64, f64); 11] = [
    (0.0,       0.5,    0.0),
    (1.0,       0.0,    0.6931471805599453),
    (1e-20,     0.5,    1e-20),
    (-1e-20,    2.0,   -1e-20),
    (0.5,       1e-18,  0.4054651081081644),
    (2.0,       0.5,    1.4641016151377546),
    (-0.5,      3.0,   -0.2916666666666667),
    (100000.0,  0.25,  67.13135422883107),
    (-0.9,      2.0,   -0.495),
    (3.0,      -1.5,    0.5833333333333334),
    (1e-12,     7.0,    1.000000000003e-12),
];

/// `(y, lambda, x, tolerance in ulps of one)`.
///
/// The last row's 2048 is conditioning, not slack. `lambda*y + 1` is `5e-4` there, a
/// thousandth of the way from the range boundary, so the rounding of the product alone is
/// `2e-13` relative once divided by it, and the exponent carries that straight through.
/// Every formulation of the inverse has it, the quantity the answer depends on being small
/// and formed by subtraction, and the point of the row is that the kernel loses nothing
/// beyond it.
#[rustfmt::skip]
const INV_BOXCOX: [(f64, f64, f64, f64); 14] = [
    (0.0,                  0.5,      1.0,                  16.0),
    (1.0,                  0.0,      2.718281828459045,    16.0),
    (-0.6931471805599453,  0.0,      0.5,                  16.0),
    (0.8284271247461901,   0.5,      2.0,                  16.0),
    (49.5,                 2.0,     10.0,                  16.0),
    (0.99,                -1.0,    100.0,                  16.0),
    (-1.21895141649746,   -1.5,      0.5,                  16.0),
    (2.3025851195035365,   1e-8,    10.0,                  16.0),
    (0.6931471805601855,   1e-12,    2.0,                  16.0),
    (-0.6931471805599453,  1e-300,   0.5,                  16.0),
    (2.0,                  3.0,      1.9129311827723892,   16.0),
    (10.0,                 0.1,   1024.0,                  16.0),
    (-0.5,                -2.0,      0.7071067811865476,   16.0),
    (-1.999,               0.5,      2.5e-07,            2048.0),
];

#[rustfmt::skip]
const INV_BOXCOX_1P: [(f64, f64, f64); 9] = [
    (0.0,                 0.5,    0.0),
    (1.0,                 0.0,    1.7182818284590453),
    (1e-20,               0.5,    1e-20),
    (0.5,                 1e-18,  0.6487212707001282),
    (1.4641016151377546,  0.5,    2.0),
    (2.0,                 3.0,    0.912931182772389),
    (-0.5,               -2.0,   -0.2928932188134525),
    (1e-18,               7.0,    1e-18),
    (-1.5,                0.5,   -0.9375),
];

/// `(y, lambda, psi(y, lambda))`, in `(y, -y)` pairs at a shared lambda: the sign fold in
/// the kernel is exactly the claim that those two rows are related, and the reference
/// computes each of them from its own published case.
#[rustfmt::skip]
const YEO_JOHNSON: [(f64, f64, f64); 21] = [
    (0.0,        0.5,    0.0),
    (1.0,        0.0,    0.6931471805599453),
    (-1.0,       2.0,   -0.6931471805599453),
    (2.0,        0.5,    1.4641016151377546),
    (-2.0,       0.5,   -2.797434948471088),
    (3.0,        1.5,    4.666666666666667),
    (-3.0,       1.5,   -2.0),
    (0.5,       -1.0,    0.3333333333333333),
    (-0.5,      -1.0,   -0.7916666666666666),
    (1e-20,      0.5,    1e-20),
    (-1e-20,     0.5,   -1e-20),
    (10.0,       0.0,    2.3978952727983707),
    (-10.0,      2.0,   -2.3978952727983707),
    (5.0,        3.0,   71.66666666666667),
    (-5.0,       3.0,   -0.8333333333333334),
    (100000.0,   0.25,  67.13135422883107),
    (-100000.0,  0.25, -321343522.971682),
    (0.25,       1e-12,  0.22314355131423466),
    (-0.25,      1e-12, -0.2812499999999663),
    (7.0,        2.0,   31.5),
    (-7.0,       0.0,  -31.5),
];

#[rustfmt::skip]
const INV_YEO_JOHNSON: [(f64, f64, f64); 11] = [
    (0.0,                  0.5,  0.0),
    (0.6931471805599453,   0.0,  1.0),
    (-0.6931471805599453,  2.0, -1.0),
    (1.4641016151377546,   0.5,  2.0),
    (-2.797434948471088,   0.5, -2.0),
    (4.666666666666667,    1.5,  3.0),
    (-2.0,                 1.5, -3.0),
    (1e-20,                0.5,  1e-20),
    (-1e-20,               0.5, -1e-20),
    (71.66666666666667,    3.0,  5.0),
    (-0.8333333333333334,  3.0, -5.000000000000003),
];

#[test]
fn they_match_the_reference() {
    for &(x, l, want) in BOXCOX_1P.iter() {
        close(
            &format!("boxcox_1p({x}, {l})"),
            boxcox_1p(x, l),
            want,
            16.0 * f64::EPSILON,
        );
    }
    for &(y, l, want, ulps) in INV_BOXCOX.iter() {
        close(
            &format!("inv_boxcox({y}, {l})"),
            inv_boxcox(y, l),
            want,
            ulps * f64::EPSILON,
        );
    }
    for &(y, l, want) in INV_BOXCOX_1P.iter() {
        close(
            &format!("inv_boxcox_1p({y}, {l})"),
            inv_boxcox_1p(y, l),
            want,
            16.0 * f64::EPSILON,
        );
    }
    for &(y, l, want) in YEO_JOHNSON.iter() {
        close(
            &format!("yeo_johnson({y}, {l})"),
            yeo_johnson(y, l),
            want,
            16.0 * f64::EPSILON,
        );
    }
    for &(z, l, want) in INV_YEO_JOHNSON.iter() {
        close(
            &format!("inv_yeo_johnson({z}, {l})"),
            inv_yeo_johnson(z, l),
            want,
            32.0 * f64::EPSILON,
        );
    }
}

#[test]
fn the_inverses_round_trip() {
    // The strongest available statement about a pair of transforms, and it catches a
    // reflection applied in one direction but not the other.
    for &l in &[0.0_f64, 1e-12, 0.25, 0.5, 1.0, 1.5, 2.0, 3.0, -0.5, -1.5] {
        for &x in &[0.25_f64, 0.5, 1.0, 2.0, 7.0, 100.0] {
            let round = inv_boxcox(D::splat(x).boxcox(D::splat(l)).extract::<0>(), l);
            close(&format!("boxcox round trip at x={x}, lambda={l}"), round, x, 1e-13);

            let round = inv_boxcox_1p(boxcox_1p(x, l), l);
            close(&format!("boxcox_1p round trip at x={x}, lambda={l}"), round, x, 1e-13);
        }

        // Yeo-Johnson's whole selling point is that this list can contain negatives.
        for &y in &[-7.0_f64, -2.0, -0.5, -1e-8, 0.0, 1e-8, 0.5, 2.0, 7.0] {
            let round = inv_yeo_johnson(yeo_johnson(y, l), l);
            let tol = if y == 0.0 { 0.0 } else { 1e-12 };
            close(&format!("yeo_johnson round trip at y={y}, lambda={l}"), round, y, tol);
        }
    }
}

#[test]
fn yeo_johnson_reproduces_its_four_published_cases() {
    // Written out branch by branch, against the folded kernel. This is the test that would
    // catch `2 - lambda` reflected the wrong way, which every other test here would pass
    // for lambda = 1.
    for &y in &[0.25_f64, 1.0, 3.0] {
        for &l in &[0.5_f64, 1.5, 2.0, -1.0, 3.0] {
            let want = ((y + 1.0).powf(l) - 1.0) / l;
            close(
                &format!("psi({y}, {l}) positive branch"),
                yeo_johnson(y, l),
                want,
                1e-13,
            );

            // lambda = 2 is where the published negative branch is 0/0; it has its own case
            // below, which is the whole reason the four are written as four.
            if l != 2.0 {
                let want = -((1.0 + y).powf(2.0 - l) - 1.0) / (2.0 - l);
                close(
                    &format!("psi({}, {l}) negative branch", -y),
                    yeo_johnson(-y, l),
                    want,
                    1e-13,
                );
            }
        }

        // lambda = 0 above zero and lambda = 2 below are the logarithmic cases, and the fold
        // maps both onto the single seam the kernel actually branches on.
        close("psi(y, 0)", yeo_johnson(y, 0.0), (y + 1.0).ln(), 8.0 * f64::EPSILON);
        close("psi(-y, 2)", yeo_johnson(-y, 2.0), -(1.0 + y).ln(), 8.0 * f64::EPSILON);
    }

    // lambda = 1 is the identity up to a shift on both sides, which pins the sign fold's
    // scale as well as its sign.
    for &y in &[-5.0_f64, -0.5, 0.0, 0.5, 5.0] {
        close(&format!("psi({y}, 1)"), yeo_johnson(y, 1.0), y, 8.0 * f64::EPSILON);
    }

    // The origin is a fixed point for every lambda, and the transform is increasing.
    for &l in &[0.0_f64, 0.5, 1.0, 2.0, -1.0, 3.5] {
        assert_eq!(yeo_johnson(0.0, l), 0.0, "psi(0, {l}) must be exactly 0");

        let mut prev = f64::NEG_INFINITY;
        for &y in &[-100.0_f64, -3.0, -0.5, 0.0, 0.5, 3.0, 100.0] {
            let v = yeo_johnson(y, l);
            assert!(v > prev, "psi is increasing, but psi({y}, {l}) = {v} <= {prev}");
            prev = v;
        }
    }
}

#[test]
fn the_1p_forms_keep_small_arguments_the_shifted_spelling_loses() {
    // `boxcox(1 + x, lambda)` and `yeo_johnson` via a shifted `boxcox` both round x away
    // once |x| drops below the epsilon of one. This is the reason the _1p forms exist, and
    // for Yeo-Johnson it is not an edge case: psi(y) ~ y near the origin, and the origin is
    // where the data is.
    for &x in &[1e-17_f64, 1e-20, 1e-30, -1e-18] {
        for &l in &[0.5_f64, 2.0, 7.0, -1.5] {
            let want = l * x; // (1+x)^l - 1 = l x + O(x^2), and x^2 is far under the ulp

            close(&format!("boxcox_1p({x}, {l})"), boxcox_1p(x, l), want / l, 1e-13);
            assert_eq!(1.0 + x, 1.0, "precondition: 1 + {x} rounds to one");

            let shifted = D::splat(1.0 + x).boxcox(D::splat(l)).extract::<0>();
            assert_eq!(shifted, 0.0, "precondition: the shifted spelling collapses");
        }

        // ... and the same statement for the transform that actually cares.
        for &l in &[0.0_f64, 0.5, 1.0, 2.0, 3.0] {
            close(&format!("psi({x}, {l})"), yeo_johnson(x, l), x, 1e-13);
        }
    }

    // The other end of the same story: an ordinary argument with a tiny lambda, where the
    // result is near zero and `pow(...) - 1` cancels instead.
    for &x in &[0.5_f64, 3.0] {
        for &l in &[1e-12_f64, 1e-18, 1e-300] {
            close(&format!("boxcox_1p({x}, {l})"), boxcox_1p(x, l), (1.0 + x).ln(), 1e-7);
            close(&format!("psi({x}, {l})"), yeo_johnson(x, l), (1.0 + x).ln(), 1e-7);
        }
        assert_eq!(
            ((1.0 + x).powf(1e-300) - 1.0) / 1e-300,
            0.0,
            "precondition: naive collapses"
        );
    }
}

#[test]
fn inv_boxcox_survives_a_lambda_the_power_spelling_cannot() {
    // Same argument in reverse. `(lambda*y + 1)^(1/lambda)` forms a number a hair above one
    // and raises it to a power of 1e12, so the rounding in the base is amplified by the
    // exponent, while the ln1p form never forms the base.
    for &l in &[1e-12_f64, 1e-16, 1e-300] {
        for &y in &[0.5_f64, -0.5, 2.0] {
            close(&format!("inv_boxcox({y}, {l})"), inv_boxcox(y, l), y.exp(), 1e-7);
        }
    }

    // Visibly wrong by 1e-12 and a flat one by 1e-300, measured here rather than asserted.
    let naive = (1e-12f64 * 0.5 + 1.0).powf(1e12);
    let err = ((naive - 0.5f64.exp()) / 0.5f64.exp()).abs();
    assert!(
        err > 1e-5,
        "precondition: naive is expected to be visibly wrong, got {naive} (err {err:e})"
    );
    assert_eq!(
        (1e-300f64 * 0.5 + 1.0).powf(1e300),
        1.0,
        "precondition: naive collapses to one"
    );
}

#[test]
fn the_domain_and_range_edges_are_pinned() {
    // boxcox_1p: x = -1 is the edge, with the same limits boxcox has at x = 0.
    close("boxcox_1p(-1, 2)", boxcox_1p(-1.0, 2.0), -0.5, 0.0);
    close("boxcox_1p(-1, 0.25)", boxcox_1p(-1.0, 0.25), -4.0, 0.0);
    assert_eq!(boxcox_1p(-1.0, 0.0), f64::NEG_INFINITY);
    assert_eq!(boxcox_1p(-1.0, -1.5), f64::NEG_INFINITY);
    assert!(boxcox_1p(-1.5, 0.5).is_nan(), "below the domain");

    // inv_boxcox: the forward transform's range is lambda*y + 1 > 0, so a y below that came
    // from no x at all. NaN, not a plausible number.
    assert!(inv_boxcox(-3.0, 0.5).is_nan(), "-3 is below -1/lambda = -2");
    assert!(inv_boxcox(3.0, -0.5).is_nan());
    assert_eq!(inv_boxcox(-2.0, 0.5), 0.0, "the boundary maps to the domain edge x = 0");
    assert_eq!(inv_boxcox(2.0, -0.5), f64::INFINITY);
    assert_eq!(inv_boxcox_1p(-2.0, 0.5), -1.0, "same boundary, shifted");

    // inv_yeo_johnson inherits the bound on each side from the lambda that side uses, so
    // which lambdas can produce an unreachable z differs above and below zero. Above, only
    // a negative lambda bounds the range. Below, only lambda > 2 does, since the reflection
    // is what has to come out negative.
    assert!(inv_yeo_johnson(3.0, -0.5).is_nan(), "z = 3 is past -1/lambda = 2");
    assert!(
        inv_yeo_johnson(-3.0, 3.5).is_nan(),
        "z = -3 is past 1/(lambda - 2) = 2/3"
    );

    // ... and the mirror-image lambdas leave both sides reachable.
    assert!(
        inv_yeo_johnson(-3.0, 0.5).is_finite(),
        "2 - lambda > 0 bounds nothing below zero"
    );
    assert!(
        inv_yeo_johnson(3.0, 3.5).is_finite(),
        "lambda > 0 bounds nothing above zero"
    );

    // Yeo-Johnson itself has no domain restriction at all, which is the whole reason it is
    // preferred over Box-Cox, so no finite input may produce a NaN at any lambda. It can
    // still overflow, and does: psi(-1e300, 0) is `(1e300)^2/2`, an honest infinity rather
    // than a failure of the transform.
    for &l in &[0.0_f64, 0.5, 1.0, 2.0, -2.0, 5.0] {
        for &y in &[-1e300_f64, -1e10, -1.0, 0.0, 1.0, 1e10, 1e300] {
            let v = yeo_johnson(y, l);
            assert!(!v.is_nan(), "psi({y}, {l}) must not be NaN, got {v}");
            assert!(
                v.is_sign_negative() == (y < 0.0) || y == 0.0,
                "psi({y}, {l}) has the wrong sign"
            );
        }

        // Over a range where nothing overflows, finite really does mean finite.
        for &y in &[-1e10_f64, -1.0, 0.0, 1.0, 1e10] {
            assert!(yeo_johnson(y, l).is_finite(), "psi({y}, {l}) must be finite");
        }
    }

    // NaN in, NaN out, in both arguments and both directions.
    for f in [
        yeo_johnson as fn(f64, f64) -> f64,
        inv_yeo_johnson,
        boxcox_1p,
        inv_boxcox,
        inv_boxcox_1p,
    ] {
        assert!(f(f64::NAN, 1.0).is_nan());
        assert!(f(1.0, f64::NAN).is_nan());
    }
}

#[test]
fn f32_tracks_the_reference() {
    for &(y, l, want) in YEO_JOHNSON.iter() {
        if want.abs() > 1e30 || (y != 0.0 && (y as f32) == 0.0) {
            continue;
        }
        let got = F::splat(y as f32).yeo_johnson(F::splat(l as f32)).extract::<0>() as f64;
        close(&format!("f32 psi({y}, {l})"), got, want, 64.0 * f32::EPSILON as f64);
    }

    for &(y, l, want, ulps) in INV_BOXCOX.iter() {
        if want.abs() > 1e30 || l.abs() < 1e-30 && l != 0.0 {
            continue;
        }
        let got = F::splat(y as f32).inv_boxcox(F::splat(l as f32)).extract::<0>() as f64;
        // The f64 row's tolerance is conditioning, so it scales to f32 rather than being
        // replaced: the same 8x margin over the epsilon of the format, in that format.
        close(
            &format!("f32 inv_boxcox({y}, {l})"),
            got,
            want,
            8.0 * ulps * f32::EPSILON as f64,
        );
    }
}

#[test]
fn lanes_stay_independent_across_every_seam() {
    use thermite::backend::scalar::Scalar;
    type D4 = thermite::simd::f64x4<Scalar>;

    // The case the blends exist for. Yeo-Johnson has two seams that the fold maps onto one,
    // so a vector with mixed signs at lambda = 0 has lanes on both sides of it.
    for &l in &[0.0_f64, 2.0, 0.5, 1e-12] {
        let ys = [2.0, -0.5, 1e-20, -7.0];
        let ls = D4::splat(l);

        let fwd = D4::new(ys).yeo_johnson(ls);
        let inv = D4::new(ys).inv_yeo_johnson(ls);
        for lane in 0..4 {
            same(
                &format!("fwd lambda={l} lane {lane}"),
                fwd.as_slice()[lane],
                yeo_johnson(ys[lane], l),
            );
            same(
                &format!("inv lambda={l} lane {lane}"),
                inv.as_slice()[lane],
                inv_yeo_johnson(ys[lane], l),
            );
        }
    }

    // And a vector where lambda itself is mixed across the seam, which is the arm no
    // realistic caller takes but every lane must still be right on.
    let ys = [2.0, -0.5, 0.25, -7.0];
    let ls = [0.0, 2.0, 1e-12, 0.5];
    let got = D4::new(ys).yeo_johnson(D4::new(ls));
    for lane in 0..4 {
        same(
            &format!("mixed lambda lane {lane}"),
            got.as_slice()[lane],
            yeo_johnson(ys[lane], ls[lane]),
        );
    }

    // The last lane here is out of `boxcox_1p`'s domain, which is the case worth having: a
    // NaN lane must not disturb the ones beside it.
    let bc = D4::new(ys).boxcox_1p(D4::new(ls));
    for lane in 0..4 {
        same(
            &format!("boxcox_1p mixed lane {lane}"),
            bc.as_slice()[lane],
            boxcox_1p(ys[lane], ls[lane]),
        );
    }
    assert!(
        bc.as_slice()[3].is_nan(),
        "precondition: that lane really is out of domain"
    );
}
