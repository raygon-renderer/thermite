//! `atan2` quadrant handling.
//!
//! REGRESSION (found 2026-08-15 by the thermite-interval ulp sweep, which
//! measured "1.4e16 ulp error" against this function and turned out to be
//! measuring the reference, not the kernel): the +-pi offset applied for
//! `x < 0` had its SIGN chosen by the sign of `x` rather than of `y`, so every
//! second-quadrant angle came out on the wrong branch. `atan2(1, -1)` gave
//! `-5pi/4`, which is not even inside the principal range `(-pi, pi]`.

use thermite::math::TranscendentalMath;
use thermite::prelude::*;
use thermite_compensated::Compensated;

type V = Vector<f64>;
type C = Compensated<V>;

fn atan2_dd(y: f64, x: f64) -> f64 {
    let r = C::new(V::splat(y)).atan2(C::new(V::splat(x)));
    r.value().extract::<0>() + r.error().extract::<0>()
}

/// Every quadrant, against `f64::atan2` (which is correctly rounded here).
#[test]
fn quadrants_match_std() {
    let vals = [-1e100, -3.0, -1.0, -1e-100, 0.0, 1e-100, 1.0, 3.0, 1e100];

    for &y in &vals {
        for &x in &vals {
            if y == 0.0 && x == 0.0 {
                continue; // atan2(0, 0) is a convention, not a value
            }
            let got = atan2_dd(y, x);
            let want = y.atan2(x);
            let err = (got - want).abs();
            assert!(
                err < 1e-14 * want.abs().max(1.0),
                "atan2({y:e}, {x:e}): got {got:.17e}, std {want:.17e}"
            );
        }
    }
}

/// The result always lies in the principal range `(-pi, pi]`.
#[test]
fn principal_range() {
    let mut state = 12345u64;
    let mut next = || {
        state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((state >> 11) as f64 / (1u64 << 53) as f64) * 20.0 - 10.0
    };

    for _ in 0..5_000 {
        let (y, x) = (next(), next());
        if y == 0.0 && x == 0.0 {
            continue;
        }
        let got = atan2_dd(y, x);
        assert!(
            got > -core::f64::consts::PI - 1e-15 && got <= core::f64::consts::PI + 1e-15,
            "atan2({y}, {x}) = {got} outside (-pi, pi]"
        );
    }
}

/// The second quadrant specifically, the branch that was wrong.
#[test]
fn second_quadrant_is_positive() {
    for (y, x) in [(1.0, -1.0), (1e-100, -1e100), (1e-300, -1.0), (0.5, -2.0)] {
        let got = atan2_dd(y, x);
        assert!(
            got > 0.0,
            "atan2({y:e}, {x:e}) = {got} must be positive (second quadrant)"
        );
        assert!(got <= core::f64::consts::PI + 1e-15, "and at most pi");
    }
    // Third quadrant stays negative.
    for (y, x) in [(-1.0, -1.0), (-1e-100, -1e100)] {
        let got = atan2_dd(y, x);
        assert!(
            got < 0.0,
            "atan2({y:e}, {x:e}) = {got} must be negative (third quadrant)"
        );
    }
}
