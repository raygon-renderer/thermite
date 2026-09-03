//! `erf`/`erfc` at arguments whose square overflows, and at the infinities.
//!
//! Regression for a NaN found while building `ndtr`: `x * x` inside the kernels overflowed
//! past `|x| = 1.34e154` (f64) / `1.8e19` (f32), and the rational then evaluated `inf/inf`.
//! Under `check_overflow` the argument is now clamped where the result has long saturated.
#![cfg(any(
    target_arch = "x86",
    target_arch = "x86_64",
    all(target_arch = "wasm32", feature = "wasm"),
    target_arch = "aarch64"
))]

use thermite::math::policy::DefaultPolicy;
use thermite::math::policy::policies::{BestPrecision, WorstPrecision};
use thermite::prelude::*;
use thermite_special::{SpecialMath, SpecialMathWithPolicy};

type D = Vector<f64>;
type F = Vector<f32>;

#[test]
fn f64_saturates_instead_of_nan() {
    for x in [40.0, 1e100, 1.3e154, 2e154, 1e300, f64::MAX, f64::INFINITY] {
        for (name, got, want) in [
            ("erf", D::splat(x).erf().extract::<0>(), 1.0),
            ("erf", D::splat(-x).erf().extract::<0>(), -1.0),
            ("erfc", D::splat(x).erfc().extract::<0>(), 0.0),
            ("erfc", D::splat(-x).erfc().extract::<0>(), 2.0),
            (
                "erf best",
                D::splat(x).erf_p::<BestPrecision<DefaultPolicy>>().extract::<0>(),
                1.0,
            ),
            (
                "erfc best",
                D::splat(-x).erfc_p::<BestPrecision<DefaultPolicy>>().extract::<0>(),
                2.0,
            ),
        ] {
            assert_eq!(got, want, "{name}({x:e})");
        }
    }
    assert!(D::splat(f64::NAN).erf().extract::<0>().is_nan());
    assert!(D::splat(f64::NAN).erfc().extract::<0>().is_nan());
}

#[test]
fn f32_saturates_instead_of_nan() {
    for x in [20.0f32, 1e19, 1e20, 1e30, f32::MAX, f32::INFINITY] {
        for (name, got, want) in [
            ("erf", F::splat(x).erf().extract::<0>(), 1.0),
            ("erf", F::splat(-x).erf().extract::<0>(), -1.0),
            ("erfc", F::splat(x).erfc().extract::<0>(), 0.0),
            ("erfc", F::splat(-x).erfc().extract::<0>(), 2.0),
            (
                "erf best",
                F::splat(x).erf_p::<BestPrecision<DefaultPolicy>>().extract::<0>(),
                1.0,
            ),
            (
                "erfc best",
                F::splat(-x).erfc_p::<BestPrecision<DefaultPolicy>>().extract::<0>(),
                2.0,
            ),
        ] {
            assert_eq!(got, want, "{name}({x:e})");
        }
        // The low tiers take the clamped polynomial path and were never affected. They
        // saturate to the polynomial's own 1 - 1/t^4 rather than to exactly 1.
        let w = F::splat(x).erf_p::<WorstPrecision<DefaultPolicy>>().extract::<0>();
        assert!(w.is_finite() && w > 0.9999, "erf worst({x:e}) = {w}");
    }
    assert!(F::splat(f32::NAN).erf().extract::<0>().is_nan());
    assert!(F::splat(f32::NAN).erfc().extract::<0>().is_nan());
}

/// The clamp sits where nothing finite changes: just below it the kernels already
/// return the saturated values bit-for-bit.
#[test]
fn clamp_is_past_saturation() {
    assert_eq!(D::splat(31.0).erfc().extract::<0>(), 0.0);
    assert_eq!(D::splat(31.0).erf().extract::<0>(), 1.0);
    assert_eq!(F::splat(15.0).erfc().extract::<0>(), 0.0);
    assert_eq!(F::splat(15.0).erf().extract::<0>(), 1.0);
}
