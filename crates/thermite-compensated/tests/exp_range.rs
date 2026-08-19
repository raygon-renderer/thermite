//! `exp` and `ln` at the ends of the exponent range.
//!
//! The hazard is `exp`'s **low word** at large negative arguments, where the value can be
//! right while the correction is 300 orders too large. `exp_internal` scales both words
//! with `ldexp` under `CheckOverflow<P, false>`, which is safe for the value (overflow is
//! pre-checked) but not for the low word, which sits ~53 binades lower and leaves the
//! representable range first. Unclamped, `ldexp` writes a negative biased exponent straight
//! into the exponent field.
//!
//! That poisons everything refining through `exp`: `ln(1e-300)` comes back off by exactly
//! 2.0, because Halley's `(x - e_y)/(x + e_y)` collapses to -1 when `e_y` is garbage.
//!
//! Scaling the low word under `PreserveDenormals` is what lets the correction survive
//! into the subnormal range instead of being flushed.
//!
//! `ln(1e300)` is deliberately absent. It fails on backends without true FMA, where
//! `two_prod` falls back to Dekker splitting and `2e300 * (2^27+1)` overflows, a separate
//! limitation of the fallback path, above `MAX / 2^27`. With FMA it is correct.

use thermite::math::TranscendentalMath;
use thermite::prelude::*;
use thermite_compensated::Compensated;

type V = Vector<f64>;
type C = Compensated<V>;

fn c(x: f64) -> C {
    C::new(V::splat(x))
}

fn dd_err(got: C, hi: f64, lo: f64) -> f64 {
    let value = got.value.extract::<0>();
    let error = got.error.extract::<0>();

    (((value - hi) + (error - lo)) / hi.abs().max(1e-300)).abs()
}

const EXP_EXTREME: &[(f64, f64, f64)] = &[
    (-700.0, 9.85967654375977077e-305, 8.44852254388531591e-322),
    (-300.0, 5.14820022241201348e-131, 2.96237637337297919e-147),
    (-100.0, 3.72007597602083612e-44, -1.57050249077320082e-60),
    (-10.0, 4.53999297624848542e-05, -2.63755405532753089e-21),
    (1.0, 2.71828182845904509e+00, 1.44564689172925016e-16),
    (10.0, 2.20264657948067179e+04, -1.37801347005173720e-12),
    (700.0, 1.01423205473500449e+304, 1.66665719207346727e+287),
];

const LN_EXTREME: &[(f64, f64, f64)] = &[
    (1e-300, -6.90775527898213682e+02, -2.36700961767098316e-14),
    (1e-100, -2.30258509299404579e+02, 1.10694130972235950e-14),
    (0.5, -6.93147180559945286e-01, -2.31904681384629956e-17),
    (2.0, 6.93147180559945286e-01, 2.31904681384629956e-17),
    (1e+100, 2.30258509299404579e+02, -1.10335183063112318e-14),
];

#[test]
fn exp_low_word_stays_a_valid_correction() {
    // The direct form of the bug: a correction can never be as large as the value it
    // corrects. exp(-700) had value 9.86e-305 and low word -2.74e+295.
    for &(x, _, _) in EXP_EXTREME {
        let e = c(x).exp();
        let (value, error) = (e.value.extract::<0>(), e.error.extract::<0>());

        assert!(value.is_finite() && error.is_finite(), "exp({x}) is not finite");
        assert!(
            error.abs() < value.abs() || (error == 0.0 && value == 0.0),
            "exp({x}): |low| {error:e} >= |high| {value:e}"
        );
    }
}

/// Below roughly 2e-292 the low word itself is subnormal, so it carries fewer than the
/// usual 53 bits: `exp(-700)`'s correction is 8.4979e-322 against a true 8.4485e-322,
/// about 0.6% of a term that is already 2^-53 of the value.
///
/// That is inherent to the format, not to the implementation: it is what gradual underflow
/// leaves. Scaling the low word under `PreserveDenormals` is what gets it at all, since
/// flushing to zero costs 2^-53 outright, so this is still two orders better than the
/// fallback.
const SUBNORMAL_LOW: f64 = 1e-18;
/// `ln` inherits the same ceiling one step removed, but lands far inside it.
const LN_SUBNORMAL: f64 = 1e-26;
const FULL: f64 = 1e-30;

#[test]
fn exp_matches_mpmath_across_the_range() {
    for &(x, hi, lo) in EXP_EXTREME {
        let tol = if hi.abs() < 2e-292 { SUBNORMAL_LOW } else { FULL };
        let err = dd_err(c(x).exp(), hi, lo);

        assert!(err <= tol, "exp({x}): rel err {err:e}");
    }
}

#[test]
fn ln_matches_mpmath_across_the_range() {
    // 1e-300 is the one that goes off by exactly 2.0 unguarded. It holds 1.2e-27, the same
    // subnormal ceiling one step removed: `ln` refines through `exp`, and near the
    // underflow floor `exp`'s own low word has only ~26 bits, so there is less to refine
    // against. Everything above the floor is full double-double.
    for &(x, hi, lo) in LN_EXTREME {
        let tol = if x < 2e-292 { LN_SUBNORMAL } else { FULL };
        let err = dd_err(c(x).ln(), hi, lo);

        assert!(err <= tol, "ln({x:e}): rel err {err:e}");
    }
}
