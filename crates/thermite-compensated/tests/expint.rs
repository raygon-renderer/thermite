//! `expint` on `Compensated`, over both regimes of the shared kernel.
//!
//! Regression: everything with x >= 1 returns NaN unguarded. That is the
//! continued-fraction half, whose Lentz sentinel is reciprocated on the first step, and
//! the inherited default, `MIN_POSITIVE`, inverts to 4.5e307, past the ~1.3e300 where
//! compensated multiplication's Dekker 2^27+1 splitter overflows. `Compensated` overrides
//! `ExpIntDetails::cf_tiny`, as `Complex` already did for the same reason.

use thermite::prelude::*;
use thermite_compensated::Compensated;
use thermite_special::SpecialMath;

type V = Vector<f64>;
type C = Compensated<V>;

fn c(x: f64) -> C {
    C::new(V::splat(x))
}

/// `(x, hi, lo)` from mpmath 1.3.0 at 45 digits.
const E1: &[(f64, f64, f64)] = &[
    (0.5, 5.59773594776160843e-01, -3.15203250418664216e-17),
    (2.0, 4.89005107080611179e-02, 1.68565551204208784e-18),
    (10.0, 4.15696892968532464e-06, -3.58275463102901105e-22),
    (30.0, 3.02155201068881243e-15, 1.12581475791289606e-31),
];

#[test]
fn expint_e1_spans_both_regimes() {
    // 0.5 takes the power series, the rest take the continued fraction.
    for &(x, hi, lo) in E1 {
        let got = c(x).expint::<1>();
        assert!(got.value.extract::<0>().is_finite(), "expint::<1>({x}) is not finite");

        let value = got.value.extract::<0>();
        let error = got.error.extract::<0>();
        let err = (((value - hi) + (error - lo)) / hi.abs()).abs();

        assert!(err <= 1e-29, "expint::<1>({x}): rel err {err:e}");
    }
}

#[test]
fn expint_order_recurrence() {
    // E_{n+1}(x) = (e^-x - x E_n(x)) / n, checked across the regime boundary.
    use thermite::math::TranscendentalMath;

    for x in [0.5f64, 2.0, 10.0] {
        let e1 = c(x).expint::<1>();
        let e2 = c(x).expint::<2>();
        let want = ((-c(x)).exp() - c(x) * e1) / C::new(V::ONE);
        let err = ((e2.value.extract::<0>() - want.value.extract::<0>()) / want.value.extract::<0>()).abs();

        assert!(err <= 1e-28, "E_2({x}) vs recurrence: err {err:e}");
    }
}
