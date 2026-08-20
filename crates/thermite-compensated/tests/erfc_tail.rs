//! `erf` / `erfc` over the continued-fraction tail (|x| >= 3).
//!
//! Regression: this whole regime returns NaN unguarded. Lentz's method seeds with a `tiny`
//! sentinel and reciprocates it on the first step, and `MIN_POSITIVE` reciprocates to
//! 4.5e307, past the ~1.3e300 where compensated multiplication's Dekker 2^27+1 splitter
//! overflows, so the next product is infinity and everything downstream NaN. Only the
//! CF branch reaches that code, which is exactly why the failure starts at |x| = 3.
//!
//! References are mpmath 1.3.0 at 45 digits, as `(hi, lo)` pairs.

use thermite::prelude::*;
use thermite_compensated::Compensated;
use thermite_special::SpecialMath;

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

const ERFC_TAIL: &[(f64, f64, f64)] = &[
    (3.0, 2.20904969985854412e-05, 1.55633779603434573e-22),
    (3.5, 7.43098372341412777e-07, -3.11706774906308902e-23),
    (4.0, 1.54172579002800200e-08, -1.14178721683710257e-24),
    (5.0, 1.53745979442803494e-12, -8.56941822207909581e-29),
    (6.5, 3.84214832712064749e-20, -2.44558168257361035e-37),
    (10.0, 2.08848758376254488e-45, -1.20065657635013813e-61),
    (-3.5, 1.99999925690162761e+00, 4.96472791872122038e-17),
    (-5.0, 1.99999999999846256e+00, -2.29499271180730108e-17),
];

#[test]
fn erfc_tail_is_finite_and_accurate() {
    for &(x, hi, lo) in ERFC_TAIL {
        let got = c(x).erfc();
        assert!(got.value.extract::<0>().is_finite(), "erfc({x}) is not finite");

        let err = dd_err(got, hi, lo);
        assert!(err <= 1e-28, "erfc({x}): rel err {err:e}");
    }
}

#[test]
fn erf_and_erfc_sum_to_one_across_the_regime_split() {
    // erf + erfc == 1 exactly. Spanning x = 3 puts the two branches either side of the
    // split, so this also checks they agree where they meet.
    for x in [2.5f64, 2.9, 2.999, 3.0, 3.001, 3.5, 5.0] {
        let s = c(x).erf() + c(x).erfc();
        let err = (s.value.extract::<0>() - 1.0).abs() + s.error.extract::<0>().abs();

        assert!(err <= 1e-30, "erf({x}) + erfc({x}) - 1 = {err:e}");
    }
}
