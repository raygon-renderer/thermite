//! `sincos_pi` on `Compensated`, and the precision it exists to protect.
//!
//! The inherited default is `sin_cos(self * PI)`, which rounds the product before any
//! reduction happens and so carries an absolute error of about `|x| * 2^-106` into the
//! argument. `Compensated` overrides it to reduce first - `sin(pi(n + r)) = (-1)^n
//! sin(pi r)` with `x - round(x)` exact - so the only rounded product involves
//! `|r| <= 1/2` and the error does not grow with the argument. That matters because the
//! gamma reflection goes through `sin_pi` at every negative argument, large ones
//! included.
//!
//! # Reference values
//!
//! mpmath 1.3.0 at 40 digits, as `(hi, lo)` pairs so both words of the result are
//! checked - collapsing to one `f64` would only ever verify the top half, which is the
//! half this type is not about.
//!
//! Generated from `mp.mpf(x)` on the Python **float**, never from the decimal string.
//! `-1000.3` is not representable: the nearest `f64` is off by 4.5e-14, and `sinpi` of
//! the two differs in the 13th digit. Referencing the decimal measures that gap rather
//! than the implementation, which is a confusing way to fail.

use thermite::math::TranscendentalMath;
use thermite::prelude::*;
use thermite_compensated::Compensated;

type V = Vector<f64>;
type C = Compensated<V>;

fn c(x: f64) -> C {
    C::new(V::splat(x))
}

/// Error against a double-double reference, across both words.
///
/// The leading subtraction is exact (the values agree to well within an octave), so no
/// precision is lost forming the residual and this really does see ~106 bits.
fn dd_err(got: C, hi: f64, lo: f64) -> f64 {
    let value = got.value.extract::<0>();
    let error = got.error.extract::<0>();

    (((value - hi) + (error - lo)) / hi.abs().max(1.0)).abs()
}

/// `(x, hi, lo)` from mpmath. See the module docs on how these were generated.
const SIN_PI: &[(f64, f64, f64)] = &[
    (0.25, 7.07106781186547573e-01, -4.83364665672645673e-17),
    (0.5, 1.00000000000000000e+00, 0.00000000000000000e+00),
    (-0.3, -8.09016994374947451e-01, 4.76617526690622591e-17),
    (1.75, -7.07106781186547573e-01, 4.83364665672645673e-17),
    (-1000.5, -1.00000000000000000e+00, 0.00000000000000000e+00),
    (-1000.3, -8.09016994374863407e-01, -4.39028270363870053e-17),
    (12345.25, -7.07106781186547573e-01, 4.83364665672645673e-17),
    (-7.125, 3.82683432365089782e-01, -1.00507726964615876e-17),
];

#[test]
fn sin_pi_matches_mpmath_to_double_double() {
    for &(x, hi, lo) in SIN_PI {
        let err = dd_err(c(x).sin_pi(), hi, lo);

        assert!(err <= 1e-30, "sin_pi({x}): rel err {err:e}");
    }
}

#[test]
fn cos_pi_matches_shifted_sin() {
    // cos(pi x) = sin(pi (x + 1/2)), an identity the implementation does not use: both
    // come out of a single reduction, so agreement here means the shared sign flip is
    // correct for each. Only half-integer shifts, which stay exact.
    for &x in &[0.25f64, 0.5, 1.75, -7.125, -1000.5] {
        let cos = c(x).cos_pi();
        let shifted = c(x + 0.5).sin_pi();

        let err = dd_err(cos, shifted.value.extract::<0>(), shifted.error.extract::<0>());

        assert!(err <= 1e-30, "cos_pi({x}) vs sin_pi({x} + 1/2): rel err {err:e}");
    }
}

#[test]
fn integers_and_half_integers_are_exact() {
    // sin(pi n) = 0 and |sin(pi(n + 1/2))| = 1 exactly - pinned by the reduction being
    // exact, not by the series converging. Both words must be clean.
    for n in [-8i32, -3, -1, 0, 1, 2, 7, 40] {
        let s = c(n as f64).sin_pi();
        assert_eq!(s.value.extract::<0>(), 0.0, "sin_pi({n}) value");
        assert_eq!(s.error.extract::<0>(), 0.0, "sin_pi({n}) error");

        let h = c(n as f64 + 0.5).sin_pi();
        assert_eq!(h.value.extract::<0>().abs(), 1.0, "|sin_pi({n} + 1/2)| value");
        assert_eq!(h.error.extract::<0>(), 0.0, "|sin_pi({n} + 1/2)| error");
    }
}

#[test]
fn precision_does_not_decay_with_argument() {
    // The whole point of the override. sin_pi(k + 1/4) is the same number for every
    // integer k, so any drift as k grows is argument-reduction error and nothing else.
    // Every input here is exactly representable, so the comparison is clean.
    //
    // Under the inherited `sin_cos(x * PI)` this degrades like |x| * 2^-106: by k = 1e9
    // that is around 1e-23, which blows the tolerance below by seven orders of magnitude.
    let (hi, lo) = (7.07106781186547573e-01, -4.83364665672645673e-17);

    for k in [0.0f64, 4.0, 1024.0, 1e6, 1e9] {
        let err = dd_err(c(k + 0.25).sin_pi(), hi, lo);

        assert!(err <= 1e-30, "sin_pi({k} + 0.25) drifted: rel err {err:e}");
    }
}
