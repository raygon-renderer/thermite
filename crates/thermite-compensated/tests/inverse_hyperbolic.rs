//! `asinh` / `atanh` near zero.
//!
//! Regression: both lost ~41 bits below |x| ~ 1e-10. Each formed a quantity that tends to
//! 1 as x -> 0 - `x + sqrt(x^2 + 1)` for asinh, `(1 + x)/(1 - x)` for atanh - and then
//! took `ln` of it. A double-double near 1 holds the part that carries the answer to only
//! `106 - log2(1/x)` bits, so at x = 1e-14 the result was good to ~65.
//!
//! Both now compute the offset from 1 in closed form and feed it to `ln_1p`, which costs
//! the same as `ln`:
//!
//!   x + sqrt(x^2 + 1) - 1 = x + x^2/(1 + sqrt(1 + x^2))
//!   (1 + x)/(1 - x)   - 1 = 2x/(1 - x)
//!
//! References are mpmath 1.3.0 at 45 digits, as `(hi, lo)` pairs.

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

    (((value - hi) + (error - lo)) / hi.abs()).abs()
}

const ASINH: &[(f64, f64, f64)] = &[
    (1e-16, 9.99999999999999979e-17, -1.66666666666667675e-49),
    (1e-14, 9.99999999999999999e-15, -1.66666666666666660e-43),
    (1e-10, 1.00000000000000004e-10, -1.66666666666666688e-31),
    (0.0001, 9.99999998333333433e-05, -4.43347621373975211e-21),
    (0.5, 4.81211825059603471e-01, -2.32578170134627362e-17),
    (5.0, 2.31243834127275250e+00, 1.24683686877521065e-16),
    (10000000000.0, 2.37189981105004009e+01, 1.29412812109419892e-15),
    (-1e-14, -9.99999999999999999e-15, 1.66666666666666660e-43),
    (-3.0, -1.81844645923206683e+00, 1.76749607778565466e-18),
];

const ATANH: &[(f64, f64, f64)] = &[
    (1e-16, 9.99999999999999979e-17, 3.33333333333335351e-49),
    (1e-14, 9.99999999999999999e-15, 3.33333333333333319e-43),
    (1e-10, 1.00000000000000004e-10, 3.33333333333333376e-31),
    (0.0001, 1.00000000333333341e-04, -1.18557472323215708e-21),
    (0.5, 5.49306144334054891e-01, -4.53564861750076498e-17),
    (0.999, 3.80020116725019941e+00, 1.79732376580802884e-16),
    (-1e-14, -9.99999999999999999e-15, -3.33333333333333319e-43),
    (-0.75, -9.72955074527656616e-01, -3.66179310395245338e-17),
];

#[test]
fn asinh_matches_mpmath() {
    for &(x, hi, lo) in ASINH {
        let err = dd_err(c(x).asinh(), hi, lo);
        assert!(err <= 1e-30, "asinh({x:e}): rel err {err:e}");
    }
}

#[test]
fn atanh_matches_mpmath() {
    for &(x, hi, lo) in ATANH {
        let err = dd_err(c(x).atanh(), hi, lo);
        assert!(err <= 1e-30, "atanh({x:e}): rel err {err:e}");
    }
}

#[test]
fn small_argument_series_agreement() {
    // asinh(x) = x - x^3/6 + O(x^5) and atanh(x) = x + x^3/3 + O(x^5). At x = 1e-10 the
    // cubic term is 1e-30 - about 2^-100 of the value - so it lands squarely in the low
    // word and needs no oracle to check. This is the term that used to be lost entirely.
    let x = 1e-10f64;

    let a = c(x).asinh();
    let want_a = -x * x * x / 6.0;
    assert!(
        ((a.error.extract::<0>() - want_a) / want_a).abs() <= 1e-6,
        "asinh low word {:e} should be ~{want_a:e}",
        a.error.extract::<0>()
    );

    let t = c(x).atanh();
    let want_t = x * x * x / 3.0;
    assert!(
        ((t.error.extract::<0>() - want_t) / want_t).abs() <= 1e-6,
        "atanh low word {:e} should be ~{want_t:e}",
        t.error.extract::<0>()
    );
}
