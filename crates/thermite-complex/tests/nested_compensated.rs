//! `Complex<Compensated<V>>`: complex arithmetic carried in double-double.
//!
//! Nothing here is written against `Compensated`; it is written against
//! `RealValue`, which `Compensated` satisfies, so the complex arithmetic and the
//! complex kernels pick up the extra precision on their own.

#![cfg(feature = "compensated")]

use thermite::math::TranscendentalMath;
use thermite::prelude::*;
use thermite_compensated::Compensated;
use thermite_complex::Complex;
use thermite_complex::prelude::ComplexMath;

/// A 1-lane f64 double-double, i.e. ~106 bits of mantissa.
type C = Compensated<Vector<f64>>;
type Z = Complex<C>;

fn z(re: f64, im: f64) -> Z {
    Complex::new(Compensated::new(Vector::splat(re)), Compensated::new(Vector::splat(im)))
}

/// The folded value of each part, back in plain f64.
fn parts(w: Z) -> (f64, f64) {
    (w.re.value().extract::<0>(), w.im.value().extract::<0>())
}

/// The error terms, which are what the extra precision lives in.
fn errors(w: Z) -> (f64, f64) {
    (w.re.error().extract::<0>(), w.im.error().extract::<0>())
}

#[test]
fn arithmetic_carries_the_error_term() {
    // (1 + 2i)(3 + 4i) = -5 + 10i, exactly representable, so the error terms stay 0.
    let p = z(1.0, 2.0) * z(3.0, 4.0);

    assert_eq!(parts(p), (-5.0, 10.0));
    assert_eq!(errors(p), (0.0, 0.0));

    // 1/3 is not representable. The double-double reciprocal keeps the bits that a
    // plain f64 would drop, so error != 0 and value+error is closer to the truth.
    let inv = z(3.0, 0.0).finv();
    let (re, _) = parts(inv);
    let (re_err, _) = errors(inv);

    assert_eq!(re, 1.0f64 / 3.0);
    assert!(re_err != 0.0, "the discarded bits of 1/3 must land in the error term");

    // value + error is a better 1/3 than f64 can hold: multiplying back by 3 in
    // double-double returns exactly 1, where f64 (1/3)*3 does not round-trip
    // through the same sequence.
    let back = inv * z(3.0, 0.0);
    assert_eq!(parts(back), (1.0, 0.0));
    assert!(errors(back).0.abs() < 1e-30);
}

/// A kernel written once over the vector traits gains ~106-bit precision by
/// instantiation alone.
#[test]
fn kernels_gain_precision() {
    // exp(i*pi) = -1. Im is sin(pi), which is only ever as close to zero as pi is
    // accurate: the f64 pi is off by ~1.2e-16, and a plain f64 sin(PI) lands there.
    // The double-double pi is good to ~1e-32 and the kernel tracks it.
    let pi = Compensated::<Vector<f64>>::PI;
    let w = Complex::new(Compensated::new(Vector::splat(0.0)), pi).exp();

    let (re, im) = parts(w);

    let f64_sin_pi = std::f64::consts::PI.sin(); // ~1.2246e-16

    assert!((re + 1.0).abs() < 1e-30, "Re exp(i*pi) = -1, got {re}");
    assert!(
        im.abs() < 1e-25,
        "Im exp(i*pi) = {im}, no better than the f64 sin(PI) of {f64_sin_pi:e}"
    );
}

/// `Complex<Compensated<V>>` is a full vector type, like the other nestings: the
/// modulus comes back as the real type, which here is a `Compensated`.
#[test]
fn complex_math_family_through_compensated() {
    let w = z(3.0, 4.0);

    let n = w.norm(); // Compensated<Vector<f64>>, i.e. Self::Real
    assert_eq!(n.value().extract::<0>(), 5.0);

    let (r, theta) = w.to_polar();
    assert_eq!(r.value().extract::<0>(), 5.0);
    assert!((theta.value().extract::<0>() - 4f64.atan2(3.0)).abs() < 1e-15);

    // sqrt(3 + 4i) = 2 + i, exactly.
    let s = w.sqrt();
    assert!((parts(s).0 - 2.0).abs() < 1e-28 && (parts(s).1 - 1.0).abs() < 1e-28);
}
