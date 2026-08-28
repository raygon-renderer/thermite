//! `Complex<Dual<V, N>>`, complex arithmetic whose parts carry derivatives.
//!
//! Nothing here is written against `Dual`. It is written against `RealValue`,
//! which `Dual` satisfies, so the complex kernels differentiate themselves. For a
//! holomorphic `f` seeded along the real axis (`dz = 1`) the dual parts are `f'(z)`:
//!
//!   f(x + iy) = u + iv,  seeded with dx = 1  =>  du/dx + i dv/dx = f'(z)
//!
//! It is the Cauchy-Riemann equations that make that the full complex derivative
//! rather than a directional one.

#![cfg(feature = "dual")]

use thermite::math::TranscendentalMath;
use thermite::prelude::*;
use thermite_complex::Complex;
use thermite_complex::prelude::ComplexMath;
use thermite_dual::Dual;

/// Inner: a 1-lane f64 vector carrying one derivative direction.
type D = Dual<Vector<f64>, 1>;
/// A complex number over that: value + d/dx of each part.
type C = Complex<D>;

/// `z` seeded so that d/dx applies to the real part (dz = 1).
fn seeded(re: f64, im: f64) -> C {
    Complex::new(
        Dual::variable(Vector::splat(re), 0), // re, with dre/dx = 1
        Dual::constant(Vector::splat(im)),    // im, with dim/dx = 0
    )
}

/// (value, derivative) of a complex result, as plain f64 pairs.
fn split(z: C) -> ((f64, f64), (f64, f64)) {
    (
        (z.re.re.extract::<0>(), z.im.re.extract::<0>()),
        (z.re.dual[0].extract::<0>(), z.im.dual[0].extract::<0>()),
    )
}

#[track_caller]
fn assert_deriv(what: &str, got: C, val: (f64, f64), d: (f64, f64), tol: f64) {
    let ((vr, vi), (dr, di)) = split(got);

    assert!(
        (vr - val.0).abs() < tol && (vi - val.1).abs() < tol,
        "{what}: value {vr} + {vi}i, want {} + {}i",
        val.0,
        val.1
    );
    assert!(
        (dr - d.0).abs() < tol && (di - d.1).abs() < tol,
        "{what}: derivative {dr} + {di}i, want {} + {}i",
        d.0,
        d.1
    );
}

/// A function written once over the vector traits, evaluated on `Complex<Dual<..>>`,
/// returns the complex value and the complex derivative.
#[test]
fn holomorphic_derivatives() {
    // f(z) = z^2  =>  f'(z) = 2z
    let z = seeded(3.0, 4.0);
    let f = z * z;
    assert_deriv("z^2", f, (9.0 - 16.0, 24.0), (6.0, 8.0), 1e-12);

    // f(z) = exp(z)  =>  f'(z) = exp(z)
    let z = seeded(0.5, 0.25);
    let f = z.exp();
    let (v, d) = split(f);
    assert_deriv("exp(z)", f, v, v, 1e-12); // derivative == value
    assert!(d == v);

    // f(z) = 1/z  =>  f'(z) = -1/z^2. At z = i: 1/i = -i, -1/i^2 = 1.
    let z = seeded(0.0, 1.0);
    assert_deriv("1/z at i", z.approx_reciprocal(), (0.0, -1.0), (1.0, 0.0), 1e-12);

    // f(z) = sin(z)  =>  f'(z) = cos(z)
    let z = seeded(0.7, -0.3);
    let (sv, _) = split(z.sin());
    let (cv, _) = split(z.cos());
    let (_, sd) = split(z.sin());
    assert!(
        (sd.0 - cv.0).abs() < 1e-12 && (sd.1 - cv.1).abs() < 1e-12,
        "d/dz sin(z) must equal cos(z): got {sd:?}, want {cv:?}"
    );
    let (_, cd) = split(z.cos());
    assert!(
        (cd.0 + sv.0).abs() < 1e-12 && (cd.1 + sv.1).abs() < 1e-12,
        "d/dz cos(z) must equal -sin(z)"
    );

    // f(z) = ln(z)  =>  f'(z) = 1/z. At z = 3+4i: 1/z = (3 - 4i)/25.
    let z = seeded(3.0, 4.0);
    assert_deriv(
        "ln(z)",
        z.ln(),
        (25f64.sqrt().ln(), 4f64.atan2(3.0)),
        (0.12, -0.16),
        1e-12,
    );

    // sqrt is built on the modulus, a non-holomorphic block, but the result is
    // holomorphic: d/dz sqrt(z) = 1/(2 sqrt(z)).
    let z = seeded(3.0, 4.0);
    let (v, d) = split(z.sqrt());
    let (sr, si) = v; // sqrt(3+4i) = 2 + i

    // 1/(2s) = conj(s) / (2|s|^2) = (2 - i)/10 = 0.2 - 0.1i
    let denom = 2.0 * (sr * sr + si * si);
    assert_deriv("sqrt(z)", z.sqrt(), (2.0, 1.0), (sr / denom, -si / denom), 1e-12);
    assert!((d.0 - 0.2).abs() < 1e-12 && (d.1 + 0.1).abs() < 1e-12, "got {d:?}");
}

/// The `ComplexMath` family threads through the nesting: `norm` returns the real
/// type, here a `Dual`, so the modulus arrives with its own derivative,
/// d|z|/dx = re/|z|.
#[test]
fn complex_math_family_through_dual() {
    let z = seeded(3.0, 4.0);

    let n = z.norm(); // Dual<Vector<f64>, 1>, i.e. Self::Real
    assert!((n.re.extract::<0>() - 5.0).abs() < 1e-12, "|3+4i| == 5");
    assert!(
        (n.dual[0].extract::<0>() - 0.6).abs() < 1e-12,
        "d|z|/dx = re/|z| = 3/5, got {}",
        n.dual[0].extract::<0>()
    );

    // arg(z) = atan2(im, re); d arg/dx = -im/|z|^2 = -4/25
    let a = z.arg();
    assert!((a.re.extract::<0>() - 4f64.atan2(3.0)).abs() < 1e-12);
    assert!((a.dual[0].extract::<0>() + 0.16).abs() < 1e-12);
}
