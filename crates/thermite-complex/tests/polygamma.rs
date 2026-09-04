//! Complex polygamma `psi_n(z)`, n >= 2, against mpmath and against itself.
//!
//! Five independent angles: the mpmath sweep (both half-planes, `polygamma_reference.py`
//! writes the table and asserts its own oracle against numerical differentiation of
//! digamma), the real-axis differential against thermite-special's real kernel (a
//! different algorithm: minimax-free recurrence/series here, walk + Bernoulli there,
//! with different transition points), conjugate symmetry, the forward recurrence
//! identity, and the delegation/limit pins.
#![cfg(feature = "special")]

use num_complex::Complex64;
use thermite::math::policy::DefaultPolicy;
use thermite::prelude::*;
use thermite_special::SpecialMathWithPolicy;

use thermite_complex::Complex;

type V = Vector<f64>;
type C = Complex<V>;
type Vf = Vector<f32>;
type Cf = Complex<Vf>;

include!("polygamma_ref/table.rs");

fn c(re: f64, im: f64) -> C {
    Complex::new(V::splat(re), V::splat(im))
}

fn parts(z: C) -> (f64, f64) {
    (z.re.extract::<0>(), z.im.extract::<0>())
}

fn pg(z: C, n: u32) -> Complex64 {
    let (re, im) = parts(z.polygamma_p::<DefaultPolicy>(n));
    Complex64::new(re, im)
}

/// Normwise relative error (see the crate's accuracy conventions: componentwise
/// claims near the axes read far worse than they are).
fn rel(got: Complex64, want: Complex64) -> f64 {
    if want.norm() == 0.0 {
        return if got.norm() == 0.0 { 0.0 } else { f64::INFINITY };
    }
    (got - want).norm() / want.norm()
}

#[test]
fn polygamma_c_matches_mpmath() {
    let mut worst = 0.0f64;
    let mut bad = 0usize;
    for &(n, zr, zi, wr, wi) in POLYGAMMA_C.iter() {
        let got = pg(c(zr, zi), n);
        let want = Complex64::new(wr, wi);
        let e = rel(got, want);
        // Reflected rows pay the exponential amplification documented on the kernel:
        // sin_pi/cos_pi carry e^(pi |Im z|), raised to ~2n powers, so their rounding
        // is amplified by ~(2n+2) pi |Im z|. The second term is that, in ulp.
        let tol = 3e-13 + ((2 * n + 2) as f64) * core::f64::consts::PI * zi.abs() * 1e-15;
        if e > tol {
            bad += 1;
            std::println!("psi_{n}({zr} + {zi}i): got {got}, want {want}, rel {e:e} (tol {tol:e})");
        }
        worst = worst.max(e);
    }
    std::println!(
        "complex f64 sweep: worst normwise rel err {worst:e} over {} rows",
        POLYGAMMA_C.len()
    );
    assert!(bad == 0, "{bad} rows over the gate, worst {worst:e}");
}

#[test]
fn polygamma_c_f32_matches_mpmath() {
    let mut worst = 0.0f64;
    let mut rows = 0usize;
    for &(n, zr, zi, wr, wi) in POLYGAMMA_C.iter() {
        // Coordinates are exact in f32 by the generator's construction. Skip rows
        // outside f32's range, past the reflected side's |Im z| ~ 28 reach (sin_pi's
        // own f32 overflow, the kernel-documented bound), and outside the
        // direct-power domain (see the real f32 sweep).
        let m = wr.abs().max(wi.abs());
        let overflow_direct = (n as f64 + 1.0) * zr.abs().max((2 + 4 * n) as f64).log10() > 36.0;
        if m > f32::MAX as f64 || m < 1e-36 || (zr < 0.5 && zi.abs() > 26.0) || overflow_direct {
            continue;
        }
        let z = Complex::new(Vf::splat(zr as f32), Vf::splat(zi as f32));
        let w = z.polygamma_p::<DefaultPolicy>(n);
        let got = Complex64::new(w.re.extract::<0>() as f64, w.im.extract::<0>() as f64);
        let e = rel(got, Complex64::new(wr, wi));
        assert!(
            e <= 1e-4,
            "f32 psi_{n}({zr} + {zi}i): got {got}, want {wr} + {wi}i, rel {e:e}"
        );
        rows += 1;
        worst = worst.max(e);
    }
    std::println!("complex f32 sweep: worst normwise rel err {worst:e} over {rows} rows");
}

#[test]
fn polygamma_c_real_axis_matches_real_kernel() {
    // On the real axis the complex routine must agree with the real kernel, which is
    // a different implementation (different transition points, minimax delegates at
    // low orders, real-only reflection form). Positive and negative axis both.
    for &n in &[2u32, 3, 5, 8, 12, 20] {
        for &x in &[0.75f64, 1.5, 3.25, 9.0, 25.0, 120.0, -0.75, -2.25, -6.5, -15.25] {
            let got = pg(c(x, 0.0), n);
            let want = V::splat(x).polygamma_p::<DefaultPolicy>(n).extract::<0>();
            assert!(
                (got.re - want).abs() <= 1e-12 * want.abs().max(1e-300),
                "psi_{n}({x}) on the axis: complex {}, real {want}",
                got.re
            );
            assert!(
                got.im == 0.0,
                "psi_{n}({x}): nonzero imaginary part {} on the real axis",
                got.im
            );
        }
    }
}

#[test]
fn polygamma_c_conjugate_symmetry() {
    // psi_n is real on the real axis, so psi_n(conj z) = conj(psi_n(z)) exactly (the
    // Schwarz reflection principle). A couple of ulp allowed for primitive rounding.
    for &n in &[2u32, 5, 10, 20] {
        for &(zr, zi) in &[(1.75, 2.5), (-3.25, 1.5), (0.25, -8.0), (12.0, 40.0)] {
            let a = pg(c(zr, zi), n);
            let b = pg(c(zr, -zi), n).conj();
            let e = rel(a, b);
            assert!(
                e <= 5e-15,
                "psi_{n} conjugate symmetry at ({zr}, {zi}): {a} vs {b}, rel {e:e}"
            );
        }
    }
}

#[test]
fn polygamma_c_recurrence() {
    // psi_n(z + 1) = psi_n(z) + (-1)^n n! z^-(n+1), both sides from the kernel, the
    // step in oracle arithmetic. The tolerance scales with what cancels.
    for &n in &[2u32, 4, 7, 11] {
        let mut fac = 1.0f64;
        for k in 2..=n {
            fac *= k as f64;
        }
        for &(zr, zi) in &[(0.75, 0.5), (2.5, -1.5), (-1.25, 0.75), (-6.5, -2.0), (0.0, 3.0)] {
            let z = Complex64::new(zr, zi);
            let lhs = pg(c(zr + 1.0, zi), n);
            let big = pg(c(zr, zi), n);
            let step = z.powi(-(n as i32 + 1)) * fac * if n % 2 == 1 { -1.0 } else { 1.0 };
            let rhs = big + step;
            let tol = 1e-13 * (1.0 + (big.norm() + step.norm()) / lhs.norm());
            let e = rel(lhs, rhs);
            assert!(
                e <= tol,
                "recurrence psi_{n} at ({zr}, {zi}): {lhs} vs {rhs}, rel {e:e} (tol {tol:e})"
            );
        }
    }
}

#[test]
fn polygamma_c_delegation_and_limits() {
    // n = 0 and 1 are the tuned complex digamma/trigamma, bit for bit.
    for &(zr, zi) in &[(1.5, 2.0), (-2.25, 0.5), (0.25, -3.0)] {
        let z = c(zr, zi);
        let (d0r, d0i) = parts(z.polygamma_p::<DefaultPolicy>(0));
        let (w0r, w0i) = parts(z.digamma_p::<DefaultPolicy>());
        let (d1r, d1i) = parts(z.polygamma_p::<DefaultPolicy>(1));
        let (w1r, w1i) = parts(z.trigamma_p::<DefaultPolicy>());
        assert!(
            d0r.to_bits() == w0r.to_bits()
                && d0i.to_bits() == w0i.to_bits()
                && d1r.to_bits() == w1r.to_bits()
                && d1i.to_bits() == w1i.to_bits(),
            "delegation at ({zr}, {zi})"
        );
    }

    // Poles on the negative real axis: the arithmetic diverges rather than lies.
    for &n in &[2u32, 3] {
        let w = pg(c(-3.0, 0.0), n);
        assert!(!w.re.is_finite(), "psi_{n}(-3) should not be finite, got {w}");
    }

    // Past the cot-pi table: reflected lanes NaN, right half-plane unaffected.
    let w = pg(c(-2.5, 1.0), 25);
    assert!(
        w.re.is_nan() && w.im.is_nan(),
        "psi_25 reflected should be NaN, got {w}"
    );
    let w = pg(c(3.5, 1.0), 25);
    assert!(w.re.is_finite() && w.im.is_finite(), "psi_25 right half-plane, got {w}");

    // Past the factorial table: NaN outright (no single signed infinity over C).
    let w = pg(c(3.5, 1.0), 200);
    assert!(w.re.is_nan(), "psi_200 past the factorial table, got {w}");
}

/// `Complex<Dual<..>>`: the low orders differentiate (trigamma runs the shared body
/// in dual arithmetic). `n >= 2` is unimplemented for this storage type and panics.
/// See the test below it.
#[cfg(feature = "dual")]
#[test]
fn polygamma_c_dual_low_orders_differentiate() {
    use thermite_dual::Dual;

    type D = Dual<V, 1>;
    type CD = Complex<D>;

    let z: CD = Complex::new(Dual::variable(V::splat(1.75), 0), Dual::constant(V::splat(0.5)));

    // psi_1 value and derivative: d/dz psi_1 = psi_2, checked against the plain
    // complex kernel's psi_2: two different code paths (dual arithmetic vs the
    // n >= 2 series).
    let w = z.polygamma_p::<DefaultPolicy>(1);
    let want_d = pg(c(1.75, 0.5), 2);
    let got_d = Complex64::new(w.re.dual[0].extract::<0>(), w.im.dual[0].extract::<0>());
    assert!(
        rel(got_d, want_d) <= 1e-11,
        "d/dz psi_1 through Complex<Dual>: got {got_d}, psi_2 {want_d}"
    );
}

// `Complex<Dual>` has no element tables for the higher orders, so `polygamma(n >= 2)` is
// unimplemented rather than undefined, and panics. It used to return NaN, which is
// indistinguishable from a genuine domain result and propagates silently. The owner's call
// (2026-08-30) is that a missing implementation should panic.
//
// Note this is only possible because `n` is a _scalar_: the per-lane NaNs elsewhere in this
// file (reflected lanes past the cot table, orders past the factorial table) are limits of an
// implemented algorithm and stay NaN, because one lane of a vector cannot panic.
#[cfg(feature = "dual")]
#[test]
#[should_panic(expected = "complex polygamma(n >= 2)")]
fn complex_dual_polygamma_high_order_is_unimplemented() {
    use thermite_dual::Dual;
    type D = Dual<V, 1>;

    let z: Complex<D> = Complex::new(Dual::variable(V::splat(2.5), 0), Dual::constant(V::splat(0.75)));
    let _ = z.polygamma_p::<DefaultPolicy>(2);
}
