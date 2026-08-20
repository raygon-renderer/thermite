//! Special functions over C (`special` feature).
//!
//! `erf` is checked against an independent oracle: the defining integral
//!
//!   erf(z) = (2/sqrt(pi)) * integral_0^z e^{-t^2} dt
//!
//! by Simpson's rule along the straight path 0 -> z (the integrand is entire, so the
//! path does not matter). It shares no code with the implementation, which uses the
//! A&S 7.1.6 expansion and a continued fraction.

#![cfg(feature = "special")]

use num_complex::Complex64;
use thermite::prelude::*;
use thermite_special::SpecialMath;

use thermite_complex::Complex;

type V = Vector<f64>;
type C = Complex<V>;

fn c(re: f64, im: f64) -> C {
    Complex::new(V::splat(re), V::splat(im))
}

fn parts(z: C) -> (f64, f64) {
    (z.re.extract::<0>(), z.im.extract::<0>())
}

fn close_c(got: C, want: Complex64, tol: f64) -> bool {
    let (re, im) = parts(got);
    let scale = want.norm().max(1.0);

    (re - want.re).abs() <= tol * scale && (im - want.im).abs() <= tol * scale
}

fn oracle(z: C) -> Complex64 {
    let (re, im) = parts(z);
    Complex64::new(re, im)
}

/// erf(z) by Simpson's rule on t in [0, 1], integrating z * e^{-(z t)^2}.
fn erf_integral(z: Complex64) -> Complex64 {
    const N: usize = 20_000; // even
    let h = 1.0 / N as f64;

    let f = |t: f64| {
        let zt = z * t;
        (-(zt * zt)).exp() * z
    };

    let mut acc = f(0.0) + f(1.0);
    for i in 1..N {
        let w = if i % 2 == 0 { 2.0 } else { 4.0 };
        acc += f(i as f64 * h) * w;
    }

    acc * (h / 3.0) * (2.0 / std::f64::consts::PI.sqrt())
}

#[track_caller]
fn assert_close(what: &str, got: C, want: Complex64, tol: f64) {
    let (re, im) = parts(got);
    let err = ((re - want.re).powi(2) + (im - want.im).powi(2)).sqrt();

    assert!(
        err <= tol * want.norm().max(1.0),
        "{what}: got {re} + {im}i, want {} + {}i (err {err:e})",
        want.re,
        want.im
    );
}

/// The series regime, against the integral oracle, over all four quadrants.
#[test]
fn erf_matches_the_defining_integral() {
    let samples = [
        (0.0, 0.0),
        (0.5, 0.0),
        (0.0, 1.0),
        (1.0, 1.0),
        (-1.0, 0.75),
        (1.5, -2.0),
        (-2.0, -1.0),
        (0.25, 2.5),
        (3.0, 0.5),
        (-0.75, 3.0),
    ];

    for (re, im) in samples {
        let z = c(re, im);
        assert_close(&format!("erf({re} + {im}i)"), z.erf(), erf_integral(oracle(z)), 1e-12);
    }
}

/// On the real axis it must reproduce libm, including past the crossover where the
/// implementation switches formulas.
#[test]
fn erf_and_erfc_match_libm_on_the_real_axis() {
    for x in [
        0.0, 0.1, 0.5, 1.0, 2.0, 3.0, 5.0, 5.9, 6.1, 8.0, 12.0, 25.0, -1.0, -4.0, -7.0,
    ] {
        let z = c(x, 0.0);

        let (re, im) = parts(z.erf());
        assert!(
            (re - libm::erf(x)).abs() < 1e-14,
            "erf({x}): got {re}, want {}",
            libm::erf(x)
        );
        assert_eq!(im, 0.0, "erf of a real must be real");

        // erfc(12) ~ 1.4e-63, which 1 - erf(z) cannot represent at all: it gives 0.
        // Past Re(z^2) = 6 the continued fraction computes erfc directly, holding
        // relative accuracy even at 1e-63. Below that line it is 1 - erf, which is
        // exact to ~1 ulp, erf not yet being close enough to 1 to cancel.
        let (re, im) = parts(z.erfc());
        let want = libm::erfc(x);
        assert!(
            (re - want).abs() <= 1e-12 * want.abs(),
            "erfc({x}): got {re}, want {want} (rel {})",
            (re - want).abs() / want.abs()
        );
        assert_eq!(im, 0.0);
    }

    // 1 - erf(12) is 0 in f64. erfc(12) is not.
    let z = c(12.0, 0.0);
    assert_eq!(parts(C::ONE - z.erf()).0, 0.0, "premise: 1 - erf(12) underflows");
    assert!((parts(z.erfc()).0 - libm::erfc(12.0)).abs() < 1e-12 * libm::erfc(12.0));
}

/// Structural identities that hold for the entire function.
#[test]
fn erf_identities() {
    for (re, im) in [(0.7, 1.3), (-2.0, 0.5), (4.0, -3.0), (0.0, 2.0)] {
        let z = c(re, im);

        // odd: erf(-z) = -erf(z)
        let a = z.erf();
        let b = (-z).erf();
        assert_close("erf(-z) == -erf(z)", b, -oracle(a), 1e-13);

        // conjugate symmetry: erf(conj z) = conj(erf z)
        let cj = Complex::new(z.re, -z.im).erf();
        let want = oracle(a).conj();
        assert_close("erf(conj z) == conj(erf z)", cj, want, 1e-13);

        // erf(z) + erfc(z) == 1
        assert_close("erf + erfc == 1", z.erf() + z.erfc(), Complex64::new(1.0, 0.0), 1e-13);
    }
}

/// The holomorphic defaults are inherited unchanged, so check they are correct over
/// C rather than just compiling.
#[test]
fn inherited_polynomial_families() {
    let z = c(0.5, 0.75);
    let o = oracle(z);

    // H_3(x) = 8x^3 - 12x
    let want = o * o * o * 8.0 - o * 12.0;
    assert_close("hermite::<3>", z.hermite::<3>(), want, 1e-13);

    // P_2(x) = (3x^2 - 1)/2
    let want = (o * o * 3.0 - Complex64::new(1.0, 0.0)) / 2.0;
    assert_close("legendre(2)", z.legendre(2, 0), want, 1e-13);

    // gaussian(x; a, c) = a e^{-(x/c)^2 / 2}
    let (a, cc) = (2.0, 1.5);
    let t = o / cc;
    let want = (-(t * t) * 0.5).exp() * a;
    assert_close("gaussian", z.gaussian(c(a, 0.0), c(cc, 0.0)), want, 1e-13);
}

/// The two overridden defaults: their real-axis stabilizations use |x| and
/// max(x, 0), which are not holomorphic. The analytic definitions are.
#[test]
fn overridden_non_holomorphic_defaults() {
    let z = c(0.6, -1.1);
    let o = oracle(z);

    // sigma(z) = 1/(1 + e^{-z})
    let want = Complex64::new(1.0, 0.0) / (Complex64::new(1.0, 0.0) + (-o).exp());
    assert_close("logistic_sigmoid", z.logistic_sigmoid(), want, 1e-13);

    // softplus(z, k=1) = ln(1 + e^z)
    let want = (Complex64::new(1.0, 0.0) + o.exp()).ln();
    assert_close("softplus", z.softplus(C::ONE, C::ONE), want, 1e-13);
}

/// erf is entire, so `Complex<Dual<..>>` differentiates it:
/// d/dz erf = (2/sqrt(pi)) e^{-z^2}.
#[cfg(feature = "dual")]
#[test]
fn erf_derivative_through_dual() {
    use thermite_dual::Dual;

    type D = Dual<Vector<f64>, 1>;
    type CD = Complex<D>;

    let (x, y) = (0.8, 0.6);

    let z: CD = Complex::new(Dual::variable(V::splat(x), 0), Dual::constant(V::splat(y)));
    let w = z.erf();

    let d = (w.re.dual[0].extract::<0>(), w.im.dual[0].extract::<0>());

    let zo = Complex64::new(x, y);
    let want = (-(zo * zo)).exp() * (2.0 / std::f64::consts::PI.sqrt());

    assert!(
        (d.0 - want.re).abs() < 1e-11 && (d.1 - want.im).abs() < 1e-11,
        "d/dz erf: got {d:?}, want {} + {}i",
        want.re,
        want.im
    );
}

/// `erfc` in the continued-fraction region, against 40-digit reference values
/// (mpmath). The integral oracle cannot reach here: erfc is as small as 1e-64, which
/// `1 - erf` cannot represent.
#[test]
fn erfc_reference_values_in_the_cf_region() {
    #[rustfmt::skip]
    let refs: [((f64, f64), (f64, f64)); 8] = [
        ((2.5, 0.0),  (0.00040695201744495894, 0.0)),
        ((3.0, 1.0),  (5.7613867986237604e-5, -7.7179563813780136e-7)),
        ((4.0, 2.0),  (-5.6521700279349374e-7, 5.1310052960818763e-7)),
        ((5.0, 0.5),  (7.3572077658981947e-13, 1.8224380770767701e-12)),
        ((6.0, 3.0),  (5.0394073504603446e-14, 1.4870834637192687e-13)),
        ((8.0, 1.0),  (-2.7719938881562651e-29, 1.2198704619504604e-29)),
        ((12.0, 0.0), (1.3562611692059042e-64, 0.0)),
        ((3.5, -1.5), (-7.4080858189666369e-7, -6.5275285463703495e-6)),
    ];

    for ((x, y), (wr, wi)) in refs {
        let want = Complex64::new(wr, wi);
        let got = c(x, y).erfc();
        let (gr, gi) = parts(got);

        // Relative to |erfc| itself: an absolute tolerance would be vacuous at 1e-64.
        let err = ((gr - wr).powi(2) + (gi - wi).powi(2)).sqrt() / want.norm();

        assert!(
            err < 1e-13,
            "erfc({x} + {y}i): got {gr} + {gi}i, want {wr} + {wi}i (rel {err:e})"
        );
    }
}

// --- trigamma ---------------------------------------------------------------
//
// Checked by the two functional equations it is *not* built from. The
// implementation reflects once and then walks the recurrence to Re z >= 16 before
// expanding, so testing the recurrence at a single step and the reflection at an
// interior point exercises identities the code never assumes pointwise.

// `trigamma` lives only on `SpecializedSpecialMath`, and `lambert_w` is on both that
// and the public `SpecialMath`, so the import is scoped to keep the two unambiguous.
mod trigamma {
    use super::*;
    use thermite::math::policy::DefaultPolicy;
    use thermite_special::specialized::SpecializedSpecialMath as _;

    #[test]
    fn trigamma_matches_the_real_axis() {
        // On the real axis the complex routine must agree with thermite-special's real
        // trigamma, which is a completely different algorithm (three minimax rationals).
        for k in 0..40 {
            let x = 0.35 + (k as f64) * 0.5;

            let got = c(x, 0.0).trigamma::<DefaultPolicy>();
            let want = V::splat(x).trigamma::<DefaultPolicy>().extract::<0>();

            let (re, im) = parts(got);
            assert!(
                (re - want).abs() <= 1e-12 * want.abs().max(1.0),
                "trigamma re @ x={x}: got {re}, real impl {want}"
            );
            assert!(im.abs() < 1e-300, "trigamma im @ x={x}: got {im}, want 0");
        }
    }

    #[test]
    fn trigamma_recurrence() {
        // psi_1(z + 1) = psi_1(z) - 1/z^2
        for &(re, im) in &[(0.7, 0.4), (2.5, -1.25), (-3.4, 2.0), (1.0, 8.0), (-0.5, -0.5)] {
            let z = c(re, im);
            let lhs = c(re + 1.0, im).trigamma::<DefaultPolicy>();

            let zo = Complex64::new(re, im);
            let rhs = oracle(z.trigamma::<DefaultPolicy>()) - 1.0 / (zo * zo);

            assert!(
                close_c(lhs, rhs, 1e-11),
                "trigamma recurrence @ {zo}: got {:?}, want {rhs}",
                parts(lhs)
            );
        }
    }

    #[test]
    fn trigamma_reflection() {
        // psi_1(z) + psi_1(1 - z) = pi^2 / sin^2(pi z)
        for &(re, im) in &[(0.3, 0.6), (-1.7, 0.9), (2.2, -1.1), (-4.25, 3.0)] {
            let z = Complex64::new(re, im);

            let lhs =
                oracle(c(re, im).trigamma::<DefaultPolicy>()) + oracle(c(1.0 - re, -im).trigamma::<DefaultPolicy>());

            let s = (Complex64::new(std::f64::consts::PI, 0.0) * z).sin();
            let rhs = std::f64::consts::PI * std::f64::consts::PI / (s * s);

            let err = (lhs - rhs).norm() / rhs.norm().max(1.0);
            assert!(
                err < 1e-11,
                "trigamma reflection @ {z}: lhs {lhs}, rhs {rhs} (rel {err})"
            );
        }
    }
}

// --- lambert_w --------------------------------------------------------------
//
// The defining equation is its own oracle: whatever branch the iteration lands on,
// w e^w must reproduce z. The branch *labelling* is checked separately against the
// real implementation, where W_0 and W_{-1} are pinned down.

#[test]
fn lambert_w_satisfies_its_defining_equation() {
    for &(re, im) in &[
        (1.0, 0.0),
        (0.5, 0.5),
        (-0.2, 0.1),
        (3.0, -2.0),
        (-5.0, 4.0),
        (10.0, 10.0),
        (0.25, -0.75),
        (-0.3, 0.0),
    ] {
        let z = Complex64::new(re, im);
        let (w0, wm1) = c(re, im).lambert_w();

        for (name, w) in [("W_0", oracle(w0)), ("W_-1", oracle(wm1))] {
            let back = w * w.exp();
            let err = (back - z).norm() / z.norm().max(1.0);

            assert!(err < 1e-9, "{name} e^{name} @ {z}: got {back}, want {z} (rel {err})");
        }
    }
}

#[test]
fn lambert_w_matches_the_real_axis() {
    // W_0 is real for x >= -1/e, and both branches are real on [-1/e, 0).
    for k in 0..30 {
        let x = -0.36 + (k as f64) * 0.4;

        let (w0, _) = c(x, 0.0).lambert_w();
        let (rw0, _) = V::splat(x).lambert_w();

        let want = rw0.extract::<0>();
        let (re, im) = parts(w0);

        assert!(
            (re - want).abs() <= 1e-9 * want.abs().max(1.0),
            "lambert W_0 re @ x={x}: got {re}, real impl {want}"
        );
        assert!(im.abs() < 1e-9, "lambert W_0 im @ x={x}: got {im}, want 0");
    }
}

#[test]
fn lambert_w_branch_point_and_zero() {
    // Both branches meet at z = -1/e with W = -1.
    let (w0, wm1) = c(-core::f64::consts::E.recip(), 0.0).lambert_w();

    assert!(
        close_c(w0, Complex64::new(-1.0, 0.0), 1e-6),
        "W_0(-1/e) = {:?}",
        parts(w0)
    );
    assert!(
        close_c(wm1, Complex64::new(-1.0, 0.0), 1e-6),
        "W_-1(-1/e) = {:?}",
        parts(wm1)
    );

    // W_0(0) = 0 exactly; W_{-1}(0) = -inf.
    let (w0, wm1) = c(0.0, 0.0).lambert_w();

    assert_eq!(parts(w0), (0.0, 0.0));
    assert_eq!(wm1.re.extract::<0>(), f64::NEG_INFINITY);
}

// --- through Dual -----------------------------------------------------------

/// `psi_2` with no tetragamma anywhere: the trigamma body is generic over
/// `RealValue`, so seeding a `Dual` real part differentiates it directly. Checked
/// against a central difference of the plain complex `trigamma`, which shares no code
/// with the dual-number path.
#[cfg(feature = "dual")]
#[test]
fn trigamma_derivative_through_dual() {
    use thermite::math::policy::DefaultPolicy;
    use thermite_dual::Dual;
    use thermite_special::specialized::SpecializedSpecialMath as _;

    type D = Dual<Vector<f64>, 1>;
    type CD = Complex<D>;

    for &(x, y) in &[(2.5, 1.0), (0.75, -0.5), (-1.6, 2.25)] {
        let z: CD = Complex::new(Dual::variable(V::splat(x), 0), Dual::constant(V::splat(y)));
        let w = z.trigamma::<DefaultPolicy>();

        // value agrees with the non-dual routine
        let plain = c(x, y).trigamma::<DefaultPolicy>();
        let (pre, pim) = parts(plain);

        assert!(
            (w.re.re.extract::<0>() - pre).abs() < 1e-12 && (w.im.re.extract::<0>() - pim).abs() < 1e-12,
            "dual trigamma value @ ({x}, {y})"
        );

        // derivative against a central difference in the real direction
        let h = 1e-5;
        let (ap, bp) = parts(c(x + h, y).trigamma::<DefaultPolicy>());
        let (am, bm) = parts(c(x - h, y).trigamma::<DefaultPolicy>());
        let want = ((ap - am) / (2.0 * h), (bp - bm) / (2.0 * h));

        let got = (w.re.dual[0].extract::<0>(), w.im.dual[0].extract::<0>());
        let scale = (want.0.abs() + want.1.abs()).max(1.0);

        assert!(
            (got.0 - want.0).abs() < 1e-5 * scale && (got.1 - want.1).abs() < 1e-5 * scale,
            "psi_2 @ ({x}, {y}): got {got:?}, central diff {want:?}"
        );
    }
}

/// Lambert W differentiates too: `W'(z) = W / (z (1 + W))`.
#[cfg(feature = "dual")]
#[test]
fn lambert_w_derivative_through_dual() {
    use thermite_dual::Dual;

    type D = Dual<Vector<f64>, 1>;
    type CD = Complex<D>;

    let (x, y) = (1.5, 0.75);

    let z: CD = Complex::new(Dual::variable(V::splat(x), 0), Dual::constant(V::splat(y)));
    let (w0, _) = z.lambert_w();

    let w = Complex64::new(w0.re.re.extract::<0>(), w0.im.re.extract::<0>());
    let zo = Complex64::new(x, y);
    let want = w / (zo * (1.0 + w));

    let got = (w0.re.dual[0].extract::<0>(), w0.im.dual[0].extract::<0>());

    assert!(
        (got.0 - want.re).abs() < 1e-9 && (got.1 - want.im).abs() < 1e-9,
        "dW/dz @ ({x}, {y}): got {got:?}, want {} + {}i",
        want.re,
        want.im
    );
}

// --- Gamma family -----------------------------------------------------------
//
// Anchored on one reference value, then held in place by the functional equations,
// which the implementation never assumes pointwise: it evaluates a Lanczos sum and
// reflects once, so the recurrence in particular is entirely independent of it.

#[test]
fn tgamma_reference_value() {
    // Gamma(1 + i), to 25 digits.
    let got = c(1.0, 1.0).tgamma();
    let want = Complex64::new(0.4980156681183560427136912, -0.1549498283018106851249551);

    assert!(
        close_c(got, want, 1e-13),
        "Gamma(1+i): got {:?}, want {want}",
        parts(got)
    );
}

#[test]
fn tgamma_matches_the_real_axis() {
    for k in 0..60 {
        let x = -6.4 + (k as f64) * 0.31; // spans the reflected half-plane too
        let want = libm::tgamma(x);

        if !want.is_finite() || want.abs() > 1e12 {
            continue;
        }

        let got = c(x, 0.0).tgamma();
        let (re, im) = parts(got);

        assert!(
            (re - want).abs() <= 1e-11 * want.abs().max(1.0),
            "tgamma re @ x={x}: got {re}, libm {want}"
        );
        assert!(im.abs() <= 1e-11 * want.abs().max(1.0), "tgamma im @ x={x}: got {im}");
    }
}

#[test]
fn tgamma_recurrence() {
    // Gamma(z + 1) = z Gamma(z)
    for &(re, im) in &[(0.7, 0.4), (2.5, -1.25), (-3.4, 2.0), (1.0, 4.0), (-0.5, -0.5)] {
        let lhs = oracle(c(re + 1.0, im).tgamma());
        let rhs = Complex64::new(re, im) * oracle(c(re, im).tgamma());

        let err = (lhs - rhs).norm() / rhs.norm().max(1e-300);
        assert!(
            err < 1e-11,
            "Gamma recurrence @ ({re}, {im}): {lhs} vs {rhs} (rel {err})"
        );
    }
}

#[test]
fn tgamma_reflection() {
    // Gamma(z) Gamma(1 - z) = pi / sin(pi z)
    for &(re, im) in &[(0.3, 0.6), (-1.7, 0.9), (2.2, -1.1), (0.25, 0.0)] {
        let z = Complex64::new(re, im);

        let lhs = oracle(c(re, im).tgamma()) * oracle(c(1.0 - re, -im).tgamma());
        let rhs = std::f64::consts::PI / (Complex64::new(std::f64::consts::PI, 0.0) * z).sin();

        let err = (lhs - rhs).norm() / rhs.norm().max(1.0);
        assert!(err < 1e-11, "Gamma reflection @ {z}: {lhs} vs {rhs} (rel {err})");
    }
}

#[test]
fn lgamma_exponentiates_to_tgamma() {
    // Only on Re z >= 1/2, where lgamma is the continuous branch and the two agree
    // exactly rather than up to 2 pi i.
    for &(re, im) in &[(0.6, 0.3), (2.0, 1.5), (5.0, -2.0), (1.25, 0.0)] {
        let lhs = oracle(c(re, im).lgamma()).exp();
        let rhs = oracle(c(re, im).tgamma());

        let err = (lhs - rhs).norm() / rhs.norm().max(1e-300);
        assert!(
            err < 1e-11,
            "exp(lgamma) vs tgamma @ ({re}, {im}): {lhs} vs {rhs} (rel {err})"
        );
    }
}

#[test]
fn lgamma_matches_the_real_axis() {
    for k in 0..40 {
        let x = 0.2 + (k as f64) * 0.7;
        let want = libm::lgamma(x);

        let got = c(x, 0.0).lgamma();
        let (re, im) = parts(got);

        assert!(
            (re - want).abs() <= 1e-11 * want.abs().max(1.0),
            "lgamma re @ x={x}: got {re}, libm {want}"
        );
        assert!(im.abs() < 1e-11, "lgamma im @ x={x}: got {im}");
    }
}

#[test]
fn digamma_recurrence() {
    // psi(z + 1) = psi(z) + 1/z
    for &(re, im) in &[(0.7, 0.4), (2.5, -1.25), (-3.4, 2.0), (1.0, 6.0), (-0.5, -0.5)] {
        let zo = Complex64::new(re, im);

        let lhs = oracle(c(re + 1.0, im).digamma());
        let rhs = oracle(c(re, im).digamma()) + 1.0 / zo;

        let err = (lhs - rhs).norm() / rhs.norm().max(1.0);
        assert!(err < 1e-11, "psi recurrence @ {zo}: {lhs} vs {rhs} (rel {err})");
    }
}

#[test]
fn digamma_matches_the_real_axis() {
    use thermite::math::policy::DefaultPolicy;
    use thermite_special::SpecialMathWithPolicy;

    for k in 0..40 {
        let x = 0.35 + (k as f64) * 0.5;

        let got = c(x, 0.0).digamma();
        let want = SpecialMathWithPolicy::digamma_p::<DefaultPolicy>(V::splat(x)).extract::<0>();

        let (re, im) = parts(got);
        assert!(
            (re - want).abs() <= 1e-10 * want.abs().max(1.0),
            "digamma re @ x={x}: got {re}, real impl {want}"
        );
        assert!(im.abs() < 1e-11, "digamma im @ x={x}: got {im}");
    }
}

/// The tightest cross-check available: `d/dz ln Gamma(z) = psi(z)`, with the left side
/// produced by AD through `lgamma` (Lanczos) and the right by `digamma` (recurrence
/// plus an asymptotic series). The two share no code path.
#[cfg(feature = "dual")]
#[test]
fn digamma_is_the_derivative_of_lgamma() {
    use thermite_dual::Dual;

    type D = Dual<Vector<f64>, 1>;
    type CD = Complex<D>;

    for &(x, y) in &[(2.5, 1.0), (0.9, -0.4), (4.0, 3.0)] {
        let z: CD = Complex::new(Dual::variable(V::splat(x), 0), Dual::constant(V::splat(y)));
        let w = z.lgamma();

        let got = (w.re.dual[0].extract::<0>(), w.im.dual[0].extract::<0>());
        let want = parts(c(x, y).digamma());

        let scale = (want.0.abs() + want.1.abs()).max(1.0);
        assert!(
            (got.0 - want.0).abs() < 1e-10 * scale && (got.1 - want.1).abs() < 1e-10 * scale,
            "d/dz lgamma vs psi @ ({x}, {y}): got {got:?}, want {want:?}"
        );
    }
}

#[test]
fn beta_is_symmetric_and_matches_gammas() {
    for &(a, b) in &[(1.5, 2.5), (0.75, 3.0), (2.0, -1.5)] {
        let ab = Complex64::new(a, 0.0);
        let bb = Complex64::new(b, 0.0);

        let got = oracle(SpecialMath::beta(c(a, 0.0), c(b, 0.0)));
        let sym = oracle(SpecialMath::beta(c(b, 0.0), c(a, 0.0)));

        let want = oracle(c(a, 0.0).tgamma()) * oracle(c(b, 0.0).tgamma()) / oracle(c(a + b, 0.0).tgamma());

        assert!(
            (got - want).norm() < 1e-10 * want.norm().max(1.0),
            "beta({ab}, {bb}): {got} vs {want}"
        );
        assert!(
            (got - sym).norm() < 1e-12 * want.norm().max(1.0),
            "beta asymmetry: {got} vs {sym}"
        );
    }
}

/// The Lanczos form needs `zgh^(w-1/2) * e^(-zgh)`, whose first factor overflows long
/// before the product does. Folding the two exponentials into one keeps the whole
/// finite range reachable: without it this dies around `Re z ~ 145`, where the true
/// value is still only ~1e249 and f64 reaches 1e308.
#[test]
fn tgamma_reaches_the_top_of_the_range() {
    for x in [140.0, 150.0, 160.0, 170.0, 171.5] {
        let want = libm::tgamma(x);
        assert!(want.is_finite(), "test premise: libm tgamma({x}) should be finite");

        let got = c(x, 0.0).tgamma();
        let (re, im) = parts(got);

        assert!(re.is_finite(), "tgamma({x}) overflowed: got {re}, libm {want}");
        assert!(
            (re - want).abs() <= 1e-11 * want.abs(),
            "tgamma({x}): got {re}, libm {want}"
        );
        assert!(im.abs() <= 1e-11 * want.abs(), "tgamma({x}) im: got {im}");
    }
}

/// The wedge: large `|Im z|` with `|z| > 8`.
///
/// This is the region a continued fraction selected on
/// `Re(z^2)`, which is very negative here, so these points fell through to a Taylor
/// series that truncates long before converging. `erfc(0.1 + 10i)` was wrong by 36
/// orders of magnitude. It is now `exp(-z^2) w(iz)` from [`thermite_complex::faddeeva`].
///
/// Reference values from mpmath at 50 digits.
#[test]
fn erfc_in_the_large_imaginary_wedge() {
    use thermite::math::policy::policies::Precision;
    use thermite_special::SpecialMathWithPolicy;

    #[rustfmt::skip]
    let refs: [((f64, f64), (f64, f64)); 7] = [
        ((1.0, 8.0), (2.6679983658195674e+25, 1.5952414853577614e+26)),
        ((0.1, 10.0), (-1.3784606413850442e+42, 6.140976128501549e+41)),
        ((0.01, 6.0), (-48528755243081.89, -408359969874301.75)),
        ((2.0, 12.0), (1.9185780595161176e+59, 2.232814946690178e+59)),
        ((0.0, 12.0), (1.0, -1.6299357995243493e+61)),
        ((3.0, 9.0), (2.937631018002194e+29, 1.0707473717244858e+30)),
        ((0.5, 20.0), (-1.0361857365910062e+172, -4.946816335504394e+171)),
    ];

    for &((x, y), (wr, wi)) in &refs {
        let got = parts(SpecialMathWithPolicy::erfc_p::<Precision>(c(x, y)));

        // Normwise: `Re erfc(iy)` is exactly 1 against a norm of 1e61, so no method can
        // hold it relatively, and the `(0, 12i)` row is here to check it is not NaN or
        // wildly wrong, not that it is exact.
        let n = (wr * wr + wi * wi).sqrt();
        let d = ((got.0 - wr).powi(2) + (got.1 - wi).powi(2)).sqrt();

        assert!(
            d <= 1e-12 * n,
            "erfc({x} + {y}i): got {got:?}, want ({wr}, {wi}), rel {:e}",
            d / n
        );
    }
}

/// `Re erfc(iy) = 1` exactly, for purely imaginary argument.
///
/// This is the one place the two regimes disagree structurally. Inside the series
/// radius it holds for free: every term of the odd-power series is purely imaginary, so
/// the real part is untouched. Outside it, it holds only because `w`'s near-real-axis
/// correction makes `Re w(-y)` exactly `exp(-y^2)`, which the `exp(y^2)` prefactor then
/// cancels, and that correction is gated at `Best`, so this is a `Precision`-and-above
/// guarantee.
#[test]
fn erfc_on_the_imaginary_axis_has_real_part_one() {
    use thermite::math::policy::policies::Precision;
    use thermite_special::SpecialMathWithPolicy;

    for y in [0.5f64, 3.0, 8.0, 12.0, 25.0] {
        let (re, _) = parts(SpecialMathWithPolicy::erfc_p::<Precision>(c(0.0, y)));

        assert!((re - 1.0).abs() < 1e-12, "Re erfc({y}i) = {re}, want 1");
    }
}

/// `erfc` at every precision tier, including the wedge.
///
/// The `w`-based branch inherits its tier's `N`, so the low policies degrade in a way a
/// continued fraction does not. This pins that they still degrade *gracefully*:
/// `UltraPerformance` is N=8 (3e-4), not garbage.
#[test]
fn erfc_degrades_gracefully_across_policies() {
    use thermite::math::policy::policies::{HighPerformance, Performance, Precision, UltraPerformance};
    use thermite_special::SpecialMathWithPolicy;

    #[rustfmt::skip]
    let refs: [((f64, f64), (f64, f64)); 4] = [
        ((1.0, 8.0), (2.6679983658195674e+25, 1.5952414853577614e+26)),
        ((0.1, 10.0), (-1.3784606413850442e+42, 6.140976128501549e+41)),
        ((2.0, 2.0), (-0.151310866398069, -0.1272916294631408)),
        ((3.0, 0.0), (2.209049699858544e-05, 0.0)),
    ];

    for &((x, y), (wr, wi)) in &refs {
        let n = (wr * wr + wi * wi).sqrt();
        let err = |g: (f64, f64)| ((g.0 - wr).powi(2) + (g.1 - wi).powi(2)).sqrt() / n;

        // Tier bounds, an order looser than the measured N for each.
        for (got, tol, name) in [
            (
                err(parts(SpecialMathWithPolicy::erfc_p::<UltraPerformance>(c(x, y)))),
                5e-3,
                "Worst/N=8",
            ),
            (
                err(parts(SpecialMathWithPolicy::erfc_p::<HighPerformance>(c(x, y)))),
                5e-6,
                "Medium/N=16",
            ),
            (
                err(parts(SpecialMathWithPolicy::erfc_p::<Performance>(c(x, y)))),
                5e-9,
                "Average/N=24",
            ),
            (
                err(parts(SpecialMathWithPolicy::erfc_p::<Precision>(c(x, y)))),
                5e-12,
                "Best/N=32",
            ),
        ] {
            assert!(got < tol, "erfc({x} + {y}i) at {name}: rel {got:e} >= {tol:e}");
        }
    }
}

/// The two regimes of `erf_erfc_positive` are skipped when no lane needs them, and
/// evaluated unconditionally under `avoid_branching`. Both paths must agree exactly:
/// the short-circuit only decides *whether* a regime runs, never what it computes.
#[test]
fn erf_regime_gating_is_transparent() {
    use thermite::math::policy::policies::{AvoidBranching, Precision};
    use thermite_special::SpecialMathWithPolicy;

    type Branchless = AvoidBranching<Precision, true>;
    type Branchy = AvoidBranching<Precision, false>;

    // All-series, all-w, and mixed vectors, so each short-circuit is exercised.
    for (x, y) in [
        (0.5f64, 0.25f64), // series
        (3.0, 0.5),        // series
        (7.0, 0.1),        // w, via the cancellation arm
        (0.1, 10.0),       // w, via the alternation arm
        (9.0, 3.0),        // w, via the radius arm
        (2.0, 2.0),
    ] {
        let a = parts(SpecialMathWithPolicy::erf_p::<Branchless>(c(x, y)));
        let b = parts(SpecialMathWithPolicy::erf_p::<Branchy>(c(x, y)));
        assert_eq!(a, b, "erf({x} + {y}i) differs between branchless and branchy");

        let a = parts(SpecialMathWithPolicy::erfc_p::<Branchless>(c(x, y)));
        let b = parts(SpecialMathWithPolicy::erfc_p::<Branchy>(c(x, y)));
        assert_eq!(a, b, "erfc({x} + {y}i) differs between branchless and branchy");
    }
}

/// A vector straddling both regimes must give each lane the same answer it would get
/// alone. The early `break` keys on `converged | use_w`, so a non-converging
/// large-|z| lane must not change what its neighbours receive.
#[test]
fn erf_lanes_straddling_the_regimes_are_independent() {
    // Lane 0 sits far out in the `w` regime and never converges in the series, while the rest
    // are ordinary series lanes.
    let elems: Vec<num_complex::Complex<f64>> = (0..C::LANES)
        .map(|i| {
            if i == 0 {
                num_complex::Complex::new(20.0, 9.0)
            } else {
                num_complex::Complex::new(0.5 + i as f64 * 0.25, 0.25)
            }
        })
        .collect();

    let mixed = unsafe { C::load_unaligned(elems.as_ptr().cast()) };
    let got = mixed.erf();

    for (i, e) in elems.iter().enumerate() {
        let alone = parts(c(e.re, e.im).erf());
        let lane = got.extractv(i);

        assert_eq!((lane.re, lane.im), alone, "lane {i} at {} + {}i", e.re, e.im);
    }
}
