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
