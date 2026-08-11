//! The complex exponential integral `E_N(z)` (`special` feature).
//!
//! `E_N` is holomorphic on the cut plane `|Arg z| < pi`, and the implementation is the
//! shared `thermite-special` series/continued-fraction body with three hooks overridden
//! (`expint_use_series`, `expint_invalid`, `expint_cf_tiny`). So what is checked here is
//! mostly that those hooks route correctly - the regime split, the absence of a
//! negative-argument hole, and the branch cut carried by the principal `ln`.
//!
//! Reference values are mpmath 1.3.0 at 30 digits; `expint_reference.py` regenerates the
//! tables below. mpmath shares no code with the implementation.
//!
//! Tolerances widen with `N` on purpose. `E_1` is at machine precision over the whole
//! set, and each order above it goes through one step of
//! `E_{n+1} = (e^-z - z E_n) / n`, which amplifies error by `|z|/n` - so the loss grows
//! like `|z|^(N-1)/(N-1)!`. That is the same contract the real path holds itself to
//! (see `RECURRENCE_THRESHOLD` in `thermite-special`), and these points sit well inside
//! it; the asymptotic series that path swaps in beyond the threshold has no complex
//! counterpart yet, so very large `|z|` at high `N` is still out of scope.

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

fn rel_err(got: C, want: Complex64) -> f64 {
    let (re, im) = parts(got);
    ((re - want.re).powi(2) + (im - want.im).powi(2)).sqrt() / want.norm().max(1.0)
}

const EXPINT_E1: &[(f64, f64, f64, f64)] = &[
    (0.5, 0.25, 4.59449988177999813e-01, -2.67511350834374773e-01),
    (0.3, -0.6, 1.73775394181994863e-01, 5.98283391653245733e-01),
    (0.9, 0.1, 2.55452810224039872e-01, -4.47492499302867414e-02),
    (2.0, 1.0, 9.38816131048446703e-03, -4.44629941413853882e-02),
    (3.5, -2.0, -4.79577613262205122e-03, 4.04704341703869194e-03),
    (8.0, 0.0, 3.76656228439248996e-05, 0.00000000000000000e+00),
    (0.5, 100.0, 3.14866779430731346e-03, -5.18249190899322792e-03),
    (-2.0, 0.1, -4.94500188562308107e+00, -2.77244757217011140e+00),
    (-2.0, -0.1, -4.94500188562308107e+00, 2.77244757217011140e+00),
    (-0.5, 0.0, -4.54219904863173596e-01, -3.14159265358979312e+00),
    (-1.0, 1e-08, -1.89511781635593679e+00, -3.14159262640697490e+00),
    (-20.0, 5.0, -7.11836971654026303e+05, -2.47452509396959022e+07),
    (-50.0, 0.5, -9.34062766231616225e+19, 4.97962700042037821e+19),
    (-100.0, 1.0, -1.49015355158554921e+41, 2.27000359111611740e+41),
    (-5.0, 0.01, -4.01840880578520512e+01, -2.84476969940322544e+00),
    (-8.0, -1e-06, -4.40379899534675246e+02, 3.14122003384141291e+00),
    (-0.999, 0.0, -1.89239953407420392e+00, -3.14159265358979312e+00),
    (-40.0, 0.0, -6.03971826361124200e+15, -3.14159265358979312e+00),
];

const EXPINT_E2: &[(f64, f64, f64, f64)] = &[
    (0.5, 0.25, 2.91072258546329066e-01, -1.31164908249475853e-01),
    (0.3, -0.6, 2.00321008456093519e-01, 3.43077651475057477e-01),
    (0.9, 0.1, 1.70156050722901214e-01, -2.58601943219883036e-02),
    (2.0, 1.0, 9.88264883570531306e-03, -3.43428870920817822e-02),
    (3.5, -2.0, -3.87541595309916827e-03, 3.70219881790318428e-03),
    (8.0, 0.0, 3.41376451511126222e-05, 0.00000000000000000e+00),
    (0.5, 100.0, 3.19930969482656001e-03, -5.14924711812216832e-03),
    (-2.0, 0.1, -2.81510693255417621e+00, -5.78806967192924482e+00),
    (-2.0, -0.1, -2.81510693255417621e+00, 5.78806967192924482e+00),
    (-0.5, 0.0, 1.42161131826854126e+00, -1.57079632679489656e+00),
    (-1.0, 1e-08, 8.23163980687182106e-01, -3.14159263463861516e+00),
    (-20.0, 5.0, -3.39974490925687947e+05, -2.61091508355149068e+07),
    (-50.0, 0.5, -9.54085357313567949e+19, 5.08363979747237888e+19),
    (-100.0, 1.0, -1.50576255074569040e+41, 2.29325480289614862e+41),
    (-5.0, 0.01, -5.25431494797942165e+01, -1.53061144720605338e+01),
    (-8.0, -1e-06, -5.42081212378384180e+02, 2.51323008488188115e+01),
    (-0.999, 0.0, 8.25057770778436894e-01, -3.13845106093620352e+00),
    (-40.0, 0.0, -6.20346370742967800e+15, -1.25663706143591725e+02),
];

const EXPINT_E3: &[(f64, f64, f64, f64)] = &[
    (0.5, 0.25, 2.04673867004194598e-01, -7.86218485670037992e-02),
    (0.3, -0.6, 1.72740384140388420e-01, 2.17783371046486568e-01),
    (0.9, 0.1, 1.24406019917377916e-01, -1.71653342096715171e-02),
    (2.0, 1.0, 9.50689041728361092e-03, -2.75387943579549176e-02),
    (3.5, -2.0, -3.20349369157912163e-03, 3.37493763696161426e-03),
    (8.0, 0.0, 3.11807333468054241e-05, 0.00000000000000000e+00),
    (0.5, 100.0, 3.24923391583645466e-03, -5.11502978274117146e-03),
    (-2.0, 0.1, 5.71560381803860151e-01, -6.01615168337720085e+00),
    (-2.0, -0.1, 5.71560381803860151e-01, 6.01615168337720085e+00),
    (-0.5, 0.0, 1.17976346491719952e+00, -3.92699081698724139e-01),
    (-1.0, 1e-08, 1.77072288886515050e+00, -1.57079633502653659e+00),
    (-20.0, 5.0, 1.38887822273020778e+05, -2.76232305777683742e+07),
    (-50.0, 0.5, -9.75007135779277373e+19, 5.19219630274108457e+19),
    (-100.0, 1.0, -1.52170562748989029e+41, 2.31699249002970160e+41),
    (-5.0, 0.01, -5.72315350186157943e+01, -3.87446238605638271e+01),
    (-8.0, -1e-06, -6.77845868559568430e+02, 1.00530422833662570e+02),
    (-0.999, 0.0, 1.76989880916311249e+00, -1.56765630493763353e+00),
    (-40.0, 0.0, -6.37664073008356200e+15, -2.51327412287183461e+03),
];

#[track_caller]
fn check(n: usize, got: C, a: f64, b: f64, wr: f64, wi: f64, tol: f64) {
    let err = rel_err(got, Complex64::new(wr, wi));
    let (re, im) = parts(got);

    assert!(err <= tol, "E_{n}({a} + {b}i): got {re} + {im}i, want {wr} + {wi}i (rel {err:e})");
}

#[test]
fn expint_e1_matches_mpmath() {
    for &(a, b, wr, wi) in EXPINT_E1 {
        check(1, c(a, b).expint::<1>(), a, b, wr, wi, 1e-14);
    }
}

#[test]
fn expint_e2_matches_mpmath() {
    for &(a, b, wr, wi) in EXPINT_E2 {
        check(2, c(a, b).expint::<2>(), a, b, wr, wi, 1e-12);
    }
}

#[test]
fn expint_e3_matches_mpmath() {
    for &(a, b, wr, wi) in EXPINT_E3 {
        check(3, c(a, b).expint::<3>(), a, b, wr, wi, 1e-10);
    }
}

#[test]
fn conjugate_symmetry_across_the_cut() {
    // E_N is real on the positive real axis, so by the Schwarz reflection principle
    // E_N(conj z) = conj(E_N(z)) everywhere off the cut. Straddling the cut is the
    // sharpest form of that: the two sides must be exact mirror images, not merely
    // close, since the only asymmetric ingredient is arg(z) inside the principal ln.
    for &(a, b) in &[(-2.0, 0.1), (-5.0, 0.01), (-20.0, 5.0), (-0.5, 0.25)] {
        let (up_re, up_im) = parts(c(a, b).expint::<1>());
        let (dn_re, dn_im) = parts(c(a, -b).expint::<1>());

        assert_eq!(up_re, dn_re, "Re E_1 differs across the cut at {a} +/- {b}i");
        assert_eq!(up_im, -dn_im, "Im E_1 is not mirrored at {a} +/- {b}i");
    }
}

#[test]
fn on_the_cut_takes_arg_pi() {
    // On (-inf, 0] the value is fixed by which side `ln` is continuous from. This crate
    // documents Arg = +pi there, and -ln z puts that straight into Im E_1 as -pi.
    for &x in &[-0.5, -0.999, -2.5] {
        let (_, im) = parts(c(x, 0.0).expint::<1>());

        assert!(
            (im + std::f64::consts::PI).abs() < 1e-15,
            "E_1({x}) should sit on Im = -pi, got {im}"
        );
    }
}

#[test]
fn negative_reals_are_in_domain() {
    // Real `expint` NaNs out x < 0; the complex principal branch must not. Regression
    // for the `expint_invalid` hook.
    for &(a, b) in &[(-2.0, 0.1), (-0.5, 0.0), (-40.0, 0.0)] {
        let (re, im) = parts(c(a, b).expint::<1>());

        assert!(re.is_finite() && im.is_finite(), "E_1({a} + {b}i) came back non-finite");
    }
}

#[test]
fn far_off_axis_small_real_part_uses_the_right_regime() {
    // 0.5 + 100i has re < 1 but |z| >> 1. A lexicographic `cmp_lt` reads that as "in the
    // unit disc" and sends it to the power series, which diverges there. Regression for
    // the `expint_use_series` hook.
    let (re, im) = parts(c(0.5, 100.0).expint::<1>());

    assert!((re - 3.14866779430731346e-03).abs() < 1e-15, "got re {re}");
    assert!((im + 5.18249190899322792e-03).abs() < 1e-15, "got im {im}");
}
