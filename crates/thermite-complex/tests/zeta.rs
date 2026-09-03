//! Complex Riemann zeta, against mpmath and against itself.
//!
//! The complex kernel is the same Euler-Maclaurin expansion the real one runs, so the tests
//! that matter are the ones a shared body could still fail: agreement with the real kernel on
//! the real axis, conjugate symmetry, the functional equation in the left half-plane, and the
//! `|Im z|` reach, which is the one property genuinely specific to the complex case, since the
//! expansion's correction terms grow like `(|z|/N)^{2k}` and `N` is fixed at 10.
#![cfg(feature = "special")]

use thermite::math::policy::DefaultPolicy;
use thermite::prelude::*;
use thermite_complex::Complex;
use thermite_special::SpecialMathWithPolicy;

type V = Vector<f64>;
type C = Complex<V>;

fn z(re: f64, im: f64) -> C {
    Complex::new(V::splat(re), V::splat(im))
}

fn zeta(re: f64, im: f64) -> (f64, f64) {
    let r = z(re, im).zeta_p::<DefaultPolicy>();
    (r.re.extract::<0>(), r.im.extract::<0>())
}

fn rel(got: (f64, f64), want: (f64, f64)) -> f64 {
    let d = ((got.0 - want.0).powi(2) + (got.1 - want.1).powi(2)).sqrt();
    let m = (want.0 * want.0 + want.1 * want.1).sqrt();
    if m == 0.0 { d } else { d / m }
}

// (re, im, zeta_re, zeta_im) from mpmath at 30 digits. All entries keep |Im z| <= 4, which is
// where N = 10 holds full accuracy. The reach itself is measured separately below.
const CZETA: &[(f64, f64, f64, f64)] = &[
    (2.0, 0.0, 1.6449340668482264365, 0.0),
    (2.0, 1.0, 1.1503557032549026717, -0.43753086591960788112),
    (0.5, 1.0, 0.14393642707718906032, -0.72209974353167308913),
    (0.5, 4.0, 0.60678376452243726922, 0.091112139972515029789),
    (1.5, -2.0, 0.7521818690342325726, 0.33397906099331399421),
    (3.0, 0.5, 1.1739287246387467673, -0.091730267113479445801),
    (-1.0, 2.0, 0.16891566977083441814, -0.070515988908254423002),
    (-2.5, 1.5, 0.038100166636025193093, 0.014861416908449377201),
    (0.25, 3.0, 0.48529811855785336912, -0.058985755815927158274),
    (4.0, -1.0, 1.0535076344416207331, 0.058042295306486234156),
];

#[test]
fn matches_mpmath() {
    for &(re, im, wr, wi) in CZETA {
        let got = zeta(re, im);
        assert!(
            rel(got, (wr, wi)) < 1e-13,
            "zeta({re} + {im}i): got {got:?}, want ({wr}, {wi})"
        );
    }
}

// On the real axis the complex kernel must reproduce the real one. Different arithmetic, same
// expansion, so a disagreement means one of the two transcribed it wrongly.
#[test]
fn agrees_with_the_real_kernel_on_the_real_axis() {
    for &s in &[0.3f64, 1.7, 2.0, 5.0, 12.0, -0.5, -3.5] {
        let (cr, ci) = zeta(s, 0.0);
        let real = V::splat(s).zeta_p::<DefaultPolicy>().extract::<0>();
        assert!(
            (cr - real).abs() <= 1e-13 * real.abs().max(1.0),
            "zeta({s}): complex {cr} vs real {real}"
        );
        assert!(ci.abs() < 1e-15, "zeta({s}) should be real, got imaginary part {ci}");
    }
}

// zeta(conj z) = conj(zeta(z)), which the kernel gets from its arithmetic rather than from any
// special handling, so this fails if a branch treats the half-planes asymmetrically.
#[test]
fn conjugate_symmetry() {
    for &(re, im) in &[(2.0f64, 1.5f64), (0.5, 3.0), (-1.5, 2.0), (4.0, 0.25)] {
        let a = zeta(re, im);
        let b = zeta(re, -im);
        assert!((a.0 - b.0).abs() < 1e-15 * a.0.abs().max(1.0), "real parts differ");
        assert!(
            (a.1 + b.1).abs() < 1e-15 * a.1.abs().max(1.0),
            "imaginary parts not negated"
        );
    }
}

// The left half-plane goes through the functional equation, so checking it against the
// relation itself exercises that arm without depending on the reference table.
#[test]
fn satisfies_the_functional_equation() {
    use core::f64::consts::PI;
    for &(re, im) in &[(-1.0f64, 1.0f64), (-2.5, 0.5), (-0.5, 2.0)] {
        let got = zeta(re, im);

        // zeta(s) = 2^s pi^(s-1) sin(pi s/2) Gamma(1-s) zeta(1-s), assembled in num-complex-free
        // arithmetic from the crate's own pieces at 1 - s, which is in the right half-plane.
        let (ar, ai) = zeta(1.0 - re, -im);
        let g = z(1.0 - re, -im).tgamma_p::<DefaultPolicy>();
        let (gr, gi) = (g.re.extract::<0>(), g.im.extract::<0>());

        // 2^s pi^(s-1) = exp((s ln2) + (s-1) ln pi), and sin(pi s / 2).
        let lr = re * 2f64.ln() + (re - 1.0) * PI.ln();
        let li = im * 2f64.ln() + im * PI.ln();
        let (er, ei) = (lr.exp() * li.cos(), lr.exp() * li.sin());
        let (sr, si) = (
            (PI * re / 2.0).sin() * (PI * im / 2.0).cosh(),
            (PI * re / 2.0).cos() * (PI * im / 2.0).sinh(),
        );

        let mul = |a: (f64, f64), b: (f64, f64)| (a.0 * b.0 - a.1 * b.1, a.0 * b.1 + a.1 * b.0);
        let want = mul(mul(mul((er, ei), (sr, si)), (gr, gi)), (ar, ai));

        assert!(
            rel(got, want) < 1e-11,
            "zeta({re} + {im}i): got {got:?}, functional equation gives {want:?}"
        );
    }
}

// The property specific to the complex case. `N = 10` is fixed, and the correction terms grow
// like `(|z|/N)^{2k}`, so accuracy is a function of |Im z| and degrades predictably past it.
// This pins the shape of that curve so a future change to N shows up here rather than silently.
#[test]
fn accuracy_degrades_with_the_imaginary_part_as_expected() {
    // (t, zeta(1/2 + it) from mpmath, tolerance). The tolerances are the _shape of the curve_,
    // not slack: N = 10 holds full accuracy to about t = 4 and then loses roughly an order of
    // magnitude per few units, which is the (|z|/N)^2k growth of the correction terms. Raising
    // N would tighten every row here, so a change to it shows up as these becoming loose rather
    // than as a silent accuracy shift.
    const LINE: &[(f64, f64, f64, f64)] = &[
        (1.0, 0.14393642707718906032, -0.72209974353167308913, 1e-14),
        (4.0, 0.60678376452243726922, 0.091112139972515029789, 1e-14),
        (8.0, 1.2416151055868185783, 0.36004758838723228549, 1e-13),
        (10.0, 1.5448952202967527669, -0.11533646527127337544, 1e-12),
        (14.0, 0.022241142609993589246, -0.1032581232664500579, 1e-8),
        (20.0, 0.42991386043784337216, -1.0642914430805891127, 1e-7),
    ];

    let mut worst_ok = 0.0f64;
    for &(t, wr, wi, tol) in LINE {
        let got = zeta(0.5, t);
        let e = rel(got, (wr, wi));
        assert!(e < tol, "zeta(1/2 + {t}i): relative error {e:.2e} exceeds {tol:.0e}");
        if e < tol {
            worst_ok = worst_ok.max(t);
        }
    }
    assert!(worst_ok >= 20.0, "the reach test stopped covering its range");

    // And the degradation is monotone in t, which is what identifies it as the expansion's
    // reach rather than noise at individual points.
    let e4 = rel(zeta(0.5, 4.0), (LINE[1].1, LINE[1].2));
    let e20 = rel(zeta(0.5, 20.0), (LINE[5].1, LINE[5].2));
    assert!(e20 > e4, "error at t = 20 ({e20:.2e}) should exceed t = 4 ({e4:.2e})");
}
