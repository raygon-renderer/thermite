//! Box-Cox transform: `(x^lambda - 1)/lambda`, `ln x` at `lambda = 0`.
//!
//! The two cases are one function, `ln x` being the limit rather than a separate rule, so the tests
//! are weighted toward the seam. References from mpmath at 60 digits, computed as
//! `expm1(lambda * ln x)/lambda` rather than `(x^lambda - 1)/lambda`, since the latter
//! cancels in the oracle for exactly the same reason it cancels in a naive kernel.
#![cfg(any(
    target_arch = "x86",
    target_arch = "x86_64",
    all(target_arch = "wasm32", feature = "wasm"),
    target_arch = "aarch64"
))]

use thermite::prelude::*;
use thermite_special::RealSpecialMath;

type D = Vector<f64>;
type F = Vector<f32>;

#[track_caller]
fn close(name: &str, got: f64, want: f64, tol: f64) {
    let rel = if want == 0.0 {
        got.abs()
    } else {
        ((got - want) / want).abs()
    };
    assert!(rel <= tol, "{name}: got {got:?}, want {want:?} (rel {rel:e})");
}

fn boxcox(x: f64, lambda: f64) -> f64 {
    D::splat(x).boxcox(D::splat(lambda)).extract::<0>()
}

// (x, lambda, boxcox) from mpmath at 60 digits. The `lambda = 0` rows are `ln x` by
// definition, so the dyadic ones land on multiples of ln 2 - oracle output, not a constant
// written out by hand, and rewriting them as `LN_2` would hide where they came from.
#[allow(clippy::approx_constant)]
#[rustfmt::skip]
const BOXCOX: [(f64, f64, f64); 14] = [
    (0.5,   0.0,    -0.6931471805599453),
    (2.0,   0.0,     0.6931471805599453),
    (1e-05, 0.0,   -11.512925464970229),
    (0.5,   1e-300, -0.6931471805599453),
    (2.0,   1e-12,   0.6931471805601855),
    (10.0,  1e-08,   2.3025851195035365),
    (0.5,   0.5,    -0.585786437626905),
    (2.0,   0.5,     0.8284271247461901),
    (10.0,  2.0,    49.5),
    (0.5,  -1.5,    -1.21895141649746),
    (1e-05,-8.0,    -1.2499999999999991e+39),
    (100.0,-1.0,     0.99),
    (1.0,   3.0,     0.0),
    (2.0,   1.0,     1.0),
];

#[test]
fn matches_the_reference() {
    for &(x, l, want) in BOXCOX.iter() {
        close(&format!("boxcox({x}, {l})"), boxcox(x, l), want, 16.0 * f64::EPSILON);
    }
}

#[test]
fn the_lambda_zero_limit_is_continuous() {
    // ln x is the limit, so approaching zero must approach it smoothly rather than jump.
    // The naive (pow - 1)/lambda is already wrong in the fifth digit at 1e-12, which is the
    // whole reason this is an entry point.
    for &x in &[0.5_f64, 2.0, 10.0, 1e-5] {
        let l0 = boxcox(x, 0.0);
        close("boxcox at zero", l0, x.ln(), 4.0 * f64::EPSILON);

        for &l in &[1e-300_f64, 1e-30, 1e-16, 1e-12, 1e-8] {
            // The true value differs from ln x by ~lambda*ln(x)^2/2, negligible at these.
            close(&format!("boxcox({x}, {l})"), boxcox(x, l), x.ln(), 1e-7);
        }

        // The naive form, evaluated here so the comparison is a measurement rather than a
        // claim about another library.
        let naive = (x.powf(1e-12) - 1.0) / 1e-12;
        let err = ((naive - x.ln()) / x.ln()).abs();
        assert!(
            err > 1e-7,
            "precondition: the naive form is expected to be visibly wrong, err {err:e}"
        );

        // ... and by 1e-300 it has collapsed to a flat zero.
        assert_eq!((x.powf(1e-300) - 1.0) / 1e-300, 0.0, "precondition: naive collapses");
        assert!(boxcox(x, 1e-300) != 0.0, "the kernel does not");
    }
}

#[test]
fn the_family_reproduces_its_named_members() {
    // lambda = 1 is a shift, 1/2 a square root, -1 a reciprocal - the transforms the family
    // interpolates. A sign or reciprocal slip shows up here immediately.
    for &x in &[0.25_f64, 1.0, 3.0, 100.0] {
        close("lambda=1", boxcox(x, 1.0), x - 1.0, 8.0 * f64::EPSILON);
        close("lambda=1/2", boxcox(x, 0.5), 2.0 * (x.sqrt() - 1.0), 8.0 * f64::EPSILON);
        close("lambda=-1", boxcox(x, -1.0), 1.0 - 1.0 / x, 8.0 * f64::EPSILON);
        close("lambda=2", boxcox(x, 2.0), (x * x - 1.0) / 2.0, 8.0 * f64::EPSILON);

        // x = 1 is a fixed point of the whole family: 1^lambda - 1 = 0.
        assert_eq!(boxcox(1.0, x), 0.0, "boxcox(1, {x}) must be exactly 0");
    }
    assert_eq!(boxcox(1.0, 0.0), 0.0, "and at lambda = 0, since ln 1 = 0");
}

#[test]
fn the_domain_edges_are_the_conventional_limits() {
    // x < 0 is out of domain.
    assert!(boxcox(-1.0, 0.5).is_nan());
    assert!(boxcox(-1.0, 0.0).is_nan());

    // x = 0: -1/lambda above zero, -inf at or below. `powf_m1(0, lambda)` delivers both on
    // its own, provided its Best-tier residual does not form inf - inf there.
    close("x=0, lambda=2", boxcox(0.0, 2.0), -0.5, 0.0);
    close("x=0, lambda=0.25", boxcox(0.0, 0.25), -4.0, 0.0);
    assert_eq!(boxcox(0.0, 0.0), f64::NEG_INFINITY);
    assert_eq!(boxcox(0.0, -1.5), f64::NEG_INFINITY);

    assert!(boxcox(f64::NAN, 1.0).is_nan());
    assert!(boxcox(2.0, f64::NAN).is_nan());
}

#[test]
fn f32_tracks_the_reference() {
    for &(x, l, want) in BOXCOX.iter() {
        // Skip rows f32 cannot hold at all.
        if want.abs() > 1e30 || (x != 0.0 && (x as f32) == 0.0) {
            continue;
        }
        let got = F::splat(x as f32).boxcox(F::splat(l as f32)).extract::<0>() as f64;
        close(&format!("f32 boxcox({x}, {l})"), got, want, 32.0 * f32::EPSILON as f64);
    }
}

#[test]
fn lanes_stay_independent_across_the_seam() {
    use thermite::backend::scalar::Scalar;
    type D4 = thermite::simd::f64x4<Scalar>;

    // A mixed vector is the case the blend exists for: one lane exactly at the seam, the
    // others away from it on both sides.
    let xs = [2.0, 0.5, 10.0, 0.0];
    let ls = [0.0, 1e-12, -1.5, 2.0];

    let got = D4::new(xs).boxcox(D4::new(ls));
    for lane in 0..4 {
        assert_eq!(got.as_slice()[lane], boxcox(xs[lane], ls[lane]), "lane {lane}");
    }
}
