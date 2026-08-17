//! `erfcx(x) = e^{x^2} erfc(x)`, the scaled complementary error function.
//!
//! References are mpmath at 40 digits. The point of the function is the range where
//! `erfc` has already underflowed to zero, so most of these probes are out past that
//! point, where any test written against `erfc` would be comparing zero to zero.
#![cfg(any(
    target_arch = "x86",
    target_arch = "x86_64",
    all(target_arch = "wasm32", feature = "wasm"),
    target_arch = "aarch64"
))]
#![allow(clippy::excessive_precision)]

use thermite::Vector;
use thermite::math::policy::DefaultPolicy;
use thermite::math::policy::policies::{AveragePrecision, BestPrecision, MediumPrecision, WorstPrecision};
use thermite::prelude::*;
use thermite_special::{SpecialMath, SpecialMathWithPolicy};

type D = Vector<f64>;
type F = Vector<f32>;

#[track_caller]
fn close(name: &str, got: f64, want: f64, tol: f64) {
    let rel = if want == 0.0 { got.abs() } else { ((got - want) / want).abs() };
    assert!(rel <= tol, "{name}: got {got:?}, want {want:?} (rel {rel:e}, tol {tol:e})");
}

/// `(x, erfcx(x))` from mpmath at 40 digits.
const REFS: &[(f64, f64)] = &[
    (0.0, 1.0),
    (0.5, 0.61569034419292587487),
    (1.0, 0.42758357615580700441),
    (2.0, 0.25539567631050574387),
    (3.0, 0.17900115118138995042),
    (5.0, 0.11070463773306862637),
    (10.0, 0.056140992743822585858),
    (26.5, 0.021275046685371105955),
    (27.0, 0.020881607990420940674),
    (50.0, 0.0112815362653237725),
    (100.0, 0.0056416137829894329036),
    (1000.0, 0.0005641893014533876542),
    (1e6, 5.6418958354747419216e-7),
    (1e15, 5.6418958354775628695e-16),
];

/// The negative side, where `erfcx` grows like `e^{x^2}`.
const NEG_REFS: &[(f64, f64)] = &[
    (-0.5, 1.9523604891825570933),
    (-1.0, 5.0089800807622834663),
    (-2.0, 108.94090438997797241),
    (-3.0, 16205.988853999586625),
    (-5.0, 144009798674.66104041),
];

#[test]
fn f64_matches_mpmath() {
    // Best takes N = 32, and the measured floor at N = 40 is ~1.2 ulp.
    for &(x, want) in REFS.iter().chain(NEG_REFS) {
        let v = D::splat(x);
        close("f64 best", v.erfcx_p::<BestPrecision<DefaultPolicy>>().extract::<0>(), want, 1e-12);
        close("f64 default", v.erfcx().extract::<0>(), want, 1e-9);
    }
}

#[test]
fn f64_lower_tiers_stay_within_their_ladder_rung() {
    // The N ladder is 8/16/24/32/40 for Worst/Medium/Average/Best/Reference, with
    // measured normwise errors 3.1e-4, 4.3e-7, 4.2e-10, 3.1e-13, 8.7e-16.
    for &(x, want) in REFS.iter().chain(NEG_REFS) {
        let v = D::splat(x);
        close("f64 worst", v.erfcx_p::<WorstPrecision<DefaultPolicy>>().extract::<0>(), want, 1e-3);
        close("f64 medium", v.erfcx_p::<MediumPrecision<DefaultPolicy>>().extract::<0>(), want, 2e-6);
        close("f64 average", v.erfcx_p::<AveragePrecision<DefaultPolicy>>().extract::<0>(), want, 1e-8);
    }
}

#[test]
fn f32_matches_mpmath() {
    // f32 clamps the ladder at N = 16 (4.3e-7), about 3.6 f32 ulp.
    for &(x, want) in REFS.iter().chain(NEG_REFS) {
        // Below about -9.3 the reflection's e^{x^2} overflows f32. The positive side
        // has no such limit (there is no exp on it), so 1e15 stays in.
        if x < -9.0 {
            continue;
        }
        let got = F::splat(x as f32).erfcx().extract::<0>();
        close("f32", got as f64, want, 1e-6);
    }
}

/// The whole reason the function exists.
#[test]
fn reaches_where_erfc_has_already_underflowed() {
    // f64: erfc underflows to exactly zero here, but erfcx is an ordinary number.
    let x = 30.0_f64;
    assert_eq!(D::splat(x).erfc().extract::<0>(), 0.0, "precondition: erfc is expected to underflow");

    let cx = D::splat(x).erfcx().extract::<0>();
    close("erfcx at 30", cx, 0.018795888861416751497, 1e-9);

    // And it is still the tail: erfc(x) = e^{-x^2} erfcx(x), recovered in the log domain.
    let ln_erfc = -x * x + cx.ln();
    close("ln erfc(30)", ln_erfc, -903.97411711064387808, 1e-12);

    // f32 underflows far earlier, at x ~ 9.3.
    let xf = 12.0_f32;
    assert_eq!(F::splat(xf).erfc().extract::<0>(), 0.0, "precondition: f32 erfc underflows");
    close("f32 erfcx at 12", F::splat(xf).erfcx().extract::<0>() as f64, 0.04685422101489376262, 1e-6);
}

#[test]
fn matches_the_defining_identity_where_both_are_representable() {
    // erfcx(x) == exp(x^2) * erfc(x) wherever the right side does not overflow or
    // underflow, which is the range the naive form would have covered.
    for &x in &[-3.0_f64, -1.0, -0.25, 0.0, 0.25, 1.0, 3.0, 5.0, 10.0, 20.0] {
        let v = D::splat(x);
        let naive = (x * x).exp() * v.erfc_p::<BestPrecision<DefaultPolicy>>().extract::<0>();
        close("identity", v.erfcx_p::<BestPrecision<DefaultPolicy>>().extract::<0>(), naive, 1e-12);
    }
}

#[test]
fn asymptotic_tail_and_special_values() {
    // erfcx(x) -> 1/(x sqrt(pi)) as x -> +inf.
    const FRAC_1_SQRT_PI: f64 = 0.5641895835477562869;
    for &x in &[1e8_f64, 1e12, 1e15] {
        close("asymptote", D::splat(x).erfcx().extract::<0>(), FRAC_1_SQRT_PI / x, 1e-9);
    }

    // erfcx(0) = 1, to the accuracy of the tier: the default is N = 24 (4.2e-10), so
    // this is not bit-exact there, only at the top of the ladder.
    close("erfcx(0)", D::splat(0.0).erfcx().extract::<0>(), 1.0, 1e-9);
    close("erfcx(0) best", D::splat(0.0).erfcx_p::<BestPrecision<DefaultPolicy>>().extract::<0>(), 1.0, 1e-12);

    assert_eq!(D::splat(f64::INFINITY).erfcx().extract::<0>(), 0.0);
    assert_eq!(D::splat(f64::NEG_INFINITY).erfcx().extract::<0>(), f64::INFINITY);
    assert!(D::splat(f64::NAN).erfcx().extract::<0>().is_nan());

    // Overflows to the left, which is the true behaviour: erfcx ~ 2 e^{x^2} there.
    assert_eq!(D::splat(-30.0).erfcx().extract::<0>(), f64::INFINITY);
}

/// A wide backend, with lanes on both sides of the sign branch in one call.
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[test]
fn wide_backend_agrees_with_scalar() {
    use thermite::simd::Simd;
    type W = Vector<<thermite::backend::x86_v3::X86V3 as Simd>::f64x4>;

    let xs = [-2.0, 0.5, 27.0, 1e6];
    let got = W::from_slice(&xs).erfcx().into_array();

    for (i, &x) in xs.iter().enumerate() {
        close("wide", got.as_slice()[i], D::splat(x).erfcx().extract::<0>(), 1e-15);
    }
}
