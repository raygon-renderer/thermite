//! `lgamma` accuracy for large arguments, across every precision policy.
//!
//! # What this guards
//!
//! The f32 path below `Best` used to evaluate `PadeApproximate[Ln[Gamma[x+1]],
//! {x, 5.000000001, 7, 9}]` over the whole domain. That cannot work and no choice of
//! coefficients would fix it: `lgamma(x) ~ x ln(x)` is not rational, so a [7/9] ratio
//! decays away from its expansion point and eventually changes sign. It returned -17690
//! at x = 300, where the answer is 1409, and `DefaultPolicy` is `Average`, so that was
//! what a plain `x.lgamma()` did on an f32 vector. It reached `tgamma` and `beta` too,
//! both of which route through `lgamma_r` at the lower tiers.
//!
//! The tiers have since been rearranged. `Average` joined `Best` on the full Lanczos
//! path, which also restored the ordering, since `Medium` used to measure _looser_ than
//! `Average` at large arguments. What remains below `Average` is a two-arm split of
//! `lgamma(x+1)`: a degree-12 minimax polynomial over `[0, 4]` and Stirling's series above
//! it, joined near where their error curves cross (1.22e-6 against 7.4e-7, or 5 and 3 f32
//! ulp, against this tier's 10000-ulp budget).
//!
//! Neither arm can take the other's range. No polynomial follows `lgamma(x) ~ x ln(x)`
//! out to infinity, and Stirling's series is asymptotic rather than convergent, so below
//! x ~ 1 it diverges (wrong sign at x = 0.1) and extra terms make it worse rather than
//! better. That is what `stirling_cannot_replace_the_polynomial_near_zero` records.
#![cfg(any(target_arch = "x86", target_arch = "x86_64"))]

use thermite::Vector;
use thermite::math::policy::DefaultPolicy;
use thermite::math::policy::policies::{AveragePrecision, BestPrecision, MediumPrecision};
use thermite::prelude::*;
use thermite_special::SpecialMathWithPolicy;

/// Spans the Pade region, the crossover, and far enough out that a rational form has no
/// chance of following.
const PROBES: &[f64] = &[0.5, 1.0, 2.0, 4.0, 5.9, 6.0, 6.1, 8.0, 10.0, 30.0, 100.0, 200.0, 300.0, 1000.0, 1e5];

#[track_caller]
fn check(tier: &str, x: f64, got: f64, tol: f64) {
    let want = libm::lgamma(x);
    let rel = if want == 0.0 { got.abs() } else { ((got - want) / want).abs() };
    assert!(rel <= tol, "{tier} lgamma({x}): got {got}, want {want} (rel {rel:e}, tol {tol:e})");
}

#[test]
fn f32_average_and_best_hold_across_the_whole_range() {
    // f32 tops out before lgamma overflows, so 1e5 is comfortably in range (~1.05e6).
    for &x in PROBES {
        let v = Vector::<f32>::splat(x as f32);
        // Both tiers stay within a few f32 ulp of the true value at every scale. Before
        // the Stirling branch, `Average` was off by 26% at x = 200.
        check(
            "f32 average",
            x,
            v.lgamma_p::<AveragePrecision<DefaultPolicy>>().extract::<0>() as f64,
            2e-6,
        );
        check("f32 best", x, v.lgamma_p::<BestPrecision<DefaultPolicy>>().extract::<0>() as f64, 2e-6);
    }
}

#[test]
fn f32_medium_is_loose_but_bounded() {
    // Medium buys a cheaper `ln`, whose error Stirling amplifies by (x + 1/2). That is a
    // fair trade at this tier, but it is worth pinning so it cannot drift further.
    for &x in PROBES {
        let got = Vector::<f32>::splat(x as f32)
            .lgamma_p::<MediumPrecision<DefaultPolicy>>()
            .extract::<0>();
        check("f32 medium", x, got as f64, 1e-4);
    }
}

/// The reflection is the worst-conditioned corner: `lgamma` has a zero near -2.5, so the
/// relative error there is amplified by however close the value sits to zero. Pinned at
/// both tiers because it is the binding constraint on the cheap arm's degree. A degree-10
/// polynomial put `Medium` at 1.0e-3 here, over its budget, where degree 12 gives 1.6e-4.
#[test]
fn f32_medium_survives_the_reflection_zero() {
    for &z in &[-2.5_f64, -2.4, -2.6, -3.5] {
        let got = Vector::<f32>::splat(z as f32)
            .lgamma_p::<MediumPrecision<DefaultPolicy>>()
            .extract::<0>();
        check("medium reflection", z, got as f64, 5e-4);
    }
}

#[test]
fn f32_handles_large_negative_arguments() {
    // The reflection path reduces to lgamma(1 - z), so a large negative argument lands
    // just as far out as a large positive one and was equally broken before.
    for &z in &[-2.5_f64, -7.5, -20.5, -100.5] {
        let got = Vector::<f32>::splat(z as f32)
            .lgamma_p::<AveragePrecision<DefaultPolicy>>()
            .extract::<0>();
        let want = libm::lgamma(z);
        let rel = ((got as f64 - want) / want).abs();
        assert!(rel <= 1e-3, "f32 average lgamma({z}): got {got}, want {want} (rel {rel:e})");
    }
}

#[test]
fn f64_is_accurate_at_every_tier() {
    for &x in PROBES.iter().chain(&[1e10]) {
        let v = Vector::<f64>::splat(x);
        for (name, got) in [
            ("average", v.lgamma_p::<AveragePrecision<DefaultPolicy>>().extract::<0>()),
            ("best", v.lgamma_p::<BestPrecision<DefaultPolicy>>().extract::<0>()),
        ] {
            check(&format!("f64 {name}"), x, got, 1e-13);
        }
    }
}

#[test]
fn crossover_is_continuous() {
    // Either side of x = 4 the two arms must agree, or the switch shows as a step.
    // Checked on the tier that actually has the split.
    for &x in &[3.9_f64, 3.99, 4.0, 4.01, 4.1] {
        let got = Vector::<f32>::splat(x as f32)
            .lgamma_p::<MediumPrecision<DefaultPolicy>>()
            .extract::<0>();
        check("crossover", x, got as f64, 1e-4);
    }
}

/// Records why the polynomial is still there: Stirling alone cannot cover the small end.
///
/// Its series is asymptotic rather than convergent, so below about x = 1 the terms grow
/// and the approximation walks away from the answer. At x = 0.1 it produces the wrong
/// sign. Adding terms makes it worse there, not better, which is why no amount of tuning
/// removes the need for a second method near zero.
#[test]
fn stirling_cannot_replace_the_polynomial_near_zero() {
    fn stirling(x: f64) -> f64 {
        // Same two Bernoulli terms the kernel uses, in exact f64 arithmetic so the
        // failure below is the method's and not the evaluation's.
        let w = 1.0 / x;
        (x - 0.5) * x.ln() - x + 0.5 * (2.0 * core::f64::consts::PI).ln() + w * (1.0 / 12.0 - w * w / 360.0)
    }

    // The tier this would have to serve is `Worst`, whose budget is 100000 ulp, about
    // 1e-2 relative at lgamma(0.5). Stirling alone does not reach even that.
    let at_half = ((stirling(0.5) - libm::lgamma(0.5)) / libm::lgamma(0.5)).abs();
    assert!(at_half > 1e-2, "stirling(0.5) rel {at_half:e}, if this is now small, revisit the split");

    // And it is not merely inaccurate at a tenth, it is the wrong sign.
    assert!(stirling(0.1) < 0.0 && libm::lgamma(0.1) > 0.0);

    // Meanwhile the shipped path is fine there, because the polynomial covers it.
    for &x in &[0.05_f64, 0.1, 0.25, 0.5, 1.5, 2.5] {
        let got = Vector::<f32>::splat(x as f32)
            .lgamma_p::<AveragePrecision<DefaultPolicy>>()
            .extract::<0>();
        check("near zero", x, got as f64, 5e-6);
    }
}
