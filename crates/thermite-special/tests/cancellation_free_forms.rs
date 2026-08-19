//! `lbeta`, `logit`, `logit_1m` and `planck`: cancellation-free or range-safe rewrites of
//! expressions whose direct spelling loses the answer.
//!
//! Reference values come from mpmath at 30 digits. As in the core suite, each function
//! that exists because the obvious form fails also carries a test showing that it does.
#![allow(clippy::excessive_precision)]
#![cfg(any(
    target_arch = "x86",
    target_arch = "x86_64",
    all(target_arch = "wasm32", feature = "wasm"),
    target_arch = "aarch64"
))]

use thermite::Vector;
use thermite::prelude::*;
use thermite_special::SpecialMath;

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

fn lbeta(a: f64, b: f64) -> f64 {
    D::splat(a).lbeta(D::splat(b)).extract::<0>()
}
fn logit(p: f64) -> f64 {
    D::splat(p).logit().extract::<0>()
}
fn logit_1m(q: f64) -> f64 {
    D::splat(q).logit_1m().extract::<0>()
}
fn planck(x: f64) -> f64 {
    D::splat(x).planck().extract::<0>()
}

// --- lbeta(a, b) = ln|B(a, b)| ---

#[test]
fn lbeta_values() {
    close("lbeta(1,1)", lbeta(1.0, 1.0) + 1.0, 1.0, 1e-15); // B(1,1) = 1, so ln = 0
    close("lbeta(2,3)", lbeta(2.0, 3.0), -2.4849066497880003102, 1e-14);
    close("lbeta(.5,.5)", lbeta(0.5, 0.5), 1.1447298858494001741, 1e-14);

    // B(a,1) = 1/a exactly, so this is -ln(200). Tolerance is looser on purpose: the
    // lgamma terms here are near 860 and the answer near -5.3, so about two digits go
    // to cancellation. That is a property of the log form, not a defect. See `lbeta`.
    close("lbeta(200,1)", lbeta(200.0, 1.0), -5.2983173665480366775, 1e-13);
}

#[test]
fn lbeta_covers_the_range_beta_cannot() {
    // B(200,200) is about 9.7e-122: representable in f64, long gone in f32. The log form
    // is an ordinary number in both.
    close("lbeta(200,200)", lbeta(200.0, 200.0), -278.64189378441853349, 1e-13);

    let direct_f32 = F::splat(200.0).beta(F::splat(200.0)).extract::<0>();
    assert_eq!(direct_f32, 0.0, "precondition: f32 beta is expected to underflow here");

    // On the default policy. This originally had to run at `Best` because the f32
    // `lgamma` below that tier was wrong for arguments this large, the defect this test
    // first tripped over. See `tests/lgamma_large_argument.rs`.
    let log_f32 = F::splat(200.0).lbeta(F::splat(200.0)).extract::<0>();
    close("lbeta f32", log_f32 as f64, -278.64189378441853349, 1e-4);
}

// --- logit(p) and logit_1m(q) ---

#[test]
fn logit_values_and_inverse() {
    close("logit(0.5)", logit(0.5) + 1.0, 1.0, 1e-15);
    close("logit(0.75)", logit(0.75), 3.0_f64.ln(), 1e-15);
    assert_eq!(logit(0.0), f64::NEG_INFINITY);
    assert_eq!(logit(1.0), f64::INFINITY);

    // Inverse of logistic_sigmoid across the interior of the range.
    for k in 1..20 {
        let p = k as f64 / 20.0;
        let back = D::splat(logit(p)).logistic_sigmoid().extract::<0>();
        close("sigmoid(logit(p))", back, p, 1e-14);
    }
}

#[test]
fn logit_1m_is_the_reflection() {
    // logit_1m(q) == -logit(q) as functions of the same number. They differ in which
    // probability the argument names.
    for k in 1..20 {
        let q = k as f64 / 20.0;
        close("logit_1m reflection", logit_1m(q), -logit(q), 1e-14);
    }
}

#[test]
fn logit_1m_reaches_where_logit_cannot() {
    // q = 1e-17. The complementary probability p = 1 - q is not representable in f64
    // (it rounds to exactly 1), so logit(p) can only return infinity. Working in q keeps
    // the answer finite and accurate.
    let q = 1e-17_f64;
    let p = 1.0 - q;
    assert_eq!(p, 1.0, "precondition: 1 - q is expected to round to one here");
    assert_eq!(logit(p), f64::INFINITY);

    close("logit_1m(1e-17)", logit_1m(q), 39.143946580898776618, 1e-14);
}

// --- planck(x) = x^3/(e^x - 1) ---

#[test]
fn planck_values_and_limit() {
    close("planck(0)", planck(0.0), 0.0, 0.0); // the 0/0, vanishing like x^2
    close("planck(1)", planck(1.0), 0.58197670686932642439, 1e-14);
    close("planck(2.8214)", planck(2.8214), 1.4214354724066744895, 1e-13);
}

#[test]
fn planck_vanishes_quadratically() {
    // x^3/(e^x - 1) -> x^2 as x -> 0. The direct form has already lost the answer at
    // 1e-17, where e^x rounds to 1 and the denominator is exactly zero.
    let x = 1e-17_f64;
    let naive = x * x * x / (x.exp() - 1.0);
    assert!(
        !naive.is_finite(),
        "precondition: the direct form is expected to be non-finite here, got {naive:?}"
    );

    close("planck tiny", planck(x), x * x, 1e-14);

    for k in 0..10 {
        let x = 1e-3 / (2.0_f64).powi(k);
        // x^2 * (1 - x/2 + x^2/12) is the leading expansion.
        let series = x * x * (1.0 - x / 2.0 + x * x / 12.0);
        close("planck series", planck(x), series, 1e-13);
    }
}

#[test]
fn planck_peaks_at_wien() {
    // The Wien displacement root of 3(1 - e^-x) = x, near 2.8214394.
    let peak = 2.821_439_372_1_f64;
    let f = planck(peak);
    assert!(
        planck(peak - 1e-3) < f && planck(peak + 1e-3) < f,
        "not a maximum at {peak}"
    );
}

// --- composites get these for free from the default bodies ---

#[test]
fn composites_inherit_the_new_forms() {
    use thermite_compensated::Compensated;

    type C = Compensated<D>;

    let p = C::new(D::splat(0.25));
    close(
        "compensated logit",
        p.logit().value().extract::<0>(),
        logit(0.25),
        1e-14,
    );

    let x = C::new(D::splat(1.0));
    close(
        "compensated planck",
        x.planck().value().extract::<0>(),
        planck(1.0),
        1e-13,
    );
}

/// `phi::<N>` on a composite takes the element-agnostic default, whose series arm
/// iterates until it converges to the element's own epsilon, so double-double gets a
/// double-double answer, not an f64 one.
#[test]
fn compensated_phi_converges_past_f64() {
    use thermite_compensated::Compensated;
    type C = Compensated<D>;

    // phi_3(0.75) from mpmath, split into the nearest f64 and its remainder.
    let (want_hi, want_lo) = (0.20325929863745107_f64, -2.887853226508e-18_f64);
    let c = C::new(D::splat(0.75)).phi::<3>();
    let (hi, lo) = (c.value().extract::<0>(), c.error().extract::<0>());
    close("compensated phi hi", hi, want_hi, 2e-16);
    // The pair need not be normalized to the nearest hi, so compare the residual of the
    // whole against the reference: (hi - want_hi) is exact by Sterbenz, and adding lo
    // to it must land on want_lo to about 1e-32 of the value.
    close("compensated phi lo", (hi - want_hi) + lo, want_lo, 1e-9);

    // And the recurrence arm, above the split.
    let far = C::new(D::splat(3.0)).phi::<3>().value().extract::<0>();
    close("compensated phi far", far, 0.4290939601180617681825, 2e-15);
}
