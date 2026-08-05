//! Special-function derivatives on `Dual` (requires the `special` feature).
#![cfg(feature = "special")]

use std::f64::consts::PI;

use thermite::prelude::*;
use thermite_dual::Dual;
use thermite_special::{RealPrimalMath, RealSpecialMath, SpecialMath};

type V = Vector<f64>;
type D = Dual<V, 2>;

fn close(a: f64, b: f64, eps: f64) -> bool {
    let d = a - b;
    let d = if d < 0.0 { -d } else { d };
    d < eps
}

#[test]
fn erf_value_and_derivative() {
    // d/dx erf(x) = 2/sqrt(pi) * e^(-x^2), at x = 0.5.
    let x = D::variable(V::splat(0.5), 0);
    let r = x.erf();

    assert!(close(r.re.extract::<0>(), libm::erf(0.5), 1e-12));
    let der = 2.0 / PI.sqrt() * (-0.25f64).exp();
    assert!(close(r.dual[0].extract::<0>(), der, 1e-12));
    assert!(close(r.dual[1].extract::<0>(), 0.0, 1e-12));
}

#[test]
fn erfinv_roundtrip_and_derivative() {
    // erf(erfinv(x)) == x, and d/dx erfinv(x) = (sqrt(pi)/2) e^(erfinv(x)^2).
    let x = D::variable(V::splat(0.5), 0);
    let r = x.erfinv();
    let v = r.re.extract::<0>();

    assert!(close(libm::erf(v), 0.5, 1e-9));
    let der = (PI.sqrt() / 2.0) * (v * v).exp();
    assert!(close(r.dual[0].extract::<0>(), der, 1e-9));
}

#[test]
fn lambert_w_principal_branch() {
    // W0 satisfies W e^W = x; W'(x) = W / (x (1 + W)). At x = 1, W0 = Omega.
    let x = D::variable(V::splat(1.0), 0);
    let (w0, _wm1) = x.lambert_w();
    let w = w0.re.extract::<0>();

    // W e^W == x == 1
    assert!(close(w * w.exp(), 1.0, 1e-9));
    assert!(close(w0.dual[0].extract::<0>(), w / (1.0 + w), 1e-9));
}

#[test]
fn dual_value_only_gelu_matches_primal_gelu_d() {
    // The value-only `gelu` on a Dual must reproduce both outputs of the primal
    // `gelu_d`: its value as `.re`, and the analytic derivative as `.dual[0]`.
    let xi = 0.7_f64;

    // AD path: value-only gelu on a Dual carries the gradient in `.dual`.
    let x = D::variable(V::splat(xi), 0);
    let g = x.gelu(D::constant(V::splat(1.0)));

    // Primal path: the (value, derivative) form on a plain vector.
    let (y, dy) = V::splat(xi).gelu_d(V::splat(1.0));

    assert!(close(g.re.extract::<0>(), y.extract::<0>(), 1e-12));
    assert!(close(g.dual[0].extract::<0>(), dy.extract::<0>(), 1e-9));
}

#[test]
fn composed_default_differentiates() {
    // logistic_sigmoid is a *default* (composes from exp): sigma' = sigma(1-sigma).
    // At x = 0: sigma = 0.5, derivative = 0.25 -- proving defaults differentiate
    // automatically through dual arithmetic.
    let x = D::variable(V::splat(0.0), 0);
    let s = x.logistic_sigmoid();

    assert!(close(s.re.extract::<0>(), 0.5, 1e-12));
    assert!(close(s.dual[0].extract::<0>(), 0.25, 1e-12));
}

// --- Gamma family -----------------------------------------------------------
//
// Derivatives are checked against a central difference of the libm reference
// rather than against thermite-special's own digamma, so a wrong chain rule
// cannot be masked by agreeing with the primitive it was built from.

/// Central difference of `f` at `x`. Truncation ~h^2, roundoff ~eps/h, so with
/// h = 1e-6 the result is good to roughly 1e-9.
fn ndiff(f: impl Fn(f64) -> f64, x: f64) -> f64 {
    let h = 1e-6;
    (f(x + h) - f(x - h)) / (2.0 * h)
}

fn beta_ref(a: f64, b: f64) -> f64 {
    (libm::lgamma(a) + libm::lgamma(b) - libm::lgamma(a + b)).exp()
}

#[test]
fn lgamma_value_and_derivative() {
    // d/dx ln|Gamma(x)| = psi(x)
    let x = D::variable(V::splat(2.5), 0);
    let r = x.lgamma();

    assert!(close(r.re.extract::<0>(), libm::lgamma(2.5), 1e-12));
    assert!(close(r.dual[0].extract::<0>(), ndiff(libm::lgamma, 2.5), 1e-6));
    assert!(close(r.dual[1].extract::<0>(), 0.0, 1e-12));
}

#[test]
fn tgamma_value_and_derivative() {
    // Gamma'(x) = Gamma(x) psi(x)
    let x = D::variable(V::splat(2.5), 0);
    let r = x.tgamma();

    assert!(close(r.re.extract::<0>(), libm::tgamma(2.5), 1e-12));
    assert!(close(r.dual[0].extract::<0>(), ndiff(libm::tgamma, 2.5), 1e-6));
}

#[test]
fn lgamma_negative_argument_derivative() {
    // psi is still the derivative of ln|Gamma| between the poles, where Gamma < 0.
    let x = D::variable(V::splat(-2.5), 0);
    let r = x.lgamma();

    assert!(close(r.re.extract::<0>(), libm::lgamma(-2.5), 1e-12));
    assert!(close(r.dual[0].extract::<0>(), ndiff(libm::lgamma, -2.5), 1e-6));
}

#[test]
fn lgamma_r_matches_lgamma_and_sign_is_constant() {
    let x = D::variable(V::splat(-2.5), 0);
    let (v, sign) = x.lgamma_r();

    // Gamma(-2.5) > 0, and the sign carries no derivative at all.
    assert!(close(sign.re.extract::<0>(), 1.0, 1e-12));
    assert!(close(sign.dual[0].extract::<0>(), 0.0, 1e-12));

    let plain = x.lgamma();
    assert!(close(v.re.extract::<0>(), plain.re.extract::<0>(), 1e-12));
    assert!(close(v.dual[0].extract::<0>(), plain.dual[0].extract::<0>(), 1e-12));
}

#[test]
fn beta_partials_in_both_arguments() {
    // dB/da = B (psi(a) - psi(a+b)), dB/db = B (psi(b) - psi(a+b)).
    // Slot 0 tracks `a`, slot 1 tracks `b`, so one call yields both partials.
    let (a, b) = (2.5, 3.5);
    let da = D::variable(V::splat(a), 0);
    let db = D::variable(V::splat(b), 1);
    let r = da.beta(db);

    assert!(close(r.re.extract::<0>(), beta_ref(a, b), 1e-12));
    assert!(close(r.dual[0].extract::<0>(), ndiff(|t| beta_ref(t, b), a), 1e-6));
    assert!(close(r.dual[1].extract::<0>(), ndiff(|t| beta_ref(a, t), b), 1e-6));
}

#[test]
fn beta_is_symmetric_including_gradients() {
    let (a, b) = (2.5, 3.5);
    let fwd = D::variable(V::splat(a), 0).beta(D::variable(V::splat(b), 1));
    let rev = D::variable(V::splat(b), 1).beta(D::variable(V::splat(a), 0));

    assert!(close(fwd.re.extract::<0>(), rev.re.extract::<0>(), 1e-12));
    assert!(close(fwd.dual[0].extract::<0>(), rev.dual[0].extract::<0>(), 1e-12));
    assert!(close(fwd.dual[1].extract::<0>(), rev.dual[1].extract::<0>(), 1e-12));
}

// psi reference by recurrence + Bernoulli asymptotic series - libm has no digamma.
// Differentiating it numerically gives psi_1, which is what `Dual::digamma` chains
// through, so this checks the trigamma port and the chain rule at once.
fn digamma_ref(mut x: f64) -> f64 {
    let mut result = 0.0;
    while x < 12.0 {
        result -= 1.0 / x;
        x += 1.0;
    }
    let inv = 1.0 / x;
    let inv2 = inv * inv;
    result += x.ln() - 0.5 * inv;
    let mut t = inv2;
    result -= t / 12.0;
    t *= inv2;
    result += t / 120.0;
    t *= inv2;
    result -= t / 252.0;
    t *= inv2;
    result += t / 240.0;
    result
}

#[test]
fn digamma_value_and_derivative() {
    // d/dx psi(x) = psi_1(x), the trigamma function.
    let x = D::variable(V::splat(2.5), 0);
    let r = x.digamma();

    assert!(close(r.re.extract::<0>(), digamma_ref(2.5), 1e-12));
    assert!(close(r.dual[0].extract::<0>(), ndiff(digamma_ref, 2.5), 1e-6));
    assert!(close(r.dual[1].extract::<0>(), 0.0, 1e-12));
}

#[test]
fn digamma_negative_argument_derivative() {
    // Both psi and psi_1 reflect here, so this exercises the reflection identity in
    // the value and in the derivative.
    let x = D::variable(V::splat(-2.5), 0);
    let r = x.digamma();

    assert!(close(r.re.extract::<0>(), digamma_ref(-2.5), 1e-12));
    assert!(close(r.dual[0].extract::<0>(), ndiff(digamma_ref, -2.5), 1e-6));
}
