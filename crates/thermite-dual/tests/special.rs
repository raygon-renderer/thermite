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

// psi reference by recurrence + Bernoulli asymptotic series, since libm has no digamma.
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

#[test]
fn expint_derivative_is_the_next_lower_order() {
    // Differentiating E_N under the integral sign pulls down a factor of -t, dropping
    // the order by one: E_N'(x) = -E_{N-1}(x). Checked against the real E_{N-1} rather
    // than a finite difference so it is an identity test, not an accuracy test.
    for &x in &[0.25, 0.75, 1.5, 4.0, 12.0] {
        let r = D::variable(V::splat(x), 0).expint::<3>();
        let expect = -V::splat(x).expint::<2>().extract::<0>();

        assert!(close(
            r.re.extract::<0>(),
            V::splat(x).expint::<3>().extract::<0>(),
            1e-15
        ));
        assert!(close(r.dual[0].extract::<0>(), expect, 1e-15), "E_3'({x}) != -E_2({x})");
        assert!(close(r.dual[1].extract::<0>(), 0.0, 1e-15));
    }
}

#[test]
fn expint_order_one_derivative_is_plain_exp() {
    // E_1'(x) = -E_0(x) = -e^-x / x, the bottom of the ladder.
    for &x in &[0.5, 2.0, 9.0] {
        let r = D::variable(V::splat(x), 0).expint::<1>();

        assert!(close(r.dual[0].extract::<0>(), -(-x).exp() / x, 1e-14), "E_1'({x})");
    }
}

#[test]
fn expint_high_order_large_argument_takes_the_guarded_path() {
    // The generic default applies the forward order recurrence unconditionally, and that
    // recurrence amplifies error by |x|^(N-1)/(N-1)!. The real path switches to an
    // asymptotic series above RECURRENCE_THRESHOLD (~33 for f32 at N=8) precisely to avoid
    // it, so `Dual` must delegate the value rather than inherit the default. These agree
    // bit for bit when it does.
    type VF = Vector<f32>;
    type DF = Dual<VF, 1>;

    for &x in &[40.0f32, 55.0, 70.0, 85.0] {
        let real = VF::splat(x).expint::<8>().extract::<0>();
        let dual = DF::variable(VF::splat(x), 0).expint::<8>().re.extract::<0>();

        assert_eq!(dual, real, "E_8({x}) diverged from the guarded real path");
    }
}

#[test]
fn expint_order_zero_is_the_closed_form() {
    // N = 0 is a closed form, not a case of E_1. Both paths return E_0(x) = e^-x / x.
    for &x in &[0.5, 3.0, 20.0] {
        let r = D::variable(V::splat(x), 0).expint::<0>();

        assert!(close(r.re.extract::<0>(), (-x).exp() / x, 1e-14), "E_0({x})");
        // E_0'(x) = -E_{-1}(x) = -e^-x (1 + 1/x) / x
        let expect = -(-x).exp() * (1.0 + 1.0 / x) / x;
        assert!(close(r.dual[0].extract::<0>(), expect, 1e-14), "E_0'({x})");
    }
}

#[test]
fn yeo_johnson_derivative_is_the_two_sided_power_rule() {
    // psi'(y, l) = (1 + |y|)^(s - 1) where s is l above zero and 2 - l below - the same
    // reflection the value uses, which is what makes psi continuously differentiable
    // through the origin. The kernel folds the sign rather than branching four ways, so
    // this is the check that the fold differentiates correctly on both sides.
    for &l in &[0.0_f64, 0.5, 1.0, 1.5, 2.0, 3.0, -1.0] {
        for &y in &[0.25_f64, 1.0, 3.0] {
            let r = D::variable(V::splat(y), 0).yeo_johnson(D::constant(V::splat(l)));
            let expect = (1.0 + y).powf(l - 1.0);
            assert!(close(r.dual[0].extract::<0>(), expect, 1e-11), "psi'({y}, {l})");

            let r = D::variable(V::splat(-y), 0).yeo_johnson(D::constant(V::splat(l)));
            let expect = (1.0 + y).powf(1.0 - l);
            assert!(close(r.dual[0].extract::<0>(), expect, 1e-11), "psi'({}, {l})", -y);
        }

        // psi'(0, l) = 1 for every lambda, from both sides: the origin is where the two
        // branches meet, and they meet smoothly.
        let r = D::variable(V::splat(0.0), 0).yeo_johnson(D::constant(V::splat(l)));
        assert!(close(r.dual[0].extract::<0>(), 1.0, 1e-12), "psi'(0, {l})");
    }
}

#[test]
fn the_boxcox_family_differentiates_to_its_own_inverses() {
    // d/dx boxcox(x, l) = x^(l-1), and the inverse transforms differentiate to the
    // reciprocal of the forward derivative evaluated at the recovered point, the identity
    // that any correct inverse pair satisfies, checked here through autodiff rather than
    // through a second closed form.
    for &l in &[0.0_f64, 0.5, 1.0, 2.0, -1.5] {
        for &x in &[0.5_f64, 1.0, 3.0] {
            let r = D::variable(V::splat(x), 0).boxcox(D::constant(V::splat(l)));
            assert!(
                close(r.dual[0].extract::<0>(), x.powf(l - 1.0), 1e-11),
                "boxcox'({x}, {l})"
            );

            let y = r.re.extract::<0>();
            let inv = D::variable(V::splat(y), 0).inv_boxcox(D::constant(V::splat(l)));
            assert!(
                close(inv.re.extract::<0>(), x, 1e-11),
                "inv_boxcox round trip at ({x}, {l})"
            );
            assert!(
                close(inv.dual[0].extract::<0>(), 1.0 / x.powf(l - 1.0), 1e-10),
                "inv_boxcox'({y}, {l})"
            );
        }
    }
}
