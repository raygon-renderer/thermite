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
