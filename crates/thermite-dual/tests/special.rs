//! Special-function derivatives on `Dual` (requires the `special` feature).
#![cfg(feature = "special")]

use std::f64::consts::PI;

use thermite::prelude::*;
use thermite_dual::Dual;
use thermite_special::{RealPrimalMath, RealSpecialMath, SpecialMath};
use thermite_special::bessel::{Ai, AiPrime, Bi, BiPrime, I, J, K, Scaled, Y};

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
    // At x = 0: sigma = 0.5, derivative = 0.25, which is defaults differentiating
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
        let r = D::variable(V::splat(x), 0).expint_n::<3>();
        let expect = -V::splat(x).expint_n::<2>().extract::<0>();

        assert!(close(
            r.re.extract::<0>(),
            V::splat(x).expint_n::<3>().extract::<0>(),
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
        let r = D::variable(V::splat(x), 0).expint_n::<1>();

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
        let real = VF::splat(x).expint_n::<8>().extract::<0>();
        let dual = DF::variable(VF::splat(x), 0).expint_n::<8>().re.extract::<0>();

        assert_eq!(dual, real, "E_8({x}) diverged from the guarded real path");
    }
}

#[test]
fn expint_order_zero_is_the_closed_form() {
    // N = 0 is a closed form, not a case of E_1. Both paths return E_0(x) = e^-x / x.
    for &x in &[0.5, 3.0, 20.0] {
        let r = D::variable(V::splat(x), 0).expint_n::<0>();

        assert!(close(r.re.extract::<0>(), (-x).exp() / x, 1e-14), "E_0({x})");
        // E_0'(x) = -E_{-1}(x) = -e^-x (1 + 1/x) / x
        let expect = -(-x).exp() * (1.0 + 1.0 / x) / x;
        assert!(close(r.dual[0].extract::<0>(), expect, 1e-14), "E_0'({x})");
    }
}

#[test]
fn yeo_johnson_derivative_is_the_two_sided_power_rule() {
    // psi'(y, l) = (1 + |y|)^(s - 1) where s is l above zero and 2 - l below, the same
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

#[test]
fn trigamma_value_and_derivative() {
    use thermite::math::policy::DefaultPolicy;
    use thermite_special::specialized::SpecializedSpecialMath;

    // Dual trigamma was a todo!() until polygamma's runtime order closed the family.
    let x = D::variable(V::splat(2.5), 0);
    let r = SpecializedSpecialMath::trigamma::<DefaultPolicy>(x);

    let f = |x: f64| SpecializedSpecialMath::trigamma::<DefaultPolicy>(V::splat(x)).extract::<0>();
    assert!(close(r.re.extract::<0>(), f(2.5), 1e-14));
    // psi_1' = psi_2. The finite difference of the value function is the check that
    // is independent of the chain rule's own psi_2 call.
    assert!(close(r.dual[0].extract::<0>(), ndiff(f, 2.5), 1e-5));
    assert!(close(r.dual[1].extract::<0>(), 0.0, 1e-14));
}

#[test]
fn polygamma_value_and_derivative() {
    // Every order differentiates to the next one up. Check value against the inner
    // vector and derivative against a finite difference of the value function, on
    // both axes. Tolerances are scaled by the derivative's own size, since psi_{n+1}
    // spans many orders of magnitude across (n, x).
    for &n in &[0u32, 1, 2, 5, 10] {
        for &xv in &[0.75_f64, 2.5, 8.0, -1.25] {
            let f = |x: f64| V::splat(x).polygamma(n).extract::<0>();
            let r = D::variable(V::splat(xv), 0).polygamma(n);

            let v = r.re.extract::<0>();
            assert!(close(v, f(xv), 1e-12 * v.abs().max(1.0)), "psi_{n}({xv}) value");

            let d = r.dual[0].extract::<0>();
            assert!(close(d, ndiff(f, xv), 1e-5 * d.abs().max(1.0)), "psi_{n}'({xv})");
        }
    }
}

// --- Jacobi elliptic functions ---
//
// The triple is closed under d/du, which is why `Dual` overrides it rather than
// differentiating the Landen ladder. These pin both that the override fires and that the
// fallback for a dual-valued modulus still produces a derivative.

#[test]
fn jacobi_elliptic_derivatives_close_the_triple() {
    // sn' = cn dn, cn' = -sn dn, dn' = -k^2 sn cn, all in u.
    for &(u, k) in &[(0.7, 0.5), (1.9, 0.9), (-2.3, 0.25), (0.0, 0.6)] {
        let x = D::variable(V::splat(u), 0);
        let kk = D::constant(V::splat(k));
        let (sn, cn, dn) = x.jacobi_elliptic(kk);

        let (s, c, d) = (sn.re.extract::<0>(), cn.re.extract::<0>(), dn.re.extract::<0>());
        let got = (
            sn.dual[0].extract::<0>(),
            cn.dual[0].extract::<0>(),
            dn.dual[0].extract::<0>(),
        );

        assert!(close(got.0, c * d, 1e-13), "sn' at u={u}, k={k}");
        assert!(close(got.1, -s * d, 1e-13), "cn' at u={u}, k={k}");
        assert!(close(got.2, -k * k * s * c, 1e-13), "dn' at u={u}, k={k}");

        // The unseeded slot stays clean, and a constant modulus contributes nothing.
        assert!(close(sn.dual[1].extract::<0>(), 0.0, 1e-15), "slot 1 leaked at u={u}");
    }
}

#[test]
fn jacobi_elliptic_derivatives_match_finite_differences() {
    // Independent of the closure identities above: differentiate the plain real function.
    for &(u, k) in &[(0.7, 0.5), (1.9, 0.9), (-2.3, 0.25)] {
        let x = D::variable(V::splat(u), 0);
        let (sn, cn, dn) = x.jacobi_elliptic(D::constant(V::splat(k)));

        let val = |i: usize| {
            move |t: f64| {
                let (s, c, d) = V::splat(t).jacobi_elliptic(V::splat(k));
                [s, c, d][i].extract::<0>()
            }
        };
        for (i, got) in [sn.dual[0], cn.dual[0], dn.dual[0]].iter().enumerate() {
            let want = ndiff(val(i), u);
            assert!(
                close(got.extract::<0>(), want, 1e-7),
                "component {i} at u={u}, k={k}: got {}, finite difference {want}",
                got.extract::<0>()
            );
        }
    }
}

#[test]
fn jacobi_elliptic_differentiates_in_the_modulus_too() {
    // A dual-valued modulus is the case the closed-form shortcut cannot serve, so this
    // exercises the fallback through the ladder. Gate is loose on purpose: it is a
    // differentiated iteration, not a closed form.
    for &(u, k) in &[(0.7, 0.5), (1.3, 0.75)] {
        let uu = D::constant(V::splat(u));
        let kk = D::variable(V::splat(k), 1);
        let (sn, cn, dn) = uu.jacobi_elliptic(kk);

        let val = |i: usize| {
            move |t: f64| {
                let (s, c, d) = V::splat(u).jacobi_elliptic(V::splat(t));
                [s, c, d][i].extract::<0>()
            }
        };
        for (i, got) in [sn.dual[1], cn.dual[1], dn.dual[1]].iter().enumerate() {
            let want = ndiff(val(i), k);
            assert!(
                close(got.extract::<0>(), want, 1e-6),
                "d/dk component {i} at u={u}, k={k}: got {}, finite difference {want}",
                got.extract::<0>()
            );
        }
        // The values themselves must be unaffected by which slot carries the seed.
        let plain = V::splat(u).jacobi_elliptic(V::splat(k));
        assert!(close(sn.re.extract::<0>(), plain.0.extract::<0>(), 1e-14));
    }
}

// --- Polylogarithm ---
//
// Li_s' = Li_{s-1}/z: the order steps down by one, so the chain closes at every depth in
// either order class, and the origin's limit is exactly 1.

#[test]
fn polylog_value_and_derivative() {
    use thermite_special::PolylogOrder;
    let orders: [PolylogOrder<f64, i64>; 7] = [
        PolylogOrder::Integer(2),
        PolylogOrder::Integer(3),
        PolylogOrder::Integer(1),
        PolylogOrder::Integer(-2),
        PolylogOrder::Real(0.5),
        PolylogOrder::Real(2.5),
        PolylogOrder::Real(-1.5),
    ];
    // The dual vector's order carries its dual element, but a constant order has none.
    let dual_order = |o: PolylogOrder<f64, i64>| -> PolylogOrder<Dual<f64, 2>, i64> {
        match o {
            PolylogOrder::Integer(n) => PolylogOrder::Integer(n),
            PolylogOrder::Real(s) => PolylogOrder::Real(Dual { re: s, dual: [0.0; 2] }),
        }
    };
    for order in orders {
        for &z in &[-30.0f64, -0.7, 0.3, 0.9, 5.0] {
            let f = |t: f64| V::splat(t).polylog(order).extract::<0>();
            let r = D::variable(V::splat(z), 0).polylog(dual_order(order));
            assert!(close(r.re.extract::<0>(), f(z), 1e-14), "value at {order:?}, z={z}");
            let want = ndiff(f, z);
            let got = r.dual[0].extract::<0>();
            assert!(
                (got - want).abs() <= 1e-6 * want.abs().max(1.0),
                "Li' at {order:?}, z={z}: got {got}, finite difference {want}"
            );
            assert!(close(r.dual[1].extract::<0>(), 0.0, 1e-15), "slot 1 leaked at {order:?}, z={z}");
        }
        // The origin: Li_s(z) ~ z, so the derivative is exactly 1.
        let r = D::variable(V::splat(0.0), 0).polylog(dual_order(order));
        assert_eq!(r.re.extract::<0>(), 0.0);
        assert_eq!(r.dual[0].extract::<0>(), 1.0, "origin slope at {order:?}");
    }
    // Nesting: the second derivative of Li_3 is (Li_1(z) - Li_2(z)) / z^2, through Dual<Dual>.
    type DD = Dual<D, 1>;
    let z = 0.4f64;
    let inner = D::variable(V::splat(z), 0);
    let x = DD::variable(inner, 0);
    let r = x.polylog(PolylogOrder::Integer(3));
    let second = r.dual[0].dual[0].extract::<0>();
    let li1 = V::splat(z).polylog(PolylogOrder::Integer(1)).extract::<0>();
    let li2 = V::splat(z).polylog(PolylogOrder::Integer(2)).extract::<0>();
    let want = (li1 - li2) / (z * z);
    assert!(close(second, want, 1e-13), "Li_3'' at {z}: {second} vs {want}");
}

// --- Riemann zeta ---
//
// Unlike the Gamma family, zeta does not close under differentiation: zeta'(s) is its own
// Dirichlet series, so the Dual impl reaches a companion kernel rather than a chain rule. The
// two arms are exercised separately because s < 0 goes through the functional equation and its
// derivative is a different expression.

#[test]
fn zeta_derivative_matches_finite_differences() {
    for &s in &[1.4f64, 2.0, 3.5, 7.0, 20.0] {
        let x = D::variable(V::splat(s), 0);
        let r = x.zeta();

        assert!(close(r.re.extract::<0>(), V::splat(s).zeta().extract::<0>(), 1e-14));
        let want = ndiff(|t| V::splat(t).zeta().extract::<0>(), s);
        assert!(
            close(r.dual[0].extract::<0>(), want, 1e-6),
            "zeta'({s}): got {}, finite difference {want}",
            r.dual[0].extract::<0>()
        );
        assert!(close(r.dual[1].extract::<0>(), 0.0, 1e-15), "slot 1 leaked at s={s}");
    }
}

// zeta and zetac differ by a constant, so their derivatives must agree exactly.
#[test]
fn zetac_shares_zetas_derivative() {
    for &s in &[1.6f64, 4.0, 12.0, -2.5] {
        let x = D::variable(V::splat(s), 0);
        let a = x.zeta().dual[0].extract::<0>();
        let b = x.zetac().dual[0].extract::<0>();
        assert_eq!(a, b, "zeta' and zetac' must be identical at s={s}");
    }
}

// The reflected arm. Its derivative comes from the product rule on the functional equation,
// with the cotangent's pole folded against chi's own sine so the trivial zeros (where
// zeta(s) = 0 and cot blows up) do not produce 0 * inf.
#[test]
fn zeta_derivative_through_the_functional_equation() {
    for &s in &[-0.5f64, -1.5, -2.5, -4.0, -7.5] {
        let x = D::variable(V::splat(s), 0);
        let got = x.zeta().dual[0].extract::<0>();
        let want = ndiff(|t| V::splat(t).zeta().extract::<0>(), s);
        assert!(
            close(got, want, 1e-5),
            "zeta'({s}): got {got}, finite difference {want}"
        );
    }

    // At the trivial zeros the value is 0 while cot(pi s / 2) diverges. The derivative is
    // finite and nonzero there, and must not come back NaN.
    for n in 1..=4 {
        let s = -2.0 * n as f64;
        let x = D::variable(V::splat(s), 0);
        let r = x.zeta();
        assert!(r.re.extract::<0>().abs() < 1e-15, "zeta({s}) is a trivial zero");
        let d = r.dual[0].extract::<0>();
        assert!(
            d.is_finite() && d != 0.0,
            "zeta'({s}) should be finite and nonzero, got {d}"
        );
    }
}

// --- Bessel, all four families ---
//
// The identities being checked reach DOWN one order, which is the whole reason these can be
// differentiated at all with only orders 0 and 1 of J/Y available:
//
//     I_N' =  I_{N-1} - (N/x) I_N      K_N' = -K_{N-1} - (N/x) K_N
//     J_N' =  J_{N-1} - (N/x) J_N      Y_N' =  Y_{N-1} - (N/x) Y_N
//
// The textbook spelling `J_N' = (J_{N-1} - J_{N+1})/2` needs order N+1, and believing that was
// the documented reason `bessel_j` sat disabled crate-wide for want of "orders beyond J_0".

/// Central difference of a scalar function, for checking a dual derivative independently.
fn central<F: Fn(f64) -> f64>(f: F, x: f64) -> f64 {
    let h = 1e-6 * x.abs().max(1.0);
    (f(x + h) - f(x - h)) / (2.0 * h)
}

#[test]
fn bessel_i_dual_matches_central_differences() {
    for &x in &[0.25f64, 1.0, 3.0, 7.0, 7.75, 9.0, 20.0] {
        for (name, got, plain) in [
            (
                "I_0",
                D::variable(V::splat(x), 0).bessel_n::<I, 0>().dual[0].extract::<0>(),
                (|t: f64| V::splat(t).bessel_n::<I, 0>().extract::<0>()) as fn(f64) -> f64,
            ),
            (
                "I_1",
                D::variable(V::splat(x), 0).bessel_n::<I, 1>().dual[0].extract::<0>(),
                (|t: f64| V::splat(t).bessel_n::<I, 1>().extract::<0>()) as fn(f64) -> f64,
            ),
            (
                "I_3",
                D::variable(V::splat(x), 0).bessel_n::<I, 3>().dual[0].extract::<0>(),
                (|t: f64| V::splat(t).bessel_n::<I, 3>().extract::<0>()) as fn(f64) -> f64,
            ),
        ] {
            let want = central(plain, x);
            let e = ((got - want) / want).abs();
            assert!(e <= 1e-8, "d/dx {name}({x}): dual {got}, central {want}, rel {e:e}");
        }
    }
}

#[test]
fn bessel_k_dual_matches_central_differences() {
    for &x in &[0.25f64, 0.9, 1.0, 1.1, 4.0, 20.0] {
        for (name, got, plain) in [
            (
                "K_0",
                D::variable(V::splat(x), 0).bessel_n::<K, 0>().dual[0].extract::<0>(),
                (|t: f64| V::splat(t).bessel_n::<K, 0>().extract::<0>()) as fn(f64) -> f64,
            ),
            (
                "K_1",
                D::variable(V::splat(x), 0).bessel_n::<K, 1>().dual[0].extract::<0>(),
                (|t: f64| V::splat(t).bessel_n::<K, 1>().extract::<0>()) as fn(f64) -> f64,
            ),
            (
                "K_3",
                D::variable(V::splat(x), 0).bessel_n::<K, 3>().dual[0].extract::<0>(),
                (|t: f64| V::splat(t).bessel_n::<K, 3>().extract::<0>()) as fn(f64) -> f64,
            ),
        ] {
            let want = central(plain, x);
            let e = ((got - want) / want).abs();
            assert!(e <= 1e-8, "d/dx {name}({x}): dual {got}, central {want}, rel {e:e}");
        }
    }
}

#[test]
fn bessel_jy_dual_matches_central_differences() {
    // Away from the zeros, where a relative comparison of the derivative is meaningful.
    for &x in &[0.5f64, 1.0, 3.0, 5.0, 7.9, 8.1, 12.0, 40.0] {
        for (name, got, plain) in [
            (
                "J_0",
                D::variable(V::splat(x), 0).bessel_n::<J, 0>().dual[0].extract::<0>(),
                (|t: f64| V::splat(t).bessel_n::<J, 0>().extract::<0>()) as fn(f64) -> f64,
            ),
            (
                "J_1",
                D::variable(V::splat(x), 0).bessel_n::<J, 1>().dual[0].extract::<0>(),
                (|t: f64| V::splat(t).bessel_n::<J, 1>().extract::<0>()) as fn(f64) -> f64,
            ),
            (
                "Y_0",
                D::variable(V::splat(x), 0).bessel_n::<Y, 0>().dual[0].extract::<0>(),
                (|t: f64| V::splat(t).bessel_n::<Y, 0>().extract::<0>()) as fn(f64) -> f64,
            ),
            (
                "Y_1",
                D::variable(V::splat(x), 0).bessel_n::<Y, 1>().dual[0].extract::<0>(),
                (|t: f64| V::splat(t).bessel_n::<Y, 1>().extract::<0>()) as fn(f64) -> f64,
            ),
        ] {
            let want = central(plain, x);
            // Scaled by the envelope, since the derivative also passes through zero.
            let amp = (2.0 / (core::f64::consts::PI * x)).sqrt();
            let e = (got - want).abs() / amp.max(want.abs());
            assert!(
                e <= 1e-8,
                "d/dx {name}({x}): dual {got}, central {want}, envelope-rel {e:e}"
            );
        }
    }
}

#[test]
fn bessel_dual_values_are_untouched() {
    // The dual override must not perturb the value half. It is the same kernel, and any
    // difference means the override recomputed rather than reused.
    for &x in &[0.5f64, 3.0, 9.0] {
        let d = D::variable(V::splat(x), 0);
        for (name, a, b) in [
            (
                "I_2",
                d.bessel_n::<I, 2>().re.extract::<0>(),
                V::splat(x).bessel_n::<I, 2>().extract::<0>(),
            ),
            (
                "K_2",
                d.bessel_n::<K, 2>().re.extract::<0>(),
                V::splat(x).bessel_n::<K, 2>().extract::<0>(),
            ),
            (
                "J_1",
                d.bessel_n::<J, 1>().re.extract::<0>(),
                V::splat(x).bessel_n::<J, 1>().extract::<0>(),
            ),
            (
                "Y_0",
                d.bessel_n::<Y, 0>().re.extract::<0>(),
                V::splat(x).bessel_n::<Y, 0>().extract::<0>(),
            ),
        ] {
            assert!(
                a.to_bits() == b.to_bits(),
                "{name}({x}) value moved under Dual: {a} vs {b}"
            );
        }
    }
}

#[test]
fn bessel_dual_closes_the_wronskian_derivative() {
    // A structural check that does not go through central differences at all: differentiating
    // the Wronskian J_1 Y_0 - J_0 Y_1 = 2/(pi x) must give -2/(pi x^2). All four derivative
    // identities participate, so a sign error in any single one shows up here.
    for &x in &[0.7f64, 2.0, 6.0, 30.0] {
        let d = D::variable(V::splat(x), 0);
        let w = d.bessel_n::<J, 1>() * d.bessel_n::<Y, 0>() - d.bessel_n::<J, 0>() * d.bessel_n::<Y, 1>();
        let got = w.dual[0].extract::<0>();
        let want = -2.0 / (core::f64::consts::PI * x * x);
        let e = ((got - want) / want).abs();
        assert!(e <= 1e-9, "d/dx Wronskian at {x}: {got}, want {want}, rel {e:e}");
    }
}

/// Negative orders differentiate the way they evaluate: reflecting the order scales the
/// function by a constant `(-1)^n`, so `d/dx J_{-n} = (-1)^n J_n'` and `K`/`I` carry no sign
/// at all. Bit-exact, because a sign flip is exact: a tolerance would hide a derivative
/// reflected by some second, slightly different route than the value.
#[test]
fn bessel_negative_order_derivatives_reflect_exactly() {
    for &x in &[1.0f64, 3.5, 8.0, 15.0] {
        let d = |v: f64| D::variable(V::splat(v), 0);

        // J and Y flip at odd order, not at even.
        assert_eq!(
            d(x).bessel_n::<J, -3>().dual[0].extract::<0>().to_bits(),
            (-d(x).bessel_n::<J, 3>().dual[0].extract::<0>()).to_bits(),
            "d/dx J_-3({x}) must be exactly -d/dx J_3"
        );
        assert_eq!(
            d(x).bessel_n::<J, -2>().dual[0].extract::<0>().to_bits(),
            d(x).bessel_n::<J, 2>().dual[0].extract::<0>().to_bits(),
            "d/dx J_-2({x}) must be exactly d/dx J_2"
        );
        assert_eq!(
            d(x).bessel_n::<Y, -1>().dual[0].extract::<0>().to_bits(),
            (-d(x).bessel_n::<Y, 1>().dual[0].extract::<0>()).to_bits(),
            "d/dx Y_-1({x}) must be exactly -d/dx Y_1"
        );

        // I and K are even in integer order, derivative included.
        assert_eq!(
            d(x).bessel_n::<Scaled<I>, -3>().dual[0].extract::<0>().to_bits(),
            d(x).bessel_n::<Scaled<I>, 3>().dual[0].extract::<0>().to_bits(),
            "d/dx I_-3({x}) must equal d/dx I_3"
        );
        assert_eq!(
            d(x).bessel_n::<Scaled<K>, -2>().dual[0].extract::<0>().to_bits(),
            d(x).bessel_n::<Scaled<K>, 2>().dual[0].extract::<0>().to_bits(),
            "d/dx K_-2({x}) must equal d/dx K_2"
        );
    }
}

// --- Airy ---
//
// The one family in the crate whose derivative rule is its own definition: `w'' = x w`, so
// `Ai' = Ai'` and `Ai'' = x Ai`. Nothing differentiates the Bessel machinery underneath.

#[test]
fn airy_dual_matches_central_differences() {
    for &x in &[-8.0f64, -3.5, -1.0, -0.25, 0.25, 1.0, 3.5, 8.0] {
        let d = D::variable(V::splat(x), 0);

        for (name, got, plain) in [
            (
                "Ai",
                d.airy::<Ai>().dual[0].extract::<0>(),
                (|t: f64| V::splat(t).airy::<Ai>().extract::<0>()) as fn(f64) -> f64,
            ),
            (
                "Ai'",
                d.airy::<AiPrime>().dual[0].extract::<0>(),
                (|t: f64| V::splat(t).airy::<AiPrime>().extract::<0>()) as fn(f64) -> f64,
            ),
            (
                "Bi",
                d.airy::<Bi>().dual[0].extract::<0>(),
                (|t: f64| V::splat(t).airy::<Bi>().extract::<0>()) as fn(f64) -> f64,
            ),
            (
                "Bi'",
                d.airy::<BiPrime>().dual[0].extract::<0>(),
                (|t: f64| V::splat(t).airy::<BiPrime>().extract::<0>()) as fn(f64) -> f64,
            ),
        ] {
            let want = central(plain, x);
            let tol = 1e-6 * (1.0 + want.abs());
            assert!(
                close(got, want, tol),
                "d/dx {name}({x}): dual {got}, central {want}"
            );
        }
    }
}

/// The derivative the chain rule produces must be the _other_ function exactly, not merely
/// close to a finite difference. `d/dx Ai = Ai'` is an identity, not an approximation, so this
/// is a bitwise claim.
#[test]
fn airy_derivatives_are_the_other_member_exactly() {
    for &x in &[-8.0f64, -1.0, 0.0, 1.0, 8.0] {
        let d = D::variable(V::splat(x), 0);
        let v = V::splat(x);

        assert_eq!(
            d.airy::<Ai>().dual[0].extract::<0>().to_bits(),
            v.airy::<AiPrime>().extract::<0>().to_bits(),
            "d/dx Ai({x}) must be Ai'({x})"
        );
        assert_eq!(
            d.airy::<Bi>().dual[0].extract::<0>().to_bits(),
            v.airy::<BiPrime>().extract::<0>().to_bits(),
            "d/dx Bi({x}) must be Bi'({x})"
        );

        // And the second derivative is the defining equation.
        assert_eq!(
            d.airy::<AiPrime>().dual[0].extract::<0>().to_bits(),
            (v * v.airy::<Ai>()).extract::<0>().to_bits(),
            "d/dx Ai'({x}) must be x Ai({x})"
        );
        assert_eq!(
            d.airy::<BiPrime>().dual[0].extract::<0>().to_bits(),
            (v * v.airy::<Bi>()).extract::<0>().to_bits(),
            "d/dx Bi'({x}) must be x Bi({x})"
        );
    }
}

/// The scaled forms carry the scaling's own derivative, `dzeta/dx = sqrt(x)`, which does not
/// cancel. These are a genuinely different rule from the unscaled ones, and the one most
/// likely to be got wrong. Checked against central differences of the scaled functions
/// themselves, which is independent of the unscaled path entirely.
#[test]
fn scaled_airy_dual_matches_central_differences() {
    for &x in &[-6.0f64, -1.0, 0.5, 2.0, 6.0, 20.0] {
        let d = D::variable(V::splat(x), 0);

        for (name, got, plain) in [
            (
                "eAi",
                d.airy::<Scaled<Ai>>().dual[0].extract::<0>(),
                (|t: f64| V::splat(t).airy::<Scaled<Ai>>().extract::<0>()) as fn(f64) -> f64,
            ),
            (
                "eAi'",
                d.airy::<Scaled<AiPrime>>().dual[0].extract::<0>(),
                (|t: f64| V::splat(t).airy::<Scaled<AiPrime>>().extract::<0>()) as fn(f64) -> f64,
            ),
            (
                "eBi",
                d.airy::<Scaled<Bi>>().dual[0].extract::<0>(),
                (|t: f64| V::splat(t).airy::<Scaled<Bi>>().extract::<0>()) as fn(f64) -> f64,
            ),
            (
                "eBi'",
                d.airy::<Scaled<BiPrime>>().dual[0].extract::<0>(),
                (|t: f64| V::splat(t).airy::<Scaled<BiPrime>>().extract::<0>()) as fn(f64) -> f64,
            ),
        ] {
            let want = central(plain, x);
            let tol = 1e-5 * (1.0 + want.abs());
            assert!(
                close(got, want, tol),
                "d/dx {name}({x}): dual {got}, central {want}"
            );
        }
    }
}

/// The tuple entry differentiates all four in one pass. Each component must match the
/// single-value entry that computes it on its own, value and derivative alike.
#[test]
fn the_airy_tuple_and_the_single_entries_agree_on_dual() {
    for &x in &[-5.0f64, -0.5, 0.5, 5.0] {
        let d = D::variable(V::splat(x), 0);
        let (ai, aip, bi, bip) = d.airy_all::<false>();

        for (name, from_tuple, single) in [
            ("Ai", ai, d.airy::<Ai>()),
            ("Ai'", aip, d.airy::<AiPrime>()),
            ("Bi", bi, d.airy::<Bi>()),
            ("Bi'", bip, d.airy::<BiPrime>()),
        ] {
            assert_eq!(
                from_tuple.re.extract::<0>().to_bits(),
                single.re.extract::<0>().to_bits(),
                "{name}({x}) value"
            );
            assert_eq!(
                from_tuple.dual[0].extract::<0>().to_bits(),
                single.dual[0].extract::<0>().to_bits(),
                "{name}({x}) derivative"
            );
        }
    }
}

// --- spherical Bessel ---
//
// `f_n' = f_{n-1} - ((n+1)/x) f_n`, with the `n+1` rather than the cylindrical `n` because
// differentiating the `sqrt(pi/2x)` between the two normalizations contributes the extra half.
// That off-by-a-half is exactly the kind of thing a central difference catches and a reread
// does not.

#[test]
fn spherical_bessel_dual_matches_central_differences() {
    for &x in &[0.3f64, 1.0, 2.5, 6.0, 15.0] {
        for (name, got, plain) in [
            (
                "j0",
                D::variable(V::splat(x), 0).sph_bessel_n::<J, 0>().dual[0].extract::<0>(),
                (|t: f64| V::splat(t).sph_bessel_n::<J, 0>().extract::<0>()) as fn(f64) -> f64,
            ),
            (
                "j3",
                D::variable(V::splat(x), 0).sph_bessel_n::<J, 3>().dual[0].extract::<0>(),
                (|t: f64| V::splat(t).sph_bessel_n::<J, 3>().extract::<0>()) as fn(f64) -> f64,
            ),
            (
                "y0",
                D::variable(V::splat(x), 0).sph_bessel_n::<Y, 0>().dual[0].extract::<0>(),
                (|t: f64| V::splat(t).sph_bessel_n::<Y, 0>().extract::<0>()) as fn(f64) -> f64,
            ),
            (
                "y4",
                D::variable(V::splat(x), 0).sph_bessel_n::<Y, 4>().dual[0].extract::<0>(),
                (|t: f64| V::splat(t).sph_bessel_n::<Y, 4>().extract::<0>()) as fn(f64) -> f64,
            ),
            (
                "i2",
                D::variable(V::splat(x), 0).sph_bessel_n::<I, 2>().dual[0].extract::<0>(),
                (|t: f64| V::splat(t).sph_bessel_n::<I, 2>().extract::<0>()) as fn(f64) -> f64,
            ),
            (
                "k2",
                D::variable(V::splat(x), 0).sph_bessel_n::<K, 2>().dual[0].extract::<0>(),
                (|t: f64| V::splat(t).sph_bessel_n::<K, 2>().extract::<0>()) as fn(f64) -> f64,
            ),
            (
                "i3e",
                D::variable(V::splat(x), 0).sph_bessel_n::<Scaled<I>, 3>().dual[0].extract::<0>(),
                (|t: f64| V::splat(t).sph_bessel_n::<Scaled<I>, 3>().extract::<0>()) as fn(f64) -> f64,
            ),
            (
                "k3e",
                D::variable(V::splat(x), 0).sph_bessel_n::<Scaled<K>, 3>().dual[0].extract::<0>(),
                (|t: f64| V::splat(t).sph_bessel_n::<Scaled<K>, 3>().extract::<0>()) as fn(f64) -> f64,
            ),
        ] {
            let want = central(plain, x);
            let tol = 1e-6 * (1.0 + want.abs());
            assert!(
                close(got, want, tol),
                "d/dx {name}({x}): dual {got}, central {want}"
            );
        }
    }
}

/// The origin, where the derivative identity `f_{n-1} - ((n+1)/x) f_n` reads `inf * 0` or
/// `inf - inf` and the kernel selects the limit instead: `j_1'(0) = i_1'(0) = 1/3`, every
/// other finite member is flat, and the singular ones are infinite with the sign opposite to
/// their value.
#[test]
fn spherical_derivatives_at_the_origin() {
    let z = V::splat(0.0);
    let d = |v: D| v.dual[0].extract::<0>();

    assert_eq!(d(D::variable(z, 0).sph_bessel_n::<J, 0>()), 0.0);
    assert_eq!(d(D::variable(z, 0).sph_bessel_n::<J, 1>()), 1.0 / 3.0);
    assert_eq!(d(D::variable(z, 0).sph_bessel_n::<J, 2>()), 0.0);
    assert_eq!(d(D::variable(z, 0).sph_bessel_n::<I, 0>()), 0.0);
    assert_eq!(d(D::variable(z, 0).sph_bessel_n::<I, 1>()), 1.0 / 3.0);
    // `e^{-x} i_0` has slope `-1` at the origin: the scaling's own derivative.
    assert_eq!(d(D::variable(z, 0).sph_bessel_n::<Scaled<I>, 0>()), -1.0);
    assert_eq!(d(D::variable(z, 0).sph_bessel_n::<Y, 0>()), f64::INFINITY);
    assert_eq!(d(D::variable(z, 0).sph_bessel_n::<Y, 3>()), f64::INFINITY);
    assert_eq!(d(D::variable(z, 0).sph_bessel_n::<K, 1>()), f64::NEG_INFINITY);
    assert_eq!(d(D::variable(z, 0).sph_bessel_n::<Scaled<K>, 2>()), f64::NEG_INFINITY);
}

/// `j_0(x) = sin(x)/x`, so `j_0'(x) = cos(x)/x - sin(x)/x^2` in closed form, a reference the
/// recurrence had no part in producing.
#[test]
fn spherical_j0_derivative_has_a_closed_form() {
    for &x in &[0.2f64, 1.0, 4.0, 11.0] {
        let got = D::variable(V::splat(x), 0).sph_bessel_n::<J, 0>().dual[0].extract::<0>();
        let want = x.cos() / x - x.sin() / (x * x);
        assert!(close(got, want, 1e-13 * (1.0 + want.abs())), "j_0'({x}): {got} vs {want}");
    }
}

/// `ndtr' = phi`, `log_ndtr' = phi/Phi` (the inverse Mills ratio, which the override takes
/// from `erfcx` so it stays finite where `Phi` has underflowed) and
/// `logerfc' = -2/(sqrt(pi) erfcx(x))`. References from mpmath at 60 digits.
#[test]
#[allow(clippy::excessive_precision)]
fn normal_cdf_family_derivatives() {
    // (x, phi(x), phi(x)/Phi(x), d/dx ln erfc(x))
    const ROWS: &[(f64, f64, f64, f64)] = &[
        // d logerfc(-30) is -7.7e-392: an underflow to -0 in f64, the true answer here.
        (-30.0, 1.4736461348785475e-196, 30.033259667433677, 0.0),
        (-1.0, 0.24197072451914335, 1.5251352761609812, -0.22527124262865746),
        (0.3, 0.38138781546052409, 0.61722085361273445, -1.5360470858030584),
        (4.0, 0.00013383022576488535, 0.00013383446446857514, -8.2363768927001761),
    ];
    let rel = |got: f64, want: f64, name: &str| {
        let tol = 1e-13 * want.abs().max(1e-300);
        assert!(close(got, want, tol), "{name}: {got:e} vs {want:e}");
    };
    for &(x, pdf, mills, dle) in ROWS {
        let xd = D::variable(V::splat(x), 0);
        let n = xd.ndtr();
        let l = xd.log_ndtr();
        let e = xd.logerfc();
        rel(n.dual[0].extract::<0>(), pdf, "ndtr'");
        rel(l.dual[0].extract::<0>(), mills, "log_ndtr'");
        rel(e.dual[0].extract::<0>(), dle, "logerfc'");
        assert_eq!(n.dual[1].extract::<0>(), 0.0);
        assert_eq!(l.dual[1].extract::<0>(), 0.0);
        assert_eq!(e.dual[1].extract::<0>(), 0.0);
        // The values are the real kernels' values.
        assert_eq!(l.re.extract::<0>(), V::splat(x).log_ndtr().extract::<0>());
    }
}

/// The Newton inverses differentiate by the implicit function theorem, never through the
/// loop: `dx/dy = 1/f'(x)` with `f'` the forward's derivative at the returned root.
#[test]
fn newton_inverses_differentiate_implicitly() {
    let d = |v: D| (v.re.extract::<0>(), v.dual[0].extract::<0>());
    let rel = |got: f64, want: f64, name: &str| {
        assert!(close(got, want, 1e-12 * want.abs().max(1e-300)), "{name}: {got:e} vs {want:e}");
    };

    for &y in &[-500.0f64, -20.0, -1.5, -1e-3] {
        let (x, dx) = d(D::variable(V::splat(y), 0).inv_log_ndtr());
        // 1 / (phi(x)/Phi(x)) = Phi(x) sqrt(2 pi) e^{x^2/2}, from the forward's own derivative.
        let mills = D::variable(V::splat(x), 0).log_ndtr().dual[0].extract::<0>();
        rel(dx, 1.0 / mills, "inv_log_ndtr'");
    }
    for &y in &[-30.0f64, -2.0, 0.5, 4.0] {
        let (x, dx) = d(D::variable(V::splat(y), 0).inv_digamma());
        rel(dx, 1.0 / V::splat(x).trigamma().extract::<0>(), "inv_digamma'");
    }
    for &x in &[-10.0f64, -1.0, 1.0, 100.0] {
        let (w, dw) = d(D::variable(V::splat(x), 0).wright_omega());
        rel(dw, w / (1.0 + w), "wright_omega'");
    }
    // A' = 1 - A^2 - (2 nu - 1) A / x, and the inverse's slope is its reciprocal.
    for &(nu, x) in &[(1.0f64, 0.5f64), (1.5, 2.0), (4.0, 3.0), (25.0, 200.0)] {
        let (a, da) = d(D::variable(V::splat(x), 0).bessel_ratio::<I>(D::constant(V::splat(nu))));
        rel(da, 1.0 - a * a - (2.0 * nu - 1.0) * a / x, "bessel_i_ratio'");
        let (k, dk) = d(D::variable(V::splat(a), 0).inv_bessel_ratio::<I>(D::constant(V::splat(nu))));
        rel(dk, 1.0 / (1.0 - a * a - (2.0 * nu - 1.0) * a / k), "inv_bessel_i_ratio'");
    }
}

// ---------------------------------------------------------------------------
// Fresnel and the trigonometric integrals. Every derivative here is closed form
// (they are the integrands the functions are defined by), so these are checked
// against the exact expression rather than a central difference.
// ---------------------------------------------------------------------------

#[test]
fn fresnel_value_and_derivative() {
    for &x in &[0.25f64, 1.3, 2.5265, 3.0, 7.5] {
        let v = D::variable(V::splat(x), 0);
        let (c, s) = v.fresnel();

        let (rc, rs) = V::splat(x).fresnel();
        assert!(close(c.re.extract::<0>(), rc.extract::<0>(), 1e-15), "C value at {x}");
        assert!(close(s.re.extract::<0>(), rs.extract::<0>(), 1e-15), "S value at {x}");

        // C' = cos(pi x^2/2), S' = sin(pi x^2/2).
        let t = PI * x * x / 2.0;
        assert!(close(c.dual[0].extract::<0>(), t.cos(), 1e-13), "C' at {x}");
        assert!(close(s.dual[0].extract::<0>(), t.sin(), 1e-13), "S' at {x}");
        assert!(close(c.dual[1].extract::<0>(), 0.0, 1e-15));
    }
}

/// The reason the rule reuses the kernel's two-word phase. Past the crossover the
/// values have settled to 1/2 plus a ripple, but the derivatives still swing over the
/// full [-1, 1]. A phase good enough for the values is not good enough here.
#[test]
fn fresnel_derivative_phase_survives_large_arguments() {
    for &x in &[123.4567f64, 1234.5678, 98765.4321] {
        let v = D::variable(V::splat(x), 0);
        let (c, s) = v.fresnel();
        let (dc, ds) = (c.dual[0].extract::<0>(), s.dual[0].extract::<0>());
        assert!(dc.abs() <= 1.0 + 1e-12 && ds.abs() <= 1.0 + 1e-12, "amplitude at {x}");
        // cos^2 + sin^2 = 1 is the phase-independent invariant, and fails loudly
        // if the two are evaluated at different reductions.
        assert!(close(dc * dc + ds * ds, 1.0, 1e-12), "C'^2 + S'^2 at {x}");
    }
}

#[test]
fn sici_value_and_derivative() {
    for &x in &[0.25f64, 1.3, 6.0, 12.0, 20.0] {
        let v = D::variable(V::splat(x), 0);
        let (si, ci) = v.sici();

        let (rsi, rci) = V::splat(x).sici();
        assert!(close(si.re.extract::<0>(), rsi.extract::<0>(), 1e-15), "Si value at {x}");
        assert!(close(ci.re.extract::<0>(), rci.extract::<0>(), 1e-15), "Ci value at {x}");

        // Si' = sin(x)/x, Ci' = cos(x)/x.
        assert!(close(si.dual[0].extract::<0>(), x.sin() / x, 1e-13), "Si' at {x}");
        assert!(close(ci.dual[0].extract::<0>(), x.cos() / x, 1e-13), "Ci' at {x}");
    }
}

/// `Si` is odd and `Ci` is evaluated at `|x|`, and the two arrive at the same
/// derivative expression from different directions: `sinc` is even, and
/// `sign(x) cos(|x|)/|x|` is `cos(x)/x`. Neither needs a fold.
#[test]
fn sici_derivative_on_the_negative_axis() {
    for &x in &[-0.75f64, -4.0, -15.0] {
        let v = D::variable(V::splat(x), 0);
        let (si, ci) = v.sici();
        assert!(close(si.dual[0].extract::<0>(), x.sin() / x, 1e-13), "Si' at {x}");
        assert!(close(ci.dual[0].extract::<0>(), x.cos() / x, 1e-13), "Ci' at {x}");
    }
}
