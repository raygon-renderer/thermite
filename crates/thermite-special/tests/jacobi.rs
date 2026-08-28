//! Jacobi polynomials `P_n^{(alpha,beta)}(x)` and their derivatives.
//!
//! `jacobi`'s `m` parameter is the derivative order, not an associated index: the kernel
//! multiplies by the Gamma ratio and shifts both weights, implementing
//!
//! ```text
//! d^m/dx^m P_n^{(a,b)}(x) = prod_{j=1..m} (n+a+b+j)/2 * P_{n-m}^{(a+m,b+m)}(x)
//! ```
//!
//! so the reference tables hold exact derivatives of the exact polynomial rather than
//! values of a shifted one. Getting that scale factor wrong is the most likely failure
//! and it is invisible at `m = 0`.
//!
//! Two kinds of check. The tables pin absolute values across five weight pairs, including
//! asymmetric integer and half-integer ones that no symmetry could accidentally satisfy.
//! The reduction tests then tie `jacobi` to two *other* kernels in this crate (Legendre
//! at `alpha = beta = 0`, and Chebyshev at `alpha = beta = -1/2`), which catches a
//! consistent-but-wrong recurrence that agrees with its own reference.
//!
//! Tolerances scale with `sum |c_k| |x|^k`, as in `legendre.rs`; the derivative rows need
//! it most, since differentiating multiplies the coefficients by their exponents.
#![cfg(any(
    target_arch = "x86",
    target_arch = "x86_64",
    all(target_arch = "wasm32", feature = "wasm"),
    target_arch = "aarch64"
))]

use thermite::prelude::*;
use thermite_special::SpecialMath;

type D = Vector<f64>;
type F = Vector<f32>;

const XS: [f64; 9] = [0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 0.875, -0.3125, -0.6875];

include!("jacobi_ref/table.rs");

#[track_caller]
fn close_cond(name: &str, got: f64, want: f64, cond: f64, eps: f64) {
    let bound = eps * cond.max(1.0);
    let err = (got - want).abs();
    assert!(
        err <= bound,
        "{name}: got {got:?}, want {want:?} (err {err:e} > bound {bound:e})"
    );
}

#[test]
fn values_and_derivatives_match_the_exact_reference() {
    for &(a, b, n, m, ref vals, ref conds) in JACOBI.iter() {
        for (j, &x) in XS.iter().enumerate() {
            let got = D::splat(x).jacobi(D::splat(a), D::splat(b), n, m).extract::<0>();
            // The Gamma-ratio prefactor and the weight shift both ride on m, so a wrong
            // scale shows up as a clean multiplicative offset here.
            close_cond(
                &format!("P_{n}^({a},{b}) d{m} at {x}"),
                got,
                vals[j],
                // A few extra ulps of headroom: the kernel builds the prefactor as a
                // running product of m terms, which the coefficient sum does not model.
                conds[j] * 8.0,
                f64::EPSILON,
            );
        }
    }
}

#[test]
fn degree_zero_is_one_for_every_weight() {
    for &(a, b, ..) in JACOBI.iter() {
        for &x in &XS {
            let got = D::splat(x).jacobi(D::splat(a), D::splat(b), 0, 0).extract::<0>();
            assert_eq!(got, 1.0, "P_0^({a},{b})({x})");
        }
    }
}

#[test]
fn reduces_to_legendre_at_zero_weights() {
    // P_n^{(0,0)} = P_n. This ties `jacobi` to a kernel with an entirely separate
    // implementation (unrolled Estrin polynomials), so a recurrence that is
    // self-consistently wrong still fails here.
    for n in 0..14u32 {
        for &x in &XS {
            let v = D::splat(x);
            let jac = v.jacobi(D::ZERO, D::ZERO, n, 0).extract::<0>();
            let leg = v.legendre(n, 0).extract::<0>();
            assert!(
                (jac - leg).abs() <= 1e-12 * leg.abs().max(1.0),
                "P_{n}^(0,0)({x}): jacobi {jac} vs legendre {leg}"
            );
        }
    }
}

#[test]
fn reduces_to_chebyshev_t_at_minus_half_weights() {
    // P_n^{(-1/2,-1/2)}(x) = [C(2n,n) / 4^n] T_n(x), the classical Gegenbauer limit. The
    // right-hand side comes from `chebyshev`, which shares no code with `jacobi`.
    for n in 1..8u32 {
        // A single non-zero coefficient isolates T_n itself.
        let mut coeffs = [0.0f64; 8];
        coeffs[n as usize] = 1.0;

        // C(2n,n) / 4^n, built as a product to stay exact for these small n.
        let mut ratio = 1.0f64;
        for k in 1..=n {
            ratio *= (2.0 * k as f64 - 1.0) / (2.0 * k as f64);
        }

        for &x in &XS {
            let v = D::splat(x);
            let jac = v.jacobi(D::splat(-0.5), D::splat(-0.5), n, 0).extract::<0>();
            let cheb = v.chebyshev_n::<1, 8>(&coeffs).extract::<0>();
            let want = ratio * cheb;
            assert!(
                (jac - want).abs() <= 1e-12 * want.abs().max(1.0),
                "P_{n}^(-1/2,-1/2)({x}): got {jac}, want {want} (= {ratio} * T_{n})"
            );
        }
    }
}

#[test]
fn swapping_the_weights_reflects_the_argument() {
    // P_n^{(a,b)}(-x) = (-1)^n P_n^{(b,a)}(x). Holds for every weight pair, and a kernel
    // that mixed up alpha and beta anywhere in the recurrence breaks it.
    for &(a, b, n, m, ..) in JACOBI.iter() {
        if m != 0 {
            continue; // the identity above is for the polynomial, not its derivative
        }
        for &x in &XS {
            let lhs = D::splat(-x).jacobi(D::splat(a), D::splat(b), n, 0).extract::<0>();
            let rhs = D::splat(x).jacobi(D::splat(b), D::splat(a), n, 0).extract::<0>();
            let want = if n % 2 == 0 { rhs } else { -rhs };
            assert!(
                (lhs - want).abs() <= 1e-11 * want.abs().max(1.0),
                "reflection P_{n}^({a},{b})(-{x}): got {lhs}, want {want}"
            );
        }
    }
}

#[test]
fn the_value_at_one_is_the_binomial_coefficient() {
    // P_n^{(a,b)}(1) = C(n+a, n), independent of beta. With integer alpha that is an
    // exact small integer, which makes this a tight check on the recurrence's endpoint.
    for a in 0..4u32 {
        for &b in &[0.0, 0.5, 2.0, -0.5] {
            for n in 0..7u32 {
                let mut want = 1.0f64;
                for k in 1..=n {
                    want *= (a as f64 + k as f64) / k as f64;
                }
                let got = D::splat(1.0)
                    .jacobi(D::splat(a as f64), D::splat(b), n, 0)
                    .extract::<0>();
                assert!(
                    (got - want).abs() <= 1e-11 * want.abs(),
                    "P_{n}^({a},{b})(1): got {got}, want {want}"
                );
            }
        }
    }
}

#[test]
fn orders_above_the_degree_vanish() {
    // m > n differentiates a degree-n polynomial more than n times.
    for &(a, b, ..) in JACOBI.iter().take(20) {
        for n in 0..5u32 {
            for m in (n + 1)..(n + 3) {
                for &x in &XS {
                    let got = D::splat(x).jacobi(D::splat(a), D::splat(b), n, m).extract::<0>();
                    assert_eq!(got, 0.0, "P_{n}^({a},{b}) d{m} at {x} should vanish, got {got}");
                }
            }
        }
    }
}

#[test]
fn lanes_carrying_different_arguments_stay_independent() {
    use thermite::backend::scalar::Scalar;
    type D4 = thermite::simd::f64x4<Scalar>;

    let xs = [XS[0], XS[3], XS[6], XS[8]];
    for &(a, b, n, m, ..) in JACOBI.iter().take(40) {
        let got = D4::new(xs).jacobi(D4::splat(a), D4::splat(b), n, m);
        for (lane, &x) in xs.iter().enumerate() {
            let want = D::splat(x).jacobi(D::splat(a), D::splat(b), n, m).extract::<0>();
            assert_eq!(got.as_slice()[lane], want, "P_{n}^({a},{b}) d{m} lane {lane}");
        }
    }
}

#[test]
fn weights_can_vary_per_lane() {
    // alpha and beta are vectors, not scalars, so unlike n and m they can differ across
    // lanes. Nothing else here exercises that.
    use thermite::backend::scalar::Scalar;
    type D4 = thermite::simd::f64x4<Scalar>;

    let alphas = [0.0, 0.5, 1.0, 2.0];
    let betas = [0.0, -0.5, 2.0, 1.5];

    for n in 0..7u32 {
        for &x in &XS {
            let got = D4::splat(x).jacobi(D4::new(alphas), D4::new(betas), n, 0);
            for lane in 0..4 {
                let want = D::splat(x)
                    .jacobi(D::splat(alphas[lane]), D::splat(betas[lane]), n, 0)
                    .extract::<0>();
                assert_eq!(got.as_slice()[lane], want, "n={n} x={x} lane={lane}");
            }
        }
    }
}

// --- f32 ---

#[test]
fn f32_tracks_the_same_reference() {
    for &(a, b, n, m, ref vals, ref conds) in JACOBI.iter() {
        if n > 5 {
            continue; // beyond this the f32 conditioning bound stops being informative
        }
        for (j, &x) in XS.iter().enumerate() {
            let got = F::splat(x as f32)
                .jacobi(F::splat(a as f32), F::splat(b as f32), n, m)
                .extract::<0>() as f64;
            close_cond(
                &format!("f32 P_{n}^({a},{b}) d{m} at {x}"),
                got,
                vals[j],
                conds[j] * 8.0,
                f32::EPSILON as f64,
            );
        }
    }
}
