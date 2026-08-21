//! Poisson mass and log-mass against mpmath, across the kernel's two branches (`k < 9`
//! plain log-Gamma, `k >= 9` Loader's saddle-point form) and lambda from zero to three
//! times the peak. References from `scripts/poisson_ref.py`.
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

include!("poisson_ref/table.rs");

/// Relative error in units of eps.
fn ulps(got: f64, want: f64, eps: f64) -> f64 {
    if want == 0.0 {
        return if got == 0.0 { 0.0 } else { f64::INFINITY };
    }
    ((got - want) / want).abs() / eps
}

/// Budget in ulp: a few for the pieces, plus the exponent's conditioning. The exponent
/// (which is `ln pmf` up to the `ln(2 pi k)/2`) is a few roundings of its own size, and
/// the exponential turns each into as many ulp of the mass, one of which is allowed. Near the peak
/// (`|k - lambda| < 0.2 (k + lambda)`) that is all. Away from it the direct
/// `k ln(k/lambda) + lambda - k` also cancels to half an ulp of its terms. Below `k = 9`
/// the plain form's `k ln lambda - lambda - lgamma(k+1)` costs its terms' size too.
fn budget(k: f64, lambda: f64, ln_pmf: f64) -> f64 {
    let expo = if ln_pmf.is_finite() { ln_pmf.abs() } else { 0.0 };
    if k < 9.0 {
        let e = if lambda > 0.0 {
            (k * lambda.ln()).abs() + lambda + 13.0
        } else {
            0.0
        };
        // 9 rather than 8: with a fused scalar mul_adde (aarch64, x86 +fma) the shifted
        // Stirling form rounds differently and k = 1, lambda = 0.9 measures 16.0 eps
        // against this branch's former 15.5.
        return 9.0 + e / 2.0;
    }
    let v = (k - lambda) / (k + lambda);
    let cancel = if v.abs() < 0.2 {
        0.0
    } else {
        (k * (k / lambda).ln()).abs() + (lambda - k).abs()
    };
    8.0 + expo + cancel / 2.0
}

#[test]
fn pmf_matches_mpmath() {
    for &(k, lambda, want, want_ln) in POIS.iter() {
        let got = D::splat(k).poisson_pmf(D::splat(lambda)).extract::<0>();
        let u = ulps(got, want, f64::EPSILON);
        let b = budget(k, lambda, want_ln);
        assert!(
            u <= b,
            "f64 P({k}; {lambda}): got {got:e} want {want:e}, {u:.1} ulp (budget {b:.1})"
        );

        // f32: skip what underflows there, and the huge k where the peak is 3.5 ulp wide.
        if want > 1e-36 && k <= 1000.0 {
            let got = F::splat(k as f32).poisson_pmf(F::splat(lambda as f32)).extract::<0>() as f64;
            let u = ulps(got, want, f32::EPSILON as f64);
            assert!(
                u <= b,
                "f32 P({k}; {lambda}): got {got:e} want {want:e}, {u:.1} ulp (budget {b:.1})"
            );
        }
    }
}

#[test]
fn log_pmf_matches_mpmath() {
    // Absolute error in the log is the relative error of the mass, so the same budget, in eps.
    for &(k, lambda, _, want) in POIS.iter() {
        let got = D::splat(k).poisson_log_pmf(D::splat(lambda)).extract::<0>();
        if want == f64::NEG_INFINITY {
            assert_eq!(got, f64::NEG_INFINITY, "ln P({k}; {lambda})");
            continue;
        }
        let err = (got - want).abs() / f64::EPSILON;
        // The log itself is O(|want|) and rounds to half an ulp of that too.
        let b = budget(k, lambda, want) + want.abs() / 2.0;
        assert!(
            err <= b,
            "f64 ln P({k}; {lambda}): got {got} want {want}, {err:.1} eps (budget {b:.1})"
        );
    }
}

#[test]
fn log_pmf_stays_finite_where_the_mass_underflows() {
    let got = D::splat(20.0).poisson_log_pmf(D::splat(2000.0)).extract::<0>();
    // 20 ln 2000 - 2000 - ln 20! = 152.03 - 2000 - 42.34
    assert!((got + 1890.31).abs() < 0.01, "{got}");
    assert_eq!(D::splat(20.0).poisson_pmf(D::splat(2000.0)).extract::<0>(), 0.0);
}

#[test]
fn edges() {
    // lambda = 0: 1 at k = 0, 0 above. k = 0: e^{-lambda}.
    assert_eq!(D::ZERO.poisson_pmf(D::ZERO).extract::<0>(), 1.0);
    assert_eq!(D::splat(3.0).poisson_pmf(D::ZERO).extract::<0>(), 0.0);
    assert_eq!(D::splat(30.0).poisson_pmf(D::ZERO).extract::<0>(), 0.0);
    // k = 0 is not special-cased: it runs the shifted Stirling form (n = 9, exponent
    // ~ -10.8) whose own budget here is ~15.75 eps. Unfused scalar arithmetic happens to
    // land within 2 eps of exp(-2.5); the fused mul_adde lowering (aarch64, x86 +fma)
    // measures 13 ulp. Both are deterministic and inside the kernel's design budget.
    let e = D::ZERO.poisson_pmf(D::splat(2.5)).extract::<0>();
    assert!((e - (-2.5f64).exp()).abs() <= 16.0 * f64::EPSILON * e, "{e}");
}

#[test]
fn lanes_stay_independent() {
    use thermite::backend::scalar::Scalar;
    type D4 = thermite::simd::f64x4<Scalar>;

    // Mixed small and large k in one vector must equal the per-lane scalars, which each
    // take a uniform branch.
    let ks = [2.0, 8.5, 9.0, 100.0];
    let ls = [1.5, 8.0, 12.0, 90.0];
    let got = D4::new(ks).poisson_pmf(D4::new(ls));
    for lane in 0..4 {
        assert_eq!(
            got.as_slice()[lane],
            D::splat(ks[lane]).poisson_pmf(D::splat(ls[lane])).extract::<0>(),
            "lane {lane}"
        );
    }
}
