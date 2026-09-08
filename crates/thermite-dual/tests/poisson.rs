//! `Dual`'s Poisson/Laguerre family: the compensated exponent, and the chain rules.
//!
//! `Dual` overrides `exp_two_sum` componentwise (see the comment on that override), so
//! the compensated exponent runs on duals and the value must be bit-identical to the
//! plain vector's. Losing the compensation is worth about 14 ulp.
//!
//! Dispatches to `X86V3` with non-integer arguments, same as `hermite_function.rs`.

#![cfg(any(target_arch = "x86", target_arch = "x86_64"))]

use thermite::isa::InstructionSet;
use thermite::math::policy::policies::Precision;
use thermite::prelude::*;
use thermite::simd::{HasIsa, Simd};
use thermite_special::SpecialMathWithPolicy;

use thermite_dual::Dual;

type S = thermite::backend::x86_v3::X86V3;
type V = Vector<<S as Simd>::f64x4>;
type D = Dual<V, 1>;

/// `(k, lambda)`. `k < 9` takes the shifted-Stirling path; `k = 13.75` the large branch.
const CASES: &[(f64, f64)] = &[
    (0.0, 0.75),
    (0.0, 37.125),
    (2.5, 3.25),
    (2.5, 11.0625),
    (7.125, 6.5),
    (7.125, 29.75),
    (13.75, 12.25),
];

#[inline(always)]
fn seed(v: f64) -> D {
    Dual::new(V::splat(v), [V::ONE])
}

#[inline(always)]
fn konst(v: f64) -> D {
    Dual::constant(V::splat(v))
}

#[thermite::dispatch(S)]
fn value_matches_the_plain_vector_imp() {
    for &(kv, lv) in CASES {
        let (k, l) = (V::splat(kv), V::splat(lv));
        let (kd, ld) = (konst(kv), konst(lv));

        let plain = <V as SpecialMathWithPolicy>::poisson_pmf_p::<Precision>(k, l);
        let dual = <D as SpecialMathWithPolicy>::poisson_pmf_p::<Precision>(kd, ld);
        assert_eq!(
            dual.re.extract::<0>().to_bits(),
            plain.extract::<0>().to_bits(),
            "poisson_pmf({kv}, {lv}): dual value differs from the plain vector, so the \
             dual is not taking the compensated exponent"
        );

        let plain = <V as SpecialMathWithPolicy>::poisson_log_pmf_p::<Precision>(k, l);
        let dual = <D as SpecialMathWithPolicy>::poisson_log_pmf_p::<Precision>(kd, ld);
        assert_eq!(
            dual.re.extract::<0>().to_bits(),
            plain.extract::<0>().to_bits(),
            "poisson_log_pmf({kv}, {lv})"
        );
    }
}

#[thermite::dispatch(S)]
fn derivative_in_lambda_imp() {
    // d/dlambda of e^-lambda lambda^k / Gamma(k+1) is P * (k/lambda - 1).
    // Verified against 40-digit mpmath, worst relative mismatch 2.2e-41.
    for &(kv, lv) in CASES {
        let got = <D as SpecialMathWithPolicy>::poisson_pmf_p::<Precision>(konst(kv), seed(lv)).dual[0]
            .extract::<0>();

        let p = <V as SpecialMathWithPolicy>::poisson_pmf_p::<Precision>(V::splat(kv), V::splat(lv))
            .extract::<0>();
        let want = p * (kv / lv - 1.0);

        let scale = want.abs().max(got.abs());
        assert!(
            (got - want).abs() <= 1e-12 * scale,
            "d/dlambda poisson_pmf({kv}, {lv}): got {got:e}, want {want:e}"
        );
    }
}

#[thermite::dispatch(S)]
fn derivative_in_k_imp() {
    // d/dk is P * (ln lambda - psi(k+1)). Verified against 40-digit mpmath, worst
    // relative mismatch 2.1e-40.
    use thermite_special::SpecialMath;

    for &(kv, lv) in CASES {
        let got = <D as SpecialMathWithPolicy>::poisson_pmf_p::<Precision>(seed(kv), konst(lv)).dual[0]
            .extract::<0>();

        let p = <V as SpecialMathWithPolicy>::poisson_pmf_p::<Precision>(V::splat(kv), V::splat(lv))
            .extract::<0>();
        let psi = (V::splat(kv) + V::ONE).digamma().extract::<0>();
        let want = p * (lv.ln() - psi);

        // `k = 0` matters: `pmf_parts` used to dodge `0 * ln 0` by selecting the whole
        // `k * ln lambda` term away, which has the right value and the wrong derivative
        // (off by 50% at lambda = 0.75, 86% at 37.125).
        let scale = want.abs().max(got.abs());
        assert!(
            (got - want).abs() <= 1e-11 * scale,
            "d/dk poisson_pmf({kv}, {lv}): got {got:e}, want {want:e}"
        );
    }
}

#[thermite::dispatch(S)]
fn laguerre_value_matches_the_plain_vector_imp() {
    // `laguerre_function` seeds through `pmf_parts`, so it inherits the same split.
    for &(xv, av) in &[(0.75, 0.0), (3.25, 1.5), (11.0625, 4.25), (29.75, 0.5)] {
        let plain =
            <V as SpecialMathWithPolicy>::laguerre_function_n_p::<Precision, 5>(V::splat(xv), V::splat(av));
        let dual =
            <D as SpecialMathWithPolicy>::laguerre_function_n_p::<Precision, 5>(konst(xv), konst(av));
        assert_eq!(
            dual.re.extract::<0>().to_bits(),
            plain.extract::<0>().to_bits(),
            "laguerre_function_n({xv}, alpha = {av})"
        );
    }
}

fn skip() -> bool {
    if <S as HasIsa>::ISA > InstructionSet::get() {
        eprintln!("skipped: {:?} unavailable (host is {:?})", <S as HasIsa>::ISA, InstructionSet::get());
        return true;
    }
    false
}

#[test]
fn value_matches_the_plain_vector() {
    if skip() {
        return;
    }
    value_matches_the_plain_vector_imp();
}

#[test]
fn derivative_in_lambda() {
    if skip() {
        return;
    }
    derivative_in_lambda_imp();
}

#[test]
fn derivative_in_k() {
    if skip() {
        return;
    }
    derivative_in_k_imp();
}

#[test]
fn laguerre_value_matches_the_plain_vector() {
    if skip() {
        return;
    }
    laguerre_value_matches_the_plain_vector_imp();
}
