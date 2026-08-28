//! `inverse_smoothstep` must reach its answer by *converging*, not by bisecting.
//!
//! For order N >= 3 there is no closed form, so the kernel runs
//! `algorithms::newtons_method` inside a bracket. That solver is bisection-safeguarded:
//! a Newton step that lands outside the bracket is silently replaced by the midpoint. The
//! safeguard is what makes a broken derivative **invisible**. The answer stays correct, it
//! just takes ~53 iterations instead of ~5, and nothing in an accuracy test can see the
//! difference.
//!
//! That is not hypothetical. The closure used to return `(fpx * dt_dx * xn1).min(HALF)`,
//! capping the derivative at 0.5 where the true value peaks at 1.875 (N = 3) and rises
//! with N. Every Newton step in the middle of the domain was inflated ~4x, overshot the
//! bracket, and was thrown away, so the whole function ran as pure bisection. Over 2^21
//! points that was **723 ms against 1.13 ms for the forward `smoothstep`**, and flat
//! across N = 3, 5 and 8, which is the tell. A converging solver costs more for a
//! higher-degree polynomial, while a bisecting one costs the same 53 halvings whatever it
//! is bisecting.
//!
//! The derivative is now returned raw. Clamping it the other way (`.max(HALF)`) fixes the
//! middle and breaks the tails far worse, because a floored derivative makes the step
//! tiny, a tiny step is inside the bracket, so the safeguard accepts it and the iteration
//! starves. Mean/max iterations over y from 1e-15 to 1-1e-15: **13.7/29 unclamped, 130/200
//! with a 0.5 floor.**
//!
//! **These tests pin the iteration budget**, which is the only thing that separates the
//! two behaviours, and the threshold sits in the gap between them rather than at either
//! end. Measured at twelve iterations from the midpoint seed:
//!
//! | | round-trip error after 12 iterations |
//! |---|---|
//! | bisection (a bracket of width 1, halved 12 times) | ~2.4e-04 |
//! | Newton, as it stands | **~3.3e-10** |
//! | the threshold below | 1e-08 |
//!
//! Four orders of margin on each side. This is deliberately NOT an accuracy test, since full
//! precision needs a few more iterations than twelve and is covered by the tier sweeps.
//! The only question here is whether the solver is converging or halving.
#![cfg(any(target_arch = "x86", target_arch = "x86_64"))]

use thermite::Vector;
use thermite::math::RealMathWithPolicy;
use thermite::math::policy::{DenormalBehavior, Policy, PolicyParameters, PrecisionPolicy};
use thermite::prelude::*;

type D = Vector<f64>;

/// `Precision` in every respect except that the solver gets twelve iterations.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
struct TightBudget;

impl Policy for TightBudget {
    const POLICY: PolicyParameters = PolicyParameters {
        check_overflow: true,
        unroll_loops: true,
        precision: PrecisionPolicy::Best,
        avoid_branching: false,
        max_iterations: 12,
        use_compensation: false,
        denormal_behavior: DenormalBehavior::FlushToZero,
    };
}

/// The order-N smoothstep evaluated in f64 scalar arithmetic, from the same
/// coefficients Thermite uses, so the round trip is checked against an independent
/// evaluation rather than against the vector kernel's own forward pass.
fn smoothstep_ref(x: f64, n: usize) -> f64 {
    // S_N(x) = x^N * sum_{k<N} C(N+k-1, k) C(2N-1, N-1-k) (-x)^k
    fn comb(a: usize, b: usize) -> f64 {
        let mut r = 1.0;
        for i in 0..b {
            r = r * (a - i) as f64 / (i + 1) as f64;
        }
        r
    }

    let mut sum = 0.0;
    for k in 0..n {
        sum += comb(n + k - 1, k) * comb(2 * n - 1, n - 1 - k) * (-x).powi(k as i32);
    }

    x.powi(n as i32) * sum
}

/// Probes across the interior. The endpoints are excluded because `smoothstep` has zero
/// derivative there, so its inverse has an infinite one and a relative error is amplified
/// by ~1e7 at y = 1e-6, a property of the function rather than of the solver.
const PROBES: &[f64] = &[
    0.02, 0.05, 0.1, 0.2, 0.3, 0.4, 0.45, 0.5, 0.55, 0.6, 0.7, 0.8, 0.9, 0.95, 0.98,
];

macro_rules! check_order {
    ($name:ident, $n:literal) => {
        #[test]
        fn $name() {
            for &y in PROBES {
                let x = D::splat(y)
                    .inverse_smoothstep_p::<TightBudget, $n>(None)
                    .extract::<0>();

                assert!(
                    (0.0..=1.0).contains(&x),
                    "n={}: inverse_smoothstep({y}) = {x}, outside [0, 1]",
                    $n
                );

                // Round trip. The forward map is well conditioned in the interior, so an
                // unconverged root shows up here directly.
                let back = smoothstep_ref(x, $n);

                assert!(
                    (back - y).abs() <= 1e-8,
                    "n={}: inverse_smoothstep({y}) = {x} round-trips to {back} \
                     (off by {:.3e}) in {} iterations. A bisecting solver cannot resolve \
                     an f64 this quickly - check the derivative returned to newtons_method.",
                    $n,
                    (back - y).abs(),
                    TightBudget::POLICY.max_iterations
                );
            }
        }
    };
}

check_order!(inverse_smoothstep_n3_converges_in_12_iterations, 3);
check_order!(inverse_smoothstep_n4_converges_in_12_iterations, 4);
check_order!(inverse_smoothstep_n5_converges_in_12_iterations, 5);
check_order!(inverse_smoothstep_n8_converges_in_12_iterations, 8);

/// The symmetry `S(1 - x) = 1 - S(x)` makes the inverse antisymmetric about (0.5, 0.5).
/// Independent of the solver's speed, and a cheap check that the bracket handling is not
/// biased toward one side.
#[test]
fn inverse_smoothstep_is_antisymmetric() {
    for &y in PROBES {
        let a = D::splat(y).inverse_smoothstep_p::<TightBudget, 3>(None).extract::<0>();
        let b = D::splat(1.0 - y)
            .inverse_smoothstep_p::<TightBudget, 3>(None)
            .extract::<0>();

        assert!(
            (a + b - 1.0).abs() <= 1e-8,
            "inverse_smoothstep({y}) + inverse_smoothstep({}) = {}, want 1",
            1.0 - y,
            a + b
        );
    }
}
