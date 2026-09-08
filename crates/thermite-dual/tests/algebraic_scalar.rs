//! The Poisson exponent's error-free transformations survive `algebraic-scalar`.
//!
//! This is not really about duals. It tests `thermite-special`'s Poisson kernel on a plain
//! `Vector<f64>`, and lives here because `thermite-special`'s test suite dev-depends on
//! `thermite-compensated`, which at the time refused the feature at compile time. Now that
//! it no longer does, this file could move to `thermite-special/tests`.
//!
//! `generic::poisson` used to spell two error-free transformations inline with `+`/`-`,
//! which the feature folds to zero. Both now route through
//! `SpecializedSpecialMath::exp_two_sum`, whose `ps`/`pd` overrides use the strict
//! `FloatVectorWithBits::two_sum`. Run both ways:
//!
//! ```text
//! cargo nextest run -p thermite-dual --test algebraic_scalar
//! cargo nextest run -p thermite-dual --features thermite/algebraic-scalar --test algebraic_scalar
//! ```
//!
//! The assertions target the transformations directly rather than end-to-end
//! `poisson_pmf` accuracy, because `exp` itself drifts under the feature and comparing
//! `poisson_pmf(0, lambda)` to `exp(-lambda)` reports over 100 ulp for unrelated reasons.
//! A `k >= 9` control with no two-word work measured 0.95 ulp in both builds.

use thermite::math::policy::policies::Precision;
use thermite::prelude::*;
use thermite_special::specialized::SpecializedSpecialMath;
use thermite_special::specialized::generic::poisson::pmf_parts;

/// The 1-lane scalar backend, the only one `algebraic-scalar` touches.
type V = Vector<f64>;

/// The hook in isolation. `1 + 2^-60` rounds to `1`, so the residual is exactly `2^-60`.
#[test]
fn the_hook_returns_an_exact_residual() {
    let (s, e) = <V as SpecializedSpecialMath<f64>>::exp_two_sum(V::splat(1.0), V::splat(2f64.powi(-60)));

    assert_eq!(s.extract::<0>(), 1.0);
    assert_eq!(
        e.extract::<0>(),
        2f64.powi(-60),
        "residual folded to zero - exp_two_sum is not routed through the strict two_sum"
    );
}

/// The same at the Poisson exponent's magnitudes: `-lambda` against a `rest` of size ~10.
#[test]
fn the_hook_is_exact_at_exponent_scale() {
    let base = -301.875f64;
    let rest = -1.0784276658208688e1f64;

    let (s, e) = <V as SpecializedSpecialMath<f64>>::exp_two_sum(V::splat(base), V::splat(rest));
    let (s, e) = (s.extract::<0>(), e.extract::<0>());

    // Plain `f64` arithmetic, which the feature does not touch.
    assert_eq!(s, base + rest);
    assert_ne!(e, 0.0, "residual folded to zero at exponent scale");
    assert_eq!(e, -5.329070518200751e-15, "residual is not the one strict arithmetic gives");
}

/// `pmf_parts` still emits a second word for the exponent. `k = 0` takes the
/// shifted-Stirling path, so both two-word steps go through the hook.
#[test]
fn pmf_parts_keeps_its_second_word() {
    let (hi, lo, _large, prod, n) = pmf_parts::<Precision, f64, V, false>(V::splat(0.0), V::splat(11.0625));
    let (hi, lo) = (hi.extract::<0>(), lo.extract::<0>());

    assert_eq!(prod.extract::<0>(), 362880.0, "9! from the Stirling shift");
    assert_eq!(n.extract::<0>(), 9.0);
    assert_eq!(hi, -1.0784276658208688e1, "the exponent's high word moved");

    // Strict gives 5.711827350024695e-16, `algebraic-scalar` 5.707240235963695e-16. Assert
    // the magnitude, not the bits.
    assert!(
        (lo - 5.71e-16).abs() < 1e-17,
        "exponent second word is {lo:e}, expected ~5.71e-16 - a value near zero means the \
         binade split's TwoSum was folded away"
    );
}

/// Cody-Waite range reductions keep their guarantee.
///
/// `exp` reduces as `x - r*ln2_hi - r*ln2_lo`. A reassociable multiply lets the optimizer
/// factor that into `r * (hi + lo)`, rounding the split constant back to one word, with
/// error scaling as `r ~ x / ln2`. Before the fix: 1.95 / 76.4 / 43.9 ulp at
/// `x = -11 / -302 / -511`. Every two-word reduction in `ps.rs`/`pd.rs` goes through the
/// scalar backend's `_e` overrides, so this one probe covers the family.
#[test]
fn cody_waite_reductions_keep_their_guarantee() {
    use thermite::math::TranscendentalMathWithPolicy;

    // Large |x|: the error scales with the reduction multiplier.
    for x in [-11.0625f64, -21.8, -100.5, -301.875, -312.659276658208682, -511.0625] {
        let got = V::splat(x).exp_p::<Precision>().extract::<0>();
        let want = x.exp();
        let ulp = (got - want).abs() / (want * f64::EPSILON);
        assert!(
            ulp <= 2.0,
            "exp({x}) is {ulp:.2} ulp. Tens of ulp, growing with |x|, means the Cody-Waite \
             reduction was reassociated - check that the scalar backend still overrides \
             mul_adde/mul_sube/nmul_adde/nmul_sube with a strict multiply."
        );
    }

    // `ln` recombines a two-word ln2 the same way, from the other direction.
    for x in [1e-300f64, 1e-8, 0.5, 3.25, 1e8, 1e300] {
        let got = V::splat(x).ln_p::<Precision>().extract::<0>();
        let want = x.ln();
        let ulp = (got - want).abs() / (want.abs() * f64::EPSILON);
        assert!(ulp <= 4.0, "ln({x}) is {ulp:.2} ulp");
    }
}

/// The trig `pi/2` reduction, three words (four in f32) subtracted in order. The sharper
/// probe: `exp` has two terms and LLVM happened not to re-bracket them once the multiply
/// was strict, but at three or four it does. At `x = 1e5` before the fix:
///
/// | scalar `nmul_adde` | sin |
/// |---|---|
/// | algebraic multiply, algebraic accumulate | 122705 ulp |
/// | strict multiply, algebraic accumulate | 122705 ulp |
/// | strict accumulate (either multiply) | 0.00 ulp |
///
/// The accumulate is what matters. `y * dp1` is exact either way.
#[test]
fn trig_reduction_keeps_its_guarantee() {
    use thermite::math::TranscendentalMathWithPolicy;

    // Below the Payne-Hanek threshold: above it a different path hides a broken
    // Cody-Waite (`1e8` measures 0.00 ulp even when `1e5` is off by 122705).
    for x in [0.7f64, 3.9, 100.5, 1000.25, 100000.125] {
        let (s, c) = V::splat(x).sin_cos_p::<Precision>();
        let (gs, gc) = (s.extract::<0>(), c.extract::<0>());

        for (got, want, name) in [(gs, x.sin(), "sin"), (gc, x.cos(), "cos")] {
            let ulp = (got - want).abs() / (want.abs() * f64::EPSILON);
            assert!(
                ulp <= 2.0,
                "{name}({x}) is {ulp:.2} ulp. Thousands, growing with x, means the pi/2 \
                 reduction was re-bracketed - check that the scalar backend's `_e` FMA \
                 overrides still delegate to the element layer, and that the reduction in \
                 ps.rs/pd.rs still goes through `nmul_adde` rather than bare operators."
            );
        }
    }
}
