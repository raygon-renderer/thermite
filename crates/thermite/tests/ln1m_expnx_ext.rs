//! `ln1m_expnx_ext(x, ln x)` is a HINT, so it must not change the answer.
//!
//! The `_ext` form exists only so a caller who already has `ln(x)` need not recompute it.
//! That makes the contract easy to state and easy to test without a reference table at
//! all: **at any tier, `ln1m_expnx_ext(x, ln x)` must agree with `ln1m_expnx(x)`**.
//! Nothing about supplying a value the function would have computed itself can
//! legitimately move the result.
//!
//! It did move the result. The float32 `_ext` had no implementation above `Medium` and
//! fell through to the naive `ln(1 - exp(-x))`, which cancels against the FLOAT GRID
//! rather than against itself: near 1 the float32 spacing is 2^-24 = 5.96e-08, so
//! `1 - exp(-x)` for x = 2.98e-08 snaps to exactly 2^-24, which is `2x`, and the answer
//! came back `ln(2x)`, too large by `ln 2`.
//!
//! That bug is why this file tests an INVARIANT between two functions rather than
//! comparing either one to a table. It was an inversion, with the cheap `Worst` and
//! `Medium` minimax form correct and `Performance` upward wrong, so grading tiers against
//! `precision` reports the RIGHT ones as broken. And `Reference` shared it, having no
//! `is_reference` arm of its own, so a `precision`-vs-`reference` cross-check passed too.
#![cfg(any(
    target_arch = "x86",
    target_arch = "x86_64",
    target_arch = "wasm32",
    target_arch = "aarch64"
))]

mod harness;

use thermite::Vector;
use thermite::math::TranscendentalMathWithPolicy;
use thermite::math::policy::DefaultPolicy;
use thermite::math::policy::policies::{HighPerformance, Performance, Precision, Reference, Size, UltraPerformance};
use thermite::prelude::*;
use thermite::simd::Simd;

macro_rules! ctx {
    () => {
        #[allow(dead_code)]
        type D = Vector<<S as Simd>::f64x4>;
        #[allow(dead_code)]
        type F = Vector<<S as Simd>::f32x8>;

        /// Spread across the two branches of the underlying kernel (it splits at `ln 2`) and well
        /// past the point where the naive form used to collapse.
        #[allow(dead_code)]
        const PROBES: &[f64] = &[
            2.98123126185601e-08,
            1e-7,
            1e-6,
            1e-4,
            1e-2,
            0.25,
            0.6931471805599453, // ln 2, the branch point itself
            1.0,
            5.0,
            10.0,
        ];

        /// Tiers whose own `ln` is accurate, so the hint is redundant and must be inert.
        ///
        /// The two fast tiers are excluded from the STRICT invariant for a reason:
        /// `ln1m_expnx` at `<= Medium` calls `_ext` with `x.ln_p::<P>()`, its OWN logarithm, and
        /// at `Worst` that carries ~0.04 absolute error. A test that hands `_ext` an exact `ln(x)`
        /// is therefore giving it a better hint than the function gives itself, and the two
        /// legitimately disagree, by 2.3e-03 relative at x = 2.98e-08, with `_ext` the more
        /// accurate of the two. **The accuracy of `_ext` is bounded by the caller's `ln(x)`**,
        /// which is the whole point of the argument existing.
        macro_rules! for_each_accurate_tier {
            (|$p:ident| $body:block) => {{
                {
                    type $p = Performance;
                    $body
                }
                {
                    type $p = Size;
                    $body
                }
                {
                    type $p = DefaultPolicy;
                    $body
                }
                {
                    type $p = Precision;
                    $body
                }
                {
                    type $p = Reference;
                    $body
                }
            }};
        }

        /// The two fast tiers, where the hint may improve on what the function computes itself.
        macro_rules! for_each_fast_tier {
            (|$p:ident| $body:block) => {{
                {
                    type $p = UltraPerformance;
                    $body
                }
                {
                    type $p = HighPerformance;
                    $body
                }
            }};
        }
    };
}

for_each_backend_concrete! {

fn the_hint_does_not_change_the_answer_f32() {
    ctx!();
    for_each_accurate_tier!(|P| {
        for &x in PROBES {
            let xv = F::splat(x as f32);
            let lnx = F::splat((x as f32).ln());

            let plain = xv.ln1m_expnx_p::<P>().extract::<0>() as f64;
            let ext = xv.ln1m_expnx_ext_p::<P>(lnx).extract::<0>() as f64;

            // Generous, because the two forms are allowed to differ in the last bits at a
            // given tier, and are not required to be the same instructions. The bug this
            // guards was off by `ln 2`, a factor of 8e6 past this bound.
            let rel = ((ext - plain) / plain).abs();

            assert!(
                rel <= 1e-4,
                "f32 x={x:e}: ln1m_expnx = {plain:e} but ln1m_expnx_ext = {ext:e} \
                 (rel {rel:e}). Supplying ln(x) must not change the answer."
            );
        }
    });
}

fn the_hint_does_not_change_the_answer_f64() {
    ctx!();
    for_each_accurate_tier!(|P| {
        for &x in PROBES {
            let xv = D::splat(x);
            let lnx = D::splat(x.ln());

            let plain = xv.ln1m_expnx_p::<P>().extract::<0>();
            let ext = xv.ln1m_expnx_ext_p::<P>(lnx).extract::<0>();

            let rel = ((ext - plain) / plain).abs();

            assert!(
                rel <= 1e-4,
                "f64 x={x:e}: ln1m_expnx = {plain:e} but ln1m_expnx_ext = {ext:e} (rel {rel:e})"
            );
        }
    });
}

/// The accurate tiers must actually be accurate in the small-x tail, where
/// `ln(1 - e^-x) -> ln(x)`. Separate from the invariant above, because both forms agreeing
/// on a wrong answer would satisfy that one.
fn small_x_tail_is_accurate_at_average_and_above() {
    ctx!();
    for &x in &[2.98123126185601e-08f64, 1e-7, 1e-6, 1e-4] {
        let want = (-((-x).exp_m1())).ln();

        macro_rules! check {
            ($p:ty, $name:literal) => {{
                let got32 = F::splat(x as f32)
                    .ln1m_expnx_ext_p::<$p>(F::splat((x as f32).ln()))
                    .extract::<0>() as f64;
                let rel = ((got32 - want) / want).abs();
                assert!(
                    rel <= 1e-5,
                    "{} f32 x={x:e}: got {got32:e}, want {want:e} (rel {rel:e})",
                    $name
                );
            }};
        }

        check!(Performance, "Performance");
        check!(Size, "Size");
        check!(DefaultPolicy, "DefaultPolicy");
        check!(Precision, "Precision");
        check!(Reference, "Reference");
    }
}

/// At `Worst` and `Medium` the hint may legitimately beat the function's own `ln(x)`, so
/// the two forms are only required to stay in the same neighbourhood and, more to the
/// point, on the same SIDE of zero. `ln(1 - e^-x)` is negative throughout its domain.
fn the_fast_tiers_stay_in_range_f32() {
    ctx!();
    for_each_fast_tier!(|P| {
        for &x in PROBES {
            let xv = F::splat(x as f32);
            let ext = xv.ln1m_expnx_ext_p::<P>(F::splat((x as f32).ln())).extract::<0>() as f64;
            let plain = xv.ln1m_expnx_p::<P>().extract::<0>() as f64;

            assert!(
                ext <= 0.0 && ext.is_finite(),
                "f32 x={x:e}: ln1m_expnx_ext = {ext:e}, must be finite and non-positive"
            );

            // ABSOLUTE, not relative, and that is the honest bound. The gap between the
            // two forms is exactly the error in the `ln(x)` the plain form feeds itself,
            // and the `Worst` log's budget is ~0.0397 ABSOLUTE, being a linear fit in the
            // bit pattern, so its error does not shrink with the result. Measured at
            // x = 1, where ln(x) is 0 and the fast log returns ~0.04: the two forms sit
            // 0.0397 apart, which is 8.7% of a result of -0.46 and would fail any
            // relative bound worth setting.
            let gap = (ext - plain).abs();
            assert!(
                gap <= 0.05,
                "f32 x={x:e}: the two forms are {gap:e} apart, more than the fast log's                  own ~0.04 absolute budget explains ({plain:e} vs {ext:e})"
            );
        }
    });
}

}
