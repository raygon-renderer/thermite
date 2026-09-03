//! The Pochhammer symbol `$(z)_m = \Gamma(z+m)/\Gamma(z)$`.
//!
//! # Three paths, and why the obvious one is not enough
//!
//! Written out, this is a ratio of two Gamma functions, and the obvious spelling
//! `exp(lgamma(z+m) - lgamma(z))` is a disaster in exactly the region the function is most
//! used. The two logarithms are large and nearly equal whenever `m` is small next to `z`,
//! so the subtraction sheds the digits that carry the answer: measured against mpmath at
//! `z = 1e8, m = 1e-4`, that form has **no correct digits at all** (2.8e-7 relative, where
//! the true value is within 1e-3 of 1). Everything below exists to avoid forming that
//! difference.
//!
//! **Integer `m`, small (`|m| <= `[`PRODUCT_CAP`]).** The definition collapses to a plain
//! product `z(z+1)...(z+m-1)`, which forms no logarithm at all and is therefore exact to
//! within its own multiplications: measured worst 4.1 ulp across the sweep, and 0 to 0.2
//! ulp on most of it. This is the dominant case, not a fast path bolted on. Hypergeometric
//! series, binomial-style coefficients and Taylor coefficients of special functions all
//! advance `m` by whole numbers. A negative integer `m` is the reciprocal
//! of the same product started at `z + m`, which is why the sign of `m` only chooses a
//! starting point and a final reciprocal. The product is also indifferent to the sign of
//! `z`, so this path covers the negative half of the domain for free, poles included:
//! `(-2)_3` contains a zero factor and correctly returns 0.
//!
//! It runs at `Average` and above, and within that is gated at runtime on any lane wanting
//! it, returning early when every lane does. That is the shape [`gamma`](super::gamma)'s
//! exact-integer branch uses, one tier lower. The tier differs because the trade does: for
//! `gamma` the Lanczos path is fast and runs regardless, so its integer branch is pure
//! accuracy spend, while here the product is _also the cheaper route_ for integer-heavy data
//! (a handful of multiplies and an early return, against the full Stirling evaluation). With
//! the runtime guard skipping it outright when no lane wants it, the only shape that pays for
//! having it is a genuinely mixed vector.
//!
//! Below `Average` it is compiled out and integer `m` goes through the Stirling difference
//! like anything else. Measured over 231 points with `m` in `0..20` and `z` across eleven
//! magnitudes, that is 4.2 ulp median and 172 worst, against 0.00 median and 4.2 worst for
//! the product. The visible difference is the exactness rather than the ulp count:
//! `(3)_1` is `3.0` on the product path and `3.0000000000000018` without it.
//!
//! **Everything else with both arguments positive.** Take the Stirling difference instead of
//! the logarithm difference. With
//! `$\ln\Gamma(x) = (x - \tfrac12)\ln x - x + \tfrac12\ln 2\pi + \mathrm{stirlerr}(x)$`,
//! the `$\tfrac12 \ln 2\pi$` cancels exactly and the rest regroups so that nothing large is
//! ever subtracted from anything large:
//!
//! ```math
//! \ln\frac{\Gamma(x+m)}{\Gamma(x)}
//!   = \left(x - \tfrac12\right)\ln\!\left(1 + \frac{m}{x}\right)
//!   + m\left(\ln(x+m) - 1\right)
//!   + \mathrm{stirlerr}(x+m) - \mathrm{stirlerr}(x)
//! ```
//!
//! Every term is `O(m)` as `m -> 0`, which is what makes the small-`m` region well behaved.
//! The `log1p` is doing the work the naive subtraction failed at. [`stirlerr`] is only valid
//! at or above [`STIRLERR_MIN`], so an argument below it is first walked up by a whole number
//! of steps and the exact product of those steps divided back out, the same shifted-product
//! trick [`pmf_parts`](super::poisson::pmf_parts) uses, and for the same reason.
//!
//! **Each argument is shifted independently**, which is what keeps every intermediate in
//! range: a product is built only for an argument that is _below_ 9, so its factors are under
//! 18 and it can never exceed `18^9`. Shifting both by a shared amount instead (the obvious
//! spelling) walks an argument that was already fine and overflows it, and the resulting
//! `inf * 0` is a NaN sitting exactly where the answer is a perfectly good infinity. The
//! independent shift is also the more accurate of the two, because the product it skips is a
//! string of roundings that never happens. See the comments on the shift for the numbers.
//!
//! Its accuracy is the floor of anything that exponentiates a logarithm: the relative error
//! of the result is the _absolute_ error of the exponent, so it tracks
//! `$|\ln (z)_m| \cdot \epsilon$` and is bounded below by nothing else. Measured against
//! mpmath over 6924 points with `z` in `[0.1, 8.9]` and non-integer `m`, the median is 2.6
//! ulp, the 99th percentile 25 ulp and the worst 51 ulp. Individual points scale with the
//! result's own logarithm, reaching 259 ulp at `z = 3.7, m = 100` where the value is near
//! `1e163` and `|ln| = 375`. It falls to **zero** error where the answer approaches 1, which
//! is precisely where the naive form was worst.
//!
//! **The residue.** A non-integer `m` (or one past the cap) with `z` or `z + m` non-positive
//! reaches neither path above, and falls back to the logarithmic form with the sign taken
//! from [`lgamma_r`](crate::RealSpecialMath::lgamma_r). It inherits that form's
//! cancellation. This is the region where `(z)_m` is a ratio across Gamma's poles and no
//! cheap rearrangement is available. It is documented rather than fixed.

use thermite::{
    const_splat,
    element::FloatElement,
    mask::GenericMask,
    math::{
        CoreMathWithPolicy as _, TranscendentalMathWithPolicy as _,
        policy::{Policy, PrecisionPolicy},
        specialized::SpecializedTranscendentalMath,
    },
    prelude::*,
};

use crate::specialized::SpecializedRealSpecialMath;

use super::poisson::{STIRLERR_MIN, stirlerr};

/// Largest `|m|` taken by the exact product path.
///
/// The product costs one multiply per step and buys two orders of magnitude of accuracy over
/// the logarithmic route, so the cap is about where the multiplications stop being free
/// rather than about where they stop being better. Integer `m` past this falls to the
/// Stirling difference, which is continuous with it.
pub const PRODUCT_CAP: usize = 20;

/// `$(z)_m = \Gamma(z+m)/\Gamma(z)$`, the Pochhammer symbol, for real `z` and real `m`.
#[inline(always)]
pub fn pochhammer<P, E, V>(z: V, m: V) -> V
where
    E: FloatElement,
    V: FloatVector<Element = E> + SpecializedTranscendentalMath<E> + SpecializedRealSpecialMath<E>,
    P: Policy,
{
    let zm = z + m;

    // --- Path A: integer m within the cap, the case nearly every caller is in. -----------
    //
    // For m >= 0 the product runs up from z. For m < 0 it runs up from z + m and is
    // reciprocated, since (z)_{-n} = 1 / (z-n)_n.
    //
    // Compiled out below `Average`. The product is the _accurate_ route rather than the fast
    // one. The Stirling difference below is already inside the tolerance those tiers ask
    // for, so at that point this is a second path earning nothing, and on a mixed vector it
    // is paid for on every lane.
    let mut use_product = GenericMask::FALSY;
    let mut by_product = V::ONE;

    if const { P::POLICY.precision.ge(PrecisionPolicy::Average) } {
        let n = m.abs();
        let negative_m = m.cmp_lt(V::ZERO);
        use_product = m.cmp_eq(m.round()) & n.cmp_le(const_splat!(int <E>: 20));

        if const { P::POLICY.avoid_branching } || thermite::unlikely(use_product.any()) {
            let base = negative_m.select(zm, z);
            let mut product = V::ONE;
            let mut step = V::ZERO;
            let mut i = 0;
            while i < PRODUCT_CAP {
                V::_loop_hint();

                // Masked by `use_product`, not just by `|m|`: a lane with a large _non-integer_
                // m would otherwise keep the loop alive for a product it never reads. Same
                // reason `gamma`'s integer branch masks its own condition by `is_int`.
                let active = use_product & step.cmp_lt(n);
                if const { !P::POLICY.avoid_branching } && !active.any() {
                    break;
                }

                // Lanes past their own m multiply by one, so a single trip count serves every lane.
                product *= active.select(base + step, V::ONE);
                step += V::ONE;
                i += 1;
            }
            by_product = negative_m.select(product.approx_reciprocal_p::<P>(), product);

            if const { !P::POLICY.avoid_branching } && use_product.all() {
                return by_product;
            }
        }
    }

    // --- Path B: both arguments positive, any real m. ------------------------------------
    //
    // Walk each argument up to STIRLERR_MIN with a whole number of unit steps and divide the
    // exact product of those steps back out.
    //
    // **Each argument gets its own shift.** Walking both by the shared `9 - min(z, z+m)` is
    // the obvious spelling and is wrong twice over: it walks an argument that is already
    // above 9, and `(z+m)^9` then overflows for `z + m` past ~1e34, leaving `inf * 0`, a NaN
    // where the answer is a perfectly good infinity. Shifting each only as far as it needs
    // means a product is built _only_ for an argument below 9, so its factors are under 18
    // and it is bounded by `18^9`, about 2e11 (nowhere near overflow, for any input at all).
    // Everything stays in range, so nothing has to be repaired afterwards.
    //
    // It is also more accurate, because the skipped product is a string of roundings that
    // never happens. Measured against mpmath over 6924 points with `z` in `[0.1, 8.9]`, the
    // median goes 3.94 -> 2.57 ulp and the 99th percentile 31.9 -> 24.9, with individual
    // large-`m` points improving much more (118 -> 10.9 ulp at `z = 0.5, m = 50.5`).
    let nine: V = const_splat!(int <E>: 9);
    let s_z = (nine - z).ceil().max(V::ZERO);
    let s_zm = (nine - zm).ceil().max(V::ZERO);

    // The one place the shifts must agree: when _both_ arguments are below 9, an unequal pair
    // makes the shifted difference `m + (s_zm - s_z)` instead of `m`, and everything staying
    // `O(m)` as `m -> 0` is the whole point of this path. Measured, letting them differ there
    // costs the small-`m` region an order of magnitude (0.48 -> 9.52 ulp at `z = 3, m = 1e-6`).
    let both_shifted = s_z.cmp_gt(V::ZERO) & s_zm.cmp_gt(V::ZERO);
    let common = s_z.max(s_zm);
    let s_z = both_shifted.select(common, s_z);
    let s_zm = both_shifted.select(common, s_zm);

    // One loop, two masked products: each argument multiplies only on the steps it needs, so
    // a lane whose `z + m` is already past 9 builds nothing for it.
    let mut num = V::ONE;
    let mut den = V::ONE;
    let mut step = V::ZERO;
    let mut i = 0;
    while i < STIRLERR_MIN as usize {
        V::_loop_hint();

        let want_num = step.cmp_lt(s_z);
        let want_den = step.cmp_lt(s_zm);
        if const { !P::POLICY.avoid_branching } && !(want_num | want_den).any() {
            break;
        }

        num *= want_num.select(z + step, V::ONE);
        den *= want_den.select(zm + step, V::ONE);
        step += V::ONE;
        i += 1;
    }

    let y = z + s_z;
    let x = zm + s_zm;

    // The shifted difference. `s_zm - s_z` is an exact small integer, and is exactly zero
    // wherever both arguments were shifted, which keeps this equal to `m` in the
    // small-`m` region rather than recovering it from `x - y` and cancelling it away.
    let mm = m + (s_zm - s_z);

    // (y - 1/2) ln(1 + mm/y) + mm (ln x - 1) + stirlerr(x) - stirlerr(y).
    // Nothing large is subtracted from anything large, and every term vanishes with mm.
    let ln_ratio = (y - V::HALF).mul_adde(
        mm.approx_div_p::<P>(y).ln_1p_p::<P>(),
        mm.mul_adde(x.ln_p::<P>() - V::ONE, stirlerr::<P, E, V>(x) - stirlerr::<P, E, V>(y)),
    );

    // Both products are bounded by 18^9, so `exp` saturating to infinity or zero is already
    // the right answer and there is nothing to guard.
    let by_stirling = ln_ratio.exp_p::<P>() * num.approx_div_p::<P>(den);

    let positive = z.cmp_gt(V::ZERO) & zm.cmp_gt(V::ZERO);
    let mut result = use_product.select(by_product, by_stirling);

    // --- Path C: the residue, where neither of the above applies. ------------------------
    let residue = !(use_product | positive);
    if const { P::POLICY.avoid_branching } || thermite::unlikely(residue.any()) {
        // The ORIGINAL arguments, not the shifted ones: the shift above is only valid where
        // `stirlerr` is, which is the case this branch exists to escape.
        let (lg_num, sign_num) = <V as SpecializedRealSpecialMath<E>>::lgamma_r::<P>(zm);
        let (lg_den, sign_den) = <V as SpecializedRealSpecialMath<E>>::lgamma_r::<P>(z);
        let by_log = (lg_num - lg_den).exp_p::<P>() * (sign_num * sign_den);
        result = residue.select(by_log, result);
    }

    result
}
