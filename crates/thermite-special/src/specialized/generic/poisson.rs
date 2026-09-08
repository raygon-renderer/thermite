//! Loader's saddle-point pieces for densities of the shape `$x^k e^{-x}/\Gamma(k+1)$`.
//!
//! The Poisson mass `$e^{-\lambda}\lambda^k/k!$`, the Gamma density, and the seed of the
//! orthonormal Laguerre functions are all this shape, and the obvious spelling
//! `exp(k ln lambda - lgamma(k+1) - lambda)` computes an `O(1)` answer as the exponential
//! of a difference of large terms: half an ulp of `lgamma(k+1) = O(k ln k)` becomes that
//! many ulp of the result. Loader (2000, "Fast and accurate computation of binomial
//! probabilities", the form R's `dpois` uses) rewrites it as
//!
//! ```math
//! \frac{\lambda^k e^{-\lambda}}{k!} = \frac{e^{-\mathrm{stirlerr}(k) - \mathrm{bd0}(k, \lambda)}}{\sqrt{2\pi k}}
//! ```
//!
//! with the two pieces below, both *small* where the density is not negligible, so the
//! exponential amplifies nothing.

use thermite::{
    element::FloatElement,
    math::{
        CoreMathWithPolicy as _, TranscendentalMathWithPolicy as _,
        policy::{Policy, PrecisionPolicy},
    },
    prelude::*,
};

use crate::specialized::SpecializedSpecialMath;


/// Below this the Stirling series in [`stirlerr`] is not accurate to binary64 at any
/// depth (it is asymptotic, and the smallest term at `n = 9` is under `1e-18`, at `n = 6`
/// it is `1e-14`). Callers handle `n < STIRLERR_MIN` some other way: a table for integers,
/// or `Gamma(n+1)` directly, which is cheap and well conditioned at small argument.
pub const STIRLERR_MIN: thermite::LargeInt = 9;

/// Terms of the `1/n^2` series in [`stirlerr`] by tier, at `n >= STIRLERR_MIN`. The
/// dropped term bounds the *absolute* error, which is the relative error of whatever
/// density it feeds: 9 terms is `1e-18`, 6 is `2.5e-15` (a measured 5 ulp at `n = 9`, so
/// `Average` keeps all 9, the three FMAs being nothing), 5 is `1e-13`, and the single
/// `1/(12n)` term is `3.8e-6`. `Worst` forgoes the series and keeps just that.
#[inline(always)]
pub const fn stirlerr_terms(precision: PrecisionPolicy) -> usize {
    match precision {
        PrecisionPolicy::Worst => 1,
        PrecisionPolicy::Medium => 5,
        _ => 9,
    }
}

/// Terms of the odd series in [`bd0`] by tier, inside `|v| < 1/5`.
///
/// The dropped term is `2k v^{2T+1}/(2T+1)`, an *absolute* error in the exponent and so
/// a relative error of `k` times `0.2^{2T+1}/(2T+1)` in the density: 12 terms is
/// `k * 7e-19`, 8 is `k * 1e-13`, 5 is `k * 4e-9`. For comparison the
/// direct form at the window edge is off by about `k * eps / 2` from its own
/// cancellation, so 12 terms matches it in binary64 and 5 in binary32.
#[inline(always)]
pub const fn bd0_terms(precision: PrecisionPolicy) -> usize {
    match precision {
        PrecisionPolicy::Worst => 5,
        PrecisionPolicy::Medium => 8,
        _ => 12,
    }
}

/// Stirling's error `$\mathrm{stirlerr}(n) = \ln n! - \left[(n + \tfrac12)\ln n - n + \tfrac12 \ln 2\pi\right]$`,
/// for `n >= STIRLERR_MIN`, by the Bernoulli series
///
/// ```math
/// \frac{1}{12n} - \frac{1}{360n^3} + \frac{1}{1260n^5} - \frac{1}{1680n^7} + \frac{1}{1188n^9} - \dots
/// ```
///
/// with [`stirlerr_terms`] terms. Cost is a reciprocal and a short Horner ladder. Below
/// `STIRLERR_MIN` the series does not converge to double precision (see there); this
/// function does not check, it just returns the truncated series.
#[inline(always)]
pub fn stirlerr<P, E, V>(n: V) -> V
where
    P: Policy,
    E: FloatElement,
    V: FloatVector<Element = E> + SpecializedSpecialMath<E>,
{
    // B_{2k} / (2k (2k-1)), k = 1..=9.
    let s: [E; 9] = [
        <E as FloatElement>::ConstRatio::<1, 12>::VALUE,
        <E as FloatElement>::ConstRatio::<1, 360>::VALUE,
        <E as FloatElement>::ConstRatio::<1, 1260>::VALUE,
        <E as FloatElement>::ConstRatio::<1, 1680>::VALUE,
        <E as FloatElement>::ConstRatio::<1, 1188>::VALUE,
        <E as FloatElement>::ConstRatio::<691, 360360>::VALUE,
        <E as FloatElement>::ConstRatio::<1, 156>::VALUE,
        <E as FloatElement>::ConstRatio::<3617, 122400>::VALUE,
        <E as FloatElement>::ConstRatio::<43867, 244188>::VALUE,
    ];
    let terms = const { stirlerr_terms(P::POLICY.precision) };

    let rn = n.approx_reciprocal_p::<P>();
    let rnn = rn * rn;

    // p = S_0 - rnn (S_1 - rnn (S_2 - ...)), then p / n.
    let mut p = V::splat(s[terms - 1]);
    let mut k = terms - 1;
    while k > 0 {
        k -= 1;
        p = rnn.nmul_adde(p, V::splat(s[k]));
    }

    p * rn
}

/// The binomial/Poisson deviance `bd0(k, lambda) = k ln(k/lambda) + lambda - k >= 0` in its
/// peak form: `(k - lambda) v + 2k sum_{j>=1} v^{2j+1}/(2j+1)` for `v = (k - lambda)/(k + lambda)`
/// (from `ln((1+v)/(1-v)) = 2 atanh v`), [`bd0_terms`] terms, full precision inside
/// `|v| < 1/5`. Away from the peak the direct form is fine and [`pmf_parts`] uses it
/// (folded with the rest of the exponent); this is only the part that needs care.
/// `diff` and `v` are passed in because callers have them.
#[inline(always)]
pub fn bd0_series<P, E, V>(k: V, diff: V, v: V) -> V
where
    P: Policy,
    E: FloatElement,
    V: FloatVector<Element = E> + SpecializedSpecialMath<E>,
{
    let terms = const { bd0_terms(P::POLICY.precision) };
    let vv = v * v;

    // sum_{j=1..T} vv^{j-1} / (2j+1), Horner.
    let mut s = V::splat(E::from_ratio(1, 2 * terms as thermite::LargeInt + 1));
    let mut j = terms;
    while j > 1 {
        j -= 1;
        s = vv.mul_adde(s, V::splat(E::from_ratio(1, 2 * j as thermite::LargeInt + 1)));
    }
    diff.mul_adde(v, ((k + k) * (vv * v)) * s)
}

/// Shift a real `k >= 0` up into the Stirling region: `n = k + m` with integer
/// `m = STIRLERR_MIN - floor(k)` (so `n` is in `[9, 10)`), and `prod = (k+1)(k+2)...(k+m)`,
/// so that `Gamma(k+1) = Gamma(n+1) / prod`. Lanes already at or past `STIRLERR_MIN` get
/// `m = 0`, `n = k`, `prod = 1`.
///
/// This is how the small-`k` case shares the large-`k` machinery instead of calling
/// `lgamma`: a masked product of at most 10 factors, then the same `stirlerr(n)`. It is
/// also more accurate than `lgamma` there, since the product's error stays relative and
/// `n` is small enough that `n ln n` is only ~20.
#[inline(always)]
fn shift_to_stirling<P, E, V>(k: V) -> (V, V, V::Mask)
where
    P: Policy,
    E: FloatElement,
    V: FloatVector<Element = E>,
{
    let min = V::splat(<E as FloatElement>::ConstInt::<STIRLERR_MIN>::VALUE);
    let large = k.cmp_ge(min);

    // The usual case (one weight per call, already in range) pays only the compare.
    if const { !P::POLICY.avoid_branching } && large.all() {
        return (k, V::ONE, large);
    }

    // m = 9 - floor(k) for k < 9, else 0. k > -1 so m <= 10.
    let m = large.select(V::ZERO, min - k.floor());
    let n = k + m;

    let mut prod = V::ONE;
    let mut i = 1;
    while i <= 10 {
        let fi = V::splat(E::from_int(i));
        let keep = m.cmp_ge(fi);
        prod = keep.select(prod * (k + fi), prod);
        i += 1;
    }
    (n, prod, large)
}

/// The shared core of every `$x^k e^{-x}/\Gamma(k+1)$` shape here: the Poisson mass, its
/// log, and the Laguerre-function seed. Returns `(rest_hi, rest_lo, large, prod, n)` such
/// that
///
/// ```text
/// P(k; lambda) = exp(rest_hi + rest_lo - [large ? 0 : lambda]) * prod / sqrt(2 pi n)
/// ```
///
/// `rest_lo` is the second word of the exponent, nonzero only on the shifted lanes (see
/// the split below) and zero for `k >= 9` and the peak series. Callers must pass it to
/// [`exp_two_sum`]; dropping it cost `poisson_pmf` 14 ulp.
///
/// where `n`, `prod` are from [`shift_to_stirling`] (`n = k`, `prod = 1` for `k >= 9`) and
/// `rest` is one of three things, per lane, always with `-stirlerr(n)`:
///
/// - `k >= 9` near the peak (`|k - lambda| < 0.2 (k + lambda)`): `-bd0(k, lambda)` as its
///   series, with nothing large in it.
/// - `k >= 9` off the peak: `-bd0` directly as `-(k ln(k/lambda) - (k - lambda))`. The
///   ratio goes into the `ln` whole and `k - lambda` is exact when they are close, so this
///   is a few ulp too, *unlike* `k (ln k - ln lambda)` (40x worse when `k ~ lambda`) or
///   splitting `-lambda` off (`k ln(k/lambda) + k` is then large on its own, and both were
///   measured at 80-150 ulp for `k = 100`, `lambda = 150`).
/// - `k < 9`: `k ln lambda - n ln n + n`, the shifted Stirling form, with `- lambda`
///   **left out** so the caller adds it with a TwoSum, it being the one large term there.
///
/// The last two share one `ln`, of `k / lambda` or `n` by lane. `ln lambda` is only formed
/// if some lane is small. Under `ALL_LARGE` (a caller who knows `k >= 9` everywhere) the
/// shift and `ln lambda` fold away. Uniform vectors skip whichever branch no lane needs.
/// `k = 0` gives `0 * ln 0 = NaN` at `lambda = 0`; callers pin that.
#[inline(always)]
pub fn pmf_parts<P, E, V, const ALL_LARGE: bool>(k: V, lambda: V) -> (V, V, V::Mask, V, V)
where
    P: Policy,
    E: FloatElement,
    V: FloatVector<Element = E> + SpecializedSpecialMath<E>,
{
    let (n, prod, large) = if const { ALL_LARGE } {
        (k, V::ONE, V::Mask::TRUTHY)
    } else {
        shift_to_stirling::<P, E, V>(k)
    };

    let st = stirlerr::<P, E, V>(n);

    // Peak lanes: |v| < 1/5, and only where the shift did nothing (bd0 is about k itself).
    let diff = n - lambda;
    let v = diff / (n + lambda);
    let near = large & v.abs().cmp_lt(V::splat(<E as FloatElement>::ConstRatio::<1, 5>::VALUE));

    if const { !P::POLICY.avoid_branching } && near.all() {
        return (-(st + bd0_series::<P, E, V>(n, diff, v)), V::ZERO, large, prod, n);
    }

    let all_large = const { ALL_LARGE } || (const { !P::POLICY.avoid_branching } && large.all());

    // Large: -(k ln(k/lambda) - diff). Small: k ln lambda - n ln n + n. One ln between them.
    let kl = if all_large {
        V::ZERO
    } else {
        // Dodge `0 * ln 0` on the NaN itself, not on `k = 0`. Selecting on `k.is_zero()`
        // has the same value but discards `d/dk = ln lambda`, which a `Dual` seeded on `k`
        // lost at `k = 0`. Only `0 * inf` (lambda zero or infinite) needs dodging.
        let kl = k * lambda.ln_p::<P>();
        large.select(V::ZERO, kl.is_nan().select(V::ZERO, kl))
    };

    // The shifted lanes never form `n ln n` as one rounded product: `exp` turns absolute
    // error in the exponent into relative error in the density, and `n ln n ~ 19.8`, so
    // one rounding is ~8 ulp (measured 14.4 ulp median on `poisson_pmf(0, lambda)`).
    // Every shifted lane is in one binade, `n = 9 + frac(k)`, so split at its left edge:
    //
    //     n ln n = 9 ln 9 + f ln 9 + n ln(1 + f/9),   f = n - 9  (exact, Sterbenz)
    //
    // Only the leading term is large enough to need two words (`E::NINE_LN_9_HI/LO`).
    // `LN_9` stays one word: it multiplies `f < 1`, and adding its low word measured no
    // change at any `k`. `ln_1p` and not `ln(1 + t)`: forming `1 + t` rounds, and `n`
    // times that is 4.5 ulp.
    let l = large.select(n / lambda, n).ln_p::<P>();

    let (plain, plain_lo) = if all_large {
        // No shifted lane to compensate.
        (n.nmul_adde(l, kl + diff) - st, V::ZERO)
    } else {
        let f = n - V::splat(<E as FloatElement>::ConstInt::<STIRLERR_MIN>::VALUE);
        let t = f * V::splat(<E as FloatElement>::ConstRatio::<1, STIRLERR_MIN>::VALUE);
        let l1 = t.ln_1p_p::<P>();

        let small = f.mul_adde(V::LN_9, n * l1);

        // TwoSum against the constant head: `hi` is the rounded `n ln n`, `lo` its
        // residual. Knuth's form, since `small` can be zero at integer `k`. Through
        // `exp_two_sum` rather than inline so `algebraic-scalar` cannot fold it.
        let (hi, resid) = V::exp_two_sum(V::NINE_LN_9_HI, small);
        let lo = resid + V::NINE_LN_9_LO;

        // Shifted lanes subtract the two-word head. `n - hi` is exact there (both are
        // multiples of `2^-49` and the difference is under 16), so keep this association.
        let a = large.select(n.nmul_adde(l, kl + diff), (n - hi) + kl);

        // Second residual, from the rounding of `a - st`. Dropping it costs 1.06 ulp
        // against 0.33. Through the same hook with `-st` (exact negation); the old
        // inline `((a - p) - st)` was foldable under `algebraic-scalar`.
        let (p, resid_p) = V::exp_two_sum(a, -st);

        (p, large.select(V::ZERO, resid_p - lo))
    };

    if const { !P::POLICY.avoid_branching } && near.none() {
        return (plain, plain_lo, large, prod, n);
    }

    (
        near.select(-(st + bd0_series::<P, E, V>(n, diff, v)), plain),
        near.select(V::ZERO, plain_lo),
        large,
        prod,
        n,
    )
}

/// `exp(base + rest_hi + rest_lo)` where `base` is a large exact-ish number (`-lambda`,
/// `x/4`) and the `rest` pair is small: TwoSum recovers the rounding of the sum, and
/// `e^{s + lo} = e^s (1 + lo)` to first order. Without it the sum rounds to half an ulp of
/// `base`, which the exponential turns into hundreds of ulp.
///
/// `rest_lo` is the caller's own second word, added to the residual this function already
/// recovers. It exists because the residual alone is not enough: `base`'s rounding is only
/// half the problem, and `rest` arrives from [`pmf_parts`] carrying an error of its own
/// that no amount of care in *this* sum can recover. Measured 2026-09-07 on
/// `poisson_pmf(0, lambda)`, which is exactly `e^-lambda`: 14.39 ulp with `rest_lo`
/// dropped, 0.33 with it.
/// The TwoSum itself is [`SpecializedSpecialMath::exp_two_sum`], not spelled here: an
/// error-free transformation written as `+` and `-` is only error-free when those are
/// strict, and on the scalar backend under `algebraic-scalar` they are not. The trait
/// method's default therefore returns a zero residual and this degrades cleanly to
/// `exp(base + rest_hi + rest_lo)`; `ps`/`pd` and `Dual` override it to recover the real
/// thing. See that method's docs.
#[inline(always)]
pub fn exp_sum<P, E, V>(base: V, rest_hi: V, rest_lo: V) -> V
where
    P: Policy,
    E: FloatElement,
    V: FloatVector<Element = E> + SpecializedSpecialMath<E>,
{
    let (s, resid) = V::exp_two_sum(base, rest_hi);
    let lo = resid + rest_lo;
    let es = s.exp_p::<P>();
    es.mul_adde(lo, es)
}

/// The Poisson mass `$e^{-\lambda}\lambda^k/k!$` at real `k >= 0`, `lambda >= 0` (`LOG =
/// false`), or its log (`LOG = true`), through [`pmf_parts`]. Real `k` because the Gamma
/// density is the same function (`dgamma(x; a) = pmf(a - 1; x)` for `a >= 1`).
///
/// Edges: `lambda = 0` gives `1` at `k = 0` and `0` above; `k = 0` is `e^{-lambda}` to a
/// few ulp (it goes through the shifted Stirling form like any other small `k`).
#[inline(always)]
pub fn poisson_pmf<P, E, V, const LOG: bool>(k: V, lambda: V) -> V
where
    P: Policy,
    E: FloatElement,
    V: FloatVector<Element = E> + SpecializedSpecialMath<E>,
{
    let (rest, rest_lo, large, prod, n) = pmf_parts::<P, E, V, false>(k, lambda);
    let tau_n = n * V::splat(E::TAU);
    let neg_half = V::splat(<E as FloatElement>::ConstRatio::<{ -1 }, 2>::VALUE);

    // Every lane at or past STIRLERR_MIN: rest already carries -lambda, prod is 1, and
    // lambda = 0 falls out (ln(k/0) = inf makes rest -inf). Nothing to pin.
    if const { !P::POLICY.avoid_branching } && large.all() {
        return if const { LOG } {
            tau_n.ln_p::<P>().mul_adde(neg_half, rest)
        } else {
            rest.exp_p::<P>().approx_div_sqrt_p::<P>(tau_n)
        };
    }

    let base = large.select(V::ZERO, -lambda);

    // lambda = 0 on a shifted lane: rest is -inf for k > 0 (0, right), but k = 0 has its
    // 0 * ln 0 dodged and would come out 1 only to a few ulp, so pin both.
    let lambda_zero = lambda.is_zero();
    let k_zero = k.is_zero();

    if const { LOG } {
        // rest + base + ln prod - ln(2 pi n) / 2
        let l = tau_n.ln_p::<P>().mul_adde(neg_half, ((base + rest) + rest_lo) + prod.ln_p::<P>());
        lambda_zero.select(k_zero.select(V::ZERO, V::NEG_INFINITY), l)
    } else {
        let p = (exp_sum::<P, E, V>(base, rest, rest_lo) * prod).approx_div_sqrt_p::<P>(tau_n);
        lambda_zero.select(k_zero.select(V::ONE, V::ZERO), p)
    }
}
