//! Modified Bessel functions of the first kind, orders 0 and 1, scaled and unscaled.
//!
//! # Two regions, not four
//!
//! `$I_0$` and `$I_1$` split at `x = 7.75` and nowhere else: an ascending series in
//! `$a = x^2/4$` below, and `$e^{x} P(1/x)/\sqrt{x}$` above. That is the whole shape.
//!
//! The obvious reference for a SIMD Bessel is fdlibm, and fdlibm is the wrong model. Its
//! `j0f` splits the _asymptotic envelope alone_ into four sub-intervals with a rational
//! apiece, which is optimal when a branch picks one and skips the rest, and pathological
//! here. A vector unit evaluates all four and discards three.
//! The tables underneath this kernel take Boost's route instead: fewer regions, higher
//! degree in each. Same trade the Faddeeva kernel makes for the same reason.
//!
//! The scalar sources carry a third region near the top of the range. It is not an accuracy
//! region. It exists so `$e^x$` cannot overflow before `$/\sqrt{x}$` brings the product back
//! down, and is reached here only under [`thermite::unlikely`], so the common path pays one
//! compare.
//!
//! f32 `$I_0$` is the one exception, and there the far fit is genuine: its `large` minimax
//! is fitted over `[7.75, 50]` and its constant term is wrong in the seventh digit, so
//! extending it to infinity costs about 16 ulp. See [`crate::tables::bessel`].
//!
//! # The scaled forms are the cheaper ones
//!
//! `$e^{-x} I_n(x)$` is not a wrapper that multiplies an exponential back out. Above 7.75
//! the tables _are_ the scaled value, so the scaled entry points skip the exponential
//! entirely and the unscaled ones pay for it. Below 7.75 the relationship inverts. Each form
//! therefore costs one transcendental in exactly one of its two arms, and neither is built
//! from the other. That is the relationship `zetac` has to `zeta`, for the same reason
//! (`$I_0(800)$` overflows f64 while `$e^{-800}I_0(800)$` is a perfectly ordinary `0.0141`).

use thermite::{
    math::{
        PrimalProjection, TranscendentalMathWithPolicy,
        policy::{Policy, PrecisionPolicy},
    },
    prelude::*,
};

use thermite::element::FloatElement;

use crate::specialized::BesselDetails;
use crate::tables::bessel::{BesselI, BesselK};

/// `$I_0(x)$`, or `$e^{-|x|} I_0(x)$` when `SCALED`.
///
/// Even in `x`, so the sign is dropped up front and never restored.
#[inline(always)]
pub fn bessel_i0_impl<P, V, const NS: usize, const NL: usize, const NF: usize, const SCALED: bool>(
    x: V,
    t: &BesselI<V::Element, NS, NL, NF>,
) -> V
where
    V: FloatVector + TranscendentalMathWithPolicy,
    P: Policy,
{
    let ax = x.abs();
    let small = ax.cmp_lt(V::splat(t.small_threshold));

    // Ascending series, `1 + a P(a)` with `a = x^2/4`. All terms positive: no cancellation
    // anywhere in this arm, at any x it is used for.
    let h = ax * V::HALF;
    let a = h * h;
    let mut lo = a.mul_adde(a.poly_n_p::<P, NS>(&t.small), V::ONE);
    if const { SCALED } {
        lo *= (-ax).exp_p::<P>();
    }

    // Asymptotic envelope. The tables give the _scaled_ value directly.
    let inv = V::ONE / ax;
    let mut hi = inv.poly_n_p::<P, NL>(&t.large) / ax.sqrt();

    // One mask, one reduction, reused for both the polynomial swap and the exponential
    // assembly below. The two are the same region and must not test it twice.
    let far = ax.cmp_ge(V::splat(t.far_threshold));
    let any_far = thermite::unlikely(far.any());
    if any_far {
        hi = far.select(inv.poly_n_p::<P, NF>(&t.far) / ax.sqrt(), hi);
    }

    if const { !SCALED } {
        // One `exp` in the common case. Past `far_threshold` the exponential is halved and
        // applied twice, which is what keeps `exp(x)` from reaching infinity before the
        // `1/sqrt(x)` and the sub-unit polynomial can bring it back down.
        let full = ax.exp_p::<P>();
        hi = if any_far {
            let half = (ax * V::HALF).exp_p::<P>();
            far.select((hi * half) * half, hi * full)
        } else {
            hi * full
        };
    }

    small.select(lo, hi)
}

/// `$I_1(x)$`, or `$e^{-|x|} I_1(x)$` when `SCALED`.
///
/// Odd in `x`: computed on `|x|` and signed at the end, so the small arm keeps its
/// all-positive series and the large arm keeps a positive reciprocal.
#[inline(always)]
pub fn bessel_i1_impl<P, V, const NS: usize, const NL: usize, const NF: usize, const SCALED: bool>(
    x: V,
    t: &BesselI<V::Element, NS, NL, NF>,
) -> V
where
    V: FloatVector + TranscendentalMathWithPolicy,
    P: Policy,
{
    let ax = x.abs();
    let small = ax.cmp_lt(V::splat(t.small_threshold));

    // `(x/2)(1 + a(1/2 + a P(a)))`, Boost's nested `Q` written out. The leading `x/2` is
    // what makes `I_1(x) ~ x/2` exact as `x -> 0` rather than a subtraction of near-equals.
    let h = ax * V::HALF;
    let a = h * h;
    let inner = a.mul_adde(a.poly_n_p::<P, NS>(&t.small), V::HALF);
    let mut lo = h * a.mul_adde(inner, V::ONE);
    if const { SCALED } {
        lo *= (-ax).exp_p::<P>();
    }

    let inv = V::ONE / ax;
    let mut hi = inv.poly_n_p::<P, NL>(&t.large) / ax.sqrt();

    // One mask, one reduction, reused for both the polynomial swap and the exponential
    // assembly below. The two are the same region and must not test it twice.
    let far = ax.cmp_ge(V::splat(t.far_threshold));
    let any_far = thermite::unlikely(far.any());
    if any_far {
        hi = far.select(inv.poly_n_p::<P, NF>(&t.far) / ax.sqrt(), hi);
    }

    if const { !SCALED } {
        let full = ax.exp_p::<P>();
        hi = if any_far {
            let half = (ax * V::HALF).exp_p::<P>();
            far.select((hi * half) * half, hi * full)
        } else {
            hi * full
        };
    }

    small.select(lo, hi).copysign(x)
}

/// The `x`-coefficient of the downward recurrence's trip count, by precision tier.
///
/// The continued fraction below needs `O(x)` iterations, a property of the method rather than
/// of this implementation, and Boost says so in its own `CF1_ik`: "|x| <= |v|, CF1_ik
/// converges rapidly; |x| > |v|, CF1_ik needs O(|x|) iterations to converge". So the _only_
/// knob is how far to run it, which makes the trip count itself the precision tier.
///
/// Measured against mpmath over `N` in 2..80 and `x` in 0.01..700, holding the flat margin at
/// [`RECURRENCE_MARGIN`]:
///
/// | tier | coefficient | worst relative error | max trips |
/// |---|---|---|---|
/// | `Best` / `Reference` | 0.35 | 5.19e-15 | 349 |
/// | `Average` (the default) | 0.25 | 2.43e-12 | 279 |
/// | `Medium` | 0.15 | 4.27e-08 | 209 |
/// | `Worst` | 0.10 | 2.84e-05 | 174 |
///
/// Monotone in precision, as the policy ladder requires. `Medium`'s 4.3e-08 is about f32
/// epsilon, which is why it sits there. Note 0.35 is not a rounded-up 0.5: 0.5 measured
/// _identically_ at 5.19e-15 while costing 454 trips instead of 349, so the extra was pure
/// waste. The margin and the coefficient interact (dropping the margin to 16 pushes even the
/// 0.35 rung to 2.08e-12), so neither is tunable alone.
#[inline(always)]
pub(super) const fn recurrence_x_coeff(p: PrecisionPolicy) -> (i64, i64) {
    match p {
        PrecisionPolicy::Best | PrecisionPolicy::Reference => (35, 100),
        PrecisionPolicy::Average => (25, 100),
        PrecisionPolicy::Medium => (15, 100),
        _ => (10, 100),
    }
}

/// The flat part of the trip count, on top of `N` and the `x`-scaled part.
///
/// 24 is the knee: 16 costs an order of magnitude at the top tier and 8 costs five.
pub(super) const RECURRENCE_MARGIN: usize = 24;

/// `I_N(x)` for `N >= 2`, or `e^{-|x|} I_N(x)` when `SCALED`, by downward recurrence on the
/// **ratios** rather than on the values.
///
/// # Why ratios
///
/// Writing `r_k = I_k(x)/I_{k-1}(x)`, the three-term recurrence
/// `I_{k-1} = I_{k+1} + (2k/x) I_k` divides through to
///
/// ```math
/// r_k = \frac{1}{2k/x + r_{k+1}}
/// ```
///
/// which is the same continued fraction Boost evaluates with Lentz's method in `CF1_ik`. The
/// point for a vector unit is that **every `r_k` lies in `(0, 1)`**, so nothing can overflow
/// and no rescaling is needed anywhere. The textbook alternative, Miller's linear downward
/// pass on the values themselves, has intermediates growing like `2^M M!/x^M`, which leaves
/// f64 range around order 50 and f32 range around order 8, and needs a per-lane rescale
/// _select_ inside the loop to survive. That is three selects per iteration to buy nothing.
///
/// Seeding is from `I_0`, which the closed form already provides, and `I_N = I_0 \prod r_k`.
/// The scaled and unscaled forms differ **only in that seed**. A ratio is scale-free, so
/// `SCALED` never reaches the loop.
///
/// # Why forward recurrence is not used
///
/// The textbook rule is "forward when `N < x`, downward otherwise", which would bound both
/// trip counts by `N` alone. Measured, it does not work: forward recurrence's amplification
/// grows with `N` faster than it decays with `x`, giving 2.1e-06 at `N = 50, x = 100`, and
/// `N = 80` never reaches 1e-13 for any `x` up to 700. No crossover rescues it: the best
/// fitted rule still left 1.4e-03. So there is one path here, and therefore no select between
/// paths at all.
///
/// # Trip count
///
/// `N + 24 + coeff * x`, with the per-lane start following that lane's own `x`. Lanes are
/// free to start _higher_ than they need, because the recurrence is self-correcting downward
/// from a zero seed. No lane is ever cut short, and the loop simply runs until every lane has
/// walked down to `k = 1`. The packet therefore pays its worst lane, which is the standing
/// trade for a data-dependent trip count here.
#[inline(always)]
pub fn bessel_in_pair_impl<
    P,
    E,
    V,
    const NS: usize,
    const NL: usize,
    const NF: usize,
    const N: i32,
    const SCALED: bool,
>(
    x: V,
    t: &BesselI<E, NS, NL, NF>,
) -> (V, V)
where
    E: FloatElement,
    V: FloatVector<Element = E> + TranscendentalMathWithPolicy,
    P: Policy,
{
    // No `const { assert!(N >= 2) }`: a statically-false `if const` arm still monomorphizes
    // the function it names, so the assert would fire from the dispatch's dead N = 0, 1
    // branches. Those orders are merely slower here, not wrong.

    // `I_{-n} = I_n` for integer `n`, so a negative order is the same computation and there is
    // no sign for the caller to apply. See `bessel_kn_recur` for why the absolute value is
    // taken in the body rather than in a const-generic argument.
    let m = const { N.unsigned_abs() as usize };

    let ax = x.abs();

    // ---- Which lanes take which arm -------------------------------------------------
    //
    // Above the crossover the asymptotic series is both cheaper and more accurate, and the
    // two properties move the same way: the recurrence gets worse with `x` (its trip count is
    // linear in it) while the series gets better _and_ shorter. Measured crossover to 1e-15,
    // against mpmath: `x ~ 20` for orders up to 5, 30 at order 12, 50 at order 20, 700 at
    // order 50, about `N^2/3`. Floored at 40 rather than 20 because near the crossover the
    // recurrence is still only ~30 divides and the series wants ~24 terms, so there is
    // nothing to win until `x` is a little higher.
    let use_asym = ax.cmp_ge(V::splat(E::from_int(
        const { asymptotic_threshold(N.unsigned_abs() as usize) } as _,
    )));
    let need_rec = !use_asym;
    let any_rec = need_rec.any();

    let mut value = V::ZERO;
    let mut prev = V::ZERO;

    // ---- Downward ratio recurrence --------------------------------------------------
    if any_rec {
        let i0 = bessel_i0_impl::<P, V, NS, NL, NF, SCALED>(ax, t);

        // 2/x. Infinite at x = 0, which is exactly right: it drives every ratio to zero, and
        // `I_N(0) = 0` for N >= 1 falls out of the product with no special case.
        let two_over_x = V::TWO / ax;

        let (cn, cd) = const { recurrence_x_coeff(P::POLICY.precision) };
        let coeff = V::splat(E::from_ratio(cn, cd));
        let start_f = V::splat(E::from_int((m + RECURRENCE_MARGIN) as _));
        let n_f = V::splat(E::from_int(m as _));

        // Each lane starts at its own `N + 24 + coeff*x`, and the asymptotic lanes start at
        // zero so they cannot drag the packet's trip count, which is the whole point of the
        // arm, since a single large `x` would otherwise cost every lane hundreds of divides.
        let mut k = need_rec.select(ax.mul_adde(coeff, start_f).ceil(), V::ZERO);
        let mut r = V::ZERO;
        let mut prod = V::ONE;
        // `I_{N-1}` for free: the same ladder, stopped one rung short. Having it is what lets
        // `Dual` differentiate without a second pass, since every derivative identity in this
        // family reaches DOWN one order and never up.
        let mut prod_prev = V::ONE;
        let nm1_f = V::splat(E::from_int((m as i64) - 1));

        loop {
            // One mask, one reduction, reused for both the ratio update and the accumulate.
            let active = k.cmp_ge(V::ONE);
            if active.none() {
                break;
            }

            r = active.select(V::ONE / two_over_x.mul_adde(k, r), r);
            // The last N rungs of each lane's own descent are its r_N .. r_1.
            prod = (active & k.cmp_le(n_f)).select(prod * r, prod);
            prod_prev = (active & k.cmp_le(nm1_f)).select(prod_prev * r, prod_prev);

            k -= V::ONE;
        }

        value = i0 * prod;
        prev = i0 * prod_prev;
    }

    // ---- Large-x asymptotic series --------------------------------------------------
    if use_asym.any() {
        let a = asymptotic_series::<P, E, V, SCALED>(ax, m, t.far_threshold);
        value = use_asym.select(a, value);
        if m > 0 {
            let b = asymptotic_series::<P, E, V, SCALED>(ax, m - 1, t.far_threshold);
            prev = use_asym.select(b, prev);
        }
    }

    // I_N is even for even N and odd for odd N. `I_{N-1}` has the opposite parity, so the two
    // take opposite sign treatment. Unlike `J_1`, `copysign` is safe for both, because
    // `I_nu` is positive on the whole positive axis at every order.
    let v = if const { N % 2 == 0 } { value } else { value.copysign(x) };
    let p = if const { N % 2 == 0 } { prev.copysign(x) } else { prev };
    (p, v)
}

/// Where [`bessel_in_impl`] hands the recurrence over to the asymptotic series.
///
/// `N^2/3`, floored at 40. The measured crossovers to 1e-15 are `x = 20` for orders 0..5,
/// 30 at order 12, 50 at order 20 and 700 at order 50. `N^2/3` clears all of them (order 20
/// gets 133, order 50 gets 833) without being so loose that the series is asked to work where
/// it cannot.
#[inline(always)]
const fn asymptotic_threshold(n: usize) -> usize {
    let t = n * n / 3;
    if t > 40 { t } else { 40 }
}

/// Terms in the asymptotic series.
///
/// The series is divergent, so this is a truncation and not a convergence count. 24 was the
/// best of 1..24 at every measured crossover point, so the optimal truncation is still
/// further out there. Past the crossover the terms shrink fast, and 24 is comfortably safe
/// everywhere the arm runs. It is a plausible future tier knob, but not one today, because
/// the arm is already cheap next to the hundreds of divides it replaces.
const ASYMPTOTIC_TERMS: usize = 24;

/// `(a e^{x}, b e^{x})` for two scaled `I` values, with the exponential halved past
/// `far_threshold`.
///
/// Every unscaled `I` arm in the family ends this way, and the halving is not decoration:
/// `$I_\nu(x) \sim e^x/\sqrt{2\pi x}$` is representable to about `x = 714`, while a single
/// `$e^x$` overflows at 709.78, so between the two a direct product returns `inf` for a finite
/// answer. `far_threshold` comes from the `BesselI` table so every arm turns the corner at
/// the same place. The pair form exists because the neighbouring order rides along for free
/// in every kernel that returns one, and should not cost a second exponential.
#[inline(always)]
pub(super) fn unscale_i_pair<P, E, V>(a: V, b: V, ax: V, far_threshold: E) -> (V, V)
where
    E: FloatElement,
    V: FloatVector<Element = E> + TranscendentalMathWithPolicy,
    P: Policy,
{
    unscale_i_pair_masked::<P, V>(a, b, ax, ax.cmp_ge(V::splat(far_threshold)))
}

/// `unscale_i_pair` with the far mask supplied, for an arithmetic where "far" is not a
/// plain comparison: over C it is `Re z >= threshold`, and `z` itself is the exponent.
#[inline(always)]
pub fn unscale_i_pair_masked<P, V>(a: V, b: V, z: V, far: V::Mask) -> (V, V)
where
    V: FloatVector + TranscendentalMathWithPolicy,
    P: Policy,
{
    let full = z.exp_p::<P>();
    if thermite::unlikely(far.any()) {
        let half = (z * V::HALF).exp_p::<P>();
        (
            far.select((a * half) * half, a * full),
            far.select((b * half) * half, b * full),
        )
    } else {
        (a * full, b * full)
    }
}

/// [`unscale_i_pair`] for one value.
#[inline(always)]
pub(super) fn unscale_i<P, E, V>(a: V, ax: V, far_threshold: E) -> V
where
    E: FloatElement,
    V: FloatVector<Element = E> + TranscendentalMathWithPolicy,
    P: Policy,
{
    unscale_i_pair::<P, E, V>(a, a, ax, far_threshold).0
}

/// `$K_0(x)$`, or `$e^{x} K_0(x)$` when `SCALED`.
///
/// Two regions, splitting at `x = 1`:
///
/// ```math
/// K_0(x) = P(x^2) - \ln(x)\,I_0(x), \qquad
/// K_0(x) = \frac{e^{-x}}{\sqrt{x}}\left(Y + \frac{P(1/x)}{Q(1/x)}\right)
/// ```
///
/// The `$I_0$` in the small arm is the shipped kernel, not a second copy of its coefficients.
/// Boost fits a cut-down `$I_0$` valid only on `[0,1]` for this. Reusing the real one trades a
/// longer Horner for one fewer table and a slightly better factor.
///
/// `$K$` has no reflection: it is undefined for `x < 0`, and NaN there rather than a mirrored
/// value. At `x = 0` it is `$+\infty$`, which falls out of `$-\ln(0)$` without a special case.
#[inline(always)]
pub fn bessel_k0_impl<
    P,
    E,
    V,
    const NS: usize,
    const DS: usize,
    const NL: usize,
    const DL: usize,
    const IS: usize,
    const IL: usize,
    const IF: usize,
    const SCALED: bool,
>(
    x: V,
    t: &BesselK<E, NS, DS, NL, DL>,
    ti: &BesselI<E, IS, IL, IF>,
) -> V
where
    E: FloatElement,
    V: FloatVector<Element = E> + TranscendentalMathWithPolicy,
    P: Policy,
{
    let small = x.cmp_le(V::splat(t.small_threshold));
    let mut value = V::ZERO;

    if small.any() {
        // `-ln(x) I_0(x)` carries the singularity and the rational carries everything else.
        let i0 = bessel_i0_impl::<P, V, IS, IL, IF, false>(x, ti);
        let mut lo = (x * x).poly_rational_n_p::<P, NS, DS>(&t.small_num, &t.small_den) - x.ln_p::<P>() * i0;
        if const { SCALED } {
            lo *= x.exp_p::<P>();
        }
        value = lo;
    }

    if !small.all() {
        let inv = V::ONE / x;
        let mut hi =
            (inv.poly_rational_n_p::<P, NL, DL>(&t.large_num, &t.large_den) + V::splat(t.large_offset)) / x.sqrt();
        if const { !SCALED } {
            // Halve the exponent where `e^-x` would underflow to zero before `1/sqrt(x)` and
            // the sub-unit rational could scale it back up (the mirror of the overflow guard
            // in the `I` kernels, reached for the same structural reason).
            let tiny = x.cmp_ge(V::splat(t.exp_split_threshold));
            hi = if thermite::unlikely(tiny.any()) {
                let half = (-x * V::HALF).exp_p::<P>();
                tiny.select((hi * half) * half, hi * (-x).exp_p::<P>())
            } else {
                hi * (-x).exp_p::<P>()
            };
        }
        value = small.select(value, hi);
    }

    // Undefined off the positive axis.
    x.cmp_lt(V::ZERO).select(V::NAN, value)
}

/// `$K_1(x)$`, or `$e^{x} K_1(x)$` when `SCALED`.
///
/// ```math
/// K_1(x) = R(x^2)\,x + \frac{1}{x} + \ln(x)\,I_1(x), \qquad
/// K_1(x) = \frac{e^{-x}}{\sqrt{x}}\left(Y + \frac{P(1/x)}{Q(1/x)}\right)
/// ```
///
/// The `$1/x$` is the singularity and is left as a bare reciprocal rather than folded into
/// the rational, because it is the entire value as `$x \to 0$` and any rearrangement that
/// mixes it with the `$O(x)$` terms cancels it away.
#[inline(always)]
pub fn bessel_k1_impl<
    P,
    E,
    V,
    const NS: usize,
    const DS: usize,
    const NL: usize,
    const DL: usize,
    const IS: usize,
    const IL: usize,
    const IF: usize,
    const SCALED: bool,
>(
    x: V,
    t: &BesselK<E, NS, DS, NL, DL>,
    ti: &BesselI<E, IS, IL, IF>,
) -> V
where
    E: FloatElement,
    V: FloatVector<Element = E> + TranscendentalMathWithPolicy,
    P: Policy,
{
    let small = x.cmp_le(V::splat(t.small_threshold));
    let mut value = V::ZERO;

    if small.any() {
        let i1 = bessel_i1_impl::<P, V, IS, IL, IF, false>(x, ti);
        let mut lo = (x * x)
            .poly_rational_n_p::<P, NS, DS>(&t.small_num, &t.small_den)
            .mul_adde(x, V::ONE / x)
            + x.ln_p::<P>() * i1;
        if const { SCALED } {
            lo *= x.exp_p::<P>();
        }
        value = lo;
    }

    if !small.all() {
        let inv = V::ONE / x;
        let mut hi =
            (inv.poly_rational_n_p::<P, NL, DL>(&t.large_num, &t.large_den) + V::splat(t.large_offset)) / x.sqrt();
        if const { !SCALED } {
            let tiny = x.cmp_ge(V::splat(t.exp_split_threshold));
            hi = if thermite::unlikely(tiny.any()) {
                let half = (-x * V::HALF).exp_p::<P>();
                tiny.select((hi * half) * half, hi * (-x).exp_p::<P>())
            } else {
                hi * (-x).exp_p::<P>()
            };
        }
        value = small.select(value, hi);
    }

    x.cmp_lt(V::ZERO).select(V::NAN, value)
}

/// `$K_N(x)$` for `N >= 2` from `$K_0$` and `$K_1$`, by **upward** recurrence.
///
/// ```math
/// K_{n+1}(x) = K_{n-1}(x) + \frac{2n}{x} K_n(x)
/// ```
///
/// Takes the two seeds rather than the coefficient tables that produce them. The recurrence
/// has nothing to do with any table, and threading four of them through so it could call the
/// order-0 and order-1 kernels itself cost **fourteen** const-generic array lengths on a
/// function whose arithmetic needs none. The one place that already names the tables
/// concretely, the per-element dispatch, builds the seeds instead.
///
/// Scaled and unscaled both work with no flag: `$e^{x}$` is a common factor of every term and
/// passes straight through, so whichever form the seeds are in is the form that comes out.
///
/// # Why upward, when `I` needs downward
///
/// `$K_\nu$` is the **dominant** solution of the modified Bessel equation and `$I_\nu$` the
/// minimal one, so the stability argument inverts exactly. Recurring `$K$` upward amplifies
/// what is already growing, which is harmless. Recurring the _minimal_ solution upward
/// destroys it, which is what forces `$I$` onto a downward continued fraction with an
/// `$O(x)$` trip count. Here the cost is `N - 1` steps: no continued fraction, no dependence
/// on `x`, and no precision tier to measure. That is why this is the one kernel in the file
/// with no `P: Policy` parameter at all. Every operation in it is exact-by-construction
/// arithmetic, so there is no approximation to pick a tier for.
///
/// # Overflow
///
/// `$K_N$` grows quickly in the _order_: `$K_{50}(1)$` is about `$2.6\times10^{78}$` and
/// `$K_{60}(1)$` leaves f64. The recurrence sums terms of like sign, so it saturates to
/// `$+\infty$` rather than returning a wrong finite value, and the scaled form buys nothing
/// here: unlike `$I$`, this overflow is in `N`, not in `x`.
#[inline(always)]
pub fn bessel_kn_recur<E, V, const N: i32>(x: V, k0: V, k1: V) -> (V, V)
where
    E: FloatElement,
    V: FloatVector<Element = E>,
{
    let mut prev = k0;
    let mut cur = k1;

    let two_over_x = V::TWO / x;

    // `K_{-n} = K_n` for integer `n`, so a negative order is genuinely the same walk. Unlike
    // `J`/`Y`, the caller has no sign to apply afterwards. The absolute value is taken here
    // rather than in a const-generic argument because `foo::<{ N.unsigned_abs() }>` needs
    // `generic_const_exprs`. A `const` block folds the same way, so this still unrolls.
    let m = const { N.unsigned_abs() as usize };

    // `m` is const, so this unrolls and every `2n` is an immediate.
    let mut n = 1usize;
    while n < m {
        let next = two_over_x.mul_adde(V::splat(E::from_int(n as _)) * cur, prev);
        prev = cur;
        cur = next;
        n += 1;
    }

    // `(K_{N-1}, K_N)`. The upward walk passes through order `N-1` anyway, so returning both
    // is free, and is what lets `Dual` use `K_N' = -K_{N-1} - (N/x)K_N` without a second
    // pass over the recurrence.
    (prev, cur)
}

/// The large-`x` asymptotic series for `$I_\nu$`, scaled or not.
///
/// ```math
/// I_\nu(x) \sim \frac{e^x}{\sqrt{2\pi x}} \sum_k \frac{(-1)^k a_k(\nu)}{x^k},
/// \qquad a_k(\nu) = \frac{\prod_j\left(4\nu^2 - (2j-1)^2\right)}{k!\,8^k}
/// ```
///
/// `nu` is a **runtime** parameter, not a const one, purely so the same body can be called at
/// `N` and `N - 1` without `generic_const_exprs`. The per-term ratio is still the same small
/// rational, `((2k+1)^2 - 4nu^2) / (8(k+1))`, so nothing here needs a coefficient table at any
/// order. What is lost against a const `nu` is only the folding of that one multiplier.
#[inline(always)]
fn asymptotic_series<P, E, V, const SCALED: bool>(ax: V, nu: usize, far_threshold: E) -> V
where
    E: FloatElement,
    V: FloatVector<Element = E> + TranscendentalMathWithPolicy,
    P: Policy,
{
    let w = V::ONE / ax;
    let four_nu2 = 4 * (nu as i64) * (nu as i64);

    let mut term = V::ONE;
    let mut sum = V::ONE;
    let mut i = 0usize;
    while i < ASYMPTOTIC_TERMS {
        let num = (2 * i as i64 + 1) * (2 * i as i64 + 1) - four_nu2;
        let den = 8 * (i as i64 + 1);
        term *= w * V::splat(E::from_ratio(num, den));
        sum += term;
        i += 1;
    }

    let a = (sum * V::FRAC_1_SQRT_TAU) / ax.sqrt();
    if const { SCALED } {
        a
    } else {
        unscale_i::<P, E, V>(a, ax, far_threshold)
    }
}

/// `$I_N(x)$` for `N >= 2`, or `$e^{-|x|} I_N(x)$` when `SCALED`.
///
/// Thin wrapper over [`bessel_in_pair_impl`], which computes `$I_{N-1}$` alongside at no extra
/// cost in the recurrence arm. Callers that want the derivative should take the pair directly.
#[inline(always)]
pub fn bessel_in_impl<P, E, V, const NS: usize, const NL: usize, const NF: usize, const N: i32, const SCALED: bool>(
    x: V,
    t: &BesselI<E, NS, NL, NF>,
) -> V
where
    E: FloatElement,
    V: FloatVector<Element = E> + TranscendentalMathWithPolicy,
    P: Policy,
{
    bessel_in_pair_impl::<P, E, V, NS, NL, NF, N, SCALED>(x, t).1
}

/// The large-`x` asymptotic series for `I_nu`, with a **per-lane** order.
///
/// Same expansion as [`asymptotic_series`], which takes a scalar `nu` and folds
/// `4 nu^2` into an immediate. Here `nu` is a vector, so the term numerator
/// `(2k+1)^2 - 4 nu^2` becomes one vector subtract and one vector multiply per term
/// (24 terms, so about 48 extra ops). The denominator `8(k+1)` is still a compile-time
/// constant and still folds.
///
/// This exists because leaving it out was measurably expensive. Without an asymptotic arm
/// the runtime-order path falls back to the recurrence at large `x`, and the recurrence's
/// trip count IS the precision tier, which put `bessel_i_scaled` at 698 to 2634 ULP on the
/// `performance` and `size` tiers over `[-400, 400]`, against 3-6 ULP at `precision`. The
/// const path never showed that, because its asymptotic arm took over exactly where the
/// recurrence starts to degrade.
#[inline(always)]
pub(super) fn asymptotic_series_v<P, E, V, const SCALED: bool>(ax: V, nu: V, far_threshold: E) -> V
where
    E: FloatElement,
    V: FloatVector<Element = E> + TranscendentalMathWithPolicy,
    P: Policy,
{
    let w = V::ONE / ax;
    let four_nu2 = (nu * nu) * V::splat(E::from_int(4));

    let mut term = V::ONE;
    let mut sum = V::ONE;
    let mut i = 0usize;
    while i < ASYMPTOTIC_TERMS {
        let odd_sq = (2 * i as i64 + 1) * (2 * i as i64 + 1);
        let num = V::splat(E::from_int(odd_sq)) - four_nu2;
        let inv_den = V::splat(E::from_ratio(1, 8 * (i as i64 + 1)));
        term *= (w * num) * inv_den;
        sum += term;
        i += 1;
    }

    let a = (sum * V::FRAC_1_SQRT_TAU) / ax.sqrt();
    if const { SCALED } {
        a
    } else {
        unscale_i::<P, E, V>(a, ax, far_threshold)
    }
}

/// `asymptotic_series_v` over a general arithmetic: the argument `z` in `C`, the order in
/// its real primal `R`, and the far mask supplied by the caller.
///
/// This is the body the real-order kernel runs, and through it the complex one. The loop is
/// the same as `asymptotic_series_v`'s. It is a separate function rather than that one's
/// body because the integer and half-integer kernels call the real form from bounds that
/// know nothing of [`BesselDetails`], and the second-term machinery below is only meaningful
/// off the real axis. Everything `z`-dependent is `C` arithmetic. The term numerators
/// `(2k+1)^2 - 4 nu^2` stay real and enter through `C: Mul<R>`, so a complex instantiation
/// pays two real multiplies per term rather than a complex one.
///
/// # The second exponential
///
/// The expansion has two exponential terms (DLMF 10.40.5), and the real line keeps one
/// because the other is `e^{-2x}` relative, below epsilon anywhere this arm runs. That is
/// a fact about the real axis. Off it the second term's modulus is `e^{-2 Re z}`, which on
/// the imaginary axis is one, so an arithmetic can ask for it through
/// [`BesselDetails::ASYM_TWO_TERMS`]. It costs one more exponential and no extra series
/// evaluation: the second sum is the first with every other sign flipped, so both are
/// accumulated in the one loop.
#[inline(always)]
pub fn asymptotic_series_g<P, E, R, C, const SCALED: bool>(z: C, nu: R, far: C::Mask) -> C
where
    E: FloatElement,
    R: FloatVector<Element = E>,
    C: FloatVector<Mask = R::Mask>
        + TranscendentalMathWithPolicy
        + PrimalProjection<Primal = R>
        + BesselDetails<C>
        + core::ops::Mul<R, Output = C>,
    P: Policy,
{
    let w = C::ONE / z;
    let four_nu2 = (nu * nu) * R::splat(E::from_int(4));

    let mut term = C::ONE;
    let mut sum = C::ONE;
    let mut sum_alt = C::ONE;
    let mut i = 0usize;
    while i < ASYMPTOTIC_TERMS {
        let odd_sq = (2 * i as i64 + 1) * (2 * i as i64 + 1);
        let num = R::splat(E::from_int(odd_sq)) - four_nu2;
        let inv_den = R::splat(E::from_ratio(1, 8 * (i as i64 + 1)));
        // Associated exactly as `asymptotic_series_v` does, so the real instantiation
        // rounds identically.
        term *= (w * num) * inv_den;
        sum += term;
        if const { C::ASYM_TWO_TERMS } {
            // `i` is the loop index, so this parity folds.
            sum_alt = if i.is_multiple_of(2) {
                sum_alt - term
            } else {
                sum_alt + term
            };
        }
        i += 1;
    }

    if const { C::ASYM_TWO_TERMS } {
        sum += sum_alt * C::asym_second_exponent(z, nu).exp_p::<P>();
    }

    let a = (sum * C::FRAC_1_SQRT_TAU) / z.sqrt();
    if const { SCALED } {
        a
    } else {
        unscale_i_pair_masked::<P, C>(a, a, z, far).0
    }
}

/// `I_n(x)` with a **per-lane** order, or `e^{-|x|} I_n(x)` when `SCALED`.
///
/// The const-generic form is the one to reach for when the order is known. This exists for the
/// case `hermitev` exists for: an order that arrives as data. Every lane may ask for a
/// different one.
///
/// # What changes, and what does not
///
/// Almost nothing. The ratio recurrence already accumulates its product under a mask
/// (`k <= N`), so making `N` a vector rather than a splat is a one-word change. The trip-count
/// start `N + 24 + c*x` was already per-lane in `x` and simply becomes per-lane in `n` too.
/// The loop still ends when the last lane reaches `k = 1`, so the packet pays for its widest
/// (order, argument) pair, the standing trade.
///
/// The asymptotic arm generalises too, via [`asymptotic_series_v`]: `4 nu^2` no longer folds,
/// which costs one vector subtract and one multiply per term. Without the arm the lower tiers
/// reached 698 to 2634 ULP on `bessel_i_scaled` past `x ~ 40`, because the recurrence's trip
/// count is the tier and its low rungs are far too short there.
#[inline(always)]
pub fn bessel_iv_impl<P, E, V, const NS: usize, const NL: usize, const NF: usize, const SCALED: bool>(
    x: V,
    n: V,
    t: &BesselI<E, NS, NL, NF>,
) -> V
where
    E: FloatElement,
    V: FloatVector<Element = E> + TranscendentalMathWithPolicy,
    P: Policy,
{
    let ax = x.abs();

    // Which lanes take which arm, exactly as the const form decides it: the asymptotic series
    // past `max(40, n^2/3)`, the recurrence below. Per-lane now, since `n` is per-lane.
    let n2_third = (n * n) * V::splat(E::from_ratio(1, 3));
    let thresh = n2_third.max(V::splat(E::from_int(40)));
    let use_asym = ax.cmp_ge(thresh);
    let need_rec = !use_asym;

    let two_over_x = V::TWO / ax;
    let mut value = V::ZERO;

    if need_rec.any() {
        let i0 = bessel_i0_impl::<P, V, NS, NL, NF, SCALED>(ax, t);

        let (cn, cd) = const { recurrence_x_coeff(P::POLICY.precision) };
        let coeff = V::splat(E::from_ratio(cn, cd));
        let margin = V::splat(E::from_int(RECURRENCE_MARGIN as _));

        // Asymptotic lanes start at zero so they cannot drag the packet's trip count.
        let mut k = need_rec.select(ax.mul_adde(coeff, n + margin).ceil(), V::ZERO);
        let mut r = V::ZERO;
        let mut prod = V::ONE;

        loop {
            let active = k.cmp_ge(V::ONE);
            if active.none() {
                break;
            }
            r = active.select(V::ONE / two_over_x.mul_adde(k, r), r);
            prod = (active & k.cmp_le(n)).select(prod * r, prod);
            k -= V::ONE;
        }

        value = i0 * prod;
    }

    if use_asym.any() {
        value = use_asym.select(asymptotic_series_v::<P, E, V, SCALED>(ax, n, t.far_threshold), value);
    }

    // Parity is per-lane: even orders are even in `x`, odd orders odd. `I_nu` is positive on
    // the positive axis at every order, so the sign is a straight per-lane negate.
    let odd_order = (n * V::HALF).fract().cmp_gt(V::ZERO);
    value.neg_c(odd_order & x.is_negative())
}

/// `K_n(x)` with a per-lane order, or `e^{x} K_n(x)` when `SCALED`.
///
/// Upward recurrence, each lane freezing once it reaches its own order, the `hermitev` shape.
/// The loop runs to the packet's largest order.
#[inline(always)]
#[allow(clippy::too_many_arguments)]
pub fn bessel_kv_impl<
    P,
    E,
    V,
    const AS: usize,
    const AD: usize,
    const AL: usize,
    const ALD: usize,
    const BS: usize,
    const BD: usize,
    const BL: usize,
    const BLD: usize,
    const IS: usize,
    const IL: usize,
    const IF: usize,
    const JS: usize,
    const JL: usize,
    const JF: usize,
    const SCALED: bool,
>(
    x: V,
    n: V,
    t0: &BesselK<E, AS, AD, AL, ALD>,
    t1: &BesselK<E, BS, BD, BL, BLD>,
    ti0: &BesselI<E, IS, IL, IF>,
    ti1: &BesselI<E, JS, JL, JF>,
) -> V
where
    E: FloatElement,
    V: FloatVector<Element = E> + TranscendentalMathWithPolicy,
    P: Policy,
{
    let k0 = bessel_k0_impl::<P, E, V, AS, AD, AL, ALD, IS, IL, IF, SCALED>(x, t0, ti0);
    let k1 = bessel_k1_impl::<P, E, V, BS, BD, BL, BLD, JS, JL, JF, SCALED>(x, t1, ti1);

    let two_over_x = V::TWO / x;
    let mut prev = k0;
    let mut cur = k1;
    let mut step = V::ONE;

    loop {
        // Freeze BOTH halves on a lane that has reached its order. Advancing only `cur` would
        // leave `prev` one rung behind and corrupt the next step, the same trap `hermitev`
        // documents for its pair.
        let cont = step.cmp_lt(n);
        if cont.none() {
            break;
        }
        let next = two_over_x.mul_adde(step * cur, prev);
        prev = cont.select(cur, prev);
        cur = cont.select(next, cur);
        step += V::ONE;
    }

    // Order 0 never entered the loop, so pick it out directly.
    n.cmp_le(V::ZERO).select(k0, cur)
}
