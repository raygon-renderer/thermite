use super::*;

use crate::vector::NumVector;

/// Shared body of the real-vector [`SpecializedCoreMath::poly_primal`] override.
///
/// A real float vector is its own primal, so the coefficients arrive already splatted:
/// this is [`poly`](SpecializedCoreMath::poly) minus the per-term `splat`, ILP lowering
/// and all. Identical for f32 and f64, so both backends delegate here.
#[inline(always)]
pub fn poly_primal_internal<V, P, N>(x: V, coeffs: &GenericArray<V, N>) -> V
where
    V: FloatVector,
    P: Policy,
    N: ArrayLength,
{
    let n = const { N::USIZE };

    if const {
        !P::POLICY.unroll_loops
            || P::POLICY.precision.ge(PrecisionPolicy::Best)
            || !V::ISA.has_instruction_level_parallelism()
    } {
        let mut res = coeffs[n - 1];
        for &c in coeffs.iter().rev().skip(1) {
            res = res.mul_adde(x, c);
        }
        return res;
    }

    // `poly_f` rather than `poly_f_n`: a typenum length cannot be passed as a const
    // generic argument on stable. `poly_f` pins fast_polynomial's own `LENGTH` to 0, so
    // its internal `assert_unchecked(n == LENGTH)` hint does NOT fire, and the 16-arm
    // length match folds only via caller inlining plus constant propagation of
    // `N::USIZE`. It does: `bin/poly_primal_probe` emits byte-identical asm for this and
    // the const-N `poly_f_n` lowering at N=13 on AVX2 (Estrin, no call, no jump table).
    // Re-run that probe if this is ever restructured.
    //
    // NumVector provides the num_traits::MulAdd implementation fast_polynomial needs.
    let res = fast_polynomial::poly_f(NumVector(x), n, |i| unsafe { NumVector(*coeffs.get_unchecked(i)) });

    res.0
}

/// [`poly_primal_internal`] with the coefficients in reverse (descending) order,
/// backing the real-vector [`SpecializedCoreMath::poly_rev_primal`] override.
#[inline(always)]
pub fn poly_rev_primal_internal<V, P, N>(x: V, coeffs: &GenericArray<V, N>) -> V
where
    V: FloatVector,
    P: Policy,
    N: ArrayLength,
{
    let n = const { N::USIZE };

    if const {
        !P::POLICY.unroll_loops
            || P::POLICY.precision.ge(PrecisionPolicy::Best)
            || !V::ISA.has_instruction_level_parallelism()
    } {
        let mut res = coeffs[0];
        for &c in coeffs.iter().skip(1) {
            res = res.mul_adde(x, c);
        }
        return res;
    }

    // See `poly_primal_internal` for why this is `poly_f` and not `poly_f_n`.
    let res = fast_polynomial::poly_f(NumVector(x), n, |i| unsafe {
        NumVector(*coeffs.get_unchecked(n - 1 - i))
    });

    res.0
}

#[inline(always)]
pub fn inverse_sqrt_internal<V, E: FloatElement, P>(x: V) -> V
where
    V: FloatVectorWithBits<Element = E> + SpecializedTranscendentalMath<E>,
    P: Policy,
{
    let mut y = x.rsqrt();

    if const { V::HAS_APPROX_RSQRT && P::POLICY.precision.gt(PrecisionPolicy::Worst) } {
        let nx2 = x.scale(const { E::ConstRatio::<{ -1 }, { 2 }>::VALUE }); // -0.5*x
        let threehalfs = V::splat(const { E::ConstRatio::<{ 3 }, { 2 }>::VALUE }); // 1.5

        // one iteration of Newton's method
        y = y * y.square().mul_adde(nx2, threehalfs);
    }

    y
}

#[inline(always)]
pub fn sinc_internal<V, E: FloatElement, P>(x: V) -> V
where
    V: FloatVectorWithBits<Element = E> + SpecializedTranscendentalMath<E>,
    P: Policy,
{
    if const { P::POLICY.precision.le(PrecisionPolicy::Medium) } {
        let mut y = V::sin::<P>(x).approx_div_p::<P>(x);

        if const { P::POLICY.check_overflow } {
            y = x.is_zero().select(V::ONE, y);
            y = x.is_infinite().select(V::ZERO, y);
        }

        return y;
    }

    let is_tiny = x.abs().cmp_le(V::FOURTH_ROOT_EPSILON);

    let x2 = x.square();

    // if branching, use Taylor series for tiny x without calling sine.
    if const { !P::POLICY.avoid_branching } && crate::unlikely(is_tiny.all()) {
        if const { P::POLICY.precision.le(PrecisionPolicy::Average) } {
            // use fma instead of division then subtraction, for improved performance
            // at the cost of a tiny bit of precision with the 120 denominator
            return x2.mul_adde(
                x2.mul_sube(V::splat(FloatElement::from_ratio(1, 120)), V::FRAC_1_6),
                V::ONE,
            );
        }

        let res = x2 / V::splat(const { E::ConstInt::<{ 120 }>::VALUE });
        return x2.mul_adde(res - V::FRAC_1_6, V::ONE);
    }

    // For very small x, sinc(x) ~ 1 - x^2/6 + x^4/120
    let num = is_tiny.select(x2, V::sin::<P>(x));
    let den = is_tiny.select(V::splat(const { E::ConstInt::<{ 120 }>::VALUE }), x);

    // combined division, since division is expensive
    let mut y = num.approx_div_p::<P>(den);

    y = is_tiny.select(x2.mul_adde(y - V::FRAC_1_6, V::ONE), y);

    if const { P::POLICY.check_overflow } {
        y = x.is_infinite().select(V::ZERO, y);
    }

    y
}

/// `$\sinh(x)/x$`, the shared body behind [`SpecializedTranscendentalMath::sinhc`].
///
/// Structurally identical to [`sinc_internal`] above, with three differences worth naming.
/// The series is `$1 + x^2/6 + x^4/120$` rather than alternating, so the `1/6` term is added
/// where `sinc` subtracts it. The large-argument limit is `$+\infty$`, not `0`, and is
/// reached from both sides: `sinh` overflows to a signed infinity past `x ~ 710` and the
/// division by `x` restores the sign, so only an exactly infinite input needs the patch
/// (`inf/inf` is NaN). And there is no zero of `sinh` to worry about away from the origin,
/// so the tiny window is the only special case in the domain.
///
/// The window is the fourth root of epsilon, which is what Boost's `sinhc_pi` uses
/// (`taylor_n_bound`) and what this crate's `sinc` already uses. Below it the series is
/// accurate to well under an ulp and, more to the point, costs no `sinh` at all.
#[inline(always)]
pub fn sinhc_internal<V, E: FloatElement, P>(x: V) -> V
where
    V: FloatVectorWithBits<Element = E> + SpecializedTranscendentalMath<E>,
    P: Policy,
{
    if const { P::POLICY.precision.le(PrecisionPolicy::Medium) } {
        let mut y = V::sinh::<P>(x).approx_div_p::<P>(x);

        if const { P::POLICY.check_overflow } {
            y = x.is_zero().select(V::ONE, y);
            y = x.is_infinite().select(V::INFINITY, y);
        }

        return y;
    }

    let is_tiny = x.abs().cmp_le(V::FOURTH_ROOT_EPSILON);

    let x2 = x.square();

    // if branching, use the Taylor series for tiny x without calling sinh.
    if const { !P::POLICY.avoid_branching } && crate::unlikely(is_tiny.all()) {
        if const { P::POLICY.precision.le(PrecisionPolicy::Average) } {
            // one FMA instead of a divide then an add, at the cost of a little accuracy
            // in the 120 denominator
            return x2.mul_adde(
                x2.mul_adde(V::splat(FloatElement::from_ratio(1, 120)), V::FRAC_1_6),
                V::ONE,
            );
        }

        let res = x2 / V::splat(const { E::ConstInt::<{ 120 }>::VALUE });
        return x2.mul_adde(res + V::FRAC_1_6, V::ONE);
    }

    // For very small x, sinhc(x) ~ 1 + x^2/6 + x^4/120
    let num = is_tiny.select(x2, V::sinh::<P>(x));
    let den = is_tiny.select(V::splat(const { E::ConstInt::<{ 120 }>::VALUE }), x);

    // combined division, since division is expensive
    let mut y = num.approx_div_p::<P>(den);

    y = is_tiny.select(x2.mul_adde(y + V::FRAC_1_6, V::ONE), y);

    if const { P::POLICY.check_overflow } {
        // sinh(inf)/inf is inf/inf = NaN; the limit is +inf from both sides.
        y = x.is_infinite().select(V::INFINITY, y);
    }

    y
}

/// High-precision `$\ln(1 - e^{-x})$`, shared by the f32 and f64 backends at
/// `Average` precision and above.
///
/// No single expression covers the domain. `(1 - exp(-x)).ln()` loses small `x` to
/// cancellation (`-inf` below one ulp of 1, ~1e13 ulp just above it) and large `x`
/// to `1 - e^{-x}` rounding to exactly 1 past `x ~ 36`, returning 0 where the answer
/// is `~-e^{-x}`, a relative error of 1. The split at `$\ln 2$` is Maechler (2012):
/// below it the subtraction lives inside `exp_m1`, above it inside `ln_1p`, and each
/// is exact where it is used. Measured at <= 0.9 ulp over `x in [1e-28, 1e6]` for f64.
///
/// Each side costs two transcendentals, and real inputs cluster (a log-domain gap is
/// usually all-small or all-large across a vector), so when the policy allows
/// branching, a side no lane needs is skipped.
///
/// Deriving the `exp` from the `exp_m1` (`e^{-x} = 1 + expm1(-x)`, exact by Sterbenz
/// for `x > ln 2`) does not work. `expm1(-x)` has already rounded to exactly -1 by
/// `x ~ 36`, which reintroduces the large-`x` failure at the same measured 8.8e15 ulp.
///
/// Domain: `x >= 0`, with `ln1m_expnx(0) = -inf` and negative `x` yielding NaN.
#[inline(always)]
pub fn ln1m_expnx_internal<V, E: FloatElement, P>(x: V) -> V
where
    V: FloatVectorWithBits<Element = E> + SpecializedTranscendentalMath<E>,
    P: Policy,
{
    let is_lo = x.cmp_le(V::LN_2);

    let mut y = V::ZERO;

    if const { P::POLICY.avoid_branching } || is_lo.any() {
        y = (-V::exp_m1::<P>(-x)).ln::<P>();
    }

    if const { P::POLICY.avoid_branching } || !is_lo.all() {
        y = is_lo.select(y, V::ln_1p::<P>(-V::exp::<P>(-x)));
    }

    y
}

pub trait SincPiConsts: FloatConsts {
    const FRAC_PI_SQR_OVER_6: Self;
    const FRAC_120_OVER_PI_SQR: Self;
    const FRAC_PI_4_OVER_120: Self;
}

impl SincPiConsts for f32 {
    const FRAC_PI_SQR_OVER_6: Self = 1.6449340668482264364724151666460251892189499012068; // pi^2/6
    const FRAC_120_OVER_PI_SQR: Self = 1.2319178705621202226983339920542432970193362224366; // 120/pi^2
    const FRAC_PI_4_OVER_120: Self = 0.81174242528335364363700277240587592708106321393905; // pi^4/120
}

impl SincPiConsts for f64 {
    const FRAC_PI_SQR_OVER_6: Self = 1.6449340668482264364724151666460251892189499012068; // pi^2/6
    const FRAC_120_OVER_PI_SQR: Self = 1.2319178705621202226983339920542432970193362224366; // 120/pi^2
    const FRAC_PI_4_OVER_120: Self = 0.81174242528335364363700277240587592708106321393905; // pi^4/120
}

#[inline(always)]
pub fn sinc_pi_internal<V, E: FloatElement + SincPiConsts, P>(x: V) -> V
where
    V: FloatVectorWithBits<Element = E> + SpecializedTranscendentalMath<E>,
    P: Policy,
{
    if const { P::POLICY.precision.le(PrecisionPolicy::Medium) } {
        // forwards to the above medium-precision sinc implementation,
        // which uses sin(x) * rcp(x)
        return V::sinc_p::<P>(x.scale(FloatConsts::PI));
    }

    let pi_2_frac_6: V = V::splat(E::FRAC_PI_SQR_OVER_6);
    let frac_120_pi_4: V = V::splat(E::FRAC_120_OVER_PI_SQR);

    let is_tiny = x.abs().cmp_le(V::FOURTH_ROOT_EPSILON);

    let x2 = x.square();

    // if branching, use Taylor series for tiny x without calling sine.
    if const { !P::POLICY.avoid_branching } && crate::unlikely(is_tiny.all()) {
        let pi_4_frac_120: V = V::splat(E::FRAC_PI_4_OVER_120);

        // unlike sinc, which has x^2/120 with 120 being an exact integer,
        // sinc_pi has pi^4/120, and since pi is irrational and imprecise anyway, we
        // can avoid the exact division by 120 in favor of multiplying by pi^4/120
        return x2.mul_adde(x2.mul_sube(pi_4_frac_120, pi_2_frac_6), V::ONE);
    }

    // for very small x, sinc_pi(x) ~ 1 - (pi^2/6)*x^2 + (pi^4/120)*x^4
    let num = is_tiny.select(x2, V::sin_pi::<P>(x));
    let den = is_tiny.select(frac_120_pi_4, x.scale(FloatConsts::PI)); // NOTE: first term is flipped for division

    // combined division, since division is expensive
    let mut y = num / den;

    y = is_tiny.select(x2.mul_adde(y - pi_2_frac_6, V::ONE), y);

    if const { P::POLICY.check_overflow } {
        y = x.is_infinite().select(V::ZERO, y);
    }

    y
}

pub trait LogNHelper: FloatConsts + Sized {
    const LOG2_TABLE: [Self; 30];

    fn fallback<P: Policy, const N: usize>() -> Self;
}

macro_rules! impl_log2_table {
    ($($value:expr),* $(,)?) => {
        impl LogNHelper for f32 {
            const LOG2_TABLE: [Self; 30] = [$($value),*];

            #[inline(always)]
            fn fallback<P: Policy, const N: usize>() -> Self {
                cfg_select! {
                    all(feature = "spirv", target_arch = "spirv") => {
                        Vector::<f32>(N as f32).log2_p::<P>().0
                    }
                    _ => libm::log2f(N as f32),
                }
            }
        }

        impl LogNHelper for f64 {
            const LOG2_TABLE: [Self; 30] = [$($value),*];

            #[inline(always)]
            fn fallback<P: Policy, const N: usize>() -> Self {
                cfg_select! {
                    all(feature = "spirv", target_arch = "spirv") => {
                        Vector::<f64>(N as f64).log2_p::<P>().0
                    }
                    _ => libm::log2(N as f64),
                }
            }
        }
    };
}

impl_log2_table![
    // precomputed N[Table[1/log_2(x), {x, 3, 32}], 20]
    0.63092975357145743710,
    0.50000000000000000000,
    0.43067655807339305067,
    0.38685280723454158687,
    0.35620718710802217651,
    0.33333333333333333333,
    0.31546487678572871855,
    0.30102999566398119521,
    0.28906482631788785927,
    0.27894294565112984319,
    0.27023815442731974129,
    0.26264953503719354798,
    0.25595802480981548939,
    0.25000000000000000000,
    0.24465054211822603039,
    0.23981246656813144474,
    0.23540891336663823645,
    0.23137821315975917426,
    0.22767024869695299798,
    0.22424382421757543948,
    0.22106472945750374615,
    0.21810429198553155923,
    0.21533827903669652534,
    0.21274605355336315361,
    0.21030991785715247903,
    0.20801459767650945760,
    0.20584683246043445731,
    0.20379504709050619003,
    0.20184908658209985072,
    0.20000000000000000000,
];

#[inline(always)]
pub fn log_n_internal<V, E: FloatElement + LogNHelper, P, const N: usize>(x: V) -> V
where
    V: FloatVectorWithBits<Element = E> + SpecializedTranscendentalMath<E>,
    P: Policy,
{
    if const { N > 32 } {
        return V::log2::<P>(x).approx_div_p::<P>(V::splat(E::fallback::<P, N>()));
    }

    match N {
        // 0 and 1 are special cases, and these are what Wolfram Alpha returns
        0 => V::ZERO,               // log(x)/log(0) = log(x)/-infinity = 0
        1 => FloatVector::INFINITY, // log(x)/log(1) = log(x)/0 = complex infinity, only return real part
        2 => V::log2::<P>(x),
        10 => V::log10::<P>(x),
        _ => V::log2::<P>(x).scale(
            const {
                if const { N <= 32 } {
                    E::LOG2_TABLE[N.saturating_sub(3)]
                } else {
                    E::ZERO
                }
            },
        ),
    }
}

/// Terms of the odd series in [`log1pmx_internal`], for a format's full precision.
///
/// The window is `|r| <= 1/3`, so `y = r^2 <= 1/9` and term `k` is `3 y^k / (2k+3)`
/// relative to the leading `1/3`: `1e-3` at k=3, `4e-5` at k=4, `4e-6` at k=5, `4e-8` at
/// k=7, and `1e-17` at k=16. That gives 17 terms in binary64 and 8 in binary32 - all FMAs,
/// and cheaper than the `ln` the window exists to avoid.
///
/// There is no per-tier trimming: `Medium` and below skip the series outright (see
/// [`log1pmx_internal`]), so every tier that reaches it wants the full count. Trimming the
/// count would be the wrong knob anyway, as the dropped term is `O(y^k)` and vanishes toward
/// `x = 0`, so it buys nothing exactly where the series is doing the work.
#[inline(always)]
pub const fn log1pmx_terms(mantissa_bits: u32) -> usize {
    if mantissa_bits > 24 { 17 } else { 8 }
}

/// `$\ln(1+x) - x$`, the shared body behind [`SpecializedTranscendentalMath::log1pmx`] for
/// real f32 and f64 vectors.
///
/// Both terms are `$O(x)$` and the answer is `$O(x^2)$`, so the direct spelling loses
/// `$2\varepsilon/|x|$`, which is everything by `$|x| \approx \varepsilon$`. Substituting
/// `$\ln(1+x) = 2\,\mathrm{atanh}(r)$` with `$r = x/(2+x)$` and subtracting the `x` *inside*
/// the series removes the cancellation entirely:
///
/// ```text
/// 2 atanh(r) - x = 2r sum_{k>=0} y^k/(2k+1) - x,   y = r^2
///                = (2r - x) + 2 r y sum_{k>=0} y^k/(2k+3)
///                = r (2 y S(y) - x)                since 2r - x = -x r
/// ```
///
/// so nothing large is ever subtracted from anything large, and `x = 0` gives exactly `0`
/// through `r = 0`. The window is `-1/2 <= x <= 1`, which is precisely where `|r| <= 1/3`
/// at *both* ends. Outside it the direct form's loss is only `2 eps / |x| <= 4 eps` and
/// that is what runs.
///
/// R evaluates the same series as a continued fraction because scalar iteration is cheap
/// for it. Here a fixed [`log1pmx_terms`]-long Horner chain is ~17 FMAs, which beats both
/// the continued fraction and the `ln_1p` it replaces, so the window is taken as wide as
/// the series allows rather than as narrow as possible.
///
/// # Precision policy
///
/// `Medium` and below drop the series entirely and return the direct `ln_1p(x) - x`, which
/// costs one compare, one blend and the whole Horner chain less. Understand what that
/// buys and what it gives up: the direct form is accurate to a few ulp for `$|x|$` down to
/// about `$10^{-3}$`, so for a caller sampling ordinary arguments the low tiers are simply
/// cheaper. Near zero it does not degrade gracefully. It degenerates. Once `$1 + x$`
/// rounds to `1` the result is `$-x$`, which is not a less precise `$-x^2/2$` but a
/// different quantity, wrong by every digit and by an unbounded factor. A caller whose
/// arguments approach zero (the case this function exists for at all) must ask for
/// `Average` or better, where the series makes `x = 0` exact.
#[inline(always)]
pub fn log1pmx_internal<V, E, P>(x: V) -> V
where
    E: FloatElementWithBits,
    V: FloatVectorWithBits<Element = E> + SpecializedTranscendentalMath<E>,
    P: Policy,
{
    // See the policy note above: below Average this is the whole function.
    if const { P::POLICY.precision.le(PrecisionPolicy::Medium) } {
        return V::ln_1p::<P>(x) - x;
    }

    let near = x.cmp_le(V::ONE) & x.cmp_ge(V::splat(const { E::ConstRatio::<{ -1 }, 2>::VALUE }));

    // NaN compares false, so it lands on the direct arm and propagates from there.
    let far = if const { !P::POLICY.avoid_branching } && crate::likely(near.all()) {
        V::ZERO
    } else {
        let d = V::ln_1p::<P>(x) - x;

        if const { !P::POLICY.avoid_branching } && crate::unlikely(near.none()) {
            return d;
        }

        d
    };

    // r = x / (2 + x). Outside the window this can be non-finite (x = -2 divides by zero),
    // which is why the tail is a blend and not arithmetic: the bad lanes are discarded.
    let r = x / (x + V::TWO);
    let y = r * r;

    let terms = const { log1pmx_terms(E::MANTISSA_BITS) };

    // S(y) = sum_{k=0..terms-1} y^k / (2k+3), Horner.
    let mut s = V::splat(E::from_ratio(1, 2 * terms as crate::LargeInt + 1));
    let mut k = terms - 1;
    while k > 0 {
        k -= 1;
        s = y.mul_adde(s, V::splat(E::from_ratio(1, 2 * k as crate::LargeInt + 3)));
    }

    let series = (y + y).mul_adde(s, -x) * r;

    if const { !P::POLICY.avoid_branching } && crate::likely(near.all()) {
        return series;
    }

    near.select(series, far)
}

/// `numer / sum(1/x_i)` with every reciprocal scaled by the smallest input, the real-vector
/// override behind [`SpecializedCoreMath::harmonic_mean`] and
/// [`SpecializedCoreMath::inv_sum_inv`].
///
/// Written directly, `sum(1/x_i)` overflows the moment any input is denormal: the reciprocal
/// saturates to infinity, the sum with it, and the answer collapses to zero when the true
/// value is merely small. Scaling by `m = min(x_i)` makes every term `m/x_i <= 1` by
/// construction, so the sum lands in `[1, N]` and cannot overflow whatever the spread of the
/// inputs. Recovering the answer is exact, since `sum(1/x_i) = s/m`.
///
/// It has to be the *smallest* element. The largest reciprocal is the one that overflows, so
/// it is the one that must normalize to 1; scaling by the largest input, which is what
/// `hypot_n` does for the opposite reason, would leave the failure exactly where it was.
/// Measured against a 60-digit oracle, this is exact across the full representable spread
/// (`5e-324` against `1e300`) where the direct form returns zero.
///
/// The scaling is what costs the two guards the direct form does not need: at `m = 0` every
/// term is `0/x_i` and the sum is `0`, giving `0/0` where the limit is `0`, and at an
/// all-infinite input every term is `inf/inf = NaN` where the limit is infinite.
///
/// This is confined to real vectors on purpose. `Complex::min` is lexicographic by
/// `(re, im)`, so it can return a large-magnitude element and the scaling would protect
/// nothing, so composites take the direct form instead.
#[inline(always)]
pub fn inv_sum_inv_internal<V, E: FloatElement, P, const N: usize>(mut values: [V; N], numer: V) -> V
where
    V: FloatVectorWithBits<Element = E> + SpecializedCoreMath<E>,
    P: Policy,
{
    if const { P::POLICY.precision.lt(PrecisionPolicy::Average) } {
        return V::inv_sum_inv_direct::<P, N>(values, numer);
    }

    // `reduce_array` copies, so `values` is still intact after this.
    let m = crate::math::algorithms::reduce_array(values, |a, b| a.min(b));

    let mut i = 0;
    while i < N {
        values[i] = m / values[i];
        i += 1;
    }

    crate::math::algorithms::reduce_in_place(&mut values, |a, b| a + b);

    let r = (numer * m) / values[0];

    if const { !P::POLICY.check_overflow } {
        return r;
    }

    m.cmp_eq(V::INFINITY)
        .select(V::INFINITY, m.cmp_eq(V::ZERO).select(V::ZERO, r))
}

/// `$\operatorname{atanh}(x)/x$`, the shared body behind
/// [`SpecializedTranscendentalMath::atanhc`].
///
/// Same shape as [`sinc_internal`] and [`sinhc_internal`], with the even series
/// `$1 + x^2/3 + x^4/5$` in place of theirs. As with those two, the window is not about
/// cancellation (`atanh(x)` is already `x` to full relative precision near zero, so the
/// quotient is accurate wherever it is defined), it is about the `0/0` at the origin, and
/// about not paying for an `atanh` to learn that the answer is 1.
///
/// The domain is `[-1, 1]`: `atanh(+-1)` is a signed infinity and the division by `x`
/// restores the sign, so both ends come out `+inf` with no patch, and `|x| > 1` is NaN from
/// `atanh` itself. Unlike `sinc` there is no infinite argument to guard, since anything past
/// 1 is out of domain already.
#[inline(always)]
pub fn atanhc_internal<V, E: FloatElement, P>(x: V) -> V
where
    V: FloatVectorWithBits<Element = E> + SpecializedTranscendentalMath<E>,
    P: Policy,
{
    if const { P::POLICY.precision.le(PrecisionPolicy::Medium) } {
        let mut y = V::atanh::<P>(x).approx_div_p::<P>(x);

        if const { P::POLICY.check_overflow } {
            y = x.is_zero().select(V::ONE, y);
        }

        return y;
    }

    let is_tiny = x.abs().cmp_le(V::FOURTH_ROOT_EPSILON);

    let x2 = x.square();

    // if branching, use the Taylor series for tiny x without calling atanh.
    if const { !P::POLICY.avoid_branching } && crate::unlikely(is_tiny.all()) {
        if const { P::POLICY.precision.le(PrecisionPolicy::Average) } {
            return x2.mul_adde(
                x2.mul_adde(V::splat(FloatElement::from_ratio(1, 5)), V::FRAC_1_3),
                V::ONE,
            );
        }

        let res = x2 / V::splat(const { E::ConstInt::<{ 5 }>::VALUE });
        return x2.mul_adde(res + V::FRAC_1_3, V::ONE);
    }

    // For very small x, atanhc(x) ~ 1 + x^2/3 + x^4/5
    let num = is_tiny.select(x2, V::atanh::<P>(x));
    let den = is_tiny.select(V::splat(const { E::ConstInt::<{ 5 }>::VALUE }), x);

    // combined division, since division is expensive
    let mut y = num.approx_div_p::<P>(den);

    y = is_tiny.select(x2.mul_adde(y + V::FRAC_1_3, V::ONE), y);

    y
}
