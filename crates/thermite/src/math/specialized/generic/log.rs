//! Logarithmic kernels: `ln(1 - e^-x)`, `log_n`, and `ln(1+x) - x`.

use super::super::*;

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
/// k=7, and `1e-17` at k=16. That gives 17 terms in binary64 and 8 in binary32, all FMAs,
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
