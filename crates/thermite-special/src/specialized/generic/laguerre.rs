use thermite::{
    element::FloatElement,
    math::{CoreMathWithPolicy as _, TranscendentalMathWithPolicy as _, policy::Policy},
    prelude::*,
};

use crate::specialized::SpecializedSpecialMath;

use super::poisson;

/// The weight, as a vector, from whichever of the two arguments `INT_ALPHA` selects.
///
/// The integer form is a scalar `i32`, so everything derived from it is a scalar constant
/// that folds to a literal whenever the caller's `alpha` is compile-time known, which the
/// float form cannot do, because the vector paths it feeds (`lgamma`, `cmp`) are built on
/// intrinsics LLVM does not constant fold.
#[inline(always)]
fn weight<E, V, const INT_ALPHA: bool>(alpha: V, alpha_int: i32) -> V
where
    E: FloatElement,
    V: FloatVector<Element = E>,
{
    if const { INT_ALPHA } {
        V::splat(E::from_int(alpha_int as thermite::LargeInt))
    } else {
        alpha
    }
}

/// `(s_k, 1/s_k)` for `s_k = sqrt((k+1)(k+alpha+1))`, the factor the recurrence divides by.
///
/// Under `INT_ALPHA` the radicand is a scalar product, and both results fold to literals
/// at a compile-time weight. The product is formed in `E` rather than in `LargeInt` on
/// purpose: `from_int` panics on a value it cannot represent exactly, and
/// `(k+1)(k+alpha+1)` leaves 2^53 for large `alpha` while the two factors separately never
/// do.
///
/// The `sqrt` and divide are taken on the *splat*, not in `E`: `FloatElement::sqrt` is
/// `libm`'s legacy-encoded `sqrtsd` under `no_std`, and one of those inside an AVX body
/// costs an SSE/AVX transition each way, measured at 30x on the whole function with a
/// runtime weight (at a literal one LLVM hoists the pure asm out of the loop and hides
/// it). A vector `sqrt` of a splat is the same one instruction, VEX-encoded, and folds.
#[inline(always)]
fn step_scale<P, E, V, const INT_ALPHA: bool>(k: usize, alpha: V, alpha_int: i32) -> (V, V)
where
    P: Policy,
    E: FloatElement,
    V: FloatVector<Element = E> + SpecializedSpecialMath<E>,
{
    if const { INT_ALPHA } {
        let k1 = E::from_int((k + 1) as thermite::LargeInt);
        let ka1 = E::from_int(k as thermite::LargeInt + alpha_int as thermite::LargeInt + 1);
        let s = V::splat(k1 * ka1).sqrt();
        (s, V::ONE / s)
    } else {
        let k1 = V::splat(E::from_int((k + 1) as thermite::LargeInt));
        let s = (k1 * (k1 + alpha)).sqrt();
        (s, s.approx_reciprocal_p::<P>())
    }
}

/// `2k + alpha + 1`, the `k`-dependent part of the recurrence's leading coefficient.
#[inline(always)]
fn two_k_a1<E, V, const INT_ALPHA: bool>(k: usize, a1: V, alpha_int: i32) -> V
where
    E: FloatElement,
    V: FloatVector<Element = E>,
{
    if const { INT_ALPHA } {
        V::splat(E::from_int(
            2 * k as thermite::LargeInt + alpha_int as thermite::LargeInt + 1,
        ))
    } else {
        V::splat(E::from_int(2 * k as thermite::LargeInt)) + a1
    }
}

/// `g = x^{alpha/2} e^{-x/4} / sqrt(alpha!)` for a small positive integer weight, as a
/// **product** rather than the exponential of a combined log (see [`seed`]).
///
/// With `alpha` a scalar integer every piece is cheap and most of it is uniform:
/// `1/sqrt(alpha!)` is a scalar factorial loop, one `sqrt` and one divide of a splat
/// (a literal at a compile-time weight), and `x^{alpha/2}` is `powi` by squaring at a
/// uniform exponent, `log2(alpha/2)` vector multiplies, plus one vector `sqrt` when
/// `alpha` is odd. That replaces a vector `ln`, `lgamma` and `exp`, and is *more*
/// accurate, not less: `lgamma(alpha+1)` is `O(alpha ln alpha)` and its half-ulp
/// absolute error becomes that many ulp of relative error once exponentiated, whereas
/// the product's error is a handful of roundings.
///
/// The price is range, which is why it is capped per arithmetic
/// (`LAGUERRE_PRODUCT_SEED_CAP`). Two things must hold: `alpha!` is finite (`alpha <= 170`
/// binary64, `34` binary32), and `x^{alpha/2}` is finite wherever `f = e^{-x/4}` has not
/// yet underflowed to zero (`x` under about 2980 / 416), so a finite `f` never meets an
/// infinite power: `2980^88` and `416^14` fit, giving `alpha <= 170` and `alpha <= 29`.
/// Past those `x` the true `l_0` is zero to the last denormal and `f` is exactly `0`,
/// so the one place the power *can* overflow is masked to `0` rather than `inf * 0`.
///
/// `x = 0` needs no patch here: `0^h = 0` and `sqrt 0 = 0` give the `l_0(0) = 0` limit
/// for `alpha > 0` directly (`alpha = 0` never reaches this function).
#[inline(always)]
fn product_seed<P, E, V>(x: V, f: V, alpha_int: i32) -> V
where
    P: Policy,
    E: FloatElement,
    V: FloatVector<Element = E> + SpecializedSpecialMath<E>,
{
    // Exact up to 22! (binary64) / 13! (binary32), a rounding per step beyond, and the sqrt
    // halves whatever accumulated. Rooted on the splat, not in E. See `step_scale`.
    let mut fact = <E as thermite::register::Element>::ONE;
    let mut i = 2;
    while i <= alpha_int {
        fact = fact * E::from_int(i as thermite::LargeInt);
        i += 1;
    }
    let c = V::ONE / V::splat(fact).sqrt();

    let mut p = x.powi_p::<P>(alpha_int / 2);
    if alpha_int & 1 != 0 {
        p *= x.sqrt();
    }

    // (x^{alpha/2} c) f: the power is O(1) or huge, c is O(1) or tiny, f is O(1) or tiny,
    // so scale the power down before the seed's own decay is applied.
    let g = (p * c) * f;

    // A zero f is the one place p may be inf (see above); the limit there is 0.
    f.is_zero().select(V::ZERO, g)
}

/// The two halves of `$l_0^{(\alpha)}(x) = x^{\alpha/2} e^{-x/2} / \sqrt{\Gamma(\alpha+1)}$`, as
/// `(f, g)` with `f = e^{-x/4}`, so `l_0 = g * f`.
///
/// The same half-split as the Hermite seed and for the same reason (see `hermite::seed`),
/// though the exponent is linear here so the ranges are far more generous: the seed
/// underflows and `l_k / f` overflows at `x/4` past the exponent range, i.e. `x` under
/// about 350 (binary32) or 2800 (binary64) is full accuracy at every degree, which covers
/// every degree up to roughly 87 / 700 everywhere on the half-line (the turning point of
/// `l_n^{(\alpha)}` sits near `4n`).
///
/// `g` carries the whole parameter dependence `x^{alpha/2} / sqrt(Gamma(alpha+1))`, and
/// how it is formed decides both range and accuracy. Not as that literal product: near the
/// peak the two factors are enormous and tiny and cancel to `$O(1)$`, but separately
/// `x^{alpha/2}` overflows binary64 near `alpha = 250` while `1/sqrt(Gamma(alpha+1))`
/// underflows near `alpha = 320`, and their overlap is `inf * 0 = NaN`. And not as the
/// exponential of the combined log `alpha/2 ln x - lgamma(alpha+1)/2 - x/4` either, which
/// keeps range but turns `lgamma`'s half-ulp *absolute* error, `O(alpha ln alpha)` in size,
/// into that many ulp of relative error. Instead, by weight:
///
/// - `alpha = 0`, the ordinary Laguerre function and the common case: `g = f` exactly.
/// - integer `alpha` up to `LAGUERRE_PRODUCT_SEED_CAP`: the direct product with an exact
///   factorial, [`product_seed`]. Cheapest and most accurate, capped where it could overflow.
/// - any other weight: Loader's saddle-point form of the Poisson mass, [`real_seed`], with
///   `alpha >= 9` directly and `alpha < 9` after a shift into the Stirling region with an
///   exact product. No `lgamma` anywhere, and no `ln` at all near the peak.
///
/// The Gamma is unavoidable in general: `alpha` is a free parameter of the family and
/// `Gamma(alpha+1)` is literally the `n = 0` normalization, which is why the Hermite seed
/// needs no such call (its `n = 0` constant is the parameter-free `pi^{-1/4}`).
#[inline(always)]
fn seed<P, E, V, const INT_ALPHA: bool>(x: V, alpha: V, alpha_int: i32) -> (V, V)
where
    P: Policy,
    E: FloatElement,
    V: FloatVector<Element = E> + SpecializedSpecialMath<E>,
{
    let f = (x * V::splat(<E as FloatElement>::ConstRatio::<{ -1 }, 4>::VALUE)).exp_p::<P>();

    // alpha = 0 is the ordinary Laguerre function and by far the common weight, and there
    // x^{alpha/2} / sqrt(Gamma(alpha+1)) is exactly 1, leaving g = f.
    if const { INT_ALPHA } {
        // A scalar test on a scalar argument: no vector compare, no reduction, and one that
        // disappears outright at a literal weight, taking the ln, lgamma and second exp
        // with it. This is the case the integer mode exists for.
        if alpha_int == 0 {
            return (f, f);
        }
        // Small positive integer weights: no ln, lgamma or second exp at all.
        if alpha_int > 0 && alpha_int <= V::LAGUERRE_PRODUCT_SEED_CAP {
            return (f, product_seed::<P, E, V>(x, f, alpha_int));
        }
    } else {
        // The vector form has to reduce a mask, which stays a *runtime* branch even at
        // a visible-constant zero: on AVX2 the compare is `_mm256_cmp_pd`, which LLVM does
        // not constant fold, so the general path remains compiled behind it.
        if const { !P::POLICY.avoid_branching } && alpha.is_zero().all() {
            return (f, f);
        }
    }

    let a = weight::<E, V, INT_ALPHA>(alpha, alpha_int);

    // Above the product cap the integer weight is far past STIRLERR_MIN, so the shift into
    // the Stirling region folds away.
    if const { INT_ALPHA } {
        return (f, real_seed::<P, E, V, true>(x, f, a));
    }

    (f, real_seed::<P, E, V, false>(x, f, alpha))
}

/// `g = x^{alpha/2} e^{-x/4} / sqrt(Gamma(alpha+1))` for a real weight, every lane
/// through one shared path: `l_0^2` is the Poisson mass at `k = alpha`, mean `x`, so
///
/// ```text
/// g = l_0 e^{x/4} = sqrt(P(alpha; x)) e^{x/4}
///   = exp(rest/2 + base) * sqrt(prod) * (2 pi n)^{-1/4}
/// ```
///
/// with `(rest, large, prod, n)` from [`poisson::pmf_parts`]: `alpha >= 9` is Loader's
/// saddle-point form (`n = alpha`, `prod = 1`, and near the peak `x ~ alpha` a series with
/// nothing large in it), `alpha < 9` is the same Stirling machinery after a shift
/// `n = alpha + m` with `prod = (alpha+1)...(alpha+m)`, and no `lgamma` anywhere. On a mixed
/// vector the two share the one `ln`, `stirlerr(n)`, the one `exp`, the TwoSum and the one
/// `inverse_sqrt`. Only the shift (plus its `ln x`) and the peak series are
/// branch-specific, and both are skipped when no lane needs them.
///
/// `base` is `+x/4` on `alpha >= 9` lanes (their `rest` already contains `-x`) and `-x/4`
/// on shifted lanes (their `rest` leaves `-x` out, see `pmf_parts`); either way it is the one
/// large term, and [`poisson::exp_two_sum`] keeps its rounding out of the result, since the
/// exponent is up to ~700 and half an ulp of that is hundreds of ulp after the `exp`.
///
/// Pins: `alpha = 0` lanes are set to exactly `f` (the shifted form would be `1` only to a
/// few ulp, and a lane's result must not depend on whether its neighbors let the vector
/// take the `g = f` shortcut), and `x = 0` lanes are `0` for `alpha > 0` (`ln 0` makes the
/// exponent `-inf`, but the TwoSum on it is `inf - inf`) and `1` at `alpha = 0`.
#[inline(always)]
fn real_seed<P, E, V, const ALL_LARGE: bool>(x: V, f: V, a: V) -> V
where
    P: Policy,
    E: FloatElement,
    V: FloatVector<Element = E> + SpecializedSpecialMath<E>,
{
    let quarter = V::splat(<E as FloatElement>::ConstRatio::<1, 4>::VALUE);

    let (rest, large, prod, n) = poisson::pmf_parts::<P, E, V, ALL_LARGE>(a, x);

    let x4 = x * quarter;
    let g = poisson::exp_two_sum::<P, E, V>(x4.neg_c(!large), rest * V::HALF);
    let g = g * (n * V::splat(E::TAU)).sqrt().inverse_sqrt_p::<P>();
    // sqrt(prod) is 1 wherever no lane was shifted, and a sqrt is not free.
    let g = if const { ALL_LARGE } || (const { !P::POLICY.avoid_branching } && large.all()) {
        g
    } else {
        g * prod.sqrt()
    };

    let g = x.is_zero().select(V::ZERO, g);
    if const { ALL_LARGE } {
        g
    } else {
        a.is_zero().select(f, g)
    }
}

/// The orthonormal generalized Laguerre function
/// `$l_N^{(\alpha)}(x) = \sqrt{N!/\Gamma(N+\alpha+1)}\, x^{\alpha/2} e^{-x/2} L_N^{(\alpha)}(x)$`.
///
/// Three-term recurrence on the functions themselves, with `s_k = sqrt((k+1)(k+alpha+1))`:
///
/// ```text
/// l_{k+1} = ((2k + alpha + 1 - x) l_k - s_{k-1} l_{k-1}) / s_k
/// ```
///
/// which keeps every intermediate `O(1)`. The `s_k` depend only on `k` and the weight, not
/// on the running values, so they sit beside the recurrence rather than on its critical
/// path. Under `INT_ALPHA` they are scalars and fold to literals at a compile-time weight;
/// otherwise each step carries a vector `sqrt` and reciprocal. See [`seed`] for the range.
#[inline(always)]
pub fn laguerre_function_n<P, E, V, const N: usize, const INT_ALPHA: bool>(x: V, alpha: V, alpha_int: i32) -> V
where
    P: Policy,
    E: FloatElement,
    V: FloatVector<Element = E> + SpecializedSpecialMath<E>,
{
    let (f, g0) = seed::<P, E, V, INT_ALPHA>(x, alpha, alpha_int);

    if const { N == 0 } {
        return g0 * f;
    }

    let a = weight::<E, V, INT_ALPHA>(alpha, alpha_int);
    let a1 = a + V::ONE;

    // s_0 = sqrt(alpha + 1); l_1 = (alpha + 1 - x) l_0 / s_0
    let (mut s_prev, d0) = step_scale::<P, E, V, INT_ALPHA>(0, alpha, alpha_int);
    let mut p0 = g0;
    let mut p1 = ((a1 - x) * g0) * d0;

    let mut k = 1;
    while k < N {
        let (s_k, d_k) = step_scale::<P, E, V, INT_ALPHA>(k, alpha, alpha_int);

        // ((2k + alpha + 1 - x) l_k - s_{k-1} l_{k-1}) / s_k
        let b = two_k_a1::<E, V, INT_ALPHA>(k, a1, alpha_int) - x;
        let next = b.mul_sube(p1, s_prev * p0) * d_k;

        s_prev = s_k;
        p0 = p1;
        p1 = next;

        k += 1;
    }

    p1 * f
}

/// The runtime-degree twin of [`laguerre_function_n`].
///
/// The same seed, the same recurrence and the same backward `s_k` order, with the degree as a
/// value. Under `INT_ALPHA` the per-step scales are computed rather than folded, which is the
/// only cost.
#[inline(always)]
pub fn laguerre_function<P, E, V, const INT_ALPHA: bool>(x: V, alpha: V, alpha_int: i32, n: u32) -> V
where
    P: Policy,
    E: FloatElement,
    V: FloatVector<Element = E> + SpecializedSpecialMath<E>,
{
    let (f, g0) = seed::<P, E, V, INT_ALPHA>(x, alpha, alpha_int);

    if n == 0 {
        return g0 * f;
    }

    let a = weight::<E, V, INT_ALPHA>(alpha, alpha_int);
    let a1 = a + V::ONE;

    let (mut s_prev, d0) = step_scale::<P, E, V, INT_ALPHA>(0, alpha, alpha_int);
    let mut p0 = g0;
    let mut p1 = ((a1 - x) * g0) * d0;

    let n = n as usize;
    let mut k = 1;
    while k < n {
        let (s_k, d_k) = step_scale::<P, E, V, INT_ALPHA>(k, alpha, alpha_int);

        let b = two_k_a1::<E, V, INT_ALPHA>(k, a1, alpha_int) - x;
        let next = b.mul_sube(p1, s_prev * p0) * d_k;

        s_prev = s_k;
        p0 = p1;
        p1 = next;

        k += 1;
    }

    p1 * f
}

/// Runtime-length form of [`laguerre_function_series`].
///
/// A genuine port of the recurrence rather than a fold over the const kernel: a series
/// carries `k`-dependent state and does not partition the way the slice reductions in
/// `thermite` do. Both forms must be edited together.
///
/// Same pre-scaling, same seed, same backward `s_k` order. Read
/// [`laguerre_function_series`] for the reasoning. `INT_ALPHA` still selects the
/// integer-weight path, but the weights are no longer folded literals at any `alpha`,
/// since `k` is not a constant, so the per-step `sqrt` is paid in full here.
///
/// The empty series is `0`, where the const form refuses to compile.
#[inline(always)]
pub fn laguerre_function_series_slice<P, E, V, const INT_ALPHA: bool>(x: V, alpha: V, alpha_int: i32, coeffs: &[E]) -> V
where
    P: Policy,
    E: FloatElement,
    V: FloatVector<Element = E> + SpecializedSpecialMath<E>,
{
    let n = coeffs.len();

    if n == 0 {
        return V::ZERO;
    }

    let (f, g0) = seed::<P, E, V, INT_ALPHA>(x, alpha, alpha_int);

    if n == 1 {
        return (f * V::splat(coeffs[0])) * g0;
    }

    let a1 = weight::<E, V, INT_ALPHA>(alpha, alpha_int) + V::ONE;

    let mut y2 = V::ZERO;
    let mut y1 = f * V::splat(coeffs[n - 1]);

    let mut k = n - 1;
    while k > 1 {
        k -= 1;
        let (s_k, d_k) = step_scale::<P, E, V, INT_ALPHA>(k, alpha, alpha_int);
        let (_, d_k1) = step_scale::<P, E, V, INT_ALPHA>(k + 1, alpha, alpha_int);

        let alpha_k = (two_k_a1::<E, V, INT_ALPHA>(k, a1, alpha_int) - x) * d_k;
        let ratio_k1 = s_k * d_k1;

        let yk = alpha_k.mul_adde(y1, y2.nmul_adde(ratio_k1, f * V::splat(coeffs[k])));
        y2 = y1;
        y1 = yk;
    }

    let (s0, d0) = step_scale::<P, E, V, INT_ALPHA>(0, alpha, alpha_int);
    let (_, d1) = step_scale::<P, E, V, INT_ALPHA>(1, alpha, alpha_int);
    let alpha_0 = (a1 - x) * d0;
    let ratio_1 = s0 * d1;

    alpha_0.mul_adde(y1, y2.nmul_adde(ratio_1, f * V::splat(coeffs[0]))) * g0
}

/// Clenshaw summation of a Laguerre-function series, `$\sum_{k=0}^{N-1} c_k l_k^{(\alpha)}(x)$`.
///
/// Clenshaw over `l_k / l_0`, with the coefficients pre-scaled by `f = e^{-x/4}` and the
/// outer factor reduced to `g = l_0 / f`, the same split as [`laguerre_function`]. With
/// `alpha_k = (2k + alpha + 1 - x) / s_k` and `beta_k = -s_{k-1} / s_k`:
///
/// ```text
/// y_k = f c_k + alpha_k y_{k+1} + beta_{k+1} y_{k+2}      k = N-1 down to 1
/// S   = g * (f c_0 + alpha_0 y_1 + beta_1 y_2)
/// ```
///
/// The recurrence runs backward, so `s_k` is needed at step `k` and `s_{k-1}` one step
/// later, the opposite order from the forward kernel. `s_{k-1}` is recomputed rather than
/// carried, since it is one `sqrt` off the critical path either way (and a folded literal
/// under `INT_ALPHA` at a compile-time weight).
#[inline(always)]
pub fn laguerre_function_series<P, E, V, const N: usize, const INT_ALPHA: bool>(
    x: V,
    alpha: V,
    alpha_int: i32,
    coeffs: &[E; N],
) -> V
where
    P: Policy,
    E: FloatElement,
    V: FloatVector<Element = E> + SpecializedSpecialMath<E>,
{
    const {
        assert!(N >= 1, "laguerre_function_series: N must be at least 1");
    }

    let (f, g0) = seed::<P, E, V, INT_ALPHA>(x, alpha, alpha_int);

    // S = c_0 l_0
    if const { N == 1 } {
        return (f * V::splat(coeffs[0])) * g0;
    }

    let a1 = weight::<E, V, INT_ALPHA>(alpha, alpha_int) + V::ONE;

    // Top step, k = N-1: y = f c_{N-1}. Below it, every step needs alpha_k and beta_{k+1}.
    let mut y2 = V::ZERO;
    let mut y1 = f * V::splat(coeffs[N - 1]);

    // k = N-2 down to 1.
    let mut k = N - 1;
    while k > 1 {
        k -= 1;
        let (s_k, d_k) = step_scale::<P, E, V, INT_ALPHA>(k, alpha, alpha_int);
        let (_, d_k1) = step_scale::<P, E, V, INT_ALPHA>(k + 1, alpha, alpha_int);

        // alpha_k = (2k + alpha + 1 - x) / s_k and beta_{k+1} = -s_k / s_{k+1}, both off
        // the chain. beta is a genuine runtime value, so its sign is taken by the FMA form.
        let alpha_k = (two_k_a1::<E, V, INT_ALPHA>(k, a1, alpha_int) - x) * d_k;
        let ratio_k1 = s_k * d_k1;

        let yk = alpha_k.mul_adde(y1, y2.nmul_adde(ratio_k1, f * V::splat(coeffs[k])));
        y2 = y1;
        y1 = yk;
    }

    // S = g0 * (alpha_0 y_1 + (f c_0 + beta_1 y_2)); s_0 = sqrt(alpha+1), s_1 = sqrt(2(alpha+2)).
    let (s0, d0) = step_scale::<P, E, V, INT_ALPHA>(0, alpha, alpha_int);
    let (_, d1) = step_scale::<P, E, V, INT_ALPHA>(1, alpha, alpha_int);
    let alpha_0 = (a1 - x) * d0;
    let ratio_1 = s0 * d1;

    alpha_0.mul_adde(y1, y2.nmul_adde(ratio_1, f * V::splat(coeffs[0]))) * g0
}
