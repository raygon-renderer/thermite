//! The trigonometric integrals `Si(x) = int_0^x sin(t)/t dt` and
//! `Ci(x) = gamma + ln x + int_0^x (cos t - 1)/t dt`.
//!
//! # Two regions
//!
//! Below a crossover (12 in f64, 6 in f32) `Si(x)/x` and `Cin(x)/x^2` are smooth
//! functions of `v = x^2`, fitted as Chebyshev series and summed by Clenshaw, where
//! `Cin = gamma + ln x - Ci` is the entire part, the piece that is _not_ the
//! logarithmic singularity. Above it, the auxiliary form
//!
//! ```text
//! Si = pi/2 - f cos x - g sin x,   Ci = f sin x - g cos x
//! ```
//!
//! with `f = P(v)/x` and `g = Q(v)/x^2` for `v = 1/x^2`, both Horner polynomials
//! tending to 1.
//!
//! # Why the crossover is so far out
//!
//! Because these auxiliaries are harder than the Fresnel ones, and for a structural
//! reason worth recording. `f(x) = int_0^inf e^{-xt}/(1+t^2) dt` (verified to 12
//! digits), so `P(v) = x f` is a _Stieltjes_ function: its asymptotic series
//! `sum (-1)^k (2k)! v^k` diverges, and the branch cut reaches `v = 0`. Polynomial
//! convergence at that endpoint is therefore sub-geometric. It shows.
//! Measured degree for f64, at contribution-weighted targets:
//!
//! | range | deg P | amplification | deg Q |
//! |---|---|---|---|
//! | `x >= 6` | 33 | 3.6e8 | 31 |
//! | `x >= 8` | 25 | 455 | 24 |
//! | `x >= 10` | 20 | 1.19 | 19 |
//! | `x >= 12` | 17 | 1.03 | 16 |
//!
//! One fit at `x >= 12` beats the multi-range ladders that were also measured
//! (`[8,20]` plus `[20,inf)` is 28 terms and a select, against 17), and Pade of the
//! divergent series is no better. `[10/10]` reaches only 8.8e-8 at `x >= 8`.
//!
//! `P` is fitted at plain relative accuracy because `|Ci| ~ f`, so `f`'s error is
//! the result's error. `Q` is relaxed by a factor `x`, contributing at `1/x^2`
//! against a `1/x` result. Re-fitting `Q` at plain relative accuracy adds degrees
//! and buys nothing.
//!
//! # Accuracy
//!
//! Measured against mpmath at 45 digits over `x` from 1e-4 to 1e15 (f64) and 1e7
//! (f32): `Si` 2.03 ulp f64 / 1.34 f32, `Ci` 1.42 / 1.99 relative to its envelope.
//!
//! Two contract points belong in the caller's head:
//!
//! - **`Ci` has zeros**, the first near `x = 0.6165`, and nothing is relatively
//!   accurate at one. The grading above is against `|gamma + ln x| + |Cin|` below
//!   the crossover and `1/x` above it, which is what the arithmetic can actually
//!   deliver.
//! - **Large-`x` accuracy inherits `sin_cos`'s argument reduction.** For `Ci` the
//!   oscillation _is_ the value, so a phase error is a relative error. Full
//!   reduction is a `Best`-tier property in this library, and below that `Ci`'s
//!   accuracy at large `x` degrades with it. `Si` is insulated: it tends to `pi/2`
//!   and the oscillation is a correction of size `1/x`.
//!
//! `Si` is `pi/2` to within half an ulp above `x = 1.147e16` (f64) / `2.136e7`
//! (f32). `Ci` has no such cutoff: it decays like `1/x` and stays representable for
//! every finite argument.

use thermite::{
    element::FloatElement,
    math::{
        CoreMathWithPolicy as _, TranscendentalMathWithPolicy as _, policy::Policy,
        specialized::SpecializedTranscendentalMath,
    },
    prelude::*,
};

use super::chebyshev::chebyshev_series;

/// `(Si(x), Ci(x))`.
///
/// `cheb_si` and `cheb_cin` are Chebyshev coefficients for `Si(x)/x` and
/// `Cin(x)/x^2` in `v = x^2` mapped onto `[-1, 1]` by `v*map - 1`. `aux_p` and
/// `aux_q` are ascending monomial coefficients in `v = 1/x^2`.
#[inline(always)]
pub fn sici_with<P, E, V, const NSI: usize, const NCI: usize, const NP: usize, const NQ: usize>(
    x: V,
    x0: E,
    map: E,
    cutoff: E,
    cheb_si: &[E; NSI],
    cheb_cin: &[E; NCI],
    aux_p: &[E; NP],
    aux_q: &[E; NQ],
) -> (V, V)
where
    E: FloatElement,
    V: FloatVector<Element = E> + SpecializedTranscendentalMath<E>,
    P: Policy,
{
    // `Si` is odd. `Ci(-x) = Ci(x) + i*pi`, so the real branch is the one at |x| and
    // the imaginary part is dropped. Both match SciPy's `sici`.
    let ax = x.abs();
    let is_small = ax.cmp_le(V::splat(x0));

    if !const { P::POLICY.avoid_branching } && is_small.all() {
        let (si, ci) = sici_small::<P, E, V, NSI, NCI>(ax, map, cheb_si, cheb_cin);
        return (si.copysign(x), ci);
    }

    // One division: rx = 1/x, then f = P(v)/x and g = Q(v)/x^2 with v = rx^2.
    let rx = ax.approx_reciprocal_p::<P>();
    let v = rx * rx;
    let f = v.poly_n_p::<P, NP>(aux_p) * rx;
    let g = v.poly_n_p::<P, NQ>(aux_q) * v;

    let (sin_x, cos_x) = ax.sin_cos_p::<P>();

    // Chained FMAs: pi/2 - f cos x - g sin x, and f sin x - g cos x.
    let mut si = g.nmul_adde(sin_x, f.nmul_adde(cos_x, V::FRAC_PI_2));
    let mut ci = g.nmul_adde(cos_x, f * sin_x);

    if const { P::POLICY.avoid_branching } || thermite::unlikely(is_small.any()) {
        let (ss, cs) = sici_small::<P, E, V, NSI, NCI>(ax, map, cheb_si, cheb_cin);
        si = is_small.select(ss, si);
        ci = is_small.select(cs, ci);
    }

    // `Si` converges: above the cutoff the oscillating correction is under half an ulp
    // of pi/2. `Ci` does NOT: it decays like 1/x and stays representable for every
    // finite argument, so it must keep coming out of the auxiliary branch. Clamping it
    // to zero on the same condition returns 0 for `Ci(9.9e8)`, whose true value is
    // -5.4e-10.
    si = ax.cmp_gt(V::splat(cutoff)).select(V::FRAC_PI_2, si);

    // At an actual infinity both limits have to be named: `rx` is 0, so `f` and `g`
    // vanish, but `sin_cos(inf)` is a NaN that would otherwise multiply through.
    let inf = ax.cmp_eq(V::INFINITY);
    si = inf.select(V::FRAC_PI_2, si);
    ci = inf.select(V::ZERO, ci);

    (si.copysign(x), ci)
}

/// `(Si, Ci)` from the Chebyshev series in `v = x^2`, for `|x|` under the crossover.
///
/// `Ci = (gamma + ln x) - x^2 B(v)` cancels near the zero at `x ~ 0.6165`, where the
/// two terms are equal and opposite. That is the function's own conditioning, not
/// the form's: `Ci` is genuinely zero there and no rearrangement recovers relative
/// accuracy. Splitting off `Cin` is what keeps everything _else_ accurate. It
/// isolates the logarithmic singularity, so the small-argument fit never has to
/// represent it.
#[inline(always)]
fn sici_small<P, E, V, const NSI: usize, const NCI: usize>(
    ax: V,
    map: E,
    cheb_si: &[E; NSI],
    cheb_cin: &[E; NCI],
) -> (V, V)
where
    E: FloatElement,
    V: FloatVector<Element = E> + SpecializedTranscendentalMath<E>,
    P: Policy,
{
    let v = ax * ax;
    let t = v.mul_sube(V::splat(map), V::ONE);
    let a = chebyshev_series::<P, E, V, 1, NSI, true>(t, cheb_si);
    let b = chebyshev_series::<P, E, V, 1, NCI, true>(t, cheb_cin);

    // ln(0) is -inf and v*b is 0 there, so Ci(0) = -inf falls out without a guard.
    let ci = (V::EULER_GAMMA + ax.ln_p::<P>()) - v * b;
    (ax * a, ci)
}
