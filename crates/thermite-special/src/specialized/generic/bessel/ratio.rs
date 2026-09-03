//! `A_nu(x) = I_nu(x) / I_{nu-1}(x)`, the modified Bessel ratio, and its inverse.
//!
//! # Where it comes from
//!
//! With `p = 2 nu` this is the mean resultant length of a von Mises-Fisher distribution
//! on the sphere `S^{p-1}` as a function of its concentration `kappa`, and its inverse is
//! the maximum-likelihood concentration from an observed mean resultant length: the one
//! step of every vMF fit that is not a matrix product. `p = 2` (`nu = 1`) is the von Mises
//! circle, `I_1/I_0`. `p = 3` (`nu = 3/2`) collapses to the elementary Langevin function
//! `coth x - 1/x`, which is the [`langevin`](crate::RealSpecialMath::langevin) family. The
//! order arrives as a plain vector, so those cases reach the integer and half-integer
//! Bessel kernels through the order simplifier, not the general real-order machinery.
//!
//! # Forward, in three arms
//!
//! The obvious quotient of the two scaled Bessel functions fails at small `x` when the
//! order is large: `e^{-x} I_nu(x)` is `(x/2)^nu / Gamma(nu+1)` there and underflows to
//! zero (`I_150(0.5)` is `1e-457`) while the ratio, about `x / 2nu`, is ordinary. And it
//! keeps failing well past that: the scaled `I_nu(x)` is below `e^{-0.17 nu}` at `x = nu`.
//!
//! - `x <= 0.9 sqrt(nu)`: the ratio of the two power series, `A = (x / 2nu) S_nu / S_{nu-1}`
//!   with `S_a = sum (x^2/4)^k / (k! (a+1)_k)`, twelve terms each (`1 / (4^12 12!)` at the
//!   edge). Never underflows: only the ratio is formed.
//! - `x < 8 nu`: the Perron continued fraction for `I_{nu}/I_{nu-1}`
//!   (`cf1_i_ratio`, the Bessel kernels' own), which converges for every `(nu, x)` in about
//!   `6 sqrt(x)` iterations.
//! - else: the quotient of the scaled Bessels from [`bessel_iv`], where `x` dominates the
//!   order and both are ordinary numbers, and the asymptotic arm inside is fast.
//!
//! # The derivative
//!
//! From the recurrence `I_{nu-1} - I_{nu+1} = (2 nu / x) I_nu`:
//! `A' = 1 - A^2 - (2 nu - 1) A / x`, a closed form in `A` itself, which is what makes
//! the inverse a one-evaluation Newton and gives `Dual` its factor for free.
//!
//! # Inverse
//!
//! Banerjee's `kappa_0 = r (p - r^2) / (1 - r^2)` (within ten percent everywhere), then
//! `newtons_method` on `A(kappa) - r` with the derivative above, bracketed by a factor of two,
//! eight iterations at most. Sra (2012) found two steps from that seed reach working
//! precision, and the tolerance is a few ulp of `r`. Below `r = 1e-8` the answer is
//! `2 nu r` outright (the next term is `r^3`).
//!
//! The inverse is ill-conditioned as `r -> 1`, where `kappa ~ (p-1)/(2(1-r))`: an ulp of
//! `r` moves `kappa` by `2 kappa^2 eps / (p-1)`, so the _relative_ error grows like
//! `kappa`, and the result is the exact inverse of the given `r` only to that extent. A
//! complement form taking `1 - r`, the way `inv_langevin_1m` does for `p = 3`, is the
//! remedy and is not built.

use thermite::{
    element::FloatElement,
    math::{
        PrimalProjection,
        algorithms::newtons_method,
        policy::{Policy, policies::MaxIterations},
    },
    prelude::*,
};

use super::ik_real::cf1_i_ratio;
use crate::specialized::generic::ndtr::residual_tolerance;
use crate::BesselOrder;
use crate::specialized::SpecializedSpecialMath;

/// `A' = 1 - A^2 - (2 nu - 1) A / x` from `A` itself.
#[inline(always)]
pub fn bessel_i_ratio_deriv<P, E, V>(a: V, x: V, nu: V) -> V
where
    P: Policy,
    E: FloatElement,
    V: FloatVector<Element = E>,
{
    let two_nu_m1 = nu.mul_sube(V::TWO, V::ONE);
    a.nmul_adde(a, V::ONE) - two_nu_m1 * a / x
}

/// `I_nu(x) / I_{nu-1}(x)`, for `x >= 0` and `nu >= 1`. Odd in `x`.
#[inline(always)]
pub fn bessel_i_ratio_impl<P, E, V>(x: V, nu: V) -> V
where
    P: Policy,
    E: FloatElement,
    V: FloatVector<Element = E> + PrimalProjection<Primal = V> + SpecializedSpecialMath<E>,
{
    let neg = x.is_negative();
    let x = x.abs();

    let two_nu = nu + nu;
    let small = x.cmp_le(nu.sqrt() * V::splat(<E as FloatElement>::ConstRatio::<9, 10>::VALUE));
    let mid = !small & x.cmp_lt(nu * V::splat(<E as FloatElement>::ConstInt::<8>::VALUE));
    let big = !(small | mid);

    let mut a = V::ZERO;

    if const { P::POLICY.avoid_branching } || small.any() {
        // S_a = sum_k (x^2/4)^k / (k! (a+1)_k) for a = nu and a = nu - 1, twelve terms,
        // as the ratio of the running terms so the two sums share the loop.
        let q = x * x * V::FRAC_1_4;
        let mut term_hi = V::ONE;
        let mut term_lo = V::ONE;
        let mut s_hi = V::ONE;
        let mut s_lo = V::ONE;
        let mut k = V::ONE;
        let mut i = 0;
        while i < 12 {
            term_hi *= q / (k * (nu + k));
            term_lo *= q / (k * (nu + k - V::ONE));
            s_hi += term_hi;
            s_lo += term_lo;
            k += V::ONE;
            i += 1;
        }
        a = x / two_nu * (s_hi / s_lo);
    }

    if const { P::POLICY.avoid_branching } || mid.any() {
        // The continued fraction evaluates I_{a+1}/I_a, with a = nu - 1.
        let cf = cf1_i_ratio::<P, E, V, V>(nu - V::ONE, x, mid);
        a = mid.select(cf, a);
    }

    if const { P::POLICY.avoid_branching } || big.any() {
        let hi = <V as SpecializedSpecialMath<E>>::bessel_iv::<P, true>(x, BesselOrder::Real(nu));
        let lo = <V as SpecializedSpecialMath<E>>::bessel_iv::<P, true>(x, BesselOrder::Real(nu - V::ONE));
        a = big.select(hi / lo, a);
    }

    // A(0) = 0 (the series gives 0/1 * 1 = 0 exactly), A(inf) = 1, A(NaN) = NaN.
    let a = x.cmp_eq(V::INFINITY).select(V::ONE, a);
    a.neg_c(neg)
}

/// `A'` from the complement `c = 1 - A`, with `1 - A^2` as `c (2 - c)`: no cancellation
/// where `A` is within an ulp of 1.
#[inline(always)]
pub fn bessel_i_ratio_deriv_1m<P, E, V>(c: V, x: V, nu: V) -> V
where
    P: Policy,
    E: FloatElement,
    V: FloatVector<Element = E>,
{
    let two_nu_m1 = nu.mul_sube(V::TWO, V::ONE);
    c * (V::TWO - c) - two_nu_m1 * (V::ONE - c) / x
}

/// `1 - A_nu(x)`, the complement of the ratio, accurate where `A` is within an ulp of 1.
///
/// For `x >= max(8 nu, 20)` the complement is evaluated directly. The order is reduced to
/// `nu_0 = nu - K` in `[1, 2)`, where the Hankel expansions of `I_{nu_0}` and `I_{nu_0 - 1}`
/// converge to `e^{-2x}` (their smallest term is at `k ~ 2x`, 40 terms at `x = 20`, and
/// the difference of the two series is formed term by term, so `1 - N/D` is never
/// evaluated), then the ratio recurrence `A_{m+1} = 1/A_m - 2m/x` written for the
/// complement, `c_{m+1} = 2m/x - c_m/(1 - c_m)`, walks the `K` orders up. The two terms
/// of that step are `(2m)/x` and about `(2m-1)/(2x)`, so the subtraction costs a bit and
/// the walk is stable as long as `c` stays small, which `x >= 8 nu` guarantees at every
/// intermediate order. Elsewhere `c = 1 - A` from the forward: below `x = 8 nu` the
/// complement is at least `1/8` and `1 - A` is within `8 eps`. In the corner `8 nu <= x < 20`
/// (only `nu < 2.5`) it is within `2x eps / (2 nu - 1)`, forty ulp at worst.
///
/// `A` is odd, so `c(-x) = 2 - c(x)`.
#[inline(always)]
pub fn bessel_i_ratio_1m_impl<P, E, V>(x: V, nu: V) -> V
where
    P: Policy,
    E: FloatElement,
    V: FloatVector<Element = E> + PrimalProjection<Primal = V> + SpecializedSpecialMath<E>,
{
    let neg = x.is_negative();
    let x = x.abs();

    let asym = x.cmp_ge(nu * V::splat(<E as FloatElement>::ConstInt::<8>::VALUE))
        & x.cmp_ge(V::splat(<E as FloatElement>::ConstInt::<20>::VALUE));

    let mut c = V::ONE;
    if const { P::POLICY.avoid_branching } || !asym.all() {
        c = V::ONE - bessel_i_ratio_impl::<P, E, V>(asym.select(V::ONE, x), nu);
    }

    if const { P::POLICY.avoid_branching } || asym.any() {
        // Reduce the order into [1, 2).
        let k = (nu - V::ONE).floor();
        let nu0 = nu - k;

        // Hankel series of I_{nu0} (hi) and I_{nu0 - 1} (lo) in 1/x: a_j = a_{j-1} (mu - (2j-1)^2)/(8j),
        // alternating. `diff` accumulates D - N term by term, `d` accumulates D.
        let mu_hi = nu0 * nu0 * V::splat(<E as FloatElement>::ConstInt::<4>::VALUE);
        let nm1 = nu0 - V::ONE;
        let mu_lo = nm1 * nm1 * V::splat(<E as FloatElement>::ConstInt::<4>::VALUE);
        let neg_xinv = -(V::ONE / x);
        let mut a_hi = V::ONE;
        let mut a_lo = V::ONE;
        let mut pw = V::ONE;
        let mut d = V::ONE;
        let mut diff = V::ZERO;
        let mut j = 1u32;
        while j <= 40 {
            let odd = E::from_int((2 * j - 1) as i64 * (2 * j - 1) as i64);
            let scale = V::splat(E::ONE / E::from_int(8 * j as i64));
            a_hi *= (mu_hi - V::splat(odd)) * scale;
            a_lo *= (mu_lo - V::splat(odd)) * scale;
            pw *= neg_xinv;
            d = a_lo.mul_adde(pw, d);
            diff = (a_lo - a_hi).mul_adde(pw, diff);
            j += 1;
        }
        let mut cm = diff / d;

        // The walk up, c_{m+1} = 2m/x - c_m/(1 - c_m), amplifies an error by 1/(1 - c_m)^2 per
        // step, about e^{nu^2/x} over the whole walk: fine while nu^2 <= x, 1e6 at nu = 150
        // and x = 3000. Above that the _downward_ map c_m = y/(1 + y), y = 2m/x - c_{m+1},
        // contracts by (1 - c_m)^2 per step instead, so those lanes descend from an order
        // M = nu + J high enough that the start's error has decayed by e^{-(M^2 - nu^2)/x}:
        // J = ceil(sqrt(nu^2 + 32x) - nu), seeded with Amos's bound at M, whose own error
        // is a few parts in a thousand and is what the 32 buys down to 1e-16.
        let two_over_x = V::TWO / x;
        let up = asym & (nu * nu).cmp_le(x);
        let down = asym & !up;

        if const { P::POLICY.avoid_branching } || up.any() {
            let mut m = nu0;
            loop {
                let live = up & m.cmp_lt(nu - V::HALF);
                if live.none() {
                    break;
                }
                let next = (m * two_over_x) - cm / (V::ONE - cm);
                cm = live.select(next, cm);
                m += V::ONE;
            }
        }

        if const { P::POLICY.avoid_branching } || down.any() {
            let j = (nu
                .mul_adde(nu, x * V::splat(<E as FloatElement>::ConstInt::<32>::VALUE))
                .sqrt()
                - nu)
                .ceil();
            let mut m = nu + j;
            // Amos: A_M(x) ~ x / (M - 1/2 + sqrt((M + 1/2)^2 + x^2)), so
            // 1 - A_M = (M - 1/2 + (sqrt(...) - x)) / (M - 1/2 + sqrt(...)), with the
            // difference of the root and x as (M + 1/2)^2 / (sqrt(...) + x).
            let mh = m + V::HALF;
            let root = mh.mul_adde(mh, x * x).sqrt();
            let denom = m - V::HALF + root;
            let mut cd = (m - V::HALF + mh * mh / (root + x)) / denom;
            loop {
                let live = down & m.cmp_gt(nu + V::HALF);
                if live.none() {
                    break;
                }
                m -= V::ONE;
                let y = (m * two_over_x) - cd;
                cd = live.select(y / (V::ONE + y), cd);
            }
            cm = down.select(cd, cm);
        }

        c = asym.select(cm, c);
    }

    // c(0) = 1, c(inf) = 0, and the reflection.
    let c = x.cmp_eq(V::INFINITY).select(V::ZERO, c);
    neg.select(V::TWO - c, c)
}

/// The `kappa` with `1 - A_nu(kappa) = t`, for `0 < t <= 2` (`t = 1 - r`), the complement
/// form of [`inv_bessel_i_ratio_impl`]: well conditioned as `t -> 0`, where the plain form
/// loses `2 kappa eps / (p - 1)` to the rounding of `r`.
#[inline(always)]
pub fn inv_bessel_i_ratio_1m_impl<P, E, V>(t: V, nu: V) -> V
where
    P: Policy,
    E: FloatElement,
    V: FloatVector<Element = E> + PrimalProjection<Primal = V> + SpecializedSpecialMath<E>,
{
    // t in (1, 2] is r < 0: kappa(r) is odd, and 2 - t is exact there.
    let mirror = t.cmp_gt(V::ONE);
    let t = mirror.select(V::TWO - t, t);

    let p = nu + nu;
    // Above t = 1/2 the plain form is well conditioned and 1 - t is exact (Sterbenz), and
    // the complement's bracket would collapse at t = 1 where kappa_0 = 0.
    let plain = t.cmp_ge(V::HALF);
    let active = t.cmp_gt(V::ZERO) & !plain;

    // Banerjee in the complement: r (p - r^2)/(1 - r^2) with r = 1 - t, 1 - r^2 = t (2 - t)
    // and p - r^2 = (p - 1) + t (2 - t), nothing cancelling.
    let u = t * (V::TWO - t);
    let k0 = (V::ONE - t) * ((p - V::ONE) + u) / u;

    let mut kappa = k0;
    if const { P::POLICY.avoid_branching } || plain.any() {
        let via_plain = inv_bessel_i_ratio_impl::<P, E, V>(V::ONE - t, nu);
        kappa = plain.select(via_plain, kappa);
    }
    if const { P::POLICY.avoid_branching } || active.any() {
        let tol = residual_tolerance::<P, E, V>(t);
        let bounds = Some((k0 * V::HALF, k0 + k0));
        let (root, _) = newtons_method::<V, MaxIterations<P, 8>, _>(k0, tol, active, bounds, |k| {
            let c = bessel_i_ratio_1m_impl::<P, E, V>(k, nu);
            (c - t, -bessel_i_ratio_deriv_1m::<P, E, V>(c, k, nu))
        });
        kappa = active.select(root, kappa);
    }

    let kappa = t.cmp_eq(V::ZERO).select(V::INFINITY, kappa);
    let kappa = t.cmp_lt(V::ZERO).select(V::NAN, kappa);
    kappa.neg_c(mirror)
}

/// The `kappa >= 0` with `I_nu(kappa) / I_{nu-1}(kappa) = r`, for `0 <= r < 1`. Odd in `r`.
#[inline(always)]
pub fn inv_bessel_i_ratio_impl<P, E, V>(r: V, nu: V) -> V
where
    P: Policy,
    E: FloatElement,
    V: FloatVector<Element = E> + PrimalProjection<Primal = V> + SpecializedSpecialMath<E>,
{
    let neg = r.is_negative();
    let r = r.abs();

    let p = nu + nu;
    let tiny = r.cmp_lt(V::splat(<E as FloatElement>::ConstRatio::<1, 100_000_000>::VALUE));
    let active = r.cmp_lt(V::ONE) & !tiny;

    // Banerjee: r (p - r^2) / (1 - r^2). No transcendental in it.
    let r2 = r * r;
    let k0 = r * (p - r2) / (V::ONE - r2);

    let mut kappa = p * r;

    if const { P::POLICY.avoid_branching } || active.any() {
        let tol = residual_tolerance::<P, E, V>(r);
        let bounds = Some((k0 * V::HALF, k0 + k0));
        let (root, _) = newtons_method::<V, MaxIterations<P, 8>, _>(k0, tol, active, bounds, |k| {
            let a = bessel_i_ratio_impl::<P, E, V>(k, nu);
            (a - r, bessel_i_ratio_deriv::<P, E, V>(a, k, nu))
        });
        kappa = active.select(root, kappa);
    }

    let kappa = r.cmp_eq(V::ONE).select(V::INFINITY, kappa);
    let kappa = r.cmp_gt(V::ONE).select(V::NAN, kappa);
    kappa.neg_c(neg)
}
