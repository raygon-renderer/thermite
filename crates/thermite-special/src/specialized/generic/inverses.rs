//! Newton inverses of shipped forwards: `inv_digamma` and `wright_omega`.
//!
//! Both are the same shape as [`inv_log_ndtr`](super::ndtr::inv_log_ndtr_impl): a cheap
//! seed one precision tier down, then `newtons_method` on the forward with its closed-form
//! derivative, stopping at a residual tolerance a few ulp above the forward's own noise and
//! capped at eight iterations. Each forward is increasing and concave on its domain, so a
//! Newton step from either side lands left of the root and the iteration is monotone from
//! there. The bracket handed to `newtons_method` only keeps a wild seed from crossing zero.
//!
//! Newton cannot beat the forward's rounding, and both functions have a region
//! where that rounding is the whole error. `digamma(x) - y` for large `x` is `ln x - y`
//! to a rounding of `eps * y`, which is `eps * y` _relative_ in `x`. `w + ln w - x` for
//! very negative `x` is the difference of two numbers near `x` whose true difference is
//! `e^x`. Each gets an analytic arm there instead: the Stirling fixed point in the
//! exponent for `inv_digamma` above `y = 6`, and the Lagrange series `sum (-n)^{n-1}/n!
//! e^{nx}` for `wright_omega` below `x = -7`.

use thermite::{
    element::FloatElement,
    math::{
        TranscendentalMathWithPolicy as _,
        algorithms::newtons_method,
        policy::{
            Policy,
            policies::{LessPrecision, MaxIterations},
        },
    },
    prelude::*,
};

use super::ndtr::residual_tolerance;
use crate::specialized::SpecializedSpecialMath;

/// The `x > 0` with `digamma(x) = y`.
///
/// Seed (Minka, "Estimating a Dirichlet distribution", appendix): `e^y + 1/2` for
/// `y >= -2.22` and `-1/(y - digamma(1))` below, both from the asymptotics at the two ends
/// and within a factor of two of the root everywhere. Newton with `trigamma` from there,
/// bracketed by that factor of two on each side.
///
/// Above `y = 3` (`x > 20`) the answer is the Stirling series solved for `x` instead.
/// `digamma(x) = ln x - t(x)` with `t = 1/(2x) + 1/(12x^2) - 1/(120x^4) + ... - 691/(32760 x^12)`
/// (5e-18 at `x = 20`), so `x = e^y e^{t(x)}`. Writing `x = e^y u`, the unknown `u = e^{t}`
/// is the root of `h(u) = u - e^{t(e^y u)}`, which Newton takes quadratically from
/// `u_0 = e^{t(e^y + 1/2)}` in two steps. Nothing in `h` subtracts `y`, so the result is
/// within an ulp or two of `e^y`'s own rounding, where Newton on `digamma` is bounded by
/// `eps * y` relative (the residual `digamma(x) - y` is `ln x - y` to a rounding of
/// `eps * y`). The plain fixed point `x <- e^y e^{t(x)}` contracts only by `1/(2x)` per
/// pass and would need seven passes at `x = 20`.
#[inline(always)]
pub fn inv_digamma_impl<P, E, V>(y: V) -> V
where
    P: Policy,
    E: FloatElement,
    V: FloatVector<Element = E> + SpecializedSpecialMath<E>,
{
    let finite = y.is_finite();
    let big = y.cmp_ge(V::splat(<E as FloatElement>::ConstInt::<3>::VALUE));
    let active = finite & !big;

    // e^y at the seed's tier serves both the seed and the analytic arm.
    let ey = y.exp_p::<LessPrecision<P>>();

    // Minka's seed, both halves cheap: the seam is at digamma(0.6) ~ -2.22.
    let seam = V::splat(<E as FloatElement>::ConstRatio::<-222, 100>::VALUE);
    let low = y.cmp_lt(seam);
    let x0 = low.select(
        (y + V::EULER_GAMMA).approx_reciprocal_p::<LessPrecision<P>>().neg(),
        ey + V::HALF,
    );

    let mut x = x0;
    if const { P::POLICY.avoid_branching } || active.any() {
        let tol = residual_tolerance::<P, E, V>(y.abs().max(V::ONE));
        let bounds = Some((x0 * V::HALF, x0 + x0));
        let (r, _) = newtons_method::<V, MaxIterations<P, 8>, _>(x0, tol, active, bounds, |x| {
            (
                <V as SpecializedSpecialMath<E>>::digamma::<P>(x) - y,
                <V as SpecializedSpecialMath<E>>::trigamma::<P>(x),
            )
        });
        x = r;
    }

    if const { P::POLICY.avoid_branching } || big.any() {
        let ey = if const { P::POLICY.precision.gt(thermite::math::policy::PrecisionPolicy::Medium) } {
            y.exp_p::<P>()
        } else {
            ey
        };

        // t(x) and t'(x) from r = 1/x. t' only needs to be right to a few percent: it is
        // Newton's slope, and h sets the step size, not h'.
        let stirling = |xs: V| -> (V, V) {
            let r = xs.approx_reciprocal_p::<P>();
            let r2 = r * r;
            // 1/(2x) is the one odd power. The rest is a polynomial in 1/x^2.
            let even = r2.poly_n_p::<P, _>(&[
                <E as FloatElement>::ConstRatio::<1, 12>::VALUE,
                <E as FloatElement>::ConstRatio::<-1, 120>::VALUE,
                <E as FloatElement>::ConstRatio::<1, 252>::VALUE,
                <E as FloatElement>::ConstRatio::<-1, 240>::VALUE,
                <E as FloatElement>::ConstRatio::<1, 132>::VALUE,
                <E as FloatElement>::ConstRatio::<-691, 32760>::VALUE,
            ]);
            let t = r.mul_adde(V::HALF, r2 * even);
            // t(x) = 1/(2x) + 1/(12x^2) + ..., so t'(x) = -(1/(2x^2))(1 + 1/(3x) + ...).
            let dt = -(r2 * V::HALF) * r.mul_adde(V::splat(<E as FloatElement>::ConstRatio::<1, 3>::VALUE), V::ONE);
            (t, dt)
        };

        let (t0, _) = stirling(ey + V::HALF);
        let mut u = t0.exp_p::<P>();
        let steps = const {
            if P::POLICY.precision.ge(thermite::math::policy::PrecisionPolicy::Best) {
                3
            } else {
                2
            }
        };
        let mut i = 0;
        while i < steps {
            let xs = ey * u;
            let (t, dt) = stirling(xs);
            let et = t.exp_p::<P>();
            // h = u - e^t, h' = 1 - e^t t'(x) e^y.
            let h = u - et;
            let dh = (et * dt).nmul_adde(ey, V::ONE);
            u -= h / dh;
            i += 1;
        }
        x = big.select(ey * u, x);
    }

    // digamma maps (0, inf) onto the whole line: +inf -> +inf, -inf -> 0.
    let x = y.cmp_eq(V::INFINITY).select(V::INFINITY, x);
    let x = y.cmp_eq(V::NEG_INFINITY).select(V::ZERO, x);
    y.is_nan().select(V::NAN, x)
}

/// The Wright omega function, the `w > 0` with `w + ln w = x`, which is `W_0(e^x)` without
/// ever forming `e^x`.
///
/// Seeds by region, after Lawrence, Corless and Jeffrey (2012, the algorithm SciPy uses):
/// `q(1 - q(1 - 3q/2))` in `q = e^x` for `x <= -2`, the series about `x = 1` for
/// `-2 < x < 1`, and `x - ln x + ln x / x` above, each one tier down. Newton on
/// `w + ln w - x` with `1 + 1/w` from there, bracketed by a factor of two.
///
/// Below `x = -7` the residual `w + ln w - x` is the difference of two numbers near `x`
/// whose true difference is `e^x < 1e-3`, so Newton is bounded by `eps * |x|` relative.
/// The Lagrange series `w = sum_{n>=1} (-n)^{n-1}/n! q^n` to six terms is `2e-17` relative
/// there and is the whole answer, one `exp` and five FMAs. Above `x = 1e20` the seed is
/// the answer to working precision and Newton's first residual is already within tolerance.
#[inline(always)]
pub fn wright_omega_impl<P, E, V>(x: V) -> V
where
    P: Policy,
    E: FloatElement,
    V: FloatVector<Element = E> + SpecializedSpecialMath<E>,
{
    let finite = x.is_finite();
    let series = x.cmp_le(-V::splat(<E as FloatElement>::ConstInt::<7>::VALUE));
    let active = finite & !series;

    let mut w = V::ZERO;

    if const { P::POLICY.avoid_branching } || series.any() {
        // (-n)^{n-1}/n!: 1, -1, 3/2, -8/3, 125/24, -54/5.
        let q = x.exp_p::<P>();
        let p = q.poly_n_p::<P, _>(&[
            E::ZERO,
            E::ONE,
            -E::ONE,
            <E as FloatElement>::ConstRatio::<3, 2>::VALUE,
            <E as FloatElement>::ConstRatio::<-8, 3>::VALUE,
            <E as FloatElement>::ConstRatio::<125, 24>::VALUE,
            <E as FloatElement>::ConstRatio::<-54, 5>::VALUE,
        ]);
        w = p;
    }

    if const { P::POLICY.avoid_branching } || active.any() {
        let left = x.cmp_le(-V::TWO);
        let right = x.cmp_ge(V::ONE);

        let mut w0 = V::ZERO;
        if const { P::POLICY.avoid_branching } || left.any() {
            let q = x.exp_p::<LessPrecision<P>>();
            let three_halves = V::splat(<E as FloatElement>::ConstRatio::<3, 2>::VALUE);
            w0 = q * q.nmul_adde(q.nmul_adde(three_halves, V::ONE), V::ONE);
        }
        if const { P::POLICY.avoid_branching } || !(left | right).all() {
            // omega about x = 1, where omega(1) = 1 exactly.
            let z = x - V::ONE;
            let mid = z.poly_n_p::<LessPrecision<P>, _>(&[
                E::ONE,
                <E as FloatElement>::ConstRatio::<1, 2>::VALUE,
                <E as FloatElement>::ConstRatio::<1, 16>::VALUE,
                <E as FloatElement>::ConstRatio::<-1, 192>::VALUE,
                <E as FloatElement>::ConstRatio::<-1, 3072>::VALUE,
                <E as FloatElement>::ConstRatio::<13, 61440>::VALUE,
            ]);
            w0 = left.select(w0, mid);
        }
        if const { P::POLICY.avoid_branching } || right.any() {
            let lx = x.ln_p::<LessPrecision<P>>();
            w0 = right.select((x - lx) + lx * x.approx_reciprocal_p::<LessPrecision<P>>(), w0);
        }

        let tol = residual_tolerance::<P, E, V>(x.abs().max(V::ONE));
        let bounds = Some((w0 * V::HALF, w0 + w0));
        let (r, _) = newtons_method::<V, MaxIterations<P, 8>, _>(w0, tol, active, bounds, |w| {
            let rw = w.approx_reciprocal_p::<P>();
            (w + w.ln_p::<P>() - x, V::ONE + rw)
        });
        w = active.select(r, w);
    }

    // omega(+inf) = +inf, omega(-inf) = 0.
    let w = x.cmp_eq(V::INFINITY).select(V::INFINITY, w);
    x.is_nan().select(V::NAN, w)
}
