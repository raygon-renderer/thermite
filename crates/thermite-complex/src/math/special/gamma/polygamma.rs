//! `polygamma(n)` over C: the same recurrence-plus-asymptotic-series kernel as the real one,
//! on the real element's Bernoulli, cot-pi and factorial tables.

use thermite::LargeInt;
use thermite::element::FloatElementWithBits;
use thermite::math::policy::Policy;
use thermite::math::{CoreMathWithPolicy as _, TranscendentalMathWithPolicy as _};
use thermite::prelude::*;
use thermite_special::tables::bernoulli::BernoulliNumbers;
use thermite_special::tables::cot_pi::CotPiDerivatives;
use thermite_special::tables::factorial::Factorials;

use crate::Complex;
use crate::math::ComplexMathWithPolicy as _;
use crate::vector::RealFloatVector;

/// Shared complex polygamma for `n >= 2` (`n = 0`/`1` delegate to the tuned
/// digamma/trigamma before reaching this).
///
/// The real kernel's structure run in complex arithmetic, with every regime gate on
/// `Re`, never the lexicographic complex compare:
///
/// * the half-plane `Re z < 1/2` reflects through
///   `$\psi_n(z) = (-1)^n[\psi_n(1-z) + \pi\,\cot^{(n)}(\pi(1-z))]$`, reusing the
///   generated [`CotPiDerivatives`] rows unchanged (the cosine polynomial is entire,
///   so it evaluates over C as it stands), with `sin_pi`/`cos_pi` taken at `z`
///   itself (equal to the `1 - z` values by periodicity, with less argument error).
/// * the recurrence `$\psi_n(w) = \psi_n(w+1) + (-1)^{n-1} n!\,w^{-(n+1)}$` walks
///   every lane to `Re w >= shift_base + 4n` (`shift_base` is the per-element tuning
///   `trigamma_impl` already uses: complex wants more margin than the real kernel's
///   `0.4 d_10`).
/// * then the Bernoulli asymptotic series by the term-ratio recurrence, coefficients
///   as exact integer element math splatted once, terms until
///   `$|term| \le \varepsilon |sum|$` (compared as `norm_sqr` against `eps^2`).
///
/// Orders past the factorial table return NaN outright: over C the overflowing
/// leading term has no single signed infinity to carry. Reflection past the cot-pi
/// table (`n > 20`) likewise NaNs the reflected lanes only. The poles at the
/// non-positive real integers come out of the arithmetic itself (`sin_pi` is exactly
/// zero there, and the even power of `pi/s` diverges), as in `trigamma_impl`.
///
/// The reflected half-plane's `|Im z|` reach is `sin_pi`/`cos_pi`'s own overflow
/// bound (they grow as `$e^{\pi|\operatorname{Im} z|}/2$`, infinite past
/// `$|\operatorname{Im} z| \approx 232$` for f64 / `$\approx 28$` for f32), thanks to
/// the bounded-variable regrouping in the reflection below. The naive
/// `$P(\cos)/\sin^{n+1}$` form dies orders of magnitude earlier, when an overflowing
/// cosine power meets an underflowing `$(\pi/s)^{n+1}$` and 0 * inf turns reflected
/// lanes NaN. Inside the reach, once `$1/s^2$` underflows the whole cot term flushes
/// to zero, which is the right answer: its true relative contribution decays as
/// `$e^{-2\pi|\operatorname{Im} z|}$`. Accuracy degrades toward the bound as
/// `$\sim 2n\,\pi|\operatorname{Im} z|\,\varepsilon$` (the exponentials amplify
/// their argument's rounding), which the tests' tolerance scaling mirrors.
#[inline(always)]
pub(crate) fn polygamma_impl<P: Policy, V: RealFloatVector>(z: Complex<V>, n: u32, shift_base: u32) -> Complex<V>
where
    V::Element: FloatElementWithBits + BernoulliNumbers + CotPiDerivatives + Factorials,
{
    let sign_neg = (n - 1) & 1 == 1;

    let (fac_nm1, fac_n) = match (
        <V::Element as Factorials>::FACTORIALS.get(n as usize - 1),
        <V::Element as Factorials>::FACTORIALS.get(n as usize),
    ) {
        (Some(&a), Some(&b)) => (a, b),
        _ => return Complex::new(V::NAN, V::NAN),
    };

    // Reflect the left half-plane. The cot term is added at the end.
    let reflect = z.re.cmp_lt(V::HALF);
    let mut w = reflect.select(Complex::ONE - z, z);

    // --- Forward recurrence up to the series' reach ---
    let shift = V::splat(<V::Element as thermite::element::FloatElement>::from_int(
        (shift_base + 4 * n) as LargeInt,
    ));
    let mut rec = Complex::<V>::ZERO;
    let mut active = w.re.cmp_lt(shift);
    while active.any() {
        V::_loop_hint();

        let t = w.powi_p::<P>(n as i32 + 1).finv_p::<P>();
        rec = active.select(rec + t, rec);
        w = active.select(w + Complex::ONE, w);
        active = w.re.cmp_lt(shift);
    }

    // --- Asymptotic tail at Re w >= shift ---
    let u = w.finv_p::<P>();
    let u2 = u * u;

    // lead = (n-1)! / w^n. The first two terms fold into lead * (1 + n/(2w)).
    let lead = w.powi_p::<P>(n as i32).finv_p::<P>() * V::splat(fac_nm1);
    let n_half = V::splat(<V::Element as thermite::element::FloatElement>::from_ratio(
        n as LargeInt,
        2,
    ));
    let mut asum = lead + (u * lead) * n_half;

    // part = (n+1)! / (2 w^(n+2)), and n(n+1)/2 is triangular, hence exact.
    let tri = <V::Element as thermite::element::FloatElement>::from_int(n as LargeInt * (n as LargeInt + 1) / 2);
    let mut part = (lead * u2) * V::splat(tri);

    let eps2 = <V as FloatVector>::EPSILON * <V as FloatVector>::EPSILON;
    let b2n = <V::Element as BernoulliNumbers>::B2N;
    let mut k = 1usize;
    loop {
        V::_loop_hint();

        let term = part * V::splat(b2n[k - 1]);
        asum += term;

        if k >= b2n.len() || term.norm_sqr().cmp_le(asum.norm_sqr() * eps2).all() {
            break;
        }

        // Term ratio with the _incremented_ k, as in the real kernel. Both integer
        // products are exact in the element, so this is one scalar division splatted.
        k += 1;
        let nk = n as LargeInt + 2 * k as LargeInt;
        let k2 = 2 * k as LargeInt;
        let ratio = <V::Element as thermite::element::FloatElement>::from_int((nk - 2) * (nk - 1))
            / <V::Element as thermite::element::FloatElement>::from_int((k2 - 1) * k2);
        part = (part * u2) * V::splat(ratio);
    }

    // Both positive-axis regions carry the same (-1)^(n-1) prefactor.
    let mut res = rec * V::splat(fac_n) + asum;
    if sign_neg {
        res = -res;
    }

    // --- Reflection: psi_n(z) = (-1)^n [psi_n(1 - z) + pi cot^(n)(pi (1 - z))] ---
    if const { P::POLICY.avoid_branching } || reflect.any() {
        let rows = <V::Element as CotPiDerivatives>::COT_PI_ROWS;
        if (n as usize) <= rows.len() {
            let row = rows[n as usize - 1];

            let s = z.sin_pi_p::<P>();
            let c = -z.cos_pi_p::<P>();

            // Bounded-variable regrouping. The naive `pi^(n+1) P(c) / s^(n+1)` pairs
            // an overflowing `c^(n-1)` against an underflowing `(pi/s)^(n+1)` as
            // |Im z| grows (both are e^(pi|Im z|) to n-ish powers), and 0 * inf is
            // NaN long before either factor is genuinely out of range. With
            // t = cot(pi z) = c/s (|t| -> 1 as |Im z| grows) and v = 1/s^2 (small
            // there), the same sum is
            //
            //   sum_j p_j c^(2j) / s^(n+1) = s^-(2+off) sum_j p_j t^(2j) v^(J-j)
            //
            // (J the top index, off the single leftover cosine power on even n), and
            // every factor stays representable to sin_pi's own overflow bound. The
            // half-integer property survives: t is exactly zero where c is.
            let sr = s.finv_p::<P>();
            let t = c * sr;
            let u = t * t;
            let v = sr * sr;

            let mut acc = Complex::real(V::splat(row[row.len() - 1]));
            let mut vk = Complex::<V>::ONE;
            let mut j = row.len() - 1;
            while j > 0 {
                j -= 1;

                vk *= v;
                acc = acc * u + vk * V::splat(row[j]);
            }

            // total = pi^(n+1) * t^off * v * acc. pi^(n+1) is finite in-format for
            // every tabulated order.
            let mut cot = acc * v * V::PI.powi_p::<P>(n as i32 + 1);
            if n & 1 == 0 {
                cot *= t;
            }

            let total = res + cot;
            res = reflect.select(if n & 1 == 1 { -total } else { total }, res);
        } else {
            // Past the cot-pi table, the same documented limit as the real kernel.
            res = reflect.select(Complex::new(V::NAN, V::NAN), res);
        }
    }

    res
}
