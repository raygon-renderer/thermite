use thermite::{
    LargeInt,
    element::FloatElementWithBits,
    math::{CoreMathWithPolicy as _, TranscendentalMathWithPolicy as _, policy::Policy},
    prelude::*,
};

use crate::specialized::SpecializedSpecialMath;
use crate::tables::bernoulli::BernoulliNumbers;
use crate::tables::cot_pi::CotPiDerivatives;
use crate::tables::factorial::Factorials;

/// Shared polygamma (`psi_n`) implementation for all real element types.
///
/// `$\psi_n(x) = \frac{\mathrm{d}^n}{\mathrm{d}x^n}\psi(x)$`, the (n+1)-th derivative of
/// `$\ln\Gamma$`. `n = 0` and `n = 1` delegate to the tuned [`digamma`] and [`trigamma`]
/// kernels. `n >= 2` runs the two-region scheme Boost.Math's `polygamma_imp` uses on the
/// positive axis, restructured for
/// vectors:
///
/// * a masked forward recurrence `$\psi_n(x) = \psi_n(x+1) + (-1)^{n-1} n!\,x^{-(n+1)}$`
///   walks every lane up to the transition point `$N = 0.4\,d_{10} + 4n$` (with `$d_{10}$`
///   the format's decimal digits), then
/// * the asymptotic expansion at large `x`,
///
/// ```math
/// \psi_n(x) = (-1)^{n-1}\left[\frac{(n-1)!}{x^n} + \frac{n!}{2x^{n+1}}
///     + \sum_{k\ge1} B_{2k}\,\frac{(2k+n-1)!}{(2k)!\,x^{2k+n}}\right]
/// ```
///
/// evaluated by the term-ratio recurrence, so the scalar order-dependent coefficients
/// `$(n+2k-2)(n+2k-1)/((2k-1)\,2k)$` are splatted and the vector work per term is one
/// multiply by `$1/x^2$`. The `$B_{2k}$` come from [`BernoulliNumbers::B2N`], and the
/// series converges well before that table ends for every `x` past the transition point.
///
/// Unlike Boost there is no separate near-zero zeta series: the leading `$n!/x^{n+1}$`
/// term dominates so completely below the recurrence range that the walk loses nothing.
///
/// Negative arguments reflect through
/// `$\psi_n(x) = (-1)^n\left[\psi_n(1-x) + \pi\,\frac{\mathrm{d}^n}{\mathrm{d}z^n}\cot(\pi z)\big|_{z=1-x}\right]$`,
/// with the cot derivative's cosine polynomial from [`CotPiDerivatives::COT_PI_ROWS`]
/// and both `sin_pi`/`cos_pi` evaluated at `x` itself, the smaller-magnitude
/// representative (they agree with the `1 - x` values exactly, by periodicity, but
/// carry less argument error). At the poles (zero and the negative integers) odd `n`
/// yields `+inf`, the correct two-sided limit. Even `n` has one-sided limits of
/// opposite sign and yields NaN when overflow checking is enabled.
///
/// # Current limits (deliberate, documented rather than patched)
///
/// * **Reflection stops at `n = 20`**, the cot-pi table's reach (Boost tabulates the
///   same range, with its runtime coefficient recurrence past it queued work). Negative
///   arguments at `n > 20` return NaN. The positive axis is unaffected.
/// * **Direct powers bound the domain.** `x^(n+1)` is formed directly at arguments up
///   to `max(x, N)`, so lanes where `(n+1) log10(max(x, N))` exceeds the format's
///   decimal exponent range (~300 for f64, ~36 for f32) flush to zero even where
///   `psi_n` itself is representable (e.g. `psi_100(1e4) ~ -9.4e-245`), and `n!`
///   likewise overflows at `n >= 171` (f64) / `n >= 35` (f32). Boost rescues both
///   with log-domain arithmetic. That is queued work, and the tests pin the boundary.
///
/// [`digamma`]: SpecializedSpecialMath::digamma
/// [`trigamma`]: SpecializedSpecialMath::trigamma
#[inline(always)]
pub fn polygamma_impl<P, E, V>(x_in: V, n: u32) -> V
where
    P: Policy,
    E: FloatElementWithBits + BernoulliNumbers + CotPiDerivatives + Factorials,
    V: FloatVectorWithBits<Element = E> + SpecializedSpecialMath<E>,
{
    if n == 0 {
        return SpecializedSpecialMath::digamma::<P>(x_in);
    }
    if n == 1 {
        return SpecializedSpecialMath::trigamma::<P>(x_in);
    }

    let x0 = x_in.flush_denormals_p::<P>();

    let reflect = x0.cmp_le(V::ZERO);
    let sign_neg = (n - 1) & 1 == 1;

    // Every order-dependent coefficient is exact integer scalar math lifted into E
    // without casts (`from_int`/`from_ratio` are exact-or-panic, and stay exact here:
    // the largest integer formed is bounded by the factorial table's reach, well under
    // 2^24), or a correctly rounded factorial from the generated table. No f64
    // arithmetic appears, so the kernel is indifferent to E's width.
    let (fac_nm1, fac_n) = match (E::FACTORIALS.get(n as usize - 1), E::FACTORIALS.get(n as usize)) {
        (Some(&a), Some(&b)) => (V::splat(a), V::splat(b)),
        (Some(&a), None) => (V::splat(a), <V as FloatVector>::INFINITY),
        _ => {
            // n! overflows E outright, the documented large-n limit (see above). The
            // reflected side's sign depends on the cot term, so it gets NaN, not inf.
            let inf = if sign_neg {
                <V as FloatVector>::NEG_INFINITY
            } else {
                <V as FloatVector>::INFINITY
            };
            return (reflect | x0.is_nan()).select(V::NAN, inf);
        }
    };

    // Transition point N = 0.4 * digits10 + 4n, Boost's choice: far enough out that the
    // Bernoulli series below converges geometrically from its first term. 12/100
    // approximates 0.4 log10(2) closely enough that every format lands on Boost's
    // integer value.
    let d4d = (12 * (E::MANTISSA_BITS + 1)) / 100;
    let threshold = V::splat(E::from_int((d4d + 4 * n) as LargeInt));

    // Reflected lanes work at z = 1 - x >= 1. The cot term is added at the end.
    let mut x = reflect.select(V::ONE - x0, x0);

    // --- Forward recurrence: psi_n(x) = psi_n(x + 1) + (-1)^(n-1) n! x^-(n+1) ---
    // The positive powi then one reciprocal (rather than powi of the reciprocal) keeps
    // the rcp's error out of the squaring chain, and overflow of x^(n+1) only happens
    // where the true term underflows anyway (the two failure regions coincide).
    let mut rec = V::ZERO;
    let mut active = x.cmp_lt(threshold);
    while active.any() {
        V::_loop_hint();

        let t = x.powi_p::<P>(n as i32 + 1).approx_reciprocal_p::<P>();
        rec = rec.add_c(active, t);
        x = x.add_c(active, V::ONE);
        active = x.cmp_lt(threshold);
    }

    // --- Asymptotic tail at x >= N ---
    let zr = x.approx_reciprocal_p::<P>();
    let z2r = zr * zr;

    // lead = (n-1)! / x^n. The first two terms fold into lead * (1 + n/(2x)).
    let lead = fac_nm1 * x.powi_p::<P>(n as i32).approx_reciprocal_p::<P>();
    let mut asum = (V::splat(E::from_ratio(n as LargeInt, 2)) * zr).mul_adde(lead, lead);

    // part = (n+1)! / (2 x^(n+2)), the k = 1 series term without its Bernoulli number.
    // n (n + 1) / 2 is a triangular number: exact.
    let mut part = lead * z2r * V::splat(E::from_int(n as LargeInt * (n as LargeInt + 1) / 2));

    let eps = <V as FloatVector>::EPSILON;
    let mut k = 1usize;
    loop {
        V::_loop_hint();

        let term = part * V::splat(E::B2N[k - 1]);
        asum += term;

        // The table ends exactly where B_2k overflows E, but convergence always wins first
        // for x past the transition point, so this bound is a backstop.
        if k >= E::B2N.len() || term.abs().cmp_le(asum.abs() * eps).all() {
            break;
        }

        // The ratio uses the _incremented_ k: part_{k+1}/part_k = (n+2k)(n+2k+1)/((2k+1)(2k+2)).
        // Both integer products are exact in E (bounded by the table reach, < 2^24), so
        // the ratio costs one scalar division's rounding, then splats.
        k += 1;
        let nk = n as LargeInt + 2 * k as LargeInt;
        let k2 = 2 * k as LargeInt;
        let ratio = E::from_int((nk - 2) * (nk - 1)) / E::from_int((k2 - 1) * k2);
        part = part * z2r * V::splat(ratio);
    }

    // Both regions carry the same (-1)^(n-1) prefactor, so it is applied once.
    let mut res = rec.mul_adde(fac_n, asum);
    if sign_neg {
        res = -res;
    }

    // --- Reflection: psi_n(x) = (-1)^n [psi_n(1 - x) + pi * cot^(n)(pi (1 - x))] ---
    if const { P::POLICY.avoid_branching } || reflect.any() {
        if (n as usize) <= E::COT_PI_ROWS.len() {
            let row = E::COT_PI_ROWS[n as usize - 1];

            // By periodicity sin_pi/cos_pi of 1 - x are +-sin_pi/cos_pi of x, and x is
            // always the smaller-magnitude representative on this path, so the argument
            // carries less error. cos powers keep every polynomial term zero at
            // half-integers, right where the derivative bottoms out, so no cancellation.
            let s = x0.sin_pi_p::<P>();
            let c = -x0.cos_pi_p::<P>();
            let c2 = c * c;

            let mut poly = V::splat(row[row.len() - 1]);
            let mut j = row.len() - 1;
            while j > 0 {
                j -= 1;
                poly = poly.mul_adde(c2, V::splat(row[j]));
            }
            if n & 1 == 0 {
                poly *= c;
            }

            // pi * pi^n P / s^(n+1) = (pi/s)^(n+1) P. At the poles s is +-0 and the
            // even power makes this +inf regardless of the zero's sign.
            let cot_term = (V::PI / s).powi_p::<P>(n as i32 + 1) * poly;

            let total = res + cot_term;
            res = reflect.select(if n & 1 == 1 { -total } else { total }, res);
        } else {
            // Past the cot-pi table, a documented limit (see above).
            res = reflect.select(V::NAN, res);
        }

        // Zero and the negative integers: odd n has a definite two-sided limit of
        // +inf (pinned here even though the arithmetic above already produces it).
        // Even n diverges with opposite signs, so checked policies get NaN.
        let pole = reflect & x0.floor().cmp_eq(x0);
        if n & 1 == 1 {
            res = pole.select(<V as FloatVector>::INFINITY, res);
        } else if const { P::POLICY.check_overflow } {
            res = pole.select(V::NAN, res);
        }
    }

    res
}
