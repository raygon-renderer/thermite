//! The Riemann zeta function over C, by the same Euler-Maclaurin expansion the real kernel
//! runs. Accuracy is governed by `|Im z|` against the expansion's `N`. See the
//! per-element hook's documentation for the measured ladder.

use thermite::math::TranscendentalMathWithPolicy as _;
use thermite::math::policy::Policy;
use thermite::prelude::*;
use thermite_special::specialized::{ZetaConsts, zeta_bernoulli_terms};
use thermite_special::tables::bernoulli::BernoulliNumbers;
use thermite_special::tables::primal::GammaPrimalTables;

use super::gamma::tgamma_impl;
use crate::Complex;
use crate::vector::RealFloatVector;

/// `zeta(z)`, or `zeta(z) - 1` when `ZETAC`, by the same Euler-Maclaurin expansion the real
/// kernel uses. See `thermite_special`'s `generic::zeta` for why that shape rather than a
/// table of minimax rationals, and for the prime factorization that keeps it to four
/// exponentials.
///
/// Nothing about the expansion is real-specific. Every tabulated piece belongs to the _real_
/// element and carries over untouched. Only the arithmetic changes. What does change is the
/// domain over which it is any good: the correction terms grow like `(|z|/N)^{2k}`, so `N` has
/// to exceed `|Im z|`. At `N = 10` that is comfortable to about `|Im z| = 4` and degrading
/// past 10.
#[inline(always)]
pub(crate) fn zeta_impl<P: Policy, V: RealFloatVector, const ZETAC: bool>(z: Complex<V>) -> Complex<V>
where
    V::Element: thermite::element::FloatElement + ZetaConsts + BernoulliNumbers,
    // Deliberately NOT bounded on `Complex<V>: SpecializedComplexSpecialMath<..>`. A
    // where-clause bound on `Complex<V>` makes its associated types opaque, which knocks out
    // the blanket `CoreMathWithPolicy` impl and with it the whole transcendental surface.
    // Every `exp`/`sin_pi` call in the body stops resolving, for a reason the error message
    // attributes to the individual method. The reflection reaches `tgamma_impl` as a free
    // function instead, exactly as the other kernels here do.
    V::Primal: GammaPrimalTables<<V::Primal as GenericVector>::Element>,
{
    let one = Complex::<V>::ONE;

    // The left half-plane reflects, exactly as on the real axis: the expansion is asymptotic
    // and gets _worse_ with more terms there, while 1 - z lands where it is at its best.
    let reflect = z.re.cmp_lt(V::ZERO);
    let s = reflect.select(one - z, z);

    let neg_s = -s;

    // Four exponentials. Every other term below is a product of these.
    let p2 = neg_s.exp2_p::<P>();
    let p3 = (neg_s * V::splat(<V::Element as ZetaConsts>::LOG2_3)).exp2_p::<P>();
    let p5 = (neg_s * V::splat(<V::Element as ZetaConsts>::LOG2_5)).exp2_p::<P>();
    let p7 = (neg_s * V::splat(<V::Element as ZetaConsts>::LOG2_7)).exp2_p::<P>();

    let p4 = p2 * p2;
    let p6 = p2 * p3;
    let p8 = p4 * p2;
    let p9 = p3 * p3;
    let n_s = p2 * p5; // 10^-z

    let direct = ((p9 + p8) + (p7 + p6)) + ((p5 + p4) + (p3 + p2));

    let ten = V::splat(<V::Element as thermite::element::FloatElement>::ConstRatio::<10, 1>::VALUE);
    let half = V::splat(<V::Element as thermite::element::FloatElement>::ConstRatio::<1, 2>::VALUE);
    let boundary = n_s * (Complex::new(ten, V::ZERO) / (s - one) + Complex::new(half, V::ZERO));

    // The correction ladder. Same recurrence and the same compile-time denominators as the
    // real kernel. The Bernoulli numbers stay real and multiply in as a scale.
    let recur: [V::Element; 8] = [
        <V::Element as thermite::element::FloatElement>::ConstRatio::<1, 1200>::VALUE,
        <V::Element as thermite::element::FloatElement>::ConstRatio::<1, 3000>::VALUE,
        <V::Element as thermite::element::FloatElement>::ConstRatio::<1, 5600>::VALUE,
        <V::Element as thermite::element::FloatElement>::ConstRatio::<1, 9000>::VALUE,
        <V::Element as thermite::element::FloatElement>::ConstRatio::<1, 13200>::VALUE,
        <V::Element as thermite::element::FloatElement>::ConstRatio::<1, 18200>::VALUE,
        <V::Element as thermite::element::FloatElement>::ConstRatio::<1, 24000>::VALUE,
        <V::Element as thermite::element::FloatElement>::ConstRatio::<1, 30600>::VALUE,
    ];
    let terms = const { zeta_bernoulli_terms(P::POLICY.precision) };

    let twentieth = V::splat(<V::Element as thermite::element::FloatElement>::ConstRatio::<1, 20>::VALUE);
    let mut u = s * n_s * twentieth;
    let mut a = s + one;
    let mut tail = Complex::<V>::ZERO;

    let mut k = 0;
    while k < terms {
        tail += u * V::splat(<V::Element as BernoulliNumbers>::B2N[k]);
        u = u * (a * (a + one)) * V::splat(recur[k]);
        a.re += V::TWO;
        k += 1;
    }

    let zc = (direct + boundary) + tail;
    let full = one + zc;

    let mut result = if const { ZETAC } { zc } else { full };

    if thermite::unlikely(reflect.any()) {
        // zeta(z) = 2^z pi^(z-1) sin(pi z / 2) Gamma(1-z) zeta(1-z), with 2^z pi^(z-1) folded
        // into a single exp2 the same way the real kernel does it.
        let scale = ((z - one) * V::LOG2_PI + z).exp2_p::<P>();
        let sin_h = (z * half).sin_pi_p::<P>();
        let reflected = scale * sin_h * tgamma_impl::<P, V>(one - z) * full;

        let out = if const { ZETAC } { reflected - one } else { reflected };
        result = reflect.select(out, result);
    }

    if const { P::POLICY.check_overflow } {
        // The simple pole, pinned as the real kernel pins it: the boundary term's division
        // by zero would otherwise hand back an `inf`/`NaN` mixture.
        let pole = z.re.cmp_eq(V::ONE) & z.im.is_zero();
        result = pole.select(Complex::real(V::INFINITY), result);
    }

    result
}
