//! The complex Gamma family: `tgamma`, `lgamma`, `digamma`, `trigamma`, and
//! [`polygamma`](self::polygamma) in its own file.
//!
//! Every body here is the real algorithm in complex arithmetic. The pieces that survive
//! the crossing are the analytic ones (the Lanczos sums, the asymptotic expansions),
//! and the pieces that do not are the minimax fits to real intervals, which is why
//! `digamma` and `trigamma` lean on a recurrence where the real versions reach for a
//! rational. See the crate-level notes in the parent module.

pub mod polygamma;

use thermite::math::policy::Policy;
use thermite::math::{CoreMathWithPolicy as _, FloatConsts, TranscendentalMathWithPolicy as _};
use thermite::prelude::*;
use thermite_special::tables::primal::GammaPrimalTables;

use crate::Complex;
use crate::math::ComplexMathWithPolicy as _;
use crate::vector::RealFloatVector;

/// Shared complex `tgamma`, by the Lanczos approximation.
///
/// Lanczos is an analytic approximation, not a minimax fit, so it carries over from
/// the real implementation unchanged apart from the arithmetic. The left half-plane
/// comes from the reflection formula `Gamma(z)Gamma(1-z) = pi/sin(pi z)`.
///
/// Unlike the real version this does not split the `pow` in two for large arguments,
/// so it overflows around `Re z ~ 171` (f64) rather than reaching the very top of the
/// range. It also has no integer fast path: over C that test would only fire on a
/// measure-zero set.
#[inline(always)]
pub(crate) fn tgamma_impl<P: Policy, V: RealFloatVector>(z: Complex<V>) -> Complex<V>
where
    V::Primal: GammaPrimalTables<<V::Primal as GenericVector>::Element>,
{
    let l = <V::Primal as GammaPrimalTables<_>>::lanczos_primal();

    let reflect = z.re.cmp_lt(V::HALF);
    let w = reflect.select(Complex::ONE - z, z);

    let gh = V::from_primal(l.g) - V::HALF;
    let zgh = Complex::new(w.re + gh, w.im);

    let lanczos = w.poly_rev_n_primal_p::<P, _>(&l.p_rev) / w.poly_rev_n_primal_p::<P, _>(&l.q_rev);

    // zgh^(w - 1/2) * e^(-zgh) * lanczos_sum(w), with the two exponentials folded into
    // one: exp((w - 1/2) ln(zgh) - zgh).
    //
    // The real version cannot do this. It calls `powf` and then divides by `exp(zgh)`,
    // so the `pow` overflows on its own well before the product does, and has to
    // split the exponent in half and square the result to compensate. Folding removes
    // the intermediate entirely: nothing overflows until the answer does. It is also
    // one transcendental cheaper, `powf` being `exp(e ln x)` underneath.
    //
    // `Re zgh >= g > 0` on this branch, so the `ln` never approaches its cut.
    let e = Complex::new(w.re - V::HALF, w.im);
    let res = (e * zgh.ln_p::<P>() - zgh).exp_p::<P>() * lanczos;

    // Gamma(z) = pi / (sin(pi z) Gamma(1 - z))
    let refl = Complex::real(<V as FloatConsts>::PI) / (z.sin_pi_p::<P>() * res);

    reflect.select(refl, res)
}

/// Shared complex `lgamma`, from the `exp(g)`-scaled Lanczos sum.
///
/// # Branch
///
/// For `Re z >= 1/2` this is the _continuous_ log-gamma, not merely a principal
/// value: both logarithms it takes are of arguments confined to the right half-plane,
/// so neither crosses the cut, and the large imaginary parts come from the
/// `(z - 1/2) ln(zgh)` product rather than from a wrapped logarithm.
///
/// The reflected half-plane is another matter: `ln(sin(pi z))` is principal there, so
/// the result can differ from the continuous branch by a multiple of `2 pi i`.
#[inline(always)]
pub(crate) fn lgamma_impl<P: Policy, V: RealFloatVector>(z: Complex<V>) -> Complex<V>
where
    V::Primal: GammaPrimalTables<<V::Primal as GenericVector>::Element>,
{
    let l = <V::Primal as GammaPrimalTables<_>>::lanczos_primal();

    let reflect = z.re.cmp_lt(V::HALF);
    let w = reflect.select(Complex::ONE - z, z);

    let b = Complex::new(w.re - V::HALF, w.im);
    let a = Complex::new(b.re + V::from_primal(l.g), b.im).ln_p::<P>() - Complex::ONE;

    let s = w.poly_n_primal_p::<P, _>(&l.p_expg_scaled) / w.poly_n_primal_p::<P, _>(&l.q);

    let res = a * b + s.ln_p::<P>();

    // ln Gamma(z) = ln(pi) - ln(sin(pi z)) - ln Gamma(1 - z)
    let refl = Complex::real(<V as FloatConsts>::LN_PI) - z.sin_pi_p::<P>().ln_p::<P>() - res;

    reflect.select(refl, res)
}

/// Shared complex `digamma`.
///
/// Only `p_large` of the real [`Digamma`] table is usable here: the `[1, 2]` rational
/// beside it is a minimax fit to a real interval and says nothing off the axis, while
/// `p_large` is a genuine asymptotic series. So the recurrence does the work the
/// rational does in the real version, walking `Re z` up to `shift` before expanding.
#[inline(always)]
pub(crate) fn digamma_impl<P: Policy, V: RealFloatVector>(z: Complex<V>) -> Complex<V>
where
    V::Primal: GammaPrimalTables<<V::Primal as GenericVector>::Element>,
{
    let p_large = <V::Primal as GammaPrimalTables<_>>::digamma_p_large();

    let reflect = z.re.cmp_lt(V::HALF);

    let mut w = reflect.select(Complex::ONE - z, z);
    let mut refl = Complex::<V>::ZERO;

    if const { P::POLICY.avoid_branching } || reflect.any() {
        // psi(z) = psi(1 - z) - pi cot(pi z)
        let (s, c) = z.sincos_pi_p::<P>();
        refl = -(c / s * Complex::real(<V as FloatConsts>::PI));
    }

    // psi(w) = psi(w + 1) - 1/w, walked until the series below applies.
    let shift = V::from_primal(<V::Primal as GammaPrimalTables<_>>::digamma_shift());
    let mut acc = Complex::<V>::ZERO;
    let mut active = w.re.cmp_lt(shift);

    while active.any() {
        acc = active.select(acc - w.finv_p::<P>(), acc);
        w = active.select(w + Complex::ONE, w);
        active = w.re.cmp_lt(shift);
    }

    // psi(w) ~ ln(w-1) + 1/(2(w-1)) - u P(u),  u = 1/(w-1)^2
    let xm1 = w - Complex::ONE;
    let u = (xm1 * xm1).finv_p::<P>();

    let psi = xm1.ln_p::<P>() + (xm1 + xm1).finv_p::<P>() - u * u.poly_n_primal_p::<P, _>(&p_large);

    let total = acc + psi;

    reflect.select(refl + total, total)
}

/// Shared complex trigamma.
///
/// Three stages, none of them the real implementation's: that one leans on minimax
/// rationals fitted to intervals of the real line, which say nothing off the axis.
/// This is the classical route instead (reflect, recurse, expand).
///
/// * `bernoulli` are `$B_2, B_4, \ldots$` in order, the asymptotic series coefficients.
/// * `shift` is the `Re z` the recurrence walks up to before that series is used.
///   Both are the per-element tuning knobs: a shorter table wants a larger shift.
#[inline(always)]
pub(crate) fn trigamma_impl<P: Policy, V: RealFloatVector, const NB: usize>(
    z: Complex<V>,
    bernoulli: &[V::Element; NB],
    shift: V::Element,
) -> Complex<V> {
    // Reflect the left half-plane: psi_1(z) + psi_1(1 - z) = pi^2 / sin^2(pi z).
    let reflect = z.re.cmp_lt(V::HALF);

    let mut w = reflect.select(Complex::ONE - z, z);
    let mut refl = Complex::<V>::ZERO;

    if const { P::POLICY.avoid_branching } || reflect.any() {
        let s = z.sin_pi_p::<P>();
        refl = Complex::real(<V as FloatConsts>::PI_SQUARED) / (s * s);
    }

    // psi_1(w) = 1/w^2 + psi_1(w + 1), walked until Re w is large enough for the
    // series below. Bounded: the reflection already put Re w >= 1/2, so this runs at
    // most `shift` times.
    let shift = V::splat(shift);
    let mut acc = Complex::<V>::ZERO;
    let mut active = w.re.cmp_lt(shift);

    while active.any() {
        let t = (w * w).finv_p::<P>();
        acc = active.select(acc + t, acc);
        w = active.select(w + Complex::ONE, w);
        active = w.re.cmp_lt(shift);
    }

    // psi_1(w) ~ 1/w + 1/(2w^2) + sum_k B_2k / w^(2k+1)
    let u = w.finv_p::<P>();
    let u2 = u * u;

    // Horner in u^2, leading term first. The coefficients are real, so each step is
    // two FMAs on the real part (the coefficient rides the second) and two on the
    // imaginary, never a complex multiply followed by a separate add.
    let mut tail = Complex::real(V::splat(bernoulli[NB - 1]));
    let mut i = NB - 1;
    while i > 0 {
        i -= 1;

        let c = V::splat(bernoulli[i]);

        tail = Complex::new(
            tail.im.nmul_adde(u2.im, tail.re.mul_adde(u2.re, c)),
            tail.re.mul_adde(u2.im, tail.im * u2.re),
        );
    }

    let psi = acc + u + u2 * V::HALF + (u2 * u) * tail;

    reflect.select(refl - psi, psi)
}
