//! Cardinal (`f(x)/x`) kernels: `sinc`, `sinhc`, `sinc_pi` and `atanhc`.

use super::super::*;

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

