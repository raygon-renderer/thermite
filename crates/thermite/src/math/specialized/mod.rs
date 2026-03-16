#![allow(clippy::excessive_precision, clippy::approx_constant)]

use core::marker::PhantomData;

use crate::{
    element::{FloatElement, FloatElementWithBits},
    mask::*,
    math::{
        CoreMathWithPolicy, FloatConsts, RealMathWithPolicy, TranscendentalMathWithPolicy, algorithms,
        policy::policies::{ExtraPrecision, LessPrecision},
    },
    register::NativeCapability,
    vector::{ops::BitAndNot as _, *},
};

// use super::MathWithPolicy;
use super::policy::{DenormalBehavior, Policy, PrecisionPolicy};

mod generic;

impl<E, V> SpecializedFloatMath<E> for V
where
    E: FloatElement,
    V: FloatVectorWithBits<Element = E>,
{
}

pub trait SpecializedFloatMath<E: FloatElementWithBits>: FloatVectorWithBits<Element = E> {
    #[inline(always)]
    fn ldexp<P: Policy>(self, exp: Self::SignedBits) -> Self {
        if const { Self::NATIVE_CAP.has(NativeCapability::LDEXP) } {
            return unsafe { Self::native_ldexp(self, exp) };
        }

        // constants
        let mantissa_bits = <Self::Element as FloatElementWithBits>::MANTISSA_BITS;
        let exp_lsb_mask: Self::Bits = crate::generic_splat!(<Self> = <S: FloatVectorWithBits>
            <S::Bits as GenericVector>::Element: <S::Element as FloatElementWithBits>::EXP_LSB_MASK);
        let sign_mantissa_mask: Self::Bits = crate::generic_splat!(<Self> = <S: FloatVectorWithBits>
            <S::Bits as GenericVector>::Element: <S::Element as FloatElementWithBits>::SIGN_MANTISSA_MASK);
        let max_biased_exp: Self::SignedBits = crate::generic_splat!(<Self> = <S: FloatVectorWithBits>
            <S::SignedBits as GenericVector>::Element: <S::Element as FloatElementWithBits>::MAX_BIASED_EXP);
        let exp_bias: Self::SignedBits = crate::generic_splat!(<Self> = <S: FloatVectorWithBits>
            <S::SignedBits as GenericVector>::Element: <S::Element as FloatElementWithBits>::EXP_BIAS);

        // special handling for denormals when we want to preserve them, since the normal path would flush them to zero
        if const {
            matches!(P::POLICY.denormal_behavior, DenormalBehavior::Preserve if <Self::Element as FloatElement>::HAS_SUBNORMALS)
        } {
            // Two-multiply approach: split the exponent in half so each
            // intermediate value stays representable, and let IEEE gradual
            // underflow produce subnormals naturally.
            let exp1 = exp.srai::<1>();
            let exp2 = exp - exp1;

            let pow2_1 = (exp1 + exp_bias).max(Self::SignedBits::ONE).min(exp_bias + exp_bias) << mantissa_bits;
            let pow2_2 = (exp2 + exp_bias).max(Self::SignedBits::ONE).min(exp_bias + exp_bias) << mantissa_bits;

            let mut result = self * Self::from_bits(pow2_1) * Self::from_bits(pow2_2);

            if const { P::POLICY.check_overflow } {
                result = self.is_nan().select(self, result);
            }

            return result;
        }

        let bits: Self::Bits = self.into_bits();

        let biased_exp = Self::SignedBits::from_bits((bits >> mantissa_bits) & exp_lsb_mask);

        let mut exp = biased_exp + exp;

        if const { P::POLICY.check_overflow } {
            // clamp exponent between 0 and max biased exponent
            exp = exp.max(Self::SignedBits::ZERO).min(max_biased_exp);
        }

        let sign_mantissa = Self::SignedBits::from_bits(bits & sign_mantissa_mask);

        let mut result = (exp << <Self::Element as FloatElementWithBits>::MANTISSA_BITS) | sign_mantissa;

        if const { P::POLICY.check_overflow } {
            let is_underflow = exp.is_negative();
            let input_was_subnormal = biased_exp.is_zero();

            result = result.z(is_underflow | input_was_subnormal); // zero result if underflow or input was subnormal
        }

        Self::from_bits(result)
    }

    #[inline(always)]
    fn frexp<P: Policy>(self) -> (Self, Self::SignedBits) {
        if const { Self::NATIVE_CAP.has(NativeCapability::FREXP) } {
            return unsafe { Self::native_frexp(self) };
        }

        let exp_lsb_mask: Self::Bits = crate::generic_splat!(<Self> = <S: FloatVectorWithBits>
            <S::Bits as GenericVector>::Element: <S::Element as FloatElementWithBits>::EXP_LSB_MASK);
        let frexp_bias_offset: Self::SignedBits = crate::generic_splat!(<Self> = <S: FloatVectorWithBits>
            <S::SignedBits as GenericVector>::Element: <S::Element as FloatElementWithBits>::FREXP_BIAS_OFFSET);
        let sign_mantissa_mask: Self::Bits = crate::generic_splat!(<Self> = <S: FloatVectorWithBits>
            <S::Bits as GenericVector>::Element: <S::Element as FloatElementWithBits>::SIGN_MANTISSA_MASK);
        let half_exp_bits: Self::Bits = crate::generic_splat!(<Self> = <S: FloatVectorWithBits>
            <S::Bits as GenericVector>::Element: <S::Element as FloatElementWithBits>::HALF_EXP_BITS);

        let mut bits: Self::Bits = self.into_bits();
        let orig_bits = bits;

        // if preserving denormals, we need to shift subnormals up to the normal range so that the exponent extraction works correctly
        let subnormal_correction = if const {
            matches!(P::POLICY.denormal_behavior, DenormalBehavior::Preserve if <Self::Element as FloatElement>::HAS_SUBNORMALS)
        } {
            let is_subnormal = self.is_subnormal();

            let exp_bias: Self::SignedBits = crate::generic_splat!(<Self> = <S: FloatVectorWithBits>
                <S::SignedBits as GenericVector>::Element: <S::Element as FloatElementWithBits>::EXP_BIAS);

            let shift_amount = Self::SignedBits::splat(unsafe {
                <E as FloatElementWithBits>::SignedBits::try_from(E::MANTISSA_BITS + 1).unwrap_unchecked()
            });

            let normalizer = Self::from_bits((exp_bias + shift_amount) << E::MANTISSA_BITS);

            // conditional multiplication to normalize subnormal
            bits = self.mul_c(is_subnormal, normalizer).into_bits();

            shift_amount.z(is_subnormal.cast()) // zero if not subnormal
        } else {
            Self::SignedBits::ZERO
        };

        // (bits >> mantissa) & mask
        let biased_exp = Self::SignedBits::from_bits((bits >> E::MANTISSA_BITS) & exp_lsb_mask);

        // subtract bias to get actual exponent
        let mut exp: Self::SignedBits = biased_exp - frexp_bias_offset;

        if const {
            matches!(P::POLICY.denormal_behavior, DenormalBehavior::Preserve if <Self::Element as FloatElement>::HAS_SUBNORMALS)
        } {
            exp -= subnormal_correction; // subtract additional amount if we had to normalize a subnormal
        }

        // extract sign and mantissa, then give it the correct exponent
        let sign_mantissa = bits & sign_mantissa_mask;
        let mut fraction = sign_mantissa | half_exp_bits;

        if const { P::POLICY.check_overflow } {
            let is_finite = self.is_finite() & biased_exp.cmp_ne(Self::SignedBits::ZERO).cast();

            exp = exp.z(is_finite.cast());
            fraction = is_finite.select(fraction, orig_bits);
        }

        (Self::from_bits(fraction), exp)
    }

    #[inline(always)]
    fn flush_denormals<P: Policy>(self) -> Self {
        if const {
            matches!(
                P::POLICY.denormal_behavior,
                DenormalBehavior::Preserve | DenormalBehavior::Ignore
            ) || !<Self::Element as FloatElement>::HAS_SUBNORMALS
        } {
            return self;
        }

        if const { matches!(P::POLICY.denormal_behavior, DenormalBehavior::Crush) } {
            let denormal_trick: Self::Bits = crate::generic_splat!(
                <Self> = <S: FloatVectorWithBits>
                <S::Bits as GenericVector>::Element: <S::Element as FloatElementWithBits>::DENORMAL_TRICK
            );

            let dt = Self::from_bits(denormal_trick);

            return (dt - (dt - self));
        }

        let abs_bits = Self::SignedBits::from_bits(self.abs());

        let max_subnormal: Self::Bits = crate::generic_splat!(
            <Self> = <S: FloatVectorWithBits>
            <S::Bits as GenericVector>::Element: <S::Element as FloatElementWithBits>::MAX_SUBNORMAL
        );

        let max_subnormal_signed: Self::SignedBits = Self::SignedBits::from_bits(max_subnormal);

        // zero self if subnormal (when cmp_gt is false)
        //
        // NOTE: Use a Signed comparison here, since that's faster than unsigned comparisons on most archs,
        // and we know that abs_bits is considered positive as an integer since the msb is zero.
        let mut res = Self::from_bits(self.z(abs_bits.cmp_gt(max_subnormal_signed).cast()));

        // we should preserve -0.0 for greater precision policies
        if const { P::POLICY.precision.gt(PrecisionPolicy::Average) && <Self::Element as FloatElement>::HAS_SIGNED_ZERO }
        {
            // get the sign by xor-ing the non-sign bits, leaving only the sign
            let sign = Self::SignedBits::from_bits(self) ^ abs_bits;

            res |= Self::from_bits(sign); // add back sign
        }

        res
    }
}

/// `AsFloatVectorWithBitsKernel` that calls `flush_denormals` on each input with the given policy.
pub struct FlushDenormals<P: Policy>(PhantomData<P>);

impl<P: Policy> FlushDenormals<P> {
    #[inline(always)]
    pub fn flush_denormals<V: FloatVector, const N: usize>(values: [V; N]) -> Option<[V; N]> {
        V::with_bits(values, FlushDenormals::<P>(PhantomData))
    }
}

impl<P: Policy, const N: usize, V: FloatVector> AsFloatVectorWithBitsKernel<V, N> for FlushDenormals<P> {
    type Output = [V; N];

    #[inline(always)]
    fn with_bits<
        W: FloatVectorWithBits<
                Element = <V>::Element,
                Lanes = <V>::Lanes,
                Mask = <V>::Mask,
                Signed = <V>::Signed,
                Unsigned = <V>::Unsigned,
                ExtendedPrecision = <V as FloatVector>::ExtendedPrecision,
            > + CastVector<V>,
    >(
        self,
        v: [W; N],
    ) -> Self::Output {
        v.map(|v| W::cast_into(v.flush_denormals::<P>()))
    }
}

pub trait SpecializedCoreMath<E>: FloatVector<Element = E> {
    #[inline(always)]
    fn poly<P: Policy, const N: usize>(self, coeffs: &[E; N]) -> Self {
        let x = self;

        if const { !P::POLICY.unroll_loops || P::POLICY.precision.ge(PrecisionPolicy::Best) } {
            // basic Horner's method that's both compact and accurate, even without FMA
            let mut res = Self::splat(coeffs[N - 1]);
            for &c in coeffs.iter().rev().skip(1) {
                res = res.mul_adde(x, Self::splat(c));
            }
            return res;
        }

        // NumVector provides the num_traits::MulAdd implementation needed for fast_polynomial
        let res = fast_polynomial::poly_f_n::<_, _, N>(crate::vector::NumVector(x), |i| unsafe {
            crate::vector::NumVector(Self::splat(*coeffs.get_unchecked(i)))
        });

        res.0
    }

    #[inline(always)]
    fn poly_rev<P: Policy, const N: usize>(self, coeffs: &[E; N]) -> Self {
        let x = self;

        if const { !P::POLICY.unroll_loops || P::POLICY.precision.ge(PrecisionPolicy::Best) } {
            // basic Horner's method that's both compact and accurate, even without FMA
            let mut res = Self::splat(coeffs[0]);
            for &c in coeffs.iter().skip(1) {
                res = res.mul_adde(x, Self::splat(c));
            }
            return res;
        }

        let res = fast_polynomial::poly_f_n::<_, _, N>(crate::vector::NumVector(x), |i| unsafe {
            crate::vector::NumVector(Self::splat(*coeffs.get_unchecked(N - 1 - i)))
        });

        res.0
    }

    #[inline(always)]
    fn poly_rational<P: Policy, const N: usize, const D: usize>(
        self,
        numerator: &[E; N],
        denominator: &[E; D],
    ) -> Self {
        let x = self;

        if const { P::POLICY.precision.le(PrecisionPolicy::Average) } {
            let n = Self::poly::<P, N>(x, numerator);
            let d = Self::poly::<P, D>(x, denominator);

            return n / d;
        }

        let invert = x.cmp_gt(Self::ONE);

        let mut n0 = Self::EMPTY;
        let mut n1 = Self::EMPTY;
        let mut d0 = Self::EMPTY;
        let mut d1 = Self::EMPTY;

        if P::POLICY.avoid_branching || !invert.all() {
            n0 = Self::poly::<P, N>(x, numerator);
            d0 = Self::poly::<P, D>(x, denominator);
        }

        let mut z = Self::EMPTY;

        if P::POLICY.avoid_branching || invert.any() {
            z = Self::reciprocal::<P>(x);
            n1 = Self::poly_rev::<P, N>(z, numerator);
            d1 = Self::poly_rev::<P, D>(z, denominator);
        }

        let n = invert.select(n1, n0);
        let d = invert.select(d1, d0);

        let res = n / d;

        // no correction needed if same degree
        if const { N == D } {
            return res;
        }

        if P::POLICY.avoid_branching || invert.any() {
            // when the degree of the numerator and denominator are different, we need to correct
            // the result by shifting over the difference in degrees
            let (mut u, mut e) = if N < D { (z, D - N) } else { (x, N - D) };

            let mut corrected = res;

            // `res = res * powi(u, e)` assuming e > 0
            // because e > 0 we can jump straight into the loop without a pre-check,
            // and avoid an extra square of u at the end
            loop {
                if e & 1 != 0 {
                    corrected *= u;
                }

                e >>= 1;

                if e == 0 {
                    // correction isn't actually needed for non-inverted case
                    return invert.select(corrected, res);
                }

                u = u.square();
            }
        }

        res
    }

    #[inline(always)]
    fn reciprocal<P: Policy>(self) -> Self {
        let mut y = self.rcp();

        // if we have approximate reciprocal and want better precision
        if const { Self::HAS_APPROX_RCP && P::POLICY.precision.gt(PrecisionPolicy::Worst) } {
            // one iteration of Newton's method
            y = y * self.nmul_adde(y, Self::TWO);
        }

        y
    }

    #[inline(always)]
    fn reciprocal_adde<P: Policy>(self, a: Self) -> Self {
        let mut y = self.rcp();

        if const { Self::HAS_APPROX_RCP && P::POLICY.precision.gt(PrecisionPolicy::Worst) } {
            // one iteration of Newton's method
            y = y.mul_adde(self.nmul_adde(y, Self::TWO), a);
        } else {
            y += a;
        }

        y
    }

    fn inverse_sqrt<P: Policy>(self) -> Self;

    // TODO: Look into better algorithms than doubling
    #[inline(always)]
    fn powi<P: Policy>(self, e: i32) -> Self {
        let mut x = self;
        let mut res = Self::ONE;

        let mut e = if e < 0 {
            x = Self::reciprocal::<P>(x);

            e.wrapping_neg() as u32
        } else {
            e as u32
        };

        while e != 0 {
            if e & 1 != 0 {
                res *= x;
            }

            x = x.square();
            e >>= 1;
        }

        res
    }

    #[inline(always)]
    fn powic<P: Policy, const N: i32>(self) -> Self {
        self.powi_p::<P>(N)
    }

    #[inline(always)]
    fn powiv<P: Policy>(self, mut e: Self::Signed) -> Self {
        let mut x = self;
        let mut res = Self::ONE;

        x = e.is_negative().select(Self::reciprocal::<P>(x), x);
        e = e.abs();

        loop {
            let nx = res * x;

            res = (e & Self::Signed::ONE).is_zero().select(res, nx);

            e >>= 1;

            if e.is_all_zero() {
                return res;
            }

            x = x.square();
        }
    }
}

pub trait SpecializedTranscendentalMath<E>: SpecializedCoreMath<E> {
    fn sin_cos<P: Policy>(self) -> (Self, Self);

    #[inline(always)]
    fn sin<P: Policy>(self) -> Self {
        Self::sin_cos::<P>(self).0
    }

    #[inline(always)]
    fn cos<P: Policy>(self) -> Self {
        Self::sin_cos::<P>(self).1
    }

    #[inline(always)]
    fn tan<P: Policy>(self) -> Self {
        let (s, c) = Self::sin_cos::<P>(self);
        s / c
    }

    #[inline(always)]
    fn sincos_pi<P: Policy>(self) -> (Self, Self) {
        Self::sin_cos::<P>(self * Self::PI)
    }

    #[inline(always)]
    fn sin_pi<P: Policy>(self) -> Self {
        Self::sincos_pi::<P>(self).0
    }

    #[inline(always)]
    fn cos_pi<P: Policy>(self) -> Self {
        Self::sincos_pi::<P>(self).1
    }

    #[inline(always)]
    fn tan_pi<P: Policy>(self) -> Self {
        let (s, c) = Self::sincos_pi::<P>(self);
        s / c
    }

    fn sinc<P: Policy>(self) -> Self;

    #[inline(always)]
    fn sinc_pi<P: Policy>(self) -> Self {
        Self::sinc::<P>(self * Self::PI)
    }

    fn sinh_cosh<P: Policy>(self) -> (Self, Self);

    #[inline(always)]
    fn sinh<P: Policy>(self) -> Self {
        Self::sinh_cosh::<P>(self).0
    }

    #[inline(always)]
    fn cosh<P: Policy>(self) -> Self {
        Self::sinh_cosh::<P>(self).1
    }

    fn tanh<P: Policy>(self) -> Self;

    fn asin<P: Policy>(self) -> Self;
    fn acos<P: Policy>(self) -> Self;
    fn atan<P: Policy>(self) -> Self;

    fn asinh<P: Policy>(self) -> Self;
    fn acosh<P: Policy>(self) -> Self;
    fn atanh<P: Policy>(self) -> Self;

    fn exp<P: Policy>(self) -> Self;
    fn exph<P: Policy>(self) -> Self;
    fn exp2<P: Policy>(self) -> Self;
    fn exp10<P: Policy>(self) -> Self;
    fn exp_m1<P: Policy>(self) -> Self;

    fn powf<P: Policy>(self, e: Self) -> Self;
    fn cbrt<P: Policy>(self) -> Self;

    #[inline(always)]
    fn nth_root<P: Policy, const N: usize>(self) -> Self {
        let mut x = self;

        match N {
            0 => Self::NAN, // undefined
            1 => x,
            2 => x.sqrt(),
            3 => x.cbrt_p::<P>(),
            _ => {
                let mut is_neg = GenericMask::FALSY;

                // for odd powers, work with absolute value and restore sign later
                if const { N & 1 == 1 } {
                    is_neg = x.is_negative();
                    x = x.abs(); // abs is faster than neg_c, just 1 AND
                }

                // initial guess using reduced precision
                let mut y = x.powf_p::<LessPrecision<P>>(Self::splat(E::from_ratio(1, N as i64)));

                // One iteration of Halley's method for nth root
                let y_n = y.powi_p::<P>(N as i32);

                let np1 = Self::splat(E::from_i64((N + 1) as i64));
                let nm1 = Self::splat(E::from_i64((N - 1) as i64));

                let n = y * (x - y_n); // half of numerator
                let d = y_n.mul_adde(np1, x * nm1);

                y += (n + n) / d;

                if const { N & 1 == 1 } {
                    y = y.neg_c(is_neg);
                }

                y
            }
        }
    }

    fn ln<P: Policy>(self) -> Self;
    fn ln_1p<P: Policy>(self) -> Self;
    fn log2<P: Policy>(self) -> Self;
    fn log10<P: Policy>(self) -> Self;

    fn log_n<P: Policy, const N: usize>(self) -> Self;

    #[inline(always)]
    fn log<P: Policy>(self, base: Self) -> Self {
        Self::ln::<P>(self) / Self::ln::<P>(base)
    }

    /// ln(1 - e^(-x))
    #[inline(always)]
    fn ln1m_expnx<P: Policy>(self) -> Self {
        Self::ln::<P>(Self::ONE - Self::exp::<P>(-self))
    }

    fn ln1m_expnx_ext<P: Policy>(self, lnx: Self) -> Self;
}

#[inline(always)]
fn hypot_n_impl<E, V, P, const N: usize, const INV: bool>(mut values: [V; N]) -> V
where
    E: FloatElement,
    V: SpecializedSpatialMath<E>,
    P: Policy,
{
    if let Some(new_values) = FlushDenormals::<P>::flush_denormals(values) {
        values = new_values;
    }

    if N == 0 {
        if INV {
            return V::INFINITY; // 1/0 == infinity
        }

        return V::ZERO;
    }

    if N == 1 {
        let mut res = values[0].abs(); // sqrt(x^2) == abs(x)

        if INV {
            res = res.reciprocal_p::<P>();
        }

        return res;
    }

    // special case N=2 which saves a couple instructions
    if N == 2 {
        let x = values[0];
        let y = values[1];

        return if const { P::POLICY.precision.le(PrecisionPolicy::Worst) } {
            // Use the worst precision method, which is usually faster
            let res = x.mul_adde(x, y.square());

            return if INV { res.inverse_sqrt_p::<P>() } else { res.sqrt() };
        } else {
            // Use a more precise method
            let x = x.abs();
            let y = y.abs();

            let max = x.max(y);
            let min = x.min(y);
            let t = min / max;

            let mut res = max * t.mul_adde(t, V::ONE);

            if INV {
                res = res.inverse_sqrt_p::<P>();

                if P::POLICY.check_overflow {
                    res = max.is_infinite().select(V::ZERO, res);
                }
            } else {
                res = res.sqrt();

                if P::POLICY.check_overflow {
                    res = max.is_infinite().select(max, res);
                }
            }

            res
        };
    }

    if const { P::POLICY.precision.le(PrecisionPolicy::Worst) } {
        // square each value in place, zero dependencies
        for value in values.iter_mut() {
            *value *= *value;
        }

        crate::math::algorithms::reduce_in_place(&mut values, |a, b| a + b);

        return if INV {
            values[0].inverse_sqrt_p::<P>()
        } else {
            values[0].sqrt()
        };
    }

    // high-precision path

    // take absolute value of each element in place, zero dependencies,
    // since we're squaring anyway this doesn't lose any information
    for x in &mut values {
        *x = x.abs();
    }

    let max_abs = crate::math::algorithms::reduce_array(values, |a, b| a.max(b));
    let is_zero = max_abs.cmp_eq(V::ZERO);

    let scale = is_zero.select(V::ONE, max_abs.reciprocal_p::<P>());

    for x in &mut values {
        *x *= scale; // scale to prevent overflow
        *x = x.square(); // square in place
    }

    // sum squares in place
    crate::math::algorithms::reduce_in_place(&mut values, |a, b| a + b);

    let mut res;

    if INV {
        res = scale * values[0].inverse_sqrt_p::<P>();

        if const { P::POLICY.check_overflow } {
            res = max_abs.is_infinite().select(V::ZERO, res);
        }
    } else {
        res = max_abs * values[0].sqrt();

        if const { P::POLICY.check_overflow } {
            res = max_abs.is_infinite().select(max_abs, res);
        }
    }

    res
}

pub trait SpecializedSpatialMath<E>: SpecializedCoreMath<E> {
    // type Scalar: SpecializedRealMath<E>;

    #[inline(always)]
    fn hypot<P: Policy>(self, y: Self) -> Self {
        Self::hypot_n::<P, 2>([self, y])
    }

    #[inline(always)]
    fn hypot_n<P: Policy, const N: usize>(values: [Self; N]) -> Self {
        hypot_n_impl::<E, Self, P, N, false>(values)
    }

    #[inline(always)]
    fn inv_hypot_n<P: Policy, const N: usize>(values: [Self; N]) -> Self {
        hypot_n_impl::<E, Self, P, N, true>(values)
    }

    fn l1_norm<P: Policy>(self) -> Self;

    #[inline(always)]
    fn l2_norm<P: Policy>(self) -> Self {
        Self::l2_norm_squared::<P>(self).sqrt()
    }

    fn l2_norm_squared<P: Policy>(self) -> Self;
}

pub trait SpecializedRealMath<E>: SpecializedTranscendentalMath<E> + SpecializedSpatialMath<E> {
    #[inline(always)]
    fn tolerance<P: Policy>() -> Self {
        Self::splat(Self::Element::from_i64(P::POLICY.precision.tolerance()) * Self::Element::EPSILON)
    }

    #[inline(always)]
    fn to_degrees<P: Policy>(self) -> Self {
        self * Self::FRAC_180_PI
    }

    #[inline(always)]
    fn to_radians<P: Policy>(self) -> Self {
        self * Self::FRAC_PI_180
    }

    #[inline(always)]
    fn wrap_angle<P: Policy>(self) -> Self {
        // self - floor((self + π) / 2π) * 2π
        (-Self::TAU).mul_adde(((self + Self::PI) * (Self::FRAC_1_PI * Self::HALF)).floor(), self)
    }

    #[inline(always)]
    fn angle_diff<P: Policy>(self, other: Self) -> Self {
        (self - other).wrap_angle_p::<P>()
    }

    fn atan2<P: Policy>(self, x: Self) -> Self;

    #[inline(always)]
    fn step<P: Policy>(self, t: Self) -> Self {
        // use z() masked zeroing to avoid branching or select
        Self::ONE.z(self.cmp_ge(t))
    }

    #[inline(always)]
    fn lerp<P: Policy>(self, a: Self, b: Self) -> Self {
        let t = self;

        if const { Self::HAS_TRUE_FMA || P::POLICY.precision.ge(PrecisionPolicy::Reference) } {
            t.mul_add(b - a, a) // Fast and accurate, if available
        } else {
            (Self::ONE - t) * a + t * b // Accurate but slower than FMA
        }
    }

    #[inline(always)]
    fn scale<P: Policy>(self, in_min: Self, in_max: Self, out_min: Self, out_max: Self) -> Self {
        let in_range = in_max - in_min;

        let mut t = self - in_min;

        t = if const { P::POLICY.precision.le(PrecisionPolicy::Worst) } {
            t * in_range.rcp()
        } else {
            t / in_range
        };

        Self::lerp::<P>(t, out_min, out_max)
    }

    #[inline(always)]
    fn smoothstep<P: Policy, const N: usize>(self, edges: Option<(Self, Self)>) -> Self {
        let mut t = self;

        if let Some(new_t) = FlushDenormals::<P>::flush_denormals([t]) {
            t = new_t[0];
        }

        if let Some((a, b)) = edges {
            let xa = t - a;
            let ba = b - a;

            t = if const { P::POLICY.precision.le(PrecisionPolicy::Worst) } {
                xa * ba.rcp()
            } else {
                xa / ba
            };
        }

        if P::POLICY.check_overflow {
            t = t.clamp(Self::ZERO, Self::ONE);
        }

        match N {
            // t was already scaled to between the edges
            0 => Self::step::<P>(t, Self::HALF),
            1 => t, // linear
            _ => {
                let coeffs = const { Smoothstep::<N>::COEFFICIENTS };
                let mut y = Self::splat(E::from_i64(coeffs[0]));

                for &c in &coeffs[1..] {
                    y = y.mul_adde(t, Self::splat(E::from_i64(c)));
                }

                y * t.powi_p::<P>(N as i32)
            }
        }
    }

    #[inline(always)]
    fn smoothstep_derivative<P: Policy, const N: usize>(self, edges: Option<(Self, Self)>) -> Self {
        let mut t = self;
        let mut dt_dx = Self::ONE;

        if let Some(new_t) = FlushDenormals::<P>::flush_denormals([t]) {
            t = new_t[0];
        }

        if let Some((a, b)) = edges {
            let xa = t - a;
            let ba = b - a;

            (dt_dx, t) = if const { P::POLICY.precision.le(PrecisionPolicy::Worst) } {
                let bar = ba.rcp();

                (bar, xa * bar)
            } else {
                (ba.reciprocal_p::<P>(), xa / ba)
            };
        }

        match N {
            // derivative of step function is infinite at 0.5, so-called Dirac delta function
            0 => t.cmp_eq(Self::HALF).select(Self::INFINITY, Self::ZERO),
            1 => dt_dx,
            _ => {
                if P::POLICY.check_overflow {
                    t = t.clamp(Self::ZERO, Self::ONE);
                }

                let coeffs = const { Smoothstep::<N>::COEFFICIENTS };
                let mut y = Self::splat(E::from_i64(coeffs[0] * (2 * N - 1) as i64));
                let mut k = 1;

                for &c in &coeffs[1..] {
                    // order - k for derivative coefficient
                    y = y.mul_adde(t, Self::splat(E::from_i64(c * (2 * N - k - 1) as i64)));
                    k += 1;
                }

                y * dt_dx * t.powi_p::<P>((N - 1) as i32)
            }
        }
    }

    #[inline(always)]
    fn inverse_smoothstep<P: Policy, const N: usize>(mut y: Self, edges: Option<(Self, Self)>) -> Self {
        let mut ba = Self::ONE;
        let mut bar = Self::ONE;
        let mut bar_a = Self::ONE; // (b - a) * a

        if let Some(new_y) = FlushDenormals::<P>::flush_denormals([y]) {
            y = new_y[0];
        }

        //                             // Initial guess: y - 2y * (1 - y) * (y - 0.5)
        // While we have a good initial guess for the inverse, S-curves are most stable at the
        // midpoint, so start there. Converges much faster this way.
        let mut x0 = Self::HALF; //(y + y).nmul_adde((Self::ONE - y) * (y - Self::HALF), y);

        if let Some((a, b)) = edges {
            ba = b - a;

            if const { P::POLICY.precision.le(PrecisionPolicy::Worst) } {
                bar = ba.rcp();
                bar_a = bar * a;
            } else {
                bar = ba.reciprocal_p::<P>();
                bar_a = a / ba;
            }

            match N {
                0 => return y.step_p::<P>(Self::HALF).mul_adde(ba, a),
                1 => return y.mul_adde(ba, a),

                // scale the initial guess to fit the edges
                _ => x0 = x0.mul_adde(ba, a),
            }
        }

        match N {
            0 => return y.step_p::<P>(Self::HALF),
            1 => return y,

            // N=2 has a closed-form solution
            2 => {
                let mut t = y.nmul_adde(Self::TWO, Self::ONE).asin_p::<P>();

                if const { P::POLICY.precision.le(PrecisionPolicy::Medium) } {
                    t *= Self::splat(E::from_ratio(1, 3)); // multiply by 1/3 for medium precision
                } else {
                    // exact division for higher precisions
                    t /= Self::splat(E::from_i64(3));
                }

                t = Self::HALF - t.sin_p::<P>();

                if let Some((a, _)) = edges {
                    // rescale to original edges
                    t = t.mul_adde(ba, a);
                }

                return t;
            }
            _ => {}
        }

        let bounds = edges.or(Some((Self::ZERO, Self::ONE)));

        #[rustfmt::skip]
        let (Ok(v) | Err(v)) = algorithms::newtons_method::<Self, P, _>(x0, Self::tolerance::<P>(), bounds, #[inline(always)] move |x: Self| {
            let mut t = x;
            let dt_dx = bar;

            if edges.is_some() {
                // adjust by precalculated scales
                t = t.mul_sube(bar, bar_a);
            }

            let xn1 = t.powi_p::<P>((N - 1) as i32);

            let coeffs = const { Smoothstep::<N>::COEFFICIENTS };

            let mut fx = Self::splat(E::from_i64(coeffs[0]));
            let mut fpx = Self::splat(E::from_i64(coeffs[0] * (2 * N - 1) as i64));

            let mut k = 1;

            for &c in &coeffs[1..] {
                fx = fx.mul_adde(t, Self::splat(E::from_i64(c)));
                fpx = fpx.mul_adde(t, Self::splat(E::from_i64(c * (2 * N - k - 1) as i64)));

                k += 1;
            }

            (t.mul_sube(xn1 * fx, y), (fpx * dt_dx * xn1).min(Self::HALF))
        });

        v
    }

    #[inline(always)]
    fn smooth_interpolator<P: Policy>(x: Self, edges: Option<(Self, Self)>, k: Self) -> Self {
        let mut t = x;

        if let Some(new_t) = FlushDenormals::<P>::flush_denormals([t]) {
            t = new_t[0];
        }

        if let Some((a, b)) = edges {
            // rescale t to [0, 1]
            t = (t - a) / (b - a);
        }

        let kt = k * t;

        // (2x-1) / (kx^2-kx)
        let e = t.mul_sube(Self::TWO, Self::ONE) / kt.mul_sube(t, kt);

        // exp(e) + 1
        let d = e.exp_p::<P>() + Self::ONE;

        // 1/(exp(e) + 1), it's important this is done in extra precision
        let mut res = d.reciprocal_p::<ExtraPrecision<P>>();

        let overflow = e.is_infinite();

        // If the denominator is small enough, it could cause overflow,
        // however that only really happens when t is very close to 0 or 1,
        // or when k is very small. So approximate it with a step function.
        if P::POLICY.avoid_branching || crate::unlikely(overflow.any()) {
            res = overflow.select(t.step_p::<P>(Self::HALF), res);
        }

        // these are important since the Exp formulation is discontinuous at 0 and 1,
        // and this maintains the asymptotes when t is outside the range [0, 1]
        res = t.cmp_ge(Self::ONE).select(Self::ONE, res);
        res = t.cmp_le(Self::ZERO).select(Self::ZERO, res);

        res
    }

    #[inline(always)]
    fn smooth_interpolator_inverse<P: Policy>(y: Self, edges: Option<(Self, Self)>, k: Self) -> Self {
        // k ln(1/y - 1)
        let l = k * (y.reciprocal_p::<P>() - Self::ONE).ln_p::<P>();

        // ((l + 2) - sqrt(l^2 + 4)) / 2l
        let a = l + Self::TWO;
        let b = l.mul_adde(l, Self::splat(E::from_i64(4))).sqrt();
        let mut t = (a - b) / (Self::TWO * l);

        // handle out-of-bounds inputs
        t = y.cmp_ge(Self::ONE).select(Self::ONE, t);
        t = y.cmp_le(Self::ZERO).select(Self::ZERO, t);

        if let Some((a, b)) = edges {
            // rescale t to the original edges
            t = t.mul_adde(b - a, a);
        }

        t
    }
}

// /// Provides specialized math routines for the given element type.
// ///
// /// Vectors implementing this will automatically have the [`Math`] and [`MathWithPolicy`] traits
// /// implemented for them.
// pub trait SpecializedMath<E>: SpecializedCoreMath<E> {
//     fn erf<P: Policy>(self) -> Self;

//     #[inline(always)]
//     fn erfc<P: Policy>(self) -> Self {
//         Self::ONE - Self::erf::<P>(self) // erfc(x) = 1 - erf(x), fallback implementation
//     }

//     fn erfinv<P: Policy>(self) -> Self;
// }

mod pd;
mod ps;

#[derive(Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
enum ExpMode {
    Exp = 0,
    Expm1,
    Exph,
    Pow2,
    Pow10,
}

const EXP_MODE_EXP: u8 = ExpMode::Exp as u8;
const EXP_MODE_EXPM1: u8 = ExpMode::Expm1 as u8;
const EXP_MODE_EXPH: u8 = ExpMode::Exph as u8;
const EXP_MODE_POW2: u8 = ExpMode::Pow2 as u8;
const EXP_MODE_POW10: u8 = ExpMode::Pow10 as u8;

const fn binomial(a: i32, b: i32) -> i64 {
    if b <= 0 {
        return 1;
    }

    let mut res: i64 = 1;
    let mut i = 0;

    while i < b {
        let n: i64 = res * (a - i) as i64;
        res = n / (i + 1) as i64;

        i += 1;
    }

    res
}

// const fn const_powi(base: i64, exp: i32) -> i64 {
//     let mut result: i64 = 1;
//     let mut b = base;
//     let mut e = exp;

//     while e > 0 {
//         if e & 1 != 0 {
//             result = result * b;
//         }

//         e >>= 1;

//         if e == 0 {
//             break;
//         }

//         b *= b;
//     }

//     result
// }

// /// Returns (numerator, denominator) of the n-th generalized harmonic number of order m
// const fn generalized_harmonic(n: i32, m: i32) -> (i64, i64) {
//     let mut n = 0;
//     let mut d = 0;

//     let mut k = 1;

//     while k <= n {
//         d += const_powi(k, m);
//         n += 1;
//     }

//     (n, d)
// }

pub struct Smoothstep<const N: usize>(PhantomData<[i64; N]>);

impl<const N: usize> Smoothstep<N> {
    // ensure these coefficients are generated at compile time
    pub const COEFFICIENTS: [i64; N] = const {
        let mut coeffs = [0; N];
        let n = (N - 1) as i32;

        let mut k = 0;
        while k < N {
            let c = binomial(-1 - n, k as i32) * binomial(n + n + 1, n - k as i32);

            // store in reverse order for easier polynomial evaluation
            coeffs[N - k - 1] = c;
            k += 1;
        }

        coeffs
    };
}
