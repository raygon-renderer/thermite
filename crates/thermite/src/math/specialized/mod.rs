#![allow(clippy::excessive_precision)]

use core::marker::PhantomData;

use crate::{
    mask::Mask,
    math::{
        CoreMathWithPolicy, FloatConsts, RealMathWithPolicy, SpatialMathWithPolicy, TranscendentalMathWithPolicy,
        algorithms, policy::policies::ExtraPrecision,
    },
    register::{FloatElement, FloatRegister, Register, SignedIntegerRegister},
    vector::{generic::*, num::NumVector},
};

// use super::MathWithPolicy;
use super::policy::{Policy, PolicyParameters, PrecisionPolicy};

pub trait SpecializedCoreMath<E>: FloatVector<Element = E> {
    #[inline(always)]
    fn ldexp<P: Policy>(self, exp: Self::Signed) -> Self {
        if const { <Self::Register as FloatRegister>::HAS_NATIVE_LDEXP } {
            let (value, exp) = (self.register(), exp.register());

            return Self::from_register(Self::Register::native_ldexp(value, exp));
        }

        let bits: Self::Bits = self.into_bits();

        let exp_lsb_mask: Self::Bits = crate::generic_splat!(
            <Self> = <S: FloatVector>
            <S::Bits as GenericVector>::Element: <S::Element as FloatElement>::EXP_LSB_MASK
        );

        let sign_mantissa_mask: Self::Bits = crate::generic_splat!(
            <Self> = <S: FloatVector>
            <S::Bits as GenericVector>::Element: <S::Element as FloatElement>::SIGN_MANTISSA_MASK
        );

        let biased_exp = Self::Signed::from_bits((bits >> <Self::Element as FloatElement>::MANTISSA) & exp_lsb_mask);

        let mut exp = biased_exp + exp;

        if const { P::POLICY.check_overflow } {
            // clamp exponent between 0 and max biased exponent
            exp = exp.max(Self::Signed::ZERO).min(crate::generic_splat!(
                <Self> = <S: FloatVector>
                <S::Signed as GenericVector>::Element: <S::Element as FloatElement>::MAX_BIASED_EXP
            ));
        }

        let sign_mantissa = Self::Signed::from_bits(bits & sign_mantissa_mask);

        let mut result = (exp << <Self::Element as FloatElement>::MANTISSA) | sign_mantissa;

        if const { P::POLICY.check_overflow } {
            let is_underflow = exp.cmp_le(Self::Signed::ZERO);
            let input_was_subnormal = biased_exp.cmp_eq(Self::Signed::ZERO);

            result = (is_underflow | input_was_subnormal).value().bitandnot(result);
        }

        Self::from_bits(result)
    }

    #[inline(always)]
    fn frexp<P: Policy>(self) -> (Self, Self::Signed) {
        if const { <Self::Register as FloatRegister>::HAS_NATIVE_FREXP } {
            let (mantissa, exp) = Self::Register::native_frexp(self.register());

            return (Self::from_register(mantissa), Self::Signed::from_register(exp));
        }

        let bits: Self::Bits = self.into_bits();

        let exp_lsb_mask: Self::Bits = crate::generic_splat!(
            <Self> = <S: FloatVector>
            <S::Bits as GenericVector>::Element: <S::Element as FloatElement>::EXP_LSB_MASK
        );

        let frexp_bias_offset: Self::Signed = crate::generic_splat!(
            <Self> = <S: FloatVector>
            <S::Signed as GenericVector>::Element: <S::Element as FloatElement>::FREXP_BIAS_OFFSET
        );

        let sign_mantissa_mask: Self::Bits = crate::generic_splat!(
            <Self> = <S: FloatVector>
            <S::Bits as GenericVector>::Element: <S::Element as FloatElement>::SIGN_MANTISSA_MASK
        );

        let half_exp_bits: Self::Bits = crate::generic_splat!(
            <Self> = <S: FloatVector>
            <S::Bits as GenericVector>::Element: <S::Element as FloatElement>::HALF_EXP_BITS
        );

        // (bits >> mantissa) & mask
        let biased_exp = Self::Signed::from_bits((bits >> E::MANTISSA) & exp_lsb_mask);

        // subtract bias to get actual exponent
        let mut exp: Self::Signed = biased_exp - frexp_bias_offset;

        // extract sign and mantissa, then give it the correct exponent
        let sign_mantissa = bits & sign_mantissa_mask;
        let mut fraction = sign_mantissa | half_exp_bits;

        if const { P::POLICY.check_overflow } {
            // if input was zero or subnormal, set fraction to zero and exponent to zero
            let is_normal = biased_exp.cmp_ne(Self::Signed::ZERO).value();
            exp &= is_normal;
            fraction &= Self::Bits::from_bits(is_normal);
        }

        (Self::from_bits(fraction), exp)
    }

    #[inline(always)]
    fn tolerance<P: Policy>() -> Self {
        Self::splat(Self::Element::from_i64(P::POLICY.precision.tolerance()) * Self::Element::EPSILON)
    }

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
        let res = fast_polynomial::poly_f_n::<_, _, N>(NumVector(x), |i| unsafe {
            NumVector(Self::splat(*coeffs.get_unchecked(i)))
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

        let res = fast_polynomial::poly_f_n::<_, _, N>(NumVector(x), |i| unsafe {
            NumVector(Self::splat(*coeffs.get_unchecked(N - 1 - i)))
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

                u *= u;
            }
        }

        res
    }

    #[inline(always)]
    fn reciprocal<P: Policy>(self) -> Self {
        if const { !Self::Register::HAS_APPROX_RCP || P::POLICY.precision.ge(PrecisionPolicy::Average) } {
            Self::ONE / self
        } else {
            let mut y = self.rcp();

            if const { P::POLICY.precision.gt(PrecisionPolicy::Worst) } {
                // one iteration of Newton's method
                y = y * self.nmul_adde(y, Self::TWO);
            }

            y
        }
    }

    #[inline(always)]
    fn reciprocal_adde<P: Policy>(self, a: Self) -> Self {
        if const { !Self::Register::HAS_APPROX_RCP || P::POLICY.precision.ge(PrecisionPolicy::Average) } {
            a + Self::ONE / self
        } else {
            let mut y = self.rcp();

            if const { P::POLICY.precision.gt(PrecisionPolicy::Worst) } {
                // one iteration of Newton's method
                y = y.mul_adde(self.nmul_adde(y, Self::TWO), a);
            } else {
                y += a;
            }

            y
        }
    }

    #[inline(always)]
    fn inverse_sqrt<P: Policy>(self) -> Self {
        if const { !Self::Register::HAS_APPROX_RSQRT || P::POLICY.precision.ge(PrecisionPolicy::Best) } {
            Self::ONE / self.sqrt()
        } else {
            let mut y = self.rsqrt();

            if const { P::POLICY.precision.gt(PrecisionPolicy::Worst) } {
                let nx2 = Self::splat(E::from_f64(-0.5));
                let threehalfs = Self::splat(E::from_f64(1.5));

                // one iteration of Newton's method
                y = y * (y * y).mul_adde(nx2, threehalfs);
            }

            y
        }
    }

    #[inline(always)]
    fn powi<P: Policy>(self, mut e: i32) -> Self {
        let mut x = self;
        let mut res = Self::ONE;

        if e < 0 {
            e = -e;
            x = Self::reciprocal::<P>(x);
        }

        while e != 0 {
            if e & 1 != 0 {
                res *= x;
            }

            x *= x;
            e >>= 1;
        }

        res
    }

    #[inline(always)]
    fn powiv<P: Policy>(self, mut e: Self::Signed) -> Self {
        let mut x = self;
        let mut res = Self::ONE;

        x = e.is_negative().select(Self::reciprocal::<P>(x), x);
        e = e.abs();

        loop {
            let mut e1 = e & Self::Signed::ONE;

            let nx = res * x;

            // NOTE: e1 is bitcast to Self when `select` is used, so we use it for the MSB_BLENDV hack
            // requirements
            res = if <Self::Register as Register>::HAS_MSB_BLENDV {
                // Move the lowest bit to the highest bit position
                e1 <<= const { core::mem::size_of::<<Self::Signed as GenericVector>::Element>() as u32 * 8 - 1 };

                // Blend the result based on the highest bit of e1
                <Self::Signed as MaskedVector>::Mask::from_unchecked(e1).select(nx, res)
            } else {
                e1.cmp_ne(Self::Signed::ZERO).select(nx, res)
            };

            x *= x;
            e >>= 1;

            if e.cmp_ne(Self::Signed::ZERO).none() {
                return res;
            }
        }

        res
    }

    #[inline(always)]
    fn lerp<P: Policy>(self, a: Self, b: Self) -> Self {
        let t = self;

        if const { Self::Register::HAS_TRUE_FMA || P::POLICY.precision.ge(PrecisionPolicy::Reference) } {
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

    #[inline(always)]
    fn sinc<P: Policy>(self) -> Self {
        let x = self;

        if const { P::POLICY.precision.le(PrecisionPolicy::Medium) } {
            return Self::sin::<P>(x) * Self::reciprocal::<P>(x);
        }

        let is_tiny = x.abs().cmp_le(Self::FOURTH_ROOT_EPSILON);

        let x2 = x * x;

        // if branching, use Taylor series for tiny x without calling sine.
        if !P::POLICY.avoid_branching && crate::unlikely(is_tiny.all()) {
            if const { P::POLICY.precision.le(PrecisionPolicy::Average) } {
                // use fma instead of division then subtraction, for improved performance
                // at the cost of a tiny bit of precision with the 120 denominator
                return x2.mul_adde(
                    x2.mul_sube(Self::splat(FloatElement::from_f64(1.0 / 120.0)), Self::FRAC_1_6),
                    Self::ONE,
                );
            }

            let res = x2 / Self::splat(FloatElement::from_i64(120));
            return x2.mul_add(res - Self::FRAC_1_6, Self::ONE);
        }

        // For very small x, sinc(x) ~ 1 - x^2/6 + x^4/120
        let num = is_tiny.select(x2, Self::sin::<P>(x));
        let den = is_tiny.select(Self::splat(FloatElement::from_i64(120)), x);

        // combined division, since division is expensive
        let mut y = num / den;

        y = is_tiny.select(x2.mul_adde(y - Self::FRAC_1_6, Self::ONE), y);

        if P::POLICY.check_overflow {
            y = x.is_infinite().select(Self::ZERO, y);
        }

        y
    }

    #[inline(always)]
    fn sinc_pi<P: Policy>(self) -> Self {
        let x = self;

        if const { P::POLICY.precision.le(PrecisionPolicy::Medium) } {
            // forwards to the above medium-precision sinc implementation,
            // which uses sin(x) * rcp(x)
            return Self::sinc::<P>(x * FloatConsts::PI);
        }

        let pi_2_frac_6: Self = Self::splat(FloatElement::from_f64(
            1.6449340668482264364724151666460251892189499012068, // pi^2/6
        ));

        let frac_120_pi_4: Self = Self::splat(FloatElement::from_f64(
            1.2319178705621202226983339920542432970193362224366, // 120/pi^4, flipped for division
        ));

        let is_tiny = x.abs().cmp_le(Self::FOURTH_ROOT_EPSILON);

        let x2 = x * x;

        // if branching, use Taylor series for tiny x without calling sine.
        if !P::POLICY.avoid_branching && crate::unlikely(is_tiny.all()) {
            let pi_4_frac_120: Self = Self::splat(FloatElement::from_f64(
                0.81174242528335364363700277240587592708106321393905,
            ));

            // unlike sinc, which has x^2/120 with 120 being an exact integer,
            // sinc_pi has pi^4/120, and since pi is irrational and imprecise anyway, we
            // can avoid the exact division by 120 in favor of multiplying by pi^4/120
            return x2.mul_add(x2.mul_sube(pi_4_frac_120, pi_2_frac_6), Self::ONE);
        }

        // for very small x, sinc_pi(x) ~ 1 - (pi^2/6)*x^2 + (pi^4/120)*x^4
        let num = is_tiny.select(x2, Self::sin_pi::<P>(x));
        let den = is_tiny.select(frac_120_pi_4, x * Self::PI); // NOTE: first term is flipped for division

        // combined division, since division is expensive
        let mut y = num / den;

        y = is_tiny.select(x2.mul_adde(y - pi_2_frac_6, Self::ONE), y);

        if P::POLICY.check_overflow {
            y = x.is_infinite().select(Self::ZERO, y);
        }

        y
    }

    fn sinh_cosh<P: Policy>(self) -> (Self, Self);

    fn sinh<P: Policy>(self) -> Self;
    fn cosh<P: Policy>(self) -> Self;
    fn tanh<P: Policy>(self) -> Self;

    fn asin<P: Policy>(self) -> Self;
    fn acos<P: Policy>(self) -> Self;
    fn atan<P: Policy>(self) -> Self;
    fn atan2<P: Policy>(self, x: Self) -> Self;

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

    fn ln<P: Policy>(self) -> Self;
    fn ln_1p<P: Policy>(self) -> Self;
    fn log2<P: Policy>(self) -> Self;
    fn log10<P: Policy>(self) -> Self;

    /// log with arbitrary base N
    #[inline(always)]
    fn log_n<P: Policy, const N: usize>(self) -> Self {
        let x = self;

        match N {
            // 0 and 1 are special cases, and these are what Wolfram Alpha returns
            0 => Self::ZERO,            // log(x)/log(0) = log(x)/-infinity = 0
            1 => FloatVector::INFINITY, // log(x)/log(1) = log(x)/0 = complex infinity, only return real part
            2 => Self::log2::<P>(x),
            10 => Self::log10::<P>(x),
            n if n <= 32 => {
                #[rustfmt::skip] #[allow(clippy::approx_constant)]
                const LOG_TABLE: [f64; 30] = [ // precomputed 1/Table[log(x), {x, 3, 32}]
                    1.0 / 1.0986122886681096913952452369225257046474905578227, 1.0 / 1.3862943611198906188344642429163531361510002687205,
                    1.0 / 1.6094379124341003746007593332261876395256013542685, 1.0 / 1.7917594692280550008124773583807022727229906921830,
                    1.0 / 1.9459101490553133051053527434431797296370847295819, 1.0 / 2.0794415416798359282516963643745297042265004030808,
                    1.0 / 2.1972245773362193827904904738450514092949811156455, 1.0 / 2.3025850929940456840179914546843642076011014886288,
                    1.0 / 2.3978952727983705440619435779651292998217068539374, 1.0 / 2.4849066497880003102297094798388788407984908265433,
                    1.0 / 2.5649493574615367360534874415653186048052679447602, 1.0 / 2.6390573296152586145225848649013562977125848639421,
                    1.0 / 2.7080502011022100659960045701487133441730919120913, 1.0 / 2.7725887222397812376689284858327062723020005374410,
                    1.0 / 2.8332133440562160802495346178731265355882030125857, 1.0 / 2.8903717578961646922077225953032279773704812500058,
                    1.0 / 2.9444389791664404600090274318878535372373792612991, 1.0 / 2.9957322735539909934352235761425407756766016229890,
                    1.0 / 3.0445224377234229965005979803657054342845752874046, 1.0 / 3.0910424533583158534791756994233058678972069882977,
                    1.0 / 3.1354942159291496908067528318101961184423803148404, 1.0 / 3.1780538303479456196469416012970554088739909609035,
                    1.0 / 3.2188758248682007492015186664523752790512027085370, 1.0 / 3.2580965380214820454707195630234951728807680791205,
                    1.0 / 3.2958368660043290741857357107675771139424716734682, 1.0 / 3.3322045101752039239398169863595328657880849983024,
                    1.0 / 3.3672958299864740271832720323619116054945129139227, 1.0 / 3.4011973816621553754132366916068899122485920464515,
                    1.0 / 3.4339872044851462459291643245423572104499389304806, 1.0 / 3.4657359027997265470861606072908828403775006718013,
                ];

                Self::ln::<P>(x) * Self::splat(E::from_f64(LOG_TABLE[n - 3]))
            }
            _ => Self::ln::<P>(x) / Self::splat(E::from_f64(libm::log(N as f64))),
        }
    }

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

pub trait SpecializedSpatialMath<E>: SpecializedCoreMath<E> {
    #[inline(always)]
    fn hypot<P: Policy>(self, y: Self) -> Self {
        let x = self;

        if const { P::POLICY.precision.le(PrecisionPolicy::Worst) } {
            // Use the worst precision method, which is usually faster
            x.mul_adde(x, y * y).sqrt()
        } else {
            // Use a more precise method
            let x = x.abs();
            let y = y.abs();

            let max = x.max(y);
            let min = x.min(y);
            let t = min / max;

            let mut res = max * t.mul_adde(t, Self::ONE).sqrt();

            if P::POLICY.check_overflow {
                // because these have already been abs, we can just use less-than
                let inf = Self::INFINITY;
                res = (x.cmp_lt(inf) & y.cmp_lt(inf) & t.cmp_lt(inf)).select(res, x + y);
            }

            res
        }
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
    fn to_degrees<P: Policy>(self) -> Self {
        self * Self::FRAC_180_PI
    }

    #[inline(always)]
    fn to_radians<P: Policy>(self) -> Self {
        self * Self::FRAC_PI_180
    }

    #[inline(always)]
    fn step<P: Policy>(self, t: Self) -> Self {
        // bitwise AND is much faster than blendv
        self.cmp_ge(t).value() & Self::ONE
    }

    #[inline(always)]
    fn smoothstep<P: Policy, const N: usize>(self, edges: Option<(Self, Self)>) -> Self {
        let mut t = self;

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
                t.powi_p::<P>(N as i32)
                    * const { Smoothstep::<E, N>::COEFFICIENTS }
                        .into_iter()
                        .fold(Self::ZERO, |res, c| res.mul_adde(t, Self::splat(E::from_i64(c))))
            }
        }
    }

    #[inline(always)]
    fn smoothstep_derivative<P: Policy, const N: usize>(self, edges: Option<(Self, Self)>) -> Self {
        let mut t = self;
        let mut dt_dx = Self::ONE;

        if let Some((a, b)) = edges {
            let xa = t - a;
            let ba = b - a;

            (dt_dx, t) = if const { P::POLICY.precision.le(PrecisionPolicy::Worst) } {
                let bar = ba.rcp();

                (bar, xa * bar)
            } else {
                (Self::ONE / ba, xa / ba)
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

                let y = const { Smoothstep::<E, N>::COEFFICIENTS }.into_iter().enumerate().fold(
                    Self::ZERO,
                    |res, (k, c)| {
                        // order - k for derivative coefficient
                        res.mul_adde(t, Self::splat(E::from_i64(c) * E::from_i64((2 * N - k - 1) as i64)))
                    },
                );

                y * dt_dx * t.powi_p::<P>((N - 1) as i32)
            }
        }
    }

    #[inline(always)]
    fn inverse_smoothstep<P: Policy, const N: usize>(y: Self, edges: Option<(Self, Self)>) -> Self {
        let mut ba = Self::ONE;
        let mut bar = Self::ONE;
        let mut bar_a = Self::ONE; // (b - a) * a

        // Start with an initial guess of 0.5, since that'll have the largest derivative
        let mut x0 = Self::HALF;

        if let Some((a, b)) = edges {
            ba = b - a;

            if const { P::POLICY.precision.le(PrecisionPolicy::Worst) } {
                bar = ba.rcp();
                bar_a = bar * a;
            } else {
                bar = Self::ONE / ba;
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
                    t *= Self::splat(E::ONE / E::from_f64(3.0));
                } else {
                    // exact division for higher precisions
                    t /= Self::splat(E::from_f64(3.0));
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
        let (Ok(v) | Err(v)) = algorithms::newtons_method::<Self, P, _>(x0, Self::tolerance::<P>(), bounds, |x: Self| {
            let mut t = x;
            let mut dt_dx = bar;

            if edges.is_some() {
                // adjust by precalculated scales
                t = t.mul_sube(bar, bar_a);
            }

            let xn1 = t.powi_p::<P>((N - 1) as i32);

            let (fx, fpx) = const { Smoothstep::<E, N>::COEFFICIENTS }.into_iter().enumerate().fold(
                (Self::ZERO, Self::ZERO),
                |(fx, fpx), (k, c)| {(
                    fx.mul_adde(t, Self::splat(E::from_i64(c))),
                    fpx.mul_adde(t, Self::splat(E::from_i64(c) * E::from_i64((2 * N - k - 1) as i64))),
                )},
            );

            (t.mul_sube(xn1 * fx, y), fpx * dt_dx * xn1)
        });

        v
    }

    #[inline(always)]
    fn smooth_interpolator<P: Policy>(x: Self, edges: Option<(Self, Self)>, k: Self) -> Self {
        let mut t = x;

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
    fn smooth_interpolator_inverse<P: Policy>(mut y: Self, edges: Option<(Self, Self)>, k: Self) -> Self {
        // k ln(1/y - 1)
        let l = k * (y.reciprocal_p::<P>() - Self::ONE).ln_p::<P>();

        // ((l + 2) - sqrt(l^2 + 4)) / 2l
        let a = (l + Self::TWO);
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

pub mod pd;
pub mod ps;

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

pub struct Smoothstep<F: FloatElement, const N: usize>(PhantomData<[F; N]>);

impl<F: FloatElement, const N: usize> Smoothstep<F, N> {
    // ensure these coefficients are generated at compile time
    pub const COEFFICIENTS: [i64; N] = const {
        let mut coeffs = [0; N];
        let n = (N - 1) as i32;

        let mut k = 0;
        while k < N {
            let c = binomial(-1 - n, k as i32) * binomial(n + n + 1, n - k as i32);

            if c.unsigned_abs() > F::MAX_U64 {
                panic!("Binomial coefficient overflow");
            }

            // store in reverse order for easier polynomial evaluation
            coeffs[N - k - 1] = c;
            k += 1;
        }

        coeffs
    };
}
