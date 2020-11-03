#![allow(unused)]

use crate::*;

/// Set of vectorized special functions optimized for both single and double precision
pub trait SimdVectorizedMath<S: Simd>: SimdFloatVector<S> {
    /// Computes the sine of a vector.
    fn sin(self) -> Self;
    /// Computes the cosine of a vector.
    fn cos(self) -> Self;
    /// Computes the tangent of a vector.
    fn tan(self) -> Self;

    /// Computes both the sine and cosine of a vector.
    fn sin_cos(self) -> (Self, Self);

    /// Computes the hyperbolic-sine of a vector.
    fn sinh(self) -> Self;
    /// Computes the hyperbolic-cosine of a vector.
    fn cosh(self) -> Self;
    /// Computes the hyperbolic-tangent of a vector.
    fn tanh(self) -> Self;

    /// Computes the hyperbolic-arcsine of a vector.
    fn asinh(self) -> Self;
    /// Computes the hyperbolic-arccosine of a vector.
    fn acosh(self) -> Self;
    /// Computes the hyperbolic-arctangent of a vector.
    fn atanh(self) -> Self;

    /// Computes the arcsine of a vector.
    fn asin(self) -> Self;
    /// Computes the arccosine of a vector.
    fn acos(self) -> Self;
    /// Computes the arctangent of a vector.
    fn atan(self) -> Self;
    /// Computes the four quadrant arc-tangent of `y`(`self`) and `x`
    fn atan2(self, x: Self) -> Self;

    /// The exponential function, returns `e^(self)`
    fn exp(self) -> Self;
    /// Half-exponential function, returns `0.5 * e^(self)`
    fn exph(self) -> Self;
    /// Binary exponential function, returns `2^(self)`
    fn exp2(self) -> Self;
    /// Exponential function minus one, `e^(self) - 1.0`,
    /// special implementation that is more accurate if the numbr if closer to zero.
    fn exp_m1(self) -> Self;

    /// Computes `x^e` where `x` is `self` and `e` is a vector of floating-point exponents
    fn powf(self, e: Self) -> Self;
    /// Computes `x^e` where `x` is `self` and `e` is a vector of integer exponents
    fn powi(self, e: S::Vi32) -> Self;

    /// Computes the natural logarithm of a vector.
    fn ln(self) -> Self;
    /// Computes `ln(1+x)` where `x` is `self`, more accurately
    /// than if operations were performed separately
    fn ln_1p(self) -> Self;
    /// Computes the base-2 logarithm of a vector
    fn log2(self) -> Self;
    /// Computes the base-10 logarithm of a vector
    fn log10(self) -> Self;

    /// Computes the error function for each value in a vector
    fn erf(self) -> Self;
    /// Computes the inverse error function for each value in a vector
    fn ierf(self) -> Self;
    /// Computes the complementary error function `1 - erf(x)` more accurately
    /// than if operations were performed separately
    fn erfc(self) -> Self;
}

#[rustfmt::skip]
impl<S: Simd, T> SimdVectorizedMath<S> for T
where
    T: SimdFloatVector<S>,
    <T as SimdVectorBase<S>>::Element: SimdVectorizedMathInternal<S, Vf = T>,
{
    #[inline(always)] fn sin(self)              -> Self         { <<Self as SimdVectorBase<S>>::Element as SimdVectorizedMathInternal<S>>::sin(self) }
    #[inline(always)] fn cos(self)              -> Self         { <<Self as SimdVectorBase<S>>::Element as SimdVectorizedMathInternal<S>>::cos(self) }
    #[inline(always)] fn tan(self)              -> Self         { <<Self as SimdVectorBase<S>>::Element as SimdVectorizedMathInternal<S>>::tan(self) }
    #[inline(always)] fn sin_cos(self)          -> (Self, Self) { <<Self as SimdVectorBase<S>>::Element as SimdVectorizedMathInternal<S>>::sin_cos(self) }
    #[inline(always)] fn sinh(self)             -> Self         { <<Self as SimdVectorBase<S>>::Element as SimdVectorizedMathInternal<S>>::sinh(self) }
    #[inline(always)] fn cosh(self)             -> Self         { <<Self as SimdVectorBase<S>>::Element as SimdVectorizedMathInternal<S>>::cosh(self) }
    #[inline(always)] fn tanh(self)             -> Self         { <<Self as SimdVectorBase<S>>::Element as SimdVectorizedMathInternal<S>>::tanh(self) }
    #[inline(always)] fn asinh(self)            -> Self         { <<Self as SimdVectorBase<S>>::Element as SimdVectorizedMathInternal<S>>::asinh(self) }
    #[inline(always)] fn acosh(self)            -> Self         { <<Self as SimdVectorBase<S>>::Element as SimdVectorizedMathInternal<S>>::acosh(self) }
    #[inline(always)] fn atanh(self)            -> Self         { <<Self as SimdVectorBase<S>>::Element as SimdVectorizedMathInternal<S>>::atanh(self) }
    #[inline(always)] fn asin(self)             -> Self         { <<Self as SimdVectorBase<S>>::Element as SimdVectorizedMathInternal<S>>::asin(self) }
    #[inline(always)] fn acos(self)             -> Self         { <<Self as SimdVectorBase<S>>::Element as SimdVectorizedMathInternal<S>>::acos(self) }
    #[inline(always)] fn atan(self)             -> Self         { <<Self as SimdVectorBase<S>>::Element as SimdVectorizedMathInternal<S>>::atan(self) }
    #[inline(always)] fn atan2(self, x: Self)   -> Self         { <<Self as SimdVectorBase<S>>::Element as SimdVectorizedMathInternal<S>>::atan2(self, x) }
    #[inline(always)] fn exp(self)              -> Self         { <<Self as SimdVectorBase<S>>::Element as SimdVectorizedMathInternal<S>>::exp(self) }
    #[inline(always)] fn exph(self)             -> Self         { <<Self as SimdVectorBase<S>>::Element as SimdVectorizedMathInternal<S>>::exph(self) }
    #[inline(always)] fn exp2(self)             -> Self         { <<Self as SimdVectorBase<S>>::Element as SimdVectorizedMathInternal<S>>::exp2(self) }
    #[inline(always)] fn exp_m1(self)           -> Self         { <<Self as SimdVectorBase<S>>::Element as SimdVectorizedMathInternal<S>>::exp_m1(self) }
    #[inline(always)] fn powf(self, e: Self)    -> Self         { <<Self as SimdVectorBase<S>>::Element as SimdVectorizedMathInternal<S>>::powf(self, e) }
    #[inline(always)] fn powi(self, e: S::Vi32) -> Self         { <<Self as SimdVectorBase<S>>::Element as SimdVectorizedMathInternal<S>>::powi(self, e) }
    #[inline(always)] fn ln(self)               -> Self         { <<Self as SimdVectorBase<S>>::Element as SimdVectorizedMathInternal<S>>::ln(self) }
    #[inline(always)] fn ln_1p(self)            -> Self         { <<Self as SimdVectorBase<S>>::Element as SimdVectorizedMathInternal<S>>::ln_1p(self) }
    #[inline(always)] fn log2(self)             -> Self         { <<Self as SimdVectorBase<S>>::Element as SimdVectorizedMathInternal<S>>::log2(self) }
    #[inline(always)] fn log10(self)            -> Self         { <<Self as SimdVectorBase<S>>::Element as SimdVectorizedMathInternal<S>>::log10(self) }
    #[inline(always)] fn erf(self)              -> Self         { <<Self as SimdVectorBase<S>>::Element as SimdVectorizedMathInternal<S>>::erf(self) }
    #[inline(always)] fn ierf(self)             -> Self         { <<Self as SimdVectorBase<S>>::Element as SimdVectorizedMathInternal<S>>::ierf(self) }
    #[inline(always)] fn erfc(self)             -> Self         { <<Self as SimdVectorBase<S>>::Element as SimdVectorizedMathInternal<S>>::erfc(self) }
}

#[doc(hidden)]
pub trait SimdVectorizedMathInternal<S: Simd>: SimdElement + From<f32> {
    type Vf: SimdFloatVector<S, Element = Self>;

    #[inline]
    fn sin(x: Self::Vf) -> Self::Vf {
        Self::sin_cos(x).0
    }

    #[inline]
    fn cos(x: Self::Vf) -> Self::Vf {
        Self::sin_cos(x).1
    }

    #[inline]
    fn tan(x: Self::Vf) -> Self::Vf {
        let (s, c) = Self::sin_cos(x);
        s / c
    }

    fn sin_cos(x: Self::Vf) -> (Self::Vf, Self::Vf);

    fn sinh(x: Self::Vf) -> Self::Vf;

    #[inline(always)]
    fn cosh(x: Self::Vf) -> Self::Vf {
        let x: Self::Vf = x.abs();
        let y: Self::Vf = x.exph(); // 0.5 * exp(x)
        y + Self::Vf::splat_any(0.25) / y // + 0.5 * exp(-x)
    }

    fn tanh(x: Self::Vf) -> Self::Vf;

    fn asin(x: Self::Vf) -> Self::Vf;
    fn acos(x: Self::Vf) -> Self::Vf;
    fn atan(x: Self::Vf) -> Self::Vf;
    fn atan2(y: Self::Vf, x: Self::Vf) -> Self::Vf;

    fn asinh(x: Self::Vf) -> Self::Vf;
    fn acosh(x: Self::Vf) -> Self::Vf;
    fn atanh(x: Self::Vf) -> Self::Vf;

    fn exp(x: Self::Vf) -> Self::Vf;
    fn exph(x: Self::Vf) -> Self::Vf;
    fn exp2(x: Self::Vf) -> Self::Vf;
    fn exp_m1(x: Self::Vf) -> Self::Vf {
        Self::exp(x) - Self::Vf::one()
    }

    fn powf(x: Self::Vf, e: Self::Vf) -> Self::Vf;

    #[inline]
    fn powi(mut x: Self::Vf, mut e: S::Vi32) -> Self::Vf {
        let mut res = Self::Vf::one();

        x = e.is_negative().select(Self::Vf::one() / x, x);
        e = e.abs();

        let zero = Vi32::<S>::zero();

        loop {
            res = (e & Vi32::<S>::one()).ne(zero).select(res * x, res);

            e >>= 1;

            let fin = e.eq(zero);

            if fin.all() {
                break;
            }

            x = fin.select(x, x * x); // x *= fin.select(1.0, x)
        }

        res
    }

    fn ln(x: Self::Vf) -> Self::Vf;
    fn ln_1p(x: Self::Vf) -> Self::Vf;
    fn log2(x: Self::Vf) -> Self::Vf;
    fn log10(x: Self::Vf) -> Self::Vf;

    fn erf(x: Self::Vf) -> Self::Vf;
    fn ierf(x: Self::Vf) -> Self::Vf;

    #[inline(always)]
    fn erfc(x: Self::Vf) -> Self::Vf {
        Self::Vf::one() - Self::erf(x)
    }
}

impl<S: Simd> SimdVectorizedMathInternal<S> for f32
where
    <S as Simd>::Vf32: SimdFloatVector<S, Element = f32>,
{
    type Vf = <S as Simd>::Vf32;

    #[inline(always)]
    fn sin_cos(xx: Self::Vf) -> (Self::Vf, Self::Vf) {
        let dp1f = Vf32::<S>::splat(0.78515625 * 2.0);
        let dp2f = Vf32::<S>::splat(2.4187564849853515625E-4 * 2.0);
        let dp3f = Vf32::<S>::splat(3.77489497744594108E-8 * 2.0);
        let p0sinf = Vf32::<S>::splat(-1.6666654611E-1);
        let p1sinf = Vf32::<S>::splat(8.3321608736E-3);
        let p2sinf = Vf32::<S>::splat(-1.9515295891E-4);
        let p0cosf = Vf32::<S>::splat(4.166664568298827E-2);
        let p1cosf = Vf32::<S>::splat(-1.388731625493765E-3);
        let p2cosf = Vf32::<S>::splat(2.443315711809948E-5);

        let xa: Vf32<S> = xx.abs();

        let y: Vf32<S> = (xa * Vf32::<S>::splat(2.0 / std::f32::consts::PI)).round();
        let q: Vu32<S> = y.cast_to::<Vi32<S>>().into_bits(); // cast to signed (faster), then transmute to unsigned

        // Reduce by extended precision modular arithmetic
        // x = ((xa - y * DP1F) - y * DP2F) - y * DP3F;
        let x = y.nmul_add(dp3f, y.nmul_add(dp2f, y.nmul_add(dp1f, xa)));

        // Taylor expansion of sin and cos, valid for -pi/4 <= x <= pi/4
        let x2: Vf32<S> = x * x;
        let x3: Vf32<S> = x2 * x;
        let x4: Vf32<S> = x2 * x2;
        let mut s = x4.mul_add(p2sinf, x2.mul_add(p1sinf, p0sinf)).mul_add(x3, x);
        let mut c = x4
            .mul_add(p2cosf, x2.mul_add(p1cosf, p0cosf))
            .mul_add(x4, Vf32::<S>::splat(0.5).nmul_add(x2, Vf32::<S>::one()));

        // swap sin and cos if odd quadrant
        let swap = (q & Vu32::<S>::one()).ne(Vu32::<S>::zero());

        let mut overflow = q.gt(Vu32::<S>::splat(0x2000000)); // q big if overflow
        overflow &= xa.is_finite().cast_to();

        s = overflow.select(Vf32::<S>::zero(), s);
        c = overflow.select(Vf32::<S>::one(), c);

        let sin1 = swap.select(c, s);
        let cos1 = swap.select(s, c);

        let signsin = Vf32::<S>::from_bits((q << 30) ^ xx.into_bits());
        let signcos = Vf32::<S>::from_bits(((q + Vu32::<S>::one()) & Vu32::<S>::splat(2)) << 30);

        // combine signs
        (sin1 ^ (signsin & Vf32::<S>::neg_zero()), cos1 ^ signcos)
    }

    fn sinh(x: Self::Vf) -> Self::Vf {
        unimplemented!()
    }

    fn tanh(x: Self::Vf) -> Self::Vf {
        unimplemented!()
    }
    fn asin(x: Self::Vf) -> Self::Vf {
        unimplemented!()
    }
    fn acos(x: Self::Vf) -> Self::Vf {
        unimplemented!()
    }
    fn atan(x: Self::Vf) -> Self::Vf {
        unimplemented!()
    }
    fn atan2(y: Self::Vf, x: Self::Vf) -> Self::Vf {
        unimplemented!()
    }
    fn asinh(x: Self::Vf) -> Self::Vf {
        unimplemented!()
    }
    fn acosh(x: Self::Vf) -> Self::Vf {
        unimplemented!()
    }
    fn atanh(x: Self::Vf) -> Self::Vf {
        unimplemented!()
    }
    fn exp(x: Self::Vf) -> Self::Vf {
        unimplemented!()
    }
    fn exph(x: Self::Vf) -> Self::Vf {
        unimplemented!()
    }
    fn exp2(x: Self::Vf) -> Self::Vf {
        unimplemented!()
    }
    fn powf(x: Self::Vf, e: Self::Vf) -> Self::Vf {
        unimplemented!()
    }
    fn ln(x: Self::Vf) -> Self::Vf {
        unimplemented!()
    }
    fn ln_1p(x: Self::Vf) -> Self::Vf {
        unimplemented!()
    }
    fn log2(x: Self::Vf) -> Self::Vf {
        unimplemented!()
    }
    fn log10(x: Self::Vf) -> Self::Vf {
        unimplemented!()
    }
    fn erf(x: Self::Vf) -> Self::Vf {
        unimplemented!()
    }
    fn ierf(x: Self::Vf) -> Self::Vf {
        unimplemented!()
    }
}

impl<S: Simd> SimdVectorizedMathInternal<S> for f64
where
    <S as Simd>::Vf64: SimdFloatVector<S, Element = f64>,
{
    type Vf = <S as Simd>::Vf64;

    fn sin_cos(xx: Self::Vf) -> (Self::Vf, Self::Vf) {
        unimplemented!()
    }

    fn sinh(x: Self::Vf) -> Self::Vf {
        unimplemented!()
    }

    fn tanh(x: Self::Vf) -> Self::Vf {
        unimplemented!()
    }
    fn asin(x: Self::Vf) -> Self::Vf {
        unimplemented!()
    }
    fn acos(x: Self::Vf) -> Self::Vf {
        unimplemented!()
    }
    fn atan(x: Self::Vf) -> Self::Vf {
        unimplemented!()
    }
    fn atan2(y: Self::Vf, x: Self::Vf) -> Self::Vf {
        unimplemented!()
    }
    fn asinh(x: Self::Vf) -> Self::Vf {
        unimplemented!()
    }
    fn acosh(x: Self::Vf) -> Self::Vf {
        unimplemented!()
    }
    fn atanh(x: Self::Vf) -> Self::Vf {
        unimplemented!()
    }
    fn exp(x: Self::Vf) -> Self::Vf {
        unimplemented!()
    }
    fn exph(x: Self::Vf) -> Self::Vf {
        unimplemented!()
    }
    fn exp2(x: Self::Vf) -> Self::Vf {
        unimplemented!()
    }
    fn powf(x: Self::Vf, e: Self::Vf) -> Self::Vf {
        unimplemented!()
    }
    fn ln(x: Self::Vf) -> Self::Vf {
        unimplemented!()
    }
    fn ln_1p(x: Self::Vf) -> Self::Vf {
        unimplemented!()
    }
    fn log2(x: Self::Vf) -> Self::Vf {
        unimplemented!()
    }
    fn log10(x: Self::Vf) -> Self::Vf {
        unimplemented!()
    }
    fn erf(x: Self::Vf) -> Self::Vf {
        unimplemented!()
    }
    fn ierf(x: Self::Vf) -> Self::Vf {
        unimplemented!()
    }
}
