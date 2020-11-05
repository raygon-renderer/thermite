#![allow(unused)]

use crate::*;

mod common;
mod pd;
mod ps;

/// Set of vectorized special functions optimized for both single and double precision
pub trait SimdVectorizedMath<S: Simd>: SimdFloatVector<S> {
    /// Scales values between `in_min` and `in_max`, to between `out_min` and `out_max`
    fn scale(self, in_min: Self, in_max: Self, out_min: Self, out_max: Self) -> Self;

    /// Linearly interpolates between `a` and `b` using `self`
    ///
    /// Equivalent to `(1 - t) * a + t * b`, but uses fused multiply-add operations
    /// to improve performance while maintaining precision
    fn lerp(self, a: Self, b: Self) -> Self;

    /// Computes the sine of a vector.
    fn sin(self) -> Self;
    /// Computes the cosine of a vector.
    fn cos(self) -> Self;
    /// Computes the tangent of a vector.
    fn tan(self) -> Self;

    /// Computes both the sine and cosine of a vector together faster than they would be computed separately.
    ///
    /// If you need both, use this.
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
    /// Base-10 exponential function, returns `10^(self)`
    fn exp10(self) -> Self;
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
    #[inline(always)]
    fn scale(self, in_min: Self, in_max: Self, out_min: Self, out_max: Self) -> Self {
        ((self - in_min) / (in_max - in_min)).mul_add(out_max - out_min, out_min)
    }

    #[inline(always)]
    fn lerp(self, a: Self, b: Self) -> Self {
        self.mul_add(b - a, a)
    }

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
    #[inline(always)] fn exp10(self)            -> Self         { <<Self as SimdVectorBase<S>>::Element as SimdVectorizedMathInternal<S>>::exp10(self) }
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
    fn exp10(x: Self::Vf) -> Self::Vf;
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
        let one = Vi32::<S>::one();

        loop {
            res = (e & one).ne(zero).select(res * x, res);

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

#[derive(Clone, Copy, PartialEq, Eq)]
enum ExpMode {
    Exp,
    Expm1,
    Exph,
    Pow2,
    Pow10,
}
