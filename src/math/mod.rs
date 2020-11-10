#![allow(unused)]

use crate::*;

mod common;
mod pd;
mod ps;

//TODO: tgamma function, beta function, cbrt, j0, y0

/// Set of vectorized special functions optimized for both single and double precision
pub trait SimdVectorizedMath<S: Simd>: SimdFloatVector<S> {
    /// Scales values between `in_min` and `in_max`, to between `out_min` and `out_max`
    fn scale(self, in_min: Self, in_max: Self, out_min: Self, out_max: Self) -> Self;

    /// Linearly interpolates between `a` and `b` using `self`
    ///
    /// Equivalent to `(1 - t) * a + t * b`, but uses fused multiply-add operations
    /// to improve performance while maintaining precision
    fn lerp(self, a: Self, b: Self) -> Self;

    /// Returns the floating-point remainder of `self / y` (rounded towards zero)
    fn fmod(self, y: Self) -> Self;

    /// Computes `sqrt(x * x + y * y)` for each element of the vector, but can be more precise with values around zero.
    ///
    /// NOTE: This is not higher-performance than the naive version, only slightly more precise.
    fn hypot(self, y: Self) -> Self;

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
    /// Computes `x^e` where `x` is `self` and `e` is a vector of integer exponents via repeated squaring
    fn powiv(self, e: S::Vi32) -> Self;
    /// Computes `x^e` where `x` is `self` and `e` is a signed integer
    fn powi(self, e: i32) -> Self;

    /// Computes the physicists' [Hermite polynomial](https://en.wikipedia.org/wiki/Hermite_polynomials)
    /// `H_n(x)` where `x` is `self` and `n` is an unsigned integer representing the polynomial degree.
    ///
    /// This uses the recurrence relation to compute the polynomial iteratively.
    fn hermite(self, n: u32) -> Self;

    /// Computes the physicists' [Hermite polynomial](https://en.wikipedia.org/wiki/Hermite_polynomials)
    /// `H_n(x)` where `x` is `self` and `n` is a vector of unsigned integers representing the polynomial degree.
    ///
    /// The polynomial is calculated independenty per-lane with the given degree in `n`.
    ///
    /// This uses the recurrence relation to compute the polynomial iteratively.
    fn hermitev(self, n: S::Vu32) -> Self;

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
    fn erfinv(self) -> Self;

    /// Finds the next representable float moving upwards to positive infinity
    fn next_float(self) -> Self;

    /// Finds the previous representable float moving downwards to negative infinity
    fn prev_float(self) -> Self;

    /// Calculates a [sigmoid-like 3rd-order interpolation function](https://en.wikipedia.org/wiki/Smoothstep#3rd-order_equation).
    ///
    /// **NOTE**: This function is only valid between 0 and 1, but does not clamp the input to maintain performance
    /// where that is not needed. Consider using `.saturate()` and `.scale` to ensure the input is within 0 to 1.
    fn smoothstep(self) -> Self;
    /// Calculates a [sigmoid-like 5th-order interpolation function](https://en.wikipedia.org/wiki/Smoothstep#5th-order_equation).
    ///
    /// **NOTE**: This function is only valid between 0 and 1, but does not clamp the input to maintain performance
    /// where that is not needed. Consider using `.saturate()` and `.scale` to ensure the input is within 0 to 1.
    fn smootherstep(self) -> Self;
    /// Calculates a [signmoid-like 7th-order interpolation function](https://en.wikipedia.org/wiki/Smoothstep#7th-order_equation).
    ///
    /// **NOTE**: This function is only valid between 0 and 1, but does not clamp the input to maintain performance
    /// where that is not needed. Consider using `.saturate()` and `.scale` to ensure the input is within 0 to 1.
    fn smootheststep(self) -> Self;
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

    #[inline(always)]
    fn fmod(self, y: Self) -> Self {
        self % y // Already implemented with operator overloads anyway
    }

    #[inline(always)]
    fn powi(self, mut e: i32) -> Self {
        let one = Self::one();

        let mut x = self;
        let mut res = one;

        if e < 0 {
            x = one / x;
            e = -e;
        }

        while e != 0 {
            if e & 1 != 0 {
                res *= x;
            }

            e >>= 1;
            x *= x;
        }

        res
    }

    #[inline(always)]
    fn powiv(self, mut e: S::Vi32) -> Self {
        let zero_i = Vi32::<S>::zero();
        let one_i = Vi32::<S>::one();
        let one = Self::one();

        let mut x = self;
        let mut res = one;

        x = e.is_negative().select(one / x, x);
        e = e.abs();

        loop {
            // TODO: Maybe try to optimize out the compare on platforms that support blendv,
            // since blendv only cares about the highest bit
            res = (e & one_i).ne(zero_i).select(res * x, res);

            e >>= 1;
            x *= x;

            if e.eq(zero_i).all() {
                return res;
            }
        }
    }

    #[inline(always)]
    fn hermite(self, mut n: u32) -> Self {
        let one = Self::one();
        let mut p0 = one;

        if unlikely!(n == 0) {
            return p0;
        }

        let x = self;

        let mut p1 = x + x; // 2 * x

        let mut c = 1;
        let mut cf = one;

        while c < n {
            // swap p0, p1
            let tmp = p0;
            p0 = p1;
            p1 = tmp;

            let next0 = x.mul_sub(p0, cf * p1);

            p1 = next0 + next0; // 2 * next0

            c += 1;
            cf += one;
        }

        p1
    }

    #[inline(always)]
    fn hermitev(self, mut n: S::Vu32) -> Self {
        let x = self;

        let one = Self::one();
        let i1 = Vu32::<S>::one();
        let n_is_zero = n.eq(Vu32::<S>::zero());

        let mut c = i1;

        // count `n = c.to_float()` separately to avoid expensive converting every iteration
        let mut cf = one;

        n -= i1; // decrement this to be able to use greater-than instead of greater-than-or-equal

        let mut p0 = one;
        let mut p1 = x + x; // 2 * x

        loop {
            let fin = c.gt(n) | n_is_zero;

            if fin.all() {
                break;
            }

            // swap p0, p1
            let tmp = p0;
            p0 = p1;
            p1 = tmp;

            let next0 = x.mul_sub(p0, cf * p1);
            let next = next0 + next0; // 2 * next0

            p1 = fin.select(p1, next);

            c += i1;
            cf += one;
        }

        p1
    }

    #[inline(always)]
    fn hypot(self, y: Self) -> Self {
        let x = self.abs();
        let y = y.abs();

        let min = x.min(y);
        let max = x.max(y);
        let t = min / max;

        let ret = max * t.mul_add(t, Self::one()).sqrt();

        min.eq(Self::zero()).select(max, ret)
    }

    #[inline] fn sin(self)              -> Self         { <<Self as SimdVectorBase<S>>::Element as SimdVectorizedMathInternal<S>>::sin(self) }
    #[inline] fn cos(self)              -> Self         { <<Self as SimdVectorBase<S>>::Element as SimdVectorizedMathInternal<S>>::cos(self) }
    #[inline] fn tan(self)              -> Self         { <<Self as SimdVectorBase<S>>::Element as SimdVectorizedMathInternal<S>>::tan(self) }
    #[inline] fn sin_cos(self)          -> (Self, Self) { <<Self as SimdVectorBase<S>>::Element as SimdVectorizedMathInternal<S>>::sin_cos(self) }
    #[inline] fn sinh(self)             -> Self         { <<Self as SimdVectorBase<S>>::Element as SimdVectorizedMathInternal<S>>::sinh(self) }
    #[inline] fn cosh(self)             -> Self         { <<Self as SimdVectorBase<S>>::Element as SimdVectorizedMathInternal<S>>::cosh(self) }
    #[inline] fn tanh(self)             -> Self         { <<Self as SimdVectorBase<S>>::Element as SimdVectorizedMathInternal<S>>::tanh(self) }
    #[inline] fn asinh(self)            -> Self         { <<Self as SimdVectorBase<S>>::Element as SimdVectorizedMathInternal<S>>::asinh(self) }
    #[inline] fn acosh(self)            -> Self         { <<Self as SimdVectorBase<S>>::Element as SimdVectorizedMathInternal<S>>::acosh(self) }
    #[inline] fn atanh(self)            -> Self         { <<Self as SimdVectorBase<S>>::Element as SimdVectorizedMathInternal<S>>::atanh(self) }
    #[inline] fn asin(self)             -> Self         { <<Self as SimdVectorBase<S>>::Element as SimdVectorizedMathInternal<S>>::asin(self) }
    #[inline] fn acos(self)             -> Self         { <<Self as SimdVectorBase<S>>::Element as SimdVectorizedMathInternal<S>>::acos(self) }
    #[inline] fn atan(self)             -> Self         { <<Self as SimdVectorBase<S>>::Element as SimdVectorizedMathInternal<S>>::atan(self) }
    #[inline] fn atan2(self, x: Self)   -> Self         { <<Self as SimdVectorBase<S>>::Element as SimdVectorizedMathInternal<S>>::atan2(self, x) }
    #[inline] fn exp(self)              -> Self         { <<Self as SimdVectorBase<S>>::Element as SimdVectorizedMathInternal<S>>::exp(self) }
    #[inline] fn exph(self)             -> Self         { <<Self as SimdVectorBase<S>>::Element as SimdVectorizedMathInternal<S>>::exph(self) }
    #[inline] fn exp2(self)             -> Self         { <<Self as SimdVectorBase<S>>::Element as SimdVectorizedMathInternal<S>>::exp2(self) }
    #[inline] fn exp10(self)            -> Self         { <<Self as SimdVectorBase<S>>::Element as SimdVectorizedMathInternal<S>>::exp10(self) }
    #[inline] fn exp_m1(self)           -> Self         { <<Self as SimdVectorBase<S>>::Element as SimdVectorizedMathInternal<S>>::exp_m1(self) }
    #[inline] fn powf(self, e: Self)    -> Self         { <<Self as SimdVectorBase<S>>::Element as SimdVectorizedMathInternal<S>>::powf(self, e) }
    #[inline] fn ln(self)               -> Self         { <<Self as SimdVectorBase<S>>::Element as SimdVectorizedMathInternal<S>>::ln(self) }
    #[inline] fn ln_1p(self)            -> Self         { <<Self as SimdVectorBase<S>>::Element as SimdVectorizedMathInternal<S>>::ln_1p(self) }
    #[inline] fn log2(self)             -> Self         { <<Self as SimdVectorBase<S>>::Element as SimdVectorizedMathInternal<S>>::log2(self) }
    #[inline] fn log10(self)            -> Self         { <<Self as SimdVectorBase<S>>::Element as SimdVectorizedMathInternal<S>>::log10(self) }
    #[inline] fn erf(self)              -> Self         { <<Self as SimdVectorBase<S>>::Element as SimdVectorizedMathInternal<S>>::erf(self) }
    #[inline] fn erfinv(self)           -> Self         { <<Self as SimdVectorBase<S>>::Element as SimdVectorizedMathInternal<S>>::erfinv(self) }
    #[inline] fn next_float(self)       -> Self         { <<Self as SimdVectorBase<S>>::Element as SimdVectorizedMathInternal<S>>::next_float(self) }
    #[inline] fn prev_float(self)       -> Self         { <<Self as SimdVectorBase<S>>::Element as SimdVectorizedMathInternal<S>>::prev_float(self) }
    #[inline] fn smoothstep(self)       -> Self         { <<Self as SimdVectorBase<S>>::Element as SimdVectorizedMathInternal<S>>::smoothstep(self) }
    #[inline] fn smootherstep(self)     -> Self         { <<Self as SimdVectorBase<S>>::Element as SimdVectorizedMathInternal<S>>::smootherstep(self) }
    #[inline] fn smootheststep(self)    -> Self         { <<Self as SimdVectorBase<S>>::Element as SimdVectorizedMathInternal<S>>::smootheststep(self) }
}

#[doc(hidden)]
pub trait SimdVectorizedMathInternal<S: Simd>: SimdElement + From<f32> + From<i16> {
    type Vf: SimdFloatVector<S, Element = Self>;

    #[inline(always)]
    fn sin(x: Self::Vf) -> Self::Vf {
        Self::sin_cos(x).0
    }

    #[inline(always)]
    fn cos(x: Self::Vf) -> Self::Vf {
        Self::sin_cos(x).1
    }

    #[inline(always)]
    fn tan(x: Self::Vf) -> Self::Vf {
        let (s, c) = Self::sin_cos(x);
        s / c
    }

    fn sin_cos(x: Self::Vf) -> (Self::Vf, Self::Vf);

    fn sinh(x: Self::Vf) -> Self::Vf;

    #[inline(always)]
    fn cosh(x: Self::Vf) -> Self::Vf {
        let y: Self::Vf = x.abs().exph(); // 0.5 * exp(x)
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

    #[inline(always)]
    fn exp_m1(x: Self::Vf) -> Self::Vf {
        Self::exp(x) - Self::Vf::one()
    }

    fn powf(x: Self::Vf, e: Self::Vf) -> Self::Vf;

    fn ln(x: Self::Vf) -> Self::Vf;
    fn ln_1p(x: Self::Vf) -> Self::Vf;
    fn log2(x: Self::Vf) -> Self::Vf;
    fn log10(x: Self::Vf) -> Self::Vf;

    fn erf(x: Self::Vf) -> Self::Vf;
    fn erfinv(x: Self::Vf) -> Self::Vf;

    fn next_float(x: Self::Vf) -> Self::Vf;
    fn prev_float(x: Self::Vf) -> Self::Vf;

    #[inline(always)]
    fn smoothstep(x: Self::Vf) -> Self::Vf {
        // use integer coefficients to ensure as-accurate-as-possible casts to f32 or f64
        x * x * x.nmul_add(Self::Vf::splat_any(2i16), Self::Vf::splat_any(3i16))
    }

    #[inline(always)]
    fn smootherstep(x: Self::Vf) -> Self::Vf {
        let c3 = Self::Vf::splat_any(10i16);
        let c4 = Self::Vf::splat_any(-15i16);
        let c5 = Self::Vf::splat_any(6i16);

        // Use Estrin's scheme here without c0-c2
        let x2 = x * x;
        let x4 = x2 * x2;

        x4.mul_add(x.mul_add(c5, c4), x2 * x * c3)
    }

    #[inline(always)]
    fn smootheststep(x: Self::Vf) -> Self::Vf {
        let c4 = Self::Vf::splat_any(35i16);
        let c5 = Self::Vf::splat_any(-84i16);
        let c6 = Self::Vf::splat_any(70i16);
        let c7 = Self::Vf::splat_any(-20i16);

        let x2 = x * x;

        x2 * x2 * x2.mul_add(x.mul_add(c7, c6), x.mul_add(c5, c4))
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
