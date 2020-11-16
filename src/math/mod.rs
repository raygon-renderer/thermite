#![allow(unused)]

use crate::*;

mod common;
mod pd;
mod ps;

//TODO: beta function, j0, y0

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

    /// Computes the sum `Σ(coefficients[i] * x^i)` from `i=0` to `coefficients.len()`
    ///
    /// **NOTE**: This has the potential to inline and unroll the inner loop for constant input
    fn poly(self, coefficients: &[Self::Element]) -> Self;

    /// Computes the sum `Σ(f(i)*x^i)` from `i=0` to `n`
    ///
    /// **NOTE**: This has the potential to inline and unroll the inner loop for constant input. For
    /// best results with dynamic input, try to precompute the input function, as it will allow for better
    /// cache utilization and lessen register pressure.
    fn poly_f<F>(self, n: usize, f: F) -> Self
    where
        F: FnMut(usize) -> Self;

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

    /// Computes the cubic-root of each lane in a vector.
    fn cbrt(self) -> Self;

    /// Computes `x^e` where `x` is `self` and `e` is a vector of floating-point exponents
    fn powf(self, e: Self) -> Self;
    /// Computes `x^e` where `x` is `self` and `e` is a vector of integer exponents via repeated squaring
    fn powiv(self, e: S::Vi32) -> Self;
    /// Computes `x^e` where `x` is `self` and `e` is a signed integer
    ///
    /// **NOTE**: Given a constant `e`, LLVM will happily unroll the inner loop
    fn powi(self, e: i32) -> Self;

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

    fn tgamma(self) -> Self;

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

    /// Computes the n-th degree physicists' [Hermite polynomial](https://en.wikipedia.org/wiki/Hermite_polynomials)
    /// `H_n(x)` where `x` is `self` and `n` is an unsigned integer representing the polynomial degree.
    ///
    /// This uses the recurrence relation to compute the polynomial iteratively.
    ///
    /// **NOTE**: Given a constant `n`, LLVM will happily unroll and optimize the inner loop where possible.
    fn hermite(self, n: u32) -> Self;

    /// Computes the n-th degree physicists' [Hermite polynomial](https://en.wikipedia.org/wiki/Hermite_polynomials)
    /// `H_n(x)` where `x` is `self` and `n` is a vector of unsigned integers representing the polynomial degree.
    ///
    /// The polynomial is calculated independenty per-lane with the given degree in `n`.
    ///
    /// This uses the recurrence relation to compute the polynomial iteratively.
    fn hermitev(self, n: S::Vu32) -> Self;

    /// Computes the n-th degree Jacobi polynomial via the 3-term recurrence relation.
    ///
    /// At the special case where α and β are both zero, the Jacobi polynomial reduces to a
    /// Legendre polynomial, and if α and β are both constant zero LLVM can optimize out expressions that rely on those.
    ///
    /// **NOTE**: Given a constant `n`, LLVM will happily unroll and optimize the inner loop where possible.
    fn jacobi(self, alpha: Self, beta: Self, n: u32) -> Self;

    /// Computes the m-th derivative of an n-th degree Jacobi polynomial
    ///
    /// A the special case where α and β are both zero, the Jacobi polynomial reduces to a
    /// Legendre polynomial.
    ///
    /// **NOTE**: Given constant α, β or `n`, LLVM will happily optimize those away and unroll loops.
    fn jacobi_d(self, alpha: Self, beta: Self, n: u32, m: u32) -> Self;

    /// Computes the m-th associated n-th degree Legendre polynomial,
    /// where m=0 signifies the regular n-th degree Legendre polynomial.
    ///
    /// If `m` is odd, the input is only valid between -1 and 1
    ///
    /// **NOTE**: Given constant `n` and/or `m`, LLVM will happily unroll and optimize inner loops.
    ///
    /// Internally, this is computed with [`jacobi_d`](#tymethod.jacobi_d)
    fn legendre_p(self, n: u32, m: u32) -> Self;
}

macro_rules! dispatch {
    (@INNER $instrset:ident #[$meta:meta] $name:ident<$($gen:ident),*>($($arg:ident:$ty:ty),*)
        -> $ret:ty $(where $([$wgen:tt: $bound:path]),*)*) => {{
        #[$meta]
        #[inline]
        unsafe fn __dispatch<S: Simd, V: SimdFloatVector<S>, $($gen),*>($($arg: $ty),*) -> $ret
        where <V as SimdVectorBase<S>>::Element: SimdVectorizedMathInternal<S, Vf = V>,
            $($($wgen: $bound),*)*
        { <<V as SimdVectorBase<S>>::Element as SimdVectorizedMathInternal<S>>::$name($($arg),*) }
        return unsafe { __dispatch::<S, Self, $($gen),*>($($arg),*) };
    }};
    ($name:ident$(<$($gen:ident),*>)*($($arg:ident:$ty:ty),*) -> $ret:ty $(where $([$wgen:tt: $bound:path]),*)*) => {
        // branch will be evaluated at compile-time
        match S::INSTRSET {
            //SimdInstructionSet::SSE2        => dispatch!(@INNER SSE2      #[target_feature(enable = "sse2")]             $name<$($($gen),*)*>($($arg: $ty),*) -> $ret $(where $([$wgen: $bound]),*)*),
            //SimdInstructionSet::SSE41       => dispatch!(@INNER SSE41     #[target_feature(enable = "sse4.1")]           $name<$($($gen),*)*>($($arg: $ty),*) -> $ret $(where $([$wgen: $bound]),*)*),
            //SimdInstructionSet::AVX         => dispatch!(@INNER AVX       #[target_feature(enable = "avx,fma")]          $name<$($($gen),*)*>($($arg: $ty),*) -> $ret $(where $([$wgen: $bound]),*)*),
            SimdInstructionSet::AVX2          => dispatch!(@INNER AVX2      #[target_feature(enable = "avx2,fma")]         $name<$($($gen),*)*>($($arg: $ty),*) -> $ret $(where $([$wgen: $bound]),*)*),
            //SimdInstructionSet::AVX512F     => dispatch!(@INNER AVX512F   #[target_feature(enable = "avx512f")]          $name<$($($gen),*)*>($($arg: $ty),*) -> $ret $(where $([$wgen: $bound]),*)*),
            //SimdInstructionSet::AVX512FBW   => dispatch!(@INNER AVX512FBW #[target_feature(enable = "avxf12f,avx512bw")] $name<$($($gen),*)*>($($arg: $ty),*) -> $ret $(where $([$wgen: $bound]),*)*),
            _ => unreachable!() // TODO: Replace with unreachable_unchecked when all instruction sets are implemented
        }
    };
}

#[rustfmt::skip]
impl<S: Simd, T> SimdVectorizedMath<S> for T
where
    T: SimdFloatVector<S>,
    <T as SimdVectorBase<S>>::Element: SimdVectorizedMathInternal<S, Vf = T>,
{

    #[inline] fn scale(self, in_min: Self, in_max: Self, out_min: Self, out_max: Self) -> Self {
        let x = self; dispatch!(scale(x: V, in_min: V, in_max: V, out_min: V, out_max: V) -> V)
    }

    #[inline] fn lerp(self, a: Self, b: Self)   -> Self { let t = self; dispatch!(lerp(t: V, a: V, b: V) -> V) }
    #[inline] fn fmod(self, y: Self)            -> Self { let x = self; dispatch!(fmod(x: V, y: V) -> V) }
    #[inline] fn hypot(self, y: Self)           -> Self { let x = self; dispatch!(hypot(x: V, y: V) -> V) }
    #[inline] fn powi(self, e: i32)             -> Self { let x = self; dispatch!(powi(x: V, e: i32) -> V) }
    #[inline] fn powiv(self, e: S::Vi32)        -> Self { let x = self; dispatch!(powiv(x: V, e: S::Vi32) -> V) }

    #[inline] fn poly_f<F>(self, n: usize, f: F) -> Self
    where
        F: FnMut(usize) -> Self,
    {
        let x = self; dispatch!(poly_f<F>(x: V, n: usize, f: F) -> V where [F: FnMut(usize) -> V])
    }

    #[inline] fn poly(self, coefficients: &[Self::Element]) -> Self {
        let x = self; dispatch!(poly(x: V, coefficients: &[V::Element]) -> V)
    }

    #[inline] fn hermite(self, n: u32)                                      -> Self { let x = self; dispatch!(hermite(x: V, n: u32) -> V) }
    #[inline] fn hermitev(self, n: S::Vu32)                                 -> Self { let x = self; dispatch!(hermitev(x: V, n: S::Vu32) -> V) }
    #[inline] fn jacobi(self, alpha: Self, beta: Self, n: u32)              -> Self { let x = self; dispatch!(jacobi(x: V, alpha: V, beta: V, n: u32) -> V) }
    #[inline] fn jacobi_d(self, alpha: Self, beta: Self, n: u32, m: u32)    -> Self { let x = self; dispatch!(jacobi_d(x: V, alpha: V, beta: V, n: u32, m: u32) -> V) }
    #[inline] fn legendre_p(self, n: u32, m: u32)                           -> Self { let x = self; dispatch!(legendre_p(x: V, n: u32, m: u32) -> V) }

    #[inline] fn sin(self)              -> Self         { let x = self; dispatch!(sin(x: V) -> V) }
    #[inline] fn cos(self)              -> Self         { let x = self; dispatch!(cos(x: V) -> V) }
    #[inline] fn tan(self)              -> Self         { let x = self; dispatch!(tan(x: V) -> V) }
    #[inline] fn sin_cos(self)          -> (Self, Self) { let x = self; dispatch!(sin_cos(x: V) -> (V, V)) }
    #[inline] fn sinh(self)             -> Self         { let x = self; dispatch!(sinh(x: V) -> V) }
    #[inline] fn cosh(self)             -> Self         { let x = self; dispatch!(cosh(x: V) -> V) }
    #[inline] fn tanh(self)             -> Self         { let x = self; dispatch!(tanh(x: V) -> V) }
    #[inline] fn asinh(self)            -> Self         { let x = self; dispatch!(asinh(x: V) -> V) }
    #[inline] fn acosh(self)            -> Self         { let x = self; dispatch!(acosh(x: V) -> V) }
    #[inline] fn atanh(self)            -> Self         { let x = self; dispatch!(atanh(x: V) -> V) }
    #[inline] fn asin(self)             -> Self         { let x = self; dispatch!(asin(x: V) -> V) }
    #[inline] fn acos(self)             -> Self         { let x = self; dispatch!(acos(x: V) -> V) }
    #[inline] fn atan(self)             -> Self         { let x = self; dispatch!(atan(x: V) -> V) }
    #[inline] fn atan2(self, x: Self)   -> Self         { let y = self; dispatch!(atan2(y: V, x: V) -> V) }
    #[inline] fn exp(self)              -> Self         { let x = self; dispatch!(exp(x: V) -> V) }
    #[inline] fn exph(self)             -> Self         { let x = self; dispatch!(exph(x: V) -> V) }
    #[inline] fn exp2(self)             -> Self         { let x = self; dispatch!(exp2(x: V) -> V) }
    #[inline] fn exp10(self)            -> Self         { let x = self; dispatch!(exp10(x: V) -> V) }
    #[inline] fn exp_m1(self)           -> Self         { let x = self; dispatch!(exp_m1(x: V) -> V) }
    #[inline] fn cbrt(self)             -> Self         { let x = self; dispatch!(cbrt(x: V) -> V) }
    #[inline] fn powf(self, e: Self)    -> Self         { let x = self; dispatch!(powf(x: V, e: V) -> V) }
    #[inline] fn ln(self)               -> Self         { let x = self; dispatch!(ln(x: V) -> V) }
    #[inline] fn ln_1p(self)            -> Self         { let x = self; dispatch!(ln_1p(x: V) -> V) }
    #[inline] fn log2(self)             -> Self         { let x = self; dispatch!(log2(x: V) -> V) }
    #[inline] fn log10(self)            -> Self         { let x = self; dispatch!(log10(x: V) -> V) }
    #[inline] fn erf(self)              -> Self         { let x = self; dispatch!(erf(x: V) -> V) }
    #[inline] fn erfinv(self)           -> Self         { let x = self; dispatch!(erfinv(x: V) -> V) }
    #[inline] fn tgamma(self)           -> Self         { let x = self; dispatch!(tgamma(x: V) -> V) }
    #[inline] fn next_float(self)       -> Self         { let x = self; dispatch!(next_float(x: V) -> V) }
    #[inline] fn prev_float(self)       -> Self         { let x = self; dispatch!(prev_float(x: V) -> V) }
    #[inline] fn smoothstep(self)       -> Self         { let x = self; dispatch!(smoothstep(x: V) -> V) }
    #[inline] fn smootherstep(self)     -> Self         { let x = self; dispatch!(smootherstep(x: V) -> V) }
    #[inline] fn smootheststep(self)    -> Self         { let x = self; dispatch!(smootheststep(x: V) -> V) }
}

#[doc(hidden)]
pub trait SimdVectorizedMathInternal<S: Simd>:
    SimdElement
    + From<f32>
    + From<i16>
    + Add<Self, Output = Self>
    + Mul<Self, Output = Self>
    + Div<Self, Output = Self>
    + PartialOrd
{
    type Vf: SimdFloatVector<S, Element = Self>;

    const __EPSILON: Self;

    fn from_u32(x: u32) -> Self;
    fn from_i32(x: i32) -> Self;

    #[inline(always)]
    fn scale(x: Self::Vf, in_min: Self::Vf, in_max: Self::Vf, out_min: Self::Vf, out_max: Self::Vf) -> Self::Vf {
        ((x - in_min) / (in_max - in_min)).mul_add(out_max - out_min, out_min)
    }

    #[inline(always)]
    fn lerp(t: Self::Vf, a: Self::Vf, b: Self::Vf) -> Self::Vf {
        t.mul_add(b - a, a)
    }

    #[inline(always)]
    fn fmod(x: Self::Vf, y: Self::Vf) -> Self::Vf {
        x % y // Already implemented with operator overloads anyway
    }

    #[inline(always)]
    fn powi(mut x: Self::Vf, mut e: i32) -> Self::Vf {
        let one = Self::Vf::one();

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
    fn powiv(mut x: Self::Vf, mut e: S::Vi32) -> Self::Vf {
        let zero_i = Vi32::<S>::zero();
        let one_i = Vi32::<S>::one();
        let one = Self::Vf::one();

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
    fn hermite(x: Self::Vf, mut n: u32) -> Self::Vf {
        let one = Self::Vf::one();
        let mut p0 = one;

        if unlikely!(n == 0) {
            return p0;
        }

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
    fn hermitev(x: Self::Vf, mut n: S::Vu32) -> Self::Vf {
        let one = Self::Vf::one();
        let i1 = Vu32::<S>::one();
        let n_is_zero = n.eq(Vu32::<S>::zero());

        let mut c = i1;

        // count `n = c.to_float()` separately to avoid expensive converting every iteration
        let mut cf = one;

        let mut p0 = one;
        let mut p1 = x + x; // 2 * x

        loop {
            let cont = c.lt(n);

            if cont.none() {
                break;
            }

            // swap p0, p1
            let tmp = p0;
            p0 = p1;
            p1 = tmp;

            let next0 = x.mul_sub(p0, cf * p1);
            let next = next0 + next0; // 2 * next0

            p1 = cont.select(next, p1);

            c += i1;
            cf += one;
        }

        n_is_zero.select(one, p1)
    }

    #[inline(always)]
    fn hypot(x: Self::Vf, y: Self::Vf) -> Self::Vf {
        let x = x.abs();
        let y = y.abs();

        let min = x.min(y);
        let max = x.max(y);
        let t = min / max;

        let ret = max * t.mul_add(t, Self::Vf::one()).sqrt();

        min.eq(Self::Vf::zero()).select(max, ret)
    }

    #[inline(always)]
    #[rustfmt::skip]
    fn poly_f<F>(x: Self::Vf, n: usize, mut c: F) -> Self::Vf
    where
        F: FnMut(usize) -> Self::Vf
    {
        use common::*;

        // max degree of hard-coded polynomials + 1 for c0
        const MAX_DEGREE_P0: usize = 16;

        // fast path for small input
        if n <= MAX_DEGREE_P0 {
            return if n < 5 {
                match n {
                    0 => Self::Vf::zero(),
                    1 => c(0),
                    2 => poly_1(x, c(0), c(1)),
                    3 => poly_2(x, x * x, c(0), c(1), c(2)),
                    4 => poly_3(x, x * x, c(0), c(1), c(2), c(3)),
                    _ => unsafe { std::hint::unreachable_unchecked() }
                }
            } else {
                let x2 = x * x;
                let x4 = x2 * x2;
                let x8 = x4 * x4;

                match n {
                    5 =>  poly_4 (x, x2, x4,     c(0), c(1), c(2), c(3), c(4)),
                    6 =>  poly_5 (x, x2, x4,     c(0), c(1), c(2), c(3), c(4), c(5)),
                    7 =>  poly_6 (x, x2, x4,     c(0), c(1), c(2), c(3), c(4), c(5), c(6)),
                    8 =>  poly_7 (x, x2, x4,     c(0), c(1), c(2), c(3), c(4), c(5), c(6), c(7)),
                    9 =>  poly_8 (x, x2, x4, x8, c(0), c(1), c(2), c(3), c(4), c(5), c(6), c(7), c(8)),
                    10 => poly_9 (x, x2, x4, x8, c(0), c(1), c(2), c(3), c(4), c(5), c(6), c(7), c(8), c(9)),
                    11 => poly_10(x, x2, x4, x8, c(0), c(1), c(2), c(3), c(4), c(5), c(6), c(7), c(8), c(9), c(10)),
                    12 => poly_11(x, x2, x4, x8, c(0), c(1), c(2), c(3), c(4), c(5), c(6), c(7), c(8), c(9), c(10), c(11)),
                    13 => poly_12(x, x2, x4, x8, c(0), c(1), c(2), c(3), c(4), c(5), c(6), c(7), c(8), c(9), c(10), c(11), c(12)),
                    14 => poly_13(x, x2, x4, x8, c(0), c(1), c(2), c(3), c(4), c(5), c(6), c(7), c(8), c(9), c(10), c(11), c(12), c(13)),
                    15 => poly_14(x, x2, x4, x8, c(0), c(1), c(2), c(3), c(4), c(5), c(6), c(7), c(8), c(9), c(10), c(11), c(12), c(13), c(14)),
                    16 => poly_15(x, x2, x4, x8, c(0), c(1), c(2), c(3), c(4), c(5), c(6), c(7), c(8), c(9), c(10), c(11), c(12), c(13), c(14), c(15)),
                    _ => unsafe { std::hint::unreachable_unchecked() }
                }
            };
        }

        macro_rules! poly {
            ($name:ident($($pows:ident),*; $j:ident + $c:ident[$($coeff:expr),*])) => {{
                $name($($pows,)* $($c($j + $coeff)),*)
            }};
        }

        let xmd = x.powi(MAX_DEGREE_P0 as i32); // hopefully inlined

        let mut sum = Self::Vf::zero();
        let mut mul = Self::Vf::one();

        let x2 = x * x;
        let x4 = x2 * x2;
        let x8 = x4 * x4;

        // Use a hybrid Estrin/Horner algorithm
        let mut j = n;
        while j >= MAX_DEGREE_P0 {
            j -= MAX_DEGREE_P0;
            sum = sum.mul_add(xmd, poly!(poly_15(x, x2, x4, x8; j + c[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15])));
        }

        // handle remaining powers
        let (rmx, res) = match j {
            0  => return sum,
            1  => (x,                                  c(0)),
            2  => (x2,          poly_1 (x,             c(0), c(1))),
            3  => (x2*x,        poly_2 (x, x2,         c(0), c(1), c(2))),
            4  => (x4,          poly_3 (x, x2,         c(0), c(1), c(2), c(3))),
            5  => (x4*x,        poly_4 (x, x2, x4,     c(0), c(1), c(2), c(3), c(4))),
            6  => (x4*x2,       poly_5 (x, x2, x4,     c(0), c(1), c(2), c(3), c(4), c(5))),
            7  => (x4*x2*x,     poly_6 (x, x2, x4,     c(0), c(1), c(2), c(3), c(4), c(5), c(6))),
            8  => (x8,          poly_7 (x, x2, x4,     c(0), c(1), c(2), c(3), c(4), c(5), c(6), c(7))),
            9  => (x8*x,        poly_8 (x, x2, x4, x8, c(0), c(1), c(2), c(3), c(4), c(5), c(6), c(7), c(8))),
            10 => (x8*x2,       poly_9 (x, x2, x4, x8, c(0), c(1), c(2), c(3), c(4), c(5), c(6), c(7), c(8), c(9))),
            11 => (x8*x2*x,     poly_10(x, x2, x4, x8, c(0), c(1), c(2), c(3), c(4), c(5), c(6), c(7), c(8), c(9), c(10))),
            12 => (x8*x4,       poly_11(x, x2, x4, x8, c(0), c(1), c(2), c(3), c(4), c(5), c(6), c(7), c(8), c(9), c(10), c(11))),
            13 => (x8*x4*x,     poly_12(x, x2, x4, x8, c(0), c(1), c(2), c(3), c(4), c(5), c(6), c(7), c(8), c(9), c(10), c(11), c(12))),
            14 => (x8*x4*x2,    poly_13(x, x2, x4, x8, c(0), c(1), c(2), c(3), c(4), c(5), c(6), c(7), c(8), c(9), c(10), c(11), c(12), c(13))),
            15 => (x8*x4*x2*x,  poly_14(x, x2, x4, x8, c(0), c(1), c(2), c(3), c(4), c(5), c(6), c(7), c(8), c(9), c(10), c(11), c(12), c(13), c(14))),
            _  => unsafe { std::hint::unreachable_unchecked() }
        };

        sum.mul_add(rmx, res)
    }

    #[inline(always)]
    fn poly(x: Self::Vf, c: &[Self]) -> Self::Vf {
        x.poly_f(c.len(), |i| unsafe { Self::Vf::splat(*c.get_unchecked(i)) })
    }

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

    fn cbrt(x: Self::Vf) -> Self::Vf;

    fn powf(x: Self::Vf, e: Self::Vf) -> Self::Vf;

    fn ln(x: Self::Vf) -> Self::Vf;
    fn ln_1p(x: Self::Vf) -> Self::Vf;
    fn log2(x: Self::Vf) -> Self::Vf;
    fn log10(x: Self::Vf) -> Self::Vf;

    fn erf(x: Self::Vf) -> Self::Vf;
    fn erfinv(x: Self::Vf) -> Self::Vf;

    fn next_float(x: Self::Vf) -> Self::Vf;
    fn prev_float(x: Self::Vf) -> Self::Vf;

    fn tgamma(x: Self::Vf) -> Self::Vf;

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

    // TODO: Add some associated forms?
    /// This is split into its own function to avoid direct recursion within `legendre_p`,
    /// as that will prevent loop unrolling or inlining and optimization at all.
    #[inline(always)]
    fn hardcoded_legendre_p(x: Self::Vf, n: u32) -> Self::Vf {
        macro_rules! c {
            ($n:expr, $d:expr) => {
                Self::Vf::splat(Self::from_i32($n) / Self::from_i32($d))
            };
        }

        let x2 = x * x;
        let x4 = x2 * x2;
        let x8 = x4 * x4;

        // hand-tuned Estrin's scheme polynomials
        match n {
            2 => x2.mul_add(c!(3, 2), c!(-1, 2)),
            3 => x * x2.mul_add(c!(5, 2), c!(-3, 2)),
            4 => x4.mul_add(c!(35, 8), x2.mul_add(c!(-15, 4), c!(3, 8))),
            5 => x * x4.mul_add(c!(63, 8), x2.mul_add(c!(-35, 4), c!(15, 8))),
            6 => x4.mul_add(
                x2.mul_add(c!(231, 16), c!(-315, 16)),
                x2.mul_add(c!(105, 16), c!(-5, 16)),
            ),
            7 => {
                x * x4.mul_add(
                    x2.mul_add(c!(429, 16), c!(-693, 16)),
                    x2.mul_add(c!(315, 16), c!(-35, 16)),
                )
            }
            8 => x8.mul_add(
                c!(6435, 128),
                x4.mul_add(
                    x2.mul_add(c!(-3003, 32), c!(3465, 64)),
                    x2.mul_add(c!(-315, 32), c!(35, 128)),
                ),
            ),
            9 => {
                x * x8.mul_add(
                    c!(12155, 128),
                    x4.mul_add(
                        x2.mul_add(c!(-6435, 32), c!(9009, 64)),
                        x2.mul_add(c!(-1155, 32), c!(315, 128)),
                    ),
                )
            }
            10 => x8.mul_add(
                x2.mul_add(c!(46189, 256), c!(-109395, 256)),
                x4.mul_add(
                    x2.mul_add(c!(45045, 128), c!(-15015, 128)),
                    x2.mul_add(c!(3465, 256), c!(-63, 256)),
                ),
            ),
            11 => {
                x * x8.mul_add(
                    x2.mul_add(c!(88179, 256), c!(-230945, 256)),
                    x4.mul_add(
                        x2.mul_add(c!(109395, 128), c!(-45045, 128)),
                        x2.mul_add(c!(15015, 256), c!(-693, 256)),
                    ),
                )
            }
            12 => x8.mul_add(
                x4.mul_add(c!(676039, 1024), x2.mul_add(c!(-969969, 512), c!(2078505, 1024))),
                x4.mul_add(
                    x2.mul_add(c!(-255255, 256), c!(225225, 1024)),
                    x2.mul_add(c!(-9009, 512), c!(231, 1024)),
                ),
            ),
            13 => {
                x * x8.mul_add(
                    x4.mul_add(c!(1300075, 1024), x2.mul_add(c!(-2028117, 512), c!(4849845, 1024))),
                    x4.mul_add(
                        x2.mul_add(c!(-692835, 256), c!(765765, 1024)),
                        x2.mul_add(c!(-45045, 512), c!(3003, 1024)),
                    ),
                )
            }
            _ => unsafe { std::hint::unreachable_unchecked() },
        }
    }

    #[inline(always)]
    fn legendre_p(x: Self::Vf, n: u32, m: u32) -> Self::Vf {
        let zero = Self::Vf::zero();
        let one = Self::Vf::one();

        match (n, m) {
            (0, 0) => return one,
            (1, 0) => return x,
            (n, 0) if n <= 13 => return Self::hardcoded_legendre_p(x, n),
            (n, 0) => {
                let mut k = 14; // set to max degree hard-coded + 1

                // these should inline
                let mut p0 = Self::hardcoded_legendre_p(x, k - 2);
                let mut p1 = Self::hardcoded_legendre_p(x, k - 1);

                while k <= n {
                    let nf = Self::Vf::splat(Self::from_u32(k));

                    let tmp = p1;
                    p1 = x.mul_sub((nf + nf).mul_sub(p1, p1), nf.mul_sub(p0, p0)) / nf;
                    p0 = tmp;

                    k += 1;
                }

                return p1;
            }
            _ => {}
        }

        let jacobi = Self::jacobi_d(x, zero, zero, n, m);

        let x12 = x.nmul_add(x, one); // (1 - x^2)

        if m & 1 == 0 {
            jacobi * x12.powi((m >> 1) as i32)
        } else {
            // negate sign for odd powers (-1)^m
            -jacobi * x12.powi(m as i32).sqrt()
        }
    }

    #[inline(always)]
    fn jacobi_d(x: Self::Vf, alpha: Self::Vf, beta: Self::Vf, n: u32, m: u32) -> Self::Vf {
        if unlikely!(m > n) {
            return Self::Vf::zero();
        }

        let one = Self::Vf::one();

        let half = Self::Vf::splat_any(0.5);
        let mut scale = one;
        let mut jf = one;
        let nf = Self::Vf::splat(Self::from_u32(n));

        let t0 = half * (nf + alpha + beta);

        for j in 0..m {
            scale *= half.mul_add(jf, t0);
            jf += one;
        }

        let mf = Self::Vf::splat(Self::from_u32(m));

        scale * Self::jacobi(x, alpha + mf, beta + mf, n - m)
    }

    // TODO: Find more places to optimize and reduce error
    #[inline(always)]
    fn jacobi(x: Self::Vf, alpha: Self::Vf, beta: Self::Vf, n: u32) -> Self::Vf {
        /*
            This implementation is a little weird since I wanted to keep it generic, but all casts
            from integers should be const-folded
        */

        let one = Self::Vf::one();

        let mut y0 = one;

        if unlikely!(n == 0) {
            return y0;
        }

        let half = Self::Vf::splat_any(0.5);
        let two = Self::Vf::splat_any(2);

        let alpha_p_beta = alpha + beta;
        let alpha2 = alpha * alpha;
        let beta2 = beta * beta;
        let alpha1 = alpha - one;
        let beta1 = beta - one;
        let alpha2beta2 = alpha2 - beta2;

        //let mut y1 = alpha + one + half * (alpha_p_beta + two) * (x - one);
        let mut y1 = half * (x.mul_add(alpha, alpha) + x.mul_sub(beta, beta) + x + x);

        let mut yk = y1;
        let mut k = Self::from_u32(2);

        let k_max = Self::from_u32(n) * (Self::from_u32(1) + Self::__EPSILON);

        while k < k_max {
            let kf = Self::Vf::splat(k);
            let kf2 = two * kf;

            let k_alpha_p_beta = kf + alpha_p_beta;
            let k2_alpha_p_beta = kf2 + alpha_p_beta;

            let k2_alpha_p_beta_m2 = k2_alpha_p_beta - two;

            let denom = kf2 * k_alpha_p_beta * k2_alpha_p_beta_m2;
            let t0 = x.mul_add(k2_alpha_p_beta * k2_alpha_p_beta_m2, alpha2beta2);
            let gamma1 = k2_alpha_p_beta.mul_sub(t0, t0);
            let gamma0 = two * (kf + alpha1) * (kf + beta1) * k2_alpha_p_beta;

            yk = gamma1.mul_sub(y1, gamma0 * y0) / denom;

            y0 = y1;
            y1 = yk;

            k = k + Self::from_u32(1);
        }

        yk
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
