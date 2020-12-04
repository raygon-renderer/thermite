//! Vectorized Math Library

#![allow(unused)]

use crate::*;

pub mod compensated;
pub mod complex;
#[cfg(feature = "nightly")]
pub mod hyperdual;
pub mod poly;

mod pd;
mod ps;

/// Execution policy used for controlling performance/precision/size tradeoffs in mathematical functions.
pub trait Policy: Debug + Clone + Copy + PartialEq + Eq + PartialOrd + Ord + core::hash::Hash {
    /// The specific policy used. This is a constant to allow for dead-code elimination of branches.
    const POLICY: Parameters;
}

/** Execution Policies (precision, performance, etc.)

To define a custom policy:
```rust,ignore
pub struct MyPolicy;

impl Policy for MyPolicy {
    const POLICY: Parameters = Parameters {
        check_overflow: false,
        unroll_loops: false,
        extra_precision: true,
    };
}

let y = x.cbrt_p::<MyPolicy>();
```
*/
pub mod policies {
    use super::Policy;

    /// Customizable Policy Parameters
    pub struct Parameters {
        /// If true, methods will check for infinity/NaN/invalid domain issues and give a well-formed standard result.
        ///
        /// If false, all of that work is avoided, and the result is undefined in those cases. Garbage in, garbage out.
        ///
        /// However, those checks can be expensive.
        pub check_overflow: bool,

        /// If true, unrolled and optimized versions of some algorithms will be used. These can be much faster than
        /// the linear variants. If code size is important, this will improve codegen when used with `opt-level=z`
        pub unroll_loops: bool,

        /// If true, extra care is taken to improve precision of results, often at the cost of some performance.
        pub extra_precision: bool,

        /// If true, methods will not try to avoid extra work by branching. Some of the internal branches are expensive,
        /// but branchless may be desired in some cases, such as minimizing code size.
        pub avoid_branching: bool,
    }

    /// Optimize for performance at the cost of precision and safety (doesn't handle special cases such as NaNs or overflow).
    ///
    /// On instruction sets with FMA, this usually doesn't hurt precision too much, but will still avoid overflow/underflow checking,
    /// which can result in undefined behavior.
    #[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
    pub struct UltraPerformance;

    /// Optimize for performance, ideally without losing precision.
    ///
    /// This is the default policy for [`SimdVectorizedMath`](super::SimdVectorizedMath),
    /// and tries to provide as much precision and performance as possible.
    #[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
    pub struct Performance;

    /// Optimize for precision, at the cost of performance if necessary.
    ///
    /// On instruction sets with FMA, performance may not be hurt too much.
    #[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
    pub struct Precision;

    /// Optimize for code size, avoids hard-coded equations or loop unrolling.
    ///
    /// Performance is not a priority for this policy.
    ///
    /// Best used in conjuction with `opt-level=z`
    #[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
    pub struct Size;

    impl Policy for UltraPerformance {
        const POLICY: Parameters = Parameters {
            check_overflow: false,
            unroll_loops: true,
            extra_precision: false,
            avoid_branching: false,
        };
    }

    impl Policy for Performance {
        const POLICY: Parameters = Parameters {
            check_overflow: true,
            unroll_loops: true,
            extra_precision: false,
            avoid_branching: false,
        };
    }

    impl Policy for Precision {
        const POLICY: Parameters = Parameters {
            check_overflow: true,
            unroll_loops: true,
            extra_precision: true,
            avoid_branching: false,
        };
    }

    impl Policy for Size {
        const POLICY: Parameters = Parameters {
            check_overflow: true,
            unroll_loops: false,
            extra_precision: false,
            avoid_branching: true,
        };
    }
}

use policies::*;

//TODO: beta function, j0, y0

/// Set of vectorized special functions allowing specific execution policies
///
/// Please refer to the documentation of [`SimdVectorizedMath`] for function reference and [`policies`] for an
/// overview of available execution policies.
pub trait SimdVectorizedMathPolicied<S: Simd>: SimdFloatVector<S> {
    fn scale_p<P: Policy>(self, in_min: Self, in_max: Self, out_min: Self, out_max: Self) -> Self;
    fn lerp_p<P: Policy>(self, a: Self, b: Self) -> Self;
    fn fmod_p<P: Policy>(self, y: Self) -> Self;
    fn hypot_p<P: Policy>(self, y: Self) -> Self;
    fn poly_p<P: Policy>(self, coefficients: &[Self::Element]) -> Self;
    fn poly_rational_p<P: Policy>(
        self,
        numerator_coefficients: &[Self::Element],
        denominator_coefficients: &[Self::Element],
    ) -> Self;

    fn poly_f_p<P: Policy, F>(self, n: usize, f: F) -> Self
    where
        F: FnMut(usize) -> Self;

    fn sin_p<P: Policy>(self) -> Self;
    fn cos_p<P: Policy>(self) -> Self;
    fn tan_p<P: Policy>(self) -> Self;
    fn sin_cos_p<P: Policy>(self) -> (Self, Self);
    fn sinh_p<P: Policy>(self) -> Self;
    fn cosh_p<P: Policy>(self) -> Self;
    fn tanh_p<P: Policy>(self) -> Self;
    fn asinh_p<P: Policy>(self) -> Self;
    fn acosh_p<P: Policy>(self) -> Self;
    fn atanh_p<P: Policy>(self) -> Self;
    fn asin_p<P: Policy>(self) -> Self;
    fn acos_p<P: Policy>(self) -> Self;
    fn atan_p<P: Policy>(self) -> Self;
    fn atan2_p<P: Policy>(self, x: Self) -> Self;
    fn exp_p<P: Policy>(self) -> Self;
    fn exph_p<P: Policy>(self) -> Self;
    fn exp2_p<P: Policy>(self) -> Self;
    fn exp10_p<P: Policy>(self) -> Self;
    fn exp_m1_p<P: Policy>(self) -> Self;
    fn cbrt_p<P: Policy>(self) -> Self;
    fn powf_p<P: Policy>(self, e: Self) -> Self;
    fn powiv_p<P: Policy>(self, e: S::Vi32) -> Self;
    fn powi_p<P: Policy>(self, e: i32) -> Self;
    fn ln_p<P: Policy>(self) -> Self;
    fn ln_1p_p<P: Policy>(self) -> Self;
    fn log2_p<P: Policy>(self) -> Self;
    fn log10_p<P: Policy>(self) -> Self;
    fn erf_p<P: Policy>(self) -> Self;
    fn erfinv_p<P: Policy>(self) -> Self;
    fn tgamma_p<P: Policy>(self) -> Self;
    fn next_float_p<P: Policy>(self) -> Self;
    fn prev_float_p<P: Policy>(self) -> Self;
    fn smoothstep_p<P: Policy>(self) -> Self;
    fn smootherstep_p<P: Policy>(self) -> Self;
    fn smootheststep_p<P: Policy>(self) -> Self;
    fn hermite_p<P: Policy>(self, n: u32) -> Self;
    fn hermitev_p<P: Policy>(self, n: S::Vu32) -> Self;
    fn jacobi_p<P: Policy>(self, alpha: Self, beta: Self, n: u32, m: u32) -> Self;
    fn legendre_p<P: Policy>(self, n: u32, m: u32) -> Self;
}

/// Set of vectorized special functions optimized for both single and double precision.
///
/// To use a specific execution policy for any function listed below, simply append `_p` to the function name and provide one of
/// the [available policies](policies) via turbofish, such as `x.sin_p::<`[`Precision`](policies::Precision)`>()`
///
/// The default execution policy for all functions in `SimdVectorizedMath` is [`Performance`](policies::Performance).
pub trait SimdVectorizedMath<S: Simd>: SimdFloatVector<S> + SimdVectorizedMathPolicied<S> {
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

    /// Computes `self.poly(numerator_coefficients) / self.poly(denominator_coefficients)` with extra precision if the policy calls for it.
    fn poly_rational(
        self,
        numerator_coefficients: &[Self::Element],
        denominator_coefficients: &[Self::Element],
    ) -> Self;

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

    //fn log2_int(self) -> Vu32<S>;

    /// Computes the error function for each value in a vector.
    fn erf(self) -> Self;
    /// Computes the inverse error function for each value in a vector.
    fn erfinv(self) -> Self;

    /// Computes the Gamma function (`Γ(z)`) for any real input, for each value in a vector.
    ///
    /// This implementation uses a few different behaviors to ensure the greatest precision where possible.
    ///
    /// * For non-integer positive inputs, it uses the Lanczos approximation.
    /// * For small non-integer negative inputs, it uses the recursive identity `Γ(z)=Γ(z+1)/z` until `z` is positive.
    /// * For large non-integer negative inputs, it uses the reflection formula `-π/(Γ(z)sin(πz)z)`.
    /// * For positive integers, it simply computes the factorial in a tight loop to ensure precision. Lookup tables could not be used with SIMD.
    /// * At zero, the result will be positive or negative infinity based on the input sign (signed zero is a thing).
    ///
    /// NOTE: The Gamma function is not defined for negative integers.
    fn tgamma(self) -> Self;

    /// Finds the next representable float moving upwards to positive infinity.
    fn next_float(self) -> Self;

    /// Finds the previous representable float moving downwards to negative infinity.
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

    /// Calculates a [sigmoid-like 7th-order interpolation function](https://en.wikipedia.org/wiki/Smoothstep#7th-order_equation).
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

    /// Computes the m-th derivative of the n-th degree Jacobi polynomial
    ///
    /// A the special case where α and β are both zero, the Jacobi polynomial reduces to a
    /// Legendre polynomial.
    ///
    /// **NOTE**: Given constant α, β or `n`, LLVM will happily optimize those away and unroll loops.
    fn jacobi(self, alpha: Self, beta: Self, n: u32, m: u32) -> Self;

    /// Computes the m-th associated n-th degree Legendre polynomial,
    /// where m=0 signifies the regular n-th degree Legendre polynomial.
    ///
    /// If `m` is odd, the input is only valid between -1 and 1
    ///
    /// **NOTE**: Given constant `n` and/or `m`, LLVM will happily unroll and optimize inner loops.
    ///
    /// Internally, this is computed with [`jacobi_d`](#tymethod.jacobi_d)
    fn legendre(self, n: u32, m: u32) -> Self;
}

#[rustfmt::skip]
#[dispatch(S, thermite = "crate")]
impl<S: Simd, T> SimdVectorizedMathPolicied<S> for T
where
    T: SimdFloatVector<S>,
    <T as SimdVectorBase<S>>::Element: SimdVectorizedMathInternal<S, Vf = T>,
{
    #[inline] fn scale_p<P: Policy>(self, in_min: Self, in_max: Self, out_min: Self, out_max: Self) -> Self {
        <<Self as SimdVectorBase<S>>::Element as SimdVectorizedMathInternal<S>>::scale::<P>(self, in_min, in_max, out_min, out_max)
    }

    #[inline] fn lerp_p<P: Policy>(self, a: Self, b: Self)   -> Self { <<Self as SimdVectorBase<S>>::Element as SimdVectorizedMathInternal<S>>::lerp::<P>(self, a, b) }
    #[inline] fn fmod_p<P: Policy>(self, y: Self)            -> Self { <<Self as SimdVectorBase<S>>::Element as SimdVectorizedMathInternal<S>>::fmod::<P>(self, y) }
    #[inline] fn hypot_p<P: Policy>(self, y: Self)           -> Self { <<Self as SimdVectorBase<S>>::Element as SimdVectorizedMathInternal<S>>::hypot::<P>(self, y) }
    #[inline] fn powi_p<P: Policy>(self, e: i32)             -> Self { <<Self as SimdVectorBase<S>>::Element as SimdVectorizedMathInternal<S>>::powi::<P>(self, e) }
    #[inline] fn powiv_p<P: Policy>(self, e: S::Vi32)        -> Self { <<Self as SimdVectorBase<S>>::Element as SimdVectorizedMathInternal<S>>::powiv::<P>(self, e) }

    #[inline] fn poly_f_p<P: Policy, F>(self, n: usize, f: F) -> Self
    where
        F: FnMut(usize) -> Self,
    {
        <<Self as SimdVectorBase<S>>::Element as SimdVectorizedMathInternal<S>>::poly_f::<F, P>(self, n, f)
    }

    #[inline] fn poly_p<P: Policy>(self, coefficients: &[Self::Element]) -> Self {
        <<Self as SimdVectorBase<S>>::Element as SimdVectorizedMathInternal<S>>::poly::<P>(self, coefficients)
    }

    #[inline] fn poly_rational_p<P: Policy>(
        self,
        numerator_coefficients: &[Self::Element],
        denominator_coefficients: &[Self::Element],
    ) -> Self {
        <<Self as SimdVectorBase<S>>::Element as SimdVectorizedMathInternal<S>>::poly_rational::<P>(self, numerator_coefficients, denominator_coefficients)
    }

    #[inline] fn hermite_p<P: Policy>(self, n: u32)                                      -> Self {
        <<Self as SimdVectorBase<S>>::Element as SimdVectorizedMathInternal<S>>::hermite::<P>(self, n)
    }
    #[inline] fn hermitev_p<P: Policy>(self, n: S::Vu32)                                 -> Self {
        <<Self as SimdVectorBase<S>>::Element as SimdVectorizedMathInternal<S>>::hermitev::<P>(self, n)
    }
    #[inline] fn jacobi_p<P: Policy>(self, alpha: Self, beta: Self, n: u32, m: u32)    -> Self {
        <<Self as SimdVectorBase<S>>::Element as SimdVectorizedMathInternal<S>>::jacobi::<P>(self, alpha, beta, n, m)
    }
    #[inline] fn legendre_p<P: Policy>(self, n: u32, m: u32)                           -> Self {
        <<Self as SimdVectorBase<S>>::Element as SimdVectorizedMathInternal<S>>::legendre::<P>(self, n, m)
    }

    #[inline] fn sin_p<P: Policy>(self)              -> Self         { <<Self as SimdVectorBase<S>>::Element as SimdVectorizedMathInternal<S>>::sin::<P>(self)  }
    #[inline] fn cos_p<P: Policy>(self)              -> Self         { <<Self as SimdVectorBase<S>>::Element as SimdVectorizedMathInternal<S>>::cos::<P>(self)  }
    #[inline] fn tan_p<P: Policy>(self)              -> Self         { <<Self as SimdVectorBase<S>>::Element as SimdVectorizedMathInternal<S>>::tan::<P>(self)  }
    #[inline] fn sin_cos_p<P: Policy>(self)          -> (Self, Self) { <<Self as SimdVectorBase<S>>::Element as SimdVectorizedMathInternal<S>>::sin_cos::<P>(self)  }
    #[inline] fn sinh_p<P: Policy>(self)             -> Self         { <<Self as SimdVectorBase<S>>::Element as SimdVectorizedMathInternal<S>>::sinh::<P>(self)  }
    #[inline] fn cosh_p<P: Policy>(self)             -> Self         { <<Self as SimdVectorBase<S>>::Element as SimdVectorizedMathInternal<S>>::cosh::<P>(self)  }
    #[inline] fn tanh_p<P: Policy>(self)             -> Self         { <<Self as SimdVectorBase<S>>::Element as SimdVectorizedMathInternal<S>>::tanh::<P>(self)  }
    #[inline] fn asinh_p<P: Policy>(self)            -> Self         { <<Self as SimdVectorBase<S>>::Element as SimdVectorizedMathInternal<S>>::asinh::<P>(self)  }
    #[inline] fn acosh_p<P: Policy>(self)            -> Self         { <<Self as SimdVectorBase<S>>::Element as SimdVectorizedMathInternal<S>>::acosh::<P>(self)  }
    #[inline] fn atanh_p<P: Policy>(self)            -> Self         { <<Self as SimdVectorBase<S>>::Element as SimdVectorizedMathInternal<S>>::atanh::<P>(self)  }
    #[inline] fn asin_p<P: Policy>(self)             -> Self         { <<Self as SimdVectorBase<S>>::Element as SimdVectorizedMathInternal<S>>::asin::<P>(self)  }
    #[inline] fn acos_p<P: Policy>(self)             -> Self         { <<Self as SimdVectorBase<S>>::Element as SimdVectorizedMathInternal<S>>::acos::<P>(self)  }
    #[inline] fn atan_p<P: Policy>(self)             -> Self         { <<Self as SimdVectorBase<S>>::Element as SimdVectorizedMathInternal<S>>::atan::<P>(self)  }
    #[inline] fn atan2_p<P: Policy>(self, x: Self)   -> Self         { <<Self as SimdVectorBase<S>>::Element as SimdVectorizedMathInternal<S>>::atan2::<P>(self, x)  }
    #[inline] fn exp_p<P: Policy>(self)              -> Self         { <<Self as SimdVectorBase<S>>::Element as SimdVectorizedMathInternal<S>>::exp::<P>(self)  }
    #[inline] fn exph_p<P: Policy>(self)             -> Self         { <<Self as SimdVectorBase<S>>::Element as SimdVectorizedMathInternal<S>>::exph::<P>(self)  }
    #[inline] fn exp2_p<P: Policy>(self)             -> Self         { <<Self as SimdVectorBase<S>>::Element as SimdVectorizedMathInternal<S>>::exp2::<P>(self)  }
    #[inline] fn exp10_p<P: Policy>(self)            -> Self         { <<Self as SimdVectorBase<S>>::Element as SimdVectorizedMathInternal<S>>::exp10::<P>(self)  }
    #[inline] fn exp_m1_p<P: Policy>(self)           -> Self         { <<Self as SimdVectorBase<S>>::Element as SimdVectorizedMathInternal<S>>::exp_m1::<P>(self)  }
    #[inline] fn cbrt_p<P: Policy>(self)             -> Self         { <<Self as SimdVectorBase<S>>::Element as SimdVectorizedMathInternal<S>>::cbrt::<P>(self)  }
    #[inline] fn powf_p<P: Policy>(self, e: Self)    -> Self         { <<Self as SimdVectorBase<S>>::Element as SimdVectorizedMathInternal<S>>::powf::<P>(self, e)  }
    #[inline] fn ln_p<P: Policy>(self)               -> Self         { <<Self as SimdVectorBase<S>>::Element as SimdVectorizedMathInternal<S>>::ln::<P>(self)  }
    #[inline] fn ln_1p_p<P: Policy>(self)            -> Self         { <<Self as SimdVectorBase<S>>::Element as SimdVectorizedMathInternal<S>>::ln_1p::<P>(self)  }
    #[inline] fn log2_p<P: Policy>(self)             -> Self         { <<Self as SimdVectorBase<S>>::Element as SimdVectorizedMathInternal<S>>::log2::<P>(self)  }
    #[inline] fn log10_p<P: Policy>(self)            -> Self         { <<Self as SimdVectorBase<S>>::Element as SimdVectorizedMathInternal<S>>::log10::<P>(self)  }
    #[inline] fn erf_p<P: Policy>(self)              -> Self         { <<Self as SimdVectorBase<S>>::Element as SimdVectorizedMathInternal<S>>::erf::<P>(self)  }
    #[inline] fn erfinv_p<P: Policy>(self)           -> Self         { <<Self as SimdVectorBase<S>>::Element as SimdVectorizedMathInternal<S>>::erfinv::<P>(self)  }
    #[inline] fn tgamma_p<P: Policy>(self)           -> Self         { <<Self as SimdVectorBase<S>>::Element as SimdVectorizedMathInternal<S>>::tgamma::<P>(self)  }
    #[inline] fn next_float_p<P: Policy>(self)       -> Self         { <<Self as SimdVectorBase<S>>::Element as SimdVectorizedMathInternal<S>>::next_float::<P>(self)  }
    #[inline] fn prev_float_p<P: Policy>(self)       -> Self         { <<Self as SimdVectorBase<S>>::Element as SimdVectorizedMathInternal<S>>::prev_float::<P>(self)  }
    #[inline] fn smoothstep_p<P: Policy>(self)       -> Self         { <<Self as SimdVectorBase<S>>::Element as SimdVectorizedMathInternal<S>>::smoothstep::<P>(self)  }
    #[inline] fn smootherstep_p<P: Policy>(self)     -> Self         { <<Self as SimdVectorBase<S>>::Element as SimdVectorizedMathInternal<S>>::smootherstep::<P>(self)  }
    #[inline] fn smootheststep_p<P: Policy>(self)    -> Self         { <<Self as SimdVectorBase<S>>::Element as SimdVectorizedMathInternal<S>>::smootheststep::<P>(self)  }
}

#[rustfmt::skip]
impl<S: Simd, T> SimdVectorizedMath<S> for T
where
    T: SimdFloatVector<S>,
    <T as SimdVectorBase<S>>::Element: SimdVectorizedMathInternal<S, Vf = T>,
{
    #[inline(always)] fn scale(self, in_min: Self, in_max: Self, out_min: Self, out_max: Self) -> Self {
        self.scale_p::<Performance>(in_min, in_max, out_min, out_max)
    }

    #[inline(always)] fn lerp(self, a: Self, b: Self)   -> Self { self.lerp_p::<Performance>(a, b) }
    #[inline(always)] fn fmod(self, y: Self)            -> Self { self.fmod_p::<Performance>(y) }
    #[inline(always)] fn hypot(self, y: Self)           -> Self { self.hypot_p::<Performance>(y) }
    #[inline(always)] fn powi(self, e: i32)             -> Self { self.powi_p::<Performance>(e) }
    #[inline(always)] fn powiv(self, e: S::Vi32)        -> Self { self.powiv_p::<Performance>(e) }

    #[inline(always)] fn poly_f<F>(self, n: usize, f: F) -> Self
    where
        F: FnMut(usize) -> Self,
    {
        self.poly_f_p::<Performance, F>(n, f)
    }

    #[inline(always)] fn poly_rational(
        self,
        numerator_coefficients: &[Self::Element],
        denominator_coefficients: &[Self::Element],
    ) -> Self {
        self.poly_rational_p::<Performance>(numerator_coefficients, denominator_coefficients)
    }

    #[inline(always)] fn poly(self, coefficients: &[Self::Element])             -> Self { self.poly_p::<Performance>(coefficients) }
    #[inline(always)] fn hermite(self, n: u32)                                  -> Self { self.hermite_p::<Performance>(n) }
    #[inline(always)] fn hermitev(self, n: S::Vu32)                             -> Self { self.hermitev_p::<Performance>(n) }
    #[inline(always)] fn jacobi(self, alpha: Self, beta: Self, n: u32, m: u32)  -> Self { self.jacobi_p::<Performance>(alpha, beta, n, m) }
    #[inline(always)] fn legendre(self, n: u32, m: u32)                         -> Self { self.legendre_p::<Performance>(n, m) }

    #[inline(always)] fn sin(self)              -> Self         { self.sin_p::<Performance>() }
    #[inline(always)] fn cos(self)              -> Self         { self.cos_p::<Performance>() }
    #[inline(always)] fn tan(self)              -> Self         { self.tan_p::<Performance>() }
    #[inline(always)] fn sin_cos(self)          -> (Self, Self) { self.sin_cos_p::<Performance>() }
    #[inline(always)] fn sinh(self)             -> Self         { self.sinh_p::<Performance>() }
    #[inline(always)] fn cosh(self)             -> Self         { self.cosh_p::<Performance>() }
    #[inline(always)] fn tanh(self)             -> Self         { self.tanh_p::<Performance>() }
    #[inline(always)] fn asinh(self)            -> Self         { self.asinh_p::<Performance>() }
    #[inline(always)] fn acosh(self)            -> Self         { self.acosh_p::<Performance>() }
    #[inline(always)] fn atanh(self)            -> Self         { self.atanh_p::<Performance>() }
    #[inline(always)] fn asin(self)             -> Self         { self.asin_p::<Performance>() }
    #[inline(always)] fn acos(self)             -> Self         { self.acos_p::<Performance>() }
    #[inline(always)] fn atan(self)             -> Self         { self.atan_p::<Performance>() }
    #[inline(always)] fn atan2(self, x: Self)   -> Self         { self.atan2_p::<Performance>(x) }
    #[inline(always)] fn exp(self)              -> Self         { self.exp_p::<Performance>() }
    #[inline(always)] fn exph(self)             -> Self         { self.exph_p::<Performance>() }
    #[inline(always)] fn exp2(self)             -> Self         { self.exp2_p::<Performance>() }
    #[inline(always)] fn exp10(self)            -> Self         { self.exp10_p::<Performance>() }
    #[inline(always)] fn exp_m1(self)           -> Self         { self.exp_m1_p::<Performance>() }
    #[inline(always)] fn cbrt(self)             -> Self         { self.cbrt_p::<Performance>() }
    #[inline(always)] fn powf(self, e: Self)    -> Self         { self.powf_p::<Performance>(e) }
    #[inline(always)] fn ln(self)               -> Self         { self.ln_p::<Performance>() }
    #[inline(always)] fn ln_1p(self)            -> Self         { self.ln_1p_p::<Performance>() }
    #[inline(always)] fn log2(self)             -> Self         { self.log2_p::<Performance>() }
    #[inline(always)] fn log10(self)            -> Self         { self.log10_p::<Performance>() }
    #[inline(always)] fn erf(self)              -> Self         { self.erf_p::<Performance>() }
    #[inline(always)] fn erfinv(self)           -> Self         { self.erfinv_p::<Performance>() }
    #[inline(always)] fn tgamma(self)           -> Self         { self.tgamma_p::<Performance>() }
    #[inline(always)] fn next_float(self)       -> Self         { self.next_float_p::<Performance>() }
    #[inline(always)] fn prev_float(self)       -> Self         { self.prev_float_p::<Performance>() }
    #[inline(always)] fn smoothstep(self)       -> Self         { self.smoothstep_p::<Performance>() }
    #[inline(always)] fn smootherstep(self)     -> Self         { self.smootherstep_p::<Performance>() }
    #[inline(always)] fn smootheststep(self)    -> Self         { self.smootheststep_p::<Performance>() }
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

    #[inline(always)]
    fn scale<P: Policy>(
        x: Self::Vf,
        in_min: Self::Vf,
        in_max: Self::Vf,
        out_min: Self::Vf,
        out_max: Self::Vf,
    ) -> Self::Vf {
        ((x - in_min) / (in_max - in_min)).mul_adde(out_max - out_min, out_min)
    }

    #[inline(always)]
    fn lerp<P: Policy>(t: Self::Vf, a: Self::Vf, b: Self::Vf) -> Self::Vf {
        if S::INSTRSET.has_true_fma() {
            t.mul_add(b - a, a)
        } else {
            (Self::Vf::one() - t) * a + t * b
        }
    }

    #[inline(always)]
    fn fmod<P: Policy>(x: Self::Vf, y: Self::Vf) -> Self::Vf {
        x % y // Already implemented with operator overloads anyway
    }

    #[inline(always)]
    fn powi<P: Policy>(mut x: Self::Vf, mut e: i32) -> Self::Vf {
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
    fn powiv<P: Policy>(mut x: Self::Vf, mut e: S::Vi32) -> Self::Vf {
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
    fn hermite<P: Policy>(x: Self::Vf, mut n: u32) -> Self::Vf {
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
    fn hermitev<P: Policy>(x: Self::Vf, mut n: S::Vu32) -> Self::Vf {
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
    fn hypot<P: Policy>(x: Self::Vf, y: Self::Vf) -> Self::Vf {
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
    fn poly_f<F, P: Policy>(x: Self::Vf, n: usize, mut c: F) -> Self::Vf
    where
        F: FnMut(usize) -> Self::Vf
    {
        // Use tiny Horner's method for code size optimization
        if !P::POLICY.unroll_loops {
            let mut idx = n - 1;
            let mut sum = c(idx);

            loop {
                idx -= 1;
                sum = sum.mul_adde(x, c(idx));
                if idx == 0 { return sum; }
            }
        }

        use poly::*;

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
                    _ => unsafe { core::hint::unreachable_unchecked() }
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
                    _ => unsafe { core::hint::unreachable_unchecked() }
                }
            };
        }

        macro_rules! poly {
            ($name:ident($($pows:ident),*; $j:ident + $c:ident[$($coeff:expr),*])) => {{
                $name($($pows,)* $($c($j + $coeff)),*)
            }};
        }

        let xmd = x.powi_p::<P>(MAX_DEGREE_P0 as i32); // hopefully inlined

        let mut sum = Self::Vf::zero();
        let mut mul = Self::Vf::one();

        let x2 = x * x;
        let x4 = x2 * x2;
        let x8 = x4 * x4;

        // Use a hybrid Estrin/Horner algorithm
        let mut j = n;
        while j >= MAX_DEGREE_P0 {
            j -= MAX_DEGREE_P0;
            sum = sum.mul_adde(xmd, poly!(poly_15(x, x2, x4, x8; j + c[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15])));
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
            _  => unsafe { core::hint::unreachable_unchecked() }
        };

        sum.mul_adde(rmx, res)
    }

    #[inline(always)]
    fn poly<P: Policy>(x: Self::Vf, c: &[Self]) -> Self::Vf {
        x.poly_f_p::<P, _>(c.len(), |i| unsafe { Self::Vf::splat(*c.get_unchecked(i)) })
    }

    #[inline(always)]
    fn poly_rational<P: Policy>(
        x: Self::Vf,
        numerator_coefficients: &[Self],
        denominator_coefficients: &[Self],
    ) -> Self::Vf {
        if P::POLICY.extra_precision {
            let one = Self::Vf::one();
            let invert = x.gt(one);
            let bitmask = invert.bitmask();

            let mut n0 = unsafe { Self::Vf::undefined() };
            let mut n1 = unsafe { Self::Vf::undefined() };

            let mut d0 = unsafe { Self::Vf::undefined() };
            let mut d1 = unsafe { Self::Vf::undefined() };

            if P::POLICY.avoid_branching || !bitmask.all() {
                n0 = Self::poly::<P>(x, numerator_coefficients);
                d0 = Self::poly::<P>(x, denominator_coefficients);
            }

            if P::POLICY.avoid_branching || bitmask.any() {
                let inv = one / x;

                n1 = Self::poly_f::<_, P>(inv, numerator_coefficients.len(), |i| unsafe {
                    Self::Vf::splat(*numerator_coefficients.get_unchecked(numerator_coefficients.len() - i - 1))
                });

                d1 = Self::poly_f::<_, P>(inv, denominator_coefficients.len(), |i| unsafe {
                    Self::Vf::splat(*denominator_coefficients.get_unchecked(denominator_coefficients.len() - i - 1))
                });
            }

            // division is slow, but select is fast, so avoid dividing in the branches
            // to save a division at the cost of one extra select.
            invert.select(n1, n0) / invert.select(d1, d0)
        } else {
            Self::poly::<P>(x, numerator_coefficients) / Self::poly::<P>(x, denominator_coefficients)
        }
    }

    #[inline(always)]
    fn sin<P: Policy>(x: Self::Vf) -> Self::Vf {
        Self::sin_cos::<P>(x).0
    }

    #[inline(always)]
    fn cos<P: Policy>(x: Self::Vf) -> Self::Vf {
        Self::sin_cos::<P>(x).1
    }

    #[inline(always)]
    fn tan<P: Policy>(x: Self::Vf) -> Self::Vf {
        let (s, c) = Self::sin_cos::<P>(x);
        s / c
    }

    fn sin_cos<P: Policy>(x: Self::Vf) -> (Self::Vf, Self::Vf);

    fn sinh<P: Policy>(x: Self::Vf) -> Self::Vf;

    #[inline(always)]
    fn cosh<P: Policy>(x: Self::Vf) -> Self::Vf {
        let y: Self::Vf = Self::exph::<P>(x.abs()); // 0.5 * exp(x)
        y + Self::Vf::splat_any(0.25) / y // + 0.5 * exp(-x)
    }

    fn tanh<P: Policy>(x: Self::Vf) -> Self::Vf;

    fn asin<P: Policy>(x: Self::Vf) -> Self::Vf;
    fn acos<P: Policy>(x: Self::Vf) -> Self::Vf;
    fn atan<P: Policy>(x: Self::Vf) -> Self::Vf;
    fn atan2<P: Policy>(y: Self::Vf, x: Self::Vf) -> Self::Vf;

    fn asinh<P: Policy>(x: Self::Vf) -> Self::Vf;
    fn acosh<P: Policy>(x: Self::Vf) -> Self::Vf;
    fn atanh<P: Policy>(x: Self::Vf) -> Self::Vf;

    fn exp<P: Policy>(x: Self::Vf) -> Self::Vf;
    fn exph<P: Policy>(x: Self::Vf) -> Self::Vf;
    fn exp2<P: Policy>(x: Self::Vf) -> Self::Vf;
    fn exp10<P: Policy>(x: Self::Vf) -> Self::Vf;

    #[inline(always)]
    fn exp_m1<P: Policy>(x: Self::Vf) -> Self::Vf {
        x.exp_p::<P>() - Self::Vf::one()
    }

    fn cbrt<P: Policy>(x: Self::Vf) -> Self::Vf;

    fn powf<P: Policy>(x: Self::Vf, e: Self::Vf) -> Self::Vf;

    fn ln<P: Policy>(x: Self::Vf) -> Self::Vf;
    fn ln_1p<P: Policy>(x: Self::Vf) -> Self::Vf;
    fn log2<P: Policy>(x: Self::Vf) -> Self::Vf;
    fn log10<P: Policy>(x: Self::Vf) -> Self::Vf;

    fn erf<P: Policy>(x: Self::Vf) -> Self::Vf;
    fn erfinv<P: Policy>(x: Self::Vf) -> Self::Vf;

    fn next_float<P: Policy>(x: Self::Vf) -> Self::Vf;
    fn prev_float<P: Policy>(x: Self::Vf) -> Self::Vf;

    fn tgamma<P: Policy>(x: Self::Vf) -> Self::Vf;

    #[inline(always)]
    fn smoothstep<P: Policy>(x: Self::Vf) -> Self::Vf {
        // use integer coefficients to ensure as-accurate-as-possible casts to f32 or f64
        x * x * x.nmul_add(Self::Vf::splat_any(2i16), Self::Vf::splat_any(3i16))
    }

    #[inline(always)]
    fn smootherstep<P: Policy>(x: Self::Vf) -> Self::Vf {
        let c3 = Self::Vf::splat_any(10i16);
        let c4 = Self::Vf::splat_any(-15i16);
        let c5 = Self::Vf::splat_any(6i16);

        // Use Estrin's scheme here without c0-c2
        let x2 = x * x;
        let x4 = x2 * x2;

        x4.mul_add(x.mul_add(c5, c4), x2 * x * c3)
    }

    #[inline(always)]
    fn smootheststep<P: Policy>(x: Self::Vf) -> Self::Vf {
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
    fn hardcoded_legendre(x: Self::Vf, n: u32) -> Self::Vf {
        macro_rules! c {
            ($n:expr, $d:expr) => {
                Self::Vf::splat(Self::cast_from($n) / Self::cast_from($d))
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
            _ => unsafe { core::hint::unreachable_unchecked() },
        }
    }

    #[inline(always)]
    fn legendre<P: Policy>(x: Self::Vf, n: u32, m: u32) -> Self::Vf {
        let zero = Self::Vf::zero();
        let one = Self::Vf::one();

        match (n, m) {
            (0, 0) => return one,
            (1, 0) => return x,
            (n, 0) if n <= 13 => return Self::hardcoded_legendre(x, n),
            (n, 0) => {
                let mut k = 14; // set to max degree hard-coded + 1

                // these should inline
                let mut p0 = Self::hardcoded_legendre(x, k - 2);
                let mut p1 = Self::hardcoded_legendre(x, k - 1);

                while k <= n {
                    let nf = Self::Vf::splat_as::<u32>(k);

                    let tmp = p1;
                    p1 = x.mul_sub((nf + nf).mul_sub(p1, p1), nf.mul_sub(p0, p0)) / nf;
                    p0 = tmp;

                    k += 1;
                }

                return p1;
            }
            _ => {}
        }

        let jacobi = Self::jacobi::<P>(x, zero, zero, n, m);

        let x12 = x.nmul_add(x, one); // (1 - x^2)

        if m & 1 == 0 {
            jacobi * Self::powi::<P>(x12, (m >> 1) as i32)
        } else {
            // negate sign for odd powers (-1)^m
            -jacobi * Self::powi::<P>(x12, m as i32).sqrt()
        }
    }

    #[inline(always)]
    fn jacobi<P: Policy>(x: Self::Vf, mut alpha: Self::Vf, mut beta: Self::Vf, mut n: u32, m: u32) -> Self::Vf {
        /*
            This implementation is a little weird since I wanted to keep it generic, but all casts
            from integers should be const-folded
        */

        if unlikely!(m > n) {
            return Self::Vf::zero();
        }

        let one = Self::Vf::one();
        let two = Self::Vf::splat_any(2);
        let half = Self::Vf::splat_any(0.5);

        let mut scale = one;

        if m > 0 {
            let mut jf = one;
            let nf = Self::Vf::splat_as::<u32>(n);

            let t0 = half * (nf + alpha + beta);

            for j in 0..m {
                scale *= half.mul_add(jf, t0);
                jf += one;
            }

            let mf = Self::Vf::splat_as::<u32>(m);

            alpha += mf;
            beta += mf;
            n -= m;
        }

        if unlikely!(n == 0) {
            return scale; // scale * one
        }

        let mut y0 = one;

        let alpha_p_beta = alpha + beta;
        let alpha_sqr = alpha * alpha;
        let beta_sqr = beta * beta;
        let alpha1 = alpha - one;
        let beta1 = beta - one;
        let alpha2beta2 = alpha_sqr - beta_sqr;

        //let mut y1 = alpha + one + half * (alpha_p_beta + two) * (x - one);
        let mut y1 = half * (x.mul_add(alpha, alpha) + x.mul_sub(beta, beta) + x + x);

        let mut yk = y1;
        let mut k = Self::cast_from(2u32);

        let k_max = Self::cast_from(n) * (Self::cast_from(1u32) + Self::__EPSILON);

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

            k = k + Self::cast_from(1u32);
        }

        scale * yk
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
