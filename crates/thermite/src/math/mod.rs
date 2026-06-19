#![allow(clippy::needless_arbitrary_self_type)]

//! Mathematical functions for floating-point vector types.
//!
//! This module provides a comprehensive set of mathematical operations tailored for floating-point vector types.
//! It includes core mathematical functions, transcendental functions, spatial computations, and real-valued operations.
//!
//! The traits defined here are designed to be flexible and efficient, allowing for different precision and performance trade-offs
//! through the use of policies. Each mathematical trait has a corresponding version that accepts a policy parameter,
//! enabling fine-tuned control over the behavior of the functions.

mod consts;
pub mod policy;

pub use consts::FloatConsts;

use crate::element::{Element, ElementExt, FloatElement, FloatElementWithBits};
use crate::vector::{FloatVector, FloatVectorWithBits};

pub mod algorithms;
pub mod specialized;

use policy::{DefaultPolicy, Policy};

pub mod prelude {
    pub use crate::vector::FloatVector;

    pub use super::FloatConsts;
    pub use super::{
        CoreMath, CoreMathWithPolicy, RealMath, RealMathWithPolicy, ScalarMath, ScalarMathWithPolicy, SpatialMath,
        SpatialMathWithPolicy, TranscendentalMath, TranscendentalMathWithPolicy,
    };
}

// this is an implementation detail, required to be public so other crates can use it,
// but it's generally not for user-consumption.
#[doc(hidden)]
pub mod scalar;
use scalar::Unwrap;

pub trait Coefficients<T, const N: usize> {
    const COEFFICIENTS: [T; N];
}

// #[macro_export]
// macro_rules! poly {
//     ()
// }

// Helper macro to declare math traits and implementations
// for both policy and default policy versions. This reduces
// boilerplate and ensures consistency between the two traits,
// though it is a bit annoying to read and write.
macro_rules! decl_math {
    ($(
        $(#[$trait_meta:meta])*
        trait $trait:ident<$element:ident> $(: $($bound:ident)&+ )? { $(
            $(#[$meta:meta])*
            fn $name:ident [ $($generics:tt)* ][$($generic_names:ident),*]( $($arg_name:ident :$arg_ty:ty),* $(,)?) -> $ret:ty
                $(where [ $($where_clause:tt)* ])?;
            )*
        }
    )*) => {paste::paste! {$(
        #[doc = "" $trait " Math functions for floating-point vectors with customizable policies."]
        $(#[$trait_meta])*
        #[doc = ""]
        #[doc = "This trait provides a set of " [<$trait:lower>] " mathematical operations that can be performed"]
        #[doc = "on floating-point vector types. Each function has a variant that accepts"]
        #[doc = "a policy parameter, allowing for fine-tuned control over precision and performance."]
        #[doc = ""]
        #[doc = "For convenience, a default implementation is also provided in the [`" $trait "Math`] trait,"]
        #[doc = "which uses the [`DefaultPolicy`]. All floating-point vector types that implement"]
        #[doc = "the necessary internal math operations will automatically implement this trait, and"]
        #[doc = "the [`" $trait "Math`] trait as well for all types that implement this one."]
        #[thermite_macros::dispatch(Self, thermite = "crate")]
        pub trait [<$trait MathWithPolicy>] $(: $($bound +)+)? {$(
            $(#[$meta])* fn [<$name _p>]<P: Policy, $($generics)*>($($arg_name: $arg_ty),*) -> $ret
                $(where $($where_clause)*)?;
        )*}

        #[doc = "" $trait " Math functions for floating-point vectors using the default policy."]
        $(#[$trait_meta])*
        #[doc = ""]
        #[doc = "This trait requires " $($("[`" $bound "`]") ", "+)? " via the bounds on [`" $trait "MathWithPolicy`]."]
        #[doc = ""]
        #[doc = "This trait provides the same set of mathematical operations as [`" $trait "MathWithPolicy`],"]
        #[doc = "but uses the [`DefaultPolicy`] for all operations. This allows for easier usage"]
        #[doc = "when specific policy customization is not required."]
        #[doc = ""]
        #[doc = "Implementors of [`" $trait "MathWithPolicy`] will automatically implement this trait as well."]
        #[doc = ""]
        #[doc = "All methods here have an associated method in [`" $trait "MathWithPolicy`] with a `_p` suffix"]
        #[doc = "that accepts a policy parameter as the first generic argument."]
        #[thermite_macros::dispatch(Self, thermite = "crate")]
        pub trait [<$trait Math>]: [<$trait MathWithPolicy>] {$(
            $(#[$meta])* #[inline(always)] fn $name<$($generics)*>($($arg_name: $arg_ty),*) -> $ret
                $(where $($where_clause)*)?
            { [<$trait MathWithPolicy>]::[<$name _p>]::<DefaultPolicy, $($generic_names),*>($($arg_name),*) }
        )*}

        impl<M> [<$trait Math>] for M where M: [<$trait MathWithPolicy>] {}

        // Note: The FloatVector<Element = E> bound is necessary to ensure E is bounded.
        #[thermite_macros::dispatch(Self, thermite = "crate")]
        impl<E: $element, V: FloatVector<Element = E> + $($($bound +)+)?> [<$trait MathWithPolicy>] for V
            where V: specialized::[<Specialized $trait Math>]<E>
        {$(
            #[cfg(not(feature = "disable_dispatch"))]
            $(#[$meta])* #[inline(always)] fn [<$name _p>]<P: Policy, $($generics)*>($($arg_name: $arg_ty),*) -> $ret
                $(where $($where_clause)*)?
            { <V as specialized::[<Specialized $trait Math>]<E>>::$name::<P, $($generic_names),*>($($arg_name),*) }

            #[cfg(feature = "disable_dispatch")]
            $(#[$meta])* #[skip_dispatch] #[inline(always)] fn [<$name _p>]<P: Policy, $($generics)*>($($arg_name: $arg_ty),*) -> $ret
                $(where $($where_clause)*)?
            { <V as specialized::[<Specialized $trait Math>]<E>>::$name::<P, $($generic_names),*>($($arg_name),*) }
        )*})*

        #[doc = "Aggregate of all scalar math traits with customizable policies."]
        #[doc = ""]
        #[doc = "This trait collects every method from the following trait families into a single"]
        #[doc = "trait implemented directly on `f32` and `f64`:"]
        #[doc = ""]
        $(#[doc = "- [`" [<$trait MathWithPolicy>] "`]"])*
        #[doc = ""]
        #[doc = "All methods are prefixed with `scalar_` to avoid conflicts with the inherent methods"]
        #[doc = "already defined on `f32`/`f64` (e.g., `f32::sin`, `f32::exp`). The policy-aware"]
        #[doc = "versions additionally carry a `_p` suffix, following the same convention as the"]
        #[doc = "vector math traits."]
        #[doc = ""]
        #[doc = "# Limitations"]
        #[doc = ""]
        #[doc = "This trait is **only** implemented for bare scalar types. Code that is generic over"]
        #[doc = "a `FloatVector` bound will not accept a bare `f32` or `f64` - the scalar must be"]
        #[doc = "wrapped in [`Vector`](crate::Vector) first (e.g., `Vector::<f32>(x)`) to satisfy"]
        #[doc = "that bound. `ScalarMath` exists purely as a convenience for call-sites that already"]
        #[doc = "hold a concrete scalar and do not need to be generic."]
        #[doc = ""]
        #[doc = "For convenience, a default-policy version is provided by [`ScalarMath`], which"]
        #[doc = "drops the `_p` suffix and uses [`DefaultPolicy`] for all operations."]
        #[thermite_macros::dispatch(Self, thermite = "crate")]
        pub trait ScalarMathWithPolicy: ElementExt<Element = Self> + FloatElementWithBits {$($(
             $(#[$meta])* fn [<scalar_ $name _p>]<P: Policy, $($generics)*>($($arg_name: $arg_ty),*) -> $ret
                $(where $($where_clause)*)?;
        )*)*}

        #[doc = "Aggregate of all scalar math traits using the default policy."]
        #[doc = ""]
        #[doc = "This trait collects every method from the following trait families into a single"]
        #[doc = "trait implemented directly on `f32` and `f64`, using [`DefaultPolicy`] for all operations:"]
        #[doc = ""]
        $(#[doc = "- [`" [<$trait Math>] "`]"])*
        #[doc = ""]
        #[doc = "All methods are prefixed with `scalar_` to avoid conflicts with the inherent methods"]
        #[doc = "already defined on `f32`/`f64`. See [`ScalarMathWithPolicy`] for the policy-aware"]
        #[doc = "variant, which additionally carries a `_p` suffix on each method."]
        #[doc = ""]
        #[doc = "# Limitations"]
        #[doc = ""]
        #[doc = "This trait is **only** implemented for bare scalar types. Code that is generic over"]
        #[doc = "a `FloatVector` bound will not accept a bare `f32` or `f64` - the scalar must be"]
        #[doc = "wrapped in [`Vector`](crate::Vector) first (e.g., `Vector::<f32>(x)`) to satisfy"]
        #[doc = "that bound. `ScalarMath` exists purely as a convenience for call-sites that already"]
        #[doc = "hold a concrete scalar and do not need to be generic."]
        #[doc = ""]
        #[doc = "All types that implement [`ScalarMathWithPolicy`] automatically implement this trait."]
        #[thermite_macros::dispatch(Self, thermite = "crate")]
        pub trait ScalarMath: ScalarMathWithPolicy {$($(
            $(#[$meta])* #[inline(always)] fn [<scalar_ $name>]<$($generics)*>($($arg_name: $arg_ty),*) -> $ret
                $(where $($where_clause)*)?
            { ScalarMathWithPolicy::[<scalar_ $name _p>]::<DefaultPolicy, $($generic_names),*>($($arg_name),*) }
        )*)*}

        impl<M> ScalarMath for M where M: ScalarMathWithPolicy {}

        #[thermite_macros::dispatch(Self, thermite = "crate")]
        impl<E: ElementExt<Element = Self> + FloatElementWithBits> ScalarMathWithPolicy for E
        where
            $crate::Vector<E>: Unwrap<Unwrapped = E> +
                FloatVectorWithBits<Element = E,
                    Signed: Unwrap<Unwrapped = <E as Element>::Signed>,
                    Unsigned: Unwrap<Unwrapped = <E as Element>::Unsigned>,
                    SignedBits: Unwrap<Unwrapped = <E as FloatElementWithBits>::SignedBits>,
                    Bits: Unwrap<Unwrapped = <E as FloatElementWithBits>::Bits>
                >
                $(+ specialized::[<Specialized $trait Math>]<E>)*,
            E: $crate::register::FloatRegister<Storage = E>,
        {$($(
            $(#[$meta])* #[skip_dispatch] #[inline(always)] fn [<scalar_ $name _p>]<P: Policy, $($generics)*>($($arg_name: $arg_ty),*) -> $ret
                $(where $($where_clause)*)?
            {
                let ($(decl_math!(@SELF $arg_name this),)*) = Unwrap::wrap(($($arg_name,)*));

                let res = <$crate::Vector<E> as specialized::[<Specialized $trait Math>]<E>>::$name::<P, $($generic_names),*>($(decl_math!(@SELF $arg_name this)),*);

                Unwrap::unwrap(res)
            }
        )*)*}
    }};

    // rename `self` to `this`. Requires an existing ident to bind to.
    (@SELF self $rename:ident) => { $rename };
    (@SELF $other:ident $rename:ident) => { $other };
}

#[cfg(test)]
mod tests {
    use super::ScalarMath;

    #[test]
    fn test_f32_scalar_math() {
        let x: f32 = 1.0;
        let _ = x.scalar_sin();
    }
}

decl_math! {
    /// Float-specific mathematical functions like `ldexp` and `frexp`.
    trait Float<FloatElementWithBits>: FloatVectorWithBits {
        /// Computes `self * 2^exp` efficiently.
        fn ldexp[][](self: Self, exp: Self::SignedBits) -> Self;

        /// Decomposes `self` into its normalized fraction and an integral power of two.
        fn frexp[][](self: Self) -> (Self, Self::SignedBits);

        /// Removes denormal/subnormal values, flushing them to zero.
        ///
        /// If the precision policy is less than [`Best`](policy::PrecisionPolicy::Best),
        /// this will NOT preserve -0.0. However, at higher precision policies the
        /// negative zero will be correctly preserved.
        ///
        /// The crate feature `preserve_denormals` will disable this for default policies,
        /// which may be useful when targeting hardware or applications where the processor
        /// will handle denormals automatically.
        ///
        /// See [`DenormalBehavior`](policy::DenormalBehavior) for more options for how to control
        /// this function, as it is used extensively internally.
        fn flush_denormals[][](self: Self) -> Self;
    }

    /// This is the core set of mathematical operations that form the basis for more advanced functions.
    trait Core<FloatElement>: FloatVector {
        /// Computes the polynomial with the given coefficients at `self`.
        ///
        /// This will use fused multiply-add instructions where available for improved performance and accuracy, but
        /// falls back to standard operations if not.
        #[skip_dispatch] fn poly[const N: usize][N](self: Self, coeffs: &[Self::Element; N]) -> Self;

        /// Computes the polynomial with the given coefficients at `self`, but with the coefficients in reverse order.
        ///
        /// This will use fused multiply-add instructions where available for improved performance and accuracy, but
        /// falls back to standard operations if not.
        #[skip_dispatch] fn poly_rev[const N: usize][N](self: Self, coeffs: &[Self::Element; N]) -> Self;

        /// Computes the ratio of two polynomials at `self`, given the numerator and denominator coefficients.
        ///
        /// Equivalent to `poly(numerator) / poly(denominator)`, but with improved numerical stability in some cases.
        ///
        /// This will use fused multiply-add instructions where available for improved performance and accuracy, but
        /// falls back to standard operations if not.
        #[skip_dispatch] fn poly_rational[const N: usize, const D: usize][N, D](
            self: Self,
            numerator: &[Self::Element; N],
            denominator: &[Self::Element; D],
        ) -> Self;

        /// Returns the multiplicative inverse of `self`, which is `1 / self`.
        ///
        /// If using the policy version, you may select lower precision policies for extra performance,
        /// at the cost of accuracy.
        fn reciprocal[][](self: Self) -> Self;

        /// Returns the result of dividing `self` by `divisor`, i.e., `self / divisor`.
        ///
        /// Depending on the precision policy and available features, this may be
        /// optimized to use approximate reciprocal and multiplication for better
        /// performance, at the cost of accuracy.
        fn approx_div[][](self: Self, divisor: Self) -> Self;

        /// Returns the inverse square root of `self`, which is `1 / sqrt(self)`.
        ///
        /// If using the policy version, you may select lower precision policies for extra performance,
        /// at the cost of accuracy.
        fn inverse_sqrt[][](self: Self) -> Self;
        /// Returns `self` raised to the signed integer power of `e`.
        fn powi[][](self: Self, e: i32) -> Self;

        /// Returns `self` raised to the signed integer power of each element in `e`.
        fn powiv[][](self: Self, e: Self::Signed) -> Self;
    }

    /// Transcendental mathematical functions like trigonometric, exponential, and logarithmic functions.
    trait Transcendental<FloatElement>: CoreMathWithPolicy {
        /// Trigonometric sine and cosine, together. This will be more efficient than calling `sin` and `cos` separately.
        fn sin_cos[][](self: Self) -> (Self, Self);
        /// Trigonometric sine
        fn sin[][](self: Self) -> Self;
        /// Trigonometric cosine
        fn cos[][](self: Self) -> Self;
        /// Trigonometric tangent
        fn tan[][](self: Self) -> Self;

        /// Returns `cos(x) - 1` of `self`, which is more precise than `cos(x) - 1` directly near zero.
        ///
        /// Evaluated as `$-2\sin^2(x/2)$`, which has no cancellation near `x = 0`.
        fn cos_m1[][](self: Self) -> Self;
        /// Returns the versine `$1 - \cos(x)$` of `self`, evaluated as `$2\sin^2(x/2)$` (accurate near zero).
        fn versin[][](self: Self) -> Self;
        /// Returns the haversine `$\tfrac{1 - \cos(x)}{2}$` of `self`, evaluated as `$\sin^2(x/2)$` (accurate near zero).
        ///
        /// This is the kernel of the haversine great-circle-distance formula.
        fn haversin[][](self: Self) -> Self;

        /// Sine and cosine of `pi * x`, together. This will be more efficient than calling `sin_pi` and `cos_pi` separately,
        /// and more precise than computing them manually with `sin(pi * x)` and `cos(pi * x)`.
        fn sincos_pi[][](self: Self) -> (Self, Self);

        /// Trigonometric sine of `pi * x`, with improved precision when the policy allows.
        fn sin_pi[][](self: Self) -> Self;
        /// Trigonometric cosine of `pi * x`, with improved precision when the policy allows.
        fn cos_pi[][](self: Self) -> Self;
        /// Trigonometric tangent of `pi * x`, with improved precision when the policy allows.
        fn tan_pi[][](self: Self) -> Self;

        /// Computes `sin(x) / x` with improved precision when the policy allows.
        fn sinc[][](self: Self) -> Self;

        /// Computes `$\frac{\sin(\pi x)}{\pi x}$` with improved precision when the policy allows.
        fn sinc_pi[][](self: Self) -> Self;

        /// Hyperbolic sine and cosine, together. This will be more efficient than calling `sinh` and `cosh` separately.
        fn sinh_cosh[][](self: Self) -> (Self, Self);
        /// Hyperbolic sine
        fn sinh[][](self: Self) -> Self;
        /// Hyperbolic cosine
        fn cosh[][](self: Self) -> Self;
        /// Hyperbolic tangent
        fn tanh[][](self: Self) -> Self;
        /// Returns the arcsine of `self`.
        fn asin[][](self: Self) -> Self;
        /// Returns the arccosine of `self`.
        fn acos[][](self: Self) -> Self;
        /// Returns the arctangent of `self`.
        fn atan[][](self: Self) -> Self;

        /// Inverse hyperbolic sine
        fn asinh[][](self: Self) -> Self;
        /// Inverse hyperbolic cosine
        fn acosh[][](self: Self) -> Self;
        /// Inverse hyperbolic tangent
        fn atanh[][](self: Self) -> Self;
        /// The exponential function, returns `e^(self)`.
        fn exp[][](self: Self) -> Self;
        /// The Half exponential function, returns `0.5 * e^(self)`.
        fn exph[][](self: Self) -> Self;
        /// The base-2 exponential function, returns `2^(self)`.
        fn exp2[][](self: Self) -> Self;
        /// The base-10 exponential function, returns `10^(self)`.
        fn exp10[][](self: Self) -> Self;
        /// Returns `exp(self) - 1` of `self`, which is more precise than calculating `exp(self) - 1` directly.
        fn exp_m1[][](self: Self) -> Self;
        /// Returns `2^(self) - 1`, which is more precise than calculating `exp2(self) - 1` directly.
        fn exp2_m1[][](self: Self) -> Self;
        /// Returns `10^(self) - 1`, which is more precise than calculating `exp10(self) - 1` directly.
        fn exp10_m1[][](self: Self) -> Self;
        /// Returns `$\sqrt{1 + x} - 1$` of `self`, which is more precise than `sqrt(1 + x) - 1` directly near zero.
        ///
        /// Evaluated as `$\frac{x}{\sqrt{1 + x} + 1}$`, which has no cancellation near `x = 0`.
        fn sqrt1pm1[][](self: Self) -> Self;
        /// Returns `self` raised to the power of `e`.
        fn powf[][](self: Self, e: Self) -> Self;
        /// Returns `$x^e - 1$` where `x = self`, computed accurately as `$e^{e \ln(x)}$`-style `expm1`.
        ///
        /// More precise than `powf(x, e) - 1` when the result is near zero (i.e. `x` near 1 or `e` near 0),
        /// e.g. compound returns/growth rates.
        fn powf_m1[][](self: Self, e: Self) -> Self;
        /// Returns `$(1 + x)^n$` where `x = self`, computed accurately near `x = 0` as `$e^{n \ln(1 + x)}$`.
        ///
        /// This is the IEEE 754 `compound` operation, and is more precise than `powf(1 + x, n)` for small `x`
        /// (e.g. compound-growth/interest over `n` periods at rate `x`).
        fn compound[][](self: Self, n: Self) -> Self;
        /// Returns the cube root of `self`.
        fn cbrt[][](self: Self) -> Self;
        /// Returns the Nth root of `self`.
        ///
        /// This is often faster _and_ more accurate than using `powf(1.0 / N as float)`. Supports
        /// negative numbers for odd N.
        fn nth_root[const N: usize][N](self: Self) -> Self;
        /// Returns the natural logarithm of `self`.
        fn ln[][](self: Self) -> Self;
        /// Returns `$\ln(1 + x)$` of `self`.
        fn ln_1p[][](self: Self) -> Self;
        /// Returns the base-2 logarithm of `self`.
        fn log2[][](self: Self) -> Self;
        /// Returns the base-10 logarithm of `self`.
        fn log10[][](self: Self) -> Self;
        /// Returns `$\log_2(1 + x)$` of `self`, which is more precise than `log2(1 + x)` directly near zero.
        fn log2_p1[][](self: Self) -> Self;
        /// Returns `$\log_{10}(1 + x)$` of `self`, which is more precise than `log10(1 + x)` directly near zero.
        fn log10_p1[][](self: Self) -> Self;

        /// Returns the logarithm of `self` with respect to the given `base`.
        fn log[][](self: Self, base: Self) -> Self;

        /// Returns the logarithm of `self` with respect to the given integer base `N`.
        ///
        /// This is efficient for bases <=32 using a lookup table, and falls back to the general `log(x)/libm::log(N)`
        /// implementation for larger bases.
        ///
        /// For bases 0 and 1, the result is 0 and Infinity respectively.
        fn log_n[const N: usize][N](self: Self) -> Self;

        /// Returns `$\ln(1 - e^{-x})$`, which depending on the policy may be
        /// an approximation more performant than the exact calculation. If you're using a policy with below
        /// average precision, and happen to have `ln(x)` available, you can use [`ln1m_expnx_ext`](TranscendentalMath::ln1m_expnx_ext) instead
        /// to provide that.
        fn ln1m_expnx[][](self: Self) -> Self;

        /// Returns `ln(1 - exp(lnx))`, which depending on the policy may be
        /// an approximation more performant than the exact calculation. If you're using a policy with below
        /// average precision, it's recommended to use this function instead of [`ln1m_expnx`](TranscendentalMath::ln1m_expnx) to provide `ln(x)` directly.
        ///
        /// Although not obvious, `ln(x)` is used internally for the approximation, and if it's already available,
        /// you may as well use this function to avoid recomputing it.
        fn ln1m_expnx_ext[][](self: Self, lnx: Self) -> Self;
    }

    // TODO: Create an associated type `Scalar` to return for spatial functions,
    // as Complex vectors may want to return real-valued norms/distances.

    /// Spatial mathematical functions like norms and distances.
    ///
    /// These functions are primarily useful in dimensions higher than one.
    trait Spatial<FloatElement>: CoreMathWithPolicy {
        /// Computes the Euclidean norm (hypotenuse) of `self` and `other`, i.e., `sqrt(self^2 + other^2)`.
        ///
        /// This is not higher performance than the naive implementation, but is more resistant to overflow and underflow.
        /// If using the worst precision policy, it becomes equivalent to the naive implementation.
        ///
        /// Check out [`hypot_n`](SpatialMath::hypot_n) for a more general version that computes the hypotenuse of N values.
        fn hypot[][](self: Self, other: Self) -> Self;

        /// Computes the Euclidean norm (hypotenuse) of N values, i.e., `$\sqrt{x_1^2 + x_2^2 + \dots + x_N^2}$`.
        ///
        /// This is typically higher performance than naively computing the sum of squares and then taking the square root,
        /// especially for larger N, and is more resistant to overflow and underflow when using average or higher precision policies.
        fn hypot_n[const N: usize][N](values: [Self; N]) -> Self;

        /// Computes the inverse Euclidean norm (inverse hypotenuse) of N values, i.e., `$1/\sqrt{x_1^2 + x_2^2 + \dots + x_N^2}$`.
        ///
        /// This is typically higher performance than naively computing the sum of squares, taking the square root, and then inverting,
        /// especially for larger N, and is more resistant to overflow and underflow when using average or higher precision policies.
        ///
        /// At lower precision policies, we can take advantage of fast approximate inverse square root implementations for better performance.
        fn inv_hypot_n[const N: usize][N](values: [Self; N]) -> Self;

        /// L1 Norm, or the "Manhattan" distance from the origin.
        ///
        /// For 1D vectors, this is equivalent to the absolute value.
        fn l1_norm[][](self: Self) -> Self;

        /// L2 Norm, or the "Euclidean" distance from the origin.
        ///
        /// For 1D vectors, this is equivalent to the absolute value.
        fn l2_norm[][](self: Self) -> Self;

        /// Squared L2 Norm, or the squared "Euclidean" distance from the origin.
        ///
        /// For 1D vectors, this is equivalent to squaring the value.
        fn l2_norm_squared[][](self: Self) -> Self;
    }

    /// Real-value mathematical functions that cannot be applied to some number types. (e.g., complex numbers)
    trait Real<FloatElement>: TranscendentalMathWithPolicy & SpatialMathWithPolicy {
        /// Returns the precision tolerance based on the selected policy. This is a good
        /// default tolerance to use for numerical methods.
        #[skip_dispatch] fn tolerance[][]() -> Self;

        /// Converts angles from radians to degrees.
        fn to_degrees[][](self: Self) -> Self;

        /// Converts angles from degrees to radians.
        fn to_radians[][](self: Self) -> Self;

        /// Wraps the angle (radians) in `self` to the range `[-π, π)`.
        ///
        /// The formula for this is `self - floor((self + π) / 2π) * 2π`
        fn wrap_angle[][](self: Self) -> Self;

        /// Computes the smallest difference between two angles (in radians),
        /// taking into account angle wrapping.
        ///
        /// To get the "distance" between two angles, use the absolute value of the result.
        fn angle_diff[][](self: Self, other: Self) -> Self;

        /// Returns the four-quadrant arctangent of `self` and `x`.
        ///
        /// This method is only defined for real-valued types.
        fn atan2[][](self: Self, x: Self) -> Self;

        /// Linearly interpolates between `a` and `b` based on the value of `self`.
        ///
        /// This operation is not clamped.
        fn lerp[][](self: Self, a: Self, b: Self) -> Self;

        /// Scales `self` from the input range `[in_min, in_max]` to the output range `[out_min, out_max]`.
        ///
        /// This operation is not clamped.
        fn rescale[][](self: Self, in_min: Self, in_max: Self, out_min: Self, out_max: Self) -> Self;

        /// Returns `$\ln(e^{a} + e^{b})$` computed in a numerically stable way that avoids overflow,
        /// where `a = self` and `b = other`.
        ///
        /// Evaluated as `$\max(a, b) + \ln(1 + e^{-|a - b|})$`, so the result is accurate even when `a`
        /// and `b` are large. This is the workhorse of stable log-domain probability arithmetic
        /// (e.g. the two-argument log-sum-exp).
        fn logaddexp[][](self: Self, other: Self) -> Self;

        /// Generalized smoothstep function of Order `2N-1`. Note: The "smoothness"
        /// for higher order is in terms of the number of continuous derivatives,
        /// not in terms of visual smoothness, though they are related in some ways.
        ///
        /// For N=0, this is equivalent to the step function. \
        /// For N=1, this is a linear line between 0 and 1. \
        /// For N=2, this is equivalent to the standard 3rd-order smoothstep function. \
        /// For N=3, this is equivalent to the 5th-order "smootherstep" function.
        ///
        /// For single precision, N can go up to 10, whereas for double precision, N can go up to 20.
        ///
        /// See [`smooth_interpolator`](RealMath::smooth_interpolator) for a more advanced interpolator with
        /// infinite differentiability.
        fn smoothstep[const N: usize][N](self: Self, edges: Option<(Self, Self)>) -> Self;

        /// Returns the inverse smoothstep of `self`, which is the value that would produce `self` when passed to `smoothstep`.
        ///
        /// N from 0..=2 have fast closed-form solutions, while higher N use numerical root-finding methods, which will inherently
        /// be much slower.
        fn inverse_smoothstep[const N: usize][N](self: Self, edges: Option<(Self, Self)>) -> Self;

        /// Derivative of the `smoothstep` function of order `2N-1`, at the given point.
        fn smoothstep_derivative[const N: usize][N](self: Self, edges: Option<(Self, Self)>) -> Self;

        /// C∞-smooth interpolation factor between the given edges (defaulting to 0 and 1).
        ///
        /// Constructs a smooth transition function using:
        ///
        /// ```text
        /// f(x) = e^(-1 / (k * x))
        /// g(x) = f(x) / (f(x) + f(1 - x))
        /// ```
        ///
        /// The result is C∞-differentiable (infinitely smooth), with all derivatives vanishing
        /// at both endpoints - making it strictly superior to polynomial smoothstep for
        /// applications requiring flatness at the edges.
        ///
        /// The `k` parameter controls the shape of the transition:
        /// - `k < 1`: sharpens the curve, concentrating the transition near the midpoint.
        /// - `k = 1`: the standard balanced sigmoid-like transition.
        /// - `k > 1`: stretches the transition region, making the curve more gradual.
        /// - `$k \approx 2/\sqrt{3}$` (~1.1547): the function becomes bimodal - use with caution above this value.
        fn smooth_interpolator[][](self: Self, edges: Option<(Self, Self)>, k: Self) -> Self;

        /// Inverse of [`smooth_interpolator`](crate::math::RealMath::smooth_interpolator).
        ///
        /// Given an output value `y` in `[0, 1]`, recovers the input `x` such that
        /// `smooth_interpolator(x, edges, k) ≈ y`.
        fn smooth_interpolator_inverse[][](self: Self, edges: Option<(Self, Self)>, k: Self) -> Self;

        /// Returns 1 if `self` is greater than or equal to `edge`, otherwise returns 0.
        fn step[][](self: Self, edge: Self) -> Self;
    }
}
