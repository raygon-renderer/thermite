#![allow(clippy::needless_arbitrary_self_type)]

mod consts;
pub mod policy;

pub use consts::FloatConsts;

use crate::vector::generic::FloatVector;

pub mod algorithms;
pub mod specialized;

use policy::{DefaultPolicy, Policy};

// Helper macro to declare math traits and implementations
// for both policy and default policy versions. This reduces
// boilerplate and ensures consistency between the two traits,
// though it is a bit annoying to read and write.
macro_rules! decl_math {
    (
        $(#[$trait_meta:meta])*
        trait $trait:ident $(: $($bound:ident)&+ )? { $(
            $(#[$meta:meta])*
            fn $name:ident [ $($generics:tt)* ][$($generic_names:ident),*]( $($arg_name:ident :$arg_ty:ty),* $(,)?) -> $ret:ty
                $(where [ $($where_clause:tt)* ])?;
            )*
        }
    ) => {paste::paste! {
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
        pub trait [<$trait Math>]: [<$trait MathWithPolicy>] {$(
            $(#[$meta])* #[inline(always)] fn $name<$($generics)*>($($arg_name: $arg_ty),*) -> $ret
                $(where $($where_clause)*)?
            {
                [<$trait MathWithPolicy>]::[<$name _p>]::<DefaultPolicy, $($generic_names),*>($($arg_name),*)
            }
        )*}

        impl<M> [<$trait Math>] for M where M: [<$trait MathWithPolicy>] {}

        // Note: The FloatVector<Element = E> bound is necessary to ensure E is bounded.
        impl<E, V: FloatVector<Element = E> + $($($bound +)+)?> [<$trait MathWithPolicy>] for V
            where V: specialized::[<Specialized $trait Math>]<E>
        {$(
            #[inline(always)] fn [<$name _p>]<P: Policy, $($generics)*>($($arg_name: $arg_ty),*) -> $ret
                $(where $($where_clause)*)?
            {
                <V as specialized::[<Specialized $trait Math>]<E>>::$name::<P, $($generic_names),*>($($arg_name),*)
            }
        )*}
    }};
}

decl_math! {
    /// This is the core set of mathematical operations that form the basis for more advanced functions.
    trait Core: FloatVector {
        /// Computes `self * 2^exp` efficiently.
        fn ldexp[][](self: Self, exp: Self::Signed) -> Self;

        /// Decomposes `self` into its normalized fraction and an integral power of two.
        fn frexp[][](self: Self) -> (Self, Self::Signed);

        /// Returns the precision tolerance based on the selected policy. This is a good
        /// default tolerance to use for numerical methods.
        fn tolerance[][]() -> Self;

        /// Computes the polynomial with the given coefficients at `self`.
        ///
        /// This will use fused multiply-add instructions where available for improved performance and accuracy, but
        /// falls back to standard operations if not.
        fn poly[const N: usize][N](self: Self, coeffs: &[Self::Element; N]) -> Self;

        /// Computes the polynomial with the given coefficients at `self`, but with the coefficients in reverse order.
        ///
        /// This will use fused multiply-add instructions where available for improved performance and accuracy, but
        /// falls back to standard operations if not.
        fn poly_rev[const N: usize][N](self: Self, coeffs: &[Self::Element; N]) -> Self;

        /// Computes the ratio of two polynomials at `self`, given the numerator and denominator coefficients.
        ///
        /// Equivalent to `poly(numerator) / poly(denominator)`, but with improved numerical stability in some cases.
        ///
        /// This will use fused multiply-add instructions where available for improved performance and accuracy, but
        /// falls back to standard operations if not.
        fn poly_rational[const N: usize, const D: usize][N, D](
            self: Self,
            numerator: &[Self::Element; N],
            denominator: &[Self::Element; D],
        ) -> Self;

        /// Linearly interpolates between `a` and `b` based on the value of `self`.
        ///
        /// This operation is not clamped.
        fn lerp[][](self: Self, a: Self, b: Self) -> Self;

        /// Scales `self` from the input range `[in_min, in_max]` to the output range `[out_min, out_max]`.
        ///
        /// This operation is not clamped.
        fn scale[][](self: Self, in_min: Self, in_max: Self, out_min: Self, out_max: Self) -> Self;

        /// Returns the multiplicative inverse of `self`, which is `1 / self`.
        ///
        /// If using the policy version, you may select lower precision policies for extra performance,
        /// at the cost of accuracy.
        fn reciprocal[][](self: Self) -> Self;

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
}

decl_math! {
    /// Transcendental mathematical functions like trigonometric, exponential, and logarithmic functions.
    trait Transcendental: CoreMathWithPolicy {
        /// Trigonometric sine and cosine, together. This will be more efficient than calling `sin` and `cos` separately.
        fn sin_cos[][](self: Self) -> (Self, Self);
        /// Trigonometric sine
        fn sin[][](self: Self) -> Self;
        /// Trigonometric cosine
        fn cos[][](self: Self) -> Self;
        /// Trigonometric tangent
        fn tan[][](self: Self) -> Self;

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

        /// Computes `sin(pi * x) / (pi * x)` with improved precision when the policy allows.
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
        /// Returns the four-quadrant arctangent of `self` and `x`.
        fn atan2[][](self: Self, x: Self) -> Self;
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
        /// Returns `self` raised to the power of `e`.
        fn powf[][](self: Self, e: Self) -> Self;
        /// Returns the cube root of `self`.
        fn cbrt[][](self: Self) -> Self;
        /// Returns the natural logarithm of `self`.
        fn ln[][](self: Self) -> Self;
        /// Returns `ln(1 + x)` of `self`.
        fn ln_1p[][](self: Self) -> Self;
        /// Returns the base-2 logarithm of `self`.
        fn log2[][](self: Self) -> Self;
        /// Returns the base-10 logarithm of `self`.
        fn log10[][](self: Self) -> Self;

        /// Returns the logarithm of `self` with respect to the given `base`.
        fn log[][](self: Self, base: Self) -> Self;

        /// Returns the logarithm of `self` with respect to the given integer base `N`.
        ///
        /// This is efficient for bases <=32 using a lookup table, and falls back to the general `log(x)/libm::log(N)`
        /// implementation for larger bases.
        ///
        /// For bases 0 and 1, the result is 0 and Infinity respectively.
        fn log_n[const N: usize][N](self: Self) -> Self;

        /// Returns `ln(1 - exp(-x))`, which depending on the policy may be
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
}

decl_math! {
    /// Spatial mathematical functions like norms and distances.
    trait Spatial: CoreMathWithPolicy {
        /// Computes the Euclidean norm (hypotenuse) of `self` and `other`, i.e., `sqrt(self^2 + other^2)`.
        ///
        /// This is not higher performance than the naive implementation, but is more resistant to overflow and underflow.
        /// If using the worst precision policy, it becomes equivalent to the naive implementation.
        fn hypot[][](self: Self, other: Self) -> Self;

        fn l1_norm[][](self: Self) -> Self;
        fn l2_norm[][](self: Self) -> Self;

        fn l2_norm_squared[][](self: Self) -> Self;
    }
}

decl_math! {
    /// Real-value mathematical functions that cannot be applied to some number types. (e.g., complex numbers)
    trait Real: TranscendentalMathWithPolicy & SpatialMathWithPolicy {
        /// Converts angles from radians to degrees.
        fn to_degrees[][](self: Self) -> Self;

        /// Converts angles from degrees to radians.
        fn to_radians[][](self: Self) -> Self;

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
        fn inverse_smoothstep[const N: usize][N](self: Self, edges: Option<(Self, Self)>) -> Self;

        /// Derivative of the `smoothstep` function of order `2N-1`, at the given point.
        fn smoothstep_derivative[const N: usize][N](self: Self, edges: Option<(Self, Self)>) -> Self;

        /// Smoothly interpolates between the given edges, which default to 0 and 1 if not provided, with infinite differentiability.
        ///
        /// This is a more advanced version of `smoothstep` that provides a mathematically smoother transition, C-infinitely differentiable.
        fn smooth_interpolator[][](self: Self, edges: Option<(Self, Self)>, k: Self) -> Self;

        fn smooth_interpolator_inverse[][](self: Self, edges: Option<(Self, Self)>, k: Self) -> Self;

        /// Returns 1 if `self` is greater than or equal to `edge`, otherwise returns 0.
        fn step[][](self: Self, edge: Self) -> Self;
    }
}
