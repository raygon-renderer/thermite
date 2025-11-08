#![allow(clippy::excessive_precision, clippy::needless_arbitrary_self_type)]

mod consts;
pub mod policy;

pub use consts::FloatConsts;

use crate::{
    Vector,
    register::{FloatRegister, Register},
};

mod internal;

use internal::MathInternal;
use policy::{DefaultPolicy, Policy};

// Helper macro to declare math traits and implementations
// for both policy and default policy versions. This reduces
// boilerplate and ensures consistency between the two traits,
// though it is a bit annoying to read and write.
macro_rules! decl_math {
    ($(
        $(#[$meta:meta])*
        fn $name:ident [ $($generics:tt)* ][$($generic_names:ident),*]( $($arg_name:ident :$arg_ty:ty),* $(,)?) -> $ret:ty
            $(where [ $($where_clause:tt)* ])?;
    )*) => {paste::paste! {
        /// Math functions for floating-point vectors with customizable policies.
        ///
        /// This trait provides a set of mathematical operations that can be performed
        /// on floating-point vector types. Each function has a variant that accepts
        /// a policy parameter, allowing for fine-tuned control over precision and performance.
        ///
        /// For convenience, a default implementation is also provided in the [`Math`] trait,
        /// which uses the [`DefaultPolicy`]. All floating-point vector types that implement
        /// the necessary internal math operations will automatically implement this trait, and
        /// the [`Math`] trait as well for all types that implement this one.
        pub trait MathWithPolicy<R: FloatRegister>: Sized {$(
            $(#[$meta])*
            fn [<$name _p>]<P: Policy, $($generics)*>($($arg_name: $arg_ty),*) -> $ret
                $(where $($where_clause)*)?;
        )*}

        /// Math functions for floating-point vectors using the default policy.
        ///
        /// This trait provides the same set of mathematical operations as [`MathWithPolicy`],
        /// but uses the [`DefaultPolicy`] for all operations. This allows for easier usage
        /// when specific policy customization is not required.
        ///
        /// Implementors of [`MathWithPolicy`] will automatically implement this trait as well.
        ///
        /// All methods here have an associated method in [`MathWithPolicy`] with a `_p` suffix
        /// that accepts a policy parameter as the first generic argument.
        pub trait Math<R: FloatRegister>: MathWithPolicy<R> {$(
            $(#[$meta])*
            #[inline(always)] fn $name<$($generics)*>($($arg_name: $arg_ty),*) -> $ret
                $(where $($where_clause)*)?
            {
                MathWithPolicy::<R>::[<$name _p>]::<DefaultPolicy, $($generic_names),*>($($arg_name),*)
            }
        )*}

        impl<M, R: FloatRegister> Math<R> for M where M: MathWithPolicy<R> {}

        impl<E, R> MathWithPolicy<R> for Vector<R>
        where
            R: MathInternal<E, Element = E>,
            E: FloatConsts,
        {$(
            #[inline(always)] fn [<$name _p>]<P: Policy, $($generics)*>($($arg_name: $arg_ty),*) -> $ret
                $(where $($where_clause)*)?
            {
                R::$name::<P, $($generic_names),*>($($arg_name),*)
            }
        )*}
    }};
}

decl_math! {
    /// Returns the precision tolerance based on the selected policy. This is a good
    /// default tolerance to use for numerical methods.
    fn tolerance[][]() -> Self;

    /// Computes the polynomial with the given coefficients at `self`.
    fn poly[const N: usize][N](self: Self, coeffs: &[R::Element; N]) -> Self;

    /// Computes the polynomial with the given coefficients at `self`, but with the coefficients in reverse order.
    fn poly_rev[const N: usize][N](self: Self, coeffs: &[R::Element; N]) -> Self;

    /// Computes the ratio of two polynomials at `self`, given the numerator and denominator coefficients.
    ///
    /// Equivalent to `poly(numerator) / poly(denominator)`, but with improved numerical stability in some cases.
    fn poly_rational[const N: usize, const D: usize][N, D](
        self: Self,
        numerator: &[R::Element; N],
        denominator: &[R::Element; D],
    ) -> Self;

    /// Computes the sum of `f(i)` for `i` in the range `[start, end)`.
    ///
    /// Returns `Ok(sum)` if the computation converged within the policy's
    /// maximum iterations and precision tolerance, otherwise returns `Err(partial_sum)`.
    ///
    /// If the precision policy is set to `Best` or higher, Kahan summation is used to improve accuracy.
    fn sum_f[F][F](start: i64, end: i64, f: F) -> Result<Self, Self> where [F: FnMut(i64) -> Self];

    /// Computes the product of `f(i)` for `i` in the range `[start, end)`.
    ///
    /// Returns `Ok(product)` if the computation converged within the policy's
    /// maximum iterations and precision tolerance, otherwise returns `Err(partial_product)`.
    fn prod_f[F][F](start: i64, end: i64, f: F) -> Result<Self, Self> where [F: FnMut(i64) -> Self];

    /// Newton's method for finding roots of a function.
    ///
    /// `f` should return a tuple of `(f(x), f'(x))`, the function value and its derivative at `x`. The
    /// value of `self` is used as the initial guess.
    ///
    /// The iteration continues until the change is within `tolerance` or the maximum iterations
    /// defined by the policy is reached. If `bounds` are provided, the result will be clamped within those bounds
    /// to prevent explosive divergence.
    ///
    /// Returns the estimated root, which may not be accurate if the method did not converge.
    fn newtons_method[F][F](self: Self, tolerance: Self, bounds: Option<(Self, Self)>, f: F) -> Self
        where [F: FnMut(Self) -> (Self, Self)];

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
    /// See [`smooth_interpolator`](Math::smooth_interpolator) for a more advanced interpolator with
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

    /// Linearly interpolates between `a` and `b` based on the value of `self`.
    fn lerp[][](self: Self, a: Self, b: Self) -> Self;

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
    fn powiv[][](self: Self, e: Vector<R::Signed>) -> Self;

    /// Computes the Euclidean norm (hypotenuse) of `self` and `other`, i.e., `sqrt(self^2 + other^2)`.
    ///
    /// This is not higher performance than the naive implementation, but is more resistant to overflow and underflow.
    /// If using the worst precision policy, it becomes equivalent to the naive implementation.
    fn hypot[][](self: Self, other: Self) -> Self;

    /// Trigonometric sine and cosine, together. This may be more efficient than calling `sin` and `cos` separately.
    fn sincos[][](self: Self) -> (Self, Self);
    /// Trigonometric sine
    fn sin[][](self: Self) -> Self;
    /// Trigonometric cosine
    fn cos[][](self: Self) -> Self;
    /// Trigonometric tangent
    fn tan[][](self: Self) -> Self;
    /// Computes `sin(pi * x) / (pi * x)` with improved precision when the policy allows.
    fn sinc[][](self: Self) -> Self;

    /// Computes `x * sin(pi * x)` with improved precision when the policy allows.
    fn sin_pix[][](self: Self) -> Self;

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
    fn ln1p[][](self: Self) -> Self;
    /// Returns the base-2 logarithm of `self`.
    fn log2[][](self: Self) -> Self;
    /// Returns the base-10 logarithm of `self`.
    fn log10[][](self: Self) -> Self;

    /// Returns `ln(1 - exp(-x))`, which depending on the policy may be
    /// an approximation more performant than the exact calculation. If you're using a policy with below
    /// average precision, and happen to have `ln(x)` available, you can use [`ln1m_expnx_ext`](Math::ln1m_expnx_ext) instead
    /// to provide that.
    fn ln1m_expnx[][](self: Self) -> Self;
    fn ln1m_expnx_ext[][](self: Self, lnx: Self) -> Self;
    /// Computes the error function.
    fn erf[][](self: Self) -> Self;
    /// Computes the complementary error function.
    fn erfc[][](self: Self) -> Self;
    /// Computes the inverse error function.
    fn erfinv[][](self: Self) -> Self;
    fn gaussian[][](self: Self, a: Self, c: Self) -> Self;
    fn gaussian_integral[][](x0: Self, x1: Self, a: Self, c: Self) -> Self;
}

impl<E, R> num_traits::Inv for Vector<R>
where
    R: MathInternal<E, Element = E>,
    E: FloatConsts,
{
    type Output = Self;

    /// Returns the multiplicative inverse of the vector,
    /// by calling [`Math::reciprocal`].
    #[inline(always)]
    fn inv(self) -> Self {
        self.reciprocal()
    }
}
