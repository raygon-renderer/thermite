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

use generic_array::{ArrayLength, GenericArray};

pub mod algorithms;
pub mod specialized;

use policy::{DefaultPolicy, Policy};

pub mod prelude {
    pub use crate::vector::FloatVector;

    pub use super::FloatConsts;
    pub use super::PrimalProjection;
    pub use super::{
        CoreMath, CoreMathWithPolicy, PrimalMath, PrimalMathWithPolicy, RealMath, RealMathWithPolicy, ScalarMath,
        ScalarMathWithPolicy, SpatialMath, SpatialMathWithPolicy, TranscendentalMath, TranscendentalMathWithPolicy,
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
        // Each supertrait bound may carry associated-type bindings, parsed
        // structurally (`Bound<Name = Ty, ...>`) because a tt-repetition cannot
        // terminate at a closing angle bracket. This is how `PrimalMath` states
        // `PrimalProjection<Primal = Self>` directly in supertrait position.
        trait $trait:ident<$element:ident> $(: $( $bound:ident $(< $($bound_assoc:ident = $bound_ty:ty),+ >)? )&+ )? {
            // Optional policy-driven methods that exist on vectors only. Same `_p` and
            // default-policy treatment as the ordinary items below, but no `scalar_`
            // form: their signatures mention associated types (`Self::Primal` from the
            // `PrimalProjection` supertrait) that a bare `f32` does not have. (Nor
            // would one be useful, since a scalar is its own primal, so the scalar
            // form would duplicate the plain method exactly.)
            $(vector_fns {
                $(
                    $(#[$vfn_meta:meta])*
                    fn $vfn_name:ident [ $($vfn_generics:tt)* ][$($vfn_generic_names:ident),*]( $($vfn_arg:ident : $vfn_ty:ty),* $(,)?) -> $vfn_ret:ty;
                )*
            })?
            $(
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
        pub trait [<$trait MathWithPolicy>] $(: $($bound $(< $($bound_assoc = $bound_ty),+ >)? +)+)? {
            $($(
                $(#[$vfn_meta])* fn [<$vfn_name _p>]<P: Policy, $($vfn_generics)*>($($vfn_arg: $vfn_ty),*) -> $vfn_ret;
            )*)?
            $(
                $(#[$meta])* fn [<$name _p>]<P: Policy, $($generics)*>($($arg_name: $arg_ty),*) -> $ret
                    $(where $($where_clause)*)?;
            )*
        }

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
        pub trait [<$trait Math>]: [<$trait MathWithPolicy>] {
            $($(
                // `<Self as ...>` explicitly: an argument typed through an associated
                // type cannot drive `Self` inference, since the projection is not
                // injective.
                $(#[$vfn_meta])* #[inline(always)] fn $vfn_name<$($vfn_generics)*>($($vfn_arg: $vfn_ty),*) -> $vfn_ret
                { <Self as [<$trait MathWithPolicy>]>::[<$vfn_name _p>]::<DefaultPolicy, $($vfn_generic_names),*>($($vfn_arg),*) }
            )*)?
            $(
                $(#[$meta])* #[inline(always)] fn $name<$($generics)*>($($arg_name: $arg_ty),*) -> $ret
                    $(where $($where_clause)*)?
                { [<$trait MathWithPolicy>]::[<$name _p>]::<DefaultPolicy, $($generic_names),*>($($arg_name),*) }
            )*
        }

        impl<M> [<$trait Math>] for M where M: [<$trait MathWithPolicy>] {}

        // Note: The FloatVector<Element = E> bound is necessary to ensure E is bounded.
        #[thermite_macros::dispatch(Self, thermite = "crate")]
        impl<E: $element, V: FloatVector<Element = E> + $($($bound $(< $($bound_assoc = $bound_ty),+ >)? +)+)?> [<$trait MathWithPolicy>] for V
            where V: specialized::[<Specialized $trait Math>]<E>
        {
            $($(
                $(#[$vfn_meta])* #[skip_dispatch] #[inline(always)]
                fn [<$vfn_name _p>]<P: Policy, $($vfn_generics)*>($($vfn_arg: $vfn_ty),*) -> $vfn_ret
                { <V as specialized::[<Specialized $trait Math>]<E>>::$vfn_name::<P, $($vfn_generic_names),*>($($vfn_arg),*) }
            )*)?
            $(
                #[cfg(not(feature = "disable_dispatch"))]
                $(#[$meta])* #[inline(always)] fn [<$name _p>]<P: Policy, $($generics)*>($($arg_name: $arg_ty),*) -> $ret
                    $(where $($where_clause)*)?
                { <V as specialized::[<Specialized $trait Math>]<E>>::$name::<P, $($generic_names),*>($($arg_name),*) }

                #[cfg(feature = "disable_dispatch")]
                $(#[$meta])* #[skip_dispatch] #[inline(always)] fn [<$name _p>]<P: Policy, $($generics)*>($($arg_name: $arg_ty),*) -> $ret
                    $(where $($where_clause)*)?
                { <V as specialized::[<Specialized $trait Math>]<E>>::$name::<P, $($generic_names),*>($($arg_name),*) }
            )*
        })*

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
        #[diagnostic::on_unimplemented(
            message = "`{Self}` is not a bare floating-point scalar",
            note = "`ScalarMathWithPolicy` is implemented only for the bare scalar types `f32` and `f64`. For SIMD vectors, bound on `FloatVector` plus the vector math traits (`CoreMath`, `TranscendentalMath`, ...) instead."
        )]
        pub trait ScalarMathWithPolicy: ElementExt<Element = Self> + FloatElementWithBits {$($(
            $(#[$meta])* fn [<scalar_ $name _p>]<P: Policy, $($generics)*>($($arg_name: $arg_ty),*) -> $ret
                $(where $($where_clause)*)?;
        )*)*

            /// Square root of `self`.
            ///
            /// Unlike the rest of this trait, `sqrt` is not a policy-driven approximation - it is
            /// exact (correctly rounded) on all supported formats, so the policy is ignored. This
            /// exists only so scalar code can spell it the same way as the other `scalar_` methods;
            /// on vectors, `sqrt` is an inherent [`FloatVector`] method rather than a math trait one.
            #[inline(always)]
            fn scalar_sqrt_p<P: Policy>(self) -> Self {
                FloatElement::sqrt(self)
            }
        }

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
        #[diagnostic::on_unimplemented(
            message = "`{Self}` is not a bare floating-point scalar",
            note = "`ScalarMath` is implemented only for the bare scalar types `f32` and `f64`. For SIMD vectors, bound on `FloatVector` plus the vector math traits (`CoreMath`, `TranscendentalMath`, ...) instead."
        )]
        pub trait ScalarMath: ScalarMathWithPolicy {$($(
            $(#[$meta])* #[inline(always)] fn [<scalar_ $name>]<$($generics)*>($($arg_name: $arg_ty),*) -> $ret
                $(where $($where_clause)*)?
            { ScalarMathWithPolicy::[<scalar_ $name _p>]::<DefaultPolicy, $($generic_names),*>($($arg_name),*) }
        )*)*

            /// Square root of `self`.
            ///
            /// See [`ScalarMathWithPolicy::scalar_sqrt_p`]. `sqrt` is exact, so this is simply the
            /// scalar spelling of the inherent [`FloatVector::sqrt`] vector method.
            #[inline(always)]
            fn scalar_sqrt(self) -> Self {
                ScalarMathWithPolicy::scalar_sqrt_p::<DefaultPolicy>(self)
            }
        }

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

        assert_eq!(4.0f32.scalar_sqrt(), 2.0);
        assert_eq!(4.0f64.scalar_sqrt(), 2.0);
    }
}

/// Projection from a float vector down to its unaugmented "primal" value type.
///
/// [`Primal`](Self::Primal) is `Self` for plain vectors, and the (recursive)
/// primal of the inner value vector for composites like `Dual` or `Complex`.
/// It is the natural type for precomputed constants and coefficient tables: a
/// constant's derivative and imaginary parts are identically zero, so storing
/// them wastes lanes. Generate and cache tables in `Self::Primal`, and let the
/// composite's kernels lift entries as needed.
///
/// `Compensated` is deliberately its own primal: the error half of a
/// double-double constant carries real precision, not augmentation.
///
/// This trait is the _single_ owner of the `Primal` projection. Both the
/// public math traits ([`CoreMathWithPolicy`]) and the specialized backing
/// traits ([`specialized::SpecializedCoreMath`]) inherit it, so `Self::Primal`
/// means the same thing on either side by construction.
///
/// The target of the projection must itself be a primal type with the full
/// real math suite ([`PrimalMath`]); `PrimalMath` in turn states the fixpoint
/// (`PrimalProjection<Primal = Self>`), so a primal type is always its own
/// primal and the projection collapses in one step.
#[diagnostic::on_unimplemented(
    message = "`{Self}` does not declare a primal (unaugmented) value type",
    label = "needs an `impl PrimalProjection for {Self}`",
    note = "Every type carrying the math traits has to answer this, because coefficient tables are stored in `Self::Primal`. There are two answers. A wide composite, one that adds fields which are zero or degenerate for a constant (`Dual`'s derivatives, `Complex`'s imaginary part, `Interval`'s width), projects to its inner vector's primal: `type Primal = V::Primal`, recursively, so towers collapse in one step. A deep composite, one that carries the same value to more precision (`Compensated`), is its own primal: `type Primal = Self`, since dropping the extra digits would throw away the thing the type exists for.",
    note = "If `{Self}` is a type parameter that is always its own primal, this is probably the projection-shadowing trap rather than a missing impl. The rigid `PrimalProjection` supertrait shadows the fixpoint blanket impl, so `{Self}::Primal` will not normalize to `{Self}` until you add an explicit `PrimalProjection<Primal = {Self}>` bound."
)]
pub trait PrimalProjection: Sized {
    /// The unaugmented value type of `Self`. See the trait docs.
    type Primal: PrimalMath;

    /// Embeds a primal value as a constant of `Self`: every non-primal field
    /// (derivative parts, imaginary part) is initialized to zero. The identity
    /// for primal types.
    ///
    /// This is the load path for tables stored in [`Primal`](Self::Primal) form.
    fn from_primal(p: Self::Primal) -> Self;

    /// Projects `self` down to its primal value, discarding every non-primal
    /// field. The identity for primal types.
    ///
    /// Lossy by design: for `Dual` this drops the derivatives, for `Complex`
    /// the imaginary part.
    fn to_primal(self) -> Self::Primal;
}

// Anything declared primal is its own primal projection: keying the blanket on
// `SpecializedPrimalMath` (the "is primal" marker, implemented for real
// `f32`/`f64` vectors and `Compensated`) covers every fixpoint type in one impl
// and is exactly the assumption needed to prove `Self: PrimalMath` for the
// associated-type bound. Wide composites (`Dual`, `Complex`) deliberately do
// not implement `SpecializedPrimalMath` and provide their own projections.
impl<E: FloatElement, V: FloatVector<Element = E> + specialized::SpecializedPrimalMath<E>> PrimalProjection for V {
    type Primal = Self;

    #[inline(always)]
    fn from_primal(p: Self::Primal) -> Self {
        p
    }

    #[inline(always)]
    fn to_primal(self) -> Self::Primal {
        self
    }
}

decl_math! {
    /// Float-specific mathematical functions like `ldexp` and `frexp`.
    #[diagnostic::on_unimplemented(
        message = "`{Self}` does not provide float bit-level math (`ldexp`, `frexp`, `flush_denormals`)",
        note = "This trait is auto-implemented for every `FloatVectorWithBits` (concrete float vectors such as `Vector<f32>` / `f32xN`). A bare `f32`/`f64` has to be wrapped in `Vector::<f32>::splat(x)`. For scalar math use `ScalarMath` instead."
    )]
    trait Float<FloatElementWithBits>: FloatVectorWithBits {
        /// Computes `self * 2^exp` efficiently.
        ///
        /// The default policy handles the full domain: overflow gives a signed
        /// infinity, underflow a signed zero (or a subnormal under a
        /// `Preserve` denormal policy), and infinities/NaNs pass through. That
        /// costs a handful of compares and selects around the exponent
        /// arithmetic.
        ///
        /// A caller whose exponent is known to stay in range (anything fed by
        /// `frexp`, for instance) can drop all of it with
        /// `ldexp_p::<CheckOverflow<P, false>>(exp)`, leaving an add, a shift
        /// and an or. Out-of-domain inputs are then garbage in, garbage out.
        fn ldexp[][](self: Self, exp: Self::SignedBits) -> Self;

        /// Decomposes `self` into its normalized fraction and an integral power of two.
        ///
        /// `self == frac * 2^exp` with `0.5 <= |frac| < 1`; `+-0` gives
        /// `(+-0, 0)`, and infinities and NaNs pass through unchanged.
        ///
        /// Unless the [`DenormalBehavior`](policy::DenormalBehavior) is set to `Ignore`,
        /// denormal/subnormal values are properly handled regardless, not flushed. Mixed
        /// workloads of normal and denormal values will be slower than all-similar workloads
        /// due to branch prediction misprediction. This was the fastest approach overall.
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
    #[diagnostic::on_unimplemented(
        message = "`{Self}` does not provide core polynomial math (`poly`, `poly_rev`, ...)",
        note = "The math traits are auto-implemented for every float vector (any `FloatVector` whose element is `f32`/`f64`) and for composite float types. A bare `f32`/`f64` does not qualify. Wrap it in `Vector::<f32>::splat(x)`, or use `ScalarMath`'s `scalar_`-prefixed methods."
    )]
    trait Core<FloatElement>: FloatVector & PrimalProjection {
        vector_fns {
            /// [`poly`](Self::poly) with the coefficients held in [`Primal`](Self::Primal) form.
            ///
            /// The augmented fields of a constant (a `Dual`'s derivatives, a `Complex`'s
            /// imaginary part) are identically zero, so carrying coefficients in `Self`
            /// stores those zeros and then adds them at every Horner step. Neither the
            /// storage nor the addition can be optimized away: `x + 0.0` is not `x` when
            /// `x` is `-0.0`, so the adds survive to run time.
            ///
            /// Taking them as `Self::Primal` removes both. The coefficient array shrinks by
            /// the augmentation factor (4x for `Dual<V, 3>`, 2x for `Complex`), and each
            /// Horner step adds to the primal component alone.
            ///
            /// For a type that is its own primal this is exactly [`poly`](Self::poly) with
            /// pre-splatted coefficients, and the default impl reduces to it.
            ///
            /// Coefficients are _vectors_, not elements: a caller with a constant table has
            /// usually splatted it once already, and the composites that benefit most are
            /// the ones for which splatting per call would be the expensive part.
            ///
            /// The length is a [`typenum`](generic_array::typenum) length rather than a
            /// `const N: usize` so that a coefficient table can be supplied by a type that
            /// knows its own length only as an associated type. Literal call sites spell it
            /// [`GenericArray::from_array`].
            fn poly_primal[N: ArrayLength][N](self: Self, coeffs: &GenericArray<Self::Primal, N>) -> Self;

            /// [`poly_rev`](Self::poly_rev) with the coefficients held in [`Primal`](Self::Primal) form.
            ///
            /// Same trade as [`poly_primal`](Self::poly_primal) (the constants carry no
            /// augmented fields to store or add), with the coefficients in reverse order.
            fn poly_rev_primal[N: ArrayLength][N](self: Self, coeffs: &GenericArray<Self::Primal, N>) -> Self;
        }

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
    #[diagnostic::on_unimplemented(
        message = "`{Self}` does not provide transcendental math (`sin`, `cos`, `exp`, `ln`, `powf`, ...)",
        label = "no transcendental math",
        note = "This trait is auto-implemented for every float vector (any `FloatVector` whose element is `f32`/`f64`) and for composite float types (`Dual`, `Complex`, `Compensated`). A bare `f32`/`f64` does not qualify. Wrap it in `Vector::<f32>::splat(x)`, or use `ScalarMath`'s `scalar_`-prefixed methods (`x.scalar_exp()`, ...).",
        note = "If `{Self}` already is a `FloatVector` and only the method call fails to resolve, bring the trait into scope: `use thermite::math::TranscendentalMath;`."
    )]
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

        /// Returns `$\frac{1 - \cos(x)}{x^2}$`, finite at `x = 0` where it takes the value `1/2`.
        ///
        /// The `$x^2$` denominator is the one worth naming: `$\frac{1-\cos x}{x}$` is simply zero at the
        /// origin and carries no removable singularity, while this ratio tends to `1/2` and is what
        /// actually appears in practice.
        ///
        /// Written directly, `$1 - \cos x$` has already lost half the mantissa by `x` of order `1e-4`.
        /// Evaluated here as `$\tfrac{1}{2}\,\mathrm{sinc}^2(x/2)$`, an exact identity that needs no series
        /// and no cutoff, and inherits [`sinc`](TranscendentalMath::sinc)'s behaviour at the origin.
        ///
        /// This is the second Rodrigues coefficient of the `SO(3)` exponential map, alongside
        /// [`sinc`](TranscendentalMath::sinc) as the first. Rigid-body and Lie-group integrators, IMU
        /// preintegration, and skinning all evaluate it once per timestep. The prevailing practice is a
        /// hand-rolled Taylor cutoff with an arbitrary epsilon.
        fn versinc[][](self: Self) -> Self;

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

        /// Returns `$\sqrt{1 - e^{-x}}$` for `x >= 0`, without the cancellation of the direct form.
        ///
        /// `$1 - e^{-x}$` annihilates for small `x`, so this is evaluated as
        /// `$\sqrt{-\mathrm{expm1}(-x)}$`, which is accurate all the way down. Negative `x` is
        /// outside the domain and gives NaN.
        ///
        /// This is the noise scaling of an exactly-integrated Ornstein-Uhlenbeck step: Langevin and
        /// Bussi-Parrinello thermostats, and the variance-preserving schedules used by diffusion models.
        fn sqrt1mexp[][](self: Self) -> Self;
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
        ///
        /// # Examples
        ///
        /// Every math function takes a precision policy via its `_p` variant; a quick
        /// sweep against a scalar reference is the cheapest way to validate that a
        /// policy choice is accurate enough for your domain:
        ///
        /// ```
        /// use thermite::prelude::*;
        /// use thermite::math::policy::policies::Precision;
        ///
        /// type V = Vector<f64>;
        ///
        /// let mut max_err = 0.0f64;
        /// for i in 1..=1000 {
        ///     let x = i as f64 * 0.05;
        ///     let y = V::splat(x).ln_p::<Precision>().extract::<0>();
        ///     max_err = max_err.max((y - x.ln()).abs() / x.ln().abs().max(1.0));
        /// }
        /// assert!(max_err < 1e-14, "max relative error {max_err}");
        /// ```
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
    #[diagnostic::on_unimplemented(
        message = "`{Self}` does not provide spatial math (`hypot`, `atan2`, ...)",
        note = "This trait is auto-implemented for every float vector (any `FloatVector` whose element is `f32`/`f64`) and for composite float types. A bare `f32`/`f64` does not qualify. Wrap it in `Vector::<f32>::splat(x)`, or use `ScalarMath`'s `scalar_`-prefixed methods."
    )]
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
    #[diagnostic::on_unimplemented(
        message = "`{Self}` does not provide real-valued math (`to_degrees`, `to_radians`, `tolerance`, ...)",
        note = "`RealMath` builds on both `TranscendentalMath` and `SpatialMath`, and is only meaningful for real-valued float vectors. Number types like `Complex` deliberately do not implement it. A bare `f32`/`f64` does not qualify either. Wrap it in `Vector::<f32>::splat(x)`, or use `ScalarMath`."
    )]
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

        /// The logarithmic mean `$L(x, y) = \frac{x - y}{\ln x - \ln y}$`, for positive `x` and `y`.
        ///
        /// Sits between the geometric and arithmetic means, and is the mean that arises whenever a
        /// quantity varies exponentially across an interval, the log-mean temperature difference of a
        /// heat exchanger being the standard example.
        ///
        /// The defining form cancels in _both_ the numerator and the denominator as `x` approaches `y`,
        /// which is the common case rather than a corner. Evaluated here as
        /// `$\frac{x - y}{2\,\mathrm{atanh}\!\left(\frac{x-y}{x+y}\right)}$`, which is stable
        /// throughout. For nearby arguments the subtraction is exact by Sterbenz's lemma and `atanh`
        /// is accurate near zero. Equal arguments return `x`, the limiting value.
        fn logmean[][](self: Self, other: Self) -> Self;

        /// Returns `$\ln\left(\sum_{i} e^{x_i}\right)$` over `N` values, computed in a
        /// numerically stable way that avoids overflow.
        ///
        /// The N-ary [`logaddexp`](RealMath::logaddexp): normalizing a set of log-weights,
        /// the denominator of a log-softmax, the forward pass of an HMM. The largest term is
        /// factored out first, so no intermediate exponential can overflow whatever the
        /// inputs are.
        ///
        /// `N = 0` gives `-inf`, the empty sum and the identity of `logaddexp`, so folding
        /// this over any partition of the inputs agrees with running it over all of them at
        /// once. Above the `Worst` precision policy the non-dominant terms go through
        /// `ln_1p`, which keeps the answer accurate when one weight dominates.
        ///
        /// # Examples
        ///
        /// ```
        /// use thermite::prelude::*;
        ///
        /// type V = Vector<f64>;
        ///
        /// // Overflows outright if evaluated as `ln(e^1000 + e^1001 + e^999)`.
        /// let y = V::logsumexp_n([V::splat(1000.0), V::splat(1001.0), V::splat(999.0)]);
        /// assert!((y.extract::<0>() - 1001.4076059644443).abs() < 1e-12);
        /// ```
        fn logsumexp_n[const N: usize][N](values: [Self; N]) -> Self;

        /// Returns `$\ln(e^{a} - e^{b})$` where `a = self` and `b = other`, computed in a
        /// numerically stable way that avoids overflow.
        ///
        /// The subtractive counterpart of [`logaddexp`](RealMath::logaddexp), for removing a
        /// term from a log-domain sum (a leave-one-out normalizer, a difference of
        /// cumulative distribution functions in log space). Evaluated as
        /// `$a + \ln(1 - e^{-(a - b)})$` via [`ln1m_expnx`](TranscendentalMath::ln1m_expnx),
        /// so no intermediate exponential overflows and the precision ladder is that
        /// kernel's.
        ///
        /// At `Average` precision and above, `$\ln(1 - e^{-x})$` is split into two regimes
        /// at `$\ln 2$`, keeping the subtraction inside `exp_m1` below the split and inside
        /// `ln_1p` above it, which is accurate at both ends of the gap. A single
        /// `$(1 - e^{-x})$` followed by a log loses the small gaps to cancellation and the
        /// large ones to `$1 - e^{-x}$` rounding to exactly 1. Below `Average`,
        /// `ln1m_expnx`'s cheaper forms apply, with the accuracy losses those tiers accept.
        ///
        /// The result exists only for `a >= b`, and is `-inf` at `a == b`. An `a < b` input
        /// is out of domain and gives NaN at `Average` precision and above.
        fn logsubexp[][](self: Self, other: Self) -> Self;

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
        ///
        /// # Examples
        ///
        /// ```
        /// use thermite::prelude::*;
        ///
        /// type V = Vector<f64>;
        ///
        /// // Standard 3rd-order smoothstep (N = 2) over the default [0, 1] edges:
        /// // 3t^2 - 2t^3
        /// let y = V::splat(0.25).smoothstep::<2>(None);
        /// assert!((y.extract::<0>() - 0.15625).abs() < 1e-15);
        /// ```
        fn smoothstep[const N: usize][N](self: Self, edges: Option<(Self, Self)>) -> Self;

        /// Returns the inverse smoothstep of `self`, which is the value that would produce `self` when passed to `smoothstep`.
        ///
        /// N from 0..=2 have fast closed-form solutions, while higher N use numerical root-finding methods, which will inherently
        /// be much slower.
        ///
        /// # Examples
        ///
        /// Round-trips [`smoothstep`](RealMath::smoothstep), even at high orders where
        /// the inverse must be found numerically:
        ///
        /// ```
        /// use thermite::prelude::*;
        ///
        /// type V = Vector<f64>;
        ///
        /// let x = V::splat(1.0 / 16.0);
        /// let y = x.smoothstep::<12>(None);
        /// let x_back = y.inverse_smoothstep::<12>(None);
        /// assert!((x_back.extract::<0>() - x.extract::<0>()).abs() < 1e-9);
        /// ```
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

    /// "Primal" math for _single-value_ real numbers: plain float vectors and
    /// `Compensated`, but never derivative- or component-carrying composites like
    /// `Dual` or `Complex`. Effectively a stricter form of real-valued math.
    ///
    /// A primal type is its own [`Primal`](PrimalProjection::Primal), which makes
    /// it the storage type for precomputed constants and coefficient tables shared
    /// with the composites built over it.
    #[diagnostic::on_unimplemented(
        message = "`{Self}` is not a primal (single-value real) float vector",
        note = "`PrimalMath` is implemented for plain real float vectors and `Compensated`, never for derivative- or component-carrying composites such as `Dual` or `Complex`.",
        note = "For a composite type, use its associated `Primal` type (`Self::Primal` via `PrimalProjection`) instead of the composite itself."
    )]
    trait Primal<FloatElement>: RealMathWithPolicy & PrimalProjection<Primal = Self> {
    }
}
