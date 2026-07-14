// The `fn name[..][..](self: Self, ..)` shape is the macro DSL's, as in
// `thermite::math`, which allows this lint at its module root for the same reason.
#![allow(clippy::needless_arbitrary_self_type)]

//! The complex math trait family.
//!
//! Thermite's math families ([`CoreMath`](thermite::math::CoreMath),
//! [`TranscendentalMath`](thermite::math::TranscendentalMath),
//! [`SpatialMath`](thermite::math::SpatialMath),
//! [`RealMath`](thermite::math::RealMath)) are all `Self -> Self`, a real vector
//! having nothing else to return. A complex number does: its modulus and argument
//! are real, and its polar form is a pair of reals. Those operations get their own
//! family here, built the way the core ones are:
//!
//! - [`ComplexVector`] carries the structure: the associated real type
//!   [`Real`](ComplexVector::Real), the component accessors, and the operations
//!   that take no [`Policy`] (`conj`, `norm_sqr`, `inv`, `norm_l1`).
//! - [`SpecializedComplexMath`] carries the algorithms, mirroring
//!   [`thermite::math::specialized`].
//! - [`ComplexMathWithPolicy`] and [`ComplexMath`] are generated from it by
//!   `decl_complex_math!` (a copy of core's `decl_math!`), giving each operation a
//!   `foo_p::<P>()` and a default-policy `foo()` form.
//!
//! So `z.norm_p::<Precision>()` behaves as `x.sin_p::<Precision>()` does, and
//! generic code bounds on `V: ComplexMath` as it would on `V: TranscendentalMath`.

use core::ops::{Add, AddAssign, Div, DivAssign, Mul, MulAssign, Rem, RemAssign, Sub, SubAssign};

use thermite::element::FloatElement;
use thermite::math::policy::{DefaultPolicy, Policy};
use thermite::prelude::*;
use thermite::vector::ops::{self, MulAddAssignExt, MulAddExt};

use crate::vector::ComplexFloatVector;

/// A vector of complex numbers over a real vector type.
///
/// The structural half of the complex math family: it names the underlying real
/// vector ([`Real`](ComplexVector::Real)) and the operations that take no
/// [`Policy`]. The policy-dependent ones (modulus, argument, polar form, ...) are
/// in [`ComplexMath`].
///
/// # Mixed complex/real arithmetic
///
/// The supertraits promise the binary operators against [`Real`](Self::Real), so
/// generic code can scale, offset and fuse by a real vector without widening it
/// into a complex one:
///
/// ```
/// use thermite::prelude::*;
/// use thermite_complex::{Complex, ComplexVector};
///
/// // Horner evaluation of a real-coefficient polynomial at a complex point.
/// fn horner<T: ComplexVector>(z: T, coeffs: &[T::Real]) -> T {
///     let mut acc = T::from_real(coeffs[0]);
///
///     for &c in &coeffs[1..] {
///         acc = acc * z + c; // Complex * Complex, then Complex + Real
///     }
///
///     acc
/// }
///
/// type V = Vector<f64>;
///
/// // z^2 + 3 at z = 1 + 2i is -3 + 4i + 3 = 4i
/// let z = Complex::new(V::splat(1.0), V::splat(2.0));
/// let r = horner(z, &[V::ONE, V::ZERO, V::splat(3.0)]);
///
/// assert_eq!((r.re.extract::<0>(), r.im.extract::<0>()), (0.0, 4.0));
/// ```
///
/// `Mul`/`Div` by a real cost two real multiplies, versus the four multiplies and
/// two adds of a complex multiply by `Complex::real(r)`, which the compiler cannot
/// recover from the widened form.
/// [`MulAddExt<Self::Real, Self>`](thermite::vector::ops::MulAddExt) is promised
/// for the same reason, and is a true single-rounding FMA (one fused op per
/// component) where the complex-by-complex FMA cannot be.
pub trait ComplexVector:
    FloatVector
    + Add<Self::Real, Output = Self>
    + Sub<Self::Real, Output = Self>
    + Mul<Self::Real, Output = Self>
    + Div<Self::Real, Output = Self>
    + Rem<Self::Real, Output = Self>
    + AddAssign<Self::Real>
    + SubAssign<Self::Real>
    + MulAssign<Self::Real>
    + DivAssign<Self::Real>
    + RemAssign<Self::Real>
    + MulAddExt<Self::Real, Self, Output = Self>
    + MulAddAssignExt<Self::Real, Self>
    + ops::AddMasked<Self::Mask, Self::Real, Output = Self>
    + ops::SubMasked<Self::Mask, Self::Real, Output = Self>
    + ops::MulMasked<Self::Mask, Self::Real, Output = Self>
    + ops::DivMasked<Self::Mask, Self::Real, Output = Self>
    + ops::MulAddExtMasked<Self::Mask, Self::Real, Self, Output = Self>
{
    /// The real vector type of each component, in which a modulus or an argument
    /// is measured.
    ///
    /// The core [`SpatialMath`] family has no such associated type, so its norms
    /// must return `Self` and come back as real-valued *complex* numbers.
    /// [`ComplexMath::norm`] returns this instead.
    ///
    /// [`SpatialMath`]: thermite::math::SpatialMath
    type Real: ComplexFloatVector;

    /// The real part.
    fn re(self) -> Self::Real;

    /// The imaginary part.
    fn im(self) -> Self::Real;

    /// Builds a complex vector from its real and imaginary parts.
    fn from_parts(re: Self::Real, im: Self::Real) -> Self;

    /// Builds a complex vector from a real part, with zero imaginary part.
    #[inline(always)]
    fn from_real(re: Self::Real) -> Self {
        Self::from_parts(re, <Self::Real as NumericVector>::ZERO)
    }

    /// The complex conjugate `re - im*i`.
    fn conj(self) -> Self;

    /// The squared modulus `$|z|^2 = re^2 + im^2$`.
    ///
    /// Cheaper than [`norm`](ComplexMath::norm) (no square root), but it squares
    /// the range, so it overflows or underflows near the limits of the format.
    fn norm_sqr(self) -> Self::Real;

    /// The L1 ("Manhattan") norm `|re| + |im|`, a real value.
    fn norm_l1(self) -> Self::Real;

    /// The multiplicative inverse `$1/z = \bar{z}/|z|^2$`.
    ///
    /// Inherits the range limits of [`norm_sqr`](ComplexVector::norm_sqr); the
    /// scaled form is [`ComplexMath::finv`].
    fn inv(self) -> Self;
}

/// Element-parameterized implementations behind [`ComplexMath`].
///
/// The complex counterpart of [`thermite::math::specialized`]: implementing this
/// for a complex vector type gives it [`ComplexMath`] and
/// [`ComplexMathWithPolicy`], as implementing `SpecializedTranscendentalMath`
/// gives it `TranscendentalMath`.
///
/// Bound on [`ComplexMath`]; this trait is for implementors.
pub trait SpecializedComplexMath<E>: ComplexVector<Element = E> {
    /// The modulus (magnitude) `|z|`.
    fn norm<P: Policy>(self) -> Self::Real;

    /// The principal argument `arg(z)`, in `(-pi, pi]`.
    fn arg<P: Policy>(self) -> Self::Real;

    /// Polar form `(r, theta)`, such that `self == r * exp(i*theta)`.
    #[inline(always)]
    fn to_polar<P: Policy>(self) -> (Self::Real, Self::Real) {
        (self.norm::<P>(), self.arg::<P>())
    }

    /// Builds a complex number from a polar representation `r * exp(i*theta)`.
    fn from_polar<P: Policy>(r: Self::Real, theta: Self::Real) -> Self;

    /// Raises `self` to a *real* power.
    fn powfr<P: Policy>(self, e: Self::Real) -> Self;

    /// Raises a *real* base to the complex power `self`.
    fn expf<P: Policy>(self, base: Self::Real) -> Self;

    /// The logarithm of `self` in an arbitrary *real* base.
    fn logr<P: Policy>(self, base: Self::Real) -> Self;

    /// `1/self`, scaling by the modulus and not its square.
    ///
    /// Survives the magnitudes where [`inv`](ComplexVector::inv) would have
    /// `norm_sqr()` overflow to infinity or underflow to zero.
    fn finv<P: Policy>(self) -> Self;

    /// `self/rhs`, scaling by the modulus and not its square.
    ///
    /// Survives the magnitudes where `/` would have `rhs.norm_sqr()` overflow to
    /// infinity or underflow to zero.
    #[inline(always)]
    fn fdiv<P: Policy>(self, rhs: Self) -> Self {
        self * rhs.finv::<P>()
    }
}

// A copy of thermite::math's (private) decl_math!, dropping the ScalarMath
// aggregate, which only makes sense for bare f32/f64. The rest is unchanged, so
// ComplexMath is generated as TranscendentalMath is, #[dispatch] trampolines and
// all.
macro_rules! decl_complex_math {
    ($(
        $(#[$trait_meta:meta])*
        trait $trait:ident<$element:ident> $(: $($bound:ident)&+ )? { $(
            $(#[$meta:meta])*
            fn $name:ident [ $($generics:tt)* ][$($generic_names:ident),*]( $($arg_name:ident : $arg_ty:ty),* $(,)?) -> $ret:ty;
        )*}
    )*) => {paste::paste! {$(
        #[doc = "" $trait " math functions with customizable policies."]
        $(#[$trait_meta])*
        #[doc = ""]
        #[doc = "Each function takes a [`Policy`] as its first generic argument. For the"]
        #[doc = "default-policy versions (same names, no `_p` suffix), see [`" $trait "Math`]."]
        #[doc = ""]
        #[doc = "Implemented automatically for every type implementing [`Specialized" $trait "Math`]."]
        #[thermite::dispatch(Self)]
        pub trait [<$trait MathWithPolicy>] $(: $($bound +)+)? {$(
            $(#[$meta])* fn [<$name _p>]<P: Policy, $($generics)*>($($arg_name: $arg_ty),*) -> $ret;
        )*}

        #[doc = "" $trait " math functions using the default policy."]
        $(#[$trait_meta])*
        #[doc = ""]
        #[doc = "Every method here has a counterpart in [`" $trait "MathWithPolicy`] with a `_p`"]
        #[doc = "suffix that takes an explicit [`Policy`]."]
        #[doc = ""]
        #[doc = "Implementors of [`" $trait "MathWithPolicy`] implement this automatically."]
        #[thermite::dispatch(Self)]
        pub trait [<$trait Math>]: [<$trait MathWithPolicy>] {$(
            $(#[$meta])* #[inline(always)] fn $name<$($generics)*>($($arg_name: $arg_ty),*) -> $ret
            { [<$trait MathWithPolicy>]::[<$name _p>]::<DefaultPolicy, $($generic_names),*>($($arg_name),*) }
        )*}

        impl<M> [<$trait Math>] for M where M: [<$trait MathWithPolicy>] {}

        // The FloatVector<Element = E> bound is what ties E down, as in core.
        #[thermite::dispatch(Self)]
        impl<E: $element, V: FloatVector<Element = E> + $($($bound +)+)?> [<$trait MathWithPolicy>] for V
            where V: [<Specialized $trait Math>]<E>
        {$(
            $(#[$meta])* #[inline(always)] fn [<$name _p>]<P: Policy, $($generics)*>($($arg_name: $arg_ty),*) -> $ret
            { <V as [<Specialized $trait Math>]<E>>::$name::<P, $($generic_names),*>($($arg_name),*) }
        )*})*
    }};
}

decl_complex_math! {
    /// Operations whose result is real (modulus, argument, polar form) or whose
    /// argument is (a real power, base, or logarithm base), which the `Self -> Self`
    /// core families cannot express.
    ///
    /// The purely complex operations (`exp`, `ln`, `sin`, `sqrt`, `powf`, ...) fit
    /// the core families and come from
    /// [`TranscendentalMath`](thermite::math::TranscendentalMath) as they do for
    /// any other vector.
    trait Complex<FloatElement>: ComplexVector {
        /// The modulus `$|z|$`, as a real value.
        ///
        /// Uses `hypot`, so it does not overflow for large components the way
        /// `sqrt(norm_sqr())` would.
        fn norm[][](self: Self) -> Self::Real;

        /// The principal argument `arg(z)`, in `(-pi, pi]`, as a real value.
        fn arg[][](self: Self) -> Self::Real;

        /// Converts to polar form `(r, theta)`, such that `self == r * exp(i*theta)`.
        fn to_polar[][](self: Self) -> (Self::Real, Self::Real);

        /// Builds a complex number from a polar representation `r * exp(i*theta)`.
        fn from_polar[][](r: Self::Real, theta: Self::Real) -> Self;

        /// Raises `self` to a real power.
        ///
        /// The complex-exponent form is [`powf`](thermite::math::TranscendentalMath::powf).
        fn powfr[][](self: Self, e: Self::Real) -> Self;

        /// Raises a real base to the complex power `self`.
        fn expf[][](self: Self, base: Self::Real) -> Self;

        /// The logarithm of `self` in an arbitrary real base.
        ///
        /// The complex-base form is [`log`](thermite::math::TranscendentalMath::log).
        fn logr[][](self: Self, base: Self::Real) -> Self;

        /// `1/self`, scaling by the modulus and not its square.
        ///
        /// Survives the magnitudes where [`inv`](ComplexVector::inv) would have
        /// `norm_sqr()` overflow to infinity or underflow to zero.
        fn finv[][](self: Self) -> Self;

        /// `self/rhs`, scaling by the modulus and not its square.
        ///
        /// Survives the magnitudes where `/` would have `rhs.norm_sqr()` overflow
        /// to infinity or underflow to zero.
        fn fdiv[][](self: Self, rhs: Self) -> Self;
    }
}
