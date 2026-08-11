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
//! - [`ComplexMathWithPolicy`](crate::math::ComplexMathWithPolicy) and [`ComplexMath`](crate::math::ComplexMath) are generated from it by
//!   `decl_complex_math!` (a copy of core's `decl_math!`), giving each operation a
//!   `foo_p::<P>()` and a default-policy `foo()` form.
//!
//! So `z.norm_p::<Precision>()` behaves as `x.sin_p::<Precision>()` does, and
//! generic code bounds on `V: ComplexMath` as it would on `V: TranscendentalMath`.

use core::ops::{Add, AddAssign, Div, DivAssign, Mul, MulAssign, Rem, RemAssign, Sub, SubAssign};

use thermite::math::policy::Policy;
use thermite::prelude::*;
use thermite::vector::ops::{self, MulAddAssignExt, MulAddExt};

use crate::vector::RealFloatVector;

#[cfg(feature = "special")]
pub use crate::math::special::SpecializedComplexSpecialMath;

/// A vector of complex numbers over a real vector type.
///
/// The structural half of the complex math family: it names the underlying real
/// vector ([`Real`](ComplexVector::Real)) and the operations that take no
/// [`Policy`]. The policy-dependent ones (modulus, argument, polar form, ...) are
/// in [`ComplexMath`](crate::math::ComplexMath).
///
/// # Mixed complex/real arithmetic
///
/// The supertraits promise the binary operators against [`Real`](Self::Real), so
/// generic code can scale, offset and fuse by a real vector without widening it
/// into a complex one:
///
/// ```
/// use thermite::prelude::*;
/// use thermite_complex::prelude::*;
///
/// // Horner evaluation of a real-coefficient polynomial at a complex point.
/// fn horner<T: ComplexVector>(z: T, coeffs: &[T::Real]) -> T {
///     let mut acc = T::real(coeffs[0]);
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
    /// [`ComplexMath::norm`](crate::math::ComplexMath::norm) returns this instead.
    ///
    /// [`SpatialMath`]: thermite::math::SpatialMath
    type Real: RealFloatVector;

    /// The real part.
    fn re(self) -> Self::Real;

    /// The imaginary part.
    fn im(self) -> Self::Real;

    /// Builds a complex vector from its real and imaginary parts.
    ///
    /// Not spelled `new`, tempting as it is to match the inherent
    /// [`Complex::new`](crate::Complex::new): [`GenericVector::new`] is already in
    /// scope on every implementor and takes a lane array, so a second `new` is
    /// ambiguous (E0034) in precisely the generic code this trait exists for.
    /// [`real`](Self::real) has no such clash and does match its inherent twin.
    fn from_parts(re: Self::Real, im: Self::Real) -> Self;

    /// Non-temporal store of the whole block to `ptr`, in `Self`'s own memory layout (the
    /// planar/SoA `[re | im]` layout for `Complex<V>`), bypassing the cache. This is for
    /// **relocating blocks within a `[Self]` buffer** - e.g. an FFT transpose whose output is
    /// too large to cache - NOT the AoS boundary (that is [`store`](thermite::prelude::GenericVector::store)
    /// / [`store_streaming`](thermite::prelude::GenericVector::store_streaming), which interleave re/im).
    ///
    /// Weakly ordered: a non-temporal store is not guaranteed visible to a later load until an
    /// `sfence`, so the caller **must fence before reading the result**. Use it only when the
    /// destination clearly exceeds last-level cache (NT forfeits cache reuse, so it loses below
    /// a few MB).
    ///
    /// The default is a plain store (correct everywhere, no NT benefit). `Complex<V>` overrides
    /// it to stream each half with the real [`store_streaming`](thermite::prelude::GenericVector::store_streaming)
    /// of its component vector (`_mm256_stream_ps` on AVX2; a plain store on backends without NT).
    ///
    /// # Safety
    /// `ptr` must be valid for writes and aligned to `Self` (a `[Self]` slot satisfies this).
    #[inline(always)]
    unsafe fn store_streaming_block(self, ptr: *mut Self) {
        unsafe { ptr.write(self) }
    }

    /// Builds a complex vector from a real part, with zero imaginary part.
    #[inline(always)]
    fn real(re: Self::Real) -> Self {
        Self::from_parts(re, <Self::Real as NumericVector>::ZERO)
    }

    /// The complex conjugate `re - im*i`.
    fn conj(self) -> Self;

    /// The squared modulus `$|z|^2 = re^2 + im^2$`.
    ///
    /// Cheaper than [`norm`](crate::math::ComplexMath::norm) (no square root), but it squares
    /// the range, so it overflows or underflows near the limits of the format.
    fn norm_sqr(self) -> Self::Real;

    /// The L1 ("Manhattan") norm `|re| + |im|`, a real value.
    fn norm_l1(self) -> Self::Real;

    /// The multiplicative inverse `$1/z = \bar{z}/|z|^2$`.
    ///
    /// Inherits the range limits of [`norm_sqr`](ComplexVector::norm_sqr); the
    /// scaled form is [`ComplexMath::finv`](crate::math::ComplexMath::finv).
    fn inv(self) -> Self;
}

/// Element-parameterized implementations behind [`ComplexMath`](crate::math::ComplexMath).
///
/// The complex counterpart of [`thermite::math::specialized`]: implementing this
/// for a complex vector type gives it [`ComplexMath`](crate::math::ComplexMath) and
/// [`ComplexMathWithPolicy`](crate::math::ComplexMathWithPolicy), as implementing `SpecializedTranscendentalMath`
/// gives it `TranscendentalMath`.
///
/// Bound on [`ComplexMath`](crate::math::ComplexMath); this trait is for implementors.
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

