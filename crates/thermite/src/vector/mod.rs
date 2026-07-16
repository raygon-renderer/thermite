#![allow(missing_docs, clippy::missing_safety_doc)]
#![deny(unconditional_recursion)] // just in case we miss one

//! User-facing vector types and the trait hierarchy that defines them.
//!
//! This module is the top of Thermite's public API. It provides the
//! [`Vector<R>`] newtype - the value you actually compute with - and the tower
//! of traits ([`GenericVector`] and its descendants) that describe what a
//! vector can do.
//!
//! # Generic over *behavior*, not over a backend
//!
//! The central idea of Thermite is that you write code against
//! [`GenericVector`] (or a more specific trait like [`NumericVector`],
//! [`FloatVector`], or [`IntegerVector`]) and let the caller pick the concrete
//! type. That concrete type decides the ISA, the lane count, and the element
//! type - your code does not name any of them:
//!
//! ```
//! use thermite::prelude::*;
//! use thermite::math::TranscendentalMath;
//!
//! // Works on any backend, any width, any float element type.
//! fn gaussian<V: FloatVector + TranscendentalMath>(v: V) -> V {
//!     (-v * v).exp()
//! }
//! ```
//!
//! Crucially, "generic" here is stronger than "generic over the hardware
//! backend". A [`GenericVector`] is not required to be a dense array of scalars
//! sitting in a hardware register at all. The trait describes an *algebra of
//! lanes*, and anything that satisfies that algebra is a first-class vector.
//!
//! # Composable abstractions all the way up
//!
//! Because the trait bounds are the only contract, wrapper types that are not
//! SIMD registers in any conventional sense can still implement the hierarchy
//! and flow through the very same generic functions:
//!
//! - **Complex numbers** - a `Complex<V>` pairing two real vectors implements
//!   the [`GenericVector`]/[`FloatVector`] traits, so a function written for
//!   real `FloatVector`s operates transparently on complex data.
//! - **Compensated arithmetic** - a double-double `Compensated<V>` that tracks
//!   rounding error implements the same traits; existing generic code gains
//!   extended precision just by being instantiated with it.
//! - **Dual / hyperdual numbers** - automatic differentiation via the same
//!   trait composition, so a generic numeric routine differentiates itself when
//!   handed a dual type.
//!
//! And these compose: `Complex<Compensated<f32x8>>` is a perfectly valid vector
//! type where every complex operation is carried out in compensated real
//! arithmetic, all still SIMD-accelerated underneath. The function you wrote
//! once against `FloatVector` does not change.
//!
//! # The trait hierarchy
//!
//! Each trait adds capability on top of the previous one; bound on the least
//! specific trait that supplies the operations you need.
//!
//! ```text
//! GenericVector          construction, lane access, memory I/O, gather/scatter,
//!   |                    reinterpretation, map/fold/reduce, interleave
//!   |- BitwiseVector     &, |, ^, !, andnot, ternlog
//!   |   \- BitshiftVector   shifts, rotations, byte-shifts
//!   \- PartialOrdVector  cmp_lt/le/gt/ge/eq/ne -> Mask
//!       \- NumericVector    +, -, *, /, %, min/max/clamp, reductions, FMA
//!            |- SignedVector     abs, signum, copysign, neg
//!            |    \- FloatVector        sqrt, rcp/rsqrt, rounding, mix, consts
//!            |         \- FloatVectorWithBits  ldexp/frexp, bit-level ops
//!            \- IntegerVector    saturating/wrapping, popcount, dividers
//!                 |              (also requires BitshiftVector)
//!                 |- SignedIntegerVector    arithmetic shift, avg
//!                 |                         (also requires SignedVector)
//!                 \- UnsignedIntegerVector  is_power_of_two, parity, avg
//! ```
//!
//! `FloatVector` and `SignedIntegerVector` both sit under [`SignedVector`];
//! `SignedIntegerVector` additionally requires [`IntegerVector`], so it is the
//! meeting point of the signed and integer branches. `IntegerVector` itself
//! does **not** require [`SignedVector`] - unsigned integer vectors are
//! integers without being signed.
//!
//! Alongside these, [`LinAlg3Vector`]/[`LinAlg4Vector`] add 3D/4D linear-algebra
//! operations, and the `Swizzle`/[`Swizzle3`]/[`Swizzle4`] traits add lane
//! permutation. Masked (`_c`/`_m`/`_z`) variants of most operations live in the
//! [`ops`] submodule.
//!
//! Three layers cooperate to make all of this work: an `Element` (the scalar),
//! a [`Register`](crate::register::Register) (the functional hardware layer),
//! and [`Vector<R>`] (this module's ergonomic wrapper). Most users only ever
//! touch the [`Vector`] layer and its traits.

use core::{
    marker::PhantomData,
    ops::{BitAnd, BitAndAssign, BitOr, BitOrAssign, BitXor, BitXorAssign, Not},
};

#[cfg(feature = "bitvec")]
use bitvec::{array::BitArray, view::BitViewSized};
use generic_array::{GenericArray, typenum};

use crate::{
    BranchfreeDivider, Divider, Mask,
    divider::{Denominator, vector::VectorDivider},
    element::{FloatElementWithBits, UnsignedIntegerElement},
    isa::InstructionSet,
    mask::{CastMask, GenericMask, GenericSelectable},
    math::{FloatConsts, policy::Policy},
    register::{Element, FloatElement, Lanes, NativeCapability},
};

mod num;

#[doc(hidden)]
pub mod splat;

#[allow(clippy::module_inception)]
mod vector;

pub mod ops;
pub mod streaming;
pub mod unaligned;

pub use self::num::NumVector;
pub use self::splat::{NewConst, NewVector, SplatConst, SplatVector, VectorValue, const_new, const_splat};
pub use self::vector::Vector;
pub use crate::register::StreamGroup;

/// Three vector types (`Self`, `A`, `B`) whose masks can all be freely cast to
/// one another.
///
/// All three must share the same [`Lanes`](GenericVector::Lanes) count, and
/// each one's [`Mask`](GenericVector::Mask) must implement [`CastMask`] into
/// the other two. This is a convenience bound for generic code that selects or
/// blends across vectors of different element types but identical width - e.g.
/// using a mask produced from a float comparison to select lanes of an integer
/// vector.
///
/// It is blanket-implemented for every triple of types satisfying the cast
/// requirements, so it never needs to be implemented manually.
pub trait MaskInteroperable<A, B>: GenericVector<Mask: CastMask<A::Mask> + CastMask<B::Mask>>
where
    A: GenericVector<Lanes = Self::Lanes, Mask: CastMask<Self::Mask> + CastMask<B::Mask>>,
    B: GenericVector<Lanes = Self::Lanes, Mask: CastMask<Self::Mask> + CastMask<A::Mask>>,
{
}

impl<T, A, B> MaskInteroperable<A, B> for T
where
    T: GenericVector<Mask: CastMask<A::Mask> + CastMask<B::Mask>>,
    A: GenericVector<Lanes = T::Lanes, Mask: CastMask<T::Mask> + CastMask<B::Mask>>,
    B: GenericVector<Lanes = T::Lanes, Mask: CastMask<T::Mask> + CastMask<A::Mask>>,
{
}

/// [`MaskInteroperable`] plus bidirectional numeric ([`CastVector`]) conversion
/// among `Self`, `A`, and `B`.
///
/// In addition to interoperable masks, this guarantees `Self`, `A`, and `B` can
/// all be numerically cast into one another in either direction (`A`/`B` into
/// `Self` *and* `Self` into `A`/`B`), so generic code can freely move operands
/// of differing element types into whichever common type it needs before
/// combining them. It does **not** require bit-level reinterpretation; for that
/// see [`FullyInteroperable`].
///
/// Blanket-implemented for every triple satisfying the bounds.
pub trait PartiallyInteroperable<A, B>:
    GenericVector<Mask: CastMask<A::Mask> + CastMask<B::Mask>>
    // casts
    + CastVector<Self>
    + CastVector<A>
    + CastVector<B>
where
    A: CastVector<Self> + GenericVector<Lanes = Self::Lanes, Mask: CastMask<Self::Mask> + CastMask<B::Mask>>,
    B: CastVector<Self> + GenericVector<Lanes = Self::Lanes, Mask: CastMask<Self::Mask> + CastMask<A::Mask>>,
{
}

impl<V, A, B> PartiallyInteroperable<A, B> for V
where
    V: GenericVector<Mask: CastMask<A::Mask> + CastMask<B::Mask>>
        // casts
        + CastVector<V>
        + CastVector<A>
        + CastVector<B>,
    A: CastVector<V> + GenericVector<Lanes = V::Lanes, Mask: CastMask<V::Mask> + CastMask<B::Mask>>,
    B: CastVector<V> + GenericVector<Lanes = V::Lanes, Mask: CastMask<V::Mask> + CastMask<A::Mask>>,
{
}

/// [`PartiallyInteroperable`] plus zero-cost bit-level reinterpretation
/// ([`BitCastVector`]) among `Self`, `A`, and `B`.
///
/// The strongest of the three interoperability bounds: masks are mutually
/// castable, the three element types convert numerically, *and* their bit
/// patterns can be reinterpreted into one another. This is what a float vector
/// needs against its own bits/signed-bits integer vectors (see
/// [`FloatVectorWithBits`]) so that bit-twiddling algorithms can hop between the
/// float view and the integer view with no instructions emitted.
///
/// Blanket-implemented for every triple satisfying the bounds.
pub trait FullyInteroperable<A, B>:
    GenericVector<Mask: CastMask<A::Mask> + CastMask<B::Mask>>
    // bits
    + BitCastVector<Self>
    + BitCastVector<A>
    + BitCastVector<B>
    // casts
    + CastVector<Self>
    + CastVector<A>
    + CastVector<B>
where
    A: BitCastVector<Self> + CastVector<Self> + GenericVector<Lanes = Self::Lanes, Mask: CastMask<Self::Mask> + CastMask<B::Mask>>,
    B: BitCastVector<Self> + CastVector<Self> + GenericVector<Lanes = Self::Lanes, Mask: CastMask<Self::Mask> + CastMask<A::Mask>>,
{
}

impl<V, A, B> FullyInteroperable<A, B> for V
where
    V: GenericVector<Mask: CastMask<A::Mask> + CastMask<B::Mask>>
        // bits
        + BitCastVector<V>
        + BitCastVector<A>
        + BitCastVector<B>
        // casts
        + CastVector<V>
        + CastVector<A>
        + CastVector<B>,
    A: BitCastVector<V> + CastVector<V> + GenericVector<Lanes = V::Lanes, Mask: CastMask<V::Mask> + CastMask<B::Mask>>,
    B: BitCastVector<V> + CastVector<V> + GenericVector<Lanes = V::Lanes, Mask: CastMask<V::Mask> + CastMask<A::Mask>>,
{
}

/// Internal helpers for generic vectors.
trait GenericVectorExt: GenericVector {
    #[inline(always)]
    fn len_to_indices<I: UnsignedIntegerVector>(len: usize) -> I {
        let Ok(len) = <<I as GenericVector>::Element as TryFrom<usize>>::try_from(len) else {
            #[cfg(feature = "std")]
            panic!("Length {} exceeds maximum supported index for this vector type", len);

            #[cfg(not(feature = "std"))]
            panic!("Length exceeds maximum supported index for this vector type");
        };

        I::splat(len)
    }
}

impl<V: GenericVector> GenericVectorExt for V {}

/// An unsigned integer vector that can be used as the index operand for
/// gather/scatter operations producing/consuming a vector of type `V`.
///
/// This is the inverse-facing companion to [`IndexableVector`]: where
/// `IndexableVector<I>` is implemented on the gathered vector type, this is
/// implemented on the index type. It is blanket-implemented for every index
/// type `I` such that `V: IndexableVector<I>`, simply forwarding to `V`'s
/// methods. The index lanes are element offsets (not byte offsets) and must
/// match `V`'s lane count.
///
/// The methods here are the raw pointer primitives; prefer the safe,
/// bounds-checked wrappers on [`GenericVector`] ([`gather`](GenericVector::gather),
/// [`scatter`](GenericVector::scatter), etc.) instead of calling these directly.
pub trait VectorIndices<V: GenericVector>: UnsignedIntegerVector<Lanes = V::Lanes> {
    /// Gather one element of `V` per lane from `ptr[indices[lane]]`.
    ///
    /// # Safety
    /// `ptr` must be valid for reads, and for every lane the offset
    /// `indices[lane]` must land within the allocation `ptr` points into
    /// (i.e. `ptr.add(indices[lane])` must be readable). Indices are not
    /// bounds-checked.
    unsafe fn gather_ptr(ptr: *const V::Element, indices: Self) -> V;

    /// Like [`gather_ptr`](Self::gather_ptr), but only lanes where `mask` is
    /// `true` are loaded; the rest are taken from `src`.
    ///
    /// # Safety
    /// Same as [`gather_ptr`](Self::gather_ptr), but only the offsets for lanes
    /// where `mask` is `true` need to be in bounds; masked-off lanes are not
    /// accessed.
    unsafe fn gather_ptr_m(src: V, mask: V::Mask, ptr: *const V::Element, indices: Self) -> V;

    /// Like [`gather_ptr_m`](Self::gather_ptr_m), but masked-off lanes are
    /// zeroed instead of taken from a source vector.
    ///
    /// # Safety
    /// Same as [`gather_ptr_m`](Self::gather_ptr_m).
    unsafe fn gather_ptr_z(mask: V::Mask, ptr: *const V::Element, indices: Self) -> V;

    /// Scatter each lane of `value` to `ptr[indices[lane]]`.
    ///
    /// # Safety
    /// `ptr` must be valid for writes, and for every lane the offset
    /// `indices[lane]` must land within the allocation `ptr` points into.
    /// Indices are not bounds-checked, and overlapping (duplicate) indices
    /// produce an unspecified winning lane.
    unsafe fn scatter_ptr(value: V, ptr: *mut V::Element, indices: Self);

    /// Like [`scatter_ptr`](Self::scatter_ptr), but only lanes where `mask` is
    /// `true` are written.
    ///
    /// # Safety
    /// Same as [`scatter_ptr`](Self::scatter_ptr), but only the offsets for
    /// lanes where `mask` is `true` need to be in bounds; masked-off lanes are
    /// not written.
    unsafe fn scatter_ptr_m(value: V, mask: V::Mask, ptr: *mut V::Element, indices: Self);
}

/// A vector type that supports gather/scatter using index vectors of type `I`.
///
/// Implemented on the gathered/scattered vector type (`Self`), parameterized by
/// the unsigned integer index vector type `I` (which must share `Self`'s lane
/// count). Backends with hardware gather/scatter (e.g. AVX2's `vpgatherdd`)
/// provide an accelerated implementation; others fall back to scalar loops.
///
/// These are the raw pointer primitives; index lanes are element offsets, not
/// byte offsets, and are not bounds-checked. Prefer the safe, bounds-checked
/// [`GenericVector`] wrappers ([`gather`](GenericVector::gather),
/// [`scatter`](GenericVector::scatter), etc.) in normal code.
pub trait IndexableVector<I: UnsignedIntegerVector<Lanes = Self::Lanes>>: GenericVector {
    /// Gather one element per lane from `ptr[indices[lane]]`.
    ///
    /// # Safety
    /// `ptr` must be valid for reads, and every offset `indices[lane]` must
    /// land within the allocation `ptr` points into. Indices are not
    /// bounds-checked.
    unsafe fn gather_ptr(ptr: *const Self::Element, indices: I) -> Self;

    /// Like [`gather_ptr`](Self::gather_ptr), but only lanes where `mask` is
    /// `true` are loaded; the rest are taken from `src`.
    ///
    /// # Safety
    /// Same as [`gather_ptr`](Self::gather_ptr), but only the offsets for lanes
    /// where `mask` is `true` need to be in bounds.
    unsafe fn gather_ptr_m(src: Self, mask: Self::Mask, ptr: *const Self::Element, indices: I) -> Self;

    /// Like [`gather_ptr_m`](Self::gather_ptr_m), but masked-off lanes are
    /// zeroed instead of taken from a source vector.
    ///
    /// # Safety
    /// Same as [`gather_ptr_m`](Self::gather_ptr_m).
    unsafe fn gather_ptr_z(mask: Self::Mask, ptr: *const Self::Element, indices: I) -> Self;

    /// Scatter each lane of `value` to `ptr[indices[lane]]`.
    ///
    /// # Safety
    /// `ptr` must be valid for writes, and every offset `indices[lane]` must
    /// land within the allocation `ptr` points into. Indices are not
    /// bounds-checked; duplicate indices produce an unspecified winning lane.
    unsafe fn scatter_ptr(value: Self, ptr: *mut Self::Element, indices: I);

    /// Like [`scatter_ptr`](Self::scatter_ptr), but only lanes where `mask` is
    /// `true` are written.
    ///
    /// # Safety
    /// Same as [`scatter_ptr`](Self::scatter_ptr), but only the offsets for
    /// lanes where `mask` is `true` need to be in bounds.
    unsafe fn scatter_ptr_m(value: Self, mask: Self::Mask, ptr: *mut Self::Element, indices: I);
}

impl<I, V> VectorIndices<V> for I
where
    I: UnsignedIntegerVector<Lanes = V::Lanes>,
    V: IndexableVector<I>,
{
    #[inline(always)]
    unsafe fn gather_ptr(ptr: *const <V as GenericVector>::Element, indices: Self) -> V {
        unsafe { V::gather_ptr(ptr, indices) }
    }

    #[inline(always)]
    unsafe fn gather_ptr_m(
        src: V,
        mask: <V as GenericVector>::Mask,
        ptr: *const <V as GenericVector>::Element,
        indices: Self,
    ) -> V {
        unsafe { V::gather_ptr_m(src, mask, ptr, indices) }
    }

    #[inline(always)]
    unsafe fn gather_ptr_z(
        mask: <V as GenericVector>::Mask,
        ptr: *const <V as GenericVector>::Element,
        indices: Self,
    ) -> V {
        unsafe { V::gather_ptr_z(mask, ptr, indices) }
    }

    #[inline(always)]
    unsafe fn scatter_ptr(value: V, ptr: *mut <V as GenericVector>::Element, indices: Self) {
        unsafe { V::scatter_ptr(value, ptr, indices) }
    }

    #[inline(always)]
    unsafe fn scatter_ptr_m(
        value: V,
        mask: <V as GenericVector>::Mask,
        ptr: *mut <V as GenericVector>::Element,
        indices: Self,
    ) {
        unsafe { V::scatter_ptr_m(value, mask, ptr, indices) }
    }
}

/// Joining two `HALF`-width values into one double-width `Self`, and splitting
/// back apart.
///
/// Implemented for both vectors and masks. `Self` has exactly twice the lane
/// count of `HALF`. Most users should go through
/// [`GenericVector::concat`] / [`GenericVector::split`] rather than naming this
/// trait directly. Because the wide type can always be narrowed back to a half,
/// `Concat` requires [`Extend`].
pub trait Concat<HALF>: Extend<HALF> {
    /// Build the double-width value from a `lo` and `hi` half, with `lo`'s lanes
    /// occupying the lower half of the result and `hi`'s the upper half.
    fn concat(lo: HALF, hi: HALF) -> Self;

    /// Split into `(lo, hi)` halves, the inverse of [`concat`](Self::concat).
    fn split(self) -> (HALF, HALF);
}

/// Zero-extend a narrower `FROM` value into a wider `Self`, and narrow back.
///
/// Implemented for both vectors and masks. Most users should go through
/// [`GenericVector::extend`] / [`GenericVector::narrow`].
pub trait Extend<FROM> {
    /// Widen `v` into `Self`, placing `v`'s lanes in the lower half and filling
    /// the upper half with zeros.
    fn extend(v: FROM) -> Self;

    /// Narrow back to `FROM` by keeping the lower lanes and discarding the
    /// upper lanes.
    fn narrow(self) -> FROM;
}

/// [`Concat`] specialized to vector types: `Self` is a [`GenericVector`] that is
/// the concatenation of two `HALF` vectors of the same element type, and whose
/// mask is likewise the concatenation of two `HALF` masks.
///
/// Blanket-implemented; this is the bound used by [`GenericVector::concat`] /
/// [`GenericVector::split`].
pub trait ConcatVector<HALF: GenericVector<Element = Self::Element>>:
    Concat<HALF> + GenericVector<Mask: Concat<HALF::Mask>>
{
}

/// Zero-extend vectors
pub trait ExtendVector<FROM: GenericVector<Element = Self::Element>>:
    Extend<FROM> + GenericVector<Mask: Extend<FROM::Mask>>
{
}

impl<V: GenericVector, H: GenericVector<Element = V::Element>> ConcatVector<H> for V
where
    V: Concat<H>,
    V::Mask: Concat<H::Mask>,
{
}
impl<V: GenericVector, F: GenericVector<Element = V::Element>> ExtendVector<F> for V
where
    V: Extend<F>,
    V::Mask: Extend<F::Mask>,
{
}

/// A [`GenericVector`] whose lanes can be permuted by the
/// [`Swizzle`](crate::swizzle::Swizzle) machinery for its lane count.
///
/// Blanket-implemented for every vector that satisfies the swizzle bound; it is
/// the prerequisite for the human-readable swizzle traits ([`Swizzle3`],
/// [`Swizzle4`]) and the [`swizzle!`](crate::swizzle) macro.
pub trait SwizzleVector: GenericVector + crate::swizzle::Swizzle<Self::Lanes> {}
impl<V> SwizzleVector for V where V: GenericVector + crate::swizzle::Swizzle<V::Lanes> {}

pub trait Interleave: Sized {
    /// Unpack and interleave elements from two vectors.
    ///
    /// The resulting two vectors contain the interleaved elements from the input vectors. e.g.,
    /// for vectors `a = [a0, a1, a2, a3]` and `b = [b0, b1, b2, b3]`, the result will be
    /// `([a0, b0, a1, b1], [a2, b2, a3, b3])`.
    ///
    /// # Note
    ///
    /// Unlike the native unpacklo/unpackhi instructions, at higher register widths
    /// this will preserve the order of all elements, not just 128-bit chunks.
    fn interleave(self, other: Self) -> (Self, Self);

    /// Pack and deinterleave elements from two vectors. This is the inverse operation of `interleave`.
    ///
    /// The resulting vector contains the deinterleaved elements from the input vectors. e.g.,
    /// for vectors `a = [a0, b0, a1, b1]` and `b = [a2, b2, a3, b3]`, the result will be
    /// `[a0, a1, a2, a3]` and `[b0, b1, b2, b3]`.
    fn deinterleave(self, other: Self) -> (Self, Self);
}

/// Core trait for generic vector types.
///
/// Provides the basis for further specialized vector traits. Every other vector
/// trait in the hierarchy (`NumericVector`, `FloatVector`, `IntegerVector`, etc.)
/// is built on top of this one.
///
/// A `GenericVector` is a fixed-length, immutable, copyable array of `Element`s
/// laid out contiguously and aligned to its register's native alignment. The
/// number of lanes is known at compile time via the [`LANES`](Self::LANES)
/// constant and the [`Lanes`](Self::Lanes) associated type (a `typenum`).
///
/// All construction, lane access, memory I/O, gather/scatter, reinterpretation,
/// and scalar-fallback (`map`/`fold`/`reduce`) operations live on this trait.
/// Arithmetic, bitwise and float operations are added by the sub-traits.
#[rustfmt::skip] #[thermite_macros::vector_trait]
pub trait GenericVector: 'static + Sized + Default + Copy + core::fmt::Debug
    + const_default::ConstDefault
    + SplatVector<Self::Element> + NewVector<Self::Element, Self::Lanes>
    + GenericSelectable<SelectableMask = Self::Mask>
    + crate::simd::HasIsa
    + CastVector<Self>
    + Interleave
{
    /// Scalar element type of the vector.
    type Element: Element;

    /// A vector with all elements zeroed.
    const EMPTY: Self;

    /// Number of lanes in the vector.
    const LANES: usize;

    /// Number of lanes in the vector, as a runtime value.
    ///
    /// Today this is always [`LANES`](Self::LANES), but prefer it over the constant in
    /// loop bounds and address arithmetic: a future scalable-vector backend (SVE /
    /// RISC-V V) can only report its lane count at runtime, and code written against
    /// `lanes()` will carry over unchanged.
    #[inline(always)]
    fn lanes() -> usize {
        Self::LANES
    }

    /// Number of lanes in the vector, as a typenum.
    type Lanes: Lanes;

    /// Unsigned Integer Type suitable for use with this vector.
    type Unsigned: UnsignedIntegerVector<
            Signed = Self::Signed,
            Unsigned = Self::Unsigned,
            Lanes = Self::Lanes,
            Element = <Self::Element as Element>::Unsigned,
            Mask: CastMask<Self::Mask> + CastMask<<Self::Signed as GenericVector>::Mask>,
        > + CastVector<Self::Signed>
        + BitCastVector<Self::Signed>;

    /// SignedBits Integer Type suitable for use with this vector.
    type Signed: SignedIntegerVector<
            Signed = Self::Signed,
            Unsigned = Self::Unsigned,
            Lanes = Self::Lanes,
            Element = <Self::Element as Element>::Signed,
            Mask: CastMask<Self::Mask> + CastMask<<Self::Unsigned as GenericVector>::Mask>,
        > + CastVector<Self::Unsigned>
        + BitCastVector<Self::Unsigned>;

    /// Mask type for this vector. Masks are semantically boolean vectors indicating
    /// true or false for each lane. They may or may not be represented as actual bits.
    type Mask: GenericMask
        + CastMask<<Self::Unsigned as GenericVector>::Mask>
        + CastMask<<Self::Signed as GenericVector>::Mask>;

    /// Create a new vector from an array of elements.
    ///
    /// The array length `N` must equal [`LANES`](Self::LANES); this is enforced
    /// at compile time by the `Const<N> == Lanes` bound.
    fn new<const N: usize>(value: [Self::Element; N]) -> Self
        where generic_array::typenum::Const<N>: generic_array::IntoArrayLength<ArrayLength = Self::Lanes>;

    /// Consume the vector and return its elements as a `GenericArray`.
    ///
    /// This is the inverse of [`new`](Self::new); it copies lane-by-lane and
    /// has no runtime cost beyond a register-to-memory store on backends where
    /// the storage and array layouts are bit-identical (the common case).
    fn into_array(self) -> GenericArray<Self::Element, Self::Lanes>;

    /// Create a new vector from a single element by splatting it across all lanes.
    #[masked] fn splat(value: Self::Element) -> Self;

    /// Create a new vector with the first lane set to the given value, and all other lanes set to zero.
    fn single(value: Self::Element) -> Self;

    /// Combine two vectors of the same type into one wider vector,
    /// with `self` as the lower half and `hi` as the upper half.
    fn concat<INTO>(self, hi: Self) -> INTO
    where
        INTO: ConcatVector<Self, Element = Self::Element>,
    {
        <INTO as Concat<Self>>::concat(self, hi)
    }

    /// Split this vector into two narrower vectors of the same type, with the lower
    /// lanes in the first vector and the upper lanes in the second vector.
    fn split<INTO: GenericVector>(self) -> (INTO, INTO)
    where
        Self: ConcatVector<INTO, Element = INTO::Element>,
    {
        <Self as Concat<INTO>>::split(self)
    }

    /// Zero-extend a narrower vector into this wider vector type, placing the
    /// original values in the lower lanes and filling the upper lanes with zeros.
    fn extend<INTO>(self) -> INTO
    where
        INTO: ExtendVector<Self, Element = Self::Element>,
    {
        <INTO as Extend<Self>>::extend(self)
    }

    /// Narrow this wider vector into a narrower vector by taking the lower lanes.
    ///
    /// The upper lanes are discarded.
    fn narrow<INTO: GenericVector>(self) -> INTO
    where
        Self: ExtendVector<INTO, Element = INTO::Element>,
    {
        <Self as Extend<INTO>>::narrow(self)
    }

    /// Align a slice of elements to the vector's lane count, returning the aligned portion and any unaligned head or tail.
    ///
    /// If the vector's size in bytes does not match the size of its elements times the lane count, this will
    /// return the entire slice as unaligned and empty aligned/remaining parts. This is rare, but may occur
    /// if using a generic vector type that doesn't correspond to an actual hardware vector (for example, a 3-lane vector).
    #[inline(always)]
    fn align_slice(slice: &[Self::Element]) -> (&[Self::Element], &[Self], &[Self::Element]) {
        if const { size_of::<Self>() != (size_of::<Self::Element>() * Self::LANES) } {
            return (slice, &[], &[]);
        };

        unsafe { slice.align_to() }
    }

    /// Align a mutable slice of elements to the vector's lane count, returning the aligned portion and any unaligned head or tail.
    ///
    /// If the vector's size in bytes does not match the size of its elements times the lane count, this will
    /// return the entire slice as unaligned and empty aligned/remaining parts. This is rare, but may occur
    /// if using a generic vector type that doesn't correspond to an actual hardware vector (for example, a 3-lane vector).
    #[inline(always)]
    fn align_slice_mut(slice: &mut [Self::Element]) -> (&mut [Self::Element], &mut [Self], &mut [Self::Element]) {
        if const { size_of::<Self>() != (size_of::<Self::Element>() * Self::LANES) } {
            return (slice, &mut [], &mut []);
        };

        unsafe { slice.align_to_mut() }
    }

    /// Create a new vector from a slice of elements. The slice must have at least as many elements as the vector's lanes.
    ///
    /// This will emit an unaligned load.
    ///
    /// If you're looking for masked variants of this, those typically only exist for aligned inputs,
    /// so you'll need an aligned pointer and use [`load_m`](Self::load_m) or [`load_z`](Self::load_z).
    fn from_slice(slice: &[Self::Element]) -> Self {
        assert!(slice.len() >= Self::lanes(), "Slice must have at least {} elements to create a vector", Self::lanes());

        unsafe { Self::load_unaligned(slice.as_ptr()) }
    }

    /// Copy the elements of the vector into a slice. The slice must have at least as many elements as the vector's lanes.
    ///
    /// This will emit an unaligned store.
    fn copy_to_slice(self, slice: &mut [Self::Element]) {
        assert!(slice.len() >= Self::lanes(), "Slice must have at least {} elements to copy from a vector", Self::lanes());

        unsafe { self.store_unaligned(slice.as_mut_ptr()) }
    }

    /// Transform a slice of element values into an unaligned iterator of vectors,
    /// returning any remaining elements as a suffix slice.
    fn iter_unaligned<'a>(values: &'a [Self::Element]) -> (unaligned::Unaligned<'a, Self>, &'a [Self::Element]) {
        let num_vectors = values.len() / Self::lanes();
        let offset = num_vectors * Self::lanes();

        let head = &values[..offset];
        let tail = &values[offset..];

        (unaligned::Unaligned(head), tail)
    }

    /// Transform a mutable slice of element values into an unaligned iterator of vectors,
    /// returning any remaining elements as a suffix slice.
    fn iter_mut_unaligned<'a>(values: &'a mut [Self::Element]) -> (unaligned::UnalignedMut<'a, Self>, &'a mut [Self::Element]) {
        let num_vectors = values.len() / Self::lanes();
        let offset = num_vectors * Self::lanes();

        let (head, tail) = values.split_at_mut(offset);

        (unaligned::UnalignedMut(head), tail)
    }

    /// Iterate over a slice of element values as Vectors using non-temporal (streaming) loads.
    ///
    /// # Panics
    ///
    /// If the slice is not aligned to the register type of the vector, or has remaining elements.
    fn stream_aligned_slice<'a>(values: &'a [Self::Element]) -> impl DoubleEndedIterator<Item = streaming::StreamingVector<'a, Self>> {
        let (&[], values, &[]) = Self::align_slice(values) else {
            panic!("Slice is not aligned to the vector type, or has remaining elements");
        };

        values.iter().map(|v| streaming::StreamingVector(v))
    }

    /// Iterate over a mutable slice of element values as Vectors using non-temporal (streaming) loads and stores.
    ///
    /// # Panics
    ///
    /// If the slice is not aligned to the register type of the vector, or has remaining elements.
    fn stream_aligned_slice_mut<'a>(values: &'a mut [Self::Element]) -> impl DoubleEndedIterator<Item = streaming::StreamingVectorMut<'a, Self>> {
        let (&mut [], values, &mut []) = Self::align_slice_mut(values) else {
            panic!("Slice is not aligned to the vector type, or has remaining elements");
        };

        values.iter_mut().map(|v| streaming::StreamingVectorMut(v))
    }

    /// Gather elements from memory at the specified indices and return a new vector with those elements.
    ///
    /// The provided indices are in number of elements, not bytes.
    ///
    /// # Panics
    /// If any index is out of bounds for the slice length, or the slice length exceeds
    /// the maximum supported index for this vector type.
    fn gather<I: VectorIndices<Self>>(slice: &[Self::Element], indices: I) -> Self {
        if indices.cmp_lt(Self::len_to_indices::<I>(slice.len())).all() {
            unsafe { I::gather_ptr(slice.as_ptr(), indices) }
        } else {
            #[cfg(feature = "std")]
            panic!("One or more indices are out of bounds for the slice length {}", slice.len());

            #[cfg(not(feature = "std"))] // avoid fmt
            panic!("One or more indices are out of bounds for the slice length");
        }
    }

    /// Gather elements from memory at the specified indices, or return `or` if the index is out of bounds.
    ///
    /// The provided indices are in number of elements, not bytes.
    ///
    /// # Panics
    /// If the slice length exceeds the maximum supported index for this vector type.
    fn gather_or<I: VectorIndices<Self>>(slice: &[Self::Element], indices: I, or: Self) -> Self
        where Self::Mask: CastMask<I::Mask>,
    {
        let in_bounds = indices.cmp_lt(Self::len_to_indices::<I>(slice.len()));

        unsafe { I::gather_ptr_m(or, in_bounds.cast(), slice.as_ptr(), indices) }
    }

    /// Gather elements from memory at the specified indices, or set the lane to zero
    /// if the index is out of bounds.
    ///
    /// The provided indices are in number of elements, not bytes.
    ///
    /// # Panics
    /// If the slice length exceeds the maximum supported index for this vector type.
    fn gather_or_zero<I: VectorIndices<Self>>(slice: &[Self::Element], indices: I) -> Self
        where Self::Mask: CastMask<I::Mask>,
    {
        let in_bounds = indices.cmp_lt(Self::len_to_indices::<I>(slice.len()));

        unsafe { I::gather_ptr_z(in_bounds.cast(), slice.as_ptr(), indices) }
    }

    /// Gather elements from memory at the specified indices, or return `or` if the `enable` mask is
    /// `false` OR if any index is out of bounds.
    ///
    /// The provided indices are in number of elements, not bytes.
    ///
    /// # Panics
    /// If the slice length exceeds the maximum supported index for this vector type.
    fn gather_if<I: VectorIndices<Self>>(slice: &[Self::Element], enable: Self::Mask, indices: I, or: Self) -> Self
    where
        Self::Mask: CastMask<I::Mask>,
        Self::Element: Default,
    {
        let in_bounds = indices.cmp_lt(Self::len_to_indices::<I>(slice.len()));

        unsafe { I::gather_ptr_m(or, enable & in_bounds.cast(), slice.as_ptr(), indices) }
    }

    /// Scatter elements from the given vector into memory at the specified indices. If the index is outside of the
    /// bounds of the provided slice, the write is suppressed without panicking.
    fn scatter<I: VectorIndices<Self>>(self, slice: &mut [Self::Element], indices: I)
        where Self::Mask: CastMask<I::Mask>,
    {
        let in_bounds = indices.cmp_lt(Self::len_to_indices::<I>(slice.len()));

        unsafe { I::scatter_ptr_m(self, in_bounds.cast(), slice.as_mut_ptr(), indices) }
    }

    /// Scatter elements from the given vector into memory at the specified indices, but only for lanes where the `enable` mask is `true`.
    /// If the index is outside of the bounds of the provided slice, the write is suppressed without panicking.
    fn scatter_if<I: VectorIndices<Self>>(self, slice: &mut [Self::Element], enable: Self::Mask, indices: I)
        where Self::Mask: CastMask<I::Mask>,
    {
        let in_bounds = indices.cmp_lt(Self::len_to_indices::<I>(slice.len()));

        unsafe { I::scatter_ptr_m(self, enable & in_bounds.cast(), slice.as_mut_ptr(), indices) }
    }

    /// Load a vector from an **aligned** pointer to its elements.
    ///
    /// # SAFETY
    /// The caller must ensure that the pointer is valid, aligned, and points to a memory region
    /// that is at least `Self::Lanes` elements long.
    #[masked] unsafe fn load(ptr: *const Self::Element) -> Self;

    /// Load a vector from an **unaligned** pointer to its elements.
    ///
    /// # SAFETY
    /// The caller must ensure that the pointer is valid and points to a memory region
    /// that is at least `Self::Lanes` elements long.
    ///
    /// Unaligned access may be slower on some older architectures.
    unsafe fn load_unaligned(ptr: *const Self::Element) -> Self;

    /// Load a vector from a pointer to its elements using non-temporal (streaming) loads.
    ///
    /// The memory region should not be accessed frequently by the CPU,
    /// as non-temporal loads are intended for data that will not be reused soon.
    ///
    /// # SAFETY
    /// The caller must ensure that the pointer is valid, aligned, and points to a memory region
    /// that is at least `Self::Lanes` elements long.
    unsafe fn load_streaming(ptr: *const Self::Element) -> Self;

    /// Store the vector to an **aligned** pointer to its elements.
    ///
    /// # SAFETY
    /// The caller must ensure that the pointer is valid, aligned, and points to a memory region
    /// that is at least `Self::Lanes` elements long.
    unsafe fn store(self, ptr: *mut Self::Element);

    /// Store the vector to an **aligned** pointer to its elements, but only for lanes where the corresponding mask lane is `true`.
    /// For lanes where the mask is `false`, the store is suppressed without panicking.
    ///
    /// # SAFETY
    /// The caller must ensure that the pointer is valid, aligned, and points to a memory region
    /// that is at least `Self::Lanes` elements long (or at least as long as the number of `true` lanes in the mask).
    unsafe fn store_masked(self, mask: Self::Mask, ptr: *mut Self::Element);

    /// Store the vector to an **unaligned** pointer to its elements.
    ///
    /// # SAFETY
    /// The caller must ensure that the pointer is valid and points to a memory region
    /// that is at least `Self::Lanes` elements long. Unaligned access may be slower on some architectures.
    ///
    /// Unaligned access may be slower on some older architectures.
    unsafe fn store_unaligned(self, ptr: *mut Self::Element);

    /// Store the vector to a pointer to its elements using non-temporal (streaming) stores.
    ///
    /// The memory region should not be accessed frequently by the CPU,
    /// as non-temporal stores are intended for data that will not be reused soon.
    ///
    /// # SAFETY
    /// The caller must ensure that the pointer is valid, aligned, and points to a memory region
    /// that is at least `Self::Lanes` elements long.
    unsafe fn store_streaming(self, ptr: *mut Self::Element);

    /// Interleave two vectors at **group granularity**: blocks of `GROUP` consecutive elements move
    /// as a unit and are never split. `GROUP == 1` is [`interleave`](Self::interleave); `GROUP == 2`
    /// is the complex interleave - `lo == [a.c0, b.c0, a.c1, b.c1, ...]` over the low half of the
    /// groups, `hi` over the high half - which lowers to the doubled-element unpack (`unpacklo_pd` +
    /// `permute2f128` on AVX2, `zip` on NEON) rather than a general permute. The primitive for
    /// complex FFT transposes and any group-structured SIMD. `GROUP` must divide `LANES`.
    ///
    /// The register-level default forwards `GROUP == 1` to [`interleave`](Self::interleave) and uses
    /// a lane-wise fallback otherwise; backends override the group sizes they do natively.
    fn interleave_by<const GROUP: usize>(self, other: Self) -> (Self, Self);

    /// The inverse of [`interleave_by`](Self::interleave_by) - group-granularity de-interleave.
    fn deinterleave_by<const GROUP: usize>(self, other: Self) -> (Self, Self);

    /// Radix-`N` interleave: the generic sibling of [`interleave`](Self::interleave)
    /// (`N == 2`). Treats the `N` inputs as one contiguous `N * LANES` span and
    /// gives `out` with `concat(out)[q * N + r] == inputs[r].extract(q)`.
    ///
    /// `N` is inferred from the array length, so no turbofish is needed:
    /// `V::interleave_radix([a, b])` is the 2-way interleave. `N == 2` reuses the
    /// native `interleave`, `N == 3` a native radix-3 register sequence; any other
    /// `N` uses a single permute+blend gather. For the AoS<->SoA memory form over
    /// arbitrary `N`, use [`load_deinterleaved`](Self::load_deinterleaved) /
    /// [`store_interleaved`](Self::store_interleaved) instead.
    fn interleave_radix<const N: usize>(inputs: [Self; N]) -> [Self; N];

    /// The inverse of [`interleave_radix`](Self::interleave_radix) - radix-`N`
    /// de-interleave: `out[r].extract(q) == concat(inputs)[q * N + r]`.
    fn deinterleave_radix<const N: usize>(inputs: [Self; N]) -> [Self; N];

    /// Group-granularity radix-`N` de-interleave: the two-axis unification of
    /// [`deinterleave_radix`](Self::deinterleave_radix) (`GROUP == 1`) and
    /// [`deinterleave_by`](Self::deinterleave_by) (`N == 2`). Each vector is viewed
    /// as `LANES / GROUP` groups of `GROUP` consecutive elements; `out[r]` group `q`
    /// is the `(q * N + r)`-th group of the concatenated input sequence, each group
    /// moving as a unit.
    ///
    /// The square case `N == LANES / GROUP` is a register-array transpose of
    /// `GROUP`-wide elements: `deinterleave_radix_by::<4, 2>` on 8-lane f32 is the
    /// 4x4 interleaved-complex transpose (8 ops on AVX2), and
    /// `deinterleave_radix_by::<4, 1>` on f64x4 is the 4x4 `f64` transpose - the
    /// primitives for FFT codelets and small matrices. `GROUP` must divide `LANES`.
    fn deinterleave_radix_by<const N: usize, const GROUP: usize>(inputs: [Self; N]) -> [Self; N];

    /// The inverse of [`deinterleave_radix_by`](Self::deinterleave_radix_by) -
    /// group-granularity radix-`N` interleave. For the square case it is the same
    /// (self-inverse) register-array transpose.
    fn interleave_radix_by<const N: usize, const GROUP: usize>(inputs: [Self; N]) -> [Self; N];

    /// Load `N` interleaved (array-of-structures) streams and de-interleave them
    /// into `N` vectors: reads `N * LANES` contiguous elements from `ptr` and
    /// returns `out` with `out[j].extract(lane) == ptr[lane * N + j]`.
    ///
    /// The AoS -> SoA load. `N == 3` over `f32` is the classic case: a
    /// `[[f32; 3]]` of `xyzxyzxyz...` becomes one vector each of `xxx`, `yyy`,
    /// `zzz`. ARM lowers this to a single `LD2`/`LD3`/`LD4` (the de-interleave
    /// happens in the load unit); elsewhere it is contiguous loads plus a
    /// cross-register permute.
    ///
    /// No alignment is required beyond that of `Element`.
    ///
    /// # SAFETY
    /// `ptr` must be valid for reads of `N * LANES` elements.
    unsafe fn load_deinterleaved<const N: usize>(ptr: *const Self::Element) -> [Self; N];

    /// Interleave `N` vectors and store them contiguously as an
    /// array-of-structures: writes `N * LANES` elements such that
    /// `ptr[lane * N + j] == values[j].extract(lane)`.
    ///
    /// The SoA -> AoS store, and the exact inverse of
    /// [`load_deinterleaved`](Self::load_deinterleaved). Lowers to `ST2`/`ST3`/`ST4`
    /// on ARM. No alignment is required beyond that of `Element`.
    ///
    /// # SAFETY
    /// `ptr` must be valid for writes of `N * LANES` elements.
    unsafe fn store_interleaved<const N: usize>(ptr: *mut Self::Element, values: [Self; N]);

    /// Load `M` interleaved AoS records of `C` components each and de-interleave
    /// them: reads `M * C * LANES` contiguous elements, and `out[j][c]` holds
    /// component `c` of record `j`
    /// (`out[j][c].extract(lane) == ptr[lane * M * C + j * C + c]`).
    ///
    /// This is the AoS -> SoA load for structured data: an array of 3D points is
    /// `M = 1, C = 3`; an array of rays (origin + direction) is `M = 2, C = 3`.
    /// See
    /// [`Register::load_deinterleaved_arrays`](crate::register::Register::load_deinterleaved_arrays)
    /// for how a backend serves it (NEON: an `LD3` per chunk).
    ///
    /// The default is a lane-wise gather - correct for ANY vector type, but
    /// scalar. [`Vector`] overrides it with the register engine.
    ///
    /// # SAFETY
    /// `ptr` must be valid for reads of `M * C * LANES` elements.
    unsafe fn load_deinterleaved_arrays<const M: usize, const C: usize>(
        ptr: *const Self::Element,
    ) -> [[Self; C]; M] {
        const { assert!(M >= 1 && C >= 1) };

        let mut out = [[Self::EMPTY; C]; M];

        let mut j = 0;
        while j < M {
            let mut c = 0;
            while c < C {
                let mut v = Self::EMPTY;

                let mut lane = 0;
                while lane < Self::LANES {
                    v = v.insertv(lane, unsafe { ptr.add(lane * (M * C) + j * C + c).read_unaligned() });
                    lane += 1;
                }

                out[j][c] = v;
                c += 1;
            }
            j += 1;
        }

        out
    }

    /// Interleave `M` records of `C` components and store them contiguously - the
    /// exact inverse of
    /// [`load_deinterleaved_arrays`](Self::load_deinterleaved_arrays), with the
    /// same lane-wise default.
    ///
    /// # SAFETY
    /// `ptr` must be valid for writes of `M * C * LANES` elements.
    unsafe fn store_interleaved_arrays<const M: usize, const C: usize>(ptr: *mut Self::Element, values: [[Self; C]; M]) {
        const { assert!(M >= 1 && C >= 1) };

        let mut j = 0;
        while j < M {
            let mut c = 0;
            while c < C {
                let v = values[j][c];

                let mut lane = 0;
                while lane < Self::LANES {
                    unsafe { ptr.add(lane * (M * C) + j * C + c).write_unaligned(v.extractv(lane)) };
                    lane += 1;
                }
                c += 1;
            }
            j += 1;
        }
    }

    /// Load `M` interleaved composite streams of `1 + TAIL` components each and
    /// de-interleave them into `M` [`StreamGroup`]s: reads
    /// `M * (TAIL + 1) * LANES` contiguous elements, and group `j`'s
    /// `head`/`tail[c - 1]` hold the de-interleaved components of composite
    /// stream `j`. See [`StreamGroup`] for why the component count is a
    /// separate const generic, and
    /// [`Register::load_deinterleaved_grouped`](crate::register::Register::load_deinterleaved_grouped)
    /// for the register-level strategy.
    ///
    /// The default is a lane-wise gather: correct for ANY vector type, but
    /// scalar. [`Vector`] overrides it with the register engine; a composite
    /// vector (dual numbers, compensated floats) instead implements its plain
    /// [`load_deinterleaved`](Self::load_deinterleaved) by calling *its inner
    /// vector's* grouped op with the composite's component count folded into
    /// `TAIL`. Only a composite nested inside another composite ever reaches
    /// this default - at that point layout-aware shuffling has run out of road,
    /// and correctness is all that is on offer.
    ///
    /// # SAFETY
    /// `ptr` must be valid for reads of `M * (TAIL + 1) * LANES` elements.
    unsafe fn load_deinterleaved_grouped<const M: usize, const TAIL: usize>(
        ptr: *const Self::Element,
    ) -> [StreamGroup<Self, TAIL>; M] {
        const { assert!(M >= 1) };

        let c = TAIL + 1;

        let mut out = [StreamGroup { head: Self::EMPTY, tail: [Self::EMPTY; TAIL] }; M];

        let mut j = 0;
        while j < M {
            let mut comp = 0;
            while comp < c {
                let mut v = Self::EMPTY;

                let mut lane = 0;
                while lane < Self::LANES {
                    v = v.insertv(lane, unsafe { ptr.add(lane * (M * c) + j * c + comp).read_unaligned() });
                    lane += 1;
                }

                if comp == 0 {
                    out[j].head = v;
                } else {
                    out[j].tail[comp - 1] = v;
                }
                comp += 1;
            }
            j += 1;
        }

        out
    }

    /// Interleave `M` [`StreamGroup`]s and store them as a contiguous
    /// array-of-structures - the exact inverse of
    /// [`load_deinterleaved_grouped`](Self::load_deinterleaved_grouped), with
    /// the same lane-wise default and the same override expectations.
    ///
    /// # SAFETY
    /// `ptr` must be valid for writes of `M * (TAIL + 1) * LANES` elements.
    unsafe fn store_interleaved_grouped<const M: usize, const TAIL: usize>(
        ptr: *mut Self::Element,
        values: [StreamGroup<Self, TAIL>; M],
    ) {
        const { assert!(M >= 1) };

        let c = TAIL + 1;

        let mut j = 0;
        while j < M {
            let mut comp = 0;
            while comp < c {
                let v = if comp == 0 { values[j].head } else { values[j].tail[comp - 1] };

                let mut lane = 0;
                while lane < Self::LANES {
                    unsafe { ptr.add(lane * (M * c) + j * c + comp).write_unaligned(v.extractv(lane)) };
                    lane += 1;
                }
                comp += 1;
            }
            j += 1;
        }
    }

    /// Assemble a vector from a slice of elements and a vector of indices
    /// into that slice. If an index is outside the bounds of the given slice,
    /// the resulting lane will be the first element of the input slice.
    ///
    /// This is semantically equivalent to `gather`, but specialized for small lookup tables approximately
    /// the same size as the vector itself. If the lookup table is too large, it will fall back to `gather`.
    fn lookup(values: &[Self::Element], indices: Self::Unsigned) -> Self {
        let in_bounds = indices.cmp_lt(Self::len_to_indices::<Self::Unsigned>(values.len()));

        unsafe { Self::lookup_unchecked(values, indices.zz(in_bounds)) }
    }

    /// Assemble a vector from a slice of elements and a vector of indices
    /// into that slice. The indices are NOT checked to be within bounds.
    ///
    /// # Safety
    /// The caller must ensure that the indices are within bounds for the given values slice,
    /// otherwise this may panic or result in undefined behavior.
    unsafe fn lookup_unchecked(values: &[Self::Element], indices: Self::Unsigned) -> Self;

    /// Broadcast the value of a single lane across all lanes of the vector.
    #[conditional] fn broadcast<const I: usize>(self) -> Self;

    /// Broadcast the value of a single lane across all lanes of the vector.
    ///
    /// # Panics
    /// If `idx` is out of bounds for the vector's lanes.
    #[conditional] fn broadcastv(self, idx: usize) -> Self;

    /// Extract a single element from the vector at the const-generic index `I`.
    ///
    /// Because `I` is known at compile time, the backend can lower this to a
    /// single instruction (e.g. `pextrd`) with no runtime branch.
    ///
    /// # Compile-time errors
    /// `I` must be less than [`LANES`](Self::LANES).
    fn extract<const I: usize>(self) -> Self::Element;

    /// Extract a single element from the vector at the runtime index `idx`.
    ///
    /// Prefer [`extract`](Self::extract) when the index is known at compile
    /// time; this variant typically lowers to a small jump table or per-lane
    /// blend and is slower.
    ///
    /// # Panics
    /// If `idx` is out of bounds for the vector's lanes.
    fn extractv(self, idx: usize) -> Self::Element;

    /// Replace a single element in the vector at the const-generic index `I`.
    ///
    /// Returns a new vector; the original is unmodified. The lane index is
    /// resolved at compile time.
    ///
    /// # Compile-time errors
    /// `I` must be less than [`LANES`](Self::LANES).
    fn insert<const I: usize>(self, value: Self::Element) -> Self;

    /// Replace a single element in the vector at the runtime index `idx`.
    ///
    /// Prefer [`insert`](Self::insert) when the index is known at compile time.
    ///
    /// # Panics
    /// If `idx` is out of bounds for the vector's lanes.
    fn insertv(self, idx: usize, value: Self::Element) -> Self;

    /// Reverse the order of the elements in the vector.
    ///
    /// For a vector `[a, b, c, d]` this returns `[d, c, b, a]`.
    #[conditional] fn reverse(self) -> Self;

    /// Swap the byte order of each element in the vector, converting between
    /// little-endian and big-endian representations lane-by-lane.
    ///
    /// Only the bytes within each element are reordered; lane order is
    /// preserved. For a `u32` vector `[0x11223344]` this returns `[0x44332211]`.
    #[conditional] fn swap_bytes(self) -> Self;

    /// (Zero If False) Zero elements if the corresponding mask lane is false; otherwise, leave unchanged.
    ///
    /// Similar to a `mask & self` operation.
    fn zz(self, mask: Self::Mask) -> Self;

    /// (Zero If True) Zero elements if the corresponding mask lane is true; otherwise, leave unchanged.
    ///
    /// Similar to a `!mask & self` operation.
    fn nz(self, mask: Self::Mask) -> Self;

    /// Construct a mask whose first `n` lanes are `true` and the remaining
    /// lanes `false`.
    ///
    /// `n` is clamped to [`LANES`](Self::LANES): `n >= LANES` yields an
    /// all-`true` mask and `n == 0` an all-`false` mask. This is the canonical
    /// tail-handling helper - given a remainder of `k < LANES` elements,
    /// `Self::prefix_mask(k)` selects exactly those lanes for a masked store,
    /// [`select`](crate::mask::GenericMask::select), or `_c`/`_m`/`_z`
    /// operation.
    ///
    /// Semantically `Self::indexed() < n` lifted into the mask domain, but
    /// available on any [`GenericVector`] (the predicate is built on
    /// [`Unsigned`](Self::Unsigned), so it does not require `Self: NumericVector`).
    #[inline(always)]
    fn prefix_mask(n: usize) -> Self::Mask {
        let n = if n > Self::lanes() { Self::lanes() } else { n };
        let limit = Self::len_to_indices::<Self::Unsigned>(n);
        Self::Unsigned::indexed().cmp_lt(limit).cast::<Self::Mask>()
    }

    /// Construct a mask whose last `n` lanes are `true` and the remaining
    /// lanes `false`.
    ///
    /// `n` is clamped to [`LANES`](Self::LANES). This is the high-lane
    /// companion to [`prefix_mask`](Self::prefix_mask); for example
    /// `Self::suffix_mask(2)` on a 4-lane vector selects lanes 2 and 3.
    #[inline(always)]
    fn suffix_mask(n: usize) -> Self::Mask {
        let n = if n > Self::lanes() { Self::lanes() } else { n };
        let start = Self::len_to_indices::<Self::Unsigned>(Self::lanes() - n);
        Self::Unsigned::indexed().cmp_ge(start).cast::<Self::Mask>()
    }

    /// Left-pack (a.k.a. `compress`): gather the lanes where `mask` is `true`
    /// into the low lanes, preserving their relative order. The unselected lanes
    /// are *kept* (not zeroed) and packed into the high lanes, also in order - a
    /// stable partition of the vector by `mask`.
    ///
    /// For `[a, b, c, d]` with `mask = [true, false, true, false]` this returns
    /// `[a, c, b, d]`. Combined with a masked store of the leading `mask`-count
    /// lanes, this is the building block for stream compaction - whitespace
    /// stripping, filtering, JSON minification, and similar. For the zero-filled
    /// tail variant, see [`compress_z`](Self::compress_z).
    ///
    /// Lowers to AVX-512 `vpcompress*` where available; otherwise a portable
    /// scalar partition (some backends accelerate it with a permute table).
    fn compress(self, mask: Self::Mask) -> Self;

    /// Zero-filling left-pack: like [`compress`](Self::compress), but the lanes
    /// beyond the `mask` population count are zeroed instead of holding the
    /// unselected elements. Matches AVX-512 zero-masking `vpcompress*`.
    ///
    /// For `[a, b, c, d]` with `mask = [true, false, true, false]` this returns
    /// `[a, c, 0, 0]`.
    fn compress_z(self, mask: Self::Mask) -> Self;

    /// Two-register element align (the `palignr` family): the window of `LANES`
    /// lanes starting at lane `OFFSET` of the concatenation `[self, other]`
    /// (`self`'s lanes first, then `other`'s). `OFFSET == 0` returns `self`,
    /// `OFFSET == LANES` returns `other`; in between, lanes spill from the tail
    /// of `self` into the head of `other`.
    ///
    /// The cross-register sliding window used for scanning multi-byte
    /// delimiters / substrings across a load boundary. Works for any element
    /// type (it is pure lane movement); integer backends accelerate it with
    /// native byte aligns.
    fn align<const OFFSET: usize>(self, other: Self) -> Self;

    /// Apply a function to each element in the vector, returning a new vector with the results.
    ///
    /// This is not explicitly SIMD-optimized, so may be slower than using native vector operations.
    fn map<F>(self, f: F) -> Self
    where
        F: Fn(Self::Element) -> Self::Element;

    /// Fold the elements of the vector using the provided function and initial value.
    ///
    /// This is not explicitly SIMD-optimized, so may be slower than using native vector operations.
    fn fold<F>(self, init: Self::Element, f: F) -> Self::Element
    where
        F: Fn(Self::Element, Self::Element) -> Self::Element;

    /// Reduce the elements of the vector using the provided function.
    ///
    /// This is not explicitly SIMD-optimized, so may be slower than using native vector operations.
    fn reduce<F>(self, f: F) -> Self::Element
    where
        F: Fn(Self::Element, Self::Element) -> Self::Element;

    /// Numeric cast to another vector type, matching the semantics of Rust's
    /// `as` operator on the underlying scalar elements.
    ///
    /// Lane count is preserved; only the element type changes. The cast may
    /// be widening, narrowing, signed/unsigned, or float/int. Out-of-range
    /// float-to-int conversions follow the same saturating behavior as
    /// scalar `as` on the host backend.
    #[inline(always)] fn cast<INTO>(self) -> INTO
    where
        INTO: CastVector<Self>,
    {
        INTO::cast_from(self)
    }

    /// Fast numeric cast to another vector type.
    ///
    /// Equivalent to [`cast`](Self::cast) when the backend has no faster path,
    /// but may relax IEEE corner cases (NaN propagation, out-of-range
    /// float-to-int handling) in exchange for fewer instructions.
    ///
    /// Use [`cast`](Self::cast) when you need the documented `as` semantics
    /// exactly; use this when you have already ruled out problematic inputs.
    #[inline(always)] fn fast_cast<INTO>(self) -> INTO
    where
        INTO: CastVector<Self>,
    {
        INTO::fast_cast_from(self)
    }

    /// Reinterpret the bits of this vector as another vector type of the same
    /// size and lane count.
    ///
    /// This is a zero-cost transmute; no conversion is performed. Typical use
    /// is moving between a float vector and its integer "bits" vector for
    /// bit-level manipulation.
    #[inline(always)] fn into_bits<INTO>(self) -> INTO
    where
        INTO: BitCastVector<Self>,
    {
        INTO::from_bits(self)
    }

    /// Narrowing cast that saturates (clamps) out-of-range values to the
    /// destination element range, rather than wrapping like [`cast`](Self::cast).
    ///
    /// Only resolves for narrowing, same-signedness integer conversions
    /// (`i64 -> ... -> i8`, `u64 -> ... -> u8`); widening or sign-changing
    /// casts have no `SaturatingCastVector` impl and must use [`cast`](Self::cast).
    /// See [`SaturatingCastVector`].
    #[inline(always)] fn saturating_cast<INTO>(self) -> INTO
    where
        INTO: SaturatingCastVector<Self>,
    {
        INTO::saturating_cast_from(self)
    }
}

#[rustfmt::skip] #[thermite_macros::vector_trait]
pub trait BitwiseVector:
    GenericVector
    + ops::BitAndMasked<Self::Mask, Self, Output = Self>
    + ops::BitAndAssignMasked<Self::Mask, Self>
    + ops::BitAndNotMasked<Self::Mask, Self, Output = Self>
    + ops::BitAndNotAssignMasked<Self::Mask, Self>
    + ops::BitOrMasked<Self::Mask, Self, Output = Self>
    + ops::BitOrAssignMasked<Self::Mask, Self>
    + ops::BitXorMasked<Self::Mask, Self, Output = Self>
    + ops::BitXorAssignMasked<Self::Mask, Self>
    + ops::NotMasked<Self::Mask, Output = Self>
{
    /// Computes an arbitrary bitwise boolean function of three inputs (`a`, `b`, `c`)
    /// based on the truth table specified by `IMM`.
    ///
    /// This function is a "programmable logic gate". It applies the logic defined in `IMM`
    /// to every bit of the inputs in parallel.
    ///
    /// # How to Calculate `IMM`
    /// The easiest way to find the correct `IMM` value is to perform your desired boolean
    /// logic on these three specific "Magic Constants":
    ///
    /// * **A** = `0xF0` (Binary `11110000`)
    /// * **B** = `0xCC` (Binary `11001100`)
    /// * **C** = `0xAA` (Binary `10101010`)
    ///
    /// ## Example: `(A OR B) XOR C`
    /// 1. `A | B` = `0xF0 | 0xCC` = `0xFC`
    /// 2. `Result ^ C` = `0xFC ^ 0xAA` = `0x56`
    /// 3. Therefore, `IMM = 0x56`.
    ///
    /// You can also use the [`ternlog_imm!`](crate::ternlog_imm) macro to compute
    /// this at compile time.
    ///
    /// # Visualization using Disjunction Normal Form (DNF)
    /// The constants `0xF0`, `0xCC`, and `0xAA` simply form a parallel truth table
    /// for all 8 possible combinations of 3 bits:
    ///
    /// |  A  |  B  |  C  |  Bit Index  |  Term Logic (Minterm) |
    /// |:---:|:---:|:---:|:-----------:|:---------------------:|
    /// |  0  |  0  |  0  |      0      | ~A & ~B & ~C          |
    /// |  0  |  0  |  1  |      1      | ~A & ~B &  C          |
    /// |  0  |  1  |  0  |      2      | ~A &  B & ~C          |
    /// |  0  |  1  |  1  |      3      | ~A &  B &  C          |
    /// |  1  |  0  |  0  |      4      |  A & ~B & ~C          |
    /// |  1  |  0  |  1  |      5      |  A & ~B &  C          |
    /// |  1  |  1  |  0  |      6      |  A &  B & ~C          |
    /// |  1  |  1  |  1  |      7      |  A &  B &  C          |
    ///
    /// If `IMM = 0x88` (Bit 3 and 7 set), the logic is:
    /// - Bit 3 (0, 1, 1): `~A & B & C`
    /// - Bit 7 (1, 1, 1): `A & B & C`
    ///
    /// As raw DNF, this becomes: `(~A & B & C) | (A & B & C)`.\
    /// `~A` and `A` cancel out, simplifying to `B & C`.
    ///
    /// For each bit set in IMM, we effectively bitwise-OR each corresponding minterm.
    ///
    /// # Common Immediate Values
    /// | Logic | Immediate | Description |
    /// | :--- | :--- | :--- |
    /// | `A ^ B ^ C` | `0x96` | **3-Way XOR** (Parity) |
    /// | `(A & B) OR (~A & C)` | `0xCA` | **Bitwise Select** (If A=1 use B, else use C) |
    /// | `(A & B) OR (A & C) OR (B & C)` | `0xE8` | **Majority** (True if 2+ inputs are 1) |
    /// | `A OR B OR C` | `0xFE` | **3-Way OR** |
    /// | `A ? B : 0` | `0xA0` | **Mask** (A & B) |
    ///
    /// # Performance Note
    /// Since `IMM` is a compile-time constant, the compiler will optimize this function
    /// into the most efficient sequence of native instructions (AND, OR, XOR, NOT)
    /// for your specific architecture. If using AVX512, there actually exists a single
    /// instruction for this.
    #[conditional] fn ternlog<const IMM: i32>(a: Self, b: Self, c: Self) -> Self;

    /// Two-input version of [`ternlog`](Self::ternlog).
    ///
    /// Computes an arbitrary bitwise boolean function of two inputs (`a`, `b`)
    /// based on the 4-bit truth table specified by the low nibble of `IMM`.
    /// Bit `i` of `IMM` selects the output when `(a, b)` equals the binary
    /// representation of `i`. As with `ternlog`, the magic constants are
    /// `A = 0xC` (`1100`) and `B = 0xA` (`1010`); evaluate your desired logic
    /// against them to obtain `IMM`. For example, `A & B == 0x8`, `A | B == 0xE`,
    /// `A ^ B == 0x6`, `!A == 0x3`.
    ///
    /// Since `IMM` is a compile-time constant, the compiler lowers this to
    /// the most efficient native instruction sequence for the target ISA.
    #[conditional] fn bilog<const IMM: i32>(a: Self, b: Self) -> Self;
}

#[rustfmt::skip] #[thermite_macros::vector_trait]
pub trait BitshiftVector:
    BitwiseVector
    + ops::ShrMasked<Self::Mask, Self::Unsigned, Output = Self>
    + ops::ShrAssignMasked<Self::Mask, Self::Unsigned>
    + ops::ShlMasked<Self::Mask, Self::Unsigned, Output = Self>
    + ops::ShlAssignMasked<Self::Mask, Self::Unsigned>
    + ops::ShrMasked<Self::Mask, u32, Output = Self>
    + ops::ShrAssignMasked<Self::Mask, u32>
    + ops::ShlMasked<Self::Mask, u32, Output = Self>
    + ops::ShlAssignMasked<Self::Mask, u32>
{
    /// `true` if the backend has a true per-lane variable shift instruction
    /// (e.g. AVX2 `vpsllvd`). When `false`, [`shlv`](Self::shlv) /
    /// [`shrv`](Self::shrv) are emulated and may be slower than splatting a
    /// scalar shift count through [`shli`](Self::shli) / [`shri`](Self::shri).
    const HAS_TRUE_SHIFTV: bool;

    /// `true` if the backend can byte-shift the entire vector as a single
    /// large integer at register widths above 128 bits without lane-boundary
    /// stitching. When `false`, [`bshli`](Self::bshli) / [`bshri`](Self::bshri)
    /// on wider vectors are emulated via shuffles.
    const HAS_WIDE_BYTE_SHIFTS: bool;

    /// Treats the entire vector as a single large integer and shifts left by the immediate value
    /// number of BYTES. Not bits, bytes.
    ///
    /// Bits shifted out at the high end are discarded; the low end is zero-filled.
    #[conditional] fn bshli<const I: i32>(self) -> Self;

    /// Treats the entire vector as a single large integer and shifts right by the immediate value
    /// number of BYTES. Not bits, bytes.
    ///
    /// Bits shifted out at the low end are discarded; the high end is zero-filled.
    #[conditional] fn bshri<const I: i32>(self) -> Self;

    /// For each lane in the vector, shift left by the immediate value.
    #[conditional] fn shli<const I: i32>(self) -> Self;

    /// For each lane in the vector, shift right by the immediate value.
    #[conditional] fn shri<const I: i32>(self) -> Self;

    /// For each lane in the vector, shift left by the given value.
    #[conditional] fn shlv(self, counts: Self::Unsigned) -> Self;

    /// For each lane in the vector, shift right by the given value.
    #[conditional] fn shrv(self, counts: Self::Unsigned) -> Self;

    /// For each element in the vector, rotate the bits to the left by the given
    /// number of bits.
    #[conditional] fn rol(self, shift: u32) -> Self;
    /// For each element in the vector, rotate the bits to the right by the given
    /// number of bits.
    #[conditional] fn ror(self, shift: u32) -> Self;
    /// For each element in the vector, rotate the bits to the left by the immediate
    /// value number of bits.
    #[conditional] fn roli<const I: i32>(self) -> Self;
    /// For each element in the vector, rotate the bits to the right by the immediate
    /// value number of bits.
    #[conditional] fn rori<const I: i32>(self) -> Self;

    /// For each element in the vector, rotate the bits to the left by the given
    /// number of bits in the corresponding lane of `counts`.
    #[conditional] fn rolv(self, counts: Self::Unsigned) -> Self;

    /// For each element in the vector, rotate the bits to the right by the given
    /// number of bits in the corresponding lane of `counts`.
    #[conditional] fn rorv(self, counts: Self::Unsigned) -> Self;

    /// For each element in the vector, reverse the bits of that element.
    #[conditional] fn reverse_bits(self) -> Self;
}

/// Per-lane numeric conversion between vector types.
///
/// Implementing `CastVector<FROM>` for `Self` means a `FROM` value can be
/// converted into `Self` with the same semantics as Rust's `as` operator on
/// the underlying scalar elements. Most users should call
/// [`GenericVector::cast`] rather than these methods directly.
pub trait CastVector<FROM: Sized>: Sized {
    /// Convert a vector of type `FROM` into `Self`, lane-by-lane, using `as`
    /// semantics on each element.
    fn cast_from(from: FROM) -> Self;

    /// Convert this vector into a vector of type `FROM`, lane-by-lane.
    fn cast_into(self) -> FROM;

    /// Like [`cast_from`](Self::cast_from), but may take a faster path that
    /// relaxes IEEE corner cases. See [`GenericVector::fast_cast`].
    #[inline(always)]
    fn fast_cast_from(from: FROM) -> Self {
        Self::cast_from(from)
    }

    /// Like [`cast_into`](Self::cast_into), but may take a faster path that
    /// relaxes IEEE corner cases. See [`GenericVector::fast_cast`].
    #[inline(always)]
    fn fast_cast_into(self) -> FROM {
        Self::cast_into(self)
    }
}

/// Zero-cost bit-level reinterpretation between vector types of the same
/// size and lane count.
///
/// Unlike [`CastVector`], no numeric conversion is performed: the underlying
/// bits are reinterpreted as the destination element type. Typical use is
/// moving between a float vector and its integer "bits" vector.
pub trait BitCastVector<FROM: Sized>: Sized {
    /// Reinterpret the bit pattern of `bits` as a value of `Self`.
    fn from_bits(bits: FROM) -> Self;
}

/// Vector-layer mirror of [`SaturatingCastRegister`](crate::register::SaturatingCastRegister):
/// a narrowing, same-signedness cast that clamps out-of-range values to the
/// destination element range instead of wrapping like [`CastVector`].
///
/// Implemented only for narrowing same-sign integer pairs (`i64 -> ... -> i8`,
/// `u64 -> ... -> u8`, including skip-level pairs). Widening and sign-changing
/// conversions are not saturating and go through [`CastVector`]. Most users
/// reach this through [`GenericVector::saturating_cast`] rather than naming the
/// trait directly.
///
/// Blanket-implemented for every `Vector<INTO>` whose register implements
/// [`SaturatingCastRegister<FROM>`](crate::register::SaturatingCastRegister).
pub trait SaturatingCastVector<FROM: Sized>: Sized {
    /// Narrow `from` into `Self`, clamping each lane to `Self`'s element range.
    fn saturating_cast_from(from: FROM) -> Self;
}

/// A `u16`/`u8` integer vector reinterpreted as a vector of *packed floats* (format `S`: fp16,
/// bfloat16, the fp8 variants, ...), transcodable to and from the wider `f32` vector `F` of the
/// same lane count.
///
/// This is the vector-layer mirror of
/// [`PackedFloatRegister`](crate::register::PackedFloatRegister): `Self` is the `Vector<u16/u8
/// register>` and `F` is the matching `Vector<f32 register>`. Both directions are exact for the
/// decode (every value of these sub-`f32` formats is representable in `f32`) and round-to-nearest
/// for the encode; backends use hardware (F16C `vcvtph2ps`) where available and a generic
/// branchless fallback otherwise.
///
/// Blanket-implemented for every `Vector<R>` whose register implements `PackedFloatRegister<S,
/// FR>`, so e.g. `u16x8<S>: PackedFloatVector<Fp16, f32x8<S>>` holds wherever the register does.
///
/// ```
/// # use thermite::prelude::*;
/// # use thermite::element::float::spec::Fp16;
/// # use thermite::vector::PackedFloatVector;
/// fn widen<U, F>(halves: U) -> F
/// where
///     U: PackedFloatVector<Fp16, F>,
/// {
///     halves.unpack()
/// }
/// ```
pub trait PackedFloatVector<S: crate::element::float::spec::FloatSpec, F>: GenericVector {
    /// Encode the `f32` vector `values` into this packed format (round to nearest, ties to even;
    /// overflow / non-finite handled per the format `S`).
    fn pack(values: F) -> Self;

    /// Decode this packed-float vector into the `f32` vector it represents (exact).
    fn unpack(self) -> F;
}

/// Per-lane comparison producing a [`Mask`](GenericVector::Mask).
///
/// Each comparison returns a mask whose lanes are `true` where the predicate
/// held for the corresponding lane pair and `false` otherwise. The mask can
/// then be used with [`select`](crate::mask::GenericMask::select),
/// `_c`/`_m`/`_z` masked variants, or reduced via
/// [`all`](crate::mask::GenericMask::all) /
/// [`any`](crate::mask::GenericMask::any).
///
/// For floating-point vectors, NaN compares unequal to everything, so e.g.
/// `cmp_lt(x, NaN)` is always `false`, matching the `<` operator on `f32`/`f64`.
pub trait PartialOrdVector: GenericVector + PartialEq {
    /// Lane-wise `self < other`.
    fn cmp_lt(self, other: Self) -> Self::Mask;
    /// Lane-wise `self <= other`.
    fn cmp_le(self, other: Self) -> Self::Mask;
    /// Lane-wise `self > other`.
    fn cmp_gt(self, other: Self) -> Self::Mask;
    /// Lane-wise `self >= other`.
    fn cmp_ge(self, other: Self) -> Self::Mask;
    /// Lane-wise `self == other`.
    fn cmp_eq(self, other: Self) -> Self::Mask;
    /// Lane-wise `self != other`.
    fn cmp_ne(self, other: Self) -> Self::Mask;
}

/// Vectors that support arithmetic and comparison operations on their elements.
///
/// This trait sits between [`PartialOrdVector`] and the more specific
/// [`SignedVector`] / [`IntegerVector`] / [`FloatVector`] traits, and provides
/// the operator overloads (`+`, `-`, `*`, `/`, `%`, their `*Assign` variants,
/// and the masked `_c`/`_m`/`_z` forms via [`ops`]).
///
/// # Overflow semantics
///
/// **For integer element types, the basic arithmetic operators (`+`, `-`, `*`,
/// `/`, `%`) are wrapping on overflow.** This matches the behavior of every
/// SIMD ISA (`paddd`, `pmulld`, etc. all wrap silently) and avoids per-lane
/// panics inside vectorized loops. Concretely, on every backend including the
/// scalar reference backend, `Vector::<i32x4>::splat(i32::MAX) + Vector::ONE`
/// produces `i32::MIN` in every lane rather than panicking.
///
/// This is intentional and is **not affected by debug vs release builds**: the
/// scalar backend uses `wrapping_add` / `wrapping_sub` / `wrapping_mul`
/// internally, so the wrapping behavior is consistent across all build
/// configurations. If you need saturation or explicit wrapping naming, use
/// [`saturating_add`](IntegerVector::saturating_add) /
/// [`saturating_sub`](IntegerVector::saturating_sub), or the
/// `num_traits::WrappingAdd` / `WrappingSub` / `WrappingMul` impls.
///
/// Integer division (`/`, `%`) panics on division by zero, matching scalar
/// Rust. Float division by zero produces an infinity or NaN per IEEE 754.
///
/// For float element types, overflow simply produces an infinity per IEEE 754;
/// there is nothing to wrap.
#[rustfmt::skip] #[thermite_macros::vector_trait]
pub trait NumericVector:
    PartialOrdVector<Element: num_traits::NumOps>
    + ops::AddMasked<Self::Mask, Self, Output = Self>
    + ops::AddAssignMasked<Self::Mask, Self>
    + ops::SubMasked<Self::Mask, Self, Output = Self>
    + ops::SubAssignMasked<Self::Mask, Self>
    + ops::MulMasked<Self::Mask, Self, Output = Self>
    + ops::MulAssignMasked<Self::Mask, Self>
    + ops::DivMasked<Self::Mask, Self, Output = Self>
    + ops::DivAssignMasked<Self::Mask, Self>
    + ops::RemMasked<Self::Mask, Self, Output = Self>
    + ops::RemAssignMasked<Self::Mask, Self>
    + ops::SquareMasked<Self::Mask, Output = Self>
    + num_traits::NumOps<Self>
    + num_traits::NumAssignOps<Self>
    + core::iter::Sum
    + core::iter::Product
{
    /// A vector of the value "0" in the element type.
    const ZERO: Self;
    /// A vector of the value "1" in the element type.
    const ONE: Self;
    /// A vector of the value "2" in the element type.
    const TWO: Self;

    /// A vector of the minimum value the element type of this vector can represent.
    const MIN: Self;
    /// A vector of the maximum value the element type of this vector can represent.
    const MAX: Self;

    /// For each element in the vector, return a mask indicating whether that element is zero.
    fn is_zero(self) -> Self::Mask;

    /// Returns `true` if all elements in the vector are zero, `false` otherwise.
    ///
    /// This can often be more performant than naive comparisons or even `is_zero().all()`
    fn is_all_zero(self) -> bool;

    /// Return the minimum of two vectors, element-wise.
    #[conditional] fn min(self, other: Self) -> Self;

    /// Return the maximum of two vectors, element-wise.
    #[conditional] fn max(self, other: Self) -> Self;

    /// Clamps the elements of the vector between the given minimum and maximum values.
    fn clamp(self, min: Self, max: Self) -> Self;

    /// Returns the minimum value in the vector.
    ///
    /// This operation has an `O(log2 n)` complexity to reduce.
    fn min_element(self) -> Self::Element;
    /// Returns the maximum value in the vector.
    ///
    /// This operation has an `O(log2 n)` complexity to reduce.
    fn max_element(self) -> Self::Element;
    /// Returns both the minimum and maximum values in the vector simultaneously.
    ///
    /// More efficient than calling [`min_element`](NumericVector::min_element) and
    /// [`max_element`](NumericVector::max_element) separately when both are needed.
    fn min_max_element(self) -> (Self::Element, Self::Element);

    /// Returns the indices of the minimum and maximum elements in the vector, respectively.
    fn arg_minmax(self) -> (usize, usize);

    /// Scales each element in the vector by the given factor.
    ///
    /// While semantically equivalent to `self * Self::splat(factor)`, this method may be optimized
    /// better on certain architectures, such as GPUs.
    #[conditional] fn scale(self, factor: Self::Element) -> Self;

    /// Sums adjacent lane pairs from `lo` and `hi`, returning a vector of the same width.
    ///
    /// Output: `[lo[0]+lo[1], lo[2]+lo[3], ..., hi[0]+hi[1], hi[2]+hi[3], ...]`
    ///
    /// The result is always in strict order: all pair sums from `lo` followed by all pair sums from `hi`.
    fn pairwise_sum(lo: Self, hi: Self) -> Self;

    /// Like [`pairwise_sum`](NumericVector::pairwise_sum), but may return a relaxed (implementation-defined)
    /// lane ordering for performance. Treat this as if randomly shuffling the result of
    /// [`pairwise_sum`](NumericVector::pairwise_sum), with better performance than `pairwise_sum`.
    ///
    /// Prefer this if you are simply summing any adjacent pairs from `lo` and `hi`, and don't
    /// care about the exact ordering of the resulting sums.
    fn relaxed_pairwise_sum(lo: Self, hi: Self) -> Self;

    /// Returns the sum of all elements in the vector.
    ///
    /// This operation has an `O(log2 n)` complexity to reduce.
    fn sum_elements(self) -> Self::Element;

    /// Returns the product of all elements in the vector.
    ///
    /// This operation has an `O(log2 n)` complexity to reduce.
    fn prod_elements(self) -> Self::Element;

    /// Returns a vector whose every lane equals [`LANES`](GenericVector::LANES),
    /// converted into the element type.
    ///
    /// Equivalent to `Self::splat(Self::LANES as Self::Element)`. Useful for
    /// stepping an [`indexed`](Self::indexed) counter forward by one full
    /// vector's worth of lanes in tight loops.
    fn offset() -> Self;

    /// Returns a vector where each lane holds its own index, cast to the
    /// element type: `[0, 1, 2, ..., LANES-1]`.
    ///
    /// This is the typical starting point for index-based vector loops. The
    /// counter can be advanced by adding [`offset`](Self::offset).
    fn indexed() -> Self;
}

/// Vectors whose elements can represent negative values.
///
/// Adds negation, absolute value, sign extraction, and sign-conditional
/// selection on top of [`NumericVector`]. Implemented for signed integer and
/// floating-point vectors; not for unsigned integer vectors.
///
/// As with the base [`NumericVector`] operators, unary `-` on a signed integer
/// vector is **wrapping**: `-Vector::<i32x4>::splat(i32::MIN)` returns
/// `i32::MIN` in every lane rather than panicking.
// TODO: Add back in some kind of `Signed` trait requirement for Element?
#[rustfmt::skip] #[thermite_macros::vector_trait]
pub trait SignedVector: NumericVector + ops::NegMasked<Self::Mask, Output = Self> {
    /// A vector of the value "-1" in the element type.
    const NEG_ONE: Self;

    /// A vector of the smallest positive (non-zero) value in the element type.
    const MIN_POSITIVE: Self;

    /// Take the absolute value of the vector, element-wise.
    #[conditional] fn abs(self) -> Self;

    /// For each element in the vector, return a new vector
    /// where each element is either -1 or +1 depending
    /// on the sign of the element.
    ///
    /// For integers, this will also return zero (0) if the
    /// element is zero. This matches Rust's behavior for integer
    /// `signum`. Floats remain only -1 or +1.
    fn signum(self) -> Self;

    /// For each element in the vector, set the sign of that
    /// element to the sign of the corresponding element in the other vector.
    #[conditional] fn copysign(self, sign: Self) -> Self;

    /// For each element in the vector, return a mask indicating
    /// whether that element is negative.
    fn is_positive(self) -> Self::Mask;

    /// For each element in the vector, return a mask indicating
    /// whether that element is positive.
    fn is_negative(self) -> Self::Mask;

    /// Based on if self is negative, select between `if_neg` and `if_pos`.
    fn select_negative(self, if_neg: Self, if_pos: Self) -> Self;
}

/// Vectors of integer elements.
///
/// Adds bitwise shifts (via [`BitshiftVector`]), bit-counting, saturating
/// arithmetic, branchfree integer division helpers, and exposes the type of
/// the per-divider precomputed structures used for vectorized division.
///
/// # Wrapping arithmetic
///
/// The basic operators (`+`, `-`, `*`, their assigning forms, and unary `-`
/// for signed integer vectors) **wrap on overflow** on every backend, in both
/// debug and release builds. See [`NumericVector`] for the rationale. The
/// `num_traits::WrappingAdd` / `WrappingSub` / `WrappingMul` impls are simply
/// renames of the operator forms; for explicit saturation use
/// [`saturating_add`](Self::saturating_add) /
/// [`saturating_sub`](Self::saturating_sub).
///
/// # Reductions
///
/// Horizontal reductions ([`sum_elements`](NumericVector::sum_elements),
/// [`prod_elements`](NumericVector::prod_elements), [`wrapping_sum`](Self::wrapping_sum),
/// [`wrapping_prod`](Self::wrapping_prod)) all wrap on overflow.
#[rustfmt::skip] #[thermite_macros::vector_trait]
pub trait IntegerVector:
    NumericVector<Element: Denominator>
    + BitshiftVector
    + ops::DivMasked<Self::Mask, Self::Divider, Output = Self>
    + ops::DivMasked<Self::Mask, Self::BranchfreeDivider, Output = Self>
    // TODO: Some of these might interfere with the methods of this trait,
    // adding ambiguity. See what we can do about that.
    + num_traits::Saturating + num_traits::SaturatingAdd
    + num_traits::SaturatingSub + num_traits::WrappingMul
    + num_traits::WrappingAdd + num_traits::WrappingSub
{
    /// Precomputed scalar divider used by per-lane division against a
    /// runtime-known but loop-invariant divisor. See [`crate::Divider`].
    type Divider: Copy;
    /// Branchfree variant of [`Divider`](Self::Divider). Slightly slower for
    /// some divisors but always emits straight-line code with no conditional
    /// branches, which is what you want inside a hot SIMD loop.
    type BranchfreeDivider: Copy;
    /// Precomputed per-lane divider produced by [`to_divider`](Self::to_divider).
    /// Used when each lane needs a different (but loop-invariant) divisor.
    type VectorizedDivider: Copy;

    /// Multiply two vectors lane-wise and return the *high* half of each
    /// double-width product.
    ///
    /// For signed `i32` lanes the result is `(a as i64 * b as i64) >> 32`;
    /// for unsigned `u32` it is the same with `u64`. Together with
    /// [`mullo`](Self::mullo) this gives the full double-width product
    /// without widening the vector type.
    #[conditional] fn mulhi(self, other: Self) -> Self;

    /// Multiply two vectors lane-wise and return the *low* half of each
    /// product, with wrapping on overflow.
    ///
    /// This is bit-identical to the `*` operator on integer vectors; the
    /// dedicated method exists because some ISAs have specialized
    /// low-half-only multiply instructions worth emitting directly.
    #[conditional] fn mullo(self, other: Self) -> Self;

    // fn wrapping_add(self, other: Self) -> Self;
    // fn wrapping_sub(self, other: Self) -> Self;
    // fn wrapping_mul(self, other: Self) -> Self;

    /// Per-lane saturating addition: instead of wrapping, the result is
    /// clamped to the element type's range (`MIN`..=`MAX`) on overflow.
    #[conditional] fn saturating_add(self, other: Self) -> Self;

    /// Per-lane saturating subtraction: instead of wrapping, the result is
    /// clamped to the element type's range (`MIN`..=`MAX`) on overflow.
    #[conditional] fn saturating_sub(self, other: Self) -> Self;

    /// Horizontal sum of all lanes, wrapping on overflow.
    ///
    /// Equivalent to [`sum_elements`](NumericVector::sum_elements) on integer
    /// vectors; the explicit name documents the wrapping behavior at the
    /// callsite.
    #[conditional] fn wrapping_sum(self) -> Self::Element;

    /// Horizontal product of all lanes, wrapping on overflow.
    ///
    /// Equivalent to [`prod_elements`](NumericVector::prod_elements); the
    /// explicit name documents the wrapping behavior at the callsite.
    #[conditional] fn wrapping_prod(self) -> Self::Element;

    /// Build a [`Divider`](Self::Divider) for a single scalar divisor `d`,
    /// suitable for repeatedly dividing many vectors by the same `d`.
    ///
    /// Construction is `O(1)` but non-trivial; build once outside the hot
    /// loop, then use `vec / divider` inside.
    fn create_divider(d: Self::Element) -> Self::Divider;

    /// Build a [`BranchfreeDivider`](Self::BranchfreeDivider) for a single
    /// scalar divisor `d`. Prefer this over [`create_divider`](Self::create_divider)
    /// inside tight SIMD loops where conditional branches would hurt
    /// throughput.
    fn create_branchfree_divider(d: Self::Element) -> Self::BranchfreeDivider;

    /// Use this vector as the denominators for a vectorized division operation.
    ///
    /// This creates a `VectorDivider` which can then be used to perform
    /// vectorized integer division with the `Div` trait. Note that for unsigned
    /// integer types, `1` is not a valid denominator and will cause a panic.
    ///
    /// This operation itself is NOT vectorized and is `O(n)` in the number of lanes.
    /// It is designed to be calculated once and then reused for multiple division operations.
    ///
    /// # Panics
    ///
    /// If unsigned, integer values of `1` present in the vector
    /// denominators will cause a panic.
    fn to_divider(self) -> Self::VectorizedDivider;

    /// For each element in the vector, count the number of bits that are set to 1.
    #[conditional] fn count_ones(self) -> Self;
    /// For each element in the vector, count the number of bits that are set to 0.
    #[conditional] fn count_zeros(self) -> Self;
    /// For each element in the vector, count the number of leading ones.
    #[conditional] fn leading_ones(self) -> Self;
    /// For each element in the vector, count the number of leading zeros.
    #[conditional] fn leading_zeros(self) -> Self;
}

#[rustfmt::skip] #[thermite_macros::vector_trait]
pub trait SignedIntegerVector: SignedVector + IntegerVector<Element: crate::element::SignedIntegerElement> {
    /// For each lane in the vector, right shift in sign bits by the immediate value.
    #[conditional] fn srai<const I: i32>(self) -> Self;
    /// For each lane in the vector, right shift in sign bits by the given value.
    #[conditional] fn sra(self, count: u32) -> Self;
    /// For each lane in the vector, right shift in sign bits by the corresponding lane in the shifts vector.
    #[conditional] fn srav(self, counts: Self::Unsigned) -> Self;

    /// Floor average: `(a + b) >> 1` rounded toward -∞, computed without overflow.
    #[conditional] fn avg_floor(self, other: Self) -> Self;
    /// Ceiling average: `(a + b + 1) >> 1` rounded toward +∞, computed without overflow.
    #[conditional] fn avg_ceil(self, other: Self) -> Self;

    /// Rounded high-half signed multiply: the fixed-point `Q(W-1)` product
    /// `(self * other + 2^(W-2)) >> (W-1)`, where `W` is the element bit width.
    ///
    /// For `i16` this is the Q15 rounded multiply (x86 `PMULHRSW`), the
    /// fixed-point DSP primitive for gain/volume, fades, and window functions.
    /// Unlike [`mulhi`](IntegerVector::mulhi) it rounds to nearest instead of
    /// truncating, avoiding a DC bias.
    #[conditional] fn mulhrs(self, other: Self) -> Self;
}

#[rustfmt::skip] #[thermite_macros::vector_trait]
pub trait UnsignedIntegerVector: IntegerVector<Element: crate::element::UnsignedIntegerElement> {
    /// Determines if each unsigned integer element in the vector is a
    /// power of two, returning a mask indicating whether or not it is.
    fn is_power_of_two(self) -> Self::Mask;

    /// Per-lane inclusive unsigned range test: a mask of `lo <= self <= hi`,
    /// assuming `lo <= hi`.
    ///
    /// Computed branchlessly as `(self - lo) <= (hi - lo)` with wrapping
    /// subtraction: a single unsigned compare instead of the two an explicit
    /// `self >= lo & self <= hi` would need. The workhorse of byte
    /// classification - testing digit/alpha/whitespace ranges.
    fn in_range(self, lo: Self, hi: Self) -> Self::Mask;

    /// Returns the next power of two minus one for each unsigned integer
    /// element in the vector.
    #[conditional] fn next_power_of_two_m1(self) -> Self;
    /// Computes log2(x) + 1 for each unsigned integer element in the vector.
    #[conditional] fn ilog2p1(self) -> Self;

    /// Compute the parity of each unsigned integer lane in the vector.
    #[conditional] fn parity(self) -> Self;

    /// Ceiling average: `(a + b + 1) >> 1`, computed without overflow.
    ///
    /// Matches x86 `PAVGB`/`PAVGW` and ARM `vrhadd` semantics.
    #[conditional] fn avg(self, other: Self) -> Self;

    /// Per-lane unsigned absolute difference `|self - other|`, without overflow.
    ///
    /// Computed branchlessly as `(self -| other) | (other -| self)` with
    /// saturating subtraction. The per-lane building block of sum-of-absolute-
    /// differences (block matching, motion estimation).
    #[conditional] fn abs_diff(self, other: Self) -> Self;

    /// Per-lane `N`-dimensional Morton code (Z-order curve index): interleave the
    /// low `floor(W / N)` bits of each of the `N` coordinate vectors into one,
    /// placing bit `i` of `values[d]` at output position `i * N + d`. `N = 2` is
    /// the classic 2D code, `N = 3` the 3D (voxel/octree) code.
    ///
    /// The workhorse for spatial sorting (BVH/octree builds, grid binning,
    /// nearest-neighbour broad-phase): compute a whole vector of codes at once,
    /// then sort. [`reverse_morton`](Self::reverse_morton) inverts it.
    fn morton<const N: usize>(values: [Self; N]) -> Self;

    /// Inverse of [`morton`](Self::morton): de-interleave a Morton code back into
    /// its `N` coordinate vectors, where `out[d]` gathers output bits
    /// `d, d + N, d + 2N, ...` into the low `floor(W / N)` bits.
    fn reverse_morton<const N: usize>(self) -> [Self; N];
}

/// Escape hatch tying a [`Vector`] to its specific underlying
/// [`Register`](crate::register::Register) type.
///
/// Provides round-trip conversion between the user-facing [`Vector`] and the
/// raw register storage. Most generic code should bound on
/// [`GenericVector`] (or a more specific vector trait) and never need this;
/// it exists so that code which deliberately specializes on a particular
/// backend can drop down to the register layer without losing the trait
/// hierarchy on the way back up.
pub trait VectorWithRegister<R: crate::register::Register>: GenericVector {
    /// Consume the vector and yield its raw register storage.
    fn into_register(self) -> crate::register::Storage<R>;

    /// Wrap a raw register storage value back into a `Vector`.
    fn from_register(reg: crate::register::Storage<R>) -> Self;

    /// Borrow the vector's elements as a slice.
    ///
    /// The lane count travels as the slice length (always
    /// [`lanes()`](GenericVector::lanes)) rather than in the type, so this is the
    /// preferred read accessor over array-typed borrows.
    fn as_slice(&self) -> &[Self::Element];

    /// Mutably borrow the vector's elements as a slice.
    ///
    /// See [`as_slice`](Self::as_slice).
    fn as_mut_slice(&mut self) -> &mut [Self::Element];
}

/// Float vector types which have an associated hardware register type.
pub trait FloatVectorWithRegister:
    FloatVectorWithBits<Mask = crate::Mask<Self::Register>> + VectorWithRegister<Self::Register>
{
    type Register: crate::register::FloatRegister<Element = Self::Element, Lanes = Self::Lanes>;
}

/// SignedBits integer vector types which have an associated hardware register type.
pub trait SignedIntegerVectorWithRegister:
    SignedIntegerVector<Mask = crate::Mask<Self::Register>> + VectorWithRegister<Self::Register>
{
    type Register: crate::register::SignedIntegerRegister<Element = Self::Element, Lanes = Self::Lanes>;
}

/// Unsigned integer vector types which have an associated hardware register type.
pub trait UnsignedIntegerVectorWithRegister:
    UnsignedIntegerVector<Mask = crate::Mask<Self::Register>> + VectorWithRegister<Self::Register>
{
    type Register: crate::register::UnsignedIntegerRegister<Element = Self::Element, Lanes = Self::Lanes>;
}

#[rustfmt::skip] #[thermite_macros::vector_trait]
pub trait FloatVector: SignedVector<Element: FloatElement>
    + FloatConsts
    + CastVector<Self::ExtendedPrecision>
    + ops::MulAddExtMasked<Self::Mask, Self, Self, Output = Self>
    + ops::MulAddAssignExtMasked<Self::Mask, Self, Self>
    + ops::AddSubExtMasked<Self::Mask, Output = Self>
{
    /// The value `0.5` represented in this vector type.
    const HALF: Self;
    /// The value `-0.0` represented in this vector type.
    const NEG_ZERO: Self;
    /// The value `infinity` represented in this vector type.
    const INFINITY: Self;
    /// The value `-infinity` represented in this vector type.
    const NEG_INFINITY: Self;
    /// The value `NaN` represented in this vector type.
    const NAN: Self;
    /// Hardware epsilon value in this vector type.
    const EPSILON: Self;

    /// If available, an extended precision floating point vector type
    /// corresponding to this vector type. E.g., for `f32` vectors, this
    /// would be an `f64` vector type.
    ///
    /// If no such type exists, this will be the same as `Self`.
    type ExtendedPrecision: FloatVector<Lanes = Self::Lanes> + CastVector<Self>;

    /// Check if each element in the vector is infinite, returning a mask.
    fn is_infinite(self) -> Self::Mask;

    /// Check if each element in the vector is finite, returning a mask.
    fn is_finite(self) -> Self::Mask;

    /// Check if each element in the vector is NaN, returning a mask.
    fn is_nan(self) -> Self::Mask;

    /// Check if each element in the vector is zero or subnormal, returning a mask.
    fn is_zero_or_subnormal(self) -> Self::Mask;

    /// Check if each element in the vector is normal, returning a mask.
    fn is_normal(self) -> Self::Mask;

    /// Check if each element in the vector is subnormal, returning a mask.
    fn is_subnormal(self) -> Self::Mask;

    /// `true` if the backend has a hardware approximate-reciprocal
    /// instruction (e.g. `rcpps` on x86). When `false`, [`rcp`](Self::rcp)
    /// falls back to a full IEEE division and provides no speed advantage
    /// over `Self::ONE / self`.
    const HAS_APPROX_RCP: bool;

    /// `true` if the backend has a hardware approximate-reciprocal-square-root
    /// instruction (e.g. `rsqrtps` on x86). When `false`, [`rsqrt`](Self::rsqrt)
    /// falls back to `Self::ONE / self.sqrt()`.
    const HAS_APPROX_RSQRT: bool;

    /// Lane-wise IEEE 754 square root.
    ///
    /// Negative inputs (other than `-0.0`) produce NaN. `sqrt(-0.0)` is `-0.0`.
    #[conditional] fn sqrt(self) -> Self;

    /// Lane-wise approximate reciprocal square root.
    ///
    /// Accuracy is hardware-dependent (typically 12 bits on x86 `rsqrtps`,
    /// closer to full precision on newer ISAs). For full-precision results
    /// or backends without hardware support, see [`HAS_APPROX_RSQRT`](Self::HAS_APPROX_RSQRT).
    #[conditional] fn rsqrt(self) -> Self;

    /// Lane-wise approximate reciprocal: `1 / self`.
    ///
    /// Accuracy is hardware-dependent (typically 12 bits on x86 `rcpps`).
    /// For full-precision results or backends without hardware support,
    /// see [`HAS_APPROX_RCP`](Self::HAS_APPROX_RCP), or use `Self::ONE / self`.
    #[conditional] fn rcp(self) -> Self;

    /// Lane-wise floor: largest integer less than or equal to each element.
    ///
    /// Result type stays the same; the value is the integer rounded toward
    /// negative infinity, kept in the float representation.
    #[conditional] fn floor(self) -> Self;

    /// Lane-wise ceiling: smallest integer greater than or equal to each
    /// element, kept in the float representation.
    #[conditional] fn ceil(self) -> Self;

    /// Lane-wise round-to-nearest.
    ///
    /// Halfway cases follow the current rounding mode of the hardware. On
    /// x86 this is round-half-to-even (banker's rounding), which differs
    /// from the scalar `f32::round` / `f64::round` half-away-from-zero
    /// convention. If you need a specific tie-breaking rule, do it explicitly.
    #[conditional] fn round(self) -> Self;

    /// Lane-wise truncation toward zero (drops the fractional part), kept
    /// in the float representation.
    #[conditional] fn trunc(self) -> Self;

    /// Lane-wise fractional part: `self - self.trunc()`.
    ///
    /// Result has the same sign as the input. For very large magnitudes the
    /// fractional part is exactly zero because the float has no fractional bits.
    #[conditional] fn fract(self) -> Self;

    /// Effectively `self * sign.signum()`, multiplying the sign bits.
    #[conditional] fn mul_sign(self, sign: Self) -> Self;

    /// Returns zero with the sign of `self`, i.e.: only the sign bit is set.
    #[conditional] fn signed_zero(self) -> Self;

    /// Returns the next representable value greater than the current value, towards positive infinity.
    #[conditional] fn next_up(self) -> Self;

    /// Returns the next representable value less than the current value, towards negative infinity.
    #[conditional] fn next_down(self) -> Self;

    /// Linearly interpolates between `a` and `b` by `self`, where `self` is typically in the range `[0, 1]`.
    ///
    /// Follows the formula: `a * (1 - self) + b * self`, but the underlying implementation
    /// may optimize into certain other formulations.
    fn mix(self, a: Self, b: Self) -> Self;

    /// Computes `$1 - x^2$` accurately, avoiding the cancellation a naive `1 - self * self`
    /// suffers as `self` approaches `±1` (where the result is small but `self * self` is near 1).
    ///
    /// With hardware FMA this is `nmul_add(self, self, 1)`: the exact product `$x^2$` is formed
    /// and subtracted from one with a single rounding. Without FMA it falls back to the factored
    /// `$(1 - x)(1 + x)$`, also cancellation-free (`1 - self` is exact for `self` near 1 by
    /// Sterbenz's lemma). Both keep full relative accuracy in the small result.
    #[inline(always)]
    fn one_minus_sq(self) -> Self {
        if const { Self::HAS_TRUE_FMA } {
            // FMA: 1 - self*self formed from the exact product with a single rounding.
            self.nmul_add(self, Self::ONE)
        } else {
            // No FMA: factored difference of squares, cancellation-free near |self| = 1.
            (Self::ONE - self) * (Self::ONE + self)
        }
    }

    /// Inhibit further LLVM auto-vectorization of code surrounding this call.
    ///
    /// LLVM sometimes tries to "vectorize the vectors" -- repacking
    /// already-SIMD code into a wider form that ends up slower. Inserting
    /// this call inside a hot loop blocks that pass at the use site. The
    /// call itself emits no instructions; only the optimizer barrier remains.
    ///
    /// # Safety
    ///
    /// Memory-safe to call, but the side effect on code generation is
    /// significant. Only reach for this when you have measured a regression
    /// caused by over-aggressive auto-vectorization.
    unsafe fn block_autovectorization(&mut self);

    /// Attempt to upcast this FloatVector to a FloatVectorWithBits,
    /// using the provided kernel. If not possible, returns None.
    fn with_bits<const N: usize, K: AsFloatVectorWithBitsKernel<Self, N>>(
        values: [Self; N],
        kernel: K,
    ) -> Option<<K as AsFloatVectorWithBitsKernel<Self, N>>::Output> {
        None // Default implementation returns None
    }
}

/// Run a closure-like block with a generic [`FloatVector`] temporarily upcast
/// to a [`FloatVectorWithBits`], when the backend supports it.
///
/// Given an array of `FloatVector` values and a body parameterized over a
/// `FloatVectorWithBits` type, this expands to a call to
/// [`FloatVector::with_bits`] with an anonymous kernel implementing
/// [`AsFloatVectorWithBitsKernel`]. The body only runs when bit access is
/// available for the concrete backend; otherwise the whole expression evaluates
/// to `None` (the return type is therefore `Option<_>`).
///
/// This is the ergonomic front-end to the [`AsFloatVectorWithBitsKernel`]
/// pattern; reach for it inside generic code bounded only on `FloatVector` that
/// wants an optional fast path requiring bit-level access.
#[macro_export]
macro_rules! with_bits {
    // (($first_value:expr $(, $value:expr)+): $ty:ty as fn($first_decl:ident: $first_alias:ident $(,$decl:ident: $alias:ident)* ) -> $ret:ty $(where $($c:ty: $constraint:ident),*)? { $($body:tt)* }) => {{
    //     $crate::with_bits!(($first_value): $ty as fn($first_decl: $first_alias) -> impl $ret $(where $($c: $constraint),*)? {
    //         $crate::with_bits!(($($value),+): $ty as fn($($decl: $alias),*) -> $ret $(where $($c: $constraint),*)? {
    //             $($body)*
    //         })
    //     })
    // }};

    ([$($values:expr),+]: [$ty:ty; $len:literal] as fn($decl:ident: [$alias:ident; _]) -> $ret:ty $(where $($c:ty: $constraint:ident),*)? { $($body:tt)* }) => {{
        struct AnonymousAsFloatVectorWithBitsKernel<V>(core::marker::PhantomData<V>);

        impl<V: FloatVector> $crate::vector::AsFloatVectorWithBitsKernel<V, $len> for AnonymousAsFloatVectorWithBitsKernel<V>
            $(where $($c: $constraint),*)?
        {
            type Output = $ret;

            fn with_bits<
                $alias: FloatVectorWithBits<
                        Element = V::Element,
                        Lanes = V::Lanes,
                        Mask = V::Mask,
                        Signed = V::Signed,
                        Unsigned = V::Unsigned,
                        ExtendedPrecision = V::ExtendedPrecision,
                    > + CastVector<V>,
            >(
                self,
                $decl: [$alias; $len],
            ) -> Self::Output {
                $($body)*
            }
        }

        <V as FloatVector>::with_bits(
            [$($values),+],
            AnonymousAsFloatVectorWithBitsKernel::<V>(core::marker::PhantomData),
        )
    }};
}

/// Some algorithms may benefit from being able to access the bitwise
/// representation of floating point vectors. However, not all vectors
/// support this functionality, and those that do may be passed as generic
/// FloatVector. Therefore, this is a way of upcasting a FloatVector
/// to a FloatVectorWithBits, if possible. If not possible, returns None.
pub trait AsFloatVectorWithBitsKernel<O: FloatVector, const N: usize> {
    type Output;

    fn with_bits<
        V: FloatVectorWithBits<
                Element = O::Element,
                Lanes = O::Lanes,
                Mask = O::Mask,
                Signed = O::Signed,
                Unsigned = O::Unsigned,
                ExtendedPrecision = O::ExtendedPrecision,
            > + CastVector<O>,
    >(
        self,
        v: [V; N],
    ) -> Self::Output;
}

/// A [`FloatVector`] that additionally exposes its raw bit representation as
/// companion integer vectors, enabling bit-level float algorithms.
///
/// On top of [`FloatVector`] this provides:
/// - the [`Bits`](Self::Bits) (unsigned) and [`SignedBits`](Self::SignedBits)
///   integer vector types matching this float's bit width and lane count, with
///   full [`FullyInteroperable`] cast/bitcast interop between all three views;
/// - hardware-accelerated `native_*` transcendentals gated by
///   [`NATIVE_CAP`](Self::NATIVE_CAP);
/// - bit-level helpers like [`total_order`](Self::total_order) /
///   [`linear_order`](Self::linear_order) for sorting and ULP math.
///
/// Not every float vector implements this (it requires the element to be a
/// [`FloatElementWithBits`]); generic code that only sometimes needs bit access
/// can attempt to obtain it via [`FloatVector::with_bits`].
///
/// The methods on this trait do **not** have masked (`_c`/`_m`/`_z`) variants.
pub trait FloatVectorWithBits:
    BitwiseVector
    + FloatVector<Element: FloatElementWithBits, Signed: CastVector<Self::SignedBits>, Unsigned: CastVector<Self::Bits>>
    + GenericVector<
        Signed: GenericVector<Mask: CastMask<<Self::SignedBits as GenericVector>::Mask>>,
        Unsigned: GenericVector<Mask: CastMask<<Self::Unsigned as GenericVector>::Mask>>,
    > + FullyInteroperable<Self::Bits, Self::SignedBits>
{
    type SignedBits: SignedIntegerVector<
            Mask: CastMask<<Self::Signed as GenericVector>::Mask>,
            Lanes = Self::Lanes,
            Divider = Divider<<Self::Element as FloatElementWithBits>::SignedBits>,
            BranchfreeDivider = BranchfreeDivider<<Self::Element as FloatElementWithBits>::SignedBits>,
            Element = <Self::Element as FloatElementWithBits>::SignedBits,
        > + FullyInteroperable<Self, Self::Bits>
        + CastVector<Self::Signed>;

    type Bits: UnsignedIntegerVector<
            Mask: CastMask<<Self::Unsigned as GenericVector>::Mask>,
            Lanes = Self::Lanes,
            Divider = Divider<<Self::Element as FloatElementWithBits>::Bits>,
            BranchfreeDivider = BranchfreeDivider<<Self::Element as FloatElementWithBits>::Bits>,
            Element = <Self::Element as FloatElementWithBits>::Bits,
        > + FullyInteroperable<Self, Self::SignedBits>
        + CastVector<Self::Unsigned>;

    /// Bit-flag set describing which `native_*` methods on this trait have a
    /// real hardware implementation on the current backend.
    ///
    /// Test with `NATIVE_CAP.has(NativeCapability::SIN)` etc. before calling
    /// the corresponding `native_*` method directly; otherwise the default
    /// implementation will panic.
    const NATIVE_CAP: NativeCapability;

    /// Hardware-accelerated `ldexp`: `self * 2^exp`, lane-wise.
    ///
    /// # Safety
    /// Only callable when [`NATIVE_CAP`](Self::NATIVE_CAP) advertises `LDEXP`.
    /// Calling on a backend without hardware support is undefined behavior
    /// (the default impl panics via `unreachable!` at the register layer).
    unsafe fn native_ldexp(self, exp: Self::SignedBits) -> Self;

    /// Hardware-accelerated `frexp`: split each lane into a normalized
    /// mantissa in `[0.5, 1.0)` and an integer exponent.
    ///
    /// # Safety
    /// Only callable when [`NATIVE_CAP`](Self::NATIVE_CAP) advertises `FREXP`.
    unsafe fn native_frexp(self) -> (Self, Self::SignedBits);

    /// Hardware-accelerated combined sine and cosine, lane-wise.
    ///
    /// # Safety
    /// Only callable when [`NATIVE_CAP`](Self::NATIVE_CAP) advertises `SIN_COS`.
    unsafe fn native_sin_cos<P: Policy>(self) -> (Self, Self);

    /// Hardware-accelerated sine, lane-wise.
    ///
    /// # Safety
    /// Only callable when [`NATIVE_CAP`](Self::NATIVE_CAP) advertises `SIN`.
    unsafe fn native_sin<P: Policy>(self) -> Self;

    /// Hardware-accelerated cosine, lane-wise.
    ///
    /// # Safety
    /// Only callable when [`NATIVE_CAP`](Self::NATIVE_CAP) advertises `COS`.
    unsafe fn native_cos<P: Policy>(self) -> Self;

    /// Hardware-accelerated tangent, lane-wise.
    ///
    /// # Safety
    /// Only callable when [`NATIVE_CAP`](Self::NATIVE_CAP) advertises `TAN`.
    unsafe fn native_tan<P: Policy>(self) -> Self;

    /// Hardware-accelerated `2^self`, lane-wise.
    ///
    /// # Safety
    /// Only callable when [`NATIVE_CAP`](Self::NATIVE_CAP) advertises `EXP2`.
    unsafe fn native_exp2<P: Policy>(self) -> Self;

    /// Hardware-accelerated `log2(self)`, lane-wise.
    ///
    /// # Safety
    /// Only callable when [`NATIVE_CAP`](Self::NATIVE_CAP) advertises `LOG2`.
    unsafe fn native_log2<P: Policy>(self) -> Self;

    /// Hardware-accelerated `e^self`, lane-wise.
    ///
    /// # Safety
    /// Only callable when [`NATIVE_CAP`](Self::NATIVE_CAP) advertises `EXP`.
    unsafe fn native_exp<P: Policy>(self) -> Self;

    /// Hardware-accelerated natural logarithm, lane-wise.
    ///
    /// # Safety
    /// Only callable when [`NATIVE_CAP`](Self::NATIVE_CAP) advertises `LN`.
    unsafe fn native_ln<P: Policy>(self) -> Self;

    /// Hardware-accelerated `self^exp`, lane-wise.
    ///
    /// # Safety
    /// Only callable when [`NATIVE_CAP`](Self::NATIVE_CAP) advertises `POWF`.
    unsafe fn native_powf<P: Policy>(self, exp: Self) -> Self;

    /// Return a signed integer vector that is capable of encapsulating
    /// the "total order" of the floating point values in this vector,
    /// such that when compared as integers, the ordering is the same
    /// as the floating point ordering, including NaNs, in the following order:
    ///
    /// - negative quiet NaN
    /// - negative signaling NaN
    /// - negative infinity
    /// - negative numbers
    /// - negative subnormal numbers
    /// - negative zero
    /// - positive zero
    /// - positive subnormal numbers
    /// - positive numbers
    /// - positive infinity
    /// - positive signaling NaN
    /// - positive quiet NaN.
    ///
    /// This is useful for sorting floating point numbers in a way that
    /// is consistent and well-defined. However, it may differ from
    /// the default floating point comparison behavior of the platform.
    ///
    /// # Example
    /// ```rust
    /// # use thermite::backend::scalar::prelude::*;
    /// let x = f32x4::NAN;
    /// let y = f32x4::ONE;
    /// let total_lt = x.total_order().cmp_lt(y.total_order());
    /// assert!(total_lt.none()); // NaN is not less than 1.0 in total order
    /// ```
    fn total_order(self) -> Self::SignedBits;

    /// Similar to [`total_order`](FloatVectorWithBits::total_order), but positive zero and negative zero are
    /// the same value. This can be used for calculating ULP differences by simply subtracting one from another.
    fn linear_order(self) -> Self::SignedBits;
}

/// Convenience accessors `x()` / `y()` automatically available on any
/// 2-lane [`GenericVector`].
///
/// Each method is just shorthand for [`extract`](GenericVector::extract) at
/// the corresponding compile-time index.
#[rustfmt::skip]
pub trait GenericVector2: GenericVector {
    /// Returns the value of lane 0.
    #[inline(always)] fn x(&self) -> Self::Element { self.extract::<0>() }
    /// Returns the value of lane 1.
    #[inline(always)] fn y(&self) -> Self::Element { self.extract::<1>() }
}

/// Convenience accessors `x()` / `y()` / `z()` automatically available on any
/// 3-lane [`GenericVector`].
///
/// Each method is just shorthand for [`extract`](GenericVector::extract) at
/// the corresponding compile-time index.
#[rustfmt::skip]
pub trait GenericVector3: GenericVector {
    /// Returns the value of lane 0.
    #[inline(always)] fn x(&self) -> Self::Element { self.extract::<0>() }
    /// Returns the value of lane 1.
    #[inline(always)] fn y(&self) -> Self::Element { self.extract::<1>() }
    /// Returns the value of lane 2.
    #[inline(always)] fn z(&self) -> Self::Element { self.extract::<2>() }
}

/// Convenience accessors `x()` / `y()` / `z()` / `w()` automatically available
/// on any 4-lane [`GenericVector`].
///
/// Each method is just shorthand for [`extract`](GenericVector::extract) at
/// the corresponding compile-time index.
#[rustfmt::skip]
pub trait GenericVector4: GenericVector {
    /// Returns the value of lane 0.
    #[inline(always)] fn x(&self) -> Self::Element { self.extract::<0>() }
    /// Returns the value of lane 1.
    #[inline(always)] fn y(&self) -> Self::Element { self.extract::<1>() }
    /// Returns the value of lane 2.
    #[inline(always)] fn z(&self) -> Self::Element { self.extract::<2>() }
    /// Returns the value of lane 3.
    #[inline(always)] fn w(&self) -> Self::Element { self.extract::<3>() }
}

impl<V: GenericVector<Lanes = typenum::U2>> GenericVector2 for V {}
impl<V: GenericVector<Lanes = typenum::U3>> GenericVector3 for V {}
impl<V: GenericVector<Lanes = typenum::U4>> GenericVector4 for V {}

#[rustfmt::skip]
macro_rules! impl_swizzle4 {
    (@ x) => { 0 };
    (@ y) => { 1 };
    (@ z) => { 2 };
    (@ w) => { 3 };

    (IMPL x x x x) => { #[inline(always)] fn xxxx(self) -> Self { self.broadcast::<0>() } };
    (IMPL y y y y) => { #[inline(always)] fn yyyy(self) -> Self { self.broadcast::<1>() } };
    (IMPL z z z z) => { #[inline(always)] fn zzzz(self) -> Self { self.broadcast::<2>() } };
    (IMPL w w w w) => { #[inline(always)] fn wwww(self) -> Self { self.broadcast::<3>() } };

    (IMPL $a:ident $b:ident $c:ident $d:ident) => {paste::paste! {
        #[inline(always)]
        fn [<$a $b $c $d>](self) -> Self {
            struct Indices;

            impl crate::swizzle::SwizzleIndices<typenum::U4> for Indices {
                const INDICES: GenericArray<u32, typenum::U4> = {
                    unsafe { $crate::generic_array::const_transmute::<_, GenericArray<u32, typenum::U4>>([
                        impl_swizzle4!(@ $a),
                        impl_swizzle4!(@ $b),
                        impl_swizzle4!(@ $c),
                        impl_swizzle4!(@ $d)
                    ]) }
                };
            }

            self.permute_const::<Indices>()
        }
    }};

    (DECL $(#[$meta:meta])* $a:ident $b:ident $c:ident $d:ident) => {paste::paste! {
        #[allow(missing_docs)]
        $(#[$meta])* fn [<$a $b $c $d>](self) -> Self;
    }};

    ($( $(#[$meta:meta])* [$a:ident $b:ident $c:ident $d:ident]),*) => {
        /// Only available for 4-lane vectors, this allows human-readable swizzle/permutations
        /// of the vector.
        pub trait Swizzle4: SwizzleVector<Lanes = typenum::U4> { $(impl_swizzle4!(DECL $(#[$meta])* $a $b $c $d);)* }

        /// Implements 4-lane swizzling for vectors.
        impl<V: SwizzleVector<Lanes = typenum::U4>> Swizzle4 for V {
            $(impl_swizzle4!(IMPL $a $b $c $d);)*
        }
    }
}

#[rustfmt::skip]
macro_rules! impl_swizzle3 {
    (IMPL x x x) => { #[inline(always)] fn xxx(self) -> Self { self.broadcast::<0>() } };
    (IMPL y y y) => { #[inline(always)] fn yyy(self) -> Self { self.broadcast::<1>() } };
    (IMPL z z z) => { #[inline(always)] fn zzz(self) -> Self { self.broadcast::<2>() } };

    (IMPL $a:ident $b:ident $c:ident) => {paste::paste! {
        #[inline(always)]
        fn [<$a $b $c>](self) -> Self {
            struct Indices;

            impl crate::swizzle::SwizzleIndices<typenum::U3> for Indices {
                const INDICES: GenericArray<u32, typenum::U3> = {
                    unsafe { $crate::generic_array::const_transmute::<_, GenericArray<u32, typenum::U3>>([
                        impl_swizzle4!(@ $a),
                        impl_swizzle4!(@ $b),
                        impl_swizzle4!(@ $c)
                    ]) }
                };
            }

            self.permute_const::<Indices>()
        }
    }};

    (DECL $(#[$meta:meta])* $a:ident $b:ident $c:ident) => {paste::paste! {
        #[allow(missing_docs)]
        $(#[$meta])* fn [<$a $b $c>](self) -> Self;
    }};

    ($( $(#[$meta:meta])* [$a:ident $b:ident $c:ident]),*) => {
        /// Only available for "3-lane" (ignoring 4th lane) [`LinAlg3Register`](crate::register::LinAlg3Register) vectors,
        /// this allows human-readable swizzle/permutations of the vector. Permutations
        /// will ignore the 4th lane of the register, leaving it unchanged.
        pub trait Swizzle3: SwizzleVector<Lanes = typenum::U3> { $(impl_swizzle3!(DECL $(#[$meta])* $a $b $c);)* }

        /// Implements 3-lane swizzling for vectors support 3-lane linear algebra operations.
        impl<V: SwizzleVector<Lanes = typenum::U3>> Swizzle3 for V {
            $(impl_swizzle3!(IMPL $a $b $c);)*
        }
    }
}

impl_swizzle3! {
    [x y z], [x x x], [x x y], [x x z], [x y x], [x y y], [x z x], [x z y], [x z z],
    [y x x], [y x y], [y x z], [y y x], [y y y], [y y z], [y z x], [y z y], [y z z],
    [z x x], [z x y], [z x z], [z y x], [z y y], [z y z], [z z x], [z z y], [z z z]
}

impl_swizzle4! {
    [x y z w], [x x x x], [x x x y], [x x x z], [x x x w], [x x y x], [x x y y], [x x y z],
    [x x y w], [x x z x], [x x z y], [x x z z], [x x z w], [x x w x], [x x w y], [x x w z],
    [x x w w], [x y x x], [x y x y], [x y x z], [x y x w], [x y y x], [x y y y], [x y y z],
    [x y y w], [x y z x], [x y z y], [x y z z], [x y w x], [x y w y], [x y w z], [x y w w],
    [x z x x], [x z x y], [x z x z], [x z x w], [x z y x], [x z y y], [x z y z], [x z y w],
    [x z z x], [x z z y], [x z z z], [x z z w], [x z w x], [x z w y], [x z w z], [x z w w],
    [x w x x], [x w x y], [x w x z], [x w x w], [x w y x], [x w y y], [x w y z], [x w y w],
    [x w z x], [x w z y], [x w z z], [x w z w], [x w w x], [x w w y], [x w w z], [x w w w],
    [y x x x], [y x x y], [y x x z], [y x x w], [y x y x], [y x y y], [y x y z], [y x y w],
    [y x z x], [y x z y], [y x z z], [y x z w], [y x w x], [y x w y], [y x w z], [y x w w],
    [y y x x], [y y x y], [y y x z], [y y x w], [y y y x], [y y y y], [y y y z], [y y y w],
    [y y z x], [y y z y], [y y z z], [y y z w], [y y w x], [y y w y], [y y w z], [y y w w],
    [y z x x], [y z x y], [y z x z], [y z x w], [y z y x], [y z y y], [y z y z], [y z y w],
    [y z z x], [y z z y], [y z z z], [y z z w], [y z w x], [y z w y], [y z w z], [y z w w],
    [y w x x], [y w x y], [y w x z], [y w x w], [y w y x], [y w y y], [y w y z], [y w y w],
    [y w z x], [y w z y], [y w z z], [y w z w], [y w w x], [y w w y], [y w w z], [y w w w],
    [z x x x], [z x x y], [z x x z], [z x x w], [z x y x], [z x y y], [z x y z], [z x y w],
    [z x z x], [z x z y], [z x z z], [z x z w], [z x w x], [z x w y], [z x w z], [z x w w],
    [z y x x], [z y x y], [z y x z], [z y x w], [z y y x], [z y y y], [z y y z], [z y y w],
    [z y z x], [z y z y], [z y z z], [z y z w], [z y w x], [z y w y], [z y w z], [z y w w],
    [z z x x], [z z x y], [z z x z], [z z x w], [z z y x], [z z y y], [z z y z], [z z y w],
    [z z z x], [z z z y], [z z z z], [z z z w], [z z w x], [z z w y], [z z w z], [z z w w],
    [z w x x], [z w x y], [z w x z], [z w x w], [z w y x], [z w y y], [z w y z], [z w y w],
    [z w z x], [z w z y], [z w z z], [z w z w], [z w w x], [z w w y], [z w w z], [z w w w],
    [w x x x], [w x x y], [w x x z], [w x x w], [w x y x], [w x y y], [w x y z], [w x y w],
    [w x z x], [w x z y], [w x z z], [w x z w], [w x w x], [w x w y], [w x w z], [w x w w],
    [w y x x], [w y x y], [w y x z], [w y x w], [w y y x], [w y y y], [w y y z], [w y y w],
    [w y z x], [w y z y], [w y z z], [w y z w], [w y w x], [w y w y], [w y w z], [w y w w],
    [w z x x], [w z x y], [w z x z], [w z x w], [w z y x], [w z y y], [w z y z], [w z y w],
    [w z z x], [w z z y], [w z z z], [w z z w], [w z w x], [w z w y], [w z w z], [w z w w],
    [w w x x], [w w x y], [w w x z], [w w x w], [w w y x], [w w y y], [w w y z], [w w y w],
    [w w z x], [w w z y], [w w z z], [w w z w], [w w w x], [w w w y], [w w w z], [w w w w]
}

/// Vector suitable for 3D linear algebra operations.
///
/// The length of this vector must be either 3 or 4 lanes.
///
/// Methods in this are specifically optimized to either ignore the fourth lane (if it exists),
/// or to use algorithms that map especially well when there are truly only three "lanes",
/// such as on GPUs.
pub trait LinAlg3Vector: FloatVector {
    /// Scalar Product using only the first three lanes of the register as a 3D vector.
    ///
    /// This is more efficient than a raw scalar product, as there is no need to
    /// zero out the last lane of the register.
    fn dot3(self, other: Self) -> Self::Element;

    /// Cross Product using only the first three lanes of the register as a 3D vector.
    ///
    /// This is more efficient than a raw cross product, as there is no need to
    /// zero out the last lane of the register.
    ///
    /// The `DOP` generic parameter indicates whether to use the
    /// "Difference of Products" method for computing the cross product,
    /// which can be more accurate in some cases, but _requires_
    /// hardware fused multiply-add instructions to be efficient.
    ///
    /// If you want the best performance, set `DOP` to `false`.\
    /// If you want the best accuracy or have FMA support, set `DOP` to `true`.
    fn cross3<const DOP: bool>(self, other: Self) -> Self;

    /// Refraction of incident vector `self` through a surface with normal `n`
    /// and relative index of refraction `eta` (`$\eta = \eta_i/\eta_t$`). `self` and `n` are
    /// assumed unit length.
    ///
    /// Total internal reflection (`1 - eta^2*(1 - dot(n,self)^2) < 0`) returns the
    /// zero vector; otherwise `eta*self - (eta*dot(n,self) + sqrt(k))*n`
    fn refract(self, n: Self, eta: Self::Element) -> Self;

    /// Efficiently set the 4th (last) lane of the register to 0.0.
    ///
    /// Useful for sanitizing 3D Homogeneous vectors.
    ///
    /// See [`LinAlg3Vector::one4`] for similar functionality for 3D points.
    fn zero4(self) -> Self;

    /// Efficiently set the 4th (last) lane of the register to 1.0.
    ///
    /// Useful for sanitizing 3D Homogeneous points.
    ///
    /// See [`LinAlg3Vector::zero4`] for similar functionality for 3D vectors.
    fn one4(self) -> Self;

    /// Returns the minimum value in the first three lanes of the register.
    fn min_element3(self) -> Self::Element;

    /// Returns the maximum value in the first three lanes of the register.
    fn max_element3(self) -> Self::Element;

    /// Returns the sum of the first three elements of the register.
    fn sum_elements3(self) -> Self::Element;

    /// Returns the product of the first three elements of the register.
    fn prod_elements3(self) -> Self::Element;

    /// 3x3 Matrix Transpose.
    ///
    /// Only the first three lanes of each output column are meaningful; the 4th
    /// lane (on 4-lane vectors) is unspecified.
    fn mat3_transpose(m: &[Self; 3]) -> [Self; 3];

    /// 3x3 Matrix-Vector multiplication, assuming `self` as the vector.
    fn mat3_vec3_product<const COLUMN_MAJOR: bool>(self, m: &[Self; 3]) -> Self;

    /// 3x3 matrix times `N` 3D vectors (small-`N` batch; see
    /// [`mat4_vec4_product_array`](LinAlg4Vector::mat4_vec4_product_array)).
    fn mat3_vec3_product_array<const COLUMN_MAJOR: bool, const N: usize>(
        m: &[Self; 3],
        vectors: &[Self; N],
    ) -> [Self; N];

    /// 3x3 Matrix-Matrix multiplication.
    ///
    /// If `COLUMN_MAJOR` is `false`, the matrices are treated as row-major and
    /// the multiplication order becomes `rhs * lhs`, mirroring
    /// [`mat4_product`](LinAlg4Vector::mat4_product).
    fn mat3_product<const COLUMN_MAJOR: bool>(lhs: &[Self; 3], rhs: &[Self; 3]) -> [Self; 3];

    /// Determinant of a column-major 3x3 matrix.
    fn mat3_det(m: &[Self; 3]) -> Self::Element;

    /// In-place 3x3 Matrix inversion; **returns the determinant**.
    ///
    /// An exactly-zero determinant leaves the matrix untouched; a near-zero
    /// (ill-conditioned) determinant gives a finite but unreliable result, so
    /// inspect the returned determinant before trusting the matrix.
    fn mat3_inverse_inplace(m: &mut [Self; 3]) -> Self::Element;

    /// 3x3 Matrix inversion.
    ///
    /// Returns `Some(inverse)`, or `None` if the matrix is exactly singular.
    /// Consider [`mat3_inverse_inplace`](Self::mat3_inverse_inplace) (which hands
    /// back the determinant) to avoid the copy and to use a custom tolerance.
    #[inline(always)]
    fn mat3_inverse(m: &[Self; 3]) -> Option<[Self; 3]> {
        let mut mat = *m;
        if Self::mat3_inverse_inplace(&mut mat) == Self::Element::ZERO {
            None
        } else {
            Some(mat)
        }
    }

    /// "Normal matrix" for transforming normals under non-uniform scale, from
    /// the cofactor cross-products of a column-major 3x3.
    ///
    /// `DIVIDE = true` gives the true inverse-transpose `$(M^{-1})^{T}$` (non-finite if
    /// singular); `DIVIDE = false` gives the un-divided cofactor matrix, which is
    /// cheaper, never singular, and points normals the same direction (use it
    /// when you re-normalize the result). Cheaper than a full inverse either way.
    fn mat3_normal<const DIVIDE: bool>(m: &[Self; 3]) -> [Self; 3];
}

/// Vector suitable for 4D linear algebra operations.
///
/// Must have exactly 4 lanes.
pub trait LinAlg4Vector: LinAlg3Vector {
    /// Scalar Product using all four lanes of the register as a 4D vector.
    fn dot4(self, other: Self) -> Self::Element;

    /// Quaternion multiplication.
    ///
    /// Method:
    /// ```text
    /// T1 = (lhs.w * rhs)
    /// T2 = (lhs.x * rhs.wzyx) * {+,-,+,-}
    /// T3 = (lhs.y * rhs.zwxy) * {+,+,-,-}
    /// T4 = (lhs.z * rhs.yxwz) * {-,+,+,-}
    /// T1 + T2 + T3 + T4
    /// ```
    fn quat4_product(self, other: Self) -> Self;

    /// Quaternion-vector multiplication.
    ///
    /// This is optimized to work best on various SIMD architectures. On
    /// architectures with permute/shuffle instructions, it uses the
    /// Double-Cross (Giesen) method. On architectures without such instructions,
    /// it falls back to the standard method of two dot products
    /// and a single cross product. This is because cross products require
    /// several shuffles/permutations to compute efficiently with SIMD.
    ///
    /// The `DOP` generic parameter indicates whether to use the
    /// "Difference of Products" method for computing the cross product(s),
    /// which can be more accurate in some cases, but _requires_
    /// hardware fused multiply-add instructions to be efficient.
    ///
    /// If you want the best performance, set `DOP` to `false`.\
    /// If you want the best accuracy or have FMA support, set `DOP` to `true`.
    fn quat4_vec3_product<const DOP: bool>(self, vec: Self) -> Self;

    /// Rotation matrix of a **unit** quaternion as 3 registers; the 4th lane of
    /// each is unspecified.
    ///
    /// `COLUMN_MAJOR` picks the storage: the rotation's columns when `true`, its
    /// rows when `false` (i.e. the transpose). The choice is free - only the
    /// compile-time sign masks differ.
    ///
    /// Trig-free - the entries are pairwise products of `{x, y, z, w}`, no
    /// `sin`/`cos`/`sqrt`. The quaternion is assumed normalized.
    ///
    /// To rotate many vectors by one quaternion, convert once here and batch via
    /// [`mat3_vec3_product`](LinAlg3Vector::mat3_vec3_product) with the matching
    /// `COLUMN_MAJOR` - cheaper than a per-vector
    /// [`quat4_vec3_product`](Self::quat4_vec3_product) for large `N`.
    fn quat_to_mat3<const COLUMN_MAJOR: bool>(self) -> [Self; 3];

    /// Homogeneous 4x4 rotation matrix of a **unit** quaternion: the
    /// [`quat_to_mat3`](Self::quat_to_mat3) rotation with each rotation
    /// register's 4th lane zeroed and a `[0, 0, 0, 1]` 4th register.
    /// `COLUMN_MAJOR` is forwarded to `quat_to_mat3`.
    fn quat_to_mat4<const COLUMN_MAJOR: bool>(self) -> [Self; 4];

    /// 4x4 Matrix Transpose.
    fn mat4_transpose(m: &[Self; 4]) -> [Self; 4];

    /// 4x4 Matrix-Vector multiplication, assuming `self` as the vector.
    ///
    /// The `COLUMN_MAJOR` generic parameter indicates whether the matrix
    /// is stored in column-major order (`true`) or row-major order (`false`).
    ///
    /// If the matrix is **NOT** in column-major order, it will need to be
    /// transposed before the actual multiplication, which will incur a performance penalty.
    ///
    /// NOTE: If you want to multiply 4 vectors by the same **column-major** matrix, consider using
    /// [`LinAlg4Vector::mat4_product`] instead. It is conceptually the same as multiplying each
    /// vector individually, but can take advantage of SIMD optimizations better.
    fn mat4_vec4_product<const COLUMN_MAJOR: bool>(self, m: &[Self; 4]) -> Self;

    /// 4x4 Matrix-Vector3 multiplication, optimized for the case where the vector is a 3D coordinate
    /// (i.e., the 4th lane is ignored).
    ///
    /// The `COLUMN_MAJOR` generic parameter indicates whether the matrix
    /// is stored in column-major order (`true`) or row-major order (`false`).
    ///
    /// If the matrix is **NOT** in column-major order, it will need to be
    /// transposed before the actual multiplication, which will incur a performance penalty.
    fn mat4_vec3_product<const COLUMN_MAJOR: bool>(self, m: &[Self; 4]) -> Self;

    /// 4x4 matrix times `N` 3D vectors (small-`N` batch; see
    /// [`mat4_vec4_product_array`](Self::mat4_vec4_product_array)).
    fn mat4_vec3_product_array<const COLUMN_MAJOR: bool, const N: usize>(
        m: &[Self; 4],
        vectors: &[Self; N],
    ) -> [Self; N];

    /// 4x4 Matrix-Matrix multiplication.
    ///
    /// If `COLUMN_MAJOR` is `false`, the matrices are assumed to be in row-major order,
    /// and the order of the multiplication will become `rhs * lhs` to account for that.
    /// This is mathematically equivalent to transposing both matrices, performing
    /// the multiplication, and then transposing the result, but is obviously more efficient.
    ///
    /// NOTE: When operating in column-major mode (`COLUMN_MAJOR = true`), the multiplication
    /// is effectively:
    /// ```text
    /// C0 = mat4_vec4_product(lhs, R0)
    /// C1 = mat4_vec4_product(lhs, R1)
    /// C2 = mat4_vec4_product(lhs, R2)
    /// C3 = mat4_vec4_product(lhs, R3)
    /// ```
    ///
    /// and is therefore, in **column-major order**, useful for transforming 4 vectors
    /// by the same matrix, but capable of being optimized better than doing
    /// 4 individual matrix-vector multiplications.
    fn mat4_product<const COLUMN_MAJOR: bool>(lhs: &[Self; 4], rhs: &[Self; 4]) -> [Self; 4];

    /// Transform `N` vectors by a single 4x4 matrix, returning the transformed array.
    ///
    /// Intended for **small** `N` (a handful of points): the array is taken and
    /// returned **by value** and the loop fully unrolls, so a large `N` will
    /// bloat code size and stack usage. For large or dynamic counts, loop
    /// [`mat4_vec4_product`](Self::mat4_vec4_product) over a slice instead.
    ///
    /// Row-major matrices are transposed once up front (amortized over `N`).
    /// Backends with a true double-width register transform two vectors per pass.
    fn mat4_vec4_product_array<const COLUMN_MAJOR: bool, const N: usize>(
        m: &[Self; 4],
        vectors: &[Self; N],
    ) -> [Self; N];

    /// In-place 4x4 Matrix inversion; **returns the determinant**.
    ///
    /// An exactly-zero determinant leaves the matrix untouched; a near-zero
    /// (ill-conditioned) determinant gives a finite but unreliable result, so
    /// inspect the returned determinant before trusting the matrix.
    fn mat4_inverse_inplace(m: &mut [Self; 4]) -> Self::Element;

    /// Compute the determinant of a 4x4 matrix without inverting it.
    fn mat4_det(m: &[Self; 4]) -> Self::Element;

    /// 4x4 Matrix inversion.
    ///
    /// Returns `Some(inverted_matrix)`, or `None` if the matrix is exactly
    /// singular. Consider [`Vector::mat4_inverse_inplace`] (which hands back the
    /// determinant) to avoid the copy and to use a custom tolerance.
    #[inline(always)]
    fn mat4_inverse(m: &[Self; 4]) -> Option<[Self; 4]> {
        let mut mat = *m;

        if crate::likely(Self::mat4_inverse_inplace(&mut mat) != Self::Element::ZERO) {
            Some(mat)
        } else {
            None
        }
    }
}
