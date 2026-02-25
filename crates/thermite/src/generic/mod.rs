#![allow(missing_docs, clippy::missing_safety_doc)]
#![deny(unconditional_recursion)] // just in case we miss one

use core::ops::{BitAnd, BitAndAssign, BitOr, BitOrAssign, BitXor, BitXorAssign, Not};

pub mod ops;

use bitvec::{array::BitArray, view::BitViewSized};
use generic_array::GenericArray;

use crate::{
    BranchfreeDivider, Divider, Mask, Vector,
    divider::{Denominator, vector::VectorDivider},
    element::{FloatElementWithBits, UnsignedIntegerElement},
    isa::InstructionSet,
    math::FloatConsts,
    register::{Element, FloatElement, Lanes},
};

/// Simple associated constant splat trait.
///
/// Used with `GenericVector::splat_const` to splat compile-time constant values into vectors.
///
/// This is effectively a workaround for the lack of `const generics` for generic types.
pub trait SplatConst<E> {
    const VALUE: E;
}

/// Macro to splat a compile-time constant value into all lanes of a generic vector.
///
/// There are three forms of this macro:
/// ```ignore
/// // 1. Generic type parameters with bounds
/// // used when the type depends on generic parameters, and especially `Self`
/// let exp_lsb_mask: Self::Bits = crate::generic_splat!(
///     <Self> = <S: FloatVector>
///     <S::Bits as GenericVector>::Element: <S::Element as FloatElement>::EXP_LSB_MASK
/// );
///
/// // 2. Static type, direct value
/// // used when the type is known and the value is a literal or const expression
/// let zero: Vector<u32x4> = crate::generic_splat!(u32: 0);
///
/// // 3. Associated const value
/// // used when the value is an associated constant of a type
/// let infinity: Vector<f32x4> = crate::generic_splat!(<f32>::INFINITY);
/// ```
#[macro_export]
macro_rules! generic_splat {
    (
        <$($real_param:ident),+> = <$($gen_param:ident $(: $bound:path)?),+ $(,)?>
        $ty:ty : $value:expr
    ) => {{
        use core::marker::PhantomData;

        struct __GenericSplatValue<$($gen_param $(: $bound)?),+>(
            PhantomData<($($gen_param),+)>
        );

        impl<$($gen_param $(: $bound)?),+> $crate::generic::SplatConst<$ty>
        for __GenericSplatValue<$($gen_param),+> {
            const VALUE: $ty = const { $value };
        }

        $crate::generic::GenericVector::splat_const::<
            __GenericSplatValue<$($real_param),+>
        >()
    }};

    // Static type, direct value
    ($ty:ty: $value:expr) => {{
        struct __ConstSplatValue;
        impl $crate::generic::SplatConst<$ty> for __ConstSplatValue {
            const VALUE: $ty = const { $value };
        }
        $crate::generic::GenericVector::splat_const::<__ConstSplatValue>()
    }};

    // Associated const value
    (<$ty:ty $(as $trait:path)?>::$associated:ident) => {{
        struct __ConstSplatValue;
        impl $crate::generic::SplatConst<$ty> for __ConstSplatValue {
            const VALUE: $ty = const { <$ty $(as $trait)?>::$associated };
        }
        $crate::generic::GenericVector::splat_const::<__ConstSplatValue>()
    }};
}

pub trait MaskInteroperable<A, B>: GenericVector<Mask: CastMask<A::Mask> + CastMask<B::Mask>>
where
    A: GenericVector<Lanes = Self::Lanes, Mask: CastMask<Self::Mask> + CastMask<B::Mask>>,
    B: GenericVector<Lanes = Self::Lanes, Mask: CastMask<Self::Mask> + CastMask<A::Mask>>,
{
}

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
    fn len_to_indices<I: VectorIndices<Self>>(len: usize) -> I {
        let Ok(len) = <<I as GenericVector>::Element as TryFrom<usize>>::try_from(len) else {
            panic!("Length {} exceeds maximum supported index for this vector type", len);
        };

        I::splat(len)
    }
}

impl<V: GenericVector> GenericVectorExt for V {}

/// Type that can be used as indices for gather/scatter operations of the vector type `V`
pub trait VectorIndices<V: GenericVector>: UnsignedIntegerVector<Lanes = V::Lanes> {
    unsafe fn gather_ptr(ptr: *const V::Element, indices: Self) -> V;
    unsafe fn gather_ptr_m(src: V, mask: V::Mask, ptr: *const V::Element, indices: Self) -> V;
    unsafe fn gather_ptr_z(mask: V::Mask, ptr: *const V::Element, indices: Self) -> V;

    unsafe fn scatter_ptr(value: V, ptr: *mut V::Element, indices: Self);
    unsafe fn scatter_ptr_m(value: V, mask: V::Mask, ptr: *mut V::Element, indices: Self);
}

pub trait IndexableVector<I: UnsignedIntegerVector<Lanes = Self::Lanes>>: GenericVector {
    unsafe fn gather_ptr(ptr: *const Self::Element, indices: I) -> Self;
    unsafe fn gather_ptr_m(src: Self, mask: Self::Mask, ptr: *const Self::Element, indices: I) -> Self;
    unsafe fn gather_ptr_z(mask: Self::Mask, ptr: *const Self::Element, indices: I) -> Self;

    unsafe fn scatter_ptr(value: Self, ptr: *mut Self::Element, indices: I);
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

pub trait Concat<HALF>: Extend<HALF> {
    fn concat(lo: HALF, hi: HALF) -> Self;
    fn split(self) -> (HALF, HALF);
}

/// Zero-extend vectors or masks
pub trait Extend<FROM> {
    fn extend(v: FROM) -> Self;
    fn narrow(self) -> FROM;
}

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

pub trait SwizzleVector: GenericVector + crate::swizzle::Swizzle<Self::Lanes> {}
impl<V> SwizzleVector for V where V: GenericVector + crate::swizzle::Swizzle<V::Lanes> {}

/// Core trait for generic vector types.
///
/// Provides the basis for further specialized vector traits.
#[rustfmt::skip] #[thermite_macros::vector_trait]
pub trait GenericVector:
    Sized
    + Default
    + Copy
    + core::fmt::Debug
    + 'static
    + GenericSelectable<SelectableMask = Self::Mask>
{
    /// Scalar element type of the vector.
    type Element: Element;

    /// A vector with all elements zeroed.
    const EMPTY: Self;

    /// Number of lanes in the vector.
    const LANES: usize;

    /// The instruction set used by this vector type.
    const ISA: InstructionSet;

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

    /// Create a new vector from a single element by splatting it across all lanes.
    #[masked] fn splat(value: Self::Element) -> Self;

    /// Splat a compile-time constant value into all lanes of the vector.
    ///
    /// Use the `generic_splat!` macro to call this function with easier syntax.
    #[inline(always)]
    fn splat_const<C>() -> Self
    where
        C: SplatConst<Self::Element>,
    {
        Self::splat(C::VALUE)
    }

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
        assert!(slice.len() >= Self::LANES, "Slice must have at least {} elements to create a vector", Self::LANES);

        unsafe { Self::load_unaligned(slice.as_ptr()) }
    }

    /// Copy the elements of the vector into a slice. The slice must have at least as many elements as the vector's lanes.
    ///
    /// This will emit an unaligned store.
    fn copy_to_slice(self, slice: &mut [Self::Element]) {
        assert!(slice.len() >= Self::LANES, "Slice must have at least {} elements to copy from a vector", Self::LANES);

        unsafe { self.store_unaligned(slice.as_mut_ptr()) }
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
            panic!("One or more indices are out of bounds for the slice length {}", slice.len());
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

    #[masked] unsafe fn load(ptr: *const Self::Element) -> Self;

    unsafe fn load_unaligned(ptr: *const Self::Element) -> Self;
    unsafe fn load_streaming(ptr: *const Self::Element) -> Self;

    unsafe fn store(self, ptr: *mut Self::Element);
    unsafe fn store_unaligned(self, ptr: *mut Self::Element);
    unsafe fn store_streaming(self, ptr: *mut Self::Element);

    /*
    /// Gather elements from memory at the specified indices and return a new vector with those elements.
    /// The provided indices are in number of elements, not bytes.
    ///
    /// For the masked variants, memory locations are never read from if the mask is false.
    ///
    /// # Safety
    /// The caller must ensure the given memory locations given by `ptr + (size_of(Element) * index)`
    /// are valid for reading for all indices where the mask is true.
    unsafe fn gather_ptr(ptr: *const Self::Element, indices: Self::Index) -> Self;

    /// Scatter elements from the given vector into memory at the specified indices.
    ///
    /// The provided indices are in number of elements, not bytes.
    #[skip_masked]
    unsafe fn scatter_ptr(self, ptr: *mut Self::Element, indices: Self::Index);

    /// Scatter elements from the given vector into memory at the specified indices,
    /// but only if the corresponding lane of the mask is true.
    ///
    /// The provided indices are in the number of elements, not bytes.
    #[skip_masked]
    unsafe fn scatter_ptr_masked(self, mask: Self::Mask, ptr: *mut Self::Element, indices: Self::Index);
    */

    /// Broadcast the value of a single lane across all lanes of the vector.
    #[conditional] fn broadcast<const I: usize>(self) -> Self;

    /// Broadcast the value of a single lane across all lanes of the vector.
    ///
    /// # Panics
    /// If `idx` is out of bounds for the vector's lanes.
    #[conditional] fn broadcastv(self, idx: usize) -> Self;

    /// Extract a single element from the vector at the given index.
    fn extract<const I: usize>(self) -> Self::Element;

    fn extractv(self, idx: usize) -> Self::Element;

    /// Replace a single element in the vector at the given index with a new value.
    fn insert<const I: usize>(self, value: Self::Element) -> Self;

    fn insertv(self, idx: usize, value: Self::Element) -> Self;

    /// Reverse the order of the elements in the vector.
    #[conditional] fn reverse(self) -> Self;

    /// Swap the byte order of each element in the vector. i.e., converts between little-endian and big-endian.
    #[conditional] fn swap_bytes(self) -> Self;

    /// (Zero If False) Zero elements if the corresponding mask lane is false; otherwise, leave unchanged.
    ///
    /// Similar to a `mask & self` operation.
    fn z(self, mask: Self::Mask) -> Self;

    /// (Zero If True) Zero elements if the corresponding mask lane is true; otherwise, leave unchanged.
    ///
    /// Similar to a `!mask & self` operation.
    fn nz(self, mask: Self::Mask) -> Self;

    /// Whether the register type has a simple unpack implementation,
    /// or requires a more complex method.
    const HAS_SIMPLE_UNPACK: bool;

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
    fn unpack(self, other: Self) -> (Self, Self);

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

    #[inline(always)] fn cast<INTO>(self) -> INTO
    where
        INTO: CastVector<Self>,
    {
        INTO::cast_from(self)
    }

    #[inline(always)] fn fast_cast<INTO>(self) -> INTO
    where
        INTO: CastVector<Self>,
    {
        INTO::fast_cast_from(self)
    }

    #[inline(always)] fn into_bits<INTO>(self) -> INTO
    where
        INTO: BitCastVector<Self>,
    {
        INTO::from_bits(self)
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
    const HAS_TRUE_SHIFTV: bool;
    const HAS_WIDE_BYTE_SHIFTS: bool;

    /// Treats the entire vector as a single large integer and shifts left by the immediate value
    /// number of BYTES. Not bits, bytes.
    #[conditional] fn bshli<const I: i32>(self) -> Self;

    /// Treats the entire vector as a single large integer and shifts right by the immediate value
    /// number of BYTES. Not bits, bytes.
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

pub trait CastVector<FROM: Sized>: Sized {
    fn cast_from(from: FROM) -> Self;
    fn cast_into(self) -> FROM;

    #[inline(always)]
    fn fast_cast_from(from: FROM) -> Self {
        Self::cast_from(from)
    }

    #[inline(always)]
    fn fast_cast_into(self) -> FROM {
        Self::cast_into(self)
    }
}

pub trait BitCastVector<FROM: Sized>: Sized {
    fn from_bits(bits: FROM) -> Self;
}

pub trait GenericMask:
    'static
    + Sized
    + Copy
    + Default
    + core::fmt::Debug
    + CastMask<Self>
    + BitAnd<Self, Output = Self>
    + BitAndAssign<Self>
    + ops::BitAndNot<Self, Output = Self>
    + ops::BitAndNotAssign<Self>
    + BitOr<Self, Output = Self>
    + BitOrAssign<Self>
    + BitXor<Self, Output = Self>
    + BitXorAssign<Self>
    + Not<Output = Self>
{
    const TRUTHY: Self;
    const FALSY: Self;

    fn all(self) -> bool;
    fn any(self) -> bool;
    fn none(self) -> bool;

    fn native_bitmask(&self) -> Option<u64>;

    fn bitmask(&self) -> BitArray<impl BitViewSized<Store = u32>>;

    #[inline(always)]
    fn select<S>(self, t: S, f: S) -> S
    where
        S: GenericSelectable<SelectableMask: CastMask<Self>>,
    {
        S::select(self, t, f)
    }

    #[inline(always)]
    fn cast<INTO>(self) -> INTO
    where
        INTO: CastMask<Self>,
    {
        INTO::mask_from(self)
    }

    #[inline(always)]
    fn swap<S>(self, a: &mut S, b: &mut S)
    where
        S: GenericSelectable<SelectableMask: CastMask<Self> + CastMask<S::SelectableMask>>,
    {
        let mask = S::SelectableMask::mask_from(self);

        let a2 = S::select(mask, *a, *b);
        let b2 = S::select(mask, *b, *a);

        *a = a2;
        *b = b2;
    }

    fn ternlog<const IMM: i32>(a: Self, b: Self, c: Self) -> Self;
}

pub trait CastMask<FROM>: Sized {
    fn mask_from(from: FROM) -> Self;
}

pub trait GenericSelectable: Copy {
    type SelectableMask: Copy;

    fn select<M>(mask: M, t: Self, f: Self) -> Self
    where
        Self::SelectableMask: CastMask<M>;
}

pub trait PartialOrdVector: GenericVector + PartialEq {
    fn cmp_lt(self, other: Self) -> Self::Mask;
    fn cmp_le(self, other: Self) -> Self::Mask;
    fn cmp_gt(self, other: Self) -> Self::Mask;
    fn cmp_ge(self, other: Self) -> Self::Mask;
    fn cmp_eq(self, other: Self) -> Self::Mask;
    fn cmp_ne(self, other: Self) -> Self::Mask;
}

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

    /// Returns the sum of all elements in the vector.
    ///
    /// This operation has an `O(log2 n)` complexity to reduce.
    fn sum_elements(self) -> Self::Element;

    /// Returns the product of all elements in the vector.
    ///
    /// This operation has an `O(log2 n)` complexity to reduce.
    fn prod_elements(self) -> Self::Element;

    /// Effectively returns `Self::splat(Self::LANES as Self::Element)`.
    fn offset() -> Self;

    /// Returns a vector where each element is the index of the lane as that element type.
    ///
    /// `[0, 1, 2, 3]`, etc.
    fn indexed() -> Self;
}

pub trait NumVector:
    NumericVector<Element: num_traits::Num>
    + num_traits::Num
    + num_traits::NumCast
    + num_traits::NumAssign
    + num_traits::ConstOne
    + num_traits::ConstZero
{
}

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

pub trait NumSignedVector: SignedVector + NumVector + num_traits::Signed {}

#[rustfmt::skip] #[thermite_macros::vector_trait]
pub trait IntegerVector:
    NumericVector<Element: Denominator>
    + BitshiftVector
    + ops::DivMasked<Self::Mask, Self::Divider, Output = Self>
    + ops::DivMasked<Self::Mask, Self::BranchfreeDivider, Output = Self>
{
    type Divider: Copy;
    type BranchfreeDivider: Copy;
    type VectorizedDivider: Copy;

    /// Multiply two vectors, returning the high half of each product.
    #[conditional] fn mulhi(self, other: Self) -> Self;

    /// Multiply two vectors, returning the low half of each product.
    ///
    /// This is usually the same as regular multiplication, but some architectures
    /// have specialized instructions for this operation.
    #[conditional] fn mullo(self, other: Self) -> Self;

    // fn wrapping_add(self, other: Self) -> Self;
    // fn wrapping_sub(self, other: Self) -> Self;
    // fn wrapping_mul(self, other: Self) -> Self;

    /// Perform saturating addition for each element of the vectors.
    #[conditional] fn saturating_add(self, other: Self) -> Self;

    /// Perform saturating subtraction for each element of the vectors.
    #[conditional] fn saturating_sub(self, other: Self) -> Self;

    #[conditional] fn wrapping_sum(self) -> Self::Element;
    #[conditional] fn wrapping_prod(self) -> Self::Element;

    fn create_divider(d: Self::Element) -> Self::Divider;
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
}

#[rustfmt::skip] #[thermite_macros::vector_trait]
pub trait UnsignedIntegerVector: IntegerVector<Element: crate::element::UnsignedIntegerElement> {
    /// Determines if each unsigned integer element in the vector is a
    /// power of two, returning a mask indicating whether or not it is.
    fn is_power_of_two(self) -> Self::Mask;

    /// Returns the next power of two minus one for each unsigned integer
    /// element in the vector.
    #[conditional] fn next_power_of_two_m1(self) -> Self;
    /// Computes log2(x) + 1 for each unsigned integer element in the vector.
    #[conditional] fn ilog2p1(self) -> Self;

    /// Compute the parity of each unsigned integer lane in the vector.
    #[conditional] fn parity(self) -> Self;
}

use num_traits::float::FloatCore as CoreFloatTrait;

pub trait NumFloatVector:
    FloatVector
    + NumSignedVector
    + num_traits::FloatConst
    + CoreFloatTrait
    + num_traits::MulAdd<Self, Self, Output = Self>
    + num_traits::MulAddAssign<Self, Self>
{
}

/// Float vector types which have an associated hardware register type.
pub trait FloatVectorWithRegister: FloatVectorWithBits<Mask = crate::Mask<Self::Register>> {
    type Register: crate::register::FloatRegister<Element = Self::Element, Lanes = Self::Lanes>;
}

/// SignedBits integer vector types which have an associated hardware register type.
pub trait SignedIntegerVectorWithRegister: SignedIntegerVector<Mask = crate::Mask<Self::Register>> {
    type Register: crate::register::SignedIntegerRegister<Element = Self::Element, Lanes = Self::Lanes>;
}

/// Unsigned integer vector types which have an associated hardware register type.
pub trait UnsignedIntegerVectorWithRegister: UnsignedIntegerVector<Mask = crate::Mask<Self::Register>> {
    type Register: crate::register::UnsignedIntegerRegister<Element = Self::Element, Lanes = Self::Lanes>;
}

#[rustfmt::skip] #[thermite_macros::vector_trait]
pub trait FloatVector: SignedVector<Element: FloatElement>
    + FloatConsts
    + CastVector<Self::ExtendedPrecision>
    + ops::MulAddExtMasked<Self::Mask, Self, Self, Output = Self>
    + ops::MulAddAssignExtMasked<Self::Mask, Self, Self>
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

    const HAS_APPROX_RCP: bool;
    const HAS_APPROX_RSQRT: bool;

    /// Square root
    #[conditional] fn sqrt(self) -> Self;

    /// Approximate reciprocal square root, hardware dependent accuracy.
    #[conditional] fn rsqrt(self) -> Self;

    /// Approximate reciprocal, hardware dependent accuracy.
    #[conditional] fn rcp(self) -> Self;

    #[conditional] fn floor(self) -> Self;
    #[conditional] fn ceil(self) -> Self;

    /// Round to nearest
    #[conditional] fn round(self) -> Self;
    /// Truncate to int
    #[conditional] fn trunc(self) -> Self;
    /// Fractional part
    #[conditional] fn fract(self) -> Self;

    /// Effectively `self * sign.signum()`, multiplying the sign bits.
    #[conditional] fn mul_sign(self, sign: Self) -> Self;

    /// Returns zero with the sign of `self`, i.e.: only the sign bit is set.
    #[conditional] fn signed_zero(self) -> Self;

    /// Returns the next representable value greater than the current value, towards positive infinity.
    #[conditional] fn next_up(self) -> Self;

    /// Returns the next representable value less than the current value, towards negative infinity.
    #[conditional] fn next_down(self) -> Self;

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

        impl<V: FloatVector> $crate::generic::AsFloatVectorWithBitsKernel<V, $len> for AnonymousAsFloatVectorWithBitsKernel<V>
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

// These do not have masked variants
pub trait FloatVectorWithBits:
    BitwiseVector + FloatVector<Element: FloatElementWithBits> + FullyInteroperable<Self::Bits, Self::SignedBits>
{
    type SignedBits: SignedIntegerVector<
            Lanes = Self::Lanes,
            Divider = Divider<<Self::Element as FloatElementWithBits>::SignedBits>,
            BranchfreeDivider = BranchfreeDivider<<Self::Element as FloatElementWithBits>::SignedBits>,
            Element = <Self::Element as FloatElementWithBits>::SignedBits,
        > + FullyInteroperable<Self, Self::Bits>;

    type Bits: UnsignedIntegerVector<
            Lanes = Self::Lanes,
            Divider = Divider<<Self::Element as FloatElementWithBits>::Bits>,
            BranchfreeDivider = BranchfreeDivider<<Self::Element as FloatElementWithBits>::Bits>,
            Element = <Self::Element as FloatElementWithBits>::Bits,
        > + FullyInteroperable<Self, Self::SignedBits>;

    const HAS_NATIVE_LDEXP: bool;
    const HAS_NATIVE_FREXP: bool;

    unsafe fn native_ldexp(self, exp: Self::SignedBits) -> Self;
    unsafe fn native_frexp(self) -> (Self, Self::SignedBits);

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
}

/// Vector suitable for 3D linear algebra operations.
///
/// The length of this vector must be either 3 or 4 lanes.
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

    /// In-place 4x4 Matrix inversion.
    ///
    /// Returns `true` if the matrix was successfully inverted,
    /// or `false` if the matrix is singular and could not be inverted.
    fn mat4_inverse_inplace(m: &mut [Self; 4]) -> bool;

    /// 4x4 Matrix inversion.
    ///
    /// Returns `Some(inverted_matrix)` if the matrix was successfully inverted,
    /// or `None` if the matrix is singular and could not be inverted.
    ///
    /// Consider using [`Vector::mat4_inverse_inplace`] if you want to avoid
    /// an extra copy.
    #[inline(always)]
    fn mat4_inverse(m: &[Self; 4]) -> Option<[Self; 4]> {
        let mut mat = *m;
        if Self::mat4_inverse_inplace(&mut mat) {
            Some(mat)
        } else {
            None
        }
    }
}

mod vector;
