#![no_std]
// stdmind for f16c instructions, core_intrinsics for likely/unlikely
#![cfg_attr(feature = "nightly", feature(stdsimd, core_intrinsics))]
#![allow(unused_imports, non_camel_case_types, non_snake_case)]

pub use thermite_dispatch::dispatch;

#[macro_use]
#[doc(hidden)]
pub mod macros;

#[macro_use]
mod runtime;

use half::f16;

pub mod arch;

#[cfg(feature = "alloc")]
mod buffer;
#[cfg(feature = "alloc")]
pub use buffer::VectorBuffer;

pub mod backends;

mod divider;
pub use divider::*;

mod pointer;
pub use self::pointer::VPtr;
use self::pointer::*;

mod mask;
pub use mask::{BitMask, Mask};

pub mod element;
pub use element::SimdElement;

#[cfg(feature = "math")]
pub mod math;
#[cfg(feature = "math")]
pub use math::{SimdVectorizedMath, *};

#[cfg(not(feature = "math"))]
/// "math" features is disabled. This trait is empty.
pub trait SimdVectorizedMath<S: Simd> {}

#[cfg(not(feature = "math"))]
impl<S: Simd, T> SimdVectorizedMath<S> for T where T: SimdFloatVector<S> {}

#[cfg(feature = "rng")]
pub mod rng;

pub mod iter;
pub use iter::*;

use core::{fmt::Debug, marker::PhantomData, mem, ops::*, ptr};

/// Describes casting from one SIMD vector type to another
///
/// This should handle extending bits correctly
pub trait SimdFromCast<S: Simd, FROM>: Sized {
    /// Casts one vector to another, performing proper numeric conversions on each element.
    ///
    /// This is equivalent to the `as` keyword in Rust, but for SIMD vectors.
    fn from_cast(from: FROM) -> Self;

    /// Casts one mask to another, not caring about the value types,
    /// but rather expanding or truncating the mask bits as efficiently as possible.
    fn from_cast_mask(from: Mask<S, FROM>) -> Mask<S, Self>;
}

impl<S: Simd, T> SimdFromCast<S, T> for T {
    #[inline(always)]
    fn from_cast(from: T) -> T {
        from
    }

    #[inline(always)]
    fn from_cast_mask(from: Mask<S, T>) -> Mask<S, Self> {
        from
    }
}

/// Describes casting to one SIMD vector type from another
pub trait SimdCastTo<S: Simd, TO>: Sized {
    /// Casts one vector to another, performing proper numeric conversions on each element.
    ///
    /// This is equivalent to the `as` keyword in Rust, but for SIMD vectors.
    fn cast(self) -> TO;

    /// Casts one mask to another, not caring about the value types,
    /// but rather expanding or truncating the mask bits as efficiently as possible.
    fn cast_mask(mask: Mask<S, Self>) -> Mask<S, TO>;
}

impl<S: Simd, FROM, TO> SimdCastTo<S, TO> for FROM
where
    TO: SimdFromCast<S, FROM>,
{
    #[inline(always)]
    fn cast(self) -> TO {
        TO::from_cast(self)
    }

    #[inline(always)]
    fn cast_mask(mask: Mask<S, Self>) -> Mask<S, TO> {
        TO::from_cast_mask(mask)
    }
}

// NOTE: Casting from a floating point value to mask will just call `f.is_normal()`
/// List of valid casts between SIMD types in an instruction set
pub trait SimdCasts<S: Simd + ?Sized>:
    Sized
    + SimdFromCast<S, S::Vi32>
    + SimdFromCast<S, S::Vu32>
    + SimdFromCast<S, S::Vu64>
    + SimdFromCast<S, S::Vf32>
    + SimdFromCast<S, S::Vf64>
    + SimdFromCast<S, S::Vi64>
{
    #[inline(always)]
    fn cast_to<T: SimdFromCast<S, Self>>(self) -> T {
        self.cast()
    }
}

impl<S: Simd + ?Sized, T> SimdCasts<S> for T where
    T: Sized
        + SimdFromCast<S, S::Vi32>
        + SimdFromCast<S, S::Vu32>
        + SimdFromCast<S, S::Vu64>
        + SimdFromCast<S, S::Vf32>
        + SimdFromCast<S, S::Vf64>
        + SimdFromCast<S, S::Vi64>
{
}

/// Helper trait for constant vector shuffles
pub trait SimdShuffleIndices {
    const INDICES: &'static [usize];
}

/// Shuffles the elements in one or two vectors into a new vector given the indices provided.
///
/// Under the hood this generates an anonymous struct implementing [`SimdShuffleIndices`],
/// then calls [`SimdVectorBase::shuffle`] on a vector or two. Shuffles will be monomorphized to generate
/// ideal code wherever possible.
///
/// The length of the indices array is not checked at compile-time, but will panic at runtime
/// if incorrect or if any of the elements are out of bounds for the vectors.
///
/// You can use a crate such as [`no-panic`](https://crates.io/crates/no-panic) to statically ensure
/// all the indices are valid. `no-panic` checks at link-time, so branches pruned via dead-code removal
/// will not contribute, allowing you to do things like:
///
/// ```ignore
/// match Vf32::NUM_ELEMENTS {
///     4 => shuffle!(x, y, [6, 2, 1, 7]),
///     8 => shuffle!(x, y, [5, 6, 10, 9, 2, 8, 6, 4]),
///     _ => unimplemented!(),
/// }
/// ```
#[macro_export]
macro_rules! shuffle {
    ($a:expr, $b:expr, [$($idx:expr),*]) => {{
        ($a).shuffle($b, {
            struct __Indices;
            impl $crate::SimdShuffleIndices for __Indices {
                const INDICES: &'static [usize] = &[$($idx),*];
            }
            __Indices
        })
    }};
    ($a:expr, [$($idx:expr),*]) => {
        shuffle!($a, $a, [$($idx),*])
    };
}

/// Basic shared vector interface
pub trait SimdVectorBase<S: Simd + ?Sized>: 'static + Sized + Copy + Debug + Default + Send + Sync {
    type Element: SimdElement;

    /// Size of element type in bytes
    const ELEMENT_SIZE: usize = core::mem::size_of::<Self::Element>();
    const NUM_ELEMENTS: usize = core::mem::size_of::<S::Vi32>() / core::mem::size_of::<i32>();
    const ALIGNMENT: usize = core::mem::align_of::<Self>();

    /// Creates a new vector with all lanes set to the given value
    fn splat(value: Self::Element) -> Self;

    /// Returns a vector containing possibly undefined or uninitialized data
    #[inline(always)]
    unsafe fn undefined() -> Self {
        Self::default()
    }

    /// Same as `splat`, but is more convenient for initializing with data that can be converted into the element type.
    #[inline(always)]
    fn splat_any(value: impl Into<Self::Element>) -> Self {
        Self::splat(value.into())
    }

    /// Splats a value by casting to the element type via `value as Element`.
    ///
    /// This is limited to primitive numeric types.
    #[inline(always)]
    fn splat_as<E>(value: E) -> Self
    where
        Self::Element: element::CastFrom<E>,
    {
        Self::splat(element::CastFrom::cast_from(value))
    }

    /// Shuffles between two vectors based on the static indices provided in `INDICES`
    ///
    /// See the [`shuffle!`] macro for more information.
    ///
    /// **NOTE**: This method will panic if the indices are the incorrect length or out of bounds.
    #[inline(always)]
    fn shuffle<INDICES: SimdShuffleIndices>(self, b: Self, indices: INDICES) -> Self {
        assert!(
            INDICES::INDICES.len() == Self::NUM_ELEMENTS
                && INDICES::INDICES.iter().all(|i| *i < Self::NUM_ELEMENTS * 2)
        );
        unsafe { self.shuffle_unchecked(b, indices) }
    }

    /// Shuffles between two vectors based on the static indices provided in `INDICES`
    ///
    /// See the [`shuffle!`] macro for more information.
    ///
    /// **NOTE**: Calling this with invalid indices (incorrect length or out-of-bounds values)
    /// will result in undefined behavior.
    unsafe fn shuffle_unchecked<INDICES: SimdShuffleIndices>(self, b: Self, indices: INDICES) -> Self;

    /// Shuffles between two vectors based on the dynamic indices provided.
    ///
    /// This differs from the [`shuffle!`] macro and [`Self::shuffle`] methods in that these indices can
    /// indeed be dynamic, at the cost of performance.
    ///
    /// **NOTE**: This method will panic if the indices are the incorrect length or out of bounds.
    #[inline(always)]
    fn shuffle_dyn(self, b: Self, indices: &[usize]) -> Self {
        assert!(indices.len() == Self::NUM_ELEMENTS);
        unsafe {
            let mut res = Self::undefined();
            for i in 0..Self::NUM_ELEMENTS {
                let idx = *indices.get_unchecked(i);
                res = res.replace_unchecked(i, {
                    if idx < Self::NUM_ELEMENTS {
                        self.extract(idx)
                    } else {
                        b.extract(idx - Self::NUM_ELEMENTS)
                    }
                });
            }
            res
        }
    }

    /// Like [`Self::shuffle_dyn`], but does not check for valid indices or input length.
    ///
    /// **NOTE**: Calling this with invalid indices (incorrect length or out-of-bounds values)
    /// will result in undefined behavior.
    #[inline(always)]
    unsafe fn shuffle_dyn_unchecked(self, b: Self, indices: &[usize]) -> Self {
        let mut res = Self::undefined();
        for i in 0..Self::NUM_ELEMENTS {
            let idx = *indices.get_unchecked(i);
            res = res.replace_unchecked(i, {
                if idx < Self::NUM_ELEMENTS {
                    self.extract_unchecked(idx)
                } else {
                    b.extract_unchecked(idx - Self::NUM_ELEMENTS)
                }
            });
        }
        res
    }

    #[inline(always)]
    #[cfg(feature = "alloc")]
    fn alloc(count: usize) -> VectorBuffer<S, Self> {
        VectorBuffer::alloc(count)
    }

    /// Extracts an element at the given lane index.
    ///
    /// **WARNING**: Will panic if the index is not less than `NUM_ELEMENTS`
    #[inline]
    fn extract(self, index: usize) -> Self::Element {
        assert!(index < Self::NUM_ELEMENTS);
        unsafe { self.extract_unchecked(index) }
    }

    /// Returns a new vector with the given value at the given lane index.
    ///
    /// **WARNING**: Will panic if the index if not less than `NUM_ELEMENTS`
    #[inline]
    fn replace(self, index: usize, value: Self::Element) -> Self {
        assert!(index < Self::NUM_ELEMENTS);
        unsafe { self.replace_unchecked(index, value) }
    }

    /// Extracts an element at the given lane index.
    ///
    /// **WARNING**: Will result in undefined behavior if the index is not less than `NUM_ELEMENTS`
    unsafe fn extract_unchecked(self, index: usize) -> Self::Element;

    /// Returns a new vector with the given value at the given lane index.
    ///
    /// **WARNING**: Will result in undefined behavior if the index is not less than `NUM_ELEMENTS`
    unsafe fn replace_unchecked(self, index: usize, value: Self::Element) -> Self;

    /// Loads a vector from a slice that has an alignment of at least `Self::ALIGNMENT`
    ///
    /// **WARNING**: Will panic if the slice is not properly aligned or is not long enough.
    #[inline]
    fn load_aligned(src: &[Self::Element]) -> Self {
        assert!(src.len() >= Self::NUM_ELEMENTS);
        let load_ptr = src.as_ptr();
        assert_eq!(
            0,
            load_ptr.align_offset(Self::ALIGNMENT),
            "source slice is not aligned properly"
        );
        unsafe { Self::load_aligned_unchecked(load_ptr) }
    }

    /// Loads a vector from a slice
    ///
    /// **WARNING**: Will panic if the slice is not long enough.
    #[inline]
    fn load_unaligned(src: &[Self::Element]) -> Self {
        assert!(src.len() >= Self::NUM_ELEMENTS);
        unsafe { Self::load_unaligned_unchecked(src.as_ptr()) }
    }

    /// Stores a vector into a slice with an alignment of at least `Self::ALIGNMENT`
    ///
    /// **WARNING**: Will panic if the target slice is not properly aligned or is not long enough.
    #[inline]
    fn store_aligned(self, dst: &mut [Self::Element]) {
        assert!(dst.len() >= Self::NUM_ELEMENTS);
        let store_ptr = dst.as_mut_ptr();
        assert_eq!(
            0,
            store_ptr.align_offset(Self::ALIGNMENT),
            "target slice is not aligned properly"
        );
        unsafe { self.store_aligned_unchecked(store_ptr) };
    }

    /// Stores a vector into a slice.
    ///
    /// **WARNING**: Will panic if the slice is not long enough.
    #[inline]
    fn store_unaligned(self, dst: &mut [Self::Element]) {
        assert!(dst.len() >= Self::NUM_ELEMENTS);
        unsafe { self.store_unaligned_unchecked(dst.as_mut_ptr()) };
    }

    /// Loads a vector from the given aligned address.
    ///
    /// **WARNING**: Will cause undefined behavior if the pointer is not properly aligned or does
    /// not point to a valid address range.
    #[inline(always)]
    unsafe fn load_aligned_unchecked(src: *const Self::Element) -> Self {
        Self::load_unaligned_unchecked(src)
    }

    /// Stores a vector to the given aligned address.
    ///
    /// **WARNING**: Will cause undefined behavior if the pointer is not properly aligned or does
    /// not point to a valid address range.
    #[inline(always)]
    unsafe fn store_aligned_unchecked(self, dst: *mut Self::Element) {
        self.store_unaligned_unchecked(dst);
    }

    /// Loads a vector from a given address (does not have to be aligned).
    ///
    /// **WARNING**: Will cause undefined behavior if the pointer does not point to a valid address range.
    #[inline(always)]
    unsafe fn load_unaligned_unchecked(src: *const Self::Element) -> Self {
        (src as *const Self).read_unaligned()
    }

    /// Stores a vector to a given address (does not have to be aligned).
    ///
    /// **WARNING**: Will cause undefined behavior if the pointer does not point to a valid address range.
    #[inline(always)]
    unsafe fn store_unaligned_unchecked(self, dst: *mut Self::Element) {
        (dst as *mut Self).write_unaligned(self)
    }

    #[inline(always)]
    unsafe fn gather_unchecked(src: *const Self::Element, indices: S::Vi32) -> Self {
        Self::gather_masked_unchecked(src, indices, Mask::truthy(), Self::default())
    }

    #[inline(always)]
    unsafe fn scatter_unchecked(self, dst: *mut Self::Element, indices: S::Vi32) {
        self.scatter_masked_unchecked(dst, indices, Mask::truthy())
    }

    /// Like `Self::gather`, but individual lanes are loaded based on the corresponding lane of the mask.
    /// If the mask lane is truthy, the source lane is loaded, otherwise it's given the lane value from `default`.
    ///
    /// Lanes with a falsey mask value do not load, and does not cause undefined behavior
    /// if the source address is invalid for that lane.
    #[inline(always)]
    unsafe fn gather_masked_unchecked(
        base_ptr: *const Self::Element,
        indices: S::Vi32,
        mask: Mask<S, Self>,
        default: Self,
    ) -> Self {
        let mut res = default;
        for i in 0..Self::NUM_ELEMENTS {
            if mask.extract_unchecked(i) {
                res = res.replace_unchecked(i, base_ptr.offset(indices.extract_unchecked(i) as isize).read());
            }
        }
        res
    }

    /// Like `self.scatter()`, but individual lanes are stored based on the corresponding lane of the mask.
    /// If the mask lane is truthy, the destination lane is written to, otherwise it is a no-op.
    ///
    /// Lanes with a falsey mask value are not written to, and does not cause undefined behavior
    /// if the destination address is invalid for that lane.
    #[inline(always)]
    unsafe fn scatter_masked_unchecked(self, base_ptr: *mut Self::Element, indices: S::Vi32, mask: Mask<S, Self>) {
        for i in 0..Self::NUM_ELEMENTS {
            if mask.extract_unchecked(i) {
                base_ptr
                    .offset(indices.extract_unchecked(i) as isize)
                    .write(self.extract_unchecked(i));
            }
        }
    }
}

/*
/// Extra scalar operator overloads, exists because stable Rust is bugged
pub trait SimdBitwiseExtra<E>:
    Sized
    + BitAnd<E, Output = Self>
    + BitOr<E, Output = Self>
    + BitXor<E, Output = Self>
    + BitAndAssign<E>
    + BitOrAssign<E>
    + BitXorAssign<E>
where
    E: Sized + BitAnd<Self, Output = Self> + BitOr<Self, Output = Self> + BitXor<Self, Output = Self>,
{
}

impl<E, V> SimdBitwiseExtra<E> for V
where
    V: Sized
        + BitAnd<E, Output = Self>
        + BitOr<E, Output = Self>
        + BitXor<E, Output = Self>
        + BitAndAssign<E>
        + BitOrAssign<E>
        + BitXorAssign<E>,
    E: Sized + BitAnd<V, Output = V> + BitOr<V, Output = V> + BitXor<V, Output = V>,
{
}
*/

/// Defines bitwise operations on vectors
pub trait SimdBitwise<S: Simd + ?Sized>:
    SimdVectorBase<S>
    + Not<Output = Self>
    + BitAnd<Self, Output = Self>
    + BitOr<Self, Output = Self>
    + BitXor<Self, Output = Self>
    + BitAndAssign<Self>
    + BitOrAssign<Self>
    + BitXorAssign<Self>
    + Shl<S::Vu32, Output = Self>
    + Shl<u32, Output = Self>
    + ShlAssign<S::Vu32>
    + ShlAssign<u32>
    + Shr<S::Vu32, Output = Self>
    + Shr<u32, Output = Self>
    + ShrAssign<S::Vu32>
    + ShrAssign<u32>
{
    /// Computes `!self & other`, may be more performant than the naive version
    #[inline(always)]
    fn and_not(self, other: Self) -> Self {
        !self & other
    }

    //fn reduce_and(self) -> Self::Element;
    //fn reduce_or(self) -> Self::Element;
    //fn reduce_xor(self) -> Self::Element;

    /// Bitmask corresponding to all lanes of the mask being truthy.
    const FULL_BITMASK: u16;

    /// Returns an integer where each bit corresponds to the binary truthy-ness of each lane from the mask.
    fn bitmask(self) -> u16;

    #[doc(hidden)]
    unsafe fn _mm_not(self) -> Self;
    #[doc(hidden)]
    unsafe fn _mm_bitand(self, rhs: Self) -> Self;
    #[doc(hidden)]
    unsafe fn _mm_bitor(self, rhs: Self) -> Self;
    #[doc(hidden)]
    unsafe fn _mm_bitxor(self, rhs: Self) -> Self;

    #[doc(hidden)]
    unsafe fn _mm_shr(self, count: Vu32<S>) -> Self;
    #[doc(hidden)]
    unsafe fn _mm_shl(self, count: Vu32<S>) -> Self;

    #[doc(hidden)]
    #[inline(always)]
    unsafe fn _mm_shri(self, count: u32) -> Self {
        self._mm_shr(S::Vu32::splat(count))
    }

    #[doc(hidden)]
    #[inline(always)]
    unsafe fn _mm_shli(self, count: u32) -> Self {
        self._mm_shl(S::Vu32::splat(count))
    }
}

/// Defines a mask type for results and selects
#[doc(hidden)]
pub trait SimdMask<S: Simd + ?Sized>: SimdVectorBase<S> + SimdBitwise<S> {
    #[inline(always)]
    unsafe fn _mm_blendv(self, t: Self, f: Self) -> Self {
        (self & t) | self.and_not(f)
    }

    /// Returns true if **all** lanes are non-zero
    #[inline(always)]
    unsafe fn _mm_all(self) -> bool {
        self.bitmask() == Self::FULL_BITMASK
    }

    /// Returns true if **any** lanes are non-zero
    #[inline(always)]
    unsafe fn _mm_any(self) -> bool {
        self.bitmask() != 0
    }

    /// Returns true if **none** of the lanes are non-zero (all zero)
    #[inline(always)]
    unsafe fn _mm_none(self) -> bool {
        !self._mm_any()
    }
}

/*
/// Extra scalar operator overloads, exists because stable Rust is bugged
pub trait SimdVectorExtra<E>:
    Sized
    + Add<E, Output = Self>
    + Sub<E, Output = Self>
    + Mul<E, Output = Self>
    + Div<E, Output = Self>
    + Rem<E, Output = Self>
    + AddAssign<E>
    + SubAssign<E>
    + MulAssign<E>
    + DivAssign<E>
    + RemAssign<E>
{
}

impl<E, V> SimdVectorExtra<E> for V where
    V: Sized
        + Add<E, Output = Self>
        + Sub<E, Output = Self>
        + Mul<E, Output = Self>
        + Div<E, Output = Self>
        + Rem<E, Output = Self>
        + AddAssign<E>
        + SubAssign<E>
        + MulAssign<E>
        + DivAssign<E>
        + RemAssign<E>
{
}
*/

/// Defines common operations on numeric vectors
pub trait SimdVector<S: Simd + ?Sized>:
    SimdVectorBase<S>
    + SimdMask<S>
    + SimdBitwise<S>
    + SimdCasts<S>
    + Add<Self, Output = Self>
    + Sub<Self, Output = Self>
    + Mul<Self, Output = Self>
    + Div<Self, Output = Self>
    + Rem<Self, Output = Self>
    + AddAssign<Self>
    + SubAssign<Self>
    + MulAssign<Self>
    + DivAssign<Self>
    + RemAssign<Self>
    + PartialEq
{
    /// Loads values from arbitrary addresses in memory based on offsets from a base address.
    #[inline(always)]
    fn gather(src: &[Self::Element], indices: S::Vu32) -> Self {
        Self::gather_masked(src, indices, Mask::truthy(), Self::default())
    }

    /// Stores values to arbitrary addresses in memory based on offsets from a base address.
    #[inline(always)]
    fn scatter(self, dst: &mut [Self::Element], indices: S::Vu32) {
        self.scatter_masked(dst, indices, Mask::truthy())
    }

    #[inline(always)]
    fn gather_masked(src: &[Self::Element], indices: S::Vu32, mask: Mask<S, Self>, default: Self) -> Self {
        // check that all indices are within the bounds of the target slice AND within i32::MAX
        let in_bounds: Mask<S, Self> = indices
            .lt(Vu32::<S>::splat(src.len().min(i32::MAX as usize) as u32))
            .cast_to();
        // if not included in the mask, it's allowed anyway
        assert!((in_bounds | !mask).all());

        unsafe { Self::gather_masked_unchecked(src.as_ptr(), indices.cast(), mask, default) }
    }

    #[inline(always)]
    fn scatter_masked(self, dst: &mut [Self::Element], indices: S::Vu32, mask: Mask<S, Self>) {
        // check that all indices are within the bounds of the target slice AND within i32::MAX
        let in_bounds: Mask<S, Self> = indices
            .lt(Vu32::<S>::splat(dst.len().min(i32::MAX as usize) as u32))
            .cast_to();
        // if not included in the mask, it's allowed anyway
        assert!((in_bounds | !mask).all());

        unsafe { self.scatter_masked_unchecked(dst.as_mut_ptr(), indices.cast(), mask) }
    }

    fn zero() -> Self;
    fn one() -> Self;

    /// Returns a vector where the first lane is zero,
    /// and each subsequent lane is one plus the previous lane.
    ///
    /// `[0, 1, 2, 3, 4, 5, 6, 7, ...]`
    fn indexed() -> Self;

    /// Maximum representable valid value
    fn min_value() -> Self;
    /// Minimum representable valid value (may be negative)
    fn max_value() -> Self;

    /// Per-lane, select the minimum value
    #[inline(always)]
    fn min(self, other: Self) -> Self {
        self.le(other).select(self, other)
    }

    /// Per-lane, select the maximum value
    #[inline(always)]
    fn max(self, other: Self) -> Self {
        self.gt(other).select(self, other)
    }

    /// Find the minimum value across all lanes
    fn min_element(self) -> Self::Element;
    /// Find the maximum value across all lanes
    fn max_element(self) -> Self::Element;

    fn eq(self, other: Self) -> Mask<S, Self>;
    fn gt(self, other: Self) -> Mask<S, Self>;

    /// Add `self` and `value` only if the corresponding lane in the given mask is true.
    #[inline(always)]
    fn conditional_add(self, value: Self, mask: Mask<S, impl SimdCastTo<S, Self>>) -> Self {
        self + (value & SimdCastTo::cast_mask(mask).value())
    }

    /// Subtracts `value` from `self` only if the corresponding lane in the given mask is true.
    #[inline(always)]
    fn conditional_sub(self, value: Self, mask: Mask<S, impl SimdCastTo<S, Self>>) -> Self {
        self - (value & SimdCastTo::cast_mask(mask).value())
    }

    #[inline(always)]
    fn ne(self, other: Self) -> Mask<S, Self> {
        !self.eq(other)
    }
    #[inline(always)]
    fn lt(self, other: Self) -> Mask<S, Self> {
        other.gt(self)
    }
    #[inline(always)]
    fn le(self, other: Self) -> Mask<S, Self> {
        other.ge(self)
    }

    #[inline(always)]
    fn ge(self, other: Self) -> Mask<S, Self> {
        self.gt(other) ^ self.eq(other)
    }

    #[doc(hidden)]
    unsafe fn _mm_add(self, rhs: Self) -> Self;
    #[doc(hidden)]
    unsafe fn _mm_sub(self, rhs: Self) -> Self;
    #[doc(hidden)]
    unsafe fn _mm_mul(self, rhs: Self) -> Self;
    #[doc(hidden)]
    unsafe fn _mm_div(self, rhs: Self) -> Self;
    #[doc(hidden)]
    unsafe fn _mm_rem(self, rhs: Self) -> Self;
}

/// Transmutations into raw bits
pub trait SimdIntoBits<S: Simd + ?Sized, B>: SimdVectorBase<S> {
    #[inline(always)]
    fn into_bits(self) -> B {
        unsafe { core::mem::transmute_copy(&self) }
    }
}

/// Transmutations from raw bits
pub trait SimdFromBits<S: Simd + ?Sized, B>: SimdVectorBase<S> {
    #[inline(always)]
    fn from_bits(bits: B) -> Self {
        unsafe { core::mem::transmute_copy(&bits) }
    }
}

/// Specialized integer division by [`Divider`]s
///
/// This primarily exists as an operator overload hack because Rust pre 1.49.0 is broken
pub trait SimdIntegerDivision<E>:
    Sized + Div<Divider<E>, Output = Self> + Div<BranchfreeDivider<E>, Output = Self>
{
}

impl<T, E> SimdIntegerDivision<E> for T where
    T: Sized + Div<Divider<E>, Output = Self> + Div<BranchfreeDivider<E>, Output = Self>
{
}

/// Integer SIMD vectors
pub trait SimdIntVector<S: Simd + ?Sized>: SimdVector<S> + Eq {
    /// Saturating addition, will not wrap
    fn saturating_add(self, rhs: Self) -> Self;
    /// Saturating subtraction, will not wrap
    fn saturating_sub(self, rhs: Self) -> Self;

    /// Sum all lanes together, wrapping the result if it can't fit in `T`
    fn wrapping_sum(self) -> Self::Element;
    /// Multiply all lanes together, wrapping the result if it can't fit in `T`
    fn wrapping_product(self) -> Self::Element;

    /// Rotates the bits in each lane to the left (towards HSB) by the number of bits specified in `cnt`
    #[inline(always)]
    fn rol(self, cnt: u32) -> Self {
        (self << cnt) | (self >> ((Self::ELEMENT_SIZE as u32 * 8) - cnt))
    }

    /// Rotates the bits in each lane to the right (towards LSB) by the number of bits specified in `cnt`
    #[inline(always)]
    fn ror(self, cnt: u32) -> Self {
        (self >> cnt) | (self << ((Self::ELEMENT_SIZE as u32 * 8) - cnt))
    }

    /// Rotates the bits in each lane to the left (towards HSB) by the number of bits specified in the corresponding lane of `cnt`
    fn rolv(self, cnt: S::Vu32) -> Self;
    /// Rotates the bits in each lane to the right (towards LSB) by the number of bits specified in the corresponding lane of `cnt`
    fn rorv(self, cnt: S::Vu32) -> Self;

    /// Reverses the bits of each lane in the vector.
    fn reverse_bits(self) -> Self;

    /// Counts the number of 1 bits in each lane of the vector.
    fn count_ones(self) -> Self;

    /// Counts the number of 0 bits in each lane of the vector.
    #[inline(always)]
    fn count_zeros(self) -> Self {
        (!self).count_ones()
    }

    /// Counts the number of leading zeros in each lane of the vector.
    fn leading_zeros(self) -> Self;

    /// Counts the number of trailing zeros in each lane of the vector.
    fn trailing_zeros(self) -> Self;

    /// Counts the number of leading ones in each lane of the vector.
    #[inline(always)]
    fn leading_ones(self) -> Self {
        (!self).leading_zeros()
    }

    /// Counts the number of trailing ones in each lane of the vector.
    #[inline(always)]
    fn trailing_ones(self) -> Self {
        (!self).trailing_zeros()
    }
}

/// Unsigned SIMD vector
pub trait SimdUnsignedIntVector<S: Simd + ?Sized>: SimdIntVector<S> {
    /// Returns `floor(log2(x)) + 1`
    #[inline(always)]
    fn log2p1(self) -> Self {
        self.next_power_of_two_m1().count_ones()
    }

    /// Returns a mask wherein if a lane was a power of two, the corresponding mask lane will be truthy
    #[inline(always)]
    fn is_power_of_two(self) -> Mask<S, Self> {
        (self & (self - Self::one())).eq(Self::zero())
    }

    /// Returns `next_power_of_two(x) - 1`
    fn next_power_of_two_m1(self) -> Self;
}

/// Signed SIMD vector, with negative numbers
pub trait SimdSignedVector<S: Simd + ?Sized>: SimdVector<S> + Neg<Output = Self> {
    fn neg_one() -> Self;

    /// Minimum positive number
    fn min_positive() -> Self;

    /// Absolute value
    #[inline(always)]
    fn abs(self) -> Self {
        self.lt(Self::zero()).select(-self, self)
    }

    /// Copies the sign from `sign` to `self`
    #[inline(always)]
    fn copysign(self, sign: Self) -> Self {
        self.abs() * sign.signum()
    }

    /// Returns `-1` if less than zero, `+1` otherwise.
    #[inline(always)]
    fn signum(self) -> Self {
        self.is_negative().select(Self::neg_one(), Self::one())
    }

    /// For each lane, if the mask is true, negate the value.
    fn conditional_neg(self, mask: Mask<S, impl SimdCastTo<S, Self>>) -> Self;

    /// Test if positive, greater or equal to zero
    #[inline(always)]
    fn is_positive(self) -> Mask<S, Self> {
        // TODO: Specialize these to get sign bit (if available)
        self.ge(Self::zero())
    }

    /// Test if negative, less than zero
    #[inline(always)]
    fn is_negative(self) -> Mask<S, Self> {
        // TODO: Specialize these to get sign bit (if available)
        self.lt(Self::zero())
    }

    /// On platforms with true "select" instructions, they often only check the HSB,
    /// which happens to correspond to the "sign" bit for both floats and twos-compliment integers,
    /// so we can save a `cmpgt(self, zero)` by calling this
    ///
    /// On platforms without true "select" instructions, this falls back to `self.is_negative().select(neg, pos)`
    #[inline(always)]
    fn select_negative(self, neg: Self, pos: Self) -> Self {
        self.is_negative().select(neg, pos)
    }

    #[doc(hidden)]
    unsafe fn _mm_neg(self) -> Self;
}

/// Floating point SIMD vectors
pub trait SimdFloatVector<S: Simd + ?Sized>: SimdVector<S> + SimdSignedVector<S> {
    type Vi: SimdIntVector<S> + SimdSignedVector<S>;
    type Vu: SimdIntVector<S>;

    fn epsilon() -> Self;
    fn infinity() -> Self;
    fn neg_infinity() -> Self;
    fn neg_zero() -> Self;
    fn nan() -> Self;

    /// Load half-precision floats and up-convert them into `Self`
    fn load_f16_unaligned(src: &[f16]) -> Self {
        assert!(src.len() >= Self::NUM_ELEMENTS);
        unsafe { Self::load_f16_unaligned_unchecked(src.as_ptr()) }
    }

    /// Down-convert `self` into half-precision and store
    fn store_f16_unaligned(&self, dst: &mut [f16]) {
        assert!(dst.len() >= Self::NUM_ELEMENTS);
        unsafe { self.store_f16_unaligned_unchecked(dst.as_mut_ptr()) };
    }

    unsafe fn load_f16_unaligned_unchecked(src: *const f16) -> Self;
    unsafe fn store_f16_unaligned_unchecked(&self, dst: *mut f16);

    /// Can convert to a signed integer faster than a regular `cast`, but may not provide
    /// correct results above a certain range.
    ///
    /// For example, `f64 -> i64` is only valid from `(-2^51, 2^51)` or so.
    ///
    /// On instruction sets where this can be done in-hardware,
    /// the results should be exact, but do not rely on this for large floats.
    unsafe fn to_int_fast(self) -> Self::Vi;

    /// Can convert to a signed integer faster than a regular `cast`, but may not provide
    /// correct results above a certain range.
    ///
    /// For example, `f64 -> u64` is only valid from `[0, 2^52)` or so.
    ///
    /// On instruction sets where this can be done in-hardware,
    /// the results should be exact, but do not rely on this for large floats.
    unsafe fn to_uint_fast(self) -> Self::Vu;

    /// Same as `self * sign.signum()` or `select(sign_bit(sign), -self, self)`, but more efficient where possible.
    #[inline(always)]
    fn combine_sign(self, sign: Self) -> Self {
        self ^ (sign & Self::neg_zero())
    }

    /// Compute the horizontal sum of all elements
    fn sum(self) -> Self::Element;
    /// Compute the horizontal product of all elements
    fn product(self) -> Self::Element;

    /// Fused multiply-add
    fn mul_add(self, m: Self, a: Self) -> Self;

    /// Fused multiply-subtract
    #[inline(always)]
    fn mul_sub(self, m: Self, s: Self) -> Self {
        self.mul_add(m, -s)
    }

    /// Fused negated multiply-add
    #[inline(always)]
    fn nmul_add(self, m: Self, a: Self) -> Self {
        self.mul_add(-m, a)
    }

    /// Fused negated multiply-subtract
    #[inline(always)]
    fn nmul_sub(self, m: Self, s: Self) -> Self {
        self.nmul_add(m, -s)
    }

    /// Fused multiply-add, with *at worst* precision equal to `x * m + a`
    #[inline(always)]
    fn mul_adde(self, m: Self, a: Self) -> Self {
        self * m + a
    }

    /// Fused multiply-subtract, with *at worst* precision equal to `x * m - s`
    #[inline(always)]
    fn mul_sube(self, m: Self, s: Self) -> Self {
        self * m - s
    }

    /// Fused negated multiply-add, with *at worst* precision equal to `a - x * m`
    #[inline(always)]
    fn nmul_adde(self, m: Self, a: Self) -> Self {
        a - self * m
    }

    /// Fused negated multiply-subtract, with *at worst* precision equal to `-x * m - s`
    #[inline(always)]
    fn nmul_sube(self, m: Self, s: Self) -> Self {
        self.nmul_adde(m, -s)
    }

    /// Rounds to the nearest representable integer.
    fn round(self) -> Self;

    /// Rounds upwards towards positive infinity.
    fn ceil(self) -> Self;

    /// Rounds downward towards negative infinity.
    fn floor(self) -> Self;

    /// Truncates any rational value towards zero
    fn trunc(self) -> Self;

    /// Returns the fractional part of a number (the part between 0 and ±1)
    #[inline(always)]
    fn fract(self) -> Self {
        self - self.trunc()
    }

    /// Calculates the square-root of each element in the vector.
    ///
    /// **NOTE**: This operation can be quite slow, so if you only need an approximation of `sqrt(x)` consider
    /// using `rsqrt` or `invsqrt`, which compute `1/sqrt(x)`. `sqrt(x) = x/sqrt(x) ≈ x * x.rsqrt()`
    fn sqrt(self) -> Self;

    /// Compute the approximate reciprocal of the square root `1/sqrt(x)`
    #[inline(always)]
    fn rsqrt(self) -> Self {
        self.sqrt().rcp()
    }

    /// Computes the approximate reciprocal/inverse of each value
    #[inline(always)]
    fn rcp(self) -> Self {
        Self::one() / self
    }

    #[inline(always)]
    fn approx_eq(self, other: Self, tolerance: Self) -> Mask<S, Self> {
        (self - other).abs().lt(tolerance)
    }

    #[inline(always)]
    fn clamp(self, min: Self, max: Self) -> Self {
        self.min(max).max(min)
    }

    /// Clamps self to between 0 and 1
    #[inline(always)]
    fn saturate(self) -> Self {
        self.clamp(Self::zero(), Self::one())
    }

    #[inline(always)]
    fn is_finite(self) -> Mask<S, Self> {
        self.abs().lt(Self::infinity())
    }

    #[inline(always)]
    fn is_infinite(self) -> Mask<S, Self> {
        self.abs().eq(Self::infinity())
    }

    #[inline(always)]
    fn is_normal(self) -> Mask<S, Self> {
        self.is_finite() & self.is_subnormal().and_not(self.ne(Self::zero()))
    }

    fn is_subnormal(self) -> Mask<S, Self>;

    #[inline(always)]
    fn is_zero_or_subnormal(self) -> Mask<S, Self> {
        self.is_subnormal() | self.eq(Self::zero())
    }

    #[inline(always)]
    fn is_nan(self) -> Mask<S, Self> {
        self.ne(self)
    }
}

/// Guarantees the vector can be used as a pointer in `VPtr`
pub trait SimdPointer<S: Simd + ?Sized>:
    SimdIntVector<S>
    + SimdPtrInternal<S, S::Vi32>
    + SimdPtrInternal<S, S::Vu32>
    + SimdPtrInternal<S, S::Vf32>
    + SimdPtrInternal<S, S::Vu64>
    + SimdPtrInternal<S, S::Vf64>
where
    <Self as SimdVectorBase<S>>::Element: pointer::AsUsize,
{
}

impl<S: Simd + ?Sized, T> SimdPointer<S> for T
where
    T: SimdIntVector<S>
        + SimdPtrInternal<S, S::Vi32>
        + SimdPtrInternal<S, S::Vu32>
        + SimdPtrInternal<S, S::Vf32>
        + SimdPtrInternal<S, S::Vu64>
        + SimdPtrInternal<S, S::Vf64>,
    <Self as SimdVectorBase<S>>::Element: pointer::AsUsize,
{
}

/// Enum of supported instruction sets
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[repr(u8)]
pub enum SimdInstructionSet {
    Scalar,

    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    SSE2,
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    SSE42,
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    AVX,
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    AVX2,

    #[cfg(all(feature = "neon", any(target_arch = "arm", target_arch = "aarch64")))]
    NEON,

    #[cfg(all(feature = "wasm32", target_arch = "wasm32"))]
    WASM32,
}

impl SimdInstructionSet {
    pub fn runtime_detect() -> SimdInstructionSet {
        unsafe {
            static mut CACHED: Option<SimdInstructionSet> = None;

            match CACHED {
                Some(value) => value,
                None => {
                    // Allow this to race, they all converge to the same result
                    let isa = Self::runtime_detect_internal();
                    CACHED = Some(isa);
                    isa
                }
            }
        }
    }

    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    fn runtime_detect_internal() -> SimdInstructionSet {
        if core_detect::is_x86_feature_detected!("fma") {
            // TODO: AVX512
            if core_detect::is_x86_feature_detected!("avx2") {
                return SimdInstructionSet::AVX2;
            }
        }

        if core_detect::is_x86_feature_detected!("avx") {
            SimdInstructionSet::AVX
        } else if core_detect::is_x86_feature_detected!("sse4.2") {
            SimdInstructionSet::SSE42
        } else if core_detect::is_x86_feature_detected!("sse2") {
            SimdInstructionSet::SSE2
        } else {
            SimdInstructionSet::Scalar
        }
    }

    #[cfg(all(feature = "neon", any(target_arch = "arm", target_arch = "aarch64")))]
    fn runtime_detect_internal() -> SimdInstructionSet {
        SimdInstructionSet::NEON
    }

    #[cfg(all(feature = "wasm32", target_arch = "wasm32"))]
    fn runtime_detect_internal() -> SimdInstructionSet {
        SimdInstructionSet::WASM32
    }

    /// True fused multiply-add instructions are only used on AVX2 and above, so this checks for that ergonomically.
    pub const fn has_true_fma(self) -> bool {
        match self {
            //SimdInstructionSet::AVX512F | SimdInstructionSet::AVX512FBW |
            SimdInstructionSet::AVX2 => true,
            _ => false,
        }
    }

    /// On older platforms, fused multiply-add instructions can be emulated (expensively),
    /// but only if the `"emulate_fma"` Cargo feature is enabled.
    pub const fn has_emulated_fma(self) -> bool {
        !self.has_true_fma() && cfg!(feature = "emulate_fma")
    }
}

/// SIMD Instruction set, contains all types
///
/// Take your time to look through this. All trait bounds contain methods and associated values which
/// encapsulate all functionality for this crate.
pub trait Simd: 'static + Debug + Send + Sync + Clone + Copy + PartialEq + Eq {
    const INSTRSET: SimdInstructionSet;

    //type Vi8: SimdIntVector<Self, Element = i8> + SimdSignedVector<Self, i8> + SimdMasked<Self, u8, Mask = Self::Vm8>;
    //type Vi16: SimdIntVector<Self, Element = i16> + SimdSignedVector<Self, i16> + SimdMasked<Self, u16, Mask = Self::Vm16>;

    /// 32-bit signed integer vector
    ///
    /// The From/Into bits traits allow it to be cast to `Vu32` at zero-cost. Some methods, such as those in
    /// [`SimdUnsignedIntVector`], are only available in unsigned vectors.
    type Vi32: SimdIntVector<Self, Element = i32>
        + SimdSignedVector<Self>
        + SimdIntegerDivision<i32>
        + SimdIntoBits<Self, Self::Vu32>
        + SimdFromBits<Self, Self::Vu32>;

    /// 64-bit signed integer vector
    ///
    /// The From/Into bits traits allow it to be cast to `Vu64` at zero-cost. Some methods, such as those in
    /// [`SimdUnsignedIntVector`], are only available in unsigned vectors.
    type Vi64: SimdIntVector<Self, Element = i64>
        + SimdSignedVector<Self>
        + SimdIntegerDivision<i64>
        + SimdIntoBits<Self, Self::Vu64>
        + SimdFromBits<Self, Self::Vu64>;

    //type Vu8: SimdIntVector<Self, Element = u8> + SimdMasked<Self, u8, Mask = Self::Vm8>;
    //type Vu16: SimdIntVector<Self, Element = u16> + SimdMasked<Self, u16, Mask = Self::Vm16>;

    /// 32-bit unsigned integer vector
    type Vu32: SimdIntVector<Self, Element = u32> + SimdUnsignedIntVector<Self> + SimdIntegerDivision<u32>;
    /// 64-bit unsigned integer vector
    type Vu64: SimdIntVector<Self, Element = u64> + SimdUnsignedIntVector<Self> + SimdIntegerDivision<u64>;

    /// Single-precision 32-bit floating point vector
    ///
    /// Note that these already implement bitwise operations between each other, but it is possible to
    /// cast to `Vu32` at zero-cost using the From/Into bits traits.
    type Vf32: SimdFloatVector<Self, Element = f32, Vu = Self::Vu32, Vi = Self::Vi32>
        + SimdIntoBits<Self, Self::Vu32>
        + SimdFromBits<Self, Self::Vu32>
        + SimdFloatVectorConsts<Self>
        + SimdVectorizedMath<Self>
        + SimdVectorizedMathPolicied<Self>;

    /// Double-precision 64-bit floating point vector
    ///
    /// Note that these already implement bitwise operations between each other, but it is possible to
    /// cast to `Vu64` at zero-cost using the From/Into bits traits.
    type Vf64: SimdFloatVector<Self, Element = f64, Vu = Self::Vu64, Vi = Self::Vi64>
        + SimdIntoBits<Self, Self::Vu64>
        + SimdFromBits<Self, Self::Vu64>
        + SimdFloatVectorConsts<Self>
        + SimdVectorizedMath<Self>
        + SimdVectorizedMathPolicied<Self>;

    #[cfg(target_pointer_width = "32")]
    type Vusize: SimdIntVector<Self, Element = u32> + SimdPointer<Self, Element = u32>;

    #[cfg(target_pointer_width = "32")]
    type Visize: SimdIntVector<Self, Element = i32> + SimdSignedVector<Self>;

    #[cfg(target_pointer_width = "64")]
    type Vusize: SimdIntVector<Self, Element = u64> + SimdPointer<Self, Element = u64>;

    #[cfg(target_pointer_width = "64")]
    type Visize: SimdIntVector<Self, Element = i64> + SimdSignedVector<Self>;
}

pub type Vi32<S> = <S as Simd>::Vi32;
pub type Vi64<S> = <S as Simd>::Vi64;
pub type Vu32<S> = <S as Simd>::Vu32;
pub type Vu64<S> = <S as Simd>::Vu64;
pub type Vf32<S> = <S as Simd>::Vf32;
pub type Vf64<S> = <S as Simd>::Vf64;

pub type Vusize<S> = <S as Simd>::Vusize;
pub type Visize<S> = <S as Simd>::Visize;

pub trait SimdAssociatedVector<S: Simd>: SimdElement {
    type V: SimdVectorBase<S>;
}

/// Associated vector type for a scalar type
pub type AssociatedVector<S, T> = <T as SimdAssociatedVector<S>>::V;

macro_rules! impl_associated {
    ($($ty:ident),*) => {paste::paste!{$(
        impl<S: Simd> SimdAssociatedVector<S> for $ty {
            type V = <S as Simd>::[<V $ty>];
        }
    )*}};
}

impl_associated!(i32, i64, u32, u64, f32, f64);

// Re-exported for procedural macro
#[doc(hidden)]
pub use core::hint::unreachable_unchecked;
