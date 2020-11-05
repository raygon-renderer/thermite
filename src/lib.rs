#![cfg_attr(feature = "nightly", feature(stdsimd, core_intrinsics))]
#![allow(unused_imports, non_camel_case_types, non_snake_case)]

#[macro_use]
mod macros;

use half::f16;

#[cfg(feature = "alloc")]
mod buffer;
#[cfg(feature = "alloc")]
pub use buffer::SimdBuffer;

pub mod backends;

//mod double;

mod pointer;
use self::pointer::*;
pub use self::pointer::{AssociatedVector, VPtr};

mod mask;
pub use mask::Mask;

mod math;
pub use math::SimdVectorizedMath;

use std::{fmt::Debug, marker::PhantomData, mem, ops::*, ptr};

/// Describes casting from one SIMD vector type to another
///
/// This should handle extending bits correctly
pub trait SimdCastFrom<S: Simd, FROM>: Sized {
    fn from_cast(from: FROM) -> Self;
    fn from_cast_mask(from: Mask<S, FROM>) -> Mask<S, Self>;
}

impl<S: Simd, T> SimdCastFrom<S, T> for T {
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
    fn cast(self) -> TO;
    fn cast_mask(mask: Mask<S, Self>) -> Mask<S, TO>;
}

impl<S: Simd, FROM, TO> SimdCastTo<S, TO> for FROM
where
    TO: SimdCastFrom<S, FROM>,
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
    + SimdCastFrom<S, S::Vi32>
    + SimdCastFrom<S, S::Vu32>
    + SimdCastFrom<S, S::Vu64>
    + SimdCastFrom<S, S::Vf32>
    + SimdCastFrom<S, S::Vf64>
{
    #[inline(always)]
    fn cast_to<T: SimdCastFrom<S, Self>>(self) -> T {
        self.cast()
    }
}

impl<S: Simd + ?Sized, T> SimdCasts<S> for T where
    T: Sized
        + SimdCastFrom<S, S::Vi32>
        + SimdCastFrom<S, S::Vu32>
        + SimdCastFrom<S, S::Vu64>
        + SimdCastFrom<S, S::Vf32>
        + SimdCastFrom<S, S::Vf64>
{
}

/// Umbrella trait for SIMD vector element bounds
pub trait SimdElement: mask::Truthy + Clone + Debug + Copy + Default + Send + Sync {}

impl<T> SimdElement for T where T: mask::Truthy + Clone + Debug + Copy + Default + Send + Sync {}

/// Basic shared vector interface
pub trait SimdVectorBase<S: Simd + ?Sized>: Sized + Copy + Debug + Default + Send + Sync {
    type Element: SimdElement;

    /// Size of element type in bytes
    const ELEMENT_SIZE: usize = std::mem::size_of::<Self::Element>();
    const NUM_ELEMENTS: usize = std::mem::size_of::<S::Vi32>() / 4;
    const ALIGNMENT: usize = std::mem::align_of::<Self>();

    fn splat(value: Self::Element) -> Self;

    /// Possibly returns a vector containing undefined or uninitialized data
    unsafe fn undefined() -> Self {
        Self::default()
    }

    #[inline(always)]
    fn splat_any(value: impl Into<Self::Element>) -> Self {
        Self::splat(value.into())
    }

    #[inline(always)]
    #[cfg(feature = "alloc")]
    fn alloc(count: usize) -> SimdBuffer<S, Self> {
        SimdBuffer::alloc(count)
    }

    #[inline]
    fn extract(self, index: usize) -> Self::Element {
        assert!(index < Self::NUM_ELEMENTS);
        unsafe { self.extract_unchecked(index) }
    }

    #[inline]
    fn replace(self, index: usize, value: Self::Element) -> Self {
        assert!(index < Self::NUM_ELEMENTS);
        unsafe { self.replace_unchecked(index, value) }
    }

    unsafe fn extract_unchecked(self, index: usize) -> Self::Element;
    unsafe fn replace_unchecked(self, index: usize, value: Self::Element) -> Self;

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

    #[inline]
    fn load_unaligned(src: &[Self::Element]) -> Self {
        assert!(src.len() >= Self::NUM_ELEMENTS);
        unsafe { Self::load_unaligned_unchecked(src.as_ptr()) }
    }

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

    #[inline]
    fn store_unaligned(self, dst: &mut [Self::Element]) {
        assert!(dst.len() >= Self::NUM_ELEMENTS);
        unsafe { self.store_unaligned_unchecked(dst.as_mut_ptr()) };
    }

    #[inline(always)]
    unsafe fn load_aligned_unchecked(src: *const Self::Element) -> Self {
        Self::load_unaligned_unchecked(src)
    }

    #[inline(always)]
    unsafe fn store_aligned_unchecked(self, dst: *mut Self::Element) {
        self.store_unaligned_unchecked(dst);
    }

    #[inline(always)]
    unsafe fn load_unaligned_unchecked(src: *const Self::Element) -> Self {
        let mut target = mem::MaybeUninit::uninit();
        ptr::copy_nonoverlapping(src as *const Self, target.as_mut_ptr(), 1);
        target.assume_init()
    }

    #[inline(always)]
    unsafe fn store_unaligned_unchecked(self, dst: *mut Self::Element) {
        ptr::copy_nonoverlapping(&self as *const Self, dst as *mut Self, 1);
    }

    #[inline(always)]
    unsafe fn gather(base_ptr: *const Self::Element, indices: S::Vi32) -> Self {
        Self::gather_masked(base_ptr, indices, Mask::truthy(), Self::default())
    }

    #[inline(always)]
    unsafe fn scatter(self, base_ptr: *mut Self::Element, indices: S::Vi32) {
        self.scatter_masked(base_ptr, indices, Mask::truthy())
    }

    #[inline(always)]
    unsafe fn gather_masked(
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

    #[inline(always)]
    unsafe fn scatter_masked(self, base_ptr: *mut Self::Element, indices: S::Vi32, mask: Mask<S, Self>) {
        for i in 0..Self::NUM_ELEMENTS {
            if mask.extract_unchecked(i) {
                base_ptr
                    .offset(indices.extract_unchecked(i) as isize)
                    .write(self.extract_unchecked(i));
            }
        }
    }
}

// TODO: Require op bounds for both Self and T
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

    const FULL_BITMASK: u16;

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
    unsafe fn _mm_shr(self, count: S::Vu32) -> Self;
    #[doc(hidden)]
    unsafe fn _mm_shl(self, count: S::Vu32) -> Self;

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

    #[inline(always)]
    unsafe fn _mm_blendv(self, t: Self, f: Self) -> Self {
        (self & t) | self.and_not(f)
    }
}

// TODO: Require op bounds for both Self and T
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
    fn zero() -> Self;
    fn one() -> Self;

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
    fn ne(self, other: Self) -> Mask<S, Self> {
        !self.eq(other)
    }
    fn lt(self, other: Self) -> Mask<S, Self> {
        other.gt(self)
    }
    fn le(self, other: Self) -> Mask<S, Self> {
        other.ge(self)
    }

    fn gt(self, other: Self) -> Mask<S, Self>;
    fn ge(self, other: Self) -> Mask<S, Self>;

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
        unsafe { std::mem::transmute_copy(&self) }
    }
}

/// Transmutations from raw bits
pub trait SimdFromBits<S: Simd + ?Sized, B>: SimdVectorBase<S> {
    #[inline(always)]
    fn from_bits(bits: B) -> Self {
        unsafe { std::mem::transmute_copy(&bits) }
    }
}

/// Integer SIMD vectors
pub trait SimdIntVector<S: Simd + ?Sized>: SimdVector<S> + Eq {
    /// Saturating addition, will not wrap
    fn saturating_add(self, _rhs: Self) -> Self;
    /// Saturating subtraction, will not wrap
    fn saturating_sub(self, _rhs: Self) -> Self;

    /// Sum all lanes together, wrapping the result if it can't fit in `T`
    fn wrapping_sum(self) -> Self::Element;
    /// Multiple all lanes together, wrapping the result if it can't fit in `T`
    fn wrapping_product(self) -> Self::Element;
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

    #[doc(hidden)]
    unsafe fn _mm_neg(self) -> Self;
}

/// Floating point SIMD vectors
pub trait SimdFloatVector<S: Simd + ?Sized>: SimdVector<S> + SimdSignedVector<S> {
    fn epsilon() -> Self;
    fn infinity() -> Self;
    fn neg_infinity() -> Self;
    fn neg_zero() -> Self;
    fn nan() -> Self;

    /// Load half-precision floats and up-convert them into `Self`
    fn load_half_unaligned(src: &[f16]) -> Self {
        assert!(src.len() >= Self::NUM_ELEMENTS);
        unsafe { Self::load_half_unaligned_unchecked(src.as_ptr()) }
    }

    /// Down-convert `self` into half-precision and store
    fn store_half_unaligned(&self, dst: &mut [f16]) {
        assert!(dst.len() >= Self::NUM_ELEMENTS);
        unsafe { self.store_half_unaligned_unchecked(dst.as_mut_ptr()) };
    }

    unsafe fn load_half_unaligned_unchecked(src: *const f16) -> Self;
    unsafe fn store_half_unaligned_unchecked(&self, dst: *mut f16);

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

    /// Fused negated multiple-add
    #[inline(always)]
    fn nmul_add(self, m: Self, a: Self) -> Self {
        self.mul_add(-m, a)
    }

    /// Fused negated multiple-subtract
    #[inline(always)]
    fn nmul_sub(self, m: Self, s: Self) -> Self {
        self.nmul_add(m, -s)
    }

    fn round(self) -> Self;
    fn ceil(self) -> Self;
    fn floor(self) -> Self;

    fn trunc(self) -> Self;

    #[inline]
    fn fract(self) -> Self {
        self - self.trunc()
    }

    fn sqrt(self) -> Self;

    /// Compute the approximate reciprocal of the square root `(1 / sqrt(x))`
    #[inline(always)]
    fn rsqrt(self) -> Self {
        self.sqrt().recepr()
    }

    /// A more precise `1 / sqrt(x)` variation, which may use faster instructions where possible
    #[inline(always)]
    fn rsqrt_precise(self) -> Self {
        Self::one() / self.sqrt()
    }

    /// Computes the approximate reciprocal/inverse of each value
    #[inline(always)]
    fn recepr(self) -> Self {
        Self::one() / self
    }

    #[inline]
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
        !(self.is_nan() | self.is_infinite())
    }

    #[inline(always)]
    fn is_infinite(self) -> Mask<S, Self> {
        self.eq(Self::infinity()) | self.eq(Self::neg_infinity())
    }

    #[inline(always)]
    fn is_normal(self) -> Mask<S, Self> {
        self.is_finite() & self.is_subnormal().and_not(self.ne(Self::zero()))
    }

    fn is_subnormal(self) -> Mask<S, Self>;

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

/// SIMD Instruction set
pub trait Simd: Debug + Send + Sync + Clone + Copy {
    //type Vi8: SimdIntVector<Self, Element = i8> + SimdSignedVector<Self, i8> + SimdMasked<Self, u8, Mask = Self::Vm8>;
    //type Vi16: SimdIntVector<Self, Element = i16> + SimdSignedVector<Self, i16> + SimdMasked<Self, u16, Mask = Self::Vm16>;
    type Vi32: SimdIntVector<Self, Element = i32>
        + SimdSignedVector<Self>
        + SimdIntoBits<Self, Self::Vu32>
        + SimdFromBits<Self, Self::Vu32>;
    //type Vi64: SimdIntVector<Self, Element = i64> + SimdSignedVector<Self>;

    //type Vu8: SimdIntVector<Self, Element = u8> + SimdMasked<Self, u8, Mask = Self::Vm8>;
    //type Vu16: SimdIntVector<Self, Element = u16> + SimdMasked<Self, u16, Mask = Self::Vm16>;
    type Vu32: SimdIntVector<Self, Element = u32>;
    type Vu64: SimdIntVector<Self, Element = u64>;

    type Vf32: SimdFloatVector<Self, Element = f32>
        + SimdIntoBits<Self, Self::Vu32>
        + SimdFromBits<Self, Self::Vu32>
        + SimdVectorizedMath<Self>;
    type Vf64: SimdFloatVector<Self, Element = f64>
        + SimdIntoBits<Self, Self::Vu64>
        + SimdFromBits<Self, Self::Vu64>
        + SimdVectorizedMath<Self>;

    #[cfg(target_pointer_width = "32")]
    type Vusize: SimdIntVector<Self, Element = u32> + SimdPointer<Self, Element = u32>;
    //#[cfg(target_pointer_width = "32")]
    //type Visize: SimdIntVector<Self, i32> + SimdSignedVector<Self, i32>;
    #[cfg(target_pointer_width = "64")]
    type Vusize: SimdIntVector<Self, Element = u64> + SimdPointer<Self, Element = u64>;
    //#[cfg(target_pointer_width = "64")]
    //type Visize: SimdIntVector<Self, i64> + SimdSignedVector<Self, i64>;
}

pub type Vi32<S> = <S as Simd>::Vi32;
pub type Vu32<S> = <S as Simd>::Vu32;
pub type Vu64<S> = <S as Simd>::Vu64;
pub type Vf32<S> = <S as Simd>::Vf32;
pub type Vf64<S> = <S as Simd>::Vf64;
