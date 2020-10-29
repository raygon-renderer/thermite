#![allow(unused)]

use crate::*;

use std::{
    fmt,
    marker::PhantomData,
    mem::{transmute, transmute_copy},
    ops::*,
};

#[cfg(target_arch = "x86")]
use std::arch::x86::*;

#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct AVX2;

impl Simd for AVX2 {
    type Vi32 = i32x8<AVX2>;
    type Vf32 = f32x8<AVX2>;
}

decl!(i32x8: i32 => __m256i);
impl<S: Simd> Default for i32x8<S> {
    #[inline(always)]
    fn default() -> Self {
        Self::new(unsafe { _mm256_setzero_si256() })
    }
}

decl!(f32x8: f32 => __m256);
impl<S: Simd> Default for f32x8<S> {
    #[inline(always)]
    fn default() -> Self {
        Self::new(unsafe { _mm256_setzero_ps() })
    }
}

impl SimdVectorBase<AVX2> for i32x8<AVX2> {
    type Element = i32;

    #[inline(always)]
    fn splat(value: Self::Element) -> Self {
        Self::new(unsafe { _mm256_set1_epi32(value) })
    }

    #[inline(always)]
    unsafe fn load_aligned_unchecked(ptr: *const Self::Element) -> Self {
        Self::new(_mm256_load_si256(ptr as *const _))
    }

    #[inline(always)]
    unsafe fn load_unaligned_unchecked(ptr: *const Self::Element) -> Self {
        Self::new(_mm256_loadu_si256(ptr as *const _))
    }

    #[inline(always)]
    unsafe fn store_aligned_unchecked(self, ptr: *mut Self::Element) {
        _mm256_store_si256(ptr as *mut _, self.value)
    }

    #[inline(always)]
    unsafe fn store_unaligned_unchecked(self, ptr: *mut Self::Element) {
        _mm256_storeu_si256(ptr as *mut _, self.value)
    }

    #[inline]
    #[target_feature(enable = "avx2")]
    unsafe fn extract_unchecked(self, index: usize) -> Self::Element {
        *transmute::<&_, *const Self::Element>(&self).add(index)
    }

    #[inline]
    #[target_feature(enable = "avx2")]
    unsafe fn replace_unchecked(mut self, index: usize, value: Self::Element) -> Self {
        *transmute::<&mut _, *mut Self::Element>(&mut self).add(index) = value;
        self
    }
}

impl SimdVectorBase<AVX2> for f32x8<AVX2> {
    type Element = f32;

    #[inline(always)]
    fn splat(value: Self::Element) -> Self {
        Self::new(unsafe { _mm256_set1_ps(value) })
    }

    #[inline(always)]
    unsafe fn load_aligned_unchecked(ptr: *const Self::Element) -> Self {
        Self::new(_mm256_load_ps(ptr))
    }

    #[inline(always)]
    unsafe fn load_unaligned_unchecked(ptr: *const Self::Element) -> Self {
        Self::new(_mm256_loadu_ps(ptr))
    }

    #[inline(always)]
    unsafe fn store_aligned_unchecked(self, ptr: *mut Self::Element) {
        _mm256_store_ps(ptr, self.value)
    }

    #[inline(always)]
    unsafe fn store_unaligned_unchecked(self, ptr: *mut Self::Element) {
        _mm256_storeu_ps(ptr, self.value)
    }

    #[inline]
    #[target_feature(enable = "avx2")]
    unsafe fn extract_unchecked(self, index: usize) -> Self::Element {
        *transmute::<&_, *const Self::Element>(&self).add(index)
    }

    #[inline]
    #[target_feature(enable = "avx2")]
    unsafe fn replace_unchecked(mut self, index: usize, value: Self::Element) -> Self {
        *transmute::<&mut _, *mut Self::Element>(&mut self).add(index) = value;
        self
    }
}

impl SimdBitwise<AVX2> for i32x8<AVX2> {
    #[inline(always)]
    fn and_not(self, other: Self) -> Self {
        Self::new(unsafe { _mm256_andnot_si256(self.value, other.value) })
    }

    const FULL_BITMASK: u16 = 0b11111111;

    #[inline(always)]
    fn bitmask(self) -> u16 {
        unsafe { _mm256_movemask_pd(transmute(self)) as u16 }
    }

    #[inline(always)]
    unsafe fn _mm_not(self) -> Self {
        self ^ Self::splat(!0)
    }

    #[inline(always)]
    unsafe fn _mm_bitand(self, rhs: Self) -> Self {
        Self::new(_mm256_and_si256(self.value, rhs.value))
    }

    #[inline(always)]
    unsafe fn _mm_bitor(self, rhs: Self) -> Self {
        Self::new(_mm256_or_si256(self.value, rhs.value))
    }

    #[inline(always)]
    unsafe fn _mm_bitxor(self, rhs: Self) -> Self {
        Self::new(_mm256_xor_si256(self.value, rhs.value))
    }
}

impl SimdBitwise<AVX2> for f32x8<AVX2> {
    fn and_not(self, other: Self) -> Self {
        Self::new(unsafe { _mm256_andnot_ps(self.value, other.value) })
    }

    const FULL_BITMASK: u16 = 0b1111;

    #[inline(always)]
    fn bitmask(self) -> u16 {
        unsafe { _mm256_movemask_ps(transmute(self)) as u16 }
    }

    #[inline(always)]
    unsafe fn _mm_not(self) -> Self {
        self ^ Self::splat(f32::from_bits(!0))
    }

    #[inline(always)]
    unsafe fn _mm_bitand(self, rhs: Self) -> Self {
        Self::new(_mm256_and_ps(self.value, rhs.value))
    }

    #[inline(always)]
    unsafe fn _mm_bitor(self, rhs: Self) -> Self {
        Self::new(_mm256_or_ps(self.value, rhs.value))
    }

    #[inline(always)]
    unsafe fn _mm_bitxor(self, rhs: Self) -> Self {
        Self::new(_mm256_xor_ps(self.value, rhs.value))
    }
}

impl PartialEq<Self> for i32x8<AVX2> {
    fn eq(&self, other: &Self) -> bool {
        <Self as SimdVector<AVX2>>::eq(*self, *other).all()
    }

    fn ne(&self, other: &Self) -> bool {
        <Self as SimdVector<AVX2>>::ne(*self, *other).any()
    }
}

impl PartialEq<Self> for f32x8<AVX2> {
    fn eq(&self, other: &Self) -> bool {
        <Self as SimdVector<AVX2>>::eq(*self, *other).all()
    }

    fn ne(&self, other: &Self) -> bool {
        <Self as SimdVector<AVX2>>::ne(*self, *other).any()
    }
}

impl Eq for i32x8<AVX2> {}

impl SimdMask<AVX2> for i32x8<AVX2> {
    #[inline(always)]
    unsafe fn _mm_blendv(self, t: Self, f: Self) -> Self {
        Self::new(_mm256_blendv_epi8(t.value, f.value, self.value))
    }
}

impl SimdMask<AVX2> for f32x8<AVX2> {
    #[inline(always)]
    unsafe fn _mm_blendv(self, t: Self, f: Self) -> Self {
        Self::new(_mm256_blendv_ps(t.value, f.value, self.value))
    }
}

impl SimdVector<AVX2> for i32x8<AVX2> {
    #[inline(always)]
    fn zero() -> Self {
        Self::new(unsafe { _mm256_setzero_si256() })
    }

    #[inline(always)]
    fn one() -> Self {
        Self::splat(1)
    }

    #[inline(always)]
    fn min_value() -> Self {
        Self::splat(i32::MIN)
    }

    #[inline(always)]
    fn max_value() -> Self {
        Self::splat(i32::MAX)
    }

    #[inline]
    fn min_element(self) -> Self::Element {
        unsafe { self.reduce2(|a, x| a.min(x)) }
    }

    #[inline]
    fn max_element(self) -> Self::Element {
        unsafe { self.reduce2(|a, x| a.max(x)) }
    }

    #[inline(always)]
    fn eq(self, other: Self) -> Mask<AVX2, Self> {
        Mask::new(Self::new(unsafe { _mm256_cmpeq_epi32(self.value, other.value) }))
    }

    #[inline(always)]
    fn gt(self, other: Self) -> Mask<AVX2, Self> {
        Mask::new(Self::new(unsafe { _mm256_cmpgt_epi32(self.value, other.value) }))
    }

    #[inline(always)]
    fn ge(self, other: Self) -> Mask<AVX2, Self> {
        self.gt(other) ^ self.eq(other)
    }

    #[inline(always)]
    unsafe fn _mm_add(self, rhs: Self) -> Self {
        Self::new(_mm256_add_epi32(self.value, rhs.value))
    }

    #[inline(always)]
    unsafe fn _mm_sub(self, rhs: Self) -> Self {
        Self::new(_mm256_sub_epi32(self.value, rhs.value))
    }

    #[inline(always)]
    unsafe fn _mm_mul(self, rhs: Self) -> Self {
        Self::new(_mm256_mullo_epi32(self.value, rhs.value))
    }

    #[inline(always)]
    unsafe fn _mm_div(self, rhs: Self) -> Self {
        Self::zip(self, rhs, Div::div)
    }

    #[inline(always)]
    unsafe fn _mm_rem(self, rhs: Self) -> Self {
        Self::zip(self, rhs, Rem::rem)
    }
}

impl SimdVector<AVX2> for f32x8<AVX2> {
    #[inline(always)]
    fn zero() -> Self {
        Self::new(unsafe { _mm256_setzero_ps() })
    }

    #[inline(always)]
    fn one() -> Self {
        Self::splat(1.0)
    }

    #[inline(always)]
    fn min_value() -> Self {
        Self::splat(f32::MIN)
    }

    #[inline(always)]
    fn max_value() -> Self {
        Self::splat(f32::MAX)
    }

    #[inline(always)]
    fn min(self, other: Self) -> Self {
        Self::new(unsafe { _mm256_min_ps(self.value, other.value) })
    }

    #[inline(always)]
    fn max(self, other: Self) -> Self {
        Self::new(unsafe { _mm256_max_ps(self.value, other.value) })
    }

    #[inline]
    fn min_element(self) -> Self::Element {
        // TODO: Replace with log-reduce
        unsafe { self.reduce2(|a, x| a.min(x)) }
    }

    #[inline]
    fn max_element(self) -> Self::Element {
        // TODO: Replace with log-reduce
        unsafe { self.reduce2(|a, x| a.max(x)) }
    }

    #[inline(always)]
    fn eq(self, other: Self) -> Mask<AVX2, Self> {
        Mask::new(Self::new(unsafe { _mm256_cmp_ps(self.value, other.value, _CMP_EQ_OQ) }))
    }

    #[inline(always)]
    fn ne(self, other: Self) -> Mask<AVX2, Self> {
        Mask::new(Self::new(unsafe {
            _mm256_cmp_ps(self.value, other.value, _CMP_NEQ_OQ)
        }))
    }

    #[inline(always)]
    fn lt(self, other: Self) -> Mask<AVX2, Self> {
        Mask::new(Self::new(unsafe { _mm256_cmp_ps(self.value, other.value, _CMP_LT_OQ) }))
    }

    #[inline(always)]
    fn le(self, other: Self) -> Mask<AVX2, Self> {
        Mask::new(Self::new(unsafe { _mm256_cmp_ps(self.value, other.value, _CMP_LE_OQ) }))
    }

    #[inline(always)]
    fn gt(self, other: Self) -> Mask<AVX2, Self> {
        Mask::new(Self::new(unsafe { _mm256_cmp_ps(self.value, other.value, _CMP_GT_OQ) }))
    }

    fn ge(self, other: Self) -> Mask<AVX2, Self> {
        Mask::new(Self::new(unsafe { _mm256_cmp_ps(self.value, other.value, _CMP_GE_OQ) }))
    }

    #[inline(always)]
    unsafe fn _mm_add(self, rhs: Self) -> Self {
        Self::new(_mm256_add_ps(self.value, rhs.value))
    }

    #[inline(always)]
    unsafe fn _mm_sub(self, rhs: Self) -> Self {
        Self::new(_mm256_sub_ps(self.value, rhs.value))
    }

    #[inline(always)]
    unsafe fn _mm_mul(self, rhs: Self) -> Self {
        Self::new(_mm256_mul_ps(self.value, rhs.value))
    }

    #[inline(always)]
    unsafe fn _mm_div(self, rhs: Self) -> Self {
        Self::new(_mm256_div_ps(self.value, rhs.value))
    }

    #[inline(always)]
    unsafe fn _mm_rem(self, rhs: Self) -> Self {
        let d = self / rhs;
        let trunc = Self::new(_mm256_round_ps(d.value, _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC));
        self - (trunc * rhs)
    }
}

impl SimdIntVector<AVX2> for i32x8<AVX2> {}

impl SimdSignedVector<AVX2> for i32x8<AVX2> {
    #[inline(always)]
    fn neg_one() -> Self {
        Self::splat(-1)
    }

    #[inline(always)]
    fn min_positive() -> Self {
        Self::splat(0)
    }

    #[inline(always)]
    fn abs(self) -> Self {
        Self::new(unsafe { _mm256_abs_epi32(self.value) })
    }

    #[inline(always)]
    unsafe fn _mm_neg(self) -> Self {
        Self::new(_mm256_sign_epi32(self.value, _mm256_set1_epi32(-1)))
    }
}

impl SimdSignedVector<AVX2> for f32x8<AVX2> {
    #[inline(always)]
    fn neg_one() -> Self {
        Self::splat(-1.0)
    }

    #[inline(always)]
    fn min_positive() -> Self {
        Self::splat(f32::MIN_POSITIVE)
    }

    #[inline(always)]
    fn signum(self) -> Self {
        Self::one() | (self & Self::neg_zero())
    }

    #[inline(always)]
    fn abs(self) -> Self {
        // clear sign bit
        self & Self::splat(f32::from_bits(0x7fffffff))
    }

    #[inline(always)]
    unsafe fn _mm_neg(self) -> Self {
        // Xor sign bit using -0.0 as a shorthand for the sign bit
        self ^ Self::neg_zero()
    }
}

impl SimdFloatVector<AVX2> for f32x8<AVX2> {
    #[inline(always)]
    fn epsilon() -> Self {
        Self::splat(f32::EPSILON)
    }
    #[inline(always)]
    fn infinity() -> Self {
        Self::splat(f32::INFINITY)
    }
    #[inline(always)]
    fn neg_infinity() -> Self {
        Self::splat(f32::NEG_INFINITY)
    }
    #[inline(always)]
    fn neg_zero() -> Self {
        Self::splat(-0.0)
    }
    #[inline(always)]
    fn nan() -> Self {
        Self::splat(f32::NAN)
    }

    fn sum(self) -> Self::Element {
        // TODO: Replace with log-reduce
        unsafe { self.reduce2(|sum, x| sum + x) }
    }

    fn product(self) -> Self::Element {
        // TODO: Replace with log-reduce
        unsafe { self.reduce2(|prod, x| x * prod) }
    }

    const HAS_TRUE_FMA: bool = true;

    #[inline(always)]
    fn mul_add(self, m: Self, a: Self) -> Self {
        Self::new(unsafe { _mm256_fmadd_ps(self.value, m.value, a.value) })
    }

    #[inline(always)]
    fn mul_sub(self, m: Self, s: Self) -> Self {
        Self::new(unsafe { _mm256_fmsub_ps(self.value, m.value, s.value) })
    }

    #[inline(always)]
    fn neg_mul_add(self, m: Self, a: Self) -> Self {
        Self::new(unsafe { _mm256_fnmadd_ps(self.value, m.value, a.value) })
    }

    #[inline(always)]
    fn neg_mul_sub(self, m: Self, s: Self) -> Self {
        Self::new(unsafe { _mm256_fnmsub_ps(self.value, m.value, s.value) })
    }

    #[inline(always)]
    fn floor(self) -> Self {
        Self::new(unsafe { _mm256_floor_ps(self.value) })
    }

    #[inline(always)]
    fn ceil(self) -> Self {
        Self::new(unsafe { _mm256_ceil_ps(self.value) })
    }

    #[inline(always)]
    fn round(self) -> Self {
        Self::new(unsafe { _mm256_round_ps(self.value, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC) })
    }

    #[inline(always)]
    fn sqrt(self) -> Self {
        Self::new(unsafe { _mm256_sqrt_ps(self.value) })
    }

    #[inline(always)]
    fn rsqrt(self) -> Self {
        Self::new(unsafe { _mm256_rsqrt_ps(self.value) })
    }

    #[inline(always)]
    fn rsqrt_precise(self) -> Self {
        let y = self.rsqrt();
        let nx2 = self * Self::splat(-0.5);
        let threehalfs = Self::splat(1.5);

        y * (y * y).mul_add(nx2, threehalfs)
    }

    #[inline(always)]
    fn recepr(self) -> Self {
        Self::new(unsafe { _mm256_rcp_ps(self.value) })
    }
}

impl_ops!(@UNARY i32x8 AVX2 => Not::not, Neg::neg);
impl_ops!(@BINARY i32x8 AVX2 => Add::add, Sub::sub, Mul::mul, Div::div, Rem::rem, BitAnd::bitand, BitOr::bitor, BitXor::bitxor);

impl_ops!(@UNARY f32x8 AVX2 => Not::not, Neg::neg);
impl_ops!(@BINARY f32x8 AVX2 => Add::add, Sub::sub, Mul::mul, Div::div, Rem::rem, BitAnd::bitand, BitOr::bitor, BitXor::bitxor);

impl SimdCastFrom<AVX2, i32x8<AVX2>> for f32x8<AVX2> {
    #[inline(always)]
    fn from_cast(from: i32x8<AVX2>) -> Self {
        Self::new(unsafe { _mm256_cvtepi32_ps(from.value) })
    }

    #[inline(always)]
    fn from_cast_mask(from: Mask<AVX2, i32x8<AVX2>>) -> Mask<AVX2, Self> {
        Self::from_cast(from.value()).ne(Self::zero())
    }
}

impl SimdCastFrom<AVX2, f32x8<AVX2>> for i32x8<AVX2> {
    #[inline(always)]
    fn from_cast(from: f32x8<AVX2>) -> Self {
        Self::new(unsafe { _mm256_cvttps_epi32(from.value) })
    }

    #[inline(always)]
    fn from_cast_mask(from: Mask<AVX2, f32x8<AVX2>>) -> Mask<AVX2, Self> {
        Self::from_cast(from.value()).ne(Self::zero())
    }
}
