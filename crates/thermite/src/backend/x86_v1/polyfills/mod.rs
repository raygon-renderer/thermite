#![allow(unsafe_op_in_unsafe_fn, clippy::missing_safety_doc, clippy::let_and_return)]

use generic_array::GenericArray;

use super::arch::*;

pub mod bits;
pub mod casts;
pub mod cmp;
pub mod divider;
pub mod math;

pub use bits::*;
pub use casts::*;
pub use cmp::*;
pub use divider::*;
pub use math::*;

#[inline(always)]
pub unsafe fn _mm_blendv_epi8x_v1(xmm0: __m128i, xmm1: __m128i, mask: __m128i) -> __m128i {
    _mm_or_si128(_mm_and_si128(mask, xmm0), _mm_andnot_si128(mask, xmm1))
}

#[inline(always)]
pub unsafe fn _mm_signbits_epi32x_v1(v: __m128i) -> __m128i {
    _mm_srai_epi32(v, 31)
}

#[inline(always)]
pub unsafe fn _mm_signbits_epi64x_v1(v: __m128i) -> __m128i {
    _mm_srai_epi32(_mm_shuffle_epi32::<{ MM_SHUFFLE!(3, 3, 1, 1) }>(v), 31)
}

#[inline(always)]
pub unsafe fn _mm_cmpeq_epi64x_v1(a: __m128i, b: __m128i) -> __m128i {
    let t = _mm_cmpeq_epi32(a, b);
    _mm_and_si128(t, _mm_shuffle_epi32(t, 0b10_11_00_01))
}

#[inline(always)]
pub unsafe fn _mm_setr_epi64x(a: i64, b: i64) -> __m128i {
    _mm_set_epi64x(b, a)
}

#[inline(always)]
pub unsafe fn _mm_set_epu32x(a: u32, b: u32, c: u32, d: u32) -> __m128i {
    _mm_set_epi32(a as i32, b as i32, c as i32, d as i32)
}

#[inline(always)]
pub unsafe fn _mm_set_epu64x(a: u64, b: u64) -> __m128i {
    _mm_set_epi64x(a as i64, b as i64)
}

#[inline(always)]
pub unsafe fn _mm_setr_epu32x(a: u32, b: u32, c: u32, d: u32) -> __m128i {
    _mm_setr_epi32(a as i32, b as i32, c as i32, d as i32)
}

#[inline(always)]
pub unsafe fn _mm_setr_epu64x(a: u64, b: u64) -> __m128i {
    _mm_setr_epi64x(a as i64, b as i64)
}

#[inline(always)]
pub unsafe fn _mm_set1_epu32x(v: u32) -> __m128i {
    _mm_set1_epi32(v as i32)
}

#[inline(always)]
pub unsafe fn _mm_set1_epu64x(v: u64) -> __m128i {
    _mm_set1_epi64x(v as i64)
}
