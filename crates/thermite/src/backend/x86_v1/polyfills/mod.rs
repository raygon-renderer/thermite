#![allow(unsafe_op_in_unsafe_fn, clippy::missing_safety_doc, clippy::let_and_return)]

pub use crate::backend::generic::polyfills::*;

use super::arch::*;

pub mod bits;
pub mod casts;
pub mod cmp;
pub mod divider;
pub mod interleave;
pub mod math;

pub use bits::*;
pub use casts::*;
pub use cmp::*;
pub use divider::*;
pub use interleave::*;
pub use math::*;

#[inline(always)]
pub const unsafe fn identity<T>(x: T) -> T {
    x
}

/// POLYFILL: Variable blend matching `_mm_blendv_epi8` argument order:
/// `mask ? b : a`.
///
/// Unlike the SSE4.1 instruction (which selects per byte on each byte's high
/// bit), this is a full bitwise select, so the mask must be lane-uniform
/// (all-ones or all-zeros per lane), as produced by the `_mm_cmp*` family
/// or the `_mm_signbits_*` helpers. Raw values are not valid masks.
#[inline(always)]
pub unsafe fn _mm_blendv_epi8x_v1(a: __m128i, b: __m128i, mask: __m128i) -> __m128i {
    _mm_or_si128(_mm_and_si128(mask, b), _mm_andnot_si128(mask, a))
}

/// POLYFILL: see [`_mm_blendv_epi8x_v1`]; same lane-uniform mask requirement.
#[inline(always)]
pub unsafe fn _mm_blendv_epi32x_v1(a: __m128i, b: __m128i, mask: __m128i) -> __m128i {
    _mm_blendv_epi8x_v1(a, b, mask)
}

/// POLYFILL: see [`_mm_blendv_epi8x_v1`]; same lane-uniform mask requirement.
#[inline(always)]
pub unsafe fn _mm_blendv_epi64x_v1(a: __m128i, b: __m128i, mask: __m128i) -> __m128i {
    _mm_blendv_epi8x_v1(a, b, mask)
}

/// POLYFILL: `_mm_blendv_ps` (`mask ? b : a`) via bitwise select.
/// The mask must be lane-uniform (e.g. a comparison result), not just sign bits.
#[inline(always)]
pub unsafe fn _mm_blendv_psx_v1(a: __m128, b: __m128, mask: __m128) -> __m128 {
    _mm_or_ps(_mm_and_ps(mask, b), _mm_andnot_ps(mask, a))
}

/// POLYFILL: `_mm_blendv_pd` (`mask ? b : a`) via bitwise select.
/// The mask must be lane-uniform (e.g. a comparison result), not just sign bits.
#[inline(always)]
pub unsafe fn _mm_blendv_pdx_v1(a: __m128d, b: __m128d, mask: __m128d) -> __m128d {
    _mm_or_pd(_mm_and_pd(mask, b), _mm_andnot_pd(mask, a))
}

/// POLYFILL: `_mm_blend_ps` (constant per-lane blend, `IMM8` bit `i` selects
/// lane `i` from `b`) via bitwise select with a compile-time mask.
#[inline(always)]
pub unsafe fn _mm_blend_psx_v1<const IMM8: i32>(a: __m128, b: __m128) -> __m128 {
    let mask = const {
        [
            f32::from_bits(if IMM8 & 0b0001 != 0 { !0 } else { 0 }),
            f32::from_bits(if IMM8 & 0b0010 != 0 { !0 } else { 0 }),
            f32::from_bits(if IMM8 & 0b0100 != 0 { !0 } else { 0 }),
            f32::from_bits(if IMM8 & 0b1000 != 0 { !0 } else { 0 }),
        ]
    };
    let mask = _mm_loadu_ps(mask.as_ptr());
    _mm_blendv_psx_v1(a, b, mask)
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
