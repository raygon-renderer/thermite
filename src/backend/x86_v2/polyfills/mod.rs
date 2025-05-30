#![allow(unsafe_op_in_unsafe_fn, clippy::missing_safety_doc, clippy::let_and_return)]

use core::arch::x86_64::_mm_srlv_epi32;

use super::arch::*;

pub use crate::backend::x86_v1::polyfills::*;

pub mod bits;
pub mod casts;
pub mod cmp;
pub mod math;

pub use bits::*;
pub use casts::*;
pub use cmp::*;
pub use math::*;

#[inline(always)]
pub const unsafe fn identity<T>(x: T) -> T {
    x
}

/// This uses _mm_shuffle_epi8 to permute the register as that's the only
/// instruction for arbitrary register-controlled shuffles available to SSE4.1
/// effectively recreating _mm_permutevar_ps/_mm_permutevar_epi32
#[inline(always)]
pub unsafe fn _mm_permutevarx_epi32x_v2(value: __m128i, indices: __m128i) -> __m128i {
    // ensure that all indices are valid by simply AND-ing them with 0b11
    let indices = _mm_and_si128(indices, _mm_set1_epi32(0x3));

    let offsets = _mm_set_epi8(3, 2, 1, 0, 3, 2, 1, 0, 3, 2, 1, 0, 3, 2, 1, 0);
    let selector = _mm_set_epi8(12, 12, 12, 12, 8, 8, 8, 8, 4, 4, 4, 4, 0, 0, 0, 0);

    let base_indices = _mm_slli_epi32(indices, 2); // index * 4
    let shuffled_bases = _mm_shuffle_epi8(base_indices, selector);

    let byte_mask = _mm_add_epi8(shuffled_bases, offsets);

    _mm_shuffle_epi8(value, byte_mask)
}

#[inline(always)]
pub unsafe fn _mm_blendv_epi32x_v2(xmm0: __m128i, xmm1: __m128i, mask: __m128i) -> __m128i {
    _mm_castps_si128(_mm_blendv_ps(
        _mm_castsi128_ps(xmm0),
        _mm_castsi128_ps(xmm1),
        _mm_castsi128_ps(mask),
    ))
}

#[inline(always)]
pub unsafe fn _mm_blendv_epi64x_v2(xmm0: __m128i, xmm1: __m128i, mask: __m128i) -> __m128i {
    _mm_castpd_si128(_mm_blendv_pd(
        _mm_castsi128_pd(xmm0),
        _mm_castsi128_pd(xmm1),
        _mm_castsi128_pd(mask),
    ))
}
