#![allow(unsafe_op_in_unsafe_fn, clippy::missing_safety_doc, clippy::let_and_return)]

use super::arch::*;

pub use crate::backend::x86_v1::polyfills::*;

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

/// This uses _mm_shuffle_epi8 to permute the register as that's the only
/// instruction for arbitrary register-controlled shuffles available to SSE4.1
/// effectively recreating _mm_permutevar_ps/_mm_permutevar_epi32
///
/// An alternative implementation is here: <https://stackoverflow.com/a/56033645/2083075>
/// but that uses _mm_mullo_epi32 which has a 10 cycle latency on some CPUs,
/// whereas these use only _mm_shuffle_epi8 and simple arithmetic/logical ops,
/// which are much faster.
#[inline(always)]
pub unsafe fn _mm_permutevarx_epi32x_v2(value: __m128i, indices: __m128i) -> __m128i {
    // ensure that all indices are valid by simply AND-ing them with 0b11
    let indices = _mm_and_si128(indices, _mm_set1_epi32(0x3));

    let shuffled_bases = _mm_shuffle_epi8(
        _mm_slli_epi32(indices, 2), // index * 4
        _mm_set_epi32(0x0C0C0C0C, 0x08080808, 0x04040404, 0x00000000),
    );

    // add 0,1,2,3 to each base index to get byte indices
    _mm_shuffle_epi8(value, _mm_add_epi8(shuffled_bases, _mm_set1_epi32(0x03020100)))
}

#[inline(always)]
pub unsafe fn _mm_permutevarx_epi64x_v2(value: __m128i, indices: __m128i) -> __m128i {
    // ensure that all indices are valid by simply AND-ing them with 0b1
    let indices = _mm_and_si128(indices, _mm_set1_epi64x(0x1));

    let shuffled_bases = _mm_shuffle_epi8(
        _mm_slli_epi64(indices, 3), // index * 8
        _mm_set_epi64x(0x0808080808080808, 0x0000000000000000),
    );

    // add 0..7 to each base index to get byte indices
    _mm_shuffle_epi8(value, _mm_add_epi8(shuffled_bases, _mm_set1_epi64x(0x0706050403020100)))
}

#[inline(always)]
pub unsafe fn _mm_permutevar_ps_v2(value: __m128, indices: __m128i) -> __m128 {
    _mm_castsi128_ps(_mm_permutevarx_epi32x_v2(_mm_castps_si128(value), indices))
}

#[inline(always)]
pub unsafe fn _mm_permutevar_pd_v2(value: __m128d, indices: __m128i) -> __m128d {
    _mm_castsi128_pd(_mm_permutevarx_epi64x_v2(_mm_castpd_si128(value), indices))
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
