#![allow(unsafe_op_in_unsafe_fn, clippy::missing_safety_doc, clippy::let_and_return)]

use generic_array::GenericArray;

use super::arch::*;

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
pub unsafe fn _mm_mullo_epi64x_v1(xmm0: __m128i, xmm1: __m128i) -> __m128i {
    let xmm2 = _mm_srli_epi64(xmm1, 32);
    let xmm3 = _mm_srli_epi64(xmm0, 32);

    let xmm2 = _mm_mul_epu32(xmm2, xmm0);
    let xmm3 = _mm_mul_epu32(xmm1, xmm3);

    let xmm2 = _mm_add_epi64(xmm3, xmm2);
    let xmm2 = _mm_slli_epi64(xmm2, 32);

    let xmm0 = _mm_mul_epu32(xmm1, xmm0);
    let xmm0 = _mm_add_epi64(xmm0, xmm2);

    xmm0
}

// SSE2 Version
#[inline(always)]
pub unsafe fn _mm_adds_epi32x_v1(lhs: __m128i, rhs: __m128i) -> __m128i {
    let res = _mm_add_epi32(lhs, rhs);

    _mm_blendv_epi8x_v1(
        res,
        _mm_blendv_epi8x_v1(
            _mm_set1_epi32(i32::MIN),
            _mm_set1_epi32(i32::MAX),
            _mm_signbits_epi32x_v1(res),
        ),
        _mm_xor_si128(rhs, _mm_cmpgt_epi32(lhs, res)),
    )
}

// SSE2 Version
#[inline(always)]
pub unsafe fn _mm_subs_epi32x_v1(lhs: __m128i, rhs: __m128i) -> __m128i {
    let res = _mm_sub_epi32(lhs, rhs);

    _mm_blendv_epi8x_v1(
        res,
        _mm_blendv_epi8x_v1(
            _mm_set1_epi32(i32::MIN),
            _mm_set1_epi32(i32::MAX),
            _mm_signbits_epi32x_v1(res),
        ),
        _mm_xor_si128(_mm_cmpgt_epi32(rhs, _mm_setzero_si128()), _mm_cmpgt_epi32(lhs, res)),
    )
}

#[inline(always)]
pub unsafe fn _mm_set1_epu32x(v: u32) -> __m128i {
    _mm_set1_epi32(v as i32)
}

#[inline(always)]
pub unsafe fn _mm_set1_epu64x(v: u64) -> __m128i {
    _mm_set1_epi64x(v as i64)
}
