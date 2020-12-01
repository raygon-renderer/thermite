use super::*;

pub unsafe fn _mm_blendv_epi8(xmm0: __m128i, xmm1: __m128i, mask: __m128i) -> __m128i {
    _mm_or_si128(_mm_and_si128(mask, xmm0), _mm_andnot_si128(mask, xmm1))
}
