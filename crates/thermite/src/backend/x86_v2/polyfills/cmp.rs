use super::*;

#[inline(always)]
pub unsafe fn _mm_cmpgt_epu16x_v2(a: __m128i, b: __m128i) -> __m128i {
    let mask = _mm_set1_epi16(i16::MIN); // 0x8000
    _mm_cmpgt_epi16(_mm_xor_si128(a, mask), _mm_xor_si128(b, mask))
}

#[inline(always)]
pub unsafe fn _mm_cmpgt_epu32x_v2(a: __m128i, b: __m128i) -> __m128i {
    let mask = _mm_set1_epu32x(1u32 << 31);
    _mm_cmpgt_epi32(_mm_xor_si128(a, mask), _mm_xor_si128(b, mask))
}

#[inline(always)]
pub unsafe fn _mm_cmpgt_epu64x_v2(a: __m128i, b: __m128i) -> __m128i {
    let mask = _mm_set1_epu64x(1u64 << 63);
    _mm_cmpgt_epi64(_mm_xor_si128(a, mask), _mm_xor_si128(b, mask))
}

#[inline(always)]
pub unsafe fn _mm_cmpge_epu32x_v2(lhs: __m128i, rhs: __m128i) -> __m128i {
    _mm_cmpeq_epi32(_mm_max_epu32(lhs, rhs), lhs)
}

#[inline(always)]
pub unsafe fn _mm_cmple_epu32x_v2(lhs: __m128i, rhs: __m128i) -> __m128i {
    _mm_cmpge_epu32x_v2(rhs, lhs)
}

// #[inline(always)]
// pub unsafe fn _mm_cmpgt_epu32x(lhs: __m128i, rhs: __m128i) -> __m128i {
//     _mm_xor_si128(_mm_cmple_epu32x(lhs, rhs), _mm_set1_epi32(-1))
// }

// TODO: Unify all these methods?
// https://outerproduct.net/trivial/2022-08-25_unsigned.html
// https://stackoverflow.com/questions/32945410/sse2-intrinsics-comparing-unsigned-integers
#[inline(always)]
pub unsafe fn _mm_cmplt_epu32x_v2(lhs: __m128i, rhs: __m128i) -> __m128i {
    _mm_cmpgt_epu32x_v2(rhs, lhs)
}

#[inline(always)]
pub unsafe fn _mm_min_epi64x_v2(a: __m128i, b: __m128i) -> __m128i {
    _mm_blendv_epi8(a, b, _mm_cmpgt_epi64(a, b))
}

#[inline(always)]
pub unsafe fn _mm_max_epi64x_v2(a: __m128i, b: __m128i) -> __m128i {
    _mm_blendv_epi8(b, a, _mm_cmpgt_epi64(a, b))
}

#[inline(always)]
pub unsafe fn _mm_max_epu64x_v2(a: __m128i, b: __m128i) -> __m128i {
    _mm_blendv_epi8(b, a, _mm_cmpgt_epu64x_v2(a, b))
}

#[inline(always)]
pub unsafe fn _mm_min_epu64x_v2(a: __m128i, b: __m128i) -> __m128i {
    _mm_blendv_epi8(a, b, _mm_cmpgt_epu64x_v2(a, b))
}
