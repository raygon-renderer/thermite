use super::*;

#[inline(always)]
pub unsafe fn _mm256_cmpgt_epu32x_v3(a: __m256i, b: __m256i) -> __m256i {
    let mask = _mm256_set1_epu32x(1u32 << 31);
    _mm256_cmpgt_epi32(_mm256_xor_si256(a, mask), _mm256_xor_si256(b, mask))
}

#[inline(always)]
pub unsafe fn _mm256_cmpgt_epu64x_v3(a: __m256i, b: __m256i) -> __m256i {
    let mask = _mm256_set1_epu64x(1u64 << 63);
    _mm256_cmpgt_epi64(_mm256_xor_si256(a, mask), _mm256_xor_si256(b, mask))
}

#[inline(always)]
pub unsafe fn _mm256_cmpge_epu32x_v3(lhs: __m256i, rhs: __m256i) -> __m256i {
    _mm256_cmpeq_epi32(_mm256_max_epu32(lhs, rhs), lhs)
}

#[inline(always)]
pub unsafe fn _mm256_cmple_epu32x_v3(lhs: __m256i, rhs: __m256i) -> __m256i {
    _mm256_cmpge_epu32x_v3(rhs, lhs)
}

// #[inline(always)]
// pub unsafe fn _mm256_cmpgt_epu32x(lhs: __m256i, rhs: __m256i) -> __m256i {
//     _mm256_xor_si256(_mm256_cmple_epu32x(lhs, rhs), _mm256_set1_epi32(-1))
// }

#[inline(always)]
pub unsafe fn _mm256_cmplt_epu32x_v3(lhs: __m256i, rhs: __m256i) -> __m256i {
    _mm256_cmpgt_epu32x_v3(rhs, lhs)
}

#[inline(always)]
pub unsafe fn _mm256_min_epi64x_v3(a: __m256i, b: __m256i) -> __m256i {
    _mm256_blendv_epi8(a, b, _mm256_cmpgt_epi64(a, b))
}

#[inline(always)]
pub unsafe fn _mm256_max_epi64x_v3(a: __m256i, b: __m256i) -> __m256i {
    _mm256_blendv_epi8(b, a, _mm256_cmpgt_epi64(a, b))
}

#[inline(always)]
pub unsafe fn _mm256_max_epu64x_v3(a: __m256i, b: __m256i) -> __m256i {
    _mm256_blendv_epi8(b, a, _mm256_cmpgt_epu64x_v3(a, b))
}

#[inline(always)]
pub unsafe fn _mm256_min_epu64x_v3(a: __m256i, b: __m256i) -> __m256i {
    _mm256_blendv_epi8(a, b, _mm256_cmpgt_epu64x_v3(a, b))
}
