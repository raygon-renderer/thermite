use super::*;

#[inline(always)]
pub unsafe fn _mm_popcnt_epi8x_v2(v: __m128i) -> __m128i {
    let lookup = _mm_setr_epi8(0, 1, 1, 2, 1, 2, 2, 3, 1, 2, 2, 3, 2, 3, 3, 4);
    let low_mask = _mm_set1_epi8(0x0f);
    let lo = _mm_and_si128(v, low_mask);
    let hi = _mm_and_si128(_mm_srli_epi16(v, 4), low_mask);
    let cnt1 = _mm_shuffle_epi8(lookup, lo);
    let cnt2 = _mm_shuffle_epi8(lookup, hi);
    _mm_add_epi8(cnt1, cnt2)
}

#[inline(always)]
pub unsafe fn _mm_popcnt_epi32x_v2(v: __m128i) -> __m128i {
    // https://stackoverflow.com/a/51106873/2083075
    _mm_madd_epi16(
        _mm_maddubs_epi16(_mm_popcnt_epi8x_v2(v), _mm_set1_epi8(1)),
        _mm_set1_epi16(1),
    )
}

#[inline(always)]
pub unsafe fn _mm_popcnt_epi64x_v2(v: __m128i) -> __m128i {
    _mm_sad_epu8(_mm_popcnt_epi8x_v2(v), _mm_setzero_si128())
}

// This kind of sucked, but I'll leave it here for later.
//
// #[inline(always)]
// pub unsafe fn _mm_log2_epu32x_v2(mut v: __m128i) -> __m128i {
//     #[inline(always)]
//     unsafe fn masked_gt(lhs: __m128i, rhs: __m128i) -> __m128i {
//         _mm_and_si128(_mm_cmpgt_epi32(lhs, rhs), _mm_set1_epi32(1))
//     }

//     let mut r;

//     let mut shift = _mm_slli_epi32::<4>(masked_gt(v, _mm_set1_epi32(0xFFFF)));
//     v = _mm_srlv_epi32(v, shift);
//     r = shift; // r = _mm_or_si128(r, shift); where r is 0 here

//     shift = _mm_slli_epi32::<3>(masked_gt(v, _mm_set1_epi32(0xFF)));
//     v = _mm_srlv_epi32(v, shift);
//     r = _mm_or_si128(r, shift);

//     shift = _mm_slli_epi32::<2>(masked_gt(v, _mm_set1_epi32(0xF)));
//     v = _mm_srlv_epi32(v, shift);
//     r = _mm_or_si128(r, shift);

//     shift = _mm_slli_epi32::<1>(masked_gt(v, _mm_set1_epi32(0x3)));
//     v = _mm_srlv_epi32(v, shift);
//     r = _mm_or_si128(r, shift);

//     r = _mm_or_si128(r, _mm_srli_epi32(v, 1));

//     r
// }

#[inline(always)]
pub unsafe fn _mm_bswap_epi16x_v2(value: __m128i) -> __m128i {
    // Mask: 1 0 | 3 2 | 5 4 | 7 6 | 9 8 | 11 10 | 13 12 | 15 14
    _mm_shuffle_epi8(
        value,
        _mm_setr_epi8(1, 0, 3, 2, 5, 4, 7, 6, 9, 8, 11, 10, 13, 12, 15, 14),
    )
}

#[inline(always)]
pub unsafe fn _mm_bswap_epi32x_v2(x: __m128i) -> __m128i {
    // Mask: 3 2 1 0 | 7 6 5 4 | 11 10 9 8 | 15 14 13 12
    _mm_shuffle_epi8(x, _mm_setr_epi8(3, 2, 1, 0, 7, 6, 5, 4, 11, 10, 9, 8, 15, 14, 13, 12))
}

#[inline(always)]
pub unsafe fn _mm_bswap_epi64x_v2(x: __m128i) -> __m128i {
    // Mask: 7 6 5 4 3 2 1 0 | 15 14 13 12 11 10 9 8
    _mm_shuffle_epi8(x, _mm_setr_epi8(7, 6, 5, 4, 3, 2, 1, 0, 15, 14, 13, 12, 11, 10, 9, 8))
}

#[inline(always)]
pub unsafe fn _mm_bswap_psx_v2(x: __m128) -> __m128 {
    _mm_castsi128_ps(_mm_bswap_epi32x_v2(_mm_castps_si128(x)))
}

#[inline(always)]
pub unsafe fn _mm_bswap_pdx_v2(x: __m128d) -> __m128d {
    _mm_castsi128_pd(_mm_bswap_epi64x_v2(_mm_castpd_si128(x)))
}
