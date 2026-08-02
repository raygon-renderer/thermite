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

// ---------------------------------------------------------------------------
// Bitmask -> lane mask. See the v1 versions for the shape; these two shorten it
// with instructions SSE2 lacks (`pcmpeqq`, `pshufb`). 32- and 16-bit lanes are
// already optimal at v1 and are inherited as-is.
// ---------------------------------------------------------------------------

/// POLYFILL: `_mm_movm_epi64` (AVX-512 VL+DQ `vpmovm2q`) - expand bits 0..=1 of
/// `bitmask` into two full-width `i64` lane masks. `pcmpeqq` compares whole
/// qwords, so this drops v1's broadcast shuffle.
#[inline(always)]
pub unsafe fn _mm_movm_epi64x_v2(bitmask: u64) -> __m128i {
    let bits = _mm_setr_epi64x(1, 2);
    let broadcast = _mm_set1_epi64x(bitmask as i64);

    _mm_cmpeq_epi64(_mm_and_si128(broadcast, bits), bits)
}

/// POLYFILL: `_mm_movm_epi8` (AVX-512 VL+BW `vpmovm2b`) - expand bits 0..=15 of
/// `bitmask` into sixteen full-width `i8` lane masks. One `pshufb` spreads the
/// low byte of `bitmask` over lanes 0..=7 and the second byte over lanes 8..=15,
/// replacing v1's scalar `0x0101..` multiplies.
#[inline(always)]
pub unsafe fn _mm_movm_epi8x_v2(bitmask: u64) -> __m128i {
    let bits = _mm_setr_epi8(
        1, 2, 4, 8, 16, 32, 64, -128, //
        1, 2, 4, 8, 16, 32, 64, -128,
    );
    let spread = _mm_setr_epi8(
        0, 0, 0, 0, 0, 0, 0, 0, //
        1, 1, 1, 1, 1, 1, 1, 1,
    );

    let broadcast = _mm_shuffle_epi8(_mm_cvtsi32_si128(bitmask as i32), spread);

    _mm_cmpeq_epi8(_mm_and_si128(broadcast, bits), bits)
}
