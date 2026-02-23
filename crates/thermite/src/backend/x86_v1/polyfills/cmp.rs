use super::*;

#[inline(always)]
pub unsafe fn _mm_max_epu32x_v1(a: __m128i, b: __m128i) -> __m128i {
    // 1. Sign bit mask
    let sign_mask = _mm_set1_epu32x(0x80000000);

    // 2. Prepare keys (Unsigned -> SignedBits)
    let a_key = _mm_xor_si128(a, sign_mask);
    let b_key = _mm_xor_si128(b, sign_mask);

    // 3. Generate Mask
    // If a_key > b_key (signed), then a > b (unsigned).
    let mask = _mm_cmpgt_epi32(a_key, b_key);

    // 4. Select
    // Returns a where mask is 1, b where mask is 0
    _mm_or_si128(_mm_and_si128(mask, a), _mm_andnot_si128(mask, b))
}

#[inline(always)]
pub unsafe fn _mm_min_epu32x_v1(a: __m128i, b: __m128i) -> __m128i {
    // 1. Sign bit mask
    let sign_mask = _mm_set1_epu32x(0x80000000);

    // 2. Prepare keys (Unsigned -> SignedBits)
    let a_key = _mm_xor_si128(a, sign_mask);
    let b_key = _mm_xor_si128(b, sign_mask);

    // 3. Generate Mask
    // If a_key < b_key (signed), then a < b (unsigned).
    let mask = _mm_cmpgt_epi32(b_key, a_key);

    // 4. Select
    // Returns a where mask is 1, b where mask is 0
    _mm_or_si128(_mm_and_si128(mask, a), _mm_andnot_si128(mask, b))
}

#[inline(always)]
pub unsafe fn _mm_cmpgt_epi64x_v1(a: __m128i, b: __m128i) -> __m128i {
    // 1. Sign bit mask (i32::MIN) to correct unsigned comparisons
    let sign_mask = _mm_set1_epu32x(0x80000000);

    // 2. Compare Low 32-bits (Unsigned)
    // Flip the sign bit to shift the range, allowing signed comparison
    // to act as unsigned comparison.
    let a_flip = _mm_xor_si128(a, sign_mask);
    let b_flip = _mm_xor_si128(b, sign_mask);
    let cmp_lo = _mm_cmpgt_epi32(a_flip, b_flip);

    // 3. Compare High 32-bits (Signed)
    let cmp_hi = _mm_cmpgt_epi32(a, b);

    // 4. Compare High 32-bits (Equality)
    let eq_hi = _mm_cmpeq_epi32(a, b);

    // 5. Combine Results
    // Result = (High >) OR (High == AND Low_Unsigned >)
    // The result is valid in the odd lanes (high bits of the 64-bit integers).
    let mask = _mm_or_si128(cmp_hi, _mm_and_si128(eq_hi, cmp_lo));

    // 6. Shuffle to fill 64-bit lanes
    // Broadcast indices 1 -> 0 and 3 -> 2.
    // _MM_SHUFFLE(3, 3, 1, 1) => 0xF5
    _mm_shuffle_epi32::<0b11_11_01_01>(mask)
}

#[inline(always)]
pub unsafe fn _mm_cmpgt_epu32x_v1(a: __m128i, b: __m128i) -> __m128i {
    let mask = _mm_set1_epu32x(0x80000000);
    _mm_cmpgt_epi32(_mm_xor_si128(a, mask), _mm_xor_si128(b, mask))
}

#[inline(always)]
pub unsafe fn _mm_cmpgt_epu64x_v1(a: __m128i, b: __m128i) -> __m128i {
    let mask = _mm_set1_epu64x(0x8000000080000000);
    _mm_cmpgt_epi64x_v1(_mm_xor_si128(a, mask), _mm_xor_si128(b, mask))
}

#[inline(always)]
pub unsafe fn _mm_cmpge_epu32x_v1(lhs: __m128i, rhs: __m128i) -> __m128i {
    _mm_cmpeq_epi32(_mm_max_epu32x_v1(lhs, rhs), lhs)
}

#[inline(always)]
pub unsafe fn _mm_cmple_epu32x_v1(lhs: __m128i, rhs: __m128i) -> __m128i {
    _mm_cmpge_epu32x_v1(rhs, lhs)
}

#[inline(always)]
pub unsafe fn _mm_cmplt_epu32x_v1(lhs: __m128i, rhs: __m128i) -> __m128i {
    _mm_cmpgt_epu32x_v1(rhs, lhs)
}

#[inline(always)]
pub unsafe fn _mm_min_epi64x_v1(a: __m128i, b: __m128i) -> __m128i {
    _mm_blendv_epi8x_v1(a, b, _mm_cmpgt_epi64x_v1(a, b))
}

#[inline(always)]
pub unsafe fn _mm_max_epi64x_v1(a: __m128i, b: __m128i) -> __m128i {
    _mm_blendv_epi8x_v1(b, a, _mm_cmpgt_epi64x_v1(a, b))
}

#[inline(always)]
pub unsafe fn _mm_max_epu64x_v1(a: __m128i, b: __m128i) -> __m128i {
    _mm_blendv_epi8x_v1(b, a, _mm_cmpgt_epu64x_v1(a, b))
}

#[inline(always)]
pub unsafe fn _mm_min_epu64x_v1(a: __m128i, b: __m128i) -> __m128i {
    _mm_blendv_epi8x_v1(a, b, _mm_cmpgt_epu64x_v1(a, b))
}
