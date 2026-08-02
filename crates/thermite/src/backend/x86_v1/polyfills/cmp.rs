use super::*;

/// POLYFILL: per-byte unsigned greater-than via the `0x80` bias trick (no `cmpgt_epu8`).
#[inline(always)]
pub unsafe fn _mm_cmpgt_epu8x_v1(a: __m128i, b: __m128i) -> __m128i {
    let bias = _mm_set1_epi8(i8::MIN); // 0x80
    _mm_cmpgt_epi8(_mm_xor_si128(a, bias), _mm_xor_si128(b, bias))
}

/// POLYFILL: per-byte signed min via cmpgt + bitwise select (no `min_epi8` pre-SSE4.1).
#[inline(always)]
pub unsafe fn _mm_min_epi8x_v1(a: __m128i, b: __m128i) -> __m128i {
    _mm_blendv_epi8x_v1(a, b, _mm_cmpgt_epi8(a, b)) // mask ? b : a => a > b ? b : a
}

/// POLYFILL: per-byte signed max via cmpgt + bitwise select (no `max_epi8` pre-SSE4.1).
#[inline(always)]
pub unsafe fn _mm_max_epi8x_v1(a: __m128i, b: __m128i) -> __m128i {
    _mm_blendv_epi8x_v1(b, a, _mm_cmpgt_epi8(a, b)) // mask ? a : b => a > b ? a : b
}

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

    // The low-dword results live in the even lanes (0, 2); move them up to the
    // odd lanes (1, 3) where the high-dword results are, so they can be combined.
    let cmp_lo = _mm_shuffle_epi32::<0b10_10_00_00>(cmp_lo);

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

/// POLYFILL: unsigned 16-bit `>` (no native unsigned compare on SSE2). Bias by 0x8000 so a
/// signed `pcmpgtw` ranks the unsigned values.
#[inline(always)]
pub unsafe fn _mm_cmpgt_epu16x_v1(a: __m128i, b: __m128i) -> __m128i {
    let bias = _mm_set1_epi16(i16::MIN); // 0x8000
    _mm_cmpgt_epi16(_mm_xor_si128(a, bias), _mm_xor_si128(b, bias))
}

/// POLYFILL: `_mm_min_epu16` (SSE4.1). `subs_epu16(a,b) = max(a-b, 0)`, so `a - that = min(a,b)`.
#[inline(always)]
pub unsafe fn _mm_min_epu16x_v1(a: __m128i, b: __m128i) -> __m128i {
    _mm_sub_epi16(a, _mm_subs_epu16(a, b))
}

/// POLYFILL: `_mm_max_epu16` (SSE4.1). `b + max(a-b, 0) = max(a,b)`.
#[inline(always)]
pub unsafe fn _mm_max_epu16x_v1(a: __m128i, b: __m128i) -> __m128i {
    _mm_add_epi16(b, _mm_subs_epu16(a, b))
}

#[inline(always)]
pub unsafe fn _mm_cmpgt_epu64x_v1(a: __m128i, b: __m128i) -> __m128i {
    // Flip only the 64-bit sign bit: the signed comparison's low half is
    // already unsigned, so touching bit 31 would corrupt it.
    let mask = _mm_set1_epu64x(1 << 63);
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

/// POLYFILL: `_mm_min_epi32` (SSE4.1)
#[inline(always)]
pub unsafe fn _mm_min_epi32x_v1(a: __m128i, b: __m128i) -> __m128i {
    _mm_blendv_epi8x_v1(a, b, _mm_cmpgt_epi32(a, b))
}

/// POLYFILL: `_mm_max_epi32` (SSE4.1)
#[inline(always)]
pub unsafe fn _mm_max_epi32x_v1(a: __m128i, b: __m128i) -> __m128i {
    _mm_blendv_epi8x_v1(b, a, _mm_cmpgt_epi32(a, b))
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

// ---------------------------------------------------------------------------
// Bitmask -> lane mask (the inverse of `movemask`).
//
// AVX-512 spells this `vpmovm2{b,w,d,q}` (`_mm_movm_epi32` and friends), hence
// the names. Everything below is the pre-AVX512 lowering: broadcast the packed
// bits, AND with a per-lane bit-select constant, then compare against that same
// constant - three instructions plus a broadcast, all SSE2.
//
// Bits at or above the lane count fall outside every lane's bit-select constant
// and so are ignored, as `MaskRegister::from_native_bitmask` requires.
// ---------------------------------------------------------------------------

/// POLYFILL: `_mm_movm_epi32` (AVX-512 VL+DQ `vpmovm2d`) - expand bits 0..=3 of
/// `bitmask` into four full-width `i32` lane masks.
#[inline(always)]
pub unsafe fn _mm_movm_epi32x_v1(bitmask: u64) -> __m128i {
    let bits = _mm_setr_epi32(1, 2, 4, 8);
    let broadcast = _mm_set1_epi32(bitmask as i32);

    _mm_cmpeq_epi32(_mm_and_si128(broadcast, bits), bits)
}

/// POLYFILL: `_mm_movm_epi64` (AVX-512 VL+DQ `vpmovm2q`) - expand bits 0..=1 of
/// `bitmask` into two full-width `i64` lane masks.
///
/// `pcmpeqq` is SSE4.1, so the compare is done on the low dword of each lane and
/// then broadcast over the lane with a shuffle.
#[inline(always)]
pub unsafe fn _mm_movm_epi64x_v1(bitmask: u64) -> __m128i {
    let bits = _mm_setr_epi32(1, 0, 2, 0);
    let broadcast = _mm_set1_epi32(bitmask as i32);

    let eq = _mm_cmpeq_epi32(_mm_and_si128(broadcast, bits), bits);

    // dwords (0, 0, 2, 2): the low dword of each qword carries the verdict.
    _mm_shuffle_epi32::<0b10_10_00_00>(eq)
}

/// POLYFILL: `_mm_movm_epi16` (AVX-512 VL+BW `vpmovm2w`) - expand bits 0..=7 of
/// `bitmask` into eight full-width `i16` lane masks.
#[inline(always)]
pub unsafe fn _mm_movm_epi16x_v1(bitmask: u64) -> __m128i {
    let bits = _mm_setr_epi16(1, 2, 4, 8, 16, 32, 64, 128);
    let broadcast = _mm_set1_epi16(bitmask as i16);

    _mm_cmpeq_epi16(_mm_and_si128(broadcast, bits), bits)
}

/// POLYFILL: `_mm_movm_epi8` (AVX-512 VL+BW `vpmovm2b`) - expand bits 0..=15 of
/// `bitmask` into sixteen full-width `i8` lane masks.
///
/// A byte lane is narrower than the bit index it tests, so the two relevant
/// bytes of `bitmask` are spread over their eight lanes each. Without `pshufb`
/// (SSSE3) that is done in the general-purpose registers, by multiplying each
/// byte with `0x0101..` and building the vector from the two halves.
#[inline(always)]
pub unsafe fn _mm_movm_epi8x_v1(bitmask: u64) -> __m128i {
    const SPREAD: u64 = 0x0101_0101_0101_0101;

    let lo = (bitmask & 0xFF) * SPREAD;
    let hi = ((bitmask >> 8) & 0xFF) * SPREAD;

    let bits = _mm_setr_epi8(
        1, 2, 4, 8, 16, 32, 64, -128, //
        1, 2, 4, 8, 16, 32, 64, -128,
    );
    let broadcast = _mm_set_epi64x(hi as i64, lo as i64);

    _mm_cmpeq_epi8(_mm_and_si128(broadcast, bits), bits)
}
