use super::*;

#[inline(always)]
pub unsafe fn _mm256_cmpgt_epu8x_v3(a: __m256i, b: __m256i) -> __m256i {
    let mask = _mm256_set1_epi8(i8::MIN); // 0x80
    _mm256_cmpgt_epi8(_mm256_xor_si256(a, mask), _mm256_xor_si256(b, mask))
}

#[inline(always)]
pub unsafe fn _mm256_cmpgt_epu16x_v3(a: __m256i, b: __m256i) -> __m256i {
    let mask = _mm256_set1_epi16(i16::MIN); // 0x8000
    _mm256_cmpgt_epi16(_mm256_xor_si256(a, mask), _mm256_xor_si256(b, mask))
}

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

// ---------------------------------------------------------------------------
// Bitmask -> lane mask, 256-bit. The 128-bit forms are inherited from v1/v2; see
// those for the shape. AVX-512 spells these `vpmovm2{b,w,d,q}`.
// ---------------------------------------------------------------------------

/// POLYFILL: `_mm256_movm_epi32` (AVX-512 VL+DQ `vpmovm2d`) - expand bits 0..=7
/// of `bitmask` into eight full-width `i32` lane masks.
#[inline(always)]
pub unsafe fn _mm256_movm_epi32x_v3(bitmask: u64) -> __m256i {
    let bits = _mm256_setr_epi32(1, 2, 4, 8, 16, 32, 64, 128);
    let broadcast = _mm256_set1_epi32(bitmask as i32);

    _mm256_cmpeq_epi32(_mm256_and_si256(broadcast, bits), bits)
}

/// POLYFILL: `_mm256_movm_epi64` (AVX-512 VL+DQ `vpmovm2q`) - expand bits 0..=3
/// of `bitmask` into four full-width `i64` lane masks.
#[inline(always)]
pub unsafe fn _mm256_movm_epi64x_v3(bitmask: u64) -> __m256i {
    let bits = _mm256_setr_epi64x(1, 2, 4, 8);
    let broadcast = _mm256_set1_epi64x(bitmask as i64);

    _mm256_cmpeq_epi64(_mm256_and_si256(broadcast, bits), bits)
}

/// POLYFILL: `_mm256_movm_epi16` (AVX-512 VL+BW `vpmovm2w`) - expand bits 0..=15
/// of `bitmask` into sixteen full-width `i16` lane masks. Sixteen lanes still fit
/// in a 16-bit lane, so a plain broadcast suffices.
#[inline(always)]
pub unsafe fn _mm256_movm_epi16x_v3(bitmask: u64) -> __m256i {
    let bits = _mm256_setr_epi16(
        1, 2, 4, 8, 16, 32, 64, 128, //
        256, 512, 1024, 2048, 4096, 8192, 16384, -32768, /* 0x8000 */
    );
    let broadcast = _mm256_set1_epi16(bitmask as i16);

    _mm256_cmpeq_epi16(_mm256_and_si256(broadcast, bits), bits)
}

/// POLYFILL: `_mm256_movm_epi8` (AVX-512 VL+BW `vpmovm2b`) - expand bits 0..=31
/// of `bitmask` into thirty-two full-width `i8` lane masks.
///
/// `vpshufb` indexes within each 128-bit half, so broadcasting the four relevant
/// bytes as a dword hands both halves the same four candidate bytes and one
/// shuffle picks bytes 0/1 for the low half and 2/3 for the high half.
#[inline(always)]
pub unsafe fn _mm256_movm_epi8x_v3(bitmask: u64) -> __m256i {
    let bits = _mm256_setr_epi8(
        1, 2, 4, 8, 16, 32, 64, -128, //
        1, 2, 4, 8, 16, 32, 64, -128, //
        1, 2, 4, 8, 16, 32, 64, -128, //
        1, 2, 4, 8, 16, 32, 64, -128,
    );
    let spread = _mm256_setr_epi8(
        0, 0, 0, 0, 0, 0, 0, 0, //
        1, 1, 1, 1, 1, 1, 1, 1, //
        2, 2, 2, 2, 2, 2, 2, 2, //
        3, 3, 3, 3, 3, 3, 3, 3,
    );

    let broadcast = _mm256_shuffle_epi8(_mm256_set1_epi32(bitmask as i32), spread);

    _mm256_cmpeq_epi8(_mm256_and_si256(broadcast, bits), bits)
}
