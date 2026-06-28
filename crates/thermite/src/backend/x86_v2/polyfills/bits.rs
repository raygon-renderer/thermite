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
pub unsafe fn _mm_popcnt_epi16x_v2(v: __m128i) -> __m128i {
    // Per-byte popcount, then sum adjacent byte pairs into 16-bit lanes via maddubs.
    _mm_maddubs_epi16(_mm_popcnt_epi8x_v2(v), _mm_set1_epi8(1))
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

// ===========================================================================
// PSHUFB-based 2D Morton (Z-order) encode.
//
// `pshufb` acts as a 16-entry LUT indexed by each byte's low nibble. The LUT
// below maps a 4-bit nibble `b3 b2 b1 b0` to its bit-spread-by-1 byte
// `0 b3 0 b2 0 b1 0 b0` (bit `i` -> bit `2i`), so one `pshufb` performs the
// within-byte 2D spread of every nibble at once. The only per-lane work is
// isolating each of a coordinate's nibbles into its own byte first - a short
// shift/mask cascade (cheaper than the full per-bit spread). This beats the
// generic cascade for the common 2D widths. The same LUT is reused at v3 (256).
// ===========================================================================

/// Spread the low 8 bits of each 16-bit lane by one (2D Morton), via the nibble LUT.
#[inline(always)]
pub unsafe fn _mm_morton2_spread_epu16x_v2(v: __m128i) -> __m128i {
    let lut = _mm_setr_epi8(
        0x00, 0x01, 0x04, 0x05, 0x10, 0x11, 0x14, 0x15, 0x40, 0x41, 0x44, 0x45, 0x50, 0x51, 0x54, 0x55,
    );
    // isolate the two nibbles of the low byte into separate bytes, then LUT-spread
    let c = _mm_and_si128(v, _mm_set1_epi16(0x00FF));
    let n = _mm_and_si128(_mm_or_si128(c, _mm_slli_epi16(c, 4)), _mm_set1_epi16(0x0F0F));
    _mm_shuffle_epi8(lut, n)
}

/// Spread the low 16 bits of each 32-bit lane by one (2D Morton), via the nibble LUT.
#[inline(always)]
pub unsafe fn _mm_morton2_spread_epu32x_v2(v: __m128i) -> __m128i {
    let lut = _mm_setr_epi8(
        0x00, 0x01, 0x04, 0x05, 0x10, 0x11, 0x14, 0x15, 0x40, 0x41, 0x44, 0x45, 0x50, 0x51, 0x54, 0x55,
    );
    // isolate the four nibbles of the low 16 bits into separate bytes, then LUT-spread
    let c = _mm_and_si128(v, _mm_set1_epi32(0x0000_FFFF));
    let c = _mm_and_si128(_mm_or_si128(c, _mm_slli_epi32(c, 8)), _mm_set1_epi32(0x00FF_00FF));
    let n = _mm_and_si128(_mm_or_si128(c, _mm_slli_epi32(c, 4)), _mm_set1_epi32(0x0F0F_0F0F));
    _mm_shuffle_epi8(lut, n)
}

/// Per-16-bit-lane 2D Morton encode (two 8-bit coords -> one 16-bit code per
/// lane): interleave the low 8 bits of `x` (even output bits) with the low 8
/// bits of `y` (odd output bits).
#[inline(always)]
pub unsafe fn _mm_morton2_epu16x_v2(x: __m128i, y: __m128i) -> __m128i {
    _mm_or_si128(
        _mm_morton2_spread_epu16x_v2(x),
        _mm_slli_epi16(_mm_morton2_spread_epu16x_v2(y), 1),
    )
}

/// Per-32-bit-lane 2D Morton encode (two 16-bit coords -> one 32-bit code per
/// lane): interleave the low 16 bits of `x` (even output bits) with the low 16
/// bits of `y` (odd output bits).
#[inline(always)]
pub unsafe fn _mm_morton2_epu32x_v2(x: __m128i, y: __m128i) -> __m128i {
    _mm_or_si128(
        _mm_morton2_spread_epu32x_v2(x),
        _mm_slli_epi32(_mm_morton2_spread_epu32x_v2(y), 1),
    )
}

// ---------------------------------------------------------------------------
// PSHUFB-based 2D Morton DECODE (the inverse spread / "compress").
//
// `reverse_morton::<2>` gathers one axis's bits (the even bit positions of the
// code, after the caller shifts the odd axis down by 1) back to a contiguous low
// half. Unlike encode, a single `pshufb` recovers only two of a byte's four even
// bits - its low nibble holds bits 0 and 2 - so the compress uses *two* LUT
// lookups per byte (low nibble + high nibble) with `MORTON2_COMPRESS_LUT`, then
// packs the resulting per-byte nibbles. (CLMUL has no decode analogue, so the
// u64 path stays on the cascade; only the pshufb widths get a decode fast path.)
// ---------------------------------------------------------------------------

// Byte -> 2-bit compress LUT: gather bits 0 and 2 of each byte's low nibble
// (`i -> bit0(i) | bit2(i) << 1`). The masked code's even bits land at `{0,2}`
// within a nibble, so this recovers a coordinate's bit pair per LUT lookup. The
// same 16 bytes are reused (replicated) by the v3 256-bit decode helpers.

/// Compress the even bits of each 16-bit lane (positions 0,2,...) back into a
/// contiguous low 8 bits - the inverse of [`_mm_morton2_spread_epu16x_v2`].
#[inline(always)]
pub unsafe fn _mm_morton2_compress_epu16x_v2(v: __m128i) -> __m128i {
    let lut = _mm_setr_epi8(0, 1, 0, 1, 2, 3, 2, 3, 0, 1, 0, 1, 2, 3, 2, 3);
    let e = _mm_and_si128(v, _mm_set1_epi16(0x5555)); // this axis's bits, at even positions
    // per byte: gather its four even bits into a nibble (low- then high-nibble LUT)
    let lo = _mm_shuffle_epi8(lut, e);
    let hi = _mm_shuffle_epi8(lut, _mm_srli_epi16(e, 4));
    let n = _mm_and_si128(_mm_or_si128(lo, _mm_slli_epi16(hi, 2)), _mm_set1_epi16(0x0F0F));
    // pack the two per-byte nibbles back into the low byte
    _mm_and_si128(_mm_or_si128(n, _mm_srli_epi16(n, 4)), _mm_set1_epi16(0x00FF))
}

/// Compress the even bits of each 32-bit lane (positions 0,2,...) back into a
/// contiguous low 16 bits - the inverse of [`_mm_morton2_spread_epu32x_v2`].
#[inline(always)]
pub unsafe fn _mm_morton2_compress_epu32x_v2(v: __m128i) -> __m128i {
    let lut = _mm_setr_epi8(0, 1, 0, 1, 2, 3, 2, 3, 0, 1, 0, 1, 2, 3, 2, 3);
    let e = _mm_and_si128(v, _mm_set1_epi32(0x5555_5555));
    let lo = _mm_shuffle_epi8(lut, e);
    let hi = _mm_shuffle_epi8(lut, _mm_srli_epi32(e, 4));
    let n = _mm_and_si128(_mm_or_si128(lo, _mm_slli_epi32(hi, 2)), _mm_set1_epi32(0x0F0F_0F0F));
    // pack the four per-byte nibbles back into the low 16 bits
    let c = _mm_and_si128(_mm_or_si128(n, _mm_srli_epi32(n, 4)), _mm_set1_epi32(0x00FF_00FF));
    _mm_and_si128(_mm_or_si128(c, _mm_srli_epi32(c, 8)), _mm_set1_epi32(0x0000_FFFF))
}
