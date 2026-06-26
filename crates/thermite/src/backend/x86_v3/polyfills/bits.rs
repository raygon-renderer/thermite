use crate::register::ZeroUpper;

use super::*;

// https://arxiv.org/pdf/1611.07612.pdf
#[inline(always)] #[rustfmt::skip]
pub unsafe fn _mm256_popcnt_epi8x_v3(v: __m256i) -> __m256i {
    let lookup = _mm256_setr_epi8(
        0, 1, 1, 2, 1, 2, 2, 3, 1, 2, 2, 3, 2, 3, 3, 4,
        0, 1, 1, 2, 1, 2, 2, 3, 1, 2, 2, 3, 2, 3, 3, 4,
    );
    let low_mask = _mm256_set1_epi8(0x0f);
    let lo = _mm256_and_si256(v, low_mask);
    let hi = _mm256_and_si256(_mm256_srli_epi32(v, 4), low_mask);
    let popcnt1 = _mm256_shuffle_epi8(lookup, lo);
    let popcnt2 = _mm256_shuffle_epi8(lookup, hi);
    _mm256_add_epi8(popcnt1, popcnt2)
}

#[inline(always)]
pub unsafe fn _mm256_popcnt_epi64x_v3(v: __m256i) -> __m256i {
    _mm256_sad_epu8(_mm256_popcnt_epi8x_v3(v), _mm256_setzero_si256())
}

#[inline(always)]
pub unsafe fn _mm256_popcnt_epi16x_v3(v: __m256i) -> __m256i {
    // Per-byte popcount, then sum adjacent byte pairs into 16-bit lanes via maddubs.
    _mm256_maddubs_epi16(_mm256_popcnt_epi8x_v3(v), _mm256_set1_epi8(1))
}

#[inline(always)]
pub unsafe fn _mm256_popcnt_epi32x_v3(v: __m256i) -> __m256i {
    // https://stackoverflow.com/a/51106873/2083075
    _mm256_madd_epi16(
        _mm256_maddubs_epi16(_mm256_popcnt_epi8x_v3(v), _mm256_set1_epi8(1)),
        _mm256_set1_epi16(1),
    )
}

#[inline(always)]
pub unsafe fn _mm256_bswap_epi16x_v3(value: __m256i) -> __m256i {
    // Mask: 1 0 | 3 2 | 5 4 | 7 6 | 9 8 | 11 10 | 13 12 | 15 14
    _mm256_shuffle_epi8(
        value,
        _mm256_setr_epi8(
            1, 0, 3, 2, 5, 4, 7, 6, 9, 8, 11, 10, 13, 12, 15, 14, // Lane 1
            1, 0, 3, 2, 5, 4, 7, 6, 9, 8, 11, 10, 13, 12, 15, 14, // Lane 2
        ),
    )
}

#[inline(always)]
pub unsafe fn _mm256_bswap_epi32x_v3(x: __m256i) -> __m256i {
    // Note: vpshufb works within 128-bit lanes.
    // We repeat the 128-bit mask for both lanes.
    let mask = _mm256_setr_epi8(
        3, 2, 1, 0, 7, 6, 5, 4, 11, 10, 9, 8, 15, 14, 13, 12, // Lane 1
        3, 2, 1, 0, 7, 6, 5, 4, 11, 10, 9, 8, 15, 14, 13, 12, // Lane 2
    );
    _mm256_shuffle_epi8(x, mask)
}

#[inline(always)]
pub unsafe fn _mm256_bswap_epi64x_v3(x: __m256i) -> __m256i {
    let mask = _mm256_setr_epi8(
        7, 6, 5, 4, 3, 2, 1, 0, 15, 14, 13, 12, 11, 10, 9, 8, // Lane 1
        7, 6, 5, 4, 3, 2, 1, 0, 15, 14, 13, 12, 11, 10, 9, 8, // Lane 2
    );
    _mm256_shuffle_epi8(x, mask)
}

#[inline(always)]
pub unsafe fn _mm256_bswap_psx_v3(x: __m256) -> __m256 {
    _mm256_castsi256_ps(_mm256_bswap_epi32x_v3(_mm256_castps_si256(x)))
}

#[inline(always)]
pub unsafe fn _mm256_bswap_pdx_v3(x: __m256d) -> __m256d {
    _mm256_castsi256_pd(_mm256_bswap_epi64x_v3(_mm256_castpd_si256(x)))
}

#[inline(always)]
pub unsafe fn _mm256_zeroupper_mask_epi32<Z: ZeroUpper>() -> __m256i {
    use crate::element::MaskElement;

    _mm256_setr_epi32(
        i32::from_bool(0 < Z::N),
        i32::from_bool(1 < Z::N),
        i32::from_bool(2 < Z::N),
        i32::from_bool(3 < Z::N),
        i32::from_bool(4 < Z::N),
        i32::from_bool(5 < Z::N),
        i32::from_bool(6 < Z::N),
        i32::from_bool(7 < Z::N),
    )
}

// ===========================================================================
// 256-bit 8-bit (byte) lane polyfills (AVX2). x86 has no native 8-bit shift or
// multiply at any width, so these emulate them with 16-bit ops + masking. Shift
// counts are assumed in `0..8`. The even/odd byte-multiply and the
// unpack/pack-based high-multiply are purely per-128-bit-lane, so no cross-lane
// fixup is needed.
// ===========================================================================

/// POLYFILL: logical left shift of each byte lane by a compile-time count.
#[inline(always)]
pub unsafe fn _mm256_slli_epi8x_v3<const IMM8: i32>(v: __m256i) -> __m256i {
    let keep = 0xFFu8.wrapping_shl(IMM8 as u32) as i8;
    _mm256_and_si256(_mm256_slli_epi16(v, IMM8), _mm256_set1_epi8(keep))
}

/// POLYFILL: logical right shift of each byte lane by a compile-time count.
#[inline(always)]
pub unsafe fn _mm256_srli_epi8x_v3<const IMM8: i32>(v: __m256i) -> __m256i {
    let keep = (0xFFu8 >> IMM8) as i8;
    _mm256_and_si256(_mm256_srli_epi16(v, IMM8), _mm256_set1_epi8(keep))
}

/// POLYFILL: arithmetic right shift of each `i8` lane by a compile-time count.
#[inline(always)]
pub unsafe fn _mm256_srai_epi8x_v3<const IMM8: i32>(v: __m256i) -> __m256i {
    let logical = _mm256_srli_epi8x_v3::<IMM8>(v);
    let m = _mm256_set1_epi8((0x80u8 >> IMM8) as i8);
    _mm256_sub_epi8(_mm256_xor_si256(logical, m), m)
}

/// POLYFILL: logical left shift of each byte lane by a runtime count.
#[inline(always)]
pub unsafe fn _mm256_sll_epi8x_v3(v: __m256i, shift: u32) -> __m256i {
    let keep = (0xFFu32.wrapping_shl(shift) as u8) as i8;
    _mm256_and_si256(
        _mm256_sll_epi16(v, _mm_cvtsi32_si128(shift as i32)),
        _mm256_set1_epi8(keep),
    )
}

/// POLYFILL: logical right shift of each byte lane by a runtime count.
#[inline(always)]
pub unsafe fn _mm256_srl_epi8x_v3(v: __m256i, shift: u32) -> __m256i {
    let keep = ((0xFFu32 >> shift.min(31)) as u8) as i8;
    _mm256_and_si256(
        _mm256_srl_epi16(v, _mm_cvtsi32_si128(shift as i32)),
        _mm256_set1_epi8(keep),
    )
}

/// POLYFILL: arithmetic right shift of each `i8` lane by a runtime count.
#[inline(always)]
pub unsafe fn _mm256_sra_epi8x_v3(v: __m256i, shift: u32) -> __m256i {
    let logical = _mm256_srl_epi8x_v3(v, shift);
    let m = _mm256_set1_epi8(((0x80u32 >> shift.min(31)) as u8) as i8);
    _mm256_sub_epi8(_mm256_xor_si256(logical, m), m)
}

/// POLYFILL: low 8 bits of each byte product (`a[i].wrapping_mul(b[i])`).
#[inline(always)]
pub unsafe fn _mm256_mullo_epi8x_v3(a: __m256i, b: __m256i) -> __m256i {
    let lo_mask = _mm256_set1_epi16(0x00FF);
    let even = _mm256_mullo_epi16(_mm256_and_si256(a, lo_mask), _mm256_and_si256(b, lo_mask));
    let odd = _mm256_mullo_epi16(_mm256_srli_epi16(a, 8), _mm256_srli_epi16(b, 8));
    _mm256_or_si256(_mm256_and_si256(even, lo_mask), _mm256_slli_epi16(odd, 8))
}

/// POLYFILL: high 8 bits of each signed byte product.
#[inline(always)]
pub unsafe fn _mm256_mulhi_epi8x_v3(a: __m256i, b: __m256i) -> __m256i {
    let zero = _mm256_setzero_si256();
    let a_lo = _mm256_srai_epi16(_mm256_unpacklo_epi8(zero, a), 8);
    let b_lo = _mm256_srai_epi16(_mm256_unpacklo_epi8(zero, b), 8);
    let a_hi = _mm256_srai_epi16(_mm256_unpackhi_epi8(zero, a), 8);
    let b_hi = _mm256_srai_epi16(_mm256_unpackhi_epi8(zero, b), 8);
    let p_lo = _mm256_srai_epi16(_mm256_mullo_epi16(a_lo, b_lo), 8);
    let p_hi = _mm256_srai_epi16(_mm256_mullo_epi16(a_hi, b_hi), 8);
    _mm256_packs_epi16(p_lo, p_hi)
}

/// POLYFILL: high 8 bits of each unsigned byte product.
#[inline(always)]
pub unsafe fn _mm256_mulhi_epu8x_v3(a: __m256i, b: __m256i) -> __m256i {
    let zero = _mm256_setzero_si256();
    let a_lo = _mm256_unpacklo_epi8(a, zero);
    let b_lo = _mm256_unpacklo_epi8(b, zero);
    let a_hi = _mm256_unpackhi_epi8(a, zero);
    let b_hi = _mm256_unpackhi_epi8(b, zero);
    let p_lo = _mm256_srli_epi16(_mm256_mullo_epi16(a_lo, b_lo), 8);
    let p_hi = _mm256_srli_epi16(_mm256_mullo_epi16(a_hi, b_hi), 8);
    _mm256_packus_epi16(p_lo, p_hi)
}

#[inline(always)]
pub unsafe fn _mm256_zeroupper_mask_epi64<Z: ZeroUpper>() -> __m256i {
    use crate::element::MaskElement;

    _mm256_setr_epi64x(
        i64::from_bool(0 < Z::N),
        i64::from_bool(1 < Z::N),
        i64::from_bool(2 < Z::N),
        i64::from_bool(3 < Z::N),
    )
}
