use super::*;

#[inline(always)]
pub unsafe fn _mm256_cvtepu32_psx_v3(x: __m256i) -> __m256 {
    let ymm0 = x;
    let ymm1 = _mm256_set1_epu32x(0x4B000000);
    let ymm1 = _mm256_blend_epi16(ymm0, ymm1, 170);
    let ymm0 = _mm256_srli_epi32(ymm0, 16);
    let ymm2 = _mm256_set1_epu32x(0x53000000);
    let ymm0 = _mm256_castsi256_ps(_mm256_blend_epi16(ymm0, ymm2, 170));
    let ymm2 = _mm256_set1_ps(f32::from_bits(0x53000080));
    let ymm0 = _mm256_sub_ps(ymm0, ymm2);
    let ymm0 = _mm256_add_ps(_mm256_castsi256_ps(ymm1), ymm0);

    ymm0
}

/// Only works for inputs in the range: [-2^51, 2^51]
#[inline(always)]
pub unsafe fn _mm256_cvtpd_epi64x_limited_v3(mut x: __m256d) -> __m256i {
    // https://stackoverflow.com/a/41148578/2083075
    let m = _mm256_set1_pd(0x0018000000000000u64 as i64 as f64);
    x = _mm256_add_pd(x, m);
    _mm256_sub_epi64(_mm256_castpd_si256(x), _mm256_castpd_si256(m))
}

/// Only works for inputs in the range: [0, 2^52)
#[inline(always)]
pub unsafe fn _mm256_cvtpd_epu64x_limited_v3(x: __m256d) -> __m256i {
    // https://stackoverflow.com/a/41148578/2083075
    let m = _mm256_set1_pd(0x0010000000000000u64 as i64 as f64);
    _mm256_castpd_si256(_mm256_xor_pd(_mm256_add_pd(x, m), m))
}

/// Only works for inputs in the range: [-2^51, 2^51]
#[inline(always)]
pub unsafe fn _mm256_cvtepi64_pdx_limited_v3(x: __m256i) -> __m256d {
    // https://stackoverflow.com/a/41223013/2083075
    let m = _mm256_set1_pd(0x0018000000000000u64 as i64 as f64);
    _mm256_sub_pd(_mm256_castsi256_pd(_mm256_add_epi64(x, _mm256_castpd_si256(m))), m)
}

/// Only works for inputs in the range: [0, 2^52)
#[inline(always)]
pub unsafe fn _mm256_cvtepu64_pdx_limited_v3(x: __m256i) -> __m256d {
    // https://stackoverflow.com/a/41223013/2083075
    let m = _mm256_set1_pd(0x0010000000000000u64 as i64 as f64);
    _mm256_sub_pd(_mm256_castsi256_pd(_mm256_or_si256(x, _mm256_castpd_si256(m))), m)
}

// https://stackoverflow.com/a/41223013/2083075
#[inline(always)]
#[rustfmt::skip]
pub unsafe fn _mm256_cvtepu64_pdx_v3(v: __m256i) -> __m256d {
    let magic_i_lo   = _mm256_set1_epi64x(0x4330000000000000);  // 2^52        encoded as floating-point
    let magic_i_hi32 = _mm256_set1_epi64x(0x4530000000000000);  // 2^84        encoded as floating-point
    let magic_i_all  = _mm256_set1_epi64x(0x4530000000100000);  // 2^84 + 2^52 encoded as floating-point
    let magic_d_all  = _mm256_castsi256_pd(magic_i_all);

    let     v_lo     = _mm256_blend_epi32(magic_i_lo, v, 0b01010101);         // Blend the 32 lowest significant bits of v with magic_int_lo
    let mut v_hi     = _mm256_srli_epi64(v, 32);                              // Extract the 32 most significant bits of v
            v_hi     = _mm256_xor_si256(v_hi, magic_i_hi32);                  // Blend v_hi with 0x45300000
    let     v_hi_dbl = _mm256_sub_pd(_mm256_castsi256_pd(v_hi), magic_d_all); // Compute in double precision:
                       _mm256_add_pd(v_hi_dbl, _mm256_castsi256_pd(v_lo))     // (v_hi - magic_d_all) + v_lo  Do not assume associativity of floating point addition !!
}

// https://stackoverflow.com/a/41223013/2083075
#[inline(always)]
#[rustfmt::skip]
pub unsafe fn _mm256_cvtepi64_pdx_v3(v: __m256i) -> __m256d {
    let magic_i_lo   = _mm256_set1_epi64x(0x4330000000000000); // 2^52               encoded as floating-point
    let magic_i_hi32 = _mm256_set1_epi64x(0x4530000080000000); // 2^84 + 2^63        encoded as floating-point
    let magic_i_all  = _mm256_set1_epi64x(0x4530000080100000); // 2^84 + 2^63 + 2^52 encoded as floating-point
    let magic_d_all  = _mm256_castsi256_pd(magic_i_all);

    let     v_lo     = _mm256_blend_epi32(magic_i_lo, v, 0b01010101);         // Blend the 32 lowest significant bits of v with magic_int_lo
    let mut v_hi     = _mm256_srli_epi64(v, 32);                              // Extract the 32 most significant bits of v
            v_hi     = _mm256_xor_si256(v_hi, magic_i_hi32);                  // Flip the msb of v_hi and blend with 0x45300000
    let     v_hi_dbl = _mm256_sub_pd(_mm256_castsi256_pd(v_hi), magic_d_all); // Compute in double precision:
                       _mm256_add_pd(v_hi_dbl, _mm256_castsi256_pd(v_lo))     // (v_hi - magic_d_all) + v_lo  Do not assume associativity of floating point addition !!
}

#[inline(always)]
pub unsafe fn _mm_cvtps_epi64x_v3(x: __m128) -> __m256i {
    let x0 = _mm_cvttss_si64(x);
    let x1 = _mm_cvttss_si64(_mm_permute_ps(x, 1));
    let x2 = _mm_cvttss_si64(_mm_permute_ps(x, 2));
    let x3 = _mm_cvttss_si64(_mm_permute_ps(x, 3));

    _mm256_setr_epi64x(x0, x1, x2, x3)
}

#[inline(always)]
pub unsafe fn _mm256_cvtpd_epi64x_v3(x: __m256d) -> __m256i {
    let low = _mm256_castpd256_pd128(x);
    let high = _mm256_extractf128_pd(x, 1);

    let x0 = _mm_cvttsd_si64(low);
    let x1 = _mm_cvttsd_si64(_mm_permute_pd(low, 1));
    let x2 = _mm_cvttsd_si64(high);
    let x3 = _mm_cvttsd_si64(_mm_permute_pd(high, 1));

    _mm256_setr_epi64x(x0, x1, x2, x3)
}

/// POLYFILL: full-range `f64x4 -> u64x4` conversion (truncating, matching
/// `f64 as u64` for in-range values). See `_mm_cvtpd_epu64x_v1`.
#[inline(always)]
pub unsafe fn _mm256_cvtpd_epu64x_v3(x: __m256d) -> __m256i {
    let bound = _mm256_set1_pd(9223372036854775808.0); // 2^63
    let big = _mm256_cmp_pd(x, bound, _CMP_GE_OQ);
    let xs = _mm256_sub_pd(x, _mm256_and_pd(big, bound));

    _mm256_or_si256(
        _mm256_cvtpd_epi64x_v3(xs),
        _mm256_and_si256(_mm256_castpd_si256(big), _mm256_set1_epi64x(i64::MIN)),
    )
}

/// POLYFILL: `f32x8 -> i32x8` saturating cast (`f32 as i32`).
/// See `_mm_cvtps_epi32_satx_v1` for the MIN-XOR trick.
#[inline(always)]
pub unsafe fn _mm256_cvtps_epi32_satx_v3(x: __m256) -> __m256i {
    let t = _mm256_cvttps_epi32(x);
    let hi = _mm256_castps_si256(_mm256_cmp_ps(x, _mm256_set1_ps(2147483648.0), _CMP_GE_OQ)); // 2^31
    let nan = _mm256_castps_si256(_mm256_cmp_ps(x, x, _CMP_UNORD_Q));

    _mm256_andnot_si256(nan, _mm256_xor_si256(t, hi))
}

/// POLYFILL: `f32x8 -> u32x8` saturating cast (`f32 as u32`).
/// See `_mm_cvtps_epu32_satx_v1`.
#[inline(always)]
pub unsafe fn _mm256_cvtps_epu32_satx_v3(x: __m256) -> __m256i {
    let x0 = _mm256_max_ps(x, _mm256_setzero_ps()); // NaN and negatives -> 0
    let t = _mm256_cvtps_epu32x_v3(x0);
    let hi = _mm256_castps_si256(_mm256_cmp_ps(x0, _mm256_set1_ps(4294967296.0), _CMP_GE_OQ)); // 2^32

    _mm256_or_si256(t, hi)
}

/// POLYFILL: `f64x4 -> i64x4` saturating cast (`f64 as i64`).
#[inline(always)]
pub unsafe fn _mm256_cvtpd_epi64_satx_v3(x: __m256d) -> __m256i {
    let t = _mm256_cvtpd_epi64x_v3(x);
    let hi = _mm256_castpd_si256(_mm256_cmp_pd(x, _mm256_set1_pd(9223372036854775808.0), _CMP_GE_OQ)); // 2^63
    let nan = _mm256_castpd_si256(_mm256_cmp_pd(x, x, _CMP_UNORD_Q));

    _mm256_andnot_si256(nan, _mm256_xor_si256(t, hi))
}

/// POLYFILL: `f64x4 -> u64x4` saturating cast (`f64 as u64`).
#[inline(always)]
pub unsafe fn _mm256_cvtpd_epu64_satx_v3(x: __m256d) -> __m256i {
    let x0 = _mm256_max_pd(x, _mm256_setzero_pd()); // NaN and negatives -> 0
    let t = _mm256_cvtpd_epu64x_v3(x0);
    let hi = _mm256_castpd_si256(_mm256_cmp_pd(x0, _mm256_set1_pd(18446744073709551616.0), _CMP_GE_OQ)); // 2^64

    _mm256_or_si256(t, hi)
}

#[inline(always)]
pub unsafe fn _mm256_cvtps_epu32x_v3(x: __m256) -> __m256i {
    // TODO: This is exactly what LLVM generates for `simd_cast(f32x4 -> u32x4)`, but it's not ideal and
    // produces different results from `f32 as u32` with negative values and values larger than some value
    let xmm0 = x;
    let xmm1 = _mm256_set1_ps(f32::from_bits(0x4f000000));
    let xmm2 = _mm256_cmp_ps(xmm0, xmm1, _CMP_LT_OQ);
    let xmm1 = _mm256_sub_ps(xmm0, xmm1);
    // `cvtt` (truncate toward zero) to match `f32 as u32` in-range.
    let xmm1 = _mm256_cvttps_epi32(xmm1);
    let xmm3 = _mm256_set1_epu32x(0x80000000);
    let xmm1 = _mm256_xor_si256(xmm1, xmm3);
    let xmm0 = _mm256_cvttps_epi32(xmm0);
    let xmm0 = _mm256_blendv_ps(_mm256_castsi256_ps(xmm1), _mm256_castsi256_ps(xmm0), xmm2);

    _mm256_castps_si256(xmm0)
}

#[inline(always)]
pub unsafe fn _mm256_cvtpd_epu32x_v3(ymm0: __m256d) -> __m128i {
    let ymm1 = _mm256_set1_pd(f64::from_bits(0x41e0000000000000));
    let ymm2 = _mm256_cmp_pd(ymm0, ymm1, _CMP_LT_OQ);
    let xmm2 = _mm256_castpd256_pd128(ymm2); // lower half of ymm2
    let xmm3 = _mm256_extractf128_pd(ymm2, 1);
    let xmm2 = _mm_packs_epi32(_mm_castpd_si128(xmm2), _mm_castpd_si128(xmm3));
    let ymm1 = _mm256_sub_pd(ymm0, ymm1);
    let xmm1 = _mm256_cvttpd_epi32(ymm1);
    let xmm3 = _mm_set1_ps(f32::from_bits(0x80000000));
    let xmm1 = _mm_xor_ps(_mm_castsi128_ps(xmm1), xmm3);
    let xmm0 = _mm256_cvttpd_epi32(ymm0);
    let xmm0 = _mm_blendv_ps(xmm1, _mm_castsi128_ps(xmm0), _mm_castsi128_ps(xmm2));

    _mm_castps_si128(xmm0)
}

#[inline(always)]
pub unsafe fn _mm256_cvtboolx8_to_epi32_mask_v3(
    value: generic_array::GenericArray<bool, generic_array::typenum::U8>,
) -> __m256i {
    let value: [u8; 8] = core::mem::transmute(value);

    #[rustfmt::skip]
    let mask = _mm_setr_epi8(
        value[0] as i8,
        value[1] as i8,
        value[2] as i8,
        value[3] as i8,
        value[4] as i8,
        value[5] as i8,
        value[6] as i8,
        value[7] as i8,
        0, 0, 0, 0,
        0, 0, 0, 0,
    );

    // take 1-byte mask, convert it to epi32, then compare it with zero to fill gaps
    _mm256_cmpgt_epi32(_mm256_cvtepi8_epi32(mask), _mm256_setzero_si256())
}

#[inline(always)]
pub unsafe fn _mm256_cvtboolx4_to_epi64_mask_v3(
    value: generic_array::GenericArray<bool, generic_array::typenum::U4>,
) -> __m256i {
    let value: [u8; 4] = core::mem::transmute(value);

    #[rustfmt::skip]
    let mask = _mm_setr_epi8(
        value[0] as i8,
        value[1] as i8,
        value[2] as i8,
        value[3] as i8,
        0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0,
    );

    // take 1-byte mask, convert it to epi64, then compare it with zero to fill gaps
    _mm256_cmpgt_epi64(_mm256_cvtepi8_epi64(mask), _mm256_setzero_si256())
}

#[inline(always)]
pub unsafe fn _mm256_cvtepi64_epi32_v3(x: __m256i) -> __m128i {
    // Take the lower 32-bits of each 64-bit integer
    _mm256_castsi256_si128(_mm256_permutevar8x32_epi32(
        x,
        _mm256_setr_epi32(0, 2, 4, 6, 0, 0, 0, 0),
    ))
}

// ===========================================================================================
// Cross-family integer narrow/widen "instructions" that x86 lacks at this level (no `as`-style
// truncating int narrow). These are the register-level polyfills behind the 8/16 <-> 16/32/64
// and 8/16 <-> f64 casts in `registers/half8.rs` / `half16.rs`. v3 is AVX2, so some forms operate
// on 256-bit `__m256i`/`__m256d` sources; multi-input (`[__m256i; N]`) forms are treated as
// instructions that read more than one register.
// ===========================================================================================

// The pshufb masks that gather lane 0 (or word 0) of each wide lane into the contiguous low
// bytes/words are identical at this level to v2's, so they are inherited from
// `x86_v2::polyfills` (re-exported in `mod.rs`): `_mm_narrow_word_to_byte_maskx_v2`,
// `_mm_narrow_dword_to_byte_maskx_v2`, `_mm_narrow_dword_to_word_maskx_v2`,
// `_mm_narrow_qword_to_byte_maskx_v2`, `_mm_narrow_qword_to_word_maskx_v2`.

// Narrow 16x i16 (256-bit) -> 16 i8 (`as`-style truncation), low byte of each lane. Split the
// 256-bit source into two 128-bit halves, pshufb each to pack its 8 low bytes into the low 64
// bits, then merge: a "cvtepi16_epi8" over a 256-bit register.
#[inline(always)]
pub unsafe fn _mm256_cvtepi16_epi8x_v3(value: __m256i) -> __m128i {
    let lo = _mm256_castsi256_si128(value); // lanes 0..8
    let hi = _mm256_extracti128_si256(value, 1); // lanes 8..16
    let mask = _mm_narrow_word_to_byte_maskx_v2();
    let lo = _mm_shuffle_epi8(lo, mask); // 8 bytes in low 64 bits
    let hi = _mm_shuffle_epi8(hi, mask);
    _mm_unpacklo_epi64(lo, hi) // 16 bytes
}

// Narrow 8x i32 (256-bit) -> 8 i8 (low byte of each lane) into the low 8 bytes of a __m128i:
// a "cvtepi32_epi8" over a 256-bit register.
#[inline(always)]
pub unsafe fn _mm256_cvtepi32_epi8x_v3(value: __m256i) -> __m128i {
    let lo = _mm256_castsi256_si128(value); // lanes 0..4
    let hi = _mm256_extracti128_si256(value, 1); // lanes 4..8
    let mask = _mm_narrow_dword_to_byte_maskx_v2();
    let lo = _mm_shuffle_epi8(lo, mask); // 4 bytes in low 32 bits
    let hi = _mm_shuffle_epi8(hi, mask);
    _mm_unpacklo_epi32(lo, hi) // 8 bytes in low 64 bits
}

// Narrow 2x i32x8 (16 i32) -> 16 i8: each half narrows to 8 bytes in the low 64 bits; merge into
// 16 bytes. A "cvtepi32_epi8" over two 256-bit registers.
#[inline(always)]
pub unsafe fn _mm_cvt2epi32x8_epi8x_v3(v: [__m256i; 2]) -> __m128i {
    _mm_unpacklo_epi64(_mm256_cvtepi32_epi8x_v3(v[0]), _mm256_cvtepi32_epi8x_v3(v[1]))
}

// Pack the 4 i64 lanes of a 256-bit register into the low 4 bytes of a __m128i (byte 0 of each):
// a "cvtepi64_epi8" over a 256-bit register.
#[inline(always)]
pub unsafe fn _mm256_cvtepi64_epi8x_v3(value: __m256i) -> __m128i {
    let lo = _mm256_castsi256_si128(value); // lanes 0..2
    let hi = _mm256_extracti128_si256(value, 1); // lanes 2..4
    let mask = _mm_narrow_qword_to_byte_maskx_v2();
    let lo = _mm_shuffle_epi8(lo, mask); // 2 bytes in low 16 bits
    let hi = _mm_shuffle_epi8(hi, mask);
    _mm_unpacklo_epi16(lo, hi) // 4 bytes in low 32 bits
}

// Narrow 2x i64x4 (8 i64) -> 8 i8: each half narrows to 4 bytes in the low 32 bits; merge into 8
// bytes. A "cvtepi64_epi8" over two 256-bit registers.
#[inline(always)]
pub unsafe fn _mm_cvt2epi64x4_epi8x_v3(v: [__m256i; 2]) -> __m128i {
    _mm_unpacklo_epi32(_mm256_cvtepi64_epi8x_v3(v[0]), _mm256_cvtepi64_epi8x_v3(v[1]))
}

// Narrow 2x f64x4 (8 f64) -> 8 i8: truncate each lane to i32x4, pshufb each to 4 bytes, merge into
// the low 64 bits. A truncating "cvtt2pd_epi8" over two 256-bit registers.
#[inline(always)]
pub unsafe fn _mm_cvt2pd4_epi8x_v3(v: [__m256d; 2]) -> __m128i {
    let mask = _mm_narrow_dword_to_byte_maskx_v2();
    let lo = _mm_shuffle_epi8(_mm256_cvttpd_epi32(v[0]), mask); // 4 bytes in low 32 bits
    let hi = _mm_shuffle_epi8(_mm256_cvttpd_epi32(v[1]), mask);
    _mm_unpacklo_epi32(lo, hi) // 8 bytes in low 64 bits
}

// Pack the 4 i64 lanes of a 256-bit register into the low 4 words of a __m128i (word 0 of each):
// a "cvtepi64_epi16" over a 256-bit register.
#[inline(always)]
pub unsafe fn _mm256_cvtepi64_epi16x_v3(value: __m256i) -> __m128i {
    let lo = _mm256_castsi256_si128(value); // lanes 0..2
    let hi = _mm256_extracti128_si256(value, 1); // lanes 2..4
    let mask = _mm_narrow_qword_to_word_maskx_v2();
    let lo = _mm_shuffle_epi8(lo, mask); // 2 words in low 32 bits
    let hi = _mm_shuffle_epi8(hi, mask);
    _mm_unpacklo_epi32(lo, hi) // 4 words in low 64 bits
}

// Narrow 2x i64x4 (8 i64) -> 8 i16: each half narrows to 4 words in the low 64 bits; merge into 8
// words. A "cvtepi64_epi16" over two 256-bit registers.
#[inline(always)]
pub unsafe fn _mm_cvt2epi64x4_epi16x_v3(v: [__m256i; 2]) -> __m128i {
    _mm_unpacklo_epi64(_mm256_cvtepi64_epi16x_v3(v[0]), _mm256_cvtepi64_epi16x_v3(v[1]))
}

// Pack the low word of each of 8 i32 lanes (256-bit) into the low 8 words of a __m128i: a
// "cvtepi32_epi16" over a 256-bit register.
#[inline(always)]
pub unsafe fn _mm256_cvtepi32_epi16x_v3(value: __m256i) -> __m128i {
    let lo = _mm256_castsi256_si128(value); // lanes 0..4
    let hi = _mm256_extracti128_si256(value, 1); // lanes 4..8
    let mask = _mm_narrow_dword_to_word_maskx_v2();
    let lo = _mm_shuffle_epi8(lo, mask); // 4 words in low 64 bits
    let hi = _mm_shuffle_epi8(hi, mask);
    _mm_unpacklo_epi64(lo, hi) // 8 words
}

// Narrow 2x f64x4 (8 f64) -> 8 i16: truncate each lane to i32x4, pshufb each to 4 words, merge
// into the low 128 bits. A truncating "cvtt2pd_epi16" over two 256-bit registers.
#[inline(always)]
pub unsafe fn _mm_cvt2pd4_epi16x_v3(v: [__m256d; 2]) -> __m128i {
    let mask = _mm_narrow_dword_to_word_maskx_v2();
    let lo = _mm_shuffle_epi8(_mm256_cvttpd_epi32(v[0]), mask); // 4 words in low 64 bits
    let hi = _mm_shuffle_epi8(_mm256_cvttpd_epi32(v[1]), mask);
    _mm_unpacklo_epi64(lo, hi) // 8 words
}
