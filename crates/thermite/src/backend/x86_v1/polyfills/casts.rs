use super::*;

/// POLYFILL: full-range `u32x4 -> f32x4` conversion.
///
/// Same magic-number algorithm as the v2 version, but the SSE4.1
/// `_mm_blend_epi16` word blends are replaced by and/or merges (the merged
/// halves never overlap, so plain bitwise ops suffice).
#[inline(always)]
pub unsafe fn _mm_cvtepu32_psx_v1(x: __m128i) -> __m128 {
    // low 16 bits of each lane with exponent 2^23
    let lo = _mm_or_si128(_mm_and_si128(x, _mm_set1_epi32(0xFFFF)), _mm_set1_epu32x(0x4B000000));
    // high 16 bits of each lane with exponent 2^39
    let hi = _mm_or_si128(_mm_srli_epi32(x, 16), _mm_set1_epu32x(0x53000000));

    // subtract the combined bias, then add the halves
    let hi = _mm_sub_ps(_mm_castsi128_ps(hi), _mm_set1_ps(f32::from_bits(0x53000080)));
    _mm_add_ps(_mm_castsi128_ps(lo), hi)
}

/// POLYFILL: `f32x4 -> u32x4` conversion (truncating, matching `f32 as u32` in range).
///
/// Same as the v2 version with the `_mm_blendv_ps` replaced by a bitwise select.
#[inline(always)]
pub unsafe fn _mm_cvtps_epu32x_v1(x: __m128) -> __m128i {
    let bound = _mm_set1_ps(f32::from_bits(0x4f000000)); // 2^31
    let in_range = _mm_cmplt_ps(x, bound);

    // high range: subtract 2^31, convert, then flip the MSB back on
    let offset = _mm_xor_si128(_mm_cvttps_epi32(_mm_sub_ps(x, bound)), _mm_set1_epu32x(0x80000000));
    let direct = _mm_cvttps_epi32(x);

    _mm_castps_si128(_mm_blendv_psx_v1(
        _mm_castsi128_ps(offset),
        _mm_castsi128_ps(direct),
        in_range,
    ))
}

/// POLYFILL: full-range `f64x2 -> u32x2` conversion (truncating). SSE2-clean
/// copy of the v2 algorithm.
#[inline(always)]
pub unsafe fn _mm_cvtpd_epu32x_v1(xmm0: __m128d) -> __m128i {
    // 1. Threshold: 2^31
    let bound = _mm_set1_pd(f64::from_bits(0x41e0000000000000));

    // 2. Mask: xmm0 < 2^31, as 64-bit lane masks
    let cmp_mask = _mm_cmplt_pd(xmm0, bound);

    // 3. Align the 64-bit masks to the packed 32-bit output lanes
    let mask_32 = _mm_shuffle_epi32(_mm_castpd_si128(cmp_mask), 0b11_11_10_00);

    // 4. High-range path (offset conversion)
    let offset_converted = _mm_xor_si128(
        _mm_cvttpd_epi32(_mm_sub_pd(xmm0, bound)),
        _mm_set1_epi32(0x80000000u32 as i32),
    );

    // 5. Low-range path (direct conversion)
    let direct_converted = _mm_cvttpd_epi32(xmm0);

    // 6. Bitwise blend: (direct & mask) | (offset & ~mask)
    _mm_or_si128(
        _mm_and_si128(mask_32, direct_converted),
        _mm_andnot_si128(mask_32, offset_converted),
    )
}

/// POLYFILL: full-range `f64x2 -> i64x2` conversion (truncating) via two
/// scalar `cvttsd` instructions (SSE2, x86-64 only - same as the v2 version).
#[inline(always)]
#[cfg(target_arch = "x86_64")]
pub unsafe fn _mm_cvtpd_epi64x_v1(x: __m128d) -> __m128i {
    let x0 = _mm_cvttsd_si64(x);
    let x1 = _mm_cvttsd_si64(_mm_shuffle_pd(x, x, 0b11));

    _mm_set_epi64x(x1, x0)
}

/// POLYFILL: full-range `u64x2 -> f64x2` conversion.
///
/// Same magic-number algorithm as the v2 version; the `_mm_blend_epi16`
/// (select the low dword of each lane from `v`) is replaced by an and/or
/// merge, since `magic_i_lo` has a zero low dword.
#[inline(always)]
#[rustfmt::skip]
pub unsafe fn _mm_cvtepu64_pdx_v1(v: __m128i) -> __m128d {
    let magic_i_lo   = _mm_set1_epi64x(0x4330000000000000);  // 2^52        encoded as floating-point
    let magic_i_hi32 = _mm_set1_epi64x(0x4530000000000000);  // 2^84        encoded as floating-point
    let magic_i_all  = _mm_set1_epi64x(0x4530000000100000);  // 2^84 + 2^52 encoded as floating-point
    let magic_d_all  = _mm_castsi128_pd(magic_i_all);

    let     v_lo     = _mm_or_si128(_mm_and_si128(v, _mm_set1_epi64x(0xFFFFFFFF)), magic_i_lo);
    let mut v_hi     = _mm_srli_epi64(v, 32);                           // Extract the 32 most significant bits of v
            v_hi     = _mm_xor_si128(v_hi, magic_i_hi32);               // Blend v_hi with 0x45300000
    let     v_hi_dbl = _mm_sub_pd(_mm_castsi128_pd(v_hi), magic_d_all); // Compute in double precision:
                       _mm_add_pd(v_hi_dbl, _mm_castsi128_pd(v_lo))     // (v_hi - magic_d_all) + v_lo  Do not assume associativity of floating point addition !!
}

/// POLYFILL: full-range `i64x2 -> f64x2` conversion. See [`_mm_cvtepu64_pdx_v1`].
#[inline(always)]
#[rustfmt::skip]
pub unsafe fn _mm_cvtepi64_pdx_v1(v: __m128i) -> __m128d {
    let magic_i_lo   = _mm_set1_epi64x(0x4330000000000000); // 2^52               encoded as floating-point
    let magic_i_hi32 = _mm_set1_epi64x(0x4530000080000000); // 2^84 + 2^63        encoded as floating-point
    let magic_i_all  = _mm_set1_epi64x(0x4530000080100000); // 2^84 + 2^63 + 2^52 encoded as floating-point
    let magic_d_all  = _mm_castsi128_pd(magic_i_all);

    let     v_lo     = _mm_or_si128(_mm_and_si128(v, _mm_set1_epi64x(0xFFFFFFFF)), magic_i_lo);
    let mut v_hi     = _mm_srli_epi64(v, 32);                           // Extract the 32 most significant bits of v
            v_hi     = _mm_xor_si128(v_hi, magic_i_hi32);               // Flip the msb of v_hi and blend with 0x45300000
    let     v_hi_dbl = _mm_sub_pd(_mm_castsi128_pd(v_hi), magic_d_all); // Compute in double precision:
                       _mm_add_pd(v_hi_dbl, _mm_castsi128_pd(v_lo))     // (v_hi - magic_d_all) + v_lo  Do not assume associativity of floating point addition !!
}

/// POLYFILL: `_mm_cvtepi32_epi64` (SSE4.1 `pmovsxdq`) - sign-extend the low
/// two `i32` lanes to `i64`.
#[inline(always)]
pub unsafe fn _mm_cvtepi32_epi64x_v1(v: __m128i) -> __m128i {
    _mm_unpacklo_epi32(v, _mm_srai_epi32(v, 31))
}

/// POLYFILL: `_mm_cvtepu32_epi64` (SSE4.1 `pmovzxdq`) - zero-extend the low
/// two `u32` lanes to `u64`.
#[inline(always)]
pub unsafe fn _mm_cvtepu32_epi64x_v1(v: __m128i) -> __m128i {
    _mm_unpacklo_epi32(v, _mm_setzero_si128())
}

/// POLYFILL: build a full-width `i32` lane mask from 4 `bool`s.
#[inline(always)]
pub unsafe fn _mm_cvtboolx4_to_epi32_mask_v1(
    value: generic_array::GenericArray<bool, generic_array::typenum::U4>,
) -> __m128i {
    // -(b as i32) is 0 or all-ones; branchless and SSE-level agnostic
    _mm_setr_epi32(
        -(value[0] as i32),
        -(value[1] as i32),
        -(value[2] as i32),
        -(value[3] as i32),
    )
}

/// POLYFILL: build a full-width `i64` lane mask from 2 `bool`s.
#[inline(always)]
pub unsafe fn _mm_cvtboolx2_to_epi64_mask_v1(
    value: generic_array::GenericArray<bool, generic_array::typenum::U2>,
) -> __m128i {
    _mm_setr_epi64x(-(value[0] as i64), -(value[1] as i64))
}

/// POLYFILL: `_mm_extract_epi32` (SSE4.1 `pextrd`) via shuffle + `movd`.
///
/// `IMM8` must be in `0..=3`; passed directly as the shuffle immediate it
/// selects lane `IMM8` into position 0 (the upper index fields are don't-cares).
#[inline(always)]
pub unsafe fn _mm_extract_epi32x_v1<const IMM8: i32>(v: __m128i) -> i32 {
    _mm_cvtsi128_si32(_mm_shuffle_epi32::<IMM8>(v))
}

/// POLYFILL: `_mm_extract_ps` (SSE4.1 `extractps`) returning the raw bits like the intrinsic.
#[inline(always)]
pub unsafe fn _mm_extract_psx_v1<const IMM8: i32>(v: __m128) -> i32 {
    _mm_extract_epi32x_v1::<IMM8>(_mm_castps_si128(v))
}

/// Only works for inputs in the range: [-2^51, 2^51]
#[inline(always)]
pub unsafe fn _mm_cvtpd_epi64x_limited_v1(mut x: __m128d) -> __m128i {
    // https://stackoverflow.com/a/41148578/2083075
    let m = _mm_set1_pd(0x0018000000000000u64 as i64 as f64);
    x = _mm_add_pd(x, m);
    _mm_sub_epi64(_mm_castpd_si128(x), _mm_castpd_si128(m))
}

/// Only works for inputs in the range: [0, 2^52)
#[inline(always)]
pub unsafe fn _mm_cvtpd_epu64x_limited_v1(x: __m128d) -> __m128i {
    // https://stackoverflow.com/a/41148578/2083075
    let m = _mm_set1_pd(0x0010000000000000u64 as i64 as f64);
    _mm_castpd_si128(_mm_xor_pd(_mm_add_pd(x, m), m))
}

/// Only works for inputs in the range: [-2^51, 2^51]
#[inline(always)]
pub unsafe fn _mm_cvtepi64_pdx_limited_v1(mut x: __m128i) -> __m128d {
    // https://stackoverflow.com/a/41223013/2083075
    let m = _mm_set1_pd(0x0018000000000000u64 as i64 as f64);
    _mm_sub_pd(_mm_castsi128_pd(_mm_add_epi64(x, _mm_castpd_si128(m))), m)
}

/// Only works for inputs in the range: [0, 2^52)
#[inline(always)]
pub unsafe fn _mm_cvtepu64_pdx_limited_v1(mut x: __m128i) -> __m128d {
    // https://stackoverflow.com/a/41223013/2083075
    let m = _mm_set1_pd(0x0010000000000000u64 as i64 as f64);
    _mm_sub_pd(_mm_castsi128_pd(_mm_or_si128(x, _mm_castpd_si128(m))), m)
}

#[inline(always)]
pub unsafe fn _mm_cvtepi64_epi32x_v1(a: __m128i, b: __m128i) -> __m128i {
    // Domain change might incur some performance penalty, but this is the simplest way to do it.
    _mm_castps_si128(_mm_shuffle_ps(_mm_castsi128_ps(a), _mm_castsi128_ps(b), 0b10_00_10_00))
}
