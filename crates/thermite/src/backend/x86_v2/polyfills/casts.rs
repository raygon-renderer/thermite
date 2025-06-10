use super::*;

#[inline(always)]
pub unsafe fn _mm_cvtepu32_psx_v2(x: __m128i) -> __m128 {
    let xmm0 = x;
    let xmm1 = _mm_set1_epu32x(0x4B000000);
    let xmm1 = _mm_blend_epi16(xmm0, xmm1, 170);
    let xmm0 = _mm_srli_epi32(xmm0, 16);
    let xmm2 = _mm_set1_epu32x(0x53000000);
    let xmm0 = _mm_castsi128_ps(_mm_blend_epi16(xmm0, xmm2, 170));
    let xmm2 = _mm_set1_ps(f32::from_bits(0x53000080));
    let xmm0 = _mm_sub_ps(xmm0, xmm2);
    let xmm0 = _mm_add_ps(_mm_castsi128_ps(xmm1), xmm0);

    xmm0
}

/// Only works for inputs in the range: [-2^51, 2^51]
#[inline(always)]
pub unsafe fn _mm_cvtpd_epi64x_limited_v2(mut x: __m128d) -> __m128i {
    // https://stackoverflow.com/a/41148578/2083075
    let m = _mm_set1_pd(0x0018000000000000u64 as i64 as f64);
    x = _mm_add_pd(x, m);
    _mm_sub_epi64(_mm_castpd_si128(x), _mm_castpd_si128(m))
}

/// Only works for inputs in the range: [0, 2^52)
#[inline(always)]
pub unsafe fn _mm_cvtpd_epu64x_limited_v2(x: __m128d) -> __m128i {
    // https://stackoverflow.com/a/41148578/2083075
    let m = _mm_set1_pd(0x0010000000000000u64 as i64 as f64);
    _mm_castpd_si128(_mm_xor_pd(_mm_add_pd(x, m), m))
}

/// Only works for inputs in the range: [-2^51, 2^51]
#[inline(always)]
pub unsafe fn _mm_cvtepi64_pdx_limited_v2(mut x: __m128i) -> __m128d {
    // https://stackoverflow.com/a/41223013/2083075
    let m = _mm_set1_pd(0x0018000000000000u64 as i64 as f64);
    _mm_sub_pd(_mm_castsi128_pd(_mm_add_epi64(x, _mm_castpd_si128(m))), m)
}

/// Only works for inputs in the range: [0, 2^52)
#[inline(always)]
pub unsafe fn _mm_cvtepu64_pdx_limited_v2(mut x: __m128i) -> __m128d {
    // https://stackoverflow.com/a/41223013/2083075
    let m = _mm_set1_pd(0x0010000000000000u64 as i64 as f64);
    _mm_sub_pd(_mm_castsi128_pd(_mm_or_si128(x, _mm_castpd_si128(m))), m)
}

#[inline(always)]
pub unsafe fn _mm_cvtpd_epi64x_v2(x: __m128d) -> __m128i {
    let x0 = _mm_cvttsd_si64(x);
    let x1 = _mm_cvttsd_si64(_mm_shuffle_pd(x, x, 0b11));

    _mm_set_epi64x(x1, x0)
}

// https://stackoverflow.com/a/41223013/2083075
#[inline(always)]
#[rustfmt::skip]
pub unsafe fn _mm_cvtepu64_pdx_v2(v: __m128i) -> __m128d {
    let magic_i_lo   = _mm_set1_epi64x(0x4330000000000000);  // 2^52        encoded as floating-point
    let magic_i_hi32 = _mm_set1_epi64x(0x4530000000000000);  // 2^84        encoded as floating-point
    let magic_i_all  = _mm_set1_epi64x(0x4530000000100000);  // 2^84 + 2^52 encoded as floating-point
    let magic_d_all  = _mm_castsi128_pd(magic_i_all);

    let     v_lo     = _mm_blend_epi16(magic_i_lo, v, 0b00110011);      // Blend the 32 lowest significant bits of v with magic_int_lo
    let mut v_hi     = _mm_srli_epi64(v, 32);                           // Extract the 32 most significant bits of v
            v_hi     = _mm_xor_si128(v_hi, magic_i_hi32);               // Blend v_hi with 0x45300000
    let     v_hi_dbl = _mm_sub_pd(_mm_castsi128_pd(v_hi), magic_d_all); // Compute in double precision:
                       _mm_add_pd(v_hi_dbl, _mm_castsi128_pd(v_lo))     // (v_hi - magic_d_all) + v_lo  Do not assume associativity of floating point addition !!
}

// https://stackoverflow.com/a/41223013/2083075
#[inline(always)]
#[rustfmt::skip]
pub unsafe fn _mm_cvtepi64_pdx_v2(v: __m128i) -> __m128d {
    let magic_i_lo   = _mm_set1_epi64x(0x4330000000000000); // 2^52               encoded as floating-point
    let magic_i_hi32 = _mm_set1_epi64x(0x4530000080000000); // 2^84 + 2^63        encoded as floating-point
    let magic_i_all  = _mm_set1_epi64x(0x4530000080100000); // 2^84 + 2^63 + 2^52 encoded as floating-point
    let magic_d_all  = _mm_castsi128_pd(magic_i_all);

    let     v_lo     = _mm_blend_epi16(magic_i_lo, v, 0b00110011);      // Blend the 32 lowest significant bits of v with magic_int_lo
    let mut v_hi     = _mm_srli_epi64(v, 32);                           // Extract the 32 most significant bits of v
            v_hi     = _mm_xor_si128(v_hi, magic_i_hi32);               // Flip the msb of v_hi and blend with 0x45300000
    let     v_hi_dbl = _mm_sub_pd(_mm_castsi128_pd(v_hi), magic_d_all); // Compute in double precision:
                       _mm_add_pd(v_hi_dbl, _mm_castsi128_pd(v_lo))     // (v_hi - magic_d_all) + v_lo  Do not assume associativity of floating point addition !!
}

#[inline(always)]
pub unsafe fn _mm_cvtps_epu32x_v2(x: __m128) -> __m128i {
    // TODO: This is exactly what LLVM generates for `simd_cast(f32x4 -> u32x4)`, but it's not ideal and
    // produces different results from `f32 as u32` with negative values and values larger than some value
    let xmm0 = x;
    let xmm1 = _mm_set1_ps(f32::from_bits(0x4f000000));
    let xmm2 = _mm_cmplt_ps(xmm0, xmm1);
    let xmm1 = _mm_sub_ps(xmm0, xmm1);
    let xmm1 = _mm_cvtps_epi32(xmm1);
    let xmm3 = _mm_set1_epu32x(0x80000000);
    let xmm1 = _mm_xor_si128(xmm1, xmm3);
    let xmm0 = _mm_cvtps_epi32(xmm0);
    let xmm0 = _mm_blendv_ps(_mm_castsi128_ps(xmm1), _mm_castsi128_ps(xmm0), xmm2);

    _mm_castps_si128(xmm0)
}

#[inline(always)]
pub unsafe fn _mm_cvtboolx4_to_epi32_mask_v2(
    value: generic_array::GenericArray<bool, generic_array::typenum::U4>,
) -> __m128i {
    let value: [u8; 4] = core::mem::transmute(value);

    #[rustfmt::skip]
    let mask = _mm_setr_epi8(
        value[0] as i8,
        value[1] as i8,
        value[2] as i8,
        value[3] as i8,
        0, 0, 0, 0,
        0, 0, 0, 0,
        0, 0, 0, 0,
    );

    // take 1-byte mask, convert it to epi32, then compare it with zero to fill gaps
    _mm_cmpgt_epi32(_mm_cvtepi8_epi32(mask), _mm_setzero_si128())
}

#[inline(always)]
pub unsafe fn _mm_cvtboolx2_to_epi64_mask_v2(
    value: generic_array::GenericArray<bool, generic_array::typenum::U2>,
) -> __m128i {
    let value: [u8; 2] = core::mem::transmute(value);

    #[rustfmt::skip]
    let mask = _mm_setr_epi8(
        value[0] as i8,
        value[1] as i8,
        0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0,
    );

    // take 1-byte mask, convert it to epi64, then compare it with zero to fill gaps
    _mm_cmpgt_epi64(_mm_cvtepi8_epi64(mask), _mm_setzero_si128())
}
