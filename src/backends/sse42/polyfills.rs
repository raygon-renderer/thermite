use super::*;

use crate::backends::sse2::polyfills::*;

#[inline(always)]
pub unsafe fn _mm_blendv_epi32x(ymm0: __m128i, ymm1: __m128i, mask: __m128i) -> __m128i {
    _mm_castps_si128(_mm_blendv_ps(
        _mm_castsi128_ps(ymm0),
        _mm_castsi128_ps(ymm1),
        _mm_castsi128_ps(mask),
    ))
}

#[inline(always)]
pub unsafe fn _mm_blendv_epi64x(ymm0: __m128i, ymm1: __m128i, mask: __m128i) -> __m128i {
    _mm_castpd_si128(_mm_blendv_pd(
        _mm_castsi128_pd(ymm0),
        _mm_castsi128_pd(ymm1),
        _mm_castsi128_pd(mask),
    ))
}

#[inline(always)]
pub unsafe fn _mm_cvtepu32_psx(x: __m128i) -> __m128 {
    let xmm0 = x;
    let xmm1 = _mm_set1_epi32(0x4B000000u32 as i32);
    let xmm1 = _mm_blend_epi16(xmm0, xmm1, 170);
    let xmm0 = _mm_srli_epi32(xmm0, 16);
    let xmm2 = _mm_set1_epi32(0x53000000u32 as i32);
    let xmm0 = _mm_castsi128_ps(_mm_blend_epi16(xmm0, xmm2, 170));
    let xmm2 = _mm_set1_ps(f32::from_bits(0x53000080));
    let xmm0 = _mm_sub_ps(xmm0, xmm2);
    let xmm0 = _mm_add_ps(_mm_castsi128_ps(xmm1), xmm0);

    xmm0
}

#[inline(always)]
pub unsafe fn _mm_cvtpd_epi64x_limited(x: __m128d) -> __m128i {
    // https://stackoverflow.com/a/41148578/2083075
    let m = _mm_set1_pd(transmute::<u64, i64>(0x0018000000000000) as f64);
    _mm_sub_epi64(_mm_castpd_si128(_mm_add_pd(x, m)), _mm_castpd_si128(m))
}

#[inline(always)]
pub unsafe fn _mm_cvtpd_epu64x_limited(x: __m128d) -> __m128i {
    // https://stackoverflow.com/a/41148578/2083075
    let m = _mm_set1_pd(transmute::<u64, i64>(0x0010000000000000) as f64);
    _mm_xor_si128(_mm_castpd_si128(_mm_add_pd(x, m)), _mm_castpd_si128(m))
}

// https://stackoverflow.com/a/41223013/2083075
#[inline(always)]
#[rustfmt::skip]
pub unsafe fn _mm_cvtepu64_pdx(v: __m128i) -> __m128d {
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
pub unsafe fn _mm_cvtepi64_pdx(v: __m128i) -> __m128d {
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
pub unsafe fn _mm_adds_epi64x(lhs: __m128i, rhs: __m128i) -> __m128i {
    let res = _mm_add_epi64(lhs, rhs);

    _mm_blendv_epi64x(
        res,
        _mm_blendv_epi64x(_mm_set1_epi64x(i64::MIN), _mm_set1_epi64x(i64::MAX), res),
        _mm_xor_si128(rhs, _mm_cmpgt_epi64(lhs, res)),
    )
}

#[inline(always)]
pub unsafe fn _mm_adds_epi32x(lhs: __m128i, rhs: __m128i) -> __m128i {
    let res = _mm_add_epi32(lhs, rhs);

    _mm_blendv_epi32x(
        res,
        _mm_blendv_epi32x(_mm_set1_epi32(i32::MIN), _mm_set1_epi32(i32::MAX), res),
        _mm_xor_si128(rhs, _mm_cmpgt_epi32(lhs, res)),
    )
}
