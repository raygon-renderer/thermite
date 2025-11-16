use super::*;

// #[inline(always)]
// pub unsafe fn _mm_cvtepu32_psx_v2(x: __m128i) -> __m128 {
//     let xmm0 = x;
//     let xmm1 = _mm_set1_epu32x(0x4B000000);
//     let xmm1 = _mm_blend_epi16(xmm0, xmm1, 170);
//     let xmm0 = _mm_srli_epi32(xmm0, 16);
//     let xmm2 = _mm_set1_epu32x(0x53000000);
//     let xmm0 = _mm_castsi128_ps(_mm_blend_epi16(xmm0, xmm2, 0b10101010));
//     let xmm2 = _mm_set1_ps(f32::from_bits(0x53000080));
//     let xmm0 = _mm_sub_ps(xmm0, xmm2);
//     let xmm0 = _mm_add_ps(_mm_castsi128_ps(xmm1), xmm0);

//     xmm0
// }

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
