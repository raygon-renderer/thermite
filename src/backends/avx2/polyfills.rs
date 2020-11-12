use super::*;

use crate::backends::avx1::polyfills::{_mm256_blendv_epi32x, _mm256_blendv_epi64x};

#[inline(always)]
pub unsafe fn _mm256_cvtepu32_psx(x: __m256i) -> __m256 {
    let ymm0 = x;
    let ymm1 = _mm256_set1_epi32(0x4B000000u32 as i32);
    let ymm1 = _mm256_blend_epi16(ymm0, ymm1, 170);
    let ymm0 = _mm256_srli_epi32(ymm0, 16);
    let ymm2 = _mm256_set1_epi32(0x53000000u32 as i32);
    let ymm0 = _mm256_castsi256_ps(_mm256_blend_epi16(ymm0, ymm2, 170));
    let ymm2 = _mm256_set1_ps(f32::from_bits(0x53000080));
    let ymm0 = _mm256_sub_ps(ymm0, ymm2);
    let ymm0 = _mm256_add_ps(_mm256_castsi256_ps(ymm1), ymm0);

    ymm0
}

#[inline(always)]
pub unsafe fn _mm256_cvtpd_epi64x_limited(x: __m256d) -> __m256i {
    // https://stackoverflow.com/a/41148578/2083075
    let m = _mm256_set1_pd(transmute::<u64, i64>(0x0018000000000000) as f64);
    _mm256_sub_epi64(_mm256_castpd_si256(_mm256_add_pd(x, m)), _mm256_castpd_si256(m))
}

#[inline(always)]
pub unsafe fn _mm256_cvtpd_epu64x_limited(x: __m256d) -> __m256i {
    // https://stackoverflow.com/a/41148578/2083075
    let m = _mm256_set1_pd(transmute::<u64, i64>(0x0010000000000000) as f64);
    _mm256_xor_si256(_mm256_castpd_si256(_mm256_add_pd(x, m)), _mm256_castpd_si256(m))
}

// https://stackoverflow.com/a/41223013/2083075
#[inline(always)]
#[rustfmt::skip]
pub unsafe fn _mm256_cvtepu64_pdx(v: __m256i) -> __m256d {
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
pub unsafe fn _mm256_cvtepi64_pdx(v: __m256i) -> __m256d {
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
pub unsafe fn _mm256_adds_epi32x(lhs: __m256i, rhs: __m256i) -> __m256i {
    let res = _mm256_add_epi32(lhs, rhs);

    _mm256_blendv_epi32x(
        res,
        // cheeky hack relying on only the highest significant bit, which is the effective "sign" bit
        _mm256_blendv_epi64x(_mm256_set1_epi32(i32::MIN), _mm256_set1_epi32(i32::MAX), res),
        _mm256_xor_si256(rhs, _mm256_cmpgt_epi32(lhs, res)),
    )
}

#[inline(always)]
pub unsafe fn _mm256_adds_epi64x(lhs: __m256i, rhs: __m256i) -> __m256i {
    let res = _mm256_add_epi64(lhs, rhs);

    _mm256_blendv_epi64x(
        res,
        _mm256_blendv_epi64x(_mm256_set1_epi64x(i64::MIN), _mm256_set1_epi64x(i64::MAX), res),
        _mm256_xor_si256(rhs, _mm256_cmpgt_epi64(lhs, res)),
    )
}

#[inline(always)]
pub unsafe fn _mm256_subs_epi32x(lhs: __m256i, rhs: __m256i) -> __m256i {
    let res = _mm256_sub_epi32(lhs, rhs);

    _mm256_blendv_epi32x(
        res,
        _mm256_blendv_epi32x(_mm256_set1_epi32(i32::MIN), _mm256_set1_epi32(i32::MAX), res),
        _mm256_xor_si256(
            _mm256_cmpgt_epi32(rhs, _mm256_setzero_si256()),
            _mm256_cmpgt_epi32(lhs, res),
        ),
    )
}

#[inline(always)]
pub unsafe fn _mm256_subs_epi64x(lhs: __m256i, rhs: __m256i) -> __m256i {
    let res = _mm256_sub_epi64(lhs, rhs);

    _mm256_blendv_epi64x(
        res,
        _mm256_blendv_epi64x(_mm256_set1_epi64x(i64::MIN), _mm256_set1_epi64x(i64::MAX), res),
        _mm256_xor_si256(
            _mm256_cmpgt_epi64(rhs, _mm256_setzero_si256()),
            _mm256_cmpgt_epi64(lhs, res),
        ),
    )
}

#[inline(always)]
pub unsafe fn _mm256_abs_epi64x(x: __m256i) -> __m256i {
    let should_negate = _mm256_xor_si256(_mm256_cmpgt_epi64(x, _mm256_setzero_si256()), _mm256_set1_epi64x(-1));

    _mm256_add_epi64(
        _mm256_xor_si256(should_negate, x),
        _mm256_and_si256(should_negate, _mm256_set1_epi64x(1)),
    )
}

#[inline(always)]
pub unsafe fn _mm256_cvtps_epi64(x: __m128) -> __m256i {
    let x0 = _mm_cvttss_si64(x);
    let x1 = _mm_cvttss_si64(_mm_permute_ps(x, 1));
    let x2 = _mm_cvttss_si64(_mm_permute_ps(x, 2));
    let x3 = _mm_cvttss_si64(_mm_permute_ps(x, 3));

    _mm256_setr_epi64x(x0, x1, x2, x3)
}

#[inline(always)]
pub unsafe fn _mm256_cvtpd_epi64x(x: __m256d) -> __m256i {
    let low = _mm256_castpd256_pd128(x);
    let high = _mm256_extractf128_pd(x, 1);

    let x0 = _mm_cvttsd_si64(low);
    let x1 = _mm_cvttsd_si64(_mm_permute_pd(low, 1));
    let x2 = _mm_cvttsd_si64(high);
    let x3 = _mm_cvttsd_si64(_mm_permute_pd(high, 1));

    _mm256_setr_epi64x(x0, x1, x2, x3)
}

#[inline(always)]
pub unsafe fn _mm256_cvtps_epu32x(x: __m256) -> __m256i {
    // TODO: This is exactly what LLVM generates for `simd_cast(f32x4 -> u32x4)`, but it's not ideal and
    // produces different results from `f32 as u32` with negaitve values and values larger than some value
    let xmm0 = x;
    let xmm1 = _mm256_set1_ps(f32::from_bits(0x4f000000));
    let xmm2 = _mm256_cmp_ps(xmm0, xmm1, _CMP_LT_OQ);
    let xmm1 = _mm256_sub_ps(xmm0, xmm1);
    let xmm1 = _mm256_cvtps_epi32(xmm1);
    let xmm3 = _mm256_set1_epi32(0x80000000u32 as i32);
    let xmm1 = _mm256_xor_si256(xmm1, xmm3);
    let xmm0 = _mm256_cvtps_epi32(xmm0);
    let xmm0 = _mm256_blendv_ps(_mm256_castsi256_ps(xmm1), _mm256_castsi256_ps(xmm0), xmm2);

    _mm256_castps_si256(xmm0)
}

#[inline(always)]
pub unsafe fn _mm256_cvtpd_epu32x(ymm0: __m256d) -> __m128i {
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
pub unsafe fn _mm256_mullo_epi64x(ymm0: __m256i, ymm1: __m256i) -> __m256i {
    let ymm2 = _mm256_srli_epi64(ymm1, 32);
    let ymm3 = _mm256_srli_epi64(ymm0, 32);

    let ymm2 = _mm256_mul_epu32(ymm2, ymm0);
    let ymm3 = _mm256_mul_epu32(ymm1, ymm3);

    let ymm2 = _mm256_add_epi64(ymm3, ymm2);
    let ymm2 = _mm256_slli_epi64(ymm2, 32);

    let ymm0 = _mm256_mul_epu32(ymm1, ymm0);
    let ymm0 = _mm256_add_epi64(ymm0, ymm2);

    ymm0
}
