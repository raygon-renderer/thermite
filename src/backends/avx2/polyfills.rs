use super::*;

pub use crate::backends::avx1::polyfills::{_mm256_blendv_epi32x, _mm256_blendv_epi64x};

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
pub unsafe fn _mm256_signbits_epi64x(v: __m256i) -> __m256i {
    _mm256_srai_epi32(_mm256_shuffle_epi32(v, _mm_shuffle(3, 3, 1, 1)), 31)
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

// https://arxiv.org/pdf/1611.07612.pdf
#[inline(always)]
pub unsafe fn _mm256_popcnt_epi8x(v: __m256i) -> __m256i {
    let lookup = _mm256_setr_epi8(
        0, 1, 1, 2, 1, 2, 2, 3, 1, 2, 2, 3, 2, 3, 3, 4, 0, 1, 1, 2, 1, 2, 2, 3, 1, 2, 2, 3, 2, 3, 3, 4,
    );
    let low_mask = _mm256_set1_epi8(0x0f);
    let lo = _mm256_and_si256(v, low_mask);
    let hi = _mm256_and_si256(_mm256_srli_epi32(v, 4), low_mask);
    let popcnt1 = _mm256_shuffle_epi8(lookup, lo);
    let popcnt2 = _mm256_shuffle_epi8(lookup, hi);
    _mm256_add_epi8(popcnt1, popcnt2)
}

#[inline(always)]
pub unsafe fn _mm256_popcnt_epi64x(v: __m256i) -> __m256i {
    _mm256_sad_epu8(_mm256_popcnt_epi8x(v), _mm256_setzero_si256())
}

#[inline(always)]
pub unsafe fn _mm256_popcnt_epi32x(v: __m256i) -> __m256i {
    // https://stackoverflow.com/a/51106873/2083075
    _mm256_madd_epi16(
        _mm256_maddubs_epi16(_mm256_popcnt_epi8x(v), _mm256_set1_epi8(1)),
        _mm256_set1_epi16(1),
    )
}

pub use divider::*;
pub mod divider {
    use super::*;

    // libdivide.h - Optimized integer division
    // https://libdivide.com
    //
    // Copyright (C) 2010 - 2019 ridiculous_fish, <libdivide@ridiculousfish.com>
    // Copyright (C) 2016 - 2019 Kim Walisch, <kim.walisch@gmail.com>

    #[rustfmt::skip]
    #[inline(always)]
    pub unsafe fn _mm256_mullhi_epu64x(x: __m256i, y: __m256i) -> __m256i {
        let lomask  = _mm256_set1_epi64x(0xffffffff);
        let xh      = _mm256_shuffle_epi32(x, 0xB1);    // x0l, x0h, x1l, x1h
        let yh      = _mm256_shuffle_epi32(y, 0xB1);    // y0l, y0h, y1l, y1h
        let w0      = _mm256_mul_epu32(x, y);           // x0l*y0l, x1l*y1l
        let w1      = _mm256_mul_epu32(x, yh);          // x0l*y0h, x1l*y1h
        let w2      = _mm256_mul_epu32(xh, y);          // x0h*y0l, x1h*y0l
        let w3      = _mm256_mul_epu32(xh, yh);         // x0h*y0h, x1h*y1h
        let w0h     = _mm256_srli_epi64(w0, 32);
        let s1      = _mm256_add_epi64(w1, w0h);
        let s1l     = _mm256_and_si256(s1, lomask);
        let s1h     = _mm256_srli_epi64(s1, 32);
        let s2      = _mm256_add_epi64(w2, s1l);
        let s2h     = _mm256_srli_epi64(s2, 32);
        let mut hi  = _mm256_add_epi64(w3, s1h);
                hi  = _mm256_add_epi64(hi, s2h);

        hi
    }

    #[inline(always)]
    pub unsafe fn _mm256_mullhi_epu32x(a: __m256i, b: __m256i) -> __m256i {
        let hi_product_0Z2Z = _mm256_srli_epi64(_mm256_mul_epu32(a, b), 32);
        let a1X3X = _mm256_srli_epi64(a, 32);
        let mask = _mm256_set_epi32(-1, 0, -1, 0, -1, 0, -1, 0);
        let hi_product_Z1Z3 = _mm256_and_si256(_mm256_mul_epu32(a1X3X, b), mask);
        _mm256_or_si256(hi_product_0Z2Z, hi_product_Z1Z3)
    }

    #[inline(always)]
    pub unsafe fn _mm256_mullhi_epi32x(a: __m256i, b: __m256i) -> __m256i {
        let hi_product_0Z2Z = _mm256_srli_epi64(_mm256_mul_epi32(a, b), 32);
        let a1X3X = _mm256_srli_epi64(a, 32);
        let mask = _mm256_set_epi32(-1, 0, -1, 0, -1, 0, -1, 0);
        let hi_product_Z1Z3 = _mm256_and_si256(_mm256_mul_epi32(a1X3X, b), mask);
        _mm256_or_si256(hi_product_0Z2Z, hi_product_Z1Z3)
    }

    #[inline(always)]
    pub unsafe fn _mm256_mullhi_epi64x(x: __m256i, y: __m256i) -> __m256i {
        let p = _mm256_mullhi_epu64x(x, y);
        let t1 = _mm256_and_si256(_mm256_signbits_epi64x(x), y);
        let t2 = _mm256_and_si256(_mm256_signbits_epi64x(y), x);
        _mm256_sub_epi64(_mm256_sub_epi64(p, t1), t2)
    }

    #[inline(always)]
    pub unsafe fn _mm256_div_epu32x(numers: __m256i, multiplier: u32, shift: u8) -> __m256i {
        if multiplier == 0 {
            return _mm256_srli_epi32(numers, shift as i32);
        }

        let q = _mm256_mullhi_epu32x(numers, _mm256_set1_epi32(multiplier as i32));

        if shift & 0x40 != 0 {
            _mm256_srli_epi32(
                _mm256_add_epi32(_mm256_srli_epi32(_mm256_sub_epi32(numers, q), 1), q),
                (shift & 0x1F) as i32,
            )
        } else {
            _mm256_srli_epi32(q, shift as i32)
        }
    }

    #[inline(always)]
    pub unsafe fn _mm256_div_epu64x(numers: __m256i, multiplier: u64, shift: u8) -> __m256i {
        if multiplier == 0 {
            return _mm256_srli_epi64(numers, shift as i32);
        }

        let q = _mm256_mullhi_epu64x(numers, _mm256_set1_epi64x(multiplier as i64));

        if shift & 0x40 != 0 {
            _mm256_srli_epi64(
                _mm256_add_epi64(_mm256_srli_epi64(_mm256_sub_epi64(numers, q), 1), q),
                (shift & 0x3F) as i32,
            )
        } else {
            _mm256_srli_epi64(q, shift as i32)
        }
    }
}
