use super::*;

// Ported from libdivide (https://libdivide.com), dual-licensed zlib / BSL-1.0.
// Copyright (C) 2010-2019 ridiculous_fish, 2016-2019 Kim Walisch.

#[inline(always)]
pub unsafe fn _mm256_mullhi_epu64x_v3(x: __m256i, y: __m256i) -> __m256i {
    let lomask = _mm256_set1_epi64x(0xffffffff);
    let xh = _mm256_shuffle_epi32(x, 0xB1); // x0l, x0h, x1l, x1h
    let yh = _mm256_shuffle_epi32(y, 0xB1); // y0l, y0h, y1l, y1h
    let w0 = _mm256_mul_epu32(x, y); // x0l*y0l, x1l*y1l
    let w1 = _mm256_mul_epu32(x, yh); // x0l*y0h, x1l*y1h
    let w2 = _mm256_mul_epu32(xh, y); // x0h*y0l, x1h*y0l
    let w3 = _mm256_mul_epu32(xh, yh); // x0h*y0h, x1h*y1h
    let w0h = _mm256_srli_epi64(w0, 32);
    let s1 = _mm256_add_epi64(w1, w0h);
    let s1l = _mm256_and_si256(s1, lomask);
    let s1h = _mm256_srli_epi64(s1, 32);
    let s2 = _mm256_add_epi64(w2, s1l);
    let s2h = _mm256_srli_epi64(s2, 32);
    let mut hi = _mm256_add_epi64(w3, s1h);

    hi = _mm256_add_epi64(hi, s2h);

    hi
}

#[inline(always)]
pub unsafe fn _mm256_mullhi_epu32x_v3(a: __m256i, b: __m256i) -> __m256i {
    let hi_product_0_z2_z = _mm256_srli_epi64(_mm256_mul_epu32(a, b), 32);
    let a1_x3_x = _mm256_srli_epi64(a, 32);
    // See _mm_mullhi_epu32x_v1: shift `b` too for a general per-lane `b`.
    let b1_x3_x = _mm256_srli_epi64(b, 32);
    let mask = _mm256_set_epi32(-1, 0, -1, 0, -1, 0, -1, 0);
    let hi_product_z1_z3 = _mm256_and_si256(_mm256_mul_epu32(a1_x3_x, b1_x3_x), mask);
    _mm256_or_si256(hi_product_0_z2_z, hi_product_z1_z3)
}

#[inline(always)]
pub unsafe fn _mm256_mullhi_epi32x_v3(a: __m256i, b: __m256i) -> __m256i {
    let hi_product_0_z2_z = _mm256_srli_epi64(_mm256_mul_epi32(a, b), 32);
    let a1_x3_x = _mm256_srli_epi64(a, 32);
    // See _mm_mullhi_epi32x_v2: shift `b` too for a general per-lane `b`.
    let b1_x3_x = _mm256_srli_epi64(b, 32);
    let mask = _mm256_set_epi32(-1, 0, -1, 0, -1, 0, -1, 0);
    let hi_product_z1_z3 = _mm256_and_si256(_mm256_mul_epi32(a1_x3_x, b1_x3_x), mask);
    _mm256_or_si256(hi_product_0_z2_z, hi_product_z1_z3)
}

#[inline(always)]
pub unsafe fn _mm256_mullhi_epi64x_v3(x: __m256i, y: __m256i) -> __m256i {
    let p = _mm256_mullhi_epu64x_v3(x, y);
    let t1 = _mm256_and_si256(_mm256_signbits_epi64x_v3(x), y);
    let t2 = _mm256_and_si256(_mm256_signbits_epi64x_v3(y), x);
    _mm256_sub_epi64(_mm256_sub_epi64(p, t1), t2)
}
