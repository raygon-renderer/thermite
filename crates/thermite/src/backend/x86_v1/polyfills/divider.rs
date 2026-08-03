use super::*;

// libdivide.h - Optimized integer division
// https://libdivide.com
//
// Copyright (C) 2010 - 2019 ridiculous_fish, <libdivide@ridiculousfish.com>
// Copyright (C) 2016 - 2019 Kim Walisch, <kim.walisch@gmail.com>

#[inline(always)]
pub unsafe fn _mm_mullhi_epu64x_v1(x: __m128i, y: __m128i) -> __m128i {
    let lomask = _mm_set1_epi64x(0xffffffff);
    let xh = _mm_shuffle_epi32(x, 0xB1); // x0l, x0h, x1l, x1h
    let yh = _mm_shuffle_epi32(y, 0xB1); // y0l, y0h, y1l, y1h
    let w0 = _mm_mul_epu32(x, y); // x0l*y0l, x1l*y1l
    let w1 = _mm_mul_epu32(x, yh); // x0l*y0h, x1l*y1h
    let w2 = _mm_mul_epu32(xh, y); // x0h*y0l, x1h*y0l
    let w3 = _mm_mul_epu32(xh, yh); // x0h*y0h, x1h*y1h
    let w0h = _mm_srli_epi64(w0, 32);
    let s1 = _mm_add_epi64(w1, w0h);
    let s1l = _mm_and_si128(s1, lomask);
    let s1h = _mm_srli_epi64(s1, 32);
    let s2 = _mm_add_epi64(w2, s1l);
    let s2h = _mm_srli_epi64(s2, 32);
    let mut hi = _mm_add_epi64(w3, s1h);

    hi = _mm_add_epi64(hi, s2h);

    hi
}

#[inline(always)]
pub unsafe fn _mm_mullhi_epu32x_v1(a: __m128i, b: __m128i) -> __m128i {
    let hi_product_0_z2_z = _mm_srli_epi64(_mm_mul_epu32(a, b), 32);
    let a1_x3_x = _mm_srli_epi64(a, 32);
    // libdivide assumes a broadcast constant `b` (b0==b1, b2==b3); for a
    // general per-lane `b` the odd lanes must be shifted into the even slots
    // too. Harmless for the broadcast (single-divisor) case.
    let b1_x3_x = _mm_srli_epi64(b, 32);
    let mask = _mm_set_epi32(-1, 0, -1, 0);
    let hi_product_z1_z3 = _mm_and_si128(_mm_mul_epu32(a1_x3_x, b1_x3_x), mask);
    _mm_or_si128(hi_product_0_z2_z, hi_product_z1_z3)
}

// Notable SSE2 version of _mm_mullhi_epi32x, as there is a better _mm_mullhi_epu32 in x86_v2
#[inline(always)]
pub unsafe fn _mm_mullhi_epi32x_v1(a: __m128i, b: __m128i) -> __m128i {
    let mut p = _mm_mullhi_epu32x_v1(a, b);

    // t1 = (a >> 31) & y, arithmetic shift
    let t1 = _mm_and_si128(_mm_srai_epi32(a, 31), b);
    let t2 = _mm_and_si128(_mm_srai_epi32(b, 31), a);
    p = _mm_sub_epi32(p, t1);
    p = _mm_sub_epi32(p, t2);

    p
}

#[inline(always)]
pub unsafe fn _mm_mullhi_epi64x_v1(x: __m128i, y: __m128i) -> __m128i {
    let p = _mm_mullhi_epu64x_v1(x, y);
    let t1 = _mm_and_si128(_mm_signbits_epi64x_v1(x), y);
    let t2 = _mm_and_si128(_mm_signbits_epi64x_v1(y), x);
    _mm_sub_epi64(_mm_sub_epi64(p, t1), t2)
}
