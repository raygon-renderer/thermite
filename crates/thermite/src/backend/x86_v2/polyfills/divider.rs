use super::*;

// libdivide.h - Optimized integer division
// https://libdivide.com
//
// Copyright (C) 2010 - 2019 ridiculous_fish, <libdivide@ridiculousfish.com>
// Copyright (C) 2016 - 2019 Kim Walisch, <kim.walisch@gmail.com>

#[inline(always)]
pub unsafe fn _mm_mullhi_epi32x_v2(a: __m128i, b: __m128i) -> __m128i {
    let hi_product_0_z2_z = _mm_srli_epi64(_mm_mul_epi32(a, b), 32);
    let a1_x3_x = _mm_srli_epi64(a, 32);
    let mask = _mm_set_epi32(-1, 0, -1, 0);
    let hi_product_z1_z3 = _mm_and_si128(_mm_mul_epi32(a1_x3_x, b), mask);
    _mm_or_si128(hi_product_0_z2_z, hi_product_z1_z3)
}
