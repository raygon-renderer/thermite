use super::*;

// Ported from libdivide (https://libdivide.com), dual-licensed zlib / BSL-1.0.
// Copyright (C) 2010-2019 ridiculous_fish, 2016-2019 Kim Walisch.

#[inline(always)]
pub unsafe fn _mm_mullhi_epi32x_v2(a: __m128i, b: __m128i) -> __m128i {
    let hi_product_0_z2_z = _mm_srli_epi64(_mm_mul_epi32(a, b), 32);
    let a1_x3_x = _mm_srli_epi64(a, 32);
    // NOTE: libdivide multiplies `a1_x3_x` by the *unshifted* `b` because there
    // `b` is a broadcast constant (b0==b1, b2==b3). For a general per-lane `b`
    // we must shift `b` too, so the odd lanes (b1,b3) - not (b0,b2) - are used.
    let b1_x3_x = _mm_srli_epi64(b, 32);
    let mask = _mm_set_epi32(-1, 0, -1, 0);
    let hi_product_z1_z3 = _mm_and_si128(_mm_mul_epi32(a1_x3_x, b1_x3_x), mask);
    _mm_or_si128(hi_product_0_z2_z, hi_product_z1_z3)
}
