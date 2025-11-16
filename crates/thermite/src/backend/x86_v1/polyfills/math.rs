use super::*;

// SSE2 Version
#[inline(always)]
pub unsafe fn _mm_adds_epi32x_v1(lhs: __m128i, rhs: __m128i) -> __m128i {
    let res = _mm_add_epi32(lhs, rhs);

    _mm_blendv_epi8x_v1(
        res,
        _mm_blendv_epi8x_v1(
            _mm_set1_epi32(i32::MIN),
            _mm_set1_epi32(i32::MAX),
            _mm_signbits_epi32x_v1(res),
        ),
        _mm_xor_si128(rhs, _mm_cmpgt_epi32(lhs, res)),
    )
}

// SSE2 Version
#[inline(always)]
pub unsafe fn _mm_subs_epi32x_v1(lhs: __m128i, rhs: __m128i) -> __m128i {
    let res = _mm_sub_epi32(lhs, rhs);

    _mm_blendv_epi8x_v1(
        res,
        _mm_blendv_epi8x_v1(
            _mm_set1_epi32(i32::MIN),
            _mm_set1_epi32(i32::MAX),
            _mm_signbits_epi32x_v1(res),
        ),
        _mm_xor_si128(_mm_cmpgt_epi32(rhs, _mm_setzero_si128()), _mm_cmpgt_epi32(lhs, res)),
    )
}

#[inline(always)]
pub unsafe fn _mm_adds_epi64x_v1(lhs: __m128i, rhs: __m128i) -> __m128i {
    let res = _mm_add_epi64(lhs, rhs);

    _mm_blendv_epi8x_v1(
        res,
        _mm_blendv_epi8x_v1(_mm_set1_epi64x(i64::MIN), _mm_set1_epi64x(i64::MAX), res),
        _mm_xor_si128(rhs, _mm_cmpgt_epi64x_v1(lhs, res)),
    )
}

#[inline(always)]
pub unsafe fn _mm_subs_epi64x_v1(lhs: __m128i, rhs: __m128i) -> __m128i {
    let res = _mm_sub_epi64(lhs, rhs);

    _mm_blendv_epi8x_v1(
        res,
        _mm_blendv_epi8x_v1(_mm_set1_epi64x(i64::MIN), _mm_set1_epi64x(i64::MAX), res),
        _mm_xor_si128(
            _mm_cmpgt_epi64x_v1(rhs, _mm_setzero_si128()),
            _mm_cmpgt_epi64x_v1(lhs, res),
        ),
    )
}

#[inline(always)]
pub unsafe fn zero4_v1(value: __m128) -> __m128 {
    // Mask: [0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF, 0x00000000]
    // value & mask -> clears the top lane
    let mask = _mm_castsi128_ps(_mm_setr_epu32x(!0, !0, !0, 0));

    _mm_and_ps(value, mask)
}

#[inline(always)]
pub unsafe fn one4_v1(value: __m128) -> __m128 {
    let mask = _mm_castsi128_ps(_mm_setr_epu32x(!0, !0, !0, 0));
    let top_one = _mm_setr_ps(0.0, 0.0, 0.0, 1.0);
    _mm_or_ps(_mm_and_ps(value, mask), top_one)
}

/// Borrowed from glam
#[inline(always)]
pub unsafe fn dot3_v1(lhs: __m128, rhs: __m128) -> f32 {
    let x2_y2_z2_w2 = _mm_mul_ps(lhs, rhs);
    let y2_0_0_0 = _mm_shuffle_ps(x2_y2_z2_w2, x2_y2_z2_w2, 0b00_00_00_01);
    let z2_0_0_0 = _mm_shuffle_ps(x2_y2_z2_w2, x2_y2_z2_w2, 0b00_00_00_10);
    let x2y2_0_0_0 = _mm_add_ss(x2_y2_z2_w2, y2_0_0_0);
    _mm_cvtss_f32(_mm_add_ss(x2y2_0_0_0, z2_0_0_0))
}

/// Borrowed from glam
#[inline(always)]
pub unsafe fn cross3_v1(lhs: __m128, rhs: __m128) -> __m128 {
    // x  <-  a.y*b.z - a.z*b.y
    // y  <-  a.z*b.x - a.x*b.z
    // z  <-  a.x*b.y - a.y*b.x
    // We can save a shuffle by grouping it in this wacky order:
    // (self.zxy() * rhs - self * rhs.zxy()).zxy()
    let lhszxy = _mm_shuffle_ps(lhs, lhs, 0b11_01_00_10);
    let rhszxy = _mm_shuffle_ps(rhs, rhs, 0b11_01_00_10);
    let lhszxy_rhs = _mm_mul_ps(lhszxy, rhs);
    let rhszxy_lhs = _mm_mul_ps(rhszxy, lhs);
    let sub = _mm_sub_ps(lhszxy_rhs, rhszxy_lhs);
    _mm_shuffle_ps(sub, sub, 0b11_01_00_10)
}

// // https://stackoverflow.com/a/76436268/2083075
// #[inline(always)]
// pub unsafe fn _mm_mullo_epi64x_v1(lhs: __m128i, rhs: __m128i) -> __m128i {
//     let bswap = _mm_shuffle_epi32(rhs, 0xB1);
//     let prodlh = _mm_mullo_epi32x_v1(lhs, bswap);

//     let prodlh2 = _mm_srli_epi64(prodlh, 32);
//     let prodlh3 = _mm_add_epi32(prodlh2, prodlh);
//     let prodlh4 = _mm_and_si128(prodlh3, _mm_set1_epi64x(0x00000000FFFFFFFF));

//     let prodll = _mm_mul_epu32(lhs, rhs);
//     let prod = _mm_add_epi64(prodll, prodlh4);

//     prod
// }

// derived from LLVM output
#[inline(always)]
pub unsafe fn _mm_mullo_epi64x_v1(xmm0: __m128i, xmm1: __m128i) -> __m128i {
    let xmm2 = _mm_srli_epi64(xmm1, 32);
    let xmm3 = _mm_srli_epi64(xmm0, 32);

    let xmm2 = _mm_mul_epu32(xmm2, xmm0);
    let xmm3 = _mm_mul_epu32(xmm1, xmm3);

    let xmm2 = _mm_add_epi64(xmm3, xmm2);
    let xmm2 = _mm_slli_epi64(xmm2, 32);

    let xmm0 = _mm_mul_epu32(xmm1, xmm0);
    let xmm0 = _mm_add_epi64(xmm0, xmm2);

    xmm0
}

// https://stackoverflow.com/a/17268337/2083075
#[inline(always)]
pub unsafe fn _mm_mullo_epi32x_v1(xmm0: __m128i, xmm1: __m128i) -> __m128i {
    let a13 = _mm_shuffle_epi32(xmm0, 0xF5); // (-,a3,-,a1)
    let b13 = _mm_shuffle_epi32(xmm1, 0xF5); // (-,b3,-,b1)
    let prod02 = _mm_mul_epu32(xmm0, xmm1); // (-,a2*b2,-,a0*b0)
    let prod13 = _mm_mul_epu32(a13, b13); // (-,a3*b3,-,a1*b1)
    let prod01 = _mm_unpacklo_epi32(prod02, prod13); // (-,-,a1*b1,a0*b0)
    let prod23 = _mm_unpackhi_epi32(prod02, prod13); // (-,-,a3*b3,a2*b2)
    let prod = _mm_unpacklo_epi64(prod01, prod23); // (ab3,ab2,ab1,ab0)

    prod
}

#[inline(always)]
pub unsafe fn _mm_mul_epi32_v1(a: __m128i, b: __m128i) -> __m128i {
    // 1. Perform the unsigned multiplication
    // Result contains: [ (u64)a[2]*b[2], (u64)a[0]*b[0] ]
    let prod = _mm_mul_epu32(a, b);

    // 2. Generate sign masks (0xFFFFFFFF if negative, 0x00000000 if positive)
    // We use srai to smear the sign bit across the lane
    let a_sign = _mm_srai_epi32(a, 31);
    let b_sign = _mm_srai_epi32(b, 31);

    // 3. Mask the operands
    // If a is negative, we capture b. If b is negative, we capture a.
    let a_correction = _mm_and_si128(a_sign, b);
    let b_correction = _mm_and_si128(b_sign, a);

    // 4. Sum the corrections
    let correction = _mm_add_epi32(a_correction, b_correction);

    // 5. Apply the shift factor (<< 32)
    // We need to subtract (corr * 2^32).
    // _mm_slli_epi64 shifts the 64-bit elements.
    // This moves the correction terms for indices 0 and 2 into the
    // upper 32 bits of the 64-bit result slots, matching the formula.
    // (Data in indices 1 and 3 is shifted out/overwritten, which is desired).
    // 6. Subtract correction from the unsigned product
    _mm_sub_epi64(prod, _mm_slli_epi64(correction, 32))
}

#[inline(always)]
pub unsafe fn _mm_copysign_epi64x_v1(lhs: __m128i, rhs: __m128i) -> __m128i {
    let change_sign = _mm_xor_si128(
        _mm_cmpgt_epi64x_v1(rhs, _mm_set1_epi64x(-1)), // rhs > -1 = rhs >= 0
        _mm_cmpgt_epi64x_v1(lhs, _mm_set1_epi64x(-1)), // lhs > -1 = lhs >= 0
    );

    _mm_add_epi64(
        _mm_xor_si128(lhs, change_sign), // invert lhs if change_sign is true
        _mm_srli_epi64(change_sign, 63), // 1 if true, 0 if false, to correct for two's complement
    )
}

#[inline(always)]
pub unsafe fn _mm_nextupps_v1(value: __m128) -> __m128 {
    let is_nan = _mm_castps_si128(_mm_cmpneq_ps(value, value));

    let bits = _mm_castps_si128(value); // switching to integer ops may add latency here
    let abs = _mm_andnot_si128(_mm_set1_epu32x(0x8000_0000), bits);

    let is_infinity = _mm_cmpeq_epi32(bits, _mm_set1_epu32x(0x7F80_0000));

    let unchanged = _mm_or_si128(is_nan, is_infinity);

    let is_positive = _mm_cmpeq_epi32(abs, bits);
    let is_zero = _mm_cmpeq_epi32(abs, _mm_setzero_si128());

    let add = _mm_add_epi32(bits, _mm_set1_epi32(1));
    let sub = _mm_sub_epi32(bits, _mm_set1_epi32(1));

    // if(positive) { bits + 1 } else { bits - 1 }
    let next_bits = _mm_blendv_epi8x_v1(sub, add, is_positive);

    // if(is_zero) { 0x1 } else { next_bits }
    let next_bits = _mm_blendv_epi8x_v1(next_bits, _mm_set1_epu32x(0x1), is_zero);

    _mm_castsi128_ps(_mm_blendv_epi8x_v1(next_bits, bits, unchanged))
}

#[inline(always)]
pub unsafe fn _mm_nextdownps_v1(value: __m128) -> __m128 {
    let is_nan = _mm_castps_si128(_mm_cmpneq_ps(value, value));

    let bits = _mm_castps_si128(value); // switching to integer ops may add latency here
    let abs = _mm_andnot_si128(_mm_set1_epu32x(0x8000_0000u32), bits);

    let is_neg_infinity = _mm_cmpeq_epi32(bits, _mm_set1_epu32x(0xFF80_0000));
    let unchanged = _mm_or_si128(is_nan, is_neg_infinity);

    let is_positive = _mm_cmpeq_epi32(abs, bits);
    let is_zero = _mm_cmpeq_epi32(abs, _mm_setzero_si128());

    let add = _mm_add_epi32(bits, _mm_set1_epi32(1));
    let sub = _mm_sub_epi32(bits, _mm_set1_epi32(1));

    // if(positive) { bits - 1 } else { bits + 1 }
    let next_bits = _mm_blendv_epi8x_v1(add, sub, is_positive);
    // if(is_zero) { 0x1 | sign_bit } else { next_bits }
    let next_bits = _mm_blendv_epi8x_v1(next_bits, _mm_set1_epu32x(0x1 | 0x8000_0000), is_zero);

    _mm_castsi128_ps(_mm_blendv_epi8x_v1(next_bits, bits, unchanged))
}

#[inline(always)]
pub unsafe fn _mm_nextuppd_v1(value: __m128d) -> __m128d {
    let is_nan = _mm_castpd_si128(_mm_cmpneq_pd(value, value));

    let bits = _mm_castpd_si128(value); // switching to integer ops may add latency here
    let abs = _mm_andnot_si128(_mm_set1_epu64x(0x8000_0000_0000_0000), bits);

    let is_infinity = _mm_cmpeq_epi64x_v1(bits, _mm_set1_epu64x(0x7FF0_0000_0000_0000));
    let is_positive = _mm_cmpeq_epi64x_v1(abs, bits);
    let is_zero = _mm_cmpeq_epi64x_v1(abs, _mm_setzero_si128());

    let add = _mm_add_epi64(bits, _mm_set1_epu64x(1));
    let sub = _mm_sub_epi64(bits, _mm_set1_epu64x(1));

    // if(positive) { bits + 1 } else { bits - 1 }
    let next_bits = _mm_blendv_epi8x_v1(sub, add, is_positive);

    // if(is_zero) { 0x1 } else { next_bits }
    let next_bits = _mm_blendv_epi8x_v1(next_bits, _mm_set1_epu64x(0x1), is_zero);

    _mm_castsi128_pd(next_bits)
}

#[inline(always)]
pub unsafe fn _mm_nextdownpd_v1(value: __m128d) -> __m128d {
    let is_nan = _mm_castpd_si128(_mm_cmpneq_pd(value, value));

    let bits = _mm_castpd_si128(value); // switching to integer ops may add latency here
    let abs = _mm_andnot_si128(_mm_set1_epu64x(0x8000_0000_0000_0000), bits);

    let is_neg_infinity = _mm_cmpeq_epi64x_v1(bits, _mm_set1_epu64x(0xFFF0_0000_0000_0000));
    let unchanged = _mm_or_si128(is_nan, is_neg_infinity);

    let is_positive = _mm_cmpeq_epi64x_v1(abs, bits);
    let is_zero = _mm_cmpeq_epi64x_v1(abs, _mm_setzero_si128());

    let add = _mm_add_epi64(bits, _mm_set1_epu64x(1));
    let sub = _mm_sub_epi64(bits, _mm_set1_epu64x(1));

    // if(positive) { bits - 1 } else { bits + 1 }
    let next_bits = _mm_blendv_epi8x_v1(add, sub, is_positive);
    // if(is_zero) { 0x1 | sign_bit } else { next_bits }
    let next_bits = _mm_blendv_epi8x_v1(next_bits, _mm_set1_epu64x(0x1 | 0x8000_0000_0000_0000), is_zero);

    _mm_castsi128_pd(_mm_blendv_epi8x_v1(next_bits, bits, unchanged))
}
