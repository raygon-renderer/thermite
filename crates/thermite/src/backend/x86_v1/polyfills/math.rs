use super::*;

// NOTE: The blendv polyfill is a full bitwise select, so every mask must be
// lane-uniform (all-ones/all-zeros). Raw operands are sign-broadcast with the
// `signbits` helpers; comparison results are already lane-uniform.

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
        _mm_xor_si128(_mm_signbits_epi32x_v1(rhs), _mm_cmpgt_epi32(lhs, res)),
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
        _mm_blendv_epi8x_v1(
            _mm_set1_epi64x(i64::MIN),
            _mm_set1_epi64x(i64::MAX),
            _mm_signbits_epi64x_v1(res),
        ),
        _mm_xor_si128(_mm_signbits_epi64x_v1(rhs), _mm_cmpgt_epi64x_v1(lhs, res)),
    )
}

#[inline(always)]
pub unsafe fn _mm_subs_epi64x_v1(lhs: __m128i, rhs: __m128i) -> __m128i {
    let res = _mm_sub_epi64(lhs, rhs);

    _mm_blendv_epi8x_v1(
        res,
        _mm_blendv_epi8x_v1(
            _mm_set1_epi64x(i64::MIN),
            _mm_set1_epi64x(i64::MAX),
            _mm_signbits_epi64x_v1(res),
        ),
        _mm_xor_si128(
            _mm_cmpgt_epi64x_v1(rhs, _mm_setzero_si128()),
            _mm_cmpgt_epi64x_v1(lhs, res),
        ),
    )
}

// ---------------------------------------------------------------------------
// Rounding (SSE4.1 `roundps`/`roundpd` polyfills)
//
// All of these use the classic "magic number" trick: adding and subtracting
// 2^23 (f32) / 2^52 (f64) forces the FPU to round to an integer in the
// current rounding mode (assumed round-to-nearest-even, the Rust default).
// Values with |x| >= 2^23 / 2^52 are already integral and are passed through,
// which also handles NaN and infinity (the `cmplt` is false for NaN).
// ---------------------------------------------------------------------------

/// POLYFILL: `_mm_round_ps(v, _MM_FROUND_TO_NEAREST_INT)` - round half to even.
#[inline(always)]
pub unsafe fn _mm_round_psx_v1(value: __m128) -> __m128 {
    let neg_zero = _mm_set1_ps(-0.0);
    let magic = _mm_set1_ps(8388608.0); // 2^23

    let sign = _mm_and_ps(value, neg_zero);
    let abs = _mm_andnot_ps(neg_zero, value);

    // round |v|, then restore the sign (also turns -0.4 into -0.0, not +0.0)
    let rounded = _mm_sub_ps(_mm_add_ps(abs, magic), magic);
    let rounded = _mm_or_ps(rounded, sign);

    // |v| < 2^23: rounded, else (already integral, inf, NaN): passthrough
    _mm_blendv_psx_v1(value, rounded, _mm_cmplt_ps(abs, magic))
}

/// POLYFILL: `_mm_floor_ps`
#[inline(always)]
pub unsafe fn _mm_floor_psx_v1(value: __m128) -> __m128 {
    let rounded = _mm_round_psx_v1(value);
    // subtract 1 where we rounded up
    _mm_sub_ps(rounded, _mm_and_ps(_mm_cmpgt_ps(rounded, value), _mm_set1_ps(1.0)))
}

/// POLYFILL: `_mm_ceil_ps`
///
/// Implemented as `-floor(-v)` rather than `round + 1` correction: the additive
/// form computes `-0.0 + 0.0` for inputs in `[-0.5, -0.0]` and loses the sign
/// of zero (IEEE: `ceil(-0.5)` is `-0.0`).
#[inline(always)]
pub unsafe fn _mm_ceil_psx_v1(value: __m128) -> __m128 {
    let neg_zero = _mm_set1_ps(-0.0);
    _mm_xor_ps(_mm_floor_psx_v1(_mm_xor_ps(value, neg_zero)), neg_zero)
}

/// POLYFILL: `_mm_round_ps(v, _MM_FROUND_TO_ZERO)` - truncate via `floor(|v|)` with the sign restored.
#[inline(always)]
pub unsafe fn _mm_trunc_psx_v1(value: __m128) -> __m128 {
    let neg_zero = _mm_set1_ps(-0.0);
    let magic = _mm_set1_ps(8388608.0); // 2^23

    let sign = _mm_and_ps(value, neg_zero);
    let abs = _mm_andnot_ps(neg_zero, value);

    let rounded = _mm_sub_ps(_mm_add_ps(abs, magic), magic);
    let rounded = _mm_blendv_psx_v1(abs, rounded, _mm_cmplt_ps(abs, magic));

    // floor(|v|): subtract 1 where we rounded up
    let trunced = _mm_sub_ps(rounded, _mm_and_ps(_mm_cmpgt_ps(rounded, abs), _mm_set1_ps(1.0)));

    _mm_or_ps(trunced, sign)
}

/// POLYFILL: `_mm_round_pd(v, _MM_FROUND_TO_NEAREST_INT)` - round half to even.
#[inline(always)]
pub unsafe fn _mm_round_pdx_v1(value: __m128d) -> __m128d {
    let neg_zero = _mm_set1_pd(-0.0);
    let magic = _mm_set1_pd(4503599627370496.0); // 2^52

    let sign = _mm_and_pd(value, neg_zero);
    let abs = _mm_andnot_pd(neg_zero, value);

    let rounded = _mm_sub_pd(_mm_add_pd(abs, magic), magic);
    let rounded = _mm_or_pd(rounded, sign);

    _mm_blendv_pdx_v1(value, rounded, _mm_cmplt_pd(abs, magic))
}

/// POLYFILL: `_mm_floor_pd`
#[inline(always)]
pub unsafe fn _mm_floor_pdx_v1(value: __m128d) -> __m128d {
    let rounded = _mm_round_pdx_v1(value);
    _mm_sub_pd(rounded, _mm_and_pd(_mm_cmpgt_pd(rounded, value), _mm_set1_pd(1.0)))
}

/// POLYFILL: `_mm_ceil_pd` - see [`_mm_ceil_psx_v1`] for why this is `-floor(-v)`.
#[inline(always)]
pub unsafe fn _mm_ceil_pdx_v1(value: __m128d) -> __m128d {
    let neg_zero = _mm_set1_pd(-0.0);
    _mm_xor_pd(_mm_floor_pdx_v1(_mm_xor_pd(value, neg_zero)), neg_zero)
}

/// POLYFILL: `_mm_round_pd(v, _MM_FROUND_TO_ZERO)` - truncate via `floor(|v|)` with the sign restored.
#[inline(always)]
pub unsafe fn _mm_trunc_pdx_v1(value: __m128d) -> __m128d {
    let neg_zero = _mm_set1_pd(-0.0);
    let magic = _mm_set1_pd(4503599627370496.0); // 2^52

    let sign = _mm_and_pd(value, neg_zero);
    let abs = _mm_andnot_pd(neg_zero, value);

    let rounded = _mm_sub_pd(_mm_add_pd(abs, magic), magic);
    let rounded = _mm_blendv_pdx_v1(abs, rounded, _mm_cmplt_pd(abs, magic));

    let trunced = _mm_sub_pd(rounded, _mm_and_pd(_mm_cmpgt_pd(rounded, abs), _mm_set1_pd(1.0)));

    _mm_or_pd(trunced, sign)
}

// ---------------------------------------------------------------------------
// 32-bit signed helpers (SSSE3 `pabsd`/`psignd` replacements)
// ---------------------------------------------------------------------------

/// POLYFILL: `_mm_abs_epi32`
#[inline(always)]
pub unsafe fn _mm_abs_epi32x_v1(value: __m128i) -> __m128i {
    let m = _mm_srai_epi32(value, 31);
    _mm_sub_epi32(_mm_xor_si128(value, m), m)
}

/// POLYFILL: true `copysign` for `i32` lanes, the magnitude of `lhs` with the
/// sign of `rhs` (negates `lhs` exactly where the signs differ).
#[inline(always)]
pub unsafe fn _mm_copysign_epi32x_v1(lhs: __m128i, rhs: __m128i) -> __m128i {
    let change_sign = _mm_xor_si128(_mm_srai_epi32(lhs, 31), _mm_srai_epi32(rhs, 31));
    _mm_sub_epi32(_mm_xor_si128(lhs, change_sign), change_sign)
}

/// POLYFILL: three-valued signum for `i32` lanes (-1 / 0 / +1), matching Rust `i32::signum`.
#[inline(always)]
pub unsafe fn _mm_signum_epi32x_v1(value: __m128i) -> __m128i {
    let zero = _mm_setzero_si128();
    let lt = _mm_cmpgt_epi32(zero, value); // -1 where value < 0
    let gt = _mm_cmpgt_epi32(value, zero); // -1 where value > 0
    _mm_sub_epi32(lt, gt)
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
