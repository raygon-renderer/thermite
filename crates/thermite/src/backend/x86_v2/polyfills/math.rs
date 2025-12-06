use super::*;

// NOTE: Saturated add/sub use the "sign bit" as the select bit,
// so when porting to SSE2 it'll need to use the `signbits` methods to properly select with `_mm_blendv_epi8x`

#[inline(always)]
pub unsafe fn _mm_adds_epi64x_v2(lhs: __m128i, rhs: __m128i) -> __m128i {
    let res = _mm_add_epi64(lhs, rhs);

    _mm_blendv_epi8(
        res,
        _mm_blendv_epi8(_mm_set1_epi64x(i64::MIN), _mm_set1_epi64x(i64::MAX), res),
        _mm_xor_si128(rhs, _mm_cmpgt_epi64(lhs, res)),
    )
}

#[inline(always)]
pub unsafe fn _mm_adds_epi32x_v2(lhs: __m128i, rhs: __m128i) -> __m128i {
    let res = _mm_add_epi32(lhs, rhs);

    _mm_blendv_epi8(
        res,
        _mm_blendv_epi8(_mm_set1_epi32(i32::MIN), _mm_set1_epi32(i32::MAX), res),
        _mm_xor_si128(rhs, _mm_cmpgt_epi32(lhs, res)),
    )
}

#[inline(always)]
pub unsafe fn _mm_subs_epi32x_v2(lhs: __m128i, rhs: __m128i) -> __m128i {
    let res = _mm_sub_epi32(lhs, rhs);

    _mm_blendv_epi8(
        res,
        _mm_blendv_epi8(_mm_set1_epi32(i32::MIN), _mm_set1_epi32(i32::MAX), res),
        _mm_xor_si128(_mm_cmpgt_epi32(rhs, _mm_setzero_si128()), _mm_cmpgt_epi32(lhs, res)),
    )
}

#[inline(always)]
pub unsafe fn _mm_subs_epi64x_v2(lhs: __m128i, rhs: __m128i) -> __m128i {
    let res = _mm_sub_epi64(lhs, rhs);

    _mm_blendv_epi8(
        res,
        _mm_blendv_epi8(_mm_set1_epi64x(i64::MIN), _mm_set1_epi64x(i64::MAX), res),
        _mm_xor_si128(_mm_cmpgt_epi64(rhs, _mm_setzero_si128()), _mm_cmpgt_epi64(lhs, res)),
    )
}

#[inline(always)]
pub unsafe fn zero4_v2(value: __m128) -> __m128 {
    // NOTE: Compiler may choose to replace the set1 with `xorps xmm, xmm`
    // this may be better than the SSE2 version that uses `andps` with a mask
    // for the last lane, simply because xorps can be paired with other instructions.
    unsafe { _mm_blend_ps(value, _mm_set1_ps(0.0), 0b1000) }
}

#[inline(always)]
pub unsafe fn one4_v2(value: __m128) -> __m128 {
    unsafe { _mm_blend_ps(value, _mm_set1_ps(1.0), 0b1000) }
}

// https://stackoverflow.com/a/76436268/2083075
#[inline(always)]
pub unsafe fn _mm_mullo_epi64x_v2(lhs: __m128i, rhs: __m128i) -> __m128i {
    let bswap = _mm_shuffle_epi32(rhs, 0xB1);
    let prodlh = _mm_mullo_epi32(lhs, bswap);

    let prodlh2 = _mm_srli_epi64(prodlh, 32);
    let prodlh3 = _mm_add_epi32(prodlh2, prodlh);
    let prodlh4 = _mm_and_si128(prodlh3, _mm_set1_epi64x(0x00000000FFFFFFFF));

    let prodll = _mm_mul_epu32(lhs, rhs);
    let prod = _mm_add_epi64(prodll, prodlh4);

    prod
}

#[inline(always)]
pub unsafe fn _mm_copysign_epi64x_v2(lhs: __m128i, rhs: __m128i) -> __m128i {
    let change_sign = _mm_xor_si128(
        _mm_cmpgt_epi64(rhs, _mm_set1_epi64x(-1)), // rhs > -1 = rhs >= 0
        _mm_cmpgt_epi64(lhs, _mm_set1_epi64x(-1)), // lhs > -1 = lhs >= 0
    );

    _mm_add_epi64(
        _mm_xor_si128(lhs, change_sign), // invert lhs if change_sign is true
        _mm_srli_epi64(change_sign, 63), // 1 if true, 0 if false, to correct for two's complement
    )
}
