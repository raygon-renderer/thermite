use super::*;

#[inline(always)]
pub unsafe fn _mm256_adds_epi32x_v3(lhs: __m256i, rhs: __m256i) -> __m256i {
    let res = _mm256_add_epi32(lhs, rhs);

    _mm256_blendv_epi32x_v3(
        res,
        _mm256_blendv_epi32x_v3(_mm256_set1_epi32(i32::MIN), _mm256_set1_epi32(i32::MAX), res),
        _mm256_xor_si256(rhs, _mm256_cmpgt_epi32(lhs, res)),
    )
}

#[inline(always)]
pub unsafe fn _mm256_adds_epi64x_v3(lhs: __m256i, rhs: __m256i) -> __m256i {
    let res = _mm256_add_epi64(lhs, rhs);

    _mm256_blendv_epi64x_v3(
        res,
        _mm256_blendv_epi64x_v3(_mm256_set1_epi64x(i64::MIN), _mm256_set1_epi64x(i64::MAX), res),
        _mm256_xor_si256(rhs, _mm256_cmpgt_epi64(lhs, res)),
    )
}

#[inline(always)]
pub unsafe fn _mm256_subs_epi32x_v3(lhs: __m256i, rhs: __m256i) -> __m256i {
    let res = _mm256_sub_epi32(lhs, rhs);

    _mm256_blendv_epi32x_v3(
        res,
        _mm256_blendv_epi32x_v3(_mm256_set1_epi32(i32::MIN), _mm256_set1_epi32(i32::MAX), res),
        _mm256_xor_si256(
            _mm256_cmpgt_epi32(rhs, _mm256_setzero_si256()),
            _mm256_cmpgt_epi32(lhs, res),
        ),
    )
}

#[inline(always)]
pub unsafe fn _mm256_subs_epi64x_v3(lhs: __m256i, rhs: __m256i) -> __m256i {
    let res = _mm256_sub_epi64(lhs, rhs);

    _mm256_blendv_epi64x_v3(
        res,
        _mm256_blendv_epi64x_v3(_mm256_set1_epi64x(i64::MIN), _mm256_set1_epi64x(i64::MAX), res),
        _mm256_xor_si256(
            _mm256_cmpgt_epi64(rhs, _mm256_setzero_si256()),
            _mm256_cmpgt_epi64(lhs, res),
        ),
    )
}

#[inline(always)]
pub unsafe fn _mm256_signbits_epi64x_v3(v: __m256i) -> __m256i {
    _mm256_srai_epi32(_mm256_shuffle_epi32::<{ MM_SHUFFLE!(3, 3, 1, 1) }>(v), 31)
}

// #[inline(always)]
// pub unsafe fn _mm256_mullo_epi64x_v3(ymm0: __m256i, ymm1: __m256i) -> __m256i {
//     let ymm2 = _mm256_srli_epi64(ymm1, 32);
//     let ymm3 = _mm256_srli_epi64(ymm0, 32);
//     let ymm2 = _mm256_mul_epu32(ymm2, ymm0);
//     let ymm3 = _mm256_mul_epu32(ymm1, ymm3);
//     let ymm2 = _mm256_add_epi64(ymm3, ymm2);
//     let ymm2 = _mm256_slli_epi64(ymm2, 32);
//     let ymm0 = _mm256_mul_epu32(ymm1, ymm0);
//     let ymm0 = _mm256_add_epi64(ymm0, ymm2);
//     ymm0
// }

// https://stackoverflow.com/a/76436268/2083075
#[inline(always)]
pub unsafe fn _mm256_mullo_epi64x_v3(lhs: __m256i, rhs: __m256i) -> __m256i {
    let bswap = _mm256_shuffle_epi32(rhs, 0xB1);
    let prodlh = _mm256_mullo_epi32(lhs, bswap);

    let prodlh2 = _mm256_srli_epi64(prodlh, 32);
    let prodlh3 = _mm256_add_epi32(prodlh2, prodlh);
    let prodlh4 = _mm256_and_si256(prodlh3, _mm256_set1_epi64x(0x00000000FFFFFFFF));

    let prodll = _mm256_mul_epu32(lhs, rhs);
    // The cross term (a_lo*b_hi + a_hi*b_lo) occupies bits 32..63, so it must
    // be shifted left by 32 before being added to the low product. The mask
    // above zeroes its high dword so the shift can't bleed garbage in.
    let prod = _mm256_add_epi64(prodll, _mm256_slli_epi64(prodlh4, 32));

    prod
}

#[inline(always)]
pub unsafe fn _mm256_nextupps_v3(value: __m256) -> __m256 {
    let is_nan = _mm256_castps_si256(_mm256_cmp_ps(value, value, _CMP_NEQ_OQ));

    let bits = _mm256_castps_si256(value); // switching to integer ops may add latency here
    let abs = _mm256_andnot_si256(_mm256_set1_epu32x(0x8000_0000), bits);

    let is_infinity = _mm256_cmpeq_epi32(bits, _mm256_set1_epu32x(0x7F80_0000));

    let unchanged = _mm256_or_si256(is_nan, is_infinity);

    let is_positive = _mm256_cmpeq_epi32(abs, bits);
    let is_zero = _mm256_cmpeq_epi32(abs, _mm256_setzero_si256());

    let add = _mm256_add_epi32(bits, _mm256_set1_epi32(1));
    let sub = _mm256_sub_epi32(bits, _mm256_set1_epi32(1));

    // if(positive) { bits + 1 } else { bits - 1 }
    let next_bits = _mm256_blendv_epi8(sub, add, is_positive);

    // if(is_zero) { 0x1 } else { next_bits }
    let next_bits = _mm256_blendv_epi8(next_bits, _mm256_set1_epu32x(0x1), is_zero);

    _mm256_castsi256_ps(_mm256_blendv_epi8(next_bits, bits, unchanged))
}

#[inline(always)]
pub unsafe fn _mm256_nextdownps_v3(value: __m256) -> __m256 {
    let is_nan = _mm256_castps_si256(_mm256_cmp_ps(value, value, _CMP_NEQ_OQ));

    let bits = _mm256_castps_si256(value); // switching to integer ops may add latency here
    let abs = _mm256_andnot_si256(_mm256_set1_epu32x(0x8000_0000u32), bits);

    let is_neg_infinity = _mm256_cmpeq_epi32(bits, _mm256_set1_epu32x(0xFF80_0000));
    let unchanged = _mm256_or_si256(is_nan, is_neg_infinity);

    let is_positive = _mm256_cmpeq_epi32(abs, bits);
    let is_zero = _mm256_cmpeq_epi32(abs, _mm256_setzero_si256());

    let add = _mm256_add_epi32(bits, _mm256_set1_epi32(1));
    let sub = _mm256_sub_epi32(bits, _mm256_set1_epi32(1));

    // if(positive) { bits - 1 } else { bits + 1 }
    let next_bits = _mm256_blendv_epi8(add, sub, is_positive);
    // if(is_zero) { 0x1 | sign_bit } else { next_bits }
    let next_bits = _mm256_blendv_epi8(next_bits, _mm256_set1_epu32x(0x1 | 0x8000_0000), is_zero);

    _mm256_castsi256_ps(_mm256_blendv_epi8(next_bits, bits, unchanged))
}

#[inline(always)]
pub unsafe fn _mm256_nextuppd_v3(value: __m256d) -> __m256d {
    let is_nan = _mm256_castpd_si256(_mm256_cmp_pd(value, value, _CMP_NEQ_OQ));

    let bits = _mm256_castpd_si256(value); // switching to integer ops may add latency here
    let abs = _mm256_andnot_si256(_mm256_set1_epu64x(0x8000_0000_0000_0000), bits);

    let is_infinity = _mm256_cmpeq_epi64(bits, _mm256_set1_epu64x(0x7FF0_0000_0000_0000));
    let is_positive = _mm256_cmpeq_epi64(abs, bits);
    let is_zero = _mm256_cmpeq_epi64(abs, _mm256_setzero_si256());

    let add = _mm256_add_epi64(bits, _mm256_set1_epu64x(1));
    let sub = _mm256_sub_epi64(bits, _mm256_set1_epu64x(1));

    // if(positive) { bits + 1 } else { bits - 1 }
    let next_bits = _mm256_blendv_epi8(sub, add, is_positive);

    // if(is_zero) { 0x1 } else { next_bits }
    let next_bits = _mm256_blendv_epi8(next_bits, _mm256_set1_epu64x(0x1), is_zero);

    _mm256_castsi256_pd(next_bits)
}

#[inline(always)]
pub unsafe fn _mm256_nextdownpd_v3(value: __m256d) -> __m256d {
    let is_nan = _mm256_castpd_si256(_mm256_cmp_pd(value, value, _CMP_NEQ_OQ));

    let bits = _mm256_castpd_si256(value); // switching to integer ops may add latency here
    let abs = _mm256_andnot_si256(_mm256_set1_epu64x(0x8000_0000_0000_0000), bits);

    let is_neg_infinity = _mm256_cmpeq_epi64(bits, _mm256_set1_epu64x(0xFFF0_0000_0000_0000));
    let unchanged = _mm256_or_si256(is_nan, is_neg_infinity);

    let is_positive = _mm256_cmpeq_epi64(abs, bits);
    let is_zero = _mm256_cmpeq_epi64(abs, _mm256_setzero_si256());

    let add = _mm256_add_epi64(bits, _mm256_set1_epu64x(1));
    let sub = _mm256_sub_epi64(bits, _mm256_set1_epu64x(1));

    // if(positive) { bits - 1 } else { bits + 1 }
    let next_bits = _mm256_blendv_epi8(add, sub, is_positive);
    // if(is_zero) { 0x1 | sign_bit } else { next_bits }
    let next_bits = _mm256_blendv_epi8(next_bits, _mm256_set1_epu64x(0x1 | 0x8000_0000_0000_0000), is_zero);

    _mm256_castsi256_pd(_mm256_blendv_epi8(next_bits, bits, unchanged))
}

#[inline(always)]
pub unsafe fn _mm256_copysign_epi64x_v3(lhs: __m256i, rhs: __m256i) -> __m256i {
    let change_sign = _mm256_xor_si256(
        _mm256_cmpgt_epi64(rhs, _mm256_set1_epi64x(-1)), // rhs > -1 = rhs >= 0
        _mm256_cmpgt_epi64(lhs, _mm256_set1_epi64x(-1)), // lhs > -1 = lhs >= 0
    );

    _mm256_add_epi64(
        _mm256_xor_si256(lhs, change_sign), // invert lhs if change_sign is true
        _mm256_srli_epi64(change_sign, 63), // 1 if true, 0 if false, to correct for two's complement
    )
}
