use super::*;

#[inline(always)]
pub unsafe fn _mm_popcnt_epi8x_v2(v: __m128i) -> __m128i {
    let lookup = _mm_setr_epi8(0, 1, 1, 2, 1, 2, 2, 3, 1, 2, 2, 3, 2, 3, 3, 4);
    let low_mask = _mm_set1_epi8(0x0f);
    let lo = _mm_and_si128(v, low_mask);
    let hi = _mm_and_si128(_mm_srli_epi16(v, 4), low_mask);
    let cnt1 = _mm_shuffle_epi8(lookup, lo);
    let cnt2 = _mm_shuffle_epi8(lookup, hi);
    _mm_add_epi8(cnt1, cnt2)
}

#[inline(always)]
pub unsafe fn _mm_popcnt_epi32x_v2(v: __m128i) -> __m128i {
    // https://stackoverflow.com/a/51106873/2083075
    _mm_madd_epi16(
        _mm_maddubs_epi16(_mm_popcnt_epi8x_v2(v), _mm_set1_epi8(1)),
        _mm_set1_epi16(1),
    )
}

#[inline(always)]
pub unsafe fn _mm_popcnt_epi64x_v2(v: __m128i) -> __m128i {
    _mm_sad_epu8(_mm_popcnt_epi8x_v2(v), _mm_setzero_si128())
}

#[inline(always)]
pub unsafe fn _mm_reverse_bits_epi32x_v2(mut value: __m128i) -> __m128i {
    let mask_a = _mm_set1_epi32(0x55555555);
    let mask_b = _mm_set1_epi32(0x33333333);
    let mask_c = _mm_set1_epi32(0x0F0F0F0F);
    let mask_d = _mm_set1_epi32(0x00FF00FF);

    value = _mm_or_si128(
        _mm_and_si128(mask_a, _mm_srli_epi32(value, 1)),
        _mm_slli_epi32(_mm_and_si128(value, mask_a), 1),
    );

    value = _mm_or_si128(
        _mm_and_si128(mask_b, _mm_srli_epi32(value, 2)),
        _mm_slli_epi32(_mm_and_si128(value, mask_b), 2),
    );

    value = _mm_or_si128(
        _mm_and_si128(mask_c, _mm_srli_epi32(value, 4)),
        _mm_slli_epi32(_mm_and_si128(value, mask_c), 4),
    );

    value = _mm_or_si128(
        _mm_and_si128(mask_d, _mm_srli_epi32(value, 8)),
        _mm_slli_epi32(_mm_and_si128(value, mask_d), 8),
    );

    value = _mm_or_si128(_mm_srli_epi32(value, 16), _mm_slli_epi32(value, 16));

    value
}

#[inline(always)]
pub unsafe fn _mm_reverse_bits_epi64x_v2(mut value: __m128i) -> __m128i {
    let mut s = 64;
    let mut mask = !0i64;

    loop {
        s >>= 1;

        if s == 0 {
            return value;
        }

        mask ^= mask << s;

        let s = _mm_set_epi32(0, 0, 0, s);

        let left = _mm_and_si128(_mm_srl_epi64(value, s), _mm_set1_epi64x(mask));
        let right = _mm_and_si128(_mm_sll_epi64(value, s), _mm_set1_epi64x(!mask));

        value = _mm_or_si128(left, right);
    }
}

#[inline(always)]
pub unsafe fn _mm_np2_m1_epu32x_v2(mut value: __m128i) -> __m128i {
    let mut s = 1;

    while s < 32 {
        value = _mm_or_si128(value, _mm_srl_epi32(value, _mm_set_epi32(0, 0, 0, s)));

        s <<= 1;
    }

    value
}

#[inline(always)]
pub unsafe fn _mm_np2_m1_epu64x_v2(mut value: __m128i) -> __m128i {
    let mut s = 1;

    while s < 64 {
        value = _mm_or_si128(value, _mm_srl_epi64(value, _mm_set_epi32(0, 0, 0, s)));

        s <<= 1;
    }

    value
}

// This kind of sucked, but I'll leave it here for later.
//
// #[inline(always)]
// pub unsafe fn _mm_log2_epu32x_v2(mut v: __m128i) -> __m128i {
//     #[inline(always)]
//     unsafe fn masked_gt(lhs: __m128i, rhs: __m128i) -> __m128i {
//         _mm_and_si128(_mm_cmpgt_epi32(lhs, rhs), _mm_set1_epi32(1))
//     }

//     let mut r;

//     let mut shift = _mm_slli_epi32::<4>(masked_gt(v, _mm_set1_epi32(0xFFFF)));
//     v = _mm_srlv_epi32(v, shift);
//     r = shift; // r = _mm_or_si128(r, shift); where r is 0 here

//     shift = _mm_slli_epi32::<3>(masked_gt(v, _mm_set1_epi32(0xFF)));
//     v = _mm_srlv_epi32(v, shift);
//     r = _mm_or_si128(r, shift);

//     shift = _mm_slli_epi32::<2>(masked_gt(v, _mm_set1_epi32(0xF)));
//     v = _mm_srlv_epi32(v, shift);
//     r = _mm_or_si128(r, shift);

//     shift = _mm_slli_epi32::<1>(masked_gt(v, _mm_set1_epi32(0x3)));
//     v = _mm_srlv_epi32(v, shift);
//     r = _mm_or_si128(r, shift);

//     r = _mm_or_si128(r, _mm_srli_epi32(v, 1));

//     r
// }
