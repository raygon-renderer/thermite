use super::*;

// https://arxiv.org/pdf/1611.07612.pdf
#[inline(always)] #[rustfmt::skip]
pub unsafe fn _mm256_popcnt_epi8x_v3(v: __m256i) -> __m256i {
    let lookup = _mm256_setr_epi8(
        0, 1, 1, 2, 1, 2, 2, 3, 1, 2, 2, 3, 2, 3, 3, 4,
        0, 1, 1, 2, 1, 2, 2, 3, 1, 2, 2, 3, 2, 3, 3, 4,
    );
    let low_mask = _mm256_set1_epi8(0x0f);
    let lo = _mm256_and_si256(v, low_mask);
    let hi = _mm256_and_si256(_mm256_srli_epi32(v, 4), low_mask);
    let popcnt1 = _mm256_shuffle_epi8(lookup, lo);
    let popcnt2 = _mm256_shuffle_epi8(lookup, hi);
    _mm256_add_epi8(popcnt1, popcnt2)
}

#[inline(always)]
pub unsafe fn _mm256_popcnt_epi64x_v3(v: __m256i) -> __m256i {
    _mm256_sad_epu8(_mm256_popcnt_epi8x_v3(v), _mm256_setzero_si256())
}

#[inline(always)]
pub unsafe fn _mm256_popcnt_epi32x_v3(v: __m256i) -> __m256i {
    // https://stackoverflow.com/a/51106873/2083075
    _mm256_madd_epi16(
        _mm256_maddubs_epi16(_mm256_popcnt_epi8x_v3(v), _mm256_set1_epi8(1)),
        _mm256_set1_epi16(1),
    )
}

#[inline(always)]
pub unsafe fn _mm256_reverse_bits_epi32x_v3(mut value: __m256i) -> __m256i {
    let mask_a = _mm256_set1_epi32(0x55555555);
    let mask_b = _mm256_set1_epi32(0x33333333);
    let mask_c = _mm256_set1_epi32(0x0F0F0F0F);
    let mask_d = _mm256_set1_epi32(0x00FF00FF);

    value = _mm256_or_si256(
        _mm256_and_si256(mask_a, _mm256_srli_epi32(value, 1)),
        _mm256_slli_epi32(_mm256_and_si256(value, mask_a), 1),
    );

    value = _mm256_or_si256(
        _mm256_and_si256(mask_b, _mm256_srli_epi32(value, 2)),
        _mm256_slli_epi32(_mm256_and_si256(value, mask_b), 2),
    );

    value = _mm256_or_si256(
        _mm256_and_si256(mask_c, _mm256_srli_epi32(value, 4)),
        _mm256_slli_epi32(_mm256_and_si256(value, mask_c), 4),
    );

    value = _mm256_or_si256(
        _mm256_and_si256(mask_d, _mm256_srli_epi32(value, 8)),
        _mm256_slli_epi32(_mm256_and_si256(value, mask_d), 8),
    );

    value = _mm256_or_si256(_mm256_srli_epi32(value, 16), _mm256_slli_epi32(value, 16));

    value
}

#[inline(always)]
pub unsafe fn _mm256_reverse_bits_epi64x_v3(mut value: __m256i) -> __m256i {
    let mut s = 64;
    let mut mask = !0i64;

    loop {
        s >>= 1;

        if s == 0 {
            return value;
        }

        mask ^= mask << s;

        let s = _mm_cvtsi32_si128(s);

        let left = _mm256_and_si256(_mm256_srl_epi64(value, s), _mm256_set1_epi64x(mask));
        let right = _mm256_and_si256(_mm256_sll_epi64(value, s), _mm256_set1_epi64x(!mask));

        value = _mm256_or_si256(left, right);
    }
}

#[inline(always)]
pub unsafe fn _mm256_np2_m1_epu32x_v3(mut value: __m256i) -> __m256i {
    let mut s = 1;

    while s < 32 {
        value = _mm256_or_si256(value, _mm256_srl_epi32(value, _mm_cvtsi32_si128(s)));

        s <<= 1;
    }

    value
}

#[inline(always)]
pub unsafe fn _mm256_np2_m1_epu64x_v3(mut value: __m256i) -> __m256i {
    let mut s = 1;

    while s < 64 {
        value = _mm256_or_si256(value, _mm256_srl_epi64(value, _mm_cvtsi32_si128(s)));

        s <<= 1;
    }

    value
}

#[inline(always)]
pub unsafe fn _mm_rolv_epi32x_v3(value: __m128i, shifts: __m128i) -> __m128i {
    let inv_shifts = _mm_sub_epi32(_mm_set1_epi32(32), shifts);
    _mm_or_si128(_mm_sllv_epi32(value, shifts), _mm_srlv_epi32(value, inv_shifts))
}

#[inline(always)]
pub unsafe fn _mm_rolv_epi64x_v3(value: __m128i, shifts: __m128i) -> __m128i {
    let inv_shifts = _mm_sub_epi64(_mm_set1_epi64x(64), shifts);
    _mm_or_si128(_mm_sllv_epi64(value, shifts), _mm_srlv_epi64(value, inv_shifts))
}

#[inline(always)]
pub unsafe fn _mm_rorv_epi32x_v3(value: __m128i, shifts: __m128i) -> __m128i {
    let inv_shifts = _mm_sub_epi32(_mm_set1_epi32(32), shifts);
    _mm_or_si128(_mm_srlv_epi32(value, shifts), _mm_sllv_epi32(value, inv_shifts))
}

#[inline(always)]
pub unsafe fn _mm_rorv_epi64x_v3(value: __m128i, shifts: __m128i) -> __m128i {
    let inv_shifts = _mm_sub_epi64(_mm_set1_epi64x(64), shifts);
    _mm_or_si128(_mm_srlv_epi64(value, shifts), _mm_sllv_epi64(value, inv_shifts))
}

#[inline(always)]
pub unsafe fn _mm256_rolv_epi32x_v3(value: __m256i, shifts: __m256i) -> __m256i {
    let inv_shifts = _mm256_sub_epi32(_mm256_set1_epi32(32), shifts);
    _mm256_or_si256(_mm256_sllv_epi32(value, shifts), _mm256_srlv_epi32(value, inv_shifts))
}

#[inline(always)]
pub unsafe fn _mm256_rolv_epi64x_v3(value: __m256i, shifts: __m256i) -> __m256i {
    let inv_shifts = _mm256_sub_epi64(_mm256_set1_epi64x(64), shifts);
    _mm256_or_si256(_mm256_sllv_epi64(value, shifts), _mm256_srlv_epi64(value, inv_shifts))
}

#[inline(always)]
pub unsafe fn _mm256_rorv_epi32x_v3(value: __m256i, shifts: __m256i) -> __m256i {
    let inv_shifts = _mm256_sub_epi32(_mm256_set1_epi32(32), shifts);
    _mm256_or_si256(_mm256_srlv_epi32(value, shifts), _mm256_sllv_epi32(value, inv_shifts))
}

#[inline(always)]
pub unsafe fn _mm256_rorv_epi64x_v3(value: __m256i, shifts: __m256i) -> __m256i {
    let inv_shifts = _mm256_sub_epi64(_mm256_set1_epi64x(64), shifts);
    _mm256_or_si256(_mm256_srlv_epi64(value, shifts), _mm256_sllv_epi64(value, inv_shifts))
}

#[inline(always)]
pub unsafe fn _mm256_bswap_epi32x_v3(x: __m256i) -> __m256i {
    // Note: vpshufb works within 128-bit lanes.
    // We repeat the 128-bit mask for both lanes.
    let mask = _mm256_setr_epi8(
        3, 2, 1, 0, 7, 6, 5, 4, 11, 10, 9, 8, 15, 14, 13, 12, // Lane 1
        3, 2, 1, 0, 7, 6, 5, 4, 11, 10, 9, 8, 15, 14, 13, 12, // Lane 2
    );
    _mm256_shuffle_epi8(x, mask)
}

#[inline(always)]
pub unsafe fn _mm256_bswap_epi64x_v3(x: __m256i) -> __m256i {
    let mask = _mm256_setr_epi8(
        7, 6, 5, 4, 3, 2, 1, 0, 15, 14, 13, 12, 11, 10, 9, 8, // Lane 1
        7, 6, 5, 4, 3, 2, 1, 0, 15, 14, 13, 12, 11, 10, 9, 8, // Lane 2
    );
    _mm256_shuffle_epi8(x, mask)
}

#[inline(always)]
pub unsafe fn _mm256_bswap_psx_v3(x: __m256) -> __m256 {
    _mm256_castsi256_ps(_mm256_bswap_epi32x_v3(_mm256_castps_si256(x)))
}

#[inline(always)]
pub unsafe fn _mm256_bswap_pdx_v3(x: __m256d) -> __m256d {
    _mm256_castsi256_pd(_mm256_bswap_epi64x_v3(_mm256_castpd_si256(x)))
}
