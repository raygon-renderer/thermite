use super::*;

/// POLYFILL: Shift right and sign extend 64-bit integers
#[inline(always)]
pub unsafe fn _mm_srai_epi64x_v1(v: __m128i, cnt: i32) -> __m128i {
    let m = _mm_set1_epi64x(1i64 << (63 - cnt));
    _mm_sub_epi64(_mm_xor_si128(_mm_srl_epi64(v, _mm_cvtsi32_si128(cnt)), m), m)
}

/// POLYFILL: Shift right 64-bit integers (variable)
///
/// <https://stackoverflow.com/a/38608465/2083075>
#[inline(always)]
pub unsafe fn _mm_srlv_epi64x_v1(value: __m128i, shifts: __m128i) -> __m128i {
    let count_high = _mm_unpackhi_epi64(shifts, shifts); // move higher 64 bits to lower 64 bits

    let shifted_low = _mm_srl_epi64(value, shifts); // uses lower 64 bits of shifts
    let mut shifted_high = _mm_srl_epi64(value, count_high); // shift value by higher 64 bits (now in lower 64 bits)

    shifted_high = _mm_unpackhi_epi64(shifted_high, shifted_high); // move result to higher 64 bits

    _mm_unpacklo_epi64(shifted_high, shifted_low) // combine results
}

#[target_feature(enable = "sse2")] // LLVM can probably auto-vectorize this to some degree
#[inline]
pub unsafe fn _mm_srlv_epi32x_v1(value: __m128i, shifts: __m128i) -> __m128i {
    let shifts: [u32; 4] = core::mem::transmute(shifts);
    let mut value: [u32; 4] = core::mem::transmute(value);

    for (value, shift) in value.iter_mut().zip(shifts) {
        *value >>= shift;
    }

    core::mem::transmute(value)
}

#[inline(always)]
pub unsafe fn _mm_sllv_epi64x_v1(value: __m128i, shifts: __m128i) -> __m128i {
    let count_high = _mm_unpackhi_epi64(shifts, shifts); // move higher 64 bits to lower 64 bits

    let shifted_low = _mm_sll_epi64(value, shifts); // uses lower 64 bits of shifts
    let mut shifted_high = _mm_sll_epi64(value, count_high); // shift value by higher 64 bits (now in lower 64 bits)

    shifted_high = _mm_unpackhi_epi64(shifted_high, shifted_high); // move result to higher 64 bits

    _mm_unpacklo_epi64(shifted_high, shifted_low) // combine results
}

#[target_feature(enable = "sse2")]
#[inline]
pub unsafe fn _mm_sllv_epi32x_v1(value: __m128i, shifts: __m128i) -> __m128i {
    let shifts: [u32; 4] = core::mem::transmute(shifts);
    let mut value: [u32; 4] = core::mem::transmute(value);

    for (value, shift) in value.iter_mut().zip(shifts) {
        *value <<= shift;
    }

    core::mem::transmute(value)
}

/// POLYFILL: Shift right and sign extend 64-bit integers (variable)
#[inline(always)]
pub unsafe fn _mm_srav_epi64x_v1(value: __m128i, shifts: __m128i) -> __m128i {
    let m = _mm_srlv_epi64x_v1(_mm_set1_epu64x(1 << 63), shifts);
    _mm_sub_epi64(_mm_xor_si128(_mm_srlv_epi64x_v1(value, shifts), m), m)
}

// This would have been like the 64-bit version, but for 32-bit integers
// it's simpler to just do it scalar-wise instead of trying to be clever
#[target_feature(enable = "sse2")]
#[inline]
pub unsafe fn _mm_srav_epi32x_v1(value: __m128i, shifts: __m128i) -> __m128i {
    let shifts: [u32; 4] = core::mem::transmute(shifts);
    let mut value: [i32; 4] = core::mem::transmute(value);

    for (value, shift) in value.iter_mut().zip(shifts) {
        *value >>= shift;
    }

    core::mem::transmute(value)
}

#[inline(always)]
pub unsafe fn _mm_rolv_epi32x_v1(value: __m128i, shifts: __m128i) -> __m128i {
    let inv_shifts = _mm_sub_epi32(_mm_set1_epi32(32), shifts);
    _mm_or_si128(_mm_sllv_epi32x_v1(value, shifts), _mm_srlv_epi32x_v1(value, inv_shifts))
}

#[inline(always)]
pub unsafe fn _mm_rolv_epi64x_v1(value: __m128i, shifts: __m128i) -> __m128i {
    let inv_shifts = _mm_sub_epi64(_mm_set1_epi64x(64), shifts);
    _mm_or_si128(_mm_sllv_epi64x_v1(value, shifts), _mm_srlv_epi64x_v1(value, inv_shifts))
}

#[inline(always)]
pub unsafe fn _mm_rorv_epi32x_v1(value: __m128i, shifts: __m128i) -> __m128i {
    let inv_shifts = _mm_sub_epi32(_mm_set1_epi32(32), shifts);
    _mm_or_si128(_mm_srlv_epi32x_v1(value, shifts), _mm_sllv_epi32x_v1(value, inv_shifts))
}

#[inline(always)]
pub unsafe fn _mm_rorv_epi64x_v1(value: __m128i, shifts: __m128i) -> __m128i {
    let inv_shifts = _mm_sub_epi64(_mm_set1_epi64x(64), shifts);
    _mm_or_si128(_mm_srlv_epi64x_v1(value, shifts), _mm_sllv_epi64x_v1(value, inv_shifts))
}

#[inline(always)]
pub unsafe fn _mm_reverse_bits_epi32x_v1(mut value: __m128i) -> __m128i {
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
pub unsafe fn _mm_reverse_bits_epi64x_v1(mut value: __m128i) -> __m128i {
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
pub unsafe fn _mm_np2_m1_epu32x_v1(mut value: __m128i) -> __m128i {
    let mut s = 1;

    while s < 32 {
        value = _mm_or_si128(value, _mm_srl_epi32(value, _mm_set_epi32(0, 0, 0, s)));

        s <<= 1;
    }

    value
}

#[inline(always)]
pub unsafe fn _mm_np2_m1_epu64x_v1(mut value: __m128i) -> __m128i {
    let mut s = 1;

    while s < 64 {
        value = _mm_or_si128(value, _mm_srl_epi64(value, _mm_set_epi32(0, 0, 0, s)));

        s <<= 1;
    }

    value
}
