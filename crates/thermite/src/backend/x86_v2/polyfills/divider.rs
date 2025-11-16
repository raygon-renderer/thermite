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

#[inline(always)]
pub unsafe fn _mm_div_epi32x_v2(numers: __m128i, multiplier: i32, shift: u8) -> __m128i {
    const SHIFT_MASK: u8 = crate::divider::Divider::<u32>::SHIFT_MASK;

    if multiplier == 0 {
        let masked_shift = shift & SHIFT_MASK;
        let mask = (1 << masked_shift) - 1;

        let round_to_zero_tweak = _mm_set1_epi32(mask);

        // q = numer + ((numer >> 31) & round_to_zero_tweak);
        let mut q = _mm_add_epi32(numers, _mm_and_si128(_mm_srai_epi32(numers, 31), round_to_zero_tweak));
        q = _mm_sra_epi32(q, _mm_cvtsi32_si128(masked_shift as i32));

        let sign = _mm_set1_epi32(((shift as i8) >> 7) as i32);

        // q = (q ^ sign) - sign;
        _mm_sub_epi32(_mm_xor_si128(q, sign), sign)
    } else {
        let mut q = _mm_mullhi_epi32x_v2(numers, _mm_set1_epi32(multiplier));

        if shift & crate::divider::ADD_MARKER != 0 {
            // must be arithmetic shift
            let sign = _mm_set1_epi32(((shift as i8) >> 7) as i32);
            // q += ((numer ^ sign) - sign);
            q = _mm_add_epi32(q, _mm_sub_epi32(_mm_xor_si128(numers, sign), sign));
        }

        // q >>= shift
        q = _mm_sra_epi32(q, _mm_cvtsi32_si128((shift & SHIFT_MASK) as i32));
        q = _mm_add_epi32(q, _mm_srli_epi32(q, 31)); // q += (q < 0)

        q
    }
}

#[inline(always)]
pub unsafe fn _mm_div_epi32x_bf_v2(numers: __m128i, multiplier: i32, shift: u8) -> __m128i {
    const SHIFT_MASK: u8 = crate::divider::Divider::<u32>::SHIFT_MASK;

    let masked_shift = shift & SHIFT_MASK;

    // must be arithmetic shift
    let sign = _mm_set1_epi32(((shift as i8) >> 7) as i32);

    let mut q = _mm_mullhi_epi32x_v2(numers, _mm_set1_epi32(multiplier));
    q = _mm_add_epi32(q, numers); // q += numers

    // If q is non-negative, we have nothing to do
    // If q is negative, we want to add either (2**shift)-1 if d is
    // a power of 2, or (2**shift) if it is not a power of 2
    let is_power_of_2 = (multiplier == 0) as u32;

    let q_sign = _mm_srai_epi32(q, 31); // q_sign = q >> 31
    let mask = _mm_set1_epi32((1u32 << masked_shift).wrapping_sub(is_power_of_2) as i32);

    q = _mm_add_epi32(q, _mm_and_si128(q_sign, mask)); // q = q + (q_sign & mask)
    q = _mm_sra_epi32(q, _mm_cvtsi32_si128(masked_shift as i32)); // q >>= shift
    q = _mm_sub_epi32(_mm_xor_si128(q, sign), sign); // q = (q ^ sign) - sign

    q
}

#[inline(always)]
pub unsafe fn _mm_divv_epi32x_bf_v2(numers: __m128i, multipliers: __m128i, shifts: __m128i) -> __m128i {
    const SHIFT_MASK: u8 = crate::divider::Divider::<u32>::SHIFT_MASK;

    let masked_shift = _mm_and_si128(shifts, _mm_set1_epi32(SHIFT_MASK as i32));

    let sign = _mm_srai_epi32(shifts, 31); // must be arithmetic shift
    let mut q = _mm_mullhi_epi32x_v2(numers, multipliers);
    q = _mm_add_epi32(q, numers); // q += numers

    // If q is non-negative, we have nothing to do
    // If q is negative, we want to add either (2**shift)-1 if d is
    // a power of 2, or (2**shift) if it is not a power of 2
    let is_power_of_2 = _mm_cmpeq_epi32(multipliers, _mm_setzero_si128());

    let q_sign = _mm_srai_epi32(q, 31); // q_sign = q >> 31
    let mask = _mm_sub_epi32(_mm_sllv_epi32x_v1(_mm_set1_epi32(1), masked_shift), is_power_of_2);
    q = _mm_add_epi32(q, _mm_and_si128(q_sign, mask)); // q = q + (q_sign & mask)
    q = _mm_srav_epi32x_v1(q, masked_shift); // q >>= shift
    q = _mm_sub_epi32(_mm_xor_si128(q, sign), sign); // q = (q ^ sign) - sign

    q
}

#[inline(always)]
pub unsafe fn _mm_divv_epi64x_bf_v2(numers: __m128i, multipliers: __m128i, shifts: __m128i) -> __m128i {
    const SHIFT_MASK: u8 = crate::divider::Divider::<u64>::SHIFT_MASK;

    let masked_shift = _mm_and_si128(shifts, _mm_set1_epi64x(SHIFT_MASK as i64));

    let sign = _mm_srai_epi64x_v1(shifts, 63); // must be arithmetic shift
    let mut q = _mm_mullhi_epi64x_v1(numers, multipliers);
    q = _mm_add_epi64(q, numers); // q += numers

    // If q is non-negative, we have nothing to do.
    // If q is negative, we want to add either (2**shift)-1 if d is
    // a power of 2, or (2**shift) if it is not a power of 2.
    let is_power_of_2 = _mm_cmpeq_epi64(multipliers, _mm_setzero_si128());

    let q_sign = _mm_srai_epi64x_v1(q, 63); // q_sign = q >> 63
    let mask = _mm_sub_epi64(_mm_sllv_epi64x_v1(_mm_set1_epi64x(1), masked_shift), is_power_of_2);
    q = _mm_add_epi64(q, _mm_and_si128(q_sign, mask)); // q = q + (q_sign & mask)
    q = _mm_srav_epi64x_v1(q, masked_shift); // q >>= shift
    q = _mm_sub_epi64(_mm_xor_si128(q, sign), sign); // q = (q ^ sign) - sign

    q
}
