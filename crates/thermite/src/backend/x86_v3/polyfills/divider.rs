use super::*;

// libdivide.h - Optimized integer division
// https://libdivide.com
//
// Copyright (C) 2010 - 2019 ridiculous_fish, <libdivide@ridiculousfish.com>
// Copyright (C) 2016 - 2019 Kim Walisch, <kim.walisch@gmail.com>

#[inline(always)]
pub unsafe fn _mm256_mullhi_epu64x_v3(x: __m256i, y: __m256i) -> __m256i {
    let lomask = _mm256_set1_epi64x(0xffffffff);
    let xh = _mm256_shuffle_epi32(x, 0xB1); // x0l, x0h, x1l, x1h
    let yh = _mm256_shuffle_epi32(y, 0xB1); // y0l, y0h, y1l, y1h
    let w0 = _mm256_mul_epu32(x, y); // x0l*y0l, x1l*y1l
    let w1 = _mm256_mul_epu32(x, yh); // x0l*y0h, x1l*y1h
    let w2 = _mm256_mul_epu32(xh, y); // x0h*y0l, x1h*y0l
    let w3 = _mm256_mul_epu32(xh, yh); // x0h*y0h, x1h*y1h
    let w0h = _mm256_srli_epi64(w0, 32);
    let s1 = _mm256_add_epi64(w1, w0h);
    let s1l = _mm256_and_si256(s1, lomask);
    let s1h = _mm256_srli_epi64(s1, 32);
    let s2 = _mm256_add_epi64(w2, s1l);
    let s2h = _mm256_srli_epi64(s2, 32);
    let mut hi = _mm256_add_epi64(w3, s1h);

    hi = _mm256_add_epi64(hi, s2h);

    hi
}

#[inline(always)]
pub unsafe fn _mm256_mullhi_epu32x_v3(a: __m256i, b: __m256i) -> __m256i {
    let hi_product_0_z2_z = _mm256_srli_epi64(_mm256_mul_epu32(a, b), 32);
    let a1_x3_x = _mm256_srli_epi64(a, 32);
    let mask = _mm256_set_epi32(-1, 0, -1, 0, -1, 0, -1, 0);
    let hi_product_z1_z3 = _mm256_and_si256(_mm256_mul_epu32(a1_x3_x, b), mask);
    _mm256_or_si256(hi_product_0_z2_z, hi_product_z1_z3)
}

#[inline(always)]
pub unsafe fn _mm256_mullhi_epi32x_v3(a: __m256i, b: __m256i) -> __m256i {
    let hi_product_0_z2_z = _mm256_srli_epi64(_mm256_mul_epi32(a, b), 32);
    let a1_x3_x = _mm256_srli_epi64(a, 32);
    let mask = _mm256_set_epi32(-1, 0, -1, 0, -1, 0, -1, 0);
    let hi_product_z1_z3 = _mm256_and_si256(_mm256_mul_epi32(a1_x3_x, b), mask);
    _mm256_or_si256(hi_product_0_z2_z, hi_product_z1_z3)
}

#[inline(always)]
pub unsafe fn _mm256_mullhi_epi64x_v3(x: __m256i, y: __m256i) -> __m256i {
    let p = _mm256_mullhi_epu64x_v3(x, y);
    let t1 = _mm256_and_si256(_mm256_signbits_epi64x_v3(x), y);
    let t2 = _mm256_and_si256(_mm256_signbits_epi64x_v3(y), x);
    _mm256_sub_epi64(_mm256_sub_epi64(p, t1), t2)
}

#[inline(always)]
pub unsafe fn _mm256_div_epu32x_v3(numers: __m256i, multiplier: u32, shift: u8) -> __m256i {
    if multiplier == 0 {
        return _mm256_srl_epi32(numers, _mm_cvtsi32_si128(shift as i32));
    }

    let q = _mm256_mullhi_epu32x_v3(numers, _mm256_set1_epi32(multiplier as i32));

    if shift & 0x40 != 0 {
        _mm256_srl_epi32(
            _mm256_add_epi32(_mm256_srli_epi32(_mm256_sub_epi32(numers, q), 1), q),
            _mm_cvtsi32_si128((shift & 0x1F) as i32),
        )
    } else {
        _mm256_srl_epi32(q, _mm_cvtsi32_si128(shift as i32))
    }
}

#[inline(always)]
pub unsafe fn _mm256_div_epu32x_bf_v3(numers: __m256i, multiplier: u32, shift: u8) -> __m256i {
    let q = _mm256_mullhi_epu32x_v3(numers, _mm256_set1_epi32(multiplier as i32));
    _mm256_srl_epi32(
        _mm256_add_epi32(_mm256_srli_epi32(_mm256_sub_epi32(numers, q), 1), q),
        _mm_cvtsi32_si128(shift as i32),
    )
}

/// multipliers and shifts are vectors of u32, same length as numers
#[inline(always)]
pub unsafe fn _mm256_divv_epu32x_bf_v3(numers: __m256i, multipliers: __m256i, shifts: __m256i) -> __m256i {
    let q = _mm256_mullhi_epu32x_v3(numers, multipliers);

    _mm256_srlv_epi32(
        _mm256_add_epi32(_mm256_srli_epi32(_mm256_sub_epi32(numers, q), 1), q),
        shifts,
    )
}

#[inline(always)]
pub unsafe fn _mm256_div_epu64x_bf_v3(numers: __m256i, multiplier: u64, shift: u8) -> __m256i {
    let q = _mm256_mullhi_epu64x_v3(numers, _mm256_set1_epi64x(multiplier as i64));
    _mm256_srl_epi64(
        _mm256_add_epi64(_mm256_srli_epi64(_mm256_sub_epi64(numers, q), 1), q),
        _mm_cvtsi32_si128(shift as i32),
    )
}

/// multipliers and shifts are vectors of u64, same length as numers
#[inline(always)]
pub unsafe fn _mm256_divv_epu64x_bf_v3(numers: __m256i, multipliers: __m256i, shifts: __m256i) -> __m256i {
    let q = _mm256_mullhi_epu64x_v3(numers, multipliers);

    _mm256_srlv_epi64(
        _mm256_add_epi64(_mm256_srli_epi64(_mm256_sub_epi64(numers, q), 1), q),
        shifts,
    )
}

#[inline(always)]
pub unsafe fn _mm256_div_epu64x_v3(numers: __m256i, multiplier: u64, shift: u8) -> __m256i {
    if multiplier == 0 {
        return _mm256_srl_epi64(numers, _mm_cvtsi32_si128(shift as i32));
    }

    let q = _mm256_mullhi_epu64x_v3(numers, _mm256_set1_epi64x(multiplier as i64));

    if shift & 0x40 != 0 {
        _mm256_srl_epi64(
            _mm256_add_epi64(_mm256_srli_epi64(_mm256_sub_epi64(numers, q), 1), q),
            _mm_cvtsi32_si128((shift & 0x3F) as i32),
        )
    } else {
        _mm256_srl_epi64(q, _mm_cvtsi32_si128(shift as i32))
    }
}

#[inline(always)]
pub unsafe fn _mm256_div_epi32x_v3(numers: __m256i, multiplier: i32, shift: u8) -> __m256i {
    const SHIFT_MASK: u8 = crate::divider::Divider::<u32>::SHIFT_MASK;

    if multiplier == 0 {
        let masked_shift = shift & SHIFT_MASK;
        let mask = (1 << masked_shift) - 1;

        let round_to_zero_tweak = _mm256_set1_epi32(mask);

        // q = numer + ((numer >> 31) & round_to_zero_tweak);
        let mut q = _mm256_add_epi32(
            numers,
            _mm256_and_si256(_mm256_srai_epi32(numers, 31), round_to_zero_tweak),
        );
        q = _mm256_sra_epi32(q, _mm_cvtsi32_si128(masked_shift as i32));

        let sign = _mm256_set1_epi32(((shift as i8) >> 7) as i32);

        // q = (q ^ sign) - sign;
        _mm256_sub_epi32(_mm256_xor_si256(q, sign), sign)
    } else {
        let mut q = _mm256_mullhi_epi32x_v3(numers, _mm256_set1_epi32(multiplier));

        if shift & crate::divider::ADD_MARKER != 0 {
            // must be arithmetic shift
            let sign = _mm256_set1_epi32(((shift as i8) >> 7) as i32);
            // q += ((numer ^ sign) - sign);
            q = _mm256_add_epi32(q, _mm256_sub_epi32(_mm256_xor_si256(numers, sign), sign));
        }

        // q >>= shift
        q = _mm256_sra_epi32(q, _mm_cvtsi32_si128((shift & SHIFT_MASK) as i32));
        q = _mm256_add_epi32(q, _mm256_srli_epi32(q, 31)); // q += (q < 0)

        q
    }
}

#[inline(always)]
pub unsafe fn _mm256_div_epi64x_v3(numers: __m256i, multiplier: i64, shift: u8) -> __m256i {
    const SHIFT_MASK: u8 = crate::divider::Divider::<u64>::SHIFT_MASK;

    if multiplier == 0 {
        let masked_shift = shift & SHIFT_MASK;
        let mask = (1i64 << masked_shift) - 1;

        let round_to_zero_tweak = _mm256_set1_epi64x(mask);

        // q = numer + ((numer >> 63) & round_to_zero_tweak);
        let mut q = _mm256_add_epi64(
            numers,
            _mm256_and_si256(_mm256_signbits_epi64x_v3(numers), round_to_zero_tweak),
        );
        q = _mm256_srai_epi64x_v3(q, masked_shift as i32);

        let sign = _mm256_set1_epi32(((shift as i8) >> 7) as i32);

        // q = (q ^ sign) - sign;
        q = _mm256_sub_epi64(_mm256_xor_si256(q, sign), sign);

        q
    } else {
        let mut q = _mm256_mullhi_epi64x_v3(numers, _mm256_set1_epi64x(multiplier));

        if shift & crate::divider::ADD_MARKER != 0 {
            // must be arithmetic shift
            let sign = _mm256_set1_epi32(((shift as i8) >> 7) as i32);
            // q += ((numer ^ sign) - sign);
            q = _mm256_add_epi64(q, _mm256_sub_epi64(_mm256_xor_si256(numers, sign), sign));
        }

        // q >>= shift
        q = _mm256_srai_epi64x_v3(q, (shift & SHIFT_MASK) as i32);
        q = _mm256_add_epi64(q, _mm256_srli_epi64(q, 63)); // q += (q < 0)

        q
    }
}

#[inline(always)]
pub unsafe fn _mm256_div_epi32x_bf_v3(numers: __m256i, multiplier: i32, shift: u8) -> __m256i {
    const SHIFT_MASK: u8 = crate::divider::Divider::<u32>::SHIFT_MASK;

    let masked_shift = shift & SHIFT_MASK;

    // must be arithmetic shift
    let sign = _mm256_set1_epi32(((shift as i8) >> 7) as i32);
    let mut q = _mm256_mullhi_epi32x_v3(numers, _mm256_set1_epi32(multiplier));
    q = _mm256_add_epi32(q, numers); // q += numers

    // If q is non-negative, we have nothing to do
    // If q is negative, we want to add either (2**shift)-1 if d is
    // a power of 2, or (2**shift) if it is not a power of 2
    let is_power_of_2 = (multiplier == 0) as u32;

    let q_sign = _mm256_srai_epi32(q, 31); // q_sign = q >> 31
    let mask = _mm256_set1_epi32((1u32 << masked_shift).wrapping_sub(is_power_of_2) as i32);
    q = _mm256_add_epi32(q, _mm256_and_si256(q_sign, mask)); // q = q + (q_sign & mask)
    q = _mm256_sra_epi32(q, _mm_cvtsi32_si128(masked_shift as i32)); // q >>= shift
    q = _mm256_sub_epi32(_mm256_xor_si256(q, sign), sign); // q = (q ^ sign) - sign

    q
}

#[inline(always)]
pub unsafe fn _mm256_div_epi64x_bf_v3(numers: __m256i, multiplier: i64, shift: u8) -> __m256i {
    const SHIFT_MASK: u8 = crate::divider::Divider::<u64>::SHIFT_MASK;

    let masked_shift = shift & SHIFT_MASK;

    // must be arithmetic shift
    let sign = _mm256_set1_epi32(((shift as i8) >> 7) as i32);

    let mut q = _mm256_mullhi_epi64x_v3(numers, _mm256_set1_epi64x(multiplier));
    q = _mm256_add_epi64(q, numers); // q += numers

    // If q is non-negative, we have nothing to do.
    // If q is negative, we want to add either (2**shift)-1 if d is
    // a power of 2, or (2**shift) if it is not a power of 2.
    let is_power_of_2 = (multiplier == 0) as u64;

    let q_sign = _mm256_signbits_epi64x_v3(q); // q_sign = q >> 63
    let mask = _mm256_set1_epi64x((1u64 << masked_shift).wrapping_sub(is_power_of_2) as i64);
    q = _mm256_add_epi64(q, _mm256_and_si256(q_sign, mask)); // q = q + (q_sign & mask)
    q = _mm256_srai_epi64x_v3(q, masked_shift as i32); // q >>= shift
    q = _mm256_sub_epi64(_mm256_xor_si256(q, sign), sign); // q = (q ^ sign) - sign

    q
}

#[inline(always)]
pub unsafe fn _mm256_divv_epi32x_bf_v3(numers: __m256i, multipliers: __m256i, shifts: __m256i) -> __m256i {
    const SHIFT_MASK: u8 = crate::divider::Divider::<u32>::SHIFT_MASK;

    let masked_shift = _mm256_and_si256(shifts, _mm256_set1_epi32(SHIFT_MASK as i32));

    let sign = _mm256_srai_epi32(shifts, 31); // must be arithmetic shift
    let mut q = _mm256_mullhi_epi32x_v3(numers, multipliers);
    q = _mm256_add_epi32(q, numers); // q += numers

    // If q is non-negative, we have nothing to do
    // If q is negative, we want to add either (2**shift)-1 if d is
    // a power of 2, or (2**shift) if it is not a power of 2
    let is_power_of_2 = _mm256_cmpeq_epi32(multipliers, _mm256_setzero_si256());

    let q_sign = _mm256_srai_epi32(q, 31); // q_sign = q >> 31
    let mask = _mm256_sub_epi32(_mm256_sllv_epi32(_mm256_set1_epi32(1), masked_shift), is_power_of_2);
    q = _mm256_add_epi32(q, _mm256_and_si256(q_sign, mask)); // q = q + (q_sign & mask)
    q = _mm256_srav_epi32(q, masked_shift); // q >>= shift
    q = _mm256_sub_epi32(_mm256_xor_si256(q, sign), sign); // q = (q ^ sign) - sign

    q
}

#[inline(always)]
pub unsafe fn _mm256_divv_epi64x_bf_v3(numers: __m256i, multipliers: __m256i, shifts: __m256i) -> __m256i {
    const SHIFT_MASK: u8 = crate::divider::Divider::<u64>::SHIFT_MASK;

    let masked_shift = _mm256_and_si256(shifts, _mm256_set1_epi64x(SHIFT_MASK as i64));

    let sign = _mm256_srai_epi64x_v3(shifts, 63); // must be arithmetic shift
    let mut q = _mm256_mullhi_epi64x_v3(numers, multipliers);
    q = _mm256_add_epi64(q, numers); // q += numers

    // If q is non-negative, we have nothing to do.
    // If q is negative, we want to add either (2**shift)-1 if d is
    // a power of 2, or (2**shift) if it is not a power of 2.
    let is_power_of_2 = _mm256_cmpeq_epi64(multipliers, _mm256_setzero_si256());

    let q_sign = _mm256_srai_epi64x_v3(q, 63); // q_sign = q >> 63
    let mask = _mm256_sub_epi64(_mm256_sllv_epi64(_mm256_set1_epi64x(1), masked_shift), is_power_of_2);
    q = _mm256_add_epi64(q, _mm256_and_si256(q_sign, mask)); // q = q + (q_sign & mask)
    q = _mm256_srav_epi64x_v3(q, masked_shift); // q >>= shift
    q = _mm256_sub_epi64(_mm256_xor_si256(q, sign), sign); // q = (q ^ sign) - sign

    q
}
