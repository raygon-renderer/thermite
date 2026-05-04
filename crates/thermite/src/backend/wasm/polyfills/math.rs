use super::*;

#[inline(always)]
pub fn f32x4_maddx(x: v128, m: v128, a: v128) -> v128 {
    // 1. Split 128-bit packed float (4 lanes) into two sets of doubles (2 lanes each)

    // Low 2 floats -> doubles (Lanes 0, 1)
    // Equivalent to: _mm_cvtps_pd(x)
    let x_lo = f64x2_promote_low_f32x4(x);
    let m_lo = f64x2_promote_low_f32x4(m);
    let a_lo = f64x2_promote_low_f32x4(a);

    // High 2 floats -> doubles
    // We shuffle lanes 2,3 to positions 0,1, then promote.
    // Equivalent to: _mm_cvtps_pd(_mm_movehl_ps(x, x))
    let x_hi_src = i32x4_shuffle::<2, 3, 2, 3>(x, x);
    let m_hi_src = i32x4_shuffle::<2, 3, 2, 3>(m, m);
    let a_hi_src = i32x4_shuffle::<2, 3, 2, 3>(a, a);

    let x_hi = f64x2_promote_low_f32x4(x_hi_src);
    let m_hi = f64x2_promote_low_f32x4(m_hi_src);
    let a_hi = f64x2_promote_low_f32x4(a_hi_src);

    // 2. Perform operation in f64
    let res_lo = f64x2_add(f64x2_mul(x_lo, m_lo), a_lo);
    let res_hi = f64x2_add(f64x2_mul(x_hi, m_hi), a_hi);

    // 3. Convert back to f32.
    // f32x4_demote_f64x2_zero converts [f64; 2] to [f32; 2] and fills the upper 64 bits with zero.
    // Result format: [A, B, 0, 0]
    let out_lo = f32x4_demote_f64x2_zero(res_lo);
    let out_hi = f32x4_demote_f64x2_zero(res_hi);

    // 4. Shuffle high results back into the upper lanes
    // We take the lower 2 lanes from out_lo (indices 0, 1)
    // And the lower 2 lanes from out_hi (indices 4, 5 referring to the second arg)
    // Equivalent to: _mm_movelh_ps(out_lo, out_hi)
    i32x4_shuffle::<0, 1, 4, 5>(out_lo, out_hi)
}

#[inline(always)]
pub fn f64x2_maddx(x: v128, m: v128, a: v128) -> v128 {
    // Constants for Veltkamp's splitting (2^27 + 1)
    let splitter = f64x2_splat(134217729.0);

    // 1. Veltkamp's Split for 'x'
    let c_x = f64x2_mul(x, splitter);
    let x_h = f64x2_sub(c_x, f64x2_sub(c_x, x));
    let x_l = f64x2_sub(x, x_h);

    // 2. Veltkamp's Split for 'm'
    let c_m = f64x2_mul(m, splitter);
    let m_h = f64x2_sub(c_m, f64x2_sub(c_m, m));
    let m_l = f64x2_sub(m, m_h);

    // 3. Dekker's Exact Product
    // p = x * m
    let p = f64x2_mul(x, m);

    // Calculate error term 'e'
    let t1 = f64x2_mul(x_h, m_h);
    let t2 = f64x2_sub(t1, p);
    let t3 = f64x2_mul(x_h, m_l);
    let t4 = f64x2_mul(x_l, m_h);
    let t5 = f64x2_mul(x_l, m_l);

    let e = f64x2_add(f64x2_add(f64x2_add(t2, t3), t4), t5);

    // 4. Knuth's TwoSum (Adding 'a' to the exact product)
    let sum = f64x2_add(p, a);

    // Recover rounding error: sum = p + a + err
    let v = f64x2_sub(sum, p);
    let z = f64x2_sub(sum, v); // Virtual p
    let err_a = f64x2_sub(a, v);
    let err_p = f64x2_sub(p, z);
    let err_add = f64x2_add(err_p, err_a);

    // 5. Final Combination
    let total_error = f64x2_add(e, err_add);

    f64x2_add(sum, total_error)
}

#[inline(always)]
pub fn i32x4_saturating_add(a: v128, b: v128) -> v128 {
    // 1. Standard wrapping addition
    let sum = i32x4_add(a, b);

    // 2. Detect Overflow
    // Logic: ~(a ^ b) checks if signs are SAME
    //        (a ^ sum) checks if sign CHANGED
    //        Both true + Sign Bit set = Overflow
    let sign_bit = i32x4_splat(i32::MIN); // 0x80000000
    let sign_check = v128_and(
        v128_not(v128_xor(a, b)), // Inputs have same sign
        v128_xor(a, sum),         // Result has different sign than a
    );
    let overflow_bits = v128_and(sign_check, sign_bit);

    // Create a mask of 0xFFFFFFFF where overflow occurred, 0 otherwise
    let mask = i32x4_shr(overflow_bits, 31);

    // 3. Calculate Saturation Target
    // If a is positive (sign bit 0): Target is INT_MAX
    // If a is negative (sign bit 1): Target is INT_MIN
    // Formula: INT_MAX ^ (a >> 31)
    //   pos: 0x7FFFFFFF ^ 0x00000000 = 0x7FFFFFFF (MAX)
    //   neg: 0x7FFFFFFF ^ 0xFFFFFFFF = 0x80000000 (MIN)
    let max = i32x4_splat(i32::MAX);
    let a_sign_extended = i32x4_shr(a, 31);
    let sat_val = v128_xor(max, a_sign_extended);

    // 4. Select Result
    v128_bitselect(sat_val, sum, mask)
}

#[inline(always)]
pub fn i32x4_saturating_sub(a: v128, b: v128) -> v128 {
    // 1. Standard wrapping subtraction
    let diff = i32x4_sub(a, b);

    // 2. Detect Overflow
    // Logic: (a ^ b) checks if signs are DIFFERENT
    //        (a ^ diff) checks if sign CHANGED from a
    //        Both true + Sign Bit set = Overflow
    let sign_bit = i32x4_splat(i32::MIN);
    let sign_check = v128_and(
        v128_xor(a, b),    // Inputs have different signs
        v128_xor(a, diff), // Result has different sign than a
    );
    let overflow_bits = v128_and(sign_check, sign_bit);

    // Create mask
    let mask = i32x4_shr(overflow_bits, 31);

    // 3. Calculate Saturation Target (Same as add)
    let max = i32x4_splat(i32::MAX);
    let a_sign_extended = i32x4_shr(a, 31);
    let sat_val = v128_xor(max, a_sign_extended);

    // 4. Select Result
    v128_bitselect(sat_val, diff, mask)
}

#[inline(always)]
pub fn i64x2_saturating_add(a: v128, b: v128) -> v128 {
    // 1. Standard wrapping addition
    let sum = i64x2_add(a, b);

    // 2. Detect Overflow
    // Logic: ~(a ^ b) checks if signs are SAME
    //        (a ^ sum) checks if sign CHANGED
    //        Both true + Sign Bit set = Overflow
    let sign_bit = i64x2_splat(i64::MIN); // 0x80000000
    let sign_check = v128_and(
        v128_not(v128_xor(a, b)), // Inputs have same sign
        v128_xor(a, sum),         // Result has different sign than a
    );
    let overflow_bits = v128_and(sign_check, sign_bit);

    // Create a mask of all-ones where overflow occurred, 0 otherwise
    let mask = i64x2_shr(overflow_bits, 63);

    // 3. Calculate Saturation Target
    // If a is positive (sign bit 0): Target is INT_MAX
    // If a is negative (sign bit 1): Target is INT_MIN
    // Formula: INT_MAX ^ (a >> 63)
    //   pos: 0x7FFFFFFFFFFFFFFF ^ 0x0000000000000000 = INT_MAX
    //   neg: 0x7FFFFFFFFFFFFFFF ^ 0xFFFFFFFFFFFFFFFF = INT_MIN
    let max = i64x2_splat(i64::MAX);
    let a_sign_extended = i64x2_shr(a, 63);
    let sat_val = v128_xor(max, a_sign_extended);

    // 4. Select Result
    v128_bitselect(sat_val, sum, mask)
}

#[inline(always)]
pub fn i64x2_saturating_sub(a: v128, b: v128) -> v128 {
    // 1. Standard wrapping subtraction
    let diff = i64x2_sub(a, b);

    // 2. Detect Overflow
    // Logic: (a ^ b) checks if signs are DIFFERENT
    //        (a ^ diff) checks if sign CHANGED from a
    //        Both true + Sign Bit set = Overflow
    let sign_bit = i64x2_splat(i64::MIN);
    let sign_check = v128_and(
        v128_xor(a, b),    // Inputs have different signs
        v128_xor(a, diff), // Result has different sign than a
    );
    let overflow_bits = v128_and(sign_check, sign_bit);

    // Create mask
    let mask = i64x2_shr(overflow_bits, 63);

    // 3. Calculate Saturation Target (Same as add)
    let max = i64x2_splat(i64::MAX);
    let a_sign_extended = i64x2_shr(a, 63);
    let sat_val = v128_xor(max, a_sign_extended);

    // 4. Select Result
    v128_bitselect(sat_val, diff, mask)
}

#[inline(always)]
pub fn u32x4_saturating_add(a: v128, b: v128) -> v128 {
    // 1. Calculate remaining space until overflow
    // u32::MAX - a is equivalent to bitwise NOT (!a)
    let remaining = v128_not(a);

    // 2. Clamp b to the remaining space
    // If b > remaining, we add remaining (resulting in MAX)
    // If b <= remaining, we add b (normal behavior)
    let safe_b = u32x4_min(b, remaining);

    // 3. Standard wrapping addition
    u32x4_add(a, safe_b)
}

#[inline(always)]
pub fn u32x4_saturating_sub(a: v128, b: v128) -> v128 {
    // 1. Clamp b so it cannot exceed a
    // If b > a, we subtract a (resulting in 0)
    // If b <= a, we subtract b (normal behavior)
    let safe_b = u32x4_min(a, b);

    // 2. Standard wrapping subtraction
    u32x4_sub(a, safe_b)
}

#[inline(always)]
pub fn u64x2_saturating_add(a: v128, b: v128) -> v128 {
    // 1. Calculate remaining space until overflow
    // u64::MAX - a is equivalent to bitwise NOT (!a)
    let remaining = v128_not(a);

    // 2. Clamp b to the remaining space
    // If b > remaining, we add remaining (resulting in MAX)
    // If b <= remaining, we add b (normal behavior)
    let safe_b = u64x2_min(b, remaining);

    // 3. Standard wrapping addition
    u64x2_add(a, safe_b)
}

#[inline(always)]
pub fn u64x2_saturating_sub(a: v128, b: v128) -> v128 {
    // 1. Clamp b so it cannot exceed a
    // If b > a, we subtract a (resulting in 0)
    // If b <= a, we subtract b (normal behavior)
    let safe_b = u64x2_min(a, b);

    // 2. Standard wrapping subtraction
    u64x2_sub(a, safe_b)
}
