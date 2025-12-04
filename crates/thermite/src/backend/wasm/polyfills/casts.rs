use super::*;

#[inline(always)]
pub fn bx4_to_i32x4x(value: generic_array::GenericArray<bool, generic_array::typenum::U4>) -> v128 {
    #[rustfmt::skip]
    let mask = i8x16(
        value[0] as i8,
        value[1] as i8,
        value[2] as i8,
        value[3] as i8,
        0, 0, 0, 0,
        0, 0, 0, 0,
        0, 0, 0, 0,
    );

    // negate the mask so that true = -1 and false = 0, then sign-extend to i32
    i32x4_extend_low_i16x8(i16x8_extend_low_i8x16(i8x16_neg(mask)))
}

#[inline(always)]
pub fn bx2_to_i64x2x(value: generic_array::GenericArray<bool, generic_array::typenum::U2>) -> v128 {
    #[rustfmt::skip]
    let mask = i8x16(
        value[0] as i8, // duplicate each value
        value[0] as i8, // to fill 16 bytes,
        value[1] as i8, // so the mask
        value[1] as i8, // fits 64-bit lanes
        0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0,
    );

    // negate the mask so that true = -1 and false = 0, then sign-extend to i32
    i32x4_extend_low_i16x8(i16x8_extend_low_i8x16(i8x16_neg(mask)))
}

#[inline(always)]
pub fn convert_u64x2_to_f64x2(v: v128) -> v128 {
    let magic_i_lo = u64x2_splat(0x4330000000000000); // 2^52
    let magic_i_hi32 = u64x2_splat(0x4530000000000000); // 2^84
    let magic_d_all = u64x2_splat(0x4530000000100000); // 2^84 + 2^52 (bits interpreted as float later)

    // Blend: Low 32 bits from 'v', High 32 bits from 'magic_i_lo'.
    // v indices (i32): 0, 1, 2, 3
    // magic indices:   4, 5, 6, 7
    // Result: [v[0], magic[1], v[2], magic[3]] -> Indices 0, 5, 2, 7
    let v_lo = i32x4_shuffle::<0, 5, 2, 7>(v, magic_i_lo);

    // Extract high 32 bits of v (logical shift)
    let v_shr = u64x2_shr(v, 32);

    // Construct v_hi by mixing high bits with magic header
    let v_hi = v128_xor(v_shr, magic_i_hi32);

    // Math (Bits are implicitly reinterpreted as f64)
    let v_hi_dbl = f64x2_sub(v_hi, magic_d_all);

    f64x2_add(v_hi_dbl, v_lo)
}

#[inline(always)]
pub fn convert_i64x2_to_f64x2(v: v128) -> v128 {
    let magic_i_lo = u64x2_splat(0x4330000000000000); // 2^52
    let magic_i_hi32 = u64x2_splat(0x4530000080000000); // 2^84 + 2^63
    let magic_d_all = u64x2_splat(0x4530000080100000); // 2^84 + 2^63 + 2^52

    // Blend: Low 32 bits from 'v', High 32 bits from 'magic_i_lo'.
    // Indices: 0 (v low), 5 (magic high), 2 (v low), 7 (magic high)
    let v_lo = i32x4_shuffle::<0, 5, 2, 7>(v, magic_i_lo);

    // Extract high 32 bits of v (logical shift)
    // Note: Use SHR (logical), not SHR_S, because we want the raw bits moved down
    let v_shr = u64x2_shr(v, 32);

    // Flip MSB of v_hi and blend
    let v_hi = v128_xor(v_shr, magic_i_hi32);

    // Math
    let v_hi_dbl = f64x2_sub(v_hi, magic_d_all);

    f64x2_add(v_hi_dbl, v_lo)
}
