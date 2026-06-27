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

// ===========================================================================================
// Cross-family integer/float narrow/widen "instructions" that WASM SIMD128 lacks at this level
// (no multi-lane `as`-style narrow/widen). These are the register-level polyfills behind the
// 8/16 <-> 16/32/64 and 8/16 <-> f32/f64 casts in `registers/half8.rs` / `half16.rs`. The
// multi-output (`[v128; N]`) forms are treated as instructions that write more than one register.
// ===========================================================================================

// Private store helpers (kept local to this module so the casts below are self-contained; the
// register files keep their own equivalents for their remaining store-rebuild call sites).
#[inline(always)]
unsafe fn store_dwords(v: v128) -> [i32; 4] {
    let mut arr = [0i32; 4];
    v128_store(arr.as_mut_ptr() as *mut _, v);
    arr
}
#[inline(always)]
unsafe fn store_qwords(v: v128) -> [i64; 2] {
    let mut arr = [0i64; 2];
    v128_store(arr.as_mut_ptr() as *mut _, v);
    arr
}
#[inline(always)]
unsafe fn store_words(v: v128) -> [i16; 8] {
    let mut arr = [0i16; 8];
    v128_store(arr.as_mut_ptr() as *mut _, v);
    arr
}

// --- 8-bit widen helpers (byte -> word -> dword via the `*_extend_low_*` ladder) ---
#[inline(always)]
pub unsafe fn widen_i8_to_i32(v: v128) -> v128 {
    i32x4_extend_low_i16x8(i16x8_extend_low_i8x16(v))
}
#[inline(always)]
pub unsafe fn widen_u8_to_u32(v: v128) -> v128 {
    i32x4_extend_low_u16x8(i16x8_extend_low_u8x16(v))
}
// Widen native 16 bytes -> 4x i32x4: each dword group shuffled into the low 4 bytes then widened.
#[inline(always)]
pub unsafe fn widen_i8x16_to_4xi32x4(v: v128) -> [v128; 4] {
    [
        widen_i8_to_i32(v),
        widen_i8_to_i32(i32x4_shuffle::<1, 0, 0, 0>(v, v)),
        widen_i8_to_i32(i32x4_shuffle::<2, 0, 0, 0>(v, v)),
        widen_i8_to_i32(i32x4_shuffle::<3, 0, 0, 0>(v, v)),
    ]
}
#[inline(always)]
pub unsafe fn widen_u8x16_to_4xu32x4(v: v128) -> [v128; 4] {
    [
        widen_u8_to_u32(v),
        widen_u8_to_u32(i32x4_shuffle::<1, 0, 0, 0>(v, v)),
        widen_u8_to_u32(i32x4_shuffle::<2, 0, 0, 0>(v, v)),
        widen_u8_to_u32(i32x4_shuffle::<3, 0, 0, 0>(v, v)),
    ]
}

// --- 8-bit narrow helpers (gather the low byte of each wide lane into contiguous low bytes) ---
// Narrow 2x i16x8 (16 i16) -> 16 i8 (low byte of each lane), indices 0..16 -> v[0], 16..32 -> v[1].
#[inline(always)]
pub unsafe fn narrow_2xi16x8_to_bytes(v: [v128; 2]) -> v128 {
    #[rustfmt::skip]
    let r = i8x16_shuffle::<
        0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30,
    >(v[0], v[1]);
    r
}
// Narrow 4x i32x4 (16 i32) -> 16 i8 (low byte of each lane).
#[inline(always)]
pub unsafe fn narrow_4xi32x4_to_bytes(v: [v128; 4]) -> v128 {
    let a = store_dwords(v[0]);
    let b = store_dwords(v[1]);
    let c = store_dwords(v[2]);
    let d = store_dwords(v[3]);
    i8x16(
        a[0] as i8, a[1] as i8, a[2] as i8, a[3] as i8,
        b[0] as i8, b[1] as i8, b[2] as i8, b[3] as i8,
        c[0] as i8, c[1] as i8, c[2] as i8, c[3] as i8,
        d[0] as i8, d[1] as i8, d[2] as i8, d[3] as i8,
    )
}

// --- 16-bit widen/narrow to/from 64-bit ---
// Widen one i16x8 (8 i16) -> 4x i64x2 (low word of each lane sign/zero-extended via `as`).
#[inline(always)]
pub unsafe fn widen_i16x8_to_4xi64x2(v: v128) -> [v128; 4] {
    let a = store_words(v);
    [
        i64x2(a[0] as i64, a[1] as i64),
        i64x2(a[2] as i64, a[3] as i64),
        i64x2(a[4] as i64, a[5] as i64),
        i64x2(a[6] as i64, a[7] as i64),
    ]
}
#[inline(always)]
pub unsafe fn widen_u16x8_to_4xu64x2(v: v128) -> [v128; 4] {
    let a = store_words(v);
    [
        u64x2(a[0] as u16 as u64, a[1] as u16 as u64),
        u64x2(a[2] as u16 as u64, a[3] as u16 as u64),
        u64x2(a[4] as u16 as u64, a[5] as u16 as u64),
        u64x2(a[6] as u16 as u64, a[7] as u16 as u64),
    ]
}
// Narrow 4x i64x2 (8 i64) -> 8 i16 (word 0 of each lane truncated via `as`).
#[inline(always)]
pub unsafe fn narrow_4xi64x2_to_words(v: [v128; 4]) -> v128 {
    let a = store_qwords(v[0]);
    let b = store_qwords(v[1]);
    let c = store_qwords(v[2]);
    let d = store_qwords(v[3]);
    i16x8(
        a[0] as i16, a[1] as i16, b[0] as i16, b[1] as i16,
        c[0] as i16, c[1] as i16, d[0] as i16, d[1] as i16,
    )
}

// --- f64 fan-out / fan-in (shared by the 8/16 <-> f64 cast paths) ---
// Truncate an F64x2Wasm to its low 2 lanes as [i32; 2] (the low byte/word of each is the value).
#[inline(always)]
pub unsafe fn f64x2_to_2xi32(v: v128) -> [i32; 2] {
    let d = store_dwords(i32x4_trunc_sat_f64x2_zero(v));
    [d[0], d[1]]
}
// Convert one i32x4 into 2 F64x2Wasm (low 2 lanes, high 2 lanes).
#[inline(always)]
pub unsafe fn i32x4_to_2xf64x2(ints: v128) -> [v128; 2] {
    [
        f64x2_convert_low_i32x4(ints),
        f64x2_convert_low_i32x4(i32x4_shuffle::<2, 3, 2, 3>(ints, ints)),
    ]
}
