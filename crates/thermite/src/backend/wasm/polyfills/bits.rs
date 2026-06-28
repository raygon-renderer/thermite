use super::*;

/// Shift the entire 128-bit register left by `IMM8` bytes, filling vacated
/// low bytes with zeros. Equivalent to x86 `PSLLDQ` / `_mm_bslli_si128`.
///
/// Since `IMM8` is a const generic, the `match` collapses at monomorphization
/// time and the compiler emits a single `i8x16.shuffle` instruction.
#[rustfmt::skip]
#[inline(always)]
pub fn wasm_bshli<const IMM8: i32>(value: v128) -> v128 {
    let z = i8x16_splat(0);
    // a=zeros, b=value; indices 0-15 select from zeros, 16-31 from value.
    // Result byte i: i < IMM8 -> zero, else value[i - IMM8].
    match IMM8 {
        0  => value,
        1  => i8x16_shuffle::<0,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30>(z, value),
        2  => i8x16_shuffle::<0, 0,16,17,18,19,20,21,22,23,24,25,26,27,28,29>(z, value),
        3  => i8x16_shuffle::<0, 0, 0,16,17,18,19,20,21,22,23,24,25,26,27,28>(z, value),
        4  => i8x16_shuffle::<0, 0, 0, 0,16,17,18,19,20,21,22,23,24,25,26,27>(z, value),
        5  => i8x16_shuffle::<0, 0, 0, 0, 0,16,17,18,19,20,21,22,23,24,25,26>(z, value),
        6  => i8x16_shuffle::<0, 0, 0, 0, 0, 0,16,17,18,19,20,21,22,23,24,25>(z, value),
        7  => i8x16_shuffle::<0, 0, 0, 0, 0, 0, 0,16,17,18,19,20,21,22,23,24>(z, value),
        8  => i8x16_shuffle::<0, 0, 0, 0, 0, 0, 0, 0,16,17,18,19,20,21,22,23>(z, value),
        9  => i8x16_shuffle::<0, 0, 0, 0, 0, 0, 0, 0, 0,16,17,18,19,20,21,22>(z, value),
        10 => i8x16_shuffle::<0, 0, 0, 0, 0, 0, 0, 0, 0, 0,16,17,18,19,20,21>(z, value),
        11 => i8x16_shuffle::<0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,16,17,18,19,20>(z, value),
        12 => i8x16_shuffle::<0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,16,17,18,19>(z, value),
        13 => i8x16_shuffle::<0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,16,17,18>(z, value),
        14 => i8x16_shuffle::<0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,16,17>(z, value),
        15 => i8x16_shuffle::<0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,16>(z, value),
        _  => i8x16_splat(0), // IMM8 >= 16: all bytes shifted out
    }
}

/// Shift the entire 128-bit register right by `IMM8` bytes, filling vacated
/// high bytes with zeros. Equivalent to x86 `PSRLDQ` / `_mm_bsrli_si128`.
///
/// Since `IMM8` is a const generic, the `match` collapses at monomorphization
/// time and the compiler emits a single `i8x16.shuffle` instruction.
#[rustfmt::skip]
#[inline(always)]
pub fn wasm_bshri<const IMM8: i32>(value: v128) -> v128 {
    let z = i8x16_splat(0);
    // a=value, b=zeros; indices 0-15 select from value, 16-31 from zeros.
    // Result byte i: i + IMM8 < 16 -> value[i + IMM8], else zero.
    match IMM8 {
        0  => value,
        1  => i8x16_shuffle::< 1, 2, 3, 4, 5, 6, 7, 8, 9,10,11,12,13,14,15,16>(value, z),
        2  => i8x16_shuffle::< 2, 3, 4, 5, 6, 7, 8, 9,10,11,12,13,14,15,16,16>(value, z),
        3  => i8x16_shuffle::< 3, 4, 5, 6, 7, 8, 9,10,11,12,13,14,15,16,16,16>(value, z),
        4  => i8x16_shuffle::< 4, 5, 6, 7, 8, 9,10,11,12,13,14,15,16,16,16,16>(value, z),
        5  => i8x16_shuffle::< 5, 6, 7, 8, 9,10,11,12,13,14,15,16,16,16,16,16>(value, z),
        6  => i8x16_shuffle::< 6, 7, 8, 9,10,11,12,13,14,15,16,16,16,16,16,16>(value, z),
        7  => i8x16_shuffle::< 7, 8, 9,10,11,12,13,14,15,16,16,16,16,16,16,16>(value, z),
        8  => i8x16_shuffle::< 8, 9,10,11,12,13,14,15,16,16,16,16,16,16,16,16>(value, z),
        9  => i8x16_shuffle::< 9,10,11,12,13,14,15,16,16,16,16,16,16,16,16,16>(value, z),
        10 => i8x16_shuffle::<10,11,12,13,14,15,16,16,16,16,16,16,16,16,16,16>(value, z),
        11 => i8x16_shuffle::<11,12,13,14,15,16,16,16,16,16,16,16,16,16,16,16>(value, z),
        12 => i8x16_shuffle::<12,13,14,15,16,16,16,16,16,16,16,16,16,16,16,16>(value, z),
        13 => i8x16_shuffle::<13,14,15,16,16,16,16,16,16,16,16,16,16,16,16,16>(value, z),
        14 => i8x16_shuffle::<14,15,16,16,16,16,16,16,16,16,16,16,16,16,16,16>(value, z),
        15 => i8x16_shuffle::<15,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16>(value, z),
        _  => i8x16_splat(0), // IMM8 >= 16: all bytes shifted out
    }
}

// ===========================================================================
// 2D Morton (Z-order) encode/decode via `i8x16.swizzle` as a nibble LUT - the
// SIMD128 analogue of the x86 `pshufb` path (see backend/x86_v2/polyfills/bits.rs
// for the method and bit-algebra). Two differences from `pshufb`: `swizzle`
// indexes by the *whole* byte and zeroes any lane whose index is >= 16 (pshufb
// masks to the low nibble), so every decode LUT index is masked to a nibble
// first; and there is no SIMD128 carry-less multiply, so the u64 2D path and all
// N != 2 stay on the generic cascade.
// ===========================================================================

/// Nibble -> spread-by-1 byte LUT (`b3 b2 b1 b0 -> 0 b3 0 b2 0 b1 0 b0`).
const MORTON2_SPREAD_LUT: v128 = u8x16(
    0x00, 0x01, 0x04, 0x05, 0x10, 0x11, 0x14, 0x15, 0x40, 0x41, 0x44, 0x45, 0x50, 0x51, 0x54, 0x55,
);
/// Nibble -> 2-bit compress LUT (gather bits 0 and 2: `i -> bit0(i) | bit2(i)<<1`).
const MORTON2_COMPRESS_LUT: v128 = u8x16(0, 1, 0, 1, 2, 3, 2, 3, 0, 1, 0, 1, 2, 3, 2, 3);

/// Spread the low 8 bits of each 16-bit lane by one (2D Morton).
#[inline(always)]
pub fn wasm_morton2_spread_epu16x(v: v128) -> v128 {
    let c = v128_and(v, u16x8_splat(0x00FF));
    let n = v128_and(v128_or(c, i16x8_shl(c, 4)), u16x8_splat(0x0F0F));
    i8x16_swizzle(MORTON2_SPREAD_LUT, n) // n bytes are already in 0..=15
}

/// Spread the low 16 bits of each 32-bit lane by one (2D Morton).
#[inline(always)]
pub fn wasm_morton2_spread_epu32x(v: v128) -> v128 {
    let c = v128_and(v, u32x4_splat(0x0000_FFFF));
    let c = v128_and(v128_or(c, i32x4_shl(c, 8)), u32x4_splat(0x00FF_00FF));
    let n = v128_and(v128_or(c, i32x4_shl(c, 4)), u32x4_splat(0x0F0F_0F0F));
    i8x16_swizzle(MORTON2_SPREAD_LUT, n)
}

/// Per-16-bit-lane 2D Morton encode: low 8 bits of `x` (even) with `y` (odd).
#[inline(always)]
pub fn wasm_morton2_epu16x(x: v128, y: v128) -> v128 {
    v128_or(wasm_morton2_spread_epu16x(x), i16x8_shl(wasm_morton2_spread_epu16x(y), 1))
}

/// Per-32-bit-lane 2D Morton encode: low 16 bits of `x` (even) with `y` (odd).
#[inline(always)]
pub fn wasm_morton2_epu32x(x: v128, y: v128) -> v128 {
    v128_or(wasm_morton2_spread_epu32x(x), i32x4_shl(wasm_morton2_spread_epu32x(y), 1))
}

/// Compress the even bits of each 16-bit lane back to a contiguous low 8 bits -
/// the inverse of [`wasm_morton2_spread_epu16x`].
#[inline(always)]
pub fn wasm_morton2_compress_epu16x(v: v128) -> v128 {
    let nib = u16x8_splat(0x0F0F);
    let e = v128_and(v, u16x8_splat(0x5555));
    // indices masked to a nibble (`swizzle` zeroes lanes whose index is >= 16)
    let lo = i8x16_swizzle(MORTON2_COMPRESS_LUT, v128_and(e, nib));
    let hi = i8x16_swizzle(MORTON2_COMPRESS_LUT, v128_and(u16x8_shr(e, 4), nib));
    let n = v128_and(v128_or(lo, i16x8_shl(hi, 2)), nib);
    v128_and(v128_or(n, u16x8_shr(n, 4)), u16x8_splat(0x00FF))
}

/// Compress the even bits of each 32-bit lane back to a contiguous low 16 bits -
/// the inverse of [`wasm_morton2_spread_epu32x`].
#[inline(always)]
pub fn wasm_morton2_compress_epu32x(v: v128) -> v128 {
    let nib = u32x4_splat(0x0F0F_0F0F);
    let e = v128_and(v, u32x4_splat(0x5555_5555));
    let lo = i8x16_swizzle(MORTON2_COMPRESS_LUT, v128_and(e, nib));
    let hi = i8x16_swizzle(MORTON2_COMPRESS_LUT, v128_and(u32x4_shr(e, 4), nib));
    let n = v128_and(v128_or(lo, i32x4_shl(hi, 2)), nib);
    let c = v128_and(v128_or(n, u32x4_shr(n, 4)), u32x4_splat(0x00FF_00FF));
    v128_and(v128_or(c, u32x4_shr(c, 8)), u32x4_splat(0x0000_FFFF))
}
