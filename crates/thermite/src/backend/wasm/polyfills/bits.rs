use core::arch::wasm32::*;

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
