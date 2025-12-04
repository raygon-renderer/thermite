#![allow(clippy::identity_op)]

pub use core::arch::wasm32::*;

pub mod bits;
pub mod casts;
pub mod cmp;
pub mod math;

pub use bits::*;
pub use casts::*;
pub use cmp::*;
pub use math::*;

pub use crate::backend::generic::polyfills::*;

#[inline(always)]
pub const fn identity<T>(value: T) -> T {
    value
}

#[rustfmt::skip]
#[inline(always)]
pub const fn x4indices(a: u8, b: u8, c: u8, d: u8) -> v128 {
    u8x16(
        a * 4, a * 4 + 1, a * 4 + 2, a * 4 + 3, // Lane A (Bytes 0-3)
        b * 4, b * 4 + 1, b * 4 + 2, b * 4 + 3, // Lane B (Bytes 4-7)
        c * 4, c * 4 + 1, c * 4 + 2, c * 4 + 3, // Lane C (Bytes 8-11)
        d * 4, d * 4 + 1, d * 4 + 2, d * 4 + 3, // Lane D (Bytes 12-15)
    )
}

#[rustfmt::skip]
#[inline(always)]
pub const fn x2indices(a: u8, b: u8) -> v128 {
    u8x16(
        // Lane A (Bytes 0-7)
        a * 8, a * 8 + 1, a * 8 + 2, a * 8 + 3,
        a * 8 + 4, a * 8 + 5, a * 8 + 6, a * 8 + 7,
        // Lane B (Bytes 8-15)
        b * 8, b * 8 + 1, b * 8 + 2, b * 8 + 3,
        b * 8 + 4, b * 8 + 5, b * 8 + 6, b * 8 + 7,
    )
}

#[inline(always)]
pub const fn imm8x2_to_indices<const IMM8: i32>() -> v128 {
    x2indices(((IMM8 >> 0) & 0b1) as u8, ((IMM8 >> 1) & 0b1) as u8)
}

#[inline(always)]
pub const fn imm8x4_to_indices<const IMM8: i32>() -> v128 {
    x4indices(
        ((IMM8 >> 0) & 0b11) as u8,
        ((IMM8 >> 2) & 0b11) as u8,
        ((IMM8 >> 4) & 0b11) as u8,
        ((IMM8 >> 6) & 0b11) as u8,
    )
}

#[inline(always)]
pub const fn imm8x2_to_mask<const IMM8: i32>() -> v128 {
    let a = -(((IMM8 >> 0) & 0b1) as i8);
    let b = -(((IMM8 >> 1) & 0b1) as i8);

    i8x16(
        a, a, a, a, a, a, a, a, // Lane A (Bytes 0-7)
        b, b, b, b, b, b, b, b, // Lane B (Bytes 8-15)
    )
}

#[inline(always)]
pub const fn imm8x4_to_mask<const IMM8: i32>() -> v128 {
    let a = -(((IMM8 >> 0) & 0b1) as i8);
    let b = -(((IMM8 >> 1) & 0b1) as i8);
    let c = -(((IMM8 >> 2) & 0b1) as i8);
    let d = -(((IMM8 >> 3) & 0b1) as i8);

    i8x16(
        a, a, a, a, // Lane A (Bytes 0-3)
        b, b, b, b, // Lane B (Bytes 4-7)
        c, c, c, c, // Lane C (Bytes 8-11)
        d, d, d, d, // Lane D (Bytes 12-15)
    )
}
