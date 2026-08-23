#![allow(clippy::identity_op, unsafe_op_in_unsafe_fn, clippy::missing_safety_doc)]

pub use super::arch::*;

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

/// `expand[j] = j / elem` - replicates each lane index across its `elem` byte
/// slots; `offsets[j] = j % elem` - the intra-lane byte offsets.
const fn lane_expand_pattern(elem: usize) -> [u8; 16] {
    let mut o = [0u8; 16];
    let mut j = 0;
    while j < 16 {
        o[j] = (j / elem) as u8;
        j += 1;
    }
    o
}

const fn lane_offset_pattern(elem: usize) -> [u8; 16] {
    let mut o = [0u8; 16];
    let mut j = 0;
    while j < 16 {
        o[j] = (j % elem) as u8;
        j += 1;
    }
    o
}

// ---------------------------------------------------------------------------
// Runtime `swizzle` control from a LIVE index register
// (`wasm_ctrl_x{16,8,4,2}`).
//
// `Register::permutev` takes the indices as a same-lane-count unsigned
// register (and on wasm every register is a `v128`), so the control build
// stays in registers, with no memory round trip and no scalar per-byte build.
//
// The mapping is `byte[i * elem + b] = idx[i] * elem + b`. Replicating a
// sub-256 value into every byte of its lane is one multiply by `0x01..01` in
// that lane width, so:
//
//   ctrl = ((idx << log2(elem)) * 0x01..01) + lane_offset_pattern(elem)
//
// three instructions plus a constant, for every width except bytes, where the
// index register already IS the control.
//
// No clamp, and deliberately NO NARROWING: the shape
// `u32x4_min` -> `u16x8_narrow_i32x4` / `u8x16_narrow_i16x8` that the previous
// scalar-sourced builder used is suspected to miscompile on stable
// (unsigned min feeding a signed-input narrow, see
// `todo/WASM_SWIZZLE_NARROW_HAZARD.md`), and a bad fold there silently
// corrupts every shuffle rather than producing a visibly wrong number. The
// multiply/add recipe touches no signed-source instruction at all.
//
// An index one past the addressable range scales to byte `LANES * elem == 16`,
// which `u8x16_swizzle` zeroes. Anything further out yields some other
// unspecified byte pattern, which the `permutev` contract permits. Every
// result is a byte shuffle, so all of it is memory-safe by construction.
// ---------------------------------------------------------------------------

/// 16 byte lanes: the index register is the swizzle control already.
#[inline(always)]
pub fn wasm_ctrl_x16(idxs: v128) -> v128 {
    idxs
}

/// 8 lanes of 16 bits: `(idx << 1) * 0x0101` puts `2i` in both bytes of the
/// lane, and the byte offsets turn them into `(2i, 2i + 1)`.
#[inline(always)]
pub fn wasm_ctrl_x8(idxs: v128) -> v128 {
    let rep = i16x8_mul(i16x8_shl(idxs, 1), u16x8_splat(0x0101));
    u8x16_add(rep, const { u8x16_from_bytes(lane_offset_pattern(2)) })
}

/// 4 lanes of 32 bits: `(idx << 2) * 0x01010101`, then the byte offsets.
#[inline(always)]
pub fn wasm_ctrl_x4(idxs: v128) -> v128 {
    let rep = i32x4_mul(i32x4_shl(idxs, 2), u32x4_splat(0x0101_0101));
    u8x16_add(rep, const { u8x16_from_bytes(lane_offset_pattern(4)) })
}

/// 2 lanes of 64 bits: `(idx << 3) * 0x0101010101010101`, then the byte
/// offsets. Unlike NEON, wasm has a real `i64x2.mul`.
#[inline(always)]
pub fn wasm_ctrl_x2(idxs: v128) -> v128 {
    let rep = i64x2_mul(i64x2_shl(idxs, 3), u64x2_splat(0x0101_0101_0101_0101));
    u8x16_add(rep, const { u8x16_from_bytes(lane_offset_pattern(8)) })
}

/// Replicate/scale/offset tail used by the byte-row path, on in-range byte lane
/// indices.
#[inline(always)]
fn wasm_lane_expand<const N: usize>(bytes: v128) -> v128 {
    if const { N == 16 } {
        return bytes;
    }

    let expanded = u8x16_swizzle(bytes, const { u8x16_from_bytes(lane_expand_pattern(16 / N)) });
    let scaled = u8x16_shl(expanded, const { (16 / N).trailing_zeros() });

    u8x16_add(scaled, const { u8x16_from_bytes(lane_offset_pattern(16 / N)) })
}

/// Byte-row entry point for the compress/expand table paths. A table row's
/// leading `LANES` entries are always `< LANES`, so neither a narrow nor a
/// clamp applies, see `neon_lane_table_row`.
///
/// # Safety
///
/// `row` must point to at least 8 readable bytes, all `< N`.
#[inline(always)]
pub unsafe fn wasm_lane_table_row<const N: usize>(row: *const u8) -> v128 {
    wasm_lane_expand::<N>(unsafe { v128_load64_zero(row as *const u64) })
}

/// `[u8; 16]` -> `v128`, for the constant patterns above.
#[inline(always)]
pub const fn u8x16_from_bytes(b: [u8; 16]) -> v128 {
    u8x16(
        b[0], b[1], b[2], b[3], b[4], b[5], b[6], b[7], b[8], b[9], b[10], b[11], b[12], b[13], b[14], b[15],
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
