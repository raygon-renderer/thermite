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

/// Runtime companion to [`x4indices`]/[`x2indices`]: the same `u8x16_swizzle`
/// byte-index control, built with SIMD instead of a scalar loop.
///
/// Those are `const fn`s, which is right for the callers that evaluate at
/// compile time (the `reduce_*` macros, the `IMM8` shuffle family, the
/// literal-index `reverse`) - there they fold to a constant. But const-evaluable
/// code has no SIMD, so the *runtime* callers - every `permutev` - lowered to a
/// scalar per-byte build. This replaced three separate scalar builders: the
/// `xNindices` pair called with runtime arguments, and the hand-rolled `[u8; 16]`
/// loops in the `i16x8` and `i8x16` registers.
///
/// `byte[j] = idx[j / elem] * elem + (j % elem)`, computed as: clamp in the
/// `u32` domain, narrow to bytes, replicate each index across its `elem` slots
/// with a constant swizzle, then scale and offset (both constants).
///
/// Clamping to `N` *before* narrowing is deliberate on two counts. It puts an
/// out-of-range lane index on byte `N * elem == 16`, which `u8x16_swizzle`
/// zeroes - matching the documented semantics - and it avoids the wrap-around
/// aliasing the old `wrapping_mul(2)` builders had. It also sidesteps the
/// narrowing ops being *signed*-input: an index above `i32::MAX` would saturate
/// to 0 and alias lane 0 rather than zeroing.
#[inline(always)]
pub fn wasm_lane_table_dyn<const N: usize>(idxs: [u32; N]) -> v128 {
    let p = idxs.as_ptr();
    let lim = u32x4_splat(N as u32);

    // SAFETY: each load reads only lanes that exist in `[u32; N]`.
    let bytes = unsafe {
        if const { N == 16 } {
            let a = u32x4_min(v128_load(p as *const v128), lim);
            let b = u32x4_min(v128_load(p.add(4) as *const v128), lim);
            let c = u32x4_min(v128_load(p.add(8) as *const v128), lim);
            let d = u32x4_min(v128_load(p.add(12) as *const v128), lim);
            u8x16_narrow_i16x8(u16x8_narrow_i32x4(a, b), u16x8_narrow_i32x4(c, d))
        } else if const { N == 8 } {
            let a = u32x4_min(v128_load(p as *const v128), lim);
            let b = u32x4_min(v128_load(p.add(4) as *const v128), lim);
            let w = u16x8_narrow_i32x4(a, b);
            u8x16_narrow_i16x8(w, w)
        } else if const { N == 4 } {
            let a = u32x4_min(v128_load(p as *const v128), lim);
            let w = u16x8_narrow_i32x4(a, a);
            u8x16_narrow_i16x8(w, w)
        } else {
            // N == 2: only a 64-bit load is in bounds.
            let a = u32x4_min(v128_load64_zero(p as *const u64), lim);
            let w = u16x8_narrow_i32x4(a, a);
            u8x16_narrow_i16x8(w, w)
        }
    };

    // Byte lanes: the index already IS the byte index.
    if const { N == 16 } {
        return bytes;
    }

    wasm_lane_expand::<N>(bytes)
}

/// Replicate/scale/offset tail of [`wasm_lane_table_dyn`], on already-clamped
/// byte lane indices.
#[inline(always)]
fn wasm_lane_expand<const N: usize>(bytes: v128) -> v128 {
    if const { N == 16 } {
        return bytes;
    }

    let expanded = u8x16_swizzle(bytes, const { u8x16_from_bytes(lane_expand_pattern(16 / N)) });
    let scaled = u8x16_shl(expanded, const { (16 / N).trailing_zeros() });

    u8x16_add(scaled, const { u8x16_from_bytes(lane_offset_pattern(16 / N)) })
}

/// Byte-row entry point for the compress/expand table paths - see
/// `neon_lane_table_row` for why the narrow and clamp are both unnecessary.
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
