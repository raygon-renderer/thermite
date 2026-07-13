#![allow(clippy::identity_op, unsafe_op_in_unsafe_fn, clippy::missing_safety_doc)]

//! NEON polyfills + the per-type "normalization" layer.
//!
//! NEON has a distinct storage type per element (`float32x4_t`, `int16x8_t`,
//! ...), unlike wasm's single `v128`. The `bits`/`cmp` modules stamp a small,
//! uniformly-named function set per type suffix (`neon_and_f32`,
//! `neon_movemask_u32`, `neon_all_u8`, ...) so the register macros can be
//! written once against `arch::[<neon_op_ $suffix>]` names, with all the
//! `vreinterpretq` plumbing centralized here. Reinterprets are zero-cost; the
//! compiler erases them entirely.

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

// ---------------------------------------------------------------------------
// Const vector constructors. NEON intrinsics are not `const fn`, so constants
// are built by transmuting element arrays (the same mechanism `reg::<R, N>`
// uses at the register layer).
// ---------------------------------------------------------------------------

macro_rules! decl_const_ctors {
    ($($name:ident: [$e:ty; $n:literal] => $ty:ty),* $(,)?) => {$(
        #[inline(always)]
        pub const fn $name(values: [$e; $n]) -> $ty {
            // SAFETY: same size; const_transmute handles alignment.
            unsafe { crate::generic_array::const_transmute(values) }
        }
    )*};
}

decl_const_ctors! {
    cu8x16: [u8; 16] => uint8x16_t,
    cu16x8: [u16; 8] => uint16x8_t,
    cu32x4: [u32; 4] => uint32x4_t,
    cu64x2: [u64; 2] => uint64x2_t,
}

/// Build a `vqtbl1q_u8` byte-index table for a per-lane permutation of a
/// 128-bit register with `elem_size`-byte lanes. Any source index `>= 16 /
/// elem_size` produces out-of-range byte indices, which `tbl` zeroes - the
/// same semantics as wasm's `i8x16.swizzle`.
///
/// `const fn`, but also callable at runtime for `permutev`.
#[inline(always)]
pub const fn neon_lane_table<const N: usize>(elem_size: usize, idxs: [u32; N]) -> uint8x16_t {
    let mut out = [0xFFu8; 16];
    let mut i = 0;
    while i < N {
        let base = idxs[i] as usize * elem_size;
        let mut b = 0;
        while b < elem_size {
            // Saturate instead of wrapping so an out-of-range lane index stays
            // out of range for `tbl` (-> zeroed) rather than aliasing a lane.
            out[i * elem_size + b] = if base + b > 255 { 0xFF } else { (base + b) as u8 };
            b += 1;
        }
        i += 1;
    }
    cu8x16(out)
}

/// Byte-index table selecting lane `idx & (N-1)` of each pair of lanes from a
/// 4-lane register, from an x86-style 2-bit-per-lane `IMM8` (`_mm_shuffle_ps`
/// index encoding).
#[inline(always)]
pub const fn neon_imm8x4_to_table<const IMM8: i32>() -> uint8x16_t {
    neon_lane_table::<4>(
        4,
        [
            ((IMM8 >> 0) & 0b11) as u32,
            ((IMM8 >> 2) & 0b11) as u32,
            ((IMM8 >> 4) & 0b11) as u32,
            ((IMM8 >> 6) & 0b11) as u32,
        ],
    )
}

/// Same as [`neon_imm8x4_to_table`] for a 2-lane (64-bit element) register.
#[inline(always)]
pub const fn neon_imm8x2_to_table<const IMM8: i32>() -> uint8x16_t {
    neon_lane_table::<2>(8, [((IMM8 >> 0) & 0b1) as u32, ((IMM8 >> 1) & 0b1) as u32])
}

const fn imm_bit(imm8: i32, b: i32) -> u64 {
    if (imm8 >> b) & 1 != 0 { !0 } else { 0 }
}

/// Per-lane select mask (all-ones where the IMM8 bit is set) for a 4-lane
/// register, as raw `u32` lanes.
#[inline(always)]
pub const fn neon_imm8x4_to_mask<const IMM8: i32>() -> uint32x4_t {
    cu32x4([
        imm_bit(IMM8, 0) as u32,
        imm_bit(IMM8, 1) as u32,
        imm_bit(IMM8, 2) as u32,
        imm_bit(IMM8, 3) as u32,
    ])
}

/// Per-lane select mask for a 2-lane register, as raw `u64` lanes.
#[inline(always)]
pub const fn neon_imm8x2_to_mask<const IMM8: i32>() -> uint64x2_t {
    cu64x2([imm_bit(IMM8, 0), imm_bit(IMM8, 1)])
}
