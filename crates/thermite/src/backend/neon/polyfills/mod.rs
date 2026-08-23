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

// `prfm` is not a NEON instruction and needs no register plumbing, so the shared,
// target-generic implementation is simply re-exported into `arch::`.
pub use crate::backend::prefetch::{HAS_PREFETCH, prefetch};

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

/// `expand[j] = j / elem` - replicates each lane index across its `elem` byte
/// slots. Constant for a given register shape.
const fn lane_expand_pattern(elem: usize) -> [u8; 16] {
    let mut o = [0u8; 16];
    let mut j = 0;
    while j < 16 {
        o[j] = (j / elem) as u8;
        j += 1;
    }
    o
}

/// `offsets[j] = j % elem` - the intra-lane byte offsets added after scaling.
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
// Runtime `tbl` control from a LIVE index register (`neon_ctrl_x{16,8,4,2}`).
//
// `Register::permutev`/`swizzle` take the indices as a same-lane-count unsigned
// register, so the whole control build stays in registers, with no memory round
// trip and no scalar byte inserts.
//
// The mapping is `byte[i * elem + b] = idx[i] * elem + b`. Replicating a
// sub-256 value into every byte of its lane is one multiply by `0x01..01` in
// that lane width, so:
//
//   ctrl = ((idx << log2(elem)) * 0x01..01) + lane_offset_pattern(elem)
//
// three instructions (shift, multiply, byte add) plus a constant load, for
// every width except bytes, where the index register already IS the control.
//
// No clamp is needed. An index one past the addressable range scales to byte
// `AVAIL * elem` (16 for `tbl`, 32 for `tbl2`, `16 * M` for the multi-register
// forms), which the consuming table lookup zeroes. A wildly out-of-range index
// produces some other unspecified byte pattern, which the `permutev` contract
// explicitly permits. Every result is a byte shuffle, so all of it is
// memory-safe by construction.
// ---------------------------------------------------------------------------

/// 16 byte lanes: the index register is the `tbl` control already.
#[inline(always)]
pub fn neon_ctrl_x16(idxs: uint8x16_t) -> uint8x16_t {
    idxs
}

/// 8 lanes of 16 bits: `(idx << 1) * 0x0101` puts `2i` in both bytes of the
/// lane, and the byte offsets turn them into `(2i, 2i + 1)`.
#[inline(always)]
pub fn neon_ctrl_x8(idxs: uint16x8_t) -> uint8x16_t {
    unsafe {
        let rep = vmulq_u16(vshlq_n_u16::<1>(idxs), vdupq_n_u16(0x0101));
        vaddq_u8(vreinterpretq_u8_u16(rep), const { cu8x16(lane_offset_pattern(2)) })
    }
}

/// 4 lanes of 32 bits: `(idx << 2) * 0x01010101`, then the byte offsets.
#[inline(always)]
pub fn neon_ctrl_x4(idxs: uint32x4_t) -> uint8x16_t {
    unsafe {
        let rep = vmulq_u32(vshlq_n_u32::<2>(idxs), vdupq_n_u32(0x0101_0101));
        vaddq_u8(vreinterpretq_u8_u32(rep), const { cu8x16(lane_offset_pattern(4)) })
    }
}

/// 2 lanes of 64 bits. NEON has no 64-bit multiply, so the replication runs in
/// `u32` lanes instead: the reinterpret gives `[lo0, hi0, lo1, hi1]` and only
/// the low word of each pair carries an in-range index, so scaling and
/// replicating those two words and copying each over its own high word (`TRN1`)
/// fills all 8 bytes of the lane. One instruction more than the other widths.
#[inline(always)]
pub fn neon_ctrl_x2(idxs: uint64x2_t) -> uint8x16_t {
    unsafe {
        let lo = vmulq_u32(vshlq_n_u32::<3>(vreinterpretq_u32_u64(idxs)), vdupq_n_u32(0x0101_0101));
        let rep = vtrn1q_u32(lo, lo);
        vaddq_u8(vreinterpretq_u8_u32(rep), const { cu8x16(lane_offset_pattern(8)) })
    }
}

/// Replicate/scale/offset tail used by the byte-row path: on already-narrowed,
/// in-range byte lane indices, replicate each index across its `elem` byte
/// slots, scale, offset.
#[inline(always)]
unsafe fn neon_lane_expand<const N: usize>(clamped: uint8x16_t) -> uint8x16_t {
    unsafe {
        if const { N == 16 } {
            return clamped;
        }

        let expanded = vqtbl1q_u8(clamped, const { cu8x16(lane_expand_pattern(16 / N)) });
        let scaled = match const { 16 / N } {
            2 => vshlq_n_u8::<1>(expanded),
            4 => vshlq_n_u8::<2>(expanded),
            _ => vshlq_n_u8::<3>(expanded),
        };

        vaddq_u8(scaled, const { cu8x16(lane_offset_pattern(16 / N)) })
    }
}

/// Byte-row entry point for the compress/expand table paths.
///
/// A `COMPRESS8`/`EXPAND8` row is already byte-sized lane indices, and its first
/// `LANES` entries are always `< LANES` (the padding lanes sort past them), so
/// no narrow and no clamp apply here. The row goes straight into the
/// replicate/scale/offset tail. That removes the `u8 -> u32 -> u8` round trip
/// those paths would otherwise make through
/// [`Register::widen_index_bytes`](crate::register::Register::widen_index_bytes)
/// and back.
///
/// # Safety
///
/// `row` must point to at least 8 readable bytes, all `< N`.
#[inline(always)]
pub unsafe fn neon_lane_table_row<const N: usize>(row: *const u8) -> uint8x16_t {
    unsafe {
        let b = vld1_u8(row);
        neon_lane_expand::<N>(vcombine_u8(b, b))
    }
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
