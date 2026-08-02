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

/// Runtime companion to [`neon_lane_table`]: the same `tbl` byte-index table,
/// built with SIMD instead of a scalar loop.
///
/// [`neon_lane_table`] is a `const fn`, which is exactly right for callers that
/// can evaluate it at compile time - `swizzle_const` and the `IMM8` shuffle
/// family, where it folds to a literal (measured: a 4-lane `swizzle_const` is 5
/// instructions, and LLVM often recognises the pattern and skips `tbl`
/// entirely). But const-evaluable code has no SIMD, so on the two *runtime*
/// callers - `permutev` and `swizzle` - it lowered verbatim to ~80 scalar
/// instructions: a saturating `cmp`/`csel` clamp plus a `mov v0.b[i]` insert for
/// every one of the 16 output bytes. Measured on a 4-lane runtime permute:
/// 83 instructions (plus a stack frame) before, 17 after.
///
/// The mapping `byte[j] = idx[j / elem] * elem + (j % elem)` is pure lane work:
///
/// 1. saturating-narrow the `u32` indices to bytes,
/// 2. clamp to `AVAIL` (see below) so an out-of-range lane index lands exactly
///    on the first byte the table lookup cannot address, which it zeroes -
///    preserving [`neon_lane_table`]'s documented out-of-range semantics,
/// 3. replicate each index across its `elem` byte slots (one `tbl` with a
///    constant pattern),
/// 4. scale by `elem` and add the intra-lane offsets (both constants).
///
/// `AVAIL` is the number of lanes the consuming `tbl` can actually address, and
/// is what step 2 clamps to: `N` for the single-register [`permutev`] form, but
/// `2 * N` for the two-register `swizzle` form, whose `tbl2` addresses 32 bytes
/// and for which indices in `N..2*N` are *valid* selections from the second
/// register. Clamping to `AVAIL` puts an out-of-range index on byte
/// `AVAIL * elem` (16 or 32 respectively), which the corresponding `tbl` zeroes.
///
/// Shapes outside `{2, 4, 8, 16}` lanes (`elem` would not divide 16) fall back
/// to the scalar builder, which is still correct.
#[inline(always)]
pub unsafe fn neon_lane_table_dyn<const N: usize, const AVAIL: usize>(idxs: [u32; N]) -> uint8x16_t {
    unsafe {
        if const { !(N == 2 || N == 4 || N == 8 || N == 16) } {
            return neon_lane_table::<N>(16 / N, idxs);
        }

        let p = idxs.as_ptr();

        // 1. Saturating narrow to bytes. Saturation keeps a wildly out-of-range
        //    index out of range rather than aliasing a valid lane.
        let bytes: uint8x16_t = if const { N == 16 } {
            let lo = vqmovn_u16(vcombine_u16(vqmovn_u32(vld1q_u32(p)), vqmovn_u32(vld1q_u32(p.add(4)))));
            let hi = vqmovn_u16(vcombine_u16(vqmovn_u32(vld1q_u32(p.add(8))), vqmovn_u32(vld1q_u32(p.add(12)))));
            vcombine_u8(lo, hi)
        } else if const { N == 8 } {
            let b = vqmovn_u16(vcombine_u16(vqmovn_u32(vld1q_u32(p)), vqmovn_u32(vld1q_u32(p.add(4)))));
            vcombine_u8(b, b)
        } else if const { N == 4 } {
            let b = vqmovn_u16(vcombine_u16(vqmovn_u32(vld1q_u32(p)), vdup_n_u16(0)));
            vcombine_u8(b, b)
        } else {
            // N == 2: only a 64-bit load is in bounds.
            let v = vcombine_u32(vld1_u32(p), vdup_n_u32(0));
            let b = vqmovn_u16(vcombine_u16(vqmovn_u32(v), vdup_n_u16(0)));
            vcombine_u8(b, b)
        };

        // 2. Clamp: index AVAIL scales to byte AVAIL*elem (16 for tbl, 32 for
        //    tbl2), which the consuming table lookup treats as out of range.
        let clamped = vminq_u8(bytes, vdupq_n_u8(const { AVAIL as u8 }));

        // 3-4. Byte registers need no replication, scaling, or offsetting.
        if const { N == 16 } {
            return clamped;
        }

        neon_lane_expand::<N>(clamped)
    }
}

/// Steps 3-4 of [`neon_lane_table_dyn`], on already-clamped byte lane indices:
/// replicate each index across its `elem` byte slots, scale, offset.
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
/// neither the saturating narrow nor the clamp that [`neon_lane_table_dyn`]
/// needs applies here - the row goes straight into the replicate/scale/offset
/// tail. That removes the `u8 -> u32 -> u8` round trip those paths would
/// otherwise make through `widen_indices` and back.
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
