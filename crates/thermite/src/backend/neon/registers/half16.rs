//! Reduced (4-lane) 16-bit registers for the NEON backend and the cast/concat glue bridging
//! the scalar/array 16-bit halves, the native 128-bit `I16x8Neon`/`U16x8Neon`, and the 32-bit
//! registers they widen into. Mirrors the wasm backend's `half16.rs` structure with NEON
//! intrinsics (`vmovl`/`vmovn` widen/truncate, `vqmovn` saturating narrows).

use generic_array::typenum::U4;

use super::arch;

use crate::register::{
    CastRegister, ConcatRegister, ExtendRegister, IndexableRegister, SaturatingCastRegister, Storage,
    array::ArrayRegister, reduced::ReducedRegister, reg,
};

// --- saturating narrows into 16-bit ---

// i32x4 -> i16x4: native `vqmovn_s32` (low half holds the result).
#[thermite_macros::inline_always]
impl SaturatingCastRegister<super::I32x4Neon> for I16x4Neon {
    fn saturating_cast_from(value: Storage<super::I32x4Neon>) -> Storage<Self> {
        unsafe { ReducedRegister::new(arch::vcombine_s16(arch::vqmovn_s32(value), arch::vdup_n_s16(0))) }
    }
}
// u32x4 -> u16x4: native `vqmovn_u32`. NEON's unsigned narrow saturates the unsigned source
// correctly by itself, so the `min(value, 0xFFFF)` pre-clamp wasm needs (its narrow reads a
// signed source) is dropped.
#[thermite_macros::inline_always]
impl SaturatingCastRegister<super::U32x4Neon> for U16x4Neon {
    fn saturating_cast_from(value: Storage<super::U32x4Neon>) -> Storage<Self> {
        unsafe { ReducedRegister::new(arch::vcombine_u16(arch::vqmovn_u32(value), arch::vdup_n_u16(0))) }
    }
}

// The 64-bit sources and the 2-lane scalar-array destinations. wasm has no 64-bit narrow and
// clamps into range (register `min`/`max`) + truncating narrow; NEON chains its native
// saturating narrows (`vqmovn_s64`/`vqmovn_s32`, unsigned analogs) instead.

// Saturating-narrow 4x i64x2 (8 i64) -> one i16x8 via chained `vqmovn`.
#[inline(always)]
fn sat_narrow_4xi64x2_to_i16x8(v: [arch::int64x2_t; 4]) -> arch::int16x8_t {
    unsafe {
        let q0 = arch::vqmovn_high_s64(arch::vqmovn_s64(v[0]), v[1]);
        let q1 = arch::vqmovn_high_s64(arch::vqmovn_s64(v[2]), v[3]);
        arch::vqmovn_high_s32(arch::vqmovn_s32(q0), q1)
    }
}
#[inline(always)]
fn sat_narrow_4xu64x2_to_u16x8(v: [arch::uint64x2_t; 4]) -> arch::uint16x8_t {
    unsafe {
        let q0 = arch::vqmovn_high_u64(arch::vqmovn_u64(v[0]), v[1]);
        let q1 = arch::vqmovn_high_u64(arch::vqmovn_u64(v[2]), v[3]);
        arch::vqmovn_high_u32(arch::vqmovn_u32(q0), q1)
    }
}

// i64x4 -> i16x4 (saturating): two `vqmovn_s64` + one `vqmovn_s32`.
#[thermite_macros::inline_always]
impl SaturatingCastRegister<ArrayRegister<super::I64x2Neon, 2>> for I16x4Neon {
    fn saturating_cast_from(value: Storage<ArrayRegister<super::I64x2Neon, 2>>) -> Storage<Self> {
        unsafe {
            let q32 = arch::vqmovn_high_s64(arch::vqmovn_s64(value.0[0]), value.0[1]);
            ReducedRegister::new(arch::vcombine_s16(arch::vqmovn_s32(q32), arch::vdup_n_s16(0)))
        }
    }
}
#[thermite_macros::inline_always]
impl SaturatingCastRegister<ArrayRegister<super::U64x2Neon, 2>> for U16x4Neon {
    fn saturating_cast_from(value: Storage<ArrayRegister<super::U64x2Neon, 2>>) -> Storage<Self> {
        unsafe {
            let q32 = arch::vqmovn_high_u64(arch::vqmovn_u64(value.0[0]), value.0[1]);
            ReducedRegister::new(arch::vcombine_u16(arch::vqmovn_u32(q32), arch::vdup_n_u16(0)))
        }
    }
}

// i32x2 -> i16x2 (saturating): `vqmovn_s32` then extract the low two lanes.
#[thermite_macros::inline_always]
impl SaturatingCastRegister<super::half::I32x2Neon> for ArrayRegister<i16, 2> {
    fn saturating_cast_from(value: Storage<super::half::I32x2Neon>) -> Storage<Self> {
        unsafe {
            let n = arch::vqmovn_s32(value.0);
            ArrayRegister([arch::vget_lane_s16::<0>(n), arch::vget_lane_s16::<1>(n)])
        }
    }
}
#[thermite_macros::inline_always]
impl SaturatingCastRegister<super::half::U32x2Neon> for ArrayRegister<u16, 2> {
    fn saturating_cast_from(value: Storage<super::half::U32x2Neon>) -> Storage<Self> {
        unsafe {
            let n = arch::vqmovn_u32(value.0);
            ArrayRegister([arch::vget_lane_u16::<0>(n), arch::vget_lane_u16::<1>(n)])
        }
    }
}

// i64x2 -> i16x2 (saturating): `vqmovn_s64` then `vqmovn_s32`, extract the low two lanes.
#[thermite_macros::inline_always]
impl SaturatingCastRegister<super::I64x2Neon> for ArrayRegister<i16, 2> {
    fn saturating_cast_from(value: Storage<super::I64x2Neon>) -> Storage<Self> {
        unsafe {
            let n32 = arch::vqmovn_s64(value);
            let n16 = arch::vqmovn_s32(arch::vcombine_s32(n32, n32));
            ArrayRegister([arch::vget_lane_s16::<0>(n16), arch::vget_lane_s16::<1>(n16)])
        }
    }
}
#[thermite_macros::inline_always]
impl SaturatingCastRegister<super::U64x2Neon> for ArrayRegister<u16, 2> {
    fn saturating_cast_from(value: Storage<super::U64x2Neon>) -> Storage<Self> {
        unsafe {
            let n32 = arch::vqmovn_u64(value);
            let n16 = arch::vqmovn_u32(arch::vcombine_u32(n32, n32));
            ArrayRegister([arch::vget_lane_u16::<0>(n16), arch::vget_lane_u16::<1>(n16)])
        }
    }
}

// i64x8 -> i16x8 (saturating): chained `vqmovn`. NOTE: the wasm backend hosts this pair in
// i16x8.rs (as a clamp + truncating narrow there); on NEON it lives here to share the
// chained-`vqmovn` helper with the x16 impls below.
#[thermite_macros::inline_always]
impl SaturatingCastRegister<ArrayRegister<super::I64x2Neon, 4>> for super::I16x8Neon {
    fn saturating_cast_from(value: Storage<ArrayRegister<super::I64x2Neon, 4>>) -> Storage<Self> {
        sat_narrow_4xi64x2_to_i16x8(value.0)
    }
}
#[thermite_macros::inline_always]
impl SaturatingCastRegister<ArrayRegister<super::U64x2Neon, 4>> for super::U16x8Neon {
    fn saturating_cast_from(value: Storage<ArrayRegister<super::U64x2Neon, 4>>) -> Storage<Self> {
        sat_narrow_4xu64x2_to_u16x8(value.0)
    }
}

// i64x16 -> i16x16 (saturating): the chained-`vqmovn` helper per 8-lane half.
#[thermite_macros::inline_always]
impl SaturatingCastRegister<ArrayRegister<super::I64x2Neon, 8>> for ArrayRegister<super::I16x8Neon, 2> {
    fn saturating_cast_from(value: Storage<ArrayRegister<super::I64x2Neon, 8>>) -> Storage<Self> {
        let v = value.0;
        ArrayRegister([
            sat_narrow_4xi64x2_to_i16x8([v[0], v[1], v[2], v[3]]),
            sat_narrow_4xi64x2_to_i16x8([v[4], v[5], v[6], v[7]]),
        ])
    }
}
#[thermite_macros::inline_always]
impl SaturatingCastRegister<ArrayRegister<super::U64x2Neon, 8>> for ArrayRegister<super::U16x8Neon, 2> {
    fn saturating_cast_from(value: Storage<ArrayRegister<super::U64x2Neon, 8>>) -> Storage<Self> {
        let v = value.0;
        ArrayRegister([
            sat_narrow_4xu64x2_to_u16x8([v[0], v[1], v[2], v[3]]),
            sat_narrow_4xu64x2_to_u16x8([v[4], v[5], v[6], v[7]]),
        ])
    }
}

/// 4-lane signed 16-bit register, backed by the low 4 lanes of a 128-bit `I16x8Neon`.
pub type I16x4Neon = ReducedRegister<super::I16x8Neon, U4>;
/// 4-lane unsigned 16-bit register, backed by the low 4 lanes of a 128-bit `U16x8Neon`.
pub type U16x4Neon = ReducedRegister<super::U16x8Neon, U4>;

// ---------------------------------------------------------------------------------------
// x4 <- x2 : build a 4-lane reduced register from two 2-lane (scalar-array) halves.
// ---------------------------------------------------------------------------------------

#[thermite_macros::inline_always]
impl ConcatRegister<ArrayRegister<i16, 2>> for I16x4Neon {
    fn concat(lo: Storage<ArrayRegister<i16, 2>>, hi: Storage<ArrayRegister<i16, 2>>) -> Storage<Self> {
        ReducedRegister::new(reg::<super::I16x8Neon, 8>([
            lo.0[0], lo.0[1], hi.0[0], hi.0[1], 0, 0, 0, 0,
        ]))
    }

    fn split(value: Storage<Self>) -> (Storage<ArrayRegister<i16, 2>>, Storage<ArrayRegister<i16, 2>>) {
        unsafe {
            (
                ArrayRegister([arch::vgetq_lane_s16::<0>(value.0), arch::vgetq_lane_s16::<1>(value.0)]),
                ArrayRegister([arch::vgetq_lane_s16::<2>(value.0), arch::vgetq_lane_s16::<3>(value.0)]),
            )
        }
    }
}

#[thermite_macros::inline_always]
impl ExtendRegister<ArrayRegister<i16, 2>> for I16x4Neon {
    fn extend(value: Storage<ArrayRegister<i16, 2>>) -> Storage<Self> {
        ReducedRegister::new(reg::<super::I16x8Neon, 8>([value.0[0], value.0[1], 0, 0, 0, 0, 0, 0]))
    }

    fn narrow(value: Storage<Self>) -> Storage<ArrayRegister<i16, 2>> {
        unsafe { ArrayRegister([arch::vgetq_lane_s16::<0>(value.0), arch::vgetq_lane_s16::<1>(value.0)]) }
    }
}

#[thermite_macros::inline_always]
impl ConcatRegister<ArrayRegister<u16, 2>> for U16x4Neon {
    fn concat(lo: Storage<ArrayRegister<u16, 2>>, hi: Storage<ArrayRegister<u16, 2>>) -> Storage<Self> {
        ReducedRegister::new(reg::<super::U16x8Neon, 8>([
            lo.0[0], lo.0[1], hi.0[0], hi.0[1], 0, 0, 0, 0,
        ]))
    }

    fn split(value: Storage<Self>) -> (Storage<ArrayRegister<u16, 2>>, Storage<ArrayRegister<u16, 2>>) {
        unsafe {
            (
                ArrayRegister([arch::vgetq_lane_u16::<0>(value.0), arch::vgetq_lane_u16::<1>(value.0)]),
                ArrayRegister([arch::vgetq_lane_u16::<2>(value.0), arch::vgetq_lane_u16::<3>(value.0)]),
            )
        }
    }
}

#[thermite_macros::inline_always]
impl ExtendRegister<ArrayRegister<u16, 2>> for U16x4Neon {
    fn extend(value: Storage<ArrayRegister<u16, 2>>) -> Storage<Self> {
        ReducedRegister::new(reg::<super::U16x8Neon, 8>([value.0[0], value.0[1], 0, 0, 0, 0, 0, 0]))
    }

    fn narrow(value: Storage<Self>) -> Storage<ArrayRegister<u16, 2>> {
        unsafe { ArrayRegister([arch::vgetq_lane_u16::<0>(value.0), arch::vgetq_lane_u16::<1>(value.0)]) }
    }
}

// ---------------------------------------------------------------------------------------
// x8 <- x4 : build the native 8-lane register from two 4-lane reduced halves.
// Both halves live in the low 64 bits of their q-register; a u64-lane zip merges them,
// and `vextq_u64::<1>` shifts the high half down for `split`.
// ---------------------------------------------------------------------------------------

#[thermite_macros::inline_always]
impl ConcatRegister<I16x4Neon> for super::I16x8Neon {
    fn concat(lo: Storage<I16x4Neon>, hi: Storage<I16x4Neon>) -> Storage<Self> {
        unsafe {
            arch::vreinterpretq_s16_u64(arch::vzip1q_u64(
                arch::vreinterpretq_u64_s16(lo.0),
                arch::vreinterpretq_u64_s16(hi.0),
            ))
        }
    }

    fn split(value: Storage<Self>) -> (Storage<I16x4Neon>, Storage<I16x4Neon>) {
        unsafe {
            (
                ReducedRegister::new(value),
                ReducedRegister::new(arch::vreinterpretq_s16_u64(arch::vextq_u64::<1>(
                    arch::vreinterpretq_u64_s16(value),
                    arch::vreinterpretq_u64_s16(value),
                ))),
            )
        }
    }
}

// NOTE: `ExtendRegister<I16x4Neon> for I16x8Neon` is provided by the ReducedRegister blanket.

#[thermite_macros::inline_always]
impl ConcatRegister<U16x4Neon> for super::U16x8Neon {
    fn concat(lo: Storage<U16x4Neon>, hi: Storage<U16x4Neon>) -> Storage<Self> {
        unsafe {
            arch::vreinterpretq_u16_u64(arch::vzip1q_u64(
                arch::vreinterpretq_u64_u16(lo.0),
                arch::vreinterpretq_u64_u16(hi.0),
            ))
        }
    }

    fn split(value: Storage<Self>) -> (Storage<U16x4Neon>, Storage<U16x4Neon>) {
        unsafe {
            (
                ReducedRegister::new(value),
                ReducedRegister::new(arch::vreinterpretq_u16_u64(arch::vextq_u64::<1>(
                    arch::vreinterpretq_u64_u16(value),
                    arch::vreinterpretq_u64_u16(value),
                ))),
            )
        }
    }
}

// NOTE: `ExtendRegister<U16x4Neon> for U16x8Neon` is provided by the ReducedRegister blanket.

// ---------------------------------------------------------------------------------------
// Widen casts to 32-bit. x4: I16x4Neon <-> I32x4Neon.  x2: ArrayRegister<i16,2> <-> I32x2Neon.
// ---------------------------------------------------------------------------------------

// --- x4 widen i16 -> i32 (sign/zero-extend the low 4 lanes via `vmovl`) ---
#[thermite_macros::inline_always]
impl CastRegister<I16x4Neon> for super::I32x4Neon {
    fn cast_from(value: Storage<I16x4Neon>) -> Storage<Self> {
        unsafe { arch::vmovl_s16(arch::vget_low_s16(value.0)) }
    }
}

#[thermite_macros::inline_always]
impl CastRegister<U16x4Neon> for super::U32x4Neon {
    fn cast_from(value: Storage<U16x4Neon>) -> Storage<Self> {
        unsafe { arch::vmovl_u16(arch::vget_low_u16(value.0)) }
    }
}

// --- x4 narrow i32 -> i16 (truncate low 16 bits per lane via `vmovn`, wasm needs a byte
//     shuffle here) ---
#[thermite_macros::inline_always]
impl CastRegister<super::I32x4Neon> for I16x4Neon {
    fn cast_from(value: Storage<super::I32x4Neon>) -> Storage<Self> {
        unsafe { ReducedRegister::new(arch::vcombine_s16(arch::vmovn_s32(value), arch::vdup_n_s16(0))) }
    }
}

#[thermite_macros::inline_always]
impl CastRegister<super::U32x4Neon> for U16x4Neon {
    fn cast_from(value: Storage<super::U32x4Neon>) -> Storage<Self> {
        unsafe { ReducedRegister::new(arch::vcombine_u16(arch::vmovn_u32(value), arch::vdup_n_u16(0))) }
    }
}

// --- x2 widen i16 -> i32 (ArrayRegister<i16,2> -> I32x2Neon reduced) ---
#[thermite_macros::inline_always]
impl CastRegister<ArrayRegister<i16, 2>> for super::half::I32x2Neon {
    fn cast_from(value: Storage<ArrayRegister<i16, 2>>) -> Storage<Self> {
        ReducedRegister::new(reg::<super::I32x4Neon, 4>([value.0[0] as i32, value.0[1] as i32, 0, 0]))
    }
}

#[thermite_macros::inline_always]
impl CastRegister<ArrayRegister<u16, 2>> for super::half::U32x2Neon {
    fn cast_from(value: Storage<ArrayRegister<u16, 2>>) -> Storage<Self> {
        ReducedRegister::new(reg::<super::U32x4Neon, 4>([value.0[0] as u32, value.0[1] as u32, 0, 0]))
    }
}

// --- x2 narrow i32 -> i16 (I32x2Neon reduced -> ArrayRegister<i16,2>) ---
#[thermite_macros::inline_always]
impl CastRegister<super::half::I32x2Neon> for ArrayRegister<i16, 2> {
    fn cast_from(value: Storage<super::half::I32x2Neon>) -> Storage<Self> {
        unsafe {
            ArrayRegister([
                arch::vgetq_lane_s32::<0>(value.0) as i16,
                arch::vgetq_lane_s32::<1>(value.0) as i16,
            ])
        }
    }
}

#[thermite_macros::inline_always]
impl CastRegister<super::half::U32x2Neon> for ArrayRegister<u16, 2> {
    fn cast_from(value: Storage<super::half::U32x2Neon>) -> Storage<Self> {
        unsafe {
            ArrayRegister([
                arch::vgetq_lane_u32::<0>(value.0) as u16,
                arch::vgetq_lane_u32::<1>(value.0) as u16,
            ])
        }
    }
}

// ---------------------------------------------------------------------------------------
// Widen/narrow casts to 64-bit (CastRegister = numeric widen / `as`-style narrow).
//
//   widen 16 -> 64: two chained `vmovl` steps (16 -> 32 -> 64), sign-extending for the
//     signed source and zero-extending for the unsigned one. (wasm stores the words and
//     rebuilds each lane pair scalar-wise; NEON stays fully in registers.)
//   narrow 64 -> 16: chained truncating `vmovn` (64 -> 32 -> 16, wrapping Rust `as`).
//   x2:  ArrayRegister<i16,2> <-> I64x2Neon (native 2-lane).
//   x4:  I16x4Neon <-> ArrayRegister<I64x2Neon, 2> (the emulated i64x4).
//   x8:  I16x8Neon <-> ArrayRegister<I64x2Neon, 4> (the emulated i64x8).
//   x16: ArrayRegister<I16x8Neon, 2> <-> ArrayRegister<I64x2Neon, 8> (i16x16 / i64x16).
// ---------------------------------------------------------------------------------------

// Widen one i16x8 (8 words) -> 4x i64x2 via two chained `vmovl` steps.
#[inline(always)]
fn widen_i16x8_to_4xi64x2(v: arch::int16x8_t) -> [arch::int64x2_t; 4] {
    unsafe {
        let lo = arch::vmovl_s16(arch::vget_low_s16(v));
        let hi = arch::vmovl_high_s16(v);
        [
            arch::vmovl_s32(arch::vget_low_s32(lo)),
            arch::vmovl_high_s32(lo),
            arch::vmovl_s32(arch::vget_low_s32(hi)),
            arch::vmovl_high_s32(hi),
        ]
    }
}
#[inline(always)]
fn widen_u16x8_to_4xu64x2(v: arch::uint16x8_t) -> [arch::uint64x2_t; 4] {
    unsafe {
        let lo = arch::vmovl_u16(arch::vget_low_u16(v));
        let hi = arch::vmovl_high_u16(v);
        [
            arch::vmovl_u32(arch::vget_low_u32(lo)),
            arch::vmovl_high_u32(lo),
            arch::vmovl_u32(arch::vget_low_u32(hi)),
            arch::vmovl_high_u32(hi),
        ]
    }
}

// Narrow 4x i64x2 (8 i64) -> one i16x8 via chained truncating `vmovn` (wrapping `as`).
#[inline(always)]
fn narrow_4xi64x2_to_i16x8(v: [arch::int64x2_t; 4]) -> arch::int16x8_t {
    unsafe {
        let q0 = arch::vmovn_high_s64(arch::vmovn_s64(v[0]), v[1]);
        let q1 = arch::vmovn_high_s64(arch::vmovn_s64(v[2]), v[3]);
        arch::vmovn_high_s32(arch::vmovn_s32(q0), q1)
    }
}
#[inline(always)]
fn narrow_4xu64x2_to_u16x8(v: [arch::uint64x2_t; 4]) -> arch::uint16x8_t {
    unsafe {
        let q0 = arch::vmovn_high_u64(arch::vmovn_u64(v[0]), v[1]);
        let q1 = arch::vmovn_high_u64(arch::vmovn_u64(v[2]), v[3]);
        arch::vmovn_high_u32(arch::vmovn_u32(q0), q1)
    }
}

// --- x2 widen i16 -> i64 (ArrayRegister<i16,2> -> I64x2Neon native) ---
#[thermite_macros::inline_always]
impl CastRegister<ArrayRegister<i16, 2>> for super::I64x2Neon {
    fn cast_from(value: Storage<ArrayRegister<i16, 2>>) -> Storage<Self> {
        reg::<super::I64x2Neon, 2>([value.0[0] as i64, value.0[1] as i64])
    }
}
#[thermite_macros::inline_always]
impl CastRegister<ArrayRegister<u16, 2>> for super::U64x2Neon {
    fn cast_from(value: Storage<ArrayRegister<u16, 2>>) -> Storage<Self> {
        reg::<super::U64x2Neon, 2>([value.0[0] as u64, value.0[1] as u64])
    }
}

// --- x2 narrow i64 -> i16 (I64x2Neon native -> ArrayRegister<i16,2>) ---
#[thermite_macros::inline_always]
impl CastRegister<super::I64x2Neon> for ArrayRegister<i16, 2> {
    fn cast_from(value: Storage<super::I64x2Neon>) -> Storage<Self> {
        unsafe {
            ArrayRegister([
                arch::vgetq_lane_s64::<0>(value) as i16,
                arch::vgetq_lane_s64::<1>(value) as i16,
            ])
        }
    }
}
#[thermite_macros::inline_always]
impl CastRegister<super::U64x2Neon> for ArrayRegister<u16, 2> {
    fn cast_from(value: Storage<super::U64x2Neon>) -> Storage<Self> {
        unsafe {
            ArrayRegister([
                arch::vgetq_lane_u64::<0>(value) as u16,
                arch::vgetq_lane_u64::<1>(value) as u16,
            ])
        }
    }
}

// --- x4 widen i16 -> i64 (low 4 words -> two 2x i64 lanes) ---
#[thermite_macros::inline_always]
impl CastRegister<I16x4Neon> for ArrayRegister<super::I64x2Neon, 2> {
    fn cast_from(value: Storage<I16x4Neon>) -> Storage<Self> {
        unsafe {
            let w = arch::vmovl_s16(arch::vget_low_s16(value.0));
            ArrayRegister([arch::vmovl_s32(arch::vget_low_s32(w)), arch::vmovl_high_s32(w)])
        }
    }
}
#[thermite_macros::inline_always]
impl CastRegister<U16x4Neon> for ArrayRegister<super::U64x2Neon, 2> {
    fn cast_from(value: Storage<U16x4Neon>) -> Storage<Self> {
        unsafe {
            let w = arch::vmovl_u16(arch::vget_low_u16(value.0));
            ArrayRegister([arch::vmovl_u32(arch::vget_low_u32(w)), arch::vmovl_high_u32(w)])
        }
    }
}

// --- x4 narrow i64 -> i16 (truncate each of 4 lanes -> low 4 words) ---
#[thermite_macros::inline_always]
impl CastRegister<ArrayRegister<super::I64x2Neon, 2>> for I16x4Neon {
    fn cast_from(value: Storage<ArrayRegister<super::I64x2Neon, 2>>) -> Storage<Self> {
        unsafe {
            let q32 = arch::vmovn_high_s64(arch::vmovn_s64(value.0[0]), value.0[1]);
            ReducedRegister::new(arch::vcombine_s16(arch::vmovn_s32(q32), arch::vdup_n_s16(0)))
        }
    }
}
#[thermite_macros::inline_always]
impl CastRegister<ArrayRegister<super::U64x2Neon, 2>> for U16x4Neon {
    fn cast_from(value: Storage<ArrayRegister<super::U64x2Neon, 2>>) -> Storage<Self> {
        unsafe {
            let q32 = arch::vmovn_high_u64(arch::vmovn_u64(value.0[0]), value.0[1]);
            ReducedRegister::new(arch::vcombine_u16(arch::vmovn_u32(q32), arch::vdup_n_u16(0)))
        }
    }
}

// --- x8 widen i16 -> i64 (8 words -> four 2x i64 lanes) ---
#[thermite_macros::inline_always]
impl CastRegister<super::I16x8Neon> for ArrayRegister<super::I64x2Neon, 4> {
    fn cast_from(value: Storage<super::I16x8Neon>) -> Storage<Self> {
        ArrayRegister(widen_i16x8_to_4xi64x2(value))
    }
}
#[thermite_macros::inline_always]
impl CastRegister<super::U16x8Neon> for ArrayRegister<super::U64x2Neon, 4> {
    fn cast_from(value: Storage<super::U16x8Neon>) -> Storage<Self> {
        ArrayRegister(widen_u16x8_to_4xu64x2(value))
    }
}

// --- x8 narrow i64 -> i16 (truncate each of 8 lanes -> 8 words) ---
#[thermite_macros::inline_always]
impl CastRegister<ArrayRegister<super::I64x2Neon, 4>> for super::I16x8Neon {
    fn cast_from(value: Storage<ArrayRegister<super::I64x2Neon, 4>>) -> Storage<Self> {
        narrow_4xi64x2_to_i16x8(value.0)
    }
}
#[thermite_macros::inline_always]
impl CastRegister<ArrayRegister<super::U64x2Neon, 4>> for super::U16x8Neon {
    fn cast_from(value: Storage<ArrayRegister<super::U64x2Neon, 4>>) -> Storage<Self> {
        narrow_4xu64x2_to_u16x8(value.0)
    }
}

// --- x16 widen i16 -> i64 (ArrayRegister<I16x8Neon, 2> -> ArrayRegister<I64x2Neon, 8>) ---
#[thermite_macros::inline_always]
impl CastRegister<ArrayRegister<super::I16x8Neon, 2>> for ArrayRegister<super::I64x2Neon, 8> {
    fn cast_from(value: Storage<ArrayRegister<super::I16x8Neon, 2>>) -> Storage<Self> {
        let lo = widen_i16x8_to_4xi64x2(value.0[0]);
        let hi = widen_i16x8_to_4xi64x2(value.0[1]);
        ArrayRegister([lo[0], lo[1], lo[2], lo[3], hi[0], hi[1], hi[2], hi[3]])
    }
}
#[thermite_macros::inline_always]
impl CastRegister<ArrayRegister<super::U16x8Neon, 2>> for ArrayRegister<super::U64x2Neon, 8> {
    fn cast_from(value: Storage<ArrayRegister<super::U16x8Neon, 2>>) -> Storage<Self> {
        let lo = widen_u16x8_to_4xu64x2(value.0[0]);
        let hi = widen_u16x8_to_4xu64x2(value.0[1]);
        ArrayRegister([lo[0], lo[1], lo[2], lo[3], hi[0], hi[1], hi[2], hi[3]])
    }
}

// --- x16 narrow i64 -> i16 (ArrayRegister<I64x2Neon, 8> -> ArrayRegister<I16x8Neon, 2>) ---
#[thermite_macros::inline_always]
impl CastRegister<ArrayRegister<super::I64x2Neon, 8>> for ArrayRegister<super::I16x8Neon, 2> {
    fn cast_from(value: Storage<ArrayRegister<super::I64x2Neon, 8>>) -> Storage<Self> {
        let v = value.0;
        ArrayRegister([
            narrow_4xi64x2_to_i16x8([v[0], v[1], v[2], v[3]]),
            narrow_4xi64x2_to_i16x8([v[4], v[5], v[6], v[7]]),
        ])
    }
}
#[thermite_macros::inline_always]
impl CastRegister<ArrayRegister<super::U64x2Neon, 8>> for ArrayRegister<super::U16x8Neon, 2> {
    fn cast_from(value: Storage<ArrayRegister<super::U64x2Neon, 8>>) -> Storage<Self> {
        let v = value.0;
        ArrayRegister([
            narrow_4xu64x2_to_u16x8([v[0], v[1], v[2], v[3]]),
            narrow_4xu64x2_to_u16x8([v[4], v[5], v[6], v[7]]),
        ])
    }
}

// ---------------------------------------------------------------------------------------
// 16 <-> f32/f64 direct casts (widen low words to i32 then native i32<->float converts;
// narrow via truncating-saturating float -> i32 then the truncating word narrows).
//   WIDEN  i16/u16 -> f32 = `vmovl` widen to 32-bit then `vcvtq_f32_s32`/`_u32` (exact).
//   NARROW f32 -> i16/u16 = `vcvtq_s32_f32` (truncate + saturate at i32 range, NaN -> 0,
//     matching wasm's `i32x4_trunc_sat_f32x4`) then a wrapping `vmovn`/`as` word narrow.
//   WIDEN  i16/u16 -> f64 = widen to i32 then `vmovl_s32` + `vcvtq_f64_s64` (2 lanes per
//     F64x2, fan out).
//   NARROW f64 -> i16/u16 = `vcvtq_s64_f64` + `vqmovn_s64` (== wasm's
//     `i32x4_trunc_sat_f64x2_zero`: truncate, saturate at i32 range) then wrapping narrows.
// Unsigned 16-bit values fit in positive i32, so the signed i32->float convert is exact
// for them. The trunc-sat-through-i32 narrow matches the 32-bit float->int cast path the
// other backends agreed on.
// ---------------------------------------------------------------------------------------

// f64x2 -> i32x2 lanes, `as`-style at i32 range: `vcvtq_s64_f64` truncates toward zero and
// handles NaN -> 0, `vqmovn_s64` saturates to the i32 range.
#[inline(always)]
fn f64x2_to_i32x2(v: arch::float64x2_t) -> arch::int32x2_t {
    unsafe { arch::vqmovn_s64(arch::vcvtq_s64_f64(v)) }
}
// i32x4 -> 2x f64x2 (exact widening convert; low 2 lanes, high 2 lanes).
#[inline(always)]
fn i32x4_to_2xf64x2(v: arch::int32x4_t) -> [arch::float64x2_t; 2] {
    unsafe {
        [
            arch::vcvtq_f64_s64(arch::vmovl_s32(arch::vget_low_s32(v))),
            arch::vcvtq_f64_s64(arch::vmovl_high_s32(v)),
        ]
    }
}

// --- x2 (ArrayRegister<i16,2> <-> ReducedRegister<F32x4Neon,2> = F32x2Neon) ---
#[thermite_macros::inline_always]
impl CastRegister<ArrayRegister<i16, 2>> for super::half::F32x2Neon {
    fn cast_from(value: Storage<ArrayRegister<i16, 2>>) -> Storage<Self> {
        let ints = reg::<super::I32x4Neon, 4>([value.0[0] as i32, value.0[1] as i32, 0, 0]);
        unsafe { ReducedRegister::new(arch::vcvtq_f32_s32(ints)) }
    }
}
#[thermite_macros::inline_always]
impl CastRegister<ArrayRegister<u16, 2>> for super::half::F32x2Neon {
    fn cast_from(value: Storage<ArrayRegister<u16, 2>>) -> Storage<Self> {
        let ints = reg::<super::I32x4Neon, 4>([value.0[0] as i32, value.0[1] as i32, 0, 0]);
        unsafe { ReducedRegister::new(arch::vcvtq_f32_s32(ints)) }
    }
}
#[thermite_macros::inline_always]
impl CastRegister<super::half::F32x2Neon> for ArrayRegister<i16, 2> {
    fn cast_from(value: Storage<super::half::F32x2Neon>) -> Storage<Self> {
        unsafe {
            let d = arch::vcvtq_s32_f32(value.0);
            ArrayRegister([arch::vgetq_lane_s32::<0>(d) as i16, arch::vgetq_lane_s32::<1>(d) as i16])
        }
    }
}
#[thermite_macros::inline_always]
impl CastRegister<super::half::F32x2Neon> for ArrayRegister<u16, 2> {
    fn cast_from(value: Storage<super::half::F32x2Neon>) -> Storage<Self> {
        unsafe {
            let d = arch::vcvtq_s32_f32(value.0);
            ArrayRegister([arch::vgetq_lane_s32::<0>(d) as u16, arch::vgetq_lane_s32::<1>(d) as u16])
        }
    }
}

// --- x2 (ArrayRegister<i16,2> <-> F64x2Neon native) ---
#[thermite_macros::inline_always]
impl CastRegister<ArrayRegister<i16, 2>> for super::F64x2Neon {
    // Two scalar lanes: direct exact scalar converts (wasm bounces through an i32x4).
    fn cast_from(value: Storage<ArrayRegister<i16, 2>>) -> Storage<Self> {
        reg::<super::F64x2Neon, 2>([value.0[0] as f64, value.0[1] as f64])
    }
}
#[thermite_macros::inline_always]
impl CastRegister<ArrayRegister<u16, 2>> for super::F64x2Neon {
    fn cast_from(value: Storage<ArrayRegister<u16, 2>>) -> Storage<Self> {
        reg::<super::F64x2Neon, 2>([value.0[0] as f64, value.0[1] as f64])
    }
}
#[thermite_macros::inline_always]
impl CastRegister<super::F64x2Neon> for ArrayRegister<i16, 2> {
    fn cast_from(value: Storage<super::F64x2Neon>) -> Storage<Self> {
        unsafe {
            let d = f64x2_to_i32x2(value);
            ArrayRegister([arch::vget_lane_s32::<0>(d) as i16, arch::vget_lane_s32::<1>(d) as i16])
        }
    }
}
#[thermite_macros::inline_always]
impl CastRegister<super::F64x2Neon> for ArrayRegister<u16, 2> {
    fn cast_from(value: Storage<super::F64x2Neon>) -> Storage<Self> {
        unsafe {
            let d = f64x2_to_i32x2(value);
            ArrayRegister([arch::vget_lane_s32::<0>(d) as u16, arch::vget_lane_s32::<1>(d) as u16])
        }
    }
}

// --- x4 (I16x4Neon <-> F32x4Neon native) ---
#[thermite_macros::inline_always]
impl CastRegister<I16x4Neon> for super::F32x4Neon {
    fn cast_from(value: Storage<I16x4Neon>) -> Storage<Self> {
        unsafe { arch::vcvtq_f32_s32(arch::vmovl_s16(arch::vget_low_s16(value.0))) }
    }
}
#[thermite_macros::inline_always]
impl CastRegister<U16x4Neon> for super::F32x4Neon {
    fn cast_from(value: Storage<U16x4Neon>) -> Storage<Self> {
        // Zero-extend + unsigned convert (equivalent to wasm's zero-extend + signed convert:
        // u16 fits in positive i32, both are exact).
        unsafe { arch::vcvtq_f32_u32(arch::vmovl_u16(arch::vget_low_u16(value.0))) }
    }
}
#[thermite_macros::inline_always]
impl CastRegister<super::F32x4Neon> for I16x4Neon {
    fn cast_from(value: Storage<super::F32x4Neon>) -> Storage<Self> {
        unsafe {
            ReducedRegister::new(arch::vcombine_s16(
                arch::vmovn_s32(arch::vcvtq_s32_f32(value)),
                arch::vdup_n_s16(0),
            ))
        }
    }
}
#[thermite_macros::inline_always]
impl CastRegister<super::F32x4Neon> for U16x4Neon {
    // Via the signed i32 `as`-cast then a wrapping narrow, matching the wasm path exactly.
    fn cast_from(value: Storage<super::F32x4Neon>) -> Storage<Self> {
        unsafe {
            ReducedRegister::new(arch::vcombine_u16(
                arch::vmovn_u32(arch::vreinterpretq_u32_s32(arch::vcvtq_s32_f32(value))),
                arch::vdup_n_u16(0),
            ))
        }
    }
}

// --- x4 (I16x4Neon <-> ArrayRegister<F64x2Neon, 2>) ---
#[thermite_macros::inline_always]
impl CastRegister<I16x4Neon> for ArrayRegister<super::F64x2Neon, 2> {
    fn cast_from(value: Storage<I16x4Neon>) -> Storage<Self> {
        unsafe { ArrayRegister(i32x4_to_2xf64x2(arch::vmovl_s16(arch::vget_low_s16(value.0)))) }
    }
}
#[thermite_macros::inline_always]
impl CastRegister<U16x4Neon> for ArrayRegister<super::F64x2Neon, 2> {
    fn cast_from(value: Storage<U16x4Neon>) -> Storage<Self> {
        // u16 zero-extends into positive i32, so the signed widen-convert path is exact.
        unsafe {
            ArrayRegister(i32x4_to_2xf64x2(arch::vreinterpretq_s32_u32(arch::vmovl_u16(
                arch::vget_low_u16(value.0),
            ))))
        }
    }
}
#[thermite_macros::inline_always]
impl CastRegister<ArrayRegister<super::F64x2Neon, 2>> for I16x4Neon {
    fn cast_from(value: Storage<ArrayRegister<super::F64x2Neon, 2>>) -> Storage<Self> {
        unsafe {
            let q32 = arch::vcombine_s32(f64x2_to_i32x2(value.0[0]), f64x2_to_i32x2(value.0[1]));
            ReducedRegister::new(arch::vcombine_s16(arch::vmovn_s32(q32), arch::vdup_n_s16(0)))
        }
    }
}
#[thermite_macros::inline_always]
impl CastRegister<ArrayRegister<super::F64x2Neon, 2>> for U16x4Neon {
    fn cast_from(value: Storage<ArrayRegister<super::F64x2Neon, 2>>) -> Storage<Self> {
        unsafe {
            let q32 = arch::vcombine_s32(f64x2_to_i32x2(value.0[0]), f64x2_to_i32x2(value.0[1]));
            ReducedRegister::new(arch::vcombine_u16(
                arch::vmovn_u32(arch::vreinterpretq_u32_s32(q32)),
                arch::vdup_n_u16(0),
            ))
        }
    }
}

// --- x8 (I16x8Neon native <-> ArrayRegister<F32x4Neon, 2>) ---
#[thermite_macros::inline_always]
impl CastRegister<super::I16x8Neon> for ArrayRegister<super::F32x4Neon, 2> {
    fn cast_from(value: Storage<super::I16x8Neon>) -> Storage<Self> {
        unsafe {
            ArrayRegister([
                arch::vcvtq_f32_s32(arch::vmovl_s16(arch::vget_low_s16(value))),
                arch::vcvtq_f32_s32(arch::vmovl_high_s16(value)),
            ])
        }
    }
}
#[thermite_macros::inline_always]
impl CastRegister<super::U16x8Neon> for ArrayRegister<super::F32x4Neon, 2> {
    fn cast_from(value: Storage<super::U16x8Neon>) -> Storage<Self> {
        unsafe {
            ArrayRegister([
                arch::vcvtq_f32_u32(arch::vmovl_u16(arch::vget_low_u16(value))),
                arch::vcvtq_f32_u32(arch::vmovl_high_u16(value)),
            ])
        }
    }
}
#[thermite_macros::inline_always]
impl CastRegister<ArrayRegister<super::F32x4Neon, 2>> for super::I16x8Neon {
    fn cast_from(value: Storage<ArrayRegister<super::F32x4Neon, 2>>) -> Storage<Self> {
        unsafe {
            arch::vmovn_high_s32(
                arch::vmovn_s32(arch::vcvtq_s32_f32(value.0[0])),
                arch::vcvtq_s32_f32(value.0[1]),
            )
        }
    }
}
#[thermite_macros::inline_always]
impl CastRegister<ArrayRegister<super::F32x4Neon, 2>> for super::U16x8Neon {
    // Via the signed i32 `as`-cast then a wrapping narrow, matching the wasm path exactly.
    fn cast_from(value: Storage<ArrayRegister<super::F32x4Neon, 2>>) -> Storage<Self> {
        unsafe {
            arch::vreinterpretq_u16_s16(arch::vmovn_high_s32(
                arch::vmovn_s32(arch::vcvtq_s32_f32(value.0[0])),
                arch::vcvtq_s32_f32(value.0[1]),
            ))
        }
    }
}

// --- x8 (I16x8Neon native <-> ArrayRegister<F64x2Neon, 4>) ---
#[thermite_macros::inline_always]
impl CastRegister<super::I16x8Neon> for ArrayRegister<super::F64x2Neon, 4> {
    fn cast_from(value: Storage<super::I16x8Neon>) -> Storage<Self> {
        unsafe {
            let a = i32x4_to_2xf64x2(arch::vmovl_s16(arch::vget_low_s16(value)));
            let b = i32x4_to_2xf64x2(arch::vmovl_high_s16(value));
            ArrayRegister([a[0], a[1], b[0], b[1]])
        }
    }
}
#[thermite_macros::inline_always]
impl CastRegister<super::U16x8Neon> for ArrayRegister<super::F64x2Neon, 4> {
    fn cast_from(value: Storage<super::U16x8Neon>) -> Storage<Self> {
        // u16 zero-extends into positive i32, so the signed widen-convert path is exact.
        unsafe {
            let a = i32x4_to_2xf64x2(arch::vreinterpretq_s32_u32(arch::vmovl_u16(arch::vget_low_u16(value))));
            let b = i32x4_to_2xf64x2(arch::vreinterpretq_s32_u32(arch::vmovl_high_u16(value)));
            ArrayRegister([a[0], a[1], b[0], b[1]])
        }
    }
}
#[thermite_macros::inline_always]
impl CastRegister<ArrayRegister<super::F64x2Neon, 4>> for super::I16x8Neon {
    fn cast_from(value: Storage<ArrayRegister<super::F64x2Neon, 4>>) -> Storage<Self> {
        unsafe {
            let q0 = arch::vcombine_s32(f64x2_to_i32x2(value.0[0]), f64x2_to_i32x2(value.0[1]));
            let q1 = arch::vcombine_s32(f64x2_to_i32x2(value.0[2]), f64x2_to_i32x2(value.0[3]));
            arch::vmovn_high_s32(arch::vmovn_s32(q0), q1)
        }
    }
}
#[thermite_macros::inline_always]
impl CastRegister<ArrayRegister<super::F64x2Neon, 4>> for super::U16x8Neon {
    fn cast_from(value: Storage<ArrayRegister<super::F64x2Neon, 4>>) -> Storage<Self> {
        unsafe {
            let q0 = arch::vcombine_s32(f64x2_to_i32x2(value.0[0]), f64x2_to_i32x2(value.0[1]));
            let q1 = arch::vcombine_s32(f64x2_to_i32x2(value.0[2]), f64x2_to_i32x2(value.0[3]));
            arch::vreinterpretq_u16_s16(arch::vmovn_high_s32(arch::vmovn_s32(q0), q1))
        }
    }
}

// --- x16 16 <-> f32: ArrayRegister<I16x8Neon, 2> <-> ArrayRegister<F32x4Neon, 4> is provided
//     for free by the ArrayRegister 2<->4 reshape cast blanket (array.rs), so no explicit impl. ---

// --- x16 (ArrayRegister<I16x8Neon, 2> <-> ArrayRegister<F64x2Neon, 8>) ---
#[thermite_macros::inline_always]
impl CastRegister<ArrayRegister<super::I16x8Neon, 2>> for ArrayRegister<super::F64x2Neon, 8> {
    fn cast_from(value: Storage<ArrayRegister<super::I16x8Neon, 2>>) -> Storage<Self> {
        let v = value.0;
        unsafe {
            let a = i32x4_to_2xf64x2(arch::vmovl_s16(arch::vget_low_s16(v[0])));
            let b = i32x4_to_2xf64x2(arch::vmovl_high_s16(v[0]));
            let c = i32x4_to_2xf64x2(arch::vmovl_s16(arch::vget_low_s16(v[1])));
            let d = i32x4_to_2xf64x2(arch::vmovl_high_s16(v[1]));
            ArrayRegister([a[0], a[1], b[0], b[1], c[0], c[1], d[0], d[1]])
        }
    }
}
#[thermite_macros::inline_always]
impl CastRegister<ArrayRegister<super::U16x8Neon, 2>> for ArrayRegister<super::F64x2Neon, 8> {
    fn cast_from(value: Storage<ArrayRegister<super::U16x8Neon, 2>>) -> Storage<Self> {
        let v = value.0;
        unsafe {
            let a = i32x4_to_2xf64x2(arch::vreinterpretq_s32_u32(arch::vmovl_u16(arch::vget_low_u16(v[0]))));
            let b = i32x4_to_2xf64x2(arch::vreinterpretq_s32_u32(arch::vmovl_high_u16(v[0])));
            let c = i32x4_to_2xf64x2(arch::vreinterpretq_s32_u32(arch::vmovl_u16(arch::vget_low_u16(v[1]))));
            let d = i32x4_to_2xf64x2(arch::vreinterpretq_s32_u32(arch::vmovl_high_u16(v[1])));
            ArrayRegister([a[0], a[1], b[0], b[1], c[0], c[1], d[0], d[1]])
        }
    }
}
#[thermite_macros::inline_always]
impl CastRegister<ArrayRegister<super::F64x2Neon, 8>> for ArrayRegister<super::I16x8Neon, 2> {
    fn cast_from(value: Storage<ArrayRegister<super::F64x2Neon, 8>>) -> Storage<Self> {
        let v = value.0;
        unsafe {
            let q0 = arch::vcombine_s32(f64x2_to_i32x2(v[0]), f64x2_to_i32x2(v[1]));
            let q1 = arch::vcombine_s32(f64x2_to_i32x2(v[2]), f64x2_to_i32x2(v[3]));
            let q2 = arch::vcombine_s32(f64x2_to_i32x2(v[4]), f64x2_to_i32x2(v[5]));
            let q3 = arch::vcombine_s32(f64x2_to_i32x2(v[6]), f64x2_to_i32x2(v[7]));
            ArrayRegister([
                arch::vmovn_high_s32(arch::vmovn_s32(q0), q1),
                arch::vmovn_high_s32(arch::vmovn_s32(q2), q3),
            ])
        }
    }
}
#[thermite_macros::inline_always]
impl CastRegister<ArrayRegister<super::F64x2Neon, 8>> for ArrayRegister<super::U16x8Neon, 2> {
    fn cast_from(value: Storage<ArrayRegister<super::F64x2Neon, 8>>) -> Storage<Self> {
        let v = value.0;
        unsafe {
            let q0 = arch::vcombine_s32(f64x2_to_i32x2(v[0]), f64x2_to_i32x2(v[1]));
            let q1 = arch::vcombine_s32(f64x2_to_i32x2(v[2]), f64x2_to_i32x2(v[3]));
            let q2 = arch::vcombine_s32(f64x2_to_i32x2(v[4]), f64x2_to_i32x2(v[5]));
            let q3 = arch::vcombine_s32(f64x2_to_i32x2(v[6]), f64x2_to_i32x2(v[7]));
            ArrayRegister([
                arch::vreinterpretq_u16_s16(arch::vmovn_high_s32(arch::vmovn_s32(q0), q1)),
                arch::vreinterpretq_u16_s16(arch::vmovn_high_s32(arch::vmovn_s32(q2), q3)),
            ])
        }
    }
}

// ---------------------------------------------------------------------------------------
// Mask-side concat: the reduced i16x4 register is its own Mask, and `FullConcatRegister`
// requires that Mask to concat from the i16x2 half's Mask (`ArrayRegister<bool, 2>`).
// ---------------------------------------------------------------------------------------

#[inline(always)]
fn bool_to_i16_mask(b: bool) -> i16 {
    if b { !0 } else { 0 }
}

#[inline(always)]
fn bool_to_u16_mask(b: bool) -> u16 {
    if b { !0 } else { 0 }
}

#[thermite_macros::inline_always]
impl ConcatRegister<ArrayRegister<bool, 2>> for I16x4Neon {
    fn concat(lo: Storage<ArrayRegister<bool, 2>>, hi: Storage<ArrayRegister<bool, 2>>) -> Storage<Self> {
        ReducedRegister::new(reg::<super::I16x8Neon, 8>([
            bool_to_i16_mask(lo.0[0]),
            bool_to_i16_mask(lo.0[1]),
            bool_to_i16_mask(hi.0[0]),
            bool_to_i16_mask(hi.0[1]),
            0,
            0,
            0,
            0,
        ]))
    }

    fn split(value: Storage<Self>) -> (Storage<ArrayRegister<bool, 2>>, Storage<ArrayRegister<bool, 2>>) {
        unsafe {
            (
                ArrayRegister([
                    arch::vgetq_lane_s16::<0>(value.0) != 0,
                    arch::vgetq_lane_s16::<1>(value.0) != 0,
                ]),
                ArrayRegister([
                    arch::vgetq_lane_s16::<2>(value.0) != 0,
                    arch::vgetq_lane_s16::<3>(value.0) != 0,
                ]),
            )
        }
    }
}

#[thermite_macros::inline_always]
impl ExtendRegister<ArrayRegister<bool, 2>> for I16x4Neon {
    fn extend(value: Storage<ArrayRegister<bool, 2>>) -> Storage<Self> {
        ReducedRegister::new(reg::<super::I16x8Neon, 8>([
            bool_to_i16_mask(value.0[0]),
            bool_to_i16_mask(value.0[1]),
            0,
            0,
            0,
            0,
            0,
            0,
        ]))
    }

    fn narrow(value: Storage<Self>) -> Storage<ArrayRegister<bool, 2>> {
        unsafe {
            ArrayRegister([
                arch::vgetq_lane_s16::<0>(value.0) != 0,
                arch::vgetq_lane_s16::<1>(value.0) != 0,
            ])
        }
    }
}

#[thermite_macros::inline_always]
impl ConcatRegister<ArrayRegister<bool, 2>> for U16x4Neon {
    fn concat(lo: Storage<ArrayRegister<bool, 2>>, hi: Storage<ArrayRegister<bool, 2>>) -> Storage<Self> {
        ReducedRegister::new(reg::<super::U16x8Neon, 8>([
            bool_to_u16_mask(lo.0[0]),
            bool_to_u16_mask(lo.0[1]),
            bool_to_u16_mask(hi.0[0]),
            bool_to_u16_mask(hi.0[1]),
            0,
            0,
            0,
            0,
        ]))
    }

    fn split(value: Storage<Self>) -> (Storage<ArrayRegister<bool, 2>>, Storage<ArrayRegister<bool, 2>>) {
        unsafe {
            (
                ArrayRegister([
                    arch::vgetq_lane_u16::<0>(value.0) != 0,
                    arch::vgetq_lane_u16::<1>(value.0) != 0,
                ]),
                ArrayRegister([
                    arch::vgetq_lane_u16::<2>(value.0) != 0,
                    arch::vgetq_lane_u16::<3>(value.0) != 0,
                ]),
            )
        }
    }
}

#[thermite_macros::inline_always]
impl ExtendRegister<ArrayRegister<bool, 2>> for U16x4Neon {
    fn extend(value: Storage<ArrayRegister<bool, 2>>) -> Storage<Self> {
        ReducedRegister::new(reg::<super::U16x8Neon, 8>([
            bool_to_u16_mask(value.0[0]),
            bool_to_u16_mask(value.0[1]),
            0,
            0,
            0,
            0,
            0,
            0,
        ]))
    }

    fn narrow(value: Storage<Self>) -> Storage<ArrayRegister<bool, 2>> {
        unsafe {
            ArrayRegister([
                arch::vgetq_lane_u16::<0>(value.0) != 0,
                arch::vgetq_lane_u16::<1>(value.0) != 0,
            ])
        }
    }
}

// Gather/scatter for the reduced 16-bit registers (indexed by the native 4-lane index types)
// falls back to scalar; only the cross-type index markers are needed. The concrete register
// types are spelled out because `Simd` is not yet implemented for `Neon` (these are what
// `<Neon as Simd>::u32x4` / `::u64x4` will alias, matching the wasm file).
impl IndexableRegister<super::U32x4Neon> for I16x4Neon {}
impl IndexableRegister<super::U32x4Neon> for U16x4Neon {}
impl IndexableRegister<ArrayRegister<super::U64x2Neon, 2>> for I16x4Neon {}
impl IndexableRegister<ArrayRegister<super::U64x2Neon, 2>> for U16x4Neon {}
