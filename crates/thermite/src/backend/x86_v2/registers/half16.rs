//! Reduced (sub-native-width) 16-bit registers for x86-v2 and the cast/concat glue that
//! bridges the scalar/array 16-bit halves, the native 128-bit `I16x8V2`/`U16x8V2`, and the
//! 32-bit registers they widen into. Mirrors the 32-bit `half.rs` structure.

use generic_array::typenum::U4;

use super::arch;

use crate::register::{
    CastRegister, ConcatRegister, ExtendRegister, IndexableRegister, NumericRegister, Register, SaturatingCastRegister,
    Storage, array::ArrayRegister, reduced::ReducedRegister,
};

// Clamp + truncating narrow for the pack-less `* -> i16` pairs: `i64 -> i16` (no SSE 64-bit
// pack) and the 2-lane `i32 -> i16` combos with a scalar `ArrayRegister` destination.
macro_rules! sat_clamp_narrow16 {
    ($(($from:ty, $fe:ty, $into:ty, $ie:ty)),* $(,)?) => {$(
        #[thermite_macros::inline_always]
        impl SaturatingCastRegister<$from> for $into {
            fn saturating_cast_from(value: Storage<$from>) -> Storage<Self> {
                let lo = <$from as Register>::splat(<$ie>::MIN as $fe);
                let hi = <$from as Register>::splat(<$ie>::MAX as $fe);
                let clamped = <$from as NumericRegister>::min(<$from as NumericRegister>::max(value, lo), hi);
                <Self as CastRegister<$from>>::cast_from(clamped)
            }
        }
    )*};
}

sat_clamp_narrow16! {
    (ArrayRegister<super::I64x2V2, 2>, i64, I16x4V2, i16),
    (ArrayRegister<super::U64x2V2, 2>, u64, U16x4V2, u16),
    (super::half::I32x2V2, i32, ArrayRegister<i16, 2>, i16),
    (super::half::U32x2V2, u32, ArrayRegister<u16, 2>, u16),
    (super::I64x2V2, i64, ArrayRegister<i16, 2>, i16),
    (super::U64x2V2, u64, ArrayRegister<u16, 2>, u16),
    (ArrayRegister<super::I64x2V2, 8>, i64, ArrayRegister<super::I16x8V2, 2>, i16),
    (ArrayRegister<super::U64x2V2, 8>, u64, ArrayRegister<super::U16x8V2, 2>, u16),
}

/// 4-lane signed 16-bit register, backed by the low 4 lanes of a 128-bit `I16x8V2`.
pub type I16x4V2 = ReducedRegister<super::I16x8V2, U4>;
/// 4-lane unsigned 16-bit register, backed by the low 4 lanes of a 128-bit `U16x8V2`.
pub type U16x4V2 = ReducedRegister<super::U16x8V2, U4>;

// ---------------------------------------------------------------------------------------
// x4 <- x2 : build a 4-lane reduced register from two 2-lane (scalar-array) halves.
// ---------------------------------------------------------------------------------------

#[thermite_macros::inline_always]
impl ConcatRegister<ArrayRegister<i16, 2>> for I16x4V2 {
    fn concat(lo: Storage<ArrayRegister<i16, 2>>, hi: Storage<ArrayRegister<i16, 2>>) -> Storage<Self> {
        ReducedRegister::new(unsafe { arch::_mm_setr_epi16(lo.0[0], lo.0[1], hi.0[0], hi.0[1], 0, 0, 0, 0) })
    }

    fn split(value: Storage<Self>) -> (Storage<ArrayRegister<i16, 2>>, Storage<ArrayRegister<i16, 2>>) {
        let mut arr = [0i16; 8];
        unsafe { arch::_mm_storeu_si128(arr.as_mut_ptr() as *mut _, value.0) };
        (ArrayRegister([arr[0], arr[1]]), ArrayRegister([arr[2], arr[3]]))
    }
}

#[thermite_macros::inline_always]
impl ExtendRegister<ArrayRegister<i16, 2>> for I16x4V2 {
    fn extend(value: Storage<ArrayRegister<i16, 2>>) -> Storage<Self> {
        ReducedRegister::new(unsafe { arch::_mm_setr_epi16(value.0[0], value.0[1], 0, 0, 0, 0, 0, 0) })
    }

    fn narrow(value: Storage<Self>) -> Storage<ArrayRegister<i16, 2>> {
        let mut arr = [0i16; 8];
        unsafe { arch::_mm_storeu_si128(arr.as_mut_ptr() as *mut _, value.0) };
        ArrayRegister([arr[0], arr[1]])
    }
}

#[thermite_macros::inline_always]
impl ConcatRegister<ArrayRegister<u16, 2>> for U16x4V2 {
    fn concat(lo: Storage<ArrayRegister<u16, 2>>, hi: Storage<ArrayRegister<u16, 2>>) -> Storage<Self> {
        ReducedRegister::new(unsafe {
            arch::_mm_setr_epi16(
                lo.0[0] as i16,
                lo.0[1] as i16,
                hi.0[0] as i16,
                hi.0[1] as i16,
                0,
                0,
                0,
                0,
            )
        })
    }

    fn split(value: Storage<Self>) -> (Storage<ArrayRegister<u16, 2>>, Storage<ArrayRegister<u16, 2>>) {
        let mut arr = [0u16; 8];
        unsafe { arch::_mm_storeu_si128(arr.as_mut_ptr() as *mut _, value.0) };
        (ArrayRegister([arr[0], arr[1]]), ArrayRegister([arr[2], arr[3]]))
    }
}

#[thermite_macros::inline_always]
impl ExtendRegister<ArrayRegister<u16, 2>> for U16x4V2 {
    fn extend(value: Storage<ArrayRegister<u16, 2>>) -> Storage<Self> {
        ReducedRegister::new(unsafe { arch::_mm_setr_epi16(value.0[0] as i16, value.0[1] as i16, 0, 0, 0, 0, 0, 0) })
    }

    fn narrow(value: Storage<Self>) -> Storage<ArrayRegister<u16, 2>> {
        let mut arr = [0u16; 8];
        unsafe { arch::_mm_storeu_si128(arr.as_mut_ptr() as *mut _, value.0) };
        ArrayRegister([arr[0], arr[1]])
    }
}

// ---------------------------------------------------------------------------------------
// x8 <- x4 : build the native 8-lane register from two 4-lane reduced halves.
// ---------------------------------------------------------------------------------------

#[thermite_macros::inline_always]
impl ConcatRegister<I16x4V2> for super::I16x8V2 {
    fn concat(lo: Storage<I16x4V2>, hi: Storage<I16x4V2>) -> Storage<Self> {
        // low 4 lanes of `lo` and low 4 lanes of `hi` -> 8 lanes. Both live in the low 64
        // bits of their __m128i, so a 64-bit unpack interleaves the halves correctly.
        unsafe { arch::_mm_unpacklo_epi64(lo.0, hi.0) }
    }

    fn split(value: Storage<Self>) -> (Storage<I16x4V2>, Storage<I16x4V2>) {
        (
            ReducedRegister::new(value),
            ReducedRegister::new(unsafe { arch::_mm_unpackhi_epi64(value, value) }),
        )
    }
}

// NOTE: `ExtendRegister<I16x4V2> for I16x8V2` is provided by the ReducedRegister blanket
// (the reduced register shares storage with its inner I16x8V2), so it is not written here.

#[thermite_macros::inline_always]
impl ConcatRegister<U16x4V2> for super::U16x8V2 {
    fn concat(lo: Storage<U16x4V2>, hi: Storage<U16x4V2>) -> Storage<Self> {
        unsafe { arch::_mm_unpacklo_epi64(lo.0, hi.0) }
    }

    fn split(value: Storage<Self>) -> (Storage<U16x4V2>, Storage<U16x4V2>) {
        (
            ReducedRegister::new(value),
            ReducedRegister::new(unsafe { arch::_mm_unpackhi_epi64(value, value) }),
        )
    }
}

// NOTE: `ExtendRegister<U16x4V2> for U16x8V2` is provided by the ReducedRegister blanket.

// ---------------------------------------------------------------------------------------
// Widen casts to 32-bit (CastRegister = numeric widen/narrow).
//
// x4: I16x4V2 <-> I32x4V2 (native 4-lane).  x2: ArrayRegister<i16,2> <-> I32x2V2 (reduced).
// ---------------------------------------------------------------------------------------

// --- x4 widen i16 -> i32 ---
#[thermite_macros::inline_always]
impl CastRegister<I16x4V2> for super::I32x4V2 {
    fn cast_from(value: Storage<I16x4V2>) -> Storage<Self> {
        unsafe { arch::_mm_cvtepi16_epi32(value.0) }
    }
}

#[thermite_macros::inline_always]
impl CastRegister<U16x4V2> for super::U32x4V2 {
    fn cast_from(value: Storage<U16x4V2>) -> Storage<Self> {
        unsafe { arch::_mm_cvtepu16_epi32(value.0) }
    }
}

// --- x4 narrow i32 -> i16 (truncate low 16 bits per lane, wrapping like `as`) ---
#[thermite_macros::inline_always]
impl CastRegister<super::I32x4V2> for I16x4V2 {
    fn cast_from(value: Storage<super::I32x4V2>) -> Storage<Self> {
        // Gather the low 2 bytes of each 32-bit lane into the low 8 bytes.
        ReducedRegister::new(unsafe {
            arch::_mm_shuffle_epi8(
                value,
                arch::_mm_setr_epi8(0, 1, 4, 5, 8, 9, 12, 13, -1, -1, -1, -1, -1, -1, -1, -1),
            )
        })
    }
}

#[thermite_macros::inline_always]
impl CastRegister<super::U32x4V2> for U16x4V2 {
    fn cast_from(value: Storage<super::U32x4V2>) -> Storage<Self> {
        ReducedRegister::new(unsafe {
            arch::_mm_shuffle_epi8(
                value,
                arch::_mm_setr_epi8(0, 1, 4, 5, 8, 9, 12, 13, -1, -1, -1, -1, -1, -1, -1, -1),
            )
        })
    }
}

// --- saturating narrows into 16-bit (low half holds the result; no lane-crossing fixup) ---

// i32x4 -> i16x4 via `packssdw`.
#[thermite_macros::inline_always]
impl SaturatingCastRegister<super::I32x4V2> for I16x4V2 {
    fn saturating_cast_from(value: Storage<super::I32x4V2>) -> Storage<Self> {
        ReducedRegister::new(unsafe { arch::_mm_packs_epi32(value, value) })
    }
}

// u32x4 -> u16x4: clamp the high end (`pminud`) then `packusdw`.
#[thermite_macros::inline_always]
impl SaturatingCastRegister<super::U32x4V2> for U16x4V2 {
    fn saturating_cast_from(value: Storage<super::U32x4V2>) -> Storage<Self> {
        let clamped = unsafe { arch::_mm_min_epu32(value, arch::_mm_set1_epi32(0xFFFF)) };
        ReducedRegister::new(unsafe { arch::_mm_packus_epi32(clamped, clamped) })
    }
}

// --- x2 widen i16 -> i32 (ArrayRegister<i16,2> -> I32x2V2 reduced) ---
#[thermite_macros::inline_always]
impl CastRegister<ArrayRegister<i16, 2>> for super::half::I32x2V2 {
    fn cast_from(value: Storage<ArrayRegister<i16, 2>>) -> Storage<Self> {
        ReducedRegister::new(unsafe { arch::_mm_setr_epi32(value.0[0] as i32, value.0[1] as i32, 0, 0) })
    }
}

#[thermite_macros::inline_always]
impl CastRegister<ArrayRegister<u16, 2>> for super::half::U32x2V2 {
    fn cast_from(value: Storage<ArrayRegister<u16, 2>>) -> Storage<Self> {
        ReducedRegister::new(unsafe { arch::_mm_setr_epi32(value.0[0] as i32, value.0[1] as i32, 0, 0) })
    }
}

// --- x2 narrow i32 -> i16 (I32x2V2 reduced -> ArrayRegister<i16,2>) ---
#[thermite_macros::inline_always]
impl CastRegister<super::half::I32x2V2> for ArrayRegister<i16, 2> {
    fn cast_from(value: Storage<super::half::I32x2V2>) -> Storage<Self> {
        let mut arr = [0i32; 4];
        unsafe { arch::_mm_storeu_si128(arr.as_mut_ptr() as *mut _, value.0) };
        ArrayRegister([arr[0] as i16, arr[1] as i16])
    }
}

#[thermite_macros::inline_always]
impl CastRegister<super::half::U32x2V2> for ArrayRegister<u16, 2> {
    fn cast_from(value: Storage<super::half::U32x2V2>) -> Storage<Self> {
        let mut arr = [0u32; 4];
        unsafe { arch::_mm_storeu_si128(arr.as_mut_ptr() as *mut _, value.0) };
        ArrayRegister([arr[0] as u16, arr[1] as u16])
    }
}

// ---------------------------------------------------------------------------------------
// Widen/narrow casts to 64-bit (CastRegister = numeric widen / `as`-style narrow).
//
//   widen 16 -> 64 via `cvtepi16_epi64`/`cvtepu16_epi64` (low 2 words -> 2x i64 per call).
//   narrow 64 -> 16 via `pshufb` gathering bytes 0,1 of each i64 lane, then combining lanes.
//   x2:  ArrayRegister<i16,2> <-> I64x2V2 (native 2-lane).
//   x4:  I16x4V2 <-> ArrayRegister<I64x2V2, 2> (the v2 i64x4).
//   x8:  I16x8V2 <-> ArrayRegister<I64x2V2, 4> (the v2 i64x8).
//   x16: ArrayRegister<I16x8V2, 2> <-> ArrayRegister<I64x2V2, 8> (i16x16 / i64x16).
// ---------------------------------------------------------------------------------------

// pshufb mask gathering bytes 0,1 of each of 2 i64 lanes (positions 0,1 and 8,9) into low 4 bytes.
// --- x2 widen i16 -> i64 (ArrayRegister<i16,2> -> I64x2V2 native) ---
#[thermite_macros::inline_always]
impl CastRegister<ArrayRegister<i16, 2>> for super::I64x2V2 {
    fn cast_from(value: Storage<ArrayRegister<i16, 2>>) -> Storage<Self> {
        unsafe { arch::_mm_set_epi64x(value.0[1] as i64, value.0[0] as i64) }
    }
}
#[thermite_macros::inline_always]
impl CastRegister<ArrayRegister<u16, 2>> for super::U64x2V2 {
    fn cast_from(value: Storage<ArrayRegister<u16, 2>>) -> Storage<Self> {
        unsafe { arch::_mm_set_epi64x(value.0[1] as i64, value.0[0] as i64) }
    }
}

// --- x2 narrow i64 -> i16 (I64x2V2 native -> ArrayRegister<i16,2>) ---
#[thermite_macros::inline_always]
impl CastRegister<super::I64x2V2> for ArrayRegister<i16, 2> {
    fn cast_from(value: Storage<super::I64x2V2>) -> Storage<Self> {
        let mut arr = [0i64; 2];
        unsafe { arch::_mm_storeu_si128(arr.as_mut_ptr() as *mut _, value) };
        ArrayRegister([arr[0] as i16, arr[1] as i16])
    }
}
#[thermite_macros::inline_always]
impl CastRegister<super::U64x2V2> for ArrayRegister<u16, 2> {
    fn cast_from(value: Storage<super::U64x2V2>) -> Storage<Self> {
        let mut arr = [0u64; 2];
        unsafe { arch::_mm_storeu_si128(arr.as_mut_ptr() as *mut _, value) };
        ArrayRegister([arr[0] as u16, arr[1] as u16])
    }
}

// --- x4 widen i16 -> i64 (low 4 words -> two 2x i64 lanes) ---
#[thermite_macros::inline_always]
impl CastRegister<I16x4V2> for ArrayRegister<super::I64x2V2, 2> {
    fn cast_from(value: Storage<I16x4V2>) -> Storage<Self> {
        unsafe {
            ArrayRegister([
                arch::_mm_cvtepi16_epi64(value.0),
                arch::_mm_cvtepi16_epi64(arch::_mm_srli_si128(value.0, 4)),
            ])
        }
    }
}
#[thermite_macros::inline_always]
impl CastRegister<U16x4V2> for ArrayRegister<super::U64x2V2, 2> {
    fn cast_from(value: Storage<U16x4V2>) -> Storage<Self> {
        unsafe {
            ArrayRegister([
                arch::_mm_cvtepu16_epi64(value.0),
                arch::_mm_cvtepu16_epi64(arch::_mm_srli_si128(value.0, 4)),
            ])
        }
    }
}

// --- x4 narrow i64 -> i16 (word 0 of each of 4 lanes -> low 4 words) ---
#[thermite_macros::inline_always]
impl CastRegister<ArrayRegister<super::I64x2V2, 2>> for I16x4V2 {
    fn cast_from(value: Storage<ArrayRegister<super::I64x2V2, 2>>) -> Storage<Self> {
        unsafe {
            let mask = arch::_mm_narrow_qword_to_word_maskx_v2();
            let lo = arch::_mm_shuffle_epi8(value.0[0], mask); // 4 bytes (2 words) in low 32 bits
            let hi = arch::_mm_shuffle_epi8(value.0[1], mask);
            ReducedRegister::new(arch::_mm_unpacklo_epi32(lo, hi))
        }
    }
}
#[thermite_macros::inline_always]
impl CastRegister<ArrayRegister<super::U64x2V2, 2>> for U16x4V2 {
    fn cast_from(value: Storage<ArrayRegister<super::U64x2V2, 2>>) -> Storage<Self> {
        unsafe {
            let mask = arch::_mm_narrow_qword_to_word_maskx_v2();
            let lo = arch::_mm_shuffle_epi8(value.0[0], mask);
            let hi = arch::_mm_shuffle_epi8(value.0[1], mask);
            ReducedRegister::new(arch::_mm_unpacklo_epi32(lo, hi))
        }
    }
}

// --- x8 widen i16 -> i64 (8 words -> four 2x i64 lanes) ---
#[thermite_macros::inline_always]
impl CastRegister<super::I16x8V2> for ArrayRegister<super::I64x2V2, 4> {
    fn cast_from(value: Storage<super::I16x8V2>) -> Storage<Self> {
        unsafe {
            ArrayRegister([
                arch::_mm_cvtepi16_epi64(value),
                arch::_mm_cvtepi16_epi64(arch::_mm_srli_si128(value, 4)),
                arch::_mm_cvtepi16_epi64(arch::_mm_srli_si128(value, 8)),
                arch::_mm_cvtepi16_epi64(arch::_mm_srli_si128(value, 12)),
            ])
        }
    }
}
#[thermite_macros::inline_always]
impl CastRegister<super::U16x8V2> for ArrayRegister<super::U64x2V2, 4> {
    fn cast_from(value: Storage<super::U16x8V2>) -> Storage<Self> {
        unsafe {
            ArrayRegister([
                arch::_mm_cvtepu16_epi64(value),
                arch::_mm_cvtepu16_epi64(arch::_mm_srli_si128(value, 4)),
                arch::_mm_cvtepu16_epi64(arch::_mm_srli_si128(value, 8)),
                arch::_mm_cvtepu16_epi64(arch::_mm_srli_si128(value, 12)),
            ])
        }
    }
}

// --- x8 narrow i64 -> i16 (word 0 of each of 8 lanes -> 8 words) ---
#[thermite_macros::inline_always]
impl CastRegister<ArrayRegister<super::I64x2V2, 4>> for super::I16x8V2 {
    fn cast_from(value: Storage<ArrayRegister<super::I64x2V2, 4>>) -> Storage<Self> {
        unsafe { arch::_mm_cvt4epi64_epi16x_v2(value.0) }
    }
}
#[thermite_macros::inline_always]
impl CastRegister<ArrayRegister<super::U64x2V2, 4>> for super::U16x8V2 {
    fn cast_from(value: Storage<ArrayRegister<super::U64x2V2, 4>>) -> Storage<Self> {
        unsafe { arch::_mm_cvt4epi64_epi16x_v2(value.0) }
    }
}

// --- x16 widen i16 -> i64 (ArrayRegister<I16x8V2, 2> -> ArrayRegister<I64x2V2, 8>) ---
#[thermite_macros::inline_always]
impl CastRegister<ArrayRegister<super::I16x8V2, 2>> for ArrayRegister<super::I64x2V2, 8> {
    fn cast_from(value: Storage<ArrayRegister<super::I16x8V2, 2>>) -> Storage<Self> {
        let (lo, hi) = unsafe {
            (
                arch::_mm_cvtepi16_4epi64x_v2(value.0[0]),
                arch::_mm_cvtepi16_4epi64x_v2(value.0[1]),
            )
        };
        ArrayRegister([lo[0], lo[1], lo[2], lo[3], hi[0], hi[1], hi[2], hi[3]])
    }
}
#[thermite_macros::inline_always]
impl CastRegister<ArrayRegister<super::U16x8V2, 2>> for ArrayRegister<super::U64x2V2, 8> {
    fn cast_from(value: Storage<ArrayRegister<super::U16x8V2, 2>>) -> Storage<Self> {
        let (lo, hi) = unsafe {
            (
                arch::_mm_cvtepu16_4epi64x_v2(value.0[0]),
                arch::_mm_cvtepu16_4epi64x_v2(value.0[1]),
            )
        };
        ArrayRegister([lo[0], lo[1], lo[2], lo[3], hi[0], hi[1], hi[2], hi[3]])
    }
}

// --- x16 narrow i64 -> i16 (ArrayRegister<I64x2V2, 8> -> ArrayRegister<I16x8V2, 2>) ---
#[thermite_macros::inline_always]
impl CastRegister<ArrayRegister<super::I64x2V2, 8>> for ArrayRegister<super::I16x8V2, 2> {
    fn cast_from(value: Storage<ArrayRegister<super::I64x2V2, 8>>) -> Storage<Self> {
        let v = value.0;
        unsafe {
            ArrayRegister([
                arch::_mm_cvt4epi64_epi16x_v2([v[0], v[1], v[2], v[3]]),
                arch::_mm_cvt4epi64_epi16x_v2([v[4], v[5], v[6], v[7]]),
            ])
        }
    }
}
#[thermite_macros::inline_always]
impl CastRegister<ArrayRegister<super::U64x2V2, 8>> for ArrayRegister<super::U16x8V2, 2> {
    fn cast_from(value: Storage<ArrayRegister<super::U64x2V2, 8>>) -> Storage<Self> {
        let v = value.0;
        unsafe {
            ArrayRegister([
                arch::_mm_cvt4epi64_epi16x_v2([v[0], v[1], v[2], v[3]]),
                arch::_mm_cvt4epi64_epi16x_v2([v[4], v[5], v[6], v[7]]),
            ])
        }
    }
}

// ---------------------------------------------------------------------------------------
// 16-bit <-> f32 / f64 direct casts.
//
// Single inline intrinsic sequence routed through a 32-bit integer intermediate of the SAME
// lane count (where the hardware int<->float converts live):
//   widen  i16/u16 -> f32 = (int widen i16/u16 -> i32, exactly the 16->32 widen above) then
//                            (i32 -> f32 via `_mm_cvtepi32_ps`). Value-preserving / exact.
//   narrow f32     -> i16 = (f32 -> i32 truncating via `_mm_cvttps_epi32`) then the existing
//                            i32 -> word narrow (pshufb gathering bytes 0,1 of each dword).
//   f64 has half the lanes per native register, so the i32 intermediate fans out across two
//   `_mm_cvtepi32_pd` / `_mm_cvttpd_epi32` calls (mirroring f32x4 <-> ArrayRegister<F64x2,2>).
// Unsigned sources zero-extend to i32 (`_mm_cvtepu16_epi32`); the widened value is a positive
// i32 so the signed `_mm_cvtepi32_ps`/`_mm_cvtepi32_pd` convert is still exact. For the float
// -> unsigned narrow the low word is identical signed or unsigned, so the signed narrow is reused.
//
// 16-bit x16 <-> f32 resolves for free (not in the missing-impl list) and is intentionally
// omitted here; only the 16-bit x16 <-> f64 pair needs explicit impls.
// ---------------------------------------------------------------------------------------

// pshufb mask gathering bytes 0,1 (word 0) of each of 4 i32 lanes into the low 8 bytes.
// --- f64 fan-out helpers (i32 <-> f64 across the 2-lane native f64 register) ---

// Widen the low 4 i32 lanes of a __m128i into two F64x2 registers (4 f64).
// Narrow two F64x2 registers (4 f64) into the low 4 i32 lanes of a __m128i (truncating).
// --- x2 16 <-> f32 (ArrayRegister<i16,2> <-> F32x2V2 reduced) ---
#[thermite_macros::inline_always]
impl CastRegister<ArrayRegister<i16, 2>> for super::half::F32x2V2 {
    fn cast_from(value: Storage<ArrayRegister<i16, 2>>) -> Storage<Self> {
        ReducedRegister::new(unsafe {
            arch::_mm_cvtepi32_ps(arch::_mm_setr_epi32(value.0[0] as i32, value.0[1] as i32, 0, 0))
        })
    }
}
#[thermite_macros::inline_always]
impl CastRegister<ArrayRegister<u16, 2>> for super::half::F32x2V2 {
    fn cast_from(value: Storage<ArrayRegister<u16, 2>>) -> Storage<Self> {
        ReducedRegister::new(unsafe {
            arch::_mm_cvtepi32_ps(arch::_mm_setr_epi32(value.0[0] as i32, value.0[1] as i32, 0, 0))
        })
    }
}
#[thermite_macros::inline_always]
impl CastRegister<super::half::F32x2V2> for ArrayRegister<i16, 2> {
    fn cast_from(value: Storage<super::half::F32x2V2>) -> Storage<Self> {
        let mut arr = [0i32; 4];
        unsafe { arch::_mm_storeu_si128(arr.as_mut_ptr() as *mut _, arch::_mm_cvttps_epi32(value.0)) };
        ArrayRegister([arr[0] as i16, arr[1] as i16])
    }
}
#[thermite_macros::inline_always]
impl CastRegister<super::half::F32x2V2> for ArrayRegister<u16, 2> {
    fn cast_from(value: Storage<super::half::F32x2V2>) -> Storage<Self> {
        let mut arr = [0i32; 4];
        unsafe { arch::_mm_storeu_si128(arr.as_mut_ptr() as *mut _, arch::_mm_cvttps_epi32(value.0)) };
        ArrayRegister([arr[0] as u16, arr[1] as u16])
    }
}

// --- x2 16 <-> f64 (ArrayRegister<i16,2> <-> F64x2V2 native) ---
#[thermite_macros::inline_always]
impl CastRegister<ArrayRegister<i16, 2>> for super::F64x2V2 {
    fn cast_from(value: Storage<ArrayRegister<i16, 2>>) -> Storage<Self> {
        unsafe { arch::_mm_cvtepi32_pd(arch::_mm_setr_epi32(value.0[0] as i32, value.0[1] as i32, 0, 0)) }
    }
}
#[thermite_macros::inline_always]
impl CastRegister<ArrayRegister<u16, 2>> for super::F64x2V2 {
    fn cast_from(value: Storage<ArrayRegister<u16, 2>>) -> Storage<Self> {
        unsafe { arch::_mm_cvtepi32_pd(arch::_mm_setr_epi32(value.0[0] as i32, value.0[1] as i32, 0, 0)) }
    }
}
#[thermite_macros::inline_always]
impl CastRegister<super::F64x2V2> for ArrayRegister<i16, 2> {
    fn cast_from(value: Storage<super::F64x2V2>) -> Storage<Self> {
        let mut arr = [0i32; 4];
        unsafe { arch::_mm_storeu_si128(arr.as_mut_ptr() as *mut _, arch::_mm_cvttpd_epi32(value)) };
        ArrayRegister([arr[0] as i16, arr[1] as i16])
    }
}
#[thermite_macros::inline_always]
impl CastRegister<super::F64x2V2> for ArrayRegister<u16, 2> {
    fn cast_from(value: Storage<super::F64x2V2>) -> Storage<Self> {
        let mut arr = [0i32; 4];
        unsafe { arch::_mm_storeu_si128(arr.as_mut_ptr() as *mut _, arch::_mm_cvttpd_epi32(value)) };
        ArrayRegister([arr[0] as u16, arr[1] as u16])
    }
}

// --- x4 16 <-> f32 (I16x4V2 reduced <-> F32x4V2 native) ---
#[thermite_macros::inline_always]
impl CastRegister<I16x4V2> for super::F32x4V2 {
    fn cast_from(value: Storage<I16x4V2>) -> Storage<Self> {
        unsafe { arch::_mm_cvtepi32_ps(arch::_mm_cvtepi16_epi32(value.0)) }
    }
}
#[thermite_macros::inline_always]
impl CastRegister<U16x4V2> for super::F32x4V2 {
    fn cast_from(value: Storage<U16x4V2>) -> Storage<Self> {
        unsafe { arch::_mm_cvtepi32_ps(arch::_mm_cvtepu16_epi32(value.0)) }
    }
}
#[thermite_macros::inline_always]
impl CastRegister<super::F32x4V2> for I16x4V2 {
    fn cast_from(value: Storage<super::F32x4V2>) -> Storage<Self> {
        ReducedRegister::new(unsafe {
            arch::_mm_shuffle_epi8(arch::_mm_cvttps_epi32(value), arch::_mm_narrow_dword_to_word_maskx_v2())
        })
    }
}
#[thermite_macros::inline_always]
impl CastRegister<super::F32x4V2> for U16x4V2 {
    fn cast_from(value: Storage<super::F32x4V2>) -> Storage<Self> {
        ReducedRegister::new(unsafe {
            arch::_mm_shuffle_epi8(arch::_mm_cvttps_epi32(value), arch::_mm_narrow_dword_to_word_maskx_v2())
        })
    }
}

// --- x4 16 <-> f64 (I16x4V2 reduced <-> ArrayRegister<F64x2V2, 2>) ---
#[thermite_macros::inline_always]
impl CastRegister<I16x4V2> for ArrayRegister<super::F64x2V2, 2> {
    fn cast_from(value: Storage<I16x4V2>) -> Storage<Self> {
        ArrayRegister(unsafe { arch::_mm_cvtepi32_2pdx_v2(arch::_mm_cvtepi16_epi32(value.0)) })
    }
}
#[thermite_macros::inline_always]
impl CastRegister<U16x4V2> for ArrayRegister<super::F64x2V2, 2> {
    fn cast_from(value: Storage<U16x4V2>) -> Storage<Self> {
        ArrayRegister(unsafe { arch::_mm_cvtepi32_2pdx_v2(arch::_mm_cvtepu16_epi32(value.0)) })
    }
}
#[thermite_macros::inline_always]
impl CastRegister<ArrayRegister<super::F64x2V2, 2>> for I16x4V2 {
    fn cast_from(value: Storage<ArrayRegister<super::F64x2V2, 2>>) -> Storage<Self> {
        ReducedRegister::new(unsafe {
            let i32s = arch::_mm_cvtt2pd_epi32x_v2(value.0);
            arch::_mm_shuffle_epi8(i32s, arch::_mm_narrow_dword_to_word_maskx_v2())
        })
    }
}
#[thermite_macros::inline_always]
impl CastRegister<ArrayRegister<super::F64x2V2, 2>> for U16x4V2 {
    fn cast_from(value: Storage<ArrayRegister<super::F64x2V2, 2>>) -> Storage<Self> {
        ReducedRegister::new(unsafe {
            let i32s = arch::_mm_cvtt2pd_epi32x_v2(value.0);
            arch::_mm_shuffle_epi8(i32s, arch::_mm_narrow_dword_to_word_maskx_v2())
        })
    }
}

// --- x8 16 <-> f32 (I16x8V2 native <-> ArrayRegister<F32x4V2, 2>) ---
#[thermite_macros::inline_always]
impl CastRegister<super::I16x8V2> for ArrayRegister<super::F32x4V2, 2> {
    fn cast_from(value: Storage<super::I16x8V2>) -> Storage<Self> {
        unsafe {
            let lo = arch::_mm_cvtepi32_ps(arch::_mm_cvtepi16_epi32(value));
            let hi = arch::_mm_cvtepi32_ps(arch::_mm_cvtepi16_epi32(arch::_mm_srli_si128(value, 8)));
            ArrayRegister([lo, hi])
        }
    }
}
#[thermite_macros::inline_always]
impl CastRegister<super::U16x8V2> for ArrayRegister<super::F32x4V2, 2> {
    fn cast_from(value: Storage<super::U16x8V2>) -> Storage<Self> {
        unsafe {
            let lo = arch::_mm_cvtepi32_ps(arch::_mm_cvtepu16_epi32(value));
            let hi = arch::_mm_cvtepi32_ps(arch::_mm_cvtepu16_epi32(arch::_mm_srli_si128(value, 8)));
            ArrayRegister([lo, hi])
        }
    }
}
#[thermite_macros::inline_always]
impl CastRegister<ArrayRegister<super::F32x4V2, 2>> for super::I16x8V2 {
    fn cast_from(value: Storage<ArrayRegister<super::F32x4V2, 2>>) -> Storage<Self> {
        unsafe {
            let mask = arch::_mm_narrow_dword_to_word_maskx_v2();
            let lo = arch::_mm_shuffle_epi8(arch::_mm_cvttps_epi32(value.0[0]), mask); // 4 words in low 64 bits
            let hi = arch::_mm_shuffle_epi8(arch::_mm_cvttps_epi32(value.0[1]), mask);
            arch::_mm_unpacklo_epi64(lo, hi) // 8 words
        }
    }
}
#[thermite_macros::inline_always]
impl CastRegister<ArrayRegister<super::F32x4V2, 2>> for super::U16x8V2 {
    fn cast_from(value: Storage<ArrayRegister<super::F32x4V2, 2>>) -> Storage<Self> {
        unsafe {
            let mask = arch::_mm_narrow_dword_to_word_maskx_v2();
            let lo = arch::_mm_shuffle_epi8(arch::_mm_cvttps_epi32(value.0[0]), mask);
            let hi = arch::_mm_shuffle_epi8(arch::_mm_cvttps_epi32(value.0[1]), mask);
            arch::_mm_unpacklo_epi64(lo, hi)
        }
    }
}

// --- x8 16 <-> f64 (I16x8V2 native <-> ArrayRegister<F64x2V2, 4>) ---
#[thermite_macros::inline_always]
impl CastRegister<super::I16x8V2> for ArrayRegister<super::F64x2V2, 4> {
    fn cast_from(value: Storage<super::I16x8V2>) -> Storage<Self> {
        unsafe {
            let lo = arch::_mm_cvtepi32_2pdx_v2(arch::_mm_cvtepi16_epi32(value));
            let hi = arch::_mm_cvtepi32_2pdx_v2(arch::_mm_cvtepi16_epi32(arch::_mm_srli_si128(value, 8)));
            ArrayRegister([lo[0], lo[1], hi[0], hi[1]])
        }
    }
}
#[thermite_macros::inline_always]
impl CastRegister<super::U16x8V2> for ArrayRegister<super::F64x2V2, 4> {
    fn cast_from(value: Storage<super::U16x8V2>) -> Storage<Self> {
        unsafe {
            let lo = arch::_mm_cvtepi32_2pdx_v2(arch::_mm_cvtepu16_epi32(value));
            let hi = arch::_mm_cvtepi32_2pdx_v2(arch::_mm_cvtepu16_epi32(arch::_mm_srli_si128(value, 8)));
            ArrayRegister([lo[0], lo[1], hi[0], hi[1]])
        }
    }
}
#[thermite_macros::inline_always]
impl CastRegister<ArrayRegister<super::F64x2V2, 4>> for super::I16x8V2 {
    fn cast_from(value: Storage<ArrayRegister<super::F64x2V2, 4>>) -> Storage<Self> {
        let v = value.0;
        unsafe {
            let lo = arch::_mm_cvtt2pd_epi32x_v2([v[0], v[1]]);
            let hi = arch::_mm_cvtt2pd_epi32x_v2([v[2], v[3]]);
            let mask = arch::_mm_narrow_dword_to_word_maskx_v2();
            let lo = arch::_mm_shuffle_epi8(lo, mask);
            let hi = arch::_mm_shuffle_epi8(hi, mask);
            arch::_mm_unpacklo_epi64(lo, hi)
        }
    }
}
#[thermite_macros::inline_always]
impl CastRegister<ArrayRegister<super::F64x2V2, 4>> for super::U16x8V2 {
    fn cast_from(value: Storage<ArrayRegister<super::F64x2V2, 4>>) -> Storage<Self> {
        let v = value.0;
        unsafe {
            let lo = arch::_mm_cvtt2pd_epi32x_v2([v[0], v[1]]);
            let hi = arch::_mm_cvtt2pd_epi32x_v2([v[2], v[3]]);
            let mask = arch::_mm_narrow_dword_to_word_maskx_v2();
            let lo = arch::_mm_shuffle_epi8(lo, mask);
            let hi = arch::_mm_shuffle_epi8(hi, mask);
            arch::_mm_unpacklo_epi64(lo, hi)
        }
    }
}

// --- x16 16 <-> f64 (ArrayRegister<I16x8V2, 2> <-> ArrayRegister<F64x2V2, 8>) ---
#[thermite_macros::inline_always]
impl CastRegister<ArrayRegister<super::I16x8V2, 2>> for ArrayRegister<super::F64x2V2, 8> {
    fn cast_from(value: Storage<ArrayRegister<super::I16x8V2, 2>>) -> Storage<Self> {
        unsafe {
            let v0 = value.0[0];
            let v1 = value.0[1];
            let a = arch::_mm_cvtepi32_2pdx_v2(arch::_mm_cvtepi16_epi32(v0));
            let b = arch::_mm_cvtepi32_2pdx_v2(arch::_mm_cvtepi16_epi32(arch::_mm_srli_si128(v0, 8)));
            let c = arch::_mm_cvtepi32_2pdx_v2(arch::_mm_cvtepi16_epi32(v1));
            let d = arch::_mm_cvtepi32_2pdx_v2(arch::_mm_cvtepi16_epi32(arch::_mm_srli_si128(v1, 8)));
            ArrayRegister([a[0], a[1], b[0], b[1], c[0], c[1], d[0], d[1]])
        }
    }
}
#[thermite_macros::inline_always]
impl CastRegister<ArrayRegister<super::U16x8V2, 2>> for ArrayRegister<super::F64x2V2, 8> {
    fn cast_from(value: Storage<ArrayRegister<super::U16x8V2, 2>>) -> Storage<Self> {
        unsafe {
            let v0 = value.0[0];
            let v1 = value.0[1];
            let a = arch::_mm_cvtepi32_2pdx_v2(arch::_mm_cvtepu16_epi32(v0));
            let b = arch::_mm_cvtepi32_2pdx_v2(arch::_mm_cvtepu16_epi32(arch::_mm_srli_si128(v0, 8)));
            let c = arch::_mm_cvtepi32_2pdx_v2(arch::_mm_cvtepu16_epi32(v1));
            let d = arch::_mm_cvtepi32_2pdx_v2(arch::_mm_cvtepu16_epi32(arch::_mm_srli_si128(v1, 8)));
            ArrayRegister([a[0], a[1], b[0], b[1], c[0], c[1], d[0], d[1]])
        }
    }
}
#[thermite_macros::inline_always]
impl CastRegister<ArrayRegister<super::F64x2V2, 8>> for ArrayRegister<super::I16x8V2, 2> {
    fn cast_from(value: Storage<ArrayRegister<super::F64x2V2, 8>>) -> Storage<Self> {
        let v = value.0;
        unsafe {
            let mask = arch::_mm_narrow_dword_to_word_maskx_v2();
            let lo0 = arch::_mm_shuffle_epi8(arch::_mm_cvtt2pd_epi32x_v2([v[0], v[1]]), mask);
            let lo1 = arch::_mm_shuffle_epi8(arch::_mm_cvtt2pd_epi32x_v2([v[2], v[3]]), mask);
            let hi0 = arch::_mm_shuffle_epi8(arch::_mm_cvtt2pd_epi32x_v2([v[4], v[5]]), mask);
            let hi1 = arch::_mm_shuffle_epi8(arch::_mm_cvtt2pd_epi32x_v2([v[6], v[7]]), mask);
            ArrayRegister([arch::_mm_unpacklo_epi64(lo0, lo1), arch::_mm_unpacklo_epi64(hi0, hi1)])
        }
    }
}
#[thermite_macros::inline_always]
impl CastRegister<ArrayRegister<super::F64x2V2, 8>> for ArrayRegister<super::U16x8V2, 2> {
    fn cast_from(value: Storage<ArrayRegister<super::F64x2V2, 8>>) -> Storage<Self> {
        let v = value.0;
        unsafe {
            let mask = arch::_mm_narrow_dword_to_word_maskx_v2();
            let lo0 = arch::_mm_shuffle_epi8(arch::_mm_cvtt2pd_epi32x_v2([v[0], v[1]]), mask);
            let lo1 = arch::_mm_shuffle_epi8(arch::_mm_cvtt2pd_epi32x_v2([v[2], v[3]]), mask);
            let hi0 = arch::_mm_shuffle_epi8(arch::_mm_cvtt2pd_epi32x_v2([v[4], v[5]]), mask);
            let hi1 = arch::_mm_shuffle_epi8(arch::_mm_cvtt2pd_epi32x_v2([v[6], v[7]]), mask);
            ArrayRegister([arch::_mm_unpacklo_epi64(lo0, lo1), arch::_mm_unpacklo_epi64(hi0, hi1)])
        }
    }
}

// ---------------------------------------------------------------------------------------
// Mask-side concat: the reduced i16x4 register is its own Mask, and `FullConcatRegister`
// requires that Mask to concat from the i16x2 half's Mask (`ArrayRegister<bool, 2>`).
// A bool lane maps to a full-width (all-ones / all-zeros) 16-bit mask lane.
// ---------------------------------------------------------------------------------------

#[inline(always)]
fn bool_to_i16_mask(b: bool) -> i16 {
    if b { !0 } else { 0 }
}

#[thermite_macros::inline_always]
impl ConcatRegister<ArrayRegister<bool, 2>> for I16x4V2 {
    fn concat(lo: Storage<ArrayRegister<bool, 2>>, hi: Storage<ArrayRegister<bool, 2>>) -> Storage<Self> {
        ReducedRegister::new(unsafe {
            arch::_mm_setr_epi16(
                bool_to_i16_mask(lo.0[0]),
                bool_to_i16_mask(lo.0[1]),
                bool_to_i16_mask(hi.0[0]),
                bool_to_i16_mask(hi.0[1]),
                0,
                0,
                0,
                0,
            )
        })
    }

    fn split(value: Storage<Self>) -> (Storage<ArrayRegister<bool, 2>>, Storage<ArrayRegister<bool, 2>>) {
        let mut arr = [0i16; 8];
        unsafe { arch::_mm_storeu_si128(arr.as_mut_ptr() as *mut _, value.0) };
        (
            ArrayRegister([arr[0] != 0, arr[1] != 0]),
            ArrayRegister([arr[2] != 0, arr[3] != 0]),
        )
    }
}

#[thermite_macros::inline_always]
impl ExtendRegister<ArrayRegister<bool, 2>> for I16x4V2 {
    fn extend(value: Storage<ArrayRegister<bool, 2>>) -> Storage<Self> {
        ReducedRegister::new(unsafe {
            arch::_mm_setr_epi16(
                bool_to_i16_mask(value.0[0]),
                bool_to_i16_mask(value.0[1]),
                0,
                0,
                0,
                0,
                0,
                0,
            )
        })
    }

    fn narrow(value: Storage<Self>) -> Storage<ArrayRegister<bool, 2>> {
        let mut arr = [0i16; 8];
        unsafe { arch::_mm_storeu_si128(arr.as_mut_ptr() as *mut _, value.0) };
        ArrayRegister([arr[0] != 0, arr[1] != 0])
    }
}

#[thermite_macros::inline_always]
impl ConcatRegister<ArrayRegister<bool, 2>> for U16x4V2 {
    fn concat(lo: Storage<ArrayRegister<bool, 2>>, hi: Storage<ArrayRegister<bool, 2>>) -> Storage<Self> {
        ReducedRegister::new(unsafe {
            arch::_mm_setr_epi16(
                bool_to_i16_mask(lo.0[0]),
                bool_to_i16_mask(lo.0[1]),
                bool_to_i16_mask(hi.0[0]),
                bool_to_i16_mask(hi.0[1]),
                0,
                0,
                0,
                0,
            )
        })
    }

    fn split(value: Storage<Self>) -> (Storage<ArrayRegister<bool, 2>>, Storage<ArrayRegister<bool, 2>>) {
        let mut arr = [0i16; 8];
        unsafe { arch::_mm_storeu_si128(arr.as_mut_ptr() as *mut _, value.0) };
        (
            ArrayRegister([arr[0] != 0, arr[1] != 0]),
            ArrayRegister([arr[2] != 0, arr[3] != 0]),
        )
    }
}

#[thermite_macros::inline_always]
impl ExtendRegister<ArrayRegister<bool, 2>> for U16x4V2 {
    fn extend(value: Storage<ArrayRegister<bool, 2>>) -> Storage<Self> {
        ReducedRegister::new(unsafe {
            arch::_mm_setr_epi16(
                bool_to_i16_mask(value.0[0]),
                bool_to_i16_mask(value.0[1]),
                0,
                0,
                0,
                0,
                0,
                0,
            )
        })
    }

    fn narrow(value: Storage<Self>) -> Storage<ArrayRegister<bool, 2>> {
        let mut arr = [0i16; 8];
        unsafe { arch::_mm_storeu_si128(arr.as_mut_ptr() as *mut _, value.0) };
        ArrayRegister([arr[0] != 0, arr[1] != 0])
    }
}

// Gather/scatter for the reduced 16-bit registers (indexed by the native 4-lane index
// types) falls back to scalar. The reduced register inherits same-type `IndexableRegister`
// from its inner register's blanket, so only the cross-type index markers are needed.
impl IndexableRegister<<super::super::X86V2 as crate::simd::Simd>::u32x4> for I16x4V2 {}
impl IndexableRegister<<super::super::X86V2 as crate::simd::Simd>::u32x4> for U16x4V2 {}
impl IndexableRegister<<super::super::X86V2 as crate::simd::Simd>::u64x4> for I16x4V2 {}
impl IndexableRegister<<super::super::X86V2 as crate::simd::Simd>::u64x4> for U16x4V2 {}
