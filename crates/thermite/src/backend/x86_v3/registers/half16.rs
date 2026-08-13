//! Reduced (sub-native-width) 16-bit registers for x86-v3 and the cast/concat glue bridging
//! the scalar/array 16-bit halves, the native 128-bit `I16x8V3`/`U16x8V3`, and the 32-bit
//! registers they widen into. Mirrors the v2 `half16.rs`.

use generic_array::typenum::U4;

use super::arch;

use crate::register::{
    CastRegister, ConcatRegister, ExtendRegister, IndexableRegister, NumericRegister, Register, Storage,
    array::ArrayRegister, reduced::ReducedRegister,
};

// Saturating narrow via clamp + the truncating narrow above, for the pack-less pairs:
// `i64 -> i16` (no AVX2 64-bit pack) and the 2-lane `i32 -> i16` combos whose destination
// is a scalar `ArrayRegister`. Clamp the source into `[INTO::MIN, INTO::MAX]` (the 64-bit
// min/max is polyfilled; the 32-bit one is native), then reuse the existing `cast_from`.
macro_rules! sat_clamp_narrow16 {
    ($from:ty, $fe:ty, $ie:ty) => {
        #[inline(always)]
        fn saturating_cast_from(value: Storage<$from>) -> Storage<Self> {
            let lo = <$from as Register>::splat(<$ie>::MIN as $fe);
            let hi = <$from as Register>::splat(<$ie>::MAX as $fe);
            let clamped = <$from as NumericRegister>::min(<$from as NumericRegister>::max(value, lo), hi);
            <Self as CastRegister<$from>>::cast_from(clamped)
        }
    };
}

/// 4-lane signed 16-bit register, backed by the low 4 lanes of a 128-bit `I16x8V3`.
pub type I16x4V3 = ReducedRegister<super::I16x8V3, U4>;
/// 4-lane unsigned 16-bit register, backed by the low 4 lanes of a 128-bit `U16x8V3`.
pub type U16x4V3 = ReducedRegister<super::U16x8V3, U4>;

// --- x4 <- x2 (two scalar-array halves into a 4-lane reduced register) ---

#[thermite_macros::inline_always]
impl ConcatRegister<ArrayRegister<i16, 2>> for I16x4V3 {
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
impl ExtendRegister<ArrayRegister<i16, 2>> for I16x4V3 {
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
impl ConcatRegister<ArrayRegister<u16, 2>> for U16x4V3 {
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
impl ExtendRegister<ArrayRegister<u16, 2>> for U16x4V3 {
    fn extend(value: Storage<ArrayRegister<u16, 2>>) -> Storage<Self> {
        ReducedRegister::new(unsafe { arch::_mm_setr_epi16(value.0[0] as i16, value.0[1] as i16, 0, 0, 0, 0, 0, 0) })
    }

    fn narrow(value: Storage<Self>) -> Storage<ArrayRegister<u16, 2>> {
        let mut arr = [0u16; 8];
        unsafe { arch::_mm_storeu_si128(arr.as_mut_ptr() as *mut _, value.0) };
        ArrayRegister([arr[0], arr[1]])
    }
}

// --- x8 <- x4 (two 4-lane reduced halves into the native 8-lane register) ---

#[thermite_macros::inline_always]
impl ConcatRegister<I16x4V3> for super::I16x8V3 {
    fn concat(lo: Storage<I16x4V3>, hi: Storage<I16x4V3>) -> Storage<Self> {
        unsafe { arch::_mm_unpacklo_epi64(lo.0, hi.0) }
    }

    fn split(value: Storage<Self>) -> (Storage<I16x4V3>, Storage<I16x4V3>) {
        (
            ReducedRegister::new(value),
            ReducedRegister::new(unsafe { arch::_mm_unpackhi_epi64(value, value) }),
        )
    }
}

// `ExtendRegister<I16x4V3> for I16x8V3` is provided by the ReducedRegister blanket.

#[thermite_macros::inline_always]
impl ConcatRegister<U16x4V3> for super::U16x8V3 {
    fn concat(lo: Storage<U16x4V3>, hi: Storage<U16x4V3>) -> Storage<Self> {
        unsafe { arch::_mm_unpacklo_epi64(lo.0, hi.0) }
    }

    fn split(value: Storage<Self>) -> (Storage<U16x4V3>, Storage<U16x4V3>) {
        (
            ReducedRegister::new(value),
            ReducedRegister::new(unsafe { arch::_mm_unpackhi_epi64(value, value) }),
        )
    }
}

// --- mask-side concat (the reduced register is its own Mask; concat from bool-array half) ---

#[inline(always)]
fn bool_to_i16_mask(b: bool) -> i16 {
    if b { !0 } else { 0 }
}

macro_rules! impl_bool_concat {
    ($ty:ty) => {
        #[thermite_macros::inline_always]
        impl ConcatRegister<ArrayRegister<bool, 2>> for $ty {
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
        impl ExtendRegister<ArrayRegister<bool, 2>> for $ty {
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
    };
}

impl_bool_concat!(I16x4V3);
impl_bool_concat!(U16x4V3);

// --- widen casts to 32-bit ---
// x4: I16x4V3 <-> I32x4V3 (native 4-lane). x2: ArrayRegister<i16,2> <-> I32x2V3 (reduced).

#[thermite_macros::inline_always]
impl CastRegister<I16x4V3> for super::I32x4V3 {
    fn cast_from(value: Storage<I16x4V3>) -> Storage<Self> {
        unsafe { arch::_mm_cvtepi16_epi32(value.0) }
    }
}

#[thermite_macros::inline_always]
impl CastRegister<U16x4V3> for super::U32x4V3 {
    fn cast_from(value: Storage<U16x4V3>) -> Storage<Self> {
        unsafe { arch::_mm_cvtepu16_epi32(value.0) }
    }
}

#[thermite_macros::inline_always]
impl CastRegister<super::I32x4V3> for I16x4V3 {
    fn cast_from(value: Storage<super::I32x4V3>) -> Storage<Self> {
        ReducedRegister::new(unsafe {
            arch::_mm_shuffle_epi8(
                value,
                arch::_mm_setr_epi8(0, 1, 4, 5, 8, 9, 12, 13, -1, -1, -1, -1, -1, -1, -1, -1),
            )
        })
    }

    // i32x4 -> i16x4 via `vpackssdw` (saturates to [i16::MIN, i16::MAX]).
    fn saturating_cast_from(value: Storage<super::I32x4V3>) -> Storage<Self> {
        ReducedRegister::new(unsafe { arch::_mm_packs_epi32(value, value) })
    }
}

#[thermite_macros::inline_always]
impl CastRegister<super::U32x4V3> for U16x4V3 {
    fn cast_from(value: Storage<super::U32x4V3>) -> Storage<Self> {
        ReducedRegister::new(unsafe {
            arch::_mm_shuffle_epi8(
                value,
                arch::_mm_setr_epi8(0, 1, 4, 5, 8, 9, 12, 13, -1, -1, -1, -1, -1, -1, -1, -1),
            )
        })
    }

    // u32x4 -> u16x4: clamp the high end (`vpminud`) then `vpackusdw` (signed-source pack).
    fn saturating_cast_from(value: Storage<super::U32x4V3>) -> Storage<Self> {
        let clamped = unsafe { arch::_mm_min_epu32(value, arch::_mm_set1_epi32(0xFFFF)) };
        ReducedRegister::new(unsafe { arch::_mm_packus_epi32(clamped, clamped) })
    }
}

// --- saturating narrows into 16-bit (low half holds the result; no lane-crossing fixup) ---

#[thermite_macros::inline_always]
impl CastRegister<ArrayRegister<i16, 2>> for super::half::I32x2V3 {
    fn cast_from(value: Storage<ArrayRegister<i16, 2>>) -> Storage<Self> {
        ReducedRegister::new(unsafe { arch::_mm_setr_epi32(value.0[0] as i32, value.0[1] as i32, 0, 0) })
    }
}

#[thermite_macros::inline_always]
impl CastRegister<ArrayRegister<u16, 2>> for super::half::U32x2V3 {
    fn cast_from(value: Storage<ArrayRegister<u16, 2>>) -> Storage<Self> {
        ReducedRegister::new(unsafe { arch::_mm_setr_epi32(value.0[0] as i32, value.0[1] as i32, 0, 0) })
    }
}

#[thermite_macros::inline_always]
impl CastRegister<super::half::I32x2V3> for ArrayRegister<i16, 2> {
    fn cast_from(value: Storage<super::half::I32x2V3>) -> Storage<Self> {
        let mut arr = [0i32; 4];
        unsafe { arch::_mm_storeu_si128(arr.as_mut_ptr() as *mut _, value.0) };
        ArrayRegister([arr[0] as i16, arr[1] as i16])
    }

    sat_clamp_narrow16!(super::half::I32x2V3, i32, i16);
}

#[thermite_macros::inline_always]
impl CastRegister<super::half::U32x2V3> for ArrayRegister<u16, 2> {
    fn cast_from(value: Storage<super::half::U32x2V3>) -> Storage<Self> {
        let mut arr = [0u32; 4];
        unsafe { arch::_mm_storeu_si128(arr.as_mut_ptr() as *mut _, value.0) };
        ArrayRegister([arr[0] as u16, arr[1] as u16])
    }

    sat_clamp_narrow16!(super::half::U32x2V3, u32, u16);
}

// ---------------------------------------------------------------------------------------
// Widen/narrow casts to 64-bit.
//   widen 16 -> 64 via `_mm256_cvtepi16_epi64` (low 4 words -> 4x i64 in a __m256i).
//   narrow 64 -> 16 by extracting the two 128-bit halves of each I64x4V3 and pshufb-gathering
//   word 0 of each i64 lane.
//   x2:  ArrayRegister<i16,2> <-> I64x2V3 (native 128-bit, 2-lane).
//   x4:  I16x4V3 <-> I64x4V3 (native 256-bit, 4-lane).
//   x8:  I16x8V3 <-> ArrayRegister<I64x4V3, 2> (the v3 i64x8).
//   x16: I16x16V3 (native 256-bit) <-> ArrayRegister<I64x4V3, 4> (the v3 i64x16).
// ---------------------------------------------------------------------------------------

// --- x2 widen / narrow (ArrayRegister<i16,2> <-> I64x2V3 native 128-bit) ---
#[thermite_macros::inline_always]
impl CastRegister<ArrayRegister<i16, 2>> for super::I64x2V3 {
    fn cast_from(value: Storage<ArrayRegister<i16, 2>>) -> Storage<Self> {
        unsafe { arch::_mm_set_epi64x(value.0[1] as i64, value.0[0] as i64) }
    }
}
#[thermite_macros::inline_always]
impl CastRegister<ArrayRegister<u16, 2>> for super::U64x2V3 {
    fn cast_from(value: Storage<ArrayRegister<u16, 2>>) -> Storage<Self> {
        unsafe { arch::_mm_set_epi64x(value.0[1] as i64, value.0[0] as i64) }
    }
}
#[thermite_macros::inline_always]
impl CastRegister<super::I64x2V3> for ArrayRegister<i16, 2> {
    fn cast_from(value: Storage<super::I64x2V3>) -> Storage<Self> {
        let mut arr = [0i64; 2];
        unsafe { arch::_mm_storeu_si128(arr.as_mut_ptr() as *mut _, value) };
        ArrayRegister([arr[0] as i16, arr[1] as i16])
    }

    sat_clamp_narrow16!(super::I64x2V3, i64, i16);
}
#[thermite_macros::inline_always]
impl CastRegister<super::U64x2V3> for ArrayRegister<u16, 2> {
    fn cast_from(value: Storage<super::U64x2V3>) -> Storage<Self> {
        let mut arr = [0u64; 2];
        unsafe { arch::_mm_storeu_si128(arr.as_mut_ptr() as *mut _, value) };
        ArrayRegister([arr[0] as u16, arr[1] as u16])
    }

    sat_clamp_narrow16!(super::U64x2V3, u64, u16);
}

// --- x4 widen i16 -> i64 (low 4 words -> 4x i64 in a __m256i) ---
#[thermite_macros::inline_always]
impl CastRegister<I16x4V3> for super::I64x4V3 {
    fn cast_from(value: Storage<I16x4V3>) -> Storage<Self> {
        unsafe { arch::_mm256_cvtepi16_epi64(value.0) }
    }
}
#[thermite_macros::inline_always]
impl CastRegister<U16x4V3> for super::U64x4V3 {
    fn cast_from(value: Storage<U16x4V3>) -> Storage<Self> {
        unsafe { arch::_mm256_cvtepu16_epi64(value.0) }
    }
}
#[thermite_macros::inline_always]
impl CastRegister<super::I64x4V3> for I16x4V3 {
    fn cast_from(value: Storage<super::I64x4V3>) -> Storage<Self> {
        ReducedRegister::new(unsafe { arch::_mm256_cvtepi64_epi16x_v3(value) })
    }

    sat_clamp_narrow16!(super::I64x4V3, i64, i16);
}
#[thermite_macros::inline_always]
impl CastRegister<super::U64x4V3> for U16x4V3 {
    fn cast_from(value: Storage<super::U64x4V3>) -> Storage<Self> {
        ReducedRegister::new(unsafe { arch::_mm256_cvtepi64_epi16x_v3(value) })
    }

    sat_clamp_narrow16!(super::U64x4V3, u64, u16);
}

// --- x8 widen i16 -> i64 (8 words of a __m128i -> two 4x i64 __m256i lanes) ---
#[thermite_macros::inline_always]
impl CastRegister<super::I16x8V3> for ArrayRegister<super::I64x4V3, 2> {
    fn cast_from(value: Storage<super::I16x8V3>) -> Storage<Self> {
        unsafe {
            ArrayRegister([
                arch::_mm256_cvtepi16_epi64(value),                          // low 4 words -> 4x i64
                arch::_mm256_cvtepi16_epi64(arch::_mm_srli_si128(value, 8)), // next 4 words
            ])
        }
    }
}
#[thermite_macros::inline_always]
impl CastRegister<super::U16x8V3> for ArrayRegister<super::U64x4V3, 2> {
    fn cast_from(value: Storage<super::U16x8V3>) -> Storage<Self> {
        unsafe {
            ArrayRegister([
                arch::_mm256_cvtepu16_epi64(value),
                arch::_mm256_cvtepu16_epi64(arch::_mm_srli_si128(value, 8)),
            ])
        }
    }
}

// --- x16 widen i16 -> i64 (native 256-bit I16x16V3 -> ArrayRegister<I64x4V3, 4>) ---
// Split the 256-bit source into two 128-bit halves (8 words each); each half widens to two
// 256-bit i64x4 lanes (4 words at a time).
#[thermite_macros::inline_always]
impl CastRegister<super::I16x16V3> for ArrayRegister<super::I64x4V3, 4> {
    fn cast_from(value: Storage<super::I16x16V3>) -> Storage<Self> {
        unsafe {
            let lo = arch::_mm256_castsi256_si128(value); // words 0..8
            let hi = arch::_mm256_extracti128_si256(value, 1); // words 8..16
            ArrayRegister([
                arch::_mm256_cvtepi16_epi64(lo),
                arch::_mm256_cvtepi16_epi64(arch::_mm_srli_si128(lo, 8)),
                arch::_mm256_cvtepi16_epi64(hi),
                arch::_mm256_cvtepi16_epi64(arch::_mm_srli_si128(hi, 8)),
            ])
        }
    }
}
#[thermite_macros::inline_always]
impl CastRegister<super::U16x16V3> for ArrayRegister<super::U64x4V3, 4> {
    fn cast_from(value: Storage<super::U16x16V3>) -> Storage<Self> {
        unsafe {
            let lo = arch::_mm256_castsi256_si128(value);
            let hi = arch::_mm256_extracti128_si256(value, 1);
            ArrayRegister([
                arch::_mm256_cvtepu16_epi64(lo),
                arch::_mm256_cvtepu16_epi64(arch::_mm_srli_si128(lo, 8)),
                arch::_mm256_cvtepu16_epi64(hi),
                arch::_mm256_cvtepu16_epi64(arch::_mm_srli_si128(hi, 8)),
            ])
        }
    }
}

// ---------------------------------------------------------------------------------------
// 16 <-> f32/f64 direct casts.
//   widen i16/u16 -> f32/f64 = (int widen to i32, same lanes) then (i32 -> float convert).
//     unsigned widens zero-extend (`cvtepu16_epi32`); the widened value fits in positive i32,
//     so the signed `cvtepi32_ps`/`cvtepi32_pd` convert is exact.
//   narrow float -> i16/u16 = (float -> i32 truncating convert, like `as`) then (i32 -> word
//     narrow). The low word is identical signed vs unsigned, so the signed narrow is reused.
//   f32 native widths: f32x8 (256-bit), f32x4 (128-bit). f64 native widths: f64x4 (256-bit),
//     f64x2 (128-bit). The f64 path fans out half as many lanes per native register.
// ---------------------------------------------------------------------------------------

// --- x2 (ArrayRegister<i16,2> <-> F32x2V3 / F64x2V3) ---
// widen: build an i32x4 [v0, v1, 0, 0] then the i32 -> float convert. narrow: float -> i32
// truncating convert, then store and take the low 2 lanes `as i16`/`as u16`.
#[thermite_macros::inline_always]
impl CastRegister<ArrayRegister<i16, 2>> for super::half::F32x2V3 {
    fn cast_from(value: Storage<ArrayRegister<i16, 2>>) -> Storage<Self> {
        ReducedRegister::new(unsafe {
            let widened = arch::_mm_setr_epi32(value.0[0] as i32, value.0[1] as i32, 0, 0);
            arch::_mm_cvtepi32_ps(widened)
        })
    }
}
#[thermite_macros::inline_always]
impl CastRegister<ArrayRegister<u16, 2>> for super::half::F32x2V3 {
    fn cast_from(value: Storage<ArrayRegister<u16, 2>>) -> Storage<Self> {
        ReducedRegister::new(unsafe {
            let widened = arch::_mm_setr_epi32(value.0[0] as i32, value.0[1] as i32, 0, 0);
            arch::_mm_cvtepi32_ps(widened)
        })
    }
}
#[thermite_macros::inline_always]
impl CastRegister<ArrayRegister<i16, 2>> for super::F64x2V3 {
    fn cast_from(value: Storage<ArrayRegister<i16, 2>>) -> Storage<Self> {
        unsafe {
            let widened = arch::_mm_setr_epi32(value.0[0] as i32, value.0[1] as i32, 0, 0);
            arch::_mm_cvtepi32_pd(widened)
        }
    }
}
#[thermite_macros::inline_always]
impl CastRegister<ArrayRegister<u16, 2>> for super::F64x2V3 {
    fn cast_from(value: Storage<ArrayRegister<u16, 2>>) -> Storage<Self> {
        unsafe {
            let widened = arch::_mm_setr_epi32(value.0[0] as i32, value.0[1] as i32, 0, 0);
            arch::_mm_cvtepi32_pd(widened)
        }
    }
}

// --- x4 (I16x4V3 <-> F32x4V3 native 128-bit / F64x4V3 native 256-bit) ---
// f32x4: widen low-4 i16 to i32x4 then `_mm_cvtepi32_ps`. f64x4: widen low-4 i16 to i32x4 then
// `_mm256_cvtepi32_pd`. narrow: float -> i32 truncating convert then pshufb the low word of each
// dword into the low 4 words.
#[thermite_macros::inline_always]
impl CastRegister<I16x4V3> for super::F32x4V3 {
    fn cast_from(value: Storage<I16x4V3>) -> Storage<Self> {
        unsafe {
            let widened = arch::_mm_cvtepi16_epi32(value.0);
            arch::_mm_cvtepi32_ps(widened)
        }
    }
}
#[thermite_macros::inline_always]
impl CastRegister<U16x4V3> for super::F32x4V3 {
    fn cast_from(value: Storage<U16x4V3>) -> Storage<Self> {
        unsafe {
            let widened = arch::_mm_cvtepu16_epi32(value.0);
            arch::_mm_cvtepi32_ps(widened)
        }
    }
}
#[thermite_macros::inline_always]
impl CastRegister<I16x4V3> for super::F64x4V3 {
    fn cast_from(value: Storage<I16x4V3>) -> Storage<Self> {
        unsafe {
            let widened = arch::_mm_cvtepi16_epi32(value.0); // low 4 words -> 4x i32 (128-bit)
            arch::_mm256_cvtepi32_pd(widened) // 4x i32 -> 4x f64 (256-bit)
        }
    }
}
#[thermite_macros::inline_always]
impl CastRegister<U16x4V3> for super::F64x4V3 {
    fn cast_from(value: Storage<U16x4V3>) -> Storage<Self> {
        unsafe {
            let widened = arch::_mm_cvtepu16_epi32(value.0);
            arch::_mm256_cvtepi32_pd(widened)
        }
    }
}

// --- x8 (I16x8V3 native 128-bit <-> F32x8V3 native 256-bit / ArrayRegister<F64x4V3, 2>) ---
// f32x8: widen 8 words to one i32x8 (`_mm256_cvtepi16_epi32`) then `_mm256_cvtepi32_ps`. f64x8:
// widen low-4 and next-4 words to two i32x4 then `_mm256_cvtepi32_pd` each. narrow reverses.
#[thermite_macros::inline_always]
impl CastRegister<super::I16x8V3> for super::F32x8V3 {
    fn cast_from(value: Storage<super::I16x8V3>) -> Storage<Self> {
        unsafe {
            let widened = arch::_mm256_cvtepi16_epi32(value); // 8 words -> 8x i32
            arch::_mm256_cvtepi32_ps(widened)
        }
    }
}
#[thermite_macros::inline_always]
impl CastRegister<super::U16x8V3> for super::F32x8V3 {
    fn cast_from(value: Storage<super::U16x8V3>) -> Storage<Self> {
        unsafe {
            let widened = arch::_mm256_cvtepu16_epi32(value);
            arch::_mm256_cvtepi32_ps(widened)
        }
    }
}
#[thermite_macros::inline_always]
impl CastRegister<super::I16x8V3> for ArrayRegister<super::F64x4V3, 2> {
    fn cast_from(value: Storage<super::I16x8V3>) -> Storage<Self> {
        unsafe {
            ArrayRegister([
                arch::_mm256_cvtepi32_pd(arch::_mm_cvtepi16_epi32(value)), // low 4 words -> f64x4
                arch::_mm256_cvtepi32_pd(arch::_mm_cvtepi16_epi32(arch::_mm_srli_si128(value, 8))), // next 4
            ])
        }
    }
}
#[thermite_macros::inline_always]
impl CastRegister<super::U16x8V3> for ArrayRegister<super::F64x4V3, 2> {
    fn cast_from(value: Storage<super::U16x8V3>) -> Storage<Self> {
        unsafe {
            ArrayRegister([
                arch::_mm256_cvtepi32_pd(arch::_mm_cvtepu16_epi32(value)),
                arch::_mm256_cvtepi32_pd(arch::_mm_cvtepu16_epi32(arch::_mm_srli_si128(value, 8))),
            ])
        }
    }
}

// --- x16 (I16x16V3 native 256-bit <-> ArrayRegister<F32x8V3, 2> / ArrayRegister<F64x4V3, 4>) ---
// f32x16: split the 256-bit source into two 128-bit halves (8 words each); widen each to one
// i32x8 then `_mm256_cvtepi32_ps`. f64x16: widen each half to two i32x4 then `_mm256_cvtepi32_pd`.
// narrow reverses, rebuilding the 256-bit result with `_mm256_set_m128i`.
#[thermite_macros::inline_always]
impl CastRegister<super::I16x16V3> for ArrayRegister<super::F32x8V3, 2> {
    fn cast_from(value: Storage<super::I16x16V3>) -> Storage<Self> {
        unsafe {
            let lo = arch::_mm256_castsi256_si128(value); // words 0..8
            let hi = arch::_mm256_extracti128_si256(value, 1); // words 8..16
            ArrayRegister([
                arch::_mm256_cvtepi32_ps(arch::_mm256_cvtepi16_epi32(lo)),
                arch::_mm256_cvtepi32_ps(arch::_mm256_cvtepi16_epi32(hi)),
            ])
        }
    }
}
#[thermite_macros::inline_always]
impl CastRegister<super::U16x16V3> for ArrayRegister<super::F32x8V3, 2> {
    fn cast_from(value: Storage<super::U16x16V3>) -> Storage<Self> {
        unsafe {
            let lo = arch::_mm256_castsi256_si128(value);
            let hi = arch::_mm256_extracti128_si256(value, 1);
            ArrayRegister([
                arch::_mm256_cvtepi32_ps(arch::_mm256_cvtepu16_epi32(lo)),
                arch::_mm256_cvtepi32_ps(arch::_mm256_cvtepu16_epi32(hi)),
            ])
        }
    }
}
#[thermite_macros::inline_always]
impl CastRegister<super::I16x16V3> for ArrayRegister<super::F64x4V3, 4> {
    fn cast_from(value: Storage<super::I16x16V3>) -> Storage<Self> {
        unsafe {
            let lo = arch::_mm256_castsi256_si128(value); // words 0..8
            let hi = arch::_mm256_extracti128_si256(value, 1); // words 8..16
            ArrayRegister([
                arch::_mm256_cvtepi32_pd(arch::_mm_cvtepi16_epi32(lo)),
                arch::_mm256_cvtepi32_pd(arch::_mm_cvtepi16_epi32(arch::_mm_srli_si128(lo, 8))),
                arch::_mm256_cvtepi32_pd(arch::_mm_cvtepi16_epi32(hi)),
                arch::_mm256_cvtepi32_pd(arch::_mm_cvtepi16_epi32(arch::_mm_srli_si128(hi, 8))),
            ])
        }
    }
}
#[thermite_macros::inline_always]
impl CastRegister<super::U16x16V3> for ArrayRegister<super::F64x4V3, 4> {
    fn cast_from(value: Storage<super::U16x16V3>) -> Storage<Self> {
        unsafe {
            let lo = arch::_mm256_castsi256_si128(value);
            let hi = arch::_mm256_extracti128_si256(value, 1);
            ArrayRegister([
                arch::_mm256_cvtepi32_pd(arch::_mm_cvtepu16_epi32(lo)),
                arch::_mm256_cvtepi32_pd(arch::_mm_cvtepu16_epi32(arch::_mm_srli_si128(lo, 8))),
                arch::_mm256_cvtepi32_pd(arch::_mm_cvtepu16_epi32(hi)),
                arch::_mm256_cvtepi32_pd(arch::_mm_cvtepu16_epi32(arch::_mm_srli_si128(hi, 8))),
            ])
        }
    }
}

// --- scalar-fallback gather markers for the reduced 16-bit registers ---
impl IndexableRegister<<super::super::X86V3 as crate::simd::Simd>::u32x4> for I16x4V3 {}
impl IndexableRegister<<super::super::X86V3 as crate::simd::Simd>::u32x4> for U16x4V3 {}
impl IndexableRegister<<super::super::X86V3 as crate::simd::Simd>::u64x4> for I16x4V3 {}
impl IndexableRegister<<super::super::X86V3 as crate::simd::Simd>::u64x4> for U16x4V3 {}
