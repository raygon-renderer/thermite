//! Reduced (sub-native-width) 8-bit registers for x86-v3. Mirrors `half16.rs` one element size
//! down; the only structural difference from the v2 `half8.rs` is the x8 widen/narrow, which goes
//! to the *native* 256-bit `I32x8V3` via `_mm256_cvtepi8_epi32`.

use generic_array::typenum::{U8, U12};

use super::arch;

use crate::register::{
    CastRegister, ConcatRegister, ExtendRegister, IndexableRegister, NumericRegister, Register, SaturatingCastRegister,
    Storage, array::ArrayRegister, reduced::ReducedRegister,
};

// Saturating narrow via clamp + the truncating narrow, for the pack-less `* -> i8` pairs:
// `i64 -> i8` (no AVX2 64-bit pack) and the 2-lane `i32 -> i8` combos with a scalar
// `ArrayRegister` destination. See `half16.rs` for the rationale; `i32 -> i8` could compose
// two packs, but the destinations here are scalar arrays, so clamp + the existing narrow is
// both simpler and the same cost.
macro_rules! sat_clamp_narrow8 {
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

sat_clamp_narrow8! {
    (super::I64x4V3, i64, I8x4V3, i8),
    (super::U64x4V3, u64, U8x4V3, u8),
    (super::half::I32x2V3, i32, ArrayRegister<i8, 2>, i8),
    (super::half::U32x2V3, u32, ArrayRegister<u8, 2>, u8),
    (super::I64x2V3, i64, ArrayRegister<i8, 2>, i8),
    (super::U64x2V3, u64, ArrayRegister<u8, 2>, u8),
}

// `ReducedRegister<R, N>` removes `N` lanes (lane count = R::Lanes - N). x4/x8 reduce over the
// native 128-bit `U8x16V3` (v3's `u8x16` is native 128-bit; x16 stays native, not reduced).
/// 4-lane signed 8-bit register, backed by the low 4 lanes of a 128-bit `I8x16V3`.
pub type I8x4V3 = ReducedRegister<super::I8x16V3, U12>;
/// 4-lane unsigned 8-bit register.
pub type U8x4V3 = ReducedRegister<super::U8x16V3, U12>;
/// 8-lane signed 8-bit register, backed by the low 8 lanes of a 128-bit `I8x16V3`.
pub type I8x8V3 = ReducedRegister<super::I8x16V3, U8>;
/// 8-lane unsigned 8-bit register.
pub type U8x8V3 = ReducedRegister<super::U8x16V3, U8>;

#[inline(always)]
fn store_bytes(v: arch::__m128i) -> [i8; 16] {
    let mut arr = [0i8; 16];
    unsafe { arch::_mm_storeu_si128(arr.as_mut_ptr() as *mut _, v) };
    arr
}

// ===========================================================================================
// x4 <- x2 (scalar-array) concat / extend.
// ===========================================================================================

macro_rules! impl_concat_x4_from_x2 {
    ($red:ty, $elem:ty) => {
        #[thermite_macros::inline_always]
        impl ConcatRegister<ArrayRegister<$elem, 2>> for $red {
            fn concat(lo: Storage<ArrayRegister<$elem, 2>>, hi: Storage<ArrayRegister<$elem, 2>>) -> Storage<Self> {
                ReducedRegister::new(unsafe {
                    arch::_mm_setr_epi8(
                        lo.0[0] as i8, lo.0[1] as i8, hi.0[0] as i8, hi.0[1] as i8,
                        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                    )
                })
            }
            fn split(value: Storage<Self>) -> (Storage<ArrayRegister<$elem, 2>>, Storage<ArrayRegister<$elem, 2>>) {
                let a = store_bytes(value.0);
                (ArrayRegister([a[0] as $elem, a[1] as $elem]), ArrayRegister([a[2] as $elem, a[3] as $elem]))
            }
        }

        #[thermite_macros::inline_always]
        impl ExtendRegister<ArrayRegister<$elem, 2>> for $red {
            fn extend(value: Storage<ArrayRegister<$elem, 2>>) -> Storage<Self> {
                ReducedRegister::new(unsafe {
                    arch::_mm_setr_epi8(value.0[0] as i8, value.0[1] as i8, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)
                })
            }
            fn narrow(value: Storage<Self>) -> Storage<ArrayRegister<$elem, 2>> {
                let a = store_bytes(value.0);
                ArrayRegister([a[0] as $elem, a[1] as $elem])
            }
        }
    };
}

impl_concat_x4_from_x2!(I8x4V3, i8);
impl_concat_x4_from_x2!(U8x4V3, u8);

// ===========================================================================================
// x8 <- x4 (both reduced over the same native register; relate them explicitly).
// ===========================================================================================

macro_rules! impl_concat_x8_from_x4 {
    ($x8:ty, $x4:ty) => {
        #[thermite_macros::inline_always]
        impl ExtendRegister<$x4> for $x8 {
            fn extend(value: Storage<$x4>) -> Storage<Self> {
                ReducedRegister::new(value.0)
            }
            fn narrow(value: Storage<Self>) -> Storage<$x4> {
                ReducedRegister::new(value.0)
            }
        }

        #[thermite_macros::inline_always]
        impl ConcatRegister<$x4> for $x8 {
            fn concat(lo: Storage<$x4>, hi: Storage<$x4>) -> Storage<Self> {
                ReducedRegister::new(unsafe { arch::_mm_unpacklo_epi32(lo.0, hi.0) })
            }
            fn split(value: Storage<Self>) -> (Storage<$x4>, Storage<$x4>) {
                (
                    ReducedRegister::new(value.0),
                    ReducedRegister::new(unsafe { arch::_mm_srli_si128(value.0, 4) }),
                )
            }
        }
    };
}

impl_concat_x8_from_x4!(I8x8V3, I8x4V3);
impl_concat_x8_from_x4!(U8x8V3, U8x4V3);

// ===========================================================================================
// x16 (native) <- x8 (reduced).
// ===========================================================================================

macro_rules! impl_concat_x16_from_x8 {
    ($native:ty, $x8:ty) => {
        #[thermite_macros::inline_always]
        impl ConcatRegister<$x8> for $native {
            fn concat(lo: Storage<$x8>, hi: Storage<$x8>) -> Storage<Self> {
                unsafe { arch::_mm_unpacklo_epi64(lo.0, hi.0) }
            }
            fn split(value: Storage<Self>) -> (Storage<$x8>, Storage<$x8>) {
                (
                    ReducedRegister::new(value),
                    ReducedRegister::new(unsafe { arch::_mm_unpackhi_epi64(value, value) }),
                )
            }
        }
    };
}

impl_concat_x16_from_x8!(super::I8x16V3, I8x8V3);
impl_concat_x16_from_x8!(super::U8x16V3, U8x8V3);

// ===========================================================================================
// 8 <-> 16 widen/narrow casts.
//   widen 8 -> 16 via `cvtepi8_epi16`/`cvtepu8_epi16`.
//   narrow 16 -> 8 via `pshufb` gathering the low byte of each i16 (even byte positions).
//   x4:  I8x4V3 <-> half16::I16x4V3 (both reduced, data in low lanes).
//   x8:  I8x8V3 <-> I16x8V3 (native 128-bit 8-lane, same as v2).
//   x16: I8x16V3 (native 128-bit) <-> I16x16V3 (native 256-bit) via 256-bit cvt + dual pshufb.
// ===========================================================================================

// --- x4 (both reduced) ---
#[thermite_macros::inline_always]
impl CastRegister<I8x4V3> for super::half16::I16x4V3 {
    fn cast_from(value: Storage<I8x4V3>) -> Storage<Self> {
        ReducedRegister::new(unsafe { arch::_mm_cvtepi8_epi16(value.0) })
    }
}
#[thermite_macros::inline_always]
impl CastRegister<U8x4V3> for super::half16::U16x4V3 {
    fn cast_from(value: Storage<U8x4V3>) -> Storage<Self> {
        ReducedRegister::new(unsafe { arch::_mm_cvtepu8_epi16(value.0) })
    }
}
#[thermite_macros::inline_always]
impl CastRegister<super::half16::I16x4V3> for I8x4V3 {
    fn cast_from(value: Storage<super::half16::I16x4V3>) -> Storage<Self> {
        ReducedRegister::new(unsafe { arch::_mm_shuffle_epi8(value.0, arch::_mm_narrow_word_to_byte_maskx_v2()) })
    }
}
#[thermite_macros::inline_always]
impl CastRegister<super::half16::U16x4V3> for U8x4V3 {
    fn cast_from(value: Storage<super::half16::U16x4V3>) -> Storage<Self> {
        ReducedRegister::new(unsafe { arch::_mm_shuffle_epi8(value.0, arch::_mm_narrow_word_to_byte_maskx_v2()) })
    }
}

// --- x8 (i16x8 is native 128-bit) ---
#[thermite_macros::inline_always]
impl CastRegister<I8x8V3> for super::I16x8V3 {
    fn cast_from(value: Storage<I8x8V3>) -> Storage<Self> {
        unsafe { arch::_mm_cvtepi8_epi16(value.0) }
    }
}
#[thermite_macros::inline_always]
impl CastRegister<U8x8V3> for super::U16x8V3 {
    fn cast_from(value: Storage<U8x8V3>) -> Storage<Self> {
        unsafe { arch::_mm_cvtepu8_epi16(value.0) }
    }
}
#[thermite_macros::inline_always]
impl CastRegister<super::I16x8V3> for I8x8V3 {
    fn cast_from(value: Storage<super::I16x8V3>) -> Storage<Self> {
        ReducedRegister::new(unsafe { arch::_mm_shuffle_epi8(value, arch::_mm_narrow_word_to_byte_maskx_v2()) })
    }
}
#[thermite_macros::inline_always]
impl CastRegister<super::U16x8V3> for U8x8V3 {
    fn cast_from(value: Storage<super::U16x8V3>) -> Storage<Self> {
        ReducedRegister::new(unsafe { arch::_mm_shuffle_epi8(value, arch::_mm_narrow_word_to_byte_maskx_v2()) })
    }
}

// --- saturating narrows into 8-bit (low half holds the result; no lane-crossing fixup) ---

// i16x8 -> i8x8 via `vpacksswb` (saturates to [i8::MIN, i8::MAX]).
#[thermite_macros::inline_always]
impl SaturatingCastRegister<super::I16x8V3> for I8x8V3 {
    fn saturating_cast_from(value: Storage<super::I16x8V3>) -> Storage<Self> {
        ReducedRegister::new(unsafe { arch::_mm_packs_epi16(value, value) })
    }
}

// u16x8 -> u8x8: clamp the high end (`vpminuw`) then `vpackuswb` (signed-source pack).
#[thermite_macros::inline_always]
impl SaturatingCastRegister<super::U16x8V3> for U8x8V3 {
    fn saturating_cast_from(value: Storage<super::U16x8V3>) -> Storage<Self> {
        let clamped = unsafe { arch::_mm_min_epu16(value, arch::_mm_set1_epi16(0xFF)) };
        ReducedRegister::new(unsafe { arch::_mm_packus_epi16(clamped, clamped) })
    }
}

// i16x4 -> i8x4 via `vpacksswb` (the 4 valid words sit in the reduced source's low half).
#[thermite_macros::inline_always]
impl SaturatingCastRegister<super::half16::I16x4V3> for I8x4V3 {
    fn saturating_cast_from(value: Storage<super::half16::I16x4V3>) -> Storage<Self> {
        ReducedRegister::new(unsafe { arch::_mm_packs_epi16(value.0, value.0) })
    }
}

// u16x4 -> u8x4: clamp the high end (`vpminuw`) then `vpackuswb`.
#[thermite_macros::inline_always]
impl SaturatingCastRegister<super::half16::U16x4V3> for U8x4V3 {
    fn saturating_cast_from(value: Storage<super::half16::U16x4V3>) -> Storage<Self> {
        let clamped = unsafe { arch::_mm_min_epu16(value.0, arch::_mm_set1_epi16(0xFF)) };
        ReducedRegister::new(unsafe { arch::_mm_packus_epi16(clamped, clamped) })
    }
}

// --- skip-level 32 -> 8: compose the 32 -> 16 and 16 -> 8 saturating packs. Saturation is
// idempotent across nested ranges, so clamping to i16 then i8 equals clamping straight to i8. ---

#[thermite_macros::inline_always]
impl SaturatingCastRegister<super::I32x4V3> for I8x4V3 {
    fn saturating_cast_from(value: Storage<super::I32x4V3>) -> Storage<Self> {
        let words = <super::half16::I16x4V3 as SaturatingCastRegister<super::I32x4V3>>::saturating_cast_from(value);
        <Self as SaturatingCastRegister<super::half16::I16x4V3>>::saturating_cast_from(words)
    }
}

#[thermite_macros::inline_always]
impl SaturatingCastRegister<super::U32x4V3> for U8x4V3 {
    fn saturating_cast_from(value: Storage<super::U32x4V3>) -> Storage<Self> {
        let words = <super::half16::U16x4V3 as SaturatingCastRegister<super::U32x4V3>>::saturating_cast_from(value);
        <Self as SaturatingCastRegister<super::half16::U16x4V3>>::saturating_cast_from(words)
    }
}

#[thermite_macros::inline_always]
impl SaturatingCastRegister<super::I32x8V3> for I8x8V3 {
    fn saturating_cast_from(value: Storage<super::I32x8V3>) -> Storage<Self> {
        let words = <super::I16x8V3 as SaturatingCastRegister<super::I32x8V3>>::saturating_cast_from(value);
        <Self as SaturatingCastRegister<super::I16x8V3>>::saturating_cast_from(words)
    }
}

#[thermite_macros::inline_always]
impl SaturatingCastRegister<super::U32x8V3> for U8x8V3 {
    fn saturating_cast_from(value: Storage<super::U32x8V3>) -> Storage<Self> {
        let words = <super::U16x8V3 as SaturatingCastRegister<super::U32x8V3>>::saturating_cast_from(value);
        <Self as SaturatingCastRegister<super::U16x8V3>>::saturating_cast_from(words)
    }
}

// --- x16 (i8x16 native 128-bit <-> i16x16 native 256-bit) ---
// widen: 16 bytes (__m128i) -> 16x i16 (__m256i) in a single AVX2 sign/zero-extend.
#[thermite_macros::inline_always]
impl CastRegister<super::I8x16V3> for super::I16x16V3 {
    fn cast_from(value: Storage<super::I8x16V3>) -> Storage<Self> {
        unsafe { arch::_mm256_cvtepi8_epi16(value) }
    }
}
#[thermite_macros::inline_always]
impl CastRegister<super::U8x16V3> for super::U16x16V3 {
    fn cast_from(value: Storage<super::U8x16V3>) -> Storage<Self> {
        unsafe { arch::_mm256_cvtepu8_epi16(value) }
    }
}
// narrow: 16x i16 (__m256i) -> 16 bytes (__m128i). See `arch::_mm256_cvtepi16_epi8x_v3`.
#[thermite_macros::inline_always]
impl CastRegister<super::I16x16V3> for super::I8x16V3 {
    fn cast_from(value: Storage<super::I16x16V3>) -> Storage<Self> {
        unsafe { arch::_mm256_cvtepi16_epi8x_v3(value) }
    }
}
#[thermite_macros::inline_always]
impl CastRegister<super::U16x16V3> for super::U8x16V3 {
    fn cast_from(value: Storage<super::U16x16V3>) -> Storage<Self> {
        unsafe { arch::_mm256_cvtepi16_epi8x_v3(value) }
    }
}

// ===========================================================================================
// Widen casts to 32-bit.
//   x4: I8x4V3 <-> I32x4V3 (native 4-lane).
//   x8: I8x8V3 <-> I32x8V3 (native 256-bit, 8-lane).
//   x2: ArrayRegister<i8,2> <-> I32x2V3 (reduced 2-lane).
// ===========================================================================================

// --- x4 widen / narrow ---
#[thermite_macros::inline_always]
impl CastRegister<I8x4V3> for super::I32x4V3 {
    fn cast_from(value: Storage<I8x4V3>) -> Storage<Self> {
        unsafe { arch::_mm_cvtepi8_epi32(value.0) }
    }
}
#[thermite_macros::inline_always]
impl CastRegister<U8x4V3> for super::U32x4V3 {
    fn cast_from(value: Storage<U8x4V3>) -> Storage<Self> {
        unsafe { arch::_mm_cvtepu8_epi32(value.0) }
    }
}
#[thermite_macros::inline_always]
impl CastRegister<super::I32x4V3> for I8x4V3 {
    fn cast_from(value: Storage<super::I32x4V3>) -> Storage<Self> {
        ReducedRegister::new(unsafe {
            arch::_mm_shuffle_epi8(value, arch::_mm_narrow_dword_to_byte_maskx_v2())
        })
    }
}
#[thermite_macros::inline_always]
impl CastRegister<super::U32x4V3> for U8x4V3 {
    fn cast_from(value: Storage<super::U32x4V3>) -> Storage<Self> {
        ReducedRegister::new(unsafe {
            arch::_mm_shuffle_epi8(value, arch::_mm_narrow_dword_to_byte_maskx_v2())
        })
    }
}

// --- x8 widen i8 -> i32 (low 8 bytes of a __m128i -> 8x i32 in a __m256i) ---
#[thermite_macros::inline_always]
impl CastRegister<I8x8V3> for super::I32x8V3 {
    fn cast_from(value: Storage<I8x8V3>) -> Storage<Self> {
        unsafe { arch::_mm256_cvtepi8_epi32(value.0) }
    }
}
#[thermite_macros::inline_always]
impl CastRegister<U8x8V3> for super::U32x8V3 {
    fn cast_from(value: Storage<U8x8V3>) -> Storage<Self> {
        unsafe { arch::_mm256_cvtepu8_epi32(value.0) }
    }
}

// --- x8 narrow i32 -> i8 (low byte of each of 8 i32 lanes -> low 8 bytes of a __m128i) ---
// See `arch::_mm256_cvtepi32_epi8x_v3`.
#[thermite_macros::inline_always]
impl CastRegister<super::I32x8V3> for I8x8V3 {
    fn cast_from(value: Storage<super::I32x8V3>) -> Storage<Self> {
        ReducedRegister::new(unsafe { arch::_mm256_cvtepi32_epi8x_v3(value) })
    }
}
#[thermite_macros::inline_always]
impl CastRegister<super::U32x8V3> for U8x8V3 {
    fn cast_from(value: Storage<super::U32x8V3>) -> Storage<Self> {
        ReducedRegister::new(unsafe { arch::_mm256_cvtepi32_epi8x_v3(value) })
    }
}

// --- x16 widen i8 -> i32 (native 16 bytes -> ArrayRegister<I32x8V3, 2>, two 256-bit widens) ---
#[thermite_macros::inline_always]
impl CastRegister<super::I8x16V3> for ArrayRegister<super::I32x8V3, 2> {
    fn cast_from(value: Storage<super::I8x16V3>) -> Storage<Self> {
        unsafe {
            ArrayRegister([
                arch::_mm256_cvtepi8_epi32(value),                          // low 8 bytes -> 8x i32
                arch::_mm256_cvtepi8_epi32(arch::_mm_srli_si128(value, 8)), // high 8 bytes
            ])
        }
    }
}
#[thermite_macros::inline_always]
impl CastRegister<super::U8x16V3> for ArrayRegister<super::U32x8V3, 2> {
    fn cast_from(value: Storage<super::U8x16V3>) -> Storage<Self> {
        unsafe {
            ArrayRegister([
                arch::_mm256_cvtepu8_epi32(value),
                arch::_mm256_cvtepu8_epi32(arch::_mm_srli_si128(value, 8)),
            ])
        }
    }
}

// --- x16 narrow i32 -> i8 (ArrayRegister<I32x8V3, 2> -> native 16 bytes) ---
// See `arch::_mm_cvt2epi32x8_epi8x_v3`.
#[thermite_macros::inline_always]
impl CastRegister<ArrayRegister<super::I32x8V3, 2>> for super::I8x16V3 {
    fn cast_from(value: Storage<ArrayRegister<super::I32x8V3, 2>>) -> Storage<Self> {
        unsafe { arch::_mm_cvt2epi32x8_epi8x_v3(value.0) }
    }
}
#[thermite_macros::inline_always]
impl CastRegister<ArrayRegister<super::U32x8V3, 2>> for super::U8x16V3 {
    fn cast_from(value: Storage<ArrayRegister<super::U32x8V3, 2>>) -> Storage<Self> {
        unsafe { arch::_mm_cvt2epi32x8_epi8x_v3(value.0) }
    }
}

// --- x2 widen / narrow (ArrayRegister<i8,2> <-> I32x2V3 reduced) ---
#[thermite_macros::inline_always]
impl CastRegister<ArrayRegister<i8, 2>> for super::half::I32x2V3 {
    fn cast_from(value: Storage<ArrayRegister<i8, 2>>) -> Storage<Self> {
        ReducedRegister::new(unsafe { arch::_mm_setr_epi32(value.0[0] as i32, value.0[1] as i32, 0, 0) })
    }
}
#[thermite_macros::inline_always]
impl CastRegister<ArrayRegister<u8, 2>> for super::half::U32x2V3 {
    fn cast_from(value: Storage<ArrayRegister<u8, 2>>) -> Storage<Self> {
        ReducedRegister::new(unsafe { arch::_mm_setr_epi32(value.0[0] as i32, value.0[1] as i32, 0, 0) })
    }
}
#[thermite_macros::inline_always]
impl CastRegister<super::half::I32x2V3> for ArrayRegister<i8, 2> {
    fn cast_from(value: Storage<super::half::I32x2V3>) -> Storage<Self> {
        let mut arr = [0i32; 4];
        unsafe { arch::_mm_storeu_si128(arr.as_mut_ptr() as *mut _, value.0) };
        ArrayRegister([arr[0] as i8, arr[1] as i8])
    }
}
#[thermite_macros::inline_always]
impl CastRegister<super::half::U32x2V3> for ArrayRegister<u8, 2> {
    fn cast_from(value: Storage<super::half::U32x2V3>) -> Storage<Self> {
        let mut arr = [0u32; 4];
        unsafe { arch::_mm_storeu_si128(arr.as_mut_ptr() as *mut _, value.0) };
        ArrayRegister([arr[0] as u8, arr[1] as u8])
    }
}

// ===========================================================================================
// Widen/narrow casts to 64-bit.
//   widen 8 -> 64 via `_mm256_cvtepi8_epi64` (low 4 bytes -> 4x i64 in a __m256i).
//   narrow 64 -> 8 by extracting the two 128-bit halves of each I64x4V3 and pshufb-gathering
//   byte 0 of each i64 lane.
//   x2:  ArrayRegister<i8,2> <-> I64x2V3 (native 128-bit, 2-lane).
//   x4:  I8x4V3 <-> I64x4V3 (native 256-bit, 4-lane).
//   x8:  I8x8V3 <-> ArrayRegister<I64x4V3, 2> (the v3 i64x8).
//   x16: I8x16V3 <-> ArrayRegister<I64x4V3, 4> (the v3 i64x16).
// ===========================================================================================

// --- x2 widen / narrow (ArrayRegister<i8,2> <-> I64x2V3 native 128-bit) ---
#[thermite_macros::inline_always]
impl CastRegister<ArrayRegister<i8, 2>> for super::I64x2V3 {
    fn cast_from(value: Storage<ArrayRegister<i8, 2>>) -> Storage<Self> {
        unsafe { arch::_mm_set_epi64x(value.0[1] as i64, value.0[0] as i64) }
    }
}
#[thermite_macros::inline_always]
impl CastRegister<ArrayRegister<u8, 2>> for super::U64x2V3 {
    fn cast_from(value: Storage<ArrayRegister<u8, 2>>) -> Storage<Self> {
        unsafe { arch::_mm_set_epi64x(value.0[1] as i64, value.0[0] as i64) }
    }
}
#[thermite_macros::inline_always]
impl CastRegister<super::I64x2V3> for ArrayRegister<i8, 2> {
    fn cast_from(value: Storage<super::I64x2V3>) -> Storage<Self> {
        let mut arr = [0i64; 2];
        unsafe { arch::_mm_storeu_si128(arr.as_mut_ptr() as *mut _, value) };
        ArrayRegister([arr[0] as i8, arr[1] as i8])
    }
}
#[thermite_macros::inline_always]
impl CastRegister<super::U64x2V3> for ArrayRegister<u8, 2> {
    fn cast_from(value: Storage<super::U64x2V3>) -> Storage<Self> {
        let mut arr = [0u64; 2];
        unsafe { arch::_mm_storeu_si128(arr.as_mut_ptr() as *mut _, value) };
        ArrayRegister([arr[0] as u8, arr[1] as u8])
    }
}

// --- x4 widen i8 -> i64 (low 4 bytes -> 4x i64 in a __m256i) ---
#[thermite_macros::inline_always]
impl CastRegister<I8x4V3> for super::I64x4V3 {
    fn cast_from(value: Storage<I8x4V3>) -> Storage<Self> {
        unsafe { arch::_mm256_cvtepi8_epi64(value.0) }
    }
}
#[thermite_macros::inline_always]
impl CastRegister<U8x4V3> for super::U64x4V3 {
    fn cast_from(value: Storage<U8x4V3>) -> Storage<Self> {
        unsafe { arch::_mm256_cvtepu8_epi64(value.0) }
    }
}
#[thermite_macros::inline_always]
impl CastRegister<super::I64x4V3> for I8x4V3 {
    fn cast_from(value: Storage<super::I64x4V3>) -> Storage<Self> {
        ReducedRegister::new(unsafe { arch::_mm256_cvtepi64_epi8x_v3(value) })
    }
}
#[thermite_macros::inline_always]
impl CastRegister<super::U64x4V3> for U8x4V3 {
    fn cast_from(value: Storage<super::U64x4V3>) -> Storage<Self> {
        ReducedRegister::new(unsafe { arch::_mm256_cvtepi64_epi8x_v3(value) })
    }
}

// --- x8 widen i8 -> i64 (low 8 bytes -> two 4x i64 __m256i lanes) ---
#[thermite_macros::inline_always]
impl CastRegister<I8x8V3> for ArrayRegister<super::I64x4V3, 2> {
    fn cast_from(value: Storage<I8x8V3>) -> Storage<Self> {
        unsafe {
            ArrayRegister([
                arch::_mm256_cvtepi8_epi64(value.0),                          // low 4 bytes -> 4x i64
                arch::_mm256_cvtepi8_epi64(arch::_mm_srli_si128(value.0, 4)), // next 4 bytes
            ])
        }
    }
}
#[thermite_macros::inline_always]
impl CastRegister<U8x8V3> for ArrayRegister<super::U64x4V3, 2> {
    fn cast_from(value: Storage<U8x8V3>) -> Storage<Self> {
        unsafe {
            ArrayRegister([
                arch::_mm256_cvtepu8_epi64(value.0),
                arch::_mm256_cvtepu8_epi64(arch::_mm_srli_si128(value.0, 4)),
            ])
        }
    }
}

// --- x8 narrow i64 -> i8 (two 4x i64 lanes -> low 8 bytes) ---
// See `arch::_mm_cvt2epi64x4_epi8x_v3`.
#[thermite_macros::inline_always]
impl CastRegister<ArrayRegister<super::I64x4V3, 2>> for I8x8V3 {
    fn cast_from(value: Storage<ArrayRegister<super::I64x4V3, 2>>) -> Storage<Self> {
        ReducedRegister::new(unsafe { arch::_mm_cvt2epi64x4_epi8x_v3(value.0) })
    }
}
#[thermite_macros::inline_always]
impl CastRegister<ArrayRegister<super::U64x4V3, 2>> for U8x8V3 {
    fn cast_from(value: Storage<ArrayRegister<super::U64x4V3, 2>>) -> Storage<Self> {
        ReducedRegister::new(unsafe { arch::_mm_cvt2epi64x4_epi8x_v3(value.0) })
    }
}

// --- x16 widen i8 -> i64 (native 16 bytes -> ArrayRegister<I64x4V3, 4>, four 256-bit widens) ---
#[thermite_macros::inline_always]
impl CastRegister<super::I8x16V3> for ArrayRegister<super::I64x4V3, 4> {
    fn cast_from(value: Storage<super::I8x16V3>) -> Storage<Self> {
        unsafe {
            ArrayRegister([
                arch::_mm256_cvtepi8_epi64(value),
                arch::_mm256_cvtepi8_epi64(arch::_mm_srli_si128(value, 4)),
                arch::_mm256_cvtepi8_epi64(arch::_mm_srli_si128(value, 8)),
                arch::_mm256_cvtepi8_epi64(arch::_mm_srli_si128(value, 12)),
            ])
        }
    }
}
#[thermite_macros::inline_always]
impl CastRegister<super::U8x16V3> for ArrayRegister<super::U64x4V3, 4> {
    fn cast_from(value: Storage<super::U8x16V3>) -> Storage<Self> {
        unsafe {
            ArrayRegister([
                arch::_mm256_cvtepu8_epi64(value),
                arch::_mm256_cvtepu8_epi64(arch::_mm_srli_si128(value, 4)),
                arch::_mm256_cvtepu8_epi64(arch::_mm_srli_si128(value, 8)),
                arch::_mm256_cvtepu8_epi64(arch::_mm_srli_si128(value, 12)),
            ])
        }
    }
}

// --- x16 narrow i64 -> i8 (ArrayRegister<I64x4V3, 4> -> native 16 bytes) ---
#[thermite_macros::inline_always]
impl CastRegister<ArrayRegister<super::I64x4V3, 4>> for super::I8x16V3 {
    fn cast_from(value: Storage<ArrayRegister<super::I64x4V3, 4>>) -> Storage<Self> {
        let v = value.0;
        unsafe {
            let lo = arch::_mm_cvt2epi64x4_epi8x_v3([v[0], v[1]]); // 8 bytes in low 64 bits
            let hi = arch::_mm_cvt2epi64x4_epi8x_v3([v[2], v[3]]);
            arch::_mm_unpacklo_epi64(lo, hi)
        }
    }
}
#[thermite_macros::inline_always]
impl CastRegister<ArrayRegister<super::U64x4V3, 4>> for super::U8x16V3 {
    fn cast_from(value: Storage<ArrayRegister<super::U64x4V3, 4>>) -> Storage<Self> {
        let v = value.0;
        unsafe {
            let lo = arch::_mm_cvt2epi64x4_epi8x_v3([v[0], v[1]]);
            let hi = arch::_mm_cvt2epi64x4_epi8x_v3([v[2], v[3]]);
            arch::_mm_unpacklo_epi64(lo, hi)
        }
    }
}

// ===========================================================================================
// 8 <-> f32/f64 direct casts.
//   widen i8/u8 -> f32/f64 = (int widen to i32, same lanes) then (i32 -> float convert).
//     unsigned widens zero-extend (`cvtepu8_epi32`); the widened value fits in positive i32,
//     so the signed `cvtepi32_ps`/`cvtepi32_pd` convert is exact.
//   narrow float -> i8/u8 = (float -> i32 truncating convert, like `as`) then (i32 -> byte
//     narrow). The low byte is identical signed vs unsigned, so the signed narrow is reused.
//   f32 native widths: f32x8 (256-bit), f32x4 (128-bit). f64 native widths: f64x4 (256-bit),
//     f64x2 (128-bit). The lane fan-out for f64 differs because each native f64 register holds
//     half as many lanes as the matching i32 register.
// ===========================================================================================

// --- x2 (ArrayRegister<i8,2> <-> F32x2V3 / F64x2V3) ---
// widen: build an i32x4 [v0, v1, 0, 0] then the i32 -> float convert. narrow: float -> i32
// truncating convert, then store and take the low 2 lanes `as i8`/`as u8`.
#[thermite_macros::inline_always]
impl CastRegister<ArrayRegister<i8, 2>> for super::half::F32x2V3 {
    fn cast_from(value: Storage<ArrayRegister<i8, 2>>) -> Storage<Self> {
        ReducedRegister::new(unsafe {
            let widened = arch::_mm_setr_epi32(value.0[0] as i32, value.0[1] as i32, 0, 0);
            arch::_mm_cvtepi32_ps(widened)
        })
    }
}
#[thermite_macros::inline_always]
impl CastRegister<ArrayRegister<u8, 2>> for super::half::F32x2V3 {
    fn cast_from(value: Storage<ArrayRegister<u8, 2>>) -> Storage<Self> {
        ReducedRegister::new(unsafe {
            let widened = arch::_mm_setr_epi32(value.0[0] as i32, value.0[1] as i32, 0, 0);
            arch::_mm_cvtepi32_ps(widened)
        })
    }
}
#[thermite_macros::inline_always]
impl CastRegister<super::half::F32x2V3> for ArrayRegister<i8, 2> {
    fn cast_from(value: Storage<super::half::F32x2V3>) -> Storage<Self> {
        let mut arr = [0i32; 4];
        unsafe {
            let ints = arch::_mm_cvttps_epi32(value.0);
            arch::_mm_storeu_si128(arr.as_mut_ptr() as *mut _, ints);
        }
        ArrayRegister([arr[0] as i8, arr[1] as i8])
    }
}
#[thermite_macros::inline_always]
impl CastRegister<super::half::F32x2V3> for ArrayRegister<u8, 2> {
    fn cast_from(value: Storage<super::half::F32x2V3>) -> Storage<Self> {
        let mut arr = [0i32; 4];
        unsafe {
            let ints = arch::_mm_cvttps_epi32(value.0);
            arch::_mm_storeu_si128(arr.as_mut_ptr() as *mut _, ints);
        }
        ArrayRegister([arr[0] as u8, arr[1] as u8])
    }
}
#[thermite_macros::inline_always]
impl CastRegister<ArrayRegister<i8, 2>> for super::F64x2V3 {
    fn cast_from(value: Storage<ArrayRegister<i8, 2>>) -> Storage<Self> {
        unsafe {
            let widened = arch::_mm_setr_epi32(value.0[0] as i32, value.0[1] as i32, 0, 0);
            arch::_mm_cvtepi32_pd(widened)
        }
    }
}
#[thermite_macros::inline_always]
impl CastRegister<ArrayRegister<u8, 2>> for super::F64x2V3 {
    fn cast_from(value: Storage<ArrayRegister<u8, 2>>) -> Storage<Self> {
        unsafe {
            let widened = arch::_mm_setr_epi32(value.0[0] as i32, value.0[1] as i32, 0, 0);
            arch::_mm_cvtepi32_pd(widened)
        }
    }
}
#[thermite_macros::inline_always]
impl CastRegister<super::F64x2V3> for ArrayRegister<i8, 2> {
    fn cast_from(value: Storage<super::F64x2V3>) -> Storage<Self> {
        let mut arr = [0i32; 4];
        unsafe {
            let ints = arch::_mm_cvttpd_epi32(value); // f64x2 -> i32 in low 2 lanes of __m128i
            arch::_mm_storeu_si128(arr.as_mut_ptr() as *mut _, ints);
        }
        ArrayRegister([arr[0] as i8, arr[1] as i8])
    }
}
#[thermite_macros::inline_always]
impl CastRegister<super::F64x2V3> for ArrayRegister<u8, 2> {
    fn cast_from(value: Storage<super::F64x2V3>) -> Storage<Self> {
        let mut arr = [0i32; 4];
        unsafe {
            let ints = arch::_mm_cvttpd_epi32(value);
            arch::_mm_storeu_si128(arr.as_mut_ptr() as *mut _, ints);
        }
        ArrayRegister([arr[0] as u8, arr[1] as u8])
    }
}

// --- x4 (I8x4V3 <-> F32x4V3 native 128-bit / F64x4V3 native 256-bit) ---
// f32x4: widen low-4 i8 to i32x4 then `_mm_cvtepi32_ps`. f64x4: widen low-4 i8 to i32x4 then
// `_mm256_cvtepi32_pd` (i32x4 -> f64x4). narrow: float -> i32 truncating convert then pshufb
// the low byte of each dword into the low 4 bytes (`arch::_mm_narrow_dword_to_byte_maskx_v2()`).
#[thermite_macros::inline_always]
impl CastRegister<I8x4V3> for super::F32x4V3 {
    fn cast_from(value: Storage<I8x4V3>) -> Storage<Self> {
        unsafe {
            let widened = arch::_mm_cvtepi8_epi32(value.0);
            arch::_mm_cvtepi32_ps(widened)
        }
    }
}
#[thermite_macros::inline_always]
impl CastRegister<U8x4V3> for super::F32x4V3 {
    fn cast_from(value: Storage<U8x4V3>) -> Storage<Self> {
        unsafe {
            let widened = arch::_mm_cvtepu8_epi32(value.0);
            arch::_mm_cvtepi32_ps(widened)
        }
    }
}
#[thermite_macros::inline_always]
impl CastRegister<super::F32x4V3> for I8x4V3 {
    fn cast_from(value: Storage<super::F32x4V3>) -> Storage<Self> {
        ReducedRegister::new(unsafe {
            let ints = arch::_mm_cvttps_epi32(value);
            arch::_mm_shuffle_epi8(ints, arch::_mm_narrow_dword_to_byte_maskx_v2())
        })
    }
}
#[thermite_macros::inline_always]
impl CastRegister<super::F32x4V3> for U8x4V3 {
    fn cast_from(value: Storage<super::F32x4V3>) -> Storage<Self> {
        ReducedRegister::new(unsafe {
            let ints = arch::_mm_cvttps_epi32(value);
            arch::_mm_shuffle_epi8(ints, arch::_mm_narrow_dword_to_byte_maskx_v2())
        })
    }
}
#[thermite_macros::inline_always]
impl CastRegister<I8x4V3> for super::F64x4V3 {
    fn cast_from(value: Storage<I8x4V3>) -> Storage<Self> {
        unsafe {
            let widened = arch::_mm_cvtepi8_epi32(value.0); // low 4 bytes -> 4x i32 (128-bit)
            arch::_mm256_cvtepi32_pd(widened) // 4x i32 -> 4x f64 (256-bit)
        }
    }
}
#[thermite_macros::inline_always]
impl CastRegister<U8x4V3> for super::F64x4V3 {
    fn cast_from(value: Storage<U8x4V3>) -> Storage<Self> {
        unsafe {
            let widened = arch::_mm_cvtepu8_epi32(value.0);
            arch::_mm256_cvtepi32_pd(widened)
        }
    }
}
#[thermite_macros::inline_always]
impl CastRegister<super::F64x4V3> for I8x4V3 {
    fn cast_from(value: Storage<super::F64x4V3>) -> Storage<Self> {
        ReducedRegister::new(unsafe {
            let ints = arch::_mm256_cvttpd_epi32(value); // f64x4 -> 4x i32 in __m128i
            arch::_mm_shuffle_epi8(ints, arch::_mm_narrow_dword_to_byte_maskx_v2())
        })
    }
}
#[thermite_macros::inline_always]
impl CastRegister<super::F64x4V3> for U8x4V3 {
    fn cast_from(value: Storage<super::F64x4V3>) -> Storage<Self> {
        ReducedRegister::new(unsafe {
            let ints = arch::_mm256_cvttpd_epi32(value);
            arch::_mm_shuffle_epi8(ints, arch::_mm_narrow_dword_to_byte_maskx_v2())
        })
    }
}

// --- x8 (I8x8V3 <-> F32x8V3 native 256-bit / ArrayRegister<F64x4V3, 2>) ---
// f32x8: widen 8 bytes to one i32x8 (`_mm256_cvtepi8_epi32`) then `_mm256_cvtepi32_ps`. f64x8:
// widen low-4 and next-4 bytes to two i32x4 then `_mm256_cvtepi32_pd` each. narrow reverses,
// reusing `arch::_mm256_cvtepi32_epi8x_v3` for f32x8 and the pshufb dword->byte mask for f64.
#[thermite_macros::inline_always]
impl CastRegister<I8x8V3> for super::F32x8V3 {
    fn cast_from(value: Storage<I8x8V3>) -> Storage<Self> {
        unsafe {
            let widened = arch::_mm256_cvtepi8_epi32(value.0); // low 8 bytes -> 8x i32
            arch::_mm256_cvtepi32_ps(widened)
        }
    }
}
#[thermite_macros::inline_always]
impl CastRegister<U8x8V3> for super::F32x8V3 {
    fn cast_from(value: Storage<U8x8V3>) -> Storage<Self> {
        unsafe {
            let widened = arch::_mm256_cvtepu8_epi32(value.0);
            arch::_mm256_cvtepi32_ps(widened)
        }
    }
}
#[thermite_macros::inline_always]
impl CastRegister<super::F32x8V3> for I8x8V3 {
    fn cast_from(value: Storage<super::F32x8V3>) -> Storage<Self> {
        ReducedRegister::new(unsafe { arch::_mm256_cvtepi32_epi8x_v3(arch::_mm256_cvttps_epi32(value)) })
    }
}
#[thermite_macros::inline_always]
impl CastRegister<super::F32x8V3> for U8x8V3 {
    fn cast_from(value: Storage<super::F32x8V3>) -> Storage<Self> {
        ReducedRegister::new(unsafe { arch::_mm256_cvtepi32_epi8x_v3(arch::_mm256_cvttps_epi32(value)) })
    }
}
#[thermite_macros::inline_always]
impl CastRegister<I8x8V3> for ArrayRegister<super::F64x4V3, 2> {
    fn cast_from(value: Storage<I8x8V3>) -> Storage<Self> {
        unsafe {
            ArrayRegister([
                arch::_mm256_cvtepi32_pd(arch::_mm_cvtepi8_epi32(value.0)), // low 4 bytes -> f64x4
                arch::_mm256_cvtepi32_pd(arch::_mm_cvtepi8_epi32(arch::_mm_srli_si128(value.0, 4))), // next 4
            ])
        }
    }
}
#[thermite_macros::inline_always]
impl CastRegister<U8x8V3> for ArrayRegister<super::F64x4V3, 2> {
    fn cast_from(value: Storage<U8x8V3>) -> Storage<Self> {
        unsafe {
            ArrayRegister([
                arch::_mm256_cvtepi32_pd(arch::_mm_cvtepu8_epi32(value.0)),
                arch::_mm256_cvtepi32_pd(arch::_mm_cvtepu8_epi32(arch::_mm_srli_si128(value.0, 4))),
            ])
        }
    }
}
// narrow f64x8 (two F64x4V3) -> low 8 bytes: see `arch::_mm_cvt2pd4_epi8x_v3`.
#[thermite_macros::inline_always]
impl CastRegister<ArrayRegister<super::F64x4V3, 2>> for I8x8V3 {
    fn cast_from(value: Storage<ArrayRegister<super::F64x4V3, 2>>) -> Storage<Self> {
        ReducedRegister::new(unsafe { arch::_mm_cvt2pd4_epi8x_v3(value.0) })
    }
}
#[thermite_macros::inline_always]
impl CastRegister<ArrayRegister<super::F64x4V3, 2>> for U8x8V3 {
    fn cast_from(value: Storage<ArrayRegister<super::F64x4V3, 2>>) -> Storage<Self> {
        ReducedRegister::new(unsafe { arch::_mm_cvt2pd4_epi8x_v3(value.0) })
    }
}

// --- x16 (I8x16V3 native 128-bit <-> ArrayRegister<F32x8V3, 2> / ArrayRegister<F64x4V3, 4>) ---
// f32x16: widen 16 bytes to two i32x8 then `_mm256_cvtepi32_ps` each. f64x16: widen 16 bytes to
// four i32x4 then `_mm256_cvtepi32_pd` each. narrow reverses, reusing the existing dword->byte
// helpers.
#[thermite_macros::inline_always]
impl CastRegister<super::I8x16V3> for ArrayRegister<super::F32x8V3, 2> {
    fn cast_from(value: Storage<super::I8x16V3>) -> Storage<Self> {
        unsafe {
            ArrayRegister([
                arch::_mm256_cvtepi32_ps(arch::_mm256_cvtepi8_epi32(value)), // low 8 bytes
                arch::_mm256_cvtepi32_ps(arch::_mm256_cvtepi8_epi32(arch::_mm_srli_si128(value, 8))), // high 8
            ])
        }
    }
}
#[thermite_macros::inline_always]
impl CastRegister<super::U8x16V3> for ArrayRegister<super::F32x8V3, 2> {
    fn cast_from(value: Storage<super::U8x16V3>) -> Storage<Self> {
        unsafe {
            ArrayRegister([
                arch::_mm256_cvtepi32_ps(arch::_mm256_cvtepu8_epi32(value)),
                arch::_mm256_cvtepi32_ps(arch::_mm256_cvtepu8_epi32(arch::_mm_srli_si128(value, 8))),
            ])
        }
    }
}
#[thermite_macros::inline_always]
impl CastRegister<ArrayRegister<super::F32x8V3, 2>> for super::I8x16V3 {
    fn cast_from(value: Storage<ArrayRegister<super::F32x8V3, 2>>) -> Storage<Self> {
        let v = value.0;
        unsafe { arch::_mm_cvt2epi32x8_epi8x_v3([arch::_mm256_cvttps_epi32(v[0]), arch::_mm256_cvttps_epi32(v[1])]) }
    }
}
#[thermite_macros::inline_always]
impl CastRegister<ArrayRegister<super::F32x8V3, 2>> for super::U8x16V3 {
    fn cast_from(value: Storage<ArrayRegister<super::F32x8V3, 2>>) -> Storage<Self> {
        let v = value.0;
        unsafe { arch::_mm_cvt2epi32x8_epi8x_v3([arch::_mm256_cvttps_epi32(v[0]), arch::_mm256_cvttps_epi32(v[1])]) }
    }
}
#[thermite_macros::inline_always]
impl CastRegister<super::I8x16V3> for ArrayRegister<super::F64x4V3, 4> {
    fn cast_from(value: Storage<super::I8x16V3>) -> Storage<Self> {
        unsafe {
            ArrayRegister([
                arch::_mm256_cvtepi32_pd(arch::_mm_cvtepi8_epi32(value)),
                arch::_mm256_cvtepi32_pd(arch::_mm_cvtepi8_epi32(arch::_mm_srli_si128(value, 4))),
                arch::_mm256_cvtepi32_pd(arch::_mm_cvtepi8_epi32(arch::_mm_srli_si128(value, 8))),
                arch::_mm256_cvtepi32_pd(arch::_mm_cvtepi8_epi32(arch::_mm_srli_si128(value, 12))),
            ])
        }
    }
}
#[thermite_macros::inline_always]
impl CastRegister<super::U8x16V3> for ArrayRegister<super::F64x4V3, 4> {
    fn cast_from(value: Storage<super::U8x16V3>) -> Storage<Self> {
        unsafe {
            ArrayRegister([
                arch::_mm256_cvtepi32_pd(arch::_mm_cvtepu8_epi32(value)),
                arch::_mm256_cvtepi32_pd(arch::_mm_cvtepu8_epi32(arch::_mm_srli_si128(value, 4))),
                arch::_mm256_cvtepi32_pd(arch::_mm_cvtepu8_epi32(arch::_mm_srli_si128(value, 8))),
                arch::_mm256_cvtepi32_pd(arch::_mm_cvtepu8_epi32(arch::_mm_srli_si128(value, 12))),
            ])
        }
    }
}
#[thermite_macros::inline_always]
impl CastRegister<ArrayRegister<super::F64x4V3, 4>> for super::I8x16V3 {
    fn cast_from(value: Storage<ArrayRegister<super::F64x4V3, 4>>) -> Storage<Self> {
        let v = value.0;
        unsafe {
            let lo = arch::_mm_cvt2pd4_epi8x_v3([v[0], v[1]]); // 8 bytes in low 64 bits
            let hi = arch::_mm_cvt2pd4_epi8x_v3([v[2], v[3]]);
            arch::_mm_unpacklo_epi64(lo, hi)
        }
    }
}
#[thermite_macros::inline_always]
impl CastRegister<ArrayRegister<super::F64x4V3, 4>> for super::U8x16V3 {
    fn cast_from(value: Storage<ArrayRegister<super::F64x4V3, 4>>) -> Storage<Self> {
        let v = value.0;
        unsafe {
            let lo = arch::_mm_cvt2pd4_epi8x_v3([v[0], v[1]]);
            let hi = arch::_mm_cvt2pd4_epi8x_v3([v[2], v[3]]);
            arch::_mm_unpacklo_epi64(lo, hi)
        }
    }
}

// ===========================================================================================
// Mask-side concat (x4 <- bool2). x8<-x4 mask concat is covered by the same-type Concat above.
// ===========================================================================================

#[inline(always)]
fn bool_to_i8_mask(b: bool) -> i8 {
    if b { !0 } else { 0 }
}

macro_rules! impl_mask_concat_x4_from_bool2 {
    ($red:ty) => {
        #[thermite_macros::inline_always]
        impl ConcatRegister<ArrayRegister<bool, 2>> for $red {
            fn concat(lo: Storage<ArrayRegister<bool, 2>>, hi: Storage<ArrayRegister<bool, 2>>) -> Storage<Self> {
                ReducedRegister::new(unsafe {
                    arch::_mm_setr_epi8(
                        bool_to_i8_mask(lo.0[0]), bool_to_i8_mask(lo.0[1]),
                        bool_to_i8_mask(hi.0[0]), bool_to_i8_mask(hi.0[1]),
                        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                    )
                })
            }
            fn split(value: Storage<Self>) -> (Storage<ArrayRegister<bool, 2>>, Storage<ArrayRegister<bool, 2>>) {
                let a = store_bytes(value.0);
                (ArrayRegister([a[0] != 0, a[1] != 0]), ArrayRegister([a[2] != 0, a[3] != 0]))
            }
        }

        #[thermite_macros::inline_always]
        impl ExtendRegister<ArrayRegister<bool, 2>> for $red {
            fn extend(value: Storage<ArrayRegister<bool, 2>>) -> Storage<Self> {
                ReducedRegister::new(unsafe {
                    arch::_mm_setr_epi8(
                        bool_to_i8_mask(value.0[0]), bool_to_i8_mask(value.0[1]),
                        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                    )
                })
            }
            fn narrow(value: Storage<Self>) -> Storage<ArrayRegister<bool, 2>> {
                let a = store_bytes(value.0);
                ArrayRegister([a[0] != 0, a[1] != 0])
            }
        }
    };
}

impl_mask_concat_x4_from_bool2!(I8x4V3);
impl_mask_concat_x4_from_bool2!(U8x4V3);

// ===========================================================================================
// Cross-type gather/scatter index markers (scalar fallback).
// ===========================================================================================

macro_rules! impl_indexable8 {
    ($idx:ty => $($ty:ty),* $(,)?) => {$( impl IndexableRegister<$idx> for $ty {} )*};
}

impl_indexable8!(<super::super::X86V3 as crate::simd::Simd>::u32x4 => I8x4V3, U8x4V3);
impl_indexable8!(<super::super::X86V3 as crate::simd::Simd>::u64x4 => I8x4V3, U8x4V3);
impl_indexable8!(<super::super::X86V3 as crate::simd::Simd>::u32x8 => I8x8V3, U8x8V3);
impl_indexable8!(<super::super::X86V3 as crate::simd::Simd>::u64x8 => I8x8V3, U8x8V3);

// i64x8 -> i8x8 / u64x8 -> u8x8: clamp down to 16-bit (no 64-bit pack), then `vpack*` to bytes.
#[thermite_macros::inline_always]
impl SaturatingCastRegister<ArrayRegister<super::I64x4V3, 2>> for I8x8V3 {
    fn saturating_cast_from(value: Storage<ArrayRegister<super::I64x4V3, 2>>) -> Storage<Self> {
        let words = <super::I16x8V3 as SaturatingCastRegister<ArrayRegister<super::I64x4V3, 2>>>::saturating_cast_from(value);
        <Self as SaturatingCastRegister<super::I16x8V3>>::saturating_cast_from(words)
    }
}
#[thermite_macros::inline_always]
impl SaturatingCastRegister<ArrayRegister<super::U64x4V3, 2>> for U8x8V3 {
    fn saturating_cast_from(value: Storage<ArrayRegister<super::U64x4V3, 2>>) -> Storage<Self> {
        let words = <super::U16x8V3 as SaturatingCastRegister<ArrayRegister<super::U64x4V3, 2>>>::saturating_cast_from(value);
        <Self as SaturatingCastRegister<super::U16x8V3>>::saturating_cast_from(words)
    }
}
