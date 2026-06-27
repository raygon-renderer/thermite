//! Reduced (sub-native-width) 8-bit registers for x86-v2 and the cast/concat glue bridging the
//! scalar/array 8-bit halves, the reduced 4-/8-lane registers, the native 128-bit
//! `I8x16V2`/`U8x16V2`, and the 32-bit registers they widen into. Mirrors `half16.rs`, one
//! element size down (`_epi8` intrinsics, byte-width shuffles).

use generic_array::typenum::{U8, U12};

use super::arch;

use crate::register::{
    CastRegister, ConcatRegister, ExtendRegister, IndexableRegister, NumericRegister, Register, SaturatingCastRegister,
    Storage, array::ArrayRegister, reduced::ReducedRegister,
};

// Clamp + truncating narrow for the pack-less `* -> i8` pairs: `i64 -> i8` and the 2-lane
// `i32 -> i8` combos with a scalar `ArrayRegister` destination.
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
    (ArrayRegister<super::I64x2V2, 2>, i64, I8x4V2, i8),
    (ArrayRegister<super::U64x2V2, 2>, u64, U8x4V2, u8),
    (ArrayRegister<super::I64x2V2, 4>, i64, I8x8V2, i8),
    (ArrayRegister<super::U64x2V2, 4>, u64, U8x8V2, u8),
    (super::half::I32x2V2, i32, ArrayRegister<i8, 2>, i8),
    (super::half::U32x2V2, u32, ArrayRegister<u8, 2>, u8),
    (super::I64x2V2, i64, ArrayRegister<i8, 2>, i8),
    (super::U64x2V2, u64, ArrayRegister<u8, 2>, u8),
}

// `ReducedRegister<R, N>` removes `N` lanes (lane count = R::Lanes - N), so over the native
// 16-lane register the 4-lane form removes 12 and the 8-lane form removes 8.
/// 4-lane signed 8-bit register, backed by the low 4 lanes of a 128-bit `I8x16V2`.
pub type I8x4V2 = ReducedRegister<super::I8x16V2, U12>;
/// 4-lane unsigned 8-bit register.
pub type U8x4V2 = ReducedRegister<super::U8x16V2, U12>;
/// 8-lane signed 8-bit register, backed by the low 8 lanes of a 128-bit `I8x16V2`.
pub type I8x8V2 = ReducedRegister<super::I8x16V2, U8>;
/// 8-lane unsigned 8-bit register.
pub type U8x8V2 = ReducedRegister<super::U8x16V2, U8>;

// Helper: read the low `n` bytes of a __m128i into a stack array.
#[inline(always)]
fn store_bytes(v: arch::__m128i) -> [i8; 16] {
    let mut arr = [0i8; 16];
    unsafe { arch::_mm_storeu_si128(arr.as_mut_ptr() as *mut _, v) };
    arr
}

// ===========================================================================================
// x4 <- x2 : build a 4-lane reduced register from two 2-lane (scalar-array) halves.
// ===========================================================================================

macro_rules! impl_concat_x4_from_x2 {
    ($red:ty, $elem:ty) => {
        #[thermite_macros::inline_always]
        impl ConcatRegister<ArrayRegister<$elem, 2>> for $red {
            fn concat(
                lo: Storage<ArrayRegister<$elem, 2>>,
                hi: Storage<ArrayRegister<$elem, 2>>,
            ) -> Storage<Self> {
                ReducedRegister::new(unsafe {
                    arch::_mm_setr_epi8(
                        lo.0[0] as i8, lo.0[1] as i8, hi.0[0] as i8, hi.0[1] as i8,
                        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                    )
                })
            }

            fn split(value: Storage<Self>) -> (Storage<ArrayRegister<$elem, 2>>, Storage<ArrayRegister<$elem, 2>>) {
                let a = store_bytes(value.0);
                (
                    ArrayRegister([a[0] as $elem, a[1] as $elem]),
                    ArrayRegister([a[2] as $elem, a[3] as $elem]),
                )
            }
        }

        #[thermite_macros::inline_always]
        impl ExtendRegister<ArrayRegister<$elem, 2>> for $red {
            fn extend(value: Storage<ArrayRegister<$elem, 2>>) -> Storage<Self> {
                ReducedRegister::new(unsafe {
                    arch::_mm_setr_epi8(
                        value.0[0] as i8, value.0[1] as i8, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                    )
                })
            }

            fn narrow(value: Storage<Self>) -> Storage<ArrayRegister<$elem, 2>> {
                let a = store_bytes(value.0);
                ArrayRegister([a[0] as $elem, a[1] as $elem])
            }
        }
    };
}

impl_concat_x4_from_x2!(I8x4V2, i8);
impl_concat_x4_from_x2!(U8x4V2, u8);

// ===========================================================================================
// x8 <- x4 : build an 8-lane reduced register from two 4-lane reduced halves.
// ===========================================================================================

macro_rules! impl_concat_x8_from_x4 {
    ($x8:ty, $x4:ty, $elem:ty) => {
        // Both x4 and x8 are ReducedRegister over the same native 16-lane register but with
        // different reduced widths, so the ReducedRegister blanket does not relate them - the
        // Extend (zero-extend low 4 lanes into 8) and Concat are written explicitly.
        #[thermite_macros::inline_always]
        impl ExtendRegister<$x4> for $x8 {
            fn extend(value: Storage<$x4>) -> Storage<Self> {
                // x4 keeps its 4 lanes in the low 4 bytes; the rest are ignored by the 8-lane view.
                ReducedRegister::new(value.0)
            }

            fn narrow(value: Storage<Self>) -> Storage<$x4> {
                ReducedRegister::new(value.0)
            }
        }

        #[thermite_macros::inline_always]
        impl ConcatRegister<$x4> for $x8 {
            fn concat(lo: Storage<$x4>, hi: Storage<$x4>) -> Storage<Self> {
                // Each x4 holds its data in the low 4 bytes; interleave the low 32 bits of each.
                ReducedRegister::new(unsafe {
                    arch::_mm_unpacklo_epi32(lo.0, hi.0)
                })
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

impl_concat_x8_from_x4!(I8x8V2, I8x4V2, i8);
impl_concat_x8_from_x4!(U8x8V2, U8x4V2, u8);

// ===========================================================================================
// x16 (native) <- x8 (reduced) : two 8-lane halves into the native 128-bit register.
// ===========================================================================================

macro_rules! impl_concat_x16_from_x8 {
    ($native:ty, $x8:ty) => {
        #[thermite_macros::inline_always]
        impl ConcatRegister<$x8> for $native {
            fn concat(lo: Storage<$x8>, hi: Storage<$x8>) -> Storage<Self> {
                // Both 8-lane halves live in the low 64 bits; a 64-bit unpack interleaves them.
                unsafe { arch::_mm_unpacklo_epi64(lo.0, hi.0) }
            }

            fn split(value: Storage<Self>) -> (Storage<$x8>, Storage<$x8>) {
                (
                    ReducedRegister::new(value),
                    ReducedRegister::new(unsafe { arch::_mm_unpackhi_epi64(value, value) }),
                )
            }
        }
        // ExtendRegister<$x8> for $native comes free from the ReducedRegister blanket.
    };
}

impl_concat_x16_from_x8!(super::I8x16V2, I8x8V2);
impl_concat_x16_from_x8!(super::U8x16V2, U8x8V2);

// ===========================================================================================
// 8 <-> 16 widen/narrow casts.
//   widen 8 -> 16 via `cvtepi8_epi16`/`cvtepu8_epi16` (low 8 bytes -> 8x i16).
//   narrow 16 -> 8 via `pshufb` gathering the low byte of each i16 (even byte positions).
//   x4:  I8x4V2 <-> I16x4V2 (both reduced, data in low lanes).
//   x8:  I8x8V2 <-> I16x8V2 (native 8-lane).
//   x16: I8x16V2 <-> ArrayRegister<I16x8V2, 2> (the i16x16).
// ===========================================================================================

// pshufb mask gathering byte 0 of each of 8 i16 lanes (positions 0,2,4,...,14) into the low 8
// bytes (rest zeroed). Built with `setr` rather than a memory load.
// --- x4 ---
#[thermite_macros::inline_always]
impl CastRegister<I8x4V2> for super::half16::I16x4V2 {
    fn cast_from(value: Storage<I8x4V2>) -> Storage<Self> {
        ReducedRegister::new(unsafe { arch::_mm_cvtepi8_epi16(value.0) })
    }
}
#[thermite_macros::inline_always]
impl CastRegister<U8x4V2> for super::half16::U16x4V2 {
    fn cast_from(value: Storage<U8x4V2>) -> Storage<Self> {
        ReducedRegister::new(unsafe { arch::_mm_cvtepu8_epi16(value.0) })
    }
}
#[thermite_macros::inline_always]
impl CastRegister<super::half16::I16x4V2> for I8x4V2 {
    fn cast_from(value: Storage<super::half16::I16x4V2>) -> Storage<Self> {
        ReducedRegister::new(unsafe { arch::_mm_shuffle_epi8(value.0, arch::_mm_narrow_word_to_byte_maskx_v2()) })
    }
}
#[thermite_macros::inline_always]
impl CastRegister<super::half16::U16x4V2> for U8x4V2 {
    fn cast_from(value: Storage<super::half16::U16x4V2>) -> Storage<Self> {
        ReducedRegister::new(unsafe { arch::_mm_shuffle_epi8(value.0, arch::_mm_narrow_word_to_byte_maskx_v2()) })
    }
}

// --- x8 (i16x8 is native) ---
#[thermite_macros::inline_always]
impl CastRegister<I8x8V2> for super::I16x8V2 {
    fn cast_from(value: Storage<I8x8V2>) -> Storage<Self> {
        unsafe { arch::_mm_cvtepi8_epi16(value.0) }
    }
}
#[thermite_macros::inline_always]
impl CastRegister<U8x8V2> for super::U16x8V2 {
    fn cast_from(value: Storage<U8x8V2>) -> Storage<Self> {
        unsafe { arch::_mm_cvtepu8_epi16(value.0) }
    }
}
#[thermite_macros::inline_always]
impl CastRegister<super::I16x8V2> for I8x8V2 {
    fn cast_from(value: Storage<super::I16x8V2>) -> Storage<Self> {
        ReducedRegister::new(unsafe { arch::_mm_shuffle_epi8(value, arch::_mm_narrow_word_to_byte_maskx_v2()) })
    }
}
#[thermite_macros::inline_always]
impl CastRegister<super::U16x8V2> for U8x8V2 {
    fn cast_from(value: Storage<super::U16x8V2>) -> Storage<Self> {
        ReducedRegister::new(unsafe { arch::_mm_shuffle_epi8(value, arch::_mm_narrow_word_to_byte_maskx_v2()) })
    }
}

// --- saturating narrows into 8-bit (low half holds the result) ---

// i16x8 -> i8x8 via `packsswb`.
#[thermite_macros::inline_always]
impl SaturatingCastRegister<super::I16x8V2> for I8x8V2 {
    fn saturating_cast_from(value: Storage<super::I16x8V2>) -> Storage<Self> {
        ReducedRegister::new(unsafe { arch::_mm_packs_epi16(value, value) })
    }
}
// u16x8 -> u8x8: clamp the high end (`pminuw`) then `packuswb`.
#[thermite_macros::inline_always]
impl SaturatingCastRegister<super::U16x8V2> for U8x8V2 {
    fn saturating_cast_from(value: Storage<super::U16x8V2>) -> Storage<Self> {
        let clamped = unsafe { arch::_mm_min_epu16(value, arch::_mm_set1_epi16(0xFF)) };
        ReducedRegister::new(unsafe { arch::_mm_packus_epi16(clamped, clamped) })
    }
}
// i16x4 -> i8x4 via `packsswb` (4 valid words in the reduced source's low half).
#[thermite_macros::inline_always]
impl SaturatingCastRegister<super::half16::I16x4V2> for I8x4V2 {
    fn saturating_cast_from(value: Storage<super::half16::I16x4V2>) -> Storage<Self> {
        ReducedRegister::new(unsafe { arch::_mm_packs_epi16(value.0, value.0) })
    }
}
// u16x4 -> u8x4: clamp the high end (`pminuw`) then `packuswb`.
#[thermite_macros::inline_always]
impl SaturatingCastRegister<super::half16::U16x4V2> for U8x4V2 {
    fn saturating_cast_from(value: Storage<super::half16::U16x4V2>) -> Storage<Self> {
        let clamped = unsafe { arch::_mm_min_epu16(value.0, arch::_mm_set1_epi16(0xFF)) };
        ReducedRegister::new(unsafe { arch::_mm_packus_epi16(clamped, clamped) })
    }
}

// --- skip-level 32 -> 8: compose the 32 -> 16 and 16 -> 8 saturating packs (idempotent). ---
#[thermite_macros::inline_always]
impl SaturatingCastRegister<super::I32x4V2> for I8x4V2 {
    fn saturating_cast_from(value: Storage<super::I32x4V2>) -> Storage<Self> {
        let w = <super::half16::I16x4V2 as SaturatingCastRegister<super::I32x4V2>>::saturating_cast_from(value);
        <Self as SaturatingCastRegister<super::half16::I16x4V2>>::saturating_cast_from(w)
    }
}
#[thermite_macros::inline_always]
impl SaturatingCastRegister<super::U32x4V2> for U8x4V2 {
    fn saturating_cast_from(value: Storage<super::U32x4V2>) -> Storage<Self> {
        let w = <super::half16::U16x4V2 as SaturatingCastRegister<super::U32x4V2>>::saturating_cast_from(value);
        <Self as SaturatingCastRegister<super::half16::U16x4V2>>::saturating_cast_from(w)
    }
}
#[thermite_macros::inline_always]
impl SaturatingCastRegister<ArrayRegister<super::I32x4V2, 2>> for I8x8V2 {
    fn saturating_cast_from(value: Storage<ArrayRegister<super::I32x4V2, 2>>) -> Storage<Self> {
        let w = <super::I16x8V2 as SaturatingCastRegister<ArrayRegister<super::I32x4V2, 2>>>::saturating_cast_from(value);
        <Self as SaturatingCastRegister<super::I16x8V2>>::saturating_cast_from(w)
    }
}
#[thermite_macros::inline_always]
impl SaturatingCastRegister<ArrayRegister<super::U32x4V2, 2>> for U8x8V2 {
    fn saturating_cast_from(value: Storage<ArrayRegister<super::U32x4V2, 2>>) -> Storage<Self> {
        let w = <super::U16x8V2 as SaturatingCastRegister<ArrayRegister<super::U32x4V2, 2>>>::saturating_cast_from(value);
        <Self as SaturatingCastRegister<super::U16x8V2>>::saturating_cast_from(w)
    }
}

// --- x16 (i16x16 = ArrayRegister<I16x8V2, 2>) ---
#[thermite_macros::inline_always]
impl CastRegister<super::I8x16V2> for ArrayRegister<super::I16x8V2, 2> {
    fn cast_from(value: Storage<super::I8x16V2>) -> Storage<Self> {
        unsafe {
            ArrayRegister([
                arch::_mm_cvtepi8_epi16(value),
                arch::_mm_cvtepi8_epi16(arch::_mm_srli_si128(value, 8)),
            ])
        }
    }
}
#[thermite_macros::inline_always]
impl CastRegister<super::U8x16V2> for ArrayRegister<super::U16x8V2, 2> {
    fn cast_from(value: Storage<super::U8x16V2>) -> Storage<Self> {
        unsafe {
            ArrayRegister([
                arch::_mm_cvtepu8_epi16(value),
                arch::_mm_cvtepu8_epi16(arch::_mm_srli_si128(value, 8)),
            ])
        }
    }
}
#[thermite_macros::inline_always]
impl CastRegister<ArrayRegister<super::I16x8V2, 2>> for super::I8x16V2 {
    fn cast_from(value: Storage<ArrayRegister<super::I16x8V2, 2>>) -> Storage<Self> {
        unsafe { arch::_mm_cvt2epi16_epi8x_v2(value.0) }
    }
}
#[thermite_macros::inline_always]
impl CastRegister<ArrayRegister<super::U16x8V2, 2>> for super::U8x16V2 {
    fn cast_from(value: Storage<ArrayRegister<super::U16x8V2, 2>>) -> Storage<Self> {
        unsafe { arch::_mm_cvt2epi16_epi8x_v2(value.0) }
    }
}

// ===========================================================================================
// Widen casts to 32-bit (CastRegister = numeric widen/narrow).
//
// x4: I8x4V2 <-> I32x4V2 (native 4-lane).
// x8: I8x8V2 <-> ArrayRegister<I32x4V2, 2> (the v2 i32x8).
// x2: ArrayRegister<i8,2> <-> I32x2V2 (reduced 2-lane).
// ===========================================================================================

// --- x4 widen i8 -> i32 (low 4 bytes -> 4x i32) ---
#[thermite_macros::inline_always]
impl CastRegister<I8x4V2> for super::I32x4V2 {
    fn cast_from(value: Storage<I8x4V2>) -> Storage<Self> {
        unsafe { arch::_mm_cvtepi8_epi32(value.0) }
    }
}
#[thermite_macros::inline_always]
impl CastRegister<U8x4V2> for super::U32x4V2 {
    fn cast_from(value: Storage<U8x4V2>) -> Storage<Self> {
        unsafe { arch::_mm_cvtepu8_epi32(value.0) }
    }
}

// --- x4 narrow i32 -> i8 (low byte of each 32-bit lane, gathered into the low 4 bytes) ---
// pshufb mask gathering byte 0 of each of 4 i32 lanes (positions 0,4,8,12) into the low 4 bytes.
#[thermite_macros::inline_always]
impl CastRegister<super::I32x4V2> for I8x4V2 {
    fn cast_from(value: Storage<super::I32x4V2>) -> Storage<Self> {
        ReducedRegister::new(unsafe {
            arch::_mm_shuffle_epi8(value, arch::_mm_narrow_dword_to_byte_maskx_v2())
        })
    }
}
#[thermite_macros::inline_always]
impl CastRegister<super::U32x4V2> for U8x4V2 {
    fn cast_from(value: Storage<super::U32x4V2>) -> Storage<Self> {
        ReducedRegister::new(unsafe {
            arch::_mm_shuffle_epi8(value, arch::_mm_narrow_dword_to_byte_maskx_v2())
        })
    }
}

// --- x8 widen i8 -> i32 (low 8 bytes -> two 4x i32 lanes) ---
#[thermite_macros::inline_always]
impl CastRegister<I8x8V2> for ArrayRegister<super::I32x4V2, 2> {
    fn cast_from(value: Storage<I8x8V2>) -> Storage<Self> {
        unsafe {
            let lo = arch::_mm_cvtepi8_epi32(value.0);
            let hi = arch::_mm_cvtepi8_epi32(arch::_mm_srli_si128(value.0, 4));
            ArrayRegister([lo, hi])
        }
    }
}
#[thermite_macros::inline_always]
impl CastRegister<U8x8V2> for ArrayRegister<super::U32x4V2, 2> {
    fn cast_from(value: Storage<U8x8V2>) -> Storage<Self> {
        unsafe {
            let lo = arch::_mm_cvtepu8_epi32(value.0);
            let hi = arch::_mm_cvtepu8_epi32(arch::_mm_srli_si128(value.0, 4));
            ArrayRegister([lo, hi])
        }
    }
}

// --- x8 narrow i32 -> i8 (low byte of each lane in both halves -> low 8 bytes) ---
#[thermite_macros::inline_always]
impl CastRegister<ArrayRegister<super::I32x4V2, 2>> for I8x8V2 {
    fn cast_from(value: Storage<ArrayRegister<super::I32x4V2, 2>>) -> Storage<Self> {
        unsafe {
            let mask = arch::_mm_narrow_dword_to_byte_maskx_v2();
            let lo = arch::_mm_shuffle_epi8(value.0[0], mask); // 4 bytes in low 32 bits
            let hi = arch::_mm_shuffle_epi8(value.0[1], mask);
            ReducedRegister::new(arch::_mm_unpacklo_epi32(lo, hi))
        }
    }
}
#[thermite_macros::inline_always]
impl CastRegister<ArrayRegister<super::U32x4V2, 2>> for U8x8V2 {
    fn cast_from(value: Storage<ArrayRegister<super::U32x4V2, 2>>) -> Storage<Self> {
        unsafe {
            let mask = arch::_mm_narrow_dword_to_byte_maskx_v2();
            let lo = arch::_mm_shuffle_epi8(value.0[0], mask);
            let hi = arch::_mm_shuffle_epi8(value.0[1], mask);
            ReducedRegister::new(arch::_mm_unpacklo_epi32(lo, hi))
        }
    }
}

// --- x16 widen i8 -> i32 (native 16 bytes -> ArrayRegister<I32x4V2, 4>) ---
#[thermite_macros::inline_always]
impl CastRegister<super::I8x16V2> for ArrayRegister<super::I32x4V2, 4> {
    fn cast_from(value: Storage<super::I8x16V2>) -> Storage<Self> {
        unsafe {
            ArrayRegister([
                arch::_mm_cvtepi8_epi32(value),
                arch::_mm_cvtepi8_epi32(arch::_mm_srli_si128(value, 4)),
                arch::_mm_cvtepi8_epi32(arch::_mm_srli_si128(value, 8)),
                arch::_mm_cvtepi8_epi32(arch::_mm_srli_si128(value, 12)),
            ])
        }
    }
}
#[thermite_macros::inline_always]
impl CastRegister<super::U8x16V2> for ArrayRegister<super::U32x4V2, 4> {
    fn cast_from(value: Storage<super::U8x16V2>) -> Storage<Self> {
        unsafe {
            ArrayRegister([
                arch::_mm_cvtepu8_epi32(value),
                arch::_mm_cvtepu8_epi32(arch::_mm_srli_si128(value, 4)),
                arch::_mm_cvtepu8_epi32(arch::_mm_srli_si128(value, 8)),
                arch::_mm_cvtepu8_epi32(arch::_mm_srli_si128(value, 12)),
            ])
        }
    }
}

// --- x16 narrow i32 -> i8 (ArrayRegister<I32x4V2, 4> -> native 16 bytes) ---
#[thermite_macros::inline_always]
impl CastRegister<ArrayRegister<super::I32x4V2, 4>> for super::I8x16V2 {
    fn cast_from(value: Storage<ArrayRegister<super::I32x4V2, 4>>) -> Storage<Self> {
        unsafe { arch::_mm_cvt4epi32_epi8x_v2(value.0) }
    }
}
#[thermite_macros::inline_always]
impl CastRegister<ArrayRegister<super::U32x4V2, 4>> for super::U8x16V2 {
    fn cast_from(value: Storage<ArrayRegister<super::U32x4V2, 4>>) -> Storage<Self> {
        unsafe { arch::_mm_cvt4epi32_epi8x_v2(value.0) }
    }
}

// --- x2 widen i8 -> i32 (ArrayRegister<i8,2> -> I32x2V2 reduced) ---
#[thermite_macros::inline_always]
impl CastRegister<ArrayRegister<i8, 2>> for super::half::I32x2V2 {
    fn cast_from(value: Storage<ArrayRegister<i8, 2>>) -> Storage<Self> {
        ReducedRegister::new(unsafe { arch::_mm_setr_epi32(value.0[0] as i32, value.0[1] as i32, 0, 0) })
    }
}
#[thermite_macros::inline_always]
impl CastRegister<ArrayRegister<u8, 2>> for super::half::U32x2V2 {
    fn cast_from(value: Storage<ArrayRegister<u8, 2>>) -> Storage<Self> {
        ReducedRegister::new(unsafe { arch::_mm_setr_epi32(value.0[0] as i32, value.0[1] as i32, 0, 0) })
    }
}

// --- x2 narrow i32 -> i8 (I32x2V2 reduced -> ArrayRegister<i8,2>) ---
#[thermite_macros::inline_always]
impl CastRegister<super::half::I32x2V2> for ArrayRegister<i8, 2> {
    fn cast_from(value: Storage<super::half::I32x2V2>) -> Storage<Self> {
        let mut arr = [0i32; 4];
        unsafe { arch::_mm_storeu_si128(arr.as_mut_ptr() as *mut _, value.0) };
        ArrayRegister([arr[0] as i8, arr[1] as i8])
    }
}
#[thermite_macros::inline_always]
impl CastRegister<super::half::U32x2V2> for ArrayRegister<u8, 2> {
    fn cast_from(value: Storage<super::half::U32x2V2>) -> Storage<Self> {
        let mut arr = [0u32; 4];
        unsafe { arch::_mm_storeu_si128(arr.as_mut_ptr() as *mut _, value.0) };
        ArrayRegister([arr[0] as u8, arr[1] as u8])
    }
}

// ===========================================================================================
// Widen/narrow casts to 64-bit (CastRegister = numeric widen / `as`-style narrow).
//
//   widen 8 -> 64 via `cvtepi8_epi64`/`cvtepu8_epi64` (low 2 bytes -> 2x i64 per call).
//   narrow 64 -> 8 via `pshufb` gathering byte 0 of each i64 lane, then combining the array lanes.
//   x2:  ArrayRegister<i8,2> <-> I64x2V2 (native 2-lane).
//   x4:  I8x4V2 <-> ArrayRegister<I64x2V2, 2> (the v2 i64x4).
//   x8:  I8x8V2 <-> ArrayRegister<I64x2V2, 4> (the v2 i64x8).
//   x16: I8x16V2 <-> ArrayRegister<I64x2V2, 8> (the v2 i64x16).
// ===========================================================================================

// pshufb mask gathering byte 0 of each of 2 i64 lanes (positions 0 and 8) into the low 2 bytes.
// --- x2 widen i8 -> i64 (ArrayRegister<i8,2> -> I64x2V2 native) ---
#[thermite_macros::inline_always]
impl CastRegister<ArrayRegister<i8, 2>> for super::I64x2V2 {
    fn cast_from(value: Storage<ArrayRegister<i8, 2>>) -> Storage<Self> {
        unsafe { arch::_mm_set_epi64x(value.0[1] as i64, value.0[0] as i64) }
    }
}
#[thermite_macros::inline_always]
impl CastRegister<ArrayRegister<u8, 2>> for super::U64x2V2 {
    fn cast_from(value: Storage<ArrayRegister<u8, 2>>) -> Storage<Self> {
        unsafe { arch::_mm_set_epi64x(value.0[1] as i64, value.0[0] as i64) }
    }
}

// --- x2 narrow i64 -> i8 (I64x2V2 native -> ArrayRegister<i8,2>) ---
#[thermite_macros::inline_always]
impl CastRegister<super::I64x2V2> for ArrayRegister<i8, 2> {
    fn cast_from(value: Storage<super::I64x2V2>) -> Storage<Self> {
        let mut arr = [0i64; 2];
        unsafe { arch::_mm_storeu_si128(arr.as_mut_ptr() as *mut _, value) };
        ArrayRegister([arr[0] as i8, arr[1] as i8])
    }
}
#[thermite_macros::inline_always]
impl CastRegister<super::U64x2V2> for ArrayRegister<u8, 2> {
    fn cast_from(value: Storage<super::U64x2V2>) -> Storage<Self> {
        let mut arr = [0u64; 2];
        unsafe { arch::_mm_storeu_si128(arr.as_mut_ptr() as *mut _, value) };
        ArrayRegister([arr[0] as u8, arr[1] as u8])
    }
}

// --- x4 widen i8 -> i64 (low 4 bytes -> two 2x i64 lanes) ---
#[thermite_macros::inline_always]
impl CastRegister<I8x4V2> for ArrayRegister<super::I64x2V2, 2> {
    fn cast_from(value: Storage<I8x4V2>) -> Storage<Self> {
        unsafe {
            ArrayRegister([
                arch::_mm_cvtepi8_epi64(value.0),
                arch::_mm_cvtepi8_epi64(arch::_mm_srli_si128(value.0, 2)),
            ])
        }
    }
}
#[thermite_macros::inline_always]
impl CastRegister<U8x4V2> for ArrayRegister<super::U64x2V2, 2> {
    fn cast_from(value: Storage<U8x4V2>) -> Storage<Self> {
        unsafe {
            ArrayRegister([
                arch::_mm_cvtepu8_epi64(value.0),
                arch::_mm_cvtepu8_epi64(arch::_mm_srli_si128(value.0, 2)),
            ])
        }
    }
}

// --- x4 narrow i64 -> i8 (byte 0 of each of 4 lanes -> low 4 bytes) ---
#[thermite_macros::inline_always]
impl CastRegister<ArrayRegister<super::I64x2V2, 2>> for I8x4V2 {
    fn cast_from(value: Storage<ArrayRegister<super::I64x2V2, 2>>) -> Storage<Self> {
        unsafe {
            let mask = arch::_mm_narrow_qword_to_byte_maskx_v2();
            let lo = arch::_mm_shuffle_epi8(value.0[0], mask); // 2 bytes in low 16 bits
            let hi = arch::_mm_shuffle_epi8(value.0[1], mask);
            ReducedRegister::new(arch::_mm_unpacklo_epi16(lo, hi))
        }
    }
}
#[thermite_macros::inline_always]
impl CastRegister<ArrayRegister<super::U64x2V2, 2>> for U8x4V2 {
    fn cast_from(value: Storage<ArrayRegister<super::U64x2V2, 2>>) -> Storage<Self> {
        unsafe {
            let mask = arch::_mm_narrow_qword_to_byte_maskx_v2();
            let lo = arch::_mm_shuffle_epi8(value.0[0], mask);
            let hi = arch::_mm_shuffle_epi8(value.0[1], mask);
            ReducedRegister::new(arch::_mm_unpacklo_epi16(lo, hi))
        }
    }
}

// --- x8 widen i8 -> i64 (low 8 bytes -> four 2x i64 lanes) ---
#[thermite_macros::inline_always]
impl CastRegister<I8x8V2> for ArrayRegister<super::I64x2V2, 4> {
    fn cast_from(value: Storage<I8x8V2>) -> Storage<Self> {
        unsafe {
            ArrayRegister([
                arch::_mm_cvtepi8_epi64(value.0),
                arch::_mm_cvtepi8_epi64(arch::_mm_srli_si128(value.0, 2)),
                arch::_mm_cvtepi8_epi64(arch::_mm_srli_si128(value.0, 4)),
                arch::_mm_cvtepi8_epi64(arch::_mm_srli_si128(value.0, 6)),
            ])
        }
    }
}
#[thermite_macros::inline_always]
impl CastRegister<U8x8V2> for ArrayRegister<super::U64x2V2, 4> {
    fn cast_from(value: Storage<U8x8V2>) -> Storage<Self> {
        unsafe {
            ArrayRegister([
                arch::_mm_cvtepu8_epi64(value.0),
                arch::_mm_cvtepu8_epi64(arch::_mm_srli_si128(value.0, 2)),
                arch::_mm_cvtepu8_epi64(arch::_mm_srli_si128(value.0, 4)),
                arch::_mm_cvtepu8_epi64(arch::_mm_srli_si128(value.0, 6)),
            ])
        }
    }
}

// --- x8 narrow i64 -> i8 (byte 0 of each of 8 lanes -> low 8 bytes) ---
#[thermite_macros::inline_always]
impl CastRegister<ArrayRegister<super::I64x2V2, 4>> for I8x8V2 {
    fn cast_from(value: Storage<ArrayRegister<super::I64x2V2, 4>>) -> Storage<Self> {
        unsafe { ReducedRegister::new(arch::_mm_cvt4epi64_epi8x_v2(value.0)) }
    }
}
#[thermite_macros::inline_always]
impl CastRegister<ArrayRegister<super::U64x2V2, 4>> for U8x8V2 {
    fn cast_from(value: Storage<ArrayRegister<super::U64x2V2, 4>>) -> Storage<Self> {
        unsafe { ReducedRegister::new(arch::_mm_cvt4epi64_epi8x_v2(value.0)) }
    }
}

// --- x16 widen i8 -> i64 (native 16 bytes -> ArrayRegister<I64x2V2, 8>) ---
#[thermite_macros::inline_always]
impl CastRegister<super::I8x16V2> for ArrayRegister<super::I64x2V2, 8> {
    fn cast_from(value: Storage<super::I8x16V2>) -> Storage<Self> {
        unsafe {
            ArrayRegister([
                arch::_mm_cvtepi8_epi64(value),
                arch::_mm_cvtepi8_epi64(arch::_mm_srli_si128(value, 2)),
                arch::_mm_cvtepi8_epi64(arch::_mm_srli_si128(value, 4)),
                arch::_mm_cvtepi8_epi64(arch::_mm_srli_si128(value, 6)),
                arch::_mm_cvtepi8_epi64(arch::_mm_srli_si128(value, 8)),
                arch::_mm_cvtepi8_epi64(arch::_mm_srli_si128(value, 10)),
                arch::_mm_cvtepi8_epi64(arch::_mm_srli_si128(value, 12)),
                arch::_mm_cvtepi8_epi64(arch::_mm_srli_si128(value, 14)),
            ])
        }
    }
}
#[thermite_macros::inline_always]
impl CastRegister<super::U8x16V2> for ArrayRegister<super::U64x2V2, 8> {
    fn cast_from(value: Storage<super::U8x16V2>) -> Storage<Self> {
        unsafe {
            ArrayRegister([
                arch::_mm_cvtepu8_epi64(value),
                arch::_mm_cvtepu8_epi64(arch::_mm_srli_si128(value, 2)),
                arch::_mm_cvtepu8_epi64(arch::_mm_srli_si128(value, 4)),
                arch::_mm_cvtepu8_epi64(arch::_mm_srli_si128(value, 6)),
                arch::_mm_cvtepu8_epi64(arch::_mm_srli_si128(value, 8)),
                arch::_mm_cvtepu8_epi64(arch::_mm_srli_si128(value, 10)),
                arch::_mm_cvtepu8_epi64(arch::_mm_srli_si128(value, 12)),
                arch::_mm_cvtepu8_epi64(arch::_mm_srli_si128(value, 14)),
            ])
        }
    }
}

// --- x16 narrow i64 -> i8 (ArrayRegister<I64x2V2, 8> -> native 16 bytes) ---
#[thermite_macros::inline_always]
impl CastRegister<ArrayRegister<super::I64x2V2, 8>> for super::I8x16V2 {
    fn cast_from(value: Storage<ArrayRegister<super::I64x2V2, 8>>) -> Storage<Self> {
        unsafe { arch::_mm_cvt8epi64_epi8x_v2(value.0) }
    }
}
#[thermite_macros::inline_always]
impl CastRegister<ArrayRegister<super::U64x2V2, 8>> for super::U8x16V2 {
    fn cast_from(value: Storage<ArrayRegister<super::U64x2V2, 8>>) -> Storage<Self> {
        unsafe { arch::_mm_cvt8epi64_epi8x_v2(value.0) }
    }
}

// ===========================================================================================
// 8-bit <-> f32 / f64 direct casts.
//
// There is no native i8->float instruction, so each body is a single inline intrinsic
// sequence routed through a 32-bit integer intermediate of the SAME lane count (where the
// hardware int<->float converts live):
//   widen  i8/u8 -> f32 = (int widen i8/u8 -> i32, exactly the 8->32 widen above) then
//                          (i32 -> f32 via `_mm_cvtepi32_ps`). Value-preserving / exact.
//   narrow f32   -> i8  = (f32 -> i32 truncating via `_mm_cvttps_epi32`) then the existing
//                          i32 -> byte narrow (`arch::_mm_narrow_dword_to_byte_maskx_v2()` pshufb / helpers).
//   f64 has half the lanes per native register, so the i32 intermediate fans out across two
//   `_mm_cvtepi32_pd` / `_mm_cvttpd_epi32` calls (mirroring f32x4 <-> ArrayRegister<F64x2,2>).
// Unsigned sources zero-extend to i32 (`_mm_cvtepu8_epi32`); the widened value is a positive
// i32 so the signed `_mm_cvtepi32_ps`/`_mm_cvtepi32_pd` convert is still exact. For the float
// -> unsigned narrow the low byte is identical signed or unsigned, so the signed narrow is reused.
// ===========================================================================================

// --- f64 fan-out helpers (i32 <-> f64 across the 2-lane native f64 register) ---

// Widen the low 4 i32 lanes of a __m128i into two F64x2 registers (4 f64).
// Narrow two F64x2 registers (4 f64) into the low 4 i32 lanes of a __m128i (truncating).
// --- x2 8 <-> f32 (ArrayRegister<i8,2> <-> F32x2V2 reduced) ---
#[thermite_macros::inline_always]
impl CastRegister<ArrayRegister<i8, 2>> for super::half::F32x2V2 {
    fn cast_from(value: Storage<ArrayRegister<i8, 2>>) -> Storage<Self> {
        ReducedRegister::new(unsafe {
            arch::_mm_cvtepi32_ps(arch::_mm_setr_epi32(value.0[0] as i32, value.0[1] as i32, 0, 0))
        })
    }
}
#[thermite_macros::inline_always]
impl CastRegister<ArrayRegister<u8, 2>> for super::half::F32x2V2 {
    fn cast_from(value: Storage<ArrayRegister<u8, 2>>) -> Storage<Self> {
        ReducedRegister::new(unsafe {
            arch::_mm_cvtepi32_ps(arch::_mm_setr_epi32(value.0[0] as i32, value.0[1] as i32, 0, 0))
        })
    }
}
#[thermite_macros::inline_always]
impl CastRegister<super::half::F32x2V2> for ArrayRegister<i8, 2> {
    fn cast_from(value: Storage<super::half::F32x2V2>) -> Storage<Self> {
        let mut arr = [0i32; 4];
        unsafe { arch::_mm_storeu_si128(arr.as_mut_ptr() as *mut _, arch::_mm_cvttps_epi32(value.0)) };
        ArrayRegister([arr[0] as i8, arr[1] as i8])
    }
}
#[thermite_macros::inline_always]
impl CastRegister<super::half::F32x2V2> for ArrayRegister<u8, 2> {
    fn cast_from(value: Storage<super::half::F32x2V2>) -> Storage<Self> {
        let mut arr = [0i32; 4];
        unsafe { arch::_mm_storeu_si128(arr.as_mut_ptr() as *mut _, arch::_mm_cvttps_epi32(value.0)) };
        ArrayRegister([arr[0] as u8, arr[1] as u8])
    }
}

// --- x2 8 <-> f64 (ArrayRegister<i8,2> <-> F64x2V2 native) ---
#[thermite_macros::inline_always]
impl CastRegister<ArrayRegister<i8, 2>> for super::F64x2V2 {
    fn cast_from(value: Storage<ArrayRegister<i8, 2>>) -> Storage<Self> {
        unsafe { arch::_mm_cvtepi32_pd(arch::_mm_setr_epi32(value.0[0] as i32, value.0[1] as i32, 0, 0)) }
    }
}
#[thermite_macros::inline_always]
impl CastRegister<ArrayRegister<u8, 2>> for super::F64x2V2 {
    fn cast_from(value: Storage<ArrayRegister<u8, 2>>) -> Storage<Self> {
        unsafe { arch::_mm_cvtepi32_pd(arch::_mm_setr_epi32(value.0[0] as i32, value.0[1] as i32, 0, 0)) }
    }
}
#[thermite_macros::inline_always]
impl CastRegister<super::F64x2V2> for ArrayRegister<i8, 2> {
    fn cast_from(value: Storage<super::F64x2V2>) -> Storage<Self> {
        let mut arr = [0i32; 4];
        unsafe { arch::_mm_storeu_si128(arr.as_mut_ptr() as *mut _, arch::_mm_cvttpd_epi32(value)) };
        ArrayRegister([arr[0] as i8, arr[1] as i8])
    }
}
#[thermite_macros::inline_always]
impl CastRegister<super::F64x2V2> for ArrayRegister<u8, 2> {
    fn cast_from(value: Storage<super::F64x2V2>) -> Storage<Self> {
        let mut arr = [0i32; 4];
        unsafe { arch::_mm_storeu_si128(arr.as_mut_ptr() as *mut _, arch::_mm_cvttpd_epi32(value)) };
        ArrayRegister([arr[0] as u8, arr[1] as u8])
    }
}

// --- x4 8 <-> f32 (I8x4V2 reduced <-> F32x4V2 native) ---
#[thermite_macros::inline_always]
impl CastRegister<I8x4V2> for super::F32x4V2 {
    fn cast_from(value: Storage<I8x4V2>) -> Storage<Self> {
        unsafe { arch::_mm_cvtepi32_ps(arch::_mm_cvtepi8_epi32(value.0)) }
    }
}
#[thermite_macros::inline_always]
impl CastRegister<U8x4V2> for super::F32x4V2 {
    fn cast_from(value: Storage<U8x4V2>) -> Storage<Self> {
        unsafe { arch::_mm_cvtepi32_ps(arch::_mm_cvtepu8_epi32(value.0)) }
    }
}
#[thermite_macros::inline_always]
impl CastRegister<super::F32x4V2> for I8x4V2 {
    fn cast_from(value: Storage<super::F32x4V2>) -> Storage<Self> {
        ReducedRegister::new(unsafe {
            arch::_mm_shuffle_epi8(
                arch::_mm_cvttps_epi32(value),
                arch::_mm_narrow_dword_to_byte_maskx_v2(),
            )
        })
    }
}
#[thermite_macros::inline_always]
impl CastRegister<super::F32x4V2> for U8x4V2 {
    fn cast_from(value: Storage<super::F32x4V2>) -> Storage<Self> {
        ReducedRegister::new(unsafe {
            arch::_mm_shuffle_epi8(
                arch::_mm_cvttps_epi32(value),
                arch::_mm_narrow_dword_to_byte_maskx_v2(),
            )
        })
    }
}

// --- x4 8 <-> f64 (I8x4V2 reduced <-> ArrayRegister<F64x2V2, 2>) ---
#[thermite_macros::inline_always]
impl CastRegister<I8x4V2> for ArrayRegister<super::F64x2V2, 2> {
    fn cast_from(value: Storage<I8x4V2>) -> Storage<Self> {
        ArrayRegister(unsafe { arch::_mm_cvtepi32_2pdx_v2(arch::_mm_cvtepi8_epi32(value.0)) })
    }
}
#[thermite_macros::inline_always]
impl CastRegister<U8x4V2> for ArrayRegister<super::F64x2V2, 2> {
    fn cast_from(value: Storage<U8x4V2>) -> Storage<Self> {
        ArrayRegister(unsafe { arch::_mm_cvtepi32_2pdx_v2(arch::_mm_cvtepu8_epi32(value.0)) })
    }
}
#[thermite_macros::inline_always]
impl CastRegister<ArrayRegister<super::F64x2V2, 2>> for I8x4V2 {
    fn cast_from(value: Storage<ArrayRegister<super::F64x2V2, 2>>) -> Storage<Self> {
        ReducedRegister::new(unsafe {
            let i32s = arch::_mm_cvtt2pd_epi32x_v2(value.0);
            arch::_mm_shuffle_epi8(i32s, arch::_mm_narrow_dword_to_byte_maskx_v2())
        })
    }
}
#[thermite_macros::inline_always]
impl CastRegister<ArrayRegister<super::F64x2V2, 2>> for U8x4V2 {
    fn cast_from(value: Storage<ArrayRegister<super::F64x2V2, 2>>) -> Storage<Self> {
        ReducedRegister::new(unsafe {
            let i32s = arch::_mm_cvtt2pd_epi32x_v2(value.0);
            arch::_mm_shuffle_epi8(i32s, arch::_mm_narrow_dword_to_byte_maskx_v2())
        })
    }
}

// --- x8 8 <-> f32 (I8x8V2 reduced <-> ArrayRegister<F32x4V2, 2>) ---
#[thermite_macros::inline_always]
impl CastRegister<I8x8V2> for ArrayRegister<super::F32x4V2, 2> {
    fn cast_from(value: Storage<I8x8V2>) -> Storage<Self> {
        unsafe {
            let lo = arch::_mm_cvtepi32_ps(arch::_mm_cvtepi8_epi32(value.0));
            let hi = arch::_mm_cvtepi32_ps(arch::_mm_cvtepi8_epi32(arch::_mm_srli_si128(value.0, 4)));
            ArrayRegister([lo, hi])
        }
    }
}
#[thermite_macros::inline_always]
impl CastRegister<U8x8V2> for ArrayRegister<super::F32x4V2, 2> {
    fn cast_from(value: Storage<U8x8V2>) -> Storage<Self> {
        unsafe {
            let lo = arch::_mm_cvtepi32_ps(arch::_mm_cvtepu8_epi32(value.0));
            let hi = arch::_mm_cvtepi32_ps(arch::_mm_cvtepu8_epi32(arch::_mm_srli_si128(value.0, 4)));
            ArrayRegister([lo, hi])
        }
    }
}
#[thermite_macros::inline_always]
impl CastRegister<ArrayRegister<super::F32x4V2, 2>> for I8x8V2 {
    fn cast_from(value: Storage<ArrayRegister<super::F32x4V2, 2>>) -> Storage<Self> {
        unsafe {
            let mask = arch::_mm_narrow_dword_to_byte_maskx_v2();
            let lo = arch::_mm_shuffle_epi8(arch::_mm_cvttps_epi32(value.0[0]), mask);
            let hi = arch::_mm_shuffle_epi8(arch::_mm_cvttps_epi32(value.0[1]), mask);
            ReducedRegister::new(arch::_mm_unpacklo_epi32(lo, hi))
        }
    }
}
#[thermite_macros::inline_always]
impl CastRegister<ArrayRegister<super::F32x4V2, 2>> for U8x8V2 {
    fn cast_from(value: Storage<ArrayRegister<super::F32x4V2, 2>>) -> Storage<Self> {
        unsafe {
            let mask = arch::_mm_narrow_dword_to_byte_maskx_v2();
            let lo = arch::_mm_shuffle_epi8(arch::_mm_cvttps_epi32(value.0[0]), mask);
            let hi = arch::_mm_shuffle_epi8(arch::_mm_cvttps_epi32(value.0[1]), mask);
            ReducedRegister::new(arch::_mm_unpacklo_epi32(lo, hi))
        }
    }
}

// --- x8 8 <-> f64 (I8x8V2 reduced <-> ArrayRegister<F64x2V2, 4>) ---
#[thermite_macros::inline_always]
impl CastRegister<I8x8V2> for ArrayRegister<super::F64x2V2, 4> {
    fn cast_from(value: Storage<I8x8V2>) -> Storage<Self> {
        unsafe {
            let lo = arch::_mm_cvtepi32_2pdx_v2(arch::_mm_cvtepi8_epi32(value.0));
            let hi = arch::_mm_cvtepi32_2pdx_v2(arch::_mm_cvtepi8_epi32(arch::_mm_srli_si128(value.0, 4)));
            ArrayRegister([lo[0], lo[1], hi[0], hi[1]])
        }
    }
}
#[thermite_macros::inline_always]
impl CastRegister<U8x8V2> for ArrayRegister<super::F64x2V2, 4> {
    fn cast_from(value: Storage<U8x8V2>) -> Storage<Self> {
        unsafe {
            let lo = arch::_mm_cvtepi32_2pdx_v2(arch::_mm_cvtepu8_epi32(value.0));
            let hi = arch::_mm_cvtepi32_2pdx_v2(arch::_mm_cvtepu8_epi32(arch::_mm_srli_si128(value.0, 4)));
            ArrayRegister([lo[0], lo[1], hi[0], hi[1]])
        }
    }
}
#[thermite_macros::inline_always]
impl CastRegister<ArrayRegister<super::F64x2V2, 4>> for I8x8V2 {
    fn cast_from(value: Storage<ArrayRegister<super::F64x2V2, 4>>) -> Storage<Self> {
        let v = value.0;
        unsafe {
            let lo = arch::_mm_cvtt2pd_epi32x_v2([v[0], v[1]]);
            let hi = arch::_mm_cvtt2pd_epi32x_v2([v[2], v[3]]);
            let mask = arch::_mm_narrow_dword_to_byte_maskx_v2();
            let lo = arch::_mm_shuffle_epi8(lo, mask);
            let hi = arch::_mm_shuffle_epi8(hi, mask);
            ReducedRegister::new(arch::_mm_unpacklo_epi32(lo, hi))
        }
    }
}
#[thermite_macros::inline_always]
impl CastRegister<ArrayRegister<super::F64x2V2, 4>> for U8x8V2 {
    fn cast_from(value: Storage<ArrayRegister<super::F64x2V2, 4>>) -> Storage<Self> {
        let v = value.0;
        unsafe {
            let lo = arch::_mm_cvtt2pd_epi32x_v2([v[0], v[1]]);
            let hi = arch::_mm_cvtt2pd_epi32x_v2([v[2], v[3]]);
            let mask = arch::_mm_narrow_dword_to_byte_maskx_v2();
            let lo = arch::_mm_shuffle_epi8(lo, mask);
            let hi = arch::_mm_shuffle_epi8(hi, mask);
            ReducedRegister::new(arch::_mm_unpacklo_epi32(lo, hi))
        }
    }
}

// --- x16 8 <-> f32 (I8x16V2 native <-> ArrayRegister<F32x4V2, 4>) ---
#[thermite_macros::inline_always]
impl CastRegister<super::I8x16V2> for ArrayRegister<super::F32x4V2, 4> {
    fn cast_from(value: Storage<super::I8x16V2>) -> Storage<Self> {
        unsafe {
            ArrayRegister([
                arch::_mm_cvtepi32_ps(arch::_mm_cvtepi8_epi32(value)),
                arch::_mm_cvtepi32_ps(arch::_mm_cvtepi8_epi32(arch::_mm_srli_si128(value, 4))),
                arch::_mm_cvtepi32_ps(arch::_mm_cvtepi8_epi32(arch::_mm_srli_si128(value, 8))),
                arch::_mm_cvtepi32_ps(arch::_mm_cvtepi8_epi32(arch::_mm_srli_si128(value, 12))),
            ])
        }
    }
}
#[thermite_macros::inline_always]
impl CastRegister<super::U8x16V2> for ArrayRegister<super::F32x4V2, 4> {
    fn cast_from(value: Storage<super::U8x16V2>) -> Storage<Self> {
        unsafe {
            ArrayRegister([
                arch::_mm_cvtepi32_ps(arch::_mm_cvtepu8_epi32(value)),
                arch::_mm_cvtepi32_ps(arch::_mm_cvtepu8_epi32(arch::_mm_srli_si128(value, 4))),
                arch::_mm_cvtepi32_ps(arch::_mm_cvtepu8_epi32(arch::_mm_srli_si128(value, 8))),
                arch::_mm_cvtepi32_ps(arch::_mm_cvtepu8_epi32(arch::_mm_srli_si128(value, 12))),
            ])
        }
    }
}
#[thermite_macros::inline_always]
impl CastRegister<ArrayRegister<super::F32x4V2, 4>> for super::I8x16V2 {
    fn cast_from(value: Storage<ArrayRegister<super::F32x4V2, 4>>) -> Storage<Self> {
        let v = value.0;
        unsafe {
            arch::_mm_cvt4epi32_epi8x_v2([
                arch::_mm_cvttps_epi32(v[0]),
                arch::_mm_cvttps_epi32(v[1]),
                arch::_mm_cvttps_epi32(v[2]),
                arch::_mm_cvttps_epi32(v[3]),
            ])
        }
    }
}
#[thermite_macros::inline_always]
impl CastRegister<ArrayRegister<super::F32x4V2, 4>> for super::U8x16V2 {
    fn cast_from(value: Storage<ArrayRegister<super::F32x4V2, 4>>) -> Storage<Self> {
        let v = value.0;
        unsafe {
            arch::_mm_cvt4epi32_epi8x_v2([
                arch::_mm_cvttps_epi32(v[0]),
                arch::_mm_cvttps_epi32(v[1]),
                arch::_mm_cvttps_epi32(v[2]),
                arch::_mm_cvttps_epi32(v[3]),
            ])
        }
    }
}

// --- x16 8 <-> f64 (I8x16V2 native <-> ArrayRegister<F64x2V2, 8>) ---
#[thermite_macros::inline_always]
impl CastRegister<super::I8x16V2> for ArrayRegister<super::F64x2V2, 8> {
    fn cast_from(value: Storage<super::I8x16V2>) -> Storage<Self> {
        unsafe {
            let a = arch::_mm_cvtepi32_2pdx_v2(arch::_mm_cvtepi8_epi32(value));
            let b = arch::_mm_cvtepi32_2pdx_v2(arch::_mm_cvtepi8_epi32(arch::_mm_srli_si128(value, 4)));
            let c = arch::_mm_cvtepi32_2pdx_v2(arch::_mm_cvtepi8_epi32(arch::_mm_srli_si128(value, 8)));
            let d = arch::_mm_cvtepi32_2pdx_v2(arch::_mm_cvtepi8_epi32(arch::_mm_srli_si128(value, 12)));
            ArrayRegister([a[0], a[1], b[0], b[1], c[0], c[1], d[0], d[1]])
        }
    }
}
#[thermite_macros::inline_always]
impl CastRegister<super::U8x16V2> for ArrayRegister<super::F64x2V2, 8> {
    fn cast_from(value: Storage<super::U8x16V2>) -> Storage<Self> {
        unsafe {
            let a = arch::_mm_cvtepi32_2pdx_v2(arch::_mm_cvtepu8_epi32(value));
            let b = arch::_mm_cvtepi32_2pdx_v2(arch::_mm_cvtepu8_epi32(arch::_mm_srli_si128(value, 4)));
            let c = arch::_mm_cvtepi32_2pdx_v2(arch::_mm_cvtepu8_epi32(arch::_mm_srli_si128(value, 8)));
            let d = arch::_mm_cvtepi32_2pdx_v2(arch::_mm_cvtepu8_epi32(arch::_mm_srli_si128(value, 12)));
            ArrayRegister([a[0], a[1], b[0], b[1], c[0], c[1], d[0], d[1]])
        }
    }
}
#[thermite_macros::inline_always]
impl CastRegister<ArrayRegister<super::F64x2V2, 8>> for super::I8x16V2 {
    fn cast_from(value: Storage<ArrayRegister<super::F64x2V2, 8>>) -> Storage<Self> {
        let v = value.0;
        unsafe {
            arch::_mm_cvt4epi32_epi8x_v2([
                arch::_mm_cvtt2pd_epi32x_v2([v[0], v[1]]),
                arch::_mm_cvtt2pd_epi32x_v2([v[2], v[3]]),
                arch::_mm_cvtt2pd_epi32x_v2([v[4], v[5]]),
                arch::_mm_cvtt2pd_epi32x_v2([v[6], v[7]]),
            ])
        }
    }
}
#[thermite_macros::inline_always]
impl CastRegister<ArrayRegister<super::F64x2V2, 8>> for super::U8x16V2 {
    fn cast_from(value: Storage<ArrayRegister<super::F64x2V2, 8>>) -> Storage<Self> {
        let v = value.0;
        unsafe {
            arch::_mm_cvt4epi32_epi8x_v2([
                arch::_mm_cvtt2pd_epi32x_v2([v[0], v[1]]),
                arch::_mm_cvtt2pd_epi32x_v2([v[2], v[3]]),
                arch::_mm_cvtt2pd_epi32x_v2([v[4], v[5]]),
                arch::_mm_cvtt2pd_epi32x_v2([v[6], v[7]]),
            ])
        }
    }
}

// ===========================================================================================
// Mask-side concat: the reduced register is its own Mask, and `FullConcatRegister` requires
// that Mask to concat from the half's Mask (`ArrayRegister<bool, 2>` for x4<-x2, the x4 mask
// for x8<-x4). A bool lane maps to a full-width (0xFF / 0x00) 8-bit mask lane.
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

impl_mask_concat_x4_from_bool2!(I8x4V2);
impl_mask_concat_x4_from_bool2!(U8x4V2);

// x8 mask <- x4 mask (the x4/x8 reduced registers are their own masks, sharing storage with the
// native register, so the x8<-x4 concat above already covers the mask side - same type).

// ===========================================================================================
// Gather/scatter for the reduced 8-bit registers (indexed by the native 4-/8-lane index types)
// falls back to scalar. Same-type IndexableRegister comes from the inner register's blanket; add
// the cross-type index markers.
// ===========================================================================================

macro_rules! impl_indexable8 {
    ($idx:ty => $($ty:ty),* $(,)?) => {$( impl IndexableRegister<$idx> for $ty {} )*};
}

impl_indexable8!(<super::super::X86V2 as crate::simd::Simd>::u32x4 => I8x4V2, U8x4V2);
impl_indexable8!(<super::super::X86V2 as crate::simd::Simd>::u64x4 => I8x4V2, U8x4V2);
impl_indexable8!(<super::super::X86V2 as crate::simd::Simd>::u32x8 => I8x8V2, U8x8V2);
impl_indexable8!(<super::super::X86V2 as crate::simd::Simd>::u64x8 => I8x8V2, U8x8V2);
