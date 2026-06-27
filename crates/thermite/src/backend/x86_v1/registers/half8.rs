//! Reduced (sub-native-width) 8-bit registers for x86-v1 (SSE2). Mirrors `half16.rs` one element
//! size down. SSE2 has no `pshufb` (SSSE3) or `pmovsx` (SSE4.1), so the widen casts sign/zero
//! extend via double `unpack` and the narrows use the store-and-rebuild pattern (the same approach
//! v1's 16-bit narrows use).

use generic_array::typenum::{U8, U12};

use super::arch;

use crate::register::{
    CastRegister, ConcatRegister, ExtendRegister, IndexableRegister, NumericRegister, Register, SaturatingCastRegister,
    Storage, array::ArrayRegister, reduced::ReducedRegister,
};

// --- saturating narrows into 8-bit ---

// Signed `i16 -> i8` via SSE2 `packsswb`.
#[thermite_macros::inline_always]
impl SaturatingCastRegister<super::I16x8V1> for I8x8V1 {
    fn saturating_cast_from(value: Storage<super::I16x8V1>) -> Storage<Self> {
        ReducedRegister::new(unsafe { arch::_mm_packs_epi16(value, value) })
    }
}
#[thermite_macros::inline_always]
impl SaturatingCastRegister<super::half16::I16x4V1> for I8x4V1 {
    fn saturating_cast_from(value: Storage<super::half16::I16x4V1>) -> Storage<Self> {
        ReducedRegister::new(unsafe { arch::_mm_packs_epi16(value.0, value.0) })
    }
}
// Unsigned `u16 -> u8`: clamp the high end (`min_epu16x_v1` polyfill) then SSE2 `packuswb`.
#[thermite_macros::inline_always]
impl SaturatingCastRegister<super::U16x8V1> for U8x8V1 {
    fn saturating_cast_from(value: Storage<super::U16x8V1>) -> Storage<Self> {
        let clamped = unsafe { arch::_mm_min_epu16x_v1(value, arch::_mm_set1_epi16(0xFF)) };
        ReducedRegister::new(unsafe { arch::_mm_packus_epi16(clamped, clamped) })
    }
}
#[thermite_macros::inline_always]
impl SaturatingCastRegister<super::half16::U16x4V1> for U8x4V1 {
    fn saturating_cast_from(value: Storage<super::half16::U16x4V1>) -> Storage<Self> {
        let clamped = unsafe { arch::_mm_min_epu16x_v1(value.0, arch::_mm_set1_epi16(0xFF)) };
        ReducedRegister::new(unsafe { arch::_mm_packus_epi16(clamped, clamped) })
    }
}

// Signed skip-level `i32 -> i8`: compose the two SSE2 packs (idempotent saturation).
#[thermite_macros::inline_always]
impl SaturatingCastRegister<super::I32x4V1> for I8x4V1 {
    fn saturating_cast_from(value: Storage<super::I32x4V1>) -> Storage<Self> {
        let w = <super::half16::I16x4V1 as SaturatingCastRegister<super::I32x4V1>>::saturating_cast_from(value);
        <Self as SaturatingCastRegister<super::half16::I16x4V1>>::saturating_cast_from(w)
    }
}
#[thermite_macros::inline_always]
impl SaturatingCastRegister<ArrayRegister<super::I32x4V1, 2>> for I8x8V1 {
    fn saturating_cast_from(value: Storage<ArrayRegister<super::I32x4V1, 2>>) -> Storage<Self> {
        let w = <super::I16x8V1 as SaturatingCastRegister<ArrayRegister<super::I32x4V1, 2>>>::saturating_cast_from(value);
        <Self as SaturatingCastRegister<super::I16x8V1>>::saturating_cast_from(w)
    }
}

// SSE2 has no `packusdw`/unsigned-32 min usable here, so every `u32 -> u8` and the `i64`/x2
// pairs clamp into range (the register `min`/`max` use the v1 polyfills) + truncating narrow.
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
    (super::U32x4V1, u32, U8x4V1, u8),
    (ArrayRegister<super::U32x4V1, 2>, u32, U8x8V1, u8),
    (ArrayRegister<super::I64x2V1, 2>, i64, I8x4V1, i8),
    (ArrayRegister<super::U64x2V1, 2>, u64, U8x4V1, u8),
    (ArrayRegister<super::I64x2V1, 4>, i64, I8x8V1, i8),
    (ArrayRegister<super::U64x2V1, 4>, u64, U8x8V1, u8),
    (super::half::I32x2V1, i32, ArrayRegister<i8, 2>, i8),
    (super::half::U32x2V1, u32, ArrayRegister<u8, 2>, u8),
    (super::I64x2V1, i64, ArrayRegister<i8, 2>, i8),
    (super::U64x2V1, u64, ArrayRegister<u8, 2>, u8),
}

// `ReducedRegister<R, N>` removes `N` lanes (lane count = R::Lanes - N); over the native 16-lane
// register the 4-lane form removes 12 and the 8-lane form removes 8.
/// 4-lane signed 8-bit register, backed by the low 4 lanes of a 128-bit `I8x16V1`.
pub type I8x4V1 = ReducedRegister<super::I8x16V1, U12>;
/// 4-lane unsigned 8-bit register.
pub type U8x4V1 = ReducedRegister<super::U8x16V1, U12>;
/// 8-lane signed 8-bit register, backed by the low 8 lanes of a 128-bit `I8x16V1`.
pub type I8x8V1 = ReducedRegister<super::I8x16V1, U8>;
/// 8-lane unsigned 8-bit register.
pub type U8x8V1 = ReducedRegister<super::U8x16V1, U8>;

#[inline(always)]
fn store_bytes(v: arch::__m128i) -> [i8; 16] {
    let mut arr = [0i8; 16];
    unsafe { arch::_mm_storeu_si128(arr.as_mut_ptr() as *mut _, v) };
    arr
}

#[inline(always)]
fn store_dwords(v: arch::__m128i) -> [i32; 4] {
    let mut arr = [0i32; 4];
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

impl_concat_x4_from_x2!(I8x4V1, i8);
impl_concat_x4_from_x2!(U8x4V1, u8);

// ===========================================================================================
// x8 <- x4 (both reduced over the same native register).
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

impl_concat_x8_from_x4!(I8x8V1, I8x4V1);
impl_concat_x8_from_x4!(U8x8V1, U8x4V1);

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

impl_concat_x16_from_x8!(super::I8x16V1, I8x8V1);
impl_concat_x16_from_x8!(super::U8x16V1, U8x8V1);

// ===========================================================================================
// 8 <-> 16 widen/narrow casts (SSE2: no pmovsx/pshufb).
//   widen 8 -> 16 via `unpacklo_epi8` against the sign (signed) or zero (unsigned) - low 8 bytes
//     become 8 sign/zero-extended i16.
//   narrow 16 -> 8 via store-and-rebuild (`_mm_setr_epi8` of the low byte of each i16).
//   x4: I8x4V1 <-> I16x4V1.  x8: I8x8V1 <-> I16x8V1 (native).  x16: I8x16V1 <-> Array<I16x8V1,2>.
// ===========================================================================================

// --- x4 ---
#[thermite_macros::inline_always]
impl CastRegister<I8x4V1> for super::half16::I16x4V1 {
    fn cast_from(value: Storage<I8x4V1>) -> Storage<Self> {
        ReducedRegister::new(unsafe { arch::_mm_cvtepi8_epi16x_v1(value.0) })
    }
}
#[thermite_macros::inline_always]
impl CastRegister<U8x4V1> for super::half16::U16x4V1 {
    fn cast_from(value: Storage<U8x4V1>) -> Storage<Self> {
        ReducedRegister::new(unsafe { arch::_mm_cvtepu8_epi16x_v1(value.0) })
    }
}
#[thermite_macros::inline_always]
impl CastRegister<super::half16::I16x4V1> for I8x4V1 {
    fn cast_from(value: Storage<super::half16::I16x4V1>) -> Storage<Self> {
        ReducedRegister::new(unsafe { arch::_mm_cvtepi16_epi8x_v1(value.0) })
    }
}
#[thermite_macros::inline_always]
impl CastRegister<super::half16::U16x4V1> for U8x4V1 {
    fn cast_from(value: Storage<super::half16::U16x4V1>) -> Storage<Self> {
        ReducedRegister::new(unsafe { arch::_mm_cvtepi16_epi8x_v1(value.0) })
    }
}

// --- x8 (i16x8 native) ---
#[thermite_macros::inline_always]
impl CastRegister<I8x8V1> for super::I16x8V1 {
    fn cast_from(value: Storage<I8x8V1>) -> Storage<Self> {
        unsafe { arch::_mm_cvtepi8_epi16x_v1(value.0) }
    }
}
#[thermite_macros::inline_always]
impl CastRegister<U8x8V1> for super::U16x8V1 {
    fn cast_from(value: Storage<U8x8V1>) -> Storage<Self> {
        unsafe { arch::_mm_cvtepu8_epi16x_v1(value.0) }
    }
}
#[thermite_macros::inline_always]
impl CastRegister<super::I16x8V1> for I8x8V1 {
    fn cast_from(value: Storage<super::I16x8V1>) -> Storage<Self> {
        ReducedRegister::new(unsafe { arch::_mm_cvtepi16_epi8x_v1(value) })
    }
}
#[thermite_macros::inline_always]
impl CastRegister<super::U16x8V1> for U8x8V1 {
    fn cast_from(value: Storage<super::U16x8V1>) -> Storage<Self> {
        ReducedRegister::new(unsafe { arch::_mm_cvtepi16_epi8x_v1(value) })
    }
}

// --- x16 (i16x16 = ArrayRegister<I16x8V1, 2>) ---
#[thermite_macros::inline_always]
impl CastRegister<super::I8x16V1> for ArrayRegister<super::I16x8V1, 2> {
    fn cast_from(value: Storage<super::I8x16V1>) -> Storage<Self> {
        unsafe {
            let lo = arch::_mm_cvtepi8_epi16x_v1(value);
            let hi = arch::_mm_cvtepi8_epi16x_v1(arch::_mm_srli_si128(value, 8));
            ArrayRegister([lo, hi])
        }
    }
}
#[thermite_macros::inline_always]
impl CastRegister<super::U8x16V1> for ArrayRegister<super::U16x8V1, 2> {
    fn cast_from(value: Storage<super::U8x16V1>) -> Storage<Self> {
        unsafe {
            let lo = arch::_mm_cvtepu8_epi16x_v1(value);
            let hi = arch::_mm_cvtepu8_epi16x_v1(arch::_mm_srli_si128(value, 8));
            ArrayRegister([lo, hi])
        }
    }
}
#[thermite_macros::inline_always]
impl CastRegister<ArrayRegister<super::I16x8V1, 2>> for super::I8x16V1 {
    fn cast_from(value: Storage<ArrayRegister<super::I16x8V1, 2>>) -> Storage<Self> {
        unsafe { arch::_mm_cvt2epi16_epi8x_v1(value.0) }
    }
}
#[thermite_macros::inline_always]
impl CastRegister<ArrayRegister<super::U16x8V1, 2>> for super::U8x16V1 {
    fn cast_from(value: Storage<ArrayRegister<super::U16x8V1, 2>>) -> Storage<Self> {
        unsafe { arch::_mm_cvt2epi16_epi8x_v1(value.0) }
    }
}

// ===========================================================================================
// Widen casts to 32-bit (SSE2: double-unpack sign/zero-extend; narrows store-and-rebuild).
//   x4: I8x4V1 <-> I32x4V1.  x8: I8x8V1 <-> ArrayRegister<I32x4V1, 2>.  x2: array <-> I32x2V1.
// ===========================================================================================

// --- x4 ---
#[thermite_macros::inline_always]
impl CastRegister<I8x4V1> for super::I32x4V1 {
    fn cast_from(value: Storage<I8x4V1>) -> Storage<Self> {
        unsafe { arch::_mm_cvtepi8_epi32x_v1(value.0) }
    }
}
#[thermite_macros::inline_always]
impl CastRegister<U8x4V1> for super::U32x4V1 {
    fn cast_from(value: Storage<U8x4V1>) -> Storage<Self> {
        unsafe { arch::_mm_cvtepu8_epi32x_v1(value.0) }
    }
}
#[thermite_macros::inline_always]
impl CastRegister<super::I32x4V1> for I8x4V1 {
    fn cast_from(value: Storage<super::I32x4V1>) -> Storage<Self> {
        let a = store_dwords(value);
        ReducedRegister::new(unsafe {
            arch::_mm_setr_epi8(
                a[0] as i8, a[1] as i8, a[2] as i8, a[3] as i8, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            )
        })
    }
}
#[thermite_macros::inline_always]
impl CastRegister<super::U32x4V1> for U8x4V1 {
    fn cast_from(value: Storage<super::U32x4V1>) -> Storage<Self> {
        let a = store_dwords(value);
        ReducedRegister::new(unsafe {
            arch::_mm_setr_epi8(
                a[0] as i8, a[1] as i8, a[2] as i8, a[3] as i8, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            )
        })
    }
}

// --- x8 (i32x8 = ArrayRegister<I32x4V1, 2>) ---
#[thermite_macros::inline_always]
impl CastRegister<I8x8V1> for ArrayRegister<super::I32x4V1, 2> {
    fn cast_from(value: Storage<I8x8V1>) -> Storage<Self> {
        unsafe {
            let lo = arch::_mm_cvtepi8_epi32x_v1(value.0);
            let hi = arch::_mm_cvtepi8_epi32x_v1(arch::_mm_srli_si128(value.0, 4));
            ArrayRegister([lo, hi])
        }
    }
}
#[thermite_macros::inline_always]
impl CastRegister<U8x8V1> for ArrayRegister<super::U32x4V1, 2> {
    fn cast_from(value: Storage<U8x8V1>) -> Storage<Self> {
        unsafe {
            let lo = arch::_mm_cvtepu8_epi32x_v1(value.0);
            let hi = arch::_mm_cvtepu8_epi32x_v1(arch::_mm_srli_si128(value.0, 4));
            ArrayRegister([lo, hi])
        }
    }
}
#[thermite_macros::inline_always]
impl CastRegister<ArrayRegister<super::I32x4V1, 2>> for I8x8V1 {
    fn cast_from(value: Storage<ArrayRegister<super::I32x4V1, 2>>) -> Storage<Self> {
        let lo = store_dwords(value.0[0]);
        let hi = store_dwords(value.0[1]);
        ReducedRegister::new(unsafe {
            arch::_mm_setr_epi8(
                lo[0] as i8, lo[1] as i8, lo[2] as i8, lo[3] as i8,
                hi[0] as i8, hi[1] as i8, hi[2] as i8, hi[3] as i8,
                0, 0, 0, 0, 0, 0, 0, 0,
            )
        })
    }
}
#[thermite_macros::inline_always]
impl CastRegister<ArrayRegister<super::U32x4V1, 2>> for U8x8V1 {
    fn cast_from(value: Storage<ArrayRegister<super::U32x4V1, 2>>) -> Storage<Self> {
        let lo = store_dwords(value.0[0]);
        let hi = store_dwords(value.0[1]);
        ReducedRegister::new(unsafe {
            arch::_mm_setr_epi8(
                lo[0] as i8, lo[1] as i8, lo[2] as i8, lo[3] as i8,
                hi[0] as i8, hi[1] as i8, hi[2] as i8, hi[3] as i8,
                0, 0, 0, 0, 0, 0, 0, 0,
            )
        })
    }
}

// --- x16 widen i8 -> i32 (native 16 bytes -> ArrayRegister<I32x4V1, 4>), SSE2 unpack ---
#[thermite_macros::inline_always]
impl CastRegister<super::I8x16V1> for ArrayRegister<super::I32x4V1, 4> {
    fn cast_from(value: Storage<super::I8x16V1>) -> Storage<Self> {
        ArrayRegister(unsafe { arch::_mm_cvtepi8_4epi32x_v1(value) })
    }
}
#[thermite_macros::inline_always]
impl CastRegister<super::U8x16V1> for ArrayRegister<super::U32x4V1, 4> {
    fn cast_from(value: Storage<super::U8x16V1>) -> Storage<Self> {
        ArrayRegister(unsafe { arch::_mm_cvtepu8_4epi32x_v1(value) })
    }
}

// --- x16 narrow i32 -> i8 (store all 4 lanes, rebuild 16 bytes; no pshufb on SSE2) ---
#[thermite_macros::inline_always]
impl CastRegister<ArrayRegister<super::I32x4V1, 4>> for super::I8x16V1 {
    fn cast_from(value: Storage<ArrayRegister<super::I32x4V1, 4>>) -> Storage<Self> {
        unsafe { arch::_mm_cvt4epi32_epi8x_v1(value.0) }
    }
}
#[thermite_macros::inline_always]
impl CastRegister<ArrayRegister<super::U32x4V1, 4>> for super::U8x16V1 {
    fn cast_from(value: Storage<ArrayRegister<super::U32x4V1, 4>>) -> Storage<Self> {
        unsafe { arch::_mm_cvt4epi32_epi8x_v1(value.0) }
    }
}

// --- x2 (ArrayRegister<i8,2> <-> I32x2V1 reduced) ---
#[thermite_macros::inline_always]
impl CastRegister<ArrayRegister<i8, 2>> for super::half::I32x2V1 {
    fn cast_from(value: Storage<ArrayRegister<i8, 2>>) -> Storage<Self> {
        ReducedRegister::new(unsafe { arch::_mm_setr_epi32(value.0[0] as i32, value.0[1] as i32, 0, 0) })
    }
}
#[thermite_macros::inline_always]
impl CastRegister<ArrayRegister<u8, 2>> for super::half::U32x2V1 {
    fn cast_from(value: Storage<ArrayRegister<u8, 2>>) -> Storage<Self> {
        ReducedRegister::new(unsafe { arch::_mm_setr_epi32(value.0[0] as i32, value.0[1] as i32, 0, 0) })
    }
}
#[thermite_macros::inline_always]
impl CastRegister<super::half::I32x2V1> for ArrayRegister<i8, 2> {
    fn cast_from(value: Storage<super::half::I32x2V1>) -> Storage<Self> {
        let a = store_dwords(value.0);
        ArrayRegister([a[0] as i8, a[1] as i8])
    }
}
#[thermite_macros::inline_always]
impl CastRegister<super::half::U32x2V1> for ArrayRegister<u8, 2> {
    fn cast_from(value: Storage<super::half::U32x2V1>) -> Storage<Self> {
        let a = store_dwords(value.0);
        ArrayRegister([a[0] as u8, a[1] as u8])
    }
}

// ===========================================================================================
// Widen/narrow casts to 64-bit (SSE2: store-and-rebuild widen/narrow; no pmovsx/pshufb).
//   widen 8 -> 64 via store-then-`_mm_set_epi64x` (sign/zero-extend each byte to i64).
//   narrow 64 -> 8 via store-then-`_mm_setr_epi8` (low byte of each i64, wrapping like `as`).
//   x2:  ArrayRegister<i8,2> <-> I64x2V1 (native 2-lane).
//   x4:  I8x4V1 <-> ArrayRegister<I64x2V1, 2> (the v1 i64x4).
//   x8:  I8x8V1 <-> ArrayRegister<I64x2V1, 4> (the v1 i64x8).
//   x16: I8x16V1 <-> ArrayRegister<I64x2V1, 8> (the v1 i64x16).
// ===========================================================================================

#[inline(always)]
fn store_qwords(v: arch::__m128i) -> [i64; 2] {
    let mut arr = [0i64; 2];
    unsafe { arch::_mm_storeu_si128(arr.as_mut_ptr() as *mut _, v) };
    arr
}

// --- x2 widen i8 -> i64 (ArrayRegister<i8,2> -> I64x2V1 native) ---
#[thermite_macros::inline_always]
impl CastRegister<ArrayRegister<i8, 2>> for super::I64x2V1 {
    fn cast_from(value: Storage<ArrayRegister<i8, 2>>) -> Storage<Self> {
        unsafe { arch::_mm_cvt2epi8_epi64x_v1(value.0[0], value.0[1]) }
    }
}
#[thermite_macros::inline_always]
impl CastRegister<ArrayRegister<u8, 2>> for super::U64x2V1 {
    fn cast_from(value: Storage<ArrayRegister<u8, 2>>) -> Storage<Self> {
        unsafe { arch::_mm_cvt2epu8_epi64x_v1(value.0[0], value.0[1]) }
    }
}

// --- x2 narrow i64 -> i8 (I64x2V1 native -> ArrayRegister<i8,2>) ---
#[thermite_macros::inline_always]
impl CastRegister<super::I64x2V1> for ArrayRegister<i8, 2> {
    fn cast_from(value: Storage<super::I64x2V1>) -> Storage<Self> {
        let a = store_qwords(value);
        ArrayRegister([a[0] as i8, a[1] as i8])
    }
}
#[thermite_macros::inline_always]
impl CastRegister<super::U64x2V1> for ArrayRegister<u8, 2> {
    fn cast_from(value: Storage<super::U64x2V1>) -> Storage<Self> {
        let a = store_qwords(value);
        ArrayRegister([a[0] as u8, a[1] as u8])
    }
}

// --- x4 widen i8 -> i64 (low 4 bytes -> two 2x i64 lanes) ---
#[thermite_macros::inline_always]
impl CastRegister<I8x4V1> for ArrayRegister<super::I64x2V1, 2> {
    fn cast_from(value: Storage<I8x4V1>) -> Storage<Self> {
        let a = store_bytes(value.0);
        ArrayRegister(unsafe {
            [
                arch::_mm_cvt2epi8_epi64x_v1(a[0], a[1]),
                arch::_mm_cvt2epi8_epi64x_v1(a[2], a[3]),
            ]
        })
    }
}
#[thermite_macros::inline_always]
impl CastRegister<U8x4V1> for ArrayRegister<super::U64x2V1, 2> {
    fn cast_from(value: Storage<U8x4V1>) -> Storage<Self> {
        let a = store_bytes(value.0);
        ArrayRegister(unsafe {
            [
                arch::_mm_cvt2epu8_epi64x_v1(a[0] as u8, a[1] as u8),
                arch::_mm_cvt2epu8_epi64x_v1(a[2] as u8, a[3] as u8),
            ]
        })
    }
}

// --- x4 narrow i64 -> i8 (byte 0 of each of 4 lanes -> low 4 bytes) ---
#[thermite_macros::inline_always]
impl CastRegister<ArrayRegister<super::I64x2V1, 2>> for I8x4V1 {
    fn cast_from(value: Storage<ArrayRegister<super::I64x2V1, 2>>) -> Storage<Self> {
        let lo = store_qwords(value.0[0]);
        let hi = store_qwords(value.0[1]);
        ReducedRegister::new(unsafe {
            arch::_mm_setr_epi8(
                lo[0] as i8, lo[1] as i8, hi[0] as i8, hi[1] as i8,
                0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            )
        })
    }
}
#[thermite_macros::inline_always]
impl CastRegister<ArrayRegister<super::U64x2V1, 2>> for U8x4V1 {
    fn cast_from(value: Storage<ArrayRegister<super::U64x2V1, 2>>) -> Storage<Self> {
        let lo = store_qwords(value.0[0]);
        let hi = store_qwords(value.0[1]);
        ReducedRegister::new(unsafe {
            arch::_mm_setr_epi8(
                lo[0] as i8, lo[1] as i8, hi[0] as i8, hi[1] as i8,
                0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            )
        })
    }
}

// --- x8 widen i8 -> i64 (low 8 bytes -> four 2x i64 lanes) ---
#[thermite_macros::inline_always]
impl CastRegister<I8x8V1> for ArrayRegister<super::I64x2V1, 4> {
    fn cast_from(value: Storage<I8x8V1>) -> Storage<Self> {
        let a = store_bytes(value.0);
        ArrayRegister(unsafe {
            [
                arch::_mm_cvt2epi8_epi64x_v1(a[0], a[1]),
                arch::_mm_cvt2epi8_epi64x_v1(a[2], a[3]),
                arch::_mm_cvt2epi8_epi64x_v1(a[4], a[5]),
                arch::_mm_cvt2epi8_epi64x_v1(a[6], a[7]),
            ]
        })
    }
}
#[thermite_macros::inline_always]
impl CastRegister<U8x8V1> for ArrayRegister<super::U64x2V1, 4> {
    fn cast_from(value: Storage<U8x8V1>) -> Storage<Self> {
        let a = store_bytes(value.0);
        ArrayRegister(unsafe {
            [
                arch::_mm_cvt2epu8_epi64x_v1(a[0] as u8, a[1] as u8),
                arch::_mm_cvt2epu8_epi64x_v1(a[2] as u8, a[3] as u8),
                arch::_mm_cvt2epu8_epi64x_v1(a[4] as u8, a[5] as u8),
                arch::_mm_cvt2epu8_epi64x_v1(a[6] as u8, a[7] as u8),
            ]
        })
    }
}

// --- x8 narrow i64 -> i8 (byte 0 of each of 8 lanes -> low 8 bytes) ---
#[thermite_macros::inline_always]
impl CastRegister<ArrayRegister<super::I64x2V1, 4>> for I8x8V1 {
    fn cast_from(value: Storage<ArrayRegister<super::I64x2V1, 4>>) -> Storage<Self> {
        ReducedRegister::new(unsafe { arch::_mm_cvt4epi64_epi8x_v1(value.0) })
    }
}
#[thermite_macros::inline_always]
impl CastRegister<ArrayRegister<super::U64x2V1, 4>> for U8x8V1 {
    fn cast_from(value: Storage<ArrayRegister<super::U64x2V1, 4>>) -> Storage<Self> {
        ReducedRegister::new(unsafe { arch::_mm_cvt4epi64_epi8x_v1(value.0) })
    }
}

// --- x16 widen i8 -> i64 (native 16 bytes -> ArrayRegister<I64x2V1, 8>) ---
#[thermite_macros::inline_always]
impl CastRegister<super::I8x16V1> for ArrayRegister<super::I64x2V1, 8> {
    fn cast_from(value: Storage<super::I8x16V1>) -> Storage<Self> {
        let a = store_bytes(value);
        ArrayRegister(unsafe {
            [
                arch::_mm_cvt2epi8_epi64x_v1(a[0], a[1]),
                arch::_mm_cvt2epi8_epi64x_v1(a[2], a[3]),
                arch::_mm_cvt2epi8_epi64x_v1(a[4], a[5]),
                arch::_mm_cvt2epi8_epi64x_v1(a[6], a[7]),
                arch::_mm_cvt2epi8_epi64x_v1(a[8], a[9]),
                arch::_mm_cvt2epi8_epi64x_v1(a[10], a[11]),
                arch::_mm_cvt2epi8_epi64x_v1(a[12], a[13]),
                arch::_mm_cvt2epi8_epi64x_v1(a[14], a[15]),
            ]
        })
    }
}
#[thermite_macros::inline_always]
impl CastRegister<super::U8x16V1> for ArrayRegister<super::U64x2V1, 8> {
    fn cast_from(value: Storage<super::U8x16V1>) -> Storage<Self> {
        let a = store_bytes(value);
        ArrayRegister(unsafe {
            [
                arch::_mm_cvt2epu8_epi64x_v1(a[0] as u8, a[1] as u8),
                arch::_mm_cvt2epu8_epi64x_v1(a[2] as u8, a[3] as u8),
                arch::_mm_cvt2epu8_epi64x_v1(a[4] as u8, a[5] as u8),
                arch::_mm_cvt2epu8_epi64x_v1(a[6] as u8, a[7] as u8),
                arch::_mm_cvt2epu8_epi64x_v1(a[8] as u8, a[9] as u8),
                arch::_mm_cvt2epu8_epi64x_v1(a[10] as u8, a[11] as u8),
                arch::_mm_cvt2epu8_epi64x_v1(a[12] as u8, a[13] as u8),
                arch::_mm_cvt2epu8_epi64x_v1(a[14] as u8, a[15] as u8),
            ]
        })
    }
}

// --- x16 narrow i64 -> i8 (ArrayRegister<I64x2V1, 8> -> native 16 bytes) ---
#[thermite_macros::inline_always]
impl CastRegister<ArrayRegister<super::I64x2V1, 8>> for super::I8x16V1 {
    fn cast_from(value: Storage<ArrayRegister<super::I64x2V1, 8>>) -> Storage<Self> {
        unsafe { arch::_mm_cvt8epi64_epi8x_v1(value.0) }
    }
}
#[thermite_macros::inline_always]
impl CastRegister<ArrayRegister<super::U64x2V1, 8>> for super::U8x16V1 {
    fn cast_from(value: Storage<ArrayRegister<super::U64x2V1, 8>>) -> Storage<Self> {
        unsafe { arch::_mm_cvt8epi64_epi8x_v1(value.0) }
    }
}

// ===========================================================================================
// 8 <-> f32/f64 direct casts (SSE2: no pmovsx/pshufb; widen via unpack chains to i32, then the
// native i32<->float converts; narrow via truncating float->i32 then the existing store-rebuild
// byte narrows).
//   WIDEN  i8/u8 -> f32 = widen_i8/u8_to_i32 then `_mm_cvtepi32_ps` (value-preserving, exact).
//   NARROW f32 -> i8/u8 = `_mm_cvttps_epi32` (truncate) then the store-rebuild byte narrow.
//   WIDEN  i8/u8 -> f64 = widen to i32 then `_mm_cvtepi32_pd` (2 lanes per F64x2V1, fan out).
//   NARROW f64 -> i8/u8 = `_mm_cvttpd_epi32` (truncate, low 2 lanes) then store-rebuild bytes.
// Unsigned 8-bit values fit in positive i32, so the signed i32->float convert is exact for them.
// ===========================================================================================

// --- x2 (ArrayRegister<i8,2> <-> ReducedRegister<F32x4V1,2> = F32x2V1) ---
#[thermite_macros::inline_always]
impl CastRegister<ArrayRegister<i8, 2>> for super::half::F32x2V1 {
    fn cast_from(value: Storage<ArrayRegister<i8, 2>>) -> Storage<Self> {
        let ints = unsafe { arch::_mm_setr_epi32(value.0[0] as i32, value.0[1] as i32, 0, 0) };
        ReducedRegister::new(unsafe { arch::_mm_cvtepi32_ps(ints) })
    }
}
#[thermite_macros::inline_always]
impl CastRegister<ArrayRegister<u8, 2>> for super::half::F32x2V1 {
    fn cast_from(value: Storage<ArrayRegister<u8, 2>>) -> Storage<Self> {
        let ints = unsafe { arch::_mm_setr_epi32(value.0[0] as i32, value.0[1] as i32, 0, 0) };
        ReducedRegister::new(unsafe { arch::_mm_cvtepi32_ps(ints) })
    }
}
#[thermite_macros::inline_always]
impl CastRegister<super::half::F32x2V1> for ArrayRegister<i8, 2> {
    fn cast_from(value: Storage<super::half::F32x2V1>) -> Storage<Self> {
        let a = store_dwords(unsafe { arch::_mm_cvttps_epi32(value.0) });
        ArrayRegister([a[0] as i8, a[1] as i8])
    }
}
#[thermite_macros::inline_always]
impl CastRegister<super::half::F32x2V1> for ArrayRegister<u8, 2> {
    fn cast_from(value: Storage<super::half::F32x2V1>) -> Storage<Self> {
        let a = store_dwords(unsafe { arch::_mm_cvttps_epi32(value.0) });
        ArrayRegister([a[0] as u8, a[1] as u8])
    }
}

// --- x2 (ArrayRegister<i8,2> <-> F64x2V1 native) ---
#[thermite_macros::inline_always]
impl CastRegister<ArrayRegister<i8, 2>> for super::F64x2V1 {
    fn cast_from(value: Storage<ArrayRegister<i8, 2>>) -> Storage<Self> {
        let ints = unsafe { arch::_mm_setr_epi32(value.0[0] as i32, value.0[1] as i32, 0, 0) };
        unsafe { arch::_mm_cvtepi32_pd(ints) }
    }
}
#[thermite_macros::inline_always]
impl CastRegister<ArrayRegister<u8, 2>> for super::F64x2V1 {
    fn cast_from(value: Storage<ArrayRegister<u8, 2>>) -> Storage<Self> {
        let ints = unsafe { arch::_mm_setr_epi32(value.0[0] as i32, value.0[1] as i32, 0, 0) };
        unsafe { arch::_mm_cvtepi32_pd(ints) }
    }
}
#[thermite_macros::inline_always]
impl CastRegister<super::F64x2V1> for ArrayRegister<i8, 2> {
    fn cast_from(value: Storage<super::F64x2V1>) -> Storage<Self> {
        let a = unsafe { arch::_mm_cvttpd_2i32x_v1(value) };
        ArrayRegister([a[0] as i8, a[1] as i8])
    }
}
#[thermite_macros::inline_always]
impl CastRegister<super::F64x2V1> for ArrayRegister<u8, 2> {
    fn cast_from(value: Storage<super::F64x2V1>) -> Storage<Self> {
        let a = unsafe { arch::_mm_cvttpd_2i32x_v1(value) };
        ArrayRegister([a[0] as u8, a[1] as u8])
    }
}

// --- x4 (I8x4V1 <-> F32x4V1 native) ---
#[thermite_macros::inline_always]
impl CastRegister<I8x4V1> for super::F32x4V1 {
    fn cast_from(value: Storage<I8x4V1>) -> Storage<Self> {
        unsafe { arch::_mm_cvtepi32_ps(arch::_mm_cvtepi8_epi32x_v1(value.0)) }
    }
}
#[thermite_macros::inline_always]
impl CastRegister<U8x4V1> for super::F32x4V1 {
    fn cast_from(value: Storage<U8x4V1>) -> Storage<Self> {
        unsafe { arch::_mm_cvtepi32_ps(arch::_mm_cvtepu8_epi32x_v1(value.0)) }
    }
}
#[thermite_macros::inline_always]
impl CastRegister<super::F32x4V1> for I8x4V1 {
    fn cast_from(value: Storage<super::F32x4V1>) -> Storage<Self> {
        let a = store_dwords(unsafe { arch::_mm_cvttps_epi32(value) });
        ReducedRegister::new(unsafe {
            arch::_mm_setr_epi8(
                a[0] as i8, a[1] as i8, a[2] as i8, a[3] as i8, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            )
        })
    }
}
#[thermite_macros::inline_always]
impl CastRegister<super::F32x4V1> for U8x4V1 {
    fn cast_from(value: Storage<super::F32x4V1>) -> Storage<Self> {
        let a = store_dwords(unsafe { arch::_mm_cvttps_epi32(value) });
        ReducedRegister::new(unsafe {
            arch::_mm_setr_epi8(
                a[0] as i8, a[1] as i8, a[2] as i8, a[3] as i8, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            )
        })
    }
}

// --- x4 (I8x4V1 <-> ArrayRegister<F64x2V1, 2>) ---
#[thermite_macros::inline_always]
impl CastRegister<I8x4V1> for ArrayRegister<super::F64x2V1, 2> {
    fn cast_from(value: Storage<I8x4V1>) -> Storage<Self> {
        unsafe {
            let ints = arch::_mm_cvtepi8_epi32x_v1(value.0);
            ArrayRegister(arch::_mm_cvtepi32_2pdx_v1(ints))
        }
    }
}
#[thermite_macros::inline_always]
impl CastRegister<U8x4V1> for ArrayRegister<super::F64x2V1, 2> {
    fn cast_from(value: Storage<U8x4V1>) -> Storage<Self> {
        unsafe {
            let ints = arch::_mm_cvtepu8_epi32x_v1(value.0);
            ArrayRegister(arch::_mm_cvtepi32_2pdx_v1(ints))
        }
    }
}
#[thermite_macros::inline_always]
impl CastRegister<ArrayRegister<super::F64x2V1, 2>> for I8x4V1 {
    fn cast_from(value: Storage<ArrayRegister<super::F64x2V1, 2>>) -> Storage<Self> {
        unsafe {
            let lo = arch::_mm_cvttpd_2i32x_v1(value.0[0]);
            let hi = arch::_mm_cvttpd_2i32x_v1(value.0[1]);
            ReducedRegister::new(arch::_mm_setr_epi8(
                lo[0] as i8, lo[1] as i8, hi[0] as i8, hi[1] as i8,
                0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            ))
        }
    }
}
#[thermite_macros::inline_always]
impl CastRegister<ArrayRegister<super::F64x2V1, 2>> for U8x4V1 {
    fn cast_from(value: Storage<ArrayRegister<super::F64x2V1, 2>>) -> Storage<Self> {
        unsafe {
            let lo = arch::_mm_cvttpd_2i32x_v1(value.0[0]);
            let hi = arch::_mm_cvttpd_2i32x_v1(value.0[1]);
            ReducedRegister::new(arch::_mm_setr_epi8(
                lo[0] as i8, lo[1] as i8, hi[0] as i8, hi[1] as i8,
                0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            ))
        }
    }
}

// --- x8 (I8x8V1 <-> ArrayRegister<F32x4V1, 2>) ---
#[thermite_macros::inline_always]
impl CastRegister<I8x8V1> for ArrayRegister<super::F32x4V1, 2> {
    fn cast_from(value: Storage<I8x8V1>) -> Storage<Self> {
        unsafe {
            let lo = arch::_mm_cvtepi8_epi32x_v1(value.0);
            let hi = arch::_mm_cvtepi8_epi32x_v1(arch::_mm_srli_si128(value.0, 4));
            ArrayRegister([arch::_mm_cvtepi32_ps(lo), arch::_mm_cvtepi32_ps(hi)])
        }
    }
}
#[thermite_macros::inline_always]
impl CastRegister<U8x8V1> for ArrayRegister<super::F32x4V1, 2> {
    fn cast_from(value: Storage<U8x8V1>) -> Storage<Self> {
        unsafe {
            let lo = arch::_mm_cvtepu8_epi32x_v1(value.0);
            let hi = arch::_mm_cvtepu8_epi32x_v1(arch::_mm_srli_si128(value.0, 4));
            ArrayRegister([arch::_mm_cvtepi32_ps(lo), arch::_mm_cvtepi32_ps(hi)])
        }
    }
}
#[thermite_macros::inline_always]
impl CastRegister<ArrayRegister<super::F32x4V1, 2>> for I8x8V1 {
    fn cast_from(value: Storage<ArrayRegister<super::F32x4V1, 2>>) -> Storage<Self> {
        let lo = store_dwords(unsafe { arch::_mm_cvttps_epi32(value.0[0]) });
        let hi = store_dwords(unsafe { arch::_mm_cvttps_epi32(value.0[1]) });
        ReducedRegister::new(unsafe {
            arch::_mm_setr_epi8(
                lo[0] as i8, lo[1] as i8, lo[2] as i8, lo[3] as i8,
                hi[0] as i8, hi[1] as i8, hi[2] as i8, hi[3] as i8,
                0, 0, 0, 0, 0, 0, 0, 0,
            )
        })
    }
}
#[thermite_macros::inline_always]
impl CastRegister<ArrayRegister<super::F32x4V1, 2>> for U8x8V1 {
    fn cast_from(value: Storage<ArrayRegister<super::F32x4V1, 2>>) -> Storage<Self> {
        let lo = store_dwords(unsafe { arch::_mm_cvttps_epi32(value.0[0]) });
        let hi = store_dwords(unsafe { arch::_mm_cvttps_epi32(value.0[1]) });
        ReducedRegister::new(unsafe {
            arch::_mm_setr_epi8(
                lo[0] as i8, lo[1] as i8, lo[2] as i8, lo[3] as i8,
                hi[0] as i8, hi[1] as i8, hi[2] as i8, hi[3] as i8,
                0, 0, 0, 0, 0, 0, 0, 0,
            )
        })
    }
}

// --- x8 (I8x8V1 <-> ArrayRegister<F64x2V1, 4>) ---
#[thermite_macros::inline_always]
impl CastRegister<I8x8V1> for ArrayRegister<super::F64x2V1, 4> {
    fn cast_from(value: Storage<I8x8V1>) -> Storage<Self> {
        unsafe {
            let lo = arch::_mm_cvtepi32_2pdx_v1(arch::_mm_cvtepi8_epi32x_v1(value.0));
            let hi = arch::_mm_cvtepi32_2pdx_v1(arch::_mm_cvtepi8_epi32x_v1(arch::_mm_srli_si128(value.0, 4)));
            ArrayRegister([lo[0], lo[1], hi[0], hi[1]])
        }
    }
}
#[thermite_macros::inline_always]
impl CastRegister<U8x8V1> for ArrayRegister<super::F64x2V1, 4> {
    fn cast_from(value: Storage<U8x8V1>) -> Storage<Self> {
        unsafe {
            let lo = arch::_mm_cvtepi32_2pdx_v1(arch::_mm_cvtepu8_epi32x_v1(value.0));
            let hi = arch::_mm_cvtepi32_2pdx_v1(arch::_mm_cvtepu8_epi32x_v1(arch::_mm_srli_si128(value.0, 4)));
            ArrayRegister([lo[0], lo[1], hi[0], hi[1]])
        }
    }
}
#[thermite_macros::inline_always]
impl CastRegister<ArrayRegister<super::F64x2V1, 4>> for I8x8V1 {
    fn cast_from(value: Storage<ArrayRegister<super::F64x2V1, 4>>) -> Storage<Self> {
        unsafe {
            let a = arch::_mm_cvttpd_2i32x_v1(value.0[0]);
            let b = arch::_mm_cvttpd_2i32x_v1(value.0[1]);
            let c = arch::_mm_cvttpd_2i32x_v1(value.0[2]);
            let d = arch::_mm_cvttpd_2i32x_v1(value.0[3]);
            ReducedRegister::new(arch::_mm_setr_epi8(
                a[0] as i8, a[1] as i8, b[0] as i8, b[1] as i8,
                c[0] as i8, c[1] as i8, d[0] as i8, d[1] as i8,
                0, 0, 0, 0, 0, 0, 0, 0,
            ))
        }
    }
}
#[thermite_macros::inline_always]
impl CastRegister<ArrayRegister<super::F64x2V1, 4>> for U8x8V1 {
    fn cast_from(value: Storage<ArrayRegister<super::F64x2V1, 4>>) -> Storage<Self> {
        unsafe {
            let a = arch::_mm_cvttpd_2i32x_v1(value.0[0]);
            let b = arch::_mm_cvttpd_2i32x_v1(value.0[1]);
            let c = arch::_mm_cvttpd_2i32x_v1(value.0[2]);
            let d = arch::_mm_cvttpd_2i32x_v1(value.0[3]);
            ReducedRegister::new(arch::_mm_setr_epi8(
                a[0] as i8, a[1] as i8, b[0] as i8, b[1] as i8,
                c[0] as i8, c[1] as i8, d[0] as i8, d[1] as i8,
                0, 0, 0, 0, 0, 0, 0, 0,
            ))
        }
    }
}

// --- x16 (I8x16V1 native <-> ArrayRegister<F32x4V1, 4>) ---
#[thermite_macros::inline_always]
impl CastRegister<super::I8x16V1> for ArrayRegister<super::F32x4V1, 4> {
    fn cast_from(value: Storage<super::I8x16V1>) -> Storage<Self> {
        unsafe {
            let q = arch::_mm_cvtepi8_4epi32x_v1(value);
            ArrayRegister([
                arch::_mm_cvtepi32_ps(q[0]),
                arch::_mm_cvtepi32_ps(q[1]),
                arch::_mm_cvtepi32_ps(q[2]),
                arch::_mm_cvtepi32_ps(q[3]),
            ])
        }
    }
}
#[thermite_macros::inline_always]
impl CastRegister<super::U8x16V1> for ArrayRegister<super::F32x4V1, 4> {
    fn cast_from(value: Storage<super::U8x16V1>) -> Storage<Self> {
        unsafe {
            let q = arch::_mm_cvtepu8_4epi32x_v1(value);
            ArrayRegister([
                arch::_mm_cvtepi32_ps(q[0]),
                arch::_mm_cvtepi32_ps(q[1]),
                arch::_mm_cvtepi32_ps(q[2]),
                arch::_mm_cvtepi32_ps(q[3]),
            ])
        }
    }
}
#[thermite_macros::inline_always]
impl CastRegister<ArrayRegister<super::F32x4V1, 4>> for super::I8x16V1 {
    fn cast_from(value: Storage<ArrayRegister<super::F32x4V1, 4>>) -> Storage<Self> {
        let v = value.0;
        unsafe {
            arch::_mm_cvt4epi32_epi8x_v1([
                arch::_mm_cvttps_epi32(v[0]),
                arch::_mm_cvttps_epi32(v[1]),
                arch::_mm_cvttps_epi32(v[2]),
                arch::_mm_cvttps_epi32(v[3]),
            ])
        }
    }
}
#[thermite_macros::inline_always]
impl CastRegister<ArrayRegister<super::F32x4V1, 4>> for super::U8x16V1 {
    fn cast_from(value: Storage<ArrayRegister<super::F32x4V1, 4>>) -> Storage<Self> {
        let v = value.0;
        unsafe {
            arch::_mm_cvt4epi32_epi8x_v1([
                arch::_mm_cvttps_epi32(v[0]),
                arch::_mm_cvttps_epi32(v[1]),
                arch::_mm_cvttps_epi32(v[2]),
                arch::_mm_cvttps_epi32(v[3]),
            ])
        }
    }
}

// --- x16 (I8x16V1 native <-> ArrayRegister<F64x2V1, 8>) ---
#[thermite_macros::inline_always]
impl CastRegister<super::I8x16V1> for ArrayRegister<super::F64x2V1, 8> {
    fn cast_from(value: Storage<super::I8x16V1>) -> Storage<Self> {
        unsafe {
            let q = arch::_mm_cvtepi8_4epi32x_v1(value);
            let a = arch::_mm_cvtepi32_2pdx_v1(q[0]);
            let b = arch::_mm_cvtepi32_2pdx_v1(q[1]);
            let c = arch::_mm_cvtepi32_2pdx_v1(q[2]);
            let d = arch::_mm_cvtepi32_2pdx_v1(q[3]);
            ArrayRegister([a[0], a[1], b[0], b[1], c[0], c[1], d[0], d[1]])
        }
    }
}
#[thermite_macros::inline_always]
impl CastRegister<super::U8x16V1> for ArrayRegister<super::F64x2V1, 8> {
    fn cast_from(value: Storage<super::U8x16V1>) -> Storage<Self> {
        unsafe {
            let q = arch::_mm_cvtepu8_4epi32x_v1(value);
            let a = arch::_mm_cvtepi32_2pdx_v1(q[0]);
            let b = arch::_mm_cvtepi32_2pdx_v1(q[1]);
            let c = arch::_mm_cvtepi32_2pdx_v1(q[2]);
            let d = arch::_mm_cvtepi32_2pdx_v1(q[3]);
            ArrayRegister([a[0], a[1], b[0], b[1], c[0], c[1], d[0], d[1]])
        }
    }
}
#[thermite_macros::inline_always]
impl CastRegister<ArrayRegister<super::F64x2V1, 8>> for super::I8x16V1 {
    fn cast_from(value: Storage<ArrayRegister<super::F64x2V1, 8>>) -> Storage<Self> {
        let v = value.0;
        unsafe {
            let a = arch::_mm_cvttpd_2i32x_v1(v[0]);
            let b = arch::_mm_cvttpd_2i32x_v1(v[1]);
            let c = arch::_mm_cvttpd_2i32x_v1(v[2]);
            let d = arch::_mm_cvttpd_2i32x_v1(v[3]);
            let e = arch::_mm_cvttpd_2i32x_v1(v[4]);
            let f = arch::_mm_cvttpd_2i32x_v1(v[5]);
            let g = arch::_mm_cvttpd_2i32x_v1(v[6]);
            let h = arch::_mm_cvttpd_2i32x_v1(v[7]);
            arch::_mm_setr_epi8(
                a[0] as i8, a[1] as i8, b[0] as i8, b[1] as i8,
                c[0] as i8, c[1] as i8, d[0] as i8, d[1] as i8,
                e[0] as i8, e[1] as i8, f[0] as i8, f[1] as i8,
                g[0] as i8, g[1] as i8, h[0] as i8, h[1] as i8,
            )
        }
    }
}
#[thermite_macros::inline_always]
impl CastRegister<ArrayRegister<super::F64x2V1, 8>> for super::U8x16V1 {
    fn cast_from(value: Storage<ArrayRegister<super::F64x2V1, 8>>) -> Storage<Self> {
        let v = value.0;
        unsafe {
            let a = arch::_mm_cvttpd_2i32x_v1(v[0]);
            let b = arch::_mm_cvttpd_2i32x_v1(v[1]);
            let c = arch::_mm_cvttpd_2i32x_v1(v[2]);
            let d = arch::_mm_cvttpd_2i32x_v1(v[3]);
            let e = arch::_mm_cvttpd_2i32x_v1(v[4]);
            let f = arch::_mm_cvttpd_2i32x_v1(v[5]);
            let g = arch::_mm_cvttpd_2i32x_v1(v[6]);
            let h = arch::_mm_cvttpd_2i32x_v1(v[7]);
            arch::_mm_setr_epi8(
                a[0] as i8, a[1] as i8, b[0] as i8, b[1] as i8,
                c[0] as i8, c[1] as i8, d[0] as i8, d[1] as i8,
                e[0] as i8, e[1] as i8, f[0] as i8, f[1] as i8,
                g[0] as i8, g[1] as i8, h[0] as i8, h[1] as i8,
            )
        }
    }
}

// ===========================================================================================
// Mask-side concat (x4 <- bool2).
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

impl_mask_concat_x4_from_bool2!(I8x4V1);
impl_mask_concat_x4_from_bool2!(U8x4V1);

// ===========================================================================================
// Cross-type gather/scatter index markers (scalar fallback).
// ===========================================================================================

macro_rules! impl_indexable8 {
    ($idx:ty => $($ty:ty),* $(,)?) => {$( impl IndexableRegister<$idx> for $ty {} )*};
}

impl_indexable8!(<super::super::X86V1 as crate::simd::Simd>::u32x4 => I8x4V1, U8x4V1);
impl_indexable8!(<super::super::X86V1 as crate::simd::Simd>::u64x4 => I8x4V1, U8x4V1);
impl_indexable8!(<super::super::X86V1 as crate::simd::Simd>::u32x8 => I8x8V1, U8x8V1);
impl_indexable8!(<super::super::X86V1 as crate::simd::Simd>::u64x8 => I8x8V1, U8x8V1);
