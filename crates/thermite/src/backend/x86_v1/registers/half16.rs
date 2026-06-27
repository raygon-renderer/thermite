//! Reduced (sub-native-width) 16-bit registers for x86-v1 and the cast/concat glue bridging
//! the scalar/array 16-bit halves, the native 128-bit `I16x8V1`/`U16x8V1`, and the 32-bit
//! registers they widen into. Mirrors the v3 `half16.rs`, but widen casts use SSE2 `unpack`
//! sign/zero extension (no SSE4.1 `cvtep*`) and narrows fall back to scalar.

use generic_array::typenum::U4;

use super::arch;

use crate::register::{
    CastRegister, ConcatRegister, ExtendRegister, IndexableRegister, NumericRegister, Register, SaturatingCastRegister,
    Storage, array::ArrayRegister, reduced::ReducedRegister,
};

// Saturating narrow i32x4 -> i16x4 via SSE2 `packssdw`.
#[thermite_macros::inline_always]
impl SaturatingCastRegister<super::I32x4V1> for I16x4V1 {
    fn saturating_cast_from(value: Storage<super::I32x4V1>) -> Storage<Self> {
        ReducedRegister::new(unsafe { arch::_mm_packs_epi32(value, value) })
    }
}

// SSE2 has no `packusdw` and no unsigned 32-bit min, so every `u32 -> *` and the `i64`/x2
// pairs clamp into range (the register's own `min`/`max` use the v1 `*_epuNNx_v1` polyfills)
// and reuse the truncating narrow. Idempotent across nested ranges.
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
    (super::U32x4V1, u32, U16x4V1, u16),
    (ArrayRegister<super::I64x2V1, 2>, i64, I16x4V1, i16),
    (ArrayRegister<super::U64x2V1, 2>, u64, U16x4V1, u16),
    (super::half::I32x2V1, i32, ArrayRegister<i16, 2>, i16),
    (super::half::U32x2V1, u32, ArrayRegister<u16, 2>, u16),
    (super::I64x2V1, i64, ArrayRegister<i16, 2>, i16),
    (super::U64x2V1, u64, ArrayRegister<u16, 2>, u16),
    (ArrayRegister<super::I64x2V1, 8>, i64, ArrayRegister<super::I16x8V1, 2>, i16),
    (ArrayRegister<super::U64x2V1, 8>, u64, ArrayRegister<super::U16x8V1, 2>, u16),
}

/// 4-lane signed 16-bit register, backed by the low 4 lanes of a 128-bit `I16x8V1`.
pub type I16x4V1 = ReducedRegister<super::I16x8V1, U4>;
/// 4-lane unsigned 16-bit register, backed by the low 4 lanes of a 128-bit `U16x8V1`.
pub type U16x4V1 = ReducedRegister<super::U16x8V1, U4>;

// --- x4 <- x2 (two scalar-array halves into a 4-lane reduced register) ---

#[thermite_macros::inline_always]
impl ConcatRegister<ArrayRegister<i16, 2>> for I16x4V1 {
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
impl ExtendRegister<ArrayRegister<i16, 2>> for I16x4V1 {
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
impl ConcatRegister<ArrayRegister<u16, 2>> for U16x4V1 {
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
impl ExtendRegister<ArrayRegister<u16, 2>> for U16x4V1 {
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
impl ConcatRegister<I16x4V1> for super::I16x8V1 {
    fn concat(lo: Storage<I16x4V1>, hi: Storage<I16x4V1>) -> Storage<Self> {
        unsafe { arch::_mm_unpacklo_epi64(lo.0, hi.0) }
    }

    fn split(value: Storage<Self>) -> (Storage<I16x4V1>, Storage<I16x4V1>) {
        (
            ReducedRegister::new(value),
            ReducedRegister::new(unsafe { arch::_mm_unpackhi_epi64(value, value) }),
        )
    }
}

// `ExtendRegister<I16x4V1> for I16x8V1` is provided by the ReducedRegister blanket.

#[thermite_macros::inline_always]
impl ConcatRegister<U16x4V1> for super::U16x8V1 {
    fn concat(lo: Storage<U16x4V1>, hi: Storage<U16x4V1>) -> Storage<Self> {
        unsafe { arch::_mm_unpacklo_epi64(lo.0, hi.0) }
    }

    fn split(value: Storage<Self>) -> (Storage<U16x4V1>, Storage<U16x4V1>) {
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

impl_bool_concat!(I16x4V1);
impl_bool_concat!(U16x4V1);

// --- widen casts to 32-bit (SSE2 unpack sign/zero-extend; narrows are scalar) ---

#[thermite_macros::inline_always]
impl CastRegister<I16x4V1> for super::I32x4V1 {
    fn cast_from(value: Storage<I16x4V1>) -> Storage<Self> {
        // sign-extend the low 4 i16 lanes into 4 i32 lanes
        unsafe { arch::_mm_unpacklo_epi16(value.0, arch::_mm_srai_epi16(value.0, 15)) }
    }
}

#[thermite_macros::inline_always]
impl CastRegister<U16x4V1> for super::U32x4V1 {
    fn cast_from(value: Storage<U16x4V1>) -> Storage<Self> {
        unsafe { arch::_mm_unpacklo_epi16(value.0, arch::_mm_setzero_si128()) }
    }
}

#[thermite_macros::inline_always]
impl CastRegister<super::I32x4V1> for I16x4V1 {
    fn cast_from(value: Storage<super::I32x4V1>) -> Storage<Self> {
        let mut a = [0i32; 4];
        unsafe { arch::_mm_storeu_si128(a.as_mut_ptr() as *mut _, value) };
        ReducedRegister::new(unsafe {
            arch::_mm_setr_epi16(a[0] as i16, a[1] as i16, a[2] as i16, a[3] as i16, 0, 0, 0, 0)
        })
    }
}

#[thermite_macros::inline_always]
impl CastRegister<super::U32x4V1> for U16x4V1 {
    fn cast_from(value: Storage<super::U32x4V1>) -> Storage<Self> {
        let mut a = [0u32; 4];
        unsafe { arch::_mm_storeu_si128(a.as_mut_ptr() as *mut _, value) };
        ReducedRegister::new(unsafe {
            arch::_mm_setr_epi16(a[0] as i16, a[1] as i16, a[2] as i16, a[3] as i16, 0, 0, 0, 0)
        })
    }
}

#[thermite_macros::inline_always]
impl CastRegister<ArrayRegister<i16, 2>> for super::half::I32x2V1 {
    fn cast_from(value: Storage<ArrayRegister<i16, 2>>) -> Storage<Self> {
        ReducedRegister::new(unsafe { arch::_mm_setr_epi32(value.0[0] as i32, value.0[1] as i32, 0, 0) })
    }
}

#[thermite_macros::inline_always]
impl CastRegister<ArrayRegister<u16, 2>> for super::half::U32x2V1 {
    fn cast_from(value: Storage<ArrayRegister<u16, 2>>) -> Storage<Self> {
        ReducedRegister::new(unsafe { arch::_mm_setr_epi32(value.0[0] as i32, value.0[1] as i32, 0, 0) })
    }
}

#[thermite_macros::inline_always]
impl CastRegister<super::half::I32x2V1> for ArrayRegister<i16, 2> {
    fn cast_from(value: Storage<super::half::I32x2V1>) -> Storage<Self> {
        let mut arr = [0i32; 4];
        unsafe { arch::_mm_storeu_si128(arr.as_mut_ptr() as *mut _, value.0) };
        ArrayRegister([arr[0] as i16, arr[1] as i16])
    }
}

#[thermite_macros::inline_always]
impl CastRegister<super::half::U32x2V1> for ArrayRegister<u16, 2> {
    fn cast_from(value: Storage<super::half::U32x2V1>) -> Storage<Self> {
        let mut arr = [0u32; 4];
        unsafe { arch::_mm_storeu_si128(arr.as_mut_ptr() as *mut _, value.0) };
        ArrayRegister([arr[0] as u16, arr[1] as u16])
    }
}

// ===========================================================================================
// Widen/narrow casts to 64-bit (SSE2: store-and-rebuild widen/narrow; no pmovsx/pshufb).
//   widen 16 -> 64 via store-then-`_mm_set_epi64x` (sign/zero-extend each word to i64).
//   narrow 64 -> 16 via store-then-`_mm_setr_epi16` (low word of each i64, wrapping like `as`).
//   x2:  ArrayRegister<i16,2> <-> I64x2V1 (native 2-lane).
//   x4:  I16x4V1 <-> ArrayRegister<I64x2V1, 2> (the v1 i64x4).
//   x8:  I16x8V1 <-> ArrayRegister<I64x2V1, 4> (the v1 i64x8).
//   x16: ArrayRegister<I16x8V1, 2> <-> ArrayRegister<I64x2V1, 8> (i16x16 / i64x16).
// ===========================================================================================

#[inline(always)]
fn store_words(v: arch::__m128i) -> [i16; 8] {
    let mut arr = [0i16; 8];
    unsafe { arch::_mm_storeu_si128(arr.as_mut_ptr() as *mut _, v) };
    arr
}

#[inline(always)]
fn store_qwords(v: arch::__m128i) -> [i64; 2] {
    let mut arr = [0i64; 2];
    unsafe { arch::_mm_storeu_si128(arr.as_mut_ptr() as *mut _, v) };
    arr
}

// --- x2 widen i16 -> i64 (ArrayRegister<i16,2> -> I64x2V1 native) ---
#[thermite_macros::inline_always]
impl CastRegister<ArrayRegister<i16, 2>> for super::I64x2V1 {
    fn cast_from(value: Storage<ArrayRegister<i16, 2>>) -> Storage<Self> {
        unsafe { arch::_mm_cvt2epi16_epi64x_v1(value.0[0], value.0[1]) }
    }
}
#[thermite_macros::inline_always]
impl CastRegister<ArrayRegister<u16, 2>> for super::U64x2V1 {
    fn cast_from(value: Storage<ArrayRegister<u16, 2>>) -> Storage<Self> {
        unsafe { arch::_mm_cvt2epu16_epi64x_v1(value.0[0], value.0[1]) }
    }
}

// --- x2 narrow i64 -> i16 (I64x2V1 native -> ArrayRegister<i16,2>) ---
#[thermite_macros::inline_always]
impl CastRegister<super::I64x2V1> for ArrayRegister<i16, 2> {
    fn cast_from(value: Storage<super::I64x2V1>) -> Storage<Self> {
        let a = store_qwords(value);
        ArrayRegister([a[0] as i16, a[1] as i16])
    }
}
#[thermite_macros::inline_always]
impl CastRegister<super::U64x2V1> for ArrayRegister<u16, 2> {
    fn cast_from(value: Storage<super::U64x2V1>) -> Storage<Self> {
        let a = store_qwords(value);
        ArrayRegister([a[0] as u16, a[1] as u16])
    }
}

// --- x4 widen i16 -> i64 (low 4 words -> two 2x i64 lanes) ---
#[thermite_macros::inline_always]
impl CastRegister<I16x4V1> for ArrayRegister<super::I64x2V1, 2> {
    fn cast_from(value: Storage<I16x4V1>) -> Storage<Self> {
        let a = store_words(value.0);
        ArrayRegister(unsafe {
            [
                arch::_mm_cvt2epi16_epi64x_v1(a[0], a[1]),
                arch::_mm_cvt2epi16_epi64x_v1(a[2], a[3]),
            ]
        })
    }
}
#[thermite_macros::inline_always]
impl CastRegister<U16x4V1> for ArrayRegister<super::U64x2V1, 2> {
    fn cast_from(value: Storage<U16x4V1>) -> Storage<Self> {
        let a = store_words(value.0);
        ArrayRegister(unsafe {
            [
                arch::_mm_cvt2epu16_epi64x_v1(a[0] as u16, a[1] as u16),
                arch::_mm_cvt2epu16_epi64x_v1(a[2] as u16, a[3] as u16),
            ]
        })
    }
}

// --- x4 narrow i64 -> i16 (word 0 of each of 4 lanes -> low 4 words) ---
#[thermite_macros::inline_always]
impl CastRegister<ArrayRegister<super::I64x2V1, 2>> for I16x4V1 {
    fn cast_from(value: Storage<ArrayRegister<super::I64x2V1, 2>>) -> Storage<Self> {
        let lo = store_qwords(value.0[0]);
        let hi = store_qwords(value.0[1]);
        ReducedRegister::new(unsafe {
            arch::_mm_setr_epi16(lo[0] as i16, lo[1] as i16, hi[0] as i16, hi[1] as i16, 0, 0, 0, 0)
        })
    }
}
#[thermite_macros::inline_always]
impl CastRegister<ArrayRegister<super::U64x2V1, 2>> for U16x4V1 {
    fn cast_from(value: Storage<ArrayRegister<super::U64x2V1, 2>>) -> Storage<Self> {
        let lo = store_qwords(value.0[0]);
        let hi = store_qwords(value.0[1]);
        ReducedRegister::new(unsafe {
            arch::_mm_setr_epi16(lo[0] as i16, lo[1] as i16, hi[0] as i16, hi[1] as i16, 0, 0, 0, 0)
        })
    }
}

// --- x8 widen i16 -> i64 (8 words -> four 2x i64 lanes) ---
#[thermite_macros::inline_always]
impl CastRegister<super::I16x8V1> for ArrayRegister<super::I64x2V1, 4> {
    fn cast_from(value: Storage<super::I16x8V1>) -> Storage<Self> {
        unsafe { ArrayRegister(arch::_mm_cvtepi16_4epi64x_v1(value)) }
    }
}
#[thermite_macros::inline_always]
impl CastRegister<super::U16x8V1> for ArrayRegister<super::U64x2V1, 4> {
    fn cast_from(value: Storage<super::U16x8V1>) -> Storage<Self> {
        unsafe { ArrayRegister(arch::_mm_cvtepu16_4epi64x_v1(value)) }
    }
}

// --- x8 narrow i64 -> i16 (word 0 of each of 8 lanes -> 8 words) ---
#[thermite_macros::inline_always]
impl CastRegister<ArrayRegister<super::I64x2V1, 4>> for super::I16x8V1 {
    fn cast_from(value: Storage<ArrayRegister<super::I64x2V1, 4>>) -> Storage<Self> {
        unsafe { arch::_mm_cvt4epi64_epi16x_v1(value.0) }
    }
}
#[thermite_macros::inline_always]
impl CastRegister<ArrayRegister<super::U64x2V1, 4>> for super::U16x8V1 {
    fn cast_from(value: Storage<ArrayRegister<super::U64x2V1, 4>>) -> Storage<Self> {
        unsafe { arch::_mm_cvt4epi64_epi16x_v1(value.0) }
    }
}

// --- x16 widen i16 -> i64 (ArrayRegister<I16x8V1, 2> -> ArrayRegister<I64x2V1, 8>) ---
#[thermite_macros::inline_always]
impl CastRegister<ArrayRegister<super::I16x8V1, 2>> for ArrayRegister<super::I64x2V1, 8> {
    fn cast_from(value: Storage<ArrayRegister<super::I16x8V1, 2>>) -> Storage<Self> {
        unsafe {
            let lo = arch::_mm_cvtepi16_4epi64x_v1(value.0[0]);
            let hi = arch::_mm_cvtepi16_4epi64x_v1(value.0[1]);
            ArrayRegister([lo[0], lo[1], lo[2], lo[3], hi[0], hi[1], hi[2], hi[3]])
        }
    }
}
#[thermite_macros::inline_always]
impl CastRegister<ArrayRegister<super::U16x8V1, 2>> for ArrayRegister<super::U64x2V1, 8> {
    fn cast_from(value: Storage<ArrayRegister<super::U16x8V1, 2>>) -> Storage<Self> {
        unsafe {
            let lo = arch::_mm_cvtepu16_4epi64x_v1(value.0[0]);
            let hi = arch::_mm_cvtepu16_4epi64x_v1(value.0[1]);
            ArrayRegister([lo[0], lo[1], lo[2], lo[3], hi[0], hi[1], hi[2], hi[3]])
        }
    }
}

// --- x16 narrow i64 -> i16 (ArrayRegister<I64x2V1, 8> -> ArrayRegister<I16x8V1, 2>) ---
#[thermite_macros::inline_always]
impl CastRegister<ArrayRegister<super::I64x2V1, 8>> for ArrayRegister<super::I16x8V1, 2> {
    fn cast_from(value: Storage<ArrayRegister<super::I64x2V1, 8>>) -> Storage<Self> {
        let v = value.0;
        unsafe {
            ArrayRegister([
                arch::_mm_cvt4epi64_epi16x_v1([v[0], v[1], v[2], v[3]]),
                arch::_mm_cvt4epi64_epi16x_v1([v[4], v[5], v[6], v[7]]),
            ])
        }
    }
}
#[thermite_macros::inline_always]
impl CastRegister<ArrayRegister<super::U64x2V1, 8>> for ArrayRegister<super::U16x8V1, 2> {
    fn cast_from(value: Storage<ArrayRegister<super::U64x2V1, 8>>) -> Storage<Self> {
        let v = value.0;
        unsafe {
            ArrayRegister([
                arch::_mm_cvt4epi64_epi16x_v1([v[0], v[1], v[2], v[3]]),
                arch::_mm_cvt4epi64_epi16x_v1([v[4], v[5], v[6], v[7]]),
            ])
        }
    }
}

// ===========================================================================================
// 16 <-> f32/f64 direct casts (SSE2: no pmovsx/pshufb; widen via unpack to i32 then native
// i32<->float converts; narrow via truncating float->i32 then the existing store-rebuild word
// narrows).
//   WIDEN  i16/u16 -> f32 = widen low 4 words to i32x4 then `_mm_cvtepi32_ps` (exact).
//   NARROW f32 -> i16/u16 = `_mm_cvttps_epi32` (truncate) then store-rebuild low word per lane.
//   WIDEN  i16/u16 -> f64 = widen to i32 then `_mm_cvtepi32_pd` (2 lanes per F64x2V1, fan out).
//   NARROW f64 -> i16/u16 = `_mm_cvttpd_epi32` (truncate, low 2 lanes) then store-rebuild words.
// Unsigned 16-bit values fit in positive i32, so the signed i32->float convert is exact for them.
// ===========================================================================================

#[inline(always)]
fn store_dwords(v: arch::__m128i) -> [i32; 4] {
    let mut arr = [0i32; 4];
    unsafe { arch::_mm_storeu_si128(arr.as_mut_ptr() as *mut _, v) };
    arr
}

// --- x2 (ArrayRegister<i16,2> <-> ReducedRegister<F32x4V1,2> = F32x2V1) ---
#[thermite_macros::inline_always]
impl CastRegister<ArrayRegister<i16, 2>> for super::half::F32x2V1 {
    fn cast_from(value: Storage<ArrayRegister<i16, 2>>) -> Storage<Self> {
        let ints = unsafe { arch::_mm_setr_epi32(value.0[0] as i32, value.0[1] as i32, 0, 0) };
        ReducedRegister::new(unsafe { arch::_mm_cvtepi32_ps(ints) })
    }
}
#[thermite_macros::inline_always]
impl CastRegister<ArrayRegister<u16, 2>> for super::half::F32x2V1 {
    fn cast_from(value: Storage<ArrayRegister<u16, 2>>) -> Storage<Self> {
        let ints = unsafe { arch::_mm_setr_epi32(value.0[0] as i32, value.0[1] as i32, 0, 0) };
        ReducedRegister::new(unsafe { arch::_mm_cvtepi32_ps(ints) })
    }
}
#[thermite_macros::inline_always]
impl CastRegister<super::half::F32x2V1> for ArrayRegister<i16, 2> {
    fn cast_from(value: Storage<super::half::F32x2V1>) -> Storage<Self> {
        let a = store_dwords(unsafe { arch::_mm_cvttps_epi32(value.0) });
        ArrayRegister([a[0] as i16, a[1] as i16])
    }
}
#[thermite_macros::inline_always]
impl CastRegister<super::half::F32x2V1> for ArrayRegister<u16, 2> {
    fn cast_from(value: Storage<super::half::F32x2V1>) -> Storage<Self> {
        let a = store_dwords(unsafe { arch::_mm_cvttps_epi32(value.0) });
        ArrayRegister([a[0] as u16, a[1] as u16])
    }
}

// --- x2 (ArrayRegister<i16,2> <-> F64x2V1 native) ---
#[thermite_macros::inline_always]
impl CastRegister<ArrayRegister<i16, 2>> for super::F64x2V1 {
    fn cast_from(value: Storage<ArrayRegister<i16, 2>>) -> Storage<Self> {
        let ints = unsafe { arch::_mm_setr_epi32(value.0[0] as i32, value.0[1] as i32, 0, 0) };
        unsafe { arch::_mm_cvtepi32_pd(ints) }
    }
}
#[thermite_macros::inline_always]
impl CastRegister<ArrayRegister<u16, 2>> for super::F64x2V1 {
    fn cast_from(value: Storage<ArrayRegister<u16, 2>>) -> Storage<Self> {
        let ints = unsafe { arch::_mm_setr_epi32(value.0[0] as i32, value.0[1] as i32, 0, 0) };
        unsafe { arch::_mm_cvtepi32_pd(ints) }
    }
}
#[thermite_macros::inline_always]
impl CastRegister<super::F64x2V1> for ArrayRegister<i16, 2> {
    fn cast_from(value: Storage<super::F64x2V1>) -> Storage<Self> {
        let a = unsafe { arch::_mm_cvttpd_2i32x_v1(value) };
        ArrayRegister([a[0] as i16, a[1] as i16])
    }
}
#[thermite_macros::inline_always]
impl CastRegister<super::F64x2V1> for ArrayRegister<u16, 2> {
    fn cast_from(value: Storage<super::F64x2V1>) -> Storage<Self> {
        let a = unsafe { arch::_mm_cvttpd_2i32x_v1(value) };
        ArrayRegister([a[0] as u16, a[1] as u16])
    }
}

// --- x4 (I16x4V1 <-> F32x4V1 native) ---
#[thermite_macros::inline_always]
impl CastRegister<I16x4V1> for super::F32x4V1 {
    fn cast_from(value: Storage<I16x4V1>) -> Storage<Self> {
        unsafe { arch::_mm_cvtepi32_ps(arch::_mm_cvtepi16_epi32x_v1(value.0)) }
    }
}
#[thermite_macros::inline_always]
impl CastRegister<U16x4V1> for super::F32x4V1 {
    fn cast_from(value: Storage<U16x4V1>) -> Storage<Self> {
        unsafe { arch::_mm_cvtepi32_ps(arch::_mm_cvtepu16_epi32x_v1(value.0)) }
    }
}
#[thermite_macros::inline_always]
impl CastRegister<super::F32x4V1> for I16x4V1 {
    fn cast_from(value: Storage<super::F32x4V1>) -> Storage<Self> {
        let a = store_dwords(unsafe { arch::_mm_cvttps_epi32(value) });
        ReducedRegister::new(unsafe {
            arch::_mm_setr_epi16(a[0] as i16, a[1] as i16, a[2] as i16, a[3] as i16, 0, 0, 0, 0)
        })
    }
}
#[thermite_macros::inline_always]
impl CastRegister<super::F32x4V1> for U16x4V1 {
    fn cast_from(value: Storage<super::F32x4V1>) -> Storage<Self> {
        let a = store_dwords(unsafe { arch::_mm_cvttps_epi32(value) });
        ReducedRegister::new(unsafe {
            arch::_mm_setr_epi16(a[0] as i16, a[1] as i16, a[2] as i16, a[3] as i16, 0, 0, 0, 0)
        })
    }
}

// --- x4 (I16x4V1 <-> ArrayRegister<F64x2V1, 2>) ---
#[thermite_macros::inline_always]
impl CastRegister<I16x4V1> for ArrayRegister<super::F64x2V1, 2> {
    fn cast_from(value: Storage<I16x4V1>) -> Storage<Self> {
        unsafe {
            let ints = arch::_mm_cvtepi16_epi32x_v1(value.0);
            ArrayRegister(arch::_mm_cvtepi32_2pdx_v1(ints))
        }
    }
}
#[thermite_macros::inline_always]
impl CastRegister<U16x4V1> for ArrayRegister<super::F64x2V1, 2> {
    fn cast_from(value: Storage<U16x4V1>) -> Storage<Self> {
        unsafe {
            let ints = arch::_mm_cvtepu16_epi32x_v1(value.0);
            ArrayRegister(arch::_mm_cvtepi32_2pdx_v1(ints))
        }
    }
}
#[thermite_macros::inline_always]
impl CastRegister<ArrayRegister<super::F64x2V1, 2>> for I16x4V1 {
    fn cast_from(value: Storage<ArrayRegister<super::F64x2V1, 2>>) -> Storage<Self> {
        unsafe {
            let lo = arch::_mm_cvttpd_2i32x_v1(value.0[0]);
            let hi = arch::_mm_cvttpd_2i32x_v1(value.0[1]);
            ReducedRegister::new(arch::_mm_setr_epi16(
                lo[0] as i16, lo[1] as i16, hi[0] as i16, hi[1] as i16, 0, 0, 0, 0,
            ))
        }
    }
}
#[thermite_macros::inline_always]
impl CastRegister<ArrayRegister<super::F64x2V1, 2>> for U16x4V1 {
    fn cast_from(value: Storage<ArrayRegister<super::F64x2V1, 2>>) -> Storage<Self> {
        unsafe {
            let lo = arch::_mm_cvttpd_2i32x_v1(value.0[0]);
            let hi = arch::_mm_cvttpd_2i32x_v1(value.0[1]);
            ReducedRegister::new(arch::_mm_setr_epi16(
                lo[0] as i16, lo[1] as i16, hi[0] as i16, hi[1] as i16, 0, 0, 0, 0,
            ))
        }
    }
}

// --- x8 (I16x8V1 native <-> ArrayRegister<F32x4V1, 2>) ---
#[thermite_macros::inline_always]
impl CastRegister<super::I16x8V1> for ArrayRegister<super::F32x4V1, 2> {
    fn cast_from(value: Storage<super::I16x8V1>) -> Storage<Self> {
        unsafe {
            let lo = arch::_mm_cvtepi16_epi32x_v1(value);
            let hi = arch::_mm_cvtepi16_epi32x_v1(arch::_mm_srli_si128(value, 8));
            ArrayRegister([arch::_mm_cvtepi32_ps(lo), arch::_mm_cvtepi32_ps(hi)])
        }
    }
}
#[thermite_macros::inline_always]
impl CastRegister<super::U16x8V1> for ArrayRegister<super::F32x4V1, 2> {
    fn cast_from(value: Storage<super::U16x8V1>) -> Storage<Self> {
        unsafe {
            let lo = arch::_mm_cvtepu16_epi32x_v1(value);
            let hi = arch::_mm_cvtepu16_epi32x_v1(arch::_mm_srli_si128(value, 8));
            ArrayRegister([arch::_mm_cvtepi32_ps(lo), arch::_mm_cvtepi32_ps(hi)])
        }
    }
}
#[thermite_macros::inline_always]
impl CastRegister<ArrayRegister<super::F32x4V1, 2>> for super::I16x8V1 {
    fn cast_from(value: Storage<ArrayRegister<super::F32x4V1, 2>>) -> Storage<Self> {
        let lo = store_dwords(unsafe { arch::_mm_cvttps_epi32(value.0[0]) });
        let hi = store_dwords(unsafe { arch::_mm_cvttps_epi32(value.0[1]) });
        unsafe {
            arch::_mm_setr_epi16(
                lo[0] as i16, lo[1] as i16, lo[2] as i16, lo[3] as i16,
                hi[0] as i16, hi[1] as i16, hi[2] as i16, hi[3] as i16,
            )
        }
    }
}
#[thermite_macros::inline_always]
impl CastRegister<ArrayRegister<super::F32x4V1, 2>> for super::U16x8V1 {
    fn cast_from(value: Storage<ArrayRegister<super::F32x4V1, 2>>) -> Storage<Self> {
        let lo = store_dwords(unsafe { arch::_mm_cvttps_epi32(value.0[0]) });
        let hi = store_dwords(unsafe { arch::_mm_cvttps_epi32(value.0[1]) });
        unsafe {
            arch::_mm_setr_epi16(
                lo[0] as i16, lo[1] as i16, lo[2] as i16, lo[3] as i16,
                hi[0] as i16, hi[1] as i16, hi[2] as i16, hi[3] as i16,
            )
        }
    }
}

// --- x8 (I16x8V1 native <-> ArrayRegister<F64x2V1, 4>) ---
#[thermite_macros::inline_always]
impl CastRegister<super::I16x8V1> for ArrayRegister<super::F64x2V1, 4> {
    fn cast_from(value: Storage<super::I16x8V1>) -> Storage<Self> {
        unsafe {
            let a = arch::_mm_cvtepi32_2pdx_v1(arch::_mm_cvtepi16_epi32x_v1(value));
            let b = arch::_mm_cvtepi32_2pdx_v1(arch::_mm_cvtepi16_epi32x_v1(arch::_mm_srli_si128(value, 8)));
            ArrayRegister([a[0], a[1], b[0], b[1]])
        }
    }
}
#[thermite_macros::inline_always]
impl CastRegister<super::U16x8V1> for ArrayRegister<super::F64x2V1, 4> {
    fn cast_from(value: Storage<super::U16x8V1>) -> Storage<Self> {
        unsafe {
            let a = arch::_mm_cvtepi32_2pdx_v1(arch::_mm_cvtepu16_epi32x_v1(value));
            let b = arch::_mm_cvtepi32_2pdx_v1(arch::_mm_cvtepu16_epi32x_v1(arch::_mm_srli_si128(value, 8)));
            ArrayRegister([a[0], a[1], b[0], b[1]])
        }
    }
}
#[thermite_macros::inline_always]
impl CastRegister<ArrayRegister<super::F64x2V1, 4>> for super::I16x8V1 {
    fn cast_from(value: Storage<ArrayRegister<super::F64x2V1, 4>>) -> Storage<Self> {
        unsafe {
            let a = arch::_mm_cvttpd_2i32x_v1(value.0[0]);
            let b = arch::_mm_cvttpd_2i32x_v1(value.0[1]);
            let c = arch::_mm_cvttpd_2i32x_v1(value.0[2]);
            let d = arch::_mm_cvttpd_2i32x_v1(value.0[3]);
            arch::_mm_setr_epi16(
                a[0] as i16, a[1] as i16, b[0] as i16, b[1] as i16,
                c[0] as i16, c[1] as i16, d[0] as i16, d[1] as i16,
            )
        }
    }
}
#[thermite_macros::inline_always]
impl CastRegister<ArrayRegister<super::F64x2V1, 4>> for super::U16x8V1 {
    fn cast_from(value: Storage<ArrayRegister<super::F64x2V1, 4>>) -> Storage<Self> {
        unsafe {
            let a = arch::_mm_cvttpd_2i32x_v1(value.0[0]);
            let b = arch::_mm_cvttpd_2i32x_v1(value.0[1]);
            let c = arch::_mm_cvttpd_2i32x_v1(value.0[2]);
            let d = arch::_mm_cvttpd_2i32x_v1(value.0[3]);
            arch::_mm_setr_epi16(
                a[0] as i16, a[1] as i16, b[0] as i16, b[1] as i16,
                c[0] as i16, c[1] as i16, d[0] as i16, d[1] as i16,
            )
        }
    }
}

// --- x16 16 <-> f32: ArrayRegister<I16x8V1, 2> <-> ArrayRegister<F32x4V1, 4> is provided for
//     free by the ArrayRegister 2<->4 reshape cast blanket (array.rs), so no explicit impl. ---

// --- x16 (ArrayRegister<I16x8V1, 2> <-> ArrayRegister<F64x2V1, 8>) ---
#[thermite_macros::inline_always]
impl CastRegister<ArrayRegister<super::I16x8V1, 2>> for ArrayRegister<super::F64x2V1, 8> {
    fn cast_from(value: Storage<ArrayRegister<super::I16x8V1, 2>>) -> Storage<Self> {
        let v = value.0;
        unsafe {
            let q0 = arch::_mm_cvtepi16_epi32x_v1(v[0]);
            let q1 = arch::_mm_cvtepi16_epi32x_v1(arch::_mm_srli_si128(v[0], 8));
            let q2 = arch::_mm_cvtepi16_epi32x_v1(v[1]);
            let q3 = arch::_mm_cvtepi16_epi32x_v1(arch::_mm_srli_si128(v[1], 8));
            let a = arch::_mm_cvtepi32_2pdx_v1(q0);
            let b = arch::_mm_cvtepi32_2pdx_v1(q1);
            let c = arch::_mm_cvtepi32_2pdx_v1(q2);
            let d = arch::_mm_cvtepi32_2pdx_v1(q3);
            ArrayRegister([a[0], a[1], b[0], b[1], c[0], c[1], d[0], d[1]])
        }
    }
}
#[thermite_macros::inline_always]
impl CastRegister<ArrayRegister<super::U16x8V1, 2>> for ArrayRegister<super::F64x2V1, 8> {
    fn cast_from(value: Storage<ArrayRegister<super::U16x8V1, 2>>) -> Storage<Self> {
        let v = value.0;
        unsafe {
            let q0 = arch::_mm_cvtepu16_epi32x_v1(v[0]);
            let q1 = arch::_mm_cvtepu16_epi32x_v1(arch::_mm_srli_si128(v[0], 8));
            let q2 = arch::_mm_cvtepu16_epi32x_v1(v[1]);
            let q3 = arch::_mm_cvtepu16_epi32x_v1(arch::_mm_srli_si128(v[1], 8));
            let a = arch::_mm_cvtepi32_2pdx_v1(q0);
            let b = arch::_mm_cvtepi32_2pdx_v1(q1);
            let c = arch::_mm_cvtepi32_2pdx_v1(q2);
            let d = arch::_mm_cvtepi32_2pdx_v1(q3);
            ArrayRegister([a[0], a[1], b[0], b[1], c[0], c[1], d[0], d[1]])
        }
    }
}
#[thermite_macros::inline_always]
impl CastRegister<ArrayRegister<super::F64x2V1, 8>> for ArrayRegister<super::I16x8V1, 2> {
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
            ArrayRegister([
                arch::_mm_setr_epi16(
                    a[0] as i16, a[1] as i16, b[0] as i16, b[1] as i16,
                    c[0] as i16, c[1] as i16, d[0] as i16, d[1] as i16,
                ),
                arch::_mm_setr_epi16(
                    e[0] as i16, e[1] as i16, f[0] as i16, f[1] as i16,
                    g[0] as i16, g[1] as i16, h[0] as i16, h[1] as i16,
                ),
            ])
        }
    }
}
#[thermite_macros::inline_always]
impl CastRegister<ArrayRegister<super::F64x2V1, 8>> for ArrayRegister<super::U16x8V1, 2> {
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
            ArrayRegister([
                arch::_mm_setr_epi16(
                    a[0] as i16, a[1] as i16, b[0] as i16, b[1] as i16,
                    c[0] as i16, c[1] as i16, d[0] as i16, d[1] as i16,
                ),
                arch::_mm_setr_epi16(
                    e[0] as i16, e[1] as i16, f[0] as i16, f[1] as i16,
                    g[0] as i16, g[1] as i16, h[0] as i16, h[1] as i16,
                ),
            ])
        }
    }
}

// --- scalar-fallback gather markers for the reduced 16-bit registers ---
impl IndexableRegister<<super::super::X86V1 as crate::simd::Simd>::u32x4> for I16x4V1 {}
impl IndexableRegister<<super::super::X86V1 as crate::simd::Simd>::u32x4> for U16x4V1 {}
impl IndexableRegister<<super::super::X86V1 as crate::simd::Simd>::u64x4> for I16x4V1 {}
impl IndexableRegister<<super::super::X86V1 as crate::simd::Simd>::u64x4> for U16x4V1 {}
