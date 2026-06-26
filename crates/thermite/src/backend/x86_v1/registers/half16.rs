//! Reduced (sub-native-width) 16-bit registers for x86-v1 and the cast/concat glue bridging
//! the scalar/array 16-bit halves, the native 128-bit `I16x8V1`/`U16x8V1`, and the 32-bit
//! registers they widen into. Mirrors the v3 `half16.rs`, but widen casts use SSE2 `unpack`
//! sign/zero extension (no SSE4.1 `cvtep*`) and narrows fall back to scalar.

use generic_array::typenum::U4;

use super::arch;

use crate::register::{
    CastRegister, ConcatRegister, ExtendRegister, IndexableRegister, Storage, array::ArrayRegister,
    reduced::ReducedRegister,
};

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

// --- scalar-fallback gather markers for the reduced 16-bit registers ---
impl IndexableRegister<<super::super::X86V1 as crate::simd::Simd>::u32x4> for I16x4V1 {}
impl IndexableRegister<<super::super::X86V1 as crate::simd::Simd>::u32x4> for U16x4V1 {}
impl IndexableRegister<<super::super::X86V1 as crate::simd::Simd>::u64x4> for I16x4V1 {}
impl IndexableRegister<<super::super::X86V1 as crate::simd::Simd>::u64x4> for U16x4V1 {}
