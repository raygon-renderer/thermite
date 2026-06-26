//! Reduced (sub-native-width) 16-bit registers for x86-v2 and the cast/concat glue that
//! bridges the scalar/array 16-bit halves, the native 128-bit `I16x8V2`/`U16x8V2`, and the
//! 32-bit registers they widen into. Mirrors the 32-bit `half.rs` structure.

use generic_array::typenum::U4;

use super::arch;

use crate::register::{
    CastRegister, ConcatRegister, ExtendRegister, IndexableRegister, Storage, array::ArrayRegister,
    reduced::ReducedRegister,
};

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
