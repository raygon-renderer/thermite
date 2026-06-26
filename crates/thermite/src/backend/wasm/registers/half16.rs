//! Reduced (4-lane) 16-bit registers for the WASM backend and the cast/concat glue bridging
//! the scalar/array 16-bit halves, the native 128-bit `I16x8Wasm`/`U16x8Wasm`, and the 32-bit
//! registers they widen into. Mirrors the x86-v2 `half16.rs` structure with WASM intrinsics.

use generic_array::typenum::U4;

use super::arch;

use crate::register::{
    CastRegister, ConcatRegister, ExtendRegister, IndexableRegister, Storage, array::ArrayRegister,
    reduced::ReducedRegister,
};

/// 4-lane signed 16-bit register, backed by the low 4 lanes of a 128-bit `I16x8Wasm`.
pub type I16x4Wasm = ReducedRegister<super::I16x8Wasm, U4>;
/// 4-lane unsigned 16-bit register, backed by the low 4 lanes of a 128-bit `U16x8Wasm`.
pub type U16x4Wasm = ReducedRegister<super::U16x8Wasm, U4>;

// ---------------------------------------------------------------------------------------
// x4 <- x2 : build a 4-lane reduced register from two 2-lane (scalar-array) halves.
// ---------------------------------------------------------------------------------------

#[thermite_macros::inline_always]
impl ConcatRegister<ArrayRegister<i16, 2>> for I16x4Wasm {
    fn concat(lo: Storage<ArrayRegister<i16, 2>>, hi: Storage<ArrayRegister<i16, 2>>) -> Storage<Self> {
        ReducedRegister::new(arch::i16x8(lo.0[0], lo.0[1], hi.0[0], hi.0[1], 0, 0, 0, 0))
    }

    fn split(value: Storage<Self>) -> (Storage<ArrayRegister<i16, 2>>, Storage<ArrayRegister<i16, 2>>) {
        let mut arr = [0i16; 8];
        unsafe { arch::v128_store(arr.as_mut_ptr() as *mut _, value.0) };
        (ArrayRegister([arr[0], arr[1]]), ArrayRegister([arr[2], arr[3]]))
    }
}

#[thermite_macros::inline_always]
impl ExtendRegister<ArrayRegister<i16, 2>> for I16x4Wasm {
    fn extend(value: Storage<ArrayRegister<i16, 2>>) -> Storage<Self> {
        ReducedRegister::new(arch::i16x8(value.0[0], value.0[1], 0, 0, 0, 0, 0, 0))
    }

    fn narrow(value: Storage<Self>) -> Storage<ArrayRegister<i16, 2>> {
        let mut arr = [0i16; 8];
        unsafe { arch::v128_store(arr.as_mut_ptr() as *mut _, value.0) };
        ArrayRegister([arr[0], arr[1]])
    }
}

#[thermite_macros::inline_always]
impl ConcatRegister<ArrayRegister<u16, 2>> for U16x4Wasm {
    fn concat(lo: Storage<ArrayRegister<u16, 2>>, hi: Storage<ArrayRegister<u16, 2>>) -> Storage<Self> {
        ReducedRegister::new(arch::u16x8(lo.0[0], lo.0[1], hi.0[0], hi.0[1], 0, 0, 0, 0))
    }

    fn split(value: Storage<Self>) -> (Storage<ArrayRegister<u16, 2>>, Storage<ArrayRegister<u16, 2>>) {
        let mut arr = [0u16; 8];
        unsafe { arch::v128_store(arr.as_mut_ptr() as *mut _, value.0) };
        (ArrayRegister([arr[0], arr[1]]), ArrayRegister([arr[2], arr[3]]))
    }
}

#[thermite_macros::inline_always]
impl ExtendRegister<ArrayRegister<u16, 2>> for U16x4Wasm {
    fn extend(value: Storage<ArrayRegister<u16, 2>>) -> Storage<Self> {
        ReducedRegister::new(arch::u16x8(value.0[0], value.0[1], 0, 0, 0, 0, 0, 0))
    }

    fn narrow(value: Storage<Self>) -> Storage<ArrayRegister<u16, 2>> {
        let mut arr = [0u16; 8];
        unsafe { arch::v128_store(arr.as_mut_ptr() as *mut _, value.0) };
        ArrayRegister([arr[0], arr[1]])
    }
}

// ---------------------------------------------------------------------------------------
// x8 <- x4 : build the native 8-lane register from two 4-lane reduced halves.
// Both halves live in the low 64 bits of their v128; an i64x2 shuffle merges them.
// ---------------------------------------------------------------------------------------

#[thermite_macros::inline_always]
impl ConcatRegister<I16x4Wasm> for super::I16x8Wasm {
    fn concat(lo: Storage<I16x4Wasm>, hi: Storage<I16x4Wasm>) -> Storage<Self> {
        arch::i64x2_shuffle::<0, 2>(lo.0, hi.0)
    }

    fn split(value: Storage<Self>) -> (Storage<I16x4Wasm>, Storage<I16x4Wasm>) {
        (
            ReducedRegister::new(value),
            ReducedRegister::new(arch::i64x2_shuffle::<1, 1>(value, value)),
        )
    }
}

// NOTE: `ExtendRegister<I16x4Wasm> for I16x8Wasm` is provided by the ReducedRegister blanket.

#[thermite_macros::inline_always]
impl ConcatRegister<U16x4Wasm> for super::U16x8Wasm {
    fn concat(lo: Storage<U16x4Wasm>, hi: Storage<U16x4Wasm>) -> Storage<Self> {
        arch::i64x2_shuffle::<0, 2>(lo.0, hi.0)
    }

    fn split(value: Storage<Self>) -> (Storage<U16x4Wasm>, Storage<U16x4Wasm>) {
        (
            ReducedRegister::new(value),
            ReducedRegister::new(arch::i64x2_shuffle::<1, 1>(value, value)),
        )
    }
}

// NOTE: `ExtendRegister<U16x4Wasm> for U16x8Wasm` is provided by the ReducedRegister blanket.

// ---------------------------------------------------------------------------------------
// Widen casts to 32-bit. x4: I16x4Wasm <-> I32x4Wasm.  x2: ArrayRegister<i16,2> <-> I32x2Wasm.
// ---------------------------------------------------------------------------------------

// --- x4 widen i16 -> i32 (low 4 lanes of the reduced register) ---
#[thermite_macros::inline_always]
impl CastRegister<I16x4Wasm> for super::I32x4Wasm {
    fn cast_from(value: Storage<I16x4Wasm>) -> Storage<Self> {
        arch::i32x4_extend_low_i16x8(value.0)
    }
}

#[thermite_macros::inline_always]
impl CastRegister<U16x4Wasm> for super::U32x4Wasm {
    fn cast_from(value: Storage<U16x4Wasm>) -> Storage<Self> {
        arch::i32x4_extend_low_u16x8(value.0)
    }
}

// --- x4 narrow i32 -> i16 (truncate low 16 bits per lane into the low 4 i16 lanes) ---
#[thermite_macros::inline_always]
impl CastRegister<super::I32x4Wasm> for I16x4Wasm {
    #[rustfmt::skip]
    fn cast_from(value: Storage<super::I32x4Wasm>) -> Storage<Self> {
        ReducedRegister::new(arch::i8x16_shuffle::<
            0, 1, 4, 5, 8, 9, 12, 13, 0, 1, 2, 3, 4, 5, 6, 7,
        >(value, value))
    }
}

#[thermite_macros::inline_always]
impl CastRegister<super::U32x4Wasm> for U16x4Wasm {
    #[rustfmt::skip]
    fn cast_from(value: Storage<super::U32x4Wasm>) -> Storage<Self> {
        ReducedRegister::new(arch::i8x16_shuffle::<
            0, 1, 4, 5, 8, 9, 12, 13, 0, 1, 2, 3, 4, 5, 6, 7,
        >(value, value))
    }
}

// --- x2 widen i16 -> i32 (ArrayRegister<i16,2> -> I32x2Wasm reduced) ---
#[thermite_macros::inline_always]
impl CastRegister<ArrayRegister<i16, 2>> for super::half::I32x2Wasm {
    fn cast_from(value: Storage<ArrayRegister<i16, 2>>) -> Storage<Self> {
        ReducedRegister::new(arch::i32x4(value.0[0] as i32, value.0[1] as i32, 0, 0))
    }
}

#[thermite_macros::inline_always]
impl CastRegister<ArrayRegister<u16, 2>> for super::half::U32x2Wasm {
    fn cast_from(value: Storage<ArrayRegister<u16, 2>>) -> Storage<Self> {
        ReducedRegister::new(arch::u32x4(value.0[0] as u32, value.0[1] as u32, 0, 0))
    }
}

// --- x2 narrow i32 -> i16 (I32x2Wasm reduced -> ArrayRegister<i16,2>) ---
#[thermite_macros::inline_always]
impl CastRegister<super::half::I32x2Wasm> for ArrayRegister<i16, 2> {
    fn cast_from(value: Storage<super::half::I32x2Wasm>) -> Storage<Self> {
        let mut arr = [0i32; 4];
        unsafe { arch::v128_store(arr.as_mut_ptr() as *mut _, value.0) };
        ArrayRegister([arr[0] as i16, arr[1] as i16])
    }
}

#[thermite_macros::inline_always]
impl CastRegister<super::half::U32x2Wasm> for ArrayRegister<u16, 2> {
    fn cast_from(value: Storage<super::half::U32x2Wasm>) -> Storage<Self> {
        let mut arr = [0u32; 4];
        unsafe { arch::v128_store(arr.as_mut_ptr() as *mut _, value.0) };
        ArrayRegister([arr[0] as u16, arr[1] as u16])
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

#[thermite_macros::inline_always]
impl ConcatRegister<ArrayRegister<bool, 2>> for I16x4Wasm {
    fn concat(lo: Storage<ArrayRegister<bool, 2>>, hi: Storage<ArrayRegister<bool, 2>>) -> Storage<Self> {
        ReducedRegister::new(arch::i16x8(
            bool_to_i16_mask(lo.0[0]),
            bool_to_i16_mask(lo.0[1]),
            bool_to_i16_mask(hi.0[0]),
            bool_to_i16_mask(hi.0[1]),
            0,
            0,
            0,
            0,
        ))
    }

    fn split(value: Storage<Self>) -> (Storage<ArrayRegister<bool, 2>>, Storage<ArrayRegister<bool, 2>>) {
        let mut arr = [0i16; 8];
        unsafe { arch::v128_store(arr.as_mut_ptr() as *mut _, value.0) };
        (
            ArrayRegister([arr[0] != 0, arr[1] != 0]),
            ArrayRegister([arr[2] != 0, arr[3] != 0]),
        )
    }
}

#[thermite_macros::inline_always]
impl ExtendRegister<ArrayRegister<bool, 2>> for I16x4Wasm {
    fn extend(value: Storage<ArrayRegister<bool, 2>>) -> Storage<Self> {
        ReducedRegister::new(arch::i16x8(
            bool_to_i16_mask(value.0[0]),
            bool_to_i16_mask(value.0[1]),
            0,
            0,
            0,
            0,
            0,
            0,
        ))
    }

    fn narrow(value: Storage<Self>) -> Storage<ArrayRegister<bool, 2>> {
        let mut arr = [0i16; 8];
        unsafe { arch::v128_store(arr.as_mut_ptr() as *mut _, value.0) };
        ArrayRegister([arr[0] != 0, arr[1] != 0])
    }
}

#[thermite_macros::inline_always]
impl ConcatRegister<ArrayRegister<bool, 2>> for U16x4Wasm {
    fn concat(lo: Storage<ArrayRegister<bool, 2>>, hi: Storage<ArrayRegister<bool, 2>>) -> Storage<Self> {
        ReducedRegister::new(arch::i16x8(
            bool_to_i16_mask(lo.0[0]),
            bool_to_i16_mask(lo.0[1]),
            bool_to_i16_mask(hi.0[0]),
            bool_to_i16_mask(hi.0[1]),
            0,
            0,
            0,
            0,
        ))
    }

    fn split(value: Storage<Self>) -> (Storage<ArrayRegister<bool, 2>>, Storage<ArrayRegister<bool, 2>>) {
        let mut arr = [0i16; 8];
        unsafe { arch::v128_store(arr.as_mut_ptr() as *mut _, value.0) };
        (
            ArrayRegister([arr[0] != 0, arr[1] != 0]),
            ArrayRegister([arr[2] != 0, arr[3] != 0]),
        )
    }
}

#[thermite_macros::inline_always]
impl ExtendRegister<ArrayRegister<bool, 2>> for U16x4Wasm {
    fn extend(value: Storage<ArrayRegister<bool, 2>>) -> Storage<Self> {
        ReducedRegister::new(arch::i16x8(
            bool_to_i16_mask(value.0[0]),
            bool_to_i16_mask(value.0[1]),
            0,
            0,
            0,
            0,
            0,
            0,
        ))
    }

    fn narrow(value: Storage<Self>) -> Storage<ArrayRegister<bool, 2>> {
        let mut arr = [0i16; 8];
        unsafe { arch::v128_store(arr.as_mut_ptr() as *mut _, value.0) };
        ArrayRegister([arr[0] != 0, arr[1] != 0])
    }
}

// Gather/scatter for the reduced 16-bit registers (indexed by the native 4-lane index types)
// falls back to scalar; only the cross-type index markers are needed.
impl IndexableRegister<<super::super::Wasm as crate::simd::Simd>::u32x4> for I16x4Wasm {}
impl IndexableRegister<<super::super::Wasm as crate::simd::Simd>::u32x4> for U16x4Wasm {}
impl IndexableRegister<<super::super::Wasm as crate::simd::Simd>::u64x4> for I16x4Wasm {}
impl IndexableRegister<<super::super::Wasm as crate::simd::Simd>::u64x4> for U16x4Wasm {}
