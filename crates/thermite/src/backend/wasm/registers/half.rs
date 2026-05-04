use crate::register::{
    CastRegister, ConcatRegister, ExtendRegister, IndexableRegister, Storage,
    reduced::{HalfRegister2, ReducedRegister},
};

use super::arch;

pub type F32x2Wasm = HalfRegister2<super::F32x4Wasm>;
pub type I32x2Wasm = HalfRegister2<super::I32x4Wasm>;
pub type U32x2Wasm = HalfRegister2<super::U32x4Wasm>;

// --- ConcatRegister / ExtendRegister for f32 ---

#[thermite_macros::inline_always]
impl ConcatRegister<f32> for F32x2Wasm {
    fn concat(lo: Storage<f32>, hi: Storage<f32>) -> Storage<Self> {
        ReducedRegister::new(arch::f32x4(lo, hi, 0.0, 0.0))
    }

    fn split(value: Storage<Self>) -> (Storage<f32>, Storage<f32>) {
        (
            arch::f32x4_extract_lane::<0>(value.0),
            arch::f32x4_extract_lane::<1>(value.0),
        )
    }
}

#[thermite_macros::inline_always]
impl ExtendRegister<f32> for F32x2Wasm {
    fn extend(value: Storage<f32>) -> Storage<Self> {
        ReducedRegister::new(arch::f32x4(value, 0.0, 0.0, 0.0))
    }

    fn narrow(value: Storage<Self>) -> Storage<f32> {
        arch::f32x4_extract_lane::<0>(value.0)
    }
}

#[thermite_macros::inline_always]
impl ConcatRegister<F32x2Wasm> for super::F32x4Wasm {
    fn concat(lo: Storage<F32x2Wasm>, hi: Storage<F32x2Wasm>) -> Storage<Self> {
        // Place lower 2 lanes of lo in positions 0,1 and lower 2 lanes of hi in positions 2,3
        arch::i32x4_shuffle::<0, 1, 4, 5>(lo.0, hi.0)
    }

    fn split(value: Storage<Self>) -> (Storage<F32x2Wasm>, Storage<F32x2Wasm>) {
        (
            ReducedRegister::new(value),
            ReducedRegister::new(arch::i32x4_shuffle::<2, 3, 2, 3>(value, value)),
        )
    }
}

// --- ConcatRegister / ExtendRegister for i32 ---

#[thermite_macros::inline_always]
impl ConcatRegister<i32> for I32x2Wasm {
    fn concat(lo: Storage<i32>, hi: Storage<i32>) -> Storage<Self> {
        ReducedRegister::new(arch::i32x4(lo, hi, 0, 0))
    }

    fn split(value: Storage<Self>) -> (Storage<i32>, Storage<i32>) {
        (
            arch::i32x4_extract_lane::<0>(value.0),
            arch::i32x4_extract_lane::<1>(value.0),
        )
    }
}

#[thermite_macros::inline_always]
impl ExtendRegister<i32> for I32x2Wasm {
    fn extend(value: Storage<i32>) -> Storage<Self> {
        ReducedRegister::new(arch::i32x4(value, 0, 0, 0))
    }

    fn narrow(value: Storage<Self>) -> Storage<i32> {
        arch::i32x4_extract_lane::<0>(value.0)
    }
}

#[thermite_macros::inline_always]
impl ConcatRegister<I32x2Wasm> for super::I32x4Wasm {
    fn concat(lo: Storage<I32x2Wasm>, hi: Storage<I32x2Wasm>) -> Storage<Self> {
        arch::i32x4_shuffle::<0, 1, 4, 5>(lo.0, hi.0)
    }

    fn split(value: Storage<Self>) -> (Storage<I32x2Wasm>, Storage<I32x2Wasm>) {
        (
            ReducedRegister::new(value),
            ReducedRegister::new(arch::i32x4_shuffle::<2, 3, 2, 3>(value, value)),
        )
    }
}

// --- ConcatRegister / ExtendRegister for u32 ---

#[thermite_macros::inline_always]
impl ConcatRegister<u32> for U32x2Wasm {
    fn concat(lo: Storage<u32>, hi: Storage<u32>) -> Storage<Self> {
        ReducedRegister::new(arch::u32x4(lo, hi, 0, 0))
    }

    fn split(value: Storage<Self>) -> (Storage<u32>, Storage<u32>) {
        (
            arch::u32x4_extract_lane::<0>(value.0),
            arch::u32x4_extract_lane::<1>(value.0),
        )
    }
}

#[thermite_macros::inline_always]
impl ExtendRegister<u32> for U32x2Wasm {
    fn extend(value: Storage<u32>) -> Storage<Self> {
        ReducedRegister::new(arch::u32x4(value, 0, 0, 0))
    }

    fn narrow(value: Storage<Self>) -> Storage<u32> {
        arch::u32x4_extract_lane::<0>(value.0)
    }
}

#[thermite_macros::inline_always]
impl ConcatRegister<U32x2Wasm> for super::U32x4Wasm {
    fn concat(lo: Storage<U32x2Wasm>, hi: Storage<U32x2Wasm>) -> Storage<Self> {
        arch::i32x4_shuffle::<0, 1, 4, 5>(lo.0, hi.0)
    }

    fn split(value: Storage<Self>) -> (Storage<U32x2Wasm>, Storage<U32x2Wasm>) {
        (
            ReducedRegister::new(value),
            ReducedRegister::new(arch::i32x4_shuffle::<2, 3, 2, 3>(value, value)),
        )
    }
}

// --- CastRegister between x2 types ---

// f32x2 <-> f64x2 (promote / demote)
#[thermite_macros::inline_always]
impl CastRegister<super::F64x2Wasm> for F32x2Wasm {
    fn cast_from(value: Storage<super::F64x2Wasm>) -> Storage<Self> {
        ReducedRegister::new(arch::f32x4_demote_f64x2_zero(value))
    }
}

#[thermite_macros::inline_always]
impl CastRegister<F32x2Wasm> for super::F64x2Wasm {
    fn cast_from(value: Storage<F32x2Wasm>) -> Storage<Self> {
        arch::f64x2_promote_low_f32x4(value.0)
    }
}

// i32x2 <-> i64x2 (sign-extend / truncate)
#[thermite_macros::inline_always]
impl CastRegister<super::I64x2Wasm> for I32x2Wasm {
    fn cast_from(value: Storage<super::I64x2Wasm>) -> Storage<Self> {
        // Truncate: take lower 32 bits of each 64-bit lane (bytes 0-3 and 8-11)
        ReducedRegister::new(arch::i8x16_shuffle::<
            0,
            1,
            2,
            3, // Lane 0 low 32 bits
            8,
            9,
            10,
            11, // Lane 1 low 32 bits
            0,
            1,
            2,
            3, // (padding, ignored by ReducedRegister)
            8,
            9,
            10,
            11, // (padding, ignored by ReducedRegister)
        >(value, value))
    }
}

#[thermite_macros::inline_always]
impl CastRegister<I32x2Wasm> for super::I64x2Wasm {
    fn cast_from(value: Storage<I32x2Wasm>) -> Storage<Self> {
        // Sign-extend the lower 2 i32 lanes to i64
        arch::i64x2_extend_low_i32x4(value.0)
    }
}

// u32x2 <-> u64x2 (zero-extend / truncate)
#[thermite_macros::inline_always]
impl CastRegister<super::U64x2Wasm> for U32x2Wasm {
    fn cast_from(value: Storage<super::U64x2Wasm>) -> Storage<Self> {
        // Truncate: take lower 32 bits of each 64-bit lane (bytes 0-3 and 8-11)
        ReducedRegister::new(arch::i8x16_shuffle::<
            0,
            1,
            2,
            3, // Lane 0 low 32 bits
            8,
            9,
            10,
            11, // Lane 1 low 32 bits
            0,
            1,
            2,
            3, // (padding, ignored by ReducedRegister)
            8,
            9,
            10,
            11, // (padding, ignored by ReducedRegister)
        >(value, value))
    }
}

#[thermite_macros::inline_always]
impl CastRegister<U32x2Wasm> for super::U64x2Wasm {
    fn cast_from(value: Storage<U32x2Wasm>) -> Storage<Self> {
        // Zero-extend the lower 2 u32 lanes to u64
        arch::u64x2_extend_low_u32x4(value.0)
    }
}

// --- IndexableRegister marker impls ---
// Note: IndexableRegister<U32x2Wasm> for the half types is auto-derived by the
// ReducedRegister blanket impl (since F32x4Wasm etc. have IndexableRegister<U32x4Wasm>).
// We only need to provide the non-auto-derived U64x2Wasm index type.

#[thermite_macros::inline_always]
impl IndexableRegister<super::U64x2Wasm> for F32x2Wasm {}
impl IndexableRegister<super::U64x2Wasm> for I32x2Wasm {}
impl IndexableRegister<super::U64x2Wasm> for U32x2Wasm {}
