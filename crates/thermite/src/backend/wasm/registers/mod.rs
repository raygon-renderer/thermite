use super::arch;

pub mod f32x4;
pub mod i32x4;
pub mod u32x4;

pub mod f64x2;
pub mod i64x2;
pub mod u64x2;

pub use f32x4::F32x4Wasm;
pub use i32x4::I32x4Wasm;
pub use u32x4::U32x4Wasm;

pub use f64x2::F64x2Wasm;
pub use i64x2::I64x2Wasm;
pub use u64x2::U64x2Wasm;

use crate::{
    isa::InstructionSet,
    register::{Storage, dp::DoublePumpRegister},
    simd::{NativeSimd, Simd},
};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Wasm;

impl NativeSimd for Wasm {
    const ISA: InstructionSet = arch::ISA;

    type Native32Width = generic_array::typenum::U4;
    type Native64Width = generic_array::typenum::U2;

    type NativeAlignment = crate::simd::Align16; // 128-bit vectors = 16 bytes

    type f32xN = F32x4Wasm;
    type i32xN = I32x4Wasm;
    type u32xN = U32x4Wasm;

    type f64xN = F64x2Wasm;
    type i64xN = I64x2Wasm;
    type u64xN = U64x2Wasm;
}

impl Simd for Wasm {
    type f32x4 = F32x4Wasm;
    type i32x4 = I32x4Wasm;
    type u32x4 = U32x4Wasm;

    type f64x2 = F64x2Wasm;
    type i64x2 = I64x2Wasm;
    type u64x2 = U64x2Wasm;

    type f32x8 = DoublePumpRegister<Self::f32x4>;
    type i32x8 = DoublePumpRegister<Self::i32x4>;
    type u32x8 = DoublePumpRegister<Self::u32x4>;

    type f64x4 = DoublePumpRegister<Self::f64x2>;
    type i64x4 = DoublePumpRegister<Self::i64x2>;
    type u64x4 = DoublePumpRegister<Self::u64x2>;

    type f32x16 = DoublePumpRegister<Self::f32x8>;
    type i32x16 = DoublePumpRegister<Self::i32x8>;
    type u32x16 = DoublePumpRegister<Self::u32x8>;

    type f64x8 = DoublePumpRegister<Self::f64x4>;
    type i64x8 = DoublePumpRegister<Self::i64x4>;
    type u64x8 = DoublePumpRegister<Self::u64x4>;

    type f64x16 = DoublePumpRegister<Self::f64x8>;
    type i64x16 = DoublePumpRegister<Self::i64x8>;
    type u64x16 = DoublePumpRegister<Self::u64x8>;
}

macro_rules! impl_identity_casts {
    ($($from:ty as $to:ty),* $(,)?) => {
        const _: () = {$(
            impl $crate::register::BitsRegister<$from> for $to {
                #[inline(always)]
                fn from_bits(value: Storage<$from>) -> Storage<Self> {
                    value // all bit casts are no-ops in Wasm
                }
            }

            impl $crate::register::CastMaskRegister<$from> for $to {
                #[inline(always)]
                fn mask_from(value: Storage<$from>) -> Storage<Self> {
                    value // all mask casts are no-ops in Wasm
                }
            }
        )*};
    };
}

macro_rules! impl_type_casts {
    ($($from:ty as $to:ty => $conv:ident $(| $fast:ident)?),* $(,)?) => {
        const _: () = {$(
            impl $crate::register::CastRegister<$from> for $to {
                #[inline(always)]
                fn cast_from(value: Storage<$from>) -> Storage<Self> {
                    arch::$conv(value)
                }

                $(
                    #[inline(always)]
                    fn fast_cast_from(value: Storage<$from>) -> Storage<Self> {
                        arch::$fast(value)
                    }
                )?
            }
        )*};
    };
}

impl_identity_casts! {
    // 32-bit
    F32x4Wasm as U32x4Wasm,
    F32x4Wasm as I32x4Wasm,
    U32x4Wasm as F32x4Wasm,
    U32x4Wasm as I32x4Wasm,
    I32x4Wasm as F32x4Wasm,
    I32x4Wasm as U32x4Wasm,

    // 64-bit
    F64x2Wasm as U64x2Wasm,
    F64x2Wasm as I64x2Wasm,
    U64x2Wasm as F64x2Wasm,
    U64x2Wasm as I64x2Wasm,
    I64x2Wasm as F64x2Wasm,
    I64x2Wasm as U64x2Wasm,

    // identity casts
    F32x4Wasm as F32x4Wasm,
    I32x4Wasm as I32x4Wasm,
    U32x4Wasm as U32x4Wasm,
    F64x2Wasm as F64x2Wasm,
    I64x2Wasm as I64x2Wasm,
    U64x2Wasm as U64x2Wasm,
}

impl_type_casts! {
    // same type casts (identity)
    F32x4Wasm as F32x4Wasm => identity,
    F64x2Wasm as F64x2Wasm => identity,
    I32x4Wasm as I32x4Wasm => identity,
    I64x2Wasm as I64x2Wasm => identity,
    U32x4Wasm as U32x4Wasm => identity,
    U64x2Wasm as U64x2Wasm => identity,

    // integer casts (also identity)
    I32x4Wasm as U32x4Wasm => identity,
    U32x4Wasm as I32x4Wasm => identity,
    I64x2Wasm as U64x2Wasm => identity,
    U64x2Wasm as I64x2Wasm => identity,

    F32x4Wasm as I32x4Wasm => i32x4_trunc_sat_f32x4 | i32x4_relaxed_trunc_f32x4,
    F32x4Wasm as U32x4Wasm => u32x4_trunc_sat_f32x4 | u32x4_relaxed_trunc_f32x4,
    I32x4Wasm as F32x4Wasm => f32x4_convert_i32x4,
    U32x4Wasm as F32x4Wasm => f32x4_convert_u32x4,
}
