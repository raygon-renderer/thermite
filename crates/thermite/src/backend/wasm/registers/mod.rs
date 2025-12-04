use super::arch;

pub mod f32x4;
pub mod i32x4;
pub mod u32x4;

pub mod f64x2;
pub mod i64x2;
pub mod u64x2;

pub use f32x4::F32x4Wasm32;
pub use i32x4::I32x4Wasm32;
pub use u32x4::U32x4Wasm32;

pub use f64x2::F64x2Wasm32;
pub use i64x2::I64x2Wasm32;
pub use u64x2::U64x2Wasm32;

use crate::{
    register::{Storage, dp::DoublePumpRegister},
    simd::{NativeSimd, Simd},
};

pub struct Wasm32;

impl NativeSimd for Wasm32 {
    type Native32Width = generic_array::typenum::U4;
    type Native64Width = generic_array::typenum::U2;

    type f32xN = F32x4Wasm32;
    type i32xN = I32x4Wasm32;
    type u32xN = U32x4Wasm32;

    type f64xN = F64x2Wasm32;
    type i64xN = I64x2Wasm32;
    type u64xN = U64x2Wasm32;
}

impl Simd for Wasm32 {
    type f32x4 = F32x4Wasm32;
    type i32x4 = I32x4Wasm32;
    type u32x4 = U32x4Wasm32;

    type f64x2 = F64x2Wasm32;
    type i64x2 = I64x2Wasm32;
    type u64x2 = U64x2Wasm32;

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
    F32x4Wasm32 as U32x4Wasm32,
    F32x4Wasm32 as I32x4Wasm32,
    U32x4Wasm32 as F32x4Wasm32,
    U32x4Wasm32 as I32x4Wasm32,
    I32x4Wasm32 as F32x4Wasm32,
    I32x4Wasm32 as U32x4Wasm32,

    // 64-bit
    F64x2Wasm32 as U64x2Wasm32,
    F64x2Wasm32 as I64x2Wasm32,
    U64x2Wasm32 as F64x2Wasm32,
    U64x2Wasm32 as I64x2Wasm32,
    I64x2Wasm32 as F64x2Wasm32,
    I64x2Wasm32 as U64x2Wasm32,

    // identity casts
    F32x4Wasm32 as F32x4Wasm32,
    I32x4Wasm32 as I32x4Wasm32,
    U32x4Wasm32 as U32x4Wasm32,
    F64x2Wasm32 as F64x2Wasm32,
    I64x2Wasm32 as I64x2Wasm32,
    U64x2Wasm32 as U64x2Wasm32,
}

impl_type_casts! {
    // same type casts (identity)
    F32x4Wasm32 as F32x4Wasm32 => identity,
    F64x2Wasm32 as F64x2Wasm32 => identity,
    I32x4Wasm32 as I32x4Wasm32 => identity,
    I64x2Wasm32 as I64x2Wasm32 => identity,
    U32x4Wasm32 as U32x4Wasm32 => identity,
    U64x2Wasm32 as U64x2Wasm32 => identity,

    // integer casts (also identity)
    I32x4Wasm32 as U32x4Wasm32 => identity,
    U32x4Wasm32 as I32x4Wasm32 => identity,
    I64x2Wasm32 as U64x2Wasm32 => identity,
    U64x2Wasm32 as I64x2Wasm32 => identity,

    F32x4Wasm32 as I32x4Wasm32 => i32x4_trunc_sat_f32x4 | i32x4_relaxed_trunc_f32x4,
    F32x4Wasm32 as U32x4Wasm32 => u32x4_trunc_sat_f32x4 | u32x4_relaxed_trunc_f32x4,
    I32x4Wasm32 as F32x4Wasm32 => f32x4_convert_i32x4,
    U32x4Wasm32 as F32x4Wasm32 => f32x4_convert_u32x4,
}
