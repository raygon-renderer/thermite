use super::arch;

pub mod f32x4;
pub mod i32x4;
pub mod u32x4;

pub mod f64x2;
pub mod i64x2;
pub mod u64x2;

pub mod half;

pub use f32x4::F32x4Wasm;
pub use i32x4::I32x4Wasm;
pub use u32x4::U32x4Wasm;

pub use f64x2::F64x2Wasm;
pub use i64x2::I64x2Wasm;
pub use u64x2::U64x2Wasm;

pub use half::{F32x2Wasm, I32x2Wasm, U32x2Wasm};

use crate::{
    element::FindUSize,
    isa::InstructionSet,
    register::{BitCastRegister, IndexableRegister, Storage, array::ArrayRegister},
    simd::{HasIsa, NativeIsa, NativeSimd, Simd},
};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Wasm;

#[thermite_macros::inline_always]
impl HasIsa for Wasm {
    const ISA: InstructionSet = arch::ISA;
}

impl NativeIsa for Wasm {
    type Registers = generic_array::typenum::U1; // WASM has no practical register count limit

    type Native32Width = generic_array::typenum::U4;
    type Native64Width = generic_array::typenum::U2;

    type NativeAlignment = crate::simd::Align16; // 128-bit vectors = 16 bytes
}

#[thermite_macros::inline_always]
impl NativeSimd for Wasm {
    type f32xN = F32x4Wasm;
    type i32xN = I32x4Wasm;
    type u32xN = U32x4Wasm;

    type f64xN = F64x2Wasm;
    type i64xN = I64x2Wasm;
    type u64xN = U64x2Wasm;
}

// Scatter/Gather is not available on WASM, so we use fallback scalar impls.
// Only provide the concrete index types; usizex* resolves via type alias to u32x* on wasm32
// and to u64x* on wasm64, so explicit usizex* impls would duplicate and conflict.
macro_rules! impl_indexable {
    ($idx:ty => $($ty:ty),* $(,)?) => {$( impl IndexableRegister<$idx> for $ty {} )*};
}

// x2: 64-bit native types need u32x2 and u64x2 index support
impl_indexable!(<Wasm as Simd>::u32x2 => F64x2Wasm, I64x2Wasm, U64x2Wasm);
impl_indexable!(<Wasm as Simd>::u64x2 => F64x2Wasm, I64x2Wasm, U64x2Wasm);

// x4: 32-bit and wider types need u32x4 and u64x4 index support
impl_indexable!(<Wasm as Simd>::u32x4 => F32x4Wasm, I32x4Wasm, U32x4Wasm,
    <Wasm as Simd>::f64x4, <Wasm as Simd>::i64x4, <Wasm as Simd>::u64x4);
impl_indexable!(<Wasm as Simd>::u64x4 => F32x4Wasm, I32x4Wasm, U32x4Wasm);

#[thermite_macros::inline_always]
impl Simd for Wasm {
    type usizex2 = <() as FindUSize<(), Self::u32x2, Self::u64x2>>::Output;
    type usizex4 = <() as FindUSize<(), Self::u32x4, Self::u64x4>>::Output;
    type usizex8 = <() as FindUSize<(), Self::u32x8, Self::u64x8>>::Output;
    type usizex16 = <() as FindUSize<(), Self::u32x16, Self::u64x16>>::Output;

    type f32x2 = F32x2Wasm;
    type i32x2 = I32x2Wasm;
    type u32x2 = U32x2Wasm;

    type f32x4 = F32x4Wasm;
    type i32x4 = I32x4Wasm;
    type u32x4 = U32x4Wasm;

    type f64x2 = F64x2Wasm;
    type i64x2 = I64x2Wasm;
    type u64x2 = U64x2Wasm;

    type f32x8 = ArrayRegister<F32x4Wasm, 2>;
    type i32x8 = ArrayRegister<I32x4Wasm, 2>;
    type u32x8 = ArrayRegister<U32x4Wasm, 2>;

    type f64x4 = ArrayRegister<F64x2Wasm, 2>;
    type i64x4 = ArrayRegister<I64x2Wasm, 2>;
    type u64x4 = ArrayRegister<U64x2Wasm, 2>;

    type f32x16 = ArrayRegister<F32x4Wasm, 4>;
    type i32x16 = ArrayRegister<I32x4Wasm, 4>;
    type u32x16 = ArrayRegister<U32x4Wasm, 4>;

    type f64x8 = ArrayRegister<F64x2Wasm, 4>;
    type i64x8 = ArrayRegister<I64x2Wasm, 4>;
    type u64x8 = ArrayRegister<U64x2Wasm, 4>;

    type f64x16 = ArrayRegister<F64x2Wasm, 8>;
    type i64x16 = ArrayRegister<I64x2Wasm, 8>;
    type u64x16 = ArrayRegister<U64x2Wasm, 8>;
}

impl_concat_bool_register2!(f32, F32x2Wasm);
impl_concat_bool_register2!(u32, U32x2Wasm);
impl_concat_bool_register2!(i32, I32x2Wasm);

impl_concat_bool_register2!(f64, F64x2Wasm);
impl_concat_bool_register2!(u64, U64x2Wasm);
impl_concat_bool_register2!(i64, I64x2Wasm);

macro_rules! impl_identity_casts {
    ($($from:ty as $to:ty),* $(,)?) => {
        const _: () = {$(
            #[thermite_macros::inline_always]
            impl $crate::register::BitCastRegister<$from> for $to {
                fn from_bits(value: Storage<$from>) -> Storage<Self> {
                    value // all bit casts are no-ops in Wasm
                }
            }

            #[thermite_macros::inline_always]
            impl $crate::register::CastMaskRegister<$from> for $to {
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
            #[thermite_macros::inline_always]
            impl $crate::register::CastRegister<$from> for $to {
                fn cast_from(value: Storage<$from>) -> Storage<Self> {
                    arch::$conv(value)
                }

                $(
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

macro_rules! impl_extend_same {
    ($($ty:ty),* $(,)?) => {$( impl crate::register::ExtendRegister<$ty> for $ty {
        #[inline(always)]
        fn extend(value: Storage<$ty>) -> Storage<Self> {
            value
        }

        #[inline(always)]
        fn narrow(value: Storage<Self>) -> Storage<$ty> {
            value
        }
    } )*};
}

impl_extend_same!(F32x4Wasm, I32x4Wasm, U32x4Wasm, F64x2Wasm, I64x2Wasm, U64x2Wasm);
