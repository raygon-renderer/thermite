use super::arch;

pub mod f32x4;
pub mod i32x4;
pub mod u32x4;

pub mod f64x2;
pub mod i64x2;
pub mod u64x2;

pub mod i16x8;
pub mod u16x8;

pub mod i8x16;
pub mod u8x16;

pub mod half;
pub mod half16;
pub mod packed; // PackedFloatRegister (16-bit float) generic-default impls for native u16 regs

pub use f32x4::F32x4Wasm;
pub use i32x4::I32x4Wasm;
pub use u32x4::U32x4Wasm;

pub use f64x2::F64x2Wasm;
pub use i64x2::I64x2Wasm;
pub use u64x2::U64x2Wasm;

pub use i16x8::I16x8Wasm;
pub use u16x8::U16x8Wasm;

pub use i8x16::I8x16Wasm;
pub use u8x16::U8x16Wasm;

pub use half::{F32x2Wasm, I32x2Wasm, U32x2Wasm};

use crate::{
    element::FindUSize,
    isa::InstructionSet,
    register::{BitCastRegister, IndexableRegister, Storage, array::ArrayRegister, reduced::ReducedRegister},
    simd::{HasIsa, NativeIsa, NativeSimd, Simd, Simd3, Simd3A, SimdExperimental},
};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Wasm;

impl_newregister!(
    F32x4Wasm, I32x4Wasm, U32x4Wasm, F64x2Wasm, I64x2Wasm, U64x2Wasm, I16x8Wasm, U16x8Wasm, I8x16Wasm, U8x16Wasm
);

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

// 16-bit gather/scatter has no hardware support on WASM; scalar-fallback marker impls.
impl_indexable!(U16x8Wasm => I16x8Wasm, U16x8Wasm);
impl_indexable!(<Wasm as Simd>::u32x8 => I16x8Wasm, U16x8Wasm);
impl_indexable!(<Wasm as Simd>::u64x8 => I16x8Wasm, U16x8Wasm);
impl_indexable!(<Wasm as Simd>::u32x2 => ArrayRegister<i16, 2>, ArrayRegister<u16, 2>);
impl_indexable!(<Wasm as Simd>::u64x2 => ArrayRegister<i16, 2>, ArrayRegister<u16, 2>);
impl_indexable!(<Wasm as Simd>::u64x16 => ArrayRegister<I16x8Wasm, 2>, ArrayRegister<U16x8Wasm, 2>);

// 8-bit: same-width self-indexing only.
impl_indexable!(U8x16Wasm => I8x16Wasm, U8x16Wasm);

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

impl SimdExperimental for Wasm {
    type Native16Width = generic_array::typenum::U8;

    type i16xN = I16x8Wasm;
    type u16xN = U16x8Wasm;

    type i16x2 = ArrayRegister<i16, 2>;
    type u16x2 = ArrayRegister<u16, 2>;

    type i16x4 = half16::I16x4Wasm;
    type u16x4 = half16::U16x4Wasm;

    type i16x8 = I16x8Wasm;
    type u16x8 = U16x8Wasm;

    type i16x16 = ArrayRegister<I16x8Wasm, 2>;
    type u16x16 = ArrayRegister<U16x8Wasm, 2>;

    type Native8Width = generic_array::typenum::U16;

    type i8xN = I8x16Wasm;
    type u8xN = U8x16Wasm;

    type i8x16 = I8x16Wasm;
    type u8x16 = U8x16Wasm;
}

impl Simd3 for Wasm {
    type usizex3 = <Self as Simd3A>::usizex3A;

    type f32x3 = <Self as Simd3A>::f32x3A;
    type i32x3 = <Self as Simd3A>::i32x3A;
    type u32x3 = <Self as Simd3A>::u32x3A;

    type f64x3 = <Self as Simd3A>::f64x3A;
    type i64x3 = <Self as Simd3A>::i64x3A;
    type u64x3 = <Self as Simd3A>::u64x3A;
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

    // 16-bit self + sibling (i16<->u16)
    I16x8Wasm as U16x8Wasm,
    U16x8Wasm as I16x8Wasm,
    I16x8Wasm as I16x8Wasm,
    U16x8Wasm as U16x8Wasm,

    // 8-bit self + sibling (i8<->u8)
    I8x16Wasm as U8x16Wasm,
    U8x16Wasm as I8x16Wasm,
    I8x16Wasm as I8x16Wasm,
    U8x16Wasm as U8x16Wasm,
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

    // 16-bit self + sibling (i16<->i32 widen/narrow live in i16x8.rs / half16.rs)
    I16x8Wasm as I16x8Wasm => identity,
    U16x8Wasm as U16x8Wasm => identity,
    I16x8Wasm as U16x8Wasm => identity,
    U16x8Wasm as I16x8Wasm => identity,

    // 8-bit self + sibling
    I8x16Wasm as I8x16Wasm => identity,
    U8x16Wasm as U8x16Wasm => identity,
    I8x16Wasm as U8x16Wasm => identity,
    U8x16Wasm as I8x16Wasm => identity,
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

impl_extend_same!(
    F32x4Wasm, I32x4Wasm, U32x4Wasm, F64x2Wasm, I64x2Wasm, U64x2Wasm, I16x8Wasm, U16x8Wasm, I8x16Wasm, U8x16Wasm
);
