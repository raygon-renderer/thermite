#![allow(clippy::useless_transmute, unnecessary_transmutes)]

pub mod float;
pub mod signed;
pub mod unsigned;

use crate::{
    isa::InstructionSet,
    register::{Element, Storage, dp::DoublePumpRegister},
    simd::{NativeSimd, Simd},
};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Scalar;

impl NativeSimd for Scalar {
    const ISA: InstructionSet = InstructionSet::Scalar;

    type Registers = generic_array::typenum::U16;

    type Native32Width = generic_array::typenum::U1;
    type Native64Width = generic_array::typenum::U1;

    type NativeAlignment = ();

    type f32xN = f32;
    type i32xN = i32;
    type u32xN = u32;

    type f64xN = f64;
    type i64xN = i64;
    type u64xN = u64;
}

impl Simd for Scalar {
    type f32x2 = DoublePumpRegister<f32>;
    type i32x2 = DoublePumpRegister<i32>;
    type u32x2 = DoublePumpRegister<u32>;

    type f32x4 = DoublePumpRegister<Self::f32x2>;
    type i32x4 = DoublePumpRegister<Self::i32x2>;
    type u32x4 = DoublePumpRegister<Self::u32x2>;

    type f32x8 = DoublePumpRegister<Self::f32x4>;
    type i32x8 = DoublePumpRegister<Self::i32x4>;
    type u32x8 = DoublePumpRegister<Self::u32x4>;

    type f64x2 = DoublePumpRegister<f64>;
    type i64x2 = DoublePumpRegister<i64>;
    type u64x2 = DoublePumpRegister<u64>;

    type f64x4 = DoublePumpRegister<Self::f64x2>;
    type i64x4 = DoublePumpRegister<Self::i64x2>;
    type u64x4 = DoublePumpRegister<Self::u64x2>;

    type f64x8 = DoublePumpRegister<Self::f64x4>;
    type i64x8 = DoublePumpRegister<Self::i64x4>;
    type u64x8 = DoublePumpRegister<Self::u64x4>;

    type f32x16 = DoublePumpRegister<Self::f32x8>;
    type i32x16 = DoublePumpRegister<Self::i32x8>;
    type u32x16 = DoublePumpRegister<Self::u32x8>;

    type f64x16 = DoublePumpRegister<Self::f64x8>;
    type i64x16 = DoublePumpRegister<Self::i64x8>;
    type u64x16 = DoublePumpRegister<Self::u64x8>;
}

decl_aliases!(Scalar);
pub use self::aliases::*;

macro_rules! impl_easy_casts {
    ($($from:ty as $to:ty),*) => {$(
        impl $crate::register::BitsRegister<$from> for $to {
            #[inline(always)]
            fn from_bits(value: Storage<$from>) -> Storage<Self> {
                unsafe { core::mem::transmute(value) }
            }
        }

        impl $crate::register::CastRegister<$from> for $to {
            #[inline(always)]
            fn cast_from(value: Storage<$from>) -> Storage<Self> {
                unsafe { value as _ }
            }
        }

        impl $crate::register::CastMaskRegister<$from> for $to {
            #[inline(always)]
            fn mask_from(value: Storage<$from>) -> Storage<Self> {
                unsafe { core::mem::transmute(value) }
            }
        }
    )*};
}

macro_rules! impl_nontrivial_casts {
    ($($from:ty as $to:ty),* $(,)?) => {$(
        impl $crate::register::CastRegister<$from> for $to {
            #[inline(always)]
            fn cast_from(value: Storage<$from>) -> Storage<Self> {
                unsafe { value as _ }
            }
        }

        impl $crate::register::CastMaskRegister<$from> for $to {
            #[inline(always)]
            fn mask_from(value: Storage<$from>) -> Storage<Self> {
                Element::from_bool(value.to_bool())
            }
        }
    )*};
}

impl_easy_casts! {
    f32 as i32,
    f32 as u32,
    f32 as f32,

    i32 as f32,
    i32 as u32,
    i32 as i32,

    u32 as f32,
    u32 as i32,
    u32 as u32,

    i64 as f64,
    i64 as u64,
    i64 as i64,

    u64 as f64,
    u64 as i64,
    u64 as u64,

    f64 as i64,
    f64 as u64,
    f64 as f64
}

impl_nontrivial_casts! {
    f32 as i64,
    f32 as u64,
    f32 as f64,

    i64 as f32,
    i64 as i32,
    i64 as u32,

    u32 as f64,
    u32 as u64,
    u32 as i64,

    f64 as f32,
    f64 as i32,
    f64 as u32,

    i32 as f64,
    i32 as i64,
    i32 as u64,

    u64 as f32,
    u64 as u32,
    u64 as i32
}
