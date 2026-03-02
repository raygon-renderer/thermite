#![allow(clippy::useless_transmute, unnecessary_transmutes)]

pub mod float;
pub mod signed;
pub mod unsigned;

use crate::{
    element::USize,
    isa::InstructionSet,
    register::{Element, ExtendRegister, MaskElement, Storage, dp::DoublePumpRegister},
    simd::{HasIsa, NativeIsa, NativeSimd, Simd},
};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Scalar;

pub mod prelude {
    pub use super::Scalar;
    pub use super::aliases::*;
    pub use crate::prelude::*;
}

impl HasIsa for Scalar {
    const ISA: InstructionSet = InstructionSet::Scalar;
}

impl NativeIsa for Scalar {
    type Registers = generic_array::typenum::U16;

    type Native32Width = generic_array::typenum::U1;
    type Native64Width = generic_array::typenum::U1;

    type NativeAlignment = ();
}

impl NativeSimd for Scalar {
    type f32xN = f32;
    type i32xN = i32;
    type u32xN = u32;

    type f64xN = f64;
    type i64xN = i64;
    type u64xN = u64;
}

impl Simd for Scalar {
    type usizex2 = DoublePumpRegister<USize>;
    type usizex4 = DoublePumpRegister<Self::usizex2>;
    type usizex8 = DoublePumpRegister<Self::usizex4>;
    type usizex16 = DoublePumpRegister<Self::usizex8>;

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

macro_rules! impl_extends {
    ($($ty:ty),* $(,)?) => {$( impl ExtendRegister<$ty> for $ty {
        #[inline(always)]
        fn extend(value: Storage<$ty>) -> Storage<Self> {
            value
        }

        #[inline(always)]
        fn narrow(value: Storage<Self>) -> Storage<$ty> {
            value
        }
    })*};
}

impl_extends!(f32, i32, u32, f64, i64, u64);

macro_rules! impl_easy_casts {
    ($($from:ty as ($($to:ty),+)),* $(,)?) => {$(
        $(
            impl $crate::register::BitCastRegister<$from> for $to {
                #[inline(always)]
                fn from_bits(value: Storage<$from>) -> Storage<Self> {
                    unsafe { core::mem::transmute(value) }
                }
            }

            impl $crate::register::CastRegister<$from> for $to {
                #[inline(always)]
                fn cast_from(value: Storage<$from>) -> Storage<Self> {
                    unsafe { value as _ } // built-in cast
                }
            }

            impl $crate::register::CastMaskRegister<$from> for $to {
                #[inline(always)]
                fn mask_from(value: Storage<$from>) -> Storage<Self> {
                    unsafe { core::mem::transmute(value) }
                }
            }
        )+
    )*};
}

macro_rules! impl_nontrivial_casts {
    ($($from:ty as ($($to:ty),+)),* $(,)?) => {$(
        $(
            impl $crate::register::CastRegister<$from> for $to {
                #[inline(always)]
                fn cast_from(value: Storage<$from>) -> Storage<Self> {
                    unsafe { value as _ } // built-in cast
                }
            }

            impl $crate::register::CastMaskRegister<$from> for $to {
                #[inline(always)]
                fn mask_from(value: Storage<$from>) -> Storage<Self> {
                    MaskElement::from_bool(value.to_bool())
                }
            }
        )+
    )*};
}

impl_easy_casts! {
    i8  as (i8,  u8),
    u8  as (u8,  i8),
    i16 as (i16, u16),
    u16 as (u16, i16),

    f32 as (f32, i32, u32),
    i32 as (f32, u32, i32),
    u32 as (f32, i32, u32),

    f64 as (f64, i64, u64),
    i64 as (f64, u64, i64),
    u64 as (f64, i64, u64),
}

// all different-sized casts
impl_nontrivial_casts! {
    //     (f32, f64, i8, i16, i32, i64, u8, u16, u32, u64)
    i8  as (f32, f64,     i16, i32, i64,     u16, u32, u64),
    u8  as (f32, f64,     i16, i32, i64,     u16, u32, u64),
    u16 as (f32, f64, i8,      i32, i64, u8,      u32, u64),
    i16 as (f32, f64, i8,      i32, i64, u8,      u32, u64),
    f32 as (     f64, i8, i16,      i64, u8, u16,      u64),
    i32 as (     f64, i8, i16,      i64, u8, u16,      u64),
    u32 as (     f64, i8, i16,      i64, u8, u16,      u64),
    f64 as (f32,      i8, i16, i32,      u8, u16, u32     ),
    i64 as (f32,      i8, i16, i32,      u8, u16, u32     ),
    u64 as (f32,      i8, i16, i32,      u8, u16, u32     ),
}
