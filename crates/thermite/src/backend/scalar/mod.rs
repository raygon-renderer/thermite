#![allow(clippy::useless_transmute, unnecessary_transmutes)]

pub mod float;
pub mod signed;
pub mod unsigned;

pub use float::{F32x1Scalar, F64x1Scalar};
pub use signed::{I32x1Scalar, I64x1Scalar};
pub use unsigned::{U32x1Scalar, U64x1Scalar};

use crate::{
    register::{Element, dp::DoublePumpRegister},
    simd::{NativeSimd, Simd},
};

pub struct Scalar;

impl NativeSimd for Scalar {
    type Native32Width = generic_array::typenum::U1;
    type Native64Width = generic_array::typenum::U1;

    type f32xN = F32x1Scalar;
    type i32xN = I32x1Scalar;
    type u32xN = U32x1Scalar;

    type f64xN = F64x1Scalar;
    type i64xN = I64x1Scalar;
    type u64xN = U64x1Scalar;
}

impl Simd for Scalar {
    type f32x4 = DoublePumpRegister<DoublePumpRegister<F32x1Scalar>>;
    type i32x4 = DoublePumpRegister<DoublePumpRegister<I32x1Scalar>>;
    type u32x4 = DoublePumpRegister<DoublePumpRegister<U32x1Scalar>>;

    type f32x8 = DoublePumpRegister<Self::f32x4>;
    type i32x8 = DoublePumpRegister<Self::i32x4>;
    type u32x8 = DoublePumpRegister<Self::u32x4>;

    type f64x2 = DoublePumpRegister<F64x1Scalar>;
    type i64x2 = DoublePumpRegister<I64x1Scalar>;
    type u64x2 = DoublePumpRegister<U64x1Scalar>;

    type f64x4 = DoublePumpRegister<Self::f64x2>;
    type i64x4 = DoublePumpRegister<Self::i64x2>;
    type u64x4 = DoublePumpRegister<Self::u64x2>;

    type f64x8 = DoublePumpRegister<Self::f64x4>;
    type i64x8 = DoublePumpRegister<Self::i64x4>;
    type u64x8 = DoublePumpRegister<Self::u64x4>;
}

decl_aliases!(Scalar);
pub use self::aliases::*;

macro_rules! impl_easy_casts {
    ($($from:ty as $to:ty),*) => {$(
        impl $crate::register::BitsRegister<$from> for $to {
            #[inline(always)]
            fn from_bits(value: <$from as $crate::register::Register>::Storage) -> Self::Storage {
                unsafe { core::mem::transmute(value) }
            }
        }

        impl $crate::register::CastRegister<$from> for $to {
            #[inline(always)]
            fn cast_from(value: <$from as $crate::register::Register>::Storage) -> Self::Storage {
                unsafe { value as _ }
            }
        }

        impl $crate::register::CastMaskRegister<$from> for $to {
            #[inline(always)]
            fn mask_from(value: <$from as $crate::register::Register>::Storage) -> Self::Storage {
                unsafe { core::mem::transmute(value) }
            }
        }
    )*};
}

macro_rules! impl_nontrivial_casts {
    ($($from:ty as $to:ty),* $(,)?) => {$(
        impl $crate::register::CastRegister<$from> for $to {
            #[inline(always)]
            fn cast_from(value: <$from as $crate::register::Register>::Storage) -> Self::Storage {
                unsafe { value as _ }
            }
        }

        impl $crate::register::CastMaskRegister<$from> for $to {
            #[inline(always)]
            fn mask_from(value: <$from as $crate::register::Register>::Storage) -> Self::Storage {
                Element::from_bool(value.to_bool())
            }
        }
    )*};
}

impl_easy_casts! {
    F32x1Scalar as I32x1Scalar,
    F32x1Scalar as U32x1Scalar,
    F32x1Scalar as F32x1Scalar,

    I32x1Scalar as F32x1Scalar,
    I32x1Scalar as U32x1Scalar,
    I32x1Scalar as I32x1Scalar,

    U32x1Scalar as F32x1Scalar,
    U32x1Scalar as I32x1Scalar,
    U32x1Scalar as U32x1Scalar,

    I64x1Scalar as F64x1Scalar,
    I64x1Scalar as U64x1Scalar,
    I64x1Scalar as I64x1Scalar,

    U64x1Scalar as F64x1Scalar,
    U64x1Scalar as I64x1Scalar,
    U64x1Scalar as U64x1Scalar,

    F64x1Scalar as I64x1Scalar,
    F64x1Scalar as U64x1Scalar,
    F64x1Scalar as F64x1Scalar
}

impl_nontrivial_casts! {
    F32x1Scalar as I64x1Scalar,
    F32x1Scalar as U64x1Scalar,
    F32x1Scalar as F64x1Scalar,

    I64x1Scalar as F32x1Scalar,
    I64x1Scalar as I32x1Scalar,
    I64x1Scalar as U32x1Scalar,

    U32x1Scalar as F64x1Scalar,
    U32x1Scalar as U64x1Scalar,
    U32x1Scalar as I64x1Scalar,

    F64x1Scalar as F32x1Scalar,
    F64x1Scalar as I32x1Scalar,
    F64x1Scalar as U32x1Scalar,

    I32x1Scalar as F64x1Scalar,
    I32x1Scalar as I64x1Scalar,
    I32x1Scalar as U64x1Scalar,

    U64x1Scalar as F32x1Scalar,
    U64x1Scalar as U32x1Scalar,
    U64x1Scalar as I32x1Scalar
}
