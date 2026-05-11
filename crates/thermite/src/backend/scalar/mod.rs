#![allow(clippy::useless_transmute, unnecessary_transmutes)]

use crate::{
    element::USize,
    isa::InstructionSet,
    register::{Element, ExtendRegister, MaskElement, Storage, array::ArrayRegister},
    simd::{HasIsa, NativeIsa, NativeSimd, Simd},
};

cfg_if::cfg_if! {
    if #[cfg(all(feature = "spirv", target_arch = "spirv"))] {
        #[path = "spirv/mod.rs"]
        mod registers;
    } else {
        #[path = "cpu/mod.rs"]
        mod registers;
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Scalar;

pub mod prelude {
    pub use super::Scalar;
    pub use super::aliases::*;
    pub use crate::prelude::*;
}

#[thermite_macros::inline_always]
impl NativeIsa for Scalar {
    type Registers = generic_array::typenum::U16;

    type Native32Width = generic_array::typenum::U1;
    type Native64Width = generic_array::typenum::U1;

    type NativeAlignment = ();

    #[cfg(all(any(target_arch = "x86", target_arch = "x86_64"), target_feature = "sse2"))]
    unsafe fn disable_denormals() -> Result<bool, crate::simd::UnsupportedError> {
        unsafe { Ok(crate::backend::x86::sse2::disable_denormals()) }
    }

    #[cfg(all(any(target_arch = "x86", target_arch = "x86_64"), target_feature = "sse2"))]
    #[allow(clippy::unit_arg)]
    unsafe fn enable_denormals() -> Result<(), crate::simd::UnsupportedError> {
        unsafe { Ok(crate::backend::x86::sse2::enable_denormals()) }
    }
}

#[thermite_macros::inline_always]
impl NativeSimd for Scalar {
    type f32xN = f32;
    type i32xN = i32;
    type u32xN = u32;

    type f64xN = f64;
    type i64xN = i64;
    type u64xN = u64;
}

#[thermite_macros::inline_always]
impl Simd for Scalar {
    type usizex2 = ArrayRegister<USize, 2>;
    type usizex4 = ArrayRegister<USize, 4>;
    type usizex8 = ArrayRegister<USize, 8>;
    type usizex16 = ArrayRegister<USize, 16>;

    type f32x2 = ArrayRegister<f32, 2>;
    type i32x2 = ArrayRegister<i32, 2>;
    type u32x2 = ArrayRegister<u32, 2>;

    type f32x4 = ArrayRegister<f32, 4>;
    type i32x4 = ArrayRegister<i32, 4>;
    type u32x4 = ArrayRegister<u32, 4>;

    type f32x8 = ArrayRegister<f32, 8>;
    type i32x8 = ArrayRegister<i32, 8>;
    type u32x8 = ArrayRegister<u32, 8>;

    type f64x2 = ArrayRegister<f64, 2>;
    type i64x2 = ArrayRegister<i64, 2>;
    type u64x2 = ArrayRegister<u64, 2>;

    type f64x4 = ArrayRegister<f64, 4>;
    type i64x4 = ArrayRegister<i64, 4>;
    type u64x4 = ArrayRegister<u64, 4>;

    type f64x8 = ArrayRegister<f64, 8>;
    type i64x8 = ArrayRegister<i64, 8>;
    type u64x8 = ArrayRegister<u64, 8>;

    type f32x16 = ArrayRegister<f32, 16>;
    type i32x16 = ArrayRegister<i32, 16>;
    type u32x16 = ArrayRegister<u32, 16>;

    type f64x16 = ArrayRegister<f64, 16>;
    type i64x16 = ArrayRegister<i64, 16>;
    type u64x16 = ArrayRegister<u64, 16>;
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

impl_newregister!(f32, i32, u32, f64, i64, u64);

impl_newregister!(u8, u16, i8, i16); // extra integers
