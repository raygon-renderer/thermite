#![allow(non_camel_case_types)]

use generic_array::{
    ArrayLength,
    typenum::{U2, U4, U8, U16},
};
use num_traits::Signed;

use crate::{
    Vector,
    register::{
        BitsRegister, CastMaskRegister, CastRegister, FloatRegister, IntegerRegister, Interoperable, Lanes,
        MaskRegister, Register, SignedIntegerRegister, SignedRegister, UnsignedIntegerRegister,
    },
};

#[rustfmt::skip]
pub trait NativeSimd {
    /// Largest native 32-bit SIMD width
    type Native32Width: Lanes;

    /// Largest native 64-bit SIMD width
    type Native64Width: Lanes;

    // Largest Native 32-bit SIMD types
    type f32xN: Interoperable<Self::i32xN, Self::u32xN, Lanes = Self::Native32Width, Element = f32, USize = Self::u32xN, ISize = Self::i32xN>
        + FloatRegister;
    type i32xN: Interoperable<Self::f32xN, Self::u32xN, Lanes = Self::Native32Width, Element = i32, USize = Self::u32xN, ISize = Self::i32xN>
        + IntegerRegister + SignedRegister;
    type u32xN: Interoperable<Self::f32xN, Self::i32xN, Lanes = Self::Native32Width, Element = u32, USize = Self::u32xN, ISize = Self::i32xN>
        + UnsignedIntegerRegister;

    // Largest Native 64-bit SIMD types
    type f64xN: Interoperable<Self::i64xN, Self::u64xN, Lanes = Self::Native64Width, Element = f64, USize = Self::u64xN, ISize = Self::i64xN>
        + FloatRegister;
    type i64xN: Interoperable<Self::f64xN, Self::u64xN, Lanes = Self::Native64Width, Element = i64, USize = Self::u64xN, ISize = Self::i64xN>
        + IntegerRegister + SignedRegister;
    type u64xN: Interoperable<Self::f64xN, Self::i64xN, Lanes = Self::Native64Width, Element = u64, USize = Self::u64xN, ISize = Self::i64xN>
        + UnsignedIntegerRegister;
}

#[rustfmt::skip]
pub trait Simd: NativeSimd {
    // 128/32-bit SIMD types
    type f32x4: Interoperable<Self::i32x4, Self::u32x4, Lanes = U4, Element = f32, USize = Self::u32x4, ISize = Self::i32x4>
        + FloatRegister<Bits = Self::u32x4, Signed = Self::i32x4> + CastRegister<Self::f64x4>;
    type i32x4: Interoperable<Self::f32x4, Self::u32x4, Lanes = U4, Element = i32, USize = Self::u32x4, ISize = Self::i32x4>
        + SignedIntegerRegister + CastRegister<Self::i64x4>;
    type u32x4: Interoperable<Self::f32x4, Self::i32x4, Lanes = U4, Element = u32, USize = Self::u32x4, ISize = Self::i32x4>
        + UnsignedIntegerRegister + CastRegister<Self::u64x4>;

    // 256/32-bit SIMD types
    type f32x8: Interoperable<Self::i32x8, Self::u32x8, Lanes = U8, Element = f32, USize = Self::u32x8, ISize = Self::i32x8>
        + FloatRegister<Bits = Self::u32x8, Signed = Self::i32x8> + CastRegister<Self::f64x8>;
    type i32x8: Interoperable<Self::f32x8, Self::u32x8, Lanes = U8, Element = i32, USize = Self::u32x8, ISize = Self::i32x8>
        + SignedIntegerRegister + CastRegister<Self::i64x8>;
    type u32x8: Interoperable<Self::f32x8, Self::i32x8, Lanes = U8, Element = u32, USize = Self::u32x8, ISize = Self::i32x8>
        + UnsignedIntegerRegister + CastRegister<Self::u64x8>;

    // 128/64-bit SIMD types
    type f64x2: Interoperable<Self::i64x2, Self::u64x2, Lanes = U2, Element = f64, USize = Self::u64x2, ISize = Self::i64x2>
        + FloatRegister<Bits = Self::u64x2, Signed = Self::i64x2>;
    type i64x2: Interoperable<Self::f64x2, Self::u64x2, Lanes = U2, Element = i64, USize = Self::u64x2, ISize = Self::i64x2>
        + SignedIntegerRegister;
    type u64x2: Interoperable<Self::f64x2, Self::i64x2, Lanes = U2, Element = u64, USize = Self::u64x2, ISize = Self::i64x2>
        + UnsignedIntegerRegister;

    // 256/64-bit SIMD types
    type f64x4: Interoperable<Self::i64x4, Self::u64x4, Lanes = U4, Element = f64, USize = Self::u64x4, ISize = Self::i64x4>
        + FloatRegister<Bits = Self::u64x4, Signed = Self::i64x4> + CastRegister<Self::f32x4>;
    type i64x4: Interoperable<Self::f64x4, Self::u64x4, Lanes = U4, Element = i64, USize = Self::u64x4, ISize = Self::i64x4>
        + SignedIntegerRegister + CastRegister<Self::i32x4>;
    type u64x4: Interoperable<Self::f64x4, Self::i64x4, Lanes = U4, Element = u64, USize = Self::u64x4, ISize = Self::i64x4>
        + UnsignedIntegerRegister + CastRegister<Self::u32x4>;

    // 512/64-bit SIMD types
    type f64x8: Interoperable<Self::i64x8, Self::u64x8, Lanes = U8, Element = f64, USize = Self::u64x8, ISize = Self::i64x8>
        + FloatRegister<Bits = Self::u64x8, Signed = Self::i64x8> + CastRegister<Self::f32x8>;
    type i64x8: Interoperable<Self::f64x8, Self::u64x8, Lanes = U8, Element = i64, USize = Self::u64x8, ISize = Self::i64x8>
        + SignedIntegerRegister + CastRegister<Self::i32x8>;
    type u64x8: Interoperable<Self::f64x8, Self::i64x8, Lanes = U8, Element = u64, USize = Self::u64x8, ISize = Self::i64x8>
        + UnsignedIntegerRegister + CastRegister<Self::u32x8>;

    // 512/32-bit SIMD types
    type f32x16: Interoperable<Self::i32x16, Self::u32x16, Lanes = U16, Element = f32, USize = Self::u32x16, ISize = Self::i32x16>
        + FloatRegister<Bits = Self::u32x16, Signed = Self::i32x16> + CastRegister<Self::f64x16>;
    type i32x16: Interoperable<Self::f32x16, Self::u32x16, Lanes = U16, Element = i32, USize = Self::u32x16, ISize = Self::i32x16>
        + SignedIntegerRegister + CastRegister<Self::i64x16>;
    type u32x16: Interoperable<Self::f32x16, Self::i32x16, Lanes = U16, Element = u32, USize = Self::u32x16, ISize = Self::i32x16>
        + UnsignedIntegerRegister + CastRegister<Self::u64x16>;

    // 1024/64-bit SIMD types
    type f64x16: Interoperable<Self::i64x16, Self::u64x16, Lanes = U16, Element = f64, USize = Self::u64x16, ISize = Self::i64x16>
        + FloatRegister<Bits = Self::u64x16, Signed = Self::i64x16> + CastRegister<Self::f32x16>;
    type i64x16: Interoperable<Self::f64x16, Self::u64x16, Lanes = U16, Element = i64, USize = Self::u64x16, ISize = Self::i64x16>
        + IntegerRegister + SignedRegister + CastRegister<Self::i32x16>;
    type u64x16: Interoperable<Self::f64x16, Self::i64x16, Lanes = U16, Element = u64, USize = Self::u64x16, ISize = Self::i64x16>
        + UnsignedIntegerRegister + CastRegister<Self::u32x16>;
}

pub type f32xN<S> = Vector<<S as NativeSimd>::f32xN>;
pub type i32xN<S> = Vector<<S as NativeSimd>::i32xN>;
pub type u32xN<S> = Vector<<S as NativeSimd>::u32xN>;

pub type f64xN<S> = Vector<<S as NativeSimd>::f64xN>;
pub type i64xN<S> = Vector<<S as NativeSimd>::i64xN>;
pub type u64xN<S> = Vector<<S as NativeSimd>::u64xN>;

pub type f32x4<S> = Vector<<S as Simd>::f32x4>;
pub type i32x4<S> = Vector<<S as Simd>::i32x4>;
pub type u32x4<S> = Vector<<S as Simd>::u32x4>;

pub type f32x8<S> = Vector<<S as Simd>::f32x8>;
pub type i32x8<S> = Vector<<S as Simd>::i32x8>;
pub type u32x8<S> = Vector<<S as Simd>::u32x8>;

pub type f64x2<S> = Vector<<S as Simd>::f64x2>;
pub type i64x2<S> = Vector<<S as Simd>::i64x2>;
pub type u64x2<S> = Vector<<S as Simd>::u64x2>;

pub type f64x4<S> = Vector<<S as Simd>::f64x4>;
pub type i64x4<S> = Vector<<S as Simd>::i64x4>;
pub type u64x4<S> = Vector<<S as Simd>::u64x4>;

pub type f64x8<S> = Vector<<S as Simd>::f64x8>;
pub type i64x8<S> = Vector<<S as Simd>::i64x8>;
pub type u64x8<S> = Vector<<S as Simd>::u64x8>;

pub type f32x16<S> = Vector<<S as Simd>::f32x16>;
pub type i32x16<S> = Vector<<S as Simd>::i32x16>;
pub type u32x16<S> = Vector<<S as Simd>::u32x16>;

pub type f64x16<S> = Vector<<S as Simd>::f64x16>;
pub type i64x16<S> = Vector<<S as Simd>::i64x16>;
pub type u64x16<S> = Vector<<S as Simd>::u64x16>;

macro_rules! decl_aliases {
    ($simd:ty) => {
        #[allow(non_camel_case_types)]
        pub mod aliases {
            use super::*;

            pub type f32xN = crate::simd::f32xN<$simd>;
            pub type i32xN = crate::simd::i32xN<$simd>;
            pub type u32xN = crate::simd::u32xN<$simd>;

            pub type f64xN = crate::simd::f64xN<$simd>;
            pub type i64xN = crate::simd::i64xN<$simd>;
            pub type u64xN = crate::simd::u64xN<$simd>;

            pub type f32x4 = crate::simd::f32x4<$simd>;
            pub type i32x4 = crate::simd::i32x4<$simd>;
            pub type u32x4 = crate::simd::u32x4<$simd>;

            pub type f32x8 = crate::simd::f32x8<$simd>;
            pub type i32x8 = crate::simd::i32x8<$simd>;
            pub type u32x8 = crate::simd::u32x8<$simd>;

            pub type f64x2 = crate::simd::f64x2<$simd>;
            pub type i64x2 = crate::simd::i64x2<$simd>;
            pub type u64x2 = crate::simd::u64x2<$simd>;

            pub type f64x4 = crate::simd::f64x4<$simd>;
            pub type i64x4 = crate::simd::i64x4<$simd>;
            pub type u64x4 = crate::simd::u64x4<$simd>;

            pub type f64x8 = crate::simd::f64x8<$simd>;
            pub type i64x8 = crate::simd::i64x8<$simd>;
            pub type u64x8 = crate::simd::u64x8<$simd>;

            pub type f32x16 = crate::simd::f32x16<$simd>;
            pub type i32x16 = crate::simd::i32x16<$simd>;
            pub type u32x16 = crate::simd::u32x16<$simd>;

            pub type f64x16 = crate::simd::f64x16<$simd>;
            pub type i64x16 = crate::simd::i64x16<$simd>;
            pub type u64x16 = crate::simd::u64x16<$simd>;
        }
    };
}
