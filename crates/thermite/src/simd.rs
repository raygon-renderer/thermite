#![allow(non_camel_case_types)]

use generic_array::{
    ArrayLength,
    typenum::{U1, U2, U4, U8, U16},
};
use num_traits::Signed;

use crate::{
    Vector,
    isa::InstructionSet,
    register::{
        BitsRegister, CastMaskRegister, CastRegister, FloatElement, FloatRegister, IntegerRegister, Interoperable,
        Lanes, LinAlg4Register, MaskRegister, NarrowRegister, Register, SignedIntegerRegister, SignedRegister,
        UnsignedIntegerRegister, WidenRegister, element::IntegerElement,
    },
};

/// Native-width SIMD types supported directly by the target architecture.
#[rustfmt::skip]
pub trait NativeSimd {
    const ISA: InstructionSet;

    type Registers: ArrayLength;

    /// Largest native 32-bit SIMD width
    type Native32Width: Lanes;

    /// Largest native 64-bit SIMD width
    type Native64Width: Lanes;

    // Largest Native 32-bit SIMD types
    type f32xN: Interoperable<Self::i32xN, Self::u32xN, Lanes = Self::Native32Width, Element = f32, USize = Self::u32xN, ISize = Self::i32xN>
        + FloatRegister<Bits = Self::u32xN, Signed = Self::i32xN>;
    type i32xN: Interoperable<Self::f32xN, Self::u32xN, Lanes = Self::Native32Width, Element = i32, USize = Self::u32xN, ISize = Self::i32xN>
        + SignedIntegerRegister;
    type u32xN: Interoperable<Self::f32xN, Self::i32xN, Lanes = Self::Native32Width, Element = u32, USize = Self::u32xN, ISize = Self::i32xN>
        + UnsignedIntegerRegister;

    // Largest Native 64-bit SIMD types
    type f64xN: Interoperable<Self::i64xN, Self::u64xN, Lanes = Self::Native64Width, Element = f64, USize = Self::u64xN, ISize = Self::i64xN>
        + FloatRegister<Bits = Self::u64xN, Signed = Self::i64xN>;
    type i64xN: Interoperable<Self::f64xN, Self::u64xN, Lanes = Self::Native64Width, Element = i64, USize = Self::u64xN, ISize = Self::i64xN>
        + SignedIntegerRegister;
    type u64xN: Interoperable<Self::f64xN, Self::i64xN, Lanes = Self::Native64Width, Element = u64, USize = Self::u64xN, ISize = Self::i64xN>
        + UnsignedIntegerRegister;
}

/// Fixed-size SIMD types of various lane counts and element sizes.
#[rustfmt::skip]
pub trait Simd: NativeSimd {
    // 64/32-bit SIMD types, almost always composite of scalar types
    type f32x2: Interoperable<Self::i32x2, Self::u32x2, Lanes = U2, Element = f32, USize = Self::u32x2, ISize = Self::i32x2>
        + FloatRegister<Bits = Self::u32x2, Signed = Self::i32x2>
        + CastRegister<Self::f64x2> + WidenRegister<f32> + NarrowRegister<f32>;
    type i32x2: Interoperable<Self::f32x2, Self::u32x2, Lanes = U2, Element = i32, USize = Self::u32x2, ISize = Self::i32x2>
        + SignedIntegerRegister
        + CastRegister<Self::i64x2> + WidenRegister<i32> + NarrowRegister<i32>;
    type u32x2: Interoperable<Self::f32x2, Self::i32x2, Lanes = U2, Element = u32, USize = Self::u32x2, ISize = Self::i32x2>
        + UnsignedIntegerRegister
        + CastRegister<Self::u64x2> + WidenRegister<u32> + NarrowRegister<u32>;

    // 128/32-bit SIMD types
    type f32x4: Interoperable<Self::i32x4, Self::u32x4, Lanes = U4, Element = f32, USize = Self::u32x4, ISize = Self::i32x4>
        + FloatRegister<Bits = Self::u32x4, Signed = Self::i32x4> + LinAlg4Register
        + CastRegister<Self::f64x4> + WidenRegister<Self::f32x2> + NarrowRegister<Self::f32x2>;
    type i32x4: Interoperable<Self::f32x4, Self::u32x4, Lanes = U4, Element = i32, USize = Self::u32x4, ISize = Self::i32x4>
        + SignedIntegerRegister
        + CastRegister<Self::i64x4> + WidenRegister<Self::i32x2> + NarrowRegister<Self::i32x2>;
    type u32x4: Interoperable<Self::f32x4, Self::i32x4, Lanes = U4, Element = u32, USize = Self::u32x4, ISize = Self::i32x4>
        + UnsignedIntegerRegister
        + CastRegister<Self::u64x4> + WidenRegister<Self::u32x2> + NarrowRegister<Self::u32x2>;

    // 256/32-bit SIMD types
    type f32x8: Interoperable<Self::i32x8, Self::u32x8, Lanes = U8, Element = f32, USize = Self::u32x8, ISize = Self::i32x8>
        + FloatRegister<Bits = Self::u32x8, Signed = Self::i32x8>
        + CastRegister<Self::f64x8> + WidenRegister<Self::f32x4> + NarrowRegister<Self::f32x4>;
    type i32x8: Interoperable<Self::f32x8, Self::u32x8, Lanes = U8, Element = i32, USize = Self::u32x8, ISize = Self::i32x8>
        + SignedIntegerRegister
        + CastRegister<Self::i64x8> + WidenRegister<Self::i32x4> + NarrowRegister<Self::i32x4>;
    type u32x8: Interoperable<Self::f32x8, Self::i32x8, Lanes = U8, Element = u32, USize = Self::u32x8, ISize = Self::i32x8>
        + UnsignedIntegerRegister
        + CastRegister<Self::u64x8> + WidenRegister<Self::u32x4> + NarrowRegister<Self::u32x4>;

    // 128/64-bit SIMD types
    type f64x2: Interoperable<Self::i64x2, Self::u64x2, Lanes = U2, Element = f64, USize = Self::u64x2, ISize = Self::i64x2>
        + FloatRegister<Bits = Self::u64x2, Signed = Self::i64x2>
        + CastRegister<Self::f32x2> + WidenRegister<f64> + NarrowRegister<f64>;
    type i64x2: Interoperable<Self::f64x2, Self::u64x2, Lanes = U2, Element = i64, USize = Self::u64x2, ISize = Self::i64x2>
        + SignedIntegerRegister
        + CastRegister<Self::i32x2> + WidenRegister<i64> + NarrowRegister<i64>;
    type u64x2: Interoperable<Self::f64x2, Self::i64x2, Lanes = U2, Element = u64, USize = Self::u64x2, ISize = Self::i64x2>
        + UnsignedIntegerRegister
        + CastRegister<Self::u32x2> + WidenRegister<u64> + NarrowRegister<u64>;

    // 256/64-bit SIMD types
    type f64x4: Interoperable<Self::i64x4, Self::u64x4, Lanes = U4, Element = f64, USize = Self::u64x4, ISize = Self::i64x4>
        + FloatRegister<Bits = Self::u64x4, Signed = Self::i64x4> + LinAlg4Register
        + CastRegister<Self::f32x4> + WidenRegister<Self::f64x2> + NarrowRegister<Self::f64x2>;
    type i64x4: Interoperable<Self::f64x4, Self::u64x4, Lanes = U4, Element = i64, USize = Self::u64x4, ISize = Self::i64x4>
        + SignedIntegerRegister
        + CastRegister<Self::i32x4> + WidenRegister<Self::i64x2> + NarrowRegister<Self::i64x2>;
    type u64x4: Interoperable<Self::f64x4, Self::i64x4, Lanes = U4, Element = u64, USize = Self::u64x4, ISize = Self::i64x4>
        + UnsignedIntegerRegister
        + CastRegister<Self::u32x4> + WidenRegister<Self::u64x2> + NarrowRegister<Self::u64x2>;

    // 512/64-bit SIMD types
    type f64x8: Interoperable<Self::i64x8, Self::u64x8, Lanes = U8, Element = f64, USize = Self::u64x8, ISize = Self::i64x8>
        + FloatRegister<Bits = Self::u64x8, Signed = Self::i64x8>
        + CastRegister<Self::f32x8> + WidenRegister<Self::f64x4> + NarrowRegister<Self::f64x4>;
    type i64x8: Interoperable<Self::f64x8, Self::u64x8, Lanes = U8, Element = i64, USize = Self::u64x8, ISize = Self::i64x8>
        + SignedIntegerRegister
        + CastRegister<Self::i32x8> + WidenRegister<Self::i64x4> + NarrowRegister<Self::i64x4>;
    type u64x8: Interoperable<Self::f64x8, Self::i64x8, Lanes = U8, Element = u64, USize = Self::u64x8, ISize = Self::i64x8>
        + UnsignedIntegerRegister
        + CastRegister<Self::u32x8> + WidenRegister<Self::u64x4> + NarrowRegister<Self::u64x4>;

    // 512/32-bit SIMD types
    type f32x16: Interoperable<Self::i32x16, Self::u32x16, Lanes = U16, Element = f32, USize = Self::u32x16, ISize = Self::i32x16>
        + FloatRegister<Bits = Self::u32x16, Signed = Self::i32x16>
        + CastRegister<Self::f64x16> + WidenRegister<Self::f32x8> + NarrowRegister<Self::f32x8>;
    type i32x16: Interoperable<Self::f32x16, Self::u32x16, Lanes = U16, Element = i32, USize = Self::u32x16, ISize = Self::i32x16>
        + SignedIntegerRegister
        + CastRegister<Self::i64x16> + WidenRegister<Self::i32x8> + NarrowRegister<Self::i32x8>;
    type u32x16: Interoperable<Self::f32x16, Self::i32x16, Lanes = U16, Element = u32, USize = Self::u32x16, ISize = Self::i32x16>
        + UnsignedIntegerRegister
        + CastRegister<Self::u64x16> + WidenRegister<Self::u32x8> + NarrowRegister<Self::u32x8>;

    // 1024/64-bit SIMD types
    type f64x16: Interoperable<Self::i64x16, Self::u64x16, Lanes = U16, Element = f64, USize = Self::u64x16, ISize = Self::i64x16>
        + FloatRegister<Bits = Self::u64x16, Signed = Self::i64x16>
        + CastRegister<Self::f32x16> + WidenRegister<Self::f64x8> + NarrowRegister<Self::f64x8>;
    type i64x16: Interoperable<Self::f64x16, Self::u64x16, Lanes = U16, Element = i64, USize = Self::u64x16, ISize = Self::i64x16>
        + SignedIntegerRegister
        + CastRegister<Self::i32x16> + WidenRegister<Self::i64x8> + NarrowRegister<Self::i64x8>;
    type u64x16: Interoperable<Self::f64x16, Self::i64x16, Lanes = U16, Element = u64, USize = Self::u64x16, ISize = Self::i64x16>
        + UnsignedIntegerRegister
        + CastRegister<Self::u32x16> + WidenRegister<Self::u64x8> + NarrowRegister<Self::u64x8>;
}

// TODO: Move these "fully formed" traits to another module? Also work on their naming.

/// A FloatElement type that can itself be used as a FloatRegister,
/// with its associated Signed and Bits types also being fully formed.
pub trait FullyFormedFloatElement:
    FloatElement<
        Signed: FullyFormedSignedIntegerElement<
            Element = <Self as FloatElement>::Signed,
            Storage = <Self as FloatElement>::Signed,
        >,
        Bits: FullyFormedUnsignedIntegerElement<
            Element = <Self as FloatElement>::Bits,
            Storage = <Self as FloatElement>::Bits,
        >,
    > + FloatRegister<Element = Self, Storage = Self>
{
}

/// A Signed IntegerElement type that can itself be used as a SignedIntegerRegister.
pub trait FullyFormedSignedIntegerElement:
    IntegerElement
    + IntegerRegister<Element = Self, Storage = Self>
    + Signed
    + SignedIntegerRegister<Element = Self, Storage = Self>
{
}

/// An Unsigned IntegerElement type that can itself be used as an UnsignedIntegerRegister.
pub trait FullyFormedUnsignedIntegerElement:
    IntegerElement
    + IntegerRegister<Element = Self, Storage = Self>
    + UnsignedIntegerRegister<Element = Self, Storage = Self>
{
}

impl<F> FullyFormedFloatElement for F where
    F: FloatElement<
            Signed: FullyFormedSignedIntegerElement<
                Element = <F as FloatElement>::Signed,
                Storage = <F as FloatElement>::Signed,
            >,
            Bits: FullyFormedUnsignedIntegerElement<
                Element = <F as FloatElement>::Bits,
                Storage = <F as FloatElement>::Bits,
            >,
        > + FloatRegister<Element = F, Storage = F>
{
}

impl<I> FullyFormedSignedIntegerElement for I where
    I: IntegerElement
        + IntegerRegister<Element = I, Storage = I>
        + Signed
        + SignedIntegerRegister<Element = I, Storage = I>
{
}

impl<U> FullyFormedUnsignedIntegerElement for U where
    U: IntegerElement + IntegerRegister<Element = U, Storage = U> + UnsignedIntegerRegister<Element = U, Storage = U>
{
}

/// SIMD types of the same element size but different lane counts, all based
/// on the given fully formed element types.
///
/// For example, `SizedSimd<f32, i32, u32>` provides all the fixed-size SIMD types
/// with 32-bit elements, and `SizedSimd<f64, i64, u64>` provides all the fixed-size SIMD types
/// with 64-bit elements.
///
/// You can use this generically to write code that works with either 32-bit or 64-bit
/// floating-point SIMD types and their associated integer types, like so:
/// ```ignore
/// fn my_func<S, T>(values: &[T]) -> T
/// where
///     T: FullyFormedFloatElement,
///     S: SizedSimd<T, <T as FloatElement>::Signed, <T as FloatElement>::Bits>,
/// {
///     // do whatever you need with S::fxN, S::ixN, S::uxN, etc.
///     let (scalar_prefix, vectors, scalar_suffix) = Vector::<S::fxN>::from_slice(values);
/// }
/// ```
#[rustfmt::skip]
pub trait SizedSimd<
    F: FullyFormedFloatElement + FloatElement<Signed = I, Bits = U>,
    I: FullyFormedSignedIntegerElement,
    U: FullyFormedUnsignedIntegerElement,
>: Simd {
    type NativeWidth: Lanes;

    type fxN: Interoperable<Self::ixN, Self::uxN, Lanes = Self::NativeWidth, Element = F, USize = Self::uxN, ISize = Self::ixN>
        + FloatRegister<Bits = Self::uxN, Signed = Self::ixN>;
    type ixN: Interoperable<Self::fxN, Self::uxN, Lanes = Self::NativeWidth, Element = I, USize = Self::uxN, ISize = Self::ixN>
        + SignedIntegerRegister<Element = <F as FloatElement>::Signed>;
    type uxN: Interoperable<Self::fxN, Self::ixN, Lanes = Self::NativeWidth, Element = U, USize = Self::uxN, ISize = Self::ixN>
        + UnsignedIntegerRegister<Element = <F as FloatElement>::Bits>;

    type fx2: Interoperable<Self::ix2, Self::ux2, Lanes = U2, Element = F, USize = Self::ux2, ISize = Self::ix2>
        + FloatRegister<Bits = Self::ux2, Signed = Self::ix2> + WidenRegister<F> + NarrowRegister<F>;
    type ix2: Interoperable<Self::fx2, Self::ux2, Lanes = U2, Element = I, USize = Self::ux2, ISize = Self::ix2>
        + SignedIntegerRegister<Element = <F as FloatElement>::Signed> + WidenRegister<I> + NarrowRegister<I>;
    type ux2: Interoperable<Self::fx2, Self::ix2, Lanes = U2, Element = U, USize = Self::ux2, ISize = Self::ix2>
        + UnsignedIntegerRegister<Element = <F as FloatElement>::Bits> + WidenRegister<U> + NarrowRegister<U>;

    type fx4: Interoperable<Self::ix4, Self::ux4, Lanes = U4, Element = F, USize = Self::ux4, ISize = Self::ix4>
        + FloatRegister<Bits = Self::ux4, Signed = Self::ix4> + LinAlg4Register
        + WidenRegister<Self::fx2> + NarrowRegister<Self::fx2>;
    type ix4: Interoperable<Self::fx4, Self::ux4, Lanes = U4, Element = I, USize = Self::ux4, ISize = Self::ix4>
        + SignedIntegerRegister<Element = <F as FloatElement>::Signed> + WidenRegister<Self::ix2> + NarrowRegister<Self::ix2>;
    type ux4: Interoperable<Self::fx4, Self::ix4, Lanes = U4, Element = U, USize = Self::ux4, ISize = Self::ix4>
        + UnsignedIntegerRegister<Element = <F as FloatElement>::Bits> + WidenRegister<Self::ux2> + NarrowRegister<Self::ux2>;

    type fx8: Interoperable<Self::ix8, Self::ux8, Lanes = U8, Element = F, USize = Self::ux8, ISize = Self::ix8>
        + FloatRegister<Bits = Self::ux8, Signed = Self::ix8> + WidenRegister<Self::fx4> + NarrowRegister<Self::fx4>;
    type ix8: Interoperable<Self::fx8, Self::ux8, Lanes = U8, Element = I, USize = Self::ux8, ISize = Self::ix8>
        + SignedIntegerRegister<Element = <F as FloatElement>::Signed> + WidenRegister<Self::ix4> + NarrowRegister<Self::ix4>;
    type ux8: Interoperable<Self::fx8, Self::ix8, Lanes = U8, Element = U, USize = Self::ux8, ISize = Self::ix8>
        + UnsignedIntegerRegister<Element = <F as FloatElement>::Bits> + WidenRegister<Self::ux4> + NarrowRegister<Self::ux4>;

    type fx16: Interoperable<Self::ix16, Self::ux16, Lanes = U16, Element = F, USize = Self::ux16, ISize = Self::ix16>
        + FloatRegister<Bits = Self::ux16, Signed = Self::ix16> + WidenRegister<Self::fx8> + NarrowRegister<Self::fx8>;
    type ix16: Interoperable<Self::fx16, Self::ux16, Lanes = U16, Element = I, USize = Self::ux16, ISize = Self::ix16>
        + SignedIntegerRegister<Element = <F as FloatElement>::Signed> + WidenRegister<Self::ix8> + NarrowRegister<Self::ix8>;
    type ux16: Interoperable<Self::fx16, Self::ix16, Lanes = U16, Element = U, USize = Self::ux16, ISize = Self::ix16>
        + UnsignedIntegerRegister<Element = <F as FloatElement>::Bits> + WidenRegister<Self::ux8> + NarrowRegister<Self::ux8>;
}

impl<S: Simd> SizedSimd<f32, i32, u32> for S {
    type NativeWidth = S::Native32Width;

    type fxN = S::f32xN;
    type ixN = S::i32xN;
    type uxN = S::u32xN;

    type fx2 = S::f32x2;
    type ix2 = S::i32x2;
    type ux2 = S::u32x2;

    type fx4 = S::f32x4;
    type ix4 = S::i32x4;
    type ux4 = S::u32x4;

    type fx8 = S::f32x8;
    type ix8 = S::i32x8;
    type ux8 = S::u32x8;

    type fx16 = S::f32x16;
    type ix16 = S::i32x16;
    type ux16 = S::u32x16;
}

impl<S: Simd> SizedSimd<f64, i64, u64> for S {
    type NativeWidth = S::Native64Width;

    type fxN = S::f64xN;
    type ixN = S::i64xN;
    type uxN = S::u64xN;

    type fx2 = S::f64x2;
    type ix2 = S::i64x2;
    type ux2 = S::u64x2;

    type fx4 = S::f64x4;
    type ix4 = S::i64x4;
    type ux4 = S::u64x4;

    type fx8 = S::f64x8;
    type ix8 = S::i64x8;
    type ux8 = S::u64x8;

    type fx16 = S::f64x16;
    type ix16 = S::i64x16;
    type ux16 = S::u64x16;
}

pub type f32xN<S> = Vector<<S as NativeSimd>::f32xN>;
pub type i32xN<S> = Vector<<S as NativeSimd>::i32xN>;
pub type u32xN<S> = Vector<<S as NativeSimd>::u32xN>;

pub type f64xN<S> = Vector<<S as NativeSimd>::f64xN>;
pub type i64xN<S> = Vector<<S as NativeSimd>::i64xN>;
pub type u64xN<S> = Vector<<S as NativeSimd>::u64xN>;

pub type f32x2<S> = Vector<<S as Simd>::f32x2>;
pub type i32x2<S> = Vector<<S as Simd>::i32x2>;
pub type u32x2<S> = Vector<<S as Simd>::u32x2>;

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

            pub type f32x2 = crate::simd::f32x2<$simd>;
            pub type i32x2 = crate::simd::i32x2<$simd>;
            pub type u32x2 = crate::simd::u32x2<$simd>;

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
