#![no_std]
// stdmind for f16c instructions, core_intrinsics for likely/unlikely
#![cfg_attr(feature = "nightly", feature(stdsimd, core_intrinsics))]
#![allow(unused_imports, non_camel_case_types, non_snake_case)]

#[macro_use]
mod macros;

pub mod arch;

pub mod backends;
pub mod iset;
pub mod widen;

pub use iset::SimdInstructionSet;

use core::{fmt::Debug, marker::PhantomData, mem, ops::*, ptr};

/// SIMD Instruction set, contains all types
///
/// Take your time to look through this. All trait bounds contain methods and associated values which
/// encapsulate all functionality for this crate.
pub trait Simd: 'static + Debug + Send + Sync + Clone + Copy + PartialEq + Eq {
    const INSTRSET: SimdInstructionSet;

    /// Largest native single-precision floating point vector, occupies one register.
    type Vf32;

    /// 32-bit single-precision floating point vector
    type Vf32x1;
    /// 64-bit single-precision floating point vector
    type Vf32x2;
    /// 128-bit single-precision floating point vector
    type Vf32x4: SimdFixedVector<Self, 4> + SimdFloatVector<Self, Element = f32> + SimdOverloads<Self>;
    /// 256-bit single-precision floating point vector
    type Vf32x8;
    /// 512-bit single-precision floating point vector
    type Vf32x16;
}

pub trait SimdVectorBase<S: Simd>: Clone + Copy {
    type Element;

    fn splat(value: Self::Element) -> Self;
}

pub trait SimdFixedVector<S: Simd, const N: usize>: SimdVectorBase<S> {
    fn set(values: [Self::Element; N]) -> Self;
}

pub trait SimdVector<S: Simd>: SimdVectorBase<S> + Add<Self, Output = Self> {
    fn zero() -> Self;
    fn one() -> Self;
    fn min_value() -> Self;
    fn max_value() -> Self;
}

pub trait SimdOverloads<S: Simd>:
    SimdVectorBase<S>
    + Add<Self::Element, Output = Self>
    + Sub<Self::Element, Output = Self>
    + Mul<Self::Element, Output = Self>
    + Div<Self::Element, Output = Self>
    + Rem<Self::Element, Output = Self>
{
}

impl<T, S: Simd> SimdOverloads<S> for T where
    T: SimdVectorBase<S>
        + Add<Self::Element, Output = Self>
        + Sub<Self::Element, Output = Self>
        + Mul<Self::Element, Output = Self>
        + Div<Self::Element, Output = Self>
        + Rem<Self::Element, Output = Self>
{
}

pub trait SimdSignedVector<S: Simd>: SimdVector<S> {
    fn abs(self) -> Self;
}

pub trait SimdFloatVector<S: Simd>: SimdSignedVector<S> {
    fn neg_one() -> Self;
    fn neg_zero() -> Self;
}
