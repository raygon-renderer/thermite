#![allow(unused)]

use crate::*;

use core::{
    fmt,
    marker::PhantomData,
    mem::{transmute, transmute_copy},
    ops::*,
};

mod polyfills;
use polyfills::*;

use half::f16;

mod vf32;
mod vf64;
mod vi32;
mod vi64;
mod vu32;
mod vu64;

pub use vf32::*;
pub use vf64::*;
pub use vi32::*;
pub use vi64::*;
pub use vu32::*;
pub use vu64::*;

type Vu32 = u32x1<Scalar>;
type Vf32 = f32x1<Scalar>;
type Vf64 = f64x1<Scalar>;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Scalar;

impl Simd for Scalar {
    const INSTRSET: SimdInstructionSet = SimdInstructionSet::Scalar;

    type Vu32 = Vu32;
    type Vf32 = Vf32;
    type Vf64 = Vf64;

    #[cfg(target_pointer_width = "32")]
    type Vusize = Vu32;

    //#[cfg(target_pointer_width = "32")]
    //type Visize = Vi32;

    /*
    type Vi32 = Vi32;
    type Vi64 = Vi64;

    type Vu64 = Vu64;

    #[cfg(target_pointer_width = "64")]
    type Vusize = Vu64;

    #[cfg(target_pointer_width = "64")]
    type Visize = Vi64;
    */
}
