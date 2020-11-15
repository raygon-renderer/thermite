#![allow(unused)]

use crate::*;

use std::{
    fmt,
    marker::PhantomData,
    mem::{transmute, transmute_copy},
    ops::*,
};

#[cfg(target_arch = "x86")]
use std::arch::x86::*;

#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

use half::f16;

pub(crate) mod polyfills;

use super::polyfills::*;
use polyfills::*;

use super::avx1::{f32x8, f64x8};

mod vf32;
mod vf64;
mod vi16;
mod vi32;
mod vi64;
mod vu32;
mod vu64;

pub use vf32::*;
pub use vf64::*;
pub use vi16::*;
pub use vi32::*;
pub use vi64::*;
pub use vu32::*;
pub use vu64::*;

type Vi16 = i16x8<AVX2>;
type Vi32 = i32x8<AVX2>;
type Vi64 = i64x8<AVX2>;
type Vu32 = u32x8<AVX2>;
type Vu64 = u64x8<AVX2>;
type Vf32 = f32x8<AVX2>;
type Vf64 = f64x8<AVX2>;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct AVX2;

impl Simd for AVX2 {
    const INSTRSET: SimdInstructionSet = SimdInstructionSet::AVX2;

    type Vi32 = Vi32;
    type Vi64 = Vi64;
    type Vu32 = Vu32;
    type Vu64 = Vu64;
    type Vf32 = Vf32;
    type Vf64 = Vf64;

    #[cfg(target_pointer_width = "32")]
    type Vusize = Vu32;

    #[cfg(target_pointer_width = "32")]
    type Visize = Vi32;

    #[cfg(target_pointer_width = "64")]
    type Vusize = Vu64;

    #[cfg(target_pointer_width = "64")]
    type Visize = Vi64;
}
