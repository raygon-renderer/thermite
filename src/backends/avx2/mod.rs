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

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct AVX2;

mod vi32;
pub use vi32::*;

mod vf32;
pub use vf32::*;

mod vu64;
pub use vu64::*;

mod vf64;
pub use vf64::*;

impl Simd for AVX2 {
    type Vi32 = i32x8<AVX2>;
    type Vu64 = u64x8<AVX2>;
    type Vf32 = f32x8<AVX2>;
    type Vf64 = f64x8<AVX2>;

    #[cfg(target_pointer_width = "64")]
    type Vusize = u64x8<AVX2>;
}
