#![allow(unused)]

use crate::*;

use std::{
    fmt,
    marker::PhantomData,
    mem::{transmute, transmute_copy},
    ops::*,
};

use super::arch::sse42::*;

use half::f16;

pub(crate) mod polyfills;

use polyfills::*;

/*
//mod vf32;
//mod vf64;
//mod vi16;
mod vi32;
//mod vi64;
mod vu32;
//mod vu64;

use vi32::i32x4;
use vu32::u32x4;

pub type Vi32 = i32x4<SSE42>;
pub type Vu32 = u32x4<SSE42>;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct SSE42;

impl Simd for SSE42 {
    const INSTRSET: SimdInstructionSet = SimdInstructionSet::SSE42;

    type Vi32 = i32x4<SSE42>;
    type Vu32 = u32x4<SSE42>;
}
*/
