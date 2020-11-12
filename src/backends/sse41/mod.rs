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

use polyfills::*;

//mod vf32;
//mod vf64;
//mod vi16;
mod vi32;
//mod vi64;
//mod vu32;
//mod vu64;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct SSE41;

impl Simd for SSE41 {}
