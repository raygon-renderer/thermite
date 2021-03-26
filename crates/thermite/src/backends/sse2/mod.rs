#![allow(unused)]

use crate::*;

use core::{
    fmt,
    marker::PhantomData,
    mem::{transmute, transmute_copy},
    ops::*,
};

use crate::arch::sse2::*;

use half::f16;

pub(crate) mod polyfills;

use super::polyfills::*;
use polyfills::*;
