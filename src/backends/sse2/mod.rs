#![allow(unused)]

use crate::*;

use std::{
    fmt,
    marker::PhantomData,
    mem::{transmute, transmute_copy},
    ops::*,
};

use super::arch::sse2::*;

use half::f16;

pub(crate) mod polyfills;

use polyfills::*;
