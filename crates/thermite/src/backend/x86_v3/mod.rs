//! x86-v3/x86-64-v3 backend, including AVX2 and FMA.

#![allow(non_camel_case_types)]

#[macro_use]
mod macros;

pub mod arch {
    pub use super::polyfills::*;
    pub use crate::backend::x86::avx2::*;
}

pub mod polyfills;
pub mod registers;

use crate::{register::DoublePump, vector::Vector};

pub use registers::X86V3;

decl_aliases!(X86V3);
pub use self::aliases::*;
