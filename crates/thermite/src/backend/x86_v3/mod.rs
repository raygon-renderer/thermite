//! x86-v3/x86-64-v3 backend, including AVX2 and FMA.

#![allow(non_camel_case_types)]

pub mod arch {
    pub use super::polyfills::*;
    pub use crate::backend::x86::avx2::*;
}

pub mod polyfills;
pub mod registers;

use crate::{register::DoublePump, vector::Vector};

pub use registers::X86V3;

pub type f32x4 = crate::simd::f32x4<X86V3>;
pub type i32x4 = crate::simd::i32x4<X86V3>;
pub type u32x4 = crate::simd::u32x4<X86V3>;
pub type f32x8 = crate::simd::f32x8<X86V3>;
pub type i32x8 = crate::simd::i32x8<X86V3>;
pub type u32x8 = crate::simd::u32x8<X86V3>;
pub type f64x2 = crate::simd::f64x2<X86V3>;
pub type i64x2 = crate::simd::i64x2<X86V3>;
pub type u64x2 = crate::simd::u64x2<X86V3>;
pub type f64x4 = crate::simd::f64x4<X86V3>;
pub type i64x4 = crate::simd::i64x4<X86V3>;
pub type u64x4 = crate::simd::u64x4<X86V3>;
pub type f64x8 = crate::simd::f64x8<X86V3>;
pub type i64x8 = crate::simd::i64x8<X86V3>;
pub type u64x8 = crate::simd::u64x8<X86V3>;
pub type f32x16 = crate::simd::f32x16<X86V3>;
pub type i32x16 = crate::simd::i32x16<X86V3>;
pub type u32x16 = crate::simd::u32x16<X86V3>;
pub type f64x16 = crate::simd::f64x16<X86V3>;
pub type i64x16 = crate::simd::i64x16<X86V3>;
pub type u64x16 = crate::simd::u64x16<X86V3>;

pub type f32xN = crate::simd::f32xN<X86V3>;
pub type i32xN = crate::simd::i32xN<X86V3>;
pub type u32xN = crate::simd::u32xN<X86V3>;
pub type f64xN = crate::simd::f64xN<X86V3>;
pub type i64xN = crate::simd::i64xN<X86V3>;
pub type u64xN = crate::simd::u64xN<X86V3>;
