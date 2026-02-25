//! Thermite: Melt your CPU
//!
//! A Rust library for portable SIMD programming, providing abstractions
//! over various SIMD instruction sets as well as generic implementations,
//! enabling high-performance vectorized computations across different hardware
//! architectures with ease.
//!
//! Thermite aims to provide a comprehensive set of SIMD-accelerated operations,
//! including arithmetic, bitwise, and mathematical functions, along with
//! support for various data types and vector sizes. It is designed to be
//! extensible and modular, allowing for easy addition of new instruction
//! sets and optimizations over time.

#![cfg_attr(not(feature = "std"), no_std)]
//
#![allow(clippy::missing_transmute_annotations, clippy::let_and_return, unused_braces, unused)]
// used for more intelligent const splat
#![cfg_attr(feature = "nightly", feature(core_intrinsics, const_eval_select))]
#![cfg_attr(feature = "nightly", allow(internal_features))]
// Enable wasm64 simd on nightly
#![cfg_attr(all(feature = "nightly", target_arch = "wasm64"), feature(simd_wasm64))]
#![cfg_attr(all(feature = "nightly", feature = "std_simd"), feature(portable_simd))]

#[cfg(feature = "nightly")]
#[rustversion::not(nightly)]
fn nightly_check() {
    compile_error!("The `nightly` feature requires a nightly compiler.");
}

pub extern crate bitvec;
pub extern crate generic_array;

/// Creates a shuffle mask for various instructions. Note
/// that the order of the arguments is reversed from the
/// normal order of the lanes, so `MM_SHUFFLE!(3, 2, 1, 0)`
/// would be the identity shuffle (unchanged).
#[macro_export]
macro_rules! MM_SHUFFLE {
    () => { 0 };
    ($v:expr) => { $v };

    ($($v:expr),* $(,)?) => {const {
        const LEN: usize = [$($v),*].len();
        assert!(LEN.is_power_of_two(), "MM_SHUFFLE! requires a power of two number of lanes");

        const SHIFT: u32 = LEN.ilog2();

        let mut mask = 0;

        $(
            mask <<= SHIFT;
            mask |= $v;
        )*

        mask
    }};
}

/// Like `MM_SHUFFLE!`, but the order of the arguments is
/// the same as the order of the lanes (reversed from
/// conventional order).
#[macro_export] #[rustfmt::skip]
macro_rules! MM_SHUFFLE_R {
    () => { 0 };
    ($v:expr) => { $v };

    ($($v:expr),* $(,)?) => {const {
        const LEN: usize = [$($v),*].len();
        assert!(LEN.is_power_of_two(), "MM_SHUFFLE_R! requires a power of two number of lanes");

        const SHIFT: u32 = LEN.ilog2();

        let mut mask = 0;
        let mut shift = 0;

        $(
            mask |= $v << shift;
            shift += SHIFT;
        )*

        mask
    }};
}

pub mod prelude {
    pub use crate::{Mask, Vector};

    pub use crate::{
        divider::{BranchfreeDivider, Divider},
        element::{Element, FloatElement},
        generic::ops::{
            AddAssignMasked as _, AddMasked as _, BitAndAssignMasked as _, BitAndMasked as _, BitAndMasked as _,
            BitAndNot as _, BitAndNotAssign as _, BitAndNotAssignMasked as _, BitAndNotMasked as _,
            BitOrAssignMasked as _, BitOrMasked as _, BitOrMasked as _, BitXorAssignMasked as _, BitXorMasked as _,
            BitXorMasked as _, DivAssignMasked as _, DivMasked as _, MulAddAssignExt as _, MulAddAssignExtMasked as _,
            MulAddExt as _, MulAddExtMasked as _, MulAssignMasked as _, MulMasked as _, NegMasked as _, NotMasked as _,
            RemAssignMasked as _, RemMasked as _, ShlAssignMasked as _, ShlMasked as _, ShrAssignMasked as _,
            ShrMasked as _, Square as _, SquareMasked as _, SubAssignMasked as _, SubMasked as _,
        },
        generic::{
            BitCastVector, BitshiftVector, BitwiseVector, CastMask, CastVector, ConcatVector, ExtendVector,
            FloatVector, FloatVectorWithBits, GenericMask, GenericVector, IndexableVector, IntegerVector,
            LinAlg3Vector, LinAlg4Vector, NumericVector, PartialOrdVector, SignedIntegerVector, SignedVector,
            SplatConst, SwizzleVector, UnsignedIntegerVector, VectorIndices,
        },
        math::{
            CoreMath as _, CoreMathWithPolicy as _, FloatMath as _, FloatMathWithPolicy as _, RealMath as _,
            RealMathWithPolicy as _, SpatialMath as _, SpatialMathWithPolicy as _, TranscendentalMath as _,
            TranscendentalMathWithPolicy as _,
        },
        math::{FloatConsts, policy::Policy},
        simd::{
            FixedWidthSimd, FloatSimd, NativeIsa, NativeSimd, NativeSimdVectors, NativeSimdVectorsWithRegisters, Simd,
            Simd3A, Simd3AVectors, Simd3AVectorsWithRegisters, SimdVectors, SimdVectorsWithRegisters, SizedSimd,
        },
        swizzle::Swizzle as _,
    };
}

#[macro_use]
mod internal_macros;

#[macro_use]
pub mod simd;
pub mod isa;
pub mod vector;

pub mod backend;
pub mod compat;
pub mod divider;
pub mod element;
pub mod generic;
pub mod mask;
pub mod math;
pub mod register;
pub mod transform;

#[doc(hidden)]
pub mod swizzle;

pub use divider::{BranchfreeDivider, Divider};
pub use mask::Mask;
pub use swizzle::Swizzle;
pub use vector::{MaskOf, Vector};

// borrows technique from https://github.com/rust-lang/hashbrown/pull/209
#[inline]
#[cold]
fn cold() {}

#[rustfmt::skip]
#[inline(always)]
pub fn likely(b: bool) -> bool {
    if !b { cold() } b
}

#[rustfmt::skip]
#[inline(always)]
pub fn unlikely(b: bool) -> bool {
    if b { cold() } b
}

/// Generate ternlog immediate constant via arbitrary expressions. The
/// constants A, B, and C are provided internally for convenience.
///
/// # Example
///
/// ```rust
/// let result = thermite::ternlog_imm!(A & B | !A & C);
/// assert_eq!(result, 0xCA);
/// ```
#[macro_export]
macro_rules! ternlog_imm {
    ($($tt:tt)*) => {
        const {
            const A: i32 = 0xF0; // Binary 11110000
            const B: i32 = 0xCC; // Binary 11001100
            const C: i32 = 0xAA; // Binary 10101010

            $($tt)*
        }
    };
}
