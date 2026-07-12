#![doc = include_str!("../README.md")]
//
#![allow(unexpected_cfgs)]
#![cfg_attr(not(feature = "std"), no_std)]
#![cfg_attr(docsrs, feature(doc_cfg))]
//
#![allow(clippy::missing_transmute_annotations, clippy::let_and_return, unused_braces, unused)]
// used for more intelligent const splat
#![cfg_attr(feature = "nightly", feature(core_intrinsics, const_eval_select))]
#![cfg_attr(
    all(feature = "nightly", feature = "spirv", target_arch = "spirv"),
    feature(asm_experimental_arch)
)]
#![cfg_attr(feature = "nightly", allow(internal_features))]
// generic_const_exprs is too unstable - causes "overly complex generic constant" errors
// throughout the codebase when enabled. Commented out until the feature matures.
// #![cfg_attr(feature = "nightly", feature(generic_const_exprs))]
// #![cfg_attr(feature = "nightly", allow(incomplete_features))]
// Enable wasm64 simd on nightly
#![cfg_attr(all(feature = "nightly", target_arch = "wasm64"), feature(simd_wasm64))]
// Scalar WASM float intrinsics (f32_sqrt, f32_floor, etc.) - still unstable
#![cfg_attr(
    all(
        feature = "nightly",
        feature = "wasm",
        any(target_arch = "wasm32", target_arch = "wasm64")
    ),
    feature(wasm_numeric_instr)
)]
#![cfg_attr(all(feature = "nightly", feature = "std_simd"), feature(portable_simd))]

#[cfg(feature = "nightly")]
#[rustversion::not(nightly)]
fn nightly_check() {
    compile_error!("The `nightly` feature requires a nightly compiler.");
}

#[cfg(feature = "bitvec")]
pub extern crate bitvec;
pub extern crate const_default;
pub extern crate generic_array;

pub use thermite_macros::{HasIsa, dispatch, dispatch_dyn};

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

pub mod guide {
    #![doc = include_str!("../GUIDE.md")]
}

/// Common imports for working with Thermite.
///
/// `use thermite::prelude::*;` brings the core types ([`Vector`], [`Mask`]) and
/// the vector/mask/math trait hierarchy into scope, with the traits imported
/// anonymously (`as _`) so their methods and operators are available without
/// cluttering the namespace. This is the recommended starting point for most
/// code.
///
/// Note that the math traits are imported anonymously: their methods are
/// callable, but the trait names are not in scope. To name one in a generic
/// bound (e.g. `fn f<V: FloatVector + TranscendentalMath>`), import it
/// explicitly with `use thermite::math::TranscendentalMath;`.
pub mod prelude {
    pub use crate::{Mask, Vector};

    pub use crate::{
        divider::{BranchfreeDivider, Divider},
        element::{Element, FloatElement},
        mask::{CastMask, GenericMask},
        math::{
            CoreMath as _, CoreMathWithPolicy as _, FloatMath as _, FloatMathWithPolicy as _, RealMath as _,
            RealMathWithPolicy as _, ScalarMath as _, ScalarMathWithPolicy as _, SpatialMath as _,
            SpatialMathWithPolicy as _, TranscendentalMath as _, TranscendentalMathWithPolicy as _,
        },
        math::{FloatConsts, policy::Policy},
        simd::{
            FixedWidthSimd, FloatSimd, NativeIsa, NativeSimd, NativeSimdVectors, NativeSimdVectorsWithRegisters, Simd,
            Simd3, Simd3A, Simd3AVectors, Simd3AVectorsWithRegisters, Simd3Vectors, Simd3VectorsWithRegisters,
            SimdVectors, SimdVectorsWithRegisters, SizedSimd,
        },
        slice::SimdSlice as _,
        swizzle::Swizzle as _,
        vector::ops::{
            AddAssignMasked as _, AddMasked as _, BitAndAssignMasked as _, BitAndMasked as _, BitAndNot as _,
            BitAndNotAssign as _, BitAndNotAssignMasked as _, BitAndNotMasked as _, BitOrAssignMasked as _,
            BitOrMasked as _, BitXorAssignMasked as _, BitXorMasked as _, DivAssignMasked as _, DivMasked as _,
            MulAddAssignExt as _, MulAddAssignExtMasked as _, MulAddExt as _, MulAddExtMasked as _,
            MulAssignMasked as _, MulMasked as _, NegMasked as _, NotMasked as _, RemAssignMasked as _, RemMasked as _,
            ShlAssignMasked as _, ShlMasked as _, ShrAssignMasked as _, ShrMasked as _, Square as _, SquareMasked as _,
            SubAssignMasked as _, SubMasked as _,
        },
        vector::{
            BitCastVector, BitshiftVector, BitwiseVector, CastVector, ConcatVector, ExtendVector, FloatVector,
            FloatVectorWithBits, GenericVector, GenericVector2 as _, GenericVector3 as _, GenericVector4 as _,
            IndexableVector, IntegerVector, Interleave, LinAlg3Vector, LinAlg4Vector, NumericVector, PackedFloatVector,
            PartialOrdVector, SignedIntegerVector, SignedVector, SplatConst, Swizzle3 as _, Swizzle4 as _,
            SwizzleVector, UnsignedIntegerVector, VectorIndices, VectorWithRegister as _,
        },
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
pub mod mask;
pub mod math;
pub mod register;
pub mod slice;
pub mod transform;

#[doc(hidden)]
pub mod swizzle;

pub use divider::{BranchfreeDivider, Divider};
pub use isa::InstructionSet;
pub use mask::Mask;
pub use simd::HasIsa;
pub use swizzle::Swizzle;
pub use vector::Vector;

/// The widest signed integer type that is efficient on the current target.
///
/// Normally `i64`. On the SPIR-V GPU backend, however, 64-bit integers require
/// the `Int64` capability, which not every device advertises; when targeting
/// SPIR-V without that capability this falls back to `i32`. Use this (and
/// [`LargeUInt`]) for index/size arithmetic that should stay native on every
/// supported target rather than hard-coding `i64`.
pub type LargeInt = cfg_select! {
    all(feature = "spirv", target_arch = "spirv", not(target_feature = "Int64")) => i32,
    _ => i64,
};

/// The widest unsigned integer type that is efficient on the current target.
///
/// The unsigned counterpart of [`LargeInt`]: `u64` everywhere except on a
/// SPIR-V target lacking the `Int64` capability, where it falls back to `u32`.
pub type LargeUInt = cfg_select! {
    all(feature = "spirv", target_arch = "spirv", not(target_feature = "Int64")) => u32,
    _ => u64,
};

cfg_if::cfg_if! {
    if #[cfg(all(feature = "spirv", target_arch = "spirv"))] {
        #[doc(hidden)] #[inline(always)] pub fn likely(b: bool) -> bool { b }
        #[doc(hidden)] #[inline(always)] pub fn unlikely(b: bool) -> bool { b }
    } else {
        // borrows technique from https://github.com/rust-lang/hashbrown/pull/209
        #[inline]
        #[cold]
        fn cold() {}

        #[rustfmt::skip]
        #[doc(hidden)] #[inline(always)]
        pub fn likely(b: bool) -> bool {
            if !b { cold() } b
        }

        #[rustfmt::skip]
        #[doc(hidden)] #[inline(always)]
        pub fn unlikely(b: bool) -> bool {
            if b { cold() } b
        }
    }
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
