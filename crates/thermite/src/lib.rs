#![doc = include_str!("../README.md")]
//
#![recursion_limit = "256"]
#![allow(unexpected_cfgs)]
#![cfg_attr(not(feature = "std"), no_std)]
#![cfg_attr(docsrs, feature(doc_cfg))]
//
#![allow(
    clippy::missing_transmute_annotations,
    clippy::let_and_return,
    unused_braces,
    unused_imports
)]
#![deny(rustdoc::invalid_rust_codeblocks)]
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

/// The SPIR-V backend is not finished, so the released crate refuses to build it.
#[cfg(all(feature = "spirv", not(thermite_unstable_spirv)))]
fn spirv_readiness_check() {
    compile_error!(
        "the `spirv` feature is incomplete and disabled in released versions of Thermite. \
         To work on it anyway, depend on thermite \
         from git and build with RUSTFLAGS='--cfg thermite_unstable_spirv'."
    );
}

/// Pre-`main` initializer for the wasm backend's relaxed-FMA detection: the
/// linker collects `.init_array` entries into `__wasm_call_ctors`, which
/// wasi-libc's `_start`/`_initialize` runs before user code. Running the
/// canaries here (instead of lazily on first use) keeps the flag read path
/// call-free, so the load is loop-invariant and hoistable out of hot loops.
///
/// Embedders that never invoke `__wasm_call_ctors` simply leave the flags
/// zeroed: `mul_add` then takes the (bit-identical, slower) emulation on
/// every call.
#[cfg(all(feature = "wasm", any(target_arch = "wasm32", target_arch = "wasm64")))]
#[used]
#[unsafe(link_section = ".init_array")]
static INIT_WASM_RELAXED_FMA: extern "C" fn() = {
    extern "C" fn init() {
        crate::backend::wasm::polyfills::math::detect_relaxed_fma();
    }
    init
};

/// Inventory of the crate features Thermite was built with, so that downstream
/// crates and algorithms can vary behavior based on them.
pub mod features {
    /// Whether the `strict_ieee754` feature is enabled: follow IEEE-754 exactly
    /// even where SIMD instructions intentionally do not, at a significant cost.
    ///
    /// Implies [`PRESERVE_DENORMALS`], and turns off the approximate
    /// reciprocal/rsqrt estimates on backends that have them. (The emulated FMA
    /// needs no strict-mode override: on backends without hardware FMA,
    /// `mul_add` and family are correctly rounded, bit-identical to a true
    /// fused multiply-add, unconditionally.)
    pub const STRICT_IEEE754: bool = cfg!(feature = "strict_ieee754");

    /// Whether the `preserve_denormals` feature is enabled, making every default
    /// math policy preserve denormal inputs rather than flushing or crushing them.
    ///
    /// Takes precedence over [`IGNORE_DENORMALS`] if both are somehow enabled.
    pub const PRESERVE_DENORMALS: bool = cfg!(feature = "preserve_denormals") || STRICT_IEEE754;

    /// Whether the `ignore_denormals` feature is enabled, making every default
    /// math policy leave denormals alone on the assumption that the hardware
    /// already flushes them.
    ///
    /// Ignored when [`PRESERVE_DENORMALS`] is also enabled.
    pub const IGNORE_DENORMALS: bool = cfg!(feature = "ignore_denormals") && !PRESERVE_DENORMALS;

    /// Whether the `algebraic-scalar` feature is enabled, i.e. whether scalar-backend
    /// float arithmetic is reassociable rather than strict IEEE-754.
    ///
    /// Exposed as a `const` so downstream crates whose algorithms depend on exact
    /// cancellation (double-double arithmetic, error-free transformations) can reject
    /// the combination with a `const` assertion.
    ///
    /// [`STRICT_IEEE754`] overrides this: asking for the spec exactly always wins over
    /// asking the optimizer to rearrange, so the two together yield strict arithmetic
    /// rather than a build error.
    pub const ALGEBRAIC_SCALAR: bool = cfg!(feature = "algebraic-scalar") && !STRICT_IEEE754;

    /// Whether the `disable_dispatch` feature is enabled, replacing every static
    /// ISA dispatch with a plain `#[inline(always)]` signature.
    ///
    /// Only correct for builds that pin the target ISA at compile time. If a
    /// dispatched function then fails to inline, it loses its target features and
    /// gets dramatically slower, which is the whole problem dispatch exists to solve.
    pub const DISABLE_DISPATCH: bool = cfg!(feature = "disable_dispatch");

    /// Whether the `avx2-f16c` feature is enabled, assuming `f16c` is present
    /// whenever AVX2 is (true of every AVX2 CPU) so half-precision conversion needs
    /// no separate runtime check. No effect off x86.
    pub const AVX2_F16C: bool = cfg!(feature = "avx2-f16c");

    /// Whether the `avx2-pclmul` feature is enabled, assuming `pclmulqdq` is present
    /// whenever AVX2 is, which enables the CLMUL 2D-Morton fast path on `u64` lanes.
    /// No effect off x86.
    pub const AVX2_PCLMUL: bool = cfg!(feature = "avx2-pclmul");
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
        sort::{Ascending, Descending, SortOrder},
        swizzle::Swizzle as _,
        vector::ops::{
            AddAssignMasked as _, AddMasked as _, AddSubExt as _, AddSubExtMasked as _, BitAndAssignMasked as _,
            BitAndMasked as _, BitAndNot as _, BitAndNotAssign as _, BitAndNotAssignMasked as _, BitAndNotMasked as _,
            BitOrAssignMasked as _, BitOrMasked as _, BitXorAssignMasked as _, BitXorMasked as _, DivAssignMasked as _,
            DivMasked as _, MulAddAssignExt as _, MulAddAssignExtMasked as _, MulAddExt as _, MulAddExtMasked as _,
            MulAssignMasked as _, MulMasked as _, NegMasked as _, NotMasked as _, RemAssignMasked as _, RemMasked as _,
            ShlAssignMasked as _, ShlMasked as _, ShrAssignMasked as _, ShrMasked as _, Square as _, SquareMasked as _,
            SubAssignMasked as _, SubMasked as _,
        },
        vector::{
            BitCastVector, BitshiftVector, BitwiseVector, CastVector, ConcatVector, ExtendVector, FloatVector,
            FloatVectorWithBits, GenericVector, GenericVector2 as _, GenericVector3 as _, GenericVector4 as _,
            IndexableVector, IntegerVector, Interleave, LinAlg3Vector, LinAlg4Vector, NumericVector, PackedFloatVector,
            PartialOrdVector, SignedIntegerVector, SignedVector, SplatConst, StreamGroup, Swizzle3 as _, Swizzle4 as _,
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
pub mod cpu;
pub mod divider;
pub mod element;
pub mod mask;
pub mod math;
pub mod register;
pub mod slice;
pub mod sort;

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
