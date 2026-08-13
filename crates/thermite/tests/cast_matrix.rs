//! Completeness gate for the same-length cast matrix: every vector of N lanes
//! must be castable to every other vector of N lanes, whatever the element types.
//!
//! Nothing here runs at test time - it all resolves during compilation, and the
//! file failing to build *is* the failure. What each pair computes is
//! `diff_cast_matrix.rs`'s job.
//!
//! Two questions get asked separately, because they have different answers and
//! each needs its own probe shape:
//!
//! 1. **Does an implementation exist?** Asked with concrete backend types, which
//!    resolve against the impls directly.
//! 2. **Can generic code name it?** Asked with a generic body over `S: Simd` /
//!    `S: SimdVectors`, which resolves against the *declared bounds*. An
//!    implementation the traits never declare is unreachable from any generic
//!    kernel, and the concrete probe cannot see that.
//!
//! Both must sit in a function **body**. The obvious `where`-clause form does
//! not work and is not a hypothetical failure - it once reported this matrix
//! complete while 106 pairs had no implementation whatsoever:
//!
//! ```ignore
//! fn p<S: Simd>() { assert_cast::<S::f32x4, S::u64x4>(); }   // checks
//! fn p<S: Simd>() where S::u64x4: CastRegister<S::f32x4> {}  // does NOT
//! ```
//!
//! Nothing instantiates the second one, so rustc never resolves the bound.
//!
//! Coverage is 90 directed pairs per lane count (10 element types, self-pairs
//! excluded) across the four power-of-two lane counts, for every backend the
//! host architecture compiles. The 3-lane `Simd3`/`Simd3A` slots carry only the
//! 32- and 64-bit types and are not covered here.
#![cfg(any(
    target_arch = "x86",
    target_arch = "x86_64",
    target_arch = "wasm32",
    target_arch = "aarch64"
))]

use thermite::register::{CastRegister, CoreRegister};
use thermite::simd::{Simd, SimdVectors};
use thermite::vector::CastVector;

fn assert_cast<FROM: CoreRegister, TO: CastRegister<FROM>>() {}

// Reached only from `generic_reachability`, which is deliberately never called.
#[allow(dead_code)]
fn assert_vcast<FROM, TO: CastVector<FROM>>() {}

/// Every directed pair over a list of slot names, using concrete backend types.
///
/// Consumes the list head-first and pairs the head against each remaining
/// element in **both** directions, then recurses on the tail. That yields each
/// unordered pair exactly once and each directed pair exactly once, without
/// needing to compare two idents for equality to skip the self-pairs - which
/// `macro_rules!` cannot do.
macro_rules! concrete_pairs {
    ($b:ty;) => {};
    ($b:ty; $head:ident $(, $tail:ident)*) => {
        $(
            assert_cast::<<$b as Simd>::$head, <$b as Simd>::$tail>();
            assert_cast::<<$b as Simd>::$tail, <$b as Simd>::$head>();
        )*
        concrete_pairs!($b; $($tail),*);
    };
}

/// As above, but generic over `S` and parameterised by which assertion to use,
/// so one macro serves the register layer (`assert_cast`, bounds from `Simd`)
/// and the vector layer (`assert_vcast`, bounds from `SimdVectors`).
macro_rules! generic_pairs {
    ($assert:ident;) => {};
    ($assert:ident; $head:ident $(, $tail:ident)*) => {
        $(
            $assert::<S::$head, S::$tail>();
            $assert::<S::$tail, S::$head>();
        )*
        generic_pairs!($assert; $($tail),*);
    };
}

/// The ten element slots at one lane count, fed to whichever pair macro.
///
/// Slot names are spelled out per lane count because the `Simd` slots are plain
/// associated-type names and there is no concatenating `i8` with `x4` without a
/// proc macro.
macro_rules! each_lane {
    ($mac:ident, $arg:tt) => {
        $mac!($arg; f32x2, f64x2, i8x2, u8x2, i16x2, u16x2, i32x2, u32x2, i64x2, u64x2);
        $mac!($arg; f32x4, f64x4, i8x4, u8x4, i16x4, u16x4, i32x4, u32x4, i64x4, u64x4);
        $mac!($arg; f32x8, f64x8, i8x8, u8x8, i16x8, u16x8, i32x8, u32x8, i64x8, u64x8);
        $mac!($arg; f32x16, f64x16, i8x16, u8x16, i16x16, u16x16, i32x16, u32x16, i64x16, u64x16);
    };
}

/// Question 1, per backend: every pair has an implementation.
macro_rules! backend_exists {
    ($($modname:ident => $b:ty),* $(,)?) => {$(
        #[test]
        fn $modname() {
            each_lane!(concrete_pairs, $b);
        }
    )*};
}

// The scalar backend is repeated per architecture rather than hoisted: its slot
// types are `ArrayRegister` composites over the host's own registers, so it is
// not the same set of impls on each target.
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
mod x86 {
    use super::*;

    backend_exists! {
        scalar_matrix_exists => thermite::backend::scalar::Scalar,
        v1_matrix_exists => thermite::backend::x86_v1::X86V1,
        v2_matrix_exists => thermite::backend::x86_v2::X86V2,
        v3_matrix_exists => thermite::backend::x86_v3::X86V3,
    }
}

#[cfg(target_arch = "wasm32")]
mod wasm {
    use super::*;

    backend_exists! {
        scalar_matrix_exists => thermite::backend::scalar::Scalar,
        wasm_matrix_exists => thermite::backend::wasm::Wasm,
    }
}

#[cfg(target_arch = "aarch64")]
mod neon {
    use super::*;

    backend_exists! {
        scalar_matrix_exists => thermite::backend::scalar::Scalar,
        neon_matrix_exists => thermite::backend::neon::Neon,
    }
}

/// Question 2: every pair is reachable from generic code, at both layers.
///
/// These are never called. A generic body is type-checked against the trait's
/// declared bounds regardless, which is the whole point - calling them would
/// only add the backend's impls back into scope and re-answer question 1.
#[allow(dead_code)]
mod generic_reachability {
    use super::*;

    /// Register layer: bounds declared on `Simd`.
    pub fn registers<S: Simd>() {
        each_lane!(generic_pairs, assert_cast);
    }

    /// Vector layer: bounds declared on `SimdVectors`, which does not inherit
    /// them from `Simd` - the mirror states its own `CastVector` bounds. This is
    /// the layer user code actually touches.
    pub fn vectors<S: SimdVectors>() {
        each_lane!(generic_pairs, assert_vcast);
    }
}
