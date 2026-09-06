//! Completeness gate for the same-length cast matrix: every vector of N lanes
//! must be castable to every other vector of N lanes, whatever the element types.
//!
//! Nothing here runs at test time. It all resolves during compilation, and the
//! file failing to build *is* the failure. What each pair computes is
//! `diff_cast_matrix.rs`'s job.
//!
//! Two questions get asked separately, because they have different answers and
//! each needs its own probe shape:
//!
//! 1. **Does an implementation exist?** Asked with concrete backend types, which
//!    resolve against the impls directly. `for_each_backend_concrete!` asks it
//!    once per compiled backend.
//! 2. **Is it reachable from a generic bound?** Asked with `S: Simd` etc., which
//!    resolve against the trait's declared bounds only.
#![cfg(any(
    target_arch = "x86",
    target_arch = "x86_64",
    target_arch = "wasm32",
    target_arch = "aarch64"
))]

mod harness;

use thermite::register::{CastRegister, CoreRegister};
use thermite::simd::{Simd, Simd3, Simd3A, SimdVectors};
use thermite::vector::CastVector;

fn assert_cast<FROM: CoreRegister, TO: CastRegister<FROM>>() {}

#[allow(dead_code)]
fn assert_vcast<FROM, TO: CastVector<FROM>>() {}

macro_rules! concrete_pairs_in {
    ($b:ty, $tr:ident;) => {};
    ($b:ty, $tr:ident; $head:ident $(, $tail:ident)*) => {
        $(
            assert_cast::<<$b as $tr>::$head, <$b as $tr>::$tail>();
            assert_cast::<<$b as $tr>::$tail, <$b as $tr>::$head>();
        )*
        concrete_pairs_in!($b, $tr; $($tail),*);
    };
}

macro_rules! concrete_pairs {
    ($b:ty; $($slot:ident),* $(,)?) => {
        concrete_pairs_in!($b, Simd; $($slot),*);
    };
}

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

macro_rules! each_lane {
    ($mac:ident, $arg:tt) => {
        $mac!($arg; f32x2, f64x2, i8x2, u8x2, i16x2, u16x2, i32x2, u32x2, i64x2, u64x2);
        $mac!($arg; f32x4, f64x4, i8x4, u8x4, i16x4, u16x4, i32x4, u32x4, i64x4, u64x4);
        $mac!($arg; f32x8, f64x8, i8x8, u8x8, i16x8, u16x8, i32x8, u32x8, i64x8, u64x8);
        $mac!($arg; f32x16, f64x16, i8x16, u8x16, i16x16, u16x16, i32x16, u32x16, i64x16, u64x16);
    };
}

macro_rules! each_lane3 {
    ($mac:ident, $b:ty, $tr:ident, $vtr:ident) => {
        $mac!($b, $tr; f32x3A, f64x3A, i32x3A, u32x3A, i64x3A, u64x3A);
        $mac!($b, $vtr; f32x3, f64x3, i32x3, u32x3, i64x3, u64x3);
    };
}

for_each_backend_concrete! {
    /// Question 1 for the x2..x16 slots.
    fn matrix_exists() {
        each_lane!(concrete_pairs, S);
    }
    /// Question 1 for the 3-lane slots (`Simd3A` and `Simd3`).
    fn lanes3_exist() {
        each_lane3!(concrete_pairs_in, S, Simd3A, Simd3);
    }
}

#[allow(dead_code)]
mod generic_reachability {
    use super::*;

    pub fn registers<S: Simd>() {
        each_lane!(generic_pairs, assert_cast);
    }

    pub fn vectors<S: SimdVectors>() {
        each_lane!(generic_pairs, assert_vcast);
    }

    use thermite::simd::{Simd3AVectors, Simd3Vectors};

    macro_rules! pairs3 {
        ($assert:ident; $($slot:ident),* $(,)?) => {
            generic_pairs!($assert; $($slot),*);
        };
    }

    pub fn registers_3a<S: Simd3A>() {
        pairs3!(assert_cast; f32x3A, f64x3A, i32x3A, u32x3A, i64x3A, u64x3A);
    }

    pub fn registers_3<S: Simd3>() {
        pairs3!(assert_cast; f32x3, f64x3, i32x3, u32x3, i64x3, u64x3);
    }

    pub fn vectors_3a<S: Simd3AVectors>() {
        pairs3!(assert_vcast; f32x3A, f64x3A, i32x3A, u32x3A, i64x3A, u64x3A);
    }

    pub fn vectors_3<S: Simd3Vectors>() {
        pairs3!(assert_vcast; f32x3, f64x3, i32x3, u32x3, i64x3, u64x3);
    }
}
