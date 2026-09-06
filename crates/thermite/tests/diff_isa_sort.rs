//! Coverage for two modules:
//!   * `isa/mod.rs`: `InstructionSet` detection plus the `num_registers`/`has_fma`
//!     classifiers (const fns, covered only when called at *runtime*).
//!   * `backend/generic/polyfills/sort.rs`: the sorting networks (`sort_2`/`_4`/
//!     `_8`/`sort_any`) reached through `NumericRegister::sort`, on every backend.
//!
//! Sorting is checked by the identity `sort([n-1, ..., 1, 0]) == [0, 1, ..., n-1]`:
//! `indexed()` is a known distinct ascending ramp, so its reverse must sort back
//! to it exactly.
#![cfg(any(
    target_arch = "x86",
    target_arch = "x86_64",
    target_arch = "wasm32",
    target_arch = "aarch64"
))]

mod harness;

use thermite::Vector;
// Which of these are used varies by backend cfg (x86 / wasm / neon).
#[allow(unused_imports)]
use thermite::isa::InstructionSet;
use thermite::prelude::*;
use thermite::register::NumericRegister;
use thermite::simd::Simd;

// ISA detection/ordering is x86-specific (on wasm `get()` returns a WASM set).
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[test]
fn instruction_set() {
    // runtime detection returns a concrete supported set on this host
    let cur = InstructionSet::get();
    assert!(
        matches!(
            cur,
            InstructionSet::Scalar
                | InstructionSet::X86V1
                | InstructionSet::X86V2
                | InstructionSet::X86V3
                | InstructionSet::X86V4
        ),
        "unexpected detected ISA: {cur:?}"
    );

    // exercise the const-fn classifiers at runtime (const-eval is not instrumented)
    // across every variant, hitting all match arms.
    let all = [
        InstructionSet::Scalar,
        InstructionSet::X86V1,
        InstructionSet::X86V2,
        InstructionSet::X86V3,
        InstructionSet::X86V4,
    ];
    for is in all {
        assert!(is.num_registers() >= 1, "{is:?} num_registers");
        let _ = is.has_fma();
        let _ = format!("{is:?}"); // Debug
    }

    // the derived ordering reflects increasing capability (the harness's ISA
    // gate relies on this)
    assert!(InstructionSet::X86V4 > InstructionSet::X86V3);
    assert!(InstructionSet::X86V3 > InstructionSet::X86V2);
    assert!(InstructionSet::X86V2 > InstructionSet::X86V1);
    assert!(InstructionSet::Scalar < InstructionSet::X86V1);
    assert!(InstructionSet::X86V3.has_fma() && !InstructionSet::X86V2.has_fma());
}

#[inline(always)]
fn check_sort<R>(label: &str)
where
    R: NumericRegister,
    Vector<R>: NumericVector<Element = R::Element>,
    R::Element: PartialEq + core::fmt::Debug,
{
    // [n-1, ..., 1, 0] must sort to [0, 1, ..., n-1] == indexed()
    let asc = Vector::<R>::indexed();
    let desc = asc.reverse();
    let sorted = Vector::<R>(<R as NumericRegister>::sort(desc.0));
    assert_eq!(
        sorted.into_array().as_slice(),
        asc.into_array().as_slice(),
        "{label}: sort(reverse(indexed)) != indexed",
    );
}

macro_rules! sort {
    ($S:ty, $reg:ident) => {
        check_sort::<<$S as Simd>::$reg>(&harness::label::<$S>(stringify!($reg)))
    };
}

for_each_backend! {
    fn f32x4<S: Simd>() { sort!(S, f32x4) }
    fn f32x8<S: Simd>() { sort!(S, f32x8) }
    fn f32x16<S: Simd>() { sort!(S, f32x16) }
    fn f64x2<S: Simd>() { sort!(S, f64x2) }
    fn f64x4<S: Simd>() { sort!(S, f64x4) }
    fn f64x8<S: Simd>() { sort!(S, f64x8) }
    fn i32x4<S: Simd>() { sort!(S, i32x4) }
    fn i32x8<S: Simd>() { sort!(S, i32x8) }
    fn i32x16<S: Simd>() { sort!(S, i32x16) }
    fn i64x2<S: Simd>() { sort!(S, i64x2) }
    fn i64x4<S: Simd>() { sort!(S, i64x4) }
    fn u32x4<S: Simd>() { sort!(S, u32x4) }
    fn u32x8<S: Simd>() { sort!(S, u32x8) }
    fn u64x2<S: Simd>() { sort!(S, u64x2) }
    fn u64x4<S: Simd>() { sort!(S, u64x4) }
    fn i16x8<S: Simd>() { sort!(S, i16x8) }
    fn u16x16<S: Simd>() { sort!(S, u16x16) }
}
