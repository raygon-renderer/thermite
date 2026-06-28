//! Coverage for two previously-0% modules:
//!   * `isa/mod.rs` — `InstructionSet` detection + the `num_registers`/`has_fma`
//!     classifiers (const fns; covered only when called at *runtime*).
//!   * `backend/generic/polyfills/sort.rs` — the sorting networks (`sort_2`/`_4`/
//!     `_8`/`sort_any`) reached through `NumericRegister::sort`.
//!
//! Sorting is checked by the identity `sort([n-1, …, 1, 0]) == [0, 1, …, n-1]`:
//! `indexed()` is a known distinct ascending ramp, so its reverse must sort back
//! to it exactly.
#![cfg(any(target_arch = "x86", target_arch = "x86_64", target_arch = "wasm32"))]

use thermite::Vector;
use thermite::isa::InstructionSet;
use thermite::prelude::*;
use thermite::register::NumericRegister;
use thermite::simd::Simd;

use thermite::backend::scalar::Scalar;

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

    // the derived ordering reflects increasing capability
    assert!(InstructionSet::X86V3 > InstructionSet::X86V2);
    assert!(InstructionSet::X86V2 > InstructionSet::X86V1);
    assert!(InstructionSet::Scalar < InstructionSet::X86V1);
    assert!(InstructionSet::X86V3.has_fma() && !InstructionSet::X86V2.has_fma());
}

fn check_sort<R>(label: &str)
where
    R: NumericRegister,
    Vector<R>: NumericVector<Element = R::Element>,
    R::Element: PartialEq + core::fmt::Debug,
{
    // [n-1, …, 1, 0] must sort to [0, 1, …, n-1] == indexed()
    let asc = Vector::<R>::indexed();
    let desc = asc.reverse();
    let sorted = Vector::<R>(<R as NumericRegister>::sort(desc.0));
    assert_eq!(
        sorted.into_array().as_slice(),
        asc.into_array().as_slice(),
        "{label}: sort(reverse(indexed)) != indexed",
    );
}

macro_rules! sort_suite {
    ($modname:ident, $backend:ty, $bl:expr) => {
        mod $modname {
            use super::*;
            macro_rules! t {
                ($name:ident, $reg:ident) => {
                    #[test]
                    fn $name() {
                        check_sort::<<$backend as Simd>::$reg>(concat!($bl, " ", stringify!($reg)));
                    }
                };
            }
            t!(f32x4, f32x4);
            t!(f32x8, f32x8);
            t!(f64x2, f64x2);
            t!(f64x4, f64x4);
            t!(i32x4, i32x4);
            t!(i32x8, i32x8);
            t!(i64x2, i64x2);
            t!(u32x4, u32x4);
            t!(u64x2, u64x2);
        }
    };
}

sort_suite!(scalar, Scalar, "scalar");

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
mod x86 {
    use super::*;
    use thermite::backend::x86_v1::X86V1;
    use thermite::backend::x86_v2::X86V2;
    use thermite::backend::x86_v3::X86V3;
    sort_suite!(v3, X86V3, "x86_v3");
    sort_suite!(v2, X86V2, "x86_v2");
    sort_suite!(v1, X86V1, "x86_v1");
}

#[cfg(target_arch = "wasm32")]
mod wasm {
    use super::*;
    use thermite::backend::wasm::Wasm;
    sort_suite!(wasm, Wasm, "wasm");
}
