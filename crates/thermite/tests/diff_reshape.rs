//! Width-changing reshape ops, tested at the **`Vector` (public API) layer**:
//! `concat` / `split` / `extend` / `narrow`. These are pure lane-routing
//! operations - exactly the family the #C2 widening bug lived in - and had no
//! lane-correctness coverage.
//!
//! Unlike the register-layer suites, this is written once as a generic
//! `fn check_reshape::<N, W>()` over a (narrow, wide) `Vector` pair and then
//! instantiated per backend/width. Testing at the `Vector` layer covers the
//! `vector/` wrapper *and* the underlying register `Concat`/`Extend` impls.
//!
//! Oracles are trivial and exact (bit-preserving, so NaN lanes must match too):
//!   concat(lo, hi) == lo ++ hi          split(concat(lo,hi)) == (lo, hi)
//!   extend(lo)     == lo ++ [0; HALF]   narrow(concat(lo,hi)) == lo
#![cfg(any(
    target_arch = "x86",
    target_arch = "x86_64",
    target_arch = "wasm32",
    target_arch = "aarch64"
))]

mod harness;

use harness::{Diff, Tol};
use thermite::Vector;
use thermite::simd::Simd;
use thermite::vector::{ConcatVector, ExtendVector, GenericVector};

use thermite::backend::scalar::Scalar;

/// Read a vector's lanes into a `Vec` for comparison.
fn lanes<V: GenericVector>(v: V) -> Vec<V::Element> {
    v.into_array().as_slice().to_vec()
}

/// Exercise concat/split/extend/narrow for a (narrow `N`, wide `W`) pair, where
/// `W` is exactly twice the lane count of `N`.
fn check_reshape<N, W>(label: &str)
where
    N: GenericVector<Element: Diff + Default>,
    W: GenericVector<Element = N::Element> + ConcatVector<N> + ExtendVector<N>,
{
    let mut rng = harness::rng();
    let nl = N::LANES;
    let zero = <N::Element as Default>::default();

    let los = harness::corpus::<N::Element>(nl, &mut rng);
    let his = harness::corpus::<N::Element>(nl, &mut rng);

    for (lo_in, hi_in) in los.iter().zip(his.iter()) {
        let lo = N::from_slice(lo_in);
        let hi = N::from_slice(hi_in);

        // concat: lo's lanes in the low half, hi's in the high half.
        let w: W = lo.concat::<W>(hi);
        let mut want = lo_in.clone();
        want.extend_from_slice(hi_in);
        harness::assert_lanes_eq(&format!("{label} [concat]"), &[], &lanes(w), &want, Tol::Exact);

        // split: the exact inverse of concat. (Fully qualified: `Concat::split`
        // is also in scope via the `ConcatVector` supertrait.)
        let (slo, shi) = GenericVector::split::<N>(w);
        harness::assert_lanes_eq(&format!("{label} [split.lo]"), &[], &lanes(slo), lo_in, Tol::Exact);
        harness::assert_lanes_eq(&format!("{label} [split.hi]"), &[], &lanes(shi), hi_in, Tol::Exact);

        // extend: lo's lanes in the low half, zeros in the high half.
        let ext: W = lo.extend::<W>();
        let mut want_ext = lo_in.clone();
        want_ext.resize(nl * 2, zero);
        harness::assert_lanes_eq(&format!("{label} [extend]"), &[], &lanes(ext), &want_ext, Tol::Exact);

        // narrow: keep the low half, drop the high half. On `concat(lo, hi)`
        // the low half is `lo`.
        let nr: N = GenericVector::narrow::<N>(w);
        harness::assert_lanes_eq(&format!("{label} [narrow]"), &[], &lanes(nr), lo_in, Tol::Exact);
    }
}

macro_rules! reshape {
    ($name:ident, $backend:ty, $narrow:ident, $wide:ident, $label:expr) => {
        #[test]
        fn $name() {
            check_reshape::<Vector<<$backend as Simd>::$narrow>, Vector<<$backend as Simd>::$wide>>($label);
        }
    };
}

// The "half" of a 128-bit x2 register (`f64x2`/`i64x2`/`u64x2`) is the *scalar*
// element register (`Vector<f64>` etc.), not a Simd-aliased SIMD type — so the
// scalar↔x2 reshape (`concat(scalar,scalar)→x2`, `split`/`extend`/`narrow`) needs
// its own pair. These are exactly the `concat`/`split`/`extend`/`narrow` lines in
// the 64-bit register files that the x2→x4 pairs never reach.
macro_rules! reshape_half {
    ($name:ident, $backend:ty, $elem:ty, $wide:ident, $label:expr) => {
        #[test]
        fn $name() {
            check_reshape::<Vector<$elem>, Vector<<$backend as Simd>::$wide>>($label);
        }
    };
}

// Reshape pair where both widths are `Simd` slots (the 8-bit ladder).
macro_rules! reshape_exp {
    ($name:ident, $backend:ty, $narrow:ident, $wide:ident, $label:expr) => {
        #[test]
        fn $name() {
            check_reshape::<Vector<<$backend as Simd>::$narrow>, Vector<<$backend as Simd>::$wide>>($label);
        }
    };
}

// scalar-element <-> x2 reshape for an experimental (8-bit) x2 slot.
macro_rules! reshape_exp_half {
    ($name:ident, $backend:ty, $elem:ty, $wide:ident, $label:expr) => {
        #[test]
        fn $name() {
            check_reshape::<Vector<$elem>, Vector<<$backend as Simd>::$wide>>($label);
        }
    };
}

// The 8-bit ReducedRegister ladder: scalar<->x2, x2<->x4, x4<->x8, x8<->x16.
macro_rules! reshape8_suite {
    ($modname:ident, $backend:ty, $bl:expr) => {
        mod $modname {
            use super::*;
            reshape_exp_half!(i8_x2, $backend, i8, i8x2, concat!($bl, " i8|i8x2"));
            reshape_exp_half!(u8_x2, $backend, u8, u8x2, concat!($bl, " u8|u8x2"));
            reshape_exp!(i8x2_x4, $backend, i8x2, i8x4, concat!($bl, " i8x2|i8x4"));
            reshape_exp!(u8x2_x4, $backend, u8x2, u8x4, concat!($bl, " u8x2|u8x4"));
            reshape_exp!(i8x4_x8, $backend, i8x4, i8x8, concat!($bl, " i8x4|i8x8"));
            reshape_exp!(u8x4_x8, $backend, u8x4, u8x8, concat!($bl, " u8x4|u8x8"));
            reshape_exp!(i8x8_x16, $backend, i8x8, i8x16, concat!($bl, " i8x8|i8x16"));
            reshape_exp!(u8x8_x16, $backend, u8x8, u8x16, concat!($bl, " u8x8|u8x16"));
        }
    };
}

macro_rules! reshape_suite {
    ($modname:ident, $backend:ty, $bl:expr) => {
        mod $modname {
            use super::*;
            reshape!(f32x4_x8, $backend, f32x4, f32x8, concat!($bl, " f32x4|f32x8"));
            reshape!(f32x8_x16, $backend, f32x8, f32x16, concat!($bl, " f32x8|f32x16"));
            reshape!(f64x2_x4, $backend, f64x2, f64x4, concat!($bl, " f64x2|f64x4"));
            reshape!(f64x4_x8, $backend, f64x4, f64x8, concat!($bl, " f64x4|f64x8"));
            reshape!(i32x4_x8, $backend, i32x4, i32x8, concat!($bl, " i32x4|i32x8"));
            reshape!(i64x2_x4, $backend, i64x2, i64x4, concat!($bl, " i64x2|i64x4"));
            reshape!(u32x4_x8, $backend, u32x4, u32x8, concat!($bl, " u32x4|u32x8"));
            reshape!(u64x2_x4, $backend, u64x2, u64x4, concat!($bl, " u64x2|u64x4"));
            // scalar↔x2 halves (the 64-bit register files' own concat/split/extend/narrow)
            reshape_half!(f64_x2, $backend, f64, f64x2, concat!($bl, " f64|f64x2"));
            reshape_half!(i64_x2, $backend, i64, i64x2, concat!($bl, " i64|i64x2"));
            reshape_half!(u64_x2, $backend, u64, u64x2, concat!($bl, " u64|u64x2"));
        }
    };
}

reshape_suite!(scalar, Scalar, "scalar");
reshape8_suite!(scalar8, Scalar, "scalar");

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
mod x86 {
    use super::*;
    use thermite::backend::x86_v1::X86V1;
    use thermite::backend::x86_v2::X86V2;
    use thermite::backend::x86_v3::X86V3;
    reshape_suite!(v3, X86V3, "x86_v3");
    reshape_suite!(v2, X86V2, "x86_v2");
    reshape_suite!(v1, X86V1, "x86_v1");
    reshape8_suite!(v3_8, X86V3, "x86_v3");
    reshape8_suite!(v2_8, X86V2, "x86_v2");
    reshape8_suite!(v1_8, X86V1, "x86_v1");
}

#[cfg(target_arch = "wasm32")]
mod wasm {
    use super::*;
    use thermite::backend::wasm::Wasm;
    reshape_suite!(wasm, Wasm, "wasm");
    reshape8_suite!(wasm8, Wasm, "wasm");
}

#[cfg(target_arch = "aarch64")]
mod neon {
    use super::*;
    use thermite::backend::neon::Neon;
    reshape_suite!(neon, Neon, "neon");
    reshape8_suite!(neon8, Neon, "neon");
}
