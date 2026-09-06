//! Width-changing reshape ops, tested at the **`Vector` (public API) layer**:
//! `concat` / `split` / `extend` / `narrow`. These are pure lane-routing
//! operations, exactly the family a widening bug hides in, and they had no
//! lane-correctness coverage.
//!
//! Written once as a generic `fn check_reshape::<N, W>()` over a (narrow, wide)
//! `Vector` pair and instantiated per backend/width by `for_each_backend!`.
//! Testing at the `Vector` layer covers the `vector/` wrapper _and_ the
//! underlying register `Concat`/`Extend` impls.
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

/// Read a vector's lanes into a `Vec` for comparison.
fn lanes<V: GenericVector>(v: V) -> Vec<V::Element> {
    v.into_array().as_slice().to_vec()
}

/// Exercise concat/split/extend/narrow for a (narrow `N`, wide `W`) pair, where
/// `W` is exactly twice the lane count of `N`.
#[inline(always)]
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

/// (narrow slot, wide slot) pair of backend `$S`.
macro_rules! reshape {
    ($S:ty, $narrow:ident, $wide:ident) => {
        check_reshape::<Vector<<$S as Simd>::$narrow>, Vector<<$S as Simd>::$wide>>(&harness::label::<$S>(concat!(
            stringify!($narrow),
            "|",
            stringify!($wide)
        )))
    };
}

// The "half" of a 128-bit x2 register (`f64x2`/`i64x2`/`u64x2`) is the *scalar*
// element register (`Vector<f64>` etc.), not a Simd-aliased SIMD type, so the
// scalar/x2 reshape (`concat(scalar,scalar) -> x2`, `split`/`extend`/`narrow`) needs
// its own stamper.
macro_rules! reshape_half {
    ($S:ty, $elem:ty, $wide:ident) => {
        check_reshape::<Vector<$elem>, Vector<<$S as Simd>::$wide>>(&harness::label::<$S>(concat!(
            stringify!($elem),
            "|",
            stringify!($wide)
        )))
    };
}

for_each_backend! {
    fn floats<S: Simd>() {
        reshape!(S, f32x4, f32x8);
        reshape!(S, f32x8, f32x16);
        reshape!(S, f64x2, f64x4);
        reshape!(S, f64x4, f64x8);
        reshape_half!(S, f64, f64x2);
    }
    fn ints<S: Simd>() {
        reshape!(S, i32x4, i32x8);
        reshape!(S, i32x8, i32x16);
        reshape!(S, i64x2, i64x4);
        reshape!(S, i64x4, i64x8);
        reshape!(S, u32x4, u32x8);
        reshape!(S, u32x8, u32x16);
        reshape!(S, u64x2, u64x4);
        reshape!(S, u64x4, u64x8);
        reshape_half!(S, i64, i64x2);
        reshape_half!(S, u64, u64x2);
    }
    /// The sub-native 8-bit ladder, scalar element up to x16.
    fn bytes<S: Simd>() {
        reshape_half!(S, i8, i8x2);
        reshape_half!(S, u8, u8x2);
        reshape!(S, i8x2, i8x4);
        reshape!(S, u8x2, u8x4);
        reshape!(S, i8x4, i8x8);
        reshape!(S, u8x4, u8x8);
        reshape!(S, i8x8, i8x16);
        reshape!(S, u8x8, u8x16);
    }
}
