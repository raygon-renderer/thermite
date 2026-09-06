//! Under `strict_ieee754`, float -> int `cast` must match Rust's `as` exactly.
//!
//! Outside that feature `cast` is only defined for in-range finite lanes, and a
//! NaN or out-of-range lane gets whatever the hardware conversion produces (on
//! x86, the "indefinite" integer `INT::MIN`). `strict_ieee754` is supposed to
//! close that gap by routing float -> int `cast_from` at the saturating
//! implementation, which is `as`-exact: NaN gives 0, out-of-range clamps to the
//! destination MIN/MAX.
//!
//! The scalar backend's `cast_from` is literally `value as _` in every
//! configuration, so it is the oracle here regardless of the feature. Each pair
//! runs the full corpus, NaN, infinities and out-of-range magnitudes included,
//! which is exactly the domain the non-strict contract leaves open.
#![cfg(feature = "strict_ieee754")]
#![cfg(any(
    target_arch = "x86",
    target_arch = "x86_64",
    target_arch = "wasm32",
    target_arch = "aarch64"
))]

mod harness;

use harness::Tol;
use thermite::Vector;
use thermite::backend::scalar::Scalar;
use thermite::register::CoreRegister;
use thermite::simd::{Simd, Simd3, Simd3A};
use thermite::vector::CastVector;

/// One float -> int pair of backend `S` (slot family `$tr`) vs the scalar `as`.
macro_rules! strict_cast_diff {
    ($tr:ident, $src:ident, $dst:ident, $se:ty) => {{
        let label = harness::label::<S>(concat!(stringify!($src), "->", stringify!($dst)));
        let mut rng = harness::rng();
        let lanes = <<<S as $tr>::$src as CoreRegister>::Lanes as generic_array::typenum::Unsigned>::USIZE;
        for input in harness::corpus::<$se>(lanes, &mut rng) {
            let src = Vector::<<S as $tr>::$src>(harness::make_array::<<S as $tr>::$src>(&input));
            let got = harness::read::<<S as $tr>::$dst>(&<Vector<<S as $tr>::$dst> as CastVector<_>>::cast_from(src).0);

            let rsrc = Vector::<<Scalar as $tr>::$src>(harness::make_array::<<Scalar as $tr>::$src>(&input));
            let want = harness::read::<<Scalar as $tr>::$dst>(
                &<Vector<<Scalar as $tr>::$dst> as CastVector<_>>::cast_from(rsrc).0,
            );
            harness::assert_lanes_eq(
                &format!("{label} [strict cast vs scalar `as`]"),
                &[],
                &got,
                &want,
                Tol::Exact,
            );
        }
    }};
}

macro_rules! strict_same_width {
    ($tr:ident, $f32:ident, $f64:ident, $i32:ident, $u32:ident, $i64:ident, $u64:ident) => {{
        strict_cast_diff!($tr, $f32, $i32, f32);
        strict_cast_diff!($tr, $f32, $u32, f32);
        strict_cast_diff!($tr, $f64, $i64, f64);
        strict_cast_diff!($tr, $f64, $u64, f64);
    }};
}

macro_rules! strict_narrowing {
    ($f32:ident, $f64:ident, [$($dst:ident),* $(,)?]) => {{
        $( strict_cast_diff!(Simd, $f32, $dst, f32); )*
        $( strict_cast_diff!(Simd, $f64, $dst, f64); )*
    }};
}

for_each_backend_concrete! {
    fn same_width_cast_is_exact() {
        strict_same_width!(Simd, f32x2, f64x2, i32x2, u32x2, i64x2, u64x2);
        strict_same_width!(Simd, f32x4, f64x4, i32x4, u32x4, i64x4, u64x4);
        strict_same_width!(Simd, f32x8, f64x8, i32x8, u32x8, i64x8, u64x8);
        strict_same_width!(Simd, f32x16, f64x16, i32x16, u32x16, i64x16, u64x16);
        strict_same_width!(Simd3A, f32x3A, f64x3A, i32x3A, u32x3A, i64x3A, u64x3A);
        strict_same_width!(Simd3, f32x3, f64x3, i32x3, u32x3, i64x3, u64x3);
    }

    fn narrowing_cast_is_exact() {
        strict_narrowing!(f32x2, f64x2, [i8x2, u8x2, i16x2, u16x2]);
        strict_narrowing!(f32x4, f64x4, [i8x4, u8x4, i16x4, u16x4]);
        strict_narrowing!(f32x8, f64x8, [i8x8, u8x8, i16x8, u16x8]);
        strict_narrowing!(f32x16, f64x16, [i8x16, u8x16, i16x16, u16x16]);
    }
}
