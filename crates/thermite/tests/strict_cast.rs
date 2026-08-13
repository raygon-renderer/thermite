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
//!
//! # The surface
//!
//! `CastRegister<f32xN/f64xN>` is required for all eight integer
//! element types in `simd.rs`, but `CastRegister` is not, so the two do not
//! cover the same pairs and only the overlap can be tested here:
//!
//! - **same-width** (`f32 -> i32/u32`, `f64 -> i64/u64`): both exist. These are
//!   the pairs `impl_float_to_int_casts!` stamps, so the toggle reaches them.
//! - **narrowing** (`f32`/`f64` into 8- and 16-bit ints): both exist, but the
//!   `CastRegister` impls are hand-written in `registers/half8.rs` and
//!   `half16.rs` and the toggle does not reach them.
//! - **cross-width 32/64** (`f32 -> i64/u64`, `f64 -> i32/u32`): saturating
//!   only. There is no `CastRegister` for these at all, so `cast` does not
//!   compile and there is nothing for the feature to override.
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

/// One float -> int `cast` pair against the scalar `as` oracle, bit-exact over
/// the raw corpus (no domain prep, which is the whole point).
///
/// Deliberately at the VECTOR layer. `strict_ieee754` redirects `cast` in the
/// `Vector<R>` impl of [`CastVector`], not in the register impls, so a register
/// -level `CastRegister::cast_from` here would bypass the feature entirely and
/// this suite would fail for the wrong reason.
macro_rules! strict_cast_diff {
    ($label:expr, $tr:ident, $b:ty, $src:ident, $dst:ident, $se:ty) => {{
        let mut rng = harness::rng();
        let lanes = <<<$b as $tr>::$src as CoreRegister>::Lanes as generic_array::typenum::Unsigned>::USIZE;
        for input in harness::corpus::<$se>(lanes, &mut rng) {
            let src = Vector::<<$b as $tr>::$src>(harness::make_array::<<$b as $tr>::$src>(&input));
            let got =
                harness::read::<<$b as $tr>::$dst>(&<Vector<<$b as $tr>::$dst> as CastVector<_>>::cast_from(src).0);

            let rsrc = Vector::<<Scalar as $tr>::$src>(harness::make_array::<<Scalar as $tr>::$src>(&input));
            let want = harness::read::<<Scalar as $tr>::$dst>(
                &<Vector<<Scalar as $tr>::$dst> as CastVector<_>>::cast_from(rsrc).0,
            );
            harness::assert_lanes_eq(
                concat!($label, " [strict cast vs scalar `as`]"),
                &[],
                &got,
                &want,
                Tol::Exact,
            );
        }
    }};
}

/// The same-width pairs, the ones `impl_float_to_int_casts!` stamps.
macro_rules! strict_same_width {
    ($tr:ident, $b:ty, $label:expr, $f32:ident, $f64:ident, $i32:ident, $u32:ident, $i64:ident, $u64:ident) => {{
        strict_cast_diff!(concat!($label, " f32->", stringify!($i32)), $tr, $b, $f32, $i32, f32);
        strict_cast_diff!(concat!($label, " f32->", stringify!($u32)), $tr, $b, $f32, $u32, f32);
        strict_cast_diff!(concat!($label, " f64->", stringify!($i64)), $tr, $b, $f64, $i64, f64);
        strict_cast_diff!(concat!($label, " f64->", stringify!($u64)), $tr, $b, $f64, $u64, f64);
    }};
}

/// Both float sources into the 8- and 16-bit integer destinations of one lane
/// count. `Simd` only carries these for the power-of-two widths.
macro_rules! strict_narrowing {
    ($b:ty, $label:expr, $f32:ident, $f64:ident, [$($dst:ident),* $(,)?]) => {{
        $( strict_cast_diff!(concat!($label, " f32->", stringify!($dst)), Simd, $b, $f32, $dst, f32); )*
        $( strict_cast_diff!(concat!($label, " f64->", stringify!($dst)), Simd, $b, $f64, $dst, f64); )*
    }};
}

/// The pairs the toggle currently reaches.
macro_rules! strict_same_width_all {
    ($b:ty, $label:expr) => {{
        strict_same_width!(
            Simd,
            $b,
            concat!($label, " x2"),
            f32x2,
            f64x2,
            i32x2,
            u32x2,
            i64x2,
            u64x2
        );
        strict_same_width!(
            Simd,
            $b,
            concat!($label, " x4"),
            f32x4,
            f64x4,
            i32x4,
            u32x4,
            i64x4,
            u64x4
        );
        strict_same_width!(
            Simd,
            $b,
            concat!($label, " x8"),
            f32x8,
            f64x8,
            i32x8,
            u32x8,
            i64x8,
            u64x8
        );
        strict_same_width!(
            Simd,
            $b,
            concat!($label, " x16"),
            f32x16,
            f64x16,
            i32x16,
            u32x16,
            i64x16,
            u64x16
        );
        strict_same_width!(
            Simd3A,
            $b,
            concat!($label, " x3A"),
            f32x3A,
            f64x3A,
            i32x3A,
            u32x3A,
            i64x3A,
            u64x3A
        );
        strict_same_width!(
            Simd3,
            $b,
            concat!($label, " x3"),
            f32x3,
            f64x3,
            i32x3,
            u32x3,
            i64x3,
            u64x3
        );
    }};
}

/// The pairs it does not. Hand-written `CastRegister` impls in `half8.rs` and
/// `half16.rs`, none of which carry the `strict_ieee754` cfg.
macro_rules! strict_narrowing_all {
    ($b:ty, $label:expr) => {{
        strict_narrowing!($b, concat!($label, " x2"), f32x2, f64x2, [i8x2, u8x2, i16x2, u16x2]);
        strict_narrowing!($b, concat!($label, " x4"), f32x4, f64x4, [i8x4, u8x4, i16x4, u16x4]);
        strict_narrowing!($b, concat!($label, " x8"), f32x8, f64x8, [i8x8, u8x8, i16x8, u16x8]);
        strict_narrowing!(
            $b,
            concat!($label, " x16"),
            f32x16,
            f64x16,
            [i8x16, u8x16, i16x16, u16x16]
        );
    }};
}

macro_rules! strict_all_widths {
    ($b:ty, $label:expr) => {{
        strict_same_width_all!($b, $label);
        strict_narrowing_all!($b, $label);
    }};
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
mod x86 {
    use super::*;
    use thermite::backend::x86_v1::X86V1;
    use thermite::backend::x86_v2::X86V2;
    use thermite::backend::x86_v3::X86V3;

    #[test]
    fn v1_same_width_cast_is_exact() {
        strict_same_width_all!(X86V1, "v1");
    }

    #[test]
    fn v2_same_width_cast_is_exact() {
        if !is_x86_feature_detected!("sse4.2") {
            return;
        }
        strict_same_width_all!(X86V2, "v2");
    }

    #[test]
    fn v3_same_width_cast_is_exact() {
        if !is_x86_feature_detected!("avx2") {
            return;
        }
        strict_same_width_all!(X86V3, "v3");
    }

    #[test]
    fn v1_narrowing_cast_is_exact() {
        strict_narrowing_all!(X86V1, "v1");
    }

    #[test]
    fn v2_narrowing_cast_is_exact() {
        if !is_x86_feature_detected!("sse4.2") {
            return;
        }
        strict_narrowing_all!(X86V2, "v2");
    }

    #[test]
    fn v3_narrowing_cast_is_exact() {
        if !is_x86_feature_detected!("avx2") {
            return;
        }
        strict_narrowing_all!(X86V3, "v3");
    }
}

#[cfg(target_arch = "aarch64")]
mod neon {
    use super::*;
    use thermite::backend::neon::Neon;

    /// `FCVTZS`/`FCVTZU` already saturate and map NaN to 0, so `cast` is exact
    /// here with or without the feature. Covered anyway to pin that down.
    #[test]
    fn neon_float_to_int_cast_is_exact() {
        strict_all_widths!(Neon, "neon");
    }
}

#[cfg(target_arch = "wasm32")]
mod wasm {
    use super::*;
    use thermite::backend::wasm::Wasm;

    /// SIMD128's `trunc_sat` ops are `as`-exact by spec, same as NEON.
    #[test]
    fn wasm_float_to_int_cast_is_exact() {
        strict_all_widths!(Wasm, "wasm");
    }
}
