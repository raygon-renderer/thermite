//! Numeric `cast` (conversion) polyfill audit.
//!
//! Int↔float and width-changing conversions are heavily polyfilled
//! (`_mm*_cvt*`, `convert_*_limited`). The scalar backend's `cast_from` is
//! literally `value as _`, so each test is a differential against Rust's
//! built-in `as` - the documented "like `as`" contract for `cast`.
//!
//! `mod gate` is the always-green correctness gate. Float→int conversions are
//! kept in the **in-range, finite** domain there (where the contract is
//! unambiguous and the `_limited` polyfills' precondition holds). The
//! out-of-range / NaN behaviour, where the x86 hardware path returns the
//! "indefinite" integer instead of saturating like `as`, is a documented
//! divergence captured (ignored) in `mod divergence`.
#![cfg(any(target_arch = "x86", target_arch = "x86_64", target_arch = "wasm32"))]

mod harness;

use harness::Tol;
use thermite::backend::scalar::Scalar;
use thermite::simd::Simd;

// identity prep (int→int, int→float, float→float: scalar uses the same `as`,
// so the differential is bit-exact even when the conversion itself rounds).
macro_rules! id {
    ($t:ty) => {
        (|x: $t| x) as fn($t) -> $t
    };
}

/// One cast pair, X86 backend vs scalar, for a given (backend, src, dst).
macro_rules! cpair {
    ($l:expr, $b:ty, $src:ident, $dst:ident, $se:ty, $prep:expr, $tol:expr) => {
        cast_diff!(
            $l,
            <$b as Simd>::$src,
            <$b as Simd>::$dst,
            <Scalar as Simd>::$src,
            <Scalar as Simd>::$dst,
            $se,
            $prep,
            $tol
        );
    };
}

// Float→int domain guards: keep strictly in-range & finite so truncation is
// unambiguous and matches `as` on both backends.
fn to_i32_dom(x: f32) -> f32 {
    if x.is_finite() { x.clamp(-2.0e9, 2.0e9) } else { 0.0 }
}
fn to_u32_dom(x: f32) -> f32 {
    if x.is_finite() { x.clamp(0.0, 4.0e9) } else { 0.0 }
}
fn to_i64_dom(x: f64) -> f64 {
    if x.is_finite() { x.clamp(-9.0e18, 9.0e18) } else { 0.0 }
}

macro_rules! cast_suite {
    ($modname:ident, $b:ty, $tag:expr) => {
        mod $modname {
            use super::*;

            // --- element swaps, same lane width (reinterpret as number) ---
            #[test]
            fn swaps_32() {
                cpair!($tag, $b, i32x4, u32x4, i32, id!(i32), Tol::Exact);
                cpair!($tag, $b, u32x4, i32x4, u32, id!(u32), Tol::Exact);
                cpair!($tag, $b, i32x4, f32x4, i32, id!(i32), Tol::Exact);
                cpair!($tag, $b, u32x4, f32x4, u32, id!(u32), Tol::Exact);
                cpair!($tag, $b, i32x8, f32x8, i32, id!(i32), Tol::Exact);
                cpair!($tag, $b, u32x8, f32x8, u32, id!(u32), Tol::Exact);
            }
            #[test]
            fn swaps_64() {
                cpair!($tag, $b, i64x2, u64x2, i64, id!(i64), Tol::Exact);
                cpair!($tag, $b, u64x2, i64x2, u64, id!(u64), Tol::Exact);
                cpair!($tag, $b, i64x2, f64x2, i64, id!(i64), Tol::Exact);
                cpair!($tag, $b, u64x2, f64x2, u64, id!(u64), Tol::Exact);
                cpair!($tag, $b, i64x4, f64x4, i64, id!(i64), Tol::Exact);
                cpair!($tag, $b, u64x4, f64x4, u64, id!(u64), Tol::Exact);
            }

            // --- width-changing int/float conversions ---
            #[test]
            fn widths() {
                cpair!($tag, $b, i32x4, i64x4, i32, id!(i32), Tol::Exact); // widen
                cpair!($tag, $b, i64x4, i32x4, i64, id!(i64), Tol::Exact); // narrow
                cpair!($tag, $b, u32x4, u64x4, u32, id!(u32), Tol::Exact);
                cpair!($tag, $b, f32x4, f64x4, f32, id!(f32), Tol::Exact);
                cpair!($tag, $b, f64x4, f32x4, f64, id!(f64), Tol::Exact);
            }

            // --- float→int, kept strictly in-range/finite ---
            #[test]
            fn f32_to_i32_inrange() {
                cpair!($tag, $b, f32x4, i32x4, f32, to_i32_dom, Tol::Exact);
            }
            #[test]
            fn f32_to_u32_inrange() {
                cpair!($tag, $b, f32x4, u32x4, f32, to_u32_dom, Tol::Exact);
            }
            #[test]
            fn f64_to_i64_inrange() {
                cpair!($tag, $b, f64x4, i64x4, f64, to_i64_dom, Tol::Exact);
            }
            // f64→u64 is NOT in the gate: it routes to the `_limited` epu64
            // polyfill, which both (a) only works on [0, 2^52) and (b) *rounds*
            // (adds 2^52 to force integer rounding) rather than truncating like
            // `as`. Fully captured in `mod divergence`.
        }
    };
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
mod x86 {
use super::*;
use thermite::backend::x86_v1::X86V1;
use thermite::backend::x86_v2::X86V2;
use thermite::backend::x86_v3::X86V3;

mod gate {
    use super::*;
    cast_suite!(v3, X86V3, "x86_v3");
    cast_suite!(v2, X86V2, "x86_v2");
    cast_suite!(v1, X86V1, "x86_v1");
}

// Out-of-range / NaN float→int: scalar saturates (Rust `as`), the x86 hardware
// path returns the "indefinite" integer (i64::MIN / i32::MIN). Documented
// divergence, not auto-failed - see TESTING.md.
mod divergence {
    use super::*;

    #[test]
    #[ignore = "DIVERGENCE: out-of-range/NaN float→int returns the hardware \
                indefinite integer instead of saturating like `as` (scalar). \
                The general `cast` contract is 'like as'; x86 needs a clamp or \
                the `_limited` precondition must be documented."]
    fn float_to_int_out_of_range() {
        cast_diff!(
            "x86_v3 f64x4->i64x4 OOR",
            <X86V3 as Simd>::f64x4,
            <X86V3 as Simd>::i64x4,
            <Scalar as Simd>::f64x4,
            <Scalar as Simd>::i64x4,
            f64,
            |x| x,
            Tol::Exact
        );
        cast_diff!(
            "x86_v3 f32x4->i32x4 OOR",
            <X86V3 as Simd>::f32x4,
            <X86V3 as Simd>::i32x4,
            <Scalar as Simd>::f32x4,
            <Scalar as Simd>::i32x4,
            f32,
            |x| x,
            Tol::Exact
        );
    }

    #[test]
    #[ignore = "DIVERGENCE: f64→u64 `cast` routes to `_mm*_cvtpd_epu64x_limited_*` \
                which (a) only works on [0, 2^52) and (b) ROUNDS (adds 2^52) \
                instead of truncating like `as` - so e.g. 2.7_f64 as u64 == 2 \
                but the cast yields 3, and values ≥ 2^52 are corrupted. \
                f64→i64 is full-range-correct and truncating; f64→u64 needs an \
                equivalent path or a documented precondition."]
    fn f64_to_u64_nonconforming() {
        // (b) rounds vs truncates, even for tiny in-range values.
        cast_diff!(
            "x86_v3 f64x4->u64x4 frac",
            <X86V3 as Simd>::f64x4,
            <X86V3 as Simd>::u64x4,
            <Scalar as Simd>::f64x4,
            <Scalar as Simd>::u64x4,
            f64,
            |x: f64| if x.is_finite() { (x.abs() % 1000.0) + 0.7 } else { 2.7 },
            Tol::Exact
        );
    }
}
}

#[cfg(target_arch = "wasm32")]
mod wasm {
use super::*;
use thermite::backend::wasm::Wasm;

mod gate {
    use super::*;
    cast_suite!(wasm, Wasm, "wasm");
}

// Out-of-range / NaN float→int: scalar saturates (Rust `as`), the wasm hardware
// path returns the "indefinite" integer (i64::MIN / i32::MIN). Documented
// divergence, not auto-failed - see TESTING.md.
mod divergence {
    use super::*;

    #[test]
    #[ignore = "DIVERGENCE: out-of-range/NaN float→int returns the hardware \
                indefinite integer instead of saturating like `as` (scalar). \
                The general `cast` contract is 'like as'; backend needs a clamp or \
                the `_limited` precondition must be documented."]
    fn float_to_int_out_of_range() {
        cast_diff!(
            "wasm f64x4->i64x4 OOR",
            <Wasm as Simd>::f64x4,
            <Wasm as Simd>::i64x4,
            <Scalar as Simd>::f64x4,
            <Scalar as Simd>::i64x4,
            f64,
            |x| x,
            Tol::Exact
        );
        cast_diff!(
            "wasm f32x4->i32x4 OOR",
            <Wasm as Simd>::f32x4,
            <Wasm as Simd>::i32x4,
            <Scalar as Simd>::f32x4,
            <Scalar as Simd>::i32x4,
            f32,
            |x| x,
            Tol::Exact
        );
    }

    #[test]
    #[ignore = "DIVERGENCE: f64→u64 `cast` routes to a `_limited` polyfill \
                which (a) only works on [0, 2^52) and (b) ROUNDS (adds 2^52) \
                instead of truncating like `as` - so e.g. 2.7_f64 as u64 == 2 \
                but the cast yields 3, and values ≥ 2^52 are corrupted. \
                f64→i64 is full-range-correct and truncating; f64→u64 needs an \
                equivalent path or a documented precondition."]
    fn f64_to_u64_nonconforming() {
        // (b) rounds vs truncates, even for tiny in-range values.
        cast_diff!(
            "wasm f64x4->u64x4 frac",
            <Wasm as Simd>::f64x4,
            <Wasm as Simd>::u64x4,
            <Scalar as Simd>::f64x4,
            <Scalar as Simd>::u64x4,
            f64,
            |x: f64| if x.is_finite() { (x.abs() % 1000.0) + 0.7 } else { 2.7 },
            Tol::Exact
        );
    }
}
}
