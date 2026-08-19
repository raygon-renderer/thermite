//! Exhaustive differential for the same-length cast matrix.
//!
//! [`diff_cast`](../diff_cast.rs) hand-picks representative pairs and is where
//! the awkward float -> int domain questions are reasoned about one at a time.
//! This file is the opposite: it stamps *every* directed pair whose contract is
//! total and unambiguous, so a newly added lowering cannot land untested.
//!
//! The scalar backend is the oracle: its `cast_from` is literally `value as _`
//! and its `saturating_cast_from` is the documented clamp-then-convert, so
//! every assertion here is a differential against the language.
//!
//! What is covered, per backend and per lane count (90 directed pairs, the whole
//! matrix, at 4 lane counts across v1/v2/v3):
//!
//! - **integer source -> any destination**, both strengths, over the raw corpus.
//!   72 pairs. No prep and `Tol::Exact`: an integer source has no out-of-range
//!   or NaN case to gate, so the contract is total in both strengths and any
//!   disagreement is a real defect.
//! - **float source -> integer destination**, both strengths. Saturating runs
//!   over the raw corpus including NaN and out-of-range, where it is exactly
//!   `as` by contract, while wrapping is gated to the in-range finite domain, since
//!   outside it `cast_from` is explicitly backend-defined (x86 yields the
//!   hardware indefinite integer).
//! - **float <-> float**, both strengths.
//!
//! The two strengths are asserted separately on purpose. For sign-changing
//! integer pairs they are deliberately the *same* conversion (scalar has no
//! saturating lowering for them, so `saturating_cast_from` falls through to the
//! wrapping `as`), and that equivalence is load-bearing: composing such a pair
//! through a same-signedness narrow leg would silently pick up that leg's clamp
//! and disagree with the language. `300u32 as i8` is `44`, not `127`.
//!
//! Existence of every pair is a separate question, checked at compile time by
//! `cast_matrix.rs`.
#![cfg(any(
    target_arch = "x86",
    target_arch = "x86_64",
    target_arch = "wasm32",
    target_arch = "aarch64"
))]

mod harness;

use harness::Tol;
use thermite::backend::scalar::Scalar;
use thermite::simd::Simd;

macro_rules! id {
    ($t:ty) => {
        (|x: $t| x) as fn($t) -> $t
    };
}

/// Every `cast_from` from one source slot into each listed destination slot.
macro_rules! row_cast {
    ($tag:expr, $b:ty, $src:ident, $se:ty, [$($dst:ident),* $(,)?]) => {$(
        cast_diff!(
            concat!($tag, " ", stringify!($src), "->", stringify!($dst)),
            <$b as Simd>::$src, <$b as Simd>::$dst,
            <Scalar as Simd>::$src, <Scalar as Simd>::$dst,
            $se, id!($se), Tol::Exact
        );
    )*};
}

/// Every `saturating_cast_from` from one source slot into each listed destination.
macro_rules! row_sat {
    ($tag:expr, $b:ty, $src:ident, $se:ty, [$($dst:ident),* $(,)?]) => {$(
        sat_cast_diff!(
            concat!($tag, " ", stringify!($src), "->", stringify!($dst)),
            <$b as Simd>::$src, <$b as Simd>::$dst,
            <Scalar as Simd>::$src, <Scalar as Simd>::$dst,
            $se
        );
    )*};
}

/// A float -> int domain guard: finite and inside the destination's range, where
/// truncation is unambiguous and `cast_from` is contractually `as`.
///
/// The 8- and 16-bit bounds are the destination's exact limits, which every
/// float format represents exactly. The 32- and 64-bit bounds are deliberately
/// slack: `i64::MAX as f64` rounds *up* to `2^63`, which is not a representable
/// `i64`, so clamping to it would push the very value being tested back out of
/// range. Rounding the bound down by a few ULP costs no coverage the saturating
/// suite is not already getting over the raw corpus.
macro_rules! dom {
    ($fe:ty, $lo:expr, $hi:expr) => {
        (|x: $fe| if x.is_finite() { x.clamp($lo, $hi) } else { 0.0 }) as fn($fe) -> $fe
    };
}

/// A float -> int guard for `fast_cast`: in range **and integral**.
///
/// Stricter than [`dom`] on purpose. The narrow-domain lowerings are
/// magic-constant tricks that round to nearest, where `as` truncates, so a
/// fractional input would fail against the oracle for a reason that is not a
/// bug, since the relaxed rounding is what `fast_cast` sells. Truncating first makes
/// the two agree and leaves the test measuring the conversion itself.
macro_rules! fdom {
    ($fe:ty, $lo:expr, $hi:expr) => {
        (|x: $fe| if x.is_finite() { x.clamp($lo, $hi).trunc() } else { 0.0 }) as fn($fe) -> $fe
    };
}

/// An int -> float guard for `fast_cast`: clamp into the lowering's domain.
macro_rules! idom {
    ($ie:ty, $lo:expr, $hi:expr) => {
        (|x: $ie| x.clamp($lo, $hi)) as fn($ie) -> $ie
    };
}

// Skipped for the same reason as `lane_suite!`: kept on the same argument
// layout as `row_cast!`/`row_sat!` above, which rustfmt leaves alone because it
// cannot descend into their repetition.
#[rustfmt::skip]
macro_rules! fast_dom {
    ($tag:expr, $b:ty, $src:ident, $se:ty, $dst:ident, $prep:expr) => {
        fast_cast_diff!(
            concat!($tag, " ", stringify!($src), "->", stringify!($dst), " [fast, in-domain]"),
            <$b as Simd>::$src, <$b as Simd>::$dst,
            <Scalar as Simd>::$src, <Scalar as Simd>::$dst,
            $se, $prep
        );
    };
}

// Skipped for the same reason as `lane_suite!`: kept on the same argument
// layout as `row_cast!`/`row_sat!` above, which rustfmt leaves alone because it
// cannot descend into their repetition.
#[rustfmt::skip]
macro_rules! cast_dom {
    ($tag:expr, $b:ty, $src:ident, $se:ty, $dst:ident, $lo:expr, $hi:expr) => {
        cast_diff!(
            concat!($tag, " ", stringify!($src), "->", stringify!($dst), " [in-range]"),
            <$b as Simd>::$src, <$b as Simd>::$dst,
            <Scalar as Simd>::$src, <Scalar as Simd>::$dst,
            $se, dom!($se, $lo, $hi), Tol::Exact
        );
    };
}

/// One lane count's worth of tests for one backend.
///
/// The slot names are spelled out rather than built from the lane count because
/// the `Simd` slots are plain associated-type names, and there is no concatenating
/// them without a proc macro, and being explicit matches how the backend cast
/// matrices are written.
///
/// `rustfmt::skip` because the point of each row is the shape of its destination
/// list (which nine of the ten slots it names), and rustfmt breaks every
/// invocation onto eight lines, which buries exactly that.
#[rustfmt::skip]
macro_rules! lane_suite {
    ($modname:ident, $b:ty, $tag:expr,
     $f32:ident, $f64:ident, $i8:ident, $u8:ident, $i16:ident, $u16:ident,
     $i32:ident, $u32:ident, $i64:ident, $u64:ident) => {
        mod $modname {
            use super::*;

            /// 8 integer sources x 9 destinations, wrapping strength.
            #[test]
            fn int_src_cast() {
                row_cast!($tag, $b, $i8,  i8,  [$f32, $f64, $u8, $i16, $u16, $i32, $u32, $i64, $u64]);
                row_cast!($tag, $b, $u8,  u8,  [$f32, $f64, $i8, $i16, $u16, $i32, $u32, $i64, $u64]);
                row_cast!($tag, $b, $i16, i16, [$f32, $f64, $i8, $u8,  $u16, $i32, $u32, $i64, $u64]);
                row_cast!($tag, $b, $u16, u16, [$f32, $f64, $i8, $u8,  $i16, $i32, $u32, $i64, $u64]);
                row_cast!($tag, $b, $i32, i32, [$f32, $f64, $i8, $u8,  $i16, $u16, $u32, $i64, $u64]);
                row_cast!($tag, $b, $u32, u32, [$f32, $f64, $i8, $u8,  $i16, $u16, $i32, $i64, $u64]);
                row_cast!($tag, $b, $i64, i64, [$f32, $f64, $i8, $u8,  $i16, $u16, $i32, $u32, $u64]);
                row_cast!($tag, $b, $u64, u64, [$f32, $f64, $i8, $u8,  $i16, $u16, $i32, $u32, $i64]);
            }

            /// The same 72 pairs at saturating strength. For the same-signedness
            /// narrows this is a genuine clamp, while for the sign-changing ones it
            /// must still agree with the wrapping `as`, which is what pins the
            /// composition order.
            #[test]
            fn int_src_saturating() {
                row_sat!($tag, $b, $i8,  i8,  [$f32, $f64, $u8, $i16, $u16, $i32, $u32, $i64, $u64]);
                row_sat!($tag, $b, $u8,  u8,  [$f32, $f64, $i8, $i16, $u16, $i32, $u32, $i64, $u64]);
                row_sat!($tag, $b, $i16, i16, [$f32, $f64, $i8, $u8,  $u16, $i32, $u32, $i64, $u64]);
                row_sat!($tag, $b, $u16, u16, [$f32, $f64, $i8, $u8,  $i16, $i32, $u32, $i64, $u64]);
                row_sat!($tag, $b, $i32, i32, [$f32, $f64, $i8, $u8,  $i16, $u16, $u32, $i64, $u64]);
                row_sat!($tag, $b, $u32, u32, [$f32, $f64, $i8, $u8,  $i16, $u16, $i32, $i64, $u64]);
                row_sat!($tag, $b, $i64, i64, [$f32, $f64, $i8, $u8,  $i16, $u16, $i32, $u32, $u64]);
                row_sat!($tag, $b, $u64, u64, [$f32, $f64, $i8, $u8,  $i16, $u16, $i32, $u32, $i64]);
            }

            /// Float -> int over the RAW corpus (NaN, infinities, out of range).
            /// Saturating is total and `as`-exact here, so nothing is gated.
            #[test]
            fn float_src_saturating() {
                row_sat!($tag, $b, $f32, f32, [$i8, $u8, $i16, $u16, $i32, $u32, $i64, $u64]);
                row_sat!($tag, $b, $f64, f64, [$i8, $u8, $i16, $u16, $i32, $u32, $i64, $u64]);
            }

            /// Float -> int at wrapping strength, gated to the in-range finite
            /// domain where `cast_from` is contractually `as`.
            #[test]
            fn float_src_cast_in_range() {
                cast_dom!($tag, $b, $f32, f32, $i8,  -128.0, 127.0);
                cast_dom!($tag, $b, $f32, f32, $u8,  0.0, 255.0);
                cast_dom!($tag, $b, $f32, f32, $i16, -32768.0, 32767.0);
                cast_dom!($tag, $b, $f32, f32, $u16, 0.0, 65535.0);
                cast_dom!($tag, $b, $f32, f32, $i32, -2.0e9, 2.0e9);
                cast_dom!($tag, $b, $f32, f32, $u32, 0.0, 4.0e9);
                cast_dom!($tag, $b, $f32, f32, $i64, -9.0e18, 9.0e18);
                cast_dom!($tag, $b, $f32, f32, $u64, 0.0, 1.8e19);

                cast_dom!($tag, $b, $f64, f64, $i8,  -128.0, 127.0);
                cast_dom!($tag, $b, $f64, f64, $u8,  0.0, 255.0);
                cast_dom!($tag, $b, $f64, f64, $i16, -32768.0, 32767.0);
                cast_dom!($tag, $b, $f64, f64, $u16, 0.0, 65535.0);
                cast_dom!($tag, $b, $f64, f64, $i32, -2.0e9, 2.0e9);
                cast_dom!($tag, $b, $f64, f64, $u32, 0.0, 4.0e9);
                cast_dom!($tag, $b, $f64, f64, $i64, -9.0e18, 9.0e18);
                cast_dom!($tag, $b, $f64, f64, $u64, 0.0, 1.8e19);
            }

            /// `fast_cast` over the domain where it is actually defined.
            ///
            /// This is the operation the `_limited` magic-constant lowerings
            /// back. A hardcoded `i32 -> f32` smoke check on `[1, 2, 3, 4]`, all
            /// positive, is not a differential. That is how a wrong bias-unbias in the signed
            /// `f64 -> i64` direction survived: it is correct for every input
            /// strictly between 0 and 2^51 and wrong for the negatives.
            ///
            /// The 64-bit bounds are the polyfills' documented domains,
            /// `[-2^51, 2^51]` signed and `[0, 2^52)` unsigned, and the corpus
            /// carries negatives and both endpoints into them.
            #[test]
            fn fast_cast_in_domain() {
                const I64_LO: i64 = -(1 << 51);
                const I64_HI: i64 = 1 << 51;
                const U64_HI: u64 = (1 << 52) - 1;

                fast_dom!($tag, $b, $f64, f64, $i64, fdom!(f64, -2251799813685248.0, 2251799813685248.0));
                fast_dom!($tag, $b, $f64, f64, $u64, fdom!(f64, 0.0, 4503599627370495.0));
                fast_dom!($tag, $b, $i64, i64, $f64, idom!(i64, I64_LO, I64_HI));
                fast_dom!($tag, $b, $u64, u64, $f64, idom!(u64, 0, U64_HI));

                // The 32-bit rungs have no narrow-domain shortcut, so `fast_cast`
                // is `cast` there and the full in-range domain applies.
                fast_dom!($tag, $b, $f32, f32, $i32, fdom!(f32, -2.0e9, 2.0e9));
                fast_dom!($tag, $b, $f32, f32, $u32, fdom!(f32, 0.0, 4.0e9));
                fast_dom!($tag, $b, $i32, i32, $f32, idom!(i32, i32::MIN, i32::MAX));
                fast_dom!($tag, $b, $u32, u32, $f32, idom!(u32, 0, u32::MAX));
            }

            /// f32 <-> f64, both strengths. The narrowing direction rounds, but
            /// scalar rounds identically, so the differential is still exact.
            #[test]
            fn float_to_float() {
                row_cast!($tag, $b, $f32, f32, [$f64]);
                row_cast!($tag, $b, $f64, f64, [$f32]);
                row_sat!($tag, $b, $f32, f32, [$f64]);
                row_sat!($tag, $b, $f64, f64, [$f32]);
            }
        }
    };
}

/// All four lane counts for one backend.
#[rustfmt::skip]
macro_rules! backend_suite {
    ($modname:ident, $b:ty, $tag:expr) => {
        mod $modname {
            use super::*;

            lane_suite!(x2, $b, concat!($tag, " x2"),
                f32x2, f64x2, i8x2, u8x2, i16x2, u16x2, i32x2, u32x2, i64x2, u64x2);
            lane_suite!(x4, $b, concat!($tag, " x4"),
                f32x4, f64x4, i8x4, u8x4, i16x4, u16x4, i32x4, u32x4, i64x4, u64x4);
            lane_suite!(x8, $b, concat!($tag, " x8"),
                f32x8, f64x8, i8x8, u8x8, i16x8, u16x8, i32x8, u32x8, i64x8, u64x8);
            lane_suite!(x16, $b, concat!($tag, " x16"),
                f32x16, f64x16, i8x16, u8x16, i16x16, u16x16, i32x16, u32x16, i64x16, u64x16);
        }
    };
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
mod x86 {
    use super::*;

    backend_suite!(v1, thermite::backend::x86_v1::X86V1, "v1");
    backend_suite!(v2, thermite::backend::x86_v2::X86V2, "v2");
    backend_suite!(v3, thermite::backend::x86_v3::X86V3, "v3");
}

#[cfg(target_arch = "wasm32")]
mod wasm {
    use super::*;

    backend_suite!(simd128, thermite::backend::wasm::Wasm, "wasm");
}

#[cfg(target_arch = "aarch64")]
mod neon {
    use super::*;

    backend_suite!(advsimd, thermite::backend::neon::Neon, "neon");
}
